#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import gc
import sys
import time
import yaml
import shutil
from datetime import datetime
from typing import Optional

from trainer.sft_config import SFTConfig
from trainer.sft_trainer import SFTTrainer
from peft import LoraConfig, get_peft_model

import fire
import wandb
# import weave
from datasets import Dataset
# import evaluate

from utils.init_functions import logger_setup, cuda_setup, random_setup
from utils.models import ModelUtils
from utils.data_io import DataIO

os.environ["HF_ALLOW_CODE_EVAL"] = "1"


class SFTTraining:

    def __init__(
            self,
            verbose: bool,
            logger,
            seed: int,
            cuda_dict: Optional[dict] = None,
            cache_dir: Optional[str] = None,
            project_root_dir: Optional[str] = None,
            ckpt_root_dir: Optional[str] = None,
            ckpt_save_dir: Optional[str] = None,
            model_ckpt_dir: Optional[str] = None,
            config_dir: Optional[str] = "config/ft",
            model_name: str = "llama3-8b",
            run_id: str = "default_run",
            label_model_name: str = "llama3-8b",
            training_task_name: str = "mmlu_training",
            training_data_dir: Optional[str] = None,
            training_data_type: str = "raw",
            use_lora: bool = False,
            lora_r: int = 16,
            lora_alpha: int = 16,
            lora_dropout: float = 0.0,
            valid_num: Optional[int] = None,
            valid_bsz: int = 1,
            valid_on_start: bool = False,
            max_seq_len: Optional[int] = 4096,
            num_train_epochs: Optional[float] = 1.0,
            learning_rate: Optional[float] = float("5e-05"),
            use_wandb: bool = False,
            push_to_hub: bool = False,
            debug: bool = False,
    ):
        self.verbose = verbose
        self.logger = logger
        self.seed = seed
        self.cuda_dict = cuda_dict
        assert os.path.isdir(project_root_dir) and os.path.isdir(ckpt_save_dir)
        self.project_root_dir = project_root_dir
        self.ckpt_root_dir = ckpt_root_dir
        self.ckpt_save_dir = ckpt_save_dir
        self.model_ckpt_dir = model_ckpt_dir
        self.model_name = model_name
        self.run_id = run_id
        self.use_wandb = use_wandb
        self.push_to_hub = push_to_hub
        self.debug = debug

        # LoRA settings
        self.use_lora = use_lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout

        # Training data
        self.label_model_name = label_model_name
        self.training_task_name = training_task_name
        if not isinstance(training_data_dir, str):
            training_data_dir = "data/ft_data"
        assert os.path.isdir(training_data_dir), training_data_dir
        training_data_fn = f"ft_data_{training_data_type}.jsonl"
        training_data_fp = os.path.join(
            training_data_dir, label_model_name, training_task_name, training_data_fn)
        assert os.path.isfile(training_data_fp), training_data_fp
        self.training_data_dir = training_data_dir
        self.training_data_type = training_data_type
        self.training_data_fn = training_data_fn
        self.training_data_fp = training_data_fp
        self.training_data_list = DataIO.load_jsonl(training_data_fp, mode="r", verbose=True)
        if self.debug:
            self.training_data_list = self.training_data_list[:1000]

        # Validation during training
        self.valid_num = valid_num
        self.valid_bsz = valid_bsz
        self.valid_on_start = valid_on_start
        valid_data_fp = os.path.join(
            training_data_dir, label_model_name, training_task_name, "ft_data_validation.jsonl")
        assert os.path.isfile(valid_data_fp), valid_data_fp
        self.valid_data_list = DataIO.load_jsonl(valid_data_fp, mode="r", verbose=True)

        # Training configurations
        self.num_train_epochs = num_train_epochs
        self.learning_rate = learning_rate
        self.max_seq_len = max_seq_len
        self.common_training_args = {}  # The training arguments used for all trainers. `common_training_args.yaml`
        self.sft_trainer_args = {}  # SFT trainer arguments. `sft_trainer_args.yaml`
        if isinstance(config_dir, str) and os.path.isdir(config_dir):
            common_training_args_fp = os.path.join(config_dir, "common_training_args.yaml")
            sft_trainer_args_fp = os.path.join(config_dir, "sft_trainer_args.yaml")
            if os.path.isfile(common_training_args_fp):
                with open(common_training_args_fp, "r", encoding="utf-8") as fp_in:
                    self.common_training_args = yaml.load(fp_in, Loader=yaml.FullLoader)
            if os.path.isfile(sft_trainer_args_fp):
                with open(sft_trainer_args_fp, "r", encoding="utf-8") as fp_in:
                    self.sft_trainer_args = yaml.load(fp_in, Loader=yaml.FullLoader)

        # Training strategy
        self.training_strategy = "sft"

        # Cache directory
        self.home_dir = os.path.expanduser("~")
        if isinstance(cache_dir, str) and os.path.isdir(cache_dir):
            self.cache_dir = cache_dir
        else:
            self.cache_dir = os.path.join(self.home_dir, ".cache/huggingface")
            # self.cache_dir = os.path.join(self.project_root_dir, ".cache/huggingface/")
            if not os.path.isdir(self.cache_dir):
                os.makedirs(self.cache_dir, exist_ok=True)
        if self.verbose:
            self.logger.info(f">>> cache_dir: {self.cache_dir}")

        os.environ["HF_HOME"] = self.cache_dir

        # Tokenizers for training and validation
        tokenizer_train = ModelUtils.initialize_tokenizer_hf(
            model_name=self.model_name, cache_dir=self.cache_dir, verbose=self.verbose,
            padding_side="right", truncation_side="right")
        tokenizer_valid = ModelUtils.initialize_tokenizer_hf(
            model_name=self.model_name, cache_dir=self.cache_dir, verbose=self.verbose,
            padding_side="left", truncation_side="left")

        self.terminators_train = [
            tokenizer_train.eos_token_id,
            # tokenizer_train.convert_tokens_to_ids("<|eot_id|>")
            tokenizer_train.convert_tokens_to_ids(tokenizer_train.eos_token)
        ]
        self.terminators_valid = [
            tokenizer_valid.eos_token_id,
            # tokenizer_valid.convert_tokens_to_ids("<|eot_id|>")
            tokenizer_valid.convert_tokens_to_ids(tokenizer_valid.eos_token)
        ]
        self.terminators_train = list(set(self.terminators_train))
        self.terminators_valid = list(set(self.terminators_valid))

        # GPT-4 context window: 128K --> We require the max sequence length to be <= 120K
        max_len = min(tokenizer_train.model_max_length, tokenizer_valid.model_max_length,
                      tokenizer_train.max_len_single_sentence, tokenizer_valid.max_len_single_sentence)
        self.MAX_GPT_WINDOW = min(120000, max_len)  # https://platform.openai.com/docs/models/gpt-4o
        self.MAX_SEQ_LEN = min(max_len, self.MAX_GPT_WINDOW)
        if self.max_seq_len is None or self.max_seq_len <= 0:
            self.max_seq_len = self.MAX_SEQ_LEN
        else:
            self.max_seq_len = min(self.max_seq_len, max_len)
        if self.verbose:
            self.logger.info(f">>> len(tokenizer_train.vocab) = {len(tokenizer_train.vocab)}")
            self.logger.info(f">>> len(tokenizer_valid.vocab) = {len(tokenizer_valid.vocab)}")
            self.logger.info(f">>> tokenizer.max_len_single_sentence = {max_len}")  # LLaMA-3.1: 131071
            self.logger.info(f"max_seq_len = {self.max_seq_len}; MAX_GPT_WINDOW = {self.MAX_GPT_WINDOW}")
        self.tokenizer_train = tokenizer_train
        self.tokenizer_valid = tokenizer_valid

        self.hub_model_id = f"YOUR_HF_ID/{self.run_id}"

    def finetune(
            self,
    ):
        data_train_list = self.training_data_list
        data_valid_list = self.valid_data_list

        data_train_list = [{"messages": _item["messages"]} for _item in data_train_list]

        data_train = Dataset.from_list(data_train_list)
        data_valid = Dataset.from_list(data_valid_list)

        # Load the base model
        if self.verbose:
            self.logger.info(f">>> model_name: {self.model_name}")
            self.logger.info(f">>> label_model_name: {self.label_model_name}")
            self.logger.info(f">>> model_ckpt_dir: {self.model_ckpt_dir}")

        if isinstance(self.model_ckpt_dir, str) and os.path.isdir(self.model_ckpt_dir):
            # Resume training
            self.logger.info(f">>> Resume training from {self.model_ckpt_dir}")
            model = ModelUtils.initialize_model_hf(
                model_name=self.model_name, cache_dir=self.cache_dir, verbose=self.verbose,
                do_train=True, do_4bit=False, do_bf16=True, do_fp16=False,
                model_ckpt_dir=self.model_ckpt_dir,
            )
            model.generation_config.pad_token_id = self.tokenizer_valid.pad_token_id  # eos_token_id
            model.train()
        else:
            # New training session
            self.logger.info(f">>> New training session: model_name = {self.model_name}")
            model = ModelUtils.initialize_model_hf(
                model_name=self.model_name, cache_dir=self.cache_dir, verbose=self.verbose,
                do_train=True, do_4bit=False, do_bf16=True, do_fp16=False,
                # model_ckpt_dir=self.model_ckpt_dir,
            )
            # model = model.to(device=self.cuda_dict["device"])
            # list(model.state_dict().keys())
            model.generation_config.pad_token_id = self.tokenizer_valid.pad_token_id  # eos_token_id
            # model.resize_token_embeddings(len(self.tokenizer_train))  # if added new special tokens (Option 1)
            model.train()

            # LoRA (PEFT training)
            if self.use_lora:
                self.logger.info(f">>> LoRA training: [rank: {self.lora_r}] "
                                 f"[alpha: {self.lora_alpha}] [dropout: {self.lora_dropout}]")
                lora_config = LoraConfig(
                    r=self.lora_r,  # any number > 0; Suggested 8, 16 (default), 32, 64, 128
                    lora_alpha=self.lora_alpha,  # Best to choose alpha = rank or rank*2 (default: 16)
                    lora_dropout=self.lora_dropout,  # default: 0.0 (optimized)
                    # target_modules=["query", "value"],  # ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head", "embed_tokens"]
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                    # modules_to_save=["classifier"],  # List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint.
                    bias="none",  # Bias type for Lora. Can be 'none', 'all' or 'lora_only'
                    init_lora_weights=True,
                    # init_lora_weights="gaussian",  # ["gaussian", "eva", "olora", "pissa", "pissa_niter_[number of iters]", "corda", "loftq", "orthogonal"]
                    # random_state=3407,  # https://arxiv.org/abs/2109.08203
                    # use_rslora=False,  # rank stabilized LoRA
                    # loftq_config=None,  # LoftQ: LoRA-Fine-Tuning-Aware Quantization
                )
                self.logger.info(f">>> LoRA: lora_r = {self.lora_r}; lora_alpha = {self.lora_alpha}; "
                                 f"lora_dropout = {self.lora_dropout}")
                model = get_peft_model(model, lora_config)
                model.print_trainable_parameters()
                model.train()

        # DataParallel. torch.nn.parallel.DistributedDataParallel
        # model = DataParallel(model)

        # Set up a TrainingArguments class with training hyperparameters
        if self.use_wandb:
            report_to = "wandb"
            # report_to = "all"
        else:
            report_to = "none"

        # Common training parameters in TrainingArguments. (Overwrite the .yaml files)
        self.common_training_args["seed"] = self.seed
        self.common_training_args["data_seed"] = self.seed
        self.common_training_args["resume_from_checkpoint"] = self.model_ckpt_dir
        self.common_training_args["run_name"] = self.ckpt_save_dir
        self.common_training_args["output_dir"] = self.ckpt_save_dir
        self.common_training_args["report_to"] = report_to
        self.common_training_args["do_eval"] = False
        self.common_training_args["eval_on_start"] = self.valid_on_start
        # self.common_training_args["metric_for_best_model"] = "loss"
        self.common_training_args["metric_for_best_model"] = "valid_score"
        self.common_training_args["eval_strategy"] = "no"
        self.common_training_args["eval_steps"] = None
        self.common_training_args["load_best_model_at_end"] = False

        self.common_training_args["num_train_epochs"] = self.num_train_epochs
        self.common_training_args["learning_rate"] = self.learning_rate

        # Training
        assert self.training_strategy == "sft"

        # SFT training parameters. (Overwrite the .yaml files)
        self.sft_trainer_args["eos_token"] = self.tokenizer_train.eos_token  # default: None
        self.sft_trainer_args["pad_token"] = self.tokenizer_train.pad_token  # default: None
        self.sft_trainer_args["max_length"] = self.max_seq_len  # If `None`, no truncation is applied. default: 1024
        config = SFTConfig(
            **self.sft_trainer_args,   # SFT training parameters
            **self.common_training_args,  # Common training parameters in TrainingArguments
        )
        extra_kwargs = {
            # Basic information
            "verbose": self.verbose,
            "logger": self.logger,
            "cache_dir": self.cache_dir,
            "project_root_dir": self.project_root_dir,
            "ckpt_save_dir": self.ckpt_save_dir,
            # Training information
            "model_name": self.model_name,
            "training_task_name": self.training_task_name,
            "training_data_dir": self.training_data_dir,
            "training_data_type": self.training_data_type,
            "label_model_name": self.label_model_name,
            # Validation settings
            "max_new_gen": 2048,  # The same as the setting during test set evaluation
            "gen_temperature": 0.0,
            "valid_num": self.valid_num,
            "valid_bsz": self.valid_bsz,
        }  # Note: these `extra_kwargs` will be used in the trainer modules of trl and transformers
        self.logger.info(f">>> [common_training_args]: {self.common_training_args}")
        self.logger.info(f">>> [sft_trainer_args]: {self.sft_trainer_args}")
        # self.logger.info(f">>> [extra_kwargs]: {extra_kwargs}")
        DataIO.show_dict(input_dict=extra_kwargs, dict_name="extra_kwargs", logger=self.logger)

        trainer = SFTTrainer(
            model=model,
            args=config,
            data_collator=None,  # default: DataCollatorForLanguageModeling
            train_dataset=data_train,
            eval_dataset=data_valid if len(data_valid_list) > 0 else None,
            processing_class=None,
            # processing_class=self.tokenizer_train,  # default: None
            compute_loss_func=None,
            compute_metrics=None,  # compute_metrics_ppl
            callbacks=None,
            optimizers=(None, None),
            optimizer_cls_and_kwargs=None,
            preprocess_logits_for_metrics=None,
            peft_config=None,
            formatting_func=None,
            **extra_kwargs
        )

        # Save the hyperparameters before training
        hyper_param_dir = os.path.join(self.ckpt_save_dir, "hyper_params")
        os.makedirs(hyper_param_dir, exist_ok=True)
        # with open(os.path.join(hyper_param_dir, "common_training_args.json"), "w", encoding="utf-8") as fp_out:
        #     json.dump(self.common_training_args, fp_out, indent=4)
        # with open(os.path.join(hyper_param_dir, "sft_trainer_args.json"), "w", encoding="utf-8") as fp_out:
        #     json.dump(self.sft_trainer_args, fp_out, indent=4)
        DataIO.save_json(os.path.join(
            hyper_param_dir, "common_training_args.json"), self.common_training_args, mode="w", indent=2)
        DataIO.save_json(os.path.join(
            hyper_param_dir, "sft_trainer_args.json"), self.sft_trainer_args, mode="w", indent=2)

        # output_disabled = model.generate(**inputs)
        # self.logger.info(self.tokenizer_train.decode(output_disabled[0], skip_special_tokens=True))

        # Run IA fine-tuning
        training_results = trainer.train(resume_from_checkpoint=self.model_ckpt_dir)
        self.logger.info(f">>> Training finished. TrainOutput:\n{training_results}")

        # Save the final model checkpoint
        ckpt_fn_list = os.listdir(self.ckpt_save_dir)
        ckpt_fn_list = [fn for fn in ckpt_fn_list if fn.startswith("checkpoint-")]
        load_best_model_at_end = self.common_training_args["load_best_model_at_end"]
        self.logger.info(f">>> load_best_model_at_end = {load_best_model_at_end}")
        if len(ckpt_fn_list) == 0:
            self.logger.info(f">>> No checkpoint saved")
        else:
            self.logger.info(f">>> Saved checkpoints: {ckpt_fn_list}")
            ckpt_fn_list.sort(key=lambda x: int(x.split("-")[-1]))
            last_ckpt_fn = ckpt_fn_list[-1]
            last_ckpt_dir = os.path.join(self.ckpt_save_dir, last_ckpt_fn)
            best_ckpt_dir = os.path.join(self.ckpt_save_dir, "last_model")
            # First, copy tokenizers and other training configurations
            shutil.copytree(last_ckpt_dir, best_ckpt_dir, dirs_exist_ok=True)
            # model.save_pretrained(best_ckpt_dir)  # To save the model state dict (params)
            trainer.save_model(best_ckpt_dir)

        if self.push_to_hub:
            model.push_to_hub(self.hub_model_id)

        return None


def main(
        model_name: str = "llama3-8b",
        cache_dir: Optional[str] = None,
        project_root_dir: Optional[str] = None,
        ckpt_root_dir: Optional[str] = None,
        model_ckpt_dir: Optional[str] = None,
        config_dir: Optional[str] = "config/ft",
        seed: int = 42,
        cuda: Optional[str] = None,
        label_model_name: str = "llama3-8b",
        training_task_name: str = "mmlu_training",
        training_data_dir: Optional[str] = None,
        training_data_type: str = "raw",
        max_seq_len: Optional[int] = 4096,
        num_train_epochs: Optional[float] = 1.0,
        learning_rate: Optional[float] = float("5e-05"),
        wandb_key: Optional[str] = None,
        train_mode: Optional[str] = None,
        lora_mode: Optional[str] = None,
        valid_mode: Optional[str] = None,
        verbose: bool = False,
        push_to_hub: bool = False,
        **kwargs
) -> None:
    """
    :param model_name: The model to be trained. E.g., "llama3-8b", "qwen2.5-7b"
    :param cache_dir: The root directory of the cache.
    :param project_root_dir: The directory of the project root.
    :param ckpt_root_dir: The directory path to save the model checkpoints.
    :param model_ckpt_dir: The directory path to the model checkpoints for resuming running.
    :param config_dir: The directory storing configuration files. (One `config_dir` means one training setting.)
    :param seed: Random seed of all modules.
    :param cuda: To specify CUDA GPU devices, e.g., "0" OR "0,1". Default: None -- Use CPU or all available GPUs.
    :param label_model_name: The labeling model that ran the training set. (`model_name` in `run_build_ft_data.py`)
    :param training_task_name: The training task. (I.e., `task_name` in `run_build_ft_data.py`)
    :param training_data_dir: The training data directory. (I.e., `ft_data_save_dir` in `run_build_ft_data.py`)
    :param training_data_type: The type of the training data. E.g., "raw", "da", "ia", or "ia_rectify"
    :param max_seq_len: The max length of input sequence (ids). -1 means max model input length.
    :param num_train_epochs: The number of training epochs.
    :param learning_rate: The initial learning rate.
    :param verbose: Verbose mode: show logs.
    :param wandb_key: The wandb key. Use wandb to save & show training logs.
    :param push_to_hub: Whether push the model checkpoints to Hugging Face Hub.
    :param train_mode: The training configurations: use_wandb, use_lora, and debug
    :param lora_mode: The LoRA configurations: rank, alpha, and dropout
    :param valid_mode: The Validation configurations: valid_num, valid_bsz, and valid_on_start

    :return: None.
    """

    timer_start = time.perf_counter()

    # Setup of the logger, CUDA gpus, and random seed
    logger = logger_setup("LM_FineTuning")
    cuda_dict = cuda_setup(cuda=cuda, logger=logger, verbose=verbose)
    random_setup(seed=seed, has_cuda=cuda_dict["has_cuda"])
    logger.info(f">>> cuda_dict:\n{cuda_dict}")

    if isinstance(kwargs, dict):
        logger.info(f">>> Extra parameters in kwargs: {kwargs}")

    if not (isinstance(project_root_dir, str) and os.path.isdir(project_root_dir)):
        project_root_dir = os.getcwd()
    assert os.path.isdir(project_root_dir)
    if not (isinstance(ckpt_root_dir, str) and os.path.isdir(ckpt_root_dir)):
        ckpt_root_dir = os.path.join(project_root_dir, "ckpt")
    os.makedirs(ckpt_root_dir, exist_ok=True)
    if not (isinstance(training_data_dir, str) and os.path.isdir(training_data_dir)):
        training_data_dir = os.path.join(project_root_dir, "data/ft_data/")
    assert os.path.isdir(training_data_dir)

    # Parse the `train_mode`
    if (isinstance(train_mode, tuple) or isinstance(train_mode, list) or
            (isinstance(train_mode, str) and len(train_mode.strip()) > 0)):
        if isinstance(train_mode, tuple):
            train_mode = list(train_mode)
        if isinstance(train_mode, str):
            train_mode = train_mode.strip()
            train_mode = train_mode.split(",")
        assert len(train_mode) == 3, (f">>> AssertionError: train_mode must have "
                                      f"3 values (like `0,0,0`), but got {train_mode}")
        # All "1"s means turning all the following boolean setting on
        # use_wandb: Whether to use WanDB for logs.
        # use_lora: Whether to use LoRA (PEFT training).
        # debug: Debugging / developing mode.
        use_wandb = str(train_mode[0]) == "1"
        use_lora = str(train_mode[1]) == "1"
        debug = str(train_mode[2]) == "1"
    else:
        # default: All "0"s
        train_mode = ["0" for _ in range(3)]
        use_wandb = use_lora = debug = False
    logger.info(f">>> train_mode = {train_mode}: [use_wandb: {use_wandb}] [use_lora: {use_lora}] [debug: {debug}]")
    logger.info(f">>> [num_train_epochs = {num_train_epochs}] [learning_rate: {learning_rate}]")

    # Parse the `lora_mode`
    if (isinstance(lora_mode, tuple) or isinstance(lora_mode, list) or
            (isinstance(lora_mode, str) and len(lora_mode.strip()) > 0)):
        if isinstance(lora_mode, tuple):
            lora_mode = list(lora_mode)
        if isinstance(lora_mode, str):
            lora_mode = lora_mode.strip()
            lora_mode = lora_mode.split(",")
        assert len(lora_mode) == 3, (f">>> AssertionError: lora_mode must have "
                                     f"3 values (like `16,16,0.0`), but got {lora_mode}")
        # All "1"s means turning all the following boolean setting on
        # lora_r: LoRA rank.
        # lora_alpha: LoRA alpha.
        # lora_dropout: LoRA dropout.
        lora_r = max(1, int(lora_mode[0]))
        lora_alpha = max(1, int(lora_mode[1]))
        lora_dropout = max(0.0, float(lora_mode[2]))
    else:
        # default: All "0"s
        lora_mode = ["16", "16", "0.0"]
        lora_r = lora_alpha = 16
        lora_dropout = float(0.0)
    logger.info(f">>> lora_mode = {lora_mode}: [rank: {lora_r}] [alpha: {lora_alpha}] [dropout: {lora_dropout}]")

    # Parse the `valid_mode`
    if (isinstance(valid_mode, tuple) or isinstance(valid_mode, list) or
            (isinstance(valid_mode, str) and len(valid_mode.strip()) > 0)):
        if isinstance(valid_mode, tuple):
            valid_mode = list(valid_mode)
        if isinstance(valid_mode, str):
            valid_mode = valid_mode.strip()
            valid_mode = valid_mode.split(",")
        assert len(valid_mode) == 3, (f">>> AssertionError: valid_mode must have "
                                      f"3 values (like `100,1,0`), but got {valid_mode}")
        # All "1"s means turning all the following boolean setting on
        # valid_num: 100 means using 100 random instances from the raw validation set
        # valid_bsz: The batch size for validation.
        # valid_on_start: Run validation before training.
        valid_num = max(10, int(valid_mode[0]))
        valid_bsz = max(1, int(valid_mode[1]))
        valid_on_start = str(valid_mode[2]) == "1"
    else:
        # default: All "0"s
        valid_mode = ["100", "1", "0"]
        valid_num = 100
        valid_bsz = 1
        valid_on_start = False
    logger.info(f">>> valid_mode = {valid_mode}: [valid_num: {valid_num}] [valid_bsz: {valid_bsz}] "
                f"[valid_on_start: {valid_on_start}]")

    # Resume fine-tuning by loading the existing model parameters and training states
    train_mode_str = "_".join([str(x) for x in train_mode])
    lora_mode_str = "_".join([str(x) for x in lora_mode])
    valid_mode_str = "_".join([str(x) for x in valid_mode])
    run_id = (f"IA-FT-{model_name}--{training_task_name}--{training_data_type}--{label_model_name}--"
              f"{train_mode_str}--{lora_mode_str}--{valid_mode_str}--{num_train_epochs}--{learning_rate}")
    cur_time = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    if isinstance(model_ckpt_dir, str) and os.path.isdir(model_ckpt_dir):
        # If `model_ckpt_dir` is a valid directory, like "ckpt/`run_id`/`cur_time`/checkpoint-1000/"
        ckpt_save_dir = os.path.abspath(os.path.join(model_ckpt_dir, os.pardir))  # "ckpt/`run_id`/`cur_time`/"
        logger.info(f">>> Resume Running: ckpt_save_dir = {ckpt_save_dir}; model_ckpt_dir = {model_ckpt_dir}")
        model_ckpt_dir = model_ckpt_dir
        # model_ckpt_dir = True
        # model_ckpt_dir = None
    else:
        # New run. Save checkpoints and training status to "ckpt/`run_id`/`cur_time`/" folder
        if isinstance(ckpt_root_dir, str) and len(ckpt_root_dir) > 0:
            os.makedirs(ckpt_root_dir, exist_ok=True)
            ckpt_save_dir = os.path.join(ckpt_root_dir, "ckpt_ia_sft", run_id, f"{cur_time}")
        else:
            ckpt_save_dir = os.path.join(project_root_dir, "ckpt_ia_sft", run_id, f"{cur_time}")
        while os.path.isdir(ckpt_save_dir):
            time.sleep(3)
            cur_time = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
            ckpt_save_dir = os.path.join(ckpt_root_dir, "ckpt_ia_sft", run_id, f"{cur_time}")
        os.makedirs(ckpt_save_dir, exist_ok=True)
        logger.info(f">>> New Running: ckpt_save_dir = {ckpt_save_dir}")
        model_ckpt_dir = None
    assert os.path.isdir(ckpt_save_dir)

    # Use wandb to save & show logs
    if use_wandb:
        if not (isinstance(wandb_key, str) and len(wandb_key) > 0):
            wandb_key = os.getenv("WANDB_API_KEY", default=None)
        if isinstance(wandb_key, str) and len(wandb_key) > 0:
            try:
                wandb.login(key=wandb_key)
                wandb.init(
                    # project=f"IA-FT",
                    project=f"IA-FT-{model_name}--{training_task_name}--{training_data_type}--{label_model_name}",
                    group=f"{model_name}--{training_task_name}--{training_data_type}--{label_model_name}",
                    name=f"{cur_time}--{run_id}",
                    # config=vars(args),
                )
                # wandb.watch(model)
            except Exception as e:
                logger.info(f">>> !!! >>> Set --use_wandb but can NOT find a valid WANDB_API_KEY")
                logger.info(e)
                wandb.init(mode="disabled")
                use_wandb = False
        else:
            logger.info(f">>> !!! >>> Set --use_wandb but can NOT find a valid WANDB_API_KEY")
            use_wandb = False
    if not use_wandb:
        wandb.init(mode="disabled")

    ia_sft = SFTTraining(
        verbose=verbose,
        logger=logger,
        seed=seed,
        cuda_dict=cuda_dict,
        cache_dir=cache_dir,
        project_root_dir=project_root_dir,
        ckpt_root_dir=ckpt_root_dir,
        ckpt_save_dir=ckpt_save_dir,
        model_ckpt_dir=model_ckpt_dir,
        config_dir=config_dir,
        model_name=model_name,
        run_id=run_id,
        label_model_name=label_model_name,
        training_task_name=training_task_name,
        training_data_dir=training_data_dir,
        training_data_type=training_data_type,
        valid_num=valid_num,
        valid_bsz=max(1, int(valid_bsz)),
        valid_on_start=valid_on_start,
        use_lora=use_lora,
        lora_r=max(4, int(lora_r)),
        lora_alpha=max(4, int(lora_alpha)),
        lora_dropout=max(0.0, float(lora_dropout)),
        max_seq_len=max_seq_len,
        num_train_epochs=max(1.0, float(num_train_epochs)),
        learning_rate=max(float("1e-08"), float(learning_rate)),
        use_wandb=use_wandb,
        push_to_hub=push_to_hub,
        debug=debug,
    )

    ia_sft.finetune()

    timer_end = time.perf_counter()
    total_sec = timer_end - timer_start
    logger.info(f"Total Running Time: {total_sec:.1f} sec ({total_sec / 60:.1f} min; {total_sec / 3600:.2f} h)")

    gc.collect()
    sys.exit(0)


if __name__ == "__main__":
    fire.Fire(main)
