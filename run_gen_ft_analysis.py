#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import gc
import re
import time
from typing import Optional

import fire
import torch
from datasets import Dataset

from tasks.tasks_utils import TASK_CLASS_DICT

from utils.models import ModelUtils
from utils.data_io import DataIO
from utils.prompting import PromptingMethods

from utils.init_functions import logger_setup, cuda_setup, random_setup


class LMGenFtAnalysis:

    def __init__(
            self,
            verbose: bool,
            logger,
            cuda_dict: Optional[dict] = None,
            seed: int = 42,
            cache_dir: Optional[str] = None,
            project_root_dir: Optional[str] = None,
            model_name: str = "llama3-8b",
            model_ckpt_dir: Optional[str] = None,
            overwrite: bool = False,
            do_bf16: bool = False,
            do_4bit: bool = False,
            bsz: int = 1,
            debug: bool = False,
            output_dir: Optional[str] = None,
            max_new_gen: int = 2048,
            gen_temperature: float = 0.0,
            top_p: Optional[float] = None,
            top_k: Optional[float] = None,
            gen_method: str = "da",
    ):
        self.verbose = verbose
        self.logger = logger
        self.cuda_dict = cuda_dict
        self.seed = seed
        self.model_name = model_name
        self.model_ckpt_dir = model_ckpt_dir
        self.do_bf16 = do_bf16
        self.do_4bit = do_4bit
        self.debug = debug
        self.overwrite = overwrite

        if isinstance(project_root_dir, str) and os.path.isdir(project_root_dir):
            self.project_root_dir = project_root_dir
        else:
            self.project_root_dir = os.getcwd()
        assert os.path.isdir(self.project_root_dir)

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

        # Tokenizer and LLM model
        self.tokenizer = ModelUtils.initialize_tokenizer_hf(
            model_name=model_name, cache_dir=cache_dir, padding_side="left", truncation_side="left",
            verbose=verbose, model_ckpt_dir=self.model_ckpt_dir)
        self.terminators_gen = [
            self.tokenizer.eos_token_id,
            # self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)
        ]
        self.terminators_gen_set = set(self.terminators_gen)
        self.terminators_gen = list(self.terminators_gen_set)
        self.model = None

        # LM Generation settings
        self.output_dir = output_dir
        self.bsz = bsz
        self.max_new_gen = max_new_gen
        self.gen_temperature = gen_temperature
        self.top_p = top_p
        self.top_k = top_k
        self.gen_method = gen_method

        # Set the filepath to the generator model
        model_path_local = ModelUtils.get_local_model_path(self.model_name, self.cache_dir)
        if isinstance(model_path_local, str) and os.path.isdir(model_path_local):
            self.model_path = model_path_local
        else:
            self.model_path = ModelUtils.OPEN_MODEL_HF[self.model_name]

    def run_inference(
            self,
            task_name: str,
            num_total: int = -1,
            data_start_idx: int = -1,
            data_end_idx: int = -1,
    ):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        assert isinstance(self.output_dir, str), "Please specify --output_dir"
        assert task_name in TASK_CLASS_DICT, \
            f"AssertionError: task name {task_name} not in TASK_CLASS_DICT"
        eval_task_class = TASK_CLASS_DICT[task_name]

        eval_task_obj = eval_task_class(
            verbose=self.verbose,
            logger=self.logger,
            cache_dir=self.cache_dir,
            project_root_dir=self.project_root_dir,
            model_name=self.model_name,
        )

        self.logger.info(f">>> Evaluation Task: {task_name}")
        task_dict = eval_task_obj.load_task()
        dataset_list = task_dict["eval"] if "eval" in task_dict else []
        num_subtasks = len(dataset_list)
        if num_subtasks == 0:
            self.logger.info(f">>> SKIP (No Eval set) [Task: {task_name}]")

        # `num_to_eval`: The # of items per subtask in "eval" set (if exists) for evaluation. -1: use all
        task_info = dict(task_dict["task_info"])
        num_to_eval = task_info["num_to_eval"]
        assert isinstance(num_to_eval, int), num_to_eval
        num_to_eval = max(num_to_eval, num_total)

        # Load the generator model
        if self.model is None:
            if self.do_bf16:
                model = ModelUtils.initialize_model_hf(
                    model_name=self.model_name, cache_dir=self.cache_dir,
                    do_train=False, do_bf16=True, do_4bit=self.do_4bit, verbose=self.verbose,
                    model_ckpt_dir=self.model_ckpt_dir)
            else:
                model = ModelUtils.initialize_model_hf(
                    model_name=self.model_name, cache_dir=self.cache_dir,
                    do_train=False, do_fp16=True, do_4bit=self.do_4bit, verbose=self.verbose,
                    model_ckpt_dir=self.model_ckpt_dir)
            model.generation_config.pad_token_id = self.tokenizer.pad_token_id
            self.model = model

        # Deal with each task (and sub-tasks)
        if self.debug:
            # Set the max number of instances to evaluate for each subtask
            if num_subtasks <= 10:
                max_num_eval = 100
            else:
                max_num_eval = 10
            show_cnt = 10
        else:
            max_num_eval = num_to_eval
            show_cnt = 100

        all_eval_data = []
        all_subset_names = []
        for dataset_dict in dataset_list:
            cur_data_items = []
            cur_subset_names = []

            ds_name, subset_name = dataset_dict["dataset_name"], dataset_dict["subset_name"]
            split_name, eval_dataset = dataset_dict["split_name"], dataset_dict["dataset"]
            assert isinstance(eval_dataset, Dataset) or isinstance(eval_dataset, list)
            # len_dataset = len(eval_dataset)
            assert isinstance(ds_name, str) and len(ds_name) > 0

            if "options" in dataset_dict:
                ds_options = list(dataset_dict["options"])
            else:
                ds_options = []

            # Construct all the input prompts for LLM generation
            skip_cnt = 0
            for idx, data_item in enumerate(eval_dataset):
                assert isinstance(data_item, dict)
                data_item["__ds_options"] = ds_options

                cur_eval_prompt = eval_task_obj.get_prompt_ft_analysis(
                    data_item=data_item, ds_name=ds_name, subset=subset_name)

                if not isinstance(cur_eval_prompt, dict) or len(cur_eval_prompt) == 0:
                    skip_cnt += 1
                else:
                    cur_subset_names.append(subset_name)
                    cur_data_items.append(data_item)
                    num_done = len(cur_data_items)
                    if num_done >= max_num_eval > 0:
                        break

            all_eval_data.extend(cur_data_items)
            all_subset_names.extend(cur_subset_names)

        assert len(all_eval_data) == len(all_subset_names)
        len_before = len(all_eval_data)
        # if 0 <= data_start_idx < data_end_idx <= len_before:
        if 0 <= data_start_idx < data_end_idx and data_start_idx < len_before:
            do_divide_conquer = True
            all_eval_data = all_eval_data[data_start_idx: data_end_idx]
            all_subset_names = all_subset_names[data_start_idx: data_end_idx]
            self.logger.info(f">>> [data_start_idx = {data_start_idx}] [data_end_idx = {data_end_idx}] "
                             f"len(all_eval_data): {len_before} --> {len(all_eval_data)}")
            output_fn = f"results_gen--{data_start_idx}__{data_end_idx}"
        else:
            do_divide_conquer = False
            output_fn = "results_gen"
        # len_dataset = len(all_eval_data)

        # Set the output dir and filepath
        output_dir = os.path.join(self.output_dir, task_name, self.model_name)
        os.makedirs(output_dir, exist_ok=True)
        output_fp = os.path.join(output_dir, output_fn + ".jsonl")
        if os.path.isfile(output_fp):
            if self.overwrite:
                all_results = []
                done_ids = set()
                self.logger.info(f"Results will be overwritten: {output_fp}")
            else:
                # Load the previous outputs to resume running
                all_results = DataIO.load_jsonl(output_fp, mode="r+", verbose=True)

                if isinstance(all_results, list) and len(all_results) > 0:
                    done_ids = set([_res["item_id_key"] for _res in all_results])
                    self.logger.info(f"Resume running (len done = {len(done_ids)}): {output_fp}")
                else:
                    all_results = []
                    done_ids = set()
                    self.logger.info(f"Results will be saved at: {output_fp}")
        else:
            all_results = []  # (list of instances)
            done_ids = set()
            self.logger.info(f"Results will be saved at: {output_fp}")

        resume_from = len(done_ids)
        if resume_from > 0:
            all_eval_data = all_eval_data[resume_from:]
            all_subset_names = all_subset_names[resume_from:]
        assert len(all_eval_data) == len(all_subset_names)
        len_dataset = len(all_eval_data)

        # Split the input list into mini batches
        assert isinstance(self.bsz, int) and self.bsz >= 1
        batches_data_items = [all_eval_data[_i: _i + self.bsz] for _i in range(0, len(all_eval_data), self.bsz)]
        batches_subset_names = [all_subset_names[_i: _i + self.bsz] for _i in range(0, len(all_subset_names), self.bsz)]
        num_batches = len(batches_data_items)

        item_idx = 0
        for batch_idx, cur_batch_data_items in enumerate(batches_data_items):
            assert isinstance(cur_batch_data_items, list) and len(cur_batch_data_items) > 0
            # cur_item_id_keys = [f"{item_idx + _idx}" for _idx in range(len(cur_batch_data_items))]
            # cur_all_done = all([_key in done_ids for _key in cur_item_id_keys])
            # if cur_all_done:  # Skip this batch if all items in this batch have been processed
            #     item_idx += len(cur_batch_data_items)
            #     continue

            cur_batch_subset_names = batches_subset_names[batch_idx]
            batch_ft_items = [eval_task_obj.get_prompt_ft_analysis(
                data_item=data_item, subset=subset) for data_item, subset in zip(
                cur_batch_data_items, cur_batch_subset_names)]
            batch_eval_items = [eval_task_obj.get_prompt_eval(
                data_item=data_item, subset=subset) for data_item, subset in zip(
                cur_batch_data_items, cur_batch_subset_names)]

            batch_prompts = []
            for ft_item in batch_ft_items:
                cur_dialog = ft_item["dialog"]
                if self.model_name.endswith("-base"):  # Base LLMs (text completion)
                    cur_prompt = "\n\n".join([str(_dialog["content"]).strip() for _dialog in cur_dialog]).strip()
                    cur_prompt += "\n\nAnswer:"
                else:  # Instruction-following LLMs
                    cur_prompt = self.tokenizer.apply_chat_template(
                        cur_dialog, tokenize=False, padding=False, return_tensors=None,
                        add_generation_prompt=True, enable_thinking=False,
                        # enable_thinking=self.thinking_mode,
                    )
                cur_prompt = str(cur_prompt).strip()
                if self.gen_method in PromptingMethods:
                    cur_prompt += "\n" + PromptingMethods[self.gen_method]
                cur_prompt = re.sub(r"[^\x00-\x7F]+", "", cur_prompt).strip()  # remove non-ASCII
                cur_prompt = cur_prompt.strip() + "\n"
                batch_prompts.append(cur_prompt)

            # Analyze the question
            gen_results = ModelUtils.open_model_gen(
                inputs=batch_prompts, model=self.model, tokenizer=self.tokenizer, need_tokenize=True,
                max_new_tokens=self.max_new_gen,
                # temperature=0.0, top_p=None, top_k=None,
                temperature=self.gen_temperature, top_p=self.top_p, top_k=self.top_k,
            )
            assert len(gen_results) == len(batch_prompts), len(gen_results)
            batch_analysis = [str(gen_result["output_text"]).strip() for gen_result in gen_results]

            break_flag = False
            for ft_item, ft_prompt, ft_analysis, eval_item in zip(
                    batch_ft_items, batch_prompts, batch_analysis, batch_eval_items):
                item_id_key = f"{item_idx}"
                # if item_id_key in done_ids:
                #     continue
                assert "dialog" in eval_item  # and "prompt" in eval_item
                ft_analysis = re.sub(r"[^\x00-\x7F]+", "", ft_analysis).strip()  # remove non-ASCII
                cur_gen_output = {
                    "method": self.gen_method,
                    "model": self.model_name,
                    "dataset_name": "",  # ds_name
                    "subset_name": "",  # subset
                    "split_name": "",   # split_name
                    "item_id_key": item_id_key,
                    "ds_id": "",  # ds_id
                    "batch_idx": batch_idx,
                    "item_idx": item_idx,
                    "prompt_ft": ft_prompt,
                    "dialog_ft": ft_item["dialog"],
                    "analysis": ft_analysis,
                    "pred_answer": "",
                    "output_text": "",
                    **eval_item
                }
                item_idx += 1

                all_results.append(cur_gen_output)
                # done_ids.add(item_id_key)
                if item_idx % show_cnt == 0:
                    cur_log_info = (f">>> Progress: [Batch (size={self.bsz}): {batch_idx + 1} / {num_batches}] "
                                    f"[Item: {item_idx} / {len_dataset}]")
                    if do_divide_conquer:
                        cur_log_info += f" [data_start_idx = {data_start_idx}] [data_end_idx = {data_end_idx}]"
                    if max_num_eval > 0:
                        cur_log_info += f" [max_num_eval = {max_num_eval}]"
                    if resume_from > 0:
                        cur_log_info += f" [resume_from = {resume_from}]"
                    if self.verbose:
                        self.logger.info(cur_log_info)

                    DataIO.save_jsonl(output_fp, all_results, mode="w", verbose=False)
                    gc.collect()
                    torch.cuda.empty_cache()

                if item_idx >= max_num_eval > 0:
                    break_flag = True
                    break

            if break_flag:
                break

        # Show logs and save the results
        if self.verbose:
            self.logger.info(
                f">>> Done. [Task: {task_name}] # = {len_dataset} "
                f"[max_num_eval = {max_num_eval}] [resume_from = {resume_from}] "
                f"[data_start_idx = {data_start_idx}] [data_end_idx = {data_end_idx}]")
        DataIO.save_jsonl(output_fp, all_results, mode="w", verbose=True)
        gc.collect()
        torch.cuda.empty_cache()
        self.logger.info(
            f">>> DONE ALL. model_name = {self.model_name}\n"
            f"gen_temperature: {self.gen_temperature}, batch_size: {self.bsz}"
        )


def main(
    task_name: Optional[str] = None,
    model_name: str = "llama3-8b",
    model_ckpt_dir: Optional[str] = None,
    cache_dir: Optional[str] = None,
    project_root_dir: Optional[str] = None,
    seed: int = 42,
    cuda: Optional[str] = None,
    bsz: int = 1,
    verbose: bool = False,
    output_dir: Optional[str] = None,
    max_new_gen: int = 2048,
    gen_temperature: float = 0.0,
    gen_top_p: Optional[float] = -1.0,
    gen_config: Optional[str] = None,
    gen_method: str = "da",
    num_total: int = -1,
    data_start_idx: int = -1,
    data_end_idx: int = -1,
    **kwargs
) -> None:
    """
    :param task_name: The name(s) of the evaluation task. (e.g., "mmlu", "bbh", or "mmlu,bbh")
    :param model_name: LLM name, e.g., "llama3-8b"
    :param model_ckpt_dir: Load checkpoint from this directory. (It overwrites the effect of `model_name`)
    :param cache_dir: The root directory of the cache.
    :param project_root_dir: The root directory of the current project/repo.
    :param seed: Random seed of all modules.
    :param cuda: To specify CUDA GPU devices, e.g., "0" OR "0,1". Default: None -- Use CPU or all available GPUs.
    :param bsz: The batch size.
    :param verbose: Verbose mode: show logs.
    :param output_dir: The path to the output file where the result metrics will be saved.
    :param max_new_gen: The maximum number of newly generated tokens.
    :param gen_temperature: The temperature used in LLM generation. Default: 0.
    :param gen_top_p: The Top-p ratio used in LLM generation.
    :param gen_config: The LLM generation configurations.
    :param gen_method: The method/baseline to use.
    :param num_total: The total number of instances for generation. -1 means using all.
    :param data_start_idx: The start index of the training data (divide and conquer).
    :param data_end_idx: The ending index of the training data (divide and conquer).

    :return: None.
    """

    timer_start = time.perf_counter()

    # Setup of the logger, CUDA gpus, and random seed
    logger = logger_setup("LM_Gen_FT_Analysis")
    cuda_dict = cuda_setup(cuda=cuda, logger=logger, verbose=verbose)
    random_setup(seed=seed, has_cuda=cuda_dict["has_cuda"])

    if isinstance(kwargs, dict):
        logger.info(f">>> Extra parameters in kwargs: {kwargs}\n")

    if isinstance(cache_dir, str) and os.path.isdir(cache_dir):
        os.environ["HF_HOME"] = cache_dir
    else:
        cache_dir = None

    # Parse the `gen_config` argument
    if (isinstance(gen_config, tuple) or isinstance(gen_config, list) or
            (isinstance(gen_config, str) and len(gen_config.strip()) > 0)):
        if isinstance(gen_config, tuple):
            gen_config = list(gen_config)
        if isinstance(gen_config, str):
            gen_config = gen_config.strip()
            gen_config = gen_config.split(",")
        # Note: For boolean parameters, "1" means True and "0" means False
        # overwrite: Whether to overwrite existing output files.
        # do_bf16: Whether to use BF16 precision mode to load models.
        # do_4bit: Whether to use 4bit quantization mode to load models.
        # debug: Debugging / developing mode.
        overwrite = str(gen_config[0]) == "1"
        do_bf16 = str(gen_config[1]) == "1"
        do_4bit = str(gen_config[2]) == "1"
        debug = str(gen_config[3]) == "1"
    else:
        # default: All "0"s
        gen_config = ["0" for _ in range(4)]
        overwrite = do_bf16 = do_4bit = debug = False
    logger.info(f">>> gen_config = {gen_config}: [overwrite: {overwrite}] "
                f"[do_bf16: {do_bf16}] [do_4bit: {do_4bit}] [debug: {debug}]")

    lm_gen_ft_analysis = LMGenFtAnalysis(
        verbose=verbose,
        logger=logger,
        cuda_dict=cuda_dict,
        seed=seed,
        cache_dir=cache_dir,
        project_root_dir=project_root_dir,
        model_name=model_name,
        model_ckpt_dir=model_ckpt_dir if (isinstance(model_ckpt_dir, str) and len(model_ckpt_dir) > 0) else None,
        bsz=max(int(bsz), 1),
        overwrite=overwrite,
        do_bf16=do_bf16,
        do_4bit=do_4bit,
        debug=debug,
        output_dir=output_dir,
        max_new_gen=max(int(max_new_gen), 128),
        gen_temperature=float(gen_temperature),
        top_p=float(gen_top_p) if float(gen_top_p) > 0.0 else None,
        top_k=None,
        gen_method=gen_method,
    )

    if isinstance(task_name, str):
        task_name = [task_name]

    if isinstance(task_name, tuple) or isinstance(task_name, list):
        for cur_task_name in task_name:
            cur_task_name = str(cur_task_name).strip()
            logger.info(f">>> <START> {cur_task_name}\n")
            lm_gen_ft_analysis.run_inference(
                task_name=cur_task_name, num_total=int(num_total),
                data_start_idx=int(data_start_idx), data_end_idx=int(data_end_idx),
            )
            logger.info(f">>> <END> {cur_task_name}\n\n\n")
    else:
        raise ValueError(f"--task_name should be a tuple/list/str: {task_name}")

    timer_end = time.perf_counter()
    total_sec = timer_end - timer_start
    logger.info(f"Total Running Time: {total_sec:.1f} sec ({total_sec / 60:.1f} min; {total_sec / 3600:.2f} h)")


if __name__ == "__main__":
    fire.Fire(main)
