#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import time
import random
from typing import Optional, List

import fire
import numpy as np

from utils.init_functions import logger_setup, cuda_setup, random_setup
from utils.models import ModelUtils
from utils.data_io import DataIO
from utils.prompting import PromptingMethods


class FtDataBuilder:

    def __init__(
            self,
            verbose: bool,
            logger,
            cuda_dict: Optional[dict] = None,
            seed: int = 42,
            cache_dir: Optional[str] = None,
            project_root_dir: Optional[str] = None,
            model_name: str = "llama3-8b",
            debug: bool = False,
            results_dir: Optional[str] = None,
            save_dir: Optional[str] = None,
            training_data_type: str = "raw",
    ):
        self.verbose = verbose
        self.logger = logger
        self.cuda_dict = cuda_dict
        self.seed = seed
        self.model_name = model_name
        self.debug = debug

        if isinstance(project_root_dir, str) and os.path.isdir(project_root_dir):
            self.project_root_dir = project_root_dir
        else:
            self.project_root_dir = os.getcwd()
        assert os.path.isdir(self.project_root_dir)

        self.results_dir = results_dir
        if not isinstance(save_dir, str):
            save_dir = os.path.join(project_root_dir, "data/ft_data")
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.training_data_type = training_data_type

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

        self.tokenizer = ModelUtils.initialize_tokenizer_hf(
            model_name=model_name, cache_dir=cache_dir, padding_side="left", truncation_side="left",
            verbose=verbose, model_ckpt_dir=None)

    def seq_len_stat(self, input_list: list) -> List[int]:
        if not isinstance(input_list, list) or len(input_list) == 0:
            return []

        # Tokenization
        input_tokens = [self.tokenizer.apply_chat_template(
            _item["messages"], return_tensors="pt", tokenize=True, return_dict=True) for _item in input_list]
        input_seq_len = [int(_tok["input_ids"].size(-1)) for _tok in input_tokens]

        # Do statistics
        self.logger.info(f">>> Data Statistics: total # token = {np.sum(input_seq_len):d}")
        self.logger.info(f">>> Data Statistics: seq len max = {np.max(input_seq_len):d}")
        self.logger.info(f">>> Data Statistics: seq len min = {np.min(input_seq_len):d}")
        self.logger.info(f">>> Data Statistics: seq len avg = {np.mean(input_seq_len):.1f}")
        self.logger.info(f">>> Data Statistics: seq len std = {np.std(input_seq_len):.1f}")
        quantile_q = [0.25, 0.5, 0.75, 0.9, 0.99, 0.999, 0.9999]
        quantiles = np.quantile(input_seq_len, q=quantile_q)
        quantiles = [float(np.round(quantile, 1)) for quantile in quantiles]
        self.logger.info(f">>> Data Statistics: seq len quantiles ({quantile_q}) = {quantiles}")

        del input_tokens
        return input_seq_len

    def build_ft_data(
            self,
            task_name: str,
            do_save: bool = True,
            do_stat: bool = False,
            num_total: int = -1,
            valid_ratio: float = 0.01,
            num_valid_min: int = 100,
            do_data_merge: bool = False,
    ) -> Optional[dict]:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        assert valid_ratio > 0 and num_valid_min > 0

        assert isinstance(self.results_dir, str) and os.path.isdir(self.results_dir), ">>> No --results_dir"
        if do_data_merge:
            results_temp = []
            cur_results_dir = os.path.join(self.results_dir, task_name, self.model_name)
            assert os.path.isdir(cur_results_dir), f">>> No results dir: {cur_results_dir}"
            cur_results_fn_list = os.listdir(cur_results_dir)
            cur_results_fn_list = [_fn for _fn in cur_results_fn_list if
                                   _fn.startswith("results_gen--") and _fn.endswith(".jsonl")]
            for cur_results_fn in cur_results_fn_list:
                # Process the indices
                cur_results_fn_idx_raw = cur_results_fn[len("results_gen--"): -len(".jsonl")]
                if "__" not in cur_results_fn_idx_raw:
                    continue
                cur_results_fn_idx_list = cur_results_fn_idx_raw.split("__")
                if not isinstance(cur_results_fn_idx_list, list) or len(cur_results_fn_idx_list) != 2:
                    continue
                data_start_idx, data_end_idx = int(cur_results_fn_idx_list[0]), int(cur_results_fn_idx_list[1])

                # Get the data
                cur_results_fp = os.path.join(cur_results_dir, cur_results_fn)
                assert os.path.isfile(cur_results_fp), f">>> No result file: {cur_results_fp}"
                cur_results = DataIO.load_jsonl(cur_results_fp, mode="r", verbose=True)
                results_temp.append([data_start_idx, cur_results])
            # Sort the data lists by the starting indices
            results_temp.sort(key=lambda x: x[0], reverse=False)
            results = []
            for res_temp in results_temp:
                results.extend(res_temp[1])
        else:
            results_fp = os.path.join(self.results_dir, task_name, self.model_name, "results_gen.jsonl")
            # results_fp = os.path.join(self.results_dir, task_name, self.model_name, "results_gen--eval.json")
            assert os.path.isfile(results_fp), f">>> No result file: {results_fp}"
            results = DataIO.load_jsonl(results_fp, mode="r", verbose=True)
        assert isinstance(results, list) and len(results) > 0
        if len(results) > num_total > 0:
            results = results[:num_total]
        num_results = len(results)

        # Deal with each task (and sub-tasks)
        self.logger.info(f">>> Task: {task_name} [num_results = {num_results}]")
        show_cnt = 1000
        ft_data_all = []
        for item_idx, item_dict in enumerate(results):
            # Get common information
            cur_method = item_dict["method"]
            cur_model = item_dict["model"]
            cur_dataset_name = item_dict["dataset_name"]
            cur_subset_name = item_dict["subset_name"]
            cur_split_name = item_dict["split_name"]
            cur_item_id_key = item_dict["item_id_key"]
            cur_ds_id = item_dict["ds_id"]
            cur_batch_idx = item_dict.get("batch_idx", 0)
            cur_item_idx = item_dict["item_idx"]
            cur_user_dialog_ft = item_dict.get("dialog_ft", "")  # the dialog for constructing the analysis
            cur_user_dialog_eval = item_dict["dialog"]  # the user prompt for eval and fine-tuning
            cur_answers = item_dict["answers"]  # valid answers: List[str]
            assert isinstance(cur_answers, list) and len(cur_answers) > 0, type(cur_answers)
            cur_info = item_dict["info"]
            assert isinstance(cur_info, dict)
            cur_info["context"] = ""
            common_info_dict = {
                "method": cur_method, "model": cur_model,
                "dataset_name": cur_dataset_name, "subset_name": cur_subset_name,
                "split_name": cur_split_name, "item_id_key": cur_item_id_key, "ds_id": cur_ds_id,
                "batch_idx": cur_batch_idx, "item_idx": cur_item_idx,
                "dialog_ft": cur_user_dialog_ft, "dialog": cur_user_dialog_eval,
                "answers": cur_answers, "info": cur_info,
            }

            # Construct the complete fine-tuning text
            correct_answer = cur_answers[0]
            if self.training_data_type == "raw":
                cur_analysis = ""
                cur_prompting = ""
                dialog_training = cur_user_dialog_eval + [
                    {"role": "assistant", "content": f"Final Answer: {correct_answer}".strip() + "\n"}]
            elif self.training_data_type == "da" or self.training_data_type == "ia":
                cur_analysis = item_dict["analysis"]
                cur_analysis = re.sub(r"[^\x00-\x7F]+", "", cur_analysis).strip()  # remove non-ASCII

                assert self.training_data_type in PromptingMethods
                cur_prompting = str(PromptingMethods[self.training_data_type]).strip()
                cur_prompting = re.sub(r"[^\x00-\x7F]+", "", cur_prompting).strip()  # remove non-ASCII

                if cur_info["task_type"] == "code":
                    dialog_training = cur_user_dialog_eval + [{"role": "assistant", "content": f"""
{cur_prompting}
{cur_analysis}
                    """.strip() + "\n"}]
                else:
                    dialog_training = cur_user_dialog_eval + [{"role": "assistant", "content": f"""
{cur_prompting}
{cur_analysis}

Final Answer: {correct_answer}
                    """.strip() + "\n"}]
            else:
                raise ValueError(f">>> !!! >>> Unknown training_data_type: {self.training_data_type}")

            analysis_info_dict = {
                "analysis": cur_analysis, "prompting": cur_prompting,
            }

            ft_data_all.append({
                "messages": dialog_training,
                "common_info_dict": common_info_dict,
                "analysis_info_dict": analysis_info_dict,
            })

            if (item_idx + 1) % show_cnt == 0:
                self.logger.info(f">>> >>> Progress: [{item_idx + 1} / {num_results}]")

        # Done constructing FT data; Obtain random samples from the raw validation set
        assert len(ft_data_all) == num_results
        ft_data_training, ft_data_validation = [], []
        random.seed(self.seed)
        valid_num = max(num_valid_min, int(num_results * valid_ratio))
        assert 0 < valid_num < num_results
        all_indices = list(range(num_results))
        valid_indices = random.sample(all_indices, valid_num)
        valid_indices_set = set(valid_indices)
        for idx, item in enumerate(ft_data_all):
            if idx in valid_indices_set:
                ft_data_validation.append(item)
            else:
                ft_data_training.append(item)

        self.logger.info(f">>> Done processing. [len(ft_data_training) = {len(ft_data_training)}] "
                         f"[len(ft_data_validation) = {len(ft_data_validation)}]")

        # Statistics of the tokenized data
        if do_stat:
            self.logger.info(f">>> seq_len_stat(ft_data_training):")
            self.seq_len_stat(ft_data_training)
            self.logger.info(f">>> seq_len_stat(ft_data_validation):")
            self.seq_len_stat(ft_data_validation)

        # Now, save the data
        if do_save:
            assert os.path.isdir(self.save_dir)
            cur_save_dir = os.path.join(self.save_dir, self.model_name, task_name)
            os.makedirs(cur_save_dir, exist_ok=True)

            DataIO.save_jsonl(os.path.join(cur_save_dir, f"ft_data_{self.training_data_type}_training.jsonl"),
                              ft_data_training, mode="w", verbose=True)
            DataIO.save_jsonl(os.path.join(cur_save_dir, f"ft_data_{self.training_data_type}_validation.jsonl"),
                              ft_data_validation, mode="w", verbose=True)

            self.logger.info(f">>> Done saving. cur_save_dir: {cur_save_dir}")

        return None


def main(
    builder_task: int = 1,
    task_name="",
    model_name: str = "llama3-8b",
    cache_dir: Optional[str] = None,
    project_root_dir: Optional[str] = None,
    seed: int = 42,
    cuda: Optional[str] = None,
    verbose: bool = False,
    debug: bool = False,
    results_dir: Optional[str] = None,
    save_dir: Optional[str] = None,
    training_data_type: str = "raw",
    num_total: int = -1,
    valid_ratio: float = 0.01,
    num_valid_min: int = 100,
    do_data_merge: bool = False,
    **kwargs
) -> None:
    """
    :param builder_task: The process for the builder to run.
    :param task_name: The name(s) of the evaluation task. (e.g., "mmlu_training")
    :param model_name: LLM name, e.g., "llama3-8b"
    :param cache_dir: The root directory of the cache.
    :param project_root_dir: The root directory of the current project/repo.
    :param seed: Random seed of all modules.
    :param cuda: To specify CUDA GPU devices, e.g., "0" OR "0,1". Default: None -- Use CPU or all available GPUs.
    :param verbose: Verbose mode: show logs.
    :param debug: Debugging / developing mode.
    :param results_dir: The dir path to the evaluated results that analyzed the questions.
    :param save_dir: The dir path to save the results.
    :param training_data_type: The training data type: "raw", "da", or "ia".
    :param num_total: The total number of instances for fine-tuning (and validation). -1 means using all.
    :param valid_ratio: The validation ratio to use.
    :param num_valid_min: The minimum number of instances for validation.
    :param do_data_merge: Whether to merge the training data files (divide and conquer).

    :return: None.
    """

    timer_start = time.perf_counter()

    # Setup of the logger, CUDA gpus, and random seed
    logger = logger_setup("Build-FT-Data")
    cuda_dict = cuda_setup(cuda=cuda, logger=logger, verbose=verbose)
    random_setup(seed=seed, has_cuda=cuda_dict["has_cuda"])

    if isinstance(kwargs, dict):
        logger.info(f">>> Extra parameters in kwargs: {kwargs}")
    logger.info(f">>> cuda_dict: {cuda_dict}")

    if isinstance(cache_dir, str) and os.path.isdir(cache_dir):
        os.environ["HF_HOME"] = cache_dir
    else:
        cache_dir = None

    ft_data_builder = FtDataBuilder(
        verbose=verbose,
        logger=logger,
        cuda_dict=cuda_dict,
        seed=seed,
        cache_dir=cache_dir,
        project_root_dir=project_root_dir,
        model_name=model_name,
        debug=debug,
        results_dir=results_dir,
        save_dir=save_dir,
        training_data_type=training_data_type,
    )

    do_stat = "do_stat" in kwargs

    builder_task = int(builder_task)
    if isinstance(task_name, str):
        task_name = [task_name]
    if isinstance(task_name, tuple) or isinstance(task_name, list):
        for cur_task_name in task_name:
            cur_task_name = str(cur_task_name).strip()
            logger.info(f">>> <START> [builder_task = {builder_task}] cur_task_name: {cur_task_name}\n")
            match builder_task:
                case 1:
                    ft_data_builder.build_ft_data(
                        task_name=cur_task_name, do_save=True, do_stat=do_stat,
                        num_total=num_total, valid_ratio=valid_ratio, num_valid_min=num_valid_min,
                        do_data_merge=do_data_merge,
                    )
                case _:
                    raise ValueError(f"ValueError: builder_task = {builder_task}")
            logger.info(f">>> <END> [builder_task = {builder_task}] cur_task_name: {cur_task_name}\n\n\n")
    else:
        raise ValueError(f"--task_name should be a tuple/list/str: {task_name}")

    timer_end = time.perf_counter()
    total_sec = timer_end - timer_start
    logger.info(f"Total Running Time: {total_sec:.1f} sec ({total_sec / 60:.1f} min; {total_sec / 3600:.2f} h)")


if __name__ == "__main__":
    fire.Fire(main)
