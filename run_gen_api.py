#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import gc
import time
from typing import Optional

import fire
from datasets import Dataset

from tasks.tasks_utils import TASK_CLASS_DICT

from utils.models import ModelUtils
from utils.data_io import DataIO
from utils.prompting import PromptingMethods

from utils.init_functions import logger_setup, cuda_setup, random_setup


class GenAI:

    def __init__(
            self,
            verbose: bool,
            logger,
            cuda_dict: Optional[dict] = None,
            seed: int = 42,
            eval_task_name: Optional[str] = None,
            cache_dir: Optional[str] = None,
            project_root_dir: Optional[str] = None,
            genai_model: str = "gpt-5.2-nano",
            genai_api_key: Optional[str] = None,
            debug: bool = False,
            overwrite: bool = False,
            output_dir: Optional[str] = None,
            max_new_gen: int = 2048,
            gen_temperature: float = 1.0,
            gen_method: str = "da",
    ):
        self.verbose = verbose
        self.logger = logger
        self.cuda_dict = cuda_dict
        self.seed = seed
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

        self.eval_task_name = eval_task_name
        self.output_dir = output_dir
        self.max_new_gen = max_new_gen
        self.gen_temperature = gen_temperature
        self.gen_method = gen_method

        # GenAI settings
        self.genai_model = genai_model
        if isinstance(genai_api_key, str) and len(genai_api_key) > 0:
            self.genai_api_key = genai_api_key
        else:
            if "gpt" in genai_model:
                self.genai_api_key = os.getenv("OPENAI_API_KEY")
            elif "gemini" in genai_model:
                self.genai_api_key = os.getenv("GEMINI_API_KEY")
            elif "claude" in genai_model:
                self.genai_api_key = os.getenv("ANTHROPIC_API_KEY")
            else:
                raise ValueError(f">>> Unsupported genai model: {genai_model}")

    def run_inference(
            self,
            task_name: str,
            show_cnt: int = 10,
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
            model_name=self.genai_model,
        )

        self.logger.info(f">>> Evaluation Task: {task_name}")
        task_dict = eval_task_obj.load_task()
        dataset_list = task_dict["eval"] if "eval" in task_dict else []
        num_subtasks = len(dataset_list)
        if num_subtasks == 0:
            self.logger.info(f">>> SKIP (No Eval set) [Task: {task_name}]")

        # Deal with each task (and sub-tasks)
        all_eval_data = dict()
        for dataset_dict in dataset_list:
            cur_data_items = []

            ds_name, subset_name = dataset_dict["dataset_name"], dataset_dict["subset_name"]
            split_name, eval_dataset = dataset_dict["split_name"], dataset_dict["dataset"]
            assert isinstance(eval_dataset, Dataset) or isinstance(eval_dataset, list)
            assert isinstance(ds_name, str) and len(ds_name) > 0
            if isinstance(subset_name, str) and len(subset_name) > 0:
                ds_id = f"{ds_name}---{subset_name}"
            else:
                ds_id = ds_name

            if "options" in dataset_dict:
                ds_options = list(dataset_dict["options"])
            else:
                ds_options = []

            # Construct all the input prompts for LLM generation
            skip_cnt = 0
            for idx, data_item in enumerate(eval_dataset):
                assert isinstance(data_item, dict)
                data_item["__ds_options"] = ds_options

                cur_eval_prompt = eval_task_obj.get_prompt_eval(
                    data_item=data_item, ds_name=ds_name, subset=subset_name)

                if not isinstance(cur_eval_prompt, dict) or len(cur_eval_prompt) == 0:
                    skip_cnt += 1
                else:
                    cur_data_items.append(data_item)

            all_eval_data[ds_id] = {
                "data_items": cur_data_items,
                "ds_info": {
                    "ds_id": ds_id,
                    "dataset_name": ds_name,
                    "subset_name": subset_name,
                    "split_name": split_name,
                    "ds_options": ds_options,
                    # "eval_dataset": eval_dataset,
                },
            }

        # Set the output dir and filepath
        output_dir = os.path.join(self.output_dir, task_name, self.genai_model)
        os.makedirs(output_dir, exist_ok=True)
        output_fp = os.path.join(output_dir, "results_gen.jsonl")
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
            all_results = []
            done_ids = set()
            self.logger.info(f"Results will be saved at: {output_fp}")

        resume_from = len(done_ids)
        if resume_from > 0:
            all_eval_data = all_eval_data[resume_from:]

        # Run generation for each item in each ds (subtask)
        for ds_id, cur_eval_data in all_eval_data.items():
            cur_ds_done_cnt = 0

            cur_data_items = list(cur_eval_data["data_items"])
            assert len(cur_data_items) > 0, len(cur_data_items)
            len_dataset = len(cur_data_items)
            ds_info = dict(cur_eval_data["ds_info"])
            ds_name, subset, split_name = ds_info["dataset_name"], ds_info["subset_name"], ds_info["split_name"]

            if self.verbose:
                self.logger.info(
                    f">>> [Dataset: {ds_id}] [Eval: {split_name}] # = {len_dataset}")

            # Run generation
            for item_idx, cur_data_item in enumerate(cur_data_items):
                item_id_key = f"{ds_id}---{item_idx}"
                if item_id_key in done_ids:
                    continue

                cur_prompt_dict = eval_task_obj.get_prompt_eval(
                    data_item=cur_data_item, ds_name=ds_name, subset=subset)
                assert isinstance(cur_prompt_dict, dict)

                # Obtain the system prompt and user prompt for GPT
                dialog = cur_prompt_dict["dialog"]
                assert isinstance(dialog, list) and len(dialog) == 1
                dialog_user = dialog[0]
                assert isinstance(dialog_user, dict) and "role" in dialog_user and dialog_user["role"] == "user"
                assert "content" in dialog_user
                system_prompt = ("You are a helpful assistant. You should first present your analysis and then "
                                 "give your final answer (starting with \"Final Answer:\") at the end of your output.")
                user_prompt = str(dialog_user["content"]).strip()
                if self.gen_method in PromptingMethods:
                    cur_prompting = PromptingMethods[self.gen_method].strip()
                    if len(cur_prompting) > 0:
                        system_prompt += " " + cur_prompting.replace("Let's", "In your analysis, you must")
                        # user_prompt += "\n\n" + cur_prompting
                        user_prompt += "\n\n" + cur_prompting.replace("Let's", "You must")
                user_prompt = user_prompt.strip() + "\n"

                # Send GenAI request
                if "gpt" in self.genai_model:
                    gpt_input_messages = [
                        {"role": "developer", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ]
                    response = ModelUtils.call_gpt(
                        openai_model_name=self.genai_model, messages=gpt_input_messages,
                        openai_api_key=self.genai_api_key,
                    )  # format_class=format_class,
                    res_message = response.choices[0].message
                    refusal = res_message.refusal
                    if refusal:  # If the model refuses to respond, get the refusal message
                        self.logger.info(f">>> !!! >>> The model refuses to respond: {refusal}")
                    output_text = str(res_message.content).strip()
                elif "gemini" in self.genai_model:
                    gemini_input_messages = [system_prompt, user_prompt]
                    response = ModelUtils.call_gemini(
                        gemini_model_name=self.genai_model, messages=gemini_input_messages,
                        gemini_api_key=self.genai_api_key,
                    )
                    res_message = response.text
                    output_text = str(res_message).strip()
                elif "claude" in self.genai_model:
                    claude_input_messages = [system_prompt, user_prompt]
                    response = ModelUtils.call_claude(
                        claude_model_name=self.genai_model, messages=claude_input_messages,
                        claude_api_key=self.genai_api_key,
                        max_output_tokens=self.max_new_gen,
                    )
                    try:
                        res_message = response.content[0].text
                    except Exception as e:
                        self.logger.info(e)
                        res_message = "NONE"
                    output_text = str(res_message).strip()
                else:
                    raise ValueError(f">>> Unsupported genai model: {self.genai_model}")

                cur_gen_output = {
                    "method": self.gen_method,
                    "model": self.genai_model,
                    "dataset_name": ds_name,
                    "subset_name": subset,
                    "split_name": split_name,
                    "item_id_key": item_id_key,
                    "ds_id": ds_id,
                    "item_idx": item_idx,
                    "prompt": user_prompt,
                    "analysis": output_text,
                    "pred_answer": output_text,
                    "output_text": output_text,
                    **cur_prompt_dict
                }

                all_results.append(cur_gen_output)
                done_ids.add(item_id_key)
                cur_ds_done_cnt += 1
                if cur_ds_done_cnt % show_cnt == 0:
                    cur_log_info = f">>> Progress: [{ds_id}] [Item: {cur_ds_done_cnt} / {len_dataset}]"
                    if resume_from > 0:
                        cur_log_info += f" [resume_from = {resume_from}]"
                    if self.verbose:
                        self.logger.info(cur_log_info)

                    DataIO.save_jsonl(output_fp, all_results, mode="w", verbose=False)

            # Show logs and save the results of the current ds (subtask)
            if self.verbose:
                self.logger.info(
                    f">>> Done. [Dataset: {ds_id}] [Eval: {split_name}] # = {len_dataset}")
            DataIO.save_jsonl(output_fp, all_results, mode="w", verbose=True)

        # Show logs and save all results
        DataIO.save_jsonl(output_fp, all_results, mode="w", verbose=True)
        gc.collect()
        self.logger.info(
            f">>> DONE ALL. genai_model = {self.genai_model}\n"
            f"gen_temperature: {self.gen_temperature}"
        )


def main(
    task_name: Optional[str] = None,
    genai_model: str = "gpt-5.2-nano",
    genai_api_key: Optional[str] = None,
    cache_dir: Optional[str] = None,
    project_root_dir: Optional[str] = None,
    seed: int = 42,
    cuda: Optional[str] = None,
    verbose: bool = False,
    debug: bool = False,
    overwrite: bool = False,
    output_dir: Optional[str] = None,
    max_new_gen: int = 2048,
    gen_temperature: float = 1.0,
    gen_method: str = "da",
    **kwargs
) -> None:
    """
    :param task_name: The name(s) of the evaluation task. (e.g., "mmlu", "bbh", or "mmlu,bbh")
    :param genai_model: e.g., "gpt-5.2", "claude-opus-4.6", or "gemini-3-flash-preview"
    :param genai_api_key: your valid API Key (OpenAI or Gemini). Default: env var ${OPENAI_API_KEY} or ${GEMINI_API_KEY}
    :param cache_dir: The root directory of the cache.
    :param project_root_dir: The root directory of the current project/repo.
    :param seed: Random seed of all modules.
    :param cuda: To specify CUDA GPU devices, e.g., "0" OR "0,1". Default: None -- Use CPU or all available GPUs.
    :param verbose: Verbose mode: show logs.
    :param debug: Debugging / developing mode.
    :param overwrite: Whether to overwrite existing output files.
    :param output_dir: The path to the output file where the result metrics will be saved.
    :param max_new_gen: The maximum number of newly generated tokens.
    :param gen_temperature: The temperature used in LLM generation. Default: 1.0
    :param gen_method: The method/baseline to use.

    :return: None.
    """

    timer_start = time.perf_counter()

    # Setup of the logger, CUDA gpus, and random seed
    logger = logger_setup("LM_Gen_API")
    cuda_dict = cuda_setup(cuda=cuda, logger=logger, verbose=verbose)
    random_setup(seed=seed, has_cuda=cuda_dict["has_cuda"])

    if isinstance(kwargs, dict):
        logger.info(f">>> Extra parameters in kwargs: {kwargs}")
    logger.info(f">>> cuda_dict: {cuda_dict}")

    api_gen = GenAI(
        verbose=verbose,
        logger=logger,
        cuda_dict=cuda_dict,
        seed=seed,
        cache_dir=cache_dir,
        project_root_dir=project_root_dir,
        genai_model=genai_model,
        genai_api_key=genai_api_key,
        debug=debug,
        overwrite=overwrite,
        output_dir=output_dir,
        max_new_gen=max(int(max_new_gen), 128),
        gen_temperature=float(gen_temperature),
        gen_method=gen_method,
    )

    if isinstance(task_name, str):
        task_name = [task_name]

    if isinstance(task_name, tuple) or isinstance(task_name, list):
        for cur_task_name in task_name:
            cur_task_name = str(cur_task_name).strip()
            logger.info(f">>> <START> {cur_task_name}\n")
            api_gen.run_inference(task_name=cur_task_name)
            logger.info(f">>> <END> {cur_task_name}\n\n\n")
    else:
        raise ValueError(f"--task_name should be a tuple/list/str: {task_name}")

    timer_end = time.perf_counter()
    total_sec = timer_end - timer_start
    logger.info(f"Total Running Time: {total_sec:.1f} sec ({total_sec / 60:.1f} min; {total_sec / 3600:.2f} h)")


if __name__ == "__main__":
    fire.Fire(main)
