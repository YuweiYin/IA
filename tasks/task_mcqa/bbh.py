# -*- coding: utf-8 -*-

import os
import re
import time
from typing import Optional, Dict, Any

from datasets import load_dataset

from tasks import TaskManager
# from tasks.task_mcqa import SYSTEM_PROMPT_MCQA_GEN


class TaskBbh(TaskManager):

    def __init__(
            self,
            verbose: bool,
            logger,
            cache_dir: Optional[str] = None,
            project_root_dir: Optional[str] = None,
            model_name: str = "llama3-8b",
    ):
        super().__init__(verbose, logger, cache_dir, project_root_dir, model_name)

        self.task_name = "bbh"
        self.task_info = {
            "task_name": self.task_name,
            "task_type": "mcqa",
            "eval": [
                # ["lukaemon/bbh", "word_sorting", "test"],  # Test = 250 (not MCQA)
                ["lukaemon/bbh", "web_of_lies", "test"],  # Test = 250
                ["lukaemon/bbh", "tracking_shuffled_objects_three_objects", "test"],  # Test = 250
                ["lukaemon/bbh", "tracking_shuffled_objects_seven_objects", "test"],  # Test = 250
                ["lukaemon/bbh", "tracking_shuffled_objects_five_objects", "test"],  # Test = 250
                ["lukaemon/bbh", "temporal_sequences", "test"],  # Test = 250
                ["lukaemon/bbh", "sports_understanding", "test"],  # Test = 250
                ["lukaemon/bbh", "snarks", "test"],  # Test = 178
                ["lukaemon/bbh", "salient_translation_error_detection", "test"],  # Test = 250
                ["lukaemon/bbh", "ruin_names", "test"],  # Test = 250
                ["lukaemon/bbh", "reasoning_about_colored_objects", "test"],  # Test = 250
                ["lukaemon/bbh", "penguins_in_a_table", "test"],  # Test = 146
                # ["lukaemon/bbh", "object_counting", "test"],  # Test = 250 (not MCQA)
                ["lukaemon/bbh", "navigate", "test"],  # Test = 250
                # ["lukaemon/bbh", "multistep_arithmetic_two", "test"],  # Test = 250 (not MCQA)
                ["lukaemon/bbh", "movie_recommendation", "test"],  # Test = 250
                ["lukaemon/bbh", "logical_deduction_three_objects", "test"],  # Test = 250
                ["lukaemon/bbh", "logical_deduction_seven_objects", "test"],  # Test = 250
                ["lukaemon/bbh", "logical_deduction_five_objects", "test"],  # Test = 250
                ["lukaemon/bbh", "hyperbaton", "test"],  # Test = 250
                ["lukaemon/bbh", "geometric_shapes", "test"],  # Test = 250
                ["lukaemon/bbh", "formal_fallacies", "test"],  # Test = 250
                # ["lukaemon/bbh", "dyck_languages", "test"],  # Test = 250 (not MCQA)
                ["lukaemon/bbh", "disambiguation_qa", "test"],  # Test = 250
                ["lukaemon/bbh", "date_understanding", "test"],  # Test = 250
                ["lukaemon/bbh", "causal_judgement", "test"],  # Test = 187
                ["lukaemon/bbh", "boolean_expressions", "test"],  # Test = 250
            ],
            "num_to_eval": -1,  # The # of items per subtask in "eval" set (if exists) for evaluation. -1: use all
        }
        # >>> Token Stat: >>> #Sub-Tasks = 23; #Items = 5511; avg_len_token: 273.5; std_len_token: 73.4; skip_cnt = 0

        self.open_qa_subtasks = {"word_sorting", "object_counting", "dyck_languages", "multistep_arithmetic_two"}
        self.options = BBH_OPTIONS

    def load_task(
            self,
    ) -> Dict[str, Any]:
        self.logger.info(f">>> [task_name: {self.task_name}]")
        assert isinstance(self.task_info, dict)
        dataset = {
            "task_name": self.task_name,
            "task_info": self.task_info,
        }

        # for task_split in ["train", "valid", "eval"]:
        for task_split in ["eval"]:
            if task_split not in self.task_info or len(self.task_info[task_split]) == 0:
                self.logger.info(f">>> [Skip] `{task_split}` split does not exist.")
                dataset[task_split] = []
                continue

            self.logger.info(f">>> task_split: {task_split}")
            cur_data = []
            ds_list = self.task_info[task_split]
            assert isinstance(ds_list, list) and len(ds_list) > 0

            option_pattern = re.compile(r"^\([A-Z]\)$")
            for ds_info in ds_list:
                time.sleep(1)
                assert isinstance(ds_info, list) and len(ds_info) == 3

                try:  # Load the subtask
                    dataset_name, subset_name, split_name = ds_info[0], ds_info[1], ds_info[2]
                    cur_ds = load_dataset(
                        dataset_name, subset_name, split=split_name,
                        cache_dir=os.path.join(self.cache_dir, "datasets"),
                        # trust_remote_code=True,
                    )

                    if subset_name not in self.open_qa_subtasks:
                        # For multiple-choice QA subtasks, obtain all possible options ("target")
                        options = list(set(list(cur_ds["target"])))
                        options = [str(_op).strip() for _op in options]
                        # ensure the format of option labels (other: Yes/No, True/False, valid/invalid)
                        # Note: this leads to certain evaluation acc=0 (can not match target and option)
                        if "(A)" in options:
                            options = [_op for _op in options if re.match(option_pattern, _op) is not None]
                        options.sort()
                    else:
                        options = None

                    ds_dict = {
                        "dataset_name": dataset_name,
                        "subset_name": subset_name,
                        "split_name": split_name,
                        "dataset": cur_ds,
                        "options": options,
                    }

                    self.logger.info(f">>> [dataset: {dataset_name} --- {subset_name}] "
                                     f"split: {split_name} [# Items = {len(cur_ds)}]")

                    cur_data.append(ds_dict)
                except Exception as e:
                    if self.verbose:
                        self.logger.info(f">>> Exception: {e}")

            dataset[task_split] = cur_data

        return dataset

    def get_prompt_eval(
            self,
            data_item,
            **kwargs
    ) -> Dict[str, Any]:
        # dialog_sys = [{"role": "system", "content": SYSTEM_PROMPT_MCQA_GEN}]
        dialog_sys = []

        # Process data ["input", "target"]
        question = str(data_item["input"]).strip()
        answer = str(data_item["target"]).strip()
        answers = [answer]

        # Get all options for the "target" in each multiple-choice QA subtask
        assert "subset" in kwargs
        subset = kwargs["subset"]
        if isinstance(subset, str) and subset in self.options and answer in list(self.options[subset]):
            subject = subset
            answer_options = list(self.options[subset])
            answer_options.sort()
            pos_a = answer_options.index(answer)

            # Set the main prompt (zero-shot)
            options_str = "\n".join([f"{_op}" for _op in answer_options])
            dialog_user = [{"role": "user", "content": f"""
Answer the following question by selecting an option:
{question}
{options_str}
            """.strip() + "\n"}]
        else:  # Not a Multiple-choice QA subtask (ignored)
            subject = None
            answer_options = None
            options_str = ""
            pos_a = None
            dialog_user = [{"role": "user", "content": f"""
Answer the following question:
{question}
            """.strip() + "\n"}]

        dialog = dialog_sys + dialog_user

        # Set the result dict
        if len(answers) == 0:
            return {}  # invalid instance, skip it
        result_dict = {
            "dialog": dialog,
            "answers": answers,
            "info": {
                "task_type": "mcqa",
                "context": "",
                "question": question,
                "options": answer_options,
                "options_str": options_str,
                "ans_idx": pos_a,
                "subject": subject,
            }
        }
        return result_dict


BBH_OPTIONS = {
    "web_of_lies": [
        "No",
        "Yes"
    ],
    "tracking_shuffled_objects_three_objects": [
        "(A)",
        "(B)",
        "(C)"
    ],
    "tracking_shuffled_objects_seven_objects": [
        "(A)",
        "(B)",
        "(C)",
        "(D)",
        "(E)",
        "(F)",
        "(G)"
    ],
    "tracking_shuffled_objects_five_objects": [
        "(A)",
        "(B)",
        "(C)",
        "(D)",
        "(E)"
    ],
    "temporal_sequences": [
        "(A)",
        "(B)",
        "(C)",
        "(D)"
    ],
    "sports_understanding": [
        "no",
        "yes"
    ],
    "snarks": [
        "(A)",
        "(B)"
    ],
    "salient_translation_error_detection": [
        "(A)",
        "(B)",
        "(C)",
        "(D)",
        "(E)",
        "(F)"
    ],
    "ruin_names": [
        "(A)",
        "(B)",
        "(C)",
        "(D)"
    ],
    "reasoning_about_colored_objects": [
        "(A)",
        "(B)",
        "(C)",
        "(D)",
        "(E)",
        "(F)",
        "(G)",
        "(H)",
        "(I)",
        "(J)",
        "(K)",
        "(L)",
        "(M)",
        "(N)",
        "(O)",
        "(P)",
        "(Q)",
        "(R)"
    ],
    "penguins_in_a_table": [
        "(A)",
        "(B)",
        "(C)",
        "(D)",
        "(E)"
    ],
    "navigate": [
        "No",
        "Yes"
    ],
    "movie_recommendation": [
        "(A)",
        "(B)",
        "(C)",
        "(D)",
        "(E)"
    ],
    "logical_deduction_three_objects": [
        "(A)",
        "(B)",
        "(C)"
    ],
    "logical_deduction_seven_objects": [
        "(A)",
        "(B)",
        "(C)",
        "(D)",
        "(E)",
        "(F)",
        "(G)"
    ],
    "logical_deduction_five_objects": [
        "(A)",
        "(B)",
        "(C)",
        "(D)",
        "(E)"
    ],
    "hyperbaton": [
        "(A)",
        "(B)"
    ],
    "geometric_shapes": [
        "(B)",
        "(C)",
        "(D)",
        "(E)",
        "(F)",
        "(G)",
        "(I)",
        "(J)",
        "(K)"
    ],
    "formal_fallacies": [
        "invalid",
        "valid"
    ],
    "disambiguation_qa": [
        "(A)",
        "(B)",
        "(C)"
    ],
    "date_understanding": [
        "(A)",
        "(B)",
        "(C)",
        "(D)",
        "(E)",
        "(F)"
    ],
    "causal_judgement": [
        "No",
        "Yes"
    ],
    "boolean_expressions": [
        "False",
        "True"
    ]
}
