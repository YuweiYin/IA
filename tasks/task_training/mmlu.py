# -*- coding: utf-8 -*-

import os
from typing import Optional, Dict, Any

from datasets import load_dataset

from tasks import TaskManager
# from tasks.task_mcqa import SYSTEM_PROMPT_MCQA_GEN


class TaskTrainingMmlu(TaskManager):

    def __init__(
            self,
            verbose: bool,
            logger,
            cache_dir: Optional[str] = None,
            project_root_dir: Optional[str] = None,
            model_name: str = "llama3-8b",
    ):
        super().__init__(verbose, logger, cache_dir, project_root_dir, model_name)

        self.task_name = "mmlu_training"
        self.task_info = {
            "task_name": self.task_name,
            "task_type": "mcqa",
            "eval": [
                ["cais/mmlu", "all", "auxiliary_train"],
            ],
            "num_to_eval": -1,  # The # of items per subtask in "eval" set (if exists) for evaluation. -1: use all
        }
        # >>> Token Stat: >>> #Sub-Tasks = 1; #Items = 99842; avg_len_token: 394.9; std_len_token: 155.0; skip_cnt = 0

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

            for ds_info in ds_list:
                assert isinstance(ds_info, list) and len(ds_info) == 3
                # self.logger.info(f">>> [dataset: {ds_info[0]} --- {ds_info[1]}]")
                try:  # Load the subtask
                    dataset_name, subset_name, split_name = ds_info[0], ds_info[1], ds_info[2]
                    cur_ds = load_dataset(
                        dataset_name, subset_name, split=split_name,
                        cache_dir=os.path.join(self.cache_dir, "datasets"),
                        # trust_remote_code=True,
                    )
                    ds_dict = {
                        "dataset_name": dataset_name,
                        "subset_name": subset_name,
                        "split_name": split_name,
                        "dataset": cur_ds,
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

        # Process data
        subject = str(data_item["subject"]).strip()
        question = str(data_item["question"]).strip()
        choices = list(data_item["choices"])
        choices = [str(_c).strip() for _c in choices]
        answer = int(data_item["answer"])

        assert isinstance(choices, list) and len(choices) == 4
        index2label = {0: "A", 1: "B", 2: "C", 3: "D"}
        assert answer in index2label

        answer_str = choices[answer]
        answer_label = index2label[answer]
        label_options = ["A", "B", "C", "D"]
        answer_options = choices

        pos_a = answer
        answers = [f"({answer_label}) {answer_str}", answer_str, answer_label, f"({answer_label})", f"{answer_label})"]

        # Set the main prompt (zero-shot)
        options_str = "\n".join([f"({_label}) {_ans}" for _label, _ans in zip(label_options, answer_options)])
        dialog_user = [{"role": "user", "content": f"""
Answer the following question by selecting an option:
{question}
{options_str}
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

    def get_prompt_ft_analysis(
            self,
            data_item,
            **kwargs
    ) -> Dict[str, Any]:
        # dialog_sys = [{"role": "system", "content": SYSTEM_PROMPT_MCQA_GEN}]
        dialog_sys = []

        # Process data
        subject = str(data_item["subject"]).strip()
        question = str(data_item["question"]).strip()
        choices = list(data_item["choices"])
        choices = [str(_c).strip() for _c in choices]
        answer = int(data_item["answer"])

        assert isinstance(choices, list) and len(choices) == 4
        index2label = {0: "A", 1: "B", 2: "C", 3: "D"}
        assert answer in index2label

        answer_str = choices[answer]
        answer_label = index2label[answer]
        label_options = ["A", "B", "C", "D"]
        answer_options = choices

        pos_a = answer
        answers = [f"({answer_label}) {answer_str}", answer_str, answer_label, f"({answer_label})", f"{answer_label})"]

        # Set the main prompt (zero-shot)
        options_str_list = [f"({_label}) {_ans}" for _label, _ans in zip(label_options, answer_options)]
        options_str = "\n".join(options_str_list)
        dialog_user = [{"role": "user", "content": f"""
You will be given a question and the correct answer, and your task is to output the analysis of the question. \
Your analysis must reasonably lead to the correct answer, but you should not reveal the answer at the beginning.

Here is the question:
{question}
{options_str}

We already know the correct answer is {options_str_list[pos_a]}

Now, you should output the analysis of the question.
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
