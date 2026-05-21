# -*- coding: utf-8 -*-

from typing import Optional, Dict, Any

from tasks import TaskManager
# from tasks.task_mcqa import SYSTEM_PROMPT_MCQA_GEN


class TaskGPQA(TaskManager):

    def __init__(
            self,
            verbose: bool,
            logger,
            cache_dir: Optional[str] = None,
            project_root_dir: Optional[str] = None,
            model_name: str = "llama3-8b",
    ):
        super().__init__(verbose, logger, cache_dir, project_root_dir, model_name)

        self.task_name = "gpqa"
        self.task_info = {
            "task_name": self.task_name,
            "task_type": "mcqa",
            "eval": [
                ["Idavidrein/gpqa", "gpqa_diamond", "train"],
            ],
            "num_to_eval": -1,  # The # of items per subtask in "eval" set (if exists) for evaluation. -1: use all
        }
        # >>> Token Stat: >>> #Sub-Tasks = 1; #Items = 198; avg_len_token: 366.0; std_len_token: 214.4; skip_cnt = 0

        self.correct_idx = 0

    def get_prompt_eval(
            self,
            data_item,
            **kwargs
    ) -> Dict[str, Any]:
        # dialog_sys = [{"role": "system", "content": SYSTEM_PROMPT_MCQA_GEN}]
        dialog_sys = []

        # Process data
        question = str(data_item["Question"]).replace("\n", " ").strip()
        # rationale = str(data_item["Explanation"]).replace("\n", " ").strip()
        answer = str(data_item["Correct Answer"]).strip()
        distractor1 = str(data_item["Incorrect Answer 1"]).strip()
        distractor2 = str(data_item["Incorrect Answer 2"]).strip()
        distractor3 = str(data_item["Incorrect Answer 3"]).strip()

        label_options = ["A", "B", "C", "D"]
        answer_str = answer
        if self.correct_idx == 0:
            answer_options = [answer, distractor1, distractor2, distractor3]
            answer_label = "A"
            pos_a = 0
        elif self.correct_idx == 1:
            answer_options = [distractor1, answer, distractor2, distractor3]
            answer_label = "B"
            pos_a = 1
        elif self.correct_idx == 2:
            answer_options = [distractor1, distractor2, answer, distractor3]
            answer_label = "C"
            pos_a = 2
        elif self.correct_idx == 3:
            answer_options = [distractor1, distractor2, distractor3, answer]
            answer_label = "D"
            pos_a = 3
        else:
            raise ValueError(f">>> ValueError: correct_idx = {self.correct_idx}")
        answers = [f"({answer_label}) {answer_str}", answer_str, answer_label, f"({answer_label})", f"{answer_label})"]
        self.correct_idx = (self.correct_idx + 1) % 4

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
            }
        }
        return result_dict
