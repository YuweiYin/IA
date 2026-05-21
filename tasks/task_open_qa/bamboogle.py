# -*- coding: utf-8 -*-

from typing import Optional, Dict, Any

from tasks import TaskManager
# from tasks.task_open_qa import SYSTEM_PROMPT_OPEN_QA_GEN


class TaskBamboogle(TaskManager):

    def __init__(
            self,
            verbose: bool,
            logger,
            cache_dir: Optional[str] = None,
            project_root_dir: Optional[str] = None,
            model_name: str = "llama3-8b",
    ):
        super().__init__(verbose, logger, cache_dir, project_root_dir, model_name)

        self.task_name = "bamboogle"
        self.task_info = {
            "task_name": self.task_name,
            "task_type": "open_qa",
            "eval": [
                ["RUC-NLPIR/FlashRAG_datasets", "bamboogle", "test"],
            ],
            "num_to_eval": -1,  # The # of items per subtask in "eval" set (if exists) for evaluation. -1: use all
        }
        # >>> Token Stat: >>> #Sub-Tasks = 1; #Items = 125; avg_len_token: 137.9; std_len_token: 3.4; skip_cnt = 0

    def get_prompt_eval(
            self,
            data_item,
            **kwargs
    ) -> Dict[str, Any]:
        # dialog_sys = [{"role": "system", "content": SYSTEM_PROMPT_OPEN_QA_GEN}]
        dialog_sys = []

        # Process data
        question = str(data_item["question"]).replace("\n", " ").strip()
        golden_answers = list(data_item["golden_answers"])
        assert isinstance(golden_answers, list) and len(golden_answers) > 0, golden_answers
        answers = [_ans.strip() for _ans in golden_answers]
        answers = list(set(answers))
        if len(answers) == 0:
            return {}  # invalid instance, skip it

        # Get the shortest answer
        answers_sort = sorted(answers, key=lambda x: len(x), reverse=False)
        # answer_str = str(answers_sort[0]).strip()

        # use_context = "use_context" in kwargs and kwargs["use_context"]
        dialog_user = [{"role": "user", "content": f"""
Answer the following question:
{question}
        """.strip() + "\n"}]

        dialog = dialog_sys + dialog_user

        # Set the result dict
        result_dict = {
            "dialog": dialog,
            "answers": answers_sort,
            "info": {
                "task_type": "open_qa",
                # "context": context,
                "context": "",
                "question": question,
            }
        }
        return result_dict
