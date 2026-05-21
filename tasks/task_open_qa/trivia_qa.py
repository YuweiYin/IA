# -*- coding: utf-8 -*-

from typing import Optional, Dict, List, Any

from tasks import TaskManager
# from tasks.task_open_qa import SYSTEM_PROMPT_OPEN_QA_GEN


class TaskTriviaQA(TaskManager):

    def __init__(
            self,
            verbose: bool,
            logger,
            cache_dir: Optional[str] = None,
            project_root_dir: Optional[str] = None,
            model_name: str = "llama3-8b",
    ):
        super().__init__(verbose, logger, cache_dir, project_root_dir, model_name)

        self.task_name = "trivia_qa"
        self.task_info = {
            "task_name": self.task_name,
            "task_type": "open_qa",
            "eval": [
                # ["mandarjoshi/trivia_qa", "rc.nocontext", "validation"],
                # ["mandarjoshi/trivia_qa", "rc", "validation"],
                # ["mandarjoshi/trivia_qa", "rc.wikipedia.nocontext", "validation"],
                ["mandarjoshi/trivia_qa", "rc.wikipedia", "validation"],
            ],
            "num_to_eval": -1,  # The # of items per subtask in "eval" set (if exists) for evaluation. -1: use all
        }
        # >>> [rc.nocontext] >>> #Sub-Tasks = 1; #Items = 17944; avg_len_token: 142.4; std_len_token: 9.2; skip_cnt = 0
        # >>> [rc.wikipedia] >>> #Sub-Tasks = 1; #Items = 7993; avg_len_token: 142.6; std_len_token: 9.4; skip_cnt = 0

    def get_prompt_eval(
            self,
            data_item,
            **kwargs
    ) -> Dict[str, Any]:
        # dialog_sys = [{"role": "system", "content": SYSTEM_PROMPT_OPEN_QA_GEN}]
        dialog_sys = []

        # Process data
        # question_id = str(data_item["question_id"]).strip()
        # question_source = str(data_item["question_source"]).strip()
        question = str(data_item["question"]).replace("\n", " ").strip()

        entity_pages = dict(data_item["entity_pages"])
        search_results = dict(data_item["search_results"])
        assert isinstance(entity_pages["title"], list) and isinstance(entity_pages["wiki_context"], list)
        assert len(entity_pages["title"]) == len(entity_pages["wiki_context"])
        assert isinstance(search_results["title"], list) and isinstance(search_results["search_context"], list)
        assert len(search_results["title"]) == len(search_results["search_context"])
        context = {
            "title": entity_pages["title"] + search_results["title"],
            "context": entity_pages["wiki_context"] + search_results["search_context"],
        }
        context_len = len(entity_pages["wiki_context"]) + len(search_results["search_context"])

        # Set the context string
        context_str = ""
        context_idx = 1
        for cur_t, cur_c in zip(context["title"], context["context"]):
            cur_t = cur_t.replace("\n", " ").strip()
            cur_c = cur_c.replace("\n", " ").strip()
            context_str += f"({context_idx}) {cur_t}: {cur_c}\n"
            context_idx += 1
        context["context_str"] = context_str

        answer_dict = dict(data_item["answer"])
        answers = ([str(answer_dict["value"]), str(answer_dict["normalized_value"])] +
                   list(answer_dict["aliases"]) + list(answer_dict["normalized_aliases"]))
        answers = [_ans.strip() for _ans in answers]
        answers = list(set(answers))
        if len(answers) == 0:
            return {}  # invalid instance, skip it

        # Get the shortest answer
        answers_sort = sorted(answers, key=lambda x: len(x), reverse=False)
        # answer_str = str(answers_sort[0]).strip()

        use_context = "use_context" in kwargs and kwargs["use_context"]
        # use_context = True

        # Set the main prompt (zero-shot)
        if use_context and context_len > 0:
            dialog_user = [{"role": "user", "content": f"""
Here is the context information for answering a question:
{context_str}

Based on the above context, answer the following question:
{question}
            """.strip() + "\n"}]
        else:
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
                "context": context_str,
                "question": question,
            }
        }
        return result_dict
