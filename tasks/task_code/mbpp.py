# -*- coding: utf-8 -*-

from typing import Optional, Dict, Any

from tasks import TaskManager
from tasks.task_code import SYSTEM_PROMPT_CODE_GEN


class TaskMBPP(TaskManager):

    def __init__(
            self,
            verbose: bool,
            logger,
            cache_dir: Optional[str] = None,
            project_root_dir: Optional[str] = None,
            model_name: str = "llama3-8b",
    ):
        super().__init__(verbose, logger, cache_dir, project_root_dir, model_name)

        self.task_name = "mbpp"
        self.task_info = {
            "task_name": self.task_name,
            "task_type": "code",
            "eval": [
                ["google-research-datasets/mbpp", "full", "test"],
            ],
            "num_to_eval": -1,  # The # of items per subtask in "eval" set (if exists) for evaluation. -1: use all
        }
        # >>> Token Stat: >>> #Sub-Tasks = 1; #Items = 500; avg_len_token: 183.8; std_len_token: 13.6; skip_cnt = 0

    def get_prompt_eval(
            self,
            data_item,
            **kwargs
    ) -> Dict[str, Any]:
        # dialog_sys = [{"role": "system", "content": SYSTEM_PROMPT_CODE_GEN}]
        dialog_sys = []

        # Process data
        # task_id = str(data_item["task_id"]).strip()
        problem = str(data_item["text"]).strip()
        solution = str(data_item["code"]).strip()
        test_case_list = list(data_item["test_list"]) + list(data_item["challenge_test_list"])
        test_setup_code = str(data_item["test_setup_code"]).strip()

        # Extract the target function name; Set the evaluation code
        func_name_list = []
        unit_tests = []
        for test_line in test_case_list:
            test_line = str(test_line).strip()
            assert test_line.startswith("assert"), test_line
            test_line_clean = test_line.lstrip("assert").strip()
            if test_line_clean.startswith("int("):
                test_line_clean = test_line_clean.lstrip("int(").strip()
            if test_line_clean.startswith("float("):
                test_line_clean = test_line_clean.lstrip("float(").strip()
            if test_line_clean.startswith("str("):
                test_line_clean = test_line_clean.lstrip("str(").strip()
            cur_func_name = test_line_clean.split("(")[0].strip()
            func_name_list.append(cur_func_name)

            if isinstance(test_setup_code, str) and len(test_setup_code) > 0:
                unit_tests.append(test_setup_code + "\n\n" + test_line)
            else:
                unit_tests.append(test_line)
        func_name_list = list(set(func_name_list))
        assert len(func_name_list) == 1, func_name_list
        func_name = func_name_list[0]

        # Extract the function header from the provided code solution
        func_header = ""
        assert f"def {func_name}" in solution, solution
        for sol_line in solution.split("\n"):
            func_header += sol_line + "\n"
            if f"def {func_name}" in sol_line:
                break

        # Note: we need to evaluate code tasks by exec() the unit tests and computing the pass@k score (e.g., k=1)
        answers = [solution]

        # Set the main prompt (zero-shot)
        dialog_user = [{"role": "user", "content": SYSTEM_PROMPT_CODE_GEN.strip() + "\n\nTask: " + f"""
Complete the following `{func_name}` function. The function header is provided below. \
Ensure your final answer is grammatically correct, complete, and executable Python code \
that fulfills the following requirements:
{problem}

```python
{func_header}
```
        """.strip()}]

        dialog = dialog_sys + dialog_user

        # Set the result dict
        if len(unit_tests) == 0:
            return {}  # invalid instance
        result_dict = {
            "dialog": dialog,
            "answers": answers,
            "info": {
                "task_type": "code",
                "problem": problem,
                "func_header": func_header,
                "unit_tests": unit_tests,
                "func_name": func_name,
                "question_interpretation": None,
                "solution_explanation": None,
            }
        }
        return result_dict
