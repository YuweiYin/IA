#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import time
import string
from typing import Optional, List

import fire
import numpy as np

import evaluate
from datasets.download.download_config import DownloadConfig

from tasks.tasks_utils import *

from utils.init_functions import logger_setup, cuda_setup, random_setup
from utils.models import ModelUtils
from utils.data_io import DataIO

os.environ["HF_ALLOW_CODE_EVAL"] = "1"


class LMEval:

    def __init__(
            self,
            verbose: bool,
            logger,
            cuda_dict: Optional[dict] = None,
            seed: int = 42,
            cache_dir: Optional[str] = None,
            project_root_dir: Optional[str] = None,
            model_name: str = "llama3-8b",
            show_generation: bool = False,
            debug: bool = False,
            output_dir: Optional[str] = None,
            overwrite: bool = False,
            add_ia_starter: bool = False,
            do_multi_stage: bool = False,
            use_analysis: bool = False,
    ):
        self.verbose = verbose
        self.logger = logger
        self.cuda_dict = cuda_dict
        self.seed = seed
        self.model_name = model_name
        self.show_generation = show_generation  # If True, show outputs during generation
        self.debug = debug
        self.add_ia_starter = add_ia_starter
        self.do_multi_stage = do_multi_stage
        self.use_analysis = use_analysis

        if isinstance(project_root_dir, str) and os.path.isdir(project_root_dir):
            self.project_root_dir = project_root_dir
        else:
            self.project_root_dir = os.getcwd()
        assert os.path.isdir(self.project_root_dir)

        self.output_dir = output_dir
        self.overwrite = overwrite

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

        self.tokenizer = None
        self.terminators_gen = None
        self.model = None

        # Evaluators
        hf_eval_cache = os.path.join(self.cache_dir, "evaluate")
        os.makedirs(hf_eval_cache, exist_ok=True)

        download_config = DownloadConfig(cache_dir=hf_eval_cache, force_download=False)
        self.eval_code = evaluate.load(
            path="evaluate_metrics/code_eval", cache_dir=hf_eval_cache, download_config=download_config)

        self.punc_remover = str.maketrans("", "", string.punctuation)  # r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
        self.space_remover = str.maketrans("", "", string.whitespace)  # " \t\n\r\v\f"

        self.special_re_token = r"<|-IA-|>"

    @staticmethod
    def extract_boxed_answers(input_str: str) -> List[str]:
        boxed_answers_1 = re.findall(r"boxed{(.*?)}", input_str)
        boxed_answers_2 = re.findall(r"\$*\\boxed(.*?)\$", input_str)
        boxed_answers = []
        for b_ans in boxed_answers_1 + boxed_answers_2:
            if not isinstance(b_ans, str):
                continue
            b_ans = b_ans.strip()
            if len(b_ans) == 0:
                continue

            boxed_answers.append(b_ans)
            # Also, consider the equivalent variants
            boxed_answers.append(r"\boxed{" + b_ans + r"}")
            boxed_answers.append(b_ans.replace("\n", "").strip())
            if len(b_ans) > 2 and b_ans.startswith("{") and b_ans.endswith("}"):
                boxed_answers.append(b_ans[1:-1])
            if len(b_ans) > 2 and b_ans.startswith("(") and b_ans.endswith(")"):
                boxed_answers.append(b_ans[1:-1])
            if len(b_ans) > 2 and b_ans.startswith("[") and b_ans.endswith("]"):
                boxed_answers.append(b_ans[1:-1])

        boxed_answers = list(set(boxed_answers))

        return boxed_answers

    @staticmethod
    def normalize_text(raw_text: str) -> str:
        def _white_space(text: str):
            return " ".join(text.split())

        def _remove_articles(text: str):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def _remove_punc(text: str):
            punc_set = set(string.punctuation)
            return "".join(ch for ch in text if ch not in punc_set)

        cleaned_text = _white_space(_remove_articles(_remove_punc(raw_text.lower())))
        return cleaned_text

    def compute_score_mcqa(
            self,
            prediction: str,
            references: List[str],
            **kwargs
    ) -> dict:
        prediction = str(prediction).strip()
        references = [str(_ref).strip() for _ref in references]

        pred_all = [prediction]

        # Extract leading answer labels
        pred_all += re.findall(r"^\(([A-Z0-9])\)", prediction, re.IGNORECASE)  # "(A)"
        pred_all += re.findall(r"^([A-Z0-9])\)", prediction, re.IGNORECASE)  # "A)"
        pred_all += re.findall(r"^([A-Z0-9]):", prediction, re.IGNORECASE)  # "A:"
        pred_all += re.findall(r"^([A-Z0-9])\.", prediction, re.IGNORECASE)  # "A."
        pred_all += re.findall(r"^([A-Z0-9])\n", prediction, re.IGNORECASE)  # "A\n"

        # Extract boxed answers (also consider the cases where \boxed{} contains "\n")
        boxed_answers = self.extract_boxed_answers(prediction)
        boxed_answers_special = self.extract_boxed_answers(prediction.replace("\n", self.special_re_token))
        boxed_answers_special = [_ans.replace(self.special_re_token, "\n").strip() for _ans in boxed_answers_special]
        pred_all += boxed_answers + boxed_answers_special

        # Extract answers in the emphasis symbols
        emphasis_answers_1 = re.findall(r"\*\*(.*?)\*\*", prediction)
        emphasis_answers_2 = re.findall(r"__(.*?)__", prediction)
        pred_all += emphasis_answers_1 + emphasis_answers_2

        # Yes/No style answers
        pred_all += re.findall(r"^(yes)", prediction, re.IGNORECASE)  # Yes
        pred_all += re.findall(r"^(no)", prediction, re.IGNORECASE)  # No
        pred_all += re.findall(r"^(true)", prediction, re.IGNORECASE)  # True
        pred_all += re.findall(r"^(false)", prediction, re.IGNORECASE)  # False
        pred_all += re.findall(r"^(valid)", prediction, re.IGNORECASE)  # Valid
        pred_all += re.findall(r"^(invalid)", prediction, re.IGNORECASE)  # Invalid

        if "info" in kwargs:
            info = dict(kwargs["info"])
            assert "options" in info, info
            options = info["options"]
            if isinstance(options, list) and len(options) > 0:
                # Match if the prediction starts with the option
                for option in options:
                    assert isinstance(option, str) and len(option) > 0, option
                    option = option.strip()
                    assert len(option) > 0, option
                    if prediction.lower().startswith(option.lower()):
                        pred_all.append(option)

        # Consider the parentheses
        pred_all = pred_all + [f"({_p})" for _p in pred_all] + [f"{_p})" for _p in pred_all]
        references = references + [f"({_r})" for _r in references] + [f"{_r})" for _r in references]

        # Remove duplication
        pred_all = list(set(pred_all))
        references = list(set(references))

        # Matching anyone in the references will have an EM score of 1; otherwise 0.
        for ref in references:
            for pred in pred_all:
                # Consider both lower and upper cases
                pred_norm = self.normalize_text(pred).strip()
                if pred_norm.endswith("."):
                    pred_norm = pred_norm[:-1].strip()
                ref_norm = self.normalize_text(ref).strip()
                if pred_norm == ref_norm:  # or pred_norm.startswith(ref_norm)
                    return {"score": float(1.0), "metric": "acc"}

        return {"score": float(0.0), "metric": "acc"}

    def compute_score_open_qa(
            self,
            prediction: str,
            references: List[str],
            # **kwargs
    ) -> dict:
        prediction = str(prediction).strip()
        references = [str(_ref).strip() for _ref in references]

        pred_all = [prediction]

        # Extract leading answer labels
        pred_all += re.findall(r"^\(([A-Z0-9])\)", prediction, re.IGNORECASE)  # "(A)"
        pred_all += re.findall(r"^([A-Z0-9])\)", prediction, re.IGNORECASE)  # "A)"
        pred_all += re.findall(r"^([A-Z0-9]):", prediction, re.IGNORECASE)  # "A:"
        pred_all += re.findall(r"^([A-Z0-9])\.", prediction, re.IGNORECASE)  # "A."
        pred_all += re.findall(r"^([A-Z0-9])\n", prediction, re.IGNORECASE)  # "A\n"

        # Extract boxed answers (also consider the cases where \boxed{} contains "\n")
        boxed_answers = self.extract_boxed_answers(prediction)
        boxed_answers_special = self.extract_boxed_answers(prediction.replace("\n", self.special_re_token))
        boxed_answers_special = [_ans.replace(self.special_re_token, "\n").strip() for _ans in boxed_answers_special]
        pred_all += boxed_answers + boxed_answers_special

        # Extract answers in the emphasis symbols
        emphasis_answers_1 = re.findall(r"\*\*(.*?)\*\*", prediction)
        emphasis_answers_2 = re.findall(r"__(.*?)__", prediction)
        pred_all += emphasis_answers_1 + emphasis_answers_2

        # Yes/No style answers
        pred_all += re.findall(r"^(yes)", prediction, re.IGNORECASE)  # Yes
        pred_all += re.findall(r"^(no)", prediction, re.IGNORECASE)  # No
        pred_all += re.findall(r"^(true)", prediction, re.IGNORECASE)  # True
        pred_all += re.findall(r"^(false)", prediction, re.IGNORECASE)  # False
        # pred_all += re.findall(r"^(valid)", prediction, re.IGNORECASE)  # Valid
        # pred_all += re.findall(r"^(invalid)", prediction, re.IGNORECASE)  # Invalid

        # Remove duplication
        pred_all = list(set(pred_all))
        references = list(set(references))

        # Normalize strings and then match.  # " ".join(prediction.split())  # Clear adjacent whitespaces
        pred_all_new, references_new = pred_all[:], references[:]
        pred_all_new += [_p.translate(self.punc_remover).strip() for _p in pred_all]  # Remove all punctuations
        # pred_all_new += [_p.translate(self.space_remover).strip() for _p in pred_all]  # Remove all whitespaces
        pred_all_new += [" ".join(_p.split()).strip() for _p in pred_all]  # Remove consecutive whitespaces
        pred_all_new = list(set(pred_all_new))
        references_new += [_r.translate(self.punc_remover).strip() for _r in references]
        # references_new += [_r.translate(self.space_remover).strip() for _r in references]
        references_new += [" ".join(_r.split()).strip() for _r in references]
        references_new = list(set(references_new))

        # Matching anyone in the references will have an EM score of 1; otherwise 0.
        for ref in references_new:
            for pred in pred_all_new:
                # Consider both lower and upper cases
                pred_norm = self.normalize_text(pred).strip()
                if pred_norm.endswith("."):
                    pred_norm = pred_norm[:-1].strip()
                ref_norm = self.normalize_text(ref).strip()
                if ref_norm == pred_norm:
                    return {"score": float(1.0), "metric": "acc"}
                if (ref_norm in pred_norm) or (pred_norm in ref_norm):
                    return {"score": float(1.0), "metric": "acc"}

        return {"score": float(0.0), "metric": "acc"}

    def compute_score_math(
            self,
            prediction: str,
            references: List[str],
            # **kwargs
    ) -> dict:
        prediction = str(prediction).strip()
        references = [str(_ref).strip() for _ref in references]

        pred_all = [prediction]

        # Extract leading answer labels
        pred_all += re.findall(r"^\(([A-Z])\)", prediction, re.IGNORECASE)  # "(A)"
        pred_all += re.findall(r"^([A-Z])\)", prediction, re.IGNORECASE)  # "A)"
        pred_all += re.findall(r"^([A-Z]):", prediction, re.IGNORECASE)  # "A:"
        pred_all += re.findall(r"^([A-Z])\.", prediction, re.IGNORECASE)  # "A."
        pred_all += re.findall(r"^([A-Z])\n", prediction, re.IGNORECASE)  # "A\n"

        # Extract leading numbers
        pred_all += re.findall(r"^([-+]?[0-9]+)", prediction, re.IGNORECASE)  # 10
        pred_all += re.findall(r"^([-+]?[0-9]*\.[0-9]+)", prediction, re.IGNORECASE)  # 10.2
        pred_all += re.findall(r"^(.*?):", prediction, re.IGNORECASE)  # "10:"
        pred_all += re.findall(r"^(.*?)\.", prediction, re.IGNORECASE)  # "10."
        pred_all += re.findall(r"^(.*?)\n", prediction, re.IGNORECASE)  # "10\n"

        # Extract boxed answers (also consider the cases where \boxed{} contains "\n")
        boxed_answers = self.extract_boxed_answers(prediction)
        boxed_answers_special = self.extract_boxed_answers(prediction.replace("\n", self.special_re_token))
        boxed_answers_special = [_ans.replace(self.special_re_token, "\n").strip() for _ans in boxed_answers_special]
        pred_all += boxed_answers + boxed_answers_special

        # Extract answers in the emphasis symbols
        emphasis_answers_1 = re.findall(r"\*\*(.*?)\*\*", prediction)
        emphasis_answers_2 = re.findall(r"__(.*?)__", prediction)
        pred_all += emphasis_answers_1 + emphasis_answers_2

        # Remove duplication
        pred_all = list(set(pred_all))
        references = list(set(references))

        # Normalize strings and then match.  # " ".join(prediction.split())  # Clear adjacent whitespaces
        pred_all_new, references_new = pred_all[:], references[:]
        pred_all_new += [_p.translate(self.punc_remover).strip() for _p in pred_all]  # Remove all punctuations
        # pred_all_new += [_p.translate(self.space_remover).strip() for _p in pred_all]  # Remove all whitespaces
        pred_all_new += [" ".join(_p.split()).strip() for _p in pred_all]  # Remove consecutive whitespaces
        pred_all_new = list(set(pred_all_new))
        references_new += [_r.translate(self.punc_remover).strip() for _r in references]
        # references_new += [_r.translate(self.space_remover).strip() for _r in references]
        references_new += [" ".join(_r.split()).strip() for _r in references]
        references_new = list(set(references_new))

        # Matching anyone in the references will have an EM score of 1; otherwise 0.
        for ref in references_new:
            for pred in pred_all_new:
                # Consider both lower and upper cases
                pred_norm = self.normalize_text(pred).strip()
                if pred_norm.endswith("."):
                    pred_norm = pred_norm[:-1].strip()
                ref_norm = self.normalize_text(ref).strip()
                if pred_norm == ref_norm:
                    return {"score": float(1.0), "metric": "acc"}

        return {"score": float(0.0), "metric": "acc"}

    def compute_score_code(
            self,
            prediction: str,
            # references: List[str],
            **kwargs
    ) -> dict:
        assert isinstance(prediction, str) and len(prediction) > 0, prediction

        py_code_starter = "```python"
        if py_code_starter in prediction:
            # Extract the code lines from "```python" to "```"
            pred_code = ""
            for code_line in prediction.split(py_code_starter)[-1].strip().split("\n"):
                if code_line.strip() == "```":
                    break
                pred_code += code_line + "\n"
            pred_code = pred_code.strip() + "\n"
        else:
            pred_code = prediction.strip() + "\n"

        assert "info" in kwargs, kwargs
        info = dict(kwargs["info"])
        assert "unit_tests" in info, info
        unit_tests = info["unit_tests"]

        assert isinstance(unit_tests, list) and len(unit_tests) > 0, unit_tests
        num_tests = len(unit_tests)

        if "k" in kwargs and isinstance(kwargs["k"], list) and len(kwargs["k"]) > 0:
            k = kwargs["k"]
        else:
            k = [1]

        if "timeout" in kwargs and isinstance(kwargs["timeout"], int) and kwargs["timeout"] > 0:
            timeout = float(kwargs["timeout"])
        else:
            # timeout = float(3.0)  # code_eval default timeout: 3 seconds
            timeout = min(float(30.0), max(float(15.0), float(5.0) * num_tests))  # no hurry...

        if "num_workers" in kwargs and isinstance(kwargs["num_workers"], int) and kwargs["num_workers"] > 0:
            num_workers = int(kwargs["timeout"])
        else:
            num_workers = int(4)  # code_eval default num_workers: 4

        try:
            pass_at_k, eval_results = self.eval_code.compute(
                references=unit_tests, predictions=[[pred_code] for _ in range(num_tests)],
                k=k, timeout=timeout, num_workers=num_workers)
            time.sleep(1.0)  # no hurry...
        except Exception as e:
            self.logger.info(f">>> Exception !!! >>> `eval_code.compute`\n{e}")
            return {"score": float(0.0), "metric": "pass@1"}

        assert "pass@1" in pass_at_k, pass_at_k
        pass_1_score = pass_at_k["pass@1"]  # "passed" -> 1.0; "failed" or "time out" -> 0.0
        return {"score": float(pass_1_score), "metric": "pass@1"}

    def lm_evaluate(
            self,
            task_name: str,
            gen_results: Optional[list] = None,
            do_save: bool = True,
    ) -> Optional[dict]:
        # Evaluation Phase: load result JSON, extract the reasoning/analysis and final answers, and compute scores

        # os.environ["HF_ALLOW_CODE_EVAL"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # Set the saving filepath
        assert isinstance(self.output_dir, str) and os.path.isdir(self.output_dir), "Please specify --output_dir"
        output_dir = os.path.join(self.output_dir, task_name, self.model_name)
        output_fn = "results_gen"
        if self.add_ia_starter:
            output_fn += "_ia"
        if self.do_multi_stage:
            output_fn += "_multi"
        output_fp = os.path.join(output_dir, output_fn + ".jsonl")
        output_eval_fp = os.path.join(output_dir, output_fn + "--eval" + ".json")
        if do_save:
            if os.path.isfile(output_eval_fp):
                if self.overwrite:
                    self.logger.info(f"Results will be overwritten: {output_eval_fp}")
                else:
                    self.logger.info(
                        f">>> model_name = {self.model_name}; output_dir: {output_dir}\n"
                        f">>> !!! >>> [SKIP; No --overwrite] File already exists: {output_eval_fp}"
                    )
                    return None
            else:
                self.logger.info(f"Results will be saved at: {output_eval_fp}")

        if not (isinstance(gen_results, list) and len(gen_results) > 0):
            if not os.path.isfile(output_fp):
                self.logger.info(
                    f">>> model_name = {self.model_name}; output_dir: {output_dir}\n"
                    f">>> !!! >>> [SKIP; No --output_fp] output_fp does not exist: {output_fp}"
                )
                return None

            # Load the generation outputs
            if output_fp.endswith(".json"):
                gen_results = DataIO.load_json(output_fp, mode="r", verbose=True)
            elif output_fp.endswith(".jsonl"):
                gen_results = DataIO.load_jsonl(output_fp, mode="r", verbose=True)
            else:
                raise ValueError(f">>> !!! >>> Only JSON and JSONL are supported. output_fp: {output_fp}")

        # Deal with each task (and sub-tasks)
        self.logger.info(f">>> Evaluation Task: {task_name}")
        assert isinstance(gen_results, list) and len(gen_results) > 0, type(gen_results)
        num_results = len(gen_results)
        all_score_dicts = []
        all_score_values = []
        eval_metric = ""
        data_item_cnt_total = 0
        miss_final_cnt_total = 0
        show_cnt = 100
        for item_idx, cur_res_dict in enumerate(gen_results):
            assert isinstance(cur_res_dict, dict) and len(cur_res_dict) > 0, type(cur_res_dict)

            # Load the attributes of the data item
            # gen_prompt = str(cur_res_dict["input_text"]).strip()  # the original inputs to the model for generation
            # analysis = str(cur_res_dict["analysis"]).strip()  # analysis (by the model) of the question
            # prediction = str(cur_res_dict["output_text"]).strip()  # model prediction to evaluate
            prediction = str(cur_res_dict["pred_answer"]).strip()  # model prediction to evaluate
            references = cur_res_dict["answers"]  # golden references (correct answers)
            info = cur_res_dict["info"]  # task-specific information

            if isinstance(references, list) and len(references) > 0:
                references = [str(_ref).strip() for _ref in references]
            else:
                references = []

            # Extract the final answer from the generated output
            assert isinstance(info, dict) and "task_type" in info
            task_type = str(info["task_type"]).strip()
            miss_final = False  # Also, count the number of missing final answer.
            final_split = "Final Answer:"

            match task_type:
                case "mcqa":
                    if self.use_analysis and "analysis" in cur_res_dict and final_split in cur_res_dict["analysis"]:
                        analysis = str(cur_res_dict["analysis"]).strip()
                        pred_final = analysis.split(final_split)[-1].strip()
                    else:
                        pred_final = prediction.strip()

                    if self.model_name not in ModelUtils.OPEN_MODEL_HF:
                        if final_split in pred_final:
                            pred_final = pred_final.split(final_split)[-1].strip()
                        while pred_final.startswith("*"):
                            pred_final = pred_final[1:].strip()
                        pred_final = pred_final.strip()

                    if len(pred_final) == 0:
                        miss_final = True
                        cur_score_dict = {"score": float(0.0), "metric": "acc"}
                    else:
                        pred_final = re.sub(r"[^\x00-\x7F]+", "", pred_final).strip()  # remove non-ASCII
                        cur_score_dict = self.compute_score_mcqa(
                            references=references, prediction=pred_final, info=info)
                case "open_qa":
                    if self.use_analysis and "analysis" in cur_res_dict and final_split in cur_res_dict["analysis"]:
                        analysis = str(cur_res_dict["analysis"]).strip()
                        pred_final = analysis.split(final_split)[-1].strip()
                    else:
                        pred_final = prediction.strip()

                    if self.model_name not in ModelUtils.OPEN_MODEL_HF:
                        if final_split in pred_final:
                            pred_final = pred_final.split(final_split)[-1].strip()
                        while pred_final.startswith("*"):
                            pred_final = pred_final[1:].strip()
                        pred_final = pred_final.strip()

                    if len(pred_final) == 0:
                        miss_final = True
                        cur_score_dict = {"score": float(0.0), "metric": "acc"}
                    else:
                        pred_final = re.sub(r"[^\x00-\x7F]+", "", pred_final).strip()  # remove non-ASCII
                        cur_score_dict = self.compute_score_open_qa(references=references, prediction=pred_final)
                case "math":
                    if self.use_analysis and "analysis" in cur_res_dict and final_split in cur_res_dict["analysis"]:
                        analysis = str(cur_res_dict["analysis"]).strip()
                        pred_final = analysis.split(final_split)[-1].strip()
                    else:
                        pred_final = prediction.strip()

                    if self.model_name not in ModelUtils.OPEN_MODEL_HF:
                        if final_split in pred_final:
                            pred_final = pred_final.split(final_split)[-1].strip()
                        while pred_final.startswith("*"):
                            pred_final = pred_final[1:].strip()
                        pred_final = pred_final.strip()

                    if len(pred_final) == 0:
                        miss_final = True
                        cur_score_dict = {"score": float(0.0), "metric": "acc"}
                    else:
                        pred_final = re.sub(r"[^\x00-\x7F]+", "", pred_final).strip()  # remove non-ASCII
                        cur_score_dict = self.compute_score_math(references=references, prediction=pred_final)
                case "code":
                    from tasks.task_code import FINAL_ANSWER_CODE

                    if "analysis" in cur_res_dict:
                        analysis = str(cur_res_dict["analysis"]).strip()  # analysis (by the model) of the question
                        pred_final = analysis
                    else:
                        pred_final = prediction
                    if self.model_name not in ModelUtils.OPEN_MODEL_HF:
                        if final_split in pred_final:
                            pred_final = pred_final.split(final_split)[-1].strip()
                        while pred_final.startswith("*"):
                            pred_final = pred_final[1:].strip()
                        pred_final = pred_final.strip()
                    else:
                        if FINAL_ANSWER_CODE in pred_final:
                            pred_final = pred_final.split(FINAL_ANSWER_CODE)[-1].strip()

                    if len(pred_final) == 0:
                        miss_final = True
                        cur_score_dict = {"score": float(0.0), "metric": "pass@1"}
                    else:
                        # Note: code evaluation does not require references/answers
                        pred_final = re.sub(r"[^\x00-\x7F]+", "", pred_final).strip()  # remove non-ASCII
                        cur_score_dict = self.compute_score_code(prediction=pred_final, info=info)
                case _:
                    raise ValueError(f"ValueError: task_type = {task_type}")

            assert "score" in cur_score_dict and "metric" in cur_score_dict
            cur_score_value = cur_score_dict["score"]
            cur_metric = str(cur_score_dict["metric"]).strip()
            assert len(cur_metric) > 0
            if len(eval_metric) == 0:
                eval_metric = cur_metric
            else:
                assert eval_metric == cur_metric, \
                    f">>> Assertion Error: `eval_metric` = {eval_metric}, but `cur_metric` = {cur_metric}"

            cur_res_dict["miss_final"] = miss_final
            if miss_final:
                miss_final_cnt_total += 1
            cur_res_dict["eval_dict"] = cur_score_dict
            cur_res_dict["eval_score"] = cur_score_value

            all_score_dicts.append(cur_res_dict)
            all_score_values.append(cur_score_value)

            if self.verbose and (item_idx + 1) % show_cnt == 0:
                self.logger.info(f">>> Progress: [{item_idx + 1} / {num_results}] "
                                 f"[miss_final: {miss_final_cnt_total}]")

        # Compute the overall score statistics of different metrics and show stats
        num_items = len(all_score_values)
        match eval_metric:
            case "acc":
                # Each value is either 1.0 (correct) or 0.0 (incorrect)
                score_avg = float(np.mean(all_score_values).item())
            case "pass@1":
                # HumanEval & Pass@k metric: https://arxiv.org/pdf/2107.03374
                # pass@k = \E(1 - \frac{\binom{n-c}{k}}{\binom{n}{k}})
                # pass@1 = \E(1 - \frac{n-c}{n}) = \E(\frac{c}{n}), where c is # of correct samples (passed tests)
                # Here, each value is the pass@1 score of an instance (generated code on unit tests)
                score_avg = float(np.mean(all_score_values).item())
            case _:
                raise ValueError(f"ValueError: eval_metric = {eval_metric}")
        assert 0.0 <= score_avg <= 1.0, score_avg

        all_score_stat = {
            "num_items": num_items,
            "metric": eval_metric,
            "score_avg": score_avg,
        }
        all_scores = {
            "all_score_dicts": all_score_dicts,
            "all_score_values": all_score_values,
            "all_score_stat": all_score_stat,
        }

        self.logger.info(
            f">>> DONE ALL. [Task: {task_name}] [# Items = {num_items}] Overall Avg Score = {score_avg:.5f} "
            f"[# Data items in total = {data_item_cnt_total}] [# Missing Final Answer = {miss_final_cnt_total}]")

        # Save the generation outputs
        if do_save:
            os.makedirs(output_dir, exist_ok=True)
            if output_eval_fp.endswith(".json"):
                DataIO.save_json(output_eval_fp, all_scores, mode="w", indent=2, verbose=True)
            elif output_eval_fp.endswith(".jsonl") and isinstance(all_scores, list):
                DataIO.save_jsonl(output_eval_fp, all_scores, mode="w", verbose=True)
            else:
                raise ValueError(f">>> !!! >>> Only JSON and JSONL are supported. output_eval_fp: {output_eval_fp}")

            self.logger.info(f">>> model_name = {self.model_name}; output_eval_fp: {output_eval_fp}")

        return all_scores


def main(
    task_name="",
    model_name: str = "llama3-8b",
    cache_dir: Optional[str] = None,
    project_root_dir: Optional[str] = None,
    seed: int = 42,
    cuda: Optional[str] = None,
    verbose: bool = False,
    debug: bool = False,
    output_dir: Optional[str] = None,
    overwrite: bool = False,
    **kwargs
) -> None:
    """
    :param task_name: The name(s) of the evaluation task. (e.g., "mmlu", "bbh", or "mmlu,bbh")
    :param model_name: LLM name, e.g., "llama3-8b"
    :param cache_dir: The root directory of the cache.
    :param project_root_dir: The root directory of the current project/repo.
    :param seed: Random seed of all modules.
    :param cuda: To specify CUDA GPU devices, e.g., "0" OR "0,1". Default: None -- Use CPU or all available GPUs.
    :param verbose: Verbose mode: show logs.
    :param debug: Debugging / developing mode.
    :param output_dir: The path to the output file where the result metrics will be saved.
    :param overwrite: Overwrite existing output files.

    :return: None.
    """

    timer_start = time.perf_counter()

    # Setup of the logger, CUDA gpus, and random seed
    logger = logger_setup("Eval_Results")
    cuda_dict = cuda_setup(cuda=cuda, logger=logger, verbose=verbose)
    random_setup(seed=seed, has_cuda=cuda_dict["has_cuda"])

    if isinstance(kwargs, dict):
        logger.info(f">>> Extra parameters in kwargs: {kwargs}")
    logger.info(f">>> cuda_dict: {cuda_dict}")

    if isinstance(cache_dir, str) and os.path.isdir(cache_dir):
        os.environ["HF_HOME"] = cache_dir
    else:
        cache_dir = None

    add_ia_starter = "add_ia_starter" in kwargs  # Whether to add the Intentional Analysis prompt to the beginning
    do_multi_stage = "do_multi_stage" in kwargs  # Whether to perform each generation stage separately.
    logger.info(f">>> [add_ia_starter: {add_ia_starter}] [do_multi_stage: {do_multi_stage}]")

    use_analysis = "use_analysis" in kwargs
    logger.info(f">>> [use_analysis: {use_analysis}]")

    lm_eval = LMEval(
        verbose=verbose,
        logger=logger,
        cuda_dict=cuda_dict,
        seed=seed,
        cache_dir=cache_dir,
        project_root_dir=project_root_dir,
        model_name=model_name,
        debug=debug,
        output_dir=output_dir,
        overwrite=overwrite,
        add_ia_starter=add_ia_starter,
        do_multi_stage=do_multi_stage,
        use_analysis=use_analysis,
    )

    if isinstance(task_name, str):
        task_name = [task_name]

    if isinstance(task_name, tuple) or isinstance(task_name, list):
        for cur_task_name in task_name:
            assert cur_task_name in TASK_CLASS_DICT, \
                f"AssertionError: task name {cur_task_name} not in task_class_dict"
            cur_task_name = str(cur_task_name).strip()
            logger.info(f">>> <START> {cur_task_name}\n")
            lm_eval.lm_evaluate(task_name=cur_task_name, gen_results=None, do_save=True)
            logger.info(f">>> <END> {cur_task_name}\n\n\n")
    else:
        raise ValueError(f"--task_name should be a tuple/list/str: {task_name}")

    timer_end = time.perf_counter()
    total_sec = timer_end - timer_start
    logger.info(f"Total Running Time: {total_sec:.1f} sec ({total_sec / 60:.1f} min; {total_sec / 3600:.2f} h)")


if __name__ == "__main__":
    fire.Fire(main)
