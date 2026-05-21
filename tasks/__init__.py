#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import logging
from typing import Optional, Dict, List, Any

import fire
import numpy as np

import re
import json
import html
from tqdm import tqdm
# from multiprocessing import Pool

from datasets import load_dataset
from datasets import Dataset

from utils.init_functions import logger_setup, random_setup
from utils.models import ModelUtils


class TaskManager:

    def __init__(
            self,
            verbose: bool,
            logger,
            cache_dir: Optional[str] = None,
            project_root_dir: Optional[str] = None,
            model_name: str = "llama3-8b",
    ):
        self.verbose = verbose
        if logger is None:
            self.logger = logging.getLogger("TaskManager")
        else:
            self.logger = logger

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

        # os.environ["TRANSFORMERS_CACHE"] = cache_dir
        os.environ["HF_HOME"] = cache_dir

        self.task_name = None
        self.task_info = None

        self.model_name = model_name
        if model_name in ModelUtils.OPEN_MODEL_HF:
            self.tokenizer = ModelUtils.initialize_tokenizer_hf(
                model_name=model_name, cache_dir=cache_dir,
                padding_side="left", truncation_side="left", verbose=verbose)
            max_len = self.tokenizer.max_len_single_sentence
            if self.verbose:
                self.logger.info(
                    f">>> len(tokenizer.vocab) = {len(self.tokenizer.vocab)}; "
                    f"tokenizer.max_len_single_sentence = {max_len}")  # LLaMA-3: 131071
        else:
            self.tokenizer = None

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
                time.sleep(1)
                assert isinstance(ds_info, list) and len(ds_info) == 3
                # self.logger.info(f">>> [dataset: {ds_info[0]} --- {ds_info[1]}]")
                try:  # Load the subset
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

    @staticmethod
    def text_cleaning(
            text_input: str,
    ) -> str:
        text_output = text_input.strip()
        if len(text_output) == 0:
            return ""

        # text_output = text_output.replace("**", "")
        text_output = text_output.replace("\n", " ")
        text_output = " ".join(text_output.split())  # replace multiple whitespaces
        text_output = text_output.strip()
        if len(text_output) == 0:
            return ""

        if not text_output.endswith("."):
            text_output += "."

        return text_output

    def get_prompt_eval(
            self,
            data_item,
            **kwargs
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def get_prompt_ft_analysis(
            self,
            data_item,
            **kwargs
    ) -> Dict[str, Any]:
        raise NotImplementedError

    @staticmethod
    def chunk_text(
            raw_corpus: List[dict],
            save_path: str = "chunk_corpus.jsonl",
            chunk_by: str = "sentence",
            chunk_size: int = 512,  # Maximum number of tokens per chunk (defaults to 512)
            seg_size: Optional[int] = 6,  # None
            stride: Optional[int] = 1,  # None
            use_chonkie: bool = False,
            tokenizer_name_or_path: str = "o200k_base",
            # num_workers: int = 1,  # 4
    ) -> None:
        documents = {}
        # To avoid duplicate pages
        for item in tqdm(raw_corpus):
            title = item["title"]
            text = item["text"]
            if title in documents:
                documents[title] += " " + text
            else:
                documents[title] = text

        logging.info("Start pre-processing...")
        documents = list(documents.items())

        def _basic_process(_title, _text):
            _title = html.unescape(_title)
            _text = html.unescape(_text)
            _text = _text.strip()

            if "(disambiguation)" in _title.lower():
                return None, None
            if "(disambiguation page)" in _title.lower():
                return None, None
            # Take out List/Index/Outline pages (mostly links)
            if re.match(r"(List of .+)|(Index of .+)|(Outline of .+)", _title):
                return None, None
            if _text.startswith("REDIRECT") or _text.startswith("redirect"):
                return None, None
            if _text.endswith(". References."):
                _text = _text[: -len(" References.")].strip()

            _text = re.sub("\{\{cite .*?\}\}", " ", _text, flags=re.DOTALL)
            _text = _text.replace(r"TABLETOREPLACE", " ")
            _text = _text.replace(r"'''", " ")
            _text = _text.replace(r"[[", " ")
            _text = _text.replace(r"]]", " ")
            _text = _text.replace(r"{{", " ")
            _text = _text.replace(r"}}", " ")
            _text = _text.replace("<br>", " ")
            _text = _text.replace("&quot;", '"')
            _text = _text.replace("&amp;", "&")
            _text = _text.replace("& amp;", "&")
            _text = _text.replace("nbsp;", " ")
            _text = _text.replace("formatnum:", "")

            # _text = re.sub('<poem.*?</poem>', ' ', _text, flags=re.DOTALL) # might have useful information?
            _text = re.sub("<math.*?</math>", "", _text, flags=re.DOTALL)
            _text = re.sub("<chem.*?</chem>", "", _text, flags=re.DOTALL)
            _text = re.sub("<score.*?</score>", "", _text, flags=re.DOTALL)

            # clean residual mess from xml dump that shouldn't have made its way here
            _text = re.sub("\| ?item[0-9]?_?style= ?.*? ", " ", _text)
            _text = re.sub("\| ?col[0-9]?_?style= ?.*? ", " ", _text)
            _text = re.sub("\| ?row[0-9]?_?style= ?.*? ", " ", _text)
            _text = re.sub("\| ?style= ?.*? ", " ", _text)
            _text = re.sub("\| ?bodystyle= ?.*? ", " ", _text)
            _text = re.sub("\| ?frame_?style= ?.*? ", " ", _text)
            _text = re.sub("\| ?data_?style= ?.*? ", " ", _text)
            _text = re.sub("\| ?label_?style= ?.*? ", " ", _text)
            _text = re.sub("\| ?headerstyle= ?.*? ", " ", _text)
            _text = re.sub("\| ?list_?style= ?.*? ", " ", _text)
            _text = re.sub("\| ?title_?style= ?.*? ", " ", _text)
            _text = re.sub("\| ?ul_?style= ?.*? ", " ", _text)
            _text = re.sub("\| ?li_?style= ?.*? ", " ", _text)
            _text = re.sub("\| ?border-style= ?.*? ", " ", _text)
            _text = re.sub('\|? ?style=".*?"', "", _text)
            _text = re.sub('\|? ?rowspan=".*?"', "", _text)
            _text = re.sub('\|? ?colspan=".*?"', "", _text)
            _text = re.sub('\|? ?scope=".*?"', "", _text)
            _text = re.sub('\|? ?align=".*?"', "", _text)
            _text = re.sub('\|? ?valign=".*?"', "", _text)
            _text = re.sub('\|? ?lang=".*?"', "", _text)
            _text = re.sub('\|? ?bgcolor=".*?"', "", _text)
            _text = re.sub("\|? ?bg=\#[a-z]+", "", _text)
            _text = re.sub('\|? ?width=".*?"', "", _text)
            _text = re.sub("\|? ?height=[0-9]+", "", _text)
            _text = re.sub("\|? ?width=[0-9]+", "", _text)
            _text = re.sub("\|? ?rowspan=[0-9]+", "", _text)
            _text = re.sub("\|? ?colspan=[0-9]+", "", _text)
            _text = re.sub(r"[\n\t]", " ", _text)
            _text = re.sub("<.*?/>", "", _text)
            _text = re.sub("\|? ?align=[a-z]+", "", _text)
            _text = re.sub("\|? ?valign=[a-z]+", "", _text)
            _text = re.sub("\|? ?scope=[a-z]+", "", _text)
            _text = re.sub("&lt;ref&gt;.*?&lt;/ref&gt;", " ", _text)
            _text = re.sub("&lt;.*?&gt;", " ", _text)
            _text = re.sub("File:[A-Za-z0-9 ]+\.[a-z]{3,4}(\|[0-9]+px)?", "", _text)
            _text = re.sub("Source: \[.*?\]", "", _text)
            _text = _text.replace("Country flag|", "country:")
            _text = _text.replace("flag|", "country:")
            _text = _text.replace("flagicon|", "country:")
            _text = _text.replace("flagcountry|", "country:")
            _text = _text.replace("Flagu|", "country:")
            _text = _text.replace("display=inline", "")
            _text = _text.replace("display=it", "")
            _text = _text.replace("abbr=on", "")
            _text = _text.replace("disp=table", "")

            _title = _title.replace("\n", " ").replace("\t", " ")

            return _title, _text

        # def _split_list(lst, n):
        #     """Split a list into n roughly equal parts."""
        #     k, m = divmod(len(lst), n)
        #     return [lst[_i * k + min(_i, m): (_i + 1) * k + min(_i + 1, m)] for _i in range(n)]

        def _single_worker(_docs):
            _results = []
            for _item in tqdm(_docs):
                _title, _text = _basic_process(_item[0], _item[1])
                if _title is None:
                    continue
                _title = f'"{_title}"'
                _results.append((_title, _text))
            return _results

        # with Pool(processes=num_workers) as p:
        #     result_list = list(p.imap(_single_worker, _split_list(documents, num_workers)))
        # result_list = sum(result_list, [])

        result_list = _single_worker(documents)

        all_title = [item[0] for item in result_list]
        all_text = [item[1] for item in result_list]

        logging.info("Start chunking...")
        idx = 0
        clean_corpus = []

        if use_chonkie:
            logging.info("Using Chonkie chunker...")

            import chonkie

            # Initialize a Chonkie chunker, based on the chunk_by argument
            if chunk_by == "token":
                chunker = chonkie.TokenChunker(
                    tokenizer=tokenizer_name_or_path, chunk_size=chunk_size)
            elif chunk_by == "sentence":
                chunker = chonkie.SentenceChunker(
                    tokenizer_or_token_counter=tokenizer_name_or_path, chunk_size=chunk_size)
            elif chunk_by == "recursive":
                chunker = chonkie.RecursiveChunker(
                    tokenizer_or_token_counter=tokenizer_name_or_path, chunk_size=chunk_size,
                    min_characters_per_chunk=1
                )
            elif chunk_by == "100w":
                chunker = chonkie.TokenChunker(tokenizer="word", chunk_size=100)
            else:
                raise ValueError(f"Invalid chunking method: {chunk_by}")

            # Chunk the text into segments, with chunker
            for title, text in tqdm(zip(all_title, all_text), total=len(all_text)):
                chunks = chunker.chunk(text)
                for chunk in chunks:
                    clean_corpus.append({"title": title, "text": chunk.text})
        else:
            logging.info("Using default chunker...")

            assert chunk_by in ["100w", "sentence"], "Only supports sentence and 100w chunking without chonkie!"
            import spacy

            nlp = spacy.load("en_core_web_lg")

            if chunk_by == "sentence":
                for doc in tqdm(nlp.pipe(all_text, n_process=1, batch_size=2000),
                                total=len(all_text)):  # n_process=num_workers
                    title = all_title[idx]
                    idx += 1
                    sentences = [sent.text.strip() for sent in doc.sents]
                    segments = []
                    for i in range(0, len(sentences), stride):
                        segment = " ".join(sentences[i: i + seg_size])
                        segments.append(segment)
                        if i + seg_size >= len(sentences):
                            break
                    for segment in segments:
                        text = segment.replace("\n", " ").replace("\t", " ")
                        clean_corpus.append({"title": title, "text": text})

            elif chunk_by == "100w":
                for doc in tqdm(nlp.pipe(all_text, n_process=1, batch_size=2000),
                                total=len(all_text)):  # n_process=num_workers
                    title = all_title[idx]
                    idx += 1
                    segments = []
                    word_count = 0
                    segment_tokens = []
                    for token in doc:
                        segment_tokens.append(token.text_with_ws)
                        if not token.is_space and not token.is_punct:
                            word_count += 1
                            if word_count == 100:
                                word_count = 0
                                segments.append("".join([token for token in segment_tokens]))
                                segment_tokens = []
                    if word_count != 0:
                        for token in doc:
                            segment_tokens.append(token.text_with_ws)
                            if not token.is_space and not token.is_punct:
                                word_count += 1
                                if word_count == 100:
                                    word_count = 0
                                    segments.append("".join([token for token in segment_tokens]))
                                    break
                    if word_count != 0:
                        segments.append("".join([token for token in segment_tokens]))

                    for segment in segments:
                        text = segment.replace("\n", " ").replace("\t", " ")
                        clean_corpus.append({"title": title, "text": text})

        logging.info("Start saving corpus...")
        with open(save_path, "w", encoding="utf-8") as f:
            for idx, item in enumerate(clean_corpus):
                title = f"\"{item['title']}\""
                item = {"id": idx, "title": title, "text": item["text"], "contents": item["text"]}
                f.write(json.dumps(item) + "\n")
        logging.info(f"Finish! (#items in clean_corpus: {len(clean_corpus)})")

    def token_stat_eval(
            self,
    ) -> None:
        if self.tokenizer is None:
            self.logger.info(f">>> [token_stat_eval] SKIP: task_name = {self.task_name} because tokenizer is None.\n")
            return

        # Load the dataset and do statistics
        cur_dataset = self.load_task()
        # for task_split in ["train", "valid", "eval"]:
        for task_split in ["eval"]:
            if task_split not in cur_dataset or len(cur_dataset[task_split]) == 0:
                self.logger.info(f">>> [Skip] `{task_split}` split is empty or it does not exist.")
                continue

            self.logger.info(f">>> task_split: {task_split}")
            ds_list = cur_dataset[task_split]  # List[dict]

            # Deal with each sub-task
            all_stat = {}
            all_len_token = []
            skip_cnt = 0
            for ds_dict in ds_list:
                ds_name, subset = ds_dict["dataset_name"], ds_dict["subset_name"]
                cur_split, cur_ds_obj = ds_dict["split_name"], ds_dict["dataset"]
                assert isinstance(cur_ds_obj, Dataset) or isinstance(cur_ds_obj, list)
                len_dataset = len(cur_ds_obj)
                assert isinstance(ds_name, str) and len(ds_name) > 0
                if isinstance(subset, str) and len(subset) > 0:
                    ds_id = f"{ds_name}---{subset}"
                else:
                    ds_id = ds_name
                if self.verbose:
                    self.logger.info(f">>> [Dataset: {ds_id}] split: {cur_split} [# Items = {len_dataset}]")

                if "options" in ds_dict and isinstance(ds_dict["options"], list):
                    ds_options = ds_dict["options"]
                else:
                    ds_options = []

                # Statistics on each data item
                cur_stat = []
                show_cnt = 10000
                for idx, data_item in enumerate(cur_ds_obj):
                    assert isinstance(data_item, dict)
                    data_item["__ds_options"] = ds_options
                    prompt_dict = self.get_prompt_eval(ds_name=ds_name, subset=subset, data_item=data_item)
                    # prompt_dict = self.set_prompt_intent_label(ds_name=ds_name, subset=subset, data_item=data_item)
                    # prompt_dict = self.set_prompt_train(ds_name=ds_name, subset=subset, data_item=data_item)

                    if not isinstance(prompt_dict, dict) or len(prompt_dict) == 0:
                        skip_cnt += 1
                    else:
                        cur_prompt = self.tokenizer.apply_chat_template(
                            prompt_dict["dialog"],
                            tokenize=False,
                            padding=False,
                            add_generation_prompt=True,
                            return_tensors=None
                        )
                        assert isinstance(cur_prompt, str)
                        # Note: we can append extra answer-triggering prompts, e.g., "Answer: Let's think step by step."

                        input_ids = self.tokenizer(
                            cur_prompt,
                            padding=True,  # truncation=True, max_length=1024
                            return_tensors="pt",
                        )
                        len_input = input_ids.data["input_ids"].size(-1)

                        cur_stat.append({
                            "prompt": cur_prompt,
                            "len_char": len(cur_prompt),
                            "len_token": len_input,
                        })
                        all_len_token.append(len_input)

                    if (idx + 1) % show_cnt == 0:
                        self.logger.info(f">>> >>> Progress: {idx + 1} / {len_dataset}")

                all_stat[ds_id] = cur_stat

            # Show logs
            assert len(all_len_token) > 0
            # avg_len_token = sum(all_len_token) / len(all_len_token)
            avg_len_token = float(np.mean(all_len_token))
            std_len_token = float(np.std(all_len_token))
            self.logger.info(
                f">>> Token Stat: [task_split = {task_split}] "
                f">>> #Sub-Tasks = {len(all_stat)}; #Items = {len(all_len_token)}; "
                f"avg_len_token: {avg_len_token:.1f}; std_len_token: {std_len_token:.1f}; "
                f"skip_cnt = {skip_cnt}\n"
            )


def main(
        cache_dir: Optional[str] = None,
        project_root_dir: Optional[str] = None,
        seed: int = 42,
        verbose: bool = False,
        **kwargs
) -> None:
    """
    :param cache_dir: The root directory of the cache.
    :param project_root_dir: The directory of the project root.
    :param seed: Random seed of all modules.
    :param verbose: Verbose mode: show logs.
    :return: None.
    """

    timer_start = time.perf_counter()

    # Setups
    logger = logger_setup("Eval_Tasks")
    random_setup(seed=seed, has_cuda=False)

    if isinstance(kwargs, dict):
        logger.info(f">>> Extra parameters in kwargs: {kwargs}\n")

    # Evaluation
    from tasks.task_mcqa.mmlu import TaskMmlu
    from tasks.task_mcqa.bbh import TaskBbh
    from tasks.task_mcqa.gpqa import TaskGPQA
    from tasks.task_open_qa.trivia_qa import TaskTriviaQA
    from tasks.task_open_qa.bamboogle import TaskBamboogle
    from tasks.task_math.amc_aime import TaskAmcAime
    from tasks.task_code.mbpp import TaskMBPP
    from tasks.task_code.humaneval import TaskHumanEval

    # Training
    from tasks.task_training.mmlu import TaskTrainingMmlu
    from tasks.task_training.trivia_qa import TaskTrainingTriviaQA

    # # Dataset loading
    # eval_tasks = TaskTrainingMmlu(
    #     verbose=verbose, logger=logger, cache_dir=cache_dir, project_root_dir=project_root_dir)
    # # cur_dataset = eval_tasks.load_task()
    # eval_tasks.token_stat_eval()

    # Token length statistics
    all_eval = [TaskMmlu, TaskBbh, TaskGPQA, TaskTriviaQA,
                TaskBamboogle, TaskAmcAime, TaskMBPP, TaskHumanEval]
    all_train = [TaskTrainingMmlu, TaskTrainingTriviaQA]
    for eval_class in all_eval + all_train:
        eval_tasks = eval_class(
            verbose=verbose, logger=logger, cache_dir=cache_dir, project_root_dir=project_root_dir,
        )
        eval_tasks.token_stat_eval()

    timer_end = time.perf_counter()
    logger.info("Total Running Time: %.1f sec (%.1f min)" % (timer_end - timer_start, (timer_end - timer_start) / 60))


if __name__ == "__main__":
    fire.Fire(main)
