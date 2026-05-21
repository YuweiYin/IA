# -*- coding: utf-8 -*-

import os
import csv
import sys
import json
import logging
from typing import Optional, Union
import numpy as np
import pandas as pd
import torch


class DataIO:

    def __init__(self):
        pass

    @staticmethod
    def handle_non_serializable(o):
        if isinstance(o, np.int64) or isinstance(o, np.int32):
            return int(o)
        elif isinstance(o, set):
            return list(o)
        else:
            return str(o)

    @staticmethod
    def load_json(
            filepath: str,
            mode: str = "r",
            encoding: str = "utf-8",
            verbose: bool = False,
            errors: Optional[str] = None,
    ) -> Union[list, dict]:
        results = []
        if os.path.isfile(filepath):
            if verbose:
                logging.info(f">>> [load_json] {filepath}")
            with open(filepath, mode, encoding=encoding, errors=errors) as fp_in:
                results = json.load(fp_in)
        else:
            if verbose:
                logging.info(f">>> [load_json] filepath does not exist: {filepath}")
        return results

    @staticmethod
    def save_json(
            filepath: str,
            data,
            mode: str = "w",
            encoding: str = "utf-8",
            indent: Optional[int] = None,
            verbose: bool = False,
    ) -> None:
        if verbose:
            logging.info(f">>> [save_json] {filepath}")
        with open(filepath, mode, encoding=encoding) as fp_out:
            json.dump(data, fp_out, indent=indent, ensure_ascii=True, default=DataIO.handle_non_serializable)

    @staticmethod
    def load_jsonl(
            filepath: str,
            mode: str = "r",
            encoding: str = "utf-8",
            verbose: bool = False,
            errors: Optional[str] = None,
    ) -> list:
        results = []
        if os.path.isfile(filepath):
            if verbose:
                logging.info(f">>> [load_jsonl] {filepath}")
            with open(filepath, mode, encoding=encoding, errors=errors) as fp_in:
                for line in fp_in:
                    results.append(json.loads(line))
        else:
            if verbose:
                logging.info(f">>> [load_jsonl] filepath does not exist: {filepath}")
        return results

    @staticmethod
    def save_jsonl(
            filepath: str,
            data: list,
            mode: str = "w",
            encoding: str = "utf-8",
            verbose: bool = False,
    ) -> None:
        if verbose:
            logging.info(f">>> [save_jsonl] {filepath}")
        with open(filepath, mode, encoding=encoding) as fp_out:
            for data_item in data:
                fp_out.write(json.dumps(data_item) + "\n")

    @staticmethod
    def show_dict(
            input_dict: dict,
            dict_name: str = "",
            logger=None,
    ):
        if logger is None:
            logger = logging.getLogger("DataIO")
        if dict_name == "":
            dict_name = "show_dict"

        if isinstance(input_dict, dict) and len(input_dict) > 0:
            for k, v in input_dict.items():
                logger.info(f">>> >>> [{dict_name}] {k}: {v}")
        else:
            logger.info(f">>> >>> [show_dict] input_dict is not dict or is empty.")

    @staticmethod
    def load_csv(
            filepath: str,
            delimiter=",",
            mode: str = "r",
            encoding: str = "utf-8",
            verbose: bool = False,
            errors: Optional[str] = None,
    ) -> Union[list, dict]:
        csv.field_size_limit(sys.maxsize)

        results = []
        if os.path.isfile(filepath):
            if verbose:
                logging.info(f">>> [load_csv] {filepath}")
            with open(filepath, mode, encoding=encoding, errors=errors) as fp_in:
                csv_reader = csv.reader(fp_in, delimiter=delimiter)
                results = [row for row in csv_reader]
        else:
            if verbose:
                logging.info(f">>> [load_csv] filepath does not exist: {filepath}")
        return results

    @staticmethod
    def load_parquet(
            filepath: str,
            engine="auto",  # "auto", "pyarrow", "fastparquet"
            # mode: str = "r",
            # encoding: str = "utf-8",
            verbose: bool = False,
            # errors: Optional[str] = None,
    ) -> pd.DataFrame:
        results = []
        if os.path.isfile(filepath):
            if verbose:
                logging.info(f">>> [load_parquet] {filepath}")
            # with open(filepath, mode, encoding=encoding, errors=errors) as fp_in:
            #     pass
            # Load a parquet object from the file path, returning a DataFrame.
            results = pd.read_parquet(filepath, engine=engine)
        else:
            if verbose:
                logging.info(f">>> [load_parquet] filepath does not exist: {filepath}")
        return results

    @staticmethod
    def save_parquet(
            filepath: str,
            data: list,
            verbose: bool = False,
            # errors: Optional[str] = None,
    ) -> None:
        if verbose:
            logging.info(f">>> [save_parquet] {filepath}")
        # Save the list (as DataFrame) as a parquet object to the file path
        df = pd.DataFrame(data, columns=None)
        df.columns = df.columns.astype(str)
        df.to_parquet(filepath)  # index=False, engine=engine
        return None

    @staticmethod
    def load_pt(
            filepath: str,
            # mode: str = "r",
            # encoding: str = "utf-8",
            verbose: bool = False,
            # errors: Optional[str] = None,
    ) -> Union[list, dict]:
        results = []
        if os.path.isfile(filepath):
            if verbose:
                logging.info(f">>> [load_pt] {filepath}")
            results = torch.load(filepath)
        else:
            if verbose:
                logging.info(f">>> [load_pt] filepath does not exist: {filepath}")
        return results

    @staticmethod
    def load_txt(
            filepath: str,
            mode: str = "r",
            encoding: str = "utf-8",
            verbose: bool = False,
            errors: Optional[str] = None,
    ) -> list:
        results = []
        if os.path.isfile(filepath):
            if verbose:
                logging.info(f">>> [load_txt] {filepath}")
            with open(filepath, mode, encoding=encoding, errors=errors) as fp_in:
                results = [line.strip() for line in fp_in.readlines()]
        else:
            if verbose:
                logging.info(f">>> [load_txt] filepath does not exist: {filepath}")
        return results
