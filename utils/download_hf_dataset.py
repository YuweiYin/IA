#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
from typing import Optional

import fire

from datasets import load_dataset
from huggingface_hub import login

from utils.init_functions import logger_setup


def main(
        hf_token: Optional[str] = None,
        hf_id: Optional[str] = None,
        subset: Optional[str] = None,
        cache_dir: Optional[str] = None,
        # trust_remote_code: bool = False,
        # verbose: bool = False,
        **kwargs
) -> None:
    """
    Download Hugging Face datasets to the local cache directory.

    :param hf_token: HuggingFace access token for login. https://huggingface.co/settings/tokens
    :param hf_id: HuggingFace dataset id (ORGANIZATION_NAME/DATASET_NAME), e.g., "allenai/ai2_arc"
    :param subset: Name of the dataset subset, e.g., "ARC-Easy" or "ARC-Challenge"
    :param cache_dir: The root directory of the cache, e.g., "${HOME}/.cache/huggingface/"
    # :param trust_remote_code: Whether to allow for datasets defined on the Hub using a dataset script.
    # :param verbose: Verbose mode: show logs.
    :return: None.
    """

    if isinstance(kwargs, dict):
        print(f">>> Extra parameters in kwargs: {kwargs}")

    timer_start = time.perf_counter()

    # Logger setup
    logger = logger_setup("DOWNLOAD")

    # Parameter checking
    if not isinstance(hf_id, str):
        raise ValueError(f"ValueError: Please specify --hf_id")
    if (cache_dir is None) or (not os.path.isdir(cache_dir)):
        raise ValueError(f"ValueError: Please specify a valid --cache_dir")

    # Hugging Face login
    if isinstance(hf_token, str) and len(hf_token) > 0:
        login(token=hf_token)

    # Cache directory setup
    cache_dir = os.path.join(cache_dir, "datasets")
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
    os.environ["HF_HOME"] = cache_dir
    logger.info(f">>> cache_dir: {cache_dir}")

    if isinstance(subset, str):
        subset = subset.strip()
        if ";" in subset:
            subset = subset.split(";")
        else:
            subset = [subset]
        subset = [_ds_name.strip() for _ds_name in subset]
    else:
        subset = [None]

    # Download the dataset
    for _ds_name in subset:
        dataset = load_dataset(
            hf_id,
            _ds_name,
            cache_dir=cache_dir,
            # trust_remote_code=trust_remote_code
        )
        logger.info(f">>> [{hf_id} --- {_ds_name}] len(dataset) = {len(dataset)}")

    timer_end = time.perf_counter()
    logger.info("Total Running Time: %.1f sec (%.1f min)" % (timer_end - timer_start, (timer_end - timer_start) / 60))


if __name__ == "__main__":
    fire.Fire(main)
