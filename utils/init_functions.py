# -*- coding: utf-8 -*-

import os
import random
import logging
from typing import Optional

import numpy as np
import torch


def logger_setup(
        name: Optional[str] = None,
) -> logging.Logger:
    """
    Logger setup.

    :param name: The name of the logger.
    :return: logging.Logger object.
    """

    logging.basicConfig(
        format="[%(asctime)s - %(levelname)s - %(name)s] -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
    )
    logger = logging.getLogger(name)

    return logger


def cuda_setup(
        cuda: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        verbose: bool = False,
) -> dict:
    """
    CUDA GPU setup.

    :param cuda: CUDA device(s), e.g., "0" OR "0,1".
    :param logger: The logger object.
    :param verbose: Verbose mode: show logs.
    :return: CUDA info dict.
    """

    if cuda is None or not isinstance(cuda, str):
        # Use CPU or all available GPUs
        has_cuda = torch.cuda.is_available()
        device_count = int(torch.cuda.device_count())
        return {
            "has_cuda": has_cuda,
            "device": torch.device("cuda" if has_cuda else "cpu"),
            "device_count": device_count,
            "gpus": list(range(device_count)),
            "ddp_able": has_cuda and device_count > 1,
        }

    assert isinstance(cuda, str), f"AssertionError: cuda_setup --- cuda = {cuda}"
    assert isinstance(logger, logging.Logger), f"AssertionError: cuda_setup --- type(logger) = {type(logger)}"

    # CUDA GPU setup (specify GPU device ID)
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda
    has_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if has_cuda else "cpu")
    gpus = cuda.split(",") if "," in cuda else [cuda]
    gpus = [int(gpu_id) for gpu_id in gpus]
    device_count = int(torch.cuda.device_count())
    ddp_able = has_cuda and len(gpus) > 1 and device_count > 1
    if verbose:
        logger.info(
            f">>> HAS_CUDA: {has_cuda}; DEVICE: {device}; "
            f"device_count: {device_count}; gpus: {gpus}; DDP able: {ddp_able}")
        logger.info(f">>> torch.__version__: {torch.__version__}")
        logger.info(f">>> torch.version.cuda: {torch.version.cuda}")
        logger.info(f">>> torch.cuda.is_available(): {torch.cuda.is_available()}")
        logger.info(f">>> torch.cuda.device_count(): {torch.cuda.device_count()}")
        logger.info(f">>> torch.backends.cudnn.version(): {torch.backends.cudnn.version()}")
        logger.info(f">>> torch.cuda.get_arch_list(): {torch.cuda.get_arch_list()}")
        if has_cuda:
            logger.info(f">>> torch.cuda.current_device(): {torch.cuda.current_device()}")
            logger.info(f">>> torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}")

    cuda_dict = {
        "has_cuda": has_cuda,
        "device": device,
        "device_count": device_count,
        "gpus": gpus,
        "ddp_able": ddp_able,
    }

    return cuda_dict


def random_setup(
        seed: Optional[int] = None,
        has_cuda: bool = False,
) -> None:
    """
    Set the random seed of all modules.

    :param seed: The name of the logger.
    :param has_cuda: True if CUDA is available.
    :return: logging.Logger object.
    """

    assert isinstance(seed, int), f"AssertionError: random_setup"

    random.seed(seed)
    np.random.seed(seed)
    if has_cuda:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
