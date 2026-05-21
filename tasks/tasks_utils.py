# -*- coding: utf-8 -*-

from tasks.task_mcqa.mmlu import TaskMmlu
from tasks.task_mcqa.bbh import TaskBbh
from tasks.task_mcqa.gpqa import TaskGPQA
from tasks.task_open_qa.trivia_qa import TaskTriviaQA
from tasks.task_open_qa.bamboogle import TaskBamboogle
from tasks.task_code.mbpp import TaskMBPP
from tasks.task_training.mmlu import TaskTrainingMmlu
from tasks.task_training.trivia_qa import TaskTrainingTriviaQA


TASK_CLASS_DICT = {
    # mcqa
    "bbh": TaskBbh,
    "mmlu": TaskMmlu,
    "gpqa": TaskGPQA,

    # open_qa
    "trivia_qa": TaskTriviaQA,
    "bamboogle": TaskBamboogle,

    # code
    "mbpp": TaskMBPP,

    # training
    "mmlu_training": TaskTrainingMmlu,
    "trivia_qa_training": TaskTrainingTriviaQA,
}

TASK_TYPE_DICT = {
    # mcqa
    "bbh": "mcqa",
    "mmlu": "mcqa",
    "gpqa": "mcqa",

    # open_qa
    "trivia_qa": "open_qa",
    "bamboogle": "open_qa",

    # math
    "amc_aime": "math",

    # code
    "humaneval": "code",
    "mbpp": "code",

    # training
    "mmlu_training": "mcqa",
    "trivia_qa_training": "open_qa",
}

MCQA_CLASS_DICT = {
    "bbh": TaskBbh,
    "mmlu": TaskMmlu,
    "gpqa": TaskGPQA,
}

OPEN_QA_CLASS_DICT = {
    "trivia_qa": TaskTriviaQA,
    "bamboogle": TaskBamboogle,
}

CODE_CLASS_DICT = {
    "mbpp": TaskMBPP,
}

TRAINING_CLASS_DICT = {
    "mmlu_training": TaskTrainingMmlu,
    "trivia_qa_training": TaskTrainingTriviaQA,
}
