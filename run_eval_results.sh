#!/bin/bash

PARAM=$1

IFS=";"  # Set ";" as the delimiter
read -ra PARAM_ARRAY <<< "${PARAM}"
idx=0
for val in "${PARAM_ARRAY[@]}";
do
  idx=$(( $((idx)) + 1 ))
  if [[ "${idx}" == "1" ]]; then
    EVAL_TASKS=${val}
  elif [[ "${idx}" == "2" ]]; then
    MODEL_NAME=${val}
  elif [[ "${idx}" == "3" ]]; then
    SEED=${val}
  elif [[ "${idx}" == "4" ]]; then
    BSZ=${val}
  fi
done

if [[ -z ${EVAL_TASKS} ]]; then
  echo -e "!!! Error EVAL_TASKS input: \"${EVAL_TASKS}\"\n"
  exit 1
fi

if [[ -z ${MODEL_NAME} ]]; then
  echo -e "!!! Error MODEL_NAME input: \"${MODEL_NAME}\"\n"
  exit 1
fi

if [[ -z ${SEED} ]]; then
  SEED=42
fi

if [[ -z ${BSZ} ]]; then
  BSZ="1"
fi

CACHE_DIR=$2
PROJECT_ROOT_DIR=$3
OUTPUT_DIR=$4
if [[ -z ${CACHE_DIR} ]]; then
  CACHE_DIR="${HOME}/.cache/huggingface"
fi
if [[ -z ${PROJECT_ROOT_DIR} ]]; then
  PROJECT_ROOT_DIR=""
fi
if [[ -z ${OUTPUT_DIR} ]]; then
  OUTPUT_DIR="results/open_llm"
fi
echo -e "CACHE_DIR: ${CACHE_DIR}"
echo -e "PROJECT_ROOT_DIR: ${PROJECT_ROOT_DIR}"
echo -e "OUTPUT_DIR: ${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

echo -e "\n\n >>> python3 run_eval_results.py"

echo -e "EVAL_TASKS: ${EVAL_TASKS}"
echo -e "MODEL_NAME SEED: ${MODEL_NAME}"
echo -e "SEED: ${SEED}"
echo -e "BSZ: ${BSZ}"

python3 run_eval_results.py \
  --task_name "${EVAL_TASKS}" \
  --model_name "${MODEL_NAME}" \
  --cache_dir "${CACHE_DIR}" \
  --project_root_dir "${PROJECT_ROOT_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --seed "${SEED}" \
  --bsz "${BSZ}" \
  --overwrite \
  --verbose
