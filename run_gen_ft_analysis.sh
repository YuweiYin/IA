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
  elif [[ "${idx}" == "5" ]]; then
    MAX_NEW_GEN=${val}
  elif [[ "${idx}" == "6" ]]; then
    GEN_TEMP=${val}
  elif [[ "${idx}" == "7" ]]; then
    GEN_TOPP=${val}
  elif [[ "${idx}" == "8" ]]; then
    GEN_CONFIG=${val}
  elif [[ "${idx}" == "9" ]]; then
    GEN_METHOD=${val}
  elif [[ "${idx}" == "10" ]]; then
    NUM_TOTAL=${val}
  elif [[ "${idx}" == "11" ]]; then
    DATA_START_IDX=${val}
  elif [[ "${idx}" == "12" ]]; then
    DATA_END_IDX=${val}
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

if [[ -z ${MAX_NEW_GEN} ]]; then
  MAX_NEW_GEN="2048"
fi

if [[ -z ${GEN_TEMP} ]]; then
  GEN_TEMP="0.0"
fi

if [[ -z ${GEN_TOPP} ]]; then
  GEN_TOPP="-1.0"
fi

if [[ -z ${GEN_CONFIG} ]]; then
  # Note: overwrite, do_bf16, do_4bit, and debug
  GEN_CONFIG="1,0,0,0"
fi

if [[ -z ${GEN_METHOD} ]]; then
  GEN_METHOD="da"
fi

if [[ -z ${NUM_TOTAL} ]]; then
  NUM_TOTAL="-1"
fi

if [[ -z ${DATA_START_IDX} ]]; then
  DATA_START_IDX="-1"
fi

if [[ -z ${DATA_END_IDX} ]]; then
  DATA_END_IDX="-1"
fi

CACHE_DIR=$2
PROJECT_ROOT_DIR=$3
OUTPUT_DIR=$4
MODEL_CKPT_DIR=$5
if [[ -z ${CACHE_DIR} ]]; then
  CACHE_DIR="${HOME}/.cache/huggingface"
fi
if [[ -z ${PROJECT_ROOT_DIR} ]]; then
  PROJECT_ROOT_DIR=""
fi
if [[ -z ${OUTPUT_DIR} ]]; then
  OUTPUT_DIR="results/run_gen_ft_analysis"  # ${MODEL_NAME} is subdir
fi
if [[ -z ${MODEL_CKPT_DIR} ]]; then
  MODEL_CKPT_DIR=""  # the model checkpoint to be loaded
fi
echo -e "CACHE_DIR: ${CACHE_DIR}"
echo -e "PROJECT_ROOT_DIR: ${PROJECT_ROOT_DIR}"
echo -e "OUTPUT_DIR: ${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

echo -e "\n\n >>> python3 run_gen_ft_analysis.py"

echo -e "EVAL_TASKS: ${EVAL_TASKS}"
echo -e "MODEL_NAME: ${MODEL_NAME}"
echo -e "MODEL_CKPT_DIR: ${MODEL_CKPT_DIR}"
echo -e "SEED: ${SEED}"
echo -e "BSZ: ${BSZ}"
echo -e "MAX_NEW_GEN: ${MAX_NEW_GEN}"
echo -e "GEN_TEMP: ${GEN_TEMP}"
echo -e "GEN_TOPP: ${GEN_TOPP}"
echo -e "GEN_CONFIG: ${GEN_CONFIG}"
echo -e "GEN_METHOD: ${GEN_METHOD}"
echo -e "NUM_TOTAL: ${NUM_TOTAL}"
echo -e "DATA_START_IDX: ${DATA_START_IDX}"
echo -e "DATA_END_IDX: ${DATA_END_IDX}"

python3 run_gen_ft_analysis.py \
  --task_name "${EVAL_TASKS}" \
  --model_name "${MODEL_NAME}" \
  --model_ckpt_dir "${MODEL_CKPT_DIR}" \
  --cache_dir "${CACHE_DIR}" \
  --project_root_dir "${PROJECT_ROOT_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --seed "${SEED}" \
  --bsz "${BSZ}" \
  --max_new_gen "${MAX_NEW_GEN}" \
  --gen_temperature "${GEN_TEMP}" \
  --gen_top_p "${GEN_TOPP}" \
  --gen_config "${GEN_CONFIG}" \
  --gen_method "${GEN_METHOD}" \
  --num_total "${NUM_TOTAL}" \
  --data_start_idx "${DATA_START_IDX}" \
  --data_end_idx "${DATA_END_IDX}" \
  --verbose
