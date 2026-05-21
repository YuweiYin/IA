#!/bin/bash

export HF_ALLOW_CODE_EVAL=1

PARAM=$1
IFS=";"  # Set ";" as the delimiter
read -ra PARAM_ARRAY <<< "${PARAM}"
idx=0
for val in "${PARAM_ARRAY[@]}";
do
  idx=$(( $((idx)) + 1 ))
  if [[ "${idx}" == "1" ]]; then
    MODEL_NAME=${val}
  elif [[ "${idx}" == "2" ]]; then
    LABEL_MODEL_NAME=${val}
  elif [[ "${idx}" == "3" ]]; then
    TRAINING_TASK_NAME=${val}
  elif [[ "${idx}" == "4" ]]; then
    TRAINING_DATA_TYPE=${val}
  elif [[ "${idx}" == "5" ]]; then
    CONFIG_DIR=${val}
  elif [[ "${idx}" == "6" ]]; then
    SEED=${val}
  elif [[ "${idx}" == "7" ]]; then
    MAX_SEQ_LEN=${val}
  elif [[ "${idx}" == "8" ]]; then
    NUM_TRAIN_EPOCHS=${val}
  elif [[ "${idx}" == "9" ]]; then
    LEARNING_RATE=${val}
  elif [[ "${idx}" == "10" ]]; then
    TRAIN_MODE=${val}
  elif [[ "${idx}" == "11" ]]; then
    LORA_MODE=${val}
  elif [[ "${idx}" == "12" ]]; then
    VALID_MODE=${val}
  fi
done

if [[ -z ${MODEL_NAME} ]]; then
  # E.g., "llama3-8b", "qwen2.5-7b"
  echo -e "!!! Error or Empty MODEL_NAME input: \"${MODEL_NAME}\"\n"
  exit 1
fi

if [[ -z ${LABEL_MODEL_NAME} ]]; then
  LABEL_MODEL_NAME="${MODEL_NAME}"
fi

if [[ -z ${TRAINING_TASK_NAME} ]]; then
  TRAINING_TASK_NAME="mmlu_training"
fi

if [[ -z ${TRAINING_DATA_TYPE} ]]; then
  # E.g., "raw", "da", and "ia"
  TRAINING_DATA_TYPE="ia"
fi

if [[ -z ${CONFIG_DIR} ]]; then
  CONFIG_DIR="config/ft"
fi

if [[ -z ${SEED} ]]; then
  SEED=42
fi

if [[ -z ${MAX_SEQ_LEN} ]]; then
  MAX_SEQ_LEN="4096"
fi

if [[ -z ${NUM_TRAIN_EPOCHS} ]]; then
  NUM_TRAIN_EPOCHS="1.0"
fi

if [[ -z ${LEARNING_RATE} ]]; then
  LEARNING_RATE="5e-05"
fi

if [[ -z ${TRAIN_MODE} ]]; then
  # Note: use_wandb, use_lora, and debug
  TRAIN_MODE="0,0,0"
fi

if [[ -z ${LORA_MODE} ]]; then
  # Note: the three numbers are LoRA rank, alpha, and dropout values
  LORA_MODE="16,16,0.0"
fi

if [[ -z ${VALID_MODE} ]]; then
  # Note: the size of valid set, validation batch size, and valid_on_start (boolean)
  VALID_MODE="100,1,0"
fi


DIR=$2
IFS=";"  # Set ";" as the delimiter
read -ra DIR_ARRAY <<< "${DIR}"
idx=0
for val in "${DIR_ARRAY[@]}";
do
  idx=$(( $((idx)) + 1 ))
  if [[ "${idx}" == "1" ]]; then
    CACHE_DIR=${val}
  elif [[ "${idx}" == "2" ]]; then
    PROJECT_ROOT_DIR=${val}
  elif [[ "${idx}" == "3" ]]; then
    CKPT_ROOT_DIR=${val}
  elif [[ "${idx}" == "4" ]]; then
    TRAINING_DATA_DIR=${val}
  elif [[ "${idx}" == "5" ]]; then
    RESUME_CKPT_DIR=${val}
  fi
done

if [[ -z ${CACHE_DIR} ]]; then
  CACHE_DIR="${HOME}/.cache/huggingface"
fi
if [[ -z ${PROJECT_ROOT_DIR} ]]; then
  PROJECT_ROOT_DIR=""
fi
if [[ -z ${CKPT_ROOT_DIR} ]]; then
  CKPT_ROOT_DIR="${PROJECT_ROOT_DIR}/ckpt"
fi
if [[ -z ${TRAINING_DATA_DIR} ]]; then
  echo -e "!!! Empty TRAINING_DATA_DIR input: \"${TRAINING_DATA_DIR}\"\n"
  exit 1
fi
if [[ -z ${RESUME_CKPT_DIR} ]]; then
  RESUME_CKPT_DIR=""
fi


WANDB_KEY=$3
if [[ -z ${WANDB_KEY} ]]; then
  WANDB_KEY="${WANDB_API_KEY}"
fi

echo -e "\n\n >>> python3 run_train_sft_unsloth.py"

echo -e "MODEL_NAME: ${MODEL_NAME}"
echo -e "LABEL_MODEL_NAME: ${LABEL_MODEL_NAME}"
echo -e "TRAINING_TASK_NAME: ${TRAINING_TASK_NAME}"
echo -e "TRAINING_DATA_TYPE: ${TRAINING_DATA_TYPE}"
echo -e "TRAINING_DATA_DIR: ${TRAINING_DATA_DIR}"
echo -e "CONFIG_DIR: ${CONFIG_DIR}"
echo -e "SEED: ${SEED}"
echo -e "MAX_SEQ_LEN: ${MAX_SEQ_LEN}"
echo -e "NUM_TRAIN_EPOCHS: ${NUM_TRAIN_EPOCHS}"
echo -e "LEARNING_RATE: ${LEARNING_RATE}"

echo -e "TRAIN_MODE: ${TRAIN_MODE}"
echo -e "LORA_MODE: ${LORA_MODE}"
echo -e "VALID_MODE: ${VALID_MODE}"

echo -e "CACHE_DIR: ${CACHE_DIR}"
echo -e "PROJECT_ROOT_DIR: ${PROJECT_ROOT_DIR}"
echo -e "CKPT_ROOT_DIR: ${CKPT_ROOT_DIR}"
echo -e "RESUME_CKPT_DIR: ${RESUME_CKPT_DIR}"

python3 run_train_sft_unsloth.py \
  --cache_dir "${CACHE_DIR}" \
  --project_root_dir "${PROJECT_ROOT_DIR}" \
  --ckpt_root_dir "${CKPT_ROOT_DIR}" \
  --resume_ckpt_dir "${RESUME_CKPT_DIR}" \
  --wandb_key "${WANDB_KEY}" \
  --config_dir "${CONFIG_DIR}" \
  --seed "${SEED}" \
  --model_name "${MODEL_NAME}" \
  --label_model_name "${LABEL_MODEL_NAME}" \
  --training_task_name "${TRAINING_TASK_NAME}" \
  --training_data_type "${TRAINING_DATA_TYPE}" \
  --training_data_dir "${TRAINING_DATA_DIR}" \
  --max_seq_len "${MAX_SEQ_LEN}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --learning_rate "${LEARNING_RATE}" \
  --train_mode "${TRAIN_MODE}" \
  --lora_mode "${LORA_MODE}" \
  --valid_mode "${VALID_MODE}" \
  --verbose
