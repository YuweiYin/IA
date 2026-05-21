#!/bin/bash

HF_ID=$1
if [[ -z ${HF_ID} ]]; then
  # E.g., "meta-llama/Llama-3.1-8B-Instruct"
  echo -e "!!! Error or Empty HF_ID input: \"${HF_ID}\"\n"
  exit 1
fi

CACHE_DIR=$2
if [[ -z ${CACHE_DIR} ]]; then
  CACHE_DIR="${HOME}/.cache/huggingface/"
fi

YOUR_HF_TOKEN=$3
if [[ -z ${YOUR_HF_TOKEN} ]]; then
  YOUR_HF_TOKEN="${HF_TOKEN}"  # https://huggingface.co/settings/tokens
fi

python3 utils/download_hf_model.py \
  --trust_remote_code --verbose \
  --cache_dir "${CACHE_DIR}" \
  --hf_token "${YOUR_HF_TOKEN}" \
  --hf_id "${HF_ID}"
