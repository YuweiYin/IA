#!/bin/bash

CACHE_DIR=$1
if [[ -z ${CACHE_DIR} ]]; then
  CACHE_DIR="${HOME}/.cache/huggingface/"
fi

HF_TOKEN=$2
if [[ -z ${HF_TOKEN} ]]; then
  HF_TOKEN="YOUR_HF_TOKEN"  # https://huggingface.co/settings/tokens
fi


python3 utils/download_hf_model.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" \
  --trust_remote_code --verbose --hf_id "meta-llama/Llama-3.1-8B-Instruct"

python3 utils/download_hf_model.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" \
  --trust_remote_code --verbose --hf_id "meta-llama/Llama-3.1-8B"
python3 utils/download_hf_model.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" \
  --trust_remote_code --verbose --hf_id "allenai/Llama-3.1-Tulu-3-8B-SFT"
python3 utils/download_hf_model.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" \
  --trust_remote_code --verbose --hf_id "allenai/Llama-3.1-Tulu-3-8B-DPO"
python3 utils/download_hf_model.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" \
  --trust_remote_code --verbose --hf_id "allenai/Llama-3.1-Tulu-3-8B"

python3 utils/download_hf_model.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" \
  --trust_remote_code --verbose --hf_id "mistralai/Mistral-7B-Instruct-v0.3"
python3 utils/download_hf_model.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" \
  --trust_remote_code --verbose --hf_id "tiiuae/Falcon3-7B-Instruct"
python3 utils/download_hf_model.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" \
  --trust_remote_code --verbose --hf_id "Qwen/Qwen2.5-7B-Instruct"
python3 utils/download_hf_model.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" \
  --trust_remote_code --verbose --hf_id "Qwen/Qwen3-32B"

python3 utils/download_hf_model.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" \
  --trust_remote_code --verbose --hf_id "hbx/Mistral-Interact"
