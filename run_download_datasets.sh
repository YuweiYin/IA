#!/bin/bash

CACHE_DIR=$1
if [[ -z ${CACHE_DIR} ]]; then
  CACHE_DIR="${HOME}/.cache/huggingface/"
fi

HF_TOKEN=$2
if [[ -z ${HF_TOKEN} ]]; then
  HF_TOKEN="YOUR_HF_TOKEN"  # https://huggingface.co/settings/tokens
fi

python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" \
  --hf_id "cais/mmlu" --subset "all"  # MMLU
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" \
  --hf_id "mandarjoshi/trivia_qa" --subset "rc.wikipedia"  # TriviaQA - wikipedia
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" \
  --hf_id "RUC-NLPIR/FlashRAG_datasets" --subset "bamboogle"  # Bamboogle
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" \
  --hf_id "google-research-datasets/mbpp" --subset "full"  # MBPP
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" \
  --hf_id "Idavidrein/gpqa" --subset "gpqa_diamond"  # GPQA

# Big-Bench Hard (BBH)
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" \
  --hf_id "lukaemon/bbh" --subset "word_sorting"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" \
  --hf_id "lukaemon/bbh" --subset "web_of_lies"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" \
  --hf_id "lukaemon/bbh" --subset "tracking_shuffled_objects_three_objects"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" \
  --hf_id "lukaemon/bbh" --subset "tracking_shuffled_objects_seven_objects"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" \
  --hf_id "lukaemon/bbh" --subset "tracking_shuffled_objects_five_objects"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" \
  --hf_id "lukaemon/bbh" --subset "temporal_sequences"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" \
  --hf_id "lukaemon/bbh" --subset "sports_understanding"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" \
  --hf_id "lukaemon/bbh" --subset "snarks"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" \
  --hf_id "lukaemon/bbh" --subset "salient_translation_error_detection"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" \
  --hf_id "lukaemon/bbh" --subset "ruin_names"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" \
  --hf_id "lukaemon/bbh" --subset "reasoning_about_colored_objects"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" \
  --hf_id "lukaemon/bbh" --subset "penguins_in_a_table"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" \
  --hf_id "lukaemon/bbh" --subset "object_counting"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" \
  --hf_id "lukaemon/bbh" --subset "navigate"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" \
  --hf_id "lukaemon/bbh" --subset "multistep_arithmetic_two"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" \
  --hf_id "lukaemon/bbh" --subset "movie_recommendation"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" \
  --hf_id "lukaemon/bbh" --subset "logical_deduction_three_objects"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" \
  --hf_id "lukaemon/bbh" --subset "logical_deduction_seven_objects"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" \
  --hf_id "lukaemon/bbh" --subset "logical_deduction_five_objects"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" \
  --hf_id "lukaemon/bbh" --subset "hyperbaton"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" \
  --hf_id "lukaemon/bbh" --subset "geometric_shapes"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" \
  --hf_id "lukaemon/bbh" --subset "formal_fallacies"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" \
  --hf_id "lukaemon/bbh" --subset "dyck_languages"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" \
  --hf_id "lukaemon/bbh" --subset "disambiguation_qa"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" \
  --hf_id "lukaemon/bbh" --subset "date_understanding"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" \
  --hf_id "lukaemon/bbh" --subset "causal_judgement"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" \
  --hf_id "lukaemon/bbh" --subset "boolean_expressions"
