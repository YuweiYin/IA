<div align="center">

# Improving Language Models with Intentional Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) &nbsp;
[![arXiv](https://img.shields.io/badge/arXiv-2502.04689-b31b1b.svg)](https://arxiv.org/abs/2502.04689)

</div>

<details open><summary>Paper Abstract</summary>

* **Improving Language Models with Intentional Analysis**
  * **Authors**: [Yuwei Yin](https://www.yuweiyin.com/) and [Giuseppe Carenini](https://www.cs.ubc.ca/~carenini/)
  * **Paper**: https://huggingface.co/papers/2502.04689

```text
Intent, a critical cognitive notion and mental state, is ubiquitous in human communication 
and problem-solving. Accurately understanding the underlying intent behind questions is 
imperative to reasoning towards correct answers. However, this significant concept has 
been largely disregarded in the rapid development of language models (LMs). To unleash 
the potential of intent and instill it into LMs, this paper introduces Intentional Analysis 
(IA), which explicitly invokes intent-aware analysis and reasoning during the problem-solving 
process. Comprehensive experiments across diverse benchmarks, model types, and configurations 
demonstrate the effectiveness, robustness, and generalizability of IA. Notably, IA consistently 
improves task performance even on SOTA proprietary models like GPT-5 and Claude-Opus-4.6. 
Moreover, IA not only outperforms Chain-of-Thought (CoT) across various experimental settings, 
but it can also synergistically work with CoT reasoning. Further qualitative analysis and case 
studies reveal that the benefits of IA stem from addressing several weaknesses in baseline 
methods, such as intent misunderstanding, hasty generalization, and mental laziness. Case 
studies also provide insights into the mechanisms underlying IA and clarify how it differs 
from CoT in mitigating these weaknesses. This study sheds light on a promising direction for 
the development of future LLMs with intentional analysis.
```

</details>


## Development Environments

<details><summary>Environment Setup</summary>

- **Python**: Python 3.10
- **GPU**: NVIDIA CUDA GPU

```bash
# Conda
git clone https://github.com/YuweiYin/IA
cd IA/
# Now, "/path/to/IA/" is the project root directory

# https://docs.conda.io/projects/miniconda/en/latest/
conda create -n ia python=3.10 -y
conda activate ia

# Packages for model inference
pip install -r requirements.txt -i https://pypi.org/simple/
pip install -e . -i https://pypi.org/simple/

# Packages for model training
pip install -r requirements_gpu.txt -i https://pypi.org/simple/
```

</details>


<details><summary>Datasets and Models</summary>

- Download the datasets and models beforehand if the computing nodes have no Internet access or HOME storage is limited.
- Please ensure `CACHE_DIR` and `HF_TOKEN` in the script are correct directories.

```bash
CACHE_DIR="${HOME}/.cache/huggingface/"  # Or your own directory to store datasets and models
#export HF_HOME="${CACHE_DIR}"
HF_TOKEN="YOUR_HF_TOKEN"  # https://huggingface.co/settings/tokens
bash run_download_datasets.sh "${CACHE_DIR}" "${HF_TOKEN}"  # Download data to "${CACHE_DIR}/datasets/"
bash run_download_models.sh "${CACHE_DIR}" "${HF_TOKEN}"  # Download models to "${CACHE_DIR}/"
```

</details>


## Experiments

### Effectiveness of Intentional Analysis

<details><summary>Experiments (Table 2)</summary>

```bash
CACHE_DIR="${HOME}/.cache/huggingface/"  # Or your own directory to store datasets and models
#export HF_HOME="${CACHE_DIR}"
OUTPUT_DIR="results/run_gen_hf"

MODEL="llama3-8b"
SEED=42
BSZ=4  # generation batch size
MAX_NEW_GEN=4096
GEN_TEMP="0.0"
GEN_TOPP="-1.0"
GEN_CONFIG="1,0,0,0"  # overwrite, do_bf16, do_4bit, and debug

for GEN_METHOD in "da" "ana" "ps" "cot" "ia"
do
  for EVAL_TASK_NAME in "mmlu" "trivia_qa" "bamboogle" "bbh" "mbpp"
  do
    # First, run LLM generation
    python3 run_gen_hf.py \
      --task_name "${EVAL_TASK_NAME}" \
      --model_name "${MODEL}" \
      --cache_dir "${CACHE_DIR}" \
      --output_dir "${OUTPUT_DIR}-${GEN_METHOD}_seed${SEED}_temp${GEN_TEMP}" \
      --seed "${SEED}" \
      --bsz "${BSZ}" \
      --max_new_gen "${MAX_NEW_GEN}" \
      --gen_temperature "${GEN_TEMP}" \
      --gen_top_p "${GEN_TOPP}" \
      --gen_config "${GEN_CONFIG}" \
      --gen_method "${GEN_METHOD}" \
      --verbose
    # Then, evaluate the results
    python3 run_eval_results.py \
      --task_name "${EVAL_TASK_NAME}" \
      --model_name "${MODEL}" \
      --cache_dir "${CACHE_DIR}" \
      --output_dir "${OUTPUT_DIR}-${GEN_METHOD}_seed${SEED}_temp${GEN_TEMP}" \
      --seed "${SEED}" \
      --overwrite \
      --verbose
  done
done
```

</details>


### Generalizability of Intentional Analysis

<details><summary>Experiments - Open Models (Table 3)</summary>

```bash
CACHE_DIR="${HOME}/.cache/huggingface/"  # Or your own directory to store datasets and models
#export HF_HOME="${CACHE_DIR}"
OUTPUT_DIR="results/run_gen_hf"

SEED=42
BSZ=4  # generation batch size
MAX_NEW_GEN=4096
GEN_TEMP="0.0"
GEN_TOPP="-1.0"
GEN_CONFIG="1,0,0,0"  # overwrite, do_bf16, do_4bit, and debug

for GEN_METHOD in "da" "cot" "ia"
do
  for EVAL_TASK_NAME in "mmlu" "trivia_qa" "bamboogle" "bbh" "mbpp"
  do
    for MODEL in "llama3-8b-base" "tulu3-8b-sft" "tulu3-8b-dpo" "tulu3-8b-rlvr" "mistral0.3-7b" "falcon3-7b"  # "llama3-8b"
    do
      # First, run LLM generation
      python3 run_gen_hf.py \
        --task_name "${EVAL_TASK_NAME}" \
        --model_name "${MODEL}" \
        --cache_dir "${CACHE_DIR}" \
        --output_dir "${OUTPUT_DIR}-${GEN_METHOD}_seed${SEED}_temp${GEN_TEMP}" \
        --seed "${SEED}" \
        --bsz "${BSZ}" \
        --max_new_gen "${MAX_NEW_GEN}" \
        --gen_temperature "${GEN_TEMP}" \
        --gen_top_p "${GEN_TOPP}" \
        --gen_config "${GEN_CONFIG}" \
        --gen_method "${GEN_METHOD}" \
        --verbose
      # Then, evaluate the results
      python3 run_eval_results.py \
        --task_name "${EVAL_TASK_NAME}" \
        --model_name "${MODEL}" \
        --cache_dir "${CACHE_DIR}" \
        --output_dir "${OUTPUT_DIR}-${GEN_METHOD}_seed${SEED}_temp${GEN_TEMP}" \
        --seed "${SEED}" \
        --overwrite \
        --verbose
    done
  done
done
```

</details>


<details><summary>Experiments - Proprietary LLMs (Table 4)</summary>

```bash
# Set GenAI API keys as environment variables
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"  # https://platform.openai.com/settings/organization/api-keys
export GEMINI_API_KEY="YOUR_GEMINI_API_KEY"  # https://aistudio.google.com/app/apikey
export ANTHROPIC_API_KEY="YOUR_ANTHROPIC_API_KEY"  # https://platform.claude.com/docs/en/api/admin/api_keys/retrieve

CACHE_DIR="${HOME}/.cache/huggingface/"  # Or your own directory to store datasets and models
#export HF_HOME="${CACHE_DIR}"
OUTPUT_DIR="results/run_gen_api"

SEED=42
#BSZ=1  # generation batch size
MAX_NEW_GEN=4096
GEN_TEMP="1.0"

for GEN_METHOD in "da" "cot" "ia"
do
  for EVAL_TASK_NAME in "bamboogle" "gpqa"
  do
    for MODEL in "gpt-5.2" "claude-opus-4.6" "gemini-3-flash"  # "gemini-3-flash-preview"
    do
      # First, run LLM generation
      python3 run_gen_api.py \
        --task_name "${EVAL_TASK_NAME}" \
        --genai_model "${MODEL}" \
        --cache_dir "${CACHE_DIR}" \
        --output_dir "${OUTPUT_DIR}-${GEN_METHOD}_seed${SEED}_temp${GEN_TEMP}" \
        --seed "${SEED}" \
        --max_new_gen "${MAX_NEW_GEN}" \
        --gen_temperature "${GEN_TEMP}" \
        --gen_method "${GEN_METHOD}" \
        --verbose
      # Then, evaluate the results
      python3 run_eval_results.py \
        --task_name "${EVAL_TASK_NAME}" \
        --model_name "${MODEL}" \
        --cache_dir "${CACHE_DIR}" \
        --output_dir "${OUTPUT_DIR}-${GEN_METHOD}_seed${SEED}_temp${GEN_TEMP}" \
        --seed "${SEED}" \
        --overwrite \
        --verbose
    done
  done
done
```

</details>


### Synergy between IA and CoT

<details><summary>Experiments - IA+CoT (Table 5)</summary>

```bash
CACHE_DIR="${HOME}/.cache/huggingface/"  # Or your own directory to store datasets and models
#export HF_HOME="${CACHE_DIR}"
OUTPUT_DIR="results/run_gen_hf"

SEED=42
BSZ=4  # generation batch size
MAX_NEW_GEN=4096
GEN_TEMP="0.0"
GEN_TOPP="-1.0"
GEN_CONFIG="1,0,0,0"  # overwrite, do_bf16, do_4bit, and debug
GEN_METHOD="ia+cot"  # "da" "ia"

for EVAL_TASK_NAME in "mmlu" "trivia_qa" "bamboogle" "bbh" "mbpp"
do
  for MODEL in "llama3-8b" "llama3-8b-base" "tulu3-8b-sft" "tulu3-8b-dpo" "tulu3-8b-rlvr" "mistral0.3-7b" "falcon3-7b"
  do
    # First, run LLM generation
    python3 run_gen_hf.py \
      --task_name "${EVAL_TASK_NAME}" \
      --model_name "${MODEL}" \
      --cache_dir "${CACHE_DIR}" \
      --output_dir "${OUTPUT_DIR}-${GEN_METHOD}_seed${SEED}_temp${GEN_TEMP}" \
      --seed "${SEED}" \
      --bsz "${BSZ}" \
      --max_new_gen "${MAX_NEW_GEN}" \
      --gen_temperature "${GEN_TEMP}" \
      --gen_top_p "${GEN_TOPP}" \
      --gen_config "${GEN_CONFIG}" \
      --gen_method "${GEN_METHOD}" \
      --verbose
    # Then, evaluate the results
    python3 run_eval_results.py \
      --task_name "${EVAL_TASK_NAME}" \
      --model_name "${MODEL}" \
      --cache_dir "${CACHE_DIR}" \
      --output_dir "${OUTPUT_DIR}-${GEN_METHOD}_seed${SEED}_temp${GEN_TEMP}" \
      --seed "${SEED}" \
      --overwrite \
      --verbose
  done
done
```

</details>


### Fine-tuning LMs with Intentional Analysis

<details><summary>Training Strategies / Settings</summary>

* **Raw-FT**: No analysis, directly generating the answer.
* **DA-FT**: Using the problem-solving analysis generated by the DA method.
  * **DA-FT-self**: Analysis generated by the target model itself (Llama3-8B or Qwen2.5-7B here) using the DA method.
  * **DA-FT-distill**: Analysis generated by the teacher model (Qwen3-32B here) using the DA method.
* **IA-FT**: Using the problem-solving analysis generated by our IA method.
  * **IA-FT-self**: Analysis generated by the target model itself (Llama3-8B or Qwen2.5-7B here) using our IA method.
  * **IA-FT-distill**: Analysis generated by the teacher model (Qwen3-32B here) using our IA method.

</details>

<details><summary>Generate the Training Data</summary>

```bash
CACHE_DIR="${HOME}/.cache/huggingface/"  # Or your own directory to store datasets and models
#export HF_HOME="${CACHE_DIR}"
OUTPUT_DIR="results/run_gen_hf"

SEED=42
BSZ=4  # generation batch size
MAX_NEW_GEN=4096
GEN_TEMP="0.0"
GEN_TOPP="-1.0"
GEN_CONFIG="1,0,0,0"  # overwrite, do_bf16, do_4bit, and debug

for GEN_METHOD in "da" "ia"
do
  for EVAL_TASK_NAME in "mmlu_training" "trivia_qa_training"
  do
    for MODEL in "llama3-8b" "qwen2.5-7b" "qwen3-32b"
    do
      if [[ "${MODEL}" == "qwen3-32b" ]]; then
        GEN_CONFIG="1,0,1,0"  # 4-bit quantization
      fi

      # First, obtain intentional analysis for fine-tuning
      python3 run_gen_ft_analysis.py \
        --task_name "${EVAL_TASK_NAME}" \
        --model_name "${MODEL}" \
        --cache_dir "${CACHE_DIR}" \
        --output_dir "${OUTPUT_DIR}-${GEN_METHOD}_seed${SEED}_temp${GEN_TEMP}" \
        --seed "${SEED}" \
        --bsz "${BSZ}" \
        --max_new_gen "${MAX_NEW_GEN}" \
        --gen_temperature "${GEN_TEMP}" \
        --gen_top_p "${GEN_TOPP}" \
        --gen_config "${GEN_CONFIG}" \
        --gen_method "${GEN_METHOD}" \
        --verbose

      # Then, build the DA or IA training data
      python3 run_build_ft_data_analysis.py \
        --task_name "${EVAL_TASK_NAME}" \
        --model_name "${MODEL}" \
        --cache_dir "${CACHE_DIR}" \
        --training_data_type "${GEN_METHOD}" \
        --save_dir "data/ft_data" \
        --valid_ratio "0.01" \
        --seed "${SEED}" \
        --overwrite \
        --do_stat \
        --verbose

      # Also, build the Raw training data (note: "raw" data can come from "da" or "ia" generated data)
      if [[ "${GEN_METHOD}" == "da" ]]; then
        python3 run_build_ft_data_analysis.py \
          --task_name "${EVAL_TASK_NAME}" \
          --model_name "${MODEL}" \
          --cache_dir "${CACHE_DIR}" \
          --training_data_type "raw" \
          --save_dir "data/ft_data" \
          --valid_ratio "0.01" \
          --seed "${SEED}" \
          --overwrite \
          --do_stat \
          --verbose
      fi
    done
  done
done
```

</details>

<details><summary>Run LLM Fine-Tuning - Experiments (Table 6)</summary>

```bash
# Set Wandb to monitor the training progress & validation scores
export WANDB_API_KEY="YOUR_WANDB_API_KEY"  # https://docs.wandb.ai/models/track/environment-variables

CACHE_DIR="${HOME}/.cache/huggingface/"  # Or your own directory to store datasets and models
#export HF_HOME="${CACHE_DIR}"
OUTPUT_DIR="results/run_gen_hf"

for GEN_METHOD in "raw" "da" "ia"
do
  for TASK_NAME in "mmlu_training" "trivia_qa_training"
  do
    for MODEL in "llama3-8b" "qwen2.5-7b"
    do
      # Run LLM fine-tuning on a single NVIDIA A-series GPU (e.g., RTX A6000)
      # (1) Self-FT (learning rate \in {2e-5, 5e-5, 2e-5}; LoRA rank \in {8, 16, 32})
      python3 run_train_sft_unsloth.py \
        --model_name "${MODEL}" \
        --label_model_name "${MODEL}" \
        --cache_dir "${CACHE_DIR}" \
        --training_task_name "${TASK_NAME}" \
        --training_data_type "${GEN_METHOD}" \
        --num_train_epochs "3" \
        --learning_rate "2e-05" \
        --train_mode "1,1,0" \
        --lora_mode "8,16,0.0" \
        --valid_mode "100,4,0" \
        --verbose

      # (2) Distill-FT (learning rate \in {2e-5, 5e-5, 2e-5}; LoRA rank \in {8, 16, 32})
      if [[ "${GEN_METHOD}" != "raw" ]]; then
        python3 run_train_sft_unsloth.py \
          --model_name "${MODEL}" \
          --label_model_name "qwen3-32b" \
          --cache_dir "${CACHE_DIR}" \
          --training_task_name "${TASK_NAME}" \
          --training_data_type "${GEN_METHOD}" \
          --num_train_epochs "3" \
          --learning_rate "2e-05" \
          --train_mode "1,1,0" \
          --lora_mode "8,16,0.0" \
          --valid_mode "100,4,0" \
          --verbose
      fi

      # Then, pick the best checkpoint based on the validation scores (see Wandb records).
      #   Assume the path to the best checkpoint directory is "/path/to/best/checkpoint/"
      # Note: Pass `--use_analysis` to `python3 run_eval_results.py` to extract final predictions from complete outputs
    done
  done
done
```

</details>


### Robustness of Intentional Analysis

<details><summary>Experiments - IA Prompt Variants (Table 9)</summary>

```bash
CACHE_DIR="${HOME}/.cache/huggingface/"  # Or your own directory to store datasets and models
#export HF_HOME="${CACHE_DIR}"
OUTPUT_DIR="results/run_gen_hf"

MODEL="llama3-8b"
SEED=42
BSZ=4  # generation batch size
MAX_NEW_GEN=4096
GEN_TEMP="0.0"
GEN_TOPP="-1.0"
GEN_CONFIG="1,0,0,0"  # overwrite, do_bf16, do_4bit, and debug

for GEN_METHOD in "ia_var1" "ia_var2" "ia_var3" "ia_var4" "ia_var5"  # "da" "ia"
do
  for EVAL_TASK_NAME in "mmlu" "trivia_qa" "bamboogle" "bbh" "mbpp"
  do
    # First, run LLM generation
    python3 run_gen_hf.py \
      --task_name "${EVAL_TASK_NAME}" \
      --model_name "${MODEL}" \
      --cache_dir "${CACHE_DIR}" \
      --output_dir "${OUTPUT_DIR}-${GEN_METHOD}_seed${SEED}_temp${GEN_TEMP}" \
      --seed "${SEED}" \
      --bsz "${BSZ}" \
      --max_new_gen "${MAX_NEW_GEN}" \
      --gen_temperature "${GEN_TEMP}" \
      --gen_top_p "${GEN_TOPP}" \
      --gen_config "${GEN_CONFIG}" \
      --gen_method "${GEN_METHOD}" \
      --verbose
    # Then, evaluate the results
    python3 run_eval_results.py \
      --task_name "${EVAL_TASK_NAME}" \
      --model_name "${MODEL}" \
      --cache_dir "${CACHE_DIR}" \
      --output_dir "${OUTPUT_DIR}-${GEN_METHOD}_seed${SEED}_temp${GEN_TEMP}" \
      --seed "${SEED}" \
      --overwrite \
      --verbose
  done
done
```

</details>


<details><summary>Experiments - Generation Temperature (Table 10)</summary>

```bash
CACHE_DIR="${HOME}/.cache/huggingface/"  # Or your own directory to store datasets and models
#export HF_HOME="${CACHE_DIR}"
OUTPUT_DIR="results/run_gen_hf"

MODEL="llama3-8b"
SEED=42
BSZ=4  # generation batch size
MAX_NEW_GEN=4096
GEN_CONFIG="1,0,0,0"  # overwrite, do_bf16, do_4bit, and debug

for GEN_METHOD in "da" "ia"
do
  for EVAL_TASK_NAME in "mmlu" "trivia_qa" "bamboogle" "bbh" "mbpp"
  do
    for GEN_TEMP in "0.25" "0.50" "0.75" "1.00"
    do
      GEN_TOPP="0.9"
      # First, run LLM generation
      python3 run_gen_hf.py \
        --task_name "${EVAL_TASK_NAME}" \
        --model_name "${MODEL}" \
        --cache_dir "${CACHE_DIR}" \
        --output_dir "${OUTPUT_DIR}-${GEN_METHOD}_seed${SEED}_temp${GEN_TEMP}" \
        --seed "${SEED}" \
        --bsz "${BSZ}" \
        --max_new_gen "${MAX_NEW_GEN}" \
        --gen_temperature "${GEN_TEMP}" \
        --gen_top_p "${GEN_TOPP}" \
        --gen_config "${GEN_CONFIG}" \
        --gen_method "${GEN_METHOD}" \
        --verbose
      # Then, evaluate the results
      python3 run_eval_results.py \
        --task_name "${EVAL_TASK_NAME}" \
        --model_name "${MODEL}" \
        --cache_dir "${CACHE_DIR}" \
        --output_dir "${OUTPUT_DIR}-${GEN_METHOD}_seed${SEED}_temp${GEN_TEMP}" \
        --seed "${SEED}" \
        --overwrite \
        --verbose
    done
  done
done
```

</details>


<details><summary>Experiments - Random Seeds (Table 11)</summary>

```bash
CACHE_DIR="${HOME}/.cache/huggingface/"  # Or your own directory to store datasets and models
#export HF_HOME="${CACHE_DIR}"
OUTPUT_DIR="results/run_gen_hf"

MODEL="llama3-8b"
BSZ=4  # generation batch size
MAX_NEW_GEN=4096
GEN_TEMP="0.7"
GEN_TOPP="0.9"
GEN_CONFIG="1,0,0,0"  # overwrite, do_bf16, do_4bit, and debug

for GEN_METHOD in "da" "ia"
do
  for EVAL_TASK_NAME in "mmlu" "trivia_qa" "bamboogle" "bbh" "mbpp"
  do
    for SEED in "7" "42" "365" "1024"
    do
      # First, run LLM generation
      python3 run_gen_hf.py \
        --task_name "${EVAL_TASK_NAME}" \
        --model_name "${MODEL}" \
        --cache_dir "${CACHE_DIR}" \
        --output_dir "${OUTPUT_DIR}-${GEN_METHOD}_seed${SEED}_temp${GEN_TEMP}" \
        --seed "${SEED}" \
        --bsz "${BSZ}" \
        --max_new_gen "${MAX_NEW_GEN}" \
        --gen_temperature "${GEN_TEMP}" \
        --gen_top_p "${GEN_TOPP}" \
        --gen_config "${GEN_CONFIG}" \
        --gen_method "${GEN_METHOD}" \
        --verbose
      # Then, evaluate the results
      python3 run_eval_results.py \
        --task_name "${EVAL_TASK_NAME}" \
        --model_name "${MODEL}" \
        --cache_dir "${CACHE_DIR}" \
        --output_dir "${OUTPUT_DIR}-${GEN_METHOD}_seed${SEED}_temp${GEN_TEMP}" \
        --seed "${SEED}" \
        --overwrite \
        --verbose
    done
  done
done
```

</details>


### Additional Comparisons

<details><summary>Experiments (Table 12)</summary>

```bash
CACHE_DIR="${HOME}/.cache/huggingface/"  # Or your own directory to store datasets and models
#export HF_HOME="${CACHE_DIR}"
OUTPUT_DIR="results/run_gen_hf"

SEED=42
BSZ=4  # generation batch size
MAX_NEW_GEN=4096
GEN_TEMP="0.0"
GEN_TOPP="-1.0"
GEN_CONFIG="1,0,0,0"  # overwrite, do_bf16, do_4bit, and debug

for GEN_METHOD in "da" "ia"
do
  for EVAL_TASK_NAME in "mmlu" "trivia_qa" "bamboogle" "bbh" "mbpp"
  do
    for MODEL in "mistral0.3-7b" "mistral-interact-7b"
    do
      # First, run LLM generation
      python3 run_gen_hf.py \
        --task_name "${EVAL_TASK_NAME}" \
        --model_name "${MODEL}" \
        --cache_dir "${CACHE_DIR}" \
        --output_dir "${OUTPUT_DIR}-${GEN_METHOD}_seed${SEED}_temp${GEN_TEMP}" \
        --seed "${SEED}" \
        --bsz "${BSZ}" \
        --max_new_gen "${MAX_NEW_GEN}" \
        --gen_temperature "${GEN_TEMP}" \
        --gen_top_p "${GEN_TOPP}" \
        --gen_config "${GEN_CONFIG}" \
        --gen_method "${GEN_METHOD}" \
        --verbose
      # Then, evaluate the results
      python3 run_eval_results.py \
        --task_name "${EVAL_TASK_NAME}" \
        --model_name "${MODEL}" \
        --cache_dir "${CACHE_DIR}" \
        --output_dir "${OUTPUT_DIR}-${GEN_METHOD}_seed${SEED}_temp${GEN_TEMP}" \
        --seed "${SEED}" \
        --overwrite \
        --verbose
    done
  done
done
```

</details>


## License

[The MIT License](https://opensource.org/license/mit): Please refer to the [LICENSE](./LICENSE) file for more details.


## Citation

* If you find our work helpful, please kindly star this GitHub repo and cite our paper. 🤗

```bibtex
@article{yin2026ia,
  title   = {Improving Language Models with Intentional Analysis},
  author  = {Yin, Yuwei and Carenini, Giuseppe},
  journal = {arXiv preprint arXiv:2502.04689},
  year    = {2026},
  url     = {https://arxiv.org/abs/2502.04689},
}
```

---
