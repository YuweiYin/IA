# -*- coding: utf-8 -*-

import os
import sys
import time
import logging
from typing import Optional, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LlamaForCausalLM, Qwen2ForCausalLM, Qwen3ForCausalLM
from peft import PeftModelForCausalLM
from transformers import BitsAndBytesConfig

import openai
from openai import APIError, APIConnectionError, RateLimitError
from google import genai
from google.genai import types
from anthropic import Anthropic


class ModelUtils:

    def __init__(self):
        pass

    @staticmethod
    def initialize_tokenizer_hf(
            model_name: str,
            cache_dir: str,
            padding_side: str = "left",
            truncation_side: str = "left",
            verbose: bool = False,
            model_ckpt_dir: Optional[str] = None,
    ):
        if isinstance(model_ckpt_dir, str) and len(model_ckpt_dir) > 0 and os.path.isdir(model_ckpt_dir):
            model_path = model_ckpt_dir
        else:
            assert model_name in ModelUtils.OPEN_MODEL_HF, f">>> Unsupported `model_name`: {model_name}"
            assert os.path.isdir(cache_dir), f">>> `cache_dir` does not exist: {cache_dir}"

            local_model_path = ModelUtils.get_local_model_path(model_name, cache_dir)
            if isinstance(local_model_path, str) and os.path.isdir(local_model_path):
                model_path = local_model_path
            else:
                model_path = ModelUtils.OPEN_MODEL_HF[model_name]

        if verbose:
            logging.info(f">>> Loading tokenizer: model_path = {model_path}")

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            cache_dir=cache_dir,
            padding_side=padding_side,
            truncation_side=truncation_side,
            trust_remote_code=True,
        )  # "right" for training, "left" for generating
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

        return tokenizer

    @staticmethod
    def get_device_count(device_type: str):
        if device_type == "cuda":
            return torch.cuda.device_count()
        elif device_type == "xpu":
            return torch.xpu.device_count()
        else:
            return 1

    @staticmethod
    def show_num_param(torch_model, torch_model_path):
        total_params = sum(p.numel() for p in torch_model.parameters())
        train_params = sum(p.numel() for p in torch_model.parameters() if p.requires_grad)
        logging.info(f">>> Model loaded from `{torch_model_path}`")
        logging.info(f">>> Number of total parameters: {total_params}")
        logging.info(f">>> Number of trainable parameters: {train_params}")

    @staticmethod
    def initialize_model_hf(
            model_name: str,
            cache_dir: str,
            do_train: bool = False,
            do_4bit: bool = False,
            do_bf16: bool = False,
            do_fp16: bool = False,
            # do_flash_attn: bool = False,
            verbose: bool = False,
            model_ckpt_dir: Optional[str] = None,
            use_mxfp4: bool = False,
    ):
        if isinstance(model_ckpt_dir, str) and len(model_ckpt_dir) > 0 and os.path.isdir(model_ckpt_dir):
            model_path = model_ckpt_dir
        else:
            assert model_name in ModelUtils.OPEN_MODEL_HF, f">>> Unsupported `model_name`: {model_name}"
            assert os.path.isdir(cache_dir), f">>> `cache_dir` does not exist: {cache_dir}"

            local_model_path = ModelUtils.get_local_model_path(model_name, cache_dir)
            if isinstance(local_model_path, str) and os.path.isdir(local_model_path):
                model_path = local_model_path
            else:
                model_path = ModelUtils.OPEN_MODEL_HF[model_name]

        # do_4bit = do_4bit or "4bit" in model_name
        if verbose:
            logging.info(f">>> Loading open LLMs: model_path = {model_path}")
            logging.info(f">>> [do_4bit]: {do_4bit}, [do_bf16]: {do_bf16}, [do_fp16]: {do_fp16}")

        if do_bf16:
            torch_dtype = torch.bfloat16
        elif do_fp16:
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        if do_4bit:
            if use_mxfp4:
                from transformers import Mxfp4Config

                mx_config_4bit = Mxfp4Config(dequantize=False)
                model = AutoModelForCausalLM.from_pretrained(
                    model_path, device_map="auto",
                    quantization_config=mx_config_4bit,
                    # dtype=torch.bfloat16,
                    dtype=torch_dtype,
                    # attn_implementation="eager",
                    # attn_implementation="flash_attention_2" if do_flash_attn else None,
                    trust_remote_code=True,
                    cache_dir=cache_dir,
                    # low_cpu_mem_usage=True,
                )
            else:
                bnb_config_4bit = BitsAndBytesConfig(
                    load_in_4bit=True,
                    load_in_8bit=False,
                    bnb_4bit_compute_dtype=torch_dtype,  # torch.bfloat16
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    quant_method="bitsandbytes",
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_path, device_map="auto",
                    quantization_config=bnb_config_4bit,
                    # attn_implementation="flash_attention_2" if do_flash_attn else None,
                    trust_remote_code=True,
                    cache_dir=cache_dir,
                )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, device_map="auto",
                # torch_dtype=torch_dtype,  # torch.bfloat16
                dtype=torch_dtype,  # torch.bfloat16
                # attn_implementation="flash_attention_2" if do_flash_attn else None,
                trust_remote_code=True,
                cache_dir=cache_dir,
            )

        if do_train:
            model.train()
        else:
            model.eval()

        ModelUtils.show_num_param(torch_model=model, torch_model_path=model_path)
        return model

    @staticmethod
    def initialize_model_hf_unsloth(
            model_name: str,
            cache_dir: str,
            do_train: bool = False,
            do_4bit: bool = False,
            # do_flash_attn: bool = False,
            verbose: bool = False,
            model_ckpt_dir: Optional[str] = None,
            max_seq_len: int = 1024,
    ):
        import unsloth

        if isinstance(model_ckpt_dir, str) and len(model_ckpt_dir) > 0 and os.path.isdir(model_ckpt_dir):
            model_path = model_ckpt_dir
        else:
            assert model_name in ModelUtils.OPEN_MODEL_HF, f">>> Unsupported `model_name`: {model_name}"
            assert os.path.isdir(cache_dir), f">>> `cache_dir` does not exist: {cache_dir}"

            local_model_path = ModelUtils.get_local_model_path(model_name, cache_dir)
            if isinstance(local_model_path, str) and os.path.isdir(local_model_path):
                model_path = local_model_path
            else:
                model_path = ModelUtils.OPEN_MODEL_HF[model_name]

        # do_4bit = do_4bit or "4bit" in model_name
        if verbose:
            logging.info(f">>> Loading open LLMs: model_path = {model_path}")
            logging.info(f">>> [do_4bit]: {do_4bit}")

        # if do_bf16:
        #     torch_dtype = torch.bfloat16
        # elif do_fp16:
        #     torch_dtype = torch.float16
        # else:
        #     torch_dtype = torch.float32

        # do_4bit = do_4bit or "4bit" in model_name
        if verbose:
            logging.info(f">>> Loading open LLMs: model_path = {model_path}")
            logging.info(f">>> [do_4bit]: {do_4bit}, [do_bf16]: torch.bfloat16")

        model, tokenizer = unsloth.FastLanguageModel.from_pretrained(
            model_path, device_map="auto",
            trust_remote_code=True,
            cache_dir=cache_dir,
            load_in_4bit=do_4bit,
            dtype="bfloat16",  # None (default), "float16", or "bfloat16"
            # torch_dtype=torch_dtype,  # unsloth auto-sets torch_dtype (`torch.bfloat16` by default)
            # attn_implementation="flash_attention_2" if do_flash_attn else None,
            max_seq_length=max_seq_len,
            full_finetuning=False,
        )  # This function is in unsloth.models.loader.py
        # disable gradient checkpointing (unsloth auto-enables it)
        model.gradient_checkpointing_disable()
        model.config.use_cache = False
        # # model.config.torch_dtype == "bfloat16"
        # if not hasattr(model.config, "torch_dtype"):
        #     setattr(model.config, "torch_dtype", "bfloat16")

        if do_train:
            unsloth.FastLanguageModel.for_training(model)
            model.train()
        else:
            unsloth.FastLanguageModel.for_inference(model)
            model.eval()

        ModelUtils.show_num_param(torch_model=model, torch_model_path=model_path)

        # return model, tokenizer
        return model

    @staticmethod
    def get_local_model_path(
            model_name: str,
            cache_dir: Optional[str] = None,
    ) -> Optional[str]:
        if model_name not in ModelUtils.OPEN_MODEL_HF:
            return None
        if not isinstance(cache_dir, str) or not os.path.isdir(cache_dir):
            return None

        hf_id = ModelUtils.OPEN_MODEL_HF[model_name]
        local_model_path = os.path.join(cache_dir, "models--" + "--".join(hf_id.split("/")), "snapshots/model")
        return local_model_path if os.path.isdir(local_model_path) else None

    @staticmethod
    def is_causal_lm(model) -> bool:
        return (isinstance(model, AutoModelForCausalLM) or isinstance(model, LlamaForCausalLM) or
                isinstance(model, Qwen2ForCausalLM) or isinstance(model, Qwen3ForCausalLM) or
                isinstance(model, PeftModelForCausalLM))

    # @torch.cuda.amp.autocast()
    @staticmethod
    @torch.no_grad()
    def open_model_gen(
            inputs,
            model,
            tokenizer,
            need_tokenize: bool = True,
            max_new_tokens: int = 1024,
            temperature: float = 1.0,
            top_p: Optional[float] = None,
            top_k: Optional[int] = None,
    ) -> List[dict]:
        if need_tokenize:
            # Note: before generation, you can optionally do `tokenizer.apply_chat_template(
            #   prompts, tokenize=False, padding=False, return_tensors=None, add_generation_prompt=True/False)`
            input_ids = tokenizer(
                inputs,
                padding=True,  # truncation=True, max_length=1024
                return_tensors="pt",
            ).to(model.device)  # batch_size=1
        else:
            input_ids = inputs
            input_ids = input_ids.to(model.device)
        # len_input = input_ids.data["input_ids"].size(-1)

        terminators_gen = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids(tokenizer.eos_token)]
        terminators_gen_set = set(terminators_gen)
        terminators_gen_list = list(terminators_gen_set)

        # model.generation_config.pad_token_id = tokenizer.pad_token_id
        with torch.no_grad():
            # with torch.cuda.amp.autocast(enabled=True, dtype=model.dtype):
            # https://huggingface.co/docs/transformers/en/main_classes/text_generation
            response = model.generate(
                **input_ids,
                max_new_tokens=max_new_tokens,
                eos_token_id=terminators_gen_list,
                do_sample=temperature > 0.0,  # False: greedy decoding (the most deterministic)
                temperature=temperature if temperature > 0.0 else None,  # defaults to 1.0
                top_p=top_p,  # defaults to 1.0
                top_k=top_k,  # defaults to 50
                # output_attentions=False,
                # output_hidden_states=False,
                # output_scores=True,
                output_logits=True,
                return_dict_in_generate=True,
            )

        # Check the last token of the current output
        output_ids = response["sequences"]

        last_token_id_list = [x.item() for x in list(output_ids[:, -1].cpu())]
        end_with_eot_list = [x in terminators_gen_set for x in last_token_id_list]

        output_text = tokenizer.batch_decode(
            output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        input_text = tokenizer.batch_decode(
            input_ids["input_ids"], skip_special_tokens=True, clean_up_tokenization_spaces=True)

        assert len(input_text) == len(output_text) == len(end_with_eot_list)
        results = []
        for _input, _output, _end_with_eot in zip(input_text, output_text, end_with_eot_list):
            _output_pure = _output[len(_input):]
            results.append({
                "input_text": _input,
                "output_text": _output_pure,
                "end_with_eot": _end_with_eot,
            })

        return results

    @staticmethod
    def call_gpt(
            openai_model_name: str,
            messages,
            openai_api_key: Optional[str] = None,
            format_class=None,
            temperature: float = 1.0,
            api_call_sleep: int = 3,
            api_retry_limit: int = 10,
            api_retry_sleep: int = 10,
    ):
        """
        :param openai_model_name: The name of the OpenAI model to call.
        :param openai_api_key: A valid OpenAI API Key or from env var ${OPENAI_API_KEY}. https://platform.openai.com/
        :param messages: The input messages for the model.
        :param format_class: A BaseModel to define the output response format.
        :param temperature: The generation temperature that controls randomness.
        :param api_call_sleep: The sleep time between API calls (to avoid hitting the limit).
        :param api_retry_limit: The number of retries before giving up API calls.
        :param api_retry_sleep: The sleep time before retrying API calls.

        :return: The raw response generated by the model.
        """

        # Set up the input prompt (dialog-style) for GPT
        if isinstance(messages, list) and len(messages) > 0:
            input_messages = messages
        elif isinstance(messages, str) and len(messages) > 0:
            input_messages = [
                {"role": "developer", "content": "You are a helpful assistant."},
                {"role": "user", "content": messages},
            ]
        else:
            raise ValueError(f">>> Unsupported `messages`: {messages}")

        if openai_model_name.startswith("gpt-5"):
            # Note: GPT-5 models only accept the default temperature value: 1.0
            temperature = float(1.0)
        else:
            temperature = float(max(0.0, temperature))

        # assert api_call_sleep > 0 and api_retry_limit > 0 and api_retry_sleep > 0
        if not (isinstance(openai_api_key, str) and len(openai_api_key) > 0):
            openai_api_key = os.getenv("OPENAI_API_KEY")
        assert isinstance(openai_api_key, str) and len(openai_api_key) > 0, \
            f">>> AssertionError: openai_api_key: {openai_api_key}"

        try_cnt = 0
        while True:
            try_cnt += 1
            try:
                # OpenAI request
                if format_class is None:
                    response = openai.OpenAI(api_key=openai_api_key).with_options(
                        timeout=900.0).chat.completions.create(
                        model=openai_model_name,
                        messages=input_messages,
                        temperature=temperature,
                        # response_format=None,
                        # service_tier="flex",
                    )
                else:
                    # response = openai.OpenAI(api_key=openai_api_key).with_options(timeout=900.0).beta.chat.completions.parse(
                    response = openai.OpenAI(api_key=openai_api_key).with_options(timeout=900.0).chat.completions.parse(
                        model=openai_model_name,
                        messages=input_messages,
                        temperature=temperature,
                        response_format=format_class,
                        # service_tier="flex",
                    )
                if api_call_sleep > 0:  # Regular sleep time before running next data item
                    time.sleep(api_call_sleep)
                break
            except APIConnectionError as e:
                logging.info(f">>> APIConnectionError !!! >>> Failed to connect to OpenAI API: {e}")
                if try_cnt >= api_retry_limit:
                    sys.exit(1)
                time.sleep(api_retry_sleep)  # Sleep time before retrying API calls
            except APIError as e:
                logging.info(f">>> APIError !!! >>> OpenAI API Error: {e}")
                if try_cnt >= api_retry_limit:
                    sys.exit(1)
                time.sleep(api_retry_sleep)  # Sleep time before retrying API calls
            except RateLimitError as e:
                logging.info(f">>> RateLimitError !!! >>> OpenAI API request exceeded rate limit: {e}")
                if try_cnt >= api_retry_limit:
                    sys.exit(1)
                time.sleep(api_retry_sleep)  # Sleep time before retrying API calls
            except Exception as e:
                logging.info(f">>> Exception !!! >>> OpenAI Exception: {e}")
                if try_cnt >= api_retry_limit:
                    sys.exit(1)
                time.sleep(api_retry_sleep)  # Sleep time before retrying API calls

        return response

    @staticmethod
    def call_gemini(
            gemini_model_name: str,
            messages: List[str],
            gemini_api_key: Optional[str] = None,
            temperature: float = 1.0,
            api_call_sleep: int = 3,
            api_retry_limit: int = 10,
            api_retry_sleep: int = 10,
    ):
        """
        :param gemini_model_name: The name of the Gemini model to call.
        :param gemini_api_key: A valid Gemini API Key or from env var ${GEMINI_API_KEY}. https://aistudio.google.com/
        :param messages: The input messages for the model. [system_prompt, user_prompt]
        :param temperature: The generation temperature that controls randomness.
        :param api_call_sleep: The sleep time between API calls (to avoid hitting the limit).
        :param api_retry_limit: The number of retries before giving up API calls.
        :param api_retry_sleep: The sleep time before retrying API calls.

        :return: The raw response generated by the model.
        """

        # assert api_call_sleep > 0 and api_retry_limit > 0 and api_retry_sleep > 0
        if not (isinstance(gemini_api_key, str) and len(gemini_api_key) > 0):
            gemini_api_key = os.getenv("GEMINI_API_KEY")
        assert isinstance(gemini_api_key, str) and len(gemini_api_key) > 0, \
            f">>> AssertionError: gemini_api_key: {gemini_api_key}"

        gemini_client = genai.Client(api_key=gemini_api_key)
        assert isinstance(messages, list) and len(messages) == 2
        system_prompt, user_prompt = messages[0], messages[1]
        assert isinstance(system_prompt, str) and isinstance(user_prompt, str)

        try_cnt = 0
        while True:
            try_cnt += 1
            try:
                response = gemini_client.models.generate_content(
                    model=gemini_model_name,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=max(0.0, float(temperature)),
                    ),
                )
                if api_call_sleep > 0:  # Regular sleep time before running next data item
                    time.sleep(api_call_sleep)
                break
            except Exception as e:
                logging.info(f">>> Exception !!! >>> Gemini Exception: {e}")
                if try_cnt >= api_retry_limit:
                    sys.exit(1)
                time.sleep(api_retry_sleep)  # Sleep time before retrying API calls

        return response

    @staticmethod
    def call_claude(
            claude_model_name: str,
            messages: List[str],
            claude_api_key: Optional[str] = None,
            max_output_tokens: Optional[int] = None,
            api_call_sleep: int = 3,
            api_retry_limit: int = 10,
            api_retry_sleep: int = 10,
    ):
        """
        :param claude_model_name: The name of the Claude model to call.
        :param claude_api_key: A valid Claude API Key or from env var ${CLAUDE_API_KEY}.
        :param messages: The input messages for the model.
        :param max_output_tokens: The maximum number of newly generated tokens.
        :param api_call_sleep: The sleep time between API calls (to avoid hitting the limit).
        :param api_retry_limit: The number of retries before giving up API calls.
        :param api_retry_sleep: The sleep time before retrying API calls.

        :return: The raw response generated by the model.
        """

        # assert api_call_sleep > 0 and api_retry_limit > 0 and api_retry_sleep > 0
        if not (isinstance(claude_api_key, str) and len(claude_api_key) > 0):
            claude_api_key = os.getenv("ANTHROPIC_API_KEY")
        assert isinstance(claude_api_key, str) and len(claude_api_key) > 0, \
            f">>> AssertionError: claude_api_key: {claude_api_key}"

        claude_client = Anthropic(api_key=claude_api_key)  # Reads ANTHROPIC_API_KEY from environment by default
        assert isinstance(messages, list) and len(messages) == 2
        system_prompt, user_prompt = messages[0], messages[1]
        assert isinstance(system_prompt, str) and isinstance(user_prompt, str)

        try_cnt = 0
        while True:
            try_cnt += 1
            try:
                response = claude_client.messages.create(
                    model=claude_model_name,
                    max_tokens=max_output_tokens,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )  # using default `temperature`
                if api_call_sleep > 0:  # Regular sleep time before running next data item
                    time.sleep(api_call_sleep)
                break
            except Exception as e:
                logging.info(f">>> Exception !!! >>> Claude Exception: {e}")
                if try_cnt >= api_retry_limit:
                    sys.exit(1)
                time.sleep(api_retry_sleep)  # Sleep time before retrying API calls

        return response

    OPEN_MODEL_HF = {
        "llama3-8b": "meta-llama/Llama-3.1-8B-Instruct",
        "llama3-8b-base": "meta-llama/Llama-3.1-8B",
        "tulu3-8b-sft": "allenai/Llama-3.1-Tulu-3-8B-SFT",
        "tulu3-8b-dpo": "allenai/Llama-3.1-Tulu-3-8B-DPO",
        "tulu3-8b-rlvr": "allenai/Llama-3.1-Tulu-3-8B",
        "mistral0.3-7b": "mistralai/Mistral-7B-Instruct-v0.3",
        "falcon3-7b": "tiiuae/Falcon3-7B-Instruct",
        "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
        "qwen3-32b": "Qwen/Qwen3-32B",
        "mistral-interact-7b": "hbx/Mistral-Interact",
        "qwen3-8b": "Qwen/Qwen3-8B",
    }
