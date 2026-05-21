# -*- coding: utf-8 -*-

FINAL_ANSWER_OPEN_QA = "Final Answer:"

SYSTEM_PROMPT_OPEN_QA_GEN = (f"""
You are a helpful assistant. \
You are good at answering questions through reading comprehension, information retrieval, and logical reasoning.
You can analyze the question first and give your final answer at the end.

Answer Format: Your final answer MUST start with "{FINAL_ANSWER_OPEN_QA}"
""".strip() + r' and the answer text MUST be put into "\boxed{}". ' +
    r'For example, you should output "' + FINAL_ANSWER_OPEN_QA + r' \boxed{Paris}" if your final answer is Paris')
