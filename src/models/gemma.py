import os

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

from ..types.chat_completion import ConversationHistory
from ..utils import extract_assistant_output
from .chat_model import ChatModel


class Gemma(ChatModel):
    def __init__(self, model_id: str) -> None:
        token = os.getenv("HUGGINGFACE_TOKEN")
        self.device = "cuda"
        self.templates = ("<start_of_turn>model\n", "<eos>")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, token=token, device_map=self.device, torch_dtype=torch.bfloat16)

    def chat_completions(self, messages: ConversationHistory) -> str:
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
        outputs = self.model.generate(input_ids=inputs.to(self.device), max_new_tokens=1000)
        return extract_assistant_output(self.tokenizer.decode(outputs[0]), self.templates)