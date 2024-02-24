import os

from transformers import AutoModelForCausalLM, AutoTokenizer

from ..types.chat_completion import ConversationHistory
from ..utils import (extract_assistant_output, get_device_config,
                     get_quantization_config)
from .chat_model import ChatModel


class Llama(ChatModel):
    def __init__(self, model_id: str, dtype: str, device: str) -> None:
        token = os.getenv("HUGGINGFACE_TOKEN")
        self.device = get_device_config(device)
        torch_dtype, quantization_config = get_quantization_config(dtype)
        self.templates = ("[/INST]", "</s>")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, token=token, device_map=self.device, torch_dtype=torch_dtype, quantization_config=quantization_config)

    def chat_completions(self, messages: ConversationHistory) -> str:
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
        outputs = self.model.generate(input_ids=inputs.to(self.device), max_new_tokens=1000)
        return extract_assistant_output(self.tokenizer.decode(outputs[0]), self.templates)
