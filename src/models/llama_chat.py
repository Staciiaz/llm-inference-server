import os

from transformers import AutoModelForCausalLM, AutoTokenizer

from ..types.chat_completion import ConversationHistory, ChatCompletionResponse
from ..utils import (extract_assistant_output, get_device_config,
                     get_quantization_config)
from ..logging import logger
from .chat_model import ChatModel


class LlamaChat(ChatModel):
    def __init__(self, model_id: str, dtype: str, device: str) -> None:
        token = os.getenv("HF_TOKEN")
        self.device = get_device_config(device)
        torch_dtype, quantization_config = get_quantization_config(dtype)
        self.templates = ("[/INST]", "</s>")
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, token=token, device_map=self.device, torch_dtype=torch_dtype, quantization_config=quantization_config)

    def chat_completions(self, messages: ConversationHistory) -> ChatCompletionResponse:
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
        logger.info(f"Inferencing with the model {self.model_id} ...")
        outputs = self.model.generate(input_ids=inputs.to(self.device), max_new_tokens=1000)
        decoded = self.tokenizer.decode(outputs[0])
        logger.info(f"Generated response:\n{decoded}")
        return ChatCompletionResponse(
            model=self.model_id,
            input_tokens=inputs.size(1),
            output_tokens=outputs.size(1),
            response_message=extract_assistant_output(decoded, self.templates)
        )
