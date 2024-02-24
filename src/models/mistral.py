from transformers import AutoModelForCausalLM, AutoTokenizer

from ..types.chat_completion import ConversationHistory, ChatCompletionResponse
from ..utils import (extract_assistant_output, get_device_config,
                     get_quantization_config)
from ..logging import logger
from .chat_model import ChatModel


class Mistral(ChatModel):
    def __init__(self, model_id: str, dtype: str, device: str) -> None:
        self.device = get_device_config(device)
        torch_dtype, quantization_config = get_quantization_config(dtype)
        self.templates = ("[/INST] ", "</s>")
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map=self.device, torch_dtype=torch_dtype, quantization_config=quantization_config)

    def chat_completions(self, messages: ConversationHistory) -> ChatCompletionResponse:
        encodeds = self.tokenizer.apply_chat_template(messages, return_tensors="pt")
        model_inputs = encodeds.to(self.device)
        logger.info(f"Inferencing with the model {self.model_id} ...")
        generated_ids = self.model.generate(model_inputs, max_new_tokens=1000, do_sample=True, pad_token_id=self.tokenizer.eos_token_id)
        decoded = self.tokenizer.batch_decode(generated_ids)
        logger.info(f"Generated response:\n{decoded[0]}")
        return ChatCompletionResponse(
            model=self.model_id,
            input_tokens=encodeds.size(1),
            output_tokens=generated_ids.size(1),
            response_message=extract_assistant_output(decoded[0], self.templates)
        )
