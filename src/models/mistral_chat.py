from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

from ..logging import logger
from ..types.chat_completion import (ChatCompletionRequest,
                                     ChatCompletionResponse)
from ..utils import (extract_assistant_output, get_device_config,
                     get_quantization_config)
from .chat_model import ChatModel


class MistralChat(ChatModel):
    def __init__(self, model_id: str, dtype: str, device: str) -> None:
        self.device = get_device_config(device)
        torch_dtype, quantization_config = get_quantization_config(dtype)
        self.templates = ("[/INST] ", "</s>")
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map=self.device, torch_dtype=torch_dtype, quantization_config=quantization_config)
        self.streamer = TextStreamer(self.tokenizer)

    def chat_completions(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        prompt = self.tokenizer.apply_chat_template(request.messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer.encode(prompt, add_special_tokens=True, return_tensors='pt').to(self.device)
        generated_kwargs = {
            "max_new_tokens": 1000,
            "pad_token_id": self.tokenizer.eos_token_id,
            "do_sample": True,
            "temperature": request.temperature,
        }
        logger.info(f"Inferencing with the model {self.model_id} ...")
        outputs = self.model.generate(inputs, streamer=self.streamer, **generated_kwargs)
        decoded = self.tokenizer.batch_decode(outputs)
        return ChatCompletionResponse(
            model=self.model_id,
            input_tokens=inputs.size(1),
            output_tokens=outputs.size(1),
            response_message=extract_assistant_output(decoded[0], self.templates)
        )
