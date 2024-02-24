from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
from io import BytesIO
from base64 import b64decode

from ..types.chat_completion import ConversationHistory, ChatCompletionResponse, ChatCompletionUserMessageParam
from ..utils import (extract_assistant_output, get_device_config,
                     get_quantization_config)
from ..logging import logger
from .chat_model import ChatModel


class Llava(ChatModel):
    def __init__(self, model_id: str, dtype: str, device: str) -> None:
        self.device = get_device_config(device)
        self.torch_dtype, _ = get_quantization_config(dtype)
        self.templates = ("ASSISTANT:", "</s>")
        self.model_id = model_id
        self.model = LlavaForConditionalGeneration.from_pretrained(model_id, device_map=self.device, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True)
        self.processor = AutoProcessor.from_pretrained(model_id)

    def find_text_input(self, user_message: ChatCompletionUserMessageParam) -> str:
        for content in user_message["content"]:
            if content["type"] == "text":
                return content["text"]
            
    def find_image_input(self, user_message: ChatCompletionUserMessageParam) -> str:
        for content in user_message["content"]:
            if content["type"] == "image_url":
                return content["image_url"]["url"].replace("data:image/jpeg;base64,", "")

    def chat_completions(self, messages: ConversationHistory) -> ChatCompletionResponse:
        text_input = self.find_text_input(messages[-1])
        image_input = self.find_image_input(messages[-1])
        prompt = f"USER: <image>\n{text_input}\nASSISTANT:"
        raw_image = Image.open(BytesIO(b64decode(image_input)))
        logger.info(f"Inferencing with the model {self.model_id} ...")
        inputs = self.processor(prompt, raw_image, return_tensors='pt').to(self.device, self.torch_dtype)
        outputs = self.model.generate(**inputs, max_new_tokens=1000, do_sample=False)
        decoded = self.processor.decode(outputs[0])
        logger.info(f"Generated response:\n{decoded}")
        return ChatCompletionResponse(
            model=self.model_id,
            input_tokens=inputs["input_ids"].size(1),
            output_tokens=outputs.size(1),
            response_message=extract_assistant_output(decoded, self.templates)
        )
