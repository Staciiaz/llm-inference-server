from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
from io import BytesIO
from base64 import b64decode

from ..types.chat_completion import ConversationHistory, ChatCompletionResponse, ChatCompletionUserMessageParam
from ..utils import (extract_assistant_output, get_device_config,
                     get_quantization_config)
from ..logging import logger
from .chat_model import ChatModel


class LlavaBase(ChatModel):
    def __init__(self, model_id: str, dtype: str, device: str) -> None:
        self.device = get_device_config(device)
        self.torch_dtype, _ = get_quantization_config(dtype)
        self.templates = ("ASSISTANT:", "</s>")
        self.model_id = model_id
        self.model = LlavaForConditionalGeneration.from_pretrained(model_id, device_map=self.device, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True)
        self.processor = AutoProcessor.from_pretrained(model_id)
            
    def get_input_prompt(self, user_message: ChatCompletionUserMessageParam) -> tuple[str, list[str]]:
        prompt = "USER: "
        b64_images = list()
        for content in user_message["content"]:
            if content["type"] == "text":
                prompt += content["text"] + '\n'
            elif content["type"] == "image_url":
                b64_images.append(content["image_url"]["url"].replace("data:image/jpeg;base64,", ""))
                prompt += "<image>\n"
        prompt += "ASSISTANT:"
        return prompt, b64_images

    def chat_completions(self, messages: ConversationHistory) -> ChatCompletionResponse:
        prompt, b64_images = self.get_input_prompt(messages[-1])
        images = [Image.open(BytesIO(b64decode(b64_image))) for b64_image in b64_images]
        logger.info(f"Inferencing with the model {self.model_id} ...")
        inputs = self.processor(prompt, images, return_tensors='pt').to(self.device, self.torch_dtype)
        outputs = self.model.generate(**inputs, max_new_tokens=1000, do_sample=False)
        decoded = self.processor.decode(outputs[0])
        logger.info(f"Generated response:\n{decoded}")
        return ChatCompletionResponse(
            model=self.model_id,
            input_tokens=inputs["input_ids"].size(1),
            output_tokens=outputs.size(1),
            response_message=extract_assistant_output(decoded, self.templates)
        )
