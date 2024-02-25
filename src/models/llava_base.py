from base64 import b64decode
from io import BytesIO

from PIL import Image
from transformers import (AutoProcessor, LlavaForConditionalGeneration,
                          TextStreamer)

from ..logging import logger
from ..types.chat_completion import (ChatCompletionRequest,
                                     ChatCompletionResponse,
                                     ChatCompletionUserMessageParam)
from ..utils import (extract_assistant_output, get_device_config,
                     get_quantization_config)
from .chat_model import ChatModel


class LlavaBase(ChatModel):
    def __init__(self, model_id: str, dtype: str, device: str) -> None:
        self.device = get_device_config(device)
        self.torch_dtype, _ = get_quantization_config(dtype)
        self.templates = ("ASSISTANT:", "</s>")
        self.model_id = model_id
        self.model = LlavaForConditionalGeneration.from_pretrained(model_id, device_map=self.device, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.streamer = TextStreamer(self.processor)
            
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

    def chat_completions(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        prompt, b64_images = self.get_input_prompt(request.messages[-1])
        images = [Image.open(BytesIO(b64decode(b64_image))) for b64_image in b64_images]
        inputs = self.processor(prompt, images, return_tensors='pt').to(self.device)
        logger.info(f"Inferencing with the model {self.model_id} ...")
        outputs = self.model.generate(**inputs, streamer=self.streamer, max_new_tokens=1000, do_sample=True, temperature=request.temperature)
        decoded = self.processor.decode(outputs[0])
        return ChatCompletionResponse(
            model=self.model_id,
            input_tokens=inputs["input_ids"].size(1),
            output_tokens=outputs.size(1),
            response_message=extract_assistant_output(decoded, self.templates)
        )
