from typing import Optional, Union

from typing_extensions import Literal, Required, TypedDict


class ImageURL(TypedDict, total=False):
    url: Required[str]
    detail: Literal["auto", "low", "high"]

class ChatCompletionContentPartImageParam(TypedDict, total=False):
    image_url: Required[ImageURL]
    type: Required[Literal["image_url"]]

class ChatCompletionContentPartTextParam(TypedDict, total=False):
    text: Required[str]
    type: Required[Literal["text"]]

ChatCompletionContentPartParam = Union[ChatCompletionContentPartTextParam, ChatCompletionContentPartImageParam]

class ChatCompletionUserMessageParam(TypedDict):
    role: Required[Literal["user"]]
    content: Required[Union[str, list[ChatCompletionContentPartParam]]]

class ChatCompletionAssistantMessageParam(TypedDict):
    role: Required[Literal["assistant"]]
    content: Optional[str]

class ChatCompletionSystemMessageParam(TypedDict):
    role: Required[Literal["system"]]
    content: Required[str]

class ChatCompletionResponse:
    def __init__(self, model: str, input_tokens: int, output_tokens: int, response_message: str) -> None:
        self.model = model
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.response_message = response_message

    @property
    def completion_tokens(self) -> int:
        return self.output_tokens - self.input_tokens
    
ChatCompletionRole = Literal["user", "assistant", "system"]
ConversationHistory = list[Union[ChatCompletionUserMessageParam, ChatCompletionAssistantMessageParam, ChatCompletionSystemMessageParam]]
