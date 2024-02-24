from typing import Optional

from typing_extensions import Literal, Required, TypedDict

type ChatCompletionRole = Literal["user", "assistant", "system"]
type ConversationHistory = list[ChatCompletionUserMessageParam | ChatCompletionAssistantMessageParam |
                                ChatCompletionSystemMessageParam]

class ChatCompletionUserMessageParam(TypedDict):
    role: Required[Literal["user"]]
    content: Required[str]

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
