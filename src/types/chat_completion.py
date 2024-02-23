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
