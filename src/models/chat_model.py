from abc import ABC, abstractmethod

from ..types.chat_completion import ConversationHistory, ChatCompletionResponse


class ChatModel(ABC):
    @abstractmethod
    def chat_completions(self, messages: ConversationHistory) -> ChatCompletionResponse:
        pass
