from abc import ABC, abstractmethod

from ..types.chat_completion import (ChatCompletionRequest,
                                     ChatCompletionResponse)


class ChatModel(ABC):
    @abstractmethod
    def chat_completions(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        pass
