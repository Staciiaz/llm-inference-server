from abc import ABC, abstractmethod

from ..types.chat_completion import ConversationHistory


class ChatModel(ABC):
    @abstractmethod
    def chat_completions(self, messages: ConversationHistory) -> str:
        pass
