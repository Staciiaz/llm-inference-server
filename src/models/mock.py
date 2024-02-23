from ..types.chat_completion import ConversationHistory
from .chat_model import ChatModel


class MockChatModel(ChatModel):
    def chat_completions(self, messages: ConversationHistory) -> str:
        return messages[-1]["content"]
