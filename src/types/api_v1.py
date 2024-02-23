from pydantic import BaseModel

from ..types.chat_completion import (ChatCompletionAssistantMessageParam,
                                     ConversationHistory)


class ChatCompletionRequest(BaseModel):
    model: str
    messages: ConversationHistory

class ModelUsage:
    def __init__(self, prompt_tokens: int, completion_tokens: int, total_tokens: int):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens

class ModelChoice:
    def __init__(self, message: str, messages: ChatCompletionAssistantMessageParam, logprobs: float, finish_reason: str, index: int):
        self.message = message
        self.messages = messages
        self.logprobs = logprobs
        self.finish_reason = finish_reason
        self.index = index

class ChatCompletionResponse:
    def __init__(self, id: str, object: str, created: int, model: str, usage: ModelUsage, choices: list[ModelChoice]):
        self.id = id
        self.object = object
        self.created = created
        self.model = model
        self.usage = usage
        self.choices = choices
    
    def to_json(self):
        return {
            "id": self.id,
            "object": self.object,
            "created": self.created,
            "model": self.model,
            "usage": {
                "prompt_tokens": self.usage.prompt_tokens,
                "completion_tokens": self.usage.completion_tokens,
                "total_tokens": self.usage.total_tokens
            },
            "choices": [
                {
                    "message": self.choices[0].message,
                    "messages": self.choices[0].messages,
                    "logprobs": self.choices[0].logprobs,
                    "finish_reason": self.choices[0].finish_reason,
                    "index": self.choices[0].index
                }
            ]
        }
