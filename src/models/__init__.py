import os

from .chat_model import ChatModel


def load_model(model_id: str, dtype: str) -> ChatModel:
    if model_id == "llama":
        from .llama import Llama
        return Llama(model_id)
    elif model_id == "gemma":
        from .gemma import Gemma
        return Gemma(model_id)
    elif model_id == "mistral":
        from .mistral import Mistral
        return Mistral(model_id)
    elif not model_id:
        from .mock import MockChatModel
        return MockChatModel()
    else:
        raise ValueError(f"Invalid model id: {model_id}")

# Load the model from the environment variable
model = load_model(os.getenv("MODEL_ID"), "FP32")

def get_model() -> ChatModel:
    return model
