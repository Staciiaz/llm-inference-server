from loguru import logger

from ..config import getenv
from .chat_model import ChatModel


def load_model(model_id: str, dtype: str) -> ChatModel:
    if model_id == "meta-llama/Llama-2-7b":
        from .llama import Llama
        return Llama(model_id)
    elif model_id == "google/gemma-7b-it":
        from .gemma import Gemma
        return Gemma(model_id)
    elif model_id == "mistralai/Mistral-7B-Instruct-v0.2":
        from .mistral import Mistral
        return Mistral(model_id)
    elif model_id is None:
        from .mock import MockChatModel
        return MockChatModel()
    else:
        raise ValueError(f"Invalid model id: {model_id}")

# Load the model from the environment variable
model_id = getenv("MODEL_ID")
logger.info(f"Load model: {model_id}")
model = load_model(model_id, "FP32")

def get_model() -> ChatModel:
    return model
