from ..config import getenv
from .chat_model import ChatModel
from .gemma import Gemma
from .llama import Llama
from .llava import Llava
from .mistral import Mistral
from .mock import MockChatModel


def load_model(model_id: str, dtype: str, device: str) -> ChatModel:
    if model_id in ("meta-llama/Llama-2-7b-chat-hf",):
        return Llama(model_id, dtype, device)
    elif model_id in ("google/gemma-2b-it", "google/gemma-7b-it"):
        return Gemma(model_id, dtype, device)
    elif model_id in ("mistralai/Mistral-7B-Instruct-v0.2",):
        return Mistral(model_id, dtype, device)
    elif model_id in ("llava-hf/llava-1.5-7b-hf",):
        return Llava(model_id, dtype, device)
    elif model_id is None:
        return MockChatModel()
    else:
        raise ValueError(f"Invalid model id: {model_id}")

# Load the model from the environment variable
model_id = getenv("MODEL_ID")
dtype = getenv("DTYPE")
device = getenv("DEVICE")
print(f"Load model: {model_id} {dtype} {device}")

if device in ("cuda", "mps") and dtype in ("int8", "int4"):
    raise ValueError(f"Invalid device: {device} for dtype: {dtype}")

model = load_model(model_id, dtype, device)

def get_model() -> ChatModel:
    return model
