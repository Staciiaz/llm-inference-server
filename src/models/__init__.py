from ..config import getenv
from .chat_model import ChatModel
from .gemma_chat import GemmaChat
from .llama_chat import LlamaChat
from .llava_base import LlavaBase
from .mistral_chat import MistralChat
from .mock import MockChatModel


def load_model(model_id: str, dtype: str, device: str) -> ChatModel:
    if model_id in ("meta-llama/Llama-2-7b-chat-hf", "meta-llama/Llama-2-13b-chat-hf"):
        return LlamaChat(model_id, dtype, device)
    elif model_id in ("google/gemma-2b-it", "google/gemma-7b-it"):
        return GemmaChat(model_id, dtype, device)
    elif model_id in ("mistralai/Mistral-7B-Instruct-v0.2", "mistralai/Mixtral-8x7B-Instruct-v0.1"):
        return MistralChat(model_id, dtype, device)
    elif model_id in ("llava-hf/llava-1.5-7b-hf", "llava-hf/llava-1.5-13b-hf"):
        return LlavaBase(model_id, dtype, device)
    elif model_id is None:
        return MockChatModel()
    else:
        raise ValueError(f"Invalid model id: {model_id}")

# Load the model from the environment variable
model_id = getenv("MODEL_ID")
dtype = getenv("DTYPE")
device = getenv("DEVICE")
print(f"Load model: {model_id} {dtype} {device}")

if device in ("cpu", "mps") and dtype in ("int8", "int4"):
    raise ValueError(f"Invalid device: {device} for dtype: {dtype}")

model = load_model(model_id, dtype, device)

def get_model() -> ChatModel:
    return model
