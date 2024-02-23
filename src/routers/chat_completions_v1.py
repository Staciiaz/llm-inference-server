import time
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from ..models import ChatModel, get_model
from ..types.api_v1 import (ChatCompletionRequest, ChatCompletionResponse,
                            ModelChoice, ModelUsage)

router = APIRouter()


@router.post("/chat/completions")
async def list(request: ChatCompletionRequest, model: Annotated[ChatModel, Depends(get_model)]):
    model_output = model.chat_completions(request.messages)
    response = ChatCompletionResponse(
        id="chatcmpl-abc123",
        object="chat.completion",
        created=int(time.time()),
        model=request.model,
        usage=ModelUsage(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0
        ),
        choices=[
            ModelChoice(
                message=model_output,
                messages={
                    "role": "assistant",
                    "content": model_output,
                },
                logprobs=None,
                finish_reason="stop",
                index=0
            )
        ]
    )
    return JSONResponse(content=response.to_json(), status_code=200)