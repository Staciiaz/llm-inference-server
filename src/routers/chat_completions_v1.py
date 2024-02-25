import time
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from ..models import ChatModel, get_model
from ..types.api_v1 import (APIChatCompletionRequest,
                            APIChatCompletionResponse, APIModelChoice,
                            APIModelUsage)

router = APIRouter()


@router.post("/")
async def list(request: APIChatCompletionRequest, model: Annotated[ChatModel, Depends(get_model)]):
    chat_completion_request = APIChatCompletionRequest(
        model=request.model,
        messages=request.messages,
        temperature=request.temperature
    )
    chat_completion_response = model.chat_completions(chat_completion_request)
    response = APIChatCompletionResponse(
        id="chatcmpl-abc123",
        object="chat.completion",
        created=int(time.time()),
        model=chat_completion_response.model,
        usage=APIModelUsage(
            prompt_tokens=chat_completion_response.input_tokens,
            completion_tokens=chat_completion_response.completion_tokens,
            total_tokens=chat_completion_response.output_tokens
        ),
        choices=[
            APIModelChoice(
                message={
                    "role": "assistant",
                    "content": chat_completion_response.response_message,
                },
                logprobs=None,
                finish_reason="stop",
                index=0
            )
        ]
    )
    return JSONResponse(content=response.to_json(), status_code=200)