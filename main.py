import os

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI

from src.routers import chat_completions_v1

app = FastAPI()
app.include_router(
    chat_completions_v1.router,
    prefix="/v1",
    tags=["chat_completions_v1"],
)

if __name__ == '__main__':
    load_dotenv()
    server_port = int(os.getenv("PORT"))
    uvicorn.run(app, port=server_port, env_file='.env')
