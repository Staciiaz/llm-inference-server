import os

from dotenv import load_dotenv

load_dotenv()

def getenv(key: str) -> str:
    return os.getenv(key)
