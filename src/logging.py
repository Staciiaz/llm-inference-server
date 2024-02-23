import sys
import logging

logger = logging.getLogger("uvicorn")
logger.addHandler(logging.StreamHandler(sys.stdout))
