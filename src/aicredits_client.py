"""
Shared AICredits client (OpenAI-compatible gateway).
Base URL: https://api.aicredits.in/v1
"""
import os
from openai import OpenAI
from loguru import logger

_client = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("AICREDITS_API_KEY")
        if not api_key:
            raise ValueError(
                "AICREDITS_API_KEY is required. "
                "Get one at https://aicredits.in → Dashboard → API Keys"
            )
        base_url = os.getenv("AICREDITS_BASE_URL", "https://api.aicredits.in/v1")
        _client = OpenAI(base_url=base_url, api_key=api_key)
        logger.info(f"AICredits client ready → {base_url}")
    return _client
