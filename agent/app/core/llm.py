"""Claude LLM 로더 (langchain-anthropic)."""

from __future__ import annotations

import logging
from functools import lru_cache

from langchain_anthropic import ChatAnthropic

from app.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache
def get_llm() -> ChatAnthropic:
    """ChatAnthropic 싱글턴. api_key 가 비어 있으면 ANTHROPIC_API_KEY 환경변수를 사용."""
    settings = get_settings()
    logger.info("LLM 로드: %s", settings.llm_model)
    return ChatAnthropic(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        api_key=settings.anthropic_api_key or None,
        timeout=60,
    )
