"""LLM 로더. 프로바이더(anthropic | ollama)에 따라 분기한다."""

from __future__ import annotations

import logging
from functools import lru_cache

from langchain_core.language_models import BaseChatModel

from app.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache
def get_llm() -> BaseChatModel:
    """채팅 LLM 싱글턴.

    ``LLM_PROVIDER`` 가 ``anthropic`` 이면 Claude(ChatAnthropic), ``ollama`` 이면
    로컬 Ollama(ChatOllama)를 반환한다.
    """
    settings = get_settings()
    provider = settings.llm_provider.lower()

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        logger.info("LLM 로드: ollama/%s @ %s", settings.ollama_llm_model, settings.ollama_base_url)
        return ChatOllama(
            model=settings.ollama_llm_model,
            base_url=settings.ollama_base_url,
            temperature=settings.llm_temperature,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        logger.info("LLM 로드: anthropic/%s", settings.llm_model)
        return ChatAnthropic(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            api_key=settings.anthropic_api_key or None,
            timeout=60,
        )

    raise ValueError(f"지원하지 않는 LLM_PROVIDER: {settings.llm_provider!r} (anthropic | ollama)")
