"""임베딩 모델 로더. 프로바이더(huggingface | ollama)에 따라 분기한다.

huggingface: CUDA 가 있으면 GPU 에서 동작(``EMBEDDING_DEVICE=auto`` 자동 감지).
한국어/다국어 성능이 좋은 ``BAAI/bge-m3`` 를 기본값으로 사용한다.
ollama: 로컬 Ollama 서버의 임베딩 모델(``embeddinggemma:300m`` 등)을 사용한다.

주의: 프로바이더/모델에 따라 벡터 차원이 다르므로(bge-m3=1024, embeddinggemma=768)
변경 시 ``scripts/ingest.py --reset`` 으로 재인덱싱이 필요하다.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from langchain_core.embeddings import Embeddings

from app.config import get_settings

logger = logging.getLogger(__name__)


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


@lru_cache
def get_embeddings() -> Embeddings:
    """임베딩 모델 싱글턴. 최초 호출 시 모델을 내려받아 로드한다."""
    settings = get_settings()
    provider = settings.embedding_provider.lower()

    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        logger.info(
            "임베딩 로드: ollama/%s @ %s",
            settings.ollama_embedding_model,
            settings.ollama_base_url,
        )
        return OllamaEmbeddings(
            model=settings.ollama_embedding_model,
            base_url=settings.ollama_base_url,
        )

    if provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings

        device = _resolve_device(settings.embedding_device)
        logger.info(
            "임베딩 로드: huggingface/%s device=%s", settings.embedding_model, device
        )
        return HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": device},
            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": settings.embedding_batch_size,
            },
        )

    raise ValueError(
        f"지원하지 않는 EMBEDDING_PROVIDER: {settings.embedding_provider!r} (huggingface | ollama)"
    )
