"""임베딩 모델 로더.

CUDA 가 있으면 GPU 에서 동작합니다. ``EMBEDDING_DEVICE=auto`` 이면 자동 감지합니다.
한국어/다국어 성능이 좋은 ``BAAI/bge-m3`` 를 기본값으로 사용합니다.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

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
def get_embeddings() -> HuggingFaceEmbeddings:
    """임베딩 모델 싱글턴. 최초 호출 시 모델을 내려받아 로드합니다."""
    settings = get_settings()
    device = _resolve_device(settings.embedding_device)
    logger.info("임베딩 모델 로드: model=%s device=%s", settings.embedding_model, device)
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": device},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": settings.embedding_batch_size,
        },
    )
