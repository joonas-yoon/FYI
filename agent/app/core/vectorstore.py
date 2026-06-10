"""Chroma 벡터 스토어 (디스크에 영속화)."""

from __future__ import annotations

import logging
from functools import lru_cache

from langchain_chroma import Chroma

from app.config import get_settings
from app.core.embeddings import get_embeddings

logger = logging.getLogger(__name__)


@lru_cache
def get_vectorstore() -> Chroma:
    """영속화된 Chroma 컬렉션 싱글턴."""
    settings = get_settings()
    settings.vector_store_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "벡터 스토어 연결: dir=%s collection=%s",
        settings.vector_store_dir,
        settings.collection_name,
    )
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=get_embeddings(),
        persist_directory=str(settings.vector_store_dir),
    )
