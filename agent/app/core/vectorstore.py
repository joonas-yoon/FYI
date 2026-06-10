"""Chroma 벡터 스토어 (디스크에 영속화).

임베딩 프로바이더/모델마다 벡터 차원이 다르므로(bge-m3=1024, embeddinggemma=768),
컬렉션 이름에 임베딩 식별자를 접미사로 붙여 서로 섞이지 않도록 분리한다.
같은 ``persist_directory`` 안에서 프로필별 인덱스가 독립적으로 영속된다.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache

from langchain_chroma import Chroma

from app.config import Settings, get_settings
from app.core.embeddings import get_embeddings

logger = logging.getLogger(__name__)


def _embedding_identity(settings: Settings) -> str:
    """현재 활성 임베딩(프로바이더+모델)을 컬렉션명에 쓸 안전한 슬러그로 변환."""
    if settings.embedding_provider.lower() == "ollama":
        raw = f"ollama_{settings.ollama_embedding_model}"
    else:
        raw = f"hf_{settings.embedding_model}"
    return re.sub(r"[^0-9a-zA-Z]+", "_", raw).strip("_").lower()


def resolve_collection_name(settings: Settings) -> str:
    """기본 컬렉션명 + 임베딩 식별자 접미사 (예: ``fyi_wiki__hf_baai_bge_m3``)."""
    return f"{settings.collection_name}__{_embedding_identity(settings)}"


@lru_cache
def get_vectorstore() -> Chroma:
    """영속화된 Chroma 컬렉션 싱글턴."""
    settings = get_settings()
    settings.vector_store_dir.mkdir(parents=True, exist_ok=True)
    collection = resolve_collection_name(settings)
    logger.info(
        "벡터 스토어 연결: dir=%s collection=%s",
        settings.vector_store_dir,
        collection,
    )
    return Chroma(
        collection_name=collection,
        embedding_function=get_embeddings(),
        persist_directory=str(settings.vector_store_dir),
    )
