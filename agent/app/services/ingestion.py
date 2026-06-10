"""인덱싱 파이프라인: 위키 문서 로드 → 청킹 → 임베딩 → 벡터 스토어 저장."""

from __future__ import annotations

import logging

from app.core.loader import load_wiki_documents
from app.core.splitter import split_documents
from app.core.vectorstore import get_vectorstore

logger = logging.getLogger(__name__)


def _chunk_ids(chunks: list) -> list[str]:
    """source + 순번으로 결정적 ID 생성 → 재인덱싱 시 중복 방지."""
    counters: dict[str, int] = {}
    ids: list[str] = []
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        idx = counters.get(source, 0)
        counters[source] = idx + 1
        ids.append(f"{source}::{idx}")
    return ids


def ingest(reset: bool = False) -> dict:
    """위키 문서를 인덱싱한다.

    Args:
        reset: True 면 기존 컬렉션을 비우고 처음부터 다시 색인.

    Returns:
        {"documents": 원문 수, "chunks": 청크 수}
    """
    if reset:
        logger.info("기존 인덱스 초기화")
        try:
            get_vectorstore().delete_collection()
        except Exception:  # noqa: BLE001 - 컬렉션이 없을 수 있음
            logger.debug("초기화할 기존 컬렉션이 없음")
        get_vectorstore.cache_clear()

    vectorstore = get_vectorstore()
    documents = load_wiki_documents()
    chunks = split_documents(documents)

    if chunks:
        vectorstore.add_documents(chunks, ids=_chunk_ids(chunks))

    logger.info("인덱싱 완료: 문서 %d → 청크 %d", len(documents), len(chunks))
    return {"documents": len(documents), "chunks": len(chunks)}
