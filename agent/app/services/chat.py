"""대화 엔진: RAG 체인에 세션별 대화 이력을 결합한다.

기본 구현은 인메모리 세션 저장소입니다 (프로세스 재시작 시 초기화).
운영에서 다중 워커/영속 이력이 필요하면 Redis 등으로 교체하세요 (seam: ``_get_history``).
"""

from __future__ import annotations

import logging
from functools import lru_cache

from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from app.core.rag import build_rag_chain

logger = logging.getLogger(__name__)

# session_id -> 대화 이력
_SESSION_STORE: dict[str, BaseChatMessageHistory] = {}


def _get_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in _SESSION_STORE:
        _SESSION_STORE[session_id] = InMemoryChatMessageHistory()
    return _SESSION_STORE[session_id]


@lru_cache
def get_chat_engine() -> RunnableWithMessageHistory:
    """대화 이력이 결합된 RAG 엔진 싱글턴."""
    chain = build_rag_chain()
    return RunnableWithMessageHistory(
        chain,
        _get_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )


def answer(message: str, session_id: str = "default") -> dict:
    """질문에 답한다.

    Returns:
        RAG 체인 결과 dict. 주요 키: ``answer`` (str), ``context`` (list[Document]).
    """
    engine = get_chat_engine()
    result = engine.invoke(
        {"input": message},
        config={"configurable": {"session_id": session_id}},
    )
    return result


def reset_session(session_id: str) -> None:
    """세션 대화 이력 삭제."""
    _SESSION_STORE.pop(session_id, None)
