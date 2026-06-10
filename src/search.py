"""검색 게이트웨이 (로컬 CLI 용 얇은 래퍼).

실제 검색/요약/대화 로직은 단일 RAG 엔진(``app.services.search``)에 있다. 이 모듈은
``cli.py`` 등 루트에서 쓰던 ``answer_query`` 시그니처와 AnswerDict 형태를 유지한다.
HTTP API 는 ``app`` 의 ``POST /search`` (watcher.py 가 재노출) 가 담당한다.
"""

from app.services.search import search as _search

from src.types import AnswerDict


def answer_query(
    query: str, session_id: str = "gateway", summarize: bool = False
) -> AnswerDict:
    """질의를 agent 엔진에 위임한다. AnswerDict 형태(dict) 로 반환."""
    return _search(query, session_id=session_id, summarize=summarize)
