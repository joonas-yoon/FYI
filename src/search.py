"""검색 게이트웨이.

루트 FYI 는 더 이상 자체 RAG 파이프라인을 구성하지 않는다. 질의는 단일 RAG 엔진인
``agent`` (``app.services.chat``) 에 위임하고, 결과를 루트가 쓰던 ``AnswerDict`` 형태
(``query`` / ``result`` / ``source_documents``) 로 변환해 반환한다.

LLM/임베딩 프로바이더(Claude ↔ Ollama)와 인덱싱은 ``agent`` 쪽 설정/스크립트로 관리한다.
"""

from app.services.chat import answer as _agent_answer

from src.types import AnswerDict


def answer_query(query: str, session_id: str = "gateway") -> AnswerDict:
    """질의를 agent 엔진에 위임하고 결과를 AnswerDict 형태로 매핑한다."""
    res = _agent_answer(query, session_id=session_id)  # {"answer", "context", ...}
    return {
        "query": query,
        "result": res["answer"],
        "source_documents": res.get("context", []),
    }
