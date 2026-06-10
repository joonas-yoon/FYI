"""검색 게이트웨이 서비스.

대화 엔진(``chat.answer``)을 호출해 루트 FYI 가 쓰던 AnswerDict 형태
(``query`` / ``result`` / ``source_documents``)로 변환해 반환한다.

- ``session_id``: 지정하면 멀티턴 대화 이력이 유지된다(컨텍스트 대화).
- ``summarize``: True 면 참고 문서 다이제스트(``summary``)를 함께 담는다.
"""

from __future__ import annotations

from app.services.chat import answer as _answer
from app.services.summarize import summarize as _summarize


def search(q: str, session_id: str = "gateway", summarize: bool = False) -> dict:
    """질의를 RAG 엔진에 위임하고 AnswerDict 형태 dict 로 반환한다."""
    res = _answer(q, session_id=session_id)  # {"answer", "context", ...}
    docs = res.get("context", [])
    out: dict = {
        "query": q,
        "result": res.get("answer", ""),
        "source_documents": docs,
    }
    if summarize:
        out["summary"] = _summarize(
            q, out["result"], [doc.metadata.get("source", "") for doc in docs]
        )
    return out
