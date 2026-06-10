"""루트 FastAPI 엔트리포인트.

FYI 와 agent 는 하나의 앱으로 통합되었다. 검색/요약/대화/인덱싱 라우트는 모두 단일
RAG 엔진 앱(``app.main``)이 제공한다. 이 모듈은 기존 실행 명령
``fastapi run watcher.py`` 호환성을 위해 그 앱을 그대로 재노출한다.

제공 엔드포인트:
    POST /search   검색 + (선택)요약(summarize) + (선택)컨텍스트 대화(session_id)
                   → AnswerDict 형태: {query, result, source_documents, summary?}
    POST /chat     멀티턴 대화
    POST /ingest   문서 인덱싱
    GET  /health   헬스체크
"""

from app.main import app

__all__ = ["app"]
