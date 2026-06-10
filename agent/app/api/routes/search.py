"""검색 엔드포인트.

AnswerDict 형태(query / result / source_documents)를 유지하면서 한 엔드포인트로
검색 · (선택)요약 · (선택)컨텍스트 대화를 제공한다. 루트 FYI 의 ``POST /search/`` 를
엔진(app) 안으로 통합한 것.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.schemas import SearchRequest, SearchResponse, SourceDocument
from app.services import search as search_service

router = APIRouter(tags=["search"])


@router.post("/search", response_model=SearchResponse, summary="검색 + (선택)요약 + (선택)컨텍스트 대화")
def search(request: SearchRequest) -> SearchResponse:
    result = search_service.search(
        request.q,
        session_id=request.session_id or "gateway",
        summarize=request.summarize,
    )
    return SearchResponse(
        query=result["query"],
        result=result["result"],
        source_documents=[
            SourceDocument(metadata=doc.metadata, page_content=doc.page_content)
            for doc in result["source_documents"]
        ],
        summary=result.get("summary"),
    )
