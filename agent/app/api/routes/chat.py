"""채팅 질의응답 엔드포인트."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.schemas import ChatRequest, ChatResponse, Source
from app.services import chat as chat_service

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse, summary="위키 문서 기반 질의응답")
def chat(request: ChatRequest) -> ChatResponse:
    result = chat_service.answer(request.message, request.session_id)

    sources = [
        Source(
            title=doc.metadata.get("title", ""),
            source=doc.metadata.get("source", ""),
            snippet=doc.page_content[:200].strip(),
        )
        for doc in result.get("context", [])
    ]
    return ChatResponse(
        answer=result.get("answer", ""),
        session_id=request.session_id,
        sources=sources,
    )


@router.delete("/chat/{session_id}", summary="대화 세션 초기화")
def reset_session(session_id: str) -> dict:
    chat_service.reset_session(session_id)
    return {"status": "reset", "session_id": session_id}
