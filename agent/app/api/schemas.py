"""API 요청/응답 스키마 (Pydantic)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., description="사용자 질문", examples=["휴가 신청은 어떻게 하나요?"])
    session_id: str = Field("default", description="대화 세션 식별자 (이력 유지용)")


class Source(BaseModel):
    title: str = Field(..., description="문서 제목")
    source: str = Field(..., description="위키 루트 기준 상대 경로")
    snippet: str = Field(..., description="참고한 문맥 일부")


class ChatResponse(BaseModel):
    answer: str = Field(..., description="모델 답변")
    session_id: str
    sources: list[Source] = Field(default_factory=list, description="참고한 문서 목록")


class IngestRequest(BaseModel):
    reset: bool = Field(False, description="기존 인덱스를 비우고 처음부터 재색인할지 여부")


class IngestResponse(BaseModel):
    documents: int = Field(..., description="색인한 원문 수")
    chunks: int = Field(..., description="생성된 청크 수")


class HealthResponse(BaseModel):
    status: str
    app: str
    env: str
