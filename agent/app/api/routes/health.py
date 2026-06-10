"""헬스체크 엔드포인트."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.schemas import HealthResponse
from app.config import get_settings

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="서비스 상태 확인")
def health() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(status="ok", app=settings.app_name, env=settings.app_env)
