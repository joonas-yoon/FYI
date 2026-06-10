"""FastAPI 애플리케이션 엔트리포인트."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import chat, health, ingest
from app.config import get_settings
from app.logging_conf import setup_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """기동 시 로깅 설정 및 무거운 싱글턴(임베딩 모델) 워밍업."""
    setup_logging()
    settings = get_settings()
    logger.info("FYI-Agent 기동: env=%s, llm=%s", settings.app_env, settings.llm_model)

    # 테스트 환경에서는 모델 로딩을 건너뛴다 (CI/단위테스트 속도).
    if settings.app_env != "test":
        try:
            from app.core.embeddings import get_embeddings

            get_embeddings()
            logger.info("임베딩 모델 로딩 완료: %s", settings.embedding_model)
        except Exception:  # noqa: BLE001 - 워밍업 실패가 기동을 막지 않도록
            logger.exception("임베딩 모델 워밍업 실패 (요청 시 재시도)")

    yield
    logger.info("FYI-Agent 종료")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description="위키 마크다운 문서에 대한 한국어 RAG 질의응답 API",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health.router)
    app.include_router(chat.router)
    app.include_router(ingest.router)
    return app


app = create_app()
