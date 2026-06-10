"""인덱싱 트리거 엔드포인트."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.schemas import IngestRequest, IngestResponse
from app.services import ingestion

router = APIRouter(tags=["ingest"])


@router.post("/ingest", response_model=IngestResponse, summary="위키 문서 인덱싱")
def run_ingest(request: IngestRequest) -> IngestResponse:
    stats = ingestion.ingest(reset=request.reset)
    return IngestResponse(**stats)
