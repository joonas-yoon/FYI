"""헬스 엔드포인트 스모크 테스트 (모델 로딩 없이 동작)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


def test_health_ok():
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["app"] == "fyi-agent"
