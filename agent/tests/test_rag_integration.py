"""엔드투엔드 통합 테스트 (opt-in).

임베딩 모델 다운로드 + 실제 Anthropic 호출이 필요하므로 기본적으로 건너뜁니다.
실행하려면:  RUN_INTEGRATION=1 ANTHROPIC_API_KEY=sk-ant-... pytest tests/test_rag_integration.py
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION") != "1",
    reason="통합 테스트는 RUN_INTEGRATION=1 일 때만 실행됩니다.",
)


def test_ingest_then_answer(tmp_path, monkeypatch):
    # 임시 위키/벡터스토어로 설정 격리.
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    (wiki / "policy.md").write_text(
        "# 휴가 정책\n신입사원의 연차 휴가는 입사 첫 해에 11일이 부여됩니다.",
        encoding="utf-8",
    )
    monkeypatch.setenv("WIKI_PATH", str(wiki))
    monkeypatch.setenv("VECTOR_STORE_DIR", str(tmp_path / "vs"))
    monkeypatch.setenv("APP_ENV", "integration")

    # 설정 캐시를 비워 새 환경변수를 반영.
    from app.config import get_settings

    get_settings.cache_clear()

    from app.services.chat import answer
    from app.services.ingestion import ingest

    stats = ingest(reset=True)
    assert stats["documents"] == 1
    assert stats["chunks"] >= 1

    result = answer("신입사원 연차는 며칠인가요?", session_id="test")
    assert "11" in result["answer"]
    assert len(result.get("context", [])) >= 1
