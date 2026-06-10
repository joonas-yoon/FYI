"""위키 로더 단위 테스트 (한국어 문서, 제목 추출, 재귀 탐색)."""

from __future__ import annotations

from pathlib import Path

from app.core.loader import load_wiki_documents


def test_loads_markdown_recursively_with_titles(tmp_path: Path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.md").write_text("# 휴가 정책\n연차는 15일입니다.", encoding="utf-8")
    (tmp_path / "sub" / "b.md").write_text("# 출장 규정\n사전 승인 필요.", encoding="utf-8")
    (tmp_path / "ignore.txt").write_text("마크다운 아님", encoding="utf-8")

    docs = load_wiki_documents(wiki_path=tmp_path, glob="**/*.md")

    assert len(docs) == 2
    titles = {d.metadata["title"] for d in docs}
    assert titles == {"휴가 정책", "출장 규정"}
    # 상대 경로는 POSIX 스타일로 정규화된다.
    sources = {d.metadata["source"] for d in docs}
    assert "a.md" in sources
    assert "sub/b.md" in sources


def test_title_falls_back_to_filename(tmp_path: Path):
    (tmp_path / "no_heading.md").write_text("제목 헤더가 없는 본문", encoding="utf-8")

    docs = load_wiki_documents(wiki_path=tmp_path, glob="**/*.md")

    assert len(docs) == 1
    assert docs[0].metadata["title"] == "no_heading"


def test_missing_directory_returns_empty(tmp_path: Path):
    docs = load_wiki_documents(wiki_path=tmp_path / "does-not-exist", glob="**/*.md")
    assert docs == []
