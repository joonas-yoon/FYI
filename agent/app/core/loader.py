"""위키 마크다운 문서 로더.

지정된 디렉터리 하위의 ``*.md`` 파일을 재귀적으로 읽어 LangChain ``Document`` 로 변환합니다.
한국어 문서를 위해 UTF-8 로 읽고, 첫 번째 ``# 제목`` 또는 파일명을 title 메타데이터로 사용합니다.
"""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.documents import Document

from app.config import get_settings

logger = logging.getLogger(__name__)


def _extract_title(text: str, fallback: str) -> str:
    """마크다운 첫 H1(``# ``) 헤더를 제목으로 추출, 없으면 fallback."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return fallback


def load_wiki_documents(
    wiki_path: Path | str | None = None,
    glob: str | None = None,
) -> list[Document]:
    """위키 디렉터리에서 마크다운 문서들을 로드한다.

    Args:
        wiki_path: 문서 루트 디렉터리. 생략 시 설정값 사용.
        glob: 매칭 패턴. 생략 시 설정값 사용.

    Returns:
        파일별 ``Document`` 리스트 (메타데이터: source, abs_path, title).
    """
    settings = get_settings()
    root = Path(wiki_path) if wiki_path is not None else settings.wiki_path
    pattern = glob or settings.wiki_glob

    if not root.exists():
        logger.warning("위키 경로가 존재하지 않음: %s", root)
        return []

    documents: list[Document] = []
    for path in sorted(root.glob(pattern)):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if not text.strip():
            continue
        rel = path.relative_to(root)
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": str(rel).replace("\\", "/"),
                    "abs_path": str(path),
                    "title": _extract_title(text, path.stem),
                },
            )
        )

    logger.info("위키 문서 로드: %d 개 (root=%s)", len(documents), root)
    return documents
