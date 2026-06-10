"""검색 결과 요약.

LLM 답변과 함께, 참고한 문서들을 중복 제거·집계해 사람이 읽기 좋은 다이제스트
문자열로 만든다. 루트 FYI 의 ``src/summerize.py`` 로직을 엔진(app) 안으로 옮긴 것.
"""

from __future__ import annotations

from collections import Counter


def _tail_path(path: str) -> str:
    """긴 경로는 마지막 3개 구간만 ``.../a/b/c`` 형태로 축약."""
    parts = path.replace("\\", "/").split("/")
    if len(parts) < 4:
        return path
    return ".../" + "/".join(parts[-3:])


def summarize(query: str, result: str, sources: list[str]) -> str:
    """질의·답변·참고 문서(경로 목록)를 다이제스트 문자열로 요약한다."""
    counter = Counter(src or "unknown" for src in sources)
    source_lines = "\n".join(
        f"* {_tail_path(src)} ({count})" for src, count in counter.most_common()
    )
    return f"Query: {query}\nAnswer: {result}\nSources:\n{source_lines}"
