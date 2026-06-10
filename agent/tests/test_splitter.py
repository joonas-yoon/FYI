"""청킹 단위 테스트."""

from __future__ import annotations

from langchain_core.documents import Document

from app.core.splitter import split_documents


def test_long_document_is_split_into_chunks():
    long_text = "# 제목\n" + ("이것은 테스트 문장입니다. " * 400)
    docs = [Document(page_content=long_text, metadata={"source": "x.md", "title": "제목"})]

    chunks = split_documents(docs)

    assert len(chunks) > 1
    # 메타데이터가 청크로 전파되어야 한다.
    assert all(c.metadata["source"] == "x.md" for c in chunks)
