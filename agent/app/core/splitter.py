"""문서 청킹.

마크다운 구조(헤더, 단락)를 우선 경계로 사용하는 재귀 분할기.
한국어 문서에서도 의미 단위가 잘 유지되도록 separators 를 구성했습니다.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings


def split_documents(docs: list[Document]) -> list[Document]:
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n# ", "\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
        keep_separator=True,
    )
    return splitter.split_documents(docs)
