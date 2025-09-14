
from langchain_core.documents.base import Document
from pydantic import BaseModel


class AnswerDict(BaseModel):
    query: str
    result: str
    source_documents: list[Document]
