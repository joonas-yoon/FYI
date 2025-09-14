
from pydantic import BaseModel
from langchain_core.documents.base import Document


class AnswerDict(BaseModel):
    query: str
    result: str
    source_documents: list[Document]
