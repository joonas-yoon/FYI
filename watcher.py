from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.params import Body
from fastapi.responses import JSONResponse

from src.search import answer_query
from src.types import AnswerDict


app = FastAPI()


class SearchRequest(BaseModel):
    q: str = Body(..., description="The search query string")


class DocumenrResponseModel(BaseModel):
    metadata: dict
    page_content: str


class SearchResponse(BaseModel):
    query: str
    result: str
    source_documents: list[DocumenrResponseModel]


class SearchResultAdapter:
    def adapt(self, answer: AnswerDict) -> SearchResponse:
        return SearchResponse(
            query=answer['query'],
            result=answer['result'],
            source_documents=[
                DocumenrResponseModel(
                    page_content=doc.page_content,
                    metadata=doc.metadata
                ) for doc in answer['source_documents']
            ]
        )


@app.post("/search/", response_model=SearchResponse)
async def search(request: SearchRequest):
    query = request.q
    answer: AnswerDict = answer_query(query)
    response = SearchResultAdapter().adapt(answer)
    return response
