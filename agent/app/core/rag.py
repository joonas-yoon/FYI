"""RAG 체인 구성.

대화 이력을 반영해 질문을 독립형으로 재작성(history-aware)한 뒤,
검색된 문맥을 바탕으로 Claude 가 한국어로 답변하도록 구성합니다.

반환되는 체인은 ``{"input", "chat_history"}`` 를 입력받아
``{"input", "chat_history", "context", "answer"}`` 를 출력합니다.
"""

from __future__ import annotations

from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable

from app.config import get_settings
from app.core.llm import get_llm
from app.core.vectorstore import get_vectorstore

# 검색된 문맥만 근거로 답하도록 강제하는 시스템 프롬프트.
ANSWER_SYSTEM_PROMPT = (
    "당신은 사내 위키 문서를 기반으로 답변하는 한국어 어시스턴트입니다. "
    "아래에 제공된 문맥(context)만을 근거로 정확하고 간결하게 답변하세요. "
    "문맥에서 답을 찾을 수 없으면 추측하지 말고 '제공된 문서에서 관련 내용을 찾지 못했습니다'라고 답하세요. "
    "가능하면 어떤 문서를 참고했는지 함께 알려주세요.\n\n"
    "문맥:\n{context}"
)

# 후속 질문을 대화 없이도 이해 가능한 독립형 질문으로 재작성하는 프롬프트.
CONTEXTUALIZE_SYSTEM_PROMPT = (
    "이전 대화 내용을 참고하여, 사용자의 마지막 질문을 대화 맥락 없이도 이해할 수 있는 "
    "독립적인 질문으로 다시 작성하세요. 질문에 답하지는 말고, 필요할 때만 재작성하고 "
    "그렇지 않으면 원래 질문을 그대로 반환하세요."
)


def build_rag_chain() -> Runnable:
    settings = get_settings()
    llm = get_llm()
    retriever = get_vectorstore().as_retriever(
        search_kwargs={"k": settings.retrieval_top_k}
    )

    contextualize_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", CONTEXTUALIZE_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_prompt
    )

    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ANSWER_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    answer_chain = create_stuff_documents_chain(llm, answer_prompt)

    return create_retrieval_chain(history_aware_retriever, answer_chain)
