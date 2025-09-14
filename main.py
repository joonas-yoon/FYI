
import os
import time

from langchain.chains import RetrievalQA

from src.setups import load_embedding_model, load_llm_model, setup_documents, setup_vector_store
from src.summerize import Summerize
from src.types import AnswerDict
from src.utils import Path, humanize_seconds


CWD = os.path.dirname(os.path.abspath(__file__))

WATCH_DIR = Path(CWD, "examples/")


documents = setup_documents(WATCH_DIR)
print(f"Loaded {len(documents)} documents...\n")  # Debugging

if not documents:
    print("No documents found. Please add supported files to the target directory.")
    exit(1)

embeddings = load_embedding_model()
llm = load_llm_model()

vector_store = setup_vector_store(
    documents=documents,
    embeddings=embeddings,
    save_dir=Path(CWD, "faiss_index"),
    index_name=WATCH_DIR.replace("/", "_").replace("\\", "_").strip("_")
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(),
    return_source_documents=True,
)


def answer_query(query: str) -> AnswerDict:
    return qa_chain.invoke({"query": query})


if __name__ == "__main__":

    while True:
        user_query = input("> Ask a question: ")
        if user_query.lower() in ["exit", "quit"]:
            break

        start_time = time.time()
        answer = answer_query(user_query)
        elapsed = time.time() - start_time

        print("=" * 40)
        print("[", humanize_seconds(elapsed), "]\n")
        print(Summerize(answer), "\n")
