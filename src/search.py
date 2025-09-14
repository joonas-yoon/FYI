
import os

from langchain.chains import RetrievalQA

from src.setups import load_embedding_model, load_llm_model, setup_documents, setup_vector_store
from src.types import AnswerDict
from src.utils import Path


CWD = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = Path(CWD, "..")

WATCH_DIR = Path(BASE_DIR, "examples/")


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
    save_dir=Path(BASE_DIR, "faiss_index"),
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
