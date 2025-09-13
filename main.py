import os
import time

from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader
from langchain.chains import RetrievalQA

from src.loaders import DocumentLoader
from src.summerize import Summerize
from src.utils import Path, humanize_seconds


CWD = os.path.dirname(os.path.abspath(__file__))
WATCH_DIR = Path(CWD, "examples/")
FAISS_DIR = Path(CWD, "faiss_index")
FAISS_INDEX_NAME = WATCH_DIR.replace("/", "_").replace("\\", "_").strip("_")

loader = DirectoryLoader(WATCH_DIR,
                         loader_cls=DocumentLoader,
                         loader_kwargs={"excludes": [
                             ".git/", "img/", "assets/"]},
                         show_progress=True,
                         use_multithreading=True,
                         recursive=True)
documents = loader.load()

print(f"Loaded {len(documents)} documents...\n")  # Debugging

if not documents:
    print("No documents found. Please add supported files to the target directory.")
    exit(1)

# Use OllamaEmbeddings for local embedding
embeddings = OllamaEmbeddings(
    model="embeddinggemma:300m",
)

try:
    vector_store = FAISS.load_local(
        FAISS_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
        index_name=FAISS_INDEX_NAME,
    )
    print("Loaded existing FAISS index.")
except Exception as e:
    print(f"Failed to load existing FAISS index", e)
    print("Creating a new FAISS index...")
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local(FAISS_DIR, index_name=FAISS_INDEX_NAME)

# Use OllamaLLM for local LLM inference
llm = OllamaLLM(
    model="gemma3:1b",
    system="You are an expert assistant to search information with given documents."
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(),
    return_source_documents=True,
)


def answer_query(query: str):
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
