import os

from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader

from src.loaders import DocumentLoader
from src.utils import Path, humanize_seconds
import time


CWD = os.path.dirname(os.path.abspath(__file__))
WATCH_DIR = Path(CWD, "target")
FAISS_DIR = Path(CWD, "faiss_index")

loader = DirectoryLoader(WATCH_DIR, loader_cls=DocumentLoader)
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
    )
    print("Loaded existing FAISS index.")
except Exception as e:
    print(f"Failed to load existing FAISS index")
    print("Creating a new FAISS index...")
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local(FAISS_DIR)

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
        user_query = input("Ask a question: ")
        if user_query.lower() in ["exit", "quit"]:
            break

        start_time = time.time()
        answer = answer_query(user_query)
        elapsed = time.time() - start_time

        print("Answer:", answer)
        print("[", humanize_seconds(elapsed), "]\n")
