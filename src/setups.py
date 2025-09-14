import os

from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader

from src.loaders import DocumentLoader
from src.utils import Path


CWD = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = Path(CWD, "..")


def setup_documents(documents_path: str):
    loader = DirectoryLoader(documents_path,
                             loader_cls=DocumentLoader,
                             loader_kwargs={"excludes": [
                                 ".git/", "img/", "assets/"]},
                             show_progress=True,
                             use_multithreading=True,
                             recursive=True)
    return loader.load()


# Use OllamaEmbeddings for local embedding
def load_embedding_model():
    return OllamaEmbeddings(
        model="embeddinggemma:300m",
        keep_alive=3000,
    )


# Use OllamaLLM for local LLM inference
def load_llm_model():
    return OllamaLLM(
        model="gemma3:1b",
        system="You are an expert assistant to search information with given documents."
    )


# Setup FAISS vector store
def setup_vector_store(documents, embeddings, save_dir: str, index_name: str):
    try:
        vector_store = FAISS.load_local(
            save_dir,
            embeddings,
            allow_dangerous_deserialization=True,
            index_name=index_name,
        )
        print("Loaded existing FAISS index.")
    except Exception as e:
        print(f"Failed to load existing FAISS index", index_name)
        print("Creating a new FAISS index...")
        vector_store = FAISS.from_documents(documents, embeddings)
        vector_store.save_local(save_dir, index_name=index_name)
    return vector_store
