from langchain_community.embeddings import HuggingFaceEmbeddings
import os
from langchain.vectorstores import FAISS
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline

from sentence_transformers import SentenceTransformer
from langchain.embeddings import SentenceTransformerEmbeddings

from src.loaders import DocumentLoader
from src.models import ModelFactory

CWD = os.getcwd()
WATCH_DIR = os.path.join(CWD, "target")

loader = DirectoryLoader(WATCH_DIR, loader_cls=DocumentLoader)
documents = loader.load()

print(f"Loaded {len(documents)} documents")  # Debugging

if not documents:
    print("No documents found. Please add supported files to the target directory.")
    exit(1)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)

vector_store = FAISS.from_documents(documents, embeddings)

# Use a local model for offline inference
# You can use any local model
local_pipeline = pipeline("text-generation", model="distilgpt2")
llm = HuggingFacePipeline(pipeline=local_pipeline)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever()
)


def answer_query(query: str):
    return qa_chain.invoke({"query": query})


if __name__ == "__main__":
    while True:
        user_query = input("Ask a question: ")
        if user_query.lower() in ["exit", "quit"]:
            break
        answer = answer_query(user_query)
        print("Answer:", answer)
