import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.document_loaders import FileValidator
from langchain.document_loaders import PDFLoader, UnstructuredHTMLLoader

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

# Get current working directory
CWD = os.getcwd()

# Directory to watch
WATCH_DIR = os.join(CWD, "target")  # Change to your target directory

# Load all text files from the directory


def get_loader_cls(path):
    if path.lower().endswith(".pdf"):
        return PDFLoader(path)
    elif path.lower().endswith(".html"):
        return UnstructuredHTMLLoader(path)
    else:
        return TextLoader(path)


loader = DirectoryLoader(WATCH_DIR, loader_cls=get_loader_cls)
documents = loader.load()

# Create embeddings and vector store
embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_documents(documents, embeddings)

# Set up retrieval QA chain
llm = OpenAI(temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever()
)


def answer_query(query: str):
    """Respond to a query using indexed documents."""
    return qa_chain.run(query)


if __name__ == "__main__":
    while True:
        user_query = input("Ask a question: ")
        if user_query.lower() in ["exit", "quit"]:
            break
        answer = answer_query(user_query)
        print("Answer:", answer)
