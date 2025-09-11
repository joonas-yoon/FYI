from langchain.document_loaders import TextLoader, UnstructuredHTMLLoader
from langchain.document_loaders import PyPDFLoader
import os


def text_base_loader(path):
    if path.lower().endswith(".pdf"):
        return PyPDFLoader(path).load()
    elif path.lower().endswith(".html"):
        return UnstructuredHTMLLoader(path).load()
    else:
        return TextLoader(path).load()


class DocumentLoader(TextLoader):
    def __init__(self, file_path):
        super().__init__(file_path)

    def load(self):
        return text_base_loader(self.file_path)
