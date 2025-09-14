import re

from langchain_core.documents.base import Document
from langchain_community.document_loaders import PyPDFLoader, \
    TextLoader, UnstructuredCSVLoader, UnstructuredHTMLLoader, \
    UnstructuredImageLoader, JSONLoader, UnstructuredMarkdownLoader
from langchain_community.document_loaders.base import BaseLoader


class DummyLoader(BaseLoader):
    def load(self) -> list[Document]:
        return []


def match_ext(path, exts) -> bool:
    return path.lower().endswith(f".{exts}")


def is_pdf(path) -> bool:
    return match_ext(path, "pdf")


def is_csv(path) -> bool:
    return match_ext(path, "csv")


def is_html(path) -> bool:
    return match_ext(path, "html") \
        or match_ext(path, "htm")


def is_text(path) -> bool:
    return match_ext(path, "txt") \
        or match_ext(path, "yml") \
        or match_ext(path, "yaml")


def is_markdown(path) -> bool:
    return match_ext(path, "md")


def is_image(path) -> bool:
    return match_ext(path, "jpg") \
        or match_ext(path, "jpeg") \
        or match_ext(path, "png")


def is_json(path) -> bool:
    return match_ext(path, "json")


def AutoLoader(path) -> BaseLoader:
    if is_pdf(path):
        return PyPDFLoader(path, mode="single")
    elif is_html(path):
        return UnstructuredHTMLLoader(path)
    elif is_text(path):
        return TextLoader(path, autodetect_encoding=True)
    elif is_markdown(path):
        return UnstructuredMarkdownLoader(path,
                                          mode="elements",
                                          strategy="fast")
    elif is_csv(path):
        return UnstructuredCSVLoader(path)
    elif is_image(path):
        return UnstructuredImageLoader(path)
    elif is_json(path):
        return JSONLoader(path)
    else:
        print(path, "is unexpected")
        return DummyLoader()


class DocumentLoader(BaseLoader):
    def __init__(self, file_path, **kwargs):
        self.file_path = file_path
        self.excludes: list[str] = kwargs.get("excludes", [])

    def load(self) -> list[Document]:
        path = self.file_path
        for pattern in self.excludes:
            if re.search(pattern, path):
                return []  # Return empty list if match found

        try:
            return AutoLoader(path).load()
        except Exception as e:
            print("Failed to load:", path, e)
            return []
