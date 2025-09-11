from langchain_ollama import OllamaLLM


class ModelFactory:
    @staticmethod
    def load_gemma():
        return OllamaLLM(model="gemma:3-270m")
