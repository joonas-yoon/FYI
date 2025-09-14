from collections import Counter

from src.types import AnswerDict


class Summerize:
    def __init__(self, answer: AnswerDict | dict):
        self.answer = AnswerDict(**answer)
        self.query = self.answer.query
        self.result = self.answer.result
        self.docs = self.answer.source_documents or []
        self.sources = [doc.metadata.get("source", "unknown")
                        for doc in self.docs]

        counter = Counter(self.sources)

        def tail_path(path: str) -> str:
            parts = path.replace("\\", "/").split("/")
            if len(parts) < 4:
                return path
            return ".../" + "/".join(parts[-3:])

        def pair_to_str(pair):
            return f"* {tail_path(pair[0])} ({pair[1]})"

        self.sources_count = "\n".join(
            [pair_to_str(p) for p in counter.most_common()]
        )

    def __str__(self) -> str:
        return f"Query: {self.query}\n" \
            f"Answer: {self.result}\n" \
            f"Sources:\n{self.sources_count}"

    def __repr__(self):
        return self.__str__()
