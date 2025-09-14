
import time

from src.search import answer_query
from src.summerize import Summerize
from src.utils import humanize_seconds


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
