"""질의 하네스: HTTP 없이 터미널에서 RAG 파이프라인을 직접 호출한다.

사용:
    python scripts/ask.py "휴가 정책 알려줘"   # 단발성 질문
    python scripts/ask.py                       # 대화형 모드
"""

from __future__ import annotations

import argparse
import uuid

from app.logging_conf import setup_logging
from app.services.chat import answer


def _print_sources(result: dict) -> None:
    for doc in result.get("context", []):
        title = doc.metadata.get("title", "")
        source = doc.metadata.get("source", "")
        print(f"  - {title} ({source})")


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="터미널에서 FYI-Agent 에 질문합니다.")
    parser.add_argument("question", nargs="*", help="단발성 질문. 생략하면 대화형 모드.")
    args = parser.parse_args()

    session_id = str(uuid.uuid4())

    if args.question:
        result = answer(" ".join(args.question), session_id)
        print(result.get("answer", ""))
        _print_sources(result)
        return

    print("FYI-Agent 대화형 모드입니다. 종료하려면 'exit' 또는 Ctrl+C.")
    while True:
        try:
            question = input("\n질문> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        result = answer(question, session_id)
        print(f"\n{result.get('answer', '')}")
        _print_sources(result)


if __name__ == "__main__":
    main()
