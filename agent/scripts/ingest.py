"""문서 인덱싱 하네스 — POST /chat 을 사용하기 전에 한 번 실행해야 합니다.

파이프라인:
  1. 로드   : WIKI_HOST_PATH 아래 *.md 파일을 재귀 탐색, UTF-8 로 읽어 Document 변환.
              첫 번째 H1 헤더(없으면 파일명)를 title 메타데이터로 보존.
  2. 청킹   : 마크다운 헤더·단락 경계를 우선 분할 기준으로 삼는 RecursiveCharacterTextSplitter.
              청크 크기 800자 / 오버랩 120자 (설정: chunk_size, chunk_overlap).
  3. 임베딩 : BAAI/bge-m3 모델로 각 청크를 1024차원 벡터로 변환 (GPU 사용 시 자동 CUDA).
  4. 저장   : Chroma 벡터 스토어(/data/vectorstore)에 "source::순번" 형식의 결정적 ID 로 upsert.
              동일 ID 가 이미 있으면 덮어쓰므로 증분 재실행이 안전.

사용:
    python scripts/ingest.py            # 신규·변경 문서만 추가 색인
    python scripts/ingest.py --reset    # 기존 컬렉션 전체 삭제 후 처음부터 재색인
"""

from __future__ import annotations

import argparse

from app.logging_conf import setup_logging
from app.services.ingestion import ingest


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="위키 마크다운을 벡터 스토어에 색인합니다.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="기존 인덱스를 비우고 처음부터 색인",
    )
    args = parser.parse_args()

    stats = ingest(reset=args.reset)
    print(f"색인 완료: 문서 {stats['documents']}개 → 청크 {stats['chunks']}개")


if __name__ == "__main__":
    main()
