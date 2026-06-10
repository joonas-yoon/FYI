# FYI-Agent 개발/운영 하네스. 사용법:  make <target>
.DEFAULT_GOAL := help

help: ## 사용 가능한 명령 목록
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install: ## 로컬 개발 의존성 설치 (editable + dev extras)
	pip install -e ".[dev]"

run: ## 로컬에서 API 서버 실행 (reload)
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

ingest: ## 위키 문서를 읽어 벡터 인덱스 재생성 (--reset)
	python scripts/ingest.py --reset

ask: ## 터미널에서 대화형으로 질의 (HTTP 없이 RAG 파이프라인 직접 호출)
	python scripts/ask.py

gpu-check: ## GPU/CUDA 및 임베딩 모델 GPU 사용 진단 (로컬)
	python scripts/check_cuda.py

test: ## 단위 테스트 실행
	pytest -q

lint: ## ruff 린트/포맷 검사
	ruff check .

docker-build: ## Docker 이미지 빌드
	docker compose build

docker-up: ## 서비스 기동 (백그라운드)
	docker compose up -d

docker-down: ## 서비스 중지
	docker compose down

docker-ingest: ## 컨테이너 안에서 인덱싱 실행
	docker compose exec fyi-agent python scripts/ingest.py --reset

docker-gpu-check: ## 컨테이너 안에서 GPU/CUDA 진단
	docker compose exec fyi-agent python scripts/check_cuda.py

docker-logs: ## 컨테이너 로그 확인
	docker compose logs -f fyi-agent

.PHONY: help install run ingest ask gpu-check test lint docker-build docker-up docker-down docker-ingest docker-gpu-check docker-logs
