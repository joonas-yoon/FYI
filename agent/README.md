# FYI-Agent

외부 위키 마크다운 문서를 대상으로 **채팅 질의응답**을 제공하는 RAG 서비스입니다.
LangChain + Claude(Anthropic) + FastAPI 로 구성되며, CUDA GPU 에서 임베딩을 수행하고
Docker 로 RESTful 하게 배포됩니다.

> 한국어 위키 문서 → 임베딩(GPU) → 벡터 검색 → Claude 답변

---

## 아키텍처

```
                  ┌────────────────────── FastAPI (REST) ──────────────────────┐
  사용자 ──HTTP──▶ │  /chat   /ingest   /health                                  │
                  └───────┬─────────────────────────┬───────────────────────────┘
                          │                          │
                  services.chat              services.ingestion
                          │                          │
              ┌───────────▼───────────┐   ┌──────────▼──────────┐
              │  RAG 체인 (core.rag)   │   │  로더→청킹→임베딩    │
              │  history-aware 검색    │   │  (core.loader/       │
              │  + Claude 답변         │   │   splitter/embeddings)│
              └───────┬───────┬────────┘   └──────────┬──────────┘
                      │       │                       │
              ChatAnthropic   └─────── Chroma 벡터 스토어 ◀──────┘
              (core.llm)            (core.vectorstore, 영속화)
                                            ▲
                                   BAAI/bge-m3 임베딩 (GPU)
```

| 컴포넌트 | 선택 | 위치 |
|---|---|---|
| LLM | Claude (`langchain-anthropic`) | [app/core/llm.py](app/core/llm.py) |
| 임베딩 | `BAAI/bge-m3` (한국어/다국어, GPU) | [app/core/embeddings.py](app/core/embeddings.py) |
| 벡터 스토어 | Chroma (디스크 영속화) | [app/core/vectorstore.py](app/core/vectorstore.py) |
| 문서 로더 | 커스텀 마크다운 로더 (UTF-8) | [app/core/loader.py](app/core/loader.py) |
| RAG 체인 | history-aware retriever | [app/core/rag.py](app/core/rag.py) |
| API | FastAPI | [app/main.py](app/main.py) |

---

## 디렉터리 구조

```
app/
  main.py            # FastAPI 엔트리포인트 + lifespan(모델 워밍업)
  config.py          # 설정 (env/.env)
  api/
    schemas.py       # 요청/응답 모델
    routes/          # /health, /chat, /ingest
  core/              # 임베딩·벡터스토어·LLM·로더·분할·RAG 체인
  services/
    ingestion.py     # 로드→청킹→임베딩→저장
    chat.py          # 세션 이력 + RAG 엔진
scripts/
  ingest.py          # 인덱싱 하네스 (CLI)
  ask.py             # 질의 하네스 (CLI, HTTP 불필요)
tests/               # 단위 테스트 + opt-in 통합 테스트
docker/Dockerfile    # CUDA + PyTorch 베이스 이미지
docker-compose.yml   # GPU 할당, 볼륨 마운트
data/wiki/           # 샘플 위키 문서 (로컬 테스트용)
```

---

## 빠른 시작 (Docker, 권장)

전제: Ubuntu + NVIDIA 드라이버 + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

```bash
# 1) 환경설정
cp .env.example .env
#  .env 편집: ANTHROPIC_API_KEY, 그리고 WIKI_HOST_PATH=/users/wiki (실제 위키 경로)

# 2) 빌드 & 기동
docker compose up -d --build

# 3) 위키 문서 인덱싱 (최초 1회 + 문서 갱신 시)
docker compose exec fyi-agent python scripts/ingest.py --reset

# 4) 질의
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "휴가는 며칠인가요?", "session_id": "user-1"}'
```

API 문서(Swagger): http://localhost:8000/docs

---

## GPU(CUDA) 동작 확인

> 참고: **Claude(LLM)는 Anthropic 클라우드 API 라 로컬 GPU 를 쓰지 않습니다.**
> 로컬 GPU 를 사용하는 것은 **임베딩 모델**(`BAAI/bge-m3`)입니다.

먼저 호스트에서 NVIDIA Container Toolkit 이 동작하는지 빠르게 확인:

```bash
docker run --rm --gpus all pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime \
  python -c "import torch; print('cuda:', torch.cuda.is_available())"
```

서비스 컨테이너 안에서 전체 진단(드라이버 → PyTorch → GPU 연산 → 임베딩 모델 GPU 사용):

```bash
docker compose exec fyi-agent python scripts/check_cuda.py
# 빠른 검사(모델 다운로드 없이):     python scripts/check_cuda.py --skip-embedding
# CI/헬스체크용(GPU 없으면 exit 1):  python scripts/check_cuda.py --strict
# Claude API 연결까지 확인:          python scripts/check_cuda.py --check-llm
```

`make docker-gpu-check` / `make gpu-check` 로도 실행할 수 있습니다.

---

## 프록시 환경 (사내망)

배포 우분투가 프록시를 사용한다면 `.env` 의 프록시 값만 채우면 **빌드·런타임의 모든 외부 호출**에 자동 적용됩니다 (한 곳에서 관리).

```bash
# .env
HTTP_PROXY=http://proxy.example.com:8080
HTTPS_PROXY=http://proxy.example.com:8080
NO_PROXY=localhost,127.0.0.1
```

| 적용 위치 | 경로 | 비고 |
|---|---|---|
| 빌드 시 `pip install` | `docker-compose.yml` 의 `build.args` → Dockerfile `ARG` | 사전정의 ARG, 이미지에 영구 저장 안 됨 |
| 런타임 외부 호출 | `env_file`(대문자) + `environment`(소문자 미러) | Claude API·HuggingFace 다운로드. httpx/requests 가 자동 사용 |

> ⚠️ `NO_PROXY` 에 `localhost,127.0.0.1` 을 꼭 포함하세요. 컨테이너 헬스체크가 `localhost:8000` 으로 접속하므로, 빠지면 프록시를 타서 실패합니다.

### 베이스 이미지 풀(pull)은 호스트 데몬 설정

`pytorch/pytorch` 베이스 이미지 다운로드는 **Docker 데몬**이 수행하므로, 빌드 인자가 아니라 **호스트 데몬 프록시**가 필요합니다 (최초 1회).

`/etc/systemd/system/docker.service.d/http-proxy.conf`:

```ini
[Service]
Environment="HTTP_PROXY=http://proxy.example.com:8080"
Environment="HTTPS_PROXY=http://proxy.example.com:8080"
Environment="NO_PROXY=localhost,127.0.0.1"
```

```bash
sudo systemctl daemon-reload && sudo systemctl restart docker
```

---

## 로컬 개발 (하네스 프로그래밍)

HTTP 서버 없이도 파이프라인을 단계별로 돌려볼 수 있는 하네스를 제공합니다.

```bash
# 의존성 (로컬은 CPU 임베딩으로도 동작)
pip install -e ".[dev]"
cp .env.example .env          # WIKI_PATH 를 로컬 경로(예: ./data/wiki)로 바꿔도 됨

# 인덱싱 하네스
python scripts/ingest.py --reset

# 질의 하네스 (대화형)
python scripts/ask.py
python scripts/ask.py "신입사원 연차는 며칠인가요?"   # 단발성

# API 서버 (reload)
uvicorn app.main:app --reload

# 테스트
pytest -q                                   # 단위 테스트 (모델/네트워크 불필요)
RUN_INTEGRATION=1 pytest tests/test_rag_integration.py   # 엔드투엔드(모델+API 키 필요)
```

`make help` 로 단축 명령을 확인하세요.

---

## REST API

| 메서드 | 경로 | 설명 |
|---|---|---|
| GET | `/health` | 상태 확인 |
| POST | `/ingest` | 위키 문서 인덱싱 (`{"reset": true}`) |
| POST | `/chat` | 질의응답 (`{"message", "session_id"}`) |
| DELETE | `/chat/{session_id}` | 대화 세션 초기화 |

`/chat` 응답 예시:

```json
{
  "answer": "신입사원의 첫 해 연차 휴가는 11일입니다.",
  "session_id": "user-1",
  "sources": [
    {"title": "FYI-Agent 샘플 위키 문서", "source": "샘플-사내위키.md", "snippet": "..."}
  ]
}
```

---

## 설정

모든 값은 환경변수 / `.env` 로 제어합니다. 전체 목록은 [.env.example](.env.example) 참고.

| 변수 | 기본값 | 설명 |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Claude API 키 (필수) |
| `LLM_MODEL` | `claude-sonnet-4-6` | `claude-opus-4-8` 로 올리면 품질↑ |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | 한국어/다국어 임베딩 |
| `EMBEDDING_DEVICE` | `auto` | `auto`/`cuda`/`cpu` |
| `WIKI_PATH` | `/data/wiki` | 컨테이너 내부 문서 경로 |
| `WIKI_HOST_PATH` | `/users/wiki` | 호스트 실제 문서 경로(compose 마운트) |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `120` | 청킹 파라미터 |
| `RETRIEVAL_TOP_K` | `5` | 검색 문서 수 |

---

## 다음 단계 (확장 지점)

이 저장소는 동작하는 **스캐폴드**입니다. 자연스러운 확장 지점:

- **세션 이력 영속화**: 현재 인메모리 → Redis 등 ([app/services/chat.py](app/services/chat.py) 의 `_get_history`)
- **스트리밍 응답**: `/chat/stream` (SSE) 추가
- **증분 인덱싱**: 변경된 파일만 재색인 (현재는 `--reset` 전체 재색인)
- **재순위화(reranker)**: 검색 정확도 향상
- **인증/레이트리밋**: 외부 노출 시 API 키 미들웨어
