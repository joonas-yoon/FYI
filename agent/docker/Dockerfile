# CUDA + PyTorch 가 미리 설치된 베이스 이미지 (GPU 임베딩용).
# 호스트에 NVIDIA Container Toolkit 이 설치되어 있어야 GPU 가 컨테이너로 전달됩니다.
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/data/hf-cache

WORKDIR /app

# 프록시 환경에서 pip 설치가 되도록 빌드 인자 수신.
# HTTP_PROXY/HTTPS_PROXY/NO_PROXY 는 Docker 사전정의 ARG → RUN 에 자동 적용되고
# 이미지/히스토리에 영구 저장되지 않음. 값은 docker-compose 의 build.args 로 전달.
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY

# 1) 의존성 먼저 설치 → 코드만 바뀌면 이 레이어는 캐시 재사용.
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# 2) 애플리케이션 코드.
COPY app ./app
COPY scripts ./scripts

EXPOSE 8000

# 컨테이너 헬스체크 (서비스 기동 확인용).
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
