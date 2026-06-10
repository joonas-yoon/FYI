"""애플리케이션 설정. 환경변수 / .env 파일에서 로드합니다.

모든 모듈은 ``get_settings()`` 를 통해 단일 설정 인스턴스를 공유합니다.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- 앱 ---
    app_name: str = "fyi-agent"
    app_env: str = "development"
    log_level: str = "INFO"

    # --- API ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: list[str] = ["*"]

    # --- Anthropic / LLM ---
    anthropic_api_key: str = Field(default="", repr=False)
    llm_model: str = "claude-sonnet-4-6"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1024

    # --- 임베딩 ---
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "auto"  # auto | cuda | cpu
    embedding_batch_size: int = 32

    # --- 위키 문서 ---
    wiki_path: Path = Path("/data/wiki")
    wiki_glob: str = "**/*.md"

    # --- 벡터 스토어 (Chroma) ---
    vector_store_dir: Path = Path("/data/vectorstore")
    collection_name: str = "fyi_wiki"

    # --- 청킹 ---
    chunk_size: int = 800
    chunk_overlap: int = 120

    # --- 검색 ---
    retrieval_top_k: int = 5


@lru_cache
def get_settings() -> Settings:
    """프로세스 전역에서 공유되는 설정 싱글턴."""
    return Settings()
