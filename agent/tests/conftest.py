"""테스트 공통 설정.

모델/네트워크에 의존하지 않는 단위 테스트를 위해, import 이전에 환경변수를 설정한다.
무거운(임베딩 다운로드 + Anthropic 호출) 통합 테스트는 RUN_INTEGRATION=1 일 때만 동작.
"""

from __future__ import annotations

import os

# 설정 싱글턴이 만들어지기 전에 테스트용 기본값을 주입.
os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("EMBEDDING_DEVICE", "cpu")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
