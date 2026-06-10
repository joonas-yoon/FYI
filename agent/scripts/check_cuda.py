"""CUDA / GPU 진단 하네스.

Docker 컨테이너 안에서 GPU(CUDA)가 제대로 전달되었는지, 그리고
LangChain 임베딩 모델이 실제로 GPU 위에서 동작하는지 검증한다.

  ┌────────────────────────────────────────────────────────────────────┐
  │ 중요: 이 서비스의 'LLM'(Claude)은 Anthropic API(클라우드)에서        │
  │ 실행되므로 로컬 GPU 를 사용하지 않습니다.                            │
  │ 로컬 GPU(CUDA)를 사용하는 것은 '임베딩 모델'(BAAI/bge-m3)입니다.     │
  │ 따라서 이 스크립트는 임베딩 모델이 CUDA 를 쓰는지를 검증합니다.      │
  └────────────────────────────────────────────────────────────────────┘

사용:
    docker compose exec fyi-agent python scripts/check_cuda.py
    python scripts/check_cuda.py                  # 로컬에서도 가능
    python scripts/check_cuda.py --skip-embedding # torch 만 빠르게 검사 (모델 다운로드 X)
    python scripts/check_cuda.py --strict         # GPU 미사용 시 exit 1 (CI/헬스체크용)
    python scripts/check_cuda.py --check-llm      # Claude API 연결도 확인 (GPU 미사용)
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

# editable 설치/PYTHONPATH 없이도 단독 실행되도록 프로젝트 루트를 경로에 추가.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _hr(title: str) -> None:
    print(f"\n{'=' * 64}\n{title}\n{'=' * 64}")


def _human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def check_system() -> None:
    _hr("1. 시스템 / 드라이버 (nvidia-smi)")
    print(f"Python : {platform.python_version()}  ({platform.platform()})")
    for var in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "HF_HOME"):
        print(f"  env {var:22}: {os.environ.get(var, '(unset)')}")
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,memory.used",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if out.returncode == 0:
            for line in out.stdout.strip().splitlines():
                print(f"  GPU: {line}")
        else:
            print("  nvidia-smi 실행 실패 — NVIDIA 드라이버 / Container Toolkit 확인 필요")
            if out.stderr.strip():
                print(f"  stderr: {out.stderr.strip()}")
    except FileNotFoundError:
        print("  nvidia-smi 없음 — GPU 가 컨테이너로 전달되지 않았을 수 있음")
        print("  (docker compose 의 deploy.resources 또는 'docker run --gpus all' 확인)")
    except Exception as exc:
        print(f"  nvidia-smi 오류: {exc}")


def check_torch():
    _hr("2. PyTorch / CUDA 빌드")
    try:
        import torch
    except ImportError:
        print("  ❌ torch 가 설치되어 있지 않습니다.")
        return False, None
    print(f"  torch             : {torch.__version__}")
    print(f"  compiled CUDA     : {torch.version.cuda}")
    print(f"  cuDNN             : {torch.backends.cudnn.version()}")
    available = torch.cuda.is_available()
    print(f"  cuda.is_available : {available}")
    return available, torch


def check_devices(torch) -> None:
    _hr("3. GPU 디바이스 정보")
    count = torch.cuda.device_count()
    print(f"  device_count: {count}")
    for i in range(count):
        name = torch.cuda.get_device_name(i)
        major, minor = torch.cuda.get_device_capability(i)
        free, total = torch.cuda.mem_get_info(i)
        print(
            f"  [{i}] {name}  compute={major}.{minor}  "
            f"mem={_human_bytes(free)} free / {_human_bytes(total)} total"
        )


def check_compute(torch) -> bool:
    _hr("4. GPU 연산 스모크 테스트 (matmul)")
    try:
        n = 2048
        x = torch.randn(n, n, device="cuda")
        y = torch.randn(n, n, device="cuda")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        z = x @ y
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        correct = torch.allclose(z.cpu(), x.cpu() @ y.cpu(), atol=1e-2, rtol=1e-2)
        print(f"  {n}x{n} matmul on GPU : {dt * 1000:.1f} ms")
        print(f"  결과 정확성 (vs CPU)  : {'OK' if correct else 'FAIL'}")
        print(f"  torch allocated       : {_human_bytes(torch.cuda.memory_allocated())}")
        del x, y, z
        torch.cuda.empty_cache()
        return bool(correct)
    except Exception as exc:
        print(f"  ❌ GPU 연산 실패: {exc}")
        return False


def _embedding_device(emb) -> str:
    """HuggingFaceEmbeddings 내부 SentenceTransformer 가 올라간 디바이스 문자열."""
    client = getattr(emb, "_client", None) or getattr(emb, "client", None)
    if client is None:
        return "unknown"
    device = getattr(client, "device", None)
    if device is not None:
        return str(device)
    try:
        return str(next(client.parameters()).device)
    except Exception:
        return "unknown"


def check_embedding(torch) -> bool:
    _hr("5. LangChain 임베딩 모델 GPU 검증 (BAAI/bge-m3)")
    print("  참고: Claude(LLM)는 클라우드 API 라 GPU 를 쓰지 않습니다.")
    print("        실제로 GPU 를 사용하는 컴포넌트는 아래 임베딩 모델입니다.\n")
    cuda_on = torch is not None and torch.cuda.is_available()
    try:
        before = torch.cuda.memory_allocated() if cuda_on else 0

        from app.core.embeddings import get_embeddings

        t0 = time.perf_counter()
        emb = get_embeddings()
        load_dt = time.perf_counter() - t0
        device = _embedding_device(emb)
        print(f"  모델 로드     : {load_dt:.1f}s,  device={device}")

        t0 = time.perf_counter()
        vector = emb.embed_query("신입사원의 연차 휴가는 며칠인가요?")
        embed_dt = time.perf_counter() - t0
        print(f"  임베딩 1건    : {embed_dt * 1000:.0f} ms,  벡터 차원={len(vector)}")

        if cuda_on:
            after = torch.cuda.memory_allocated()
            print(f"  GPU 메모리(모델): +{_human_bytes(after - before)} (총 {_human_bytes(after)})")

        on_gpu = "cuda" in device
        if on_gpu:
            print("\n  ✅ 임베딩 모델이 CUDA(GPU) 위에서 동작 중입니다.")
        else:
            print(f"\n  ⚠️  임베딩 모델이 GPU 가 아닌 '{device}' 에서 동작 중입니다.")
        return on_gpu
    except Exception as exc:
        print(f"  ❌ 임베딩 검증 실패: {exc}")
        return False


def check_llm() -> bool:
    _hr("6. Claude LLM 연결 확인 (참고: GPU 미사용 — 클라우드 API)")
    try:
        from app.core.llm import get_llm

        llm = get_llm()
        response = llm.invoke("한 단어로만 답하세요: 안녕하세요?")
        text = getattr(response, "content", str(response))
        print(f"  응답: {str(text)[:80]}")
        print("  ✅ Anthropic API 연결 OK (이 호출은 GPU 가 아니라 클라우드에서 처리됨)")
        return True
    except Exception as exc:
        print(f"  ❌ Claude 호출 실패: {exc}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="CUDA/GPU 및 임베딩 모델 GPU 사용 진단")
    parser.add_argument(
        "--strict", action="store_true", help="GPU 미사용 시 비정상 종료(exit 1)"
    )
    parser.add_argument(
        "--skip-embedding",
        action="store_true",
        help="임베딩 모델 로드 테스트 생략 (모델 다운로드 없이 빠른 검사)",
    )
    parser.add_argument(
        "--check-llm", action="store_true", help="Claude API 연결도 확인 (GPU 미사용)"
    )
    args = parser.parse_args()

    # Windows 콘솔(cp949 등)에서 이모지/한글 출력이 깨지거나 죽지 않도록 UTF-8 로 강제.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    results: dict[str, bool] = {}

    check_system()
    cuda_ok, torch_mod = check_torch()
    results["PyTorch CUDA 사용 가능"] = cuda_ok

    if cuda_ok and torch_mod is not None:
        check_devices(torch_mod)
        results["GPU 연산(matmul)"] = check_compute(torch_mod)
    else:
        print("\n  ⚠️  CUDA 를 사용할 수 없어 GPU 연산 테스트를 건너뜁니다.")

    if not args.skip_embedding:
        results["임베딩 모델 GPU 사용"] = check_embedding(torch_mod)

    if args.check_llm:
        results["Claude API 연결"] = check_llm()

    _hr("결과 요약")
    for name, ok in results.items():
        print(f"  {'✅' if ok else '❌'}  {name}")
    all_ok = bool(results) and all(results.values())

    if args.strict and not all_ok:
        print("\nSTRICT 모드: 일부 검사 실패 → exit 1")
        return 1
    print("\n진단 완료.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
