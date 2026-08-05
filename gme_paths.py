"""Repository-relative paths and handoff-friendly external overrides.

The GME project can live next to the benchmark workspace as in the original
layout, or inside another project. External dependencies are discovered from
that layout by default and can be overridden with environment variables.
"""
from __future__ import annotations

import os
from pathlib import Path

GME_ROOT = Path(__file__).resolve().parent


def _path_env(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default.resolve()


BENCHMARK_ROOT = _path_env("GME_BENCHMARK_ROOT", GME_ROOT.parent / "benchmark")
OPENVINO_INSTALL = _path_env("GME_OPENVINO_INSTALL", BENCHMARK_ROOT / "openvino" / "install")
SOURCE_MODEL = _path_env(
    "GME_SOURCE_MODEL",
    BENCHMARK_ROOT / "models" / "gme-Qwen2-VL-2B-Instruct",
)
PACKAGE = _path_env(
    "GME_PACKAGE",
    GME_ROOT / "ov_models" / "gme-qwen2-vl-2b-instruct-openvino",
)
ARTIFACTS = _path_env("GME_ARTIFACTS", GME_ROOT / "artifacts")
CPP_BINARY = _path_env("GME_CPP_BINARY", GME_ROOT / "cpp" / "build" / "gme_embed")
OPENVINO_SETUP = OPENVINO_INSTALL / "setupvars.sh"
LOGO_IMAGE = SOURCE_MODEL / "images" / "gme_logo.png"

LLAMA_ROOT = _path_env("GME_LLAMA_CPP_ROOT", GME_ROOT.parent / "llama.cpp")
LLAMA_EMBEDDING = _path_env(
    "GME_LLAMA_EMBEDDING",
    LLAMA_ROOT / "build-openvino" / "bin" / "llama-embedding",
)
GGUF_MODEL = _path_env(
    "GME_GGUF_MODEL",
    BENCHMARK_ROOT / "models" / "gme-Qwen2-VL-2B-Instruct-GGUF"
    / "gme-Qwen2-VL-2B-Instruct-Q4_K_M.gguf",
)


def require_file(path: Path, description: str) -> Path:
    """Raise a useful handoff error instead of failing later in a subprocess."""
    if not path.is_file():
        raise FileNotFoundError(
            f"{description} not found: {path}\n"
            f"Set the corresponding GME_* environment variable if your layout differs."
        )
    return path


def require_dir(path: Path, description: str) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(
            f"{description} not found: {path}\n"
            f"Set the corresponding GME_* environment variable if your layout differs."
        )
    return path
