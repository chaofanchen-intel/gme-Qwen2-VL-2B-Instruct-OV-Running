"""Shared paths for the GME hotspot experiments.

The defaults follow the repository layout after this directory was moved under
GME-Qwen2-VL-2B-Instruct-Running. Environment variables override them for a
checkout in a different workspace.
"""
from __future__ import annotations

import os
from pathlib import Path

HOTSPOT_ROOT = Path(__file__).resolve().parent
GME_ROOT = Path(os.environ.get("GME_ROOT", HOTSPOT_ROOT.parent))
MODEL_BENCH_ROOT = Path(os.environ.get("MODEL_BENCH_ROOT", GME_ROOT.parent))
OPENVINO_ROOT = Path(os.environ.get("OPENVINO_ROOT", MODEL_BENCH_ROOT / "benchmark/openvino"))
MODELS_ROOT = Path(os.environ.get("MODELS_ROOT", MODEL_BENCH_ROOT / "benchmark/models"))
GME_PACKAGE = GME_ROOT / "ov_models" / "gme-qwen2-vl-2b-instruct-openvino"
