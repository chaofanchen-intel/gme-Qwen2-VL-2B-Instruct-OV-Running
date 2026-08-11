#!/usr/bin/env python3
"""Dump GME language-IR outputs for a fixed set of inputs, for A/B numerical comparison.

Run once against the stock OpenVINO 2026.3.0 wheel and once against the patched local
build, then diff the two .npz files with compare_outputs.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import openvino as ov
from openvino.properties import hint

from paths import GME_PACKAGE

PACKAGE = GME_PACKAGE
IR = {
    "int4": "gme_language_int4.xml",
    "int4_gs64": "gme_language_int4_gs64.xml",
    "int8": "gme_language_int8.xml",
}
LENGTHS = (1, 7, 32, 129, 512, 1024)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", default="int4", choices=sorted(IR))
    ap.add_argument("--device", default="GPU")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    core = ov.Core()
    compiled = core.compile_model(
        PACKAGE / IR[args.precision], args.device,
        {hint.performance_mode: hint.PerformanceMode.LATENCY},
    )
    request = compiled.create_infer_request()

    rng = np.random.default_rng(20260809)
    out: dict[str, np.ndarray] = {}
    for seq in LENGTHS:
        # Deterministic pseudo-random token ids well inside the vocabulary.
        ids = rng.integers(1000, 150000, size=(1, seq), dtype=np.int64)
        result = request.infer({
            "input_ids": ids,
            "attention_mask": np.ones((1, seq), dtype=np.int64),
            "image_embeds": np.empty((0, 1536), dtype=np.float32),
        })
        out[f"seq{seq}"] = np.asarray(next(iter(result.values())), dtype=np.float32)

    # A two-row batch exercises the padded batch path the GME encoder actually uses.
    ids = rng.integers(1000, 150000, size=(2, 512), dtype=np.int64)
    mask = np.ones((2, 512), dtype=np.int64)
    mask[1, 300:] = 0
    result = request.infer({
        "input_ids": ids, "attention_mask": mask,
        "image_embeds": np.empty((0, 1536), dtype=np.float32),
    })
    out["batch2_padded"] = np.asarray(next(iter(result.values())), dtype=np.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **out)
    print(f"{ov.__version__}  ->  {args.output}")
    for k, v in out.items():
        print(f"  {k:16s} shape={v.shape} norm={np.linalg.norm(v):.6f}")


if __name__ == "__main__":
    main()
