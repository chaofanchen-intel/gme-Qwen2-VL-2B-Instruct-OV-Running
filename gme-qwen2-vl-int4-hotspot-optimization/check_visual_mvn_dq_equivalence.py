#!/usr/bin/env python3
"""Check that visual MVN-to-DynamicQuantize fusion is bit-identical.

The control and candidate use the same INT4 visual IR and GPU plugin. The control
keeps the default dynamic-quantization threshold, so the graph matcher leaves MVN
standalone; the candidate sets the threshold to zero and folds eligible MVN+affine
nodes into per-token DynamicQuantize. Every test grid has more than 64 raw patch
rows, so DynamicQuantize executes in both graphs and the threshold itself does not
change their arithmetic.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import openvino as ov
from openvino.properties import hint

from paths import GME_PACKAGE

PATCH_ROW = 1176
CASES = [
    ("square_448", (1, 32, 32)),
    ("wide_cap768", (1, 18, 162)),
    ("square_672", (1, 48, 48)),
    ("logo_quality", (1, 24, 210)),
]


def cosine(lhs: np.ndarray, rhs: np.ndarray) -> float:
    lhs64 = lhs.astype(np.float64).reshape(-1)
    rhs64 = rhs.astype(np.float64).reshape(-1)
    return float(np.dot(lhs64, rhs64) / (np.linalg.norm(lhs64) * np.linalg.norm(rhs64)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ir", default="gme_visual_int4_nomask_headfirst_rank3.xml")
    parser.add_argument("--device", default="GPU")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    core = ov.Core()
    config = {hint.performance_mode: hint.PerformanceMode.LATENCY}
    control = core.compile_model(GME_PACKAGE / args.ir, args.device, config)
    fused = core.compile_model(
        GME_PACKAGE / args.ir,
        args.device,
        {**config, "GPU_DYNAMIC_QUANTIZATION_THRESHOLD": 0},
    )

    rows = []
    for name, grid in CASES:
        patches = int(np.prod(grid))
        rng = np.random.default_rng(patches)
        feed = {
            "pixel_values": rng.standard_normal((patches, PATCH_ROW), dtype=np.float32),
            "image_grid_thw": np.array([grid], dtype=np.int64),
        }
        reference = np.asarray(next(iter(control(feed).values())), dtype=np.float32)
        candidate = np.asarray(next(iter(fused(feed).values())), dtype=np.float32)
        if reference.shape != candidate.shape:
            raise SystemExit(f"{name}: shape mismatch {reference.shape} vs {candidate.shape}")

        row = {
            "case": name,
            "grid": list(grid),
            "patches": patches,
            "tokens": patches // 4,
            "shape": list(reference.shape),
            "cosine": cosine(reference, candidate),
            "max_abs_diff": float(np.max(np.abs(reference - candidate))),
            "bit_identical": bool(np.array_equal(reference, candidate)),
        }
        rows.append(row)
        print(
            f"{name:<14} grid={','.join(map(str, grid)):<10} "
            f"cos={row['cosine']:.9f} max_abs={row['max_abs_diff']:.3e} "
            f"bit_identical={row['bit_identical']}"
        )

    payload = {
        "device": args.device,
        "ir": args.ir,
        "control_dynamic_quantization_threshold": 64,
        "fused_dynamic_quantization_threshold": 0,
        "all_bit_identical": all(row["bit_identical"] for row in rows),
        "worst_max_abs_diff": max(row["max_abs_diff"] for row in rows),
        "cases": rows,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.output}")

    if not payload["all_bit_identical"]:
        raise SystemExit("visual MVN-to-DynamicQuantize fusion is not bit-identical")


if __name__ == "__main__":
    main()
