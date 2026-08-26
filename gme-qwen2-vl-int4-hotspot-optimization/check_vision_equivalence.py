#!/usr/bin/env python
"""Is the mask-free visual IR numerically identical to the mask-carrying one?

Same argument, and the same reason for making it this way, as
``check_causal_equivalence.py``: re-running NNCF against a different source IR produces
materially different weights, so a quantized-vs-golden cosine cannot isolate a graph
change of this size.  Compare the two **FP32** IRs directly, on CPU, on identical inputs.

What is being tested.  Qwen2-VL's vision tower packs every image's patch rows into one
sequence and restricts attention to each image's own block via ``cu_seqlens``.  The shipped
IR spells that out as a rank-3 BOOL ``[1, L, L]`` mask; ``gme_visual_fp32_nomask.xml`` drops
it and runs dense attention over the whole sequence.  Those two agree **iff there is exactly
one block**, i.e. one image per call.

That premise is not an assumption this script makes -- it is a property both IRs already
have.  ``rot_pos_emb`` iterates over ``grid_thw`` rows in Python, so the trace bakes the
image count in (``02_export.py`` emits a ``TracerWarning`` saying so), and
``gme_openvino.py:_encode_image_batch`` splits a multi-image batch into one visual request
per image for exactly that reason.  So this checks the contract that is actually in force.

Grids are varied deliberately: the sequence dimension stays dynamic in both IRs, so a shape
this trace never saw is the case most likely to expose a baked-in constant.

    python check_vision_equivalence.py
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import openvino as ov
from openvino.properties import hint

from paths import GME_PACKAGE

PATCH_ROW = 1176  # 3 channels * 2 temporal * 14 * 14

# (name, (t, h, w)).  h, w are multiples of the spatial merge size 2.  448x448 is the
# package's floor (min_image_tokens=256); 52x92 is the non-square 720p-scale case; 24x210
# is the extreme aspect ratio of the repo's own gme_logo.png.
CASES = [
    ("grid_1x2x2_minimal", (1, 2, 2)),
    ("grid_1x32x32_448px", (1, 32, 32)),
    ("grid_1x34x46_nonsquare", (1, 34, 46)),
    ("grid_1x48x48_672px", (1, 48, 48)),
    ("grid_1x24x210_logo_aspect", (1, 24, 210)),
]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="CPU",
                    help="CPU keeps this a pure fp32 graph-equivalence question")
    ap.add_argument("--reference", default="gme_visual_fp32.xml")
    ap.add_argument("--candidate", default="gme_visual_fp32_nomask.xml")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    core = ov.Core()
    cfg = {hint.inference_precision: ov.Type.f32}
    ref = core.compile_model(GME_PACKAGE / args.reference, args.device, cfg)
    cand = core.compile_model(GME_PACKAGE / args.candidate, args.device, cfg)

    rng = np.random.default_rng(0)
    rows = []
    for name, (t, h, w) in CASES:
        patches = t * h * w
        # Real pixel_values are normalized patches; standard normal is the same order of
        # magnitude and avoids the denormals a zero-filled tensor would produce.
        pixel_values = rng.standard_normal((patches, PATCH_ROW)).astype(np.float32)
        feed = {
            "pixel_values": pixel_values,
            "image_grid_thw": np.array([[t, h, w]], dtype=np.int64),
        }
        a = np.asarray(ref(feed)[ref.output(0)])
        b = np.asarray(cand(feed)[cand.output(0)])
        if a.shape != b.shape:
            raise SystemExit(f"{name}: shape mismatch {a.shape} vs {b.shape}")
        max_abs = float(np.max(np.abs(a - b)))
        # Per-row cosine as well: the language IR consumes these as K separate token
        # embeddings, so a single flattened cosine could hide one bad row.
        row_cos = [cosine(a[i], b[i]) for i in range(a.shape[0])]
        rows.append({
            "case": name,
            "grid": [t, h, w],
            "patches": patches,
            "tokens": patches // 4,
            "shape": list(a.shape),
            "cosine": cosine(a, b),
            "row_cosine_min": float(np.min(row_cos)),
            "max_abs_diff": max_abs,
            "bit_identical": bool(max_abs == 0.0),
        })
        r = rows[-1]
        print(f"{name:<26} grid={t},{h},{w:<4} K={r['tokens']:<5} "
              f"cos={r['cosine']:.9f} row_cos_min={r['row_cosine_min']:.9f} "
              f"max_abs={max_abs:.3e} bit_identical={r['bit_identical']}")

    worst = max(row["max_abs_diff"] for row in rows)
    identical = all(row["bit_identical"] for row in rows)
    print(f"\nworst max_abs_diff={worst:.3e}  all_bit_identical={identical}")

    if args.output:
        payload = {
            "device": args.device,
            "reference": args.reference,
            "candidate": args.candidate,
            "worst_max_abs_diff": worst,
            "all_bit_identical": identical,
            "cases": rows,
        }
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        print(f"wrote {args.output}")

    if worst > 1e-4:
        raise SystemExit("mask-free visual IR is NOT equivalent to the mask-carrying one")


if __name__ == "__main__":
    main()
