#!/usr/bin/env python
"""Is the causal-SDPA language IR numerically identical to the mask-carrying one?

The quantized-vs-golden cosine in 04_validate_python.py cannot answer this.  Re-running
NNCF (AWQ + scale estimation over 30 calibration texts) against a different source IR
produces materially different weights, and the resulting spread in ``text_cos_min`` --
measured at +0.0064 / -0.0120 between configurations that should be equivalent -- is several
times larger than the effect being tested.

So compare the two **FP32** IRs directly, on CPU, on identical inputs.  Both are exported by
the same transformers; the only difference is the explicit ``[B,1,S,S]`` mask versus
``causal=True``.  If dropping the mask is the identity this argument claims it is, the
outputs must agree to fp32 round-off.

The right-padded batch is the case that actually matters: the claim is that under right
padding every padding position sits at a strictly larger index than any token whose output is
read, so causal masking alone already excludes them.  A padded row that disagreed would
falsify exactly that.

    python check_causal_equivalence.py
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import openvino as ov
from openvino.properties import hint

from paths import GME_PACKAGE

CASES = [
    # (name, batch of valid token counts).  Ragged batches exercise right padding.
    ("seq1", [1]),
    ("seq32", [32]),
    ("seq512", [512]),
    ("seq1024", [1024]),
    ("batch2_padded", [1024, 300]),
    ("batch3_padded", [512, 129, 7]),
]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="CPU",
                    help="CPU keeps this a pure fp32 graph-equivalence question")
    ap.add_argument("--reference", default="gme_language_fp32_v5ctrl.xml")
    ap.add_argument("--candidate", default="gme_language_fp32_causal.xml")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    core = ov.Core()
    cfg = {hint.inference_precision: ov.Type.f32}
    ref = core.compile_model(GME_PACKAGE / args.reference, args.device, cfg)
    cand = core.compile_model(GME_PACKAGE / args.candidate, args.device, cfg)

    rng = np.random.default_rng(0)
    rows = []
    for name, lengths in CASES:
        width = max(lengths)
        # 151655 is the image-token id; keep it out so the masked_scatter path stays inert
        # and the image_embeds input can legitimately be empty.
        input_ids = rng.integers(1000, 150000, size=(len(lengths), width), dtype=np.int64)
        attention_mask = np.zeros((len(lengths), width), dtype=np.int64)
        for row, valid in enumerate(lengths):
            attention_mask[row, :valid] = 1
        feed = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image_embeds": np.empty((0, 1536), dtype=np.float32),
        }
        a = np.asarray(ref(feed)[ref.output(0)])
        b = np.asarray(cand(feed)[cand.output(0)])
        per_row = [cosine(a[i], b[i]) for i in range(a.shape[0])]
        rows.append({
            "case": name,
            "lengths": lengths,
            "cosine": cosine(a, b),
            "worst_row_cosine": min(per_row),
            "max_abs_diff": float(np.abs(a - b).max()),
        })
        r = rows[-1]
        print(f"{name:<16} lengths={str(lengths):<16} cosine={r['cosine']:.9f} "
              f"worst_row={r['worst_row_cosine']:.9f} max_abs={r['max_abs_diff']:.3e}")

    worst = min(rows, key=lambda r: r["worst_row_cosine"])
    print(f"\nworst row cosine: {worst['worst_row_cosine']:.9f} ({worst['case']})")
    print(f"largest max_abs_diff: {max(r['max_abs_diff'] for r in rows):.3e}")
    if args.output:
        payload = {"reference": args.reference, "candidate": args.candidate,
                   "device": args.device, "cases": rows}
        with open(args.output, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
