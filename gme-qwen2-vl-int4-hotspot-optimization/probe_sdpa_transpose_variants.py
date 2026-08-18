#!/usr/bin/env python3
"""Isolate which side of the SDPA transpose fusion breaks, per SDPA rank and order.

Builds small synthetic attention blocks that differ only in which transposes surround the
SDPA, then scores GPU against CPU. Variant names say what the graph contains:

    in   -- Transpose on Q/K/V only          (stock TransposeSDPAMatcher)
    out  -- Transpose on the SDPA result only (opt-2 TransposeSDPATransposeMatcher)
    both -- both, which is the Qwen3-VL vision tower shape
    none -- neither, as a sanity floor

Rank 3 ([heads, seq, head_size], no batch axis) is the Qwen3-VL vision layout; rank 4
([batch, heads, seq, head_size]) is the usual LLM layout.
"""
from __future__ import annotations

import argparse
import itertools

import numpy as np
import openvino as ov
import openvino.opset13 as opset

HEADS, HEAD_SIZE = 16, 64


def build(rank: int, variant: str, seq: int, dynamic: bool = False):
    """Attention block whose Q/K/V parameters are always in *untransposed* layout."""
    if rank == 3:
        untransposed = [seq, HEADS, HEAD_SIZE]  # [seq, heads, head_size]
        order = [1, 0, 2]
    else:
        untransposed = [1, seq, HEADS, HEAD_SIZE]  # [batch, seq, heads, head_size]
        order = [0, 2, 1, 3]

    fuse_in = variant in ("in", "both")
    fuse_out = variant in ("out", "both")

    # Without an input transpose in the graph the parameter must already be [.., heads, seq, ..].
    param_shape = untransposed if fuse_in else [untransposed[i] for i in order]
    # The real vision tower is compiled with a dynamic token count, which is what forces the
    # SDPA output layout to be re-inferred at run time rather than baked in at compile time.
    decl_shape = [-1 if d == seq else d for d in param_shape] if dynamic else param_shape
    params = [opset.parameter(decl_shape, ov.Type.f32, name=n) for n in ("q", "k", "v")]

    qkv = []
    for p in params:
        # A Multiply keeps the transpose from folding into the Parameter itself.
        scaled = opset.multiply(p, opset.constant(np.float32(1.0)))
        qkv.append(opset.transpose(scaled, opset.constant(order, ov.Type.i64)) if fuse_in else scaled)

    mask_shape = [1, seq, seq] if rank == 3 else [1, 1, seq, seq]
    mask = opset.parameter([-1 if d == seq else d for d in mask_shape] if dynamic else mask_shape,
                           ov.Type.f32, name="mask")
    out = opset.scaled_dot_product_attention(*qkv, mask, causal=False)

    if fuse_out:
        out = opset.transpose(out, opset.constant(order, ov.Type.i64))
    # Reshape to rows x (heads*head_size), the o_proj input in a real block.
    out = opset.reshape(out, opset.constant([-1, HEADS * HEAD_SIZE], ov.Type.i64), special_zero=False)
    out = opset.multiply(out, opset.constant(np.float32(1.0)))

    return ov.Model([out], params + [mask], f"rank{rank}_{variant}"), param_shape, mask_shape


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=208)
    ap.add_argument("--ranks", nargs="*", type=int, default=[3, 4])
    ap.add_argument("--variants", nargs="*", default=["none", "in", "out", "both"])
    ap.add_argument("--dynamic", action="store_true", help="declare the token axis as dynamic")
    args = ap.parse_args()

    core = ov.Core()
    print(f"openvino {ov.get_version()}  seq={args.seq}  dynamic={args.dynamic}")
    worst = 1.0
    for rank, variant in itertools.product(args.ranks, args.variants):
        model, pshape, mshape = build(rank, variant, args.seq, args.dynamic)
        rng = np.random.default_rng(rank * 100 + len(variant))
        feed = {n: rng.standard_normal(pshape, dtype=np.float32) for n in ("q", "k", "v")}
        feed["mask"] = np.zeros(mshape, dtype=np.float32)

        ref = core.compile_model(model, "CPU").create_infer_request().infer(feed)[0]
        try:
            got = core.compile_model(model, "GPU").create_infer_request().infer(feed)[0]
        except Exception as exc:  # noqa: BLE001
            print(f"  rank{rank:<2} {variant:<5} GPU FAILED: {str(exc).splitlines()[0]}")
            worst = 0.0
            continue

        if ref.shape != got.shape:
            print(f"  rank{rank:<2} {variant:<5} SHAPE MISMATCH cpu={ref.shape} gpu={got.shape}")
            worst = 0.0
            continue
        a, b = ref.ravel().astype(np.float64), got.ravel().astype(np.float64)
        cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
        worst = min(worst, cos)
        print(f"  rank{rank:<2} {variant:<5} shape={tuple(ref.shape)} cos={cos:.6f} "
              f"max_abs={np.abs(a - b).max():.4g}  {'ok' if cos > 0.9999 else 'BAD'}")

    print(f"worst cos = {worst:.6f}")
    return 0 if worst > 0.9999 else 1


if __name__ == "__main__":
    raise SystemExit(main())
