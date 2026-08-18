#!/usr/bin/env python3
"""Compare the Qwen3-VL-Embedding-2B vision merger on GPU against the CPU reference.

Same inputs are fed to both devices so a GPU-plugin change can be scored on accuracy,
not just on "did it crash". Deterministic per case: the RNG is re-seeded from the patch
count, so a case's inputs do not depend on which other cases were requested.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import openvino as ov

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from paths import MODELS_ROOT  # noqa: E402

IMAGE_TO_PATCHES = {"64x64": 16, "224x224": 196, "448x448": 784, "896x896": 3136}


def build_inputs(n_patches: int):
    rng = np.random.default_rng(n_patches)
    return {
        "hidden_states": rng.standard_normal((n_patches, 1024), dtype=np.float32) * 0.02,
        "attention_mask": np.zeros((1, n_patches, n_patches), dtype=np.float32),
        "rotary_pos_emb": rng.standard_normal((n_patches, 32), dtype=np.float32) * 0.02,
    }


def run(core, xml, device, cases):
    req = core.compile_model(str(xml), device).create_infer_request()
    out = {}
    for case in cases:
        req.infer(build_inputs(IMAGE_TO_PATCHES[case]))
        out[case] = req.get_output_tensor(0).data.astype(np.float64).copy()
    return out


def cosine(a, b):
    a, b = a.ravel(), b.ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", default="INT8")
    ap.add_argument("--cases", nargs="*", default=list(IMAGE_TO_PATCHES))
    ap.add_argument("--json", default=None)
    ap.add_argument("--save-gpu", default=None, help="npz to write the raw GPU outputs to")
    ap.add_argument("--compare-gpu", default=None, help="npz from a previous run to diff against")
    args = ap.parse_args()

    xml = MODELS_ROOT / f"Qwen3-VL-Embedding-2B-{args.precision}" / "openvino_vision_embeddings_merger_model.xml"
    core = ov.Core()
    print(f"openvino {ov.get_version()}  model {xml.parent.name}")

    ref = run(core, xml, "CPU", args.cases)
    got = run(core, xml, "GPU", args.cases)

    if args.save_gpu:
        np.savez(args.save_gpu, **got)
        print(f"  saved GPU outputs to {args.save_gpu}")
    if args.compare_gpu:
        base = np.load(args.compare_gpu)
        for case in args.cases:
            d = float(np.abs(base[case] - got[case]).max())
            print(f"  {case:>9}  vs saved GPU baseline: max_abs_diff={d:.6g} "
                  f"{'IDENTICAL' if d == 0.0 else 'DIFFERS'}")

    # An INT8/INT4 vision tower on GPU (f16 compute) legitimately drifts from the f32 CPU
    # reference; the bar here is "no worse than the unpatched plugin", so this only screens for
    # gross corruption. Use --save-gpu / --compare-gpu for an exact stock-vs-patched diff.
    records, ok = [], True
    for case in args.cases:
        r, g = ref[case], got[case]
        cos = cosine(r, g)
        max_abs = float(np.abs(r - g).max())
        rel = max_abs / max(float(np.abs(r).max()), 1e-9)
        passed = cos > 0.95
        ok &= passed
        records.append({"case": case, "cos": cos, "max_abs": max_abs, "rel": rel,
                        "cpu_absmax": float(np.abs(r).max()), "gpu_absmax": float(np.abs(g).max()),
                        "pass": passed})
        print(f"  {case:>9}  cos={cos:.6f}  max_abs={max_abs:.4g}  rel={rel:.4g}  "
              f"cpu_absmax={np.abs(r).max():.4g} gpu_absmax={np.abs(g).max():.4g}  "
              f"{'PASS' if passed else 'FAIL'}")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"precision": args.precision, "version": ov.get_version(), "cases": records}, fh, indent=2)
    print("ALL PASS" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
