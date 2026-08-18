#!/usr/bin/env python3
"""Isolated probe for the Qwen3-VL-Embedding-2B vision merger on GPU.

The merger is the sub-model that fails with CL_OUT_OF_RESOURCES when the three GME
plugin patches are installed. Running it standalone removes the tokenizer / language
model / GenAI pipeline from the picture, so a failure here is attributable to the
vision attention block alone.

Usage:
    python probe_qwen3_vision_merger.py [--precision INT8|INT4] [--device GPU]
                                        [--lens 256 1024 4096 ...]
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback

import numpy as np
import openvino as ov

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from paths import MODELS_ROOT  # noqa: E402

# Qwen3-VL vision tower: 16 heads x 64 head_dim = 1024 hidden.
# `hidden_states` rows are patches; a WxH image at patch 16 / merge 2 gives
# (W/16)*(H/16) rows before the merger, which is what we sweep here.
IMAGE_TO_PATCHES = {
    "64x64": 16,       # (64/16)^2
    "224x224": 196,    # (224/16)^2
    "448x448": 784,
    "896x896": 3136,
}


def build_inputs(n_patches: int, rng: np.random.Generator):
    hidden = rng.standard_normal((n_patches, 1024), dtype=np.float32) * 0.02
    # Full (non-causal) window: the merger's attention_mask is [1, q, k] additive.
    mask = np.zeros((1, n_patches, n_patches), dtype=np.float32)
    rotary = rng.standard_normal((n_patches, 32), dtype=np.float32) * 0.02
    return {"hidden_states": hidden, "attention_mask": mask, "rotary_pos_emb": rotary}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", default="INT8", choices=["INT8", "INT4", "FP32"])
    ap.add_argument("--device", default="GPU")
    ap.add_argument("--cases", nargs="*", default=list(IMAGE_TO_PATCHES))
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    model_dir = MODELS_ROOT / f"Qwen3-VL-Embedding-2B-{args.precision}"
    xml = model_dir / "openvino_vision_embeddings_merger_model.xml"
    if not xml.exists():
        print(f"missing model: {xml}", file=sys.stderr)
        return 2

    core = ov.Core()
    print(f"openvino {ov.get_version()}")
    print(f"device   {args.device}  {core.get_property(args.device, 'FULL_DEVICE_NAME')}")
    print(f"model    {xml}")

    compiled = core.compile_model(str(xml), args.device)
    req = compiled.create_infer_request()
    rng = np.random.default_rng(0)

    results = []
    for case in args.cases:
        n = IMAGE_TO_PATCHES[case]
        rec = {"case": case, "patches": n}
        try:
            req.infer(build_inputs(n, rng))
            out = req.get_output_tensor(0).data
            rec.update(status="OK", shape=list(out.shape),
                       finite=bool(np.isfinite(out).all()),
                       absmax=float(np.abs(out).max()))
            print(f"  {case:>9} patches={n:<5} OK   shape={rec['shape']} "
                  f"finite={rec['finite']} absmax={rec['absmax']:.4g}")
        except Exception as exc:  # noqa: BLE001 - we want the plugin error text
            rec.update(status="FAIL", error=str(exc).strip().splitlines()[0:6])
            print(f"  {case:>9} patches={n:<5} FAIL")
            traceback.print_exc(limit=1)
            results.append(rec)
            break
        results.append(rec)

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"precision": args.precision, "device": args.device,
                       "version": ov.get_version(), "results": results}, fh, indent=2)
    return 0 if all(r["status"] == "OK" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
