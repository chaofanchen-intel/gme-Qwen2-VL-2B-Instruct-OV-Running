#!/usr/bin/env python
"""Full accuracy sweep for the packaged GME OpenVINO runtime.

For every precision in the package (fp32/int8/int4/...) x device (CPU/GPU) x mode
(text/query/image/fused) it reports cosine vs the PyTorch fp32 golden. FP32 must
reproduce the reference (>0.9999); quantized rows are reported so the accuracy
loss (esp. INT4 vision) is visible and can be judged.
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

from gme_openvino import GmeOpenVINO
from gme_paths import ARTIFACTS as ART, PACKAGE, SOURCE_MODEL as MODEL_DIR

HERE = Path(__file__).resolve().parent


def cos_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


def ensure_fused_golden(meta: dict) -> np.ndarray:
    """Torch fp32 fused reference for texts[0]+logo, cached to golden_fused.npy."""
    path = ART / "golden_fused.npy"
    if path.exists():
        return np.load(path)
    sys.path.insert(0, str(MODEL_DIR))
    GmeQwen2VL = importlib.import_module("gme_inference").GmeQwen2VL
    gme = GmeQwen2VL(model_path=str(MODEL_DIR), device="cpu")
    gme.base = gme.base.float().eval()
    with torch.no_grad():
        fused = gme.get_fused_embeddings([meta["texts"][0]], [meta["image"]]).float().numpy()
    np.save(path, fused)
    return fused


def main() -> None:
    meta = json.loads((ART / "golden_meta.json").read_text())
    golden = {
        "text": np.load(ART / "golden_text.npy"),
        "query": np.load(ART / "golden_query.npy"),
        "image": np.load(ART / "golden_image.npy"),
        "fused": ensure_fused_golden(meta),
    }

    cfg = json.loads((PACKAGE / "gme_openvino_config.json").read_text())
    precisions = list(cfg["precisions"])
    available = GmeOpenVINO(PACKAGE, precision="fp32").core.available_devices
    devices = [d for d in ("CPU", "GPU") if d in available]
    print(f"[validate] devices={devices} precisions={precisions}\n")

    report: dict = {}
    for device in devices:
        report[device] = {}
        for prec in precisions:
            enc = GmeOpenVINO(PACKAGE, device=device, precision=prec)
            text = enc.get_text_embeddings(meta["texts"])                       # 5 rows, default instr
            query = enc.get_text_embeddings([meta["texts"][0]], instruction=meta["instruction"])
            image = enc.get_image_embeddings(meta["image"])                     # 1 row
            fused = enc.get_fused_embeddings(meta["texts"][0], meta["image"])   # 1 row
            row = {
                "text_cos_min": round(float(cos_rows(text, golden["text"]).min()), 6),
                "query_cos": round(float(cos_rows(query, golden["query"][:1])[0]), 6),
                "image_cos": round(float(cos_rows(image, golden["image"])[0]), 6),
                "fused_cos": round(float(cos_rows(fused, golden["fused"])[0]), 6),
                "fused_norm": round(float(np.linalg.norm(fused[0])), 5),
            }
            report[device][prec] = row
            print(f"[{device}] {prec:11s} text={row['text_cos_min']:.6f} query={row['query_cos']:.6f} "
                  f"image={row['image_cos']:.6f} fused={row['fused_cos']:.6f}")
            if prec == "fp32" and min(row["text_cos_min"], row["query_cos"], row["image_cos"], row["fused_cos"]) < 0.9999:
                raise RuntimeError(f"FP32 on {device} failed exact-match acceptance")
        print()

    (ART / "openvino_full_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"[validate] wrote {ART / 'openvino_full_validation.json'}")


if __name__ == "__main__":
    main()
