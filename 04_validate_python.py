#!/usr/bin/env python
"""Full accuracy sweep for the packaged GME OpenVINO runtime.

For every precision in the package (fp32/int8/int4/...) x device (CPU/GPU) x mode
(text/query/image/fused) it reports cosine vs the PyTorch fp32 golden. FP32 must
reproduce the reference (>0.9999); quantized rows are reported so the accuracy
loss (esp. INT4 vision) is visible and can be judged.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

from gme_openvino import GmeOpenVINO

HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE_MODEL = HERE.parent / "benchmark" / "models" / "gme-Qwen2-VL-2B-Instruct"
DEFAULT_PACKAGE = HERE / "ov_models" / "gme-qwen2-vl-2b-instruct-openvino"
DEFAULT_ARTIFACTS = HERE / "artifacts"


def cos_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


def ensure_fused_golden(meta: dict, artifacts: Path, model_dir: Path) -> np.ndarray:
    """Torch fp32 fused reference for texts[0]+logo, cached to golden_fused.npy."""
    path = artifacts / "golden_fused.npy"
    if path.exists():
        return np.load(path)
    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"source model directory not found while creating {path.name}: {model_dir}"
        )
    sys.path.insert(0, str(model_dir))
    GmeQwen2VL = importlib.import_module("gme_inference").GmeQwen2VL
    gme = GmeQwen2VL(model_path=str(model_dir), device="cpu")
    gme.base = gme.base.float().eval()
    with torch.no_grad():
        fused = gme.get_fused_embeddings([meta["texts"][0]], [meta["image"]]).float().numpy()
    np.save(path, fused)
    return fused


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE,
                        help="exported OpenVINO package directory")
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS,
                        help="directory containing golden vectors and report output")
    parser.add_argument("--source-model", type=Path, default=DEFAULT_SOURCE_MODEL,
                        help="GME checkpoint used only when golden_fused.npy is missing")
    parser.add_argument("--devices", default="CPU,GPU",
                        help="comma-separated OpenVINO devices to test (default: CPU,GPU)")
    parser.add_argument("--precisions",
                        help="comma-separated package precision keys (default: all package precisions)")
    parser.add_argument("--output", type=Path,
                        help="validation JSON path (default: <artifacts>/openvino_full_validation.json)")
    args = parser.parse_args()

    package = args.package.expanduser().resolve()
    artifacts = args.artifacts.expanduser().resolve()
    model_dir = args.source_model.expanduser().resolve()
    if not package.is_dir():
        raise SystemExit(f"package directory not found: {package}")
    artifacts.mkdir(parents=True, exist_ok=True)
    meta_path = artifacts / "golden_meta.json"
    if not meta_path.is_file():
        raise SystemExit(f"golden metadata not found: {meta_path}; run 01_reference.py first")

    meta = json.loads(meta_path.read_text())
    golden = {
        "text": np.load(artifacts / "golden_text.npy"),
        "query": np.load(artifacts / "golden_query.npy"),
        "image": np.load(artifacts / "golden_image.npy"),
        "fused": ensure_fused_golden(meta, artifacts, model_dir),
    }

    cfg = json.loads((package / "gme_openvino_config.json").read_text())
    precisions = [item.strip() for item in args.precisions.split(",") if item.strip()] if args.precisions else list(cfg["precisions"])
    unknown_precisions = [item for item in precisions if item not in cfg["precisions"]]
    if unknown_precisions:
        raise SystemExit(f"unknown precision(s) {unknown_precisions}; choose from {list(cfg['precisions'])}")
    requested_devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    available = GmeOpenVINO(package, precision="fp32").core.available_devices
    devices = [device for device in requested_devices if device in available]
    if not devices:
        raise SystemExit(f"none of the requested devices are available: requested={requested_devices}, found={available}")
    print(f"[validate] devices={devices} precisions={precisions}\n")

    report: dict = {}
    for device in devices:
        report[device] = {}
        for prec in precisions:
            enc = GmeOpenVINO(package, device=device, precision=prec)
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

    output = (args.output or artifacts / "openvino_full_validation.json").expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[validate] wrote {output}")


if __name__ == "__main__":
    main()
