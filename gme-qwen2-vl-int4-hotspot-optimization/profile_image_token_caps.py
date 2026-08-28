#!/usr/bin/env python3
"""Profile image-token cap latency and embedding accuracy through the public GME API."""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from paths import GME_PACKAGE

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_META = ROOT / "artifacts" / "golden_meta.json"


def cosine(lhs: np.ndarray, rhs: np.ndarray) -> float:
    lhs64 = lhs.astype(np.float64).reshape(-1)
    rhs64 = rhs.astype(np.float64).reshape(-1)
    return float(np.dot(lhs64, rhs64) / (np.linalg.norm(lhs64) * np.linalg.norm(rhs64)))


def timed(call, warmup: int, repeat: int) -> tuple[np.ndarray, list[float]]:
    result = call()
    for _ in range(warmup - 1):
        result = call()
    samples = []
    for _ in range(repeat):
        started = time.perf_counter()
        result = call()
        samples.append((time.perf_counter() - started) * 1000)
    return result, samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=GME_PACKAGE)
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--precision", default="int4_causal_vopt")
    parser.add_argument("--device", default="GPU")
    parser.add_argument("--caps", default="256,384,512,768,1024,1280")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.warmup < 1 or args.repeat < 1:
        raise SystemExit("--warmup and --repeat must be positive")
    caps = sorted({int(value) for value in args.caps.split(",")})
    if not caps or caps[0] < 1:
        raise SystemExit("--caps must contain positive integers")

    meta = json.loads(args.meta.read_text())
    image_path = Path(meta["image"])
    image = Image.open(image_path).convert("RGB")
    text = meta["texts"][0]
    golden_image = np.load(args.meta.parent / "golden_image.npy")
    golden_fused = np.load(args.meta.parent / "golden_fused.npy")

    from gme_openvino import GmeOpenVINO

    rows = []
    embeddings: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for cap in caps:
        started = time.perf_counter()
        encoder = GmeOpenVINO(
            args.package,
            device=args.device,
            precision=args.precision,
            max_image_tokens=cap,
        )
        compile_seconds = time.perf_counter() - started

        prompt = encoder._prompt(None, has_image=True, instruction=encoder._instruction(None, True))
        inputs = encoder.processor(
            text=[prompt],
            images=[image],
            padding=True,
            truncation=True,
            max_length=encoder.max_length,
            return_tensors="np",
        )
        grid = [int(value) for value in inputs["image_grid_thw"][0]]
        visual_tokens = int(np.prod(grid) // 4)
        sequence_tokens = int(inputs["attention_mask"][0].sum())

        image_embedding, image_samples = timed(
            lambda: encoder.get_image_embeddings(image, batch_size=None, show_progress_bar=False),
            args.warmup,
            args.repeat,
        )
        fused_embedding, fused_samples = timed(
            lambda: encoder.get_fused_embeddings(text, image, batch_size=None, show_progress_bar=False),
            args.warmup,
            args.repeat,
        )
        embeddings[cap] = (image_embedding.copy(), fused_embedding.copy())

        row = {
            "max_image_tokens": cap,
            "image_grid_thw": grid,
            "visual_tokens": visual_tokens,
            "sequence_tokens": sequence_tokens,
            "compile_seconds": round(compile_seconds, 3),
            "image_mean_ms": round(float(np.mean(image_samples)), 3),
            "image_p50_ms": round(float(np.percentile(image_samples, 50)), 3),
            "fused_mean_ms": round(float(np.mean(fused_samples)), 3),
            "fused_p50_ms": round(float(np.percentile(fused_samples, 50)), 3),
            "image_cos_vs_fp32": round(cosine(image_embedding, golden_image), 9),
            "fused_cos_vs_fp32": round(cosine(fused_embedding, golden_fused), 9),
        }
        rows.append(row)
        print(
            f"cap={cap:4d} visual_tokens={visual_tokens:4d} "
            f"image_mean={row['image_mean_ms']:8.1f}ms fused_mean={row['fused_mean_ms']:8.1f}ms "
            f"cos_fp32={row['image_cos_vs_fp32']:.6f}/{row['fused_cos_vs_fp32']:.6f}"
        )
        del encoder
        gc.collect()

    reference_cap = max(caps)
    reference_image, reference_fused = embeddings[reference_cap]
    for row in rows:
        image_embedding, fused_embedding = embeddings[row["max_image_tokens"]]
        row["image_cos_vs_max_cap"] = round(cosine(image_embedding, reference_image), 9)
        row["fused_cos_vs_max_cap"] = round(cosine(fused_embedding, reference_fused), 9)

    report = {
        "precision": args.precision,
        "device": args.device,
        "image": str(image_path),
        "source_size_wh": list(image.size),
        "reference_cap": reference_cap,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()