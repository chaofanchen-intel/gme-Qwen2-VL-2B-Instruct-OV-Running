#!/usr/bin/env python3
"""Profile Qwen3-VL-Embedding-2B text latency with per-call samples."""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import openvino_genai

from paths import MODELS_ROOT

MODELS = MODELS_ROOT


def make_text(tokenizer: openvino_genai.Tokenizer, target: int) -> str:
    for words in range(target * 4 + 64):
        text = " ".join(["a"] * words)
        if int(tokenizer.encode(text).input_ids.shape[1]) == target:
            return text
    raise RuntimeError(f"cannot construct exact {target}-token text")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", choices=("INT8", "INT4"), default="INT4")
    parser.add_argument("--length", type=int, choices=(32, 128, 512, 1024), default=1024)
    parser.add_argument("--warmup", type=int, default=14)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model_dir = MODELS / f"Qwen3-VL-Embedding-2B-{args.precision}"
    tokenizer = openvino_genai.Tokenizer(str(model_dir))
    text = make_text(tokenizer, args.length)
    pipeline = openvino_genai.EmbeddingPipeline(str(model_dir), "GPU")
    for _ in range(args.warmup):
        pipeline.embed(text)
    samples = []
    shape = None
    for _ in range(args.repeat):
        started = time.perf_counter()
        result = pipeline.embed(text)
        samples.append((time.perf_counter() - started) * 1000)
        shape = list(result.embeddings.shape)
    output = {
        "model": str(model_dir),
        "precision": args.precision,
        "device": "GPU",
        "target_tokens": args.length,
        "tokenizer_tokens": int(tokenizer.encode(text).input_ids.shape[1]),
        "shape": shape,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "samples_ms": [round(x, 3) for x in samples],
        "mean_ms": round(statistics.mean(samples), 3),
        "median_ms": round(statistics.median(samples), 3),
        "p10_ms": round(min(samples), 3),
        "p90_ms": round(max(samples), 3),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({key: output[key] for key in ("precision", "target_tokens", "shape", "mean_ms", "median_ms", "p10_ms", "p90_ms")}))


if __name__ == "__main__":
    main()
