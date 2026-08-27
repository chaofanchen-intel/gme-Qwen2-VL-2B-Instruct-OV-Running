#!/usr/bin/env python3
"""Profile GME Qwen2-VL INT4 text latency with per-call samples."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from paths import GME_PACKAGE, GME_ROOT

PACKAGE = GME_PACKAGE
INSTRUCTION = "Find an image that matches the given text."


def sequence_length(encoder, text: str, instruction: str) -> int:
    prompt = encoder._prompt(text, has_image=False, instruction=instruction)
    inputs = encoder.processor(text=[prompt], padding=True, truncation=True,
                               max_length=encoder.max_length, return_tensors="np")
    return int(inputs["attention_mask"][0].sum())


def make_text(encoder, target: int, instruction: str) -> str:
    for words in range(target * 4 + 64):
        text = " ".join(["a"] * words)
        if sequence_length(encoder, text, instruction) == target:
            return text
    raise RuntimeError(f"cannot construct exact {target}-token text")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, choices=(32, 128, 512, 1024), default=1024)
    parser.add_argument("--warmup", type=int, default=14)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(GME_ROOT))
    from gme_openvino import GmeOpenVINO

    encoder = GmeOpenVINO(PACKAGE, device="GPU", precision="int4_causal_vopt", streams=1, batch_mode="padded")
    text = make_text(encoder, args.length, encoder._instruction(None, False))
    for _ in range(args.warmup):
        encoder.get_text_embeddings([text], is_query=False, batch_size=1, show_progress_bar=False)
    samples = []
    shape = None
    for _ in range(args.repeat):
        started = time.perf_counter()
        result = encoder.get_text_embeddings([text], is_query=False, batch_size=1, show_progress_bar=False)
        samples.append((time.perf_counter() - started) * 1000)
        shape = list(result.shape)
    output = {
        "model": str(PACKAGE), "precision": "INT4", "device": "GPU",
        "target_tokens": args.length, "effective_tokens": sequence_length(encoder, text, encoder._instruction(None, False)),
        "shape": shape, "warmup": args.warmup, "repeat": args.repeat,
        "samples_ms": [round(x, 3) for x in samples],
        "mean_ms": round(statistics.mean(samples), 3),
        "median_ms": round(statistics.median(samples), 3),
        "p10_ms": round(min(samples), 3), "p90_ms": round(max(samples), 3),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({key: output[key] for key in ("target_tokens", "shape", "mean_ms", "median_ms", "p10_ms", "p90_ms")}))


if __name__ == "__main__":
    main()
