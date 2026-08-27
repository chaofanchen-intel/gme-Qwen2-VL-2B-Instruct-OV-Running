#!/usr/bin/env python3
"""Profile steady-state GME Qwen2-VL GPU text/query inference."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
from paths import GME_PACKAGE, GME_ROOT

PACKAGE = GME_PACKAGE
INSTRUCTION = "Find an image that matches the given text."


def sequence_length(encoder, text: str, instruction: str) -> int:
    prompt = encoder._prompt(text, has_image=False, instruction=instruction)
    inputs = encoder.processor(
        text=[prompt], padding=True, truncation=True,
        max_length=encoder.max_length, return_tensors="np",
    )
    return int(inputs["attention_mask"][0].sum())


WORKLOAD_CACHE = HERE / "results" / "workloads.json"


def text_for_length(encoder, target: int, instruction: str) -> str:
    """Shortest ``"a a a ..."`` string whose prompt tokenizes to exactly `target` tokens.

    The linear search costs ~`target` tokenizer calls, which dominates a short benchmark run
    and shows up as CPU time in a profiler capture, so results are memoized on disk.
    """
    cache = json.loads(WORKLOAD_CACHE.read_text()) if WORKLOAD_CACHE.exists() else {}
    key = f"{target}|{instruction}"
    if key in cache:
        return cache[key]
    for count in range(0, target * 4 + 64):
        text = " ".join(["a"] * count)
        actual = sequence_length(encoder, text, instruction)
        if actual == target:
            cache[key] = text
            WORKLOAD_CACHE.parent.mkdir(parents=True, exist_ok=True)
            WORKLOAD_CACHE.write_text(json.dumps(cache, indent=2) + "\n", encoding="utf-8")
            return text
        if actual > target:
            break
    raise RuntimeError(f"Could not construct an exact {target}-token workload")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=1024, choices=(32, 128, 512, 1024))
    parser.add_argument("--warmup", type=int, default=14)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--hold-seconds", type=int, default=0,
                        help="keep the process alive after timing for VTune attach")
    parser.add_argument("--output", type=Path, default=HERE / "results" / "profile_run.json")
    parser.add_argument("--batch-mode", choices=("padded", "per_sample"), default="padded")
    parser.add_argument("--streams", default="1")
    parser.add_argument("--precision", default="int4_causal_vopt",
                        help="package precision key (default: int4_causal_vopt)")
    args = parser.parse_args()

    sys.path.insert(0, str(GME_ROOT))
    from gme_openvino import GmeOpenVINO

    streams: int | str = "auto" if args.streams == "auto" else int(args.streams)
    encoder = GmeOpenVINO(
        PACKAGE,
        device="GPU",
        precision=args.precision,
        streams=streams,
        batch_mode=args.batch_mode,
    )
    text = text_for_length(encoder, args.length, encoder._instruction(None, False))
    query = text_for_length(encoder, args.length, INSTRUCTION)
    workloads = {"text": text, "query": query}
    output = {
        "package": str(PACKAGE),
        "device": "GPU",
        "precision": args.precision,
        "batch_mode": encoder.batch_mode,
        "streams": args.streams,
        "target_tokens": args.length,
        "workloads": {},
        "warmup": args.warmup,
        "repeat": args.repeat,
    }

    for name, prompt in workloads.items():
        call = (
            lambda prompt=prompt, name=name: encoder.get_text_embeddings(
                [prompt],
                instruction=INSTRUCTION if name == "query" else None,
                is_query=False if name == "text" else True,
                batch_size=1,
                show_progress_bar=False,
            )
        )
        for _ in range(args.warmup):
            call()
        samples = []
        for _ in range(args.repeat):
            started = time.perf_counter()
            result = call()
            samples.append((time.perf_counter() - started) * 1000)
        output["workloads"][name] = {
            "effective_tokens": sequence_length(encoder, prompt, encoder._instruction(INSTRUCTION if name == "query" else None, name == "query")),
            "shape": list(result.shape),
            "mean_ms": round(float(np.mean(samples)), 3),
            "p50_ms": round(float(np.percentile(samples, 50)), 3),
            "min_ms": round(float(np.min(samples)), 3),
            "max_ms": round(float(np.max(samples)), 3),
        }
        print(
            f"{name}: tokens={output['workloads'][name]['effective_tokens']} "
            f"shape={result.shape} mean_ms={output['workloads'][name]['mean_ms']}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    if args.hold_seconds:
        print(f"holding process for {args.hold_seconds}s for profiler attach")
        time.sleep(args.hold_seconds)


if __name__ == "__main__":
    main()
