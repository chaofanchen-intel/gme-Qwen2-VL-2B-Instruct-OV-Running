#!/usr/bin/env python3
"""Greedy-decode regression check for the KV-cache (generative) SDPA path.

The two patched passes live in the shared GPU SDPA fusion path, so a stateful LLM has to
produce the same tokens before and after. Run once per build and diff the JSON.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import openvino_genai as ov_genai

from paths import MODELS_ROOT

PROMPTS = [
    "Explain in one paragraph why grouped-query attention reduces memory bandwidth.",
    "List three prime numbers greater than 100.",
    "Write a haiku about integrated GPUs.",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=MODELS_ROOT / "Qwen3-1.7B-INT4")
    ap.add_argument("--device", default="GPU")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    pipe = ov_genai.LLMPipeline(str(args.model), args.device)
    config = ov_genai.GenerationConfig()
    config.max_new_tokens = args.max_new_tokens
    config.do_sample = False
    config.apply_chat_template = False

    out = {"model": str(args.model), "device": args.device, "results": []}
    for prompt in PROMPTS:
        started = time.perf_counter()
        text = str(pipe.generate(prompt, config))
        out["results"].append({
            "prompt": prompt,
            "text": text,
            "elapsed_s": round(time.perf_counter() - started, 3),
        })
        print(f"[{out['results'][-1]['elapsed_s']}s] {text[:80]!r}...")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
