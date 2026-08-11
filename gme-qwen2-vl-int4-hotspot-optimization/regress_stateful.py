#!/usr/bin/env python3
"""GenAI-free greedy decode on a stateful LLM IR, to isolate the OpenVINO GPU plugin.

regress_generative.py goes through openvino_genai, whose local build differs from the wheel,
so it cannot attribute a difference to the plugin. This drives the stateful IR directly with
plain OpenVINO and dumps the argmax token ids plus the raw first-step logits.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import openvino as ov
from openvino.properties import hint

from paths import MODELS_ROOT

PROMPT_IDS = [
    [3838, 374, 279, 6722, 315, 9625, 30],
    [151644, 872, 198, 8420, 374, 264, 1140, 315, 10250, 5109, 25],
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=MODELS_ROOT / "Qwen3-1.7B-INT4/openvino_model.xml")
    ap.add_argument("--device", default="GPU")
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    core = ov.Core()
    compiled = core.compile_model(str(args.model), args.device,
                                  {hint.performance_mode: hint.PerformanceMode.LATENCY})
    input_names = {p.get_any_name() for p in compiled.inputs}

    out = {"model": str(args.model), "device": args.device, "runs": []}
    for prompt in PROMPT_IDS:
        request = compiled.create_infer_request()
        ids = np.array([prompt], dtype=np.int64)
        past = 0
        generated: list[int] = []
        first_logits = None
        for step in range(args.steps):
            n = ids.shape[1]
            feed = {
                "input_ids": ids,
                "attention_mask": np.ones((1, past + n), dtype=np.int64),
            }
            if "position_ids" in input_names:
                feed["position_ids"] = np.arange(past, past + n, dtype=np.int64).reshape(1, n)
            if "beam_idx" in input_names:
                feed["beam_idx"] = np.zeros(1, dtype=np.int32)
            logits = np.asarray(request.infer(feed)[compiled.output(0)])
            if step == 0:
                first_logits = logits[0, -1].astype(np.float32).copy()
            token = int(np.argmax(logits[0, -1]))
            generated.append(token)
            past += n
            ids = np.array([[token]], dtype=np.int64)
        out["runs"].append({
            "prompt_ids": prompt,
            "generated_ids": generated,
            "first_step_logits_sha": float(np.sum(first_logits.astype(np.float64) ** 2)),
            "first_step_logits_top8": [round(float(v), 6) for v in np.sort(first_logits)[-8:]],
        })
        print(f"prompt {prompt[:4]}... -> {generated[:12]}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
