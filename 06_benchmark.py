#!/usr/bin/env python
"""Latency benchmark for the GME OpenVINO runtime (Python and C++).

Measures warm end-to-end per-call latency (tokenize/preprocess + inference,
excluding one-time model load/compile) for device x precision x mode. Uses one
short text + the local logo so numbers are comparable across cells. Pass
``--sequence-lengths 128,256`` to benchmark text/query at exact final token
lengths (after GME prompt templating), without running image/fused modes.

Writes artifacts/benchmark.json and prints a table.
"""
import argparse
import json
import re
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image

from gme_openvino import GmeOpenVINO, load_image
from gme_paths import ARTIFACTS as ART, CPP_BINARY as BIN, LOGO_IMAGE, OPENVINO_SETUP as SETUP, PACKAGE as PKG

HERE = Path(__file__).resolve().parent
IMAGE = str(LOGO_IMAGE)

TEXT = "What kind of car is this?"
INSTR = "Find an image that matches the given text."
DEVICES = ["CPU", "GPU"]
PRECISIONS = ["fp32", "int8", "int4"]
REPEAT = 5


def describe_workload():
    """Return the post-template and post-processor shapes behind each timing."""
    encoder = GmeOpenVINO(PKG, device="CPU", precision="fp32")

    def text_shape(instruction):
        prompt = encoder._prompt(TEXT, has_image=False, instruction=instruction)
        inputs = encoder.processor(
            text=[prompt], padding=True, truncation=True,
            max_length=encoder.max_length, return_tensors="np",
        )
        return {
            "raw_text_chars": len(TEXT),
            "instruction_chars": len(instruction) if instruction != encoder._instruction(None, False) else 0,
            "prompt_chars": len(prompt),
            "sequence_tokens": int(inputs["attention_mask"][0].sum()),
        }

    source = Image.open(IMAGE)
    resized = load_image(IMAGE)

    def image_shape(text):
        prompt = encoder._prompt(text, has_image=True, instruction=encoder._instruction(None, False))
        inputs = encoder.processor(
            text=[prompt], images=[resized], padding=True, truncation=True,
            max_length=encoder.max_length, return_tensors="np",
        )
        grid_t, grid_h, grid_w = (int(value) for value in inputs["image_grid_thw"][0])
        return {
            "raw_text_chars": len(text) if text else 0,
            "prompt_chars": len(prompt),
            "sequence_tokens": int(inputs["attention_mask"][0].sum()),
            "image_grid_thw": [grid_t, grid_h, grid_w],
            "raw_patch_rows": int(inputs["pixel_values"].shape[0]),
            "visual_tokens": grid_t * grid_h * grid_w // 4,
        }

    return {
        "batch_size": 1,
        "text": text_shape(encoder._instruction(None, False)),
        "query": text_shape(INSTR),
        "image": {
            "source_size_wh": list(source.size),
            "gme_fetch_resize_wh": list(resized.size),
            **image_shape(None),
        },
        "fused": image_shape(TEXT),
    }


def sequence_length(encoder, text, instruction):
    prompt = encoder._prompt(text, has_image=False, instruction=instruction)
    inputs = encoder.processor(
        text=[prompt], padding=True, truncation=True,
        max_length=encoder.max_length, return_tensors="np",
    )
    return int(inputs["attention_mask"][0].sum())


def text_for_sequence_length(encoder, target, instruction):
    """Create harmless ASCII filler whose final GME sequence has ``target`` tokens."""
    base = sequence_length(encoder, "", instruction)
    if target < base:
        raise ValueError(f"target {target} is shorter than the {base}-token GME template")
    for count in range(target - base + 2):
        text = " ".join(["a"] * count)
        actual = sequence_length(encoder, text, instruction)
        if actual == target:
            return text
        if actual > target:
            break
    raise RuntimeError(
        f"cannot construct a {target}-token prompt using the current tokenizer; "
        f"last length was {actual}"
    )


def sequence_length_workloads(targets):
    """Build independently token-exact text and query requests for each target."""
    encoder = GmeOpenVINO(PKG, device="CPU", precision="fp32")
    result = {}
    default_instruction = encoder._instruction(None, False)
    for target in targets:
        text = text_for_sequence_length(encoder, target, default_instruction)
        query = text_for_sequence_length(encoder, target, INSTR)
        result[str(target)] = {
            "text": {
                "raw_text": text,
                "raw_text_chars": len(text),
                "sequence_tokens": sequence_length(encoder, text, default_instruction),
            },
            "query": {
                "raw_text": query,
                "raw_text_chars": len(query),
                "instruction": INSTR,
                "instruction_chars": len(INSTR),
                "sequence_tokens": sequence_length(encoder, query, INSTR),
            },
        }
    return result


def bench_python(text=None, query_text=None):
    out = {}
    for dev in DEVICES:
        avail = GmeOpenVINO(PKG, precision="fp32").core.available_devices
        if dev not in avail:
            continue
        out[dev] = {}
        for prec in PRECISIONS:
            enc = GmeOpenVINO(PKG, device=dev, precision=prec)
            if text is None:
                modes = {
                    "text": lambda: enc.get_text_embeddings([TEXT], is_query=False),
                    "query": lambda: enc.get_text_embeddings([TEXT], instruction=INSTR),
                    "image": lambda: enc.get_image_embeddings(IMAGE),
                    "fused": lambda: enc.get_fused_embeddings(TEXT, IMAGE),
                }
            else:
                modes = {
                    "text": lambda: enc.get_text_embeddings([text], is_query=False),
                    "query": lambda: enc.get_text_embeddings([query_text], instruction=INSTR),
                }
            row = {}
            for name, fn in modes.items():
                fn()  # warmup
                t0 = time.perf_counter()
                for _ in range(REPEAT):
                    fn()
                row[name] = round((time.perf_counter() - t0) / REPEAT * 1000, 2)
            out[dev][prec] = row
            print(f"[py ] {dev} {prec:5s} " + " ".join(f"{k}={v}ms" for k, v in row.items()))
    return out


def bench_cpp(text=None, query_text=None):
    out = {}
    for dev in DEVICES:
        for prec in PRECISIONS:
            row = {}
            requests = [("text,query,image,fused", TEXT)] if text is None else [
                ("text", text), ("query", query_text),
            ]
            for modes, request_text in requests:
                cmd = (f'set +u; source "{SETUP}" >/dev/null 2>&1; "{BIN}" --package "{PKG}" '
                       f'--device {dev} --precision {prec} --benchmark --benchmark-modes {modes} --repeat {REPEAT} '
                       f'--instruction "{INSTR}" --image "{IMAGE}" "{request_text}"')
                run = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=900)
                for match in re.finditer(r"mode=(\w+) avg_ms=([\d.]+)", run.stdout):
                    row[match.group(1)] = round(float(match.group(2)), 2)
            if not row:
                print(f"[cpp] {dev} {prec}: no timing (device unavailable?)\n{run.stderr[-300:]}")
                continue
            out.setdefault(dev, {})[prec] = row
            print(f"[cpp] {dev} {prec:5s} " + " ".join(f"{k}={v}ms" for k, v in row.items()))
    return out


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sequence-lengths",
        help="comma-separated final effective token lengths for text/query, for example 128,256",
    )
    parser.add_argument("--repeat", type=int, default=REPEAT, help=f"warm calls per measurement (default: {REPEAT})")
    return parser.parse_args()


def main():
    args = parse_args()
    global REPEAT
    REPEAT = args.repeat
    if REPEAT < 1:
        raise ValueError("--repeat must be positive")
    print(f"repeat={REPEAT}, warm end-to-end per-call latency (ms)\n")

    if args.sequence_lengths:
        targets = [int(value) for value in args.sequence_lengths.split(",") if value.strip()]
        if not targets or any(value < 1 for value in targets):
            raise ValueError("--sequence-lengths needs positive comma-separated integers")
        if len(set(targets)) != len(targets):
            raise ValueError("--sequence-lengths must not contain duplicates")
        workloads = sequence_length_workloads(targets)
        report = {
            "repeat": REPEAT,
            "unit": "ms/call (warm, end-to-end incl. tokenize/preprocess)",
            "benchmark_type": "text-query-sequence-length-sweep",
            "sequence_length_definition": "effective unpadded tokens after GME prompt templating",
            "sequence_length_workloads": workloads,
            "results": {},
        }
        for target, workload in workloads.items():
            print(f"== Sequence length {target} ==")
            print(json.dumps(workload, ensure_ascii=False, indent=2))
            report["results"][target] = {
                "python": bench_python(workload["text"]["raw_text"], workload["query"]["raw_text"]),
                "cpp": bench_cpp(workload["text"]["raw_text"], workload["query"]["raw_text"]),
            }
        out_path = ART / "benchmark_sequence_lengths.json"
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(f"\nwrote {out_path}")
        return

    workload = describe_workload()
    print("== Workload ==")
    print(json.dumps(workload, ensure_ascii=False, indent=2))
    print("== Python ==")
    py = bench_python()
    print("\n== C++ ==")
    cpp = bench_cpp()
    report = {"repeat": REPEAT, "unit": "ms/call (warm, end-to-end incl. tokenize/preprocess)",
              "workload": workload,
              "python": py, "cpp": cpp}
    (ART / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {ART / 'benchmark.json'}")


if __name__ == "__main__":
    main()
