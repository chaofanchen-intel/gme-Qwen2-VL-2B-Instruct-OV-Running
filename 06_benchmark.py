#!/usr/bin/env python
"""Latency benchmark for the GME OpenVINO runtime (Python and C++).

Measures warm end-to-end per-call latency (tokenize/preprocess + inference,
excluding one-time model load/compile) for device x precision x mode. Uses one
short text + the local logo so numbers are comparable across cells.

Writes artifacts/benchmark.json and prints a table.
"""
import json
import re
import subprocess
import time
from pathlib import Path

import numpy as np

from gme_openvino import GmeOpenVINO

HERE = Path(__file__).resolve().parent
PKG = HERE / "ov_models" / "gme-qwen2-vl-2b-instruct-openvino"
ART = HERE / "artifacts"
BIN = HERE / "cpp" / "build" / "gme_embed"
SETUP = HERE.parent / "benchmark" / "openvino" / "install" / "setupvars.sh"
IMAGE = str(HERE.parent / "benchmark" / "models" / "gme-Qwen2-VL-2B-Instruct" / "images" / "gme_logo.png")

TEXT = "What kind of car is this?"
INSTR = "Find an image that matches the given text."
DEVICES = ["CPU", "GPU"]
PRECISIONS = ["fp32", "int8", "int4"]
REPEAT = 5


def bench_python():
    out = {}
    for dev in DEVICES:
        avail = GmeOpenVINO(PKG, precision="fp32").core.available_devices
        if dev not in avail:
            continue
        out[dev] = {}
        for prec in PRECISIONS:
            enc = GmeOpenVINO(PKG, device=dev, precision=prec)
            modes = {
                "text": lambda: enc.get_text_embeddings([TEXT], is_query=False),
                "query": lambda: enc.get_text_embeddings([TEXT], instruction=INSTR),
                "image": lambda: enc.get_image_embeddings(IMAGE),
                "fused": lambda: enc.get_fused_embeddings(TEXT, IMAGE),
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


def bench_cpp():
    out = {}
    for dev in DEVICES:
        for prec in PRECISIONS:
            cmd = (f'set +u; source "{SETUP}" >/dev/null 2>&1; "{BIN}" --package "{PKG}" '
                   f'--device {dev} --precision {prec} --benchmark --repeat {REPEAT} '
                   f'--instruction "{INSTR}" --image "{IMAGE}" "{TEXT}"')
            r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=900)
            row = {}
            for m in re.finditer(r"mode=(\w+) avg_ms=([\d.]+)", r.stdout):
                row[m.group(1)] = round(float(m.group(2)), 2)
            if not row:
                print(f"[cpp] {dev} {prec}: no timing (device unavailable?)\n{r.stderr[-300:]}")
                continue
            out.setdefault(dev, {})[prec] = row
            print(f"[cpp] {dev} {prec:5s} " + " ".join(f"{k}={v}ms" for k, v in row.items()))
    return out


def main():
    print(f"repeat={REPEAT}, warm end-to-end per-call latency (ms)\n")
    print("== Python ==")
    py = bench_python()
    print("\n== C++ ==")
    cpp = bench_cpp()
    report = {"repeat": REPEAT, "unit": "ms/call (warm, end-to-end incl. tokenize/preprocess)",
              "python": py, "cpp": cpp}
    (ART / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {ART / 'benchmark.json'}")


if __name__ == "__main__":
    main()
