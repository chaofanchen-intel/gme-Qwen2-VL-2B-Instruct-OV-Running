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
import shlex
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image

from gme_openvino import GmeOpenVINO, load_image

HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE_MODEL = HERE.parent / "benchmark" / "models" / "gme-Qwen2-VL-2B-Instruct"
DEFAULT_PACKAGE = HERE / "ov_models" / "gme-qwen2-vl-2b-instruct-openvino"
DEFAULT_ARTIFACTS = HERE / "artifacts"
DEFAULT_CPP_BINARY = HERE / "cpp" / "build" / "gme_embed"
DEFAULT_OPENVINO_INSTALL = HERE.parent / "benchmark" / "openvino" / "install"

TEXT = "What kind of car is this?"
INSTR = "Find an image that matches the given text."
DEVICES = ["CPU", "GPU"]
PRECISIONS = ["fp32", "int8", "int4"]
REPEAT = 5


def describe_workload(package, image, text, instruction):
    """Return the post-template and post-processor shapes behind each timing."""
    encoder = GmeOpenVINO(package, device="CPU", precision="fp32")

    def text_shape(instruction):
        prompt = encoder._prompt(text, has_image=False, instruction=instruction)
        inputs = encoder.processor(
            text=[prompt], padding=True, truncation=True,
            max_length=encoder.max_length, return_tensors="np",
        )
        return {
            "raw_text_chars": len(text),
            "instruction_chars": len(instruction) if instruction != encoder._instruction(None, False) else 0,
            "prompt_chars": len(prompt),
            "sequence_tokens": int(inputs["attention_mask"][0].sum()),
        }

    source = Image.open(image)
    resized = load_image(image)

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
        "query": text_shape(instruction),
        "image": {
            "source_size_wh": list(source.size),
            "gme_fetch_resize_wh": list(resized.size),
            **image_shape(None),
        },
        "fused": image_shape(text),
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


def sequence_length_workloads(package, targets, instruction):
    """Build independently token-exact text and query requests for each target."""
    encoder = GmeOpenVINO(package, device="CPU", precision="fp32")
    result = {}
    default_instruction = encoder._instruction(None, False)
    for target in targets:
        text = text_for_sequence_length(encoder, target, default_instruction)
        query = text_for_sequence_length(encoder, target, instruction)
        result[str(target)] = {
            "text": {
                "raw_text": text,
                "raw_text_chars": len(text),
                "sequence_tokens": sequence_length(encoder, text, default_instruction),
            },
            "query": {
                "raw_text": query,
                "raw_text_chars": len(query),
                "instruction": instruction,
                "instruction_chars": len(instruction),
                "sequence_tokens": sequence_length(encoder, query, instruction),
            },
        }
    return result


def bench_python(package, devices, precisions, repeat, image, default_text, instruction,
                 text=None, query_text=None):
    out = {}
    for dev in devices:
        avail = GmeOpenVINO(package, precision="fp32").core.available_devices
        if dev not in avail:
            continue
        out[dev] = {}
        for prec in precisions:
            enc = GmeOpenVINO(package, device=dev, precision=prec)
            if text is None:
                modes = {
                    "text": lambda: enc.get_text_embeddings([default_text], is_query=False),
                    "query": lambda: enc.get_text_embeddings([default_text], instruction=instruction),
                    "image": lambda: enc.get_image_embeddings(image),
                    "fused": lambda: enc.get_fused_embeddings(default_text, image),
                }
            else:
                modes = {
                    "text": lambda: enc.get_text_embeddings([text], is_query=False),
                    "query": lambda: enc.get_text_embeddings([query_text], instruction=instruction),
                }
            row = {}
            for name, fn in modes.items():
                fn()  # warmup
                t0 = time.perf_counter()
                for _ in range(repeat):
                    fn()
                row[name] = round((time.perf_counter() - t0) / repeat * 1000, 2)
            out[dev][prec] = row
            print(f"[py ] {dev} {prec:5s} " + " ".join(f"{k}={v}ms" for k, v in row.items()))
    return out


def bench_cpp(package, binary, setup, devices, precisions, repeat, image, default_text, instruction,
               text=None, query_text=None):
    out = {}
    for dev in devices:
        for prec in precisions:
            row = {}
            requests = [("text,query,image,fused", default_text)] if text is None else [
                ("text", text), ("query", query_text),
            ]
            for modes, request_text in requests:
                command = [
                    str(binary), "--package", str(package), "--device", dev, "--precision", prec,
                    "--benchmark", "--benchmark-modes", modes, "--repeat", str(repeat),
                    "--instruction", instruction,
                ]
                if text is None:
                    command += ["--image", str(image)]
                command.append(request_text)
                shell = (
                    f"set +u; source {shlex.quote(str(setup))} >/dev/null 2>&1; "
                    f"exec {shlex.join(command)}"
                )
                run = subprocess.run(["bash", "-c", shell], capture_output=True, text=True, timeout=900)
                for match in re.finditer(r"mode=(\w+) avg_ms=([\d.]+)", run.stdout):
                    row[match.group(1)] = round(float(match.group(2)), 2)
            if not row:
                print(f"[cpp] {dev} {prec}: no timing (device unavailable?)\n{run.stderr[-300:]}")
                continue
            out.setdefault(dev, {})[prec] = row
            print(f"[cpp] {dev} {prec:5s} " + " ".join(f"{k}={v}ms" for k, v in row.items()))
    return out


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE,
                        help="exported OpenVINO package directory")
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS,
                        help="directory for default benchmark reports")
    parser.add_argument("--source-model", type=Path, default=DEFAULT_SOURCE_MODEL,
                        help="used for the default logo image path")
    parser.add_argument("--image", type=Path,
                        help="image for image/fused timing (default: <source-model>/images/gme_logo.png)")
    parser.add_argument("--cpp-binary", type=Path, default=DEFAULT_CPP_BINARY,
                        help="gme_embed executable")
    parser.add_argument("--openvino-install", type=Path, default=DEFAULT_OPENVINO_INSTALL,
                        help="OpenVINO installation containing setupvars.sh")
    parser.add_argument("--backends", default="python,cpp",
                        help="comma-separated backends: python,cpp")
    parser.add_argument("--devices", default=",".join(DEVICES),
                        help="comma-separated OpenVINO devices")
    parser.add_argument("--precisions", default=",".join(PRECISIONS),
                        help="comma-separated package precisions")
    parser.add_argument("--text", default=TEXT, help="short text used by the fixed workload")
    parser.add_argument("--instruction", default=INSTR, help="query instruction")
    parser.add_argument(
        "--sequence-lengths",
        help="comma-separated final effective token lengths for text/query, for example 128,256",
    )
    parser.add_argument("--repeat", type=int, default=5, help="warm calls per measurement (default: 5)")
    parser.add_argument("--output", type=Path,
                        help="report path (default: <artifacts>/benchmark.json or benchmark_sequence_lengths.json)")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.repeat < 1:
        raise ValueError("--repeat must be positive")

    package = args.package.expanduser().resolve()
    artifacts = args.artifacts.expanduser().resolve()
    source_model = args.source_model.expanduser().resolve()
    image = (args.image or source_model / "images" / "gme_logo.png").expanduser().resolve()
    binary = args.cpp_binary.expanduser().resolve()
    setup = args.openvino_install.expanduser().resolve() / "setupvars.sh"
    if not package.is_dir():
        raise SystemExit(f"package directory not found: {package}")
    if not args.sequence_lengths and not image.is_file():
        raise SystemExit(f"benchmark image not found: {image}")
    backends = [item.strip() for item in args.backends.split(",") if item.strip()]
    unknown_backends = [item for item in backends if item not in {"python", "cpp"}]
    if unknown_backends:
        raise SystemExit(f"unknown backend(s) {unknown_backends}; choose python and/or cpp")
    if not backends:
        raise SystemExit("--backends must include python and/or cpp")
    devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    precisions = [item.strip() for item in args.precisions.split(",") if item.strip()]
    if "cpp" in backends:
        if not binary.is_file():
            raise SystemExit(f"C++ binary not found: {binary}")
        if not setup.is_file():
            raise SystemExit(f"OpenVINO setupvars.sh not found: {setup}")
    artifacts.mkdir(parents=True, exist_ok=True)
    print(f"repeat={args.repeat}, warm end-to-end per-call latency (ms)\n")

    if args.sequence_lengths:
        targets = [int(value) for value in args.sequence_lengths.split(",") if value.strip()]
        if not targets or any(value < 1 for value in targets):
            raise ValueError("--sequence-lengths needs positive comma-separated integers")
        if len(set(targets)) != len(targets):
            raise ValueError("--sequence-lengths must not contain duplicates")
        workloads = sequence_length_workloads(package, targets, args.instruction)
        report = {
            "repeat": args.repeat,
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
                "python": bench_python(
                    package, devices, precisions, args.repeat, image, args.text, args.instruction,
                    workload["text"]["raw_text"], workload["query"]["raw_text"],
                ) if "python" in backends else {},
                "cpp": bench_cpp(
                    package, binary, setup, devices, precisions, args.repeat, image,
                    args.text, args.instruction, workload["text"]["raw_text"], workload["query"]["raw_text"],
                ) if "cpp" in backends else {},
            }
        out_path = (args.output or artifacts / "benchmark_sequence_lengths.json").expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(f"\nwrote {out_path}")
        return

    workload = describe_workload(package, image, args.text, args.instruction)
    print("== Workload ==")
    print(json.dumps(workload, ensure_ascii=False, indent=2))
    print("== Python ==")
    py = bench_python(
        package, devices, precisions, args.repeat, image, args.text, args.instruction,
    ) if "python" in backends else {}
    print("\n== C++ ==")
    cpp = bench_cpp(
        package, binary, setup, devices, precisions, args.repeat, image, args.text, args.instruction,
    ) if "cpp" in backends else {}
    report = {"repeat": args.repeat, "unit": "ms/call (warm, end-to-end incl. tokenize/preprocess)",
              "backends": backends, "devices": devices, "precisions": precisions,
              "workload": workload,
              "python": py, "cpp": cpp}
    out_path = (args.output or artifacts / "benchmark.json").expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
