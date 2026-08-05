#!/usr/bin/env python
"""Performance matrix for the GME OpenVINO runtime: modes x length x size x batch.

06_benchmark.py answers "how long does one call take on the fixed reference
workload". This script answers the questions that one cannot:

* text/query at a **specified final token length**, up to 1024, not just the
  27-token reference prompt;
* image/fused at **ordinary image sizes**, not only ``gme_logo.png`` -- that file
  is a 5164x600 banner that expands to 1260 visual tokens, essentially the 1280
  ceiling, so it measures the worst case and nothing else;
* **batch scaling**: ms per call and ms per embedding as N grows, for both
  ``batch_mode`` values and both ``streams`` settings, which is where CPU and GPU
  behave oppositely.

Every cell reports ms/call and ms/embedding. ms/embedding is the number to compare
across N; ms/call is what a caller waits.

Suites (``--suite``, default all four):

  latency      N=1, all four modes, every precision, per device -- the base table.
  text-lengths text/query at each --text-lengths target.
  image-sizes  image/fused at each --image-sizes entry.
  batch        N sweep x batch_mode x streams, for text and image.

Written for report generation, so the matrix is wide: check the printed estimate
before running the full thing, and cut it down with the flags. Nothing here is
required for correctness -- 08_validate_batch.py owns that.

    source "${GME_OPENVINO_INSTALL:-../benchmark/openvino/install}/setupvars.sh"
    ./env-ref/bin/python 09_benchmark_matrix.py --suite latency --devices GPU
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image

from gme_openvino import GmeOpenVINO, load_image
from gme_paths import ARTIFACTS as ART, CPP_BINARY as BIN, LOGO_IMAGE as LOGO, OPENVINO_SETUP as SETUP, PACKAGE as PKG

HERE = Path(__file__).resolve().parent
BENCH_IMAGES = ART / "bench_images"

INSTRUCTION = "Find an image that matches the given text."
# Effective token counts after GME prompt templating, not raw text length. The
# package truncates at 1800, so 1024 is the last power-of-two that fits with room.
TEXT_LENGTHS = [32, 128, 512, 1024]
# Common capture sizes, each a multiple of 28 so neither GME's fetch_image nor the
# Qwen2-VL processor rescales them: the visual grid is then exactly (H/14, W/14)
# and the visual token count is H*W/784. 448x448 is the floor -- anything smaller
# is upscaled to 256 tokens by min_image_tokens, so it cannot be measured faster.
IMAGE_SIZES = ["448x448", "644x476", "672x672", "896x896", "1288x728"]
BATCH_SIZES = [1, 2, 4, 8, 16, 32]
# Images cost 10-100x a short text per sample, so the image batch sweep stops earlier.
IMAGE_BATCH_SIZES = [1, 2, 4, 8]
DEVICES = ["CPU", "GPU"]
PRECISIONS = ["fp32", "int8", "int4"]


# --------------------------------------------------------------- workload setup
def parse_size(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"(\d+)x(\d+)", value.strip())
    if not match:
        raise ValueError(f"--image-sizes entries must look like 448x448, got {value!r}")
    width, height = int(match.group(1)), int(match.group(2))
    if width % 28 or height % 28:
        print(f"[warn] {value} is not a multiple of 28; the processor will rescale it "
              "and the measured grid will not match the requested size")
    return width, height


def make_bench_image(width: int, height: int) -> Path:
    """A deterministic gradient at the exact requested size.

    Synthetic rather than a resized logo: content does not affect latency, and a
    neutral image keeps the file reproducible and avoids implying that these
    timings say anything about the logo's 1260-token worst case.
    """
    BENCH_IMAGES.mkdir(parents=True, exist_ok=True)
    path = BENCH_IMAGES / f"bench_{width}x{height}.png"
    if path.exists():
        return path
    x = np.linspace(0, 255, width, dtype=np.float32)[None, :]
    y = np.linspace(0, 255, height, dtype=np.float32)[:, None]
    pixels = np.stack([np.broadcast_to(x, (height, width)),
                       np.broadcast_to(y, (height, width)),
                       (x + y) / 2], axis=-1)
    Image.fromarray(pixels.astype(np.uint8)).save(path)
    return path


def sequence_length(encoder: GmeOpenVINO, text: str, instruction: str) -> int:
    prompt = encoder._prompt(text, has_image=False, instruction=instruction)
    inputs = encoder.processor(
        text=[prompt], padding=True, truncation=True,
        max_length=encoder.max_length, return_tensors="np",
    )
    return int(inputs["attention_mask"][0].sum())


def text_for_length(encoder: GmeOpenVINO, target: int, instruction: str) -> str:
    """ASCII filler whose templated prompt lands on exactly ``target`` tokens."""
    base = sequence_length(encoder, "", instruction)
    if target < base:
        raise ValueError(f"target {target} is below the {base}-token GME template for this instruction")
    words, actual = 0, base
    while actual < target:
        words += 1
        actual = sequence_length(encoder, " ".join(["a"] * words), instruction)
    if actual != target:
        raise RuntimeError(f"cannot hit exactly {target} tokens; nearest was {actual}")
    return " ".join(["a"] * words)


def describe_image(encoder: GmeOpenVINO, path: Path, text: str | None) -> dict:
    resized = load_image(path)
    prompt = encoder._prompt(text, has_image=True, instruction=encoder._instruction(None, False))
    inputs = encoder.processor(
        text=[prompt], images=[resized], padding=True, truncation=True,
        max_length=encoder.max_length, return_tensors="np",
    )
    grid = [int(value) for value in inputs["image_grid_thw"][0]]
    return {
        "file": path.name,
        "source_size_wh": list(Image.open(path).size),
        "processor_size_wh": list(resized.size),
        "image_grid_thw": grid,
        "raw_patch_rows": int(inputs["pixel_values"].shape[0]),
        "visual_tokens": grid[0] * grid[1] * grid[2] // 4,
        "sequence_tokens": int(inputs["attention_mask"][0].sum()),
    }


def build_workloads(text_lengths: list[int], image_sizes: list[str], include_logo: bool) -> dict:
    """Resolve every workload once, on CPU, so all backends measure the same input."""
    encoder = GmeOpenVINO(PKG, device="CPU", precision="int8")
    default_instruction = encoder._instruction(None, False)
    texts = {}
    for target in text_lengths:
        text = text_for_length(encoder, target, default_instruction)
        query = text_for_length(encoder, target, INSTRUCTION)
        texts[str(target)] = {
            "text": {"raw_text": text, "raw_text_chars": len(text),
                     "sequence_tokens": sequence_length(encoder, text, default_instruction)},
            "query": {"raw_text": query, "raw_text_chars": len(query), "instruction": INSTRUCTION,
                      "sequence_tokens": sequence_length(encoder, query, INSTRUCTION)},
        }
    paths = [make_bench_image(*parse_size(size)) for size in image_sizes]
    if include_logo:
        paths.append(LOGO)
    images = {}
    for path in paths:
        key = f"{Image.open(path).size[0]}x{Image.open(path).size[1]}"
        images[key] = {"path": str(path),
                       "image": describe_image(encoder, path, None),
                       "fused": describe_image(encoder, path, texts[str(text_lengths[0])]["text"]["raw_text"])}
    return {"texts": texts, "images": images,
            "note": "sequence_tokens is the effective unpadded length after GME templating, "
                    "including the expanded image-pad run; visual_tokens is grid_t*grid_h*grid_w/4"}


# --------------------------------------------------------------- timing helpers
def timed(call, repeat: int, batch: int) -> dict:
    call()  # warm up: excludes compile and first-run allocation
    start = time.perf_counter()
    for _ in range(repeat):
        call()
    elapsed = (time.perf_counter() - start) / repeat * 1000
    return {"ms_per_call": round(elapsed, 2), "ms_per_embedding": round(elapsed / batch, 2)}


def run_cpp(arguments: list[str], timeout: int = 3600) -> dict[str, dict]:
    command = (f'set +u; source "{SETUP}" >/dev/null 2>&1; "{BIN}" --package "{PKG}" '
               + " ".join(arguments))
    result = subprocess.run(["bash", "-c", command], capture_output=True, text=True, timeout=timeout)
    rows = {}
    for match in re.finditer(
        r"mode=(\w+) batch=(\d+) streams=(\S+) batch_mode=(\S+) avg_ms=([\d.eE+-]+) ms_per_embedding=([\d.eE+-]+)",
        result.stdout,
    ):
        rows[match.group(1)] = {
            "ms_per_call": round(float(match.group(5)), 2),
            "ms_per_embedding": round(float(match.group(6)), 2),
        }
    if not rows:
        rows["error"] = {"stderr": result.stderr.strip()[-300:] or "no timing produced"}
    return rows


def available_devices(devices: list[str]) -> list[str]:
    present = GmeOpenVINO(PKG, device="CPU", precision="int8").core.available_devices
    usable = [device for device in devices if device in present]
    for device in devices:
        if device not in present:
            print(f"[skip] device {device} unavailable (found {present})")
    return usable


# --------------------------------------------------------------------- suites
def suite_latency(config: dict, workloads: dict) -> dict:
    """N=1 across mode x precision x device: the base comparison table."""
    length = str(config["text_lengths"][0])
    text = workloads["texts"][length]["text"]["raw_text"]
    query = workloads["texts"][length]["query"]["raw_text"]
    size = config["image_sizes"][0]
    path = workloads["images"][size]["path"]
    results: dict = {"text_length": length, "image_size": size, "batch": 1, "cells": {}}
    for device in config["devices"]:
        for precision in config["precisions"]:
            key = f"{device}/{precision}"
            if "python" in config["backends"]:
                encoder = GmeOpenVINO(PKG, device=device, precision=precision)
                results["cells"][f"python/{key}"] = {
                    "text": timed(lambda: encoder.get_text_embeddings(
                        [text], is_query=False, show_progress_bar=False), config["repeat"], 1),
                    "query": timed(lambda: encoder.get_text_embeddings(
                        [query], instruction=INSTRUCTION, show_progress_bar=False), config["repeat"], 1),
                    "image": timed(lambda: encoder.get_image_embeddings(
                        [path], show_progress_bar=False), config["repeat"], 1),
                    "fused": timed(lambda: encoder.get_fused_embeddings(
                        [text], [path], show_progress_bar=False), config["repeat"], 1),
                }
                report_cell(f"python/{key}", results["cells"][f"python/{key}"])
            if "cpp" in config["backends"]:
                results["cells"][f"cpp/{key}"] = run_cpp([
                    "--device", device, "--precision", precision, "--benchmark",
                    "--benchmark-modes", "text,query,image,fused", "--repeat", str(config["repeat"]),
                    "--instruction", f'"{INSTRUCTION}"', "--image", f'"{path}"', f'"{text}"',
                ])
                report_cell(f"cpp/{key}", results["cells"][f"cpp/{key}"])
    return results


def suite_text_lengths(config: dict, workloads: dict) -> dict:
    """text/query at each exact final token length."""
    results: dict = {"batch": 1, "cells": {}}
    for length in config["text_lengths"]:
        entry = workloads["texts"][str(length)]
        text, query = entry["text"]["raw_text"], entry["query"]["raw_text"]
        for device in config["devices"]:
            for precision in config["precisions"]:
                key = f"{device}/{precision}/{length}tok"
                if "python" in config["backends"]:
                    encoder = GmeOpenVINO(PKG, device=device, precision=precision)
                    cell = {
                        "text": timed(lambda: encoder.get_text_embeddings(
                            [text], is_query=False, show_progress_bar=False), config["repeat"], 1),
                        "query": timed(lambda: encoder.get_text_embeddings(
                            [query], instruction=INSTRUCTION, show_progress_bar=False), config["repeat"], 1),
                    }
                    results["cells"][f"python/{key}"] = cell
                    report_cell(f"python/{key}", cell)
                if "cpp" in config["backends"]:
                    cell = run_cpp(["--device", device, "--precision", precision, "--benchmark",
                                    "--benchmark-modes", "text", "--repeat", str(config["repeat"]), f'"{text}"'])
                    cell.update({f"query_{name}": value for name, value in run_cpp([
                        "--device", device, "--precision", precision, "--benchmark",
                        "--benchmark-modes", "query", "--repeat", str(config["repeat"]),
                        "--instruction", f'"{INSTRUCTION}"', f'"{query}"']).items()})
                    results["cells"][f"cpp/{key}"] = cell
                    report_cell(f"cpp/{key}", cell)
    return results


def suite_image_sizes(config: dict, workloads: dict) -> dict:
    """image/fused across the image-size ladder."""
    text = workloads["texts"][str(config["text_lengths"][0])]["text"]["raw_text"]
    results: dict = {"batch": 1, "cells": {}}
    for size in config["image_sizes"] + (["logo"] if config["include_logo"] else []):
        entry = workloads["images"][size] if size in workloads["images"] else None
        if entry is None:  # the logo keys by its own pixel size
            entry = next(value for key, value in workloads["images"].items()
                         if value["image"]["file"] == LOGO.name)
        path = entry["path"]
        tokens = entry["image"]["visual_tokens"]
        for device in config["devices"]:
            for precision in config["precisions"]:
                key = f"{device}/{precision}/{size}({tokens}vt)"
                if "python" in config["backends"]:
                    encoder = GmeOpenVINO(PKG, device=device, precision=precision)
                    cell = {
                        "image": timed(lambda: encoder.get_image_embeddings(
                            [path], show_progress_bar=False), config["repeat"], 1),
                        "fused": timed(lambda: encoder.get_fused_embeddings(
                            [text], [path], show_progress_bar=False), config["repeat"], 1),
                    }
                    results["cells"][f"python/{key}"] = cell
                    report_cell(f"python/{key}", cell)
                if "cpp" in config["backends"]:
                    cell = run_cpp(["--device", device, "--precision", precision, "--benchmark",
                                    "--benchmark-modes", "image,fused", "--repeat", str(config["repeat"]),
                                    "--image", f'"{path}"', f'"{text}"'])
                    results["cells"][f"cpp/{key}"] = cell
                    report_cell(f"cpp/{key}", cell)
    return results


def suite_batch(config: dict, workloads: dict) -> dict:
    """N sweep x batch_mode x streams, for text and image.

    The two knobs interact with the device, not with each other: on CPU one request
    already fills every core, so throughput comes from `streams`; on GPU the
    per-submission cost dominates and only a padded call amortizes it. That is the
    whole reason `batch_mode="auto"` exists, and this suite is what re-measures it.
    """
    length = str(config["batch_text_length"])
    text = workloads["texts"][length]["text"]["raw_text"]
    size = config["image_sizes"][0]
    path = workloads["images"][size]["path"]
    precision = config["batch_precision"]
    results: dict = {"text_length": length, "image_size": size, "precision": precision, "cells": {}}
    for device in config["devices"]:
        for batch_mode in config["batch_modes"]:
            for streams in config["streams"]:
                tag = f"{device}/{batch_mode}/streams={streams}"
                if "python" in config["backends"]:
                    encoder = GmeOpenVINO(PKG, device=device, precision=precision,
                                          streams=streams, batch_mode=batch_mode)
                    for count in config["batch_sizes"]:
                        cell = {"text": timed(lambda: encoder.get_text_embeddings(
                            [text] * count, is_query=False, batch_size=None,
                            show_progress_bar=False), config["repeat"], count)}
                        if count in config["image_batch_sizes"]:
                            cell["image"] = timed(lambda: encoder.get_image_embeddings(
                                [path] * count, batch_size=None, show_progress_bar=False),
                                config["repeat"], count)
                        results["cells"][f"python/{tag}/N={count}"] = cell
                        report_cell(f"python/{tag}/N={count}", cell)
                if "cpp" in config["backends"]:
                    for count in config["batch_sizes"]:
                        modes = "text,image" if count in config["image_batch_sizes"] else "text"
                        cell = run_cpp([
                            "--device", device, "--precision", precision, "--benchmark",
                            "--benchmark-modes", modes, "--repeat", str(config["repeat"]),
                            "--batch", str(count), "--batch-mode", batch_mode,
                            "--streams", "0" if streams == "auto" else str(streams),
                            "--batch-size", "0",  # one chunk: measure the batch, not the chunking
                            "--image", f'"{path}"', f'"{text}"',
                        ])
                        results["cells"][f"cpp/{tag}/N={count}"] = cell
                        report_cell(f"cpp/{tag}/N={count}", cell)
    return results


def report_cell(label: str, cell: dict) -> None:
    if "error" in cell:
        print(f"  {label:44s} ERROR {cell['error']['stderr'][:80]}")
        return
    parts = " ".join(
        f"{mode}={value['ms_per_call']:.1f}ms({value['ms_per_embedding']:.1f}/emb)"
        for mode, value in cell.items() if isinstance(value, dict) and "ms_per_call" in value
    )
    print(f"  {label:44s} {parts}")


# ----------------------------------------------------------------------- driver
SUITES = {"latency": suite_latency, "text-lengths": suite_text_lengths,
          "image-sizes": suite_image_sizes, "batch": suite_batch}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", default=",".join(SUITES),
                        help=f"comma-separated subset of {list(SUITES)} (default: all)")
    parser.add_argument("--devices", default=",".join(DEVICES))
    parser.add_argument("--precisions", default=",".join(PRECISIONS))
    parser.add_argument("--backends", default="python,cpp")
    parser.add_argument("--repeat", type=int, default=5,
                        help="warm calls per cell (default 5; use >=20 for anything quoted as a threshold)")
    parser.add_argument("--text-lengths", default=",".join(str(value) for value in TEXT_LENGTHS),
                        help="effective token counts after GME templating")
    parser.add_argument("--image-sizes", default=",".join(IMAGE_SIZES),
                        help="WxH, ideally multiples of 28 so the processor does not rescale")
    parser.add_argument("--include-logo", action="store_true",
                        help="also measure gme_logo.png (1260 visual tokens, the worst case)")
    parser.add_argument("--batch-sizes", default=",".join(str(value) for value in BATCH_SIZES))
    parser.add_argument("--image-batch-sizes", default=",".join(str(value) for value in IMAGE_BATCH_SIZES),
                        help="batch suite: which N also run the image mode (images are far more expensive)")
    parser.add_argument("--batch-modes", default="per_sample,padded")
    parser.add_argument("--streams", default="1,auto")
    parser.add_argument("--batch-precision", default="int8",
                        help="batch suite precision; int8 keeps the sweep affordable")
    parser.add_argument("--batch-text-length", type=int, default=128)
    parser.add_argument("--output", type=Path, default=ART / "benchmark_matrix.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repeat < 1:
        raise ValueError("--repeat must be positive")
    suites = [name.strip() for name in args.suite.split(",") if name.strip()]
    unknown = [name for name in suites if name not in SUITES]
    if unknown:
        raise ValueError(f"unknown suite(s) {unknown}; choose from {list(SUITES)}")

    def ints(value: str) -> list[int]:
        return [int(item) for item in value.split(",") if item.strip()]

    config = {
        "devices": available_devices([d.strip() for d in args.devices.split(",") if d.strip()]),
        "precisions": [p.strip() for p in args.precisions.split(",") if p.strip()],
        "backends": [b.strip() for b in args.backends.split(",") if b.strip()],
        "repeat": args.repeat,
        "text_lengths": ints(args.text_lengths),
        "image_sizes": [s.strip() for s in args.image_sizes.split(",") if s.strip()],
        "include_logo": args.include_logo,
        "batch_sizes": ints(args.batch_sizes),
        "image_batch_sizes": ints(args.image_batch_sizes),
        "batch_modes": [m.strip() for m in args.batch_modes.split(",") if m.strip()],
        "streams": [1 if s.strip() == "1" else s.strip() for s in args.streams.split(",") if s.strip()],
        "batch_precision": args.batch_precision,
        "batch_text_length": args.batch_text_length,
    }
    if not config["devices"]:
        raise SystemExit("no requested device is available")
    if config["batch_text_length"] not in config["text_lengths"]:
        config["text_lengths"] = sorted(set(config["text_lengths"] + [config["batch_text_length"]]))

    print("resolving workloads (tokenizing filler text, writing benchmark images) ...")
    workloads = build_workloads(config["text_lengths"], config["image_sizes"], config["include_logo"])
    for length, entry in workloads["texts"].items():
        print(f"  text {length:>5} tokens <- {entry['text']['raw_text_chars']} chars "
              f"(query: {entry['query']['raw_text_chars']} chars + instruction)")
    for size, entry in workloads["images"].items():
        image = entry["image"]
        print(f"  image {size:>10} -> grid {image['image_grid_thw']}, {image['raw_patch_rows']} patch rows, "
              f"{image['visual_tokens']} visual tokens, image-mode sequence {image['sequence_tokens']} tokens")

    report = {
        "unit": "ms, warm, end-to-end per call (tokenize/preprocess + inference), excludes load/compile",
        "repeat": config["repeat"],
        "config": {key: value for key, value in config.items()},
        "workloads": workloads,
        "suites": {},
    }
    for name in suites:
        print(f"\n=== suite: {name} ===")
        report["suites"][name] = SUITES[name](config, workloads)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
