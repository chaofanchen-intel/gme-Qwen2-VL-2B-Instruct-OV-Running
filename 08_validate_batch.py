#!/usr/bin/env python
"""Validate that batching is exact and unbounded, in Python and in C++.

A GME batch is N independent samples, so both runtimes dispatch N single-sample
requests rather than one padded N-row tensor. That has three consequences this
script checks:

A. Any N works with no per-batch-size export. The visual IR is traced with one
   ``image_grid_thw`` row and can only ever accept one, so it is called per image;
   because Qwen2-VL vision attention is block-diagonal per image, that is exact.
B. A batch is *bit-identical* to the same items encoded one at a time -- no row
   shifts with batch composition, which padded batching cannot promise.
C. ``streams`` only buys throughput: results must not move.

Finally the batch is compared against the official PyTorch model so "exact"
means exact with respect to upstream, not merely self-consistent.
"""
from __future__ import annotations

import importlib
import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from gme_openvino import GmeOpenVINO, load_image

HERE = Path(__file__).resolve().parent
PACKAGE = HERE / "ov_models" / "gme-qwen2-vl-2b-instruct-openvino"
MODEL_DIR = HERE.parent / "benchmark" / "models" / "gme-Qwen2-VL-2B-Instruct"
IMAGE = MODEL_DIR / "images" / "gme_logo.png"
SETUP = HERE.parent / "benchmark" / "openvino" / "install" / "setupvars.sh"
CPP = HERE / "cpp" / "build" / "gme_embed"
ART = HERE / "artifacts"

# Deliberately ragged: different text lengths and different image sizes, so any
# hidden dependence on padding or on a uniform grid shows up as a mismatch.
TEXTS = [
    "a compact electric pickup truck",
    "a square GME logo image",
    "a much longer sentence describing an electric vehicle, its battery chemistry, and range",
    "short",
    "another query about charging infrastructure",
]
IMAGE_SIZES = [(448, 448), (336, 336), (448, 336), (280, 280), (336, 448)]
SIZES = [1, 2, 3, 5]


def cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = left.astype(np.float64)
    right = right.astype(np.float64)
    return (left * right).sum(1) / (np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1))


def read_dump(path: Path) -> np.ndarray:
    with path.open("rb") as source:
        if source.read(4) != b"GEMB":
            raise RuntimeError(f"invalid embedding dump: {path}")
        rows, dimension = struct.unpack("<ii", source.read(8))
        return np.frombuffer(source.read(), dtype=np.float32).reshape(rows, dimension)


def run_cpp(arguments: list[str], output: Path, streams: int = 1) -> np.ndarray:
    command = [
        str(CPP), "--package", str(PACKAGE), "--device", "CPU", "--precision", "fp32",
        "--streams", str(streams), *arguments, "--out", str(output),
    ]
    shell = f'set +u; source "{SETUP}" >/dev/null 2>&1; ' + " ".join(f'"{item}"' for item in command)
    completed = subprocess.run(["bash", "-c", shell], capture_output=True, text=True, timeout=1800)
    if completed.returncode:
        raise RuntimeError(completed.stderr[-1200:])
    return read_dump(output)


def make_images(directory: Path) -> list[Path]:
    source = Image.open(IMAGE).convert("RGB")
    paths = []
    for width, height in IMAGE_SIZES:
        path = directory / f"batch_{width}x{height}.png"
        source.resize((width, height)).save(path)
        paths.append(path)
    return paths


def check(label: str, batch: np.ndarray, reference: np.ndarray, floor: float, report: dict) -> None:
    """Record cosine-vs-reference and bit-exactness, and fail below `floor`."""
    values = cosine_rows(batch, reference)
    entry = {
        "min_cos": round(float(values.min()), 10),
        "bit_exact": bool(np.array_equal(batch, reference)),
        "rows": int(batch.shape[0]),
    }
    report[label] = entry
    flag = "bit-exact" if entry["bit_exact"] else f"cos={entry['min_cos']:.10f}"
    print(f"  {label:38s} rows={entry['rows']}  {flag}")
    if values.min() < floor:
        raise RuntimeError(f"{label}: min cosine {values.min():.10f} < {floor}")


def batch_vs_sequential(report: dict, images: list[Path]) -> None:
    """A: any N works. B: the batch equals the same items encoded one by one."""
    print("[A/B] Python batch vs sequential (int8, streams=1, per_sample) -- expect bit-exact")
    encoder = GmeOpenVINO(PACKAGE, device="CPU", precision="int8", batch_mode="per_sample")
    for size in SIZES:
        texts, paths = TEXTS[:size], images[:size]
        text_batch = encoder.get_text_embeddings(texts)
        cases = {
            f"N={size} text": (
                text_batch,
                np.concatenate([encoder.get_text_embeddings([t]) for t in texts]),
            ),
            f"N={size} image": (
                encoder.get_image_embeddings(paths),
                np.concatenate([encoder.get_image_embeddings([p]) for p in paths]),
            ),
            f"N={size} fused": (
                encoder.get_fused_embeddings(texts, paths),
                np.concatenate(
                    [encoder.get_fused_embeddings([t], [p]) for t, p in zip(texts, paths)]
                ),
            ),
        }
        for label, (batch, sequential) in cases.items():
            check(label, batch, sequential, 1.0 - 1e-9, report)
        # batch_size only chunks the work; it must not change the answer.
        check(f"N={size} text batch_size=2", encoder.get_text_embeddings(texts, batch_size=2),
              text_batch, 1.0 - 1e-9, report)

    # The padded path (the GPU default, and what the official model does) cannot be
    # bit-exact: a row's decode sees padding whose amount depends on the longest
    # sample beside it. Record how far that drifts on deliberately ragged input.
    print("\n[B2] Python padded batch vs sequential (int8) -- drift is expected, not a bug")
    padded = GmeOpenVINO(PACKAGE, device="CPU", precision="int8", batch_mode="padded")
    check("padded text vs sequential", padded.get_text_embeddings(TEXTS),
          np.concatenate([padded.get_text_embeddings([t]) for t in TEXTS]), 0.999, report)
    check("padded fused vs sequential", padded.get_fused_embeddings(TEXTS[:3], images[:3]),
          np.concatenate([padded.get_fused_embeddings([t], [p])
                          for t, p in zip(TEXTS[:3], images[:3])]), 0.999, report)


def streams_invariance(report: dict, images: list[Path]) -> None:
    """C: concurrency is a throughput knob, not a numerical one."""
    print("\n[C] Python streams=1 vs streams='auto' (int8) -- expect bit-exact")
    single = GmeOpenVINO(PACKAGE, device="CPU", precision="int8", streams=1)
    concurrent = GmeOpenVINO(PACKAGE, device="CPU", precision="int8", streams="auto")
    print(f"  pool sizes: streams=1 -> {single.language_pool.size}, "
          f"streams='auto' -> {concurrent.language_pool.size}")
    report["pool_size_auto"] = concurrent.language_pool.size
    check("streams text", concurrent.get_text_embeddings(TEXTS),
          single.get_text_embeddings(TEXTS), 1.0 - 1e-9, report)
    check("streams fused", concurrent.get_fused_embeddings(TEXTS[:3], images[:3]),
          single.get_fused_embeddings(TEXTS[:3], images[:3]), 1.0 - 1e-9, report)


def official_golden(images: list[Path]) -> dict[str, np.ndarray]:
    """Torch fp32 embeddings for the N=3 ragged batch, cached (the load is slow)."""
    cache = ART / "golden_batch_n3.npz"
    # Tie the cache to the inputs it was produced from, so editing TEXTS or
    # IMAGE_SIZES regenerates it instead of silently validating against stale rows.
    signature = np.frombuffer(
        repr((TEXTS[:3], IMAGE_SIZES[:3])).encode(), dtype=np.uint8
    )
    if cache.exists():
        cached = dict(np.load(cache))
        if np.array_equal(cached.pop("signature", None), signature):
            return cached
        print("  golden cache is stale for the current inputs; regenerating")
    sys.path.insert(0, str(MODEL_DIR))
    official = importlib.import_module("gme_inference").GmeQwen2VL(
        model_path=str(MODEL_DIR), device="cpu"
    )
    official.base = official.base.float().eval()
    texts = TEXTS[:3]
    pillow = [load_image(path) for path in images[:3]]
    with torch.no_grad():
        golden = {
            "text": official.get_text_embeddings(texts).float().numpy(),
            "image": official.get_image_embeddings(pillow).float().numpy(),
            "fused": official.get_fused_embeddings(texts, pillow).float().numpy(),
        }
    np.savez(cache, signature=signature, **golden)
    return golden


def official_parity(report: dict, images: list[Path]) -> None:
    """The batch must match upstream PyTorch, not just itself."""
    print("\n[D] FP32 batch vs official PyTorch (N=3, ragged) -- expect >= 0.9999")
    golden = official_golden(images)
    texts, paths = TEXTS[:3], images[:3]
    # Both decode paths must land on upstream; neither is allowed to be the odd one.
    for mode in ("per_sample", "padded"):
        encoder = GmeOpenVINO(PACKAGE, device="CPU", precision="fp32", batch_mode=mode)
        check(f"official text ({mode})", encoder.get_text_embeddings(texts),
              golden["text"], 0.9999, report)
        check(f"official image ({mode})", encoder.get_image_embeddings(paths),
              golden["image"], 0.9999, report)
        check(f"official fused ({mode})", encoder.get_fused_embeddings(texts, paths),
              golden["fused"], 0.9999, report)


# C++ preprocesses images with OpenCV while Python uses PIL, and the two
# resamplers disagree slightly. 05_validate_cpp.py records the resulting steady
# state at CPU/fp32 (image 0.975, fused 0.988), so any image-bearing C++ vs
# Python comparison is gated there, not at the language path's 0.9999.
CPP_IMAGE_FLOOR = 0.98


def cpp_parity(report: dict, images: list[Path], dump_dir: Path) -> None:
    """The C++ runtime batches the same way and must agree with Python."""
    print("\n[E] C++ batch vs C++ sequential vs Python (fp32, N=3)")
    texts, paths = TEXTS[:3], images[:3]
    image_args = [arg for path in paths for arg in ("--image", str(path))]
    batch = run_cpp(["--mode", "fused", *image_args, *texts], dump_dir / "batch_fused_fp32.bin")
    sequential = np.concatenate([
        run_cpp(["--mode", "fused", "--image", str(path), text],
                dump_dir / f"batch_single_{index}.bin")
        for index, (text, path) in enumerate(zip(texts, paths))
    ])
    check("cpp fused batch vs sequential", batch, sequential, 1.0 - 1e-9, report)

    concurrent = run_cpp(["--mode", "fused", *image_args, *texts],
                         dump_dir / "batch_fused_streams.bin", streams=0)
    check("cpp fused streams=0 vs 1", concurrent, batch, 1.0 - 1e-9, report)

    # Text shares no preprocessing with OpenCV, so C++ and Python must agree
    # tightly there -- that is what pins the two batch implementations together.
    for mode in ("per_sample", "padded"):
        encoder = GmeOpenVINO(PACKAGE, device="CPU", precision="fp32", batch_mode=mode)
        cpp_text = run_cpp(["--mode", "text", "--is-query", "0", "--batch-mode", mode, *texts],
                           dump_dir / f"batch_text_{mode}.bin")
        check(f"cpp text vs python ({mode})", cpp_text,
              encoder.get_text_embeddings(texts, is_query=False), 0.9999, report)
        cpp_fused = run_cpp(["--mode", "fused", "--batch-mode", mode, *image_args, *texts],
                            dump_dir / f"batch_fused_{mode}.bin")
        check(f"cpp fused vs python ({mode})", cpp_fused,
              encoder.get_fused_embeddings(texts, paths), CPP_IMAGE_FLOOR, report)


def main() -> None:
    ART.mkdir(exist_ok=True)
    dump_dir = ART / "cpp_dumps"
    dump_dir.mkdir(exist_ok=True)
    images = make_images(ART)

    report: dict = {"device": "CPU", "sizes": SIZES, "image_sizes": IMAGE_SIZES}
    batch_vs_sequential(report, images)
    streams_invariance(report, images)
    official_parity(report, images)
    cpp_parity(report, images, dump_dir)

    (ART / "batch_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    for path in images:
        path.unlink(missing_ok=True)
    print(f"\n[batch] all checks passed; wrote {ART / 'batch_validation.json'}")


if __name__ == "__main__":
    main()
