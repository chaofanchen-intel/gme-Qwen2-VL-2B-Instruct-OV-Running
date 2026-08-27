#!/usr/bin/env python
"""Validate that batching is exact and unbounded, in Python and in C++.

A GME batch is N independent samples, so both runtimes dispatch N single-sample
requests rather than one padded N-row tensor. Checks:

A. Any N works with no per-batch-size export. The visual IR is traced with one
   ``image_grid_thw`` row and can only ever accept one, so it is called per image;
   because Qwen2-VL vision attention is block-diagonal per image, that is exact.
B. A batch is *bit-identical* to the same items encoded one at a time -- no row
   shifts with batch composition, which padded batching cannot promise.
C. ``streams`` only buys throughput: results must not move.
D. The batch matches the official PyTorch model, so "exact" means exact with
   respect to upstream and not merely self-consistent.
E. The C++ runtime batches the same way and agrees with Python.
F. The official API surface behaves as upstream does: ``embed`` is the
   single-batch primitive, ``batch_size`` chunking is invisible in the result,
   an all-``None`` image list means text-only, and the aliases agree.
G. Inputs past ``max_length`` truncate identically in both runtimes. That is not
   free: the tokenizer IR only truncates because 03_finalize.py converts it with
   ``max_length``, and without that C++ silently kept 3021 tokens where Python
   and the official model keep 1800.
H. The claim behind A, measured rather than asserted: the B=1-traced visual IR
   rejects a second ``image_grid_thw`` row, and a deliberately B=2-traced IR is
   compared against two per-image calls. Skipped unless the b2 diagnostic IR is
   present (see experiments/probe_visual_batch.py for the full sweep).
I. Distinct samples that share a tensor shape stay distinct. This is the
   regression for the AsyncInferQueue bug described in ``_RequestPool``; every
   other check here uses ragged input, which cannot expose it.
"""
from __future__ import annotations

import argparse
import importlib
import json
import shlex
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from gme_openvino import GmeOpenVINO, load_image

HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE_MODEL = HERE.parent / "benchmark" / "models" / "gme-Qwen2-VL-2B-Instruct"
DEFAULT_PACKAGE = HERE / "ov_models" / "gme-qwen2-vl-2b-instruct-openvino"
DEFAULT_ARTIFACTS = HERE / "artifacts"
DEFAULT_CPP_BINARY = HERE / "cpp" / "build" / "gme_embed"
DEFAULT_OPENVINO_INSTALL = HERE.parent / "benchmark" / "openvino" / "install"


@dataclass(frozen=True)
class BatchPaths:
    package: Path
    artifacts: Path
    source_model: Path
    cpp_binary: Path
    openvino_setup: Path
    image: Path

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


def run_cpp(
    arguments: list[str],
    output: Path,
    paths: BatchPaths,
    streams: int = 1,
    precision: str = "fp32",
) -> np.ndarray:
    command = [
        str(paths.cpp_binary), "--package", str(paths.package), "--device", "CPU", "--precision", precision,
        "--streams", str(streams), *arguments, "--out", str(output),
    ]
    shell = (
        f"set +u; source {shlex.quote(str(paths.openvino_setup))} >/dev/null 2>&1; "
        f"exec {shlex.join(command)}"
    )
    completed = subprocess.run(["bash", "-c", shell], capture_output=True, text=True, timeout=1800)
    if completed.returncode:
        raise RuntimeError(completed.stderr[-1200:])
    return read_dump(output)


def make_images(directory: Path, source_image: Path) -> list[Path]:
    source = Image.open(source_image).convert("RGB")
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


def batch_vs_sequential(report: dict, images: list[Path], paths: BatchPaths) -> None:
    """A: any N works. B: the batch equals the same items encoded one by one."""
    print("[A/B] Python batch vs sequential (int8_vnomask, streams=1, per_sample) -- expect bit-exact")
    encoder = GmeOpenVINO(paths.package, device="CPU", precision="int8_vnomask", batch_mode="per_sample")
    for size in SIZES:
        texts, image_paths = TEXTS[:size], images[:size]
        text_batch = encoder.get_text_embeddings(texts)
        cases = {
            f"N={size} text": (
                text_batch,
                np.concatenate([encoder.get_text_embeddings([t]) for t in texts]),
            ),
            f"N={size} image": (
                encoder.get_image_embeddings(image_paths),
                np.concatenate([encoder.get_image_embeddings([p]) for p in image_paths]),
            ),
            f"N={size} fused": (
                encoder.get_fused_embeddings(texts, image_paths),
                np.concatenate(
                    [encoder.get_fused_embeddings([t], [p]) for t, p in zip(texts, image_paths)]
                ),
            ),
        }
        for label, (batch, sequential) in cases.items():
            check(label, batch, sequential, 1.0 - 1e-9, report)
        # batch_size only chunks the work; it must not change the answer.
        check(f"N={size} text batch_size=2",
              encoder.get_text_embeddings(texts, batch_size=2, show_progress_bar=False),
              text_batch, 1.0 - 1e-9, report)

    # The padded path (the GPU default, and what the official model does) cannot be
    # bit-exact: a row's decode sees padding whose amount depends on the longest
    # sample beside it. Record how far that drifts on deliberately ragged input.
    print("\n[B2] Python padded batch vs sequential (int8_vnomask) -- drift is expected, not a bug")
    padded = GmeOpenVINO(paths.package, device="CPU", precision="int8_vnomask", batch_mode="padded")
    check("padded text vs sequential", padded.get_text_embeddings(TEXTS),
          np.concatenate([padded.get_text_embeddings([t]) for t in TEXTS]), 0.999, report)
    check("padded fused vs sequential", padded.get_fused_embeddings(TEXTS[:3], images[:3]),
          np.concatenate([padded.get_fused_embeddings([t], [p])
                          for t, p in zip(TEXTS[:3], images[:3])]), 0.999, report)


def streams_invariance(report: dict, images: list[Path], paths: BatchPaths) -> None:
    """C: concurrency is a throughput knob, not a numerical one."""
    print("\n[C] Python streams=1 vs streams='auto' (int8_vnomask) -- expect bit-exact")
    single = GmeOpenVINO(paths.package, device="CPU", precision="int8_vnomask", streams=1)
    concurrent = GmeOpenVINO(paths.package, device="CPU", precision="int8_vnomask", streams="auto")
    print(f"  pool sizes: streams=1 -> {single.language_pool.size}, "
          f"streams='auto' -> {concurrent.language_pool.size}")
    report["pool_size_auto"] = concurrent.language_pool.size
    check("streams text", concurrent.get_text_embeddings(TEXTS),
          single.get_text_embeddings(TEXTS), 1.0 - 1e-9, report)
    check("streams fused", concurrent.get_fused_embeddings(TEXTS[:3], images[:3]),
          single.get_fused_embeddings(TEXTS[:3], images[:3]), 1.0 - 1e-9, report)


def official_golden(images: list[Path], paths: BatchPaths) -> dict[str, np.ndarray]:
    """Torch fp32 embeddings for the N=3 ragged batch, cached (the load is slow)."""
    cache = paths.artifacts / "golden_batch_n3.npz"
    # Tie the cache to the inputs it was produced from, so editing TEXTS or
    # IMAGE_SIZES regenerates it instead of silently validating against stale rows.
    signature = np.frombuffer(
        repr((TEXTS[:3], IMAGE_SIZES[:3], str(paths.image))).encode(), dtype=np.uint8
    )
    if cache.exists():
        cached = dict(np.load(cache))
        if np.array_equal(cached.pop("signature", None), signature):
            return cached
        print("  golden cache is stale for the current inputs; regenerating")
    sys.path.insert(0, str(paths.source_model))
    official = importlib.import_module("gme_inference").GmeQwen2VL(
        model_path=str(paths.source_model), device="cpu"
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


def official_parity(report: dict, images: list[Path], paths: BatchPaths) -> None:
    """The batch must match upstream PyTorch, not just itself."""
    print("\n[D] FP32 batch vs official PyTorch (N=3, ragged) -- expect >= 0.9999")
    golden = official_golden(images, paths)
    texts, image_paths = TEXTS[:3], images[:3]
    # Both decode paths must land on upstream; neither is allowed to be the odd one.
    for mode in ("per_sample", "padded"):
        encoder = GmeOpenVINO(paths.package, device="CPU", precision="fp32", batch_mode=mode)
        check(f"official text ({mode})", encoder.get_text_embeddings(texts),
              golden["text"], 0.9999, report)
        check(f"official image ({mode})", encoder.get_image_embeddings(image_paths),
              golden["image"], 0.9999, report)
        check(f"official fused ({mode})", encoder.get_fused_embeddings(texts, image_paths),
              golden["fused"], 0.9999, report)


# C++ preprocesses images with OpenCV while Python uses PIL, and the two
# resamplers disagree slightly. 05_validate_cpp.py records the resulting steady
# state at CPU/fp32 (image 0.975, fused 0.988), so any image-bearing C++ vs
# Python comparison is gated there, not at the language path's 0.9999.
CPP_IMAGE_FLOOR = 0.98


def cpp_parity(report: dict, images: list[Path], dump_dir: Path, paths: BatchPaths) -> None:
    """The C++ runtime batches the same way and must agree with Python."""
    print("\n[E] C++ batch vs C++ sequential vs Python (fp32, N=3)")
    texts, image_paths = TEXTS[:3], images[:3]
    image_args = [arg for path in image_paths for arg in ("--image", str(path))]
    batch = run_cpp(["--mode", "fused", *image_args, *texts], dump_dir / "batch_fused_fp32.bin", paths)
    sequential = np.concatenate([
        run_cpp(["--mode", "fused", "--image", str(path), text],
            dump_dir / f"batch_single_{index}.bin", paths)
        for index, (text, path) in enumerate(zip(texts, image_paths))
    ])
    check("cpp fused batch vs sequential", batch, sequential, 1.0 - 1e-9, report)

    concurrent = run_cpp(["--mode", "fused", *image_args, *texts],
                         dump_dir / "batch_fused_streams.bin", paths, streams=0)
    check("cpp fused streams=0 vs 1", concurrent, batch, 1.0 - 1e-9, report)

    # --batch-size chunks the C++ batch the way official GME's batch_size does; under
    # per_sample it must be invisible, including when it does not divide the input.
    for size in ("0", "1", "2"):
        check(f"cpp fused --batch-size {size}",
              run_cpp(["--mode", "fused", "--batch-mode", "per_sample", "--batch-size", size,
                       *image_args, *texts], dump_dir / f"batch_fused_bs{size}.bin", paths),
              batch, 1.0 - 1e-9, report)

    # Text shares no preprocessing with OpenCV, so C++ and Python must agree
    # tightly there -- that is what pins the two batch implementations together.
    for mode in ("per_sample", "padded"):
        encoder = GmeOpenVINO(paths.package, device="CPU", precision="fp32", batch_mode=mode)
        cpp_text = run_cpp(["--mode", "text", "--is-query", "0", "--batch-mode", mode, *texts],
                   dump_dir / f"batch_text_{mode}.bin", paths)
        check(f"cpp text vs python ({mode})", cpp_text,
              encoder.get_text_embeddings(texts, is_query=False), 0.9999, report)
        cpp_fused = run_cpp(["--mode", "fused", "--batch-mode", mode, *image_args, *texts],
                    dump_dir / f"batch_fused_{mode}.bin", paths)
        check(f"cpp fused vs python ({mode})", cpp_fused,
              encoder.get_fused_embeddings(texts, image_paths), CPP_IMAGE_FLOOR, report)


def api_parity(report: dict, images: list[Path], paths: BatchPaths) -> None:
    """F: the official API surface, including the parts only upstream's loop shows."""
    print("\n[F] official API surface (int8_vnomask, per_sample)")
    encoder = GmeOpenVINO(paths.package, device="CPU", precision="int8_vnomask", batch_mode="per_sample")
    reference = encoder.get_text_embeddings(TEXTS, show_progress_bar=False)

    # embed() is the single-batch primitive the chunked entry points call, so the
    # default (32 > 5 texts, i.e. one chunk) must be exactly embed().
    check("embed() == get_text_embeddings", encoder.embed(texts=TEXTS), reference, 1.0 - 1e-9, report)
    # Upstream's default is batch_size=32; sweep across and past it.
    for size in (1, 2, 5, 32):
        check(f"batch_size={size} chunking",
              encoder.get_text_embeddings(TEXTS, batch_size=size, show_progress_bar=False),
              reference, 1.0 - 1e-9, report)
    check("batch_size=None (one chunk)",
          encoder.get_text_embeddings(TEXTS, batch_size=None, show_progress_bar=False),
          reference, 1.0 - 1e-9, report)
    # Aliases must not quietly change the prompt. encode/encode_queries are queries
    # with no instruction, which resolves to the default instruction -- same as text.
    check("encode() alias", encoder.encode(TEXTS, prompt_name="ignored", show_progress_bar=False),
          reference, 1.0 - 1e-9, report)
    check("encode_queries() alias", encoder.encode_queries(TEXTS, show_progress_bar=False),
          reference, 1.0 - 1e-9, report)
    check("encode_corpus() == is_query=False",
          encoder.encode_corpus([{"text": text} for text in TEXTS], show_progress_bar=False),
          encoder.get_text_embeddings(TEXTS, is_query=False, show_progress_bar=False),
          1.0 - 1e-9, report)
    # Official signals "text-only" with a list of None, and its fused path zips
    # texts against images so a longer text list is truncated, not rejected.
    check("images=[None]*N -> text path", encoder.embed(texts=TEXTS, images=[None] * len(TEXTS)),
          reference, 1.0 - 1e-9, report)
    check("fused, texts longer than images",
          encoder.embed(texts=TEXTS, images=images[:2]),
          encoder.get_fused_embeddings(TEXTS[:2], images[:2], show_progress_bar=False),
          1.0 - 1e-9, report)
    for label, call in (
        ("mixed None images", lambda: encoder.embed(texts=TEXTS, images=[None, images[0], None, None, None])),
        ("fewer texts than images", lambda: encoder.embed(texts=TEXTS[:1], images=images[:2])),
        ("empty input", lambda: encoder.get_text_embeddings([])),
    ):
        try:
            call()
        except ValueError as error:
            print(f"  {label:38s} rejected: {str(error).splitlines()[0][:60]}")
            report[f"rejects {label}"] = True
        else:
            raise RuntimeError(f"{label} should have raised ValueError")


def truncation_parity(report: dict, images: list[Path], dump_dir: Path, paths: BatchPaths) -> None:
    """G: past max_length, both runtimes must cut at the same token."""
    print("\n[G] max_length truncation parity (fp32)")
    encoder = GmeOpenVINO(paths.package, device="CPU", precision="fp32")
    long_text = "word " * 3000
    tokens = encoder.processor.tokenizer(
        [encoder._prompt(long_text, has_image=False, instruction="You are a helpful assistant.")],
        padding=True, truncation=True, max_length=encoder.max_length, return_tensors="np",
    )["input_ids"].shape[1]
    report["truncated_tokens"] = int(tokens)
    print(f"  a {len(long_text.split())}-word input tokenizes to {tokens} tokens (max_length={encoder.max_length})")
    if tokens != encoder.max_length:
        raise RuntimeError(f"expected truncation to {encoder.max_length}, got {tokens}")
    python = encoder.get_text_embeddings([long_text], is_query=False, show_progress_bar=False)
    cpp = run_cpp(["--mode", "text", "--is-query", "0", long_text], dump_dir / "batch_text_truncated.bin", paths)
    # Text shares no image preprocessing, so this is the language path's own floor.
    check("cpp vs python, >max_length text", cpp, python, 0.9999, report)


def same_shape_rows(report: dict, paths: BatchPaths) -> None:
    """I: distinct samples that share a tensor shape must not collapse onto each other.

    Regression for a real bug: the request pool used ``ov.AsyncInferQueue``, which
    returned a stale output for a later job, duplicating row 0 into row 1. It only
    showed when two feeds had the *same* shape -- two 448x448 images, two texts of
    equal token length -- which is the common case, and it was intermittent. The
    earlier ragged-input checks could not catch it, because ragged feeds differ in
    shape. This one is deliberately non-ragged.
    """
    print("\n[I] distinct samples sharing a tensor shape (int8_vnomask, per_sample)")
    encoder = GmeOpenVINO(paths.package, device="CPU", precision="int8_vnomask", batch_mode="per_sample")

    # Equal token length, different content -> identical feed shapes.
    same_length = ["red car", "big dog", "hot sun", "old man"]
    lengths = {
        int(encoder.processor.tokenizer(
            encoder._prompt(text, has_image=False, instruction="You are a helpful assistant."),
            return_tensors="np")["input_ids"].shape[1])
        for text in same_length
    }
    if len(lengths) != 1:
        raise RuntimeError(f"the same-shape texts must tokenize to one length, got {lengths}")
    report["same_shape_token_length"] = lengths.pop()

    # Sequential first, then batched, then batched again: the bug depended on how the
    # compiled model had been used before, so check both orders.
    sequential = np.concatenate([
        encoder.get_text_embeddings([text], show_progress_bar=False) for text in same_length
    ])
    check("same-length text batch", encoder.get_text_embeddings(same_length, show_progress_bar=False),
          sequential, 1.0 - 1e-9, report)
    fresh = GmeOpenVINO(paths.package, device="CPU", precision="int8_vnomask", batch_mode="per_sample")
    check("same-length text, batch first", fresh.get_text_embeddings(same_length, show_progress_bar=False),
          sequential, 1.0 - 1e-9, report)

    # Rows must also be distinct from each other -- a pool that returns row 0 for
    # every job would still be "bit-exact" against a sequential run of one sample.
    batch = encoder.get_text_embeddings(same_length, show_progress_bar=False)
    collisions = [(i, j) for i in range(len(batch)) for j in range(i + 1, len(batch))
                  if np.array_equal(batch[i], batch[j])]
    report["same_shape_row_collisions"] = len(collisions)
    print(f"  {'distinct rows':38s} rows={len(batch)}  collisions={len(collisions)}")
    if collisions:
        raise RuntimeError(f"batch rows collapsed onto each other: {collisions}")

    # Same test on the visual path: two different images that land on one grid.
    source = Image.open(paths.image).convert("RGB")
    pair = [source.resize((448, 448)), source.crop((0, 0, 2000, 600)).resize((336, 336))]
    grids = encoder.processor(
        text=["<|vision_start|><|image_pad|><|vision_end|>"] * 2,
        images=[load_image(image) for image in pair], padding=True, return_tensors="np",
    )["image_grid_thw"]
    if len({tuple(row) for row in grids.tolist()}) != 1:
        raise RuntimeError(f"the same-shape images must share a grid, got {grids.tolist()}")
    report["same_shape_image_grid"] = grids[0].tolist()
    image_sequential = np.concatenate([
        encoder.get_image_embeddings([image], show_progress_bar=False) for image in pair
    ])
    check("same-grid image batch", encoder.get_image_embeddings(pair, show_progress_bar=False),
          image_sequential, 1.0 - 1e-9, report)
    if np.array_equal(image_sequential[0], image_sequential[1]):
        raise RuntimeError("the two probe images produce identical embeddings; the check is vacuous")


def visual_any_n(report: dict, images: list[Path], paths: BatchPaths) -> None:
    """H: why the visual IR is driven per image, measured on the IRs themselves."""
    print("\n[H] visual IR: dynamic image count")
    import openvino as ov

    encoder = GmeOpenVINO(paths.package, device="CPU", precision="int8_vnomask")
    inputs = encoder.processor(
        text=["<|vision_start|><|image_pad|><|vision_end|>"] * 2,
        images=[load_image(path) for path in images[:2]],
        padding=True, return_tensors="np",
    )
    grid = inputs["image_grid_thw"]
    bounds = np.concatenate([[0], np.cumsum(np.prod(grid, axis=1, dtype=np.int64))])
    feeds = [
        {"pixel_values": inputs["pixel_values"][bounds[i]:bounds[i + 1]], "image_grid_thw": grid[i:i + 1]}
        for i in range(grid.shape[0])
    ]
    per_image = np.concatenate(encoder.visual_pool.run(feeds), axis=0)

    # A separate compiled model for the negative test: a throwing inference is not
    # something the rest of this function should have to reason about.
    probe = ov.Core().compile_model(str(paths.package / "gme_visual_int8.xml"), "CPU")
    try:  # the traced B=1 graph must refuse 2 rows; that is the whole reason for per-image
        probe({"pixel_values": inputs["pixel_values"], "image_grid_thw": grid})
    except Exception as error:  # noqa: BLE001 - the rejection is the result
        report["b1_ir_rejects_two_rows"] = True
        print(f"  B=1 IR with 2 grid rows      rejected ({str(error).strip().splitlines()[-1][:60]})")
    else:
        report["b1_ir_rejects_two_rows"] = False
        print("  B=1 IR with 2 grid rows      ACCEPTED (visual IR is natively dynamic here)")

    b2_path = paths.package / "gme_visual_b2_int8.xml"
    if not b2_path.exists():
        report["b2_ir_available"] = False
        print(f"  {b2_path.name} absent; skipping the batched-IR comparison")
        return
    report["b2_ir_available"] = True
    batched = ov.Core().compile_model(str(b2_path), "CPU")
    rows = np.array(batched({"pixel_values": inputs["pixel_values"], "image_grid_thw": grid})[batched.output(0)],
                    copy=True)
    identical = bool(np.array_equal(rows, per_image.astype(rows.dtype)))
    report["b2_ir_bit_identical_to_per_image"] = identical
    print(f"  B=2-traced IR vs per-image   {'bit-exact' if identical else 'DIFFERS'} "
          f"(max abs diff {np.abs(rows - per_image).max():.3e}, rows={rows.shape[0]})")
    # Not a hard failure either way: the batched IR is a diagnostic artifact, and its
    # agreement is resolution-dependent (experiments/probe_visual_batch.py --only q2
    # sweeps that). The shipped path is the per-image one, which section D pins to
    # official PyTorch.
    if not identical:
        print("     ^ expected for resolutions it was not traced on; see probe_visual_batch.py")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE,
                        help="exported OpenVINO package directory")
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS,
                        help="directory for generated batch images, dumps, cache, and report")
    parser.add_argument("--source-model", type=Path, default=DEFAULT_SOURCE_MODEL,
                        help="GME checkpoint used when regenerating the official cache")
    parser.add_argument("--image", type=Path,
                        help="source image for generated probes (default: <source-model>/images/gme_logo.png)")
    parser.add_argument("--cpp-binary", type=Path, default=DEFAULT_CPP_BINARY,
                        help="gme_embed executable")
    parser.add_argument("--openvino-install", type=Path, default=DEFAULT_OPENVINO_INSTALL,
                        help="OpenVINO installation containing setupvars.sh")
    parser.add_argument("--output", type=Path,
                        help="report path (default: <artifacts>/batch_validation.json)")
    args = parser.parse_args()

    package = args.package.expanduser().resolve()
    artifacts = args.artifacts.expanduser().resolve()
    source_model = args.source_model.expanduser().resolve()
    image = (args.image or source_model / "images" / "gme_logo.png").expanduser().resolve()
    cpp_binary = args.cpp_binary.expanduser().resolve()
    openvino_setup = args.openvino_install.expanduser().resolve() / "setupvars.sh"
    if not package.is_dir():
        raise SystemExit(f"package directory not found: {package}")
    if not source_model.is_dir():
        raise SystemExit(f"source model directory not found: {source_model}")
    if not image.is_file():
        raise SystemExit(f"probe image not found: {image}")
    if not cpp_binary.is_file():
        raise SystemExit(f"C++ binary not found: {cpp_binary}")
    if not openvino_setup.is_file():
        raise SystemExit(f"OpenVINO setupvars.sh not found: {openvino_setup}")

    paths = BatchPaths(package, artifacts, source_model, cpp_binary, openvino_setup, image)
    artifacts.mkdir(parents=True, exist_ok=True)
    dump_dir = artifacts / "cpp_dumps"
    dump_dir.mkdir(parents=True, exist_ok=True)
    images = make_images(artifacts, image)

    report: dict = {"device": "CPU", "sizes": SIZES, "image_sizes": IMAGE_SIZES}
    batch_vs_sequential(report, images, paths)
    streams_invariance(report, images, paths)
    official_parity(report, images, paths)
    cpp_parity(report, images, dump_dir, paths)
    api_parity(report, images, paths)
    truncation_parity(report, images, dump_dir, paths)
    visual_any_n(report, images, paths)
    same_shape_rows(report, paths)

    output = (args.output or artifacts / "batch_validation.json").expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    for path in images:
        path.unlink(missing_ok=True)
    print(f"\n[batch] all checks passed; wrote {output}")


if __name__ == "__main__":
    main()
