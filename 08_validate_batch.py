#!/usr/bin/env python
"""Validate dual-IR batch results against sequential inference on safe inputs."""
from __future__ import annotations

import json
import struct
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from gme_openvino import GmeOpenVINO

HERE = Path(__file__).resolve().parent
PACKAGE = HERE / "ov_models" / "gme-qwen2-vl-2b-instruct-openvino"
IMAGE = HERE.parent / "benchmark" / "models" / "gme-Qwen2-VL-2B-Instruct" / "images" / "gme_logo.png"
SETUP = HERE.parent / "benchmark" / "openvino" / "install" / "setupvars.sh"
CPP = HERE / "cpp" / "build" / "gme_embed"
ART = HERE / "artifacts"

TEXTS = ["a compact electric pickup truck", "a square GME logo image"]


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


def run_cpp(arguments: list[str], output: Path) -> np.ndarray:
    command = [str(CPP), "--package", str(PACKAGE), "--device", "CPU", "--precision", "fp32", *arguments, "--out", str(output)]
    shell = f'set +u; source "{SETUP}" >/dev/null 2>&1; ' + " ".join(f'"{item}"' for item in command)
    completed = subprocess.run(["bash", "-c", shell], capture_output=True, text=True, timeout=600)
    if completed.returncode:
        raise RuntimeError(completed.stderr[-1200:])
    return read_dump(output)


def main() -> None:
    ART.mkdir(exist_ok=True)
    small_image = ART / "batch_validation_448.png"
    Image.open(IMAGE).convert("RGB").resize((448, 448)).save(small_image)
    images = [small_image, small_image]
    encoder = GmeOpenVINO(PACKAGE, device="CPU", precision="fp32")

    text_batch = encoder.get_text_embeddings(TEXTS, batch_size=2)
    text_single = np.concatenate([encoder.get_text_embeddings([text]) for text in TEXTS])
    image_batch = encoder.get_image_embeddings(images, batch_size=2)
    image_single = np.concatenate([encoder.get_image_embeddings([image]) for image in images])
    fused_batch = encoder.get_fused_embeddings(TEXTS, images, batch_size=2)
    fused_single = np.concatenate(
        [encoder.get_fused_embeddings([text], [image]) for text, image in zip(TEXTS, images)], axis=0
    )

    dump_dir = ART / "cpp_dumps"
    dump_dir.mkdir(exist_ok=True)
    cpp_batch = run_cpp(
        ["--mode", "fused", "--image", str(small_image), "--image", str(small_image), *TEXTS],
        dump_dir / "batch_fused_fp32.bin",
    )
    cpp_single = np.concatenate(
        [
            run_cpp(["--mode", "fused", "--image", str(small_image), text], dump_dir / f"batch_single_{index}.bin")
            for index, text in enumerate(TEXTS)
        ],
        axis=0,
    )

    report = {
        "device": "CPU",
        "precision": "fp32",
        "batch_size": 2,
        "image_size": [448, 448],
        "python": {
            "text_cos_vs_sequential": cosine_rows(text_batch, text_single).tolist(),
            "image_cos_vs_sequential": cosine_rows(image_batch, image_single).tolist(),
            "fused_cos_vs_sequential": cosine_rows(fused_batch, fused_single).tolist(),
        },
        "cpp": {"fused_cos_vs_sequential": cosine_rows(cpp_batch, cpp_single).tolist()},
    }
    for values in [*report["python"].values(), report["cpp"]["fused_cos_vs_sequential"]]:
        if min(values) < 0.9999:
            raise RuntimeError(f"batch output differs from sequential output: {values}")
    (ART / "batch_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    small_image.unlink(missing_ok=True)
    print(json.dumps(report, indent=2))
    print(f"[batch] wrote {ART / 'batch_validation.json'}")


if __name__ == "__main__":
    main()