#!/usr/bin/env python
"""Validate the single-IR image/fused path against dual IR and GME goldens."""
from __future__ import annotations

import json
import struct
import subprocess
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from gme_openvino import GmeOpenVINO
from gme_openvino_monolithic import GmeOpenVINOMonolithic

HERE = Path(__file__).resolve().parent
PACKAGE = HERE / "ov_models" / "gme-qwen2-vl-2b-instruct-openvino"
ART = HERE / "artifacts"
BIN = HERE / "cpp" / "build" / "gme_embed"
SETUP = HERE.parent / "benchmark" / "openvino" / "install" / "setupvars.sh"


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = left.reshape(-1).astype(np.float64)
    right = right.reshape(-1).astype(np.float64)
    return float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)))


def load_dump(path: Path) -> np.ndarray:
    with path.open("rb") as output:
        if output.read(4) != b"GEMB":
            raise RuntimeError(f"invalid C++ embedding dump: {path}")
        rows, dimension = struct.unpack("<ii", output.read(8))
        return np.frombuffer(output.read(), dtype=np.float32).reshape(rows, dimension)


def run_cpp(
    device: str,
    precision: str,
    monolithic: bool,
    mode: str,
    image: str,
    text: str,
    output: Path,
) -> np.ndarray:
    args = [
        str(BIN), "--package", str(PACKAGE), "--device", device, "--precision", precision,
        "--mode", mode, "--image", image, "--out", str(output),
    ]
    if monolithic:
        args.append("--monolithic")
    if mode == "fused":
        args.append(text)
    shell = f'set +u; source "{SETUP}" >/dev/null 2>&1; ' + " ".join(f'"{arg}"' for arg in args)
    completed = subprocess.run(["bash", "-c", shell], capture_output=True, text=True, timeout=600)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr[-1200:])
    return load_dump(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--precisions",
        help="comma-separated monolithic precisions; defaults to all variants declared in the package",
    )
    parser.add_argument(
        "--devices",
        default="CPU",
        help="comma-separated devices to execute; default CPU",
    )
    parser.add_argument(
        "--workloads",
        default="logo_448_approximately_256_visual_tokens",
        help="comma-separated workloads; default is the 448x448 image only",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="run large model compilation/inference; omitted by default to prevent accidental resource pressure",
    )
    parser.add_argument(
        "--include-large-logo",
        action="store_true",
        help="allow the 1260-visual-token logo workload, which may exhaust iGPU resources for monolithic inference",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.execute:
        print(
            "[validate] no inference was run. Pass --execute explicitly; "
            "use --include-large-logo only when sufficient resources are available."
        )
        return
    meta = json.loads((ART / "golden_meta.json").read_text())
    package_config = json.loads((PACKAGE / "gme_openvino_config.json").read_text())
    available_precisions = package_config.get("monolithic_models", {"fp32": "gme_monolithic_fp32.xml"})
    precisions = args.precisions.split(",") if args.precisions else list(available_precisions)
    unknown = [precision for precision in precisions if precision not in available_precisions]
    if unknown:
        raise ValueError(f"unknown monolithic precision(s): {unknown}; available={list(available_precisions)}")
    devices = [device.strip() for device in args.devices.split(",") if device.strip()]
    invalid_devices = [device for device in devices if device not in ("CPU", "GPU")]
    if not devices or invalid_devices:
        raise ValueError("--devices must be a comma-separated subset of CPU,GPU")
    goldens = {
        "image": np.load(ART / "golden_image.npy"),
        "fused": np.load(ART / "golden_fused.npy"),
    }
    available = GmeOpenVINO(PACKAGE, precision="fp32").core.available_devices
    output_dir = ART / "monolithic_cpp_dumps"
    output_dir.mkdir(exist_ok=True)
    small_image = ART / "monolithic_test_448.png"
    if not small_image.exists():
        Image.open(meta["image"]).convert("RGB").resize((448, 448)).save(small_image)
    workloads = {
        "golden_logo_1260_visual_tokens": {
            "image": meta["image"],
            "goldens": goldens,
            "must_run": "CPU",
        },
        "logo_448_approximately_256_visual_tokens": {
            "image": str(small_image),
            "goldens": None,
            "must_run": "CPU,GPU",
        },
    }
    requested_workloads = [name.strip() for name in args.workloads.split(",") if name.strip()]
    unknown_workloads = [name for name in requested_workloads if name not in workloads]
    if not requested_workloads or unknown_workloads:
        raise ValueError(f"unknown --workloads value(s): {unknown_workloads}; available={list(workloads)}")
    if "golden_logo_1260_visual_tokens" in requested_workloads and not args.include_large_logo:
        raise ValueError("the 1260-token logo workload requires --include-large-logo")
    report: dict[str, dict] = {"precisions": precisions, "workloads": {}}

    for device in devices:
        if device not in available:
            continue
        report["workloads"][device] = {}
        for precision in precisions:
            dual = GmeOpenVINO(PACKAGE, device=device, precision=precision)
            mono = GmeOpenVINOMonolithic(PACKAGE, device=device, precision=precision)
            report["workloads"][device][precision] = {}
            for workload_name in requested_workloads:
                workload = workloads[workload_name]
                row = {"python": {}, "cpp": {}}
                report["workloads"][device][precision][workload_name] = row
                for mode in ("image", "fused"):
                    try:
                        if mode == "image":
                            dual_value = dual.get_image_embeddings(workload["image"])
                            mono_value = mono.get_image_embeddings(workload["image"])
                        else:
                            dual_value = dual.get_fused_embeddings(meta["texts"][0], workload["image"])
                            mono_value = mono.get_fused_embeddings(meta["texts"][0], workload["image"])
                        values = {"cos_vs_dual": round(cosine(mono_value, dual_value), 8)}
                        if workload["goldens"] is not None:
                            values["cos_vs_golden"] = round(cosine(mono_value, workload["goldens"][mode]), 8)
                        row["python"][mode] = values
                    except Exception as error:
                        row["python"][mode] = {"error": repr(error)[:500]}

                    try:
                        cpp_dual = run_cpp(
                            device, precision, False, mode, workload["image"], meta["texts"][0],
                            output_dir / f"dual_{device}_{precision}_{workload_name}_{mode}.bin",
                        )
                        cpp_mono = run_cpp(
                            device, precision, True, mode, workload["image"], meta["texts"][0],
                            output_dir / f"mono_{device}_{precision}_{workload_name}_{mode}.bin",
                        )
                        values = {"cos_vs_dual": round(cosine(cpp_mono, cpp_dual), 8)}
                        if workload["goldens"] is not None:
                            values["cos_vs_golden"] = round(cosine(cpp_mono, workload["goldens"][mode]), 8)
                        row["cpp"][mode] = values
                    except Exception as error:
                        row["cpp"][mode] = {"error": repr(error)[:500]}

                    py_result = row["python"][mode]
                    cpp_result = row["cpp"][mode]
                    print(f"[{device}] {precision} {workload_name} {mode}: python={py_result} cpp={cpp_result}")

                    # FP32 must remain numerically identical where the workload
                    # is mandatory. Quantized variants are measured, not asserted,
                    # because their independent NNCF graphs need not equal dual IR.
                    mandatory = precision == "fp32" and device in workload["must_run"].split(",")
                    if mandatory:
                        for backend, values in (("Python", py_result), ("C++", cpp_result)):
                            if "error" in values or values["cos_vs_dual"] < 0.9999:
                                raise RuntimeError(f"{backend} monolithic {device}/{workload_name}/{mode} failed: {values}")
    (ART / "monolithic_validation_full.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"[validate] wrote {ART / 'monolithic_validation_full.json'}")


if __name__ == "__main__":
    main()