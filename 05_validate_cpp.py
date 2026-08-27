#!/usr/bin/env python
"""Validate the C++ gme_embed runtime against the PyTorch golden embeddings.

Drives the compiled C++ CLI for text/query/image/fused across devices and
precisions, then compares its dumped embeddings (cosine) to the golden vectors.
Text/query/language are expected near-exact at fp32; the image path carries the
accepted OpenCV-vs-PIL resize delta.
"""
import argparse
import json
import shlex
import struct
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE_MODEL = HERE.parent / "benchmark" / "models" / "gme-Qwen2-VL-2B-Instruct"
DEFAULT_PACKAGE = HERE / "ov_models" / "gme-qwen2-vl-2b-instruct-openvino"
DEFAULT_ARTIFACTS = HERE / "artifacts"
DEFAULT_CPP_BINARY = HERE / "cpp" / "build" / "gme_embed"
DEFAULT_OPENVINO_INSTALL = HERE.parent / "benchmark" / "openvino" / "install"


def run(args, out_path: Path, binary: Path, setup: Path, timeout: int) -> None:
    out_path.unlink(missing_ok=True)
    command = [str(binary), *[str(value) for value in args], "--out", str(out_path)]
    shell = (
        f"set +u; source {shlex.quote(str(setup))} >/dev/null 2>&1; "
        f"exec {shlex.join(command)}"
    )
    r = subprocess.run(["bash", "-c", shell], capture_output=True, text=True, timeout=timeout)
    if r.returncode:
        raise RuntimeError(f"gme_embed failed with exit {r.returncode}.\nSTDERR:\n{r.stderr[-1500:]}")
    if not Path(out_path).exists():
        raise RuntimeError(f"gme_embed produced no output.\nSTDERR:\n{r.stderr[-1500:]}")


def load_dump(path):
    with open(path, "rb") as f:
        magic = f.read(4)
        assert magic == b"GEMB", magic
        rows, dim = struct.unpack("<ii", f.read(8))
        data = np.frombuffer(f.read(), dtype=np.float32).reshape(rows, dim)
    return data.astype(np.float64)


def cos_rows(a, b):
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE,
                        help="exported OpenVINO package directory")
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS,
                        help="directory containing golden vectors and report output")
    parser.add_argument("--source-model", type=Path, default=DEFAULT_SOURCE_MODEL,
                        help="used for the default logo image path")
    parser.add_argument("--image", type=Path,
                        help="image used for image/fused checks (default: <source-model>/images/gme_logo.png)")
    parser.add_argument("--cpp-binary", type=Path, default=DEFAULT_CPP_BINARY,
                        help="gme_embed executable")
    parser.add_argument("--openvino-install", type=Path, default=DEFAULT_OPENVINO_INSTALL,
                        help="OpenVINO installation containing setupvars.sh")
    parser.add_argument("--devices", default="CPU,GPU",
                        help="comma-separated devices to test (default: CPU,GPU)")
    parser.add_argument("--precisions", default="fp32,int8_causal_vopt,int4_causal_vopt,int4_gs64_causal_vopt",
                        help="comma-separated precisions to test")
    parser.add_argument("--timeout", type=int, default=600,
                        help="per-inference subprocess timeout in seconds")
    parser.add_argument("--output", type=Path,
                        help="validation JSON path (default: <artifacts>/cpp_validation.json)")
    args = parser.parse_args()

    package = args.package.expanduser().resolve()
    artifacts = args.artifacts.expanduser().resolve()
    source_model = args.source_model.expanduser().resolve()
    image = (args.image or source_model / "images" / "gme_logo.png").expanduser().resolve()
    binary = args.cpp_binary.expanduser().resolve()
    setup = args.openvino_install.expanduser().resolve() / "setupvars.sh"
    if not package.is_dir():
        raise SystemExit(f"package directory not found: {package}")
    if not binary.is_file():
        raise SystemExit(f"C++ binary not found: {binary}")
    if not setup.is_file():
        raise SystemExit(f"OpenVINO setupvars.sh not found: {setup}")
    if not image.is_file():
        raise SystemExit(f"validation image not found: {image}")
    artifacts.mkdir(parents=True, exist_ok=True)
    required = [artifacts / "golden_meta.json", artifacts / "golden_text.npy",
                artifacts / "golden_query.npy", artifacts / "golden_image.npy",
                artifacts / "golden_fused.npy"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"missing golden files: {missing}; run 01_reference.py and 04_validate_python.py first")

    meta = json.loads((artifacts / "golden_meta.json").read_text())
    texts = meta["texts"]
    instr = meta["instruction"]
    golden = {k: np.load(artifacts / f"golden_{k}.npy").astype(np.float64) for k in ["text", "query", "image", "fused"]}
    tmp = artifacts / "cpp_dumps"
    tmp.mkdir(parents=True, exist_ok=True)

    devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    precisions = [item.strip() for item in args.precisions.split(",") if item.strip()]
    report = {}
    for dev in devices:
        report[dev] = {}
        for prec in precisions:
            base = ["--package", str(package), "--device", dev, "--precision", prec]
            try:
                run(base + ["--mode", "text", "--is-query", "0", *texts], tmp / f"text_{dev}_{prec}.bin", binary, setup, args.timeout)
                run(base + ["--mode", "text", "--is-query", "1", "--instruction", instr, texts[0]], tmp / f"query_{dev}_{prec}.bin", binary, setup, args.timeout)
                run(base + ["--mode", "image", "--image", str(image)], tmp / f"image_{dev}_{prec}.bin", binary, setup, args.timeout)
                run(base + ["--mode", "fused", "--image", str(image), texts[0]], tmp / f"fused_{dev}_{prec}.bin", binary, setup, args.timeout)
                row = {
                    "text_cos_min": round(float(cos_rows(load_dump(tmp / f"text_{dev}_{prec}.bin"), golden["text"]).min()), 6),
                    "query_cos": round(float(cos_rows(load_dump(tmp / f"query_{dev}_{prec}.bin"), golden["query"][:1])[0]), 6),
                    "image_cos": round(float(cos_rows(load_dump(tmp / f"image_{dev}_{prec}.bin"), golden["image"])[0]), 6),
                    "fused_cos": round(float(cos_rows(load_dump(tmp / f"fused_{dev}_{prec}.bin"), golden["fused"])[0]), 6),
                }
                report[dev][prec] = row
                print(f"[{dev}] {prec:5s} text={row['text_cos_min']:.6f} query={row['query_cos']:.6f} "
                      f"image={row['image_cos']:.6f} fused={row['fused_cos']:.6f}")
            except Exception as e:
                report[dev][prec] = {"error": str(e)[:400]}
                print(f"[{dev}] {prec}: ERROR {str(e)[:200]}")

    output = (args.output or artifacts / "cpp_validation.json").expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[cpp] wrote {output}")


if __name__ == "__main__":
    main()
