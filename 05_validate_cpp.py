#!/usr/bin/env python
"""Validate the C++ gme_embed runtime against the PyTorch golden embeddings.

Drives the compiled C++ CLI for text/query/image/fused across devices and
precisions, then compares its dumped embeddings (cosine) to the golden vectors.
Text/query/language are expected near-exact at fp32; the image path carries the
accepted OpenCV-vs-PIL resize delta.
"""
import json
import struct
import subprocess
from pathlib import Path

import numpy as np
from gme_paths import ARTIFACTS as ART, CPP_BINARY as BIN, LOGO_IMAGE, OPENVINO_SETUP as SETUP, PACKAGE as PKG

HERE = Path(__file__).resolve().parent
IMAGE = str(LOGO_IMAGE)


def run(args, out_path):
    cmd = f'set +u; source "{SETUP}" >/dev/null 2>&1; "{BIN}" ' + " ".join(
        f'"{a}"' for a in args) + f' --out "{out_path}"'
    r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=600)
    if not Path(out_path).exists():
        raise RuntimeError(f"gme_embed produced no output.\nSTDERR:\n{r.stderr[-1500:]}")
    return r


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
    meta = json.loads((ART / "golden_meta.json").read_text())
    texts = meta["texts"]
    instr = meta["instruction"]
    golden = {k: np.load(ART / f"golden_{k}.npy").astype(np.float64) for k in ["text", "query", "image", "fused"]}
    tmp = ART / "cpp_dumps"; tmp.mkdir(exist_ok=True)

    devices = ["CPU", "GPU"]
    precisions = ["fp32", "int8", "int4"]
    report = {}
    for dev in devices:
        report[dev] = {}
        for prec in precisions:
            base = ["--package", str(PKG), "--device", dev, "--precision", prec]
            try:
                run(base + ["--mode", "text", "--is-query", "0", *texts], tmp / f"text_{dev}_{prec}.bin")
                run(base + ["--mode", "text", "--is-query", "1", "--instruction", instr, texts[0]], tmp / f"query_{dev}_{prec}.bin")
                run(base + ["--mode", "image", "--image", IMAGE], tmp / f"image_{dev}_{prec}.bin")
                run(base + ["--mode", "fused", "--image", IMAGE, texts[0]], tmp / f"fused_{dev}_{prec}.bin")
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

    (ART / "cpp_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"[cpp] wrote {ART / 'cpp_validation.json'}")


if __name__ == "__main__":
    main()
