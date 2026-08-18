#!/usr/bin/env python3
"""End-to-end smoke test for Qwen3-VL-Embedding-2B on GPU through openvino_genai.

This is the path that reported `CL_OUT_OF_RESOURCES` /
`FullyConnectedCompressed could not set scales primitive attribute` once the GME plugin
patches were installed. Every text / image / fused workload is run in one process, so a
regression that only shows up after the first shape change is still caught.

With --save the 2048-dim embeddings are written to an npz; with --compare a previous npz is
diffed element-wise, which is how the patched build is checked against stock.
"""
from __future__ import annotations

import argparse
import sys
import traceback

import numpy as np
import openvino as ov
import openvino_genai
from PIL import Image

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from paths import MODELS_ROOT  # noqa: E402

IMAGE_SIZES = [(64, 64), (224, 224), (448, 448), (672, 672), (960, 480)]
TEXT_LENGTHS = [32, 128, 512, 1024]


def gradient_image(width: int, height: int) -> ov.Tensor:
    x = np.linspace(0, 255, width, dtype=np.float32)[None, :]
    y = np.linspace(0, 255, height, dtype=np.float32)[:, None]
    pixels = np.stack([np.broadcast_to(x, (height, width)),
                       np.broadcast_to(y, (height, width)),
                       (x + y) / 2], axis=-1).astype(np.uint8)
    return ov.Tensor(np.array(Image.fromarray(pixels, mode="RGB"))[None])


def as_vector(result) -> np.ndarray:
    tensor = result.embeddings
    return np.asarray(tensor.data, dtype=np.float32).reshape(tuple(tensor.shape)).reshape(-1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--precisions", nargs="*", default=["INT8", "INT4"])
    ap.add_argument("--device", default="GPU")
    ap.add_argument("--save", default=None)
    ap.add_argument("--compare", default=None)
    args = ap.parse_args()

    print(f"openvino {ov.get_version()}  device {args.device}")
    vectors: dict[str, np.ndarray] = {}
    failures = 0

    for precision in args.precisions:
        model_dir = MODELS_ROOT / f"Qwen3-VL-Embedding-2B-{precision}"
        tokenizer = openvino_genai.Tokenizer(str(model_dir))
        pipeline = openvino_genai.EmbeddingPipeline(str(model_dir), args.device)
        # "word " repeated is close enough to one token per word for a length sweep.
        texts = {n: " ".join(["word"] * n) for n in TEXT_LENGTHS}
        for n, text in texts.items():
            token_count = len(tokenizer.encode(text).input_ids.data[0])
            for mode, size in [("text", None)] + [("image", s) for s in IMAGE_SIZES] + \
                              [("fused", s) for s in IMAGE_SIZES]:
                if mode != "text" and n != TEXT_LENGTHS[0]:
                    continue  # image/fused only need one text length
                key = f"{precision}/{mode}/{n if mode == 'text' else f'{size[0]}x{size[1]}'}"
                try:
                    if mode == "text":
                        out = pipeline.embed(text)
                    elif mode == "image":
                        out = pipeline.embed(images=[gradient_image(*size)])
                    else:
                        out = pipeline.embed(text, images=[gradient_image(*size)])
                    vec = as_vector(out)
                    vectors[key] = vec
                    ok = np.isfinite(vec).all()
                    failures += not ok
                    print(f"  {key:<28} OK dim={vec.size} tokens={token_count} "
                          f"norm={np.linalg.norm(vec):.4f} finite={ok}")
                except Exception as exc:  # noqa: BLE001
                    failures += 1
                    print(f"  {key:<28} FAIL {str(exc).strip().splitlines()[0]}")
                    traceback.print_exc(limit=1)

    if args.save:
        np.savez(args.save, **vectors)
        print(f"saved {len(vectors)} embeddings to {args.save}")
    if args.compare:
        base = np.load(args.compare)
        worst = 0.0
        for key in sorted(set(base.files) & set(vectors)):
            d = float(np.abs(base[key] - vectors[key]).max())
            worst = max(worst, d)
            if d != 0.0:
                print(f"  DIFF {key}: max_abs={d:.6g}")
        missing = sorted(set(base.files) ^ set(vectors))
        if missing:
            print(f"  keys only on one side: {missing}")
        print(f"vs baseline: worst max_abs_diff={worst:.6g} "
              f"{'IDENTICAL' if worst == 0.0 and not missing else 'DIFFERS'}")

    print(f"{'ALL OK' if failures == 0 else f'{failures} FAILURES'}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
