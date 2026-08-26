#!/usr/bin/env python3
"""End-to-end GME image/fused latency, split into the stages that can be optimized.

``profile_gme_int4_gpu.py`` covers the text/query path. This is its image/fused
counterpart, and it exists because the single end-to-end number those modes produce
is not actionable: it mixes CPU-side image preprocessing, the visual IR, and the
language IR, and the three scale with completely different things.

Reported per size:

* ``preprocess`` -- PIL load/resize plus the Qwen2-VL processor, on CPU. Scales with
  pixel count.
* ``visual`` -- the visual IR alone, fed the processor's output. Scales with the raw
  **patch** count ``L = t*h*w``, and its attention is O(L^2).
* ``language`` -- the language IR alone, fed the visual embeddings. Scales with the
  **merged token** count ``K = L/4`` plus the text tokens.
* ``end_to_end`` -- the public ``get_image_embeddings`` / ``get_fused_embeddings``
  call. The gap between it and the sum of the stages is harness overhead.

Sizes are multiples of 28 so that GME's ``fetch_image`` and the Qwen2-VL processor
both leave them alone and ``grid = (H/14, W/14)`` is exactly predictable; see
PERFORMANCE_REPORT.md S4.2. 448x448 is a hard floor -- anything smaller is upscaled
to the same 256 tokens.

    python profile_gme_image.py --precision int4_causal --sizes 448x448,672x672
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from paths import GME_PACKAGE, GME_ROOT

HERE = Path(__file__).resolve().parent
INSTRUCTION = "Find an image that matches the given text."
# 32-token filler for fused mode, so the text side is a fixed small constant and any
# change across sizes is attributable to the image.
FUSED_TEXT = " ".join(["a"] * 24)

DEFAULT_SIZES = "448x448,644x476,672x672,896x896,1288x728"


def gradient_image(height: int, width: int):
    """Deterministic RGB gradient. Content does not affect latency; fixing it keeps
    runs comparable and avoids the denormals a constant image could produce."""
    from PIL import Image

    y, x = np.mgrid[0:height, 0:width]
    pixels = np.stack([
        x * 255 // max(width - 1, 1),
        y * 255 // max(height - 1, 1),
        (x + y) * 255 // max(width + height - 2, 1),
    ], axis=-1)
    return Image.fromarray(pixels.astype(np.uint8))


def percentile(values: list[float], q: float = 50) -> float:
    return round(float(np.percentile(values, q)), 3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", default="int4_causal")
    ap.add_argument("--device", default="GPU")
    ap.add_argument("--sizes", default=DEFAULT_SIZES,
                    help="comma-separated HxW list; multiples of 28 avoid processor rescaling")
    ap.add_argument("--modes", default="image,fused")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--repeat", type=int, default=6)
    ap.add_argument("--streams", default="1")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    sys.path.insert(0, str(GME_ROOT))
    from gme_openvino import GmeOpenVINO

    streams: int | str = "auto" if args.streams == "auto" else int(args.streams)
    encoder = GmeOpenVINO(GME_PACKAGE, device=args.device, precision=args.precision,
                          streams=streams)
    config = encoder.config["precisions"][args.precision]
    print(f"precision={args.precision} language={config['language']} visual={config['visual']} "
          f"device={args.device} batch_mode={encoder.batch_mode}")

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    sizes = []
    for item in args.sizes.split(","):
        height, width = (int(v) for v in item.strip().lower().split("x"))
        sizes.append((height, width))

    # One request, reused: matches how the runtime pool behaves and keeps the
    # per-stage numbers free of request-creation cost.
    visual_request = encoder.visual.create_infer_request()
    rows = []
    for height, width in sizes:
        image = gradient_image(height, width)
        for mode in modes:
            text = FUSED_TEXT if mode == "fused" else None
            prompt = encoder._prompt(text, has_image=True,
                                     instruction=encoder._instruction(INSTRUCTION, True))

            def preprocess():
                return encoder.processor(text=[prompt], images=[image], padding=True,
                                         truncation=True, max_length=encoder.max_length,
                                         return_tensors="np")

            inputs = preprocess()
            grid = inputs["image_grid_thw"]
            patches = int(np.prod(grid, axis=1).sum())
            tokens = patches // 4
            seq = int(inputs["attention_mask"].sum())
            visual_feed = {"pixel_values": inputs["pixel_values"], "image_grid_thw": grid}

            public = (encoder.get_image_embeddings if mode == "image"
                      else encoder.get_fused_embeddings)
            public_kwargs = ({"images": [image]} if mode == "image"
                             else {"texts": [text], "images": [image]})

            for _ in range(args.warmup):
                embeds = np.asarray(visual_request.infer(visual_feed)[encoder.visual.output(0)])
                encoder._language_embedding(inputs, [embeds])
                public(show_progress_bar=False, **public_kwargs)

            pre_ms, vis_ms, lang_ms, e2e_ms = [], [], [], []
            for _ in range(args.repeat):
                started = time.perf_counter()
                preprocess()
                pre_ms.append((time.perf_counter() - started) * 1000)

                started = time.perf_counter()
                embeds = np.asarray(visual_request.infer(visual_feed)[encoder.visual.output(0)])
                vis_ms.append((time.perf_counter() - started) * 1000)

                started = time.perf_counter()
                encoder._language_embedding(inputs, [embeds])
                lang_ms.append((time.perf_counter() - started) * 1000)

                started = time.perf_counter()
                result = public(show_progress_bar=False, **public_kwargs)
                e2e_ms.append((time.perf_counter() - started) * 1000)

            row = {
                "mode": mode,
                "size": f"{height}x{width}",
                "grid": grid.tolist(),
                "patches": patches,
                "tokens": tokens,
                "sequence": seq,
                "shape": list(np.asarray(result).shape),
                "preprocess_p50_ms": percentile(pre_ms),
                "visual_p50_ms": percentile(vis_ms),
                "language_p50_ms": percentile(lang_ms),
                "stage_sum_ms": round(percentile(pre_ms) + percentile(vis_ms)
                                      + percentile(lang_ms), 3),
                "end_to_end_p50_ms": percentile(e2e_ms),
                "end_to_end_min_ms": round(float(np.min(e2e_ms)), 3),
                "end_to_end_max_ms": round(float(np.max(e2e_ms)), 3),
            }
            rows.append(row)
            print(f"{mode:<6} {row['size']:<9} L={patches:<5} K={tokens:<5} S={seq:<5} "
                  f"preproc={row['preprocess_p50_ms']:8.1f} "
                  f"visual={row['visual_p50_ms']:9.1f} "
                  f"language={row['language_p50_ms']:8.1f} "
                  f"sum={row['stage_sum_ms']:9.1f} "
                  f"e2e={row['end_to_end_p50_ms']:9.1f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "precision": args.precision,
            "language_ir": config["language"],
            "visual_ir": config["visual"],
            "device": args.device,
            "batch_mode": encoder.batch_mode,
            "streams": args.streams,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "fused_text_tokens": 32,
            "rows": rows,
        }, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
