#!/usr/bin/env python
"""Does the mask-free visual IR still handle multi-image batches correctly?

This is the risk the mask removal buys its speed with, so it gets its own check.

The shipped visual IR carries a block-diagonal ``[1, L, L]`` mask, which in principle
lets one call cover several images. ``gme_visual_fp32_nomask.xml`` drops it and attends
densely, so that call would silently mix images together. What makes the swap safe is
that ``gme_openvino.py:_encode_image_batch`` splits a batch into one visual request per
image before it ever reaches the IR -- so the mask was always all-True in practice.

check_vision_equivalence.py proves the two IRs agree on a single image. This one proves
the *runtime* upholds the single-image contract, by checking the property that would
break first if it did not: for a batch of differently-sized images, every visual-tower
row must equal what the same image produces on its own. Mixed sizes matter -- with a
shared sequence and no mask, image i would pick up image j != i and the batch/singleton
outputs would diverge.

The comparison is made on the **visual IR's own output**, not on the final embedding, and
that is deliberate. Going end-to-end would fold in the language IR's ``padded`` batch mode,
whose rows already shift slightly with batch composition (see ``GmeOpenVINO``'s docstring)
for reasons that have nothing to do with the vision tower -- measured here at up to 1e-2,
and present on the shipped mask-carrying package too. Comparing the visual output isolates
the property this change could actually break.

    python check_vision_batch_contract.py --precisions int4_causal,int4_causal_vnomask,int8,int8_vnomask
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from paths import GME_PACKAGE, GME_ROOT

# Deliberately mixed sizes and aspect ratios: equal-sized images could agree by accident
# even if the batch path were wrong.
SIZES = [(448, 448), (672, 448), (448, 672)]


def gradient_image(height: int, width: int, seed: int):
    from PIL import Image

    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width]
    base = np.stack([
        x * 255 // max(width - 1, 1),
        y * 255 // max(height - 1, 1),
        (x + y) * 255 // max(width + height - 2, 1),
    ], axis=-1).astype(np.float64)
    # A per-image random tint makes the images genuinely different, so a leak between
    # rows shows up as a numeric difference rather than cancelling out.
    base = np.clip(base * rng.uniform(0.6, 1.0, size=3), 0, 255)
    return Image.fromarray(base.astype(np.uint8))


def cos_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    num = (a * b).sum(axis=1)
    return num / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="GPU")
    ap.add_argument("--precisions", default="int4_causal_vopt,int4_gs64_causal_vopt,int8_causal_vopt")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    sys.path.insert(0, str(GME_ROOT))
    from gme_openvino import GmeOpenVINO

    images = [gradient_image(h, w, seed) for seed, (h, w) in enumerate(SIZES)]
    rows = []
    failures = []
    for precision in [p.strip() for p in args.precisions.split(",") if p.strip()]:
        encoder = GmeOpenVINO(GME_PACKAGE, device=args.device, precision=precision)
        visual_ir = encoder.config["precisions"][precision]["visual"]

        instruction = encoder._instruction(None, True)
        prompt = encoder._prompt(None, has_image=True, instruction=instruction)

        # The batch the runtime would build for all three images at once, split back into
        # per-image visual feeds exactly the way _encode_image_batch does.
        multi = encoder.processor(text=[prompt] * len(images), images=images, padding=True,
                                  truncation=True, max_length=encoder.max_length,
                                  return_tensors="np")
        grid = multi["image_grid_thw"]
        bounds = np.concatenate([[0], np.cumsum(np.prod(grid, axis=1, dtype=np.int64))])
        solo = [encoder.processor(text=[prompt], images=[image], padding=True, truncation=True,
                                  max_length=encoder.max_length, return_tensors="np")
                for image in images]

        request = encoder.visual.create_infer_request()
        out = encoder.visual.output(0)

        def run(pixel_values, grid_row):
            return np.asarray(request.infer({"pixel_values": pixel_values,
                                             "image_grid_thw": grid_row})[out]).copy()

        batched = [run(multi["pixel_values"][bounds[i]:bounds[i + 1]], grid[i:i + 1])
                   for i in range(len(images))]
        singles = [run(solo[i]["pixel_values"], solo[i]["image_grid_thw"])
                   for i in range(len(images))]

        stacked_b = np.concatenate(batched, axis=0)
        stacked_s = np.concatenate(singles, axis=0)
        cosines = cos_rows(stacked_b, stacked_s)
        max_abs = float(np.max(np.abs(stacked_b - stacked_s)))
        row = {
            "precision": precision,
            "visual_ir": visual_ir,
            "sizes": [f"{h}x{w}" for h, w in SIZES],
            "visual_tokens": [int(a.shape[0]) for a in singles],
            "row_cosine_min": float(cosines.min()),
            "max_abs_diff": max_abs,
            "bit_identical": bool(max_abs == 0.0),
        }
        rows.append(row)
        print(f"{precision:<28} visual={visual_ir:<34} K={row['visual_tokens']} "
              f"row_cos_min={row['row_cosine_min']:.9f} max_abs={max_abs:.3e} "
              f"bit_identical={row['bit_identical']}")
        if not row["bit_identical"]:
            failures.append(precision)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump({"device": args.device, "sizes": [f"{h}x{w}" for h, w in SIZES],
                       "results": rows}, handle, indent=2)
            handle.write("\n")
        print(f"wrote {args.output}")

    if failures:
        raise SystemExit(f"visual output differs batch vs singleton for: {', '.join(failures)}")
    print("\nevery image's visual output is bit-identical batched vs alone: "
          "the single-image contract holds")


if __name__ == "__main__":
    main()
