#!/usr/bin/env python
"""Was the vision attention mask ever load-bearing in this package's exported IR?

docs/11 removes the Qwen2-VL vision tower's block-diagonal ``cu_seqlens`` mask on the
grounds that it is always all-True here. That claim deserves adversarial checking rather
than assertion, because **official GME does not work this way**: ``gme_inference.py`` calls
``self.base.visual(pixel_values, grid_thw=image_grid_thw)`` once with the whole batch's
patches packed into one sequence, and there the mask genuinely separates images.

So this script enumerates every input shape that could make the mask matter, and reports
what each of the two IRs actually does with it:

1. one image, ``t=1``           -- the only shape GME's processor produces; mask all-True
2. several grid rows in one call -- what official HF does; mask genuinely block-diagonal
3. one row with ``t>1`` (video)  -- ``cu_seqlens`` has several blocks even from one row

and separately, in **PyTorch fp32** (no OpenVINO involved), it measures:

4. packed-with-mask vs per-image  -- is the runtime's per-image split actually equivalent?
5. packed-without-mask vs per-image -- if the mask were wrongly dropped, how loud is it?

(4) is what justifies ``_encode_image_batch`` splitting at all, and (5) establishes that a
mistake here would be obvious rather than a silent quality regression.

    python check_vision_mask_necessity.py --output results/r4_20260826/r4_mask_necessity.json
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from paths import GME_PACKAGE, GME_ROOT

PATCH_ROW = 1176
DEFAULT_SOURCE = "/home/chaofan/model_benchmark/benchmark/models/gme-Qwen2-VL-2B-Instruct"


def ir_cases(device: str) -> list[dict]:
    """Feed both IRs the three shapes and record what actually happens."""
    import openvino as ov
    from openvino.properties import hint

    core = ov.Core()
    cfg = {hint.inference_precision: ov.Type.f32}
    rng = np.random.default_rng(0)
    results = []
    for name in ("gme_visual_fp32.xml", "gme_visual_fp32_nomask.xml"):
        model = core.compile_model(GME_PACKAGE / name, device, cfg)
        out = model.output(0)

        def attempt(pixel_values, grid):
            try:
                value = np.asarray(model({"pixel_values": pixel_values,
                                          "image_grid_thw": np.array(grid, dtype=np.int64)})[out])
            except Exception as exc:  # noqa: BLE001 - the failure mode is the result
                return {"status": "raised", "error": f"{type(exc).__name__}: {str(exc)[:160]}"}
            return {"status": "ok", "shape": list(value.shape),
                    "nan_count": int(np.isnan(value).sum()),
                    "all_finite": bool(np.isfinite(value).all())}

        g1 = (1, 32, 32)
        pv1 = rng.standard_normal((int(np.prod(g1)), PATCH_ROW)).astype(np.float32)
        g2 = (1, 16, 16)
        pv2 = rng.standard_normal((int(np.prod(g2)), PATCH_ROW)).astype(np.float32)
        gv = (2, 16, 16)
        pvv = rng.standard_normal((int(np.prod(gv)), PATCH_ROW)).astype(np.float32)

        results.append({
            "ir": name,
            "single_image_t1": attempt(pv1, [g1]),
            "two_grid_rows": attempt(np.concatenate([pv1, pv2]), [g1, g2]),
            "video_t2_single_row": attempt(pvv, [gv]),
        })
        row = results[-1]
        print(f"\n== {name}")
        for key in ("single_image_t1", "two_grid_rows", "video_t2_single_row"):
            print(f"   {key:<22} {row[key]}")
    return results


def torch_equivalence(source_model: str) -> dict:
    """Packed vs per-image, and the size of a genuine cross-image leak, in PyTorch fp32."""
    import importlib.util
    from pathlib import Path

    import torch
    from PIL import Image
    from transformers import AutoProcessor

    spec = importlib.util.spec_from_file_location("gme_export", Path(GME_ROOT) / "02_export.py")
    export = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(export)

    visual = export.load_visual_encoder(Path(source_model))
    processor = AutoProcessor.from_pretrained(str(GME_PACKAGE), min_pixels=256 * 28 * 28,
                                              max_pixels=1280 * 28 * 28, use_fast=False)

    def image(height, width, seed):
        rng = np.random.default_rng(seed)
        y, x = np.mgrid[0:height, 0:width]
        base = np.stack([x * 255 // (width - 1), y * 255 // (height - 1),
                         (x + y) * 255 // (width + height - 2)], -1).astype(np.float64)
        return Image.fromarray(np.clip(base * rng.uniform(0.6, 1.0, 3), 0, 255).astype(np.uint8))

    images = [image(448, 448, 0), image(672, 448, 1), image(448, 672, 2)]
    encoded = processor(text=["<|vision_start|><|image_pad|><|vision_end|>"] * len(images),
                        images=images, padding=True, return_tensors="pt")
    grid, pixel_values = encoded["image_grid_thw"], encoded["pixel_values"]

    def pooled(value):
        return getattr(value, "pooler_output", value)

    with torch.no_grad():
        packed = pooled(visual(pixel_values, grid_thw=grid))
        bounds = np.concatenate([[0], np.cumsum(np.prod(grid.numpy(), axis=1))])
        per_image = torch.cat([
            pooled(visual(pixel_values[bounds[i]:bounds[i + 1]], grid_thw=grid[i:i + 1]))
            for i in range(len(images))
        ], dim=0)
        with export.unmasked_vision_attention(_allow_multi_block=True):
            leaked = pooled(visual(pixel_values, grid_thw=grid))

    def compare(a, b):
        cos = torch.nn.functional.cosine_similarity(a, b, dim=1)
        return {"max_abs_diff": float((a - b).abs().max()),
                "relative": float((a - b).abs().max() / b.abs().max()),
                "row_cosine_min": float(cos.min())}

    noise = compare(packed, per_image)
    leak = compare(leaked, per_image)
    result = {
        "grid_rows": grid.tolist(),
        "packed_with_mask_vs_per_image": noise,
        "packed_without_mask_vs_per_image": leak,
        "leak_over_noise_ratio": leak["max_abs_diff"] / noise["max_abs_diff"],
    }
    print("\n== PyTorch fp32, 3 images")
    print(f"   packed WITH mask vs per-image : row_cos_min={noise['row_cosine_min']:.9f} "
          f"max_abs={noise['max_abs_diff']:.3e}  <- reduction-order noise")
    print(f"   packed NO   mask vs per-image : row_cos_min={leak['row_cosine_min']:.9f} "
          f"max_abs={leak['max_abs_diff']:.3e}  <- a real cross-image leak")
    print(f"   leak / noise = {result['leak_over_noise_ratio']:.0f}x")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="CPU")
    ap.add_argument("--source-model", default=DEFAULT_SOURCE)
    ap.add_argument("--skip-torch", action="store_true",
                    help="skip the PyTorch half (needs the original checkpoint)")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    sys.path.insert(0, str(GME_ROOT))
    payload = {"device": args.device, "ir_cases": ir_cases(args.device)}
    if not args.skip_torch:
        payload["torch_equivalence"] = torch_equivalence(args.source_model)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
