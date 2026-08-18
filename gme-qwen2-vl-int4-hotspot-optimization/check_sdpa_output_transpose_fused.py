#!/usr/bin/env python3
"""Report whether opt-2 folded the post-SDPA Transpose, per model, from the GPU runtime graph.

When the fusion fires the `permute` primitive between the SDPA and the o_proj FullyConnected is
gone; when it does not, one permute per attention block remains. Counting them is a direct check
that the optimization still applies where it is supposed to (GME language tower, rank-4 SDPA) and
that it stays away from where it must not (Qwen3-VL vision tower, rank-3 SDPA).
"""
from __future__ import annotations

import argparse
import collections
import sys

import openvino as ov

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from paths import GME_PACKAGE, MODELS_ROOT  # noqa: E402

TARGETS = {
    "gme-lm-int4": GME_PACKAGE / "gme_language_int4.xml",
    "gme-vision-int4": GME_PACKAGE / "gme_visual_int4.xml",
    "qwen3vl-vision-merger-int8": MODELS_ROOT / "Qwen3-VL-Embedding-2B-INT8" / "openvino_vision_embeddings_merger_model.xml",
    "qwen3vl-lm-int8": MODELS_ROOT / "Qwen3-VL-Embedding-2B-INT8" / "openvino_language_model.xml",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="*", default=list(TARGETS))
    ap.add_argument("--device", default="GPU")
    args = ap.parse_args()

    core = ov.Core()
    print(f"openvino {ov.get_version()}  device {args.device}")
    for name in args.targets:
        xml = TARGETS[name]
        if not xml.exists():
            print(f"  {name:<28} SKIP (missing {xml})")
            continue
        rt = core.compile_model(str(xml), args.device).get_runtime_model()
        def layer_type(node):
            info = node.get_rt_info()
            return info["layerType"].get() if "layerType" in info else node.get_type_name()

        types = collections.Counter(layer_type(n) for n in rt.get_ops())
        sdpa = sum(v for k, v in types.items() if "sdpa" in k.lower() or "scaled" in k.lower())
        permute = sum(v for k, v in types.items() if k.lower() in ("permute", "transpose"))
        print(f"  {name:<28} sdpa={sdpa:<4} permute/transpose={permute}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
