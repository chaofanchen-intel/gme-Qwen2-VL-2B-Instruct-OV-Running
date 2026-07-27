#!/usr/bin/env python
"""Create independent INT8/INT4 variants of the monolithic GME embedding IR.

This script only reads ``gme_monolithic_fp32.xml`` and writes new files.  It
does not modify the retained dual-IR FP32/INT8/INT4 artifacts.  The visual
rotary-frequency subgraph must stay in floating point: compressing its outer
product MatMul collapses visual embedding quality.
"""
from __future__ import annotations

import json
from pathlib import Path

import nncf
import openvino as ov
from nncf.quantization.advanced_parameters import AdvancedCompressionParameters, GroupSizeFallbackMode

HERE = Path(__file__).resolve().parent
PACKAGE = HERE / "ov_models" / "gme-qwen2-vl-2b-instruct-openvino"
SOURCE = PACKAGE / "gme_monolithic_fp32.xml"


def compress(destination: Path, **kwargs) -> None:
    if destination.exists() and destination.with_suffix(".bin").exists():
        print(f"[monolithic] {destination.name} exists, skip")
        return
    print(f"[monolithic] compressing {SOURCE.name} -> {destination.name}")
    model = ov.Core().read_model(SOURCE)
    compressed = nncf.compress_weights(model, **kwargs)
    ov.save_model(compressed, destination)
    print(f"[monolithic] wrote {destination.name} ({destination.with_suffix('.bin').stat().st_size / 1e6:.0f} MB)")


def main() -> None:
    if not SOURCE.exists():
        raise SystemExit("run 08_compose_monolithic.py first (FP32 monolithic IR is missing)")

    # Qwen2-VL visual RoPE is expressed as an outer-product MatMul. It carries
    # position frequencies, not learned weights, and must not be compressed.
    visual_rotary = nncf.IgnoredScope(patterns=[".*visual.*rotary.*"], validate=False)
    fallback = AdvancedCompressionParameters(group_size_fallback_mode=GroupSizeFallbackMode.ADJUST)
    compress(
        PACKAGE / "gme_monolithic_int8.xml",
        mode=nncf.CompressWeightsMode.INT8_ASYM,
        ignored_scope=visual_rotary,
    )
    compress(
        PACKAGE / "gme_monolithic_int4.xml",
        mode=nncf.CompressWeightsMode.INT4_ASYM,
        group_size=128,
        ratio=1.0,
        ignored_scope=visual_rotary,
        advanced_parameters=fallback,
    )
    compress(
        PACKAGE / "gme_monolithic_int4_gs64.xml",
        mode=nncf.CompressWeightsMode.INT4_ASYM,
        group_size=64,
        ratio=1.0,
        ignored_scope=visual_rotary,
        advanced_parameters=fallback,
    )

    config_path = PACKAGE / "gme_openvino_config.json"
    config = json.loads(config_path.read_text())
    config["monolithic_models"] = {
        "fp32": "gme_monolithic_fp32.xml",
        "int8": "gme_monolithic_int8.xml",
        "int4": "gme_monolithic_int4.xml",
        "int4_gs64": "gme_monolithic_int4_gs64.xml",
    }
    config["monolithic_model"] = {
        "models": config["monolithic_models"],
        "modes": ["image", "fused"],
        "text_query_behavior": "use the retained language-only dual-IR path at the requested precision",
        "rotary_quantization": "visual rotary subgraph excluded from NNCF compression",
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    print(f"[monolithic] updated {config_path.name}")


if __name__ == "__main__":
    main()