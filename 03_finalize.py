#!/usr/bin/env python
"""Finalize the GME OpenVINO package for C++ consumption.

Adds, into the exported package directory:
  * openvino_tokenizer.xml/.bin  -- the Qwen2-VL tokenizer as an OpenVINO model
    (via openvino-tokenizers), so C++ tokenizes natively with NO Python. Verified
    bit-exact against the HF tokenizer on the golden templated prompts.
  * INT8 / INT4 weight-compressed variants of the language and visual IRs (NNCF),
    plus an accuracy-recovered INT4 (smaller group size).
  * an updated gme_openvino_config.json describing every precision.

Run after 02_export.py has produced the FP32 IRs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import openvino as ov
import nncf
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
SOURCE_MODEL = (HERE.parent / "benchmark" / "models" / "gme-Qwen2-VL-2B-Instruct").resolve()
PACKAGE = HERE / "ov_models" / "gme-qwen2-vl-2b-instruct-openvino"
ART = HERE / "artifacts"


# ---------------------------------------------------------------- tokenizer
def convert_and_check_tokenizer() -> None:
    from openvino_tokenizers import convert_tokenizer

    out = PACKAGE / "openvino_tokenizer.xml"
    tok = AutoTokenizer.from_pretrained(str(SOURCE_MODEL))
    tok.padding_side = "right"  # GME pools the last valid token -> must right-pad
    ov_tok = convert_tokenizer(tok, with_detokenizer=False)
    ov.save_model(ov_tok, str(out))
    print(f"[tok] saved {out.name}")

    # bit-exact check against the golden HF tokenization
    meta = json.loads((ART / "golden_meta.json").read_text())
    prompts = meta["templated_text_default"]
    gold_ids = meta["input_ids"]
    gold_mask = meta["attention_mask"]

    core = ov.Core()
    compiled = core.compile_model(ov_tok, "CPU")
    res = compiled(prompts)
    names = {n for out in compiled.outputs for n in out.get_names()}
    key_ids = "input_ids" if "input_ids" in names else compiled.outputs[0].get_any_name()
    ids = None
    mask = None
    for o in compiled.outputs:
        nm = o.get_any_name()
        if "input_ids" in nm:
            ids = np.asarray(res[o])
        elif "attention_mask" in nm:
            mask = np.asarray(res[o])
    assert ids is not None, f"tokenizer outputs: {names}"
    ok_ids = ids.shape == np.array(gold_ids).shape and np.array_equal(ids, np.array(gold_ids, dtype=ids.dtype))
    ok_mask = mask is not None and np.array_equal(mask, np.array(gold_mask, dtype=mask.dtype))
    print(f"[tok] shape ov={ids.shape} hf={np.array(gold_ids).shape}  ids_match={ok_ids} mask_match={ok_mask}")
    if not ok_ids:
        # show first mismatch row for debugging
        for i, (a, b) in enumerate(zip(ids.tolist(), gold_ids)):
            if a != b:
                print(f"[tok]   row {i} MISMATCH\n     ov={a}\n     hf={b}")
                break
        raise RuntimeError("OpenVINO tokenizer does not match HF tokenizer bit-exactly")
    print("[tok] OK: OpenVINO tokenizer is bit-exact with the HF tokenizer")


# ---------------------------------------------------------------- quantize
def compress(src_xml: Path, dst_xml: Path, **kw) -> float:
    if dst_xml.exists():
        size = (dst_xml.with_suffix(".bin")).stat().st_size / 1e6
        print(f"[nncf] {dst_xml.name} exists ({size:.0f} MB), skip")
        return size
    core = ov.Core()
    model = core.read_model(str(src_xml))
    comp = nncf.compress_weights(model, **kw)
    ov.save_model(comp, str(dst_xml))
    size = dst_xml.with_suffix(".bin").stat().st_size / 1e6
    print(f"[nncf] wrote {dst_xml.name} ({size:.0f} MB)  [{kw}]")
    return size


def main() -> None:
    if not (PACKAGE / "gme_language_fp32.xml").exists():
        raise SystemExit("run 02_export.py first (FP32 IRs missing)")

    convert_and_check_tokenizer()

    from nncf.quantization.advanced_parameters import (
        AdvancedCompressionParameters,
        GroupSizeFallbackMode,
    )

    lang_fp32 = PACKAGE / "gme_language_fp32.xml"
    vis_fp32 = PACKAGE / "gme_visual_fp32.xml"

    # CRITICAL for the vision tower: exclude the ViT rotary-embedding constant
    # (`rotary_pos_emb` inv_freq, an aten::outer MatMul). Quantizing it corrupts the
    # rotary frequencies and collapses the visual output (row-cos ~0.32 vs fp32 for
    # BOTH int8 and int4). Excluding it restores image accuracy (int8 image ~0.99).
    # The language tower does not need this (its rotary uses cos/sin buffers, not a
    # quantizable MatMul).
    vis_ignore = nncf.IgnoredScope(patterns=[".*rotary.*"], validate=False)
    # A few tiny vision matmuls have channel size not divisible by the group size ->
    # ADJUST shrinks the group for those layers instead of erroring.
    vis_int4_adv = AdvancedCompressionParameters(group_size_fallback_mode=GroupSizeFallbackMode.ADJUST)

    # INT8 (near-lossless for embeddings): both towers.
    compress(lang_fp32, PACKAGE / "gme_language_int8.xml", mode=nncf.CompressWeightsMode.INT8_ASYM)
    compress(vis_fp32, PACKAGE / "gme_visual_int8.xml",
             mode=nncf.CompressWeightsMode.INT8_ASYM, ignored_scope=vis_ignore)

    # INT4 language (default gs128) and accuracy-recovered (gs64).
    compress(lang_fp32, PACKAGE / "gme_language_int4.xml",
             mode=nncf.CompressWeightsMode.INT4_ASYM, group_size=128, ratio=1.0)
    compress(lang_fp32, PACKAGE / "gme_language_int4_gs64.xml",
             mode=nncf.CompressWeightsMode.INT4_ASYM, group_size=64, ratio=1.0)
    # INT4 vision (rotary excluded), so a fully-INT4 package exists.
    compress(vis_fp32, PACKAGE / "gme_visual_int4.xml",
             mode=nncf.CompressWeightsMode.INT4_ASYM, group_size=128, ratio=1.0,
             ignored_scope=vis_ignore, advanced_parameters=vis_int4_adv)

    cfg_path = PACKAGE / "gme_openvino_config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["tokenizer_model"] = "openvino_tokenizer.xml"
    cfg["precisions"] = {
        "fp32": {"language": "gme_language_fp32.xml", "visual": "gme_visual_fp32.xml"},
        "int8": {"language": "gme_language_int8.xml", "visual": "gme_visual_int8.xml"},
        # fully-INT4 package (both towers INT4)
        "int4": {"language": "gme_language_int4.xml", "visual": "gme_visual_int4.xml"},
        # accuracy-recovered language INT4 (gs64) + INT4 vision
        "int4_gs64": {"language": "gme_language_int4_gs64.xml", "visual": "gme_visual_int4.xml"},
        # mixed: INT4 language + INT8 vision (keeps the quant-sensitive vision tower safer)
        "int4_visint8": {"language": "gme_language_int4.xml", "visual": "gme_visual_int8.xml"},
    }
    cfg["default_precision"] = "int8"
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"[cfg] updated {cfg_path.name} with precisions: {list(cfg['precisions'])}")


if __name__ == "__main__":
    main()
