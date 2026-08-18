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

compress() now fingerprints each call's actual config (nncf_fingerprints.json next
to the package) and only skips an existing output when the fingerprint matches, so
an edited group_size/mode/ignored_scope/dataset for the same output filename can no
longer be silently ignored by the "file already exists" check. One consequence: the
first run of this script after adding that check has no fingerprint file yet, so it
recompresses every precision once to establish one. For the data-free calls (INT8,
both towers; INT4 vision) that is expected and safe -- ratio=1.0 and no dataset=,
so recompressing reproduces the same artifact bit-for-bit. The two INT4 *language*
calls are now data-aware (dataset=, awq=True, scale_estimation=True, see
_calibration_dataset), so their recompression is only bit-for-bit reproducible in
the sense that AWQ/scale-estimation search is itself deterministic given the same
inputs -- it is not a no-op skip.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import openvino as ov
import nncf
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE_MODEL = HERE.parent / "benchmark" / "models" / "gme-Qwen2-VL-2B-Instruct"
DEFAULT_PACKAGE = HERE / "ov_models" / "gme-qwen2-vl-2b-instruct-openvino"
DEFAULT_ARTIFACTS = HERE / "artifacts"


# ---------------------------------------------------------------- tokenizer
def convert_and_check_tokenizer(
    source_model: Path,
    package: Path,
    artifacts: Path,
    max_length: int,
) -> None:
    from openvino_tokenizers import convert_tokenizer

    out = package / "openvino_tokenizer.xml"
    tok = AutoTokenizer.from_pretrained(str(source_model))
    tok.padding_side = "right"  # GME pools the last valid token -> must right-pad
    # max_length bakes GME's truncation into the IR. Without it the C++ runtime would
    # happily tokenize past 1800 tokens while Python (HF `truncation=True`) and the
    # official model cut there, so the two runtimes would disagree on long inputs and
    # C++ would spend unbounded decode time on them. It truncates only: short prompts
    # keep their true length, padding still goes to the batch maximum.
    ov_tok = convert_tokenizer(tok, with_detokenizer=False, max_length=max_length)
    ov.save_model(ov_tok, str(out))
    print(f"[tok] saved {out.name} (truncating at max_length={max_length})")

    # bit-exact check against the golden HF tokenization
    meta = json.loads((artifacts / "golden_meta.json").read_text())
    prompts = meta["templated_text_default"]
    gold_ids = meta["input_ids"]
    gold_mask = meta["attention_mask"]

    core = ov.Core()
    compiled = core.compile_model(ov_tok, "CPU")
    res = compiled(prompts)
    names = {n for out in compiled.outputs for n in out.get_names()}
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

    # The golden prompts are short, so they cannot exercise truncation. Check it
    # separately, batched with a short prompt so padding and truncation interact.
    long_prompt = prompts[0].replace("<|im_end|>\n<|im_start|>assistant", "word " * 3000 + "<|im_end|>\n<|im_start|>assistant", 1)
    batch = [prompts[0], long_prompt]
    result = compiled(batch)
    reference = tok(batch, padding=True, truncation=True, max_length=max_length, return_tensors="np")
    for out_port in compiled.outputs:
        name = out_port.get_any_name()
        key = "input_ids" if "input_ids" in name else "attention_mask" if "attention_mask" in name else None
        if key is None:
            continue
        actual = np.asarray(result[out_port])
        if not np.array_equal(actual, reference[key].astype(actual.dtype)):
            raise RuntimeError(f"tokenizer truncation differs from HF on {key}: {actual.shape} vs {reference[key].shape}")
    print(f"[tok] OK: truncation at max_length={max_length} matches HF (long input -> {np.asarray(result[compiled.outputs[0]]).shape})")


# ---------------------------------------------------------------- quantize

# INT4 weight-only compression is data-free (ratio=1.0, no dataset) for every tower
# except this one: measured accuracy (04_validate_python.py, artifacts/
# openvino_full_validation.json) shows plain INT4 hurts *text* embeddings much more
# than image/fused (cosine ~0.89-0.92 vs official FP32, against ~0.94-0.96 for
# image/fused) -- last-token pooling over a decoder is more sensitive to per-channel
# weight error than the vision tower's patch-averaged output. AWQ + scale estimation
# search per-channel scales that minimize *this model's* actual activation error, so
# they need representative activations to search over, unlike plain data-free RTN.
#
# The calibration set is deliberately not the golden evaluation texts
# (01_reference.py): calibration should not be scored on its own test data, even
# though weight-only PTQ calibration (collecting activation-magnitude statistics,
# not gradient updates) is a much weaker form of "seeing the test set" than
# fine-tuning would be. Diversity of length/language/register matters far more here
# than domain realism -- this mirrors how NNCF's own LLM examples calibrate from a
# handful of representative prompts, not a full corpus. Only the language IR is
# calibrated: the vision tower does not show this problem, so it stays data-free.
_CALIBRATION_TEXTS: list[tuple[str, str | None]] = [
    # (text, instruction) -- instruction=None means the corpus-style default GME
    # instruction ("You are a helpful assistant."), matching is_query=False.
    ("How do I reset my password on this website?", None),
    ("What time does the museum open on Sundays?", None),
    ("The quarterly earnings report showed a 12% increase in revenue year over year.", None),
    ("Mix 2 cups of flour with a teaspoon of baking soda, then fold in the melted butter.", None),
    ("def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)", None),
    ("SELECT customer_id, SUM(total) FROM orders GROUP BY customer_id ORDER BY 2 DESC LIMIT 10;", None),
    ("A large orange cat sleeping on a windowsill in the afternoon sun.", None),
    ("The Amazon rainforest produces roughly 20% of the world's oxygen supply.", None),
    ("请解释一下什么是卷积神经网络，以及它在图像识别中的应用。", None),
    ("北京的秋天天气凉爽，是登长城的最佳季节之一。", None),
    ("Photosynthesis converts carbon dioxide and water into glucose and oxygen using light energy.", None),
    ("On 2024-03-14 at 09:30 UTC, flight AC142 departed gate B17 for a 6h45m transatlantic crossing.", None),
    (
        "Neural network quantization reduces model size and inference latency by representing "
        "weights and activations with fewer bits, typically 8-bit integers instead of 32-bit "
        "floating point, at the cost of some accuracy that can often be recovered with "
        "calibration-aware or quantization-aware training techniques.",
        None,
    ),
    (
        "The French Revolution, beginning in 1789, fundamentally transformed French society by "
        "abolishing the monarchy and feudal privileges, establishing in their place a series of "
        "republics and empires whose legal and political innovations reverberated across Europe "
        "for the following century.",
        None,
    ),
    ("Battery electric vehicles convert over 77% of electrical energy from the grid to power at the wheels.", None),
    ("The train to Boston leaves platform 4 at quarter past six every weekday morning.", None),
    ("import numpy as np\narr = np.zeros((3, 3), dtype=np.float32)\narr[1, 1] = 1.0", None),
    ("A weathered wooden pier extending into a calm lake at sunrise, mist rising off the water.", None),
    ("Interest rates rose by 25 basis points, the third consecutive increase this fiscal year.", None),
    ("为什么天空是蓝色的？这与太阳光在大气层中的瑞利散射有关。", None),
    ("Please summarize the key differences between TCP and UDP for a networking course.", None),
    ("A red mountain bike leaning against a brick wall in an alleyway.", None),
    ("The recipe calls for simmering the sauce on low heat for at least forty-five minutes.", None),
    ("Global smartphone shipments declined 3.2% in the first quarter compared to the prior year.", None),
    ("192.168.1.1 is a commonly used private IPv4 address for home network routers.", None),
    ("Can you recommend a good beginner book on machine learning?", None),
    (
        "In distributed systems, the CAP theorem states that a system can provide at most two of "
        "the following three guarantees simultaneously: consistency, availability, and partition "
        "tolerance, which forces architects to make explicit tradeoffs depending on workload.",
        None,
    ),
    ("A stack of pancakes topped with maple syrup and a pat of melting butter.", None),
    ("水的沸点在标准大气压下是攝氏一百度。", None),
    # A few query-style prompts (is_query=True), covering the other GME instruction:
    ("What kind of car is this?", "Find an image that matches the given text."),
    ("best hiking trails near Denver with waterfalls", "Represent this sentence for searching relevant passages."),
    ("How does gradient descent converge for a convex loss function?", "Find a passage that answers this question."),
]


def _prompt(text: str, instruction: str | None) -> str:
    """The same GME chat template used at inference (02_export.py/gme_openvino.py)."""
    inst = instruction if instruction is not None else "You are a helpful assistant."
    return (
        f"<|im_start|>system\n{inst}<|im_end|>\n"
        f"<|im_start|>user\n{text}<|im_end|>\n"
        f"<|im_start|>assistant\n<|endoftext|>"
    )


def _calibration_dataset(source_model: Path, max_length: int, embedding_dim: int) -> "nncf.Dataset":
    """Build the nncf.Dataset used to calibrate the language IR's INT4 variants.

    Each item is tokenized independently (its own [1, S] shape), matching how the
    language IR is actually invoked in per_sample batch_mode and exercising a range
    of real sequence lengths rather than one padded shape. image_embeds is the same
    empty [0, dim] tensor text-only inference uses -- see _language_embedding.
    """
    tok = AutoTokenizer.from_pretrained(str(source_model))
    tok.padding_side = "right"
    prompts = [_prompt(text, instruction) for text, instruction in _CALIBRATION_TEXTS]

    def transform(prompt: str) -> dict[str, np.ndarray]:
        encoded = tok([prompt], truncation=True, max_length=max_length, return_tensors="np")
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "image_embeds": np.empty((0, embedding_dim), dtype=np.float32),
        }

    dataset = nncf.Dataset(prompts, transform)
    # See _fingerprint(): nncf.Dataset has no value-based repr, so compress()'s
    # staleness check would otherwise treat every data-aware call as "changed".
    dataset._gme_fingerprint = repr(prompts)
    return dataset


def _fingerprint(src_xml: Path, kw: dict) -> str:
    """Stable hash of one compress_weights() call, for staleness detection below.

    repr() on nncf's dataclass-based kwargs (CompressWeightsMode, IgnoredScope,
    AdvancedCompressionParameters, ...) expands every field, including ones nncf
    itself defaults -- checked deterministic across separately-constructed but
    equal objects. So this also catches an nncf upgrade that silently changes a
    default we never touched, not only edits to the arguments listed in main().

    A ``dataset=`` value is the one exception: nncf.Dataset has no value-based
    __repr__ (two datasets built from identical calibration data repr as distinct
    objects, by memory address), so it is fingerprinted via the stable marker
    _calibration_dataset() attaches instead of the object itself -- otherwise every
    data-aware call would look "changed" on every run and never skip.
    """
    kw = dict(kw)
    dataset = kw.get("dataset")
    if dataset is not None:
        kw["dataset"] = getattr(dataset, "_gme_fingerprint", repr(dataset))
    payload = repr((src_xml.name, sorted(kw.items())))
    return hashlib.sha256(payload.encode()).hexdigest()


def _read_fingerprints(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _write_fingerprint(path: Path, name: str, fingerprint: str) -> None:
    fingerprints = _read_fingerprints(path)
    fingerprints[name] = fingerprint
    path.write_text(json.dumps(fingerprints, indent=2) + "\n")


def compress(src_xml: Path, dst_xml: Path, fingerprints_path: Path, force: bool = False, **kw) -> float:
    fingerprint = _fingerprint(src_xml, kw)
    if dst_xml.exists():
        binary = dst_xml.with_suffix(".bin")
        size = binary.stat().st_size / 1e6 if binary.exists() else 0
        if not force and binary.exists() and _read_fingerprints(fingerprints_path).get(dst_xml.name) == fingerprint:
            print(f"[nncf] {dst_xml.name} exists ({size:.0f} MB), config unchanged, skip")
            return size
        # Existing file's compression config no longer matches this call -- e.g. a
        # changed group_size/mode/ignored_scope, or an upgraded nncf with different
        # defaults. Re-running is the safe choice: silently keeping the stale file
        # here previously meant an edited quantization recipe never actually took
        # effect until someone noticed the .xml's mtime predated the edit.
        print(f"[nncf] {dst_xml.name} exists but its compression config changed -- recompressing")
    core = ov.Core()
    model = core.read_model(str(src_xml))
    comp = nncf.compress_weights(model, **kw)
    ov.save_model(comp, str(dst_xml))
    size = dst_xml.with_suffix(".bin").stat().st_size / 1e6
    _write_fingerprint(fingerprints_path, dst_xml.name, fingerprint)
    # nncf.Dataset's default repr is a bare memory address; swap in something
    # readable for the log line (the fingerprint above already covers correctness).
    logged_kw = dict(kw)
    if logged_kw.get("dataset") is not None:
        logged_kw["dataset"] = "<calibration dataset>"
    print(f"[nncf] wrote {dst_xml.name} ({size:.0f} MB)  [{logged_kw}]")
    return size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", type=Path, default=DEFAULT_SOURCE_MODEL,
                        help="directory containing the GME checkpoint")
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE,
                        help="exported package directory containing the FP32 IRs")
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS,
                        help="directory containing golden_meta.json")
    parser.add_argument("--max-length", type=int,
                        help="tokenizer truncation limit (default: value in package config)")
    parser.add_argument("--force", action="store_true",
                        help="recompress existing artifacts even when their fingerprint matches")
    args = parser.parse_args()

    source_model = args.source_model.expanduser().resolve()
    package = args.package.expanduser().resolve()
    artifacts = args.artifacts.expanduser().resolve()
    if not source_model.is_dir():
        raise SystemExit(f"source model directory not found: {source_model}")
    if not package.is_dir():
        raise SystemExit(f"package directory not found: {package}")
    if not artifacts.is_dir():
        raise SystemExit(f"artifacts directory not found: {artifacts}")
    if not (package / "gme_language_fp32.xml").exists():
        raise SystemExit("run 02_export.py first (FP32 IRs missing)")

    cfg_path = package / "gme_openvino_config.json"
    package_config = json.loads(cfg_path.read_text())
    max_length = package_config["processor"]["max_length"] if args.max_length is None else args.max_length
    if max_length < 1:
        raise SystemExit("--max-length must be positive")
    embedding_dim = int(package_config["embedding_dim"])
    fingerprints_path = package / "nncf_fingerprints.json"
    convert_and_check_tokenizer(source_model, package, artifacts, max_length)

    from nncf.quantization.advanced_parameters import (
        AdvancedCompressionParameters,
        GroupSizeFallbackMode,
    )

    lang_fp32 = package / "gme_language_fp32.xml"
    vis_fp32 = package / "gme_visual_fp32.xml"

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
    compress(lang_fp32, package / "gme_language_int8.xml", fingerprints_path, args.force,
             mode=nncf.CompressWeightsMode.INT8_ASYM)
    compress(vis_fp32, package / "gme_visual_int8.xml", fingerprints_path, args.force,
             mode=nncf.CompressWeightsMode.INT8_ASYM, ignored_scope=vis_ignore)

    # INT4 language (default gs128) and accuracy-recovered (gs64): data-aware, see
    # _calibration_dataset for why -- plain (data-free) INT4 hurts text embeddings
    # noticeably more than image/fused.
    lang_calibration = _calibration_dataset(source_model, max_length, embedding_dim)
    compress(lang_fp32, package / "gme_language_int4.xml", fingerprints_path, args.force,
             mode=nncf.CompressWeightsMode.INT4_ASYM, group_size=128, ratio=1.0,
             dataset=lang_calibration, awq=True, scale_estimation=True,
             subset_size=len(_CALIBRATION_TEXTS))
    compress(lang_fp32, package / "gme_language_int4_gs64.xml", fingerprints_path, args.force,
             mode=nncf.CompressWeightsMode.INT4_ASYM, group_size=64, ratio=1.0,
             dataset=lang_calibration, awq=True, scale_estimation=True,
             subset_size=len(_CALIBRATION_TEXTS))
    # Per-output-channel language INT4: group_size=-1 means one scale (and, for the
    # ASYM variant, one zero-point) per output channel instead of one per 128-element
    # group along K. This is the requantized counterpart of the oneDNN GEMM-probe
    # candidates in gme-qwen2-vl-int4-hotspot-optimization/onednn/gemm_probe.cpp
    # (docs/06_round2_onednn_investigation.md SS6.3/6.6): D_cab_per_oc measured 1.55x
    # on the GEMM alone, but that case is ALSO symmetric (wei_zp=false) -- group_size
    # and symmetry are two separate knobs there. Two variants isolate them:
    #   int4_per_oc      ASYM + group_size=-1  (only removes the K-grouping; the probe
    #                     never priced this exact combination on its own, since its
    #                     per-OC case also went symmetric)
    #   int4_per_oc_sym  SYM  + group_size=-1  (matches D_cab_per_oc exactly)
    compress(lang_fp32, package / "gme_language_int4_per_oc.xml", fingerprints_path, args.force,
             mode=nncf.CompressWeightsMode.INT4_ASYM, group_size=-1, ratio=1.0,
             dataset=lang_calibration, awq=True, scale_estimation=True,
             subset_size=len(_CALIBRATION_TEXTS))
    compress(lang_fp32, package / "gme_language_int4_per_oc_sym.xml", fingerprints_path, args.force,
             mode=nncf.CompressWeightsMode.INT4_SYM, group_size=-1, ratio=1.0,
             dataset=lang_calibration, awq=True, scale_estimation=True,
             subset_size=len(_CALIBRATION_TEXTS))
    # INT4 vision (rotary excluded), so a fully-INT4 package exists. Left data-free:
    # measured accuracy shows the vision tower does not have the text tower's INT4
    # degradation (see _calibration_dataset), so there is nothing here to recover.
    compress(vis_fp32, package / "gme_visual_int4.xml", fingerprints_path, args.force,
             mode=nncf.CompressWeightsMode.INT4_ASYM, group_size=128, ratio=1.0,
             ignored_scope=vis_ignore, advanced_parameters=vis_int4_adv)

    cfg = json.loads(cfg_path.read_text())
    cfg["processor"]["max_length"] = max_length
    cfg["tokenizer_model"] = "openvino_tokenizer.xml"
    cfg["precisions"] = {
        "fp32": {"language": "gme_language_fp32.xml", "visual": "gme_visual_fp32.xml"},
        "int8": {"language": "gme_language_int8.xml", "visual": "gme_visual_int8.xml"},
        # fully-INT4 package (both towers INT4)
        "int4": {"language": "gme_language_int4.xml", "visual": "gme_visual_int4.xml"},
        # accuracy-recovered language INT4 (gs64) + INT4 vision
        "int4_gs64": {"language": "gme_language_int4_gs64.xml", "visual": "gme_visual_int4.xml"},
        # per-output-channel language INT4 (group_size=-1), asymmetric vs symmetric --
        # see docs/08_per_oc_quantization.md
        "int4_per_oc": {"language": "gme_language_int4_per_oc.xml", "visual": "gme_visual_int4.xml"},
        "int4_per_oc_sym": {"language": "gme_language_int4_per_oc_sym.xml", "visual": "gme_visual_int4.xml"},
        # mixed: INT4 language + INT8 vision (keeps the quant-sensitive vision tower safer)
        "int4_visint8": {"language": "gme_language_int4.xml", "visual": "gme_visual_int8.xml"},
    }
    cfg["default_precision"] = "int8"
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"[cfg] updated {cfg_path.name} with precisions: {list(cfg['precisions'])}")


if __name__ == "__main__":
    main()
