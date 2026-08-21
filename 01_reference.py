#!/usr/bin/env python
"""
Golden reference generator for gme-Qwen2-VL-2B-Instruct.

Runs the OFFICIAL model code in fp32 and saves embeddings for a fixed set of
inputs. OpenVINO is validated by cosine-similarity against these vectors. The
script also records the exact templated strings and input_ids so the runtime
can tokenize identically.

Env: gme-Qwen2-VL-2B-Instruct-Running/env-ref  (torch cpu, transformers==4.51.3)
"""
import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np

from gme_compat import patch_transformers
import torch

HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE_MODEL = HERE.parent / "benchmark" / "models" / "gme-Qwen2-VL-2B-Instruct"
DEFAULT_ARTIFACTS = HERE / "artifacts"

torch.manual_seed(0)

# ---- Fixed, network-free test inputs --------------------------------------
TEXTS = [
    "What kind of car is this?",
    "The Tesla Cybertruck is a battery electric pickup truck built by Tesla, Inc. since 2023.",
    "A photo of a cat sitting on a sofa.",
    "如何在 Intel 的 GPU 上高效运行大语言模型？",
    "Neural network quantization reduces model size with minimal accuracy loss.",
]
INSTRUCTION = "Find an image that matches the given text."


def build_templated(texts, instruction=None, has_image=False):
    """Replicate GmeQwen2VL.embed() prompt templating (for the record)."""
    out = []
    default = "You are a helpful assistant."
    inst = instruction or default
    for t in texts:
        s = ""
        if has_image:
            s += "<|vision_start|><|image_pad|><|vision_end|>"
        if t is not None:
            s += t
        out.append(
            f"<|im_start|>system\n{inst}<|im_end|>\n"
            f"<|im_start|>user\n{s}<|im_end|>\n"
            f"<|im_start|>assistant\n<|endoftext|>"
        )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-model",
        type=Path,
        default=DEFAULT_SOURCE_MODEL,
        help="directory containing the GME checkpoint and gme_inference.py",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="regenerate the goldens even if they exist (invalidates published cosines)",
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=DEFAULT_ARTIFACTS,
        help="directory for golden vectors and metadata",
    )
    parser.add_argument(
        "--image",
        type=Path,
        help="image used for the golden image/fused vectors (default: <source-model>/images/gme_logo.png)",
    )
    parser.add_argument("--device", default="cpu", help="PyTorch device (default: cpu)")
    parser.add_argument("--max-length", type=int, default=1800, help="tokenization limit (default: 1800)")
    return parser.parse_args()


def main():
    args = parse_args()
    model_dir = args.source_model.expanduser().resolve()
    artifacts = args.artifacts.expanduser().resolve()
    image = (args.image or model_dir / "images" / "gme_logo.png").expanduser().resolve()
    if not model_dir.is_dir():
        raise SystemExit(f"source model directory not found: {model_dir}")
    if not image.is_file():
        raise SystemExit(f"reference image not found: {image}")
    if args.max_length < 1:
        raise SystemExit("--max-length must be positive")
    artifacts.mkdir(parents=True, exist_ok=True)

    # Every published cosine in docs/04, docs/08 and docs/09 is measured against these exact
    # arrays, so silently regenerating them would make all historical numbers incomparable --
    # especially now that the pipeline runs on a different transformers major than the one
    # that produced them. Refuse unless asked.
    existing = [name for name in ("golden_text.npy", "golden_query.npy", "golden_image.npy",
                                  "golden_meta.json") if (artifacts / name).exists()]
    if existing and not args.force:
        print(f"[ref] goldens already present in {artifacts} ({len(existing)} files), skipping.")
        print("[ref] pass --force to regenerate -- this invalidates every published cosine.")
        return

    print(f"[ref] model dir: {model_dir}")
    print(f"[ref] torch {torch.__version__}  threads={torch.get_num_threads()}")

    sys.path.insert(0, str(model_dir))
    # The checkpoint's gme_inference.py still imports transformers 4.x names.
    patch_transformers()
    gme_inference = importlib.import_module("gme_inference")

    gme = gme_inference.GmeQwen2VL(model_path=str(model_dir), device=args.device)
    gme.base = gme.base.float()  # fp32 canonical golden (weights are fp32 on disk)
    gme.base.eval()

    # --- text embeddings (default instruction, is_query=False semantics) ---
    e_text = gme.get_text_embeddings(texts=TEXTS).float().cpu().numpy()
    # --- text-as-query embeddings (with instruction) ---
    e_query = gme.get_text_embeddings(texts=TEXTS, instruction=INSTRUCTION).float().cpu().numpy()
    # --- image embedding ---
    e_image = gme.get_image_embeddings(images=[str(image)]).float().cpu().numpy()

    print(f"[ref] e_text shape {e_text.shape}  norm[0]={np.linalg.norm(e_text[0]):.5f}")
    print(f"[ref] e_image shape {e_image.shape}")

    # text-text cosine matrix (embeddings already L2-normalized)
    cos_tt = e_text @ e_text.T
    print("[ref] text-text cosine matrix:")
    np.set_printoptions(precision=4, suppress=True)
    print(cos_tt)

    # text-image cosine
    cos_ti = (e_query @ e_image.T).ravel()
    print(f"[ref] query-image cosine (instruction): {cos_ti}")

    # record templated strings + input_ids for text path
    templated = build_templated(TEXTS, instruction=None, has_image=False)
    tok = gme.processor.tokenizer(templated, padding=True, truncation=True, max_length=args.max_length)
    rec = {
        "texts": TEXTS,
        "instruction": INSTRUCTION,
        "image": str(image),
        "templated_text_default": templated,
        "templated_text_query": build_templated(TEXTS, instruction=INSTRUCTION, has_image=False),
        "input_ids": tok["input_ids"],
        "attention_mask": tok["attention_mask"],
        "eos_token_id": gme.processor.tokenizer.eos_token_id,
        "max_length": args.max_length,
        "embedding_dim": int(e_text.shape[1]),
        "dtype": "float32",
        "pooling": "last-token (index attention_mask.sum(-1)-1); L2 normalized",
    }
    np.save(artifacts / "golden_text.npy", e_text)
    np.save(artifacts / "golden_query.npy", e_query)
    np.save(artifacts / "golden_image.npy", e_image)
    (artifacts / "golden_meta.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2))
    print(f"[ref] saved golden_text/query/image.npy + golden_meta.json -> {artifacts}")


if __name__ == "__main__":
    main()
