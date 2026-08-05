#!/usr/bin/env python
"""
Golden reference generator for gme-Qwen2-VL-2B-Instruct.

Runs the OFFICIAL model code (benchmark/models/.../gme_inference.py) in fp32 on
CPU and saves embeddings for a fixed set of inputs. Every other backend
(OpenVINO, llama.cpp, ...) is validated by cosine-similarity against these
vectors. Also records the exact templated strings + input_ids so other backends
tokenize identically.

Env: gme-Qwen2-VL-2B-Instruct-Running/env-ref  (torch cpu, transformers==4.51.3)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from gme_paths import ARTIFACTS as ART, LOGO_IMAGE, SOURCE_MODEL as MODEL_DIR

HERE = Path(__file__).resolve().parent
ART.mkdir(exist_ok=True)

# Import the official standalone inference class (no remote-code needed:
# AutoModelForVision2Seq maps qwen2_vl -> Qwen2VLForConditionalGeneration).
sys.path.insert(0, str(MODEL_DIR))
import gme_inference  # noqa: E402

torch.manual_seed(0)

# ---- Fixed, network-free test inputs --------------------------------------
TEXTS = [
    "What kind of car is this?",
    "The Tesla Cybertruck is a battery electric pickup truck built by Tesla, Inc. since 2023.",
    "A photo of a cat sitting on a sofa.",
    "如何在 Intel 的 GPU 上高效运行大语言模型？",
    "Neural network quantization reduces model size with minimal accuracy loss.",
]
IMAGE = str(LOGO_IMAGE)
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


def main():
    print(f"[ref] model dir: {MODEL_DIR}")
    print(f"[ref] torch {torch.__version__}  threads={torch.get_num_threads()}")

    gme = gme_inference.GmeQwen2VL(model_path=str(MODEL_DIR), device="cpu")
    gme.base = gme.base.float()  # fp32 canonical golden (weights are fp32 on disk)
    gme.base.eval()

    # --- text embeddings (default instruction, is_query=False semantics) ---
    e_text = gme.get_text_embeddings(texts=TEXTS).float().cpu().numpy()
    # --- text-as-query embeddings (with instruction) ---
    e_query = gme.get_text_embeddings(texts=TEXTS, instruction=INSTRUCTION).float().cpu().numpy()
    # --- image embedding ---
    e_image = gme.get_image_embeddings(images=[IMAGE]).float().cpu().numpy()

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
    tok = gme.processor.tokenizer(templated, padding=True, truncation=True, max_length=1800)
    rec = {
        "texts": TEXTS,
        "instruction": INSTRUCTION,
        "image": IMAGE,
        "templated_text_default": templated,
        "templated_text_query": build_templated(TEXTS, instruction=INSTRUCTION, has_image=False),
        "input_ids": tok["input_ids"],
        "attention_mask": tok["attention_mask"],
        "eos_token_id": gme.processor.tokenizer.eos_token_id,
        "embedding_dim": int(e_text.shape[1]),
        "dtype": "float32",
        "pooling": "last-token (index attention_mask.sum(-1)-1); L2 normalized",
    }
    np.save(ART / "golden_text.npy", e_text)
    np.save(ART / "golden_query.npy", e_query)
    np.save(ART / "golden_image.npy", e_image)
    (ART / "golden_meta.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2))
    print(f"[ref] saved golden_text/query/image.npy + golden_meta.json -> {ART}")


if __name__ == "__main__":
    main()
