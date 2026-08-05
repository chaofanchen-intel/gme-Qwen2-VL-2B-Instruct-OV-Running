#!/usr/bin/env python
"""
Validate the llama.cpp GGUF text-embedding path against the PyTorch golden.

Feeds the EXACT GME text template (ending in <|endoftext|>) to llama-embedding
with --pooling last and special-token parsing (confirmed in embedding.cpp:
common_tokenize(ctx, prompt, add_special=true, parse_special=true)). Uses a
custom --embd-separator so the multi-line template is treated as ONE prompt,
and writes each prompt WITHOUT a trailing newline so the final token stays
<|endoftext|> (what GME pools).

Compares cosine vs artifacts/golden_text.npy (fp32 reference).
"""
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from gme_paths import ARTIFACTS as ART, GGUF_MODEL as GGUF, LLAMA_EMBEDDING as LLAMA

HERE = Path(__file__).resolve().parent
SEP = "###NO_SPLIT_SEPARATOR###"


def run_one(prompt_text, extra_env=None):
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(prompt_text)  # NO trailing newline
        path = f.name
    cmd = [
        str(LLAMA), "-m", str(GGUF), "-f", path,
        "--pooling", "last", "--embd-separator", SEP,
        "--embd-output-format", "array",
        "-c", "2048", "-b", "2048", "--ctx-size", "2048",
        # NOTE: the experimental ggml-openvino backend in this build ABORTS on the
        # embeddings path (GGML_ASSERT "failed to get sequence embeddings"). Force
        # pure CPU: -dev none (no offload device) + -fa off (no flash-attn op that
        # would otherwise be handed to the OpenVINO device).
        "-dev", "none", "-fa", "off",
    ]
    env = None
    if extra_env:
        import os
        env = {**os.environ, **extra_env}
    out = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)
    Path(path).unlink(missing_ok=True)
    # last non-empty stdout line is the array
    lines = [l for l in out.stdout.splitlines() if l.strip()]
    if not lines:
        raise RuntimeError(f"no output. stderr tail:\n{out.stderr[-800:]}")
    vec = json.loads(lines[-1])
    while isinstance(vec, list) and len(vec) == 1 and isinstance(vec[0], list):
        vec = vec[0]
    return np.array(vec, dtype=np.float64)


def cos(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float((a * b).sum())


def main():
    meta = json.loads((ART / "golden_meta.json").read_text())
    gold = np.load(ART / "golden_text.npy").astype(np.float64)
    templated = meta["templated_text_default"]
    texts = meta["texts"]

    print(f"[llamacpp] {LLAMA.name} on {GGUF.name}")
    print(f"[llamacpp] validating {len(templated)} templated prompts (--pooling last)\n")

    report = {"backend": "build-openvino (CPU+OpenVINO auto)", "pooling": "last", "per_text": []}
    csum = []
    for i, (t, tp) in enumerate(zip(texts, templated)):
        try:
            emb = run_one(tp)
            c = cos(emb, gold[i])
            cross = [round(cos(emb, gold[j]), 3) for j in range(len(gold))]
            csum.append(c)
            print(f"[{i}] dim={emb.size} cos_vs_golden={c:.4f}  argmax_match={int(np.argmax(cross))==i}  text={t[:40]!r}")
            print(f"     cross-cos vs all golden: {cross}")
            report["per_text"].append({"i": i, "dim": int(emb.size), "cos_vs_golden": round(c, 5),
                                       "cross_cos": cross, "argmax_is_self": bool(int(np.argmax(cross)) == i)})
        except Exception as e:
            print(f"[{i}] ERROR {e}")
            report["per_text"].append({"i": i, "error": str(e)[:300]})

    # raw (no template) control on text[0] to show the template matters
    try:
        raw = run_one(texts[0])
        report["raw_no_template_cos_text0"] = round(cos(raw, gold[0]), 5)
        print(f"\n[control] raw text[0] WITHOUT GME template: cos_vs_golden={report['raw_no_template_cos_text0']:.4f}")
    except Exception as e:
        print(f"[control] raw error {e}")

    if csum:
        report["cos_mean"] = round(float(np.mean(csum)), 5)
        report["cos_min"] = round(float(np.min(csum)), 5)
        print(f"\n[llamacpp] cos_vs_golden mean={np.mean(csum):.4f} min={np.min(csum):.4f}")
    (ART / "llamacpp_validation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[llamacpp] wrote {ART/'llamacpp_validation.json'}")


if __name__ == "__main__":
    main()
