#!/usr/bin/env python
"""Export the complete GME-Qwen2-VL embedding runtime to OpenVINO.

The output directory contains two composable IRs and the original tokenizer /
image-processor assets:

* ``gme_language_fp32.xml``: token lookup, visual-token insertion, decoder,
  last-valid-token pooling and L2 normalization.
* ``gme_visual_fp32.xml``: Qwen2-VL's visual encoder.

Splitting the two neural subgraphs avoids the excessive conversion-memory peak
of a complete visual-language graph. The runtime only passes the visual
encoder output to the language IR; all learned model computation remains in
OpenVINO while the official processor retains tokenization and pixel-preprocess
parity.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import openvino as ov
import torch
import torch.nn.functional as functional
from transformers import AutoConfig, AutoProcessor

from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VisionTransformerPretrainedModel
from safetensors import safe_open
from openvino.properties import hint

from gme_compat import auto_vlm_class

# Resolving the class also installs the transformers 4.x aliases that the checkpoint's own
# gme_inference.py needs (imported lazily in make_example for fetch_image).
# See gme_compat and docs/09 S9.3.
_AutoVLM = auto_vlm_class()

HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE_MODEL = HERE.parent / "benchmark" / "models" / "gme-Qwen2-VL-2B-Instruct"
DEFAULT_OUTPUT = HERE / "ov_models" / "gme-qwen2-vl-2b-instruct-openvino"

DEFAULT_INSTRUCTION = "You are a helpful assistant."
MIN_IMAGE_TOKENS = 256
MAX_IMAGE_TOKENS = 1280
MAX_LENGTH = 1800


class GmeLanguageEmbed(torch.nn.Module):
    """Decoder path for text, image, and fused embeddings.

    ``causal_attention`` drops the explicit ``[B, 1, S, S]`` attention mask and lets the
    exported SDPA carry ``causal=True`` instead.  For GME this is an identity, not an
    approximation, and it is worth ~10 ms of the 1024-token GPU latency:

    * The kernel really does skip the upper triangle -- ``sdpa_micro.cl`` clamps its K-loop
      bound to ``causal_k = min(k, wg_j0 + wg_tile_n)`` under ``IS_CAUSAL``, rather than
      computing the full S x S and masking afterwards.
    * The padding half of the mask is redundant here.  The package is right-padded
      (``gme_openvino.py`` rejects anything else) and pooling reads the *last valid* token,
      so every padding position sits at a strictly larger index than any token whose output
      is read -- causal masking already excludes them.
    * ``position_ids`` are supplied explicitly as ``arange(S)`` instead of letting
      ``get_rope_index()`` derive them from ``attention_mask.cumsum(-1) - 1``.  Under right
      padding those agree on the whole valid prefix, and passing them keeps the cumsum out
      of the graph.

    The mask input stays on the signature either way: it still selects the pooling index.
    """

    def __init__(self, base: torch.nn.Module, causal_attention: bool = False) -> None:
        super().__init__()
        self.base = base
        self.image_token_id = base.config.image_token_id
        self.causal_attention = causal_attention
        # transformers < 5 hangs the embedding off Qwen2VLModel directly; >= 5 moved it
        # down to Qwen2VLModel.language_model.  get_input_embeddings() is stable on both.
        self.embed_tokens = base.model.get_input_embeddings()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        image_embeds: torch.Tensor,
    ) -> torch.Tensor:
        inputs_embeds = self.embed_tokens(input_ids)
        image_mask = (input_ids.reshape(-1) == self.image_token_id).unsqueeze(1)
        flat_embeds = inputs_embeds.reshape(-1, inputs_embeds.shape[-1])
        flat_embeds = flat_embeds.masked_scatter(
            image_mask, image_embeds.to(dtype=flat_embeds.dtype).reshape(-1)
        )
        inputs_embeds = flat_embeds.reshape_as(inputs_embeds)
        if self.causal_attention:
            batch, seq_len = inputs_embeds.shape[0], inputs_embeds.shape[1]
            # Qwen2-VL's mrope wants [3, batch, seq].
            position_ids = (
                torch.arange(seq_len, device=inputs_embeds.device)
                .view(1, 1, -1)
                .expand(3, batch, -1)
            )
            outputs = self.base.model(
                input_ids=None,
                attention_mask=None,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                use_cache=False,
            )
        else:
            outputs = self.base.model(
                input_ids=None,
                attention_mask=attention_mask,
                position_ids=None,
                inputs_embeds=inputs_embeds,
                use_cache=False,
            )
        sequence_lengths = attention_mask.sum(dim=1) - 1
        embedding = outputs.last_hidden_state[
            torch.arange(outputs.last_hidden_state.shape[0]), sequence_lengths
        ]
        return functional.normalize(embedding, p=2, dim=1)


class GmeVisualEncoder(torch.nn.Module):
    """Qwen2-VL visual encoder that produces image-token embeddings."""

    def __init__(self, visual: torch.nn.Module) -> None:
        super().__init__()
        self.visual = visual
    def forward(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        return self.visual(pixel_values, grid_thw=image_grid_thw)


def prompt(text: str | None, has_image: bool, instruction: str = DEFAULT_INSTRUCTION) -> str:
    user_content = "<|vision_start|><|image_pad|><|vision_end|>" if has_image else ""
    if text is not None:
        user_content += text
    return (
        f"<|im_start|>system\n{instruction}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n<|endoftext|>"
    )


def processor_for(
    model_dir: Path,
    min_image_tokens: int = MIN_IMAGE_TOKENS,
    max_image_tokens: int = MAX_IMAGE_TOKENS,
) -> AutoProcessor:
    processor = AutoProcessor.from_pretrained(
        str(model_dir),
        min_pixels=min_image_tokens * 28 * 28,
        max_pixels=max_image_tokens * 28 * 28,
        use_fast=False,
    )
    processor.tokenizer.padding_side = "right"
    return processor


def make_example(
    processor: AutoProcessor,
    model_dir: Path,
    image_path: Path,
) -> dict[str, torch.Tensor]:
    sys.path.insert(0, str(model_dir))
    fetch_image = importlib.import_module("gme_inference").fetch_image

    image = fetch_image(str(image_path))
    inputs = processor(
        text=[prompt("A photo of a cat sitting on a sofa.", has_image=True)],
        images=[image],
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    return {name: value for name, value in inputs.items()}


def load_visual_encoder(model_dir: Path) -> torch.nn.Module:
    """Load only the visual weights, avoiding the 2B language-model allocation."""
    config = AutoConfig.from_pretrained(str(model_dir))
    visual = Qwen2VisionTransformerPretrainedModel._from_config(config.vision_config)
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    visual_keys = [name for name in index["weight_map"] if name.startswith("visual.")]
    shards = sorted({index["weight_map"][name] for name in visual_keys})
    state_dict: dict[str, torch.Tensor] = {}
    for shard_name in shards:
        with safe_open(model_dir / shard_name, framework="pt", device="cpu") as shard:
            for name in shard.keys():
                if name.startswith("visual."):
                    state_dict[name.removeprefix("visual.")] = shard.get_tensor(name)
    missing, unexpected = visual.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"visual checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    return visual.eval()


def label_ports(model: ov.Model, names: list[str]) -> None:
    for port, name in zip(model.inputs, names, strict=True):
        port.get_tensor().set_names({name})
    model.outputs[0].get_tensor().set_names({"embedding"})


@contextlib.contextmanager
def suppress_causal_mask_materialization():
    """Make HF hand SDPA ``attn_mask=None, is_causal=True`` while tracing.

    ``masking_utils._ignore_causal_mask_sdpa()`` short-circuits on ``not is_tracing(...)``,
    so under ``torch.jit.trace`` (which ``ov.convert_model`` uses) transformers *always*
    materializes an explicit S x S mask -- even when ``attention_mask is None``.  That is a
    deliberate guard against baking ``is_causal`` into an exported graph
    (pytorch/pytorch#108108), and it is the reason the shipped IR has 28 SDPA nodes with
    ``causal="false"`` plus a mask built from Select/masked_fill.

    Here baking it in is exactly what we want, so ``create_causal_mask`` is stubbed to
    ``None`` for the duration of the trace.  HF's own ``sdpa_attention_forward`` then
    computes ``is_causal = q_len > 1 and attention_mask is None``, so the flag comes from
    upstream logic rather than from us hand-editing the graph.

    Only ``create_causal_mask`` is patched for the mask itself: this checkpoint's 28 layers
    are all ``full_attention`` (no ``sliding_attention``), so the sliding-window builder
    never runs.

    ``use_gqa_in_sdpa`` is pinned to False for a second, GPU-plugin-specific reason.  It
    returns True as soon as ``attention_mask is None``, which makes transformers skip
    ``repeat_kv`` and pass ``enable_gqa=True`` instead -- and that exports a **rank-5** SDPA
    (Q ``[B,2,6,S,128]``, K/V ``[B,2,1,S,128]``).  The Intel GPU plugin does not lower a
    rank-5 generic ScaledDotProductAttention: it decomposes it into explicit MatMul +
    Softmax, measured at 428 ms vs 240 ms for the shipped IR.  Keeping ``repeat_kv`` holds
    SDPA at rank 4, where it lowers to ``ocl::sdpa::opt``, and the resulting
    Unsqueeze/Broadcast/Reshape is precisely what opt-1 already eliminates in the plugin.
    """
    from transformers.integrations import sdpa_attention
    from transformers.models.qwen2_vl import modeling_qwen2_vl as qwen2_vl

    saved_mask = qwen2_vl.create_causal_mask
    saved_gqa = sdpa_attention.use_gqa_in_sdpa
    qwen2_vl.create_causal_mask = lambda **_kwargs: None
    sdpa_attention.use_gqa_in_sdpa = lambda *_args, **_kwargs: False
    try:
        yield
    finally:
        qwen2_vl.create_causal_mask = saved_mask
        sdpa_attention.use_gqa_in_sdpa = saved_gqa


def export_model(
    module: torch.nn.Module,
    example_inputs: tuple[torch.Tensor, ...],
    output_path: Path,
    input_names: list[str],
    shapes: list[ov.PartialShape],
    causal_attention: bool = False,
) -> None:
    print(f"[export] converting {output_path.name} ...")
    start = time.monotonic()
    with contextlib.ExitStack() as stack:
        if causal_attention:
            stack.enter_context(suppress_causal_mask_materialization())
        model = ov.convert_model(module, example_input=example_inputs)
    label_ports(model, input_names)
    model.reshape(dict(enumerate(shapes)))
    ov.save_model(model, str(output_path), compress_to_fp16=False)
    print(f"[export] saved {output_path.name} in {time.monotonic() - start:.1f}s")


def link_or_copy(source: Path, destination: Path) -> None:
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def package_processor_assets(source: Path, output: Path, runtime_module: Path) -> None:
    asset_names = [
        "config.json",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "chat_template.json",
        "vocab.json",
        "merges.txt",
    ]
    for name in asset_names:
        asset = source / name
        if asset.exists():
            shutil.copy2(asset, output / name)
    destination = output / "gme_openvino.py"
    if runtime_module.resolve() != destination.resolve():
        shutil.copy2(runtime_module, destination)


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = left.reshape(-1).astype(np.float64)
    right = right.reshape(-1).astype(np.float64)
    return float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)))


def validate_pipeline(
    output: Path,
    device: str,
    language_reference: GmeLanguageEmbed,
    visual_reference: GmeVisualEncoder,
    inputs: dict[str, torch.Tensor],
) -> float:
    with torch.no_grad():
        golden_visual = visual_reference(inputs["pixel_values"], inputs["image_grid_thw"])
        golden = language_reference(
            inputs["input_ids"],
            inputs["attention_mask"],
            golden_visual,
        ).cpu().numpy()

    core = ov.Core()
    if device not in core.available_devices:
        raise RuntimeError(f"requested device {device!r} is unavailable: {core.available_devices}")
    compile_config = {hint.inference_precision: ov.Type.f32}
    visual = core.compile_model(core.read_model(str(output / "gme_visual_fp32.xml")), device, compile_config)
    visual_result = visual(
        {
            "pixel_values": inputs["pixel_values"].numpy(),
            "image_grid_thw": inputs["image_grid_thw"].numpy(),
        }
    )
    visual_embeddings = np.asarray(visual_result[visual.output(0)])
    language = core.compile_model(
        core.read_model(str(output / "gme_language_fp32.xml")), device, compile_config
    )
    result = language(
        {
            "input_ids": inputs["input_ids"].numpy(),
            "attention_mask": inputs["attention_mask"].numpy(),
            "image_embeds": visual_embeddings,
        }
    )
    actual = np.asarray(result[language.output(0)])
    value = cosine(actual, golden)
    print(f"[validate] image/fused FP32 pipeline on {device}: cosine={value:.8f}")
    if value < 0.9999:
        raise RuntimeError("OpenVINO embedding differs from the official model")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-model",
        type=Path,
        default=DEFAULT_SOURCE_MODEL,
        help="directory containing the GME checkpoint and gme_inference.py",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--example-image",
        type=Path,
        help="image used for the export/validation trace (default: <source-model>/images/gme_logo.png)",
    )
    parser.add_argument(
        "--runtime-module",
        type=Path,
        default=HERE / "gme_openvino.py",
        help="runtime module copied into the exported package",
    )
    parser.add_argument("--device", default="CPU", help="OpenVINO device used for the post-export check")
    parser.add_argument("--force", action="store_true", help="re-export models that already exist")
    parser.add_argument(
        "--stage",
        choices=("all", "language", "visual", "validate"),
        default="all",
        help="all runs the isolated stages in separate processes to limit peak memory",
    )
    parser.add_argument(
        "--language-suffix",
        default=None,
        help="suffix for the language IR filename, e.g. --language-suffix _v5ctrl writes "
             "gme_language_fp32_v5ctrl.xml; defaults to _causal with --causal-sdpa and "
             "empty otherwise. Lets a control and an experiment IR coexist for A/B.",
    )
    parser.add_argument(
        "--causal-sdpa",
        action="store_true",
        help="export the language IR with causal=True SDPA and no explicit attention mask, "
             "into gme_language_fp32_causal.xml (an identity for this right-padded, "
             "last-valid-token-pooled package -- see GmeLanguageEmbed)",
    )
    args = parser.parse_args()

    source_model = args.source_model.expanduser().resolve()
    output = args.output.expanduser().resolve()
    example_image = (args.example_image or source_model / "images" / "gme_logo.png").expanduser().resolve()
    runtime_module = args.runtime_module.expanduser().resolve()
    if not source_model.is_dir():
        raise SystemExit(f"source model directory not found: {source_model}")
    if not example_image.is_file():
        raise SystemExit(f"example image not found: {example_image}")
    if not runtime_module.is_file():
        raise SystemExit(f"runtime module not found: {runtime_module}")
    output.mkdir(parents=True, exist_ok=True)
    package_processor_assets(source_model, output, runtime_module)

    if args.stage == "all":
        # "language" runs twice: once for the mask-carrying IR that every earlier round was
        # measured on, once for the causal IR that docs/09 adopts. Both are products -- the
        # first keeps the historical baseline reproducible, the second is the recommended
        # default -- so an in-order 01..05 run has to produce both.
        stages: list[tuple[str, list[str]]] = [
            ("language", []),
            ("language", ["--causal-sdpa"]),
            ("visual", []),
            ("validate", []),
        ]
        for stage, extra in stages:
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--output",
                str(output),
                "--source-model",
                str(source_model),
                "--example-image",
                str(example_image),
                "--runtime-module",
                str(runtime_module),
                "--device",
                args.device,
                "--stage",
                stage,
                *extra,
            ]
            if args.force and stage != "validate":
                command.append("--force")
            label = f"{stage}{' (causal)' if extra else ''}"
            print(f"[export] starting isolated {label} stage")
            subprocess.run(command, check=True)
        return

    export_processor = processor_for(source_model, MIN_IMAGE_TOKENS, MIN_IMAGE_TOKENS)
    vision_inputs = make_example(export_processor, source_model, example_image)

    # The causal variant is written to its own file so both IRs coexist and can be A/B'd
    # inside one build/session, the way every other measurement in this project is done.
    suffix = args.language_suffix if args.language_suffix is not None else (
        "_causal" if args.causal_sdpa else "")
    language_path = output / f"gme_language_fp32{suffix}.xml"
    if args.stage == "language" and not args.force and language_path.exists():
        print(f"[export] {language_path.name} exists, skipping (--force to re-export)")
    if args.stage == "language" and (args.force or not language_path.exists()):
        print(f"[export] loading language checkpoint from {source_model}")
        base = _AutoVLM.from_pretrained(str(source_model), torch_dtype=torch.float32)
        base.eval()
        # transformers >= 5 moved the text dims onto config.text_config.
        hidden_size = getattr(base.config, "hidden_size", None) or base.config.get_text_config().hidden_size
        language_module = GmeLanguageEmbed(base, causal_attention=args.causal_sdpa).eval()
        export_model(
            language_module,
            (
                vision_inputs["input_ids"],
                vision_inputs["attention_mask"],
                torch.empty((1260, hidden_size), dtype=torch.float32),
            ),
            language_path,
            ["input_ids", "attention_mask", "image_embeds"],
            [
                ov.PartialShape([-1, -1]),
                ov.PartialShape([-1, -1]),
                ov.PartialShape([-1, hidden_size]),
            ],
            causal_attention=args.causal_sdpa,
        )

    visual_path = output / "gme_visual_fp32.xml"
    if args.stage == "visual" and not args.force and visual_path.exists():
        print(f"[export] {visual_path.name} exists, skipping (--force to re-export)")
    if args.stage == "visual" and (args.force or not visual_path.exists()):
        print(f"[export] loading visual-only checkpoint from {source_model}")
        visual_module = GmeVisualEncoder(load_visual_encoder(source_model)).eval()
        export_model(
            visual_module,
            (vision_inputs["pixel_values"], vision_inputs["image_grid_thw"]),
            visual_path,
            ["pixel_values", "image_grid_thw"],
            [ov.PartialShape([-1, 1176]), ov.PartialShape([-1, 3])],
        )

    if args.stage != "validate":
        return
    if not language_path.exists() or not visual_path.exists():
        raise RuntimeError("both language and visual IRs must exist before validation")
    print(f"[validate] loading reference checkpoint from {source_model}")
    base = _AutoVLM.from_pretrained(str(source_model), torch_dtype=torch.float32)
    base.eval()
    language_module = GmeLanguageEmbed(base).eval()
    # >= 5 moved the visual tower under Qwen2VLModel.
    visual_module = GmeVisualEncoder(getattr(base, "visual", None) or base.model.visual).eval()
    validation_inputs = make_example(processor_for(source_model), source_model, example_image)
    config_path = output / "gme_openvino_config.json"
    report = json.loads(config_path.read_text()) if config_path.exists() else {}
    report.update({
        "format": 1,
        "model": "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",
        "embedding_dim": 1536,
        "pooling": "last valid token, L2 normalized",
        "language_model": language_path.name,
        "visual_model": visual_path.name,
    })
    processor = dict(report.get("processor", {}))
    processor.update({
        "min_image_tokens": MIN_IMAGE_TOKENS,
        "max_image_tokens": MAX_IMAGE_TOKENS,
        "max_length": MAX_LENGTH,
        "padding_side": "right",
    })
    report["processor"] = processor
    report["vision_fp32_reference_cosine"] = validate_pipeline(
        output, args.device, language_module, visual_module, validation_inputs
    )
    config_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[export] complete OpenVINO package: {output}")


if __name__ == "__main__":
    main()