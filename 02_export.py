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
from transformers import AutoConfig, AutoModelForVision2Seq, AutoProcessor
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VisionTransformerPretrainedModel
from safetensors import safe_open
from openvino.properties import hint

HERE = Path(__file__).resolve().parent
SOURCE_MODEL = (HERE.parent / "benchmark" / "models" / "gme-Qwen2-VL-2B-Instruct").resolve()
DEFAULT_OUTPUT = HERE / "ov_models" / "gme-qwen2-vl-2b-instruct-openvino"
DEFAULT_INSTRUCTION = "You are a helpful assistant."
MIN_IMAGE_TOKENS = 256
MAX_IMAGE_TOKENS = 1280
MAX_LENGTH = 1800


class GmeLanguageEmbed(torch.nn.Module):
    """Decoder path for text, image, and fused embeddings."""

    def __init__(self, base: torch.nn.Module) -> None:
        super().__init__()
        self.base = base
        self.image_token_id = base.config.image_token_id

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        image_embeds: torch.Tensor,
    ) -> torch.Tensor:
        inputs_embeds = self.base.model.embed_tokens(input_ids)
        image_mask = (input_ids.reshape(-1) == self.image_token_id).unsqueeze(1)
        flat_embeds = inputs_embeds.reshape(-1, inputs_embeds.shape[-1])
        flat_embeds = flat_embeds.masked_scatter(
            image_mask, image_embeds.to(dtype=flat_embeds.dtype).reshape(-1)
        )
        inputs_embeds = flat_embeds.reshape_as(inputs_embeds)
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


def make_example(processor: AutoProcessor, model_dir: Path) -> dict[str, torch.Tensor]:
    sys.path.insert(0, str(model_dir))
    fetch_image = importlib.import_module("gme_inference").fetch_image

    image = fetch_image(str(model_dir / "images" / "gme_logo.png"))
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


def export_model(
    module: torch.nn.Module,
    example_inputs: tuple[torch.Tensor, ...],
    output_path: Path,
    input_names: list[str],
    shapes: list[ov.PartialShape],
) -> None:
    print(f"[export] converting {output_path.name} ...")
    start = time.monotonic()
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


def package_processor_assets(source: Path, output: Path) -> None:
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
    shutil.copy2(HERE / "gme_openvino.py", output / "gme_openvino.py")


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="CPU", help="OpenVINO device used for the post-export check")
    parser.add_argument("--force", action="store_true", help="re-export models that already exist")
    parser.add_argument(
        "--stage",
        choices=("all", "language", "visual", "validate"),
        default="all",
        help="all runs the isolated stages in separate processes to limit peak memory",
    )
    args = parser.parse_args()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    package_processor_assets(SOURCE_MODEL, output)

    if args.stage == "all":
        for stage in ("language", "visual", "validate"):
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--output",
                str(output),
                "--device",
                args.device,
                "--stage",
                stage,
            ]
            if args.force and stage != "validate":
                command.append("--force")
            print(f"[export] starting isolated {stage} stage")
            subprocess.run(command, check=True)
        return

    export_processor = processor_for(SOURCE_MODEL, MIN_IMAGE_TOKENS, MIN_IMAGE_TOKENS)
    vision_inputs = make_example(export_processor, SOURCE_MODEL)

    language_path = output / "gme_language_fp32.xml"
    if args.stage == "language" and (args.force or not language_path.exists()):
        print(f"[export] loading language checkpoint from {SOURCE_MODEL}")
        base = AutoModelForVision2Seq.from_pretrained(str(SOURCE_MODEL), torch_dtype=torch.float32)
        base.eval()
        language_module = GmeLanguageEmbed(base).eval()
        export_model(
            language_module,
            (
                vision_inputs["input_ids"],
                vision_inputs["attention_mask"],
                torch.empty((1260, base.config.hidden_size), dtype=torch.float32),
            ),
            language_path,
            ["input_ids", "attention_mask", "image_embeds"],
            [
                ov.PartialShape([-1, -1]),
                ov.PartialShape([-1, -1]),
                ov.PartialShape([-1, base.config.hidden_size]),
            ],
        )

    visual_path = output / "gme_visual_fp32.xml"
    if args.stage == "visual" and (args.force or not visual_path.exists()):
        print(f"[export] loading visual-only checkpoint from {SOURCE_MODEL}")
        visual_module = GmeVisualEncoder(load_visual_encoder(SOURCE_MODEL)).eval()
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
    print(f"[validate] loading reference checkpoint from {SOURCE_MODEL}")
    base = AutoModelForVision2Seq.from_pretrained(str(SOURCE_MODEL), torch_dtype=torch.float32)
    base.eval()
    language_module = GmeLanguageEmbed(base).eval()
    visual_module = GmeVisualEncoder(base.visual).eval()
    validation_inputs = make_example(processor_for(SOURCE_MODEL), SOURCE_MODEL)
    report = {
        "format": 1,
        "model": "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",
        "embedding_dim": 1536,
        "pooling": "last valid token, L2 normalized",
        "language_model": language_path.name,
        "visual_model": visual_path.name,
        "processor": {
            "min_image_tokens": MIN_IMAGE_TOKENS,
            "max_image_tokens": MAX_IMAGE_TOKENS,
            "max_length": MAX_LENGTH,
            "padding_side": "right",
        },
    }
    report["vision_fp32_reference_cosine"] = validate_pipeline(
        output, args.device, language_module, visual_module, validation_inputs
    )
    (output / "gme_openvino_config.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"[export] complete OpenVINO package: {output}")


if __name__ == "__main__":
    main()