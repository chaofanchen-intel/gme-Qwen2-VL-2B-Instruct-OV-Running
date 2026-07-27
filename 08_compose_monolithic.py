#!/usr/bin/env python
"""Export one FP32 OpenVINO IR for GME image/fused embedding.

``--method torch`` traces the visual encoder, visual-token injection, decoder,
pooling and normalization in one PyTorch module. It uses a minimum 256-token
visual example to control conversion memory. The resulting IR is for image and
fused embedding; text-only requests should retain the dual-IR language-only fast
path because a monolithic graph must still execute a valid visual input.

``--method graph`` preserves an experiment that attempts to connect existing
IRs. The graph validates in memory but OpenVINO 2026.2 cannot serialize cross-
model ownership, so it is deliberately not the default.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import openvino as ov
from openvino.properties import hint
import torch
import torch.nn.functional as functional
from transformers import AutoModelForVision2Seq, AutoProcessor

from gme_openvino import GmeOpenVINO, load_image

HERE = Path(__file__).resolve().parent
PACKAGE = HERE / "ov_models" / "gme-qwen2-vl-2b-instruct-openvino"
ART = HERE / "artifacts"
MONOLITHIC_XML = PACKAGE / "gme_monolithic_fp32.xml"
SOURCE_MODEL = HERE.parent / "benchmark" / "models" / "gme-Qwen2-VL-2B-Instruct"
MIN_IMAGE_TOKENS = 256
MAX_LENGTH = 1800


class GmeMonolithicEmbed(torch.nn.Module):
    """Official GME visual + language embedding computation in one module."""

    def __init__(self, base: torch.nn.Module) -> None:
        super().__init__()
        self.base = base
        self.image_token_id = base.config.image_token_id

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        inputs_embeds = self.base.model.embed_tokens(input_ids)
        image_embeds = self.base.visual(pixel_values, grid_thw=image_grid_thw)
        image_mask = (input_ids.reshape(-1) == self.image_token_id).unsqueeze(1)
        flat_embeds = inputs_embeds.reshape(-1, inputs_embeds.shape[-1])
        flat_embeds = flat_embeds.masked_scatter(
            image_mask, image_embeds.to(dtype=flat_embeds.dtype).reshape(-1)
        )
        outputs = self.base.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=None,
            inputs_embeds=flat_embeds.reshape_as(inputs_embeds),
            use_cache=False,
        )
        sequence_lengths = attention_mask.sum(dim=1) - 1
        embedding = outputs.last_hidden_state[
            torch.arange(outputs.last_hidden_state.shape[0]), sequence_lengths
        ]
        return functional.normalize(embedding, p=2, dim=1)


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = left.reshape(-1).astype(np.float64)
    right = right.reshape(-1).astype(np.float64)
    return float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)))


def compose_graph(force: bool) -> None:
    if MONOLITHIC_XML.exists() and not force:
        print(f"[compose] {MONOLITHIC_XML.name} exists, skip")
        return

    core = ov.Core()
    language = core.read_model(PACKAGE / "gme_language_fp32.xml")
    visual = core.read_model(PACKAGE / "gme_visual_fp32.xml")

    # Both tensors are dynamic [K,1536]. Replacing the language Parameter
    # preserves every decoder/pooling op and incorporates the visual graph.
    for target in list(language.input("image_embeds").get_target_inputs()):
        target.replace_source_output(visual.output(0))
    parameters = [
        language.input("input_ids").get_node(),
        language.input("attention_mask").get_node(),
        visual.input("pixel_values").get_node(),
        visual.input("image_grid_thw").get_node(),
    ]
    combined = ov.Model(language.outputs, parameters, "gme_monolithic_embedding")
    combined.validate_nodes_and_infer_types()
    combined.output(0).get_tensor().set_names({"embedding"})
    ov.save_model(combined, MONOLITHIC_XML, compress_to_fp16=False)
    print(f"[compose] saved {MONOLITHIC_XML.name} ({len(combined.get_ordered_ops())} ops)")


def compose_torch(force: bool) -> None:
    if MONOLITHIC_XML.exists() and not force:
        print(f"[compose] {MONOLITHIC_XML.name} exists, skip")
        return
    print(f"[compose] loading complete GME checkpoint from {SOURCE_MODEL}")
    base = AutoModelForVision2Seq.from_pretrained(str(SOURCE_MODEL), torch_dtype=torch.float32)
    module = GmeMonolithicEmbed(base.eval()).eval()
    processor = AutoProcessor.from_pretrained(
        str(SOURCE_MODEL),
        min_pixels=MIN_IMAGE_TOKENS * 28 * 28,
        max_pixels=MIN_IMAGE_TOKENS * 28 * 28,
        use_fast=False,
    )
    processor.tokenizer.padding_side = "right"
    image = load_image(str(SOURCE_MODEL / "images" / "gme_logo.png"))
    inputs = processor(
        text=[GmeOpenVINO._prompt("A photo of a cat sitting on a sofa.", True, "You are a helpful assistant.")],
        images=[image], padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt",
    )
    print(f"[compose] converting monolithic graph with {inputs['pixel_values'].shape[0] // 4} visual tokens ...")
    model = ov.convert_model(
        module,
        example_input=(
            inputs["input_ids"], inputs["attention_mask"],
            inputs["pixel_values"], inputs["image_grid_thw"],
        ),
    )
    for port, name in zip(model.inputs, ["input_ids", "attention_mask", "pixel_values", "image_grid_thw"], strict=True):
        port.get_tensor().set_names({name})
    model.output(0).get_tensor().set_names({"embedding"})
    model.reshape({
        0: ov.PartialShape([-1, -1]),
        1: ov.PartialShape([-1, -1]),
        2: ov.PartialShape([-1, 1176]),
        3: ov.PartialShape([-1, 3]),
    })
    ov.save_model(model, MONOLITHIC_XML, compress_to_fp16=False)
    print(f"[compose] saved {MONOLITHIC_XML.name} ({len(model.get_ordered_ops())} ops)")


def validate(device: str) -> float:
    meta = json.loads((ART / "golden_meta.json").read_text())
    encoder = GmeOpenVINO(PACKAGE, device=device, precision="fp32")
    image = load_image(meta["image"])
    prompt = encoder._prompt(meta["texts"][0], has_image=True, instruction=encoder._instruction(None, True))
    inputs = encoder.processor(
        text=[prompt], images=[image], padding=True, truncation=True,
        max_length=encoder.max_length, return_tensors="np",
    )

    core = ov.Core()
    if device not in core.available_devices:
        raise RuntimeError(f"requested device {device!r} unavailable: {core.available_devices}")
    compiled = core.compile_model(MONOLITHIC_XML, device, {hint.inference_precision: ov.Type.f32})
    result = compiled(
        {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "pixel_values": inputs["pixel_values"],
            "image_grid_thw": inputs["image_grid_thw"],
        }
    )
    actual = np.asarray(result[compiled.output(0)])
    golden = np.load(ART / "golden_fused.npy")
    value = cosine(actual, golden)
    print(f"[validate] monolithic fused FP32 {device}: cosine={value:.8f}")
    if value < 0.9999:
        raise RuntimeError("monolithic IR does not match the GME golden embedding")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--method", choices=("torch", "graph"), default="torch")
    args = parser.parse_args()
    if args.method == "torch":
        compose_torch(args.force)
    else:
        compose_graph(args.force)
    value = validate(args.device)
    config_path = PACKAGE / "gme_openvino_config.json"
    config = json.loads(config_path.read_text())
    config["monolithic_model"] = {
        "model": MONOLITHIC_XML.name,
        "precision": "fp32",
        "modes": ["image", "fused"],
        "text_query_behavior": "use the retained language-only dual-IR path",
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    report = {"model": MONOLITHIC_XML.name, "device": args.device, "fused_cosine": value}
    (ART / "monolithic_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"[validate] wrote {ART / 'monolithic_validation.json'}")


if __name__ == "__main__":
    main()