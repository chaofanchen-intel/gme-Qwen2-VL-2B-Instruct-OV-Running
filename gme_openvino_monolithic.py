"""Runtime for the single-IR GME image/fused embedding experiment.

``gme_monolithic_<precision>.xml`` contains the visual encoder through final
embedding output. Text/query have no visual tensor to feed this graph, so they
lazily use the preserved dual-IR language-only runtime instead of executing a
dummy vision input.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import openvino as ov
from openvino.properties import hint
from transformers import AutoProcessor

from gme_openvino import DEFAULT_INSTRUCTION, IMAGE_FACTOR, GmeOpenVINO, _as_list, load_image


class GmeOpenVINOMonolithic:
    """GME runtime using one IR for image/fused inference and dual IR for text."""

    def __init__(self, model_dir: str | Path, device: str = "CPU", precision: str = "fp32") -> None:
        self.model_dir = Path(model_dir)
        self.device = device
        self.config = json.loads((self.model_dir / "gme_openvino_config.json").read_text())
        models = self.config.get("monolithic_models")
        if models is None:
            models = {"fp32": self.config.get("monolithic_model", {}).get("model", "gme_monolithic_fp32.xml")}
        if precision not in models:
            raise ValueError(f"monolithic precision {precision!r} unavailable; choose from {list(models)}")
        self.precision = precision
        self.model_path = self.model_dir / models[precision]
        if not self.model_path.exists():
            raise FileNotFoundError(f"monolithic IR is missing: {self.model_path}")
        processor_config = self.config["processor"]
        self.processor = AutoProcessor.from_pretrained(
            str(self.model_dir),
            min_pixels=processor_config["min_image_tokens"] * IMAGE_FACTOR * IMAGE_FACTOR,
            max_pixels=processor_config["max_image_tokens"] * IMAGE_FACTOR * IMAGE_FACTOR,
            use_fast=False,
        )
        self.processor.tokenizer.padding_side = processor_config["padding_side"]
        self.max_length = processor_config["max_length"]
        self.core = ov.Core()
        if device not in self.core.available_devices:
            raise RuntimeError(f"OpenVINO device {device!r} unavailable; found {self.core.available_devices}")
        compile_config = {hint.inference_precision: ov.Type.f32} if precision == "fp32" else {}
        self.monolithic = self.core.compile_model(self.model_path, device, compile_config)
        self._text_runtime: GmeOpenVINO | None = None

    @staticmethod
    def _instruction(instruction: str | None, is_query: bool) -> str:
        return instruction if is_query and instruction is not None else DEFAULT_INSTRUCTION

    @staticmethod
    def _prompt(text: str | None, has_image: bool, instruction: str) -> str:
        return GmeOpenVINO._prompt(text, has_image, instruction)

    def _text(self) -> GmeOpenVINO:
        if self._text_runtime is None:
            self._text_runtime = GmeOpenVINO(self.model_dir, self.device, precision=self.precision)
        return self._text_runtime

    def _encode_with_image(self, text: str | None, image, instruction: str) -> np.ndarray:
        inputs = self.processor(
            text=[self._prompt(text, has_image=True, instruction=instruction)],
            images=[load_image(image)],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )
        result = self.monolithic(
            {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "pixel_values": inputs["pixel_values"],
                "image_grid_thw": inputs["image_grid_thw"],
            }
        )
        return np.asarray(result[self.monolithic.output(0)], dtype=np.float32)

    def get_text_embeddings(self, texts: str | Sequence[str], instruction: str | None = None, is_query: bool = True) -> np.ndarray:
        """Use the preserved dual-IR language-only path for text/query."""
        return self._text().get_text_embeddings(texts, instruction=instruction, is_query=is_query)

    def get_image_embeddings(self, images, instruction: str | None = None, is_query: bool = True) -> np.ndarray:
        values = _as_list(images)
        if not values:
            raise ValueError("images must be a non-empty image or image sequence")
        prompt = self._instruction(instruction, is_query)
        return np.concatenate([self._encode_with_image(None, image, prompt) for image in values], axis=0)

    def get_fused_embeddings(
        self,
        texts: str | Sequence[str] | None = None,
        images=None,
        instruction: str | None = None,
        is_query: bool = True,
    ) -> np.ndarray:
        text_values = _as_list(texts)
        image_values = _as_list(images)
        prompt = self._instruction(instruction, is_query)
        if image_values is None:
            if text_values is None:
                raise ValueError("provide texts, images, or both")
            return self._text().get_text_embeddings(text_values, instruction=instruction, is_query=is_query)
        if text_values is None:
            return np.concatenate([self._encode_with_image(None, image, prompt) for image in image_values], axis=0)
        if len(text_values) != len(image_values):
            raise ValueError("texts and images must have the same length")
        return np.concatenate(
            [self._encode_with_image(text, image, prompt) for text, image in zip(text_values, image_values)], axis=0
        )