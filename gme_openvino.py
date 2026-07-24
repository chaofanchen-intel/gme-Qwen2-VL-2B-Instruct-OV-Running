"""OpenVINO Runtime implementation of Alibaba-NLP/gme-Qwen2-VL-2B-Instruct.

The exported package contains the learned computation in two OpenVINO IRs. This
module deliberately retains Hugging Face's Qwen2-VL processor for tokenization
and pixel preprocessing, which is necessary to exactly preserve GME's special
tokens, dynamic image resizing, and image-token expansion.
"""
from __future__ import annotations

import base64
import math
from io import BytesIO
from pathlib import Path
from typing import Sequence

import numpy as np
import openvino as ov
import requests
from PIL import Image
from transformers import AutoProcessor
from openvino.properties import hint

DEFAULT_INSTRUCTION = "You are a helpful assistant."
IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200


def _as_list(value: str | Image.Image | Sequence[str | Image.Image] | None) -> list | None:
    if value is None:
        return None
    if isinstance(value, (str, Image.Image)):
        return [value]
    return list(value)


def _round_by_factor(value: int, factor: int) -> int:
    return round(value / factor) * factor


def _resize_for_gme(image: Image.Image) -> Image.Image:
    width, height = image.size
    resized_height = max(IMAGE_FACTOR, _round_by_factor(height, IMAGE_FACTOR))
    resized_width = max(IMAGE_FACTOR, _round_by_factor(width, IMAGE_FACTOR))
    if resized_height * resized_width > MAX_PIXELS:
        scale = math.sqrt((height * width) / MAX_PIXELS)
        resized_height = math.floor(height / scale / IMAGE_FACTOR) * IMAGE_FACTOR
        resized_width = math.floor(width / scale / IMAGE_FACTOR) * IMAGE_FACTOR
    elif resized_height * resized_width < MIN_PIXELS:
        scale = math.sqrt(MIN_PIXELS / (height * width))
        resized_height = math.ceil(height * scale / IMAGE_FACTOR) * IMAGE_FACTOR
        resized_width = math.ceil(width * scale / IMAGE_FACTOR) * IMAGE_FACTOR
    if max(resized_height, resized_width) / min(resized_height, resized_width) > MAX_RATIO:
        raise ValueError(f"image aspect ratio must not exceed {MAX_RATIO}:1")
    return image.resize((resized_width, resized_height))


def load_image(value: str | Image.Image) -> Image.Image:
    """Load and resize an image exactly as the official GME inference helper."""
    if isinstance(value, Image.Image):
        image = value
    elif value.startswith(("http://", "https://")):
        image = Image.open(requests.get(value, stream=True, timeout=30).raw)
    elif value.startswith("file://"):
        image = Image.open(value[7:])
    elif value.startswith("data:image") and "base64," in value:
        image = Image.open(BytesIO(base64.b64decode(value.split("base64,", 1)[1])))
    else:
        image = Image.open(value)
    return _resize_for_gme(image.convert("RGB"))


class GmeOpenVINO:
    """GME encoder backed entirely by OpenVINO Runtime neural computation.

    One image per input is supported, matching the official model limitation.
    Image-containing batches are evaluated one sample at a time because the
    visual IR is exported for the model's one-image-per-sample contract; text
    inputs are batched in one language inference.
    """

    def __init__(
        self,
        model_dir: str | Path,
        device: str = "CPU",
        precision: str | None = None,
        fp32_math: bool | None = None,
    ) -> None:
        """precision: one of the keys in the package config's ``precisions`` map
        (e.g. ``fp32``/``int8``/``int4``); defaults to the package ``default_precision``.
        fp32_math: force f32 inference-precision hint (defaults to True only for the
        fp32 weights, so the reference is reproduced exactly; quantized models use the
        device default for best speed)."""
        self.model_dir = Path(model_dir)
        config_path = self.model_dir / "gme_openvino_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"not an exported GME OpenVINO package: {self.model_dir}")
        self.config = __import__("json").loads(config_path.read_text())
        self.device = device

        precisions = self.config.get("precisions")
        if precisions:
            precision = precision or self.config.get("default_precision", "int8")
            if precision not in precisions:
                raise ValueError(f"precision {precision!r} not in {list(precisions)}")
            language_model = precisions[precision]["language"]
            visual_model = precisions[precision]["visual"]
        else:  # backward compatibility with older single-precision packages
            precision = precision or "fp32"
            language_model = self.config["language_model"]
            visual_model = self.config["visual_model"]
        self.precision = precision

        processor_config = self.config["processor"]
        self.processor = AutoProcessor.from_pretrained(
            str(self.model_dir),
            min_pixels=processor_config["min_image_tokens"] * IMAGE_FACTOR * IMAGE_FACTOR,
            max_pixels=processor_config["max_image_tokens"] * IMAGE_FACTOR * IMAGE_FACTOR,
            use_fast=False,
        )
        self.processor.tokenizer.padding_side = self.config["processor"]["padding_side"]
        self.max_length = self.config["processor"]["max_length"]
        self.core = ov.Core()
        if device not in self.core.available_devices:
            raise RuntimeError(f"OpenVINO device {device!r} unavailable; found {self.core.available_devices}")
        if fp32_math is None:
            fp32_math = precision == "fp32"
        compile_config = {hint.inference_precision: ov.Type.f32} if fp32_math else {}
        self.language = self.core.compile_model(self.model_dir / language_model, device, compile_config)
        self.visual = self.core.compile_model(self.model_dir / visual_model, device, compile_config)

    @staticmethod
    def _prompt(text: str | None, has_image: bool, instruction: str) -> str:
        content = "<|vision_start|><|image_pad|><|vision_end|>" if has_image else ""
        if text is not None:
            content += text
        return (
            f"<|im_start|>system\n{instruction}<|im_end|>\n"
            f"<|im_start|>user\n{content}<|im_end|>\n"
            f"<|im_start|>assistant\n<|endoftext|>"
        )

    @staticmethod
    def _instruction(instruction: str | None, is_query: bool) -> str:
        return instruction if is_query and instruction is not None else DEFAULT_INSTRUCTION

    @staticmethod
    def _output(result: dict, request: ov.CompiledModel) -> np.ndarray:
        return np.asarray(result[request.output(0)], dtype=np.float32)

    def _language_embedding(self, inputs: dict[str, np.ndarray], image_embeds: np.ndarray) -> np.ndarray:
        result = self.language(
            {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "image_embeds": image_embeds,
            }
        )
        return self._output(result, self.language)

    def _encode_text_batch(self, texts: list[str], instruction: str) -> np.ndarray:
        prompts = [self._prompt(text, has_image=False, instruction=instruction) for text in texts]
        inputs = self.processor(
            text=prompts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )
        empty_images = np.empty((0, self.config["embedding_dim"]), dtype=np.float32)
        return self._language_embedding(inputs, empty_images)

    def _encode_with_image(self, text: str | None, image: str | Image.Image, instruction: str) -> np.ndarray:
        inputs = self.processor(
            text=[self._prompt(text, has_image=True, instruction=instruction)],
            images=[load_image(image)],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )
        visual_result = self.visual(
            {
                "pixel_values": inputs["pixel_values"],
                "image_grid_thw": inputs["image_grid_thw"],
            }
        )
        image_embeds = self._output(visual_result, self.visual)
        return self._language_embedding(inputs, image_embeds)

    def get_text_embeddings(
        self,
        texts: str | Sequence[str],
        instruction: str | None = None,
        is_query: bool = True,
    ) -> np.ndarray:
        values = _as_list(texts)
        if not values or not all(isinstance(text, str) for text in values):
            raise ValueError("texts must be a non-empty string or sequence of strings")
        return self._encode_text_batch(values, self._instruction(instruction, is_query))

    def get_image_embeddings(
        self,
        images: str | Image.Image | Sequence[str | Image.Image],
        instruction: str | None = None,
        is_query: bool = True,
    ) -> np.ndarray:
        values = _as_list(images)
        if not values:
            raise ValueError("images must be a non-empty image or image sequence")
        prompt = self._instruction(instruction, is_query)
        return np.concatenate([self._encode_with_image(None, image, prompt) for image in values], axis=0)

    def get_fused_embeddings(
        self,
        texts: str | Sequence[str] | None = None,
        images: str | Image.Image | Sequence[str | Image.Image] | None = None,
        instruction: str | None = None,
        is_query: bool = True,
    ) -> np.ndarray:
        text_values = _as_list(texts)
        image_values = _as_list(images)
        prompt = self._instruction(instruction, is_query)
        if image_values is None:
            if text_values is None:
                raise ValueError("provide texts, images, or both")
            return self._encode_text_batch(text_values, prompt)
        if text_values is None:
            return np.concatenate([self._encode_with_image(None, image, prompt) for image in image_values], axis=0)
        if len(text_values) != len(image_values):
            raise ValueError("texts and images must have the same length")
        return np.concatenate(
            [self._encode_with_image(text, image, prompt) for text, image in zip(text_values, image_values)],
            axis=0,
        )

    def encode(
        self,
        sentences: str | Sequence[str],
        *,
        instruction: str | None = None,
        is_query: bool = True,
    ) -> np.ndarray:
        """Compatibility alias for text encoding used by embedding clients."""
        return self.get_text_embeddings(sentences, instruction=instruction, is_query=is_query)

    def encode_queries(
        self,
        queries: str | Sequence[str],
        *,
        instruction: str | None = None,
    ) -> np.ndarray:
        return self.get_text_embeddings(queries, instruction=instruction, is_query=True)

    def encode_corpus(self, corpus: Sequence[dict[str, str]] | dict[str, Sequence[str]]) -> np.ndarray:
        """Encode sentence-transformers-style corpus rows with GME corpus prompts."""
        if isinstance(corpus, dict):
            texts = corpus.get("text")
            titles = corpus.get("title")
            if texts is None:
                raise ValueError("corpus dictionary must contain 'text'")
            if titles is not None and len(titles) != len(texts):
                raise ValueError("corpus 'title' and 'text' must have the same length")
            rows = [f"{titles[index]} {text}".strip() if titles else text for index, text in enumerate(texts)]
        else:
            rows = [f"{row.get('title', '')} {row['text']}".strip() for row in corpus]
        return self.get_text_embeddings(rows, is_query=False)