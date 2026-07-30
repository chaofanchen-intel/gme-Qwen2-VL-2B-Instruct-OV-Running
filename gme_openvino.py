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


def _as_list(value: str | Path | Image.Image | Sequence[str | Path | Image.Image] | None) -> list | None:
    if value is None:
        return None
    if isinstance(value, (str, Path, Image.Image)):
        return [value]
    values = list(value)
    # Official GME accepts a DataLoader for images. Its batches are normally
    # lists (the official collate function is identity), so flatten those while
    # keeping every actual image object/path as one sample.
    if values and all(isinstance(item, (list, tuple)) for item in values):
        return [item for batch in values for item in batch]
    return values


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


def load_image(value: str | Path | Image.Image) -> Image.Image:
    """Load and resize an image exactly as the official GME inference helper."""
    if isinstance(value, Image.Image):
        image = value
    else:
        value = str(value)
        if value.startswith(("http://", "https://")):
            image = Image.open(requests.get(value, stream=True, timeout=30).raw)
        elif value.startswith("file://"):
            image = Image.open(value[7:])
        elif value.startswith("data:image") and "base64," in value:
            image = Image.open(BytesIO(base64.b64decode(value.split("base64,", 1)[1])))
        else:
            image = Image.open(value)
    return _resize_for_gme(image.convert("RGB"))


class _RequestPool:
    """Run independent inference requests over one compiled model, in order.

    A GME "batch" is N mutually independent samples, so it maps onto N infer
    requests rather than one padded N-row tensor. That keeps every sample at its
    own true sequence length (no padding compute, no dependence on what else
    shares the batch) and lets the plugin overlap samples across streams.
    """

    def __init__(self, compiled: ov.CompiledModel) -> None:
        self.compiled = compiled
        self.size = max(1, int(compiled.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS")))
        self._queue: ov.AsyncInferQueue | None = None

    def run(self, feeds: list[dict[str, np.ndarray]]) -> list[np.ndarray]:
        if not feeds:
            return []
        if len(feeds) == 1:  # avoid queue overhead on the single-sample path
            result = self.compiled(feeds[0])
            return [np.asarray(result[self.compiled.output(0)], dtype=np.float32)]
        if self._queue is None:  # only pay for the pool's activations once a batch needs it
            self._queue = ov.AsyncInferQueue(self.compiled, self.size)
        results: list[np.ndarray | None] = [None] * len(feeds)

        def collect(request: ov.InferRequest, index: int) -> None:
            # The request's output buffer is reused by the next job: copy now.
            results[index] = request.get_output_tensor(0).data.astype(np.float32, copy=True)

        self._queue.set_callback(collect)
        for index, feed in enumerate(feeds):
            self._queue.start_async(feed, index)  # blocks while every request is busy
        self._queue.wait_all()
        missing = [index for index, row in enumerate(results) if row is None]
        if missing:
            raise RuntimeError(f"inference produced no result for samples {missing}")
        return results  # type: ignore[return-value]


class GmeOpenVINO:
    """GME encoder backed entirely by OpenVINO Runtime neural computation.

    One image per input is supported, matching the official model limitation.

    Batching works for any number of samples with no per-batch-size export. Both
    IRs are driven one sample at a time:

    * The visual IR is invoked once per image. Qwen2-VL's vision tower has no
      batch dimension at all -- images are concatenated into one patch sequence
      whose attention is block-diagonal per image -- so per-image invocation is
      *bit-identical* to feeding several ``image_grid_thw`` rows at once, and it
      is cheaper: a shared sequence costs O((sum L)^2) attention instead of
      O(sum L^2). Only the number of images can be batched in that IR anyway,
      and tracing bakes it in (``rot_pos_emb`` loops over the grid rows in
      Python), which is why a traced IR rejects a row count it was not traced
      with.
    * The language IR is driven by ``batch_mode``. ``"per_sample"`` invokes it once
      per sample at that sample's true length, which makes a batch bit-identical to
      the same items encoded one by one; ``"padded"`` sends one [B, S] call the way
      the official model does, where rows shift slightly with batch composition.

    Which one is faster is a property of the device, so ``batch_mode="auto"``
    (the default) picks per measurement:

    * CPU: one request already saturates every core, so a padded N-row call is no
      faster than N sequential calls (measured 1.00x at N=4 and N=8). Per-sample
      wins because it drops the padding work and, with ``streams``, overlaps
      samples for ~1.9x.
    * GPU: per-request submission overhead dominates short sequences, so folding
      the batch into one call is 2.4x faster per embedding (22 ms vs 53 ms at
      N=4) and extra streams do not recover it. Padded wins.
    """

    def __init__(
        self,
        model_dir: str | Path,
        device: str = "CPU",
        precision: str | None = None,
        fp32_math: bool | None = None,
        streams: int | str = 1,
        batch_mode: str = "auto",
    ) -> None:
        """precision: one of the keys in the package config's ``precisions`` map
        (e.g. ``fp32``/``int8``/``int4``); defaults to the package ``default_precision``.
        fp32_math: force f32 inference-precision hint (defaults to True only for the
        fp32 weights, so the reference is reproduced exactly; quantized models use the
        device default for best speed).
        streams: execution streams per IR, trading single-sample latency for batch
        throughput. ``1`` (default) gives one stream all cores -- lowest latency for
        one sample. ``"auto"`` lets the plugin pick (4 on a 16-core CPU), or pass an
        int to pin it; multi-sample calls then run ~1.9x faster while a lone sample
        is ~1.3x slower and each extra stream costs ~0.2 GB. Output is unaffected.
        batch_mode: how the language IR consumes a batch -- ``"per_sample"`` (one
        request per sample, bit-identical to sequential), ``"padded"`` (one [B, S]
        request, like the official model), or ``"auto"`` (default: per_sample on
        CPU, padded on GPU, matching where each is faster)."""
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
        padding_side = self.config["processor"]["padding_side"]
        if padding_side != "right":
            # Per-sample slicing below assumes valid tokens come first.
            raise ValueError(f"package must be right-padded, got padding_side={padding_side!r}")
        self.processor.tokenizer.padding_side = padding_side
        self.max_length = self.config["processor"]["max_length"]
        self.embedding_dim = int(self.config["embedding_dim"])
        self.core = ov.Core()
        if device not in self.core.available_devices:
            raise RuntimeError(f"OpenVINO device {device!r} unavailable; found {self.core.available_devices}")
        if fp32_math is None:
            fp32_math = precision == "fp32"
        compile_config = {hint.inference_precision: ov.Type.f32} if fp32_math else {}
        compile_config.update(self._stream_config(streams))
        self.streams = streams
        if batch_mode not in ("auto", "per_sample", "padded"):
            raise ValueError(f"batch_mode must be auto/per_sample/padded, got {batch_mode!r}")
        # Per-sample drops padding work and overlaps across CPU streams; a padded
        # call amortizes the GPU's per-submission overhead. See the class docstring.
        self.batch_mode = (
            ("per_sample" if device.upper().startswith("CPU") else "padded")
            if batch_mode == "auto"
            else batch_mode
        )
        self.language = self.core.compile_model(self.model_dir / language_model, device, compile_config)
        self.visual = self.core.compile_model(self.model_dir / visual_model, device, compile_config)
        self.language_pool = _RequestPool(self.language)
        self.visual_pool = _RequestPool(self.visual)
        self._no_images = np.empty((0, self.embedding_dim), dtype=np.float32)

    @staticmethod
    def _stream_config(streams: int | str) -> dict:
        """LATENCY gives one request every core; THROUGHPUT overlaps requests."""
        if streams == "auto":
            return {hint.performance_mode: hint.PerformanceMode.THROUGHPUT}
        if not isinstance(streams, int) or streams < 1:
            raise ValueError(f"streams must be a positive int or 'auto', got {streams!r}")
        if streams == 1:
            return {hint.performance_mode: hint.PerformanceMode.LATENCY}
        return {
            hint.performance_mode: hint.PerformanceMode.THROUGHPUT,
            ov.properties.num_streams: streams,
        }

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

    def _language_embedding(
        self,
        inputs: dict[str, np.ndarray],
        image_embeds: list[np.ndarray] | None = None,
    ) -> np.ndarray:
        """Decode the batch, per ``batch_mode``.

        ``image_embeds`` holds each sample's visual tokens, or None for text-only.
        """
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        if self.batch_mode == "padded":
            # One [B, S] call, exactly the official layout: every sample's visual
            # tokens are concatenated and masked_scatter distributes them across the
            # flattened batch, one image-pad slot per row.
            return self.language_pool.run([{
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "image_embeds": (
                    self._no_images if image_embeds is None
                    else np.concatenate(image_embeds, axis=0)
                ),
            }])[0]

        # Per-sample: trimming the right padding makes the result independent of the
        # batch it was encoded in, and skips the decoder work the padding would cost.
        lengths = attention_mask.sum(axis=1).astype(np.int64)
        feeds = [
            {
                "input_ids": input_ids[row : row + 1, : lengths[row]],
                "attention_mask": attention_mask[row : row + 1, : lengths[row]],
                "image_embeds": self._no_images if image_embeds is None else image_embeds[row],
            }
            for row in range(input_ids.shape[0])
        ]
        return np.concatenate(self.language_pool.run(feeds), axis=0)

    def _encode_text_batch(self, texts: list[str], instruction: str) -> np.ndarray:
        prompts = [self._prompt(text, has_image=False, instruction=instruction) for text in texts]
        inputs = self.processor(
            text=prompts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )
        return self._language_embedding(inputs)

    @staticmethod
    def _batches(values: list, batch_size: int | None):
        if batch_size is None:
            yield values
            return
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        for start in range(0, len(values), batch_size):
            yield values[start : start + batch_size]

    def _encode_text(self, texts: list[str], instruction: str, batch_size: int | None) -> np.ndarray:
        return np.concatenate(
            [self._encode_text_batch(batch, instruction) for batch in self._batches(texts, batch_size)], axis=0
        )

    def _encode_image_batch(
        self,
        texts: list[str | None],
        images: list[str | Image.Image],
        instruction: str,
    ) -> np.ndarray:
        if len(texts) != len(images) or not images:
            raise ValueError("image batch needs matching, non-empty texts and images")
        inputs = self.processor(
            text=[self._prompt(text, has_image=True, instruction=instruction) for text in texts],
            images=[load_image(image) for image in images],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )
        # The processor concatenates every image's patches into one array; split it
        # back so each image becomes its own visual request. Block-diagonal vision
        # attention makes this bit-identical to one multi-row call, and faster.
        grid_thw = inputs["image_grid_thw"]
        bounds = np.concatenate([[0], np.cumsum(np.prod(grid_thw, axis=1, dtype=np.int64))])
        visual_feeds = [
            {
                "pixel_values": inputs["pixel_values"][bounds[index] : bounds[index + 1]],
                "image_grid_thw": grid_thw[index : index + 1],
            }
            for index in range(grid_thw.shape[0])
        ]
        return self._language_embedding(inputs, self.visual_pool.run(visual_feeds))

    def get_text_embeddings(
        self,
        texts: str | Sequence[str],
        instruction: str | None = None,
        is_query: bool = True,
        batch_size: int | None = None,
    ) -> np.ndarray:
        values = _as_list(texts)
        if not values or not all(isinstance(text, str) for text in values):
            raise ValueError("texts must be a non-empty string or sequence of strings")
        return self._encode_text(values, self._instruction(instruction, is_query), batch_size)

    def get_image_embeddings(
        self,
        images: str | Image.Image | Sequence[str | Image.Image],
        instruction: str | None = None,
        is_query: bool = True,
        batch_size: int | None = None,
    ) -> np.ndarray:
        values = _as_list(images)
        if not values:
            raise ValueError("images must be a non-empty image or image sequence")
        prompt = self._instruction(instruction, is_query)
        return np.concatenate(
            [
                self._encode_image_batch([None] * len(batch), batch, prompt)
                for batch in self._batches(values, batch_size)
            ],
            axis=0,
        )

    def get_fused_embeddings(
        self,
        texts: str | Sequence[str] | None = None,
        images: str | Image.Image | Sequence[str | Image.Image] | None = None,
        instruction: str | None = None,
        is_query: bool = True,
        batch_size: int | None = None,
    ) -> np.ndarray:
        text_values = _as_list(texts)
        image_values = _as_list(images)
        prompt = self._instruction(instruction, is_query)
        if image_values is None:
            if text_values is None:
                raise ValueError("provide texts, images, or both")
            return self._encode_text(text_values, prompt, batch_size)
        if text_values is None:
            return self.get_image_embeddings(image_values, instruction=instruction, is_query=is_query, batch_size=batch_size)
        if len(text_values) != len(image_values):
            raise ValueError("texts and images must have the same length")
        return np.concatenate(
            [
                self._encode_image_batch(
                    [text for text, _ in batch], [image for _, image in batch], prompt
                )
                for batch in self._batches(list(zip(text_values, image_values)), batch_size)
            ],
            axis=0,
        )

    def encode(
        self,
        sentences: str | Sequence[str],
        *,
        instruction: str | None = None,
        is_query: bool = True,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Compatibility alias for text encoding used by embedding clients."""
        return self.get_text_embeddings(sentences, instruction=instruction, is_query=is_query, batch_size=batch_size)

    def encode_queries(
        self,
        queries: str | Sequence[str],
        *,
        instruction: str | None = None,
        batch_size: int | None = None,
    ) -> np.ndarray:
        return self.get_text_embeddings(queries, instruction=instruction, is_query=True, batch_size=batch_size)

    def encode_corpus(
        self,
        corpus: Sequence[dict[str, str]] | dict[str, Sequence[str]],
        *,
        batch_size: int | None = None,
    ) -> np.ndarray:
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
        return self.get_text_embeddings(rows, is_query=False, batch_size=batch_size)