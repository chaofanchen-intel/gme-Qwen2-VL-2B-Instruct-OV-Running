"""Embedding backend adapters for MMDocIR."""
from __future__ import annotations

import importlib
import inspect
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image


class EmbeddingBackend(ABC):
    """Minimal interface required by the dataset and evaluation layers."""

    model_name: str
    embedding_dim: int

    @abstractmethod
    def encode_queries(self, texts: Sequence[str], batch_size: int | None = None) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def encode_page_images(
        self, images: Sequence[Image.Image], batch_size: int | None = None
    ) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def encode_layout_texts(
        self, texts: Sequence[str], batch_size: int | None = None
    ) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def encode_layout_fused(
        self,
        texts: Sequence[str],
        images: Sequence[Image.Image],
        batch_size: int | None = None,
    ) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        raise NotImplementedError


class GmeOpenVINOBackend(EmbeddingBackend):
    def __init__(
        self,
        package: Path,
        precision: str,
        device: str,
        batch_size: int,
        running_dir: Path,
        cache_dir: Path | None = None,
        batch_mode: str = "auto",
        max_image_tokens: int | None = None,
        query_instruction: str | None = None,
    ) -> None:
        sys.path.insert(0, str(running_dir))
        from gme_openvino import GmeOpenVINO

        self.model_name = "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct"
        self.package = package
        self.precision = precision
        self.device = device
        self.batch_size = batch_size
        self.batch_mode = batch_mode
        self.max_image_tokens = max_image_tokens
        self.query_instruction = query_instruction
        started = time.perf_counter()
        self.encoder = GmeOpenVINO(
            package,
            device=device,
            precision=precision,
            batch_mode=batch_mode,
            cache_dir=cache_dir,
            max_image_tokens=max_image_tokens,
        )
        self.load_seconds = time.perf_counter() - started
        self.embedding_dim = self.encoder.embedding_dim

    def encode_queries(self, texts: Sequence[str], batch_size: int | None = None) -> np.ndarray:
        return self.encoder.get_text_embeddings(
            texts,
            instruction=self.query_instruction,
            is_query=True,
            batch_size=batch_size or self.batch_size,
            show_progress_bar=False,
        )

    def encode_page_images(
        self, images: Sequence[Image.Image], batch_size: int | None = None
    ) -> np.ndarray:
        return self.encoder.get_image_embeddings(
            images,
            is_query=False,
            batch_size=batch_size or self.batch_size,
            show_progress_bar=False,
        )

    def encode_layout_texts(
        self, texts: Sequence[str], batch_size: int | None = None
    ) -> np.ndarray:
        return self.encoder.get_text_embeddings(
            texts,
            is_query=False,
            batch_size=batch_size or self.batch_size,
            show_progress_bar=False,
        )

    def encode_layout_fused(
        self,
        texts: Sequence[str],
        images: Sequence[Image.Image],
        batch_size: int | None = None,
    ) -> np.ndarray:
        return self.encoder.get_fused_embeddings(
            texts=texts,
            images=images,
            is_query=False,
            batch_size=batch_size or self.batch_size,
            show_progress_bar=False,
        )

    def metadata(self) -> dict[str, Any]:
        device_properties: dict[str, Any] = {}
        if self.device.split(".", 1)[0] == "GPU":
            property_device = self.device if self.device != "GPU" else "GPU"
            for key in ("FULL_DEVICE_NAME", "GPU_DEVICE_TOTAL_MEM_SIZE", "GPU_DEVICE_MAX_ALLOC_MEM_SIZE"):
                try:
                    device_properties[key.lower()] = self.encoder.core.get_property(property_device, key)
                except RuntimeError:
                    device_properties[key.lower()] = None
        return {
            "adapter": "gme-openvino",
            "model": self.model_name,
            "package": str(self.package.resolve()),
            "precision": self.precision,
            "device": self.device,
            "batch_size": self.batch_size,
            "batch_mode": self.batch_mode,
            "max_image_tokens": self.max_image_tokens,
            "query_instruction": self.query_instruction,
            "embedding_dim": self.embedding_dim,
            "load_seconds": self.load_seconds,
            "device_properties": device_properties,
        }

    def resource_snapshot(self) -> dict[str, Any]:
        if self.device.split(".", 1)[0] != "GPU":
            return {}
        try:
            values = self.encoder.core.get_property(self.device, "GPU_MEMORY_STATISTICS")
        except RuntimeError:
            return {}
        return {"gpu_memory_bytes": {str(key): int(value) for key, value in values.items()}}


def create_backend(name: str, **kwargs: Any) -> EmbeddingBackend:
    """Create a built-in backend or a `module:factory` plugin."""
    if name == "gme-openvino":
        return GmeOpenVINOBackend(**kwargs)
    if ":" not in name:
        raise ValueError(f"unknown backend {name!r}; use gme-openvino or module:factory")
    module_name, factory_name = name.split(":", 1)
    factory = getattr(importlib.import_module(module_name), factory_name)
    signature = inspect.signature(factory)
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    factory_kwargs = kwargs if accepts_kwargs else {
        key: value for key, value in kwargs.items() if key in signature.parameters
    }
    backend = factory(**factory_kwargs)
    required = (
        "model_name",
        "embedding_dim",
        "encode_queries",
        "encode_page_images",
        "encode_layout_texts",
        "encode_layout_fused",
        "metadata",
    )
    missing = [name for name in required if not hasattr(backend, name)]
    if missing:
        raise TypeError(f"{name} backend is missing: {', '.join(missing)}")
    return backend