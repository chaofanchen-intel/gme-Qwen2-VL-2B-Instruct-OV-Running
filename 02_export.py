#!/usr/bin/env python
"""Export the complete GME-Qwen2-VL embedding runtime to OpenVINO.

The output directory contains two composable IRs and the original tokenizer /
image-processor assets:

* ``gme_language_fp32.xml``: token lookup, visual-token insertion, decoder,
  last-valid-token pooling and L2 normalization.
* ``gme_visual_fp32.xml``: Qwen2-VL's historical mask-carrying visual encoder.
* ``gme_visual_fp32_nomask.xml``: current single-image mask-free visual encoder.

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
    """Qwen2-VL visual encoder that produces image-token embeddings.

    Returns only the **merged** ``[K, 1536]`` tensor the language IR consumes.  That
    unwrapping is load-bearing: transformers 4 returned it directly, but transformers 5
    returns ``BaseModelOutputWithPooling(last_hidden_state=[L, 1280], pooler_output=[K, 1536])``.
    Tracing the wrapper as-is therefore exports a **two-output** IR whose output 0 is the
    unmerged 1280-dim hidden state -- and ``gme_openvino.py`` reads ``visual.output(0)``,
    so the package would silently feed the language IR the wrong tensor at the wrong width.
    Selecting the merged output here keeps the exported signature identical across both
    transformers majors.
    """

    def __init__(self, visual: torch.nn.Module) -> None:
        super().__init__()
        self.visual = visual

    def forward(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        output = self.visual(pixel_values, grid_thw=image_grid_thw)
        pooled = getattr(output, "pooler_output", None)
        return output if pooled is None else pooled


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


@contextlib.contextmanager
def unmasked_vision_attention(fuse_input_transposes: bool = False,
                              qkv_layout: str = "upstream",
                              rank3_linear: bool = False,
                              _allow_multi_block: bool = False):
    """Trace the ViT blocks with one full-sequence, mask-free SDPA call.

    Qwen2-VL's vision tower is *packed*, not batched: every image's patch rows are
    concatenated into one sequence and attention is restricted to each image's own
    block by ``cu_seqlens``.  Both of the upstream code paths express that in a way
    that exports badly:

    * the mask path (what the shipped ``gme_visual_fp32.xml`` was traced from) builds
      ``torch.zeros([1, L, L], dtype=torch.bool)`` and fills the diagonal blocks with
      a slice-assign.  The Intel GPU plugin will not lower a ScaledDotProductAttention
      whose mask is a **rank-3 BOOL** tensor, so all 32 attention layers decompose into
      MatMul + Select + Softmax + MatMul with permutes around them -- the same failure
      mode docs/10 documents for the language ``_v5ctrl`` control export;
    * the split path (transformers >= 5) calls ``lengths.tolist()`` and slices with the
      resulting Python ints, which bakes this trace's ``L`` into the graph as constants
      and destroys the dynamic sequence dimension the runtime needs.

    ``gme_openvino.py`` invokes the visual IR **once per image** -- that is a documented,
    bit-identity-tested property of ``_encode_image_batch``, not an accident -- so
    ``cu_seqlens`` is always ``[0, L]``, the block-diagonal mask is always all-True, and
    dropping it is an identity.  This patch therefore calls SDPA once over the whole
    sequence with ``attn_mask=None``: rank-4 Q/K/V, no mask, ``is_causal=False``, which
    is exactly the shape the plugin lowers to ``ocl::sdpa::opt``.

    The narrowing this buys performance with: an IR traced this way computes *dense*
    attention over whatever it is fed, so feeding it several images' patches in one call
    would silently attend across image boundaries.  The single-image contract is what
    makes it correct, and ``_encode_image_batch`` already guarantees it.

    ``fuse_input_transposes`` emits Q/K/V as ``Unsqueeze -> Transpose(rank 4) -> SDPA``
    instead of upstream's ``Transpose(rank 3) -> Unsqueeze -> SDPA``.  Same tensors, but
    only the first order lets ``TransposeSDPAMatcher`` absorb the transposes into the SDPA
    primitive's input order, which removes three ``permute_ref`` copies per layer.  It is
    **off by default** because that trade reverses with sequence length: the permutes it
    deletes are O(L) while the strided reads it hands the SDPA kernel cost O(L^2).
    Measured at L=2304 it is -31.8 ms, at L=4096 it is -6.0 ms, and at L=5040 it is
    **+103.8 ms** -- and this package's ``max_image_tokens=1280`` puts the worst case
    inside the supported range.  See docs/11 for the full crossover table.

    ``qkv_layout`` selects how Q/K/V reach SDPA's [1, H, L, D]:

    * ``"upstream"`` -- verbatim upstream: ``permute(1,0,2,3)`` to split QKV, then a
      ``transpose(0,1)`` on each of the three. Two data movements per layer.
    * ``"unbind"``   -- ``unbind(1)``; identical tensors, one fewer graph node. A control
      for whether the cost is the permute node or the copy it implies.
    * ``"headfirst"`` -- ``permute(1,2,0,3)`` reaches [3,H,L,D] in **one** movement, so the
      three per-tensor transposes vanish. Rotary is rewritten for the [H,L,D] layout
      (same arithmetic; ``rotate_half`` only touches the last axis).

    docs/12 measures all three.

    ``rank3_linear`` feeds every ViT projection a ``[1, L, C]`` input instead of ``[L, C]``.
    The tensors are identical -- only a leading unit axis is added -- but the GPU plugin's
    INT4 weight-reorder (opt-3, docs/06 S6.9) starts with ``prim_input_size != 3 -> reject``,
    so the whole vision tower is excluded from it purely for being rank 2.  Of the ViT's
    projections only ``mlp.fc1`` clears opt-3's other gate (``N >= 4K``: 5120 vs 4x1280,
    exactly on the boundary), so this is a bounded experiment, not a general win.
    docs/12 S12.3 reports what it actually measured.

    ``_allow_multi_block`` disables the single-block guard.  It exists for exactly one
    caller -- ``check_vision_mask_necessity.py``, which deliberately runs a packed
    multi-image sequence *without* the mask to measure how large a genuine cross-image
    leak is (row cosine 0.059, ~4400x the fp32 noise floor).  Never set it when exporting.
    """
    from transformers.models.qwen2_vl import modeling_qwen2_vl as qwen2_vl

    def forward(self, hidden_states, cu_seqlens, rotary_pos_emb=None,
                position_embeddings=None, **_kwargs):
        # Trace-time guard, not a graph op. `cu_seqlens` has one entry per attention block
        # plus a leading 0, so length 2 means exactly one block and therefore an all-True
        # mask -- the only case in which dropping it is an identity. This fires if someone
        # later traces with a multi-image example, or with a t>1 (video) grid, both of which
        # would otherwise silently produce an IR that attends across block boundaries.
        # Measured on the real checkpoint, that is not a subtle error: row cosine drops to
        # 0.059 versus per-image, ~4400x the fp32 reduction-order noise floor.
        if len(cu_seqlens) != 2 and not _allow_multi_block:
            raise RuntimeError(
                f"--unmasked-vision requires a single-block example (cu_seqlens length 2), "
                f"got {len(cu_seqlens)} from grid rows with more than one image or t > 1. "
                "The mask-free IR is only an identity for one image per call; export the "
                "mask-carrying IR instead."
            )
        # shape[-2] so this works whether the tower runs rank 2 ([L, C]) or rank 3
        # ([1, L, C]); under `rank3_linear` the unsqueeze happens once for the whole tower
        # (see tower_forward) rather than around every projection.
        seq_length = hidden_states.shape[-2]
        qkv_source = hidden_states
        if qkv_layout == "headfirst":
            # One permute does the whole job. Upstream needs two data movements per layer:
            # `permute(1,0,2,3)` to split QKV, then a `transpose(0,1)` on each of Q/K/V to
            # reach SDPA's [1,H,L,D]. Measured at L=1024 that is 18.3 ms + 20.1 ms of
            # `permute_ref` per call. `permute(1,2,0,3)` lands on [3,H,L,D] directly, so the
            # three transposes disappear and only one copy of the same tensor remains.
            #
            # This is NOT the rejected `fuse_input_transposes` (docs/11 S11.5): that one
            # deleted the copies by making SDPA read strided, which costs O(L^2). Here SDPA
            # still receives contiguous [1,H,L,D] -- only the redundant second pass is gone.
            #
            # Rotary then has to run on [H,L,D] instead of [L,H,D], which is why it is
            # written out below instead of calling apply_rotary_pos_emb_vision: that helper
            # hardcodes `cos.unsqueeze(-2)` for the seq-first layout. rotate_half only ever
            # touches the last axis, so the arithmetic is unchanged.
            query_states, key_states, value_states = (
                self.qkv(qkv_source)
                .reshape(seq_length, 3, self.num_heads, -1)
                .permute(1, 2, 0, 3)
                .unbind(0)
            )
        elif qkv_layout == "unbind":
            # Same tensors as upstream with one fewer node: unbinding dim 1 of [L,3,H,D] is
            # exactly what `permute(1,0,2,3).unbind(0)` computes, without the explicit
            # Transpose. Kept as a control for how much of the 18.3 ms is the permute node
            # itself versus the data movement it implies.
            query_states, key_states, value_states = (
                self.qkv(qkv_source)
                .reshape(seq_length, 3, self.num_heads, -1)
                .unbind(1)
            )
        else:
            query_states, key_states, value_states = (
                self.qkv(qkv_source)
                .reshape(seq_length, 3, self.num_heads, -1)
                .permute(1, 0, 2, 3)
                .unbind(0)
            )
        cos, sin = position_embeddings
        if qkv_layout == "headfirst":
            # apply_rotary_pos_emb_vision inlined for [H, L, D]: the helper's
            # `cos.unsqueeze(-2)` assumes seq-first, so broadcast over heads instead.
            # Everything else -- the float() casts, the multiply order, rotate_half -- is
            # copied verbatim so the arithmetic matches upstream bit for bit.
            original_dtype = query_states.dtype
            query_float, key_float = query_states.float(), key_states.float()
            cos_b, sin_b = cos.unsqueeze(0).float(), sin.unsqueeze(0).float()
            rotate_half = qwen2_vl.rotate_half
            query_states = ((query_float * cos_b)
                            + (rotate_half(query_float) * sin_b)).to(original_dtype)
            key_states = ((key_float * cos_b)
                          + (rotate_half(key_float) * sin_b)).to(original_dtype)
        else:
            query_states, key_states = qwen2_vl.apply_rotary_pos_emb_vision(
                query_states, key_states, cos, sin
            )
        # Both branches produce the same [1, H, L, D] tensor; only the op order differs,
        # and that order is what decides whether the plugin fuses the transpose. See the
        # docstring for why the fused form is not the default.
        if qkv_layout == "headfirst":
            # Already [H, L, D]; only the batch axis is missing.
            query_states = query_states.unsqueeze(0)
            key_states = key_states.unsqueeze(0)
            value_states = value_states.unsqueeze(0)
        elif fuse_input_transposes:
            query_states = query_states.unsqueeze(0).transpose(1, 2)
            key_states = key_states.unsqueeze(0).transpose(1, 2)
            value_states = value_states.unsqueeze(0).transpose(1, 2)
        else:
            query_states = query_states.transpose(0, 1).unsqueeze(0)
            key_states = key_states.transpose(0, 1).unsqueeze(0)
            value_states = value_states.transpose(0, 1).unsqueeze(0)
        # `scale` is deliberately left at the default. VisionAttention sets
        # `self.scaling = head_dim ** -0.5`, which *is* SDPA's default `1/sqrt(q.size(-1))`,
        # and passing it explicitly exports a 5-input SDPA (Q, K, V, scalar-0 mask, scale)
        # instead of the 3-input form the plugin's fusion matches. check_vision_equivalence.py
        # confirms the result is bit-identical to the mask-carrying IR either way.
        attn_output = functional.scaled_dot_product_attention(
            query_states, key_states, value_states,
            attn_mask=None, dropout_p=0.0, is_causal=False,
        )
        # Match upstream's layout exactly: [1, H, L, D] -> [1, L, H, D] -> [L, dim].
        # Reshaping straight to [1, L, C] under rank3_linear keeps `proj` rank 3 without a
        # separate Unsqueeze node -- the per-projection Unsqueeze/Squeeze pairs an earlier
        # attempt used cost 64 `generic_eltwise_ref` copies per call (docs/12 S12.3).
        if rank3_linear:
            attn_output = attn_output.transpose(1, 2).reshape(1, seq_length, -1).contiguous()
        else:
            attn_output = attn_output.transpose(1, 2).reshape(seq_length, -1).contiguous()
        return self.proj(attn_output)

    saved_tower = qwen2_vl.Qwen2VisionTransformerPretrainedModel.forward

    def tower_forward(self, hidden_states, grid_thw, **kwargs):
        # Upstream's own forward, with one change: the block stack runs on [1, L, C].
        # VisionMlp and the residual adds are shape-agnostic, and VisionAttention above
        # reads shape[-2], so nothing else needs patching -- and the merger's first op is
        # a `view(-1, hidden_size)` that flattens the leading axis away again, so no
        # squeeze is needed either. Total cost: two metadata-only reshapes per call.
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())
        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(dim=0, dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32)
        cu_seqlens = functional.pad(cu_seqlens, (1, 0), value=0)

        hidden_states = hidden_states.unsqueeze(0)
        for blk in self.blocks:
            hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens,
                                position_embeddings=position_embeddings, **kwargs)
        hidden_states = hidden_states.squeeze(0)
        return qwen2_vl.BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=self.merger(hidden_states),
        )

    saved = qwen2_vl.VisionAttention.forward
    qwen2_vl.VisionAttention.forward = forward
    if rank3_linear:
        qwen2_vl.Qwen2VisionTransformerPretrainedModel.forward = tower_forward
    try:
        yield
    finally:
        qwen2_vl.VisionAttention.forward = saved
        qwen2_vl.Qwen2VisionTransformerPretrainedModel.forward = saved_tower


def export_model(
    module: torch.nn.Module,
    example_inputs: tuple[torch.Tensor, ...],
    output_path: Path,
    input_names: list[str],
    shapes: list[ov.PartialShape],
    causal_attention: bool = False,
    unmasked_vision: bool = False,
    fuse_vision_input_transposes: bool = False,
    vision_qkv_layout: str = "upstream",
    vision_rank3_linear: bool = False,
) -> None:
    print(f"[export] converting {output_path.name} ...")
    start = time.monotonic()
    with contextlib.ExitStack() as stack:
        if causal_attention:
            stack.enter_context(suppress_causal_mask_materialization())
        if unmasked_vision:
            stack.enter_context(unmasked_vision_attention(fuse_vision_input_transposes,
                                                          vision_qkv_layout,
                                                          vision_rank3_linear))
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
    parser.add_argument(
        "--visual-suffix",
        default=None,
        help="suffix for the visual IR filename; defaults to _nomask with "
             "--unmasked-vision and empty otherwise",
    )
    parser.add_argument(
        "--unmasked-vision",
        action="store_true",
        help="export the visual IR with one full-sequence, mask-free SDPA per block, "
             "into gme_visual_fp32_nomask.xml (an identity for the one-image-per-call "
             "contract gme_openvino.py already enforces -- see unmasked_vision_attention)",
    )
    parser.add_argument(
        "--vision-qkv-layout",
        choices=("upstream", "unbind", "headfirst"),
        default="upstream",
        help="with --unmasked-vision, how Q/K/V reach SDPA's [1,H,L,D]. 'headfirst' does it "
             "in one permute instead of a permute plus three transposes (docs/12); the "
             "default suffix becomes _nomask_<layout> for anything but 'upstream'",
    )
    parser.add_argument(
        "--vision-rank3-linear",
        action="store_true",
        help="with --unmasked-vision, give ViT projections a [1,L,C] input so the plugin's "
             "INT4 weight reorder (opt-3) is not rejected for rank 2 alone (docs/12 S12.3)",
    )
    parser.add_argument(
        "--fuse-vision-input-transposes",
        action="store_true",
        help="with --unmasked-vision, emit Q/K/V so the plugin folds their Transpose into "
             "SDPA, into gme_visual_fp32_nomask_qkvfused.xml. Analysis-only: faster below "
             "~4K patch rows, slower above (docs/11), so it is not the shipped default",
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
        # first keeps the historical baseline reproducible, the second is the current
        # causal language option -- so an in-order 01..05 run has to produce both.
        # "language" and "visual" each run twice: once for the mask-carrying IR that the
        # earlier rounds were measured on, once for the mask-free IR used by the current
        # image/fused package (docs/11 for vision). Both are products -- the first keeps
        # the historical baseline reproducible, the second is the current visual option --
        # so an in-order 01..05 run has to produce both.
        stages: list[tuple[str, list[str]]] = [
            ("language", []),
            ("language", ["--causal-sdpa"]),
            ("visual", []),
            ("visual", ["--unmasked-vision"]),
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
            label = f"{stage}{' (' + extra[0].lstrip('-') + ')' if extra else ''}"
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

    # Like the language IR's _causal variant, the unmasked vision IR gets its own file so
    # the mask-carrying IR every earlier round was measured on stays reproducible and the
    # two can be A/B'd inside one session.
    visual_suffix = args.visual_suffix if args.visual_suffix is not None else (
        ("_nomask_qkvfused" if args.fuse_vision_input_transposes
         else "_nomask" if args.vision_qkv_layout == "upstream" and not args.vision_rank3_linear
         else f"_nomask_{args.vision_qkv_layout}"
              f"{'_rank3' if args.vision_rank3_linear else ''}")
        if args.unmasked_vision else "")
    visual_path = output / f"gme_visual_fp32{visual_suffix}.xml"
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
            unmasked_vision=args.unmasked_vision,
            fuse_vision_input_transposes=args.fuse_vision_input_transposes,
            vision_qkv_layout=args.vision_qkv_layout,
            vision_rank3_linear=args.vision_rank3_linear,
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