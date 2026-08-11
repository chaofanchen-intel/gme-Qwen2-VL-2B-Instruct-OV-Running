# Qwen3-VL-Embedding-2B: GME opt1/opt2 applicability test

Date: 2026-08-11

## Scope

This A/B tests the two GME GPU plugin optimizations:

- opt-1: stateless GQA `Unsqueeze -> Broadcast -> Reshape` / `repeat_kv` fusion;
- opt-2: SDPA output transpose fusion plus the associated `sdpa_gen_micro` output-order fix.

The test uses the same Qwen3-VL-Embedding-2B INT8/INT4 OpenVINO models, the same GPU,
text inputs, tokenizer lengths, repeat/warmup, and OpenVINO runtime. Only the Intel GPU
plugin binary changes:

```text
stock:     plugin saved before applying all_gme_int4_gpu_opts.patch
optimized: plugin built from benchmark/openvino after applying all_gme_int4_gpu_opts.patch
```

The OpenVINO source tree was restored after the test. The GME patch is based on official
OpenVINO 2026.3.0, commit `8a17657b995fd3b4a52f8484acfcf2bb61214623`.

## Text-only A/B

Three alternating rounds, GPU, native 2048 output, `repeat=5`, `warmup=3`, batch N=1:

| Precision | Tokens | Stock median ms | Optimized median ms | Median speedup |
|---|---:|---:|---:|---:|
| INT8 | 512 | 176.0 | 176.6 | 0.997x |
| INT8 | 1024 | 366.6 | 359.5 | **1.020x** |
| INT4 | 512 | 169.3 | 170.7 | 0.991x |
| INT4 | 1024 | 355.7 | 351.2 | **1.013x** |

The per-round raw JSON is stored beside this report under `stock_r*` and `optimized_r*`.
These are small effects compared with the GME result, where the same optimizations produced
about 5--7% at 512/1024 tokens. Qwen3's text-only graph does not receive the same useful
fusion pattern at the same frequency, or the gain is hidden by other language graph costs.

## Image/fused A/B result

The stock plugin completed the existing Qwen3 detailed image/fused matrix. With the
optimized plugin, INT8 reached the first 64x64 image/fused cases, then the next visual
case failed:

```text
Failed to select implementation for
fullyconnectedcompressed:__module.blocks.0.attn.proj/ov_ext::linear/MatMul
original_type: FullyConnectedCompressed could not set scales primitive attribute

GPU clFinish, error code: -5 CL_OUT_OF_RESOURCES
```

An independent optimized-plugin probe reproduced the failure in one process:

```text
64x64 image:  [1,2048]  OK
224x224 image: CL_OUT_OF_RESOURCES / FullyConnectedCompressed failure
```

The same stock-plugin probe completed 64x64, 224x224 and 448x448 image-only calls with
`[1,2048]` output. Therefore image/fused optimized results are **not accepted as valid
performance data**; the plugin change causes a visual-path resource/regression issue for
this Qwen3 model/workload.

## Conclusion

| Path | Result for Qwen3-VL-Embedding-2B |
|---|---|
| INT8 text 512 | no meaningful gain, 0.997x in this A/B |
| INT8 text 1024 | small gain, 1.020x |
| INT4 text 512 | small regression, 0.991x |
| INT4 text 1024 | small gain, 1.013x |
| image/fused | **regression: CL_OUT_OF_RESOURCES** |

The GME opt1/opt2 patch should **not** be applied as a Qwen3-VL-Embedding production
optimization. It is not generally safe for this model's visual graph. The text-only
micro-gains are too small and inconsistent to justify accepting the multimodal regression.

This test does not prove that no Qwen3-specific optimization is possible. It proves that
these two GME transformations are not a drop-in optimization for Qwen3-VL-Embedding-2B.
A Qwen3-specific follow-up would need graph-level kernel/shape analysis and a fix for the
visual `FullyConnectedCompressed` resource failure before any deployment claim.
