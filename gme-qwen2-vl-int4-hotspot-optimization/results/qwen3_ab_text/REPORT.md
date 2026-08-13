# Qwen3-VL-Embedding-2B: GME GPU optimization applicability test

Date: 2026-08-11

## Scope

This report tests the three GME GPU plugin optimizations:

- opt-1: stateless GQA `Unsqueeze -> Broadcast -> Reshape` / `repeat_kv` fusion;
- opt-2: SDPA output transpose fusion plus the associated `sdpa_gen_micro` output-order fix.
- opt-3: wide INT4 fully-connected weights reordered to N-contiguous layout when `N >= 4K`
	and the prefill input has at least 128 rows.

The original A/B below tests opt-1 + opt-2. A second validation on 2026-08-13 tests the
current opt-1 + opt-2 + opt-3 install. The same Qwen3-VL-Embedding-2B INT8/INT4 OpenVINO
models, GPU, text inputs, tokenizer lengths, and warmup/repeat conventions are used:

```text
stock:     plugin saved before applying all_gme_int4_gpu_opts.patch
optimized: plugin built from benchmark/openvino after applying the relevant GME patches
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

## Opt-3 incremental validation

The opt-1 + opt-2 A/B above predates opt-3. With the current OpenVINO install containing
all three patches, repeated warm text-only measurements at exactly 1024 tokenizer tokens
were compared against a same-build opt-1 + opt-2 measurement. These two rows were run in
separate builds, so this is a scope check rather than a new alternating stock-vs-patched
campaign; the previous alternating campaign remains the authoritative opt-1 + opt-2 number.

| Precision | opt-1 + opt-2 p50 | opt-1 + opt-2 + opt-3 p50 | opt-3 incremental speedup |
|---|---:|---:|---:|
| INT8 | 359.73 ms | 359.34 ms | 1.0011x (+0.11%) |
| INT4 | 351.04 ms | 350.76 ms | 1.0008x (+0.08%) |

The tiny delta is expected from the model shape, not evidence of a useful opt-3 speedup.
Qwen3-VL-Embedding-2B's language config is:

```text
hidden_size=2048, intermediate_size=6144
N/K = 6144/2048 = 3.0 < 4
```

The opt-3 selector requires `N >= 4K`, so the Qwen3 language MLP gate/up projections do
not enter the N-contiguous reorder path. The opt-3 optimization therefore has **no
material effect on Qwen3 text embedding**. It is useful for the GME language tower because
GME's corresponding ratio is `8960/1536 = 5.83`.

The raw samples and visual smoke-test outcomes are recorded in
[`opt123_incremental_20260813.json`](opt123_incremental_20260813.json).

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

The current all-three build was also checked in fresh processes. INT8 and INT4 both
completed 64x64 image/fused calls, but 224x224 and 448x448 failed with the same
`FullyConnectedCompressed` / `CL_OUT_OF_RESOURCES` error. Because this failure was already
present in the opt-1 + opt-2 campaign, this test cannot attribute it to opt-3. The vision
config has `hidden_size=1024, intermediate_size=4096` (`N/K=4`), but the resource failure
prevents a valid opt-3-only visual performance experiment.

## Conclusion

| Path | Result for Qwen3-VL-Embedding-2B |
|---|---|
| INT8 text 512 | no meaningful gain, 0.997x in this A/B |
| INT8 text 1024 | small gain, 1.020x |
| INT4 text 512 | small regression, 0.991x |
| INT4 text 1024 | small gain, 1.013x |
| image/fused with opt-1 + opt-2 | **regression: CL_OUT_OF_RESOURCES** |
| opt-3 on text | **no material effect**; Qwen3 language MLP is below `N >= 4K` |
| opt-3 on image/fused | **not isolatable** because the earlier visual regression blocks the workload |

The three GME patches should **not** be applied as a Qwen3-VL-Embedding production
optimization. Opt-3 does not match the Qwen3 language MLP and provides no useful text
speedup; opt-1 + opt-2 provide only small text changes and still cause the known visual
`CL_OUT_OF_RESOURCES` regression. The text-only micro-gains are too small and inconsistent
to justify accepting the multimodal regression.

This test does not prove that no Qwen3-specific optimization is possible. It proves that
these three GME transformations are not a drop-in optimization for Qwen3-VL-Embedding-2B.
A Qwen3-specific follow-up would need graph-level kernel/shape analysis and a fix for the
visual `FullyConnectedCompressed` resource failure before any deployment claim.
