# Round 10: Python-only final performance rebenchmark

日期：2026-09-07

本轮只测试 Python API；按用户要求不重编、不测试、不引用 C++ binary。目标是验证当前所有已应用 patch 的 GME-Qwen2-VL-2B package/runtime 性能，重点覆盖 `int4_causal_vopt` 的 `text/query/image/fused` 和 batch，并与最初 stock baseline 对比。

## 17.1 Provenance

| 项 | 当前值 |
| --- | --- |
| GME revision | `9ae29bd` |
| OpenVINO runtime | `2026.3.0-22455-2080527e54b-qwen-ttft-optimizations` |
| package config SHA256 | `0f7bcad4977f7078a425d1b2b11b1ae1f7b5b9e444f963e71f2ec765cb83bee7` |
| device | Intel Graphics iGPU |
| backend | Python API only |
| visual threshold | `GPU_DYNAMIC_QUANTIZATION_THRESHOLD=0` |
| public repeat | warm-up 1 + measured repeat 5 |
| batch profile | GPU `padded + streams=1`, 128 effective text tokens, 448x448 image |
| metric | arithmetic mean of measured samples |

原始结果：

- [Python public matrix](../results/r10_latest/benchmark_matrix_python_vdq0_formal.json)
- [INT8 batch](../results/r10_latest/benchmark_batch_int8_causal_vopt_vdq0.json)
- [INT4 batch](../results/r10_latest/benchmark_batch_int4_causal_vopt_vdq0.json)
- [INT4 gs64 batch](../results/r10_latest/benchmark_batch_int4_gs64_causal_vdq0.json)
- [Cosine validation](../results/r10_latest/openvino_validation_r10_vdq0.json)
- [Baseline comparison](../results/r10_latest/baseline_vs_r10_python_comparison.json)

## 17.2 N=1 public latency

单位：ms/call；`text/query` 和 `image/fused` 按顺序排列。

### Text/query

| effective tokens | INT8 | INT4 causal | INT4 gs64 |
| ---: | ---: | ---: | ---: |
| 32 | 44.91 / 44.92 | **33.19 / 33.18** | 37.52 / 37.86 |
| 128 | 55.93 / 55.75 | **43.55 / 43.36** | 44.64 / 44.83 |
| 512 | 129.94 / 130.10 | **120.87 / 120.23** | 139.83 / 139.91 |
| 1024 | 267.31 / 266.85 | **228.18 / 228.96** | 241.05 / 241.44 |

### Image/fused

| source size | visual tokens | INT8 | INT4 causal | INT4 gs64 |
| --- | ---: | ---: | ---: | ---: |
| 448x448 | 256 | 285.68 / 284.25 | **248.36 / 248.38** | 259.60 / 258.55 |
| 672x672 | 576 | 677.06 / 723.51 | **602.19 / 600.55** | 621.93 / 620.96 |
| 1288x728 | 1196 | 1759.95 / 1742.62 | **1647.54 / 1639.14** | 1683.12 / 1671.75 |

`int4_causal_vopt` 是本轮三个正式 precision 中 N=1 text/query 和 image/fused
整体最低延迟的组合；`int4_gs64_causal_vopt` 以更高精度为目标，延迟通常介于
INT8 与 INT4 causal 之间。

## 17.3 Batch=8

单位：`ms/call / ms/embedding`。视觉塔仍按单图 request 执行，因此 image/fused 的
batch 不是一个真正的 `[N,...]` visual padded tensor。

| precision | text | query | image | fused |
| --- | ---: | ---: | ---: | ---: |
| `int8_causal_vopt` | 253.06 / 31.63 | 268.94 / 33.62 | 2047.63 / 255.95 | 2235.92 / 279.49 |
| `int4_causal_vopt` | **217.42 / 27.18** | **234.32 / 29.29** | **1899.56 / 237.45** | **2111.79 / 263.97** |
| `int4_gs64_causal_vopt` | 229.61 / 28.70 | 257.86 / 32.23 | 1947.85 / 243.48 | 2167.38 / 270.92 |

完整 `N=1/2/4/8/16/32`、`per_sample/padded`、`streams=1/auto` 保留在 batch JSON
中；正式推荐 GPU 配置仍为 `padded + streams=1`。

## 17.4 与最初 stock baseline

### INT8

| workload | initial stock text/query | r10 text/query | speedup |
| --- | ---: | ---: | ---: |
| 32 tokens | 51.55 / 49.41 | 44.91 / 44.92 | 1.148x / 1.100x |
| 128 tokens | 64.07 / 62.74 | 55.93 / 55.75 | 1.146x / 1.125x |
| 512 tokens | 144.65 / 139.36 | 129.94 / 130.10 | 1.113x / 1.071x |
| 1024 tokens | 296.57 / 296.01 | 267.31 / 266.85 | 1.109x / 1.109x |
| 448x448 image/fused | 476.69 / 468.07 | 285.68 / 284.25 | 1.669x / 1.647x |
| 672x672 image/fused | 1372.30 / 1338.03 | 677.06 / 723.51 | 2.027x / 1.849x |
| 1288x728 image/fused | 4654.46 / 4600.58 | 1759.95 / 1742.62 | 2.645x / 2.640x |

### INT4

INT4 32-token stock arithmetic mean和绝对 masked image/fused mean没有保留，因此不伪造
这两项的初始 baseline。text/query 使用三轮 stock raw-run mean；image/fused 使用最早
保留的完整 pre-vopt mean（mask-free、尚未 headfirst/rank3）。

| workload | baseline text/query or image/fused | r10 | speedup |
| --- | ---: | ---: | ---: |
| 128 tokens | 56.967 / 46.514 | 43.55 / 43.36 | 1.308x / 1.073x |
| 512 tokens | 157.318 / 146.188 | 120.87 / 120.23 | 1.302x / 1.216x |
| 1024 tokens | 291.443 / 280.598 | 228.18 / 228.96 | 1.277x / 1.226x |
| 448x448 image/fused pre-vopt | 296.37 / 296.43 | 248.36 / 248.38 | 1.193x / 1.193x |
| 672x672 image/fused pre-vopt | 699.25 / 694.68 | 602.19 / 600.55 | 1.161x / 1.157x |
| 1288x728 image/fused pre-vopt | 1800.29 / 1795.29 | 1647.54 / 1639.14 | 1.093x / 1.095x |

完整逐行 reduction、r10 相对 round-9 变化和 baseline provenance 见
[baseline_vs_r10_python_comparison.json](../results/r10_latest/baseline_vs_r10_python_comparison.json)。

## 17.5 Round-9 对照与解释

r10 使用同一 package config 和同一 OpenVINO source revision，但在本次测量中相对
round-9 的 N=1 latency 普遍高约 1--14%，INT4 1024-token text/query 高约 5.3--5.5%。
这说明当前结果仍明显优于初始 stock，但不能把 r10 的测量波动解释成新 patch 收益。
正式累计优化结论以“initial stock -> r10”为准；round-9 仅作为同 plugin 的近期复测
对照。可能影响包括 GPU clock state、进程/缓存状态和运行顺序。

## 17.6 Document embedding

当前 GME commit 相对 round-9 只增加 MMDocIR evaluator、cache/resume 和文档改动，没有
改变 `gme_openvino.py`、package IR/config 或 plugin kernel。当前 dedicated INT4 Python
document run 已完成，结果为
[`benchmark_gme_r10_int4_vopt_vdq0_20260907.json`](../../document_embedding_eval_int8_int4_gpu/results/benchmark_gme_r10_int4_vopt_vdq0_20260907.json)。
INT8 和 INT4-gs64 继续使用 threshold=0 的同 runtime round-9 reference：
[`benchmark_gme_round9_vopt_vdq0_20260829.json`](../../document_embedding_eval_int8_int4_gpu/results/benchmark_gme_round9_vopt_vdq0_20260829.json)。

| representation, batch=8 | INT8 reference | INT4 causal r10 | INT4 gs64 reference |
| --- | ---: | ---: | ---: |
| page_text | 494.103 | 435.708 | 490.767 |
| text_chunk_256 | 494.869 | 438.270 | - |
| page_image_150 | 25235.720 | 24171.267 | 24641.368 |
| page_fused_150 | 25651.721 | 24661.086 | 25130.626 |

相对最初 document stock baseline，INT4 causal 的 batch=8 speedup 为 page_text
`1.275x`、text_chunk `1.265x`、page_image `2.806x`、page_fused `2.796x`；INT8
对应为 `1.150x/1.149x/2.714x/2.774x`。本轮曾尝试 dedicated r10 document
process；INT4 dedicated output 已写盘，详情和 baseline 表见
[document REPORT](../../document_embedding_eval_int8_int4_gpu/REPORT.md)。

## 17.7 Correctness

同一当前 runtime 的 `openvino_validation_r10_vdq0.json` 显示：FP32 四模式 cosine
为 1.0；INT4 causal 的 text/query/image/fused cosine 为
`0.941730/0.955173/0.943164/0.955472`，没有因为 r10 性能复测引入新的数值路径。
