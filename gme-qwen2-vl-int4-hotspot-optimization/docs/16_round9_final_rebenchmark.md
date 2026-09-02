# 第九轮：latest package / runtime 完整复测

本轮目标是用当前 package 和最新 custom OpenVINO runtime 重新测量 GME-Qwen2-VL-2B 的
`text`、`query`、`image`、`fused`，包含 batch，并重新计算相对于 FP32 golden 的 cosine。
同时重跑 document embedding evaluator，并检查 MMDocIR backend 是否能显式启用视觉
MVN-to-DynamicQuantize fusion。

所有性能结论使用原始 measured samples 的 arithmetic mean，不使用 p50 作优化判断。

## 16.1 测试边界和 provenance

| 项 | 本轮值 |
| --- | --- |
| GME revision | `d3cf918` |
| OpenVINO source revision | `2080527e54` |
| OpenVINO runtime | `2026.3.0-22455-2080527e54b-qwen-ttft-optimizations` |
| device | Intel(R) Graphics (iGPU) |
| package | `ov_models/gme-qwen2-vl-2b-instruct-openvino` |
| package config SHA256 | `0f7bcad4977f7078a425d1b2b11b1ae1f7b5b9e444f963e71f2ec765cb83bee7` |
| package processor | `min_image_tokens=256`, `max_image_tokens=1280`, `max_length=1800` |
| visual threshold | `GPU_DYNAMIC_QUANTIZATION_THRESHOLD=0`，只传给 visual IR |
| public benchmark | Python API，warm-up 1 次，measured repeat=5 |
| batch benchmark | GPU，`batch_mode=padded`，`streams=1`，128 effective text tokens，448x448 image |
| document benchmark | 14-page PDF，150 DPI，batch `1/4/8/14`，warm-up 1 次后 repeat=3 |

本轮公共矩阵原始结果：
[`benchmark_matrix_python_vdq0_formal.json`](../results/r9_latest/benchmark_matrix_python_vdq0_formal.json)。
Batch 原始结果分别位于：

- [`benchmark_batch_int8_causal_vopt_vdq0.json`](../results/r9_latest/benchmark_batch_int8_causal_vopt_vdq0.json)
- [`benchmark_batch_int4_causal_vopt_vdq0.json`](../results/r9_latest/benchmark_batch_int4_causal_vopt_vdq0.json)
- [`benchmark_batch_int4_gs64_causal_vopt_vdq0.json`](../results/r9_latest/benchmark_batch_int4_gs64_causal_vopt_vdq0.json)

初始 baseline 与 round-9 的统一计算结果见
[`baseline_vs_round9_comparison.json`](../results/r9_latest/baseline_vs_round9_comparison.json)。

## 16.2 正式部署 precision

| precision | language IR | visual IR |
| --- | --- | --- |
| `int8_causal_vopt` | `gme_language_int8_causal.xml` | `gme_visual_int8_nomask_headfirst_rank3.xml` |
| `int4_causal_vopt` | `gme_language_int4_causal.xml` | `gme_visual_int4_nomask_headfirst_rank3.xml` |
| `int4_gs64_causal_vopt` | `gme_language_int4_gs64_causal.xml` | `gme_visual_int4_nomask_headfirst_rank3.xml` |

`visual threshold=0` 使 INT4 visual 的 accepted opt-5 MVN fusion 可被启用；INT8
visual 仍是 `group_size=128 + precomputed_reduction`，不满足该 fusion 的适用条件。

## 16.3 Public API arithmetic mean

### Text / query，N=1，单位 ms/call

| effective tokens | INT8 text/query | INT4 text/query | INT4-gs64 text/query |
| ---: | ---: | ---: | ---: |
| 32 | 39.84 / 39.37 | 28.80 / 27.91 | 33.06 / 32.37 |
| 128 | 50.60 / 49.02 | 35.48 / 35.10 | 38.38 / 38.50 |
| 512 | 122.90 / 123.04 | 114.36 / 114.15 | 134.44 / 134.18 |
| 1024 | 253.36 / 252.94 | 216.73 / 217.05 | 231.18 / 231.32 |

### Image / fused，N=1，单位 ms/call

| source size | visual tokens | INT8 image/fused | INT4 image/fused | INT4-gs64 image/fused |
| --- | ---: | ---: | ---: | ---: |
| 448x448 | 256 | 275.23 / 277.43 | 242.23 / 241.77 | 253.20 / 252.83 |
| 672x672 | 576 | 661.96 / 658.72 | 595.29 / 592.59 | 614.78 / 612.65 |
| 1288x728 | 1196 | 1728.10 / 1731.73 | 1618.10 / 1623.11 | 1657.60 / 1658.56 |

`1288x728` 经 processor 得到 1196 visual tokens，仍在 package 的 1280 上限以内。
这些值是独立 workload 的 arithmetic mean；不要把不同 workload 的 N=1 结果当成同一
进程内 A/B 配对。

结论：当前三档中，`int4_causal_vopt` 在 text/query 和 image/fused 均为最低延迟；
`int4_gs64_causal_vopt` 是质量优先的折中；`int8_causal_vopt` 的 text/query cosine
最好，但视觉路径并不比 INT4 快。

### 16.3.1 与最初 baseline 的累计优化效果

最初 public INT8 baseline 来自无 GME patch 的
[`benchmark_matrix_fresh_r1.json`](../../artifacts/benchmark_matrix_fresh_r1.json)，
使用同样的 text/query 长度和 image size。最初 public INT4 的 text/query 使用
[`results/ab/`](../results/ab/) 中三轮 stock raw-run mean 的平均值。所有百分比为
`(baseline - round9) / baseline`，speedup 为 `baseline / round9`。

#### INT8：最初 stock public API -> round 9

| workload | baseline text/query | round-9 text/query | reduction | speedup |
| --- | ---: | ---: | ---: | ---: |
| 32 tokens | 51.55 / 49.41 | 39.84 / 39.37 | 22.72% / 20.32% | 1.294x / 1.255x |
| 128 tokens | 64.07 / 62.74 | 50.60 / 49.02 | 21.02% / 21.87% | 1.266x / 1.280x |
| 512 tokens | 144.65 / 139.36 | 122.90 / 123.04 | 15.04% / 11.71% | 1.177x / 1.133x |
| 1024 tokens | 296.57 / 296.01 | 253.36 / 252.94 | 14.57% / 14.55% | 1.171x / 1.170x |
| 448x448 image/fused | 476.69 / 468.07 | 275.23 / 277.43 | 42.26% / 40.73% | 1.732x / 1.687x |
| 672x672 image/fused | 1372.30 / 1338.03 | 661.96 / 658.72 | 51.76% / 50.77% | 2.073x / 2.031x |
| 1288x728 image/fused | 4654.46 / 4600.58 | 1728.10 / 1731.73 | 62.87% / 62.36% | 2.693x / 2.657x |

#### INT4：最初 stock text/query -> round 9

| workload | baseline text/query | round-9 text/query | reduction | speedup |
| --- | ---: | ---: | ---: | ---: |
| 128 tokens | 56.967 / 46.514 | 35.48 / 35.10 | 37.72% / 24.54% | 1.606x / 1.325x |
| 512 tokens | 157.318 / 146.188 | 114.36 / 114.15 | 27.31% / 21.92% | 1.376x / 1.281x |
| 1024 tokens | 291.443 / 280.598 | 216.73 / 217.05 | 25.64% / 22.65% | 1.345x / 1.293x |

最初 INT4 的 32-token public raw mean 没有保存在 baseline 产物中，旧的 32-token
p50 不纳入本轮比较。最初 masked INT4 image/fused 的端到端文件也只保留 p50；为
避免混用统计量，本报告不把它伪装成 mean。最早保留的完整 INT4 image/fused
arithmetic-mean 矩阵是 round-4 的 `int4_causal_vnomask`，即 mask-free 但尚未
加入 headfirst/rank3 的 pre-vopt 对照：

| workload | pre-vopt INT4 image/fused | round-9 image/fused | reduction | speedup |
| --- | ---: | ---: | ---: | ---: |
| 448x448 | 296.37 / 296.43 | 242.23 / 241.77 | 18.27% / 18.44% | 1.224x / 1.226x |
| 672x672 | 699.25 / 694.68 | 595.29 / 592.59 | 14.87% / 14.70% | 1.175x / 1.172x |
| 1288x728 | 1800.29 / 1795.29 | 1618.10 / 1623.11 | 10.12% / 9.59% | 1.113x / 1.106x |

这张 INT4 图像表是“最早可比的 mean pre-vopt”而不是绝对初始 masked baseline；
绝对初始 visual-only masked mean 仍可在 `r4_20260826/vis_base_int4_*.json` 中核查。

## 16.4 Batch arithmetic mean

每个 cell 是 `ms/call / ms/embedding`。图像只测 N=1/2/4/8，因为 API 当前按单图执行
visual IR，较大 N 不代表一个真正的 `[N,...]` visual padded batch。

### `int8_causal_vopt`

| N | text | query | image | fused |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 49.13 / 49.13 | 51.92 / 51.92 | 276.05 / 276.05 | 294.55 / 294.55 |
| 2 | 66.67 / 33.34 | 82.77 / 41.39 | 525.94 / 262.97 | 586.70 / 293.35 |
| 4 | 122.60 / 30.65 | 139.87 / 34.97 | 1045.41 / 261.35 | 1147.69 / 286.92 |
| 8 | 249.86 / 31.23 | 266.13 / 33.27 | 2087.05 / 260.88 | 2265.55 / 283.19 |
| 16 | 481.56 / 30.10 | 517.73 / 32.36 | - | - |
| 32 | 1003.24 / 31.35 | 1023.83 / 31.99 | - | - |

### `int4_causal_vopt`

| N | text | query | image | fused |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 35.52 / 35.52 | 42.45 / 42.45 | 241.37 / 241.37 | 270.84 / 270.84 |
| 2 | 53.95 / 26.97 | 66.08 / 33.04 | 468.15 / 234.08 | 522.79 / 261.40 |
| 4 | 111.14 / 27.79 | 115.67 / 28.92 | 925.72 / 231.43 | 1044.63 / 261.16 |
| 8 | 211.68 / 26.46 | 226.80 / 28.35 | 1875.74 / 234.47 | 2077.76 / 259.72 |
| 16 | 461.89 / 28.87 | 453.74 / 28.36 | - | - |
| 32 | 881.29 / 27.54 | 903.99 / 28.25 | - | - |

### `int4_gs64_causal_vopt`

| N | text | query | image | fused |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 38.67 / 38.67 | 48.52 / 48.52 | 252.90 / 252.90 | 285.11 / 285.11 |
| 2 | 62.25 / 31.12 | 77.01 / 38.50 | 486.48 / 243.24 | 548.82 / 274.41 |
| 4 | 132.16 / 33.04 | 134.41 / 33.60 | 955.21 / 238.80 | 1084.49 / 271.12 |
| 8 | 226.21 / 28.28 | 255.19 / 31.90 | 1934.85 / 241.86 | 2159.03 / 269.88 |
| 16 | 520.66 / 32.54 | 507.61 / 31.73 | - | - |
| 32 | 985.15 / 30.79 | 1005.16 / 31.41 | - | - |

N=8 时，正式三档的 text/query per embedding 分别为：

| precision | text | query |
| --- | ---: | ---: |
| INT8 | 31.23 ms | 33.27 ms |
| INT4 | **26.46 ms** | **28.35 ms** |
| INT4-gs64 | 28.28 ms | 31.90 ms |

## 16.5 FP32 cosine validation

命令入口是 `04_validate_python.py`，结果为
[`openvino_validation_round9_latest.json`](../../artifacts/openvino_validation_round9_latest.json)。
所有值都是相对于 PyTorch FP32 golden 的 cosine；`text` 是五条 golden text 的最小值。

| precision | text min | query | image | fused | fused norm |
| --- | ---: | ---: | ---: | ---: | ---: |
| `fp32` | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.00032 |
| `int8_causal_vopt` | 0.997899 | 0.998784 | 0.987115 | 0.996531 | 1.00032 |
| `int4_causal_vopt` | 0.941730 | 0.955173 | 0.943164 | 0.955472 | 1.00032 |
| `int4_gs64_causal_vopt` | 0.944205 | 0.957576 | 0.943657 | 0.958768 | 1.00030 |
| `int4_causal_vnomask_qkvfused` | 0.941730 | 0.955173 | 0.943164 | 0.955472 | 1.00032 |
| `int4_visint8` | 0.944538 | 0.955146 | 0.954492 | 0.964321 | 0.99983 |

新增的 `int4_causal_vnomask_qkvfused` 与 `int4_causal_vopt` 在该 golden probe 上逐项
相同，但本轮只做了 correctness smoke，没有将其纳入正式性能矩阵。`int4_visint8` 的
cosine 也完成了 smoke，但其大图 fused 性能在 1288x728 触发 GPU
`CL_OUT_OF_RESOURCES`，所以不作为部署候选或完整矩阵结论。

## 16.6 Document embedding evaluator

使用：
[`benchmark_gme_round9_vopt_vdq0_20260829.json`](../../document_embedding_eval_int8_int4_gpu/results/benchmark_gme_round9_vopt_vdq0_20260829.json)。
设置 `VISUAL_DQ_THRESHOLD=0`，三档正式 precision 均完成 14-page PDF 的
`page_text`、`text_chunk_256`、`page_image_150`、`page_fused_150`，batch 为
`1/4/8/14`，性能表中的主值是 arithmetic mean。

### 16.6.1 Document embedding：最初 INT8/INT4 baseline -> round 9

以下为 batch=8 的同一份 14-page PDF、同一四种表示。baseline 来自
[`document_embedding_benchmark_before_optimized.json`](../../document_embedding_eval_int8_int4_gpu/results/document_embedding_benchmark_before_optimized.json)，
round-9 来自本节上面的 JSON。gs64 没有对应的初始 baseline，因此不放入 speedup 表。

| representation | INT8 baseline -> round 9 | INT8 speedup | INT4 baseline -> round 9 | INT4 speedup |
| --- | ---: | ---: | ---: | ---: |
| page_text | 568.104 -> 494.103 ms | 1.150x (-13.03%) | 555.459 -> 437.855 ms | 1.269x (-21.17%) |
| text_chunk_256 | 568.450 -> 494.869 ms | 1.149x (-12.94%) | 554.420 -> 435.908 ms | 1.272x (-21.38%) |
| page_image_150 | 68482.061 -> 25235.720 ms | 2.714x (-63.15%) | 67823.817 -> 24254.520 ms | 2.796x (-64.24%) |
| page_fused_150 | 71149.129 -> 25651.721 ms | 2.774x (-63.95%) | 68941.041 -> 24666.638 ms | 2.795x (-64.22%) |

质量 probe 相对 baseline 的变化很小，但样本只有 8 queries/14 pages：INT8 的
page_text 与 page_image 指标保持不变，page_fused MRR 从 `0.4568` 到 `0.4506`；
INT4 的 page_text R@3 从 `0.625` 到 `0.500`、MRR 从 `0.4100` 到 `0.3996`，
page_image MRR 从 `0.5208` 到 `0.5193`，page_fused MRR 保持 `0.4464`。这些值
只能作为方向性 probe，不能替代完整业务检索评测。

### Mean ms/call

| representation / batch | INT8 b1 / b4 / b8 / b14 | INT4 b1 / b4 / b8 / b14 | gs64 b1 / b4 / b8 / b14 |
| --- | ---: | ---: | ---: |
| page_text | 784.18 / 506.54 / 494.10 / 551.35 | 595.36 / 451.71 / 437.86 / 512.26 | 656.48 / 513.71 / 490.77 / 577.08 |
| text_chunk_256 | 767.62 / 509.14 / 494.87 / 549.82 | 600.11 / 452.88 / 435.91 / 512.77 | 657.80 / 513.33 / 493.14 / 576.99 |
| page_image_150 | 25982.21 / 25249.64 / 25235.72 / 25313.74 | 24798.70 / 24294.12 / 24254.52 / 24219.29 | 25089.98 / 24679.77 / 24641.37 / 24632.99 |
| page_fused_150 | 26410.11 / 25812.15 / 25651.72 / 25658.71 | 25254.89 / 24590.84 / 24666.64 / 24749.83 | 25532.28 / 25032.14 / 25130.63 / 25199.75 |

### 8-query document probe

| precision | page_text R@1/R@3/R@5/MRR | page_image R@1/R@3/R@5/MRR | page_fused R@1/R@3/R@5/MRR |
| --- | --- | --- | --- |
| INT8 | 0.125 / 0.500 / 0.750 / 0.4020 | 0.250 / 0.750 / 0.875 / 0.5218 | 0.125 / 0.750 / 0.875 / 0.4506 |
| INT4 | 0.125 / 0.500 / 0.750 / 0.3996 | 0.250 / 0.750 / 0.875 / 0.5193 | 0.125 / 0.750 / 0.750 / 0.4464 |
| INT4-gs64 | 0.125 / 0.625 / 0.750 / 0.4124 | 0.375 / 0.875 / 0.875 / 0.5923 | 0.125 / 0.750 / 0.875 / 0.4360 |

这组 8-query/14-page 指标是工程 probe，只用于观察量化趋势，不替代业务检索集。
本轮没有看到明显的结构性质量崩溃；INT4 上线仍应由实际 Recall@K、nDCG 和 Top-K
评测决定。

## 16.7 MMDocIR balanced_200 质量结果和 audit

本轮对 `mmdocir_openvino_eval` 做了如下更新：

- `run_benchmark.py` 增加 `--visual-dq-threshold`；
- `run_campaign.sh` 增加 `VISUAL_DQ_THRESHOLD`，并把值加入默认输出目录名，例如
  `_vdq0`；
- `GmeOpenVINOBackend` 把 threshold 传给 visual IR；
- backend metadata 和 page/layout cache fingerprint 均包含
  `visual_dynamic_quantization_threshold`，不同 threshold 不会复用同一 embedding cache；
- 编码阶段增加 progress heartbeat、单次调用 watchdog、checkpoint 和 resume，减少
  oneDNN/ GPU plugin hang 时从头开始的风险；
- `README.md` 增加 workaround、质量结果和 cache 说明。

当前结果位于
[`balanced_200_recommended_vdq0/combined/summary.json`](../../mmdocir_openvino_eval/results/balanced_200_recommended_vdq0/combined/summary.json)。
它使用 200 queries、10 个 domain、34 documents、607 pages、7,956 layouts，完成
FP32、INT8、INT4 的 query/page/layout 编码和 close/open 评估。

### 质量主表

| precision | scope | page R@5/R@10 | page nDCG@5/@10 | layout R@5/R@10 | layout nDCG@5/@10 |
| --- | --- | ---: | ---: | ---: | ---: |
| FP32 | close | 0.8650 / 0.9333 | 0.7699 / 0.7968 | 0.5995 / 0.6774 | 0.4853 / 0.5143 |
| INT8 | close | 0.8663 / 0.9329 | 0.7671 / 0.7932 | 0.5824 / 0.6841 | 0.4728 / 0.5101 |
| INT4 | close | 0.8529 / 0.9121 | 0.7580 / 0.7802 | 0.5661 / 0.6603 | 0.4657 / 0.5008 |
| FP32 | open | 0.7571 / 0.8100 | 0.6678 / 0.6871 | 0.5208 / 0.5920 | 0.4232 / 0.4495 |
| INT8 | open | 0.7513 / 0.8108 | 0.6633 / 0.6844 | 0.5267 / 0.5899 | 0.4208 / 0.4441 |
| INT4 | open | 0.7425 / 0.8083 | 0.6519 / 0.6766 | 0.4825 / 0.5967 | 0.3992 / 0.4398 |

相对 FP32 的质量变化（percentage points）为：

| precision | scope | page R@5 | page R@10 | page nDCG@5 | page nDCG@10 | layout R@5 | layout R@10 | layout nDCG@5 | layout nDCG@10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| INT8 | close | +0.13 | -0.04 | -0.28 | -0.37 | -1.71 | +0.67 | -1.25 | -0.42 |
| INT4 | close | -1.21 | -2.12 | -1.19 | -1.66 | -3.35 | -1.71 | -1.97 | -1.35 |
| INT8 | open | -0.58 | +0.08 | -0.46 | -0.27 | +0.59 | -0.21 | -0.24 | -0.54 |
| INT4 | open | -1.46 | -0.17 | -1.60 | -1.05 | -3.83 | +0.47 | -2.40 | -0.97 |

结论：INT8 在这个 balanced_200 子集上最接近 FP32；INT4 的 page 和 layout 指标均出现
更明显下降，尤其是 layout Recall@5。该子集适合筛查趋势，但最终上线仍需完整
1,658-query 业务评测。

### INT4 死锁 workaround 和时间口径

2026-08-29 的默认 INT4 run 在 layout 阶段进入 GPU plugin/oneDNN primitive-cache 死锁；
2026-09-02 使用 `ONEDNN_PRIMITIVE_CACHE_CAPACITY=0` 重新完成全部 7,956 layouts。
该 workaround 绕过 cache key 哈希路径，但会显著增加 INT4 运行时间。因此本节**不展示
INT4 与 FP32/INT8 的 encoding latency、throughput、ETA 或 speedup 对比**；JSON 中的时间
字段只用于记录本次运行过程，不能作为 precision 间性能结论。

INT4 standalone 结果中的 `system.onednn_primitive_cache_capacity` 为 `0`；FP32/INT8
质量向量和 INT4 质量向量使用同一 balanced_200 selection，combined summary 只用于
汇总 quality delta。此前的 `r9_smoke_first1_rebenchmark_vdq0` 仍保留为单文档链路 smoke，
不再作为 balanced_200 当前质量结论。

## 16.8 复现入口

公共 API：

```bash
cd /home/chaofan/model_benchmark/gme-Qwen2-VL-2B-Instruct-Running
source /home/chaofan/model_benchmark/benchmark/openvino/install/setupvars.sh
./env-ref/bin/python 08_benchmark_matrix.py \
  --suite latency,text-lengths,image-sizes \
  --devices GPU --backends python \
  --precisions int8_causal_vopt,int4_causal_vopt,int4_gs64_causal_vopt \
  --visual-dq-threshold 0 --repeat 5
```

Batch：

```bash
./env-ref/bin/python 08_benchmark_matrix.py --suite batch \
  --devices GPU --backends python \
  --precisions int4_causal_vopt --batch-precision int4_causal_vopt \
  --visual-dq-threshold 0 --batch-modes padded --streams 1 --repeat 5
```

Document embedding：

```bash
cd document_embedding_eval_int8_int4_gpu
VISUAL_DQ_THRESHOLD=0 \
PRECISIONS=int8_causal_vopt,int4_causal_vopt,int4_gs64_causal_vopt \
OUTPUT="$PWD/results/benchmark_gme_round9_vopt_vdq0_20260829.json" \
/home/chaofan/envs/qwen_env/bin/python3 -u run_eval.py
```

MMDocIR：

```bash
cd mmdocir_openvino_eval
VISUAL_DQ_THRESHOLD=0 MAX_SAMPLES=1 \
PRECISIONS=fp32,int8_causal_vopt,int4_causal_vopt \
./run_campaign.sh
```
