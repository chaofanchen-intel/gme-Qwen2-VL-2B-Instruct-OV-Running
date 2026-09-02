# GME Qwen2-VL-2B-Instruct — GPU text/query/image/fused hotspot 分析与优化

针对 `gme-Qwen2-VL-2B-Instruct` 在 Intel iGPU 上 INT4 精度做纯文本
`text` / `query` embedding 时延迟偏高（历史 1024 token ≈ 281 ms）以及
`image` / `fused` visual SDPA 未充分 lower 的问题，做了完整的 hotspot 定位、
源码优化、数值验证和回归测试。

**所有源码改动基于 OpenVINO 官方 release tag `2026.3.0`（`8a17657b99`），
不掺杂本仓库既有的任何本地提交或工作区改动**，patch 见 [`patches/`](patches/)。

---

## TL;DR

| | |
| --- | --- |
| **结果（plugin patch，第一/二轮）** | 1024 token **1.170–1.172x**、512 token **1.180–1.184x**、128 token **1.151–1.158x**、32 token **1.017–1.018x** |
| **结果（第三轮，导出侧 SDPA causal，可叠加）** | 1024 token 再 **1.073x**（gs128 238.5 → **222.3 ms**）/ **1.072x**（gs64 252.9 → **236.0 ms**）；SDPA 单项 **2.24x**。累计相对最初 stock gs128 **1.273x**。见 [`docs/09`](docs/09_round3_further_latency_optimization.md) |
| **结果（第四轮，导出侧 mask-free 视觉 SDPA）** | **image/fused 端到端 1.79–2.50x**（448x448 563.6 → **288.7 ms**；1288x728 4494.4 → **1798.3 ms**）；视觉 IR 单项 1.53–2.83x，收益随分辨率上升。见 [`docs/11`](docs/11_round4_image_fused_optimization.md) |
| **结果（第五轮，视觉塔布局 + 补 int8_causal）** | image 端到端再 **1.07–1.13x**（448x448 288.7 → **256.6 ms**；1288x728 1798.3 → **1677.6 ms**），**累计 2.20–2.70x**；text/query 的 **`int8_causal` 补漏 −15.2 ms（1.058x）**。见 [`docs/12`](docs/12_round5_vision_layout_and_int8_causal.md) |
| **结果（第六轮，DQ/RMS + image token cap）** | arithmetic mean：DQ reuse 140→112 calls；长序列 RMS→DQ opt-in 在 1024 token **220.28 → 211.63 ms（1.041x）**，129+ token 与 padded batch 对 control 逐位相同。显式 `max_image_tokens=768` 在超宽 logo 上 **1.94x**，MMDocIR `balanced_100` page/layout encoding **1.183x/1.460x**。见 [`docs/13`](docs/13_round6_rms_dq_and_image_token_cap.md) |
| **结果（第七轮，视觉 MVN→DQ）** | arithmetic mean：INT4 视觉 IR **1.036–1.053x**；public image/fused 端到端 **1.027–1.039x**。63 个 block LayerNorm 融入 DQ，MVN 65→2 calls；四个 grid、真实 public API 和 cache reload 全部 **bit-identical**。见 [`docs/14`](docs/14_round7_visual_mvn_dq_fusion.md) |
| **结果（第八轮，QKV transpose）** | arithmetic mean：专用 `permute_qkv_headfirst` 让 INT4 visual **1.005–1.020x**、public image/fused **1.004–1.017x**；INT8 visual **1.006–1.020x**。三精度四-grid、golden 和 cache reload 全部 bit-identical。INT8 grouped MVN→DQ 虽逐位正确但回归 1.7–3.7%，否决。见 [`docs/15`](docs/15_round8_qkv_permute_and_int8_grouped_mvn.md) |
| **结果（第九轮，latest package/runtime final rerun）** | GPU Python public API arithmetic mean：1024-token text/query 为 `253.36/252.94` ms（INT8）、`216.73/217.05` ms（INT4）、`231.18/231.32` ms（gs64）；448x448 image/fused 为 `275.23/277.43`、`242.23/241.77`、`253.20/252.83` ms。visual threshold=0；新增 `int4_visint8` 在 1288x728 fused 触发 `CL_OUT_OF_RESOURCES`，不作为部署候选。完整数据见 [`docs/16`](docs/16_round9_final_rebenchmark.md) |
| **正确性** | text/语言 IR 结果见 §4；image-only 三尺寸 bit-identical；fused 仍可运行但部分尺寸有小幅稳定漂移，见 `docs/04_validation_and_results.md` §4.1c。第三轮 causal 改造在 FP32 上**逐位相同**（含不等长右 padding 批），docs/09 §9.2.5。第四轮 mask-free 视觉塔在 FP32 上**逐位相同**（5 个 grid，`max_abs_diff = 0.0`），docs/11 §11.3.3 |
| **回归** | 有状态生成式模型 greedy 解码 token id、首步 logits SHA 与 top-8 逐位相同 |
| **改动** | opt1–5 见 round-7 combined；第八轮另提供增量 `opt6_qkv_headfirst_permute.patch`（4 files）和 opt1–6 combined patch。第六至八轮包含 GPU plugin 与 Python 配置/测量入口 |
| **改了什么** | ① GQA Broadcast；② SDPA Transpose；③ 宽 INT4 layout；④ language causal；⑤ visual mask-free；⑥ headfirst + rank3；⑦ DQ reuse；⑧ RMS→DQ；⑨ visual MVN→DQ；⑩ **QKV head-first 专用 half8 transpose kernel** |
| **推荐精度** | package 默认 precision 是 **`int8_causal_vopt`**；低延迟/均衡用 **`int4_causal_vopt`**，更高精度的 INT4 选 `int4_gs64_causal_vopt`。当前 package `max_image_tokens=1280`；低时延可显式设 768 |
| **推荐运行配置** | 默认 language threshold=64；512+ token 的独立 INT4 实例可设 `dynamic_quantization_threshold=0`。INT4 image/fused 实例设 `visual_dynamic_quantization_threshold=0` 启用本轮融合；它不影响 language。INT8 visual 是 group=128 + precomputed reduction，当前不触发 |
| **剩下的大头** | 文本侧仍是 oneDNN INT4 GEMM。视觉 L=1024 主要是 GEMM/SDPA，L=5040 是全注意力 SDPA。普通 rank-4 QKV direct-write 已证实会掉出 compressed i8 路径；INT8 grouped MVN 已证实性能回归。下一步必须深入 oneDNN blocked QKV dst 或 SDPA，而不是继续图层面改写 |

上表中的 plugin profiling 和 stage-split 数字属于历史轮次。历史 package 的独立
第五轮 final rerun（2026-08-27，Python/GPU，repeat=5，主要看 mean）使用
`int8_causal_vopt`、`int4_causal_vopt` 和 `int4_gs64_causal_vopt`，原始结果见
[`results/r5_final_20260827/`](results/r5_final_20260827/)。最初未优化 stock
结果保留在 `results/base_ops_int4_1024.json`、`results/r4_20260826/` 和
根目录 `artifacts/benchmark_matrix_fresh_r1.json`，用于累计对照。

下面第一至四轮表格中的 `wall p50` 是旧报告的存档字段；第六轮起的重新测量、当前
性能结论和新增 profiler 均使用 arithmetic mean，不再用 p50 判断优化收益。

## 优化前后的 GPU 时间构成（语言 IR，seq=1024，INT4/GPU）

| exec_type | baseline (ms) | opt1+2 (ms) | opt1+2+3 (ms) | Δ 总计 |
| --- | ---: | ---: | ---: | ---: |
| `jit:gemm:any__i8`（oneDNN，所有线性层） | 177.99 | 177.51 | **151.10** | **−26.9** |
| `ocl::sdpa::opt__f16` | 28.03 | 26.05 | 25.90 | −2.1 |
| `dynamic_quantize_gpu_opt__f16` | 26.55 | 25.24 | 25.57 | −1.0 |
| `broadcast_gpu_ref__f16`（GQA repeat_kv） | **10.10** | **0** | **0** | **−10.1** |
| `rms_gpu_bfyx_opt__f16` | 9.14 | 9.42 | 9.38 | +0.2 |
| `permute_ref__f16`（attention transpose） | **4.93** | **0.88** | **0.88** | **−4.1** |
| `ocl::rope::opt___f16` | 4.65 | 4.64 | 4.66 | 0 |
| 其它 | 2.41 | 2.40 | 2.40 | 0 |
| **profiled 合计** | **263.8** | **246.2** | **219.9** | **−43.9** |
| **wall p50** | **283.0** | **264.1** | **238.2** | **−44.8 (−15.8%)** |

opt-3 的 26.4 ms **全部来自 `gate_proj`（63.00 → 49.55）和 `up_proj`（47.43 → 35.24）**，
其余线性层和其余算子不动 —— 这正是该优化的适用范围（见 docs/06 §6.9）。

### 第三轮叠加导出侧 SDPA causal 之后（语言 IR，seq=1024，INT4/GPU，gs128）

| exec_type | opt1+2+3 | + causal 导出 | Δ |
| --- | ---: | ---: | ---: |
| `jit:gemm:any__i8` | 152.45 | 152.81 | +0.4 |
| **`ocl::sdpa::opt__f16`** | **25.99** | **11.62** | **−14.4** |
| `dynamic_quantize_gpu_opt__f16` | 27.04 | 25.49 | −1.6 |
| `rms_gpu_bfyx_opt__f16` | 9.34 | 9.39 | 0 |
| `permute_ref__f16` | 0.88 | 0.88 | 0 |
| host（wall − profiled） | 18.03 | 16.50 | −1.5 |
| **wall p50** | **238.48** | **222.33** | **−16.2 (1.073x)** |

SDPA 之外的收益来自被删掉的 mask 构造算子（`Select` 等）和随之下降的 primitive 数。
GEMM 完全不动——这一笔与 GEMM 侧的任何后续优化正交。详见 docs/09 §9.2。

### 第四轮：视觉塔（image / fused 路径，INT4/GPU，`448x448` → L=1024 patch 行）

前三轮只动语言 IR，image/fused 因此一直没变快。第四轮发现视觉塔的 32 个 SDPA 全部带着一张
**rank-3 BOOL `[1,L,L]` mask**，GPU plugin 拒绝 lower，注意力被拆成手工 MatMul+Softmax；
而这张 mask 对本包（一次只喂一张图）**恒为全 True**。

| exec_type | 优化前 | mask-free | Δ |
| --- | ---: | ---: | ---: |
| `jit:gemm:any__i8`（权重线性层） | 89.75 | 90.18 | +0.4 |
| `jit:gemm:any__f16`（QKᵀ / PV） | **67.44** | **0.01** | **−67.4** |
| `softmax_gpu_bf__f16` | **54.28** | **0.00** | **−54.3** |
| `permute_ref__f16` | 48.26 | 38.36 | −9.9 |
| **`ocl::sdpa::opt__f16`** | **0.00** | **33.82** | **+33.8** |
| `select_gpu_ref__u8`（mask） | **3.03** | **0.00** | **−3.0** |
| host（wall − profiled） | 28.10 | 15.17 | −12.9 |
| **wall p50** | **326.66** | **212.99** | **−113.7 (1.53x)** |

视觉塔的序列长度是 **merge 前的 patch 行数 L = 4K**，注意力 O(L²)，所以收益随分辨率上升：
L=1024 上 1.53x，L=5040 上 **2.83x**。端到端 image/fused **1.79–2.50x**。详见 docs/11。

## 历史第五轮 final rerun

第五轮当时的 end-to-end 结果来自同一套 custom core/plugin/Python binding：
`2026.3.0-22451-8a17657b995-qwen-ttft-optimizations`。没有使用官方 wheel
runtime，也没有重新运行最初 stock baseline。

| workload | `int8_causal_vopt` | `int4_causal_vopt` | `int4_gs64_causal_vopt` |
| --- | ---: | ---: | ---: |
| text/query, 32 tokens (mean ms/call) | 43.75 / 43.37 | 32.47 / 32.07 | 36.64 / 36.27 |
| text/query, 128 tokens (mean ms/call) | 54.13 / 53.53 | 40.30 / 39.78 | 43.42 / 42.92 |
| text/query, 512 tokens (mean ms/call) | 128.60 / 127.97 | 119.25 / 119.01 | 139.35 / 138.66 |
| text/query, 1024 tokens (mean ms/call) | 260.95 / 262.29 | 225.47 / 225.54 | 240.39 / 239.83 |
| image/fused, 448x448 (mean ms/call) | 285.13 / 284.54 | 262.76 / 257.84 | 269.60 / 268.09 |
| image/fused, 1288x728 (mean ms/call) | 1750.35 / 1757.35 | 1688.70 / 1688.79 | 1707.59 / 1711.78 |

GPU `padded + streams=1`、text=128 的 N=8 为 `31.90`、`27.53`、`29.35`
ms/embedding（INT8 causal、INT4 causal、INT4 gs64 causal）；448x448 image
对应为 `260.55`、`247.79`、`253.42` ms/embedding。完整 N sweep 和
`streams=auto` 数据见：

- [`benchmark_text_image_matrix.json`](results/r5_final_20260827/benchmark_text_image_matrix.json)
- [`benchmark_batch_int8_causal_vopt.json`](results/r5_final_20260827/benchmark_batch_int8_causal_vopt.json)
- [`benchmark_batch_int4_causal_vopt.json`](results/r5_final_20260827/benchmark_batch_int4_causal_vopt.json)
- [`benchmark_batch_int4_gs64_causal_vopt.json`](results/r5_final_20260827/benchmark_batch_int4_gs64_causal_vopt.json)

初始 stock INT8 在相同 text/image workload 上的完整数据位于
[`../artifacts/benchmark_matrix_fresh_r1.json`](../artifacts/benchmark_matrix_fresh_r1.json)。
以 1024-token text 为例，初始 stock text/query 为 296.57/296.01 ms，
第五轮 `int8_causal_vopt` 为 260.95/262.29 ms；448x448 image/fused
初始为 476.69/468.07 ms，最终为 285.13/284.54 ms。

## 当前第九轮 latest package/runtime rerun

第九轮使用当前 package `gme-Qwen2-VL-2B-Instruct`（GME `d3cf918`）和
OpenVINO `2026.3.0-22455-2080527e54b-qwen-ttft-optimizations`，GPU visual
threshold 显式设为 0。所有性能数字是 arithmetic mean；完整结果、batch、FP32
cosine、document embedding 和 MMDocIR smoke 见
[`docs/16`](docs/16_round9_final_rebenchmark.md)。

| workload | `int8_causal_vopt` | `int4_causal_vopt` | `int4_gs64_causal_vopt` |
| --- | ---: | ---: | ---: |
| text/query, 1024 tokens (ms/call) | 253.36 / 252.94 | **216.73 / 217.05** | 231.18 / 231.32 |
| image/fused, 448x448 (ms/call) | 275.23 / 277.43 | **242.23 / 241.77** | 253.20 / 252.83 |
| image/fused, 1288x728 (ms/call) | 1728.10 / 1731.73 | **1618.10 / 1623.11** | 1657.60 / 1658.56 |

N=8、128-token、GPU `padded + streams=1` 的 text/query ms/embedding 为
`31.23/33.27`（INT8）、`26.46/28.35`（INT4）、`28.28/31.90`（gs64）。新增
`int4_causal_vnomask_qkvfused` 和 `int4_visint8` 完成 cosine smoke；后者在
1288x728 fused 上触发 `CL_OUT_OF_RESOURCES`，均不纳入正式部署结论。

## 历史第四轮 fresh rerun

当前可引用的 end-to-end 结果来自同一套 custom core/plugin/Python binding：
`2026.3.0-22451-8a17657b995-qwen-ttft-optimizations`。它不是官方 wheel
runtime，也不是旧的 masked visual 组合。

| workload | `int8_vnomask` | `int4_causal_vnomask` |
| --- | ---: | ---: |
| text/query, 32 tokens (mean ms/call) | 44.12 / 43.80 | 32.32 / 31.99 |
| text/query, 128 tokens (mean ms/call) | 54.18 / 54.15 | 41.86 / 41.65 |
| text/query, 512 tokens (mean ms/call) | 135.56 / 135.63 | 119.36 / 119.35 |
| text/query, 1024 tokens (mean ms/call) | 279.68 / 280.77 | 226.00 / 226.16 |
| image/fused, 448x448 (mean ms/call) | 310.53 / 306.94 | 296.37 / 296.43 |
| image/fused, 1288x728 (mean ms/call) | 1850.57 / 1850.86 | 1800.29 / 1795.29 |

batch 使用 text=128 tokens 和 image=448x448；GPU `padded + streams=1` 在 N=8
时分别为 `32.47` 和 `27.64 ms/embedding`（INT8/causal INT4 text），
image 为 `284.46` 和 `279.71 ms/embedding`。完整的 N=1/2/4/8/16/32、
`per_sample`/`padded`、`streams=1/auto` 数据见：

- [`benchmark_text_image_matrix.json`](results/r4_rerun_20260826/benchmark_text_image_matrix.json)
- [`benchmark_batch_int8_vnomask.json`](results/r4_rerun_20260826/benchmark_batch_int8_vnomask.json)
- [`benchmark_batch_int4_causal_vnomask.json`](results/r4_rerun_20260826/benchmark_batch_int4_causal_vnomask.json)

这次 fresh matrix 的详细 document embedding 评测另见
[`document_embedding_eval_int8_int4_gpu/REPORT.md`](../document_embedding_eval_int8_int4_gpu/REPORT.md)。

---

## 文档

| 文档 | 内容 |
| --- | --- |
| [`docs/01_methodology_and_baseline.md`](docs/01_methodology_and_baseline.md) | 被测对象、为什么必须重建干净基线、三套测量工具、理论算力标尺 |
| [`docs/02_hotspot_analysis.md`](docs/02_hotspot_analysis.md) | VTune GPU hotspots、算子级归因、GEMM 为什么只有 15 TFLOPS、两处图融合缺口 |
| [`docs/03_optimizations.md`](docs/03_optimizations.md) | 三个 patch 的问题、改法、为什么安全、逐项收益 |
| [`docs/04_validation_and_results.md`](docs/04_validation_and_results.md) | 数值验证（opt1+2 bit-identical / opt3 的漂移量级）、生成式回归、交替 A/B 性能、VTune 复核 |
| [`docs/05_rejected_and_future_work.md`](docs/05_rejected_and_future_work.md) | 试过且**否掉**的 3 个方案（附实测数字）、后续 5 个方向、与既有 Qwen3.5 patch 的关系 |
| [`docs/06_round2_onednn_investigation.md`](docs/06_round2_onednn_investigation.md) | **第二轮**：oneDNN 单独立项、INT4 GEMM 成本分解 probe、VTune 复采、opt-3 的落地与两次规则修正（§6.9） |
| [`docs/08_per_oc_quantization.md`](docs/08_per_oc_quantization.md) | per-OC INT4 量化方案（docs/06 §6.3 挂起项落地）：两个新精度、性能基线、精度评估——**收益大但精度不达标，不建议采纳** |
| [`docs/09_round3_further_latency_optimization.md`](docs/09_round3_further_latency_optimization.md) | **第三轮**：gs64 首次算子级基线、oneDNN 定价表 round-3、**导出侧 SDPA causal（采纳，1.073x）**、分组对称量化（可选）、以及 4 个带实测数字的否决项 |
| [`docs/10_round3_gme_causal_performance_report.md`](docs/10_round3_gme_causal_performance_report.md) | **第三轮统一性能报告**：普通/causal、gs128/gs64、per-OC/对称量化的 1024-token wall、GEMM、SDPA、plugin 阶段和数据缺口 |
| [`docs/12_round5_vision_layout_and_int8_causal.md`](docs/12_round5_vision_layout_and_int8_causal.md) | **第五轮**：补上漏掉的 **`int8_causal`**（−15.2 ms，且落在默认精度上）；视觉塔两笔纯重排优化——**`headfirst`**（四趟 Q/K/V 搬运并成一趟，`permute_ref` 128→32 calls）与 **`rank3`**（给 ViT 投影加 batch 维，解锁 opt-3，`mlp.fc1` −28~32%）；含一个失败尝试的实测（per-Linear rank3 被 64 次 eltwise 拷贝反噬） |
| [`docs/11_round4_image_fused_optimization.md`](docs/11_round4_image_fused_optimization.md) | **第四轮**：image/fused 的四段拆分、视觉塔 hotspot（rank-3 BOOL mask 让 SDPA 不 lower）、**mask-free 视觉 SDPA（采纳，端到端 1.79–2.50x）**、transformers 5 下的视觉输出接错 bug、一个带实测交叉点的否决项，以及两个设计问题的实测答案：**为什么逐图而不是像官方那样打包**（§11.2.6）、**batch 在 image 上能带来多少**（§11.6.3） |
| [`patches/README.md`](patches/README.md) | patch 的 base、应用方式、影响面、上游建议 |
| [`results/qwen3_ab_text/REPORT.md`](results/qwen3_ab_text/REPORT.md) | 三个 GME patch 对 Qwen3-VL-Embedding-2B 的适用性复测：opt-3 text 基本无收益，image/fused `CL_OUT_OF_RESOURCES` 回归 |

## 脚本

| 脚本 | 用途 |
| --- | --- |
| `profile_gme_int4_gpu.py` | 端到端 GME encoder 延迟（含 processor/tokenizer），workload 结果磁盘缓存 |
| `profile_ops.py` | 语言 IR 的 per-op GPU profiling，按 exec_type + node name 双维度聚合 |
| `profile_visual_ops.py` | **视觉 IR** 的 per-op GPU profiling（`profile_ops.py` 的视觉侧对应物），`--grid t,h,w` 或 `--tokens K` |
| `profile_gme_image.py` | image/fused 端到端，拆成 preprocess / visual / language / end-to-end 四段 |
| `check_vision_equivalence.py` | mask-free 视觉 IR 与原 IR 的 FP32 逐位等价性（多个 grid） |
| `check_vision_batch_contract.py` | 多图 batch 下每张图的视觉输出是否与单独跑逐位相同（单图契约） |
| `check_vision_mask_necessity.py` | **那张 mask 到底有没有用**：枚举单图 / 多行 grid / `t>1` 三种形态在两个 IR 上的实际行为，并在 PyTorch 上量化「打包≡逐图」与「真串味」的差距 |
| `dump_outputs.py` / `compare_outputs.py` | 固定输入下 dump 输出并做 cosine / max-abs-diff 对比 |
| `regress_stateful.py` | **GenAI-free** 有状态 IR greedy 解码回归（推荐） |
| `regress_generative.py` | 走 `openvino_genai` 的回归；⚠️ 本地 GenAI 构建与 wheel 不同，会有假阳性，见 docs/04 §4.2 |
| `bench_ab.sh` | 交替 A/B 端到端 benchmark（应对 GPU 双档频率） |

## 复现

```bash
cd /path/to/gme-Qwen2-VL-2B-Instruct-Running/gme-qwen2-vl-int4-hotspot-optimization
export HOTSPOT_ROOT="$PWD"
export GME_ROOT="${GME_ROOT:-$(cd "$HOTSPOT_ROOT/.." && pwd)}"
export MODEL_BENCH_ROOT="${MODEL_BENCH_ROOT:-$(cd "$GME_ROOT/.." && pwd)}"
export OPENVINO_ROOT="${OPENVINO_ROOT:-$MODEL_BENCH_ROOT/benchmark/openvino}"
export QWEN_ENV="${QWEN_ENV:-/home/chaofan/envs/qwen_env}"

source /opt/intel/oneapi/setvars.sh                       # VTune
source "$QWEN_ENV/bin/activate"

# --- 基线（官方 2026.3.0 wheel）---
python profile_ops.py --precision int4 --length 1024 --warmup 12 --repeat 8
python dump_outputs.py --precision int4 --output results/acc_int4_stock.npz

# --- 打 patch 并重编 GPU plugin ---
cd "$OPENVINO_ROOT"
git apply "$HOTSPOT_ROOT/patches/all_gme_int4_gpu_opts.patch"
cmake --build build --target openvino_intel_gpu_plugin -j && cmake --install build

# --- patched ---
source "$OPENVINO_ROOT/install/setupvars.sh"
python profile_ops.py --precision int4 --length 1024 --warmup 12 --repeat 8
python dump_outputs.py --precision int4 --output results/acc_int4_round2.npz
python compare_outputs.py results/acc_int4_stock.npz results/acc_int4_round2.npz

# --- 端到端交替 A/B + 回归 ---
./bench_ab.sh                     # 默认 1024 / 512 / 128 / 32
python regress_stateful.py --output results/r2_regress_stateful_patched.json

# --- VTune ---
vtune -collect gpu-hotspots -result-dir results/vtune-gpu-hotspots-int4-1024-patched \
      -- python profile_gme_int4_gpu.py --length 1024 --warmup 14 --repeat 30 --hold-seconds 30
```

> `--hold-seconds` 是必须的：加了 workload 缓存之后进程结束太快，VTune 会报
> "main thread is a zombie thread" 而 attach 失败。

## 目录

```
├── README.md                        本文件
├── docs/                            分析过程 / hotspot / 优化 / 测试结论
├── patches/                         基于官方 2026.3.0 的可交付 patch
├── onednn/                          第二轮的独立 oneDNN probe（脱离 OpenVINO 直接定价 GEMM）
│   ├── *_probe.cpp                  GEMM 成本分解 / u4 重排 / scale 排布 / layout 精度
│   ├── build_probe.sh               链接 oneDNN-ov2026.3 (babb7375ff)
│   └── results/                     probe 输出
├── results/                         profiling 输出、A/B 原始数据、精度对比、VTune 导出
│   ├── README.md                    每个结果文件的口径说明（务必先看这个）
│   └── ab/                          交替 A/B 的逐轮 JSON
├── preexisting-worktree-backup/     开工前 openvino / openvino.genai 工作区改动的备份
└── build-openvino-official/         早期尝试的独立官方源码构建目录（未使用，已 gitignore）
```

本目录现在位于 `gme-Qwen2-VL-2B-Instruct-Running/` 下。Python 脚本会按该布局
自动推导路径；迁移到其他工作区时可通过 `GME_ROOT`、`MODEL_BENCH_ROOT`、
`OPENVINO_ROOT` 和 `MODELS_ROOT` 覆盖默认值。`QWEN_ENV` 用于指定 Python 虚拟环境。

`.gitignore` 排除了 VTune 原始 result 目录（数百 MB、机器相关）。summary 导出为
`results/vtune_gpu_summary{,_patched}.txt`，按 computing-task 的完整指标表导出为
`results/vtune_tasks_{baseline,patched}.tsv`。

## 关于工作区状态

开工时 `benchmark/openvino` 工作区里 apply 着 Qwen3.5 的 5 个 patch，其中
`opt2_permute_lws_f_axis` 和 `opt3_dynamic_quantize_reuse` **正落在 GME 的热点路径上**，
用它做基线会污染结论。处理方式：

1. 先 `git diff` 备份到 `preexisting-worktree-backup/`（**未丢失，可随时恢复**）；
2. `git checkout -- src/` 回到 `2026.3.0`；
3. 重编 GPU plugin，以此为唯一基线。

`benchmark/openvino.genai` **未做任何修改**，其工作区改动也已备份。恢复方式：

```bash
cd benchmark/openvino
git apply ../../gme-qwen2-vl-int4-hotspot-optimization/preexisting-worktree-backup/openvino-worktree-before-gme-work.diff
```
