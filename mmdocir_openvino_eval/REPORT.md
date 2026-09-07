# MMDocIR GME-2B OpenVINO Evaluation Report

状态：`balanced_200` 子集已完成 FP32、INT8、INT4 三档质量评测；full set 的
`int4_causal_vopt` 已完成 close/open 质量评测。full FP32/INT8 尚未完成，因此 full
set 暂不做相对 FP32 的精度 delta。本报告的 balanced 三精度结论使用
`results/balanced_200_recommended_vdq0/combined/summary.json`。

## 当前 full INT4 质量结果（2026-09-05）

结果文件为
[`results/full_recommended/int4_causal_vopt/summary.json`](results/full_recommended/int4_causal_vopt/summary.json)，
覆盖完整 MMDocIR Evaluation set：313 documents、1,658 queries、20,395 pages 和
170,338 layouts。运行使用 GPU、`int4_causal_vopt`、recommended profile，并在评测
阶段设置 `ONEDNN_PRIMITIVE_CACHE_CAPACITY=0`；已完成的 page/layout embedding cache
被直接复用，没有重新编码 170,338 个 layouts。

Page 指标包含全部 1,658 个 query。Layout 指标包含 1,657 个 query；唯一排除项是
上游 `query_id=700` 的零面积 bbox `[300,100,300,450]`，具体记录在 summary 的
`layout_exclusions` 中。该处理只影响 layout 聚合，不影响 page 聚合。

| scope | page Recall@5/10 | page nDCG@5/10 | layout Recall@5/10 | layout nDCG@5/10 |
|---|---:|---:|---:|---:|
| close | 0.8086 / 0.8812 | 0.6942 / 0.7204 | 0.5248 / 0.6193 | 0.4202 / 0.4532 |
| open | 0.5957 / 0.6615 | 0.5109 / 0.5341 | 0.3987 / 0.4691 | 0.3194 / 0.3434 |

本轮 INT4 使用 `ONEDNN_PRIMITIVE_CACHE_CAPACITY=0`，因此不把这次 full INT4 的
encoding 时间或 throughput 与其它 capacity 配置下的 FP32/INT8 运行作横向性能比较。
这些数值是当前 full-set 的 INT4 质量结果，不是三精度回归结论。

## 当前 balanced_200 质量结果（2026-09-02）

固定子集包含 200 queries、10 个 domain（每个 20 条）、34 documents、607 pages 和
7,956 layouts。三档均使用当前 custom OpenVINO runtime、GPU、
`visual_dynamic_quantization_threshold=0`、recommended batch profile；结果同时覆盖
close-domain 和 open-domain。

本轮的目标是质量分析，不是吞吐 benchmark。INT4 的 standalone run 使用
`ONEDNN_PRIMITIVE_CACHE_CAPACITY=0` 绕过 oneDNN primitive-cache 死锁；该 workaround
会显著增加运行时间，因此 INT4 与 FP32/INT8 的本轮 encoding 时间、throughput、
ETA 和 speedup **全部不作横向比较**。时间字段仍保留在 JSON 中作为运行记录。

### Page / layout 指标

指标格式为 `Recall@5 / Recall@10` 和 `nDCG@5 / nDCG@10`；nDCG 与 layout 指标遵循
本评测目录前文定义。

| precision | scope | page Recall | page nDCG | layout Recall | layout nDCG |
|---|---|---:|---:|---:|---:|
| FP32 | close | 0.8650 / 0.9333 | 0.7699 / 0.7968 | 0.5995 / 0.6774 | 0.4853 / 0.5143 |
| INT8 | close | 0.8663 / 0.9329 | 0.7671 / 0.7932 | 0.5824 / 0.6841 | 0.4728 / 0.5101 |
| INT4 | close | 0.8529 / 0.9121 | 0.7580 / 0.7802 | 0.5661 / 0.6603 | 0.4657 / 0.5008 |
| FP32 | open | 0.7571 / 0.8100 | 0.6678 / 0.6871 | 0.5208 / 0.5920 | 0.4232 / 0.4495 |
| INT8 | open | 0.7513 / 0.8108 | 0.6633 / 0.6844 | 0.5267 / 0.5899 | 0.4208 / 0.4441 |
| INT4 | open | 0.7425 / 0.8083 | 0.6519 / 0.6766 | 0.4825 / 0.5967 | 0.3992 / 0.4398 |

### 相对 FP32 的变化（percentage points）

正值表示量化结果高于 FP32，负值表示下降。该表只分析质量，不涉及本轮耗时。

| precision | scope | page R@5 | page R@10 | page nDCG@5 | page nDCG@10 | layout R@5 | layout R@10 | layout nDCG@5 | layout nDCG@10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| INT8 | close | +0.13 | -0.04 | -0.28 | -0.37 | -1.71 | +0.67 | -1.25 | -0.42 |
| INT4 | close | -1.21 | -2.12 | -1.19 | -1.66 | -3.35 | -1.71 | -1.97 | -1.35 |
| INT8 | open | -0.58 | +0.08 | -0.46 | -0.27 | +0.59 | -0.21 | -0.24 | -0.54 |
| INT4 | open | -1.46 | -0.17 | -1.60 | -1.05 | -3.83 | +0.47 | -2.40 | -0.97 |

### 质量结论

- **INT8 最接近 FP32**：page 指标变化均在约 0.6 percentage points 以内；layout
	Recall@5 的最大下降为 close-domain 的 1.71 points，open-domain 反而上升 0.59 points。
- **INT4 存在稳定但中等幅度的质量下降**：page Recall@5/10 在 close/open 分别下降
	`1.21/2.12` 和 `1.46/0.17` points；layout Recall@5 分别下降 `3.35` 和 `3.83`
	points。nDCG 也呈相同方向，尤其是 open layout nDCG@5 下降 2.40 points。
- 这些结果支持 `int8_causal_vopt` 作为质量优先默认候选；`int4_causal_vopt` 可以作为
	低资源候选，但上线前仍需在完整 1,658-query 集合上确认业务容忍度。
- 由于 INT4 使用 `capacity=0`，本轮不能从 encoding wall time 得出 INT4 相对 FP32 或
	INT8 的性能结论。此前 README/旧报告中的时间估算只作历史容量规划，不能与本轮 INT4
	数字混用。

### 结果与复现信息

- combined 质量报告：`results/balanced_200_recommended_vdq0/combined/summary.json`
- FP32：`results/balanced_200_recommended_vdq0/fp32/summary.json`
- INT8：`results/balanced_200_recommended_vdq0/int8_causal_vopt/summary.json`
- INT4：`results/balanced_200_recommended_vdq0/int4_causal_vopt/summary.json`
- INT4 结果的 `system.onednn_primitive_cache_capacity` 为 `0`；quality delta 使用同一
	balanced_200 选择和同一批 embedding cache 生成。

full INT4 结果见本报告第一节；由于 full FP32/INT8 尚未完成，本节 balanced_200
的三精度 delta 不能外推为 full-set 三精度质量结论。

## 对照目标

同事在 A100 上、batch=8 的 GME-2B 参考值如下。这些值用于识别数量级差异，不作为 OpenVINO FP32 的质量 baseline。

| Scope | Page Recall@5/10 | Page nDCG@5/10 | Layout Recall@5/10 | Layout nDCG@5/10 | Page throughput | Text layout throughput | Image layout throughput |
|---|---:|---:|---:|---:|---:|---:|---:|
| Close | 81.1/89.8 | 71.2/74.4 | 48.7/57.9 | 45.5/48.4 | 1.38 pages/s | 156.42 layouts/s | 1.00 layouts/s |
| Open | 61.2/66.7 | 53.4/55.4 | 38.5/45.9 | 36.6/38.9 | 同一索引 | 同一索引 | 同一索引 |

以下单文档表格是历史 smoke 结果，当前 balanced_200 结论以本报告顶部为准。

| Precision | Scope | Page R@5/10 | Page nDCG@5/10 | Layout R@5/10 | Layout nDCG@5/10 | vs FP32 conclusion |
|---|---|---:|---:|---:|---:|---|
| FP32 | Close | 100.0/100.0 | 81.0/81.0 | 39.0/39.0 | 27.5/27.1 | baseline |
| INT8 optimized | Close | 100.0/100.0 | 81.0/81.0 | 39.0/39.0 | 27.6/27.3 | page/Recall 无损，layout nDCG +0.15pp |
| INT4 optimized | Close | 100.0/100.0 | 86.1/86.1 | 35.7/39.0 | 24.0/24.6 | layout R@5 -3.24pp，nDCG@5 -3.51pp |
| FP32 | Open smoke | 100.0/100.0 | 81.0/81.0 | 39.0/39.0 | 27.5/27.1 | 单文档候选集与 Close 相同 |
| INT8 optimized | Open smoke | 100.0/100.0 | 81.0/81.0 | 39.0/39.0 | 27.6/27.3 | 单文档候选集与 Close 相同 |
| INT4 optimized | Open smoke | 100.0/100.0 | 86.1/86.1 | 35.7/39.0 | 24.0/24.6 | 单文档候选集与 Close 相同 |

上述只覆盖第一份真实文档 `PH_2016.06.08_Economy-Final.pdf`：6 queries、23 pages、153 layouts。样本过小，INT4 page nDCG 的 +5.13pp 是排序抖动，不代表量化提升；目前唯一合理的方向性信号是 INT8 很稳定、INT4 layout top-5 可能退化，必须由全量 1,658 queries 验证。

## 历史 Batch=1 初步性能（不代表当前 INT4 可比耗时）

下表为排除每种模态首个 batch 后的 steady-state 指标。Page 输入为整页 image；text layout 包括 text/equation；fused layout 包括 image/table 的 VLM text + crop image。

| Precision | Query latency | Page throughput | Text layout throughput | Fused layout throughput |
|---|---:|---:|---:|---:|
| FP32 | 281.9 ms | 0.159 pages/s | 3.53 layouts/s | 0.112 layouts/s |
| INT8 optimized | 123.3 ms | 1.335 pages/s | 15.63 layouts/s | 0.951 layouts/s |
| INT4 optimized | 263.5 ms | 1.444 pages/s | 17.72 layouts/s | 1.014 layouts/s |

这些 throughput 数字是 workaround 之前的历史记录；INT4 当前使用 `capacity=0`，因此
不应据此与 FP32/INT8 或 balanced_200 的运行时间比较。该文档图像尺寸分布较小且
batch=1，不替代当前质量结论。

## 历史 Batch sweep 结论（不代表当前 INT4 可比耗时）

已完成统一 batch=1/2 和量化模型混合 batch 4/8 的真实输入测试，并以 200 条平衡 query 做了 batch=1/2/4/8、每档 3 次复测。最终推荐：

| Precision | Query | Page | Text layout | Fused layout |
|---|---:|---:|---:|---:|
| FP32 | 8 | 1 | 1 | 1 |
| INT8 optimized | 8 | 1 | 4 | 1 |
| INT4 optimized | 8 | 1 | 4 | 1 |

200-query throughput（queries/s）：

| Precision | B1 | B2 | B4 | B8 | Selected |
|---|---:|---:|---:|---:|---:|
| FP32 | 5.83 | 6.09 | 7.83 | **8.16** | 8 |
| INT8 optimized | 18.62 | 29.05 | 49.32 | **65.60** | 8 |
| INT4 optimized | 20.10 | 35.03 | 55.63 | **73.05** | 8 |

Document 路径的 batch=1/2 或 1/2/4/8 关键结果：

| Precision/path | B1 | B2 | B4 | B8 | Selected |
|---|---:|---:|---:|---:|---:|
| FP32 page (pages/s) | **0.159** | 0.121 | - | - | 1 |
| FP32 text layout (layouts/s) | **3.53** | 2.94 | - | - | 1 |
| FP32 fused layout (layouts/s) | **0.112** | 0.0467 | - | - | 1 |
| INT8 text layout (layouts/s) | 15.63 | 23.28 | **26.23** | 22.26 | 4 |
| INT4 text layout (layouts/s) | 17.72 | 23.54 | **26.20** | 24.00 | 4 |
| INT8 fused layout (layouts/s) | **0.951** | 0.894 | - | - | 1 |
| INT4 fused layout (layouts/s) | **1.014** | 0.927 | - | - | 1 |

FP32 batch=2 的 page+layout wall time相对 batch=1 从约 324 秒升至 547 秒；更大 FP32 document batch 不再继续测试。量化 page batch=1/2 基本持平，选择 batch=1 是因为 fused 更快、显存更低且配置统一。所有 profile 的 Recall/nDCG 与 batch=1 完全一致。运行器和周末脚本现默认使用上述 precision-aware `recommended` profile。

## 资源与匹配

测试设备为 `Intel(R) Graphics (iGPU)`，OpenVINO 报告 shared total memory 30.7 GB。每个 inference call 前后采样的 observed GPU memory peak：FP32 14.70 GiB、INT8 2.65 GiB、INT4 2.22 GiB；内部瞬时峰值可能更高。对应整个 layout 阶段 observed process RSS peak 为 3.95/4.08/4.15 GiB。

预热后的 close-domain CPU cosine score + top-k 平均 latency：

| Precision | Page match | Layout match |
|---|---:|---:|
| FP32 | 0.046 ms | 0.275 ms |
| INT8 optimized | 0.058 ms | 0.254 ms |
| INT4 optimized | 0.039 ms | 0.250 ms |

当前 `qwen_env` 是 CUDA PyTorch build，`torch.xpu` 不可用，所以 GPU-resident match latency 明确标为 unavailable；OpenVINO embedding 编码始终运行在 GPU。

## 历史全量容量规划

全量实际构成为 20,395 pages、155,144 text layouts、292 equation layouts、7,596 image layouts、7,306 table layouts。下列 batch=1 时间是 workaround 之前的历史容量规划，不能用于当前 INT4 或三精度耗时比较；full 质量评测仍需单独安排。

## 推荐初步对比（质量结果已完成）

`balanced_200` 已完成 FP32/INT8/INT4 质量评测；INT4 使用 `ONEDNN_PRIMITIVE_CACHE_CAPACITY=0`，
因此旧的时间估算只作历史容量规划，不用于当前耗时比较。质量结果见本报告顶部。

`balanced_100` 约 2.6 小时，但最坏比例指标标准误约 5.0pp；`balanced_200` 约 3.5pp，增加约 1.4 小时即可获得更稳定的初步信号，因此周末脚本默认使用 200-query 版本。两个子集都按 domain 等额采样，避免 annotation 前 N 项以及 137-query News 文档带来的偏置。

自动执行入口为 `./run_weekend_suite.sh`：三精度分别运行并保存日志，最后复用 cache 生成 combined FP32 delta。完整命令和 systemd 周末调度方式见 `README.md`。

## 可比性说明

- A100 表格的 layout page 输入说明与本评测一致：page image；table/image fused；text/formula text。
- 上游 MMDocIR 只实现 page nDCG 和 layout area Recall。本报告的 layout nDCG 使用 graded bbox intersection 定义，需与 A100 结果确认口径后才能直接比较绝对值。
- batch sweep 的吞吐只比较非 cache-hit run；质量比较始终使用相同 query/data selection。
- close-domain 应先通过完整 313-document 结果，再运行 open-domain。子集 smoke 的数值不能用于模型质量结论。
- 数据已固定至 Hugging Face revision `bdcb36ecb3eee73667180ee3fb24fe433f6dd2a4`，三文件字节数与远端 metadata 完全一致。
- backend 确认为本地优化 build：`OpenVINO 2026.3.0-22455-2080527e54b-qwen-ttft-optimizations`。