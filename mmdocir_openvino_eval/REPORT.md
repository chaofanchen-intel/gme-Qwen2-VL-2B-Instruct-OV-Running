# MMDocIR GME-2B OpenVINO Evaluation Report

状态：评测框架、完整数据下载和真实数据 smoke test 已完成；全量结果待运行。canonical 初步结果为 `results/smoke_resource_b1/summary_b1.json`。

## 对照目标

同事在 A100 上、batch=8 的 GME-2B 参考值如下。这些值用于识别数量级差异，不作为 OpenVINO FP32 的质量 baseline。

| Scope | Page Recall@5/10 | Page nDCG@5/10 | Layout Recall@5/10 | Layout nDCG@5/10 | Page throughput | Text layout throughput | Image layout throughput |
|---|---:|---:|---:|---:|---:|---:|---:|
| Close | 81.1/89.8 | 71.2/74.4 | 48.7/57.9 | 45.5/48.4 | 1.38 pages/s | 156.42 layouts/s | 1.00 layouts/s |
| Open | 61.2/66.7 | 53.4/55.4 | 38.5/45.9 | 36.6/38.9 | 同一索引 | 同一索引 | 同一索引 |

本次最终判断以 OpenVINO FP32 为 baseline，分别报告 `int8_causal_vopt` 和 `int4_causal_vopt` 的指标变化：

| Precision | Scope | Page R@5/10 | Page nDCG@5/10 | Layout R@5/10 | Layout nDCG@5/10 | vs FP32 conclusion |
|---|---|---:|---:|---:|---:|---|
| FP32 | Close | 100.0/100.0 | 81.0/81.0 | 39.0/39.0 | 27.5/27.1 | baseline |
| INT8 optimized | Close | 100.0/100.0 | 81.0/81.0 | 39.0/39.0 | 27.6/27.3 | page/Recall 无损，layout nDCG +0.15pp |
| INT4 optimized | Close | 100.0/100.0 | 86.1/86.1 | 35.7/39.0 | 24.0/24.6 | layout R@5 -3.24pp，nDCG@5 -3.51pp |
| FP32 | Open smoke | 100.0/100.0 | 81.0/81.0 | 39.0/39.0 | 27.5/27.1 | 单文档候选集与 Close 相同 |
| INT8 optimized | Open smoke | 100.0/100.0 | 81.0/81.0 | 39.0/39.0 | 27.6/27.3 | 单文档候选集与 Close 相同 |
| INT4 optimized | Open smoke | 100.0/100.0 | 86.1/86.1 | 35.7/39.0 | 24.0/24.6 | 单文档候选集与 Close 相同 |

上述只覆盖第一份真实文档 `PH_2016.06.08_Economy-Final.pdf`：6 queries、23 pages、153 layouts。样本过小，INT4 page nDCG 的 +5.13pp 是排序抖动，不代表量化提升；目前唯一合理的方向性信号是 INT8 很稳定、INT4 layout top-5 可能退化，必须由全量 1,658 queries 验证。

## Batch=1 初步性能

下表为排除每种模态首个 batch 后的 steady-state 指标。Page 输入为整页 image；text layout 包括 text/equation；fused layout 包括 image/table 的 VLM text + crop image。

| Precision | Query latency | Page throughput | Text layout throughput | Fused layout throughput |
|---|---:|---:|---:|---:|
| FP32 | 281.9 ms | 0.159 pages/s | 3.53 layouts/s | 0.112 layouts/s |
| INT8 optimized | 123.3 ms | 1.335 pages/s | 15.63 layouts/s | 0.951 layouts/s |
| INT4 optimized | 263.5 ms | 1.444 pages/s | 17.72 layouts/s | 1.014 layouts/s |

相对 FP32，INT8 page/text/fused throughput 分别约 `8.4x/4.4x/8.5x`；INT4 分别约 `9.1x/5.0x/9.1x`。该文档图像尺寸分布较小且 batch=1，不替代全量 batch sweep。

## Batch sweep 结论

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

## 全量容量规划

全量实际构成为 20,395 pages、155,144 text layouts、292 equation layouts、7,596 image layouts、7,306 table layouts。按 smoke batch=1 steady-state 粗估：FP32 约 85 小时、INT8 约 11 小时、INT4 约 10 小时，三精度串行约 4.4 天。建议先对 5-10 documents 跑 batch `1/2/4/8`，选吞吐与内存平衡点后执行 `./run_campaign.sh`。

## 推荐初步对比

已新增 `subsets/balanced_200.json` 作为全量前的推荐筛选集：10 个 domain 各 20 queries，共 34 documents、607 pages、7,363 text/equation layouts、593 image/table fused layouts。旧统一 batch=1 估算为 FP32 约 3.11 小时、INT8 约 0.43 小时、INT4 约 0.40 小时；recommended profile 会加速 query 和量化 text layout，实际总时间应略低，仍建议预留约 4 小时。

`balanced_100` 约 2.6 小时，但最坏比例指标标准误约 5.0pp；`balanced_200` 约 3.5pp，增加约 1.4 小时即可获得更稳定的初步信号，因此周末脚本默认使用 200-query 版本。两个子集都按 domain 等额采样，避免 annotation 前 N 项以及 137-query News 文档带来的偏置。

自动执行入口为 `./run_weekend_suite.sh`：三精度分别运行并保存日志，最后复用 cache 生成 combined FP32 delta。完整命令和 systemd 周末调度方式见 `README.md`。

## 可比性说明

- A100 表格的 layout page 输入说明与本评测一致：page image；table/image fused；text/formula text。
- 上游 MMDocIR 只实现 page nDCG 和 layout area Recall。本报告的 layout nDCG 使用 graded bbox intersection 定义，需与 A100 结果确认口径后才能直接比较绝对值。
- batch sweep 的吞吐只比较非 cache-hit run；质量比较始终使用相同 query/data selection。
- close-domain 应先通过完整 313-document 结果，再运行 open-domain。子集 smoke 的数值不能用于模型质量结论。
- 数据已固定至 Hugging Face revision `bdcb36ecb3eee73667180ee3fb24fe433f6dd2a4`，三文件字节数与远端 metadata 完全一致。
- backend 确认为本地优化 build：`OpenVINO 2026.3.0-22451-8a17657b995-qwen-ttft-optimizations`。