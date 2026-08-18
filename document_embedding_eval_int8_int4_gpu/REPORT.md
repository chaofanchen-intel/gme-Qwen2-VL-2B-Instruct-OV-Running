# GME 文档 Embedding 评测报告：GPU INT8 / INT4

## 1. 结论摘要

本次只评测 GME-Qwen2-VL-2B-Instruct 的 GPU `int8` 和 `int4`，没有加入 CPU、FP32 或其他精度。对象是 14 页工程 PDF `060_OT_Downlook_PR_Error.pdf`，比较四种文档表示：每页文本、256-token 文本 chunk、每页图像、每页图文 fused。

本次结果使用本地优化后的 OpenVINO GPU plugin：`2026.3.0-22453-32f59cd8191-qwen-ttft-optimizations`。优化前结果保存在 `results/document_embedding_benchmark_before_optimized.json`，本报告中的性能结论以相同脚本口径对比该基线与当前 `results/benchmark.json`。

主要结论：

- **文本 embedding 是优化收益最大的路径**：batch 8 下，INT8 page text p50 为 514.9 ms、27.17 items/s，INT4 为 456.5 ms、30.65 items/s；相比基线分别提升 **1.104x** 和 **1.216x**。
- **页面图像/fused embedding 仍是主要成本**：优化后 14 页 150 DPI 的 page image 约 65.8 s，page fused 约 66.9 s，约比文本路径慢两个数量级；视觉路径提升约 3--6%。
- **优化后 INT4 文本吞吐明显优于 INT8**：batch 8 的 INT4 page text 比 INT8 快约 11.4%；单次模型加载分别为 INT4 9.2 s、INT8 10.5 s，但加载时间受编译缓存和运行状态影响，不作为稳定性能结论。
- **峰值 RSS 仍由视觉和 INT4 路径主导**：优化后 INT8 约 1.53 GB（文本）/ 2.03 GB（图像），INT4 约 1.83 GB（文本）/ 2.19 GB（图像）。
- **当前小型人工质量集上，页面图像 embedding 的页面定位优于文本和 fused**：image 的 Recall@5 为 87.5%，text 为 75.0%，fused 为 87.5%。这说明该 PDF 中的示意图和界面截图携带了重要信息。
- **优化没有改变聚合质量指标**：INT8/INT4 的 Recall@1/3/5 和 MRR 与基线一致；INT4 视觉表示只有少量 cosine 小数值和非关键页面排序变化，相关页面排名没有变化。
- **fused 没有在当前 8 条 query 上稳定超过 image-only**，batch 8 下只比 image-only 慢约 1.6%；不能仅凭“图文一起输入”假设质量一定更好。
- 这份 PDF 的每页原生文本最多只有 149 tokens，因此 `text_chunk_256` 实际没有发生页面内二次切分，和 `page_text` 的 item 数及质量结果相同。要研究 chunk size，需要另选更长文档或把多页合并后再切 chunk。

## 2. 测试环境

- Model package：`ov_models/gme-qwen2-vl-2b-instruct-openvino`
- Device：`GPU`
- Precision：`int8`、`int4`
- OpenVINO runtime：本地 `benchmark/openvino/install`，版本 `2026.3.0-22453-32f59cd8191-qwen-ttft-optimizations`
- Embedding dimension：1536，输出 L2 normalized
- PDF：14 pages，375,895 bytes，PowerPoint 横向页面，720 x 540 pt
- Page rendering：150 DPI，1500 x 1125 pixels，14 张 PNG 总计约 7.52 MB
- Batch sizes：1、4、8、14
- 每个 batch：1 次 warm-up + 3 次 measured calls
- 计时：GME Python API 调用的端到端 wall time，包含 tokenizer/image preprocessing、OpenVINO inference 和 embedding 返回；不包含一次性模型 load
- 优化后结果原文件：[results/benchmark.json](results/benchmark.json)
- 优化前基线：[results/document_embedding_benchmark_before_optimized.json](results/document_embedding_benchmark_before_optimized.json)
- 评测脚本：[run_eval.py](run_eval.py)

模型加载时间单独统计，不混入每次 embedding latency。

## 3. 文档解析统计

PDF 有原生文本层，本次使用 PyMuPDF 提取文本，没有使用 OCR。OCR 不作为默认方案，因为原生文本可用且工程术语容易被 OCR 误识别。

| 指标 | 结果 |
|---|---:|
| PDF pages | 14 |
| PDF bytes | 375,895 |
| extracted text chars | 5,213 |
| PDF open | 22.1 ms |
| text extraction mean/page | 4.28 ms |
| page render mean/page @150 DPI | 112.6 ms |
| page render total | 1.58 s |
| rendered image size | 1500 x 1125 |

### 文本 token 分布

Token 数由 GME package 自带 tokenizer 统计，表示原始页面文本进入 GME prompt 前的文本 token 数。

| 表示 | items | min | p50 | mean | max |
|---|---:|---:|---:|---:|---:|
| page_text | 14 | 10 | 64.5 | 75.07 | 149 |
| text_chunk_256 | 14 | 10 | 64.5 | 75.07 | 149 |

因为本文件每页最多 149 tokens，256-token chunk 没有拆分页面。它验证的是“chunk 管线”而不是长文本 chunk scaling。

## 4. 性能结果

下面的每行是 `batch_size` 对应的 API batch 参数。`mean/p50` 是 3 次 measured calls 的毫秒数；items/s 按全部 14 个 item 除以本次调用 wall time 计算。图像和 fused 的 item 是页面，文本 chunk 的 item 是 chunk。

本节为优化后 patched runtime 的完整矩阵；与优化前的逐项 speedup 以 batch 8 的汇总表为主，完整基线数据保存在 `results/document_embedding_benchmark_before_optimized.json`。

### 4.1 INT8：完整 batch 矩阵

| 表示 | batch | mean ms | p50 ms | items/s | tokens/s | peak RSS MB |
|---|---:|---:|---:|---:|---:|---:|
| page_text | 1 | 923.1 | 923.2 | 15.17 | 1138.6 | 1524 |
| page_text | 4 | 526.8 | 527.5 | 26.58 | 1995.0 | 1529 |
| page_text | 8 | 515.3 | 514.9 | 27.17 | 2039.6 | 1530 |
| page_text | 14 | 579.9 | 580.0 | 24.14 | 1812.3 | 1531 |
| text_chunk_256 | 1 | 789.6 | 789.9 | 17.73 | 1331.1 | 1531 |
| text_chunk_256 | 4 | 525.7 | 525.8 | 26.63 | 1999.4 | 1532 |
| text_chunk_256 | 8 | 515.1 | 515.2 | 27.18 | 2040.4 | 1532 |
| text_chunk_256 | 14 | 581.4 | 581.2 | 24.08 | 1807.8 | 1532 |
| page_image_150 | 1 | 66764.0 | 66735.3 | 0.210 | -- | 1630 |
| page_image_150 | 4 | 65946.3 | 65921.7 | 0.212 | -- | 1715 |
| page_image_150 | 8 | 65859.7 | 65842.1 | 0.213 | -- | 1794 |
| page_image_150 | 14 | 65983.6 | 65982.8 | 0.212 | -- | 2026 |
| page_fused_150 | 1 | 68172.6 | 67659.5 | 0.205 | -- | 1823 |
| page_fused_150 | 4 | 67086.6 | 67095.4 | 0.209 | -- | 1940 |
| page_fused_150 | 8 | 66930.6 | 66901.2 | 0.209 | -- | 1911 |
| page_fused_150 | 14 | 67379.1 | 67396.7 | 0.208 | -- | 2009 |

### 4.2 INT4：完整 batch 矩阵

| 表示 | batch | mean ms | p50 ms | items/s | tokens/s | peak RSS MB |
|---|---:|---:|---:|---:|---:|---:|
| page_text | 1 | 624.8 | 624.1 | 22.41 | 1682.1 | 1824 |
| page_text | 4 | 472.5 | 470.6 | 29.63 | 2224.2 | 1830 |
| page_text | 8 | 456.8 | 456.5 | 30.65 | 2300.8 | 1831 |
| page_text | 14 | 541.9 | 541.7 | 25.84 | 1939.6 | 1831 |
| text_chunk_256 | 1 | 625.5 | 624.9 | 22.38 | 1680.1 | 1831 |
| text_chunk_256 | 4 | 468.0 | 468.0 | 29.92 | 2245.9 | 1831 |
| text_chunk_256 | 8 | 455.9 | 456.0 | 30.71 | 2305.3 | 1832 |
| text_chunk_256 | 14 | 541.1 | 541.4 | 25.87 | 1942.3 | 1832 |
| page_image_150 | 1 | 66505.4 | 66246.5 | 0.211 | -- | 1977 |
| page_image_150 | 4 | 65798.2 | 65802.5 | 0.213 | -- | 2014 |
| page_image_150 | 8 | 65832.7 | 65828.3 | 0.213 | -- | 2081 |
| page_image_150 | 14 | 65840.1 | 65849.6 | 0.213 | -- | 2186 |
| page_fused_150 | 1 | 67280.0 | 67279.8 | 0.208 | -- | 1998 |
| page_fused_150 | 4 | 66596.3 | 66591.3 | 0.210 | -- | 2011 |
| page_fused_150 | 8 | 66899.8 | 66896.1 | 0.209 | -- | 2080 |
| page_fused_150 | 14 | 67535.0 | 67473.9 | 0.207 | -- | 2180 |

### 4.3 优化前基线：完整 batch 矩阵

以下数据来自优化前的 `benchmark.json` 备份，运行在官方 OpenVINO 2026.3 GPU wheel 上。保留完整矩阵是为了和上面的 patched 结果按相同的 batch、表示和统计口径直接对照。

#### INT8 基线

| 表示 | batch | mean ms | p50 ms | items/s | tokens/s | peak RSS MB |
|---|---:|---:|---:|---:|---:|---:|
| page_text | 1 | 1097.6 | 1054.6 | 12.76 | 957.6 | 1176 |
| page_text | 4 | 573.8 | 574.5 | 24.40 | 1831.5 | 1187 |
| page_text | 8 | 568.1 | 568.3 | 24.64 | 1850.0 | 1188 |
| page_text | 14 | 643.6 | 643.3 | 21.75 | 1633.0 | 1189 |
| text_chunk_256 | 1 | 840.6 | 839.5 | 16.66 | 1250.4 | 1189 |
| text_chunk_256 | 4 | 578.4 | 577.9 | 24.20 | 1817.0 | 1189 |
| text_chunk_256 | 8 | 568.5 | 568.5 | 24.63 | 1848.9 | 1190 |
| text_chunk_256 | 14 | 640.8 | 640.9 | 21.85 | 1640.0 | 1190 |
| page_image_150 | 1 | 69518.9 | 69509.2 | 0.201 | -- | 1311 |
| page_image_150 | 4 | 68615.9 | 68557.3 | 0.204 | -- | 1544 |
| page_image_150 | 8 | 68482.1 | 68501.9 | 0.204 | -- | 1623 |
| page_image_150 | 14 | 68685.3 | 68675.3 | 0.204 | -- | 1855 |
| page_fused_150 | 1 | 70964.4 | 70970.9 | 0.197 | -- | 1655 |
| page_fused_150 | 4 | 71006.9 | 71023.0 | 0.197 | -- | 1772 |
| page_fused_150 | 8 | 71149.1 | 71150.0 | 0.197 | -- | 1744 |
| page_fused_150 | 14 | 71583.5 | 71587.8 | 0.196 | -- | 1860 |

#### INT4 基线

| 表示 | batch | mean ms | p50 ms | items/s | tokens/s | peak RSS MB |
|---|---:|---:|---:|---:|---:|---:|
| page_text | 1 | 718.2 | 710.9 | 19.49 | 1463.4 | 2290 |
| page_text | 4 | 555.9 | 556.2 | 25.18 | 1890.5 | 2291 |
| page_text | 8 | 555.5 | 554.9 | 25.20 | 1892.1 | 2292 |
| page_text | 14 | 648.0 | 647.9 | 21.61 | 1621.9 | 2293 |
| text_chunk_256 | 1 | 719.3 | 716.0 | 19.46 | 1461.2 | 2293 |
| text_chunk_256 | 4 | 552.0 | 552.1 | 25.36 | 1904.0 | 2293 |
| text_chunk_256 | 8 | 554.4 | 554.5 | 25.25 | 1895.7 | 2293 |
| text_chunk_256 | 14 | 649.1 | 649.0 | 21.57 | 1619.1 | 2293 |
| page_image_150 | 1 | 68472.5 | 68472.6 | 0.204 | -- | 2402 |
| page_image_150 | 4 | 67838.3 | 67822.8 | 0.206 | -- | 2421 |
| page_image_150 | 8 | 67823.8 | 67828.1 | 0.206 | -- | 2492 |
| page_image_150 | 14 | 67806.1 | 67809.7 | 0.206 | -- | 2618 |
| page_fused_150 | 1 | 69410.2 | 69379.6 | 0.202 | -- | 2488 |
| page_fused_150 | 4 | 68805.3 | 68794.8 | 0.203 | -- | 2437 |
| page_fused_150 | 8 | 68941.0 | 68953.0 | 0.203 | -- | 2495 |
| page_fused_150 | 14 | 69518.7 | 69497.9 | 0.201 | -- | 2619 |

### 4.4 优化前后 p50 对比

`speedup` 定义为 `优化前 p50 / 优化后 p50`；大于 1 表示优化后更快。

| Precision | 表示 | batch | 优化前 p50 ms | 优化后 p50 ms | speedup | 延迟变化 |
|---|---|---:|---:|---:|---:|---:|
| INT8 | page_text | 1 | 1054.6 | 923.2 | 1.142x | -12.5% |
| INT8 | page_text | 4 | 574.5 | 527.5 | 1.089x | -8.2% |
| INT8 | page_text | 8 | 568.3 | 514.9 | 1.104x | -9.4% |
| INT8 | page_text | 14 | 643.3 | 580.0 | 1.109x | -9.8% |
| INT8 | text_chunk_256 | 1 | 839.5 | 789.9 | 1.063x | -5.9% |
| INT8 | text_chunk_256 | 4 | 577.9 | 525.8 | 1.099x | -9.0% |
| INT8 | text_chunk_256 | 8 | 568.5 | 515.2 | 1.103x | -9.4% |
| INT8 | text_chunk_256 | 14 | 640.9 | 581.2 | 1.103x | -9.3% |
| INT8 | page_image_150 | 1 | 69509.2 | 66735.3 | 1.042x | -4.0% |
| INT8 | page_image_150 | 4 | 68557.3 | 65921.7 | 1.040x | -3.8% |
| INT8 | page_image_150 | 8 | 68501.9 | 65842.1 | 1.040x | -3.9% |
| INT8 | page_image_150 | 14 | 68675.3 | 65982.8 | 1.041x | -3.9% |
| INT8 | page_fused_150 | 1 | 70970.9 | 67659.5 | 1.049x | -4.7% |
| INT8 | page_fused_150 | 4 | 71023.0 | 67095.4 | 1.059x | -5.5% |
| INT8 | page_fused_150 | 8 | 71150.0 | 66901.2 | 1.064x | -6.0% |
| INT8 | page_fused_150 | 14 | 71587.8 | 67396.7 | 1.062x | -5.9% |
| INT4 | page_text | 1 | 710.9 | 624.1 | 1.139x | -12.2% |
| INT4 | page_text | 4 | 556.2 | 470.6 | 1.182x | -15.4% |
| INT4 | page_text | 8 | 554.9 | 456.5 | 1.216x | -17.7% |
| INT4 | page_text | 14 | 647.9 | 541.7 | 1.196x | -16.4% |
| INT4 | text_chunk_256 | 1 | 716.0 | 624.9 | 1.146x | -12.7% |
| INT4 | text_chunk_256 | 4 | 552.1 | 468.0 | 1.180x | -15.2% |
| INT4 | text_chunk_256 | 8 | 554.5 | 456.0 | 1.216x | -17.8% |
| INT4 | text_chunk_256 | 14 | 649.0 | 541.4 | 1.199x | -16.6% |
| INT4 | page_image_150 | 1 | 68472.6 | 66246.5 | 1.034x | -3.3% |
| INT4 | page_image_150 | 4 | 67822.8 | 65802.5 | 1.031x | -3.0% |
| INT4 | page_image_150 | 8 | 67828.1 | 65828.3 | 1.030x | -2.9% |
| INT4 | page_image_150 | 14 | 67809.7 | 65849.6 | 1.030x | -2.9% |
| INT4 | page_fused_150 | 1 | 69379.6 | 67279.8 | 1.031x | -3.0% |
| INT4 | page_fused_150 | 4 | 68794.8 | 66591.3 | 1.033x | -3.2% |
| INT4 | page_fused_150 | 8 | 68953.0 | 66896.1 | 1.031x | -3.0% |
| INT4 | page_fused_150 | 14 | 69497.9 | 67473.9 | 1.030x | -2.9% |

## 5. INT8 vs INT4 对比

以 batch 8 为主比较；下表使用 mean，完整 p50 数据见上面的矩阵：

| 表示 | INT8 mean | INT4 mean | INT4 相对 INT8 | INT8 items/s | INT4 items/s |
|---|---:|---:|---:|---:|---:|
| page_text | 515.3 ms | 456.8 ms | **-11.4%** | 27.17 | 30.65 |
| text_chunk_256 | 515.1 ms | 455.9 ms | **-11.5%** | 27.18 | 30.71 |
| page_image_150 | 65859.7 ms | 65832.7 ms | **-0.0%** | 0.213 | 0.213 |
| page_fused_150 | 66930.6 ms | 66899.8 ms | **-0.0%** | 0.209 | 0.209 |

模型加载和内存：

| Precision | 基线 load ms | 优化后 load ms | 基线 peak RSS，text阶段 | 优化后 peak RSS，text阶段 | 基线 peak RSS，image/fused阶段 | 优化后 peak RSS，image/fused阶段 |
|---|---:|---:|---:|---:|---:|---:|
| INT8 | 8,636 | 10,492 | 1,190 MB | 1,532 MB | 1,861 MB | 2,026 MB |
| INT4 | 12,470 | 9,197 | 2,293 MB | 1,832 MB | 2,619 MB | 2,186 MB |

本次优化运行中 INT4 的 steady-state 文本 embedding 比 INT8 快约 11.4%，但 INT4 峰值 RSS 仍高于 INT8。基线与优化后的 load/RSS 受编译缓存、权重重排和 GPU 状态影响，不能据单次测量判断稳定的启动差异；生产环境若复用同一个 encoder，load 只发生一次，若频繁创建短生命周期进程则应单独评估启动成本。

## 6. Embedding 表征质量

质量测试不是 RAG 评测，而是用人工工程 query 检查 embedding 是否把 query 对齐到相关页面。共 8 条 query，相关页由人工标注。由于集合很小，结果用于方案比较和冒烟验证，不应视为统计意义上的通用 benchmark。

### 6.1 Query 到页面/Chunk 的定位

| Precision | 表示 | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---|---:|---:|---:|---:|
| INT8 | page_text | 12.5% | 50.0% | 75.0% | 0.402 |
| INT8 | text_chunk_256 | 12.5% | 50.0% | 75.0% | 0.402 |
| INT8 | page_image_150 | **25.0%** | **75.0%** | **87.5%** | **0.522** |
| INT8 | page_fused_150 | 12.5% | **75.0%** | **87.5%** | 0.457 |
| INT4 | page_text | 12.5% | **62.5%** | 75.0% | **0.410** |
| INT4 | text_chunk_256 | 12.5% | **62.5%** | 75.0% | **0.410** |
| INT4 | page_image_150 | **25.0%** | **75.0%** | **87.5%** | 0.521 |
| INT4 | page_fused_150 | 12.5% | **75.0%** | 75.0% | 0.446 |

与 `results/document_embedding_benchmark_before_optimized.json` 对比，以上 Recall/MRR 聚合指标全部保持不变。INT8 输出的质量数值逐项一致；INT4 的 image/fused 只有少量 cosine 小数值和非相关页面排序变化，8 条 query 的相关页面排名没有变化。

### 6.2 对应页面跨表示 cosine

这些分数只用于观察同一页的不同表示是否相近；不能默认 text/image/fused 向量空间完全等价。

| Precision | text vs image mean/min/max | text vs fused mean/min/max | image vs fused mean/min/max |
|---|---|---|---|
| INT8 | 0.701 / 0.476 / 0.778 | 0.835 / 0.550 / 0.898 | 0.910 / 0.867 / 0.941 |
| INT4 | 0.725 / 0.522 / 0.780 | 0.838 / 0.585 / 0.890 | 0.924 / 0.901 / 0.946 |

## 7. 方案判断

### 推荐用于纯文本工程文档

`page_text` 或更长文档的 `text_chunk_256`：

- 优化后 batch 8 速度约 0.46--0.52 s/14 pages，INT4 约 30.65 items/s，INT8 约 27.17 items/s
- INT4 比 INT8 快约 11.4%，是本轮优化最主要的收益
- 内存低于图像/fused 路径，但 INT4 仍高于 INT8
- 结果可追溯到页码和原始文本

当前 PDF 每页小于 256 tokens，所以 page_text 和 chunk 结果相同。更长文档应重新测试 128/256/384/512 token chunk，不能从本次 PDF 推断 chunk size 的普遍结论。

### 推荐用于包含工程图和 UI 截图的文档

`page_image_150`：

- 当前质量 probe 上 Recall@5 最高或并列最高
- 能保留 halo、bump、threshold、Vision Browser 等文本层之外的视觉信息
- 优化后代价约 65.8 s/14 pages，远高于文本路径；本轮只获得约 3% 加速

### fused 的使用建议

`page_fused_150` 暂不作为默认方案：

- batch 8 下比 image-only 慢约 1.6%
- 当前 query 集上没有稳定质量增益
- 如果后续文档的文字和图像必须联合理解，再单独验证 fused；不要仅因为输入同时有文字和图片就默认 fused 更好

### INT8 / INT4 选择

- 只看 warm embedding throughput：文本路径优先 INT4，batch 8 比 INT8 快约 11.4%；image/fused 两者基本相同。
- 关注峰值内存：INT8 更合适，优化后文本峰值约 1.53 GB，INT4 约 1.83 GB；图像路径分别约 2.03 GB 和 2.19 GB。
- 模型 load 仅测得一次，不能据此判断长期稳定的冷启动差异。
- 对这份 PDF 的 8 条质量 probe，INT4 和 INT8 的 image 结果基本相同；text 的 INT4 Recall@3 略高，但差异不足以形成通用质量结论。

## 8. 评测边界和后续扩展

本次已经覆盖用户当前限定的 GPU INT8/INT4，并完成 text/image/fused 的实际性能和初步质量对比。以下项目没有混入当前结论：

- 本报告的优化组使用本地 OpenVINO patched runtime；基线原始结果保存在 `results/document_embedding_benchmark_before_optimized.json`，两组使用同一份评测脚本和输入。
- OCR：PDF 原生文本可用，未把 OCR 误差混入基线；
- 300 DPI：本次固定 150 DPI，300 DPI 应作为单独图像分辨率实验；
- 长文本 chunk scaling：本 PDF 每页很短，256-token chunk 未拆分；
- 大规模质量集：当前只有 8 条人工工程 query，适合作为方案筛选，不足以代表通用检索质量；
- C++ runtime：本报告评测 Python GME wrapper，C++ 应作为独立 runtime benchmark，不与 Python 数字混合。

## 9. 复现命令

```bash
source /home/chaofan/envs/qwen_env/bin/activate
source /home/chaofan/model_benchmark/benchmark/openvino/install/setupvars.sh
export PYTHONPATH=/home/chaofan/model_benchmark/gme-Qwen2-VL-2B-Instruct-Running:$PYTHONPATH
cd /home/chaofan/model_benchmark/gme-Qwen2-VL-2B-Instruct-Running/document_embedding_eval_int8_int4_gpu
python -u run_eval.py
```
