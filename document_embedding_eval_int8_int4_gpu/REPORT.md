# GME 文档 Embedding 评测报告：GPU INT8 / INT4

## 1. 结论摘要

本次只评测 GME-Qwen2-VL-2B-Instruct 的 GPU `int8` 和 `int4`，没有加入 CPU、FP32 或其他精度。对象是 14 页工程 PDF `060_OT_Downlook_PR_Error.pdf`，比较四种文档表示：每页文本、256-token 文本 chunk、每页图像、每页图文 fused。

主要结论：

- **文本 embedding 是可用的低成本路径**：14 页文本在最佳 batch 下约 0.55--0.57 s，约 24--25 items/s；INT4 比 INT8 略快。
- **页面图像/fused embedding 是主要成本**：14 页 150 DPI 页面约 67.8--71.6 s，约 0.20 items/s，约比文本路径慢两个数量级。
- **INT4 的推理吞吐略优于 INT8，但加载时间和峰值 RSS 更高**：本次 INT4 GPU 模型加载约 12.5 s，INT8 约 8.6 s；文本 batch 8 的 INT4 约 555 ms，INT8 约 568 ms。
- **当前小型人工质量集上，页面图像 embedding 的页面定位优于文本和 fused**：image 的 Recall@5 为 87.5%，text 为 75.0%，fused 为 87.5%。这说明该 PDF 中的示意图和界面截图携带了重要信息。
- **fused 没有在当前 8 条 query 上稳定超过 image-only**，而且比 image-only 更慢约 3--4%；不能仅凭“图文一起输入”假设质量一定更好。
- 这份 PDF 的每页原生文本最多只有 149 tokens，因此 `text_chunk_256` 实际没有发生页面内二次切分，和 `page_text` 的 item 数及质量结果相同。要研究 chunk size，需要另选更长文档或把多页合并后再切 chunk。

## 2. 测试环境

- Model package：`ov_models/gme-qwen2-vl-2b-instruct-openvino`
- Device：`GPU`
- Precision：`int8`、`int4`
- Embedding dimension：1536，输出 L2 normalized
- PDF：14 pages，375,895 bytes，PowerPoint 横向页面，720 x 540 pt
- Page rendering：150 DPI，1500 x 1125 pixels，14 张 PNG 总计约 7.52 MB
- Batch sizes：1、4、8、14
- 每个 batch：1 次 warm-up + 3 次 measured calls
- 计时：GME Python API 调用的端到端 wall time，包含 tokenizer/image preprocessing、OpenVINO inference 和 embedding 返回；不包含一次性模型 load
- 结果原文件：[results/benchmark.json](results/benchmark.json)
- 评测脚本：[run_eval.py](run_eval.py)

模型加载时间单独统计，不混入每次 embedding latency。

## 3. 文档解析统计

PDF 有原生文本层，本次使用 PyMuPDF 提取文本，没有使用 OCR。OCR 不作为默认方案，因为原生文本可用且工程术语容易被 OCR 误识别。

| 指标 | 结果 |
|---|---:|
| PDF pages | 14 |
| PDF bytes | 375,895 |
| extracted text chars | 5,213 |
| PDF open | 13.8 ms |
| text extraction mean/page | 3.85 ms |
| page render mean/page @150 DPI | 112.1 ms |
| page render total | 1.57 s |
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

### 4.1 INT8：完整 batch 矩阵

| 表示 | batch | mean ms | p50 ms | items/s | tokens/s | peak RSS MB |
|---|---:|---:|---:|---:|---:|---:|
| page_text | 1 | 1097.6 | 1054.6 | 12.76 | 957.6 | 1176 |
| page_text | 4 | 573.9 | 574.5 | 24.40 | 1831.5 | 1187 |
| page_text | 8 | 568.1 | 568.3 | 24.64 | 1850.0 | 1188 |
| page_text | 14 | 643.6 | 643.3 | 21.75 | 1633.0 | 1189 |
| text_chunk_256 | 1 | 840.6 | 839.5 | 16.66 | 1250.4 | 1189 |
| text_chunk_256 | 4 | 578.4 | 577.9 | 24.20 | 1817.0 | 1189 |
| text_chunk_256 | 8 | 568.5 | 568.5 | 24.63 | 1848.9 | 1190 |
| text_chunk_256 | 14 | 640.8 | 640.9 | 21.85 | 1640.0 | 1190 |
| page_image_150 | 1 | 69518.9 | 69509.2 | 0.20 | -- | 1311 |
| page_image_150 | 4 | 68615.9 | 68557.3 | 0.20 | -- | 1544 |
| page_image_150 | 8 | 68482.1 | 68501.9 | 0.20 | -- | 1623 |
| page_image_150 | 14 | 68685.3 | 68675.3 | 0.20 | -- | 1855 |
| page_fused_150 | 1 | 70964.4 | 70970.9 | 0.20 | -- | 1655 |
| page_fused_150 | 4 | 71006.9 | 71023.0 | 0.20 | -- | 1772 |
| page_fused_150 | 8 | 71149.1 | 71150.0 | 0.20 | -- | 1744 |
| page_fused_150 | 14 | 71583.5 | 71587.8 | 0.20 | -- | 1861 |

### 4.2 INT4：完整 batch 矩阵

| 表示 | batch | mean ms | p50 ms | items/s | tokens/s | peak RSS MB |
|---|---:|---:|---:|---:|---:|---:|
| page_text | 1 | 718.2 | 710.9 | 19.49 | 1463.4 | 2290 |
| page_text | 4 | 556.0 | 556.2 | 25.18 | 1890.5 | 2291 |
| page_text | 8 | 555.5 | 554.9 | 25.20 | 1892.1 | 2292 |
| page_text | 14 | 648.0 | 647.9 | 21.61 | 1621.9 | 2293 |
| text_chunk_256 | 1 | 719.3 | 716.0 | 19.46 | 1461.2 | 2293 |
| text_chunk_256 | 4 | 552.0 | 552.1 | 25.36 | 1904.0 | 2293 |
| text_chunk_256 | 8 | 554.4 | 554.5 | 25.25 | 1895.7 | 2293 |
| text_chunk_256 | 14 | 649.1 | 649.0 | 21.57 | 1619.1 | 2293 |
| page_image_150 | 1 | 68472.5 | 68472.6 | 0.20 | -- | 2402 |
| page_image_150 | 4 | 67838.3 | 67822.9 | 0.21 | -- | 2421 |
| page_image_150 | 8 | 67823.8 | 67828.1 | 0.21 | -- | 2492 |
| page_image_150 | 14 | 67806.1 | 67809.7 | 0.21 | -- | 2618 |
| page_fused_150 | 1 | 69410.2 | 69379.6 | 0.20 | -- | 2488 |
| page_fused_150 | 4 | 68805.3 | 68794.8 | 0.20 | -- | 2437 |
| page_fused_150 | 8 | 68941.0 | 68953.0 | 0.20 | -- | 2495 |
| page_fused_150 | 14 | 69518.7 | 69497.9 | 0.20 | -- | 2619 |

## 5. INT8 vs INT4 对比

以最稳定的 batch 8 为主比较：

| 表示 | INT8 mean | INT4 mean | INT4 相对 INT8 | INT8 items/s | INT4 items/s |
|---|---:|---:|---:|---:|---:|
| page_text | 568.1 ms | 555.5 ms | **-2.2%** | 24.64 | 25.20 |
| text_chunk_256 | 568.5 ms | 554.4 ms | **-2.5%** | 24.63 | 25.25 |
| page_image_150 | 68482.1 ms | 67823.8 ms | **-1.0%** | 0.204 | 0.206 |
| page_fused_150 | 71149.1 ms | 68941.0 ms | **-3.1%** | 0.197 | 0.203 |

模型加载和内存：

| Precision | load ms | peak RSS，text阶段 | peak RSS，image/fused阶段 |
|---|---:|---:|---:|
| INT8 | 8,636 | 1,190 MB | 1,861 MB |
| INT4 | 12,470 | 2,293 MB | 2,619 MB |

本次测量中 INT4 的 steady-state embedding 略快，但冷加载明显更慢，峰值 RSS 也更高。生产环境若复用同一个 encoder，load 只发生一次；若频繁创建短生命周期进程，load 成本需要单独考虑。

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

### 6.2 对应页面跨表示 cosine

这些分数只用于观察同一页的不同表示是否相近；不能默认 text/image/fused 向量空间完全等价。

| Precision | text vs image mean/min/max | text vs fused mean/min/max | image vs fused mean/min/max |
|---|---|---|---|
| INT8 | 0.701 / 0.476 / 0.778 | 0.835 / 0.550 / 0.898 | 0.910 / 0.867 / 0.941 |
| INT4 | 0.726 / 0.514 / 0.780 | 0.840 / 0.588 / 0.890 | 0.924 / 0.901 / 0.946 |

## 7. 方案判断

### 推荐用于纯文本工程文档

`page_text` 或更长文档的 `text_chunk_256`：

- 速度约 0.55--0.65 s/14 pages
- INT4 略快于 INT8
- 内存低于图像/fused 路径
- 结果可追溯到页码和原始文本

当前 PDF 每页小于 256 tokens，所以 page_text 和 chunk 结果相同。更长文档应重新测试 128/256/384/512 token chunk，不能从本次 PDF 推断 chunk size 的普遍结论。

### 推荐用于包含工程图和 UI 截图的文档

`page_image_150`：

- 当前质量 probe 上 Recall@5 最高或并列最高
- 能保留 halo、bump、threshold、Vision Browser 等文本层之外的视觉信息
- 代价是约 68 s/14 pages，远高于文本路径

### fused 的使用建议

`page_fused_150` 暂不作为默认方案：

- 比 image-only 更慢约 3--4%
- 当前 query 集上没有稳定质量增益
- 如果后续文档的文字和图像必须联合理解，再单独验证 fused；不要仅因为输入同时有文字和图片就默认 fused 更好

### INT8 / INT4 选择

- 只看 warm embedding throughput：优先 INT4，文本约快 2--3%，image/fused 约快 1--3%。
- 关注冷启动时间或峰值内存：INT8 更合适，本次 load 快约 31%，峰值 RSS 低约 1.1--1.5 GB。
- 对这份 PDF 的 8 条质量 probe，INT4 和 INT8 的 image 结果基本相同；text 的 INT4 Recall@3 略高，但差异不足以形成通用质量结论。

## 8. 评测边界和后续扩展

本次已经覆盖用户当前限定的 GPU INT8/INT4，并完成 text/image/fused 的实际性能和初步质量对比。以下项目没有混入当前结论：

- OCR：PDF 原生文本可用，未把 OCR 误差混入基线；
- 300 DPI：本次固定 150 DPI，300 DPI 应作为单独图像分辨率实验；
- 长文本 chunk scaling：本 PDF 每页很短，256-token chunk 未拆分；
- 大规模质量集：当前只有 8 条人工工程 query，适合作为方案筛选，不足以代表通用检索质量；
- C++ runtime：本报告评测 Python GME wrapper，C++ 应作为独立 runtime benchmark，不与 Python 数字混合。

## 9. 复现命令

```bash
source /home/chaofan/envs/qwen_env/bin/activate
export PYTHONPATH=/home/chaofan/model_benchmark/gme-Qwen2-VL-2B-Instruct-Running:$PYTHONPATH
cd /home/chaofan/model_benchmark/gme-Qwen2-VL-2B-Instruct-Running/document_embedding_eval_int8_int4_gpu
python run_eval.py
```
