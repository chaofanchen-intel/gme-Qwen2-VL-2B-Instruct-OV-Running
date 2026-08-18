# GME 文档 Embedding 评测：GPU INT8 / INT4

本目录专门评测 GME-Qwen2-VL-2B-Instruct 对 PDF 文档做 embedding 的性能和表征质量。

本评测不实现 RAG，也不测 CPU、FP32 或其他精度。只测：

- device: `GPU`
- precision: `int8`, `int4`
- document: `../060_OT_Downlook_PR_Error.pdf`

## 文档特征

- 14 页，PowerPoint 横向页面，720 x 540 pt
- 有原生 PDF 文本层，约 5 KB 可提取文本
- 页面包含工程示意图、bump/halo effect 图、threshold 标注和 Vision Browser 操作截图

## 被比较的表示方案

| 表示 | 输入 | 目的 |
|---|---|---|
| `page_text` | 每页原生 PDF 文本 | 页面文本 embedding 基线 |
| `text_chunk_256` | 每页文本按 256 token、64 token overlap 切块 | 文本长度和细粒度切分影响 |
| `page_image_150` | 每页渲染为 150 DPI PNG | 图像页面 embedding |
| `page_fused_150` | 每页文本 + 150 DPI 页面图像 | 图文联合 embedding |

文本使用 GME 的 document/corpus 模式（`is_query=False`）；质量测试的 query 使用 query 模式（`is_query=True`）。

## 性能矩阵

每个 precision 加载一次模型，然后测所有表示：

- text page/chunk: batch `1, 4, 8, 14`
- image/fused page: batch `1, 4, 8, 14`
- 每个 batch 重复 3 次，第一次作为 warm-up，报告 warm p50/mean
- 记录：PDF open/text extraction/render、模型 load、embedding wall time、items/s、tokens/s、peak RSS、向量维度

性能分为两种口径：

- `embedding_wall_ms`: 从调用 GME embedding API 到返回的端到端时间，包含 tokenizer/image preprocessing 和 OpenVINO inference
- `pure_model_load_ms`: GME wrapper 构造和 compile 时间

## 质量矩阵

用 8 个工程 query，人工标注相关页面：

- halo effect / edge bump
- Downlook PR error cause
- flux coverage
- PR learning substrate
- threshold and area
- lighting adjustment
- Vision inspection log
- Vision Dump / system files

对 `page_text`、`text_chunk_256`、`page_image_150`、`page_fused_150` 计算：

- Recall@1/3/5
- MRR
- query 到同一表示内部的同义改写 cosine
- text/image/fused 的跨表示 cosine（仅作为观察指标，不假设不同模态必然同空间等价）

## 运行

```bash
source /home/chaofan/envs/qwen_env/bin/activate
source /home/chaofan/model_benchmark/benchmark/openvino/install/setupvars.sh
cd /home/chaofan/model_benchmark/gme-Qwen2-VL-2B-Instruct-Running/document_embedding_eval_int8_int4_gpu
python -u run_eval.py
```

结果写入 `results/`：

- `benchmark.json`: 全部性能和质量结果
- `document_embedding_benchmark_before_optimized.json`: 优化前基线结果，用于 A/B 对比
- `pages/`: 150 DPI 中间页面图像
- `REPORT.md`: 结果报告

本次实测结论见 [REPORT.md](REPORT.md)。

## 扩展评测：per-OC INT4（`int4_per_oc` / `int4_per_oc_sym`）

`run_eval.py` 支持用 `PRECISIONS` 环境变量（逗号分隔的 package precision key）
和 `OUTPUT` 覆盖默认的 `int8,int4` 范围和 `results/benchmark.json` 输出路径，
默认行为不变：

```bash
PRECISIONS=int4_per_oc,int4_per_oc_sym python -u run_eval.py
# -> results/benchmark_int4_per_oc_int4_per_oc_sym.json（不覆盖 benchmark.json）
```

这是为对比
[`../gme-qwen2-vl-int4-hotspot-optimization/docs/08_per_oc_quantization.md`](../gme-qwen2-vl-int4-hotspot-optimization/docs/08_per_oc_quantization.md)
里新增的两个语言塔量化方案而做的扩展，结果和结论见
[REPORT_per_oc.md](REPORT_per_oc.md)，不改变本目录默认评测范围（GPU int8/int4）。

## 注意事项

- 文本 token 数使用 GME package tokenizer，并记录每个页面/chunk 的 token 统计。
- PDF 原生文本优先；本评测不默认 OCR，因为该 PDF 有可用文本层。OCR 只应作为单独实验，不能与原生文本结果混合。
- 150 DPI 是页面图像默认点；如需扩展分辨率，应新增结果文件，不覆盖本次结果。
