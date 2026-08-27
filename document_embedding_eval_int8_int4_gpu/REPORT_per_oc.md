# GME 文档 Embedding 评测：per-OC INT4 扩展对比（int8 / int4 / int4_per_oc / int4_per_oc_sym）

补充评测，对应
[`../gme-qwen2-vl-int4-hotspot-optimization/docs/08_per_oc_quantization.md`](../gme-qwen2-vl-int4-hotspot-optimization/docs/08_per_oc_quantization.md)
里新增的两个语言塔量化方案：`int4_per_oc`（group_size=-1，非对称）和
`int4_per_oc_sym`（group_size=-1，对称）。本报告用同一份 PDF 文档 embedding
评测（`run_eval.py`），把这两个新方案和 [REPORT.md](REPORT.md) 里已有的
`int8`/`int4`（gs128）放进同一张表对比。

**不改变当时 [README.md](README.md) 声明的历史评测范围**（GPU int8/int4）；这是一次
独立的扩展运行，结果写在单独的文件里，不覆盖 `results/benchmark.json`。

```bash
source /home/chaofan/envs/qwen_env/bin/activate
source /home/chaofan/model_benchmark/benchmark/openvino/install/setupvars.sh
export PYTHONPATH=/home/chaofan/model_benchmark/gme-Qwen2-VL-2B-Instruct-Running:$PYTHONPATH
cd /home/chaofan/model_benchmark/gme-Qwen2-VL-2B-Instruct-Running/document_embedding_eval_int8_int4_gpu
PRECISIONS=int4_per_oc,int4_per_oc_sym python -u run_eval.py
# -> results/benchmark_int4_per_oc_int4_per_oc_sym.json
```

测试环境、PDF、query 集与 [REPORT.md](REPORT.md) 完全一致（同一次运行环境，
OpenVINO `2026.3.0-22453-32f59cd8191-qwen-ttft-optimizations`），只是新增了两个
precision，可以直接跨表对照。

## 1. 结论摘要

- **文本路径性能进一步提升**：batch 8，`int4_per_oc` page_text 424.3 ms
  （32.99 items/s），`int4_per_oc_sym` 412.5 ms（33.94 items/s）——比已经优化过的
  `int4`（gs128，456.8 ms / 30.65 items/s）再快 **7.6%\~10.7%**，比 `int8`
  （515.3 ms）快 **17.6%\~24.9%**。方向与语言塔 IR 级 benchmark 一致
  （docs/08 §8.4），但幅度小一些——这份 PDF 每页只有 64.5\~149 token，
  远短于 IR benchmark 用的 1024 token 压力测试，GEMM 占比更低，量化方案切换的
  收益被 tokenizer/预处理等固定开销摊薄了。
- **图像/fused 路径几乎不受影响**（65.2\~66.3 s，和 int4/int8 的 65.8\~66.9 s
  在同一量级），符合预期：per-OC 只改了语言塔的量化方案，视觉塔权重和计算
  完全没变。
- **⚠️ 这份 8-query 检索质量 probe 上，per-OC 并没有表现出 docs/08 §8.5(a)
  测到的严重精度下降**：Recall@1/3/5、MRR 与 `int4`/`int8` 同一量级，
  `page_text` MRR 甚至略高（`int4_per_oc` 0.431 vs `int4` 0.410），
  `page_fused_150` 的 Recall@1 从 12.5% 提到 25.0%。**这不代表 per-OC 精度问题
  不存在，而是这两种评测测的是不同的东西**，见下面 §3 的解释——docs/08 的
  cosine-vs-FP32-golden 测量仍然是更严谨的结论来源，本报告只是如实记录这份
  小样本 probe 上没有复现同等幅度的质量下降，不能反过来推翻 docs/08 的否决结论。
- 模型 load 时间 `int4_per_oc`（7.6 s）、`int4_per_oc_sym`（6.8 s）比
  `int4`（9.2 s）、`int8`（10.5 s）更短，但和 REPORT.md §5 的告诫一样，
  单次测量受编译缓存和 GPU 状态影响，不作为稳定结论。

## 2. 性能对比（batch 8，与 REPORT.md §5 同口径）

| 精度 | page_text mean ms | items/s | tokens/s | page_image_150 mean ms | page_fused_150 mean ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| int8 | 515.3 | 27.17 | 2039.6 | 65859.7 | 66930.6 |
| int4（gs128，历史默认） | 456.8 | 30.65 | 2300.8 | 65832.7 | 66899.8 |
| int4_per_oc | **424.3** | **32.99** | **2476.8** | 65298.5 | 66277.2 |
| int4_per_oc_sym | **412.5** | **33.94** | **2547.6** | 65252.5 | 66257.5 |

`text_chunk_256` 与 `page_text` 数字几乎相同（这份 PDF 每页 < 256 token，没有
真正切分，见 REPORT.md §3），从略。

相对 `int4`（gs128）的加速：

| 精度 | page_text speedup | page_image_150 speedup | page_fused_150 speedup |
| --- | ---: | ---: | ---: |
| int4_per_oc | 1.077x | 1.008x | 1.009x |
| int4_per_oc_sym | **1.108x** | 1.009x | 1.010x |

完整 batch 矩阵（1/4/8/14）：

| 精度 | 表示 | batch | mean ms | p50 ms | items/s | peak RSS MB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| int4_per_oc | page_text | 1 | 808.4 | 810.9 | 17.32 | 1197 |
| int4_per_oc | page_text | 4 | 412.9 | 412.3 | 33.90 | 1202 |
| int4_per_oc | page_text | 8 | 424.3 | 424.2 | 32.99 | 1203 |
| int4_per_oc | page_text | 14 | 496.7 | 496.1 | 28.19 | 1204 |
| int4_per_oc_sym | page_text | 1 | 784.8 | 779.5 | 17.84 | 1623 |
| int4_per_oc_sym | page_text | 4 | 431.0 | 432.9 | 32.48 | 1626 |
| int4_per_oc_sym | page_text | 8 | 412.5 | 412.4 | 33.94 | 1627 |
| int4_per_oc_sym | page_text | 14 | 491.4 | 491.5 | 28.49 | 1627 |
| int4_per_oc | page_image_150 | 1/4/8/14 | 66214 / 65265 / 65299 / 65274 | -- | 0.211\~0.215 | 1365\~1894 |
| int4_per_oc_sym | page_image_150 | 1/4/8/14 | 65918 / 65279 / 65253 / 65264 | -- | 0.212\~0.215 | 1623\~1973 |
| int4_per_oc | page_fused_150 | 1/4/8/14 | 66927 / 66153 / 66277 / 66684 | -- | 0.209\~0.212 | 1711\~1901 |
| int4_per_oc_sym | page_fused_150 | 1/4/8/14 | 66780 / 66135 / 66258 / 66713 | -- | 0.209\~0.212 | 1790\~1837 |

峰值 RSS 与 `int4`（文本阶段 ~1.83 GB、图像阶段 ~2.19 GB，见 REPORT.md §5）大致
持平或略低，符合 docs/08 §8.2 的"per-OC 权重文件比 gs128 更小"的结论
（scale/zp 元数据更少）。

## 3. 检索质量对比：为什么这里没有复现 docs/08 的精度下降

| 精度 | 表示 | Recall@1 | Recall@3 | Recall@5 | MRR |
| --- | --- | ---: | ---: | ---: | ---: |
| int8 | page_text | 12.5% | 50.0% | 75.0% | 0.402 |
| int4 | page_text | 12.5% | 62.5% | 75.0% | 0.410 |
| int4_per_oc | page_text | 12.5% | 62.5% | 75.0% | **0.431** |
| int4_per_oc_sym | page_text | 12.5% | 62.5% | 75.0% | 0.406 |
| int8 | page_image_150 | 25.0% | 75.0% | 87.5% | 0.522 |
| int4 | page_image_150 | 25.0% | 75.0% | 87.5% | 0.521 |
| int4_per_oc | page_image_150 | 25.0% | 75.0% | 87.5% | 0.513 |
| int4_per_oc_sym | page_image_150 | 25.0% | 75.0% | 87.5% | 0.520 |
| int8 | page_fused_150 | 12.5% | 75.0% | 87.5% | 0.457 |
| int4 | page_fused_150 | 12.5% | 75.0% | 75.0% | 0.446 |
| int4_per_oc | page_fused_150 | **25.0%** | 75.0% | 87.5% | **0.513** |
| int4_per_oc_sym | page_fused_150 | **25.0%** | 75.0% | 75.0% | 0.510 |

`text_chunk_256` 与 `page_text` 结果相同（未真正切分），从略。

这份表**看起来**和 docs/08 §8.5(a)"text/query cosine 从 0.945/0.955 掉到
0.83\~0.88"的结论矛盾，但两者测的是不同的量，并不矛盾：

- **docs/08 §8.5(a)（`04_validate_python.py`）** 测的是"同一个输入，量化后的
  向量相对 **PyTorch FP32 参考向量**的绝对 cosine"——衡量的是量化引入的绝对
  误差有多大，样本是覆盖 text/query/image/fused 多种长度和内容的通用校验集。
- **本报告的 Recall/MRR** 测的是"query 向量和一组 document 向量，在**同一个
  量化方案内部**的相对排序是否正确"——只要量化噪声在这批向量上大致同分布
  （不是针对某几个特定 page/query 的系统性偏移），排序结果可以在绝对误差很大
  的情况下依然大致不变。这正是本报告观察到的情况。
- 本报告的检索集**只有 8 条 query、14 个候选页面**（REPORT.md §8 已经声明
  "适合方案筛选，不足以代表通用检索质量"），统计功效很低，测不出
  docs/08 那种量级的绝对误差是否会在更大规模检索里体现为排序质量下降。

**结论：不能用这份小样本检索 probe 反驳 docs/08 的否决结论。** docs/08 的
cosine-vs-FP32-golden 测量样本更系统、衡量的是量化本身引入的绝对误差，仍然是
判断"是否采纳 per-OC 量化"的权威依据；本报告只是如实记录一个有意思的现象——
**在自洽的（query 和 document 用同一套量化权重编码）小规模检索场景里，
这个量级的绝对精度损失不一定能被 Recall/MRR 这种相对排序指标捕捉到**。
如果生产场景涉及跨精度比较向量（例如迁移量化方案后旧向量库不重新编码）、
或者检索库规模远大于 14 个候选、query 分布更广，绝对误差引入的排序错误
更有可能显现出来，这一点本报告的样本量无法验证。

## 4. 建议

- **维持 docs/08 的结论**：不采纳 `int4_per_oc` / `int4_per_oc_sym` 作为语言塔
  默认量化方案，性能收益（本报告 batch 8 上 7.6%\~10.8%，语言塔 IR 级 benchmark
  上最高到 ~28%）不足以覆盖 §8.5(a) 测到的绝对精度损失风险。
- 如果确实想在生产检索场景里验证"绝对精度损失是否影响真实排序质量"，需要一个
  比这份 8-query/14-page probe 大得多的检索评测集（更多 query、更多候选文档、
  覆盖更广的内容分布），而不是依赖这份方案筛选级别的 probe 得出结论。
- 完整原始数据：
  [`results/benchmark_int4_per_oc_int4_per_oc_sym.json`](results/benchmark_int4_per_oc_int4_per_oc_sym.json)，
  与 [`results/benchmark.json`](results/benchmark.json)（int8/int4）用相同脚本、
  相同 PDF、相同 query 集产出，可直接逐字段对照。
