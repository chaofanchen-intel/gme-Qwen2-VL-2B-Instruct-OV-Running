# GME 文档 Embedding 评测报告：GPU mask-free INT8 / causal INT4

## 当前第四轮结果（2026-08-26）

本节是当前 document embedding 结果的主入口。使用的原始文件为
[`benchmark_gme_round4_vnomask_int8_int4_causal_20260826.json`](results/benchmark_gme_round4_vnomask_int8_int4_causal_20260826.json)。
本次没有重新运行未优化 baseline；对照文件仍是
[`document_embedding_benchmark_before_optimized.json`](results/document_embedding_benchmark_before_optimized.json)。
上一轮 masked visual 的优化结果
[`benchmark_gme_opt_int8_int4_causal_20260825.json`](results/benchmark_gme_opt_int8_int4_causal_20260825.json)
也保留，用于拆分第四轮视觉收益。

### 运行时与模型组合

| 项目 | 当前值 |
| --- | --- |
| OpenVINO source | `benchmark/openvino`，官方 base `8a17657b99` + GME opt1/opt2/opt3 |
| core/plugin/Python binding | `2026.3.0-22451-8a17657b995-qwen-ttft-optimizations`，三者同源 custom build |
| device | GPU |
| package | `ov_models/gme-qwen2-vl-2b-instruct-openvino` |
| document | 14 pages, 375,895 bytes, 5,213 extracted characters |
| page rendering | 150 DPI, 1500 x 1125 pixels |
| batch sizes | 1, 4, 8, 14 |
| measurement | 1 warm-up + 3 measured calls per cell，表格以 mean 为主，同时保留 p50/p95/min/max |
| embedding | 1536 dimensions, float32, L2 normalized |

当前 precision 的实际映射如下：

| precision key | language IR | visual IR | 语义 |
| --- | --- | --- | --- |
| `int8_vnomask` | `gme_language_int8.xml` | `gme_visual_int8_nomask.xml` | 普通 INT8 language + mask-free INT8 visual |
| `int4_causal_vnomask` | `gme_language_int4_causal.xml` | `gme_visual_int4_nomask.xml` | causal INT4 language + mask-free INT4 visual |

`_vnomask` 只改变 visual IR。mask-free visual IR 对当前 runtime 的单图 image/fused
路径去掉恒为全 True 的 rank-3 BOOL attention mask；visual FP32 已逐位等价验证。
package 中没有 `int8_causal`，本次没有为不存在的模型做外推或重新导出。

### 性能摘要（mean）

| 表示 / batch | `int8_vnomask` mean ms | `int4_causal_vnomask` mean ms |
| --- | ---: | ---: |
| `page_text` / 1 | 956.107 | 597.904 |
| `page_text` / 4 | 526.701 | 455.796 |
| `page_text` / 8 | 515.698 | 444.465 |
| `page_text` / 14 | 576.975 | 519.235 |
| `text_chunk_256` / 1 | 793.292 | 609.459 |
| `text_chunk_256` / 4 | 524.235 | 462.605 |
| `text_chunk_256` / 8 | 513.616 | 442.226 |
| `text_chunk_256` / 14 | 578.592 | 518.671 |
| `page_image_150` / 1 | 27,898.644 | 27,212.842 |
| `page_image_150` / 4 | 27,191.059 | 26,771.652 |
| `page_image_150` / 8 | 27,109.498 | 26,738.608 |
| `page_image_150` / 14 | 27,218.030 | 26,774.130 |
| `page_fused_150` / 1 | 28,998.338 | 27,692.052 |
| `page_fused_150` / 4 | 28,181.784 | 27,151.353 |
| `page_fused_150` / 8 | 28,132.805 | 27,209.941 |
| `page_fused_150` / 14 | 28,567.512 | 27,289.400 |

文本在 batch 8 的吞吐为 `27.1477 items/s`（INT8）和 `31.4985 items/s`
（causal INT4）；image 为 `0.5164` 和 `0.5236 items/s`；fused 为
`0.4976` 和 `0.5145 items/s`。图像 batch 的 visual 阶段仍逐图执行，所以
batch 对 image/fused 的收益远小于 text。

### `int8_vnomask` 完整性能矩阵

| 表示 | batch | mean ms | p50 ms | p95 ms | min ms | max ms | items/s | tokens/s | peak RSS MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| page_text | 1 | 956.107 | 945.535 | 991.971 | 925.656 | 997.131 | 1.04643 | 1099.249 | 1181 |
| page_text | 4 | 526.701 | 527.001 | 527.297 | 525.772 | 527.330 | 26.58055 | 1995.440 | 1192 |
| page_text | 8 | 515.698 | 515.953 | 516.154 | 514.965 | 516.176 | 27.14766 | 2038.013 | 1193 |
| page_text | 14 | 576.975 | 577.137 | 578.086 | 575.596 | 578.191 | 24.26450 | 1821.571 | 1194 |
| text_chunk_256 | 1 | 793.292 | 790.806 | 798.632 | 789.568 | 799.502 | 1.25965 | 1324.859 | 1195 |
| text_chunk_256 | 4 | 524.235 | 524.392 | 524.584 | 523.708 | 524.605 | 26.70559 | 2004.827 | 1195 |
| text_chunk_256 | 8 | 513.616 | 513.351 | 514.558 | 512.804 | 514.692 | 27.25774 | 2046.277 | 1195 |
| text_chunk_256 | 14 | 578.592 | 578.534 | 579.127 | 578.050 | 579.193 | 24.19666 | 1816.478 | 1195 |
| page_image_150 | 1 | 27898.644 | 27898.140 | 27908.239 | 27888.431 | 27909.362 | 0.50182 | -- | 1400 |
| page_image_150 | 4 | 27191.059 | 27198.178 | 27239.745 | 27130.635 | 27244.364 | 0.51488 | -- | 1643 |
| page_image_150 | 8 | 27109.498 | 27114.599 | 27153.941 | 27055.582 | 27158.312 | 0.51642 | -- | 1714 |
| page_image_150 | 14 | 27218.030 | 27230.957 | 27234.547 | 27188.188 | 27234.946 | 0.51436 | -- | 1931 |
| page_fused_150 | 1 | 28998.338 | 28998.992 | 29014.893 | 28979.364 | 29016.659 | 0.48279 | -- | 1751 |
| page_fused_150 | 4 | 28181.784 | 28161.459 | 28216.803 | 28160.942 | 28222.952 | 0.49677 | -- | 1810 |
| page_fused_150 | 8 | 28132.805 | 28105.540 | 28205.080 | 28076.735 | 28216.140 | 0.49764 | -- | 1839 |
| page_fused_150 | 14 | 28567.512 | 28574.401 | 28578.707 | 28548.951 | 28579.185 | 0.49007 | -- | 1942 |

### `int4_causal_vnomask` 完整性能矩阵

| 表示 | batch | mean ms | p50 ms | p95 ms | min ms | max ms | items/s | tokens/s | peak RSS MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| page_text | 1 | 597.904 | 596.910 | 600.297 | 596.129 | 600.673 | 1.67249 | 1757.807 | 1762 |
| page_text | 4 | 455.796 | 455.152 | 457.291 | 454.708 | 457.529 | 30.71546 | 2305.854 | 1765 |
| page_text | 8 | 444.465 | 444.651 | 444.782 | 443.949 | 444.796 | 31.49853 | 2364.640 | 1766 |
| page_text | 14 | 519.235 | 519.279 | 519.728 | 518.647 | 519.778 | 26.96275 | 2024.132 | 1766 |
| text_chunk_256 | 1 | 609.459 | 608.018 | 618.693 | 600.480 | 619.879 | 1.64079 | 1724.480 | 1767 |
| text_chunk_256 | 4 | 462.605 | 464.597 | 465.492 | 457.626 | 465.591 | 30.26343 | 2271.919 | 1767 |
| text_chunk_256 | 8 | 442.226 | 442.551 | 442.582 | 441.542 | 442.585 | 31.65802 | 2376.613 | 1767 |
| text_chunk_256 | 14 | 518.671 | 518.694 | 519.252 | 518.006 | 519.314 | 26.99204 | 2026.331 | 1767 |
| page_image_150 | 1 | 27212.842 | 27201.078 | 27244.794 | 27187.799 | 27249.651 | 0.51446 | -- | 1921 |
| page_image_150 | 4 | 26771.652 | 26780.049 | 26780.216 | 26754.674 | 26780.234 | 0.52294 | -- | 1931 |
| page_image_150 | 8 | 26738.608 | 26738.647 | 26748.302 | 26727.802 | 26749.375 | 0.52359 | -- | 2002 |
| page_image_150 | 14 | 26774.130 | 26752.488 | 26814.926 | 26748.037 | 26821.864 | 0.52289 | -- | 2130 |
| page_fused_150 | 1 | 27692.052 | 27685.965 | 27737.248 | 27647.245 | 27742.946 | 0.50556 | -- | 2003 |
| page_fused_150 | 4 | 27151.353 | 27156.038 | 27174.400 | 27121.582 | 27176.440 | 0.51563 | -- | 1948 |
| page_fused_150 | 8 | 27209.941 | 27190.124 | 27248.580 | 27184.624 | 27255.075 | 0.51452 | -- | 2014 |
| page_fused_150 | 14 | 27289.400 | 27266.726 | 27348.547 | 27243.838 | 27357.637 | 0.51302 | -- | 2138 |

### 与已有结果的对照（batch 8 mean）

| 表示 | 未优化普通 INT8 | 当前 `int8_vnomask` | 未优化普通 INT4 | 当前 `int4_causal_vnomask` | 历史 masked 优化 INT8 | 历史 masked 优化 causal INT4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| page_text | 568.104 | 515.698 | 555.459 | 444.465 | 515.255 | 444.425 |
| text_chunk_256 | 568.450 | 513.616 | 554.420 | 442.226 | 521.777 | 443.674 |
| page_image_150 | 68,482.061 | 27,109.498 | 67,823.817 | 26,738.608 | 70,477.171 | 65,637.068 |
| page_fused_150 | 71,149.129 | 28,132.805 | 68,941.041 | 27,209.941 | 71,514.594 | 66,165.529 |

解读边界很重要：

1. 普通 INT8 baseline -> `int8_vnomask` 同时包含 opt1/opt2/opt3 plugin 优化和
    masked -> mask-free visual 导出变化；文本路径的 language IR 相同，视觉路径不能
    把全部差异解释成 plugin 收益。
2. 普通 INT4 baseline -> `int4_causal_vnomask` 同时包含 plugin 优化、普通 -> causal
    language 导出和 masked -> mask-free visual 导出，不能声称是 plugin-only 对比。
3. 历史 masked optimized -> 当前 vnomask 只比较 visual IR（language IR 相同）：
    batch 8 image/fused mean 从 70.477/71.515 s 降到 27.109/28.133 s（INT8），
    causal INT4 从 65.637/66.166 s 降到 26.739/27.210 s，说明第四轮视觉优化是
    document image/fused 端到端主要变化来源。

### 当前质量结果

8 个工程 query、14 个页面的内部检索 probe：

| precision | 表示 | Recall@1 | Recall@3 | Recall@5 | MRR |
| --- | --- | ---: | ---: | ---: | ---: |
| `int8_vnomask` | page_text | 12.5% | 50.0% | 75.0% | 0.4020 |
| `int8_vnomask` | text_chunk_256 | 12.5% | 50.0% | 75.0% | 0.4020 |
| `int8_vnomask` | page_image_150 | 25.0% | 75.0% | 87.5% | 0.5218 |
| `int8_vnomask` | page_fused_150 | 12.5% | 75.0% | 87.5% | 0.4568 |
| `int4_causal_vnomask` | page_text | 12.5% | 50.0% | 75.0% | 0.3996 |
| `int4_causal_vnomask` | text_chunk_256 | 12.5% | 50.0% | 75.0% | 0.3996 |
| `int4_causal_vnomask` | page_image_150 | 25.0% | 75.0% | 87.5% | 0.5193 |
| `int4_causal_vnomask` | page_fused_150 | 12.5% | 75.0% | 75.0% | 0.4464 |

对应页面的 cross-representation cosine：

| precision | text vs image mean/min/max | text vs fused mean/min/max | image vs fused mean/min/max |
| --- | --- | --- | --- |
| `int8_vnomask` | 0.7029 / 0.4825 / 0.7730 | 0.8372 / 0.5607 / 0.8983 | 0.9102 / 0.8723 / 0.9348 |
| `int4_causal_vnomask` | 0.7231 / 0.4968 / 0.7788 | 0.8353 / 0.5759 / 0.8945 | 0.9189 / 0.8805 / 0.9391 |

这些 Recall/MRR 数字只是 8-query/14-page 的工程 probe，不能替代真实业务检索集。
INT8 与历史普通 INT8 的质量基本保持同一量级；causal INT4 的普通 INT4 对比仍
混合了 language causal 与 visual mask-free 两项图变化，因此质量差异不能单独归因
于 plugin。

### Load、token 和内存

| precision | load mean ms | page token mean/min/max | chunk token mean/min/max |
| --- | ---: | --- | --- |
| `int8_vnomask` | 7938.734 | 75.07 / 10 / 149 | 75.07 / 10 / 149 |
| `int4_causal_vnomask` | 7710.813 | 75.07 / 10 / 149 | 75.07 / 10 / 149 |

单次 load 会受 compile cache、GPU 状态和进程生命周期影响；不要把这两个 load 值
当成通用冷启动 SLA。完整的 min/max、RSS 和每个 query 的 top pages 都在原始 JSON。

### 当前复现命令

不要让官方 wheel 的 Python binding 或 package-local official libraries 抢先加载：

```bash
ROOT=/home/chaofan/model_benchmark
OV=$ROOT/benchmark/openvino
GME=$ROOT/gme-Qwen2-VL-2B-Instruct-Running
EVAL=$GME/document_embedding_eval_int8_int4_gpu
CUSTOM_PY=$OV/bin/intel64/Release/python
CUSTOM_LIB=$OV/bin/intel64/Release
TBB=$OV/temp/Linux_x86_64/tbb/lib
PY=/home/chaofan/envs/qwen_env/bin/python3

env -i HOME="$HOME" PATH=/usr/bin:/bin \
  PYTHONPATH="$CUSTOM_PY:$GME" \
  LD_LIBRARY_PATH="$CUSTOM_LIB:$TBB" \
  OPENVINO_LIB_PATHS="$CUSTOM_LIB" \
  OPENVINO_PLUGIN_PATH="$CUSTOM_LIB" \
  OUTPUT="$EVAL/results/benchmark_gme_round4_vnomask_int8_int4_causal_20260826.json" \
  "$PY" -u "$EVAL/run_eval.py"
```

脚本默认 precision 是 `int8_vnomask,int4_causal_vnomask`。历史结果仍保留，
不要用默认 fallback `results/benchmark.json` 覆盖它们。

## 历史 masked visual 报告

以下原有正文记录 2026-08-25 的 masked visual optimized run 和未优化 ordinary
baseline。它保留用于实验 provenance 和纵向对照；当前结论以上面的第四轮章节为准。

## 1. 结论摘要

这段历史评测使用当时的 GME GPU plugin 优化：

- OpenVINO 官方基线：`2026.3.0`，commit `8a17657b99`
- GME plugin 优化：`opt-1 + opt-2 + opt-3`
- 当时 custom core/plugin：`2026.3.0-22451-8a17657b995-qwen-ttft-optimizations`
- device：`GPU`
- document：14 页工程 PDF `060_OT_Downlook_PR_Error.pdf`

本次没有重新测试未优化基线，直接使用已有的
[document_embedding_benchmark_before_optimized.json](results/document_embedding_benchmark_before_optimized.json)
作为对照。历史优化结果来自
[benchmark_gme_opt_int8_int4_causal_20260825.json](results/benchmark_gme_opt_int8_int4_causal_20260825.json)。

需要区分两个对比：

1. **INT8**：未优化普通 `int8` 对 GME 全优化普通 `int8`，可用于观察 plugin 优化收益。
2. **INT4**：未优化普通 `int4` 对 GME 全优化 `int4_causal`，同时包含 plugin 优化和 causal 导出收益，不能解释成纯 plugin 优化收益。

package 中没有 `int8_causal` 导出，因此没有对 int8 causal 做外推，也没有重新导出不存在的模型。
历史 `int4_causal` 只替换 language IR，visual IR 仍使用同一个 `gme_visual_int4.xml`。

### batch 8 主结果

性能变化使用均值计算：`(baseline mean - optimized mean) / baseline mean`。
正值表示优化后更快。

| Precision 对比 | 表示 | Baseline mean (ms) | Optimized mean (ms) | 变化 | Baseline items/s | Optimized items/s |
|---|---|---:|---:|---:|---:|---:|
| INT8 -> INT8 | page_text | 568.104 | 515.255 | +9.30% | 24.643 | 27.171 |
| INT8 -> INT8 | text_chunk_256 | 568.450 | 521.777 | +8.21% | 24.628 | 26.831 |
| INT8 -> INT8 | page_image_150 | 68482.061 | 70477.171 | -2.91% | 0.204 | 0.199 |
| INT8 -> INT8 | page_fused_150 | 71149.129 | 71514.594 | -0.51% | 0.197 | 0.196 |
| INT4 -> INT4 causal | page_text | 555.459 | 444.425 | +19.99% | 25.204 | 31.501 |
| INT4 -> INT4 causal | text_chunk_256 | 554.420 | 443.674 | +19.98% | 25.252 | 31.555 |
| INT4 -> INT4 causal | page_image_150 | 67823.817 | 65637.068 | +3.22% | 0.206 | 0.213 |
| INT4 -> INT4 causal | page_fused_150 | 68941.041 | 66165.529 | +4.03% | 0.203 | 0.212 |

文本 embedding 是本次最稳定的收益来源。INT8 普通模型的 page_text 均值下降约 9.3%；
普通 INT4 到 causal INT4 的 page_text 均值下降约 20.0%，其中包含 causal SDPA 导出减少
显式 mask 的收益。INT8 图像路径在本次单次矩阵中略慢，不能把文本路径收益外推到所有视觉路径。

## 2. 测试环境和模型选择

| 项目 | 内容 |
|---|---|
| CPU/GPU | Intel Core Ultra 7 356H，Intel Graphics，GPU device 0 |
| OpenVINO source | `benchmark/openvino`，官方基线 commit `8a17657b99` + GME `all_gme_int4_gpu_opts.patch` |
| OpenVINO core/plugin | `2026.3.0-22451-8a17657b995-qwen-ttft-optimizations` |
| Python binding | 同一源码构建的 `benchmark/openvino/bin/intel64/Release/python` custom binding |
| Model package | `ov_models/gme-qwen2-vl-2b-instruct-openvino` |
| Device | `GPU` |
| PDF | 14 pages, 375,895 bytes, 5,213 extracted characters |
| Page rendering | 150 DPI, 1500 x 1125 pixels |
| Batch sizes | 1, 4, 8, 14 |
| 每个 batch | 1 次 warm-up + 3 次 measured calls |
| 计时 | GME Python API 端到端 wall time，包含 preprocessing 和 OpenVINO inference，不包含模型 load |
| Embedding dimension | 1536，L2 normalized |

| Precision key | Language IR | Visual IR | 本次用途 |
|---|---|---|---|
| `int8` | `gme_language_int8.xml` | `gme_visual_int8.xml` | 普通 INT8 plugin 对比 |
| `int4` baseline | `gme_language_int4.xml` | `gme_visual_int4.xml` | 未应用 patch 的已有基线 |
| `int4_causal` optimized | `gme_language_int4_causal.xml` | `gme_visual_int4.xml` | GME 优化后的当前结果 |

package precision 配置中没有 `int8_causal`。因此 INT8 使用当前已有的普通模型即可，
不重新导出；INT4 使用最新已有的 `int4_causal` 模型。

## 3. 性能指标定义

- `mean ms`：该 batch 的 3 次 measured API wall time 均值。
- `p50 ms`：该 batch 的 3 次 measured API wall time 中位数。
- `items/s`：全部 item 数除以 mean wall time。
- `tokens/s`：文本 item 的 token 总数除以 mean wall time。
- `peak RSS MB`：评测进程观测到的峰值 RSS。
- `pure_model_load_ms`：GME wrapper 构造和两个 IR compile 时间，不计入 embedding latency。
- `speedup`：baseline latency / optimized latency。

## 4. GME 优化后的完整性能矩阵

以下是 custom GME core、完整 GME plugin patch 和当前模型选择下的新结果。
每个单元格为 `mean / p50`，时间单位为 ms。

### 4.1 INT8 普通模型

| 表示 | batch | mean ms | p50 ms | items/s | tokens/s | peak RSS MB |
|---|---:|---:|---:|---:|---:|---:|
| page_text | 1 | 919.363 | 918.893 | 15.228 | 1143.182 | 1189 |
| page_text | 4 | 524.585 | 524.652 | 26.688 | 2003.489 | 1195 |
| page_text | 8 | 515.255 | 515.166 | 27.171 | 2039.765 | 1196 |
| page_text | 14 | 577.427 | 576.583 | 24.245 | 1820.143 | 1197 |
| text_chunk_256 | 1 | 788.235 | 787.756 | 17.761 | 1333.359 | 1197 |
| text_chunk_256 | 4 | 526.980 | 527.467 | 26.566 | 1994.383 | 1197 |
| text_chunk_256 | 8 | 521.777 | 514.620 | 26.831 | 2014.270 | 1198 |
| text_chunk_256 | 14 | 579.979 | 579.600 | 24.139 | 1812.134 | 1198 |
| page_image_150 | 1 | 71342.713 | 71248.104 | 0.196 | -- | 1302 |
| page_image_150 | 4 | 70451.272 | 70433.314 | 0.199 | -- | 1532 |
| page_image_150 | 8 | 70477.171 | 70501.150 | 0.199 | -- | 1613 |
| page_image_150 | 14 | 70805.562 | 70803.834 | 0.198 | -- | 1832 |
| page_fused_150 | 1 | 72242.379 | 72247.859 | 0.194 | -- | 1649 |
| page_fused_150 | 4 | 71492.363 | 71509.557 | 0.196 | -- | 1767 |
| page_fused_150 | 8 | 71514.594 | 71490.648 | 0.196 | -- | 1738 |
| page_fused_150 | 14 | 72301.756 | 72321.716 | 0.194 | -- | 1855 |

### 4.2 INT4 causal 模型

| 表示 | batch | mean ms | p50 ms | items/s | tokens/s | peak RSS MB |
|---|---:|---:|---:|---:|---:|---:|
| page_text | 1 | 600.395 | 599.951 | 23.318 | 1750.514 | 1649 |
| page_text | 4 | 458.773 | 458.217 | 30.516 | 2290.891 | 1664 |
| page_text | 8 | 444.425 | 444.452 | 31.501 | 2364.856 | 1664 |
| page_text | 14 | 519.600 | 519.261 | 26.944 | 2022.711 | 1665 |
| text_chunk_256 | 1 | 604.967 | 603.668 | 23.142 | 1737.285 | 1668 |
| text_chunk_256 | 4 | 457.612 | 457.227 | 30.594 | 2296.707 | 1668 |
| text_chunk_256 | 8 | 443.674 | 443.928 | 31.555 | 2368.855 | 1668 |
| text_chunk_256 | 14 | 519.002 | 518.905 | 26.975 | 2025.041 | 1668 |
| page_image_150 | 1 | 65853.121 | 65831.167 | 0.213 | -- | 1879 |
| page_image_150 | 4 | 65459.968 | 65468.412 | 0.214 | -- | 1839 |
| page_image_150 | 8 | 65637.068 | 65632.688 | 0.213 | -- | 1915 |
| page_image_150 | 14 | 66171.858 | 66176.046 | 0.212 | -- | 2023 |
| page_fused_150 | 1 | 66374.947 | 66358.822 | 0.211 | -- | 1908 |
| page_fused_150 | 4 | 65809.940 | 65812.387 | 0.213 | -- | 1850 |
| page_fused_150 | 8 | 66165.529 | 66172.347 | 0.212 | -- | 1903 |
| page_fused_150 | 14 | 66217.454 | 66218.399 | 0.211 | -- | 1854 |

## 5. 未应用 patch 基线矩阵

基线数据来自已有的
[document_embedding_benchmark_before_optimized.json](results/document_embedding_benchmark_before_optimized.json)。
它包含普通 `int8` 和普通 `int4` 的完整 batch 矩阵；本次没有重新运行基线。

### 5.1 Baseline INT8

| 表示 | batch | mean ms | p50 ms | items/s | tokens/s | peak RSS MB |
|---|---:|---:|---:|---:|---:|---:|
| page_text | 1 | 1097.574 | 1054.600 | 12.755 | 957.566 | 1176 |
| page_text | 4 | 573.847 | 574.534 | 24.397 | 1831.497 | 1187 |
| page_text | 8 | 568.104 | 568.338 | 24.643 | 1850.012 | 1188 |
| page_text | 14 | 643.599 | 643.282 | 21.753 | 1633.005 | 1189 |
| text_chunk_256 | 1 | 840.562 | 839.525 | 16.656 | 1250.354 | 1189 |
| text_chunk_256 | 4 | 578.438 | 577.871 | 24.203 | 1816.963 | 1189 |
| text_chunk_256 | 8 | 568.450 | 568.536 | 24.628 | 1848.886 | 1190 |
| text_chunk_256 | 14 | 640.842 | 640.918 | 21.846 | 1640.031 | 1190 |
| page_image_150 | 1 | 69518.912 | 69509.182 | 0.201 | -- | 1311 |
| page_image_150 | 4 | 68615.930 | 68557.304 | 0.204 | -- | 1544 |
| page_image_150 | 8 | 68482.061 | 68501.917 | 0.204 | -- | 1623 |
| page_image_150 | 14 | 68685.297 | 68675.332 | 0.204 | -- | 1855 |
| page_fused_150 | 1 | 70964.440 | 70970.930 | 0.197 | -- | 1655 |
| page_fused_150 | 4 | 71006.911 | 71022.951 | 0.197 | -- | 1772 |
| page_fused_150 | 8 | 71149.129 | 71150.001 | 0.197 | -- | 1744 |
| page_fused_150 | 14 | 71583.543 | 71587.751 | 0.196 | -- | 1860 |

### 5.2 Baseline INT4 普通模型

| 表示 | batch | mean ms | p50 ms | items/s | tokens/s | peak RSS MB |
|---|---:|---:|---:|---:|---:|---:|
| page_text | 1 | 718.208 | 710.877 | 19.493 | 1463.365 | 2290 |
| page_text | 4 | 555.946 | 556.170 | 25.182 | 1890.473 | 2291 |
| page_text | 8 | 555.459 | 554.938 | 25.204 | 1892.128 | 2292 |
| page_text | 14 | 648.011 | 647.880 | 21.605 | 1621.887 | 2293 |
| text_chunk_256 | 1 | 719.262 | 716.003 | 19.464 | 1461.221 | 2293 |
| text_chunk_256 | 4 | 552.007 | 552.122 | 25.362 | 1903.960 | 2293 |
| text_chunk_256 | 8 | 554.420 | 554.454 | 25.252 | 1895.677 | 2293 |
| text_chunk_256 | 14 | 649.126 | 648.960 | 21.567 | 1619.099 | 2293 |
| page_image_150 | 1 | 68472.498 | 68472.554 | 0.204 | -- | 2402 |
| page_image_150 | 4 | 67838.338 | 67822.847 | 0.206 | -- | 2421 |
| page_image_150 | 8 | 67823.817 | 67828.071 | 0.206 | -- | 2492 |
| page_image_150 | 14 | 67806.142 | 67809.702 | 0.206 | -- | 2618 |
| page_fused_150 | 1 | 69410.219 | 69379.598 | 0.202 | -- | 2488 |
| page_fused_150 | 4 | 68805.302 | 68794.784 | 0.203 | -- | 2437 |
| page_fused_150 | 8 | 68941.041 | 68952.987 | 0.203 | -- | 2495 |
| page_fused_150 | 14 | 69518.718 | 69497.893 | 0.201 | -- | 2619 |

## 6. 优化前后完整对比

### 6.1 INT8 普通模型 -> GME 优化 INT8

`speedup` 为 baseline mean / optimized mean；`p50 speedup` 为 baseline p50 / optimized p50。

| 表示 | batch | Baseline mean | Optimized mean | Mean speedup | Baseline p50 | Optimized p50 | P50 speedup | RSS baseline -> optimized |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| page_text | 1 | 1097.574 | 919.363 | 1.194x | 1054.600 | 918.893 | 1.148x | 1176 -> 1189 MB |
| page_text | 4 | 573.847 | 524.585 | 1.094x | 574.534 | 524.652 | 1.095x | 1187 -> 1195 MB |
| page_text | 8 | 568.104 | 515.255 | 1.103x | 568.338 | 515.166 | 1.103x | 1188 -> 1196 MB |
| page_text | 14 | 643.599 | 577.427 | 1.115x | 643.282 | 576.583 | 1.116x | 1189 -> 1197 MB |
| text_chunk_256 | 1 | 840.562 | 788.235 | 1.066x | 839.525 | 787.756 | 1.066x | 1189 -> 1197 MB |
| text_chunk_256 | 4 | 578.438 | 526.980 | 1.098x | 577.871 | 527.467 | 1.096x | 1189 -> 1197 MB |
| text_chunk_256 | 8 | 568.450 | 521.777 | 1.090x | 568.536 | 514.620 | 1.105x | 1190 -> 1198 MB |
| text_chunk_256 | 14 | 640.842 | 579.979 | 1.105x | 640.918 | 579.600 | 1.105x | 1190 -> 1198 MB |
| page_image_150 | 1 | 69518.912 | 71342.713 | 0.975x | 69509.182 | 71248.104 | 0.976x | 1311 -> 1302 MB |
| page_image_150 | 4 | 68615.930 | 70451.272 | 0.974x | 68557.304 | 70433.314 | 0.973x | 1544 -> 1532 MB |
| page_image_150 | 8 | 68482.061 | 70477.171 | 0.971x | 68501.917 | 70501.150 | 0.972x | 1623 -> 1613 MB |
| page_image_150 | 14 | 68685.297 | 70805.562 | 0.970x | 68675.332 | 70803.834 | 0.970x | 1855 -> 1832 MB |
| page_fused_150 | 1 | 70964.440 | 72242.379 | 0.982x | 70970.930 | 72247.859 | 0.982x | 1655 -> 1649 MB |
| page_fused_150 | 4 | 71006.911 | 71492.363 | 0.993x | 71022.951 | 71509.557 | 0.993x | 1772 -> 1767 MB |
| page_fused_150 | 8 | 71149.129 | 71514.594 | 0.995x | 71150.001 | 71490.648 | 0.995x | 1744 -> 1738 MB |
| page_fused_150 | 14 | 71583.543 | 72301.756 | 0.990x | 71587.751 | 72321.716 | 0.990x | 1860 -> 1855 MB |

### 6.2 普通 INT4 -> GME 优化 causal INT4

该表同时反映 GME plugin 优化和普通 INT4 -> causal INT4 导出变化。

| 表示 | batch | Baseline mean | Optimized mean | Mean speedup | Baseline p50 | Optimized p50 | P50 speedup | RSS baseline -> optimized |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| page_text | 1 | 718.208 | 600.395 | 1.196x | 710.877 | 599.951 | 1.185x | 2290 -> 1649 MB |
| page_text | 4 | 555.946 | 458.773 | 1.212x | 556.170 | 458.217 | 1.214x | 2291 -> 1664 MB |
| page_text | 8 | 555.459 | 444.425 | 1.250x | 554.938 | 444.452 | 1.249x | 2292 -> 1664 MB |
| page_text | 14 | 648.011 | 519.600 | 1.247x | 647.880 | 519.261 | 1.248x | 2293 -> 1665 MB |
| text_chunk_256 | 1 | 719.262 | 604.967 | 1.189x | 716.003 | 603.668 | 1.186x | 2293 -> 1668 MB |
| text_chunk_256 | 4 | 552.007 | 457.612 | 1.206x | 552.122 | 457.227 | 1.208x | 2293 -> 1668 MB |
| text_chunk_256 | 8 | 554.420 | 443.674 | 1.250x | 554.454 | 443.928 | 1.249x | 2293 -> 1668 MB |
| text_chunk_256 | 14 | 649.126 | 519.002 | 1.250x | 648.960 | 518.905 | 1.251x | 2293 -> 1668 MB |
| page_image_150 | 1 | 68472.498 | 65853.121 | 1.040x | 68472.554 | 65831.167 | 1.040x | 2402 -> 1879 MB |
| page_image_150 | 4 | 67838.338 | 65459.968 | 1.036x | 67822.847 | 65468.412 | 1.036x | 2421 -> 1839 MB |
| page_image_150 | 8 | 67823.817 | 65637.068 | 1.033x | 67828.071 | 65632.688 | 1.033x | 2492 -> 1915 MB |
| page_image_150 | 14 | 67806.142 | 66171.858 | 1.025x | 67809.702 | 66176.046 | 1.025x | 2618 -> 2023 MB |
| page_fused_150 | 1 | 69410.219 | 66374.947 | 1.046x | 69379.598 | 66358.822 | 1.045x | 2488 -> 1908 MB |
| page_fused_150 | 4 | 68805.302 | 65809.940 | 1.046x | 68794.784 | 65812.387 | 1.045x | 2437 -> 1850 MB |
| page_fused_150 | 8 | 68941.041 | 66165.529 | 1.042x | 68952.987 | 66172.347 | 1.042x | 2495 -> 1903 MB |
| page_fused_150 | 14 | 69518.718 | 66217.454 | 1.050x | 69497.893 | 66218.399 | 1.050x | 2619 -> 1854 MB |

## 7. 质量结果

质量测试使用 8 条工程 query，相关页面由人工标注；它不是大规模 RAG benchmark。

### 7.1 聚合质量指标

| Precision | 表示 | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---|---:|---:|---:|---:|
| Baseline INT8 | page_text | 12.5% | 50.0% | 75.0% | 0.402 |
| Baseline INT8 | text_chunk_256 | 12.5% | 50.0% | 75.0% | 0.402 |
| Baseline INT8 | page_image_150 | 25.0% | 75.0% | 87.5% | 0.522 |
| Baseline INT8 | page_fused_150 | 12.5% | 75.0% | 87.5% | 0.457 |
| Optimized INT8 | page_text | 12.5% | 50.0% | 75.0% | 0.402 |
| Optimized INT8 | text_chunk_256 | 12.5% | 50.0% | 75.0% | 0.402 |
| Optimized INT8 | page_image_150 | 25.0% | 75.0% | 87.5% | 0.522 |
| Optimized INT8 | page_fused_150 | 12.5% | 75.0% | 87.5% | 0.457 |
| Baseline INT4 | page_text | 12.5% | 62.5% | 75.0% | 0.410 |
| Baseline INT4 | text_chunk_256 | 12.5% | 62.5% | 75.0% | 0.410 |
| Baseline INT4 | page_image_150 | 25.0% | 75.0% | 87.5% | 0.521 |
| Baseline INT4 | page_fused_150 | 12.5% | 75.0% | 75.0% | 0.446 |
| Optimized INT4 causal | page_text | 12.5% | 50.0% | 75.0% | 0.400 |
| Optimized INT4 causal | text_chunk_256 | 12.5% | 50.0% | 75.0% | 0.400 |
| Optimized INT4 causal | page_image_150 | 25.0% | 75.0% | 87.5% | 0.519 |
| Optimized INT4 causal | page_fused_150 | 12.5% | 75.0% | 75.0% | 0.426 |

INT8 的四种表示质量与基线一致。INT4 causal 与普通 INT4 的指标存在小幅差异，
因为这里同时换了导出图（ordinary -> causal）和 plugin 状态，不能把差异全部归因于
plugin patch；当前 8 条 query 上没有观察到 Recall@5 的下降。

### 7.2 跨表示 cosine

| Precision | text vs image mean/min/max | text vs fused mean/min/max | image vs fused mean/min/max |
|---|---|---|---|
| Baseline INT8 | 0.701 / 0.476 / 0.778 | 0.835 / 0.550 / 0.898 | 0.910 / 0.867 / 0.941 |
| Optimized INT8 | 0.701 / 0.476 / 0.778 | 0.835 / 0.550 / 0.898 | 0.910 / 0.867 / 0.941 |
| Baseline INT4 | 0.726 / 0.514 / 0.780 | 0.840 / 0.588 / 0.890 | 0.924 / 0.901 / 0.946 |
| Optimized INT4 causal | 0.722 / 0.501 / 0.788 | 0.837 / 0.590 / 0.896 | 0.919 / 0.890 / 0.946 |

## 8. 模型 load 和内存

| Precision comparison | Baseline load ms | Optimized load ms | Text batch 8 RSS baseline -> optimized | Image batch 8 RSS baseline -> optimized | Fused batch 8 RSS baseline -> optimized |
|---|---:|---:|---:|---:|---:|
| INT8 -> INT8 | 8636.4 | 7818.0 | 1188 -> 1196 MB | 1623 -> 1613 MB | 1744 -> 1738 MB |
| INT4 -> INT4 causal | 12469.8 | 7959.0 | 2292 -> 1664 MB | 2492 -> 1915 MB | 2495 -> 1903 MB |

模型 load 和 RSS 会受 compile cache、权重布局、GPU 状态和进程生命周期影响，不能
仅依据单次 load 时间作通用冷启动结论。INT4 causal 的内存下降是当前两个模型/导出
结果的观测值，不应外推到所有 INT4 导出方案。

## 9. 复现命令

以下命令使用同一套带 `-qwen-ttft-optimizations` 后缀的 custom core、Python binding
和 GPU plugin，不使用 qwen_env 中官方 OpenVINO wheel 的 Python binding：

```bash
ROOT=/home/chaofan/model_benchmark
OV=$ROOT/benchmark/openvino
GME=$ROOT/gme-Qwen2-VL-2B-Instruct-Running
cd $GME/document_embedding_eval_int8_int4_gpu

env -i HOME="$HOME" PATH=/usr/bin:/bin \
    PYTHONPATH="$OV/bin/intel64/Release/python:$GME" \
    LD_LIBRARY_PATH="$OV/bin/intel64/Release:$OV/temp/Linux_x86_64/tbb/lib" \
    OPENVINO_LIB_PATHS="$OV/bin/intel64/Release" \
    OPENVINO_PLUGIN_PATH="$OV/bin/intel64/Release" \
    OUTPUT="$PWD/results/benchmark_gme_round4_vnomask_int8_int4_causal_20260826.json" \
    /home/chaofan/envs/qwen_env/bin/python3 -u run_eval.py
```

当前脚本默认使用：

```text
PRECISIONS=int8_vnomask,int4_causal_vnomask
BATCHES=1,4,8,14
REPEATS=3 measured calls per batch, with one warm-up call
```

## 10. 原始结果和边界

- 当前第四轮结果：[results/benchmark_gme_round4_vnomask_int8_int4_causal_20260826.json](results/benchmark_gme_round4_vnomask_int8_int4_causal_20260826.json)
- 历史 masked visual 优化结果：[results/benchmark_gme_opt_int8_int4_causal_20260825.json](results/benchmark_gme_opt_int8_int4_causal_20260825.json)
- 未应用 patch 的已有基线：[results/document_embedding_benchmark_before_optimized.json](results/document_embedding_benchmark_before_optimized.json)
- 旧的 `results/benchmark.json` 保留为历史结果，不作为本报告当前数据源。
- image/fused 阶段拆分：[profile_gme_image_int8_vnomask.json](../gme-qwen2-vl-int4-hotspot-optimization/results/r4_rerun_20260826/profile_gme_image_int8_vnomask.json)、[profile_gme_image_int4_causal_vnomask.json](../gme-qwen2-vl-int4-hotspot-optimization/results/r4_rerun_20260826/profile_gme_image_int4_causal_vnomask.json)
- 当前评测脚本：[run_eval.py](run_eval.py)
- 当前运行说明：[README.md](README.md)
- GME 完整优化 patch：[../gme-qwen2-vl-int4-hotspot-optimization/patches/all_gme_int4_gpu_opts.patch](../gme-qwen2-vl-int4-hotspot-optimization/patches/all_gme_int4_gpu_opts.patch)

当前第四轮结论见本报告顶部：在当前机器和这份 PDF 上，`int8_vnomask` 与
`int4_causal_vnomask` 已完成四表示和 batch 1/4/8/14 重测；旧的 plugin/causal
收益结论仍保留在下方历史章节，且普通 INT4 -> causal INT4 的差异不能解释成
plugin-only。`int8_causal` 不存在，也没有为它做外推或重新导出。
