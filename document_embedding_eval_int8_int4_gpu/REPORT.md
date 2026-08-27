# GME 文档 Embedding 评测报告：第五轮 final causal / visual optimized

## 当前第五轮 final（2026-08-27）

本节是当前 document embedding 结果的主入口。原始文件为
[`benchmark_gme_round5_final_vopt_20260827.json`](results/benchmark_gme_round5_final_vopt_20260827.json)。
最初未优化的普通 INT8/INT4 baseline 没有重复运行，直接保留并引用
[`document_embedding_benchmark_before_optimized.json`](results/document_embedding_benchmark_before_optimized.json)。
第四轮和更早结果也保留在本报告后部，作为中间阶段和 provenance。

### 运行时与模型组合

| 项目 | 当前值 |
| --- | --- |
| OpenVINO source | `benchmark/openvino`，official base `8a17657b99` + GME opt1/opt2/opt3 |
| core/plugin/Python binding | `2026.3.0-22451-8a17657b995-qwen-ttft-optimizations`，三者同源 custom build |
| device | GPU |
| package | `ov_models/gme-qwen2-vl-2b-instruct-openvino` |
| document | 14 pages, 375,895 bytes, 5,213 extracted characters |
| page rendering | 150 DPI, 1500 x 1125 pixels |
| batch sizes | 1, 4, 8, 14 |
| measurement | 1 warm-up + 3 measured calls per cell；mean 为主，同时保留 p50/p95/min/max |
| embedding | 1536 dimensions, float32, L2 normalized |

当前最终 precision 映射：

| precision key | language IR | visual IR | 语义 |
| --- | --- | --- | --- |
| `int8_causal_vopt` | `gme_language_int8_causal.xml` | `gme_visual_int8_nomask_headfirst_rank3.xml` | causal INT8 language + 第五轮最终 visual |
| `int4_causal_vopt` | `gme_language_int4_causal.xml` | `gme_visual_int4_nomask_headfirst_rank3.xml` | causal INT4 language + 第五轮最终 visual |
| `int4_gs64_causal_vopt` | `gme_language_int4_gs64_causal.xml` | `gme_visual_int4_nomask_headfirst_rank3.xml` | 精度优先 INT4 + 第五轮最终 visual |

第五轮的 `_vopt` 在第四轮 mask-free visual 基础上加入 headfirst Q/K/V 布局和整塔
rank-3 输入；后者使既有 plugin opt-3 能命中视觉 `mlp.fc1`。第五轮还补齐了之前
漏掉的 `int8_causal`。本轮没有新增 GPU plugin 改动。

### 性能摘要（mean ms/call）

| 表示 / batch | `int8_causal_vopt` | `int4_causal_vopt` | `int4_gs64_causal_vopt` |
| --- | ---: | ---: | ---: |
| `page_text` / 1 | 915.197 | 606.515 | 662.619 |
| `page_text` / 4 | 512.863 | 457.144 | 523.786 |
| `page_text` / 8 | 501.612 | 443.450 | 503.384 |
| `page_text` / 14 | 558.613 | 519.118 | 587.220 |
| `text_chunk_256` / 1 | 770.105 | 600.166 | 661.570 |
| `text_chunk_256` / 4 | 512.260 | 464.533 | 519.343 |
| `text_chunk_256` / 8 | 500.042 | 443.483 | 499.023 |
| `text_chunk_256` / 14 | 558.726 | 520.500 | 611.374 |
| `page_image_150` / 1 | 26,115.822 | 25,606.405 | 25,900.757 |
| `page_image_150` / 4 | 25,440.547 | 25,049.039 | 25,459.527 |
| `page_image_150` / 8 | 25,437.627 | 25,069.491 | 25,448.049 |
| `page_image_150` / 14 | 25,469.491 | 25,068.985 | 25,418.578 |
| `page_fused_150` / 1 | 26,577.032 | 26,091.805 | 26,355.902 |
| `page_fused_150` / 4 | 25,999.614 | 25,410.562 | 25,845.653 |
| `page_fused_150` / 8 | 25,827.996 | 25,490.157 | 25,910.498 |
| `page_fused_150` / 14 | 25,822.423 | 25,561.208 | 25,936.868 |

图像路径的 visual 阶段仍逐图执行，因此 batch 对 image/fused 的收益主要来自后续
language/request 编排；不要把 image batch 当成真正的视觉 `[N,...]` padded batch。

### `int8_causal_vopt` 完整性能矩阵

| 表示 | batch | mean ms | p50 ms | p95 ms | min ms | max ms | items/s | tokens/s | peak RSS MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| page_text | 1 | 915.197 | 916.369 | 917.167 | 911.968 | 917.255 | 15.29725 | 1148.386 | 1180 |
| page_text | 4 | 512.863 | 512.997 | 513.207 | 512.361 | 513.230 | 27.29775 | 2049.281 | 1186 |
| page_text | 8 | 501.612 | 501.759 | 501.869 | 501.194 | 501.882 | 27.91004 | 2095.247 | 1187 |
| page_text | 14 | 558.613 | 557.402 | 561.229 | 556.782 | 561.654 | 25.06209 | 1881.447 | 1188 |
| text_chunk_256 | 1 | 770.105 | 769.934 | 770.635 | 769.670 | 770.712 | 18.17933 | 1364.748 | 1188 |
| text_chunk_256 | 4 | 512.260 | 511.674 | 513.317 | 511.605 | 513.499 | 27.32989 | 2051.694 | 1189 |
| text_chunk_256 | 8 | 500.042 | 499.940 | 500.247 | 499.904 | 500.281 | 27.99766 | 2101.824 | 1189 |
| text_chunk_256 | 14 | 558.726 | 558.552 | 559.131 | 558.433 | 559.195 | 25.05699 | 1881.064 | 1189 |
| page_image_150 | 1 | 26115.822 | 26125.865 | 26126.748 | 26094.755 | 26126.847 | 0.53607 | -- | 1352 |
| page_image_150 | 4 | 25440.547 | 25440.021 | 25474.024 | 25403.817 | 25477.802 | 0.55030 | -- | 1435 |
| page_image_150 | 8 | 25437.627 | 25429.999 | 25450.652 | 25429.936 | 25452.947 | 0.55037 | -- | 1632 |
| page_image_150 | 14 | 25469.491 | 25487.520 | 25495.605 | 25424.451 | 25496.503 | 0.54968 | -- | 1846 |
| page_fused_150 | 1 | 26577.032 | 26576.665 | 26591.223 | 26561.591 | 26592.840 | 0.52677 | -- | 1698 |
| page_fused_150 | 4 | 25999.614 | 26013.538 | 26047.545 | 25933.979 | 26051.324 | 0.53847 | -- | 1654 |
| page_fused_150 | 8 | 25827.996 | 25819.548 | 25843.406 | 25818.383 | 25846.057 | 0.54205 | -- | 1749 |
| page_fused_150 | 14 | 25822.423 | 25822.347 | 25857.201 | 25783.849 | 25861.074 | 0.54216 | -- | 1852 |

### `int4_causal_vopt` 完整性能矩阵

| 表示 | batch | mean ms | p50 ms | p95 ms | min ms | max ms | items/s | tokens/s | peak RSS MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| page_text | 1 | 606.515 | 604.649 | 610.169 | 604.114 | 610.782 | 23.08269 | 1732.850 | 1661 |
| page_text | 4 | 457.144 | 456.859 | 458.073 | 456.364 | 458.208 | 30.62494 | 2299.058 | 1669 |
| page_text | 8 | 443.450 | 443.499 | 443.851 | 442.960 | 443.890 | 31.57065 | 2370.054 | 1669 |
| page_text | 14 | 519.118 | 518.995 | 519.805 | 518.466 | 519.895 | 26.96880 | 2024.586 | 1671 |
| text_chunk_256 | 1 | 600.166 | 600.950 | 601.408 | 598.090 | 601.459 | 23.32686 | 1751.181 | 1671 |
| text_chunk_256 | 4 | 464.533 | 467.225 | 469.797 | 456.290 | 470.083 | 30.13782 | 2262.489 | 1671 |
| text_chunk_256 | 8 | 443.483 | 443.662 | 444.202 | 442.524 | 444.262 | 31.56831 | 2369.878 | 1671 |
| text_chunk_256 | 14 | 520.500 | 520.467 | 520.877 | 520.110 | 520.922 | 26.89723 | 2019.214 | 1672 |
| page_image_150 | 1 | 25606.405 | 25591.013 | 25669.444 | 25550.044 | 25678.159 | 0.54674 | -- | 1731 |
| page_image_150 | 4 | 25049.039 | 25045.412 | 25064.279 | 25035.331 | 25066.375 | 0.55890 | -- | 1787 |
| page_image_150 | 8 | 25069.491 | 25064.577 | 25086.115 | 25055.388 | 25088.508 | 0.55845 | -- | 1851 |
| page_image_150 | 14 | 25068.985 | 25070.647 | 25074.629 | 25061.237 | 25075.071 | 0.55846 | -- | 1983 |
| page_fused_150 | 1 | 26091.805 | 26105.792 | 26132.681 | 26033.955 | 26135.668 | 0.53657 | -- | 1827 |
| page_fused_150 | 4 | 25410.562 | 25432.714 | 25436.862 | 25361.648 | 25437.323 | 0.55095 | -- | 1807 |
| page_fused_150 | 8 | 25490.157 | 25502.171 | 25543.435 | 25420.280 | 25548.020 | 0.54923 | -- | 1865 |
| page_fused_150 | 14 | 25561.208 | 25581.427 | 25610.513 | 25488.453 | 25613.745 | 0.54770 | -- | 1990 |

### `int4_gs64_causal_vopt` 完整性能矩阵

| 表示 | batch | mean ms | p50 ms | p95 ms | min ms | max ms | items/s | tokens/s | peak RSS MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| page_text | 1 | 662.619 | 662.200 | 665.076 | 660.261 | 665.396 | 21.12828 | 1586.130 | 1685 |
| page_text | 4 | 523.786 | 521.776 | 530.812 | 517.767 | 531.816 | 26.72846 | 2006.543 | 1689 |
| page_text | 8 | 503.384 | 502.298 | 505.279 | 502.243 | 505.610 | 27.81178 | 2087.870 | 1690 |
| page_text | 14 | 587.220 | 586.964 | 588.488 | 586.039 | 588.658 | 23.84113 | 1789.788 | 1692 |
| text_chunk_256 | 1 | 661.570 | 661.566 | 662.063 | 661.024 | 662.118 | 21.16180 | 1588.646 | 1692 |
| text_chunk_256 | 4 | 519.343 | 520.121 | 521.745 | 515.984 | 521.925 | 26.95711 | 2023.709 | 1693 |
| text_chunk_256 | 8 | 499.023 | 499.455 | 499.477 | 498.134 | 499.480 | 28.05482 | 2106.115 | 1693 |
| text_chunk_256 | 14 | 611.374 | 596.213 | 645.327 | 587.126 | 650.784 | 22.89924 | 1719.079 | 1693 |
| page_image_150 | 1 | 25900.757 | 25897.870 | 25934.610 | 25865.708 | 25938.692 | 0.54052 | -- | 1885 |
| page_image_150 | 4 | 25459.527 | 25479.069 | 25481.729 | 25417.488 | 25482.024 | 0.54989 | -- | 1884 |
| page_image_150 | 8 | 25448.049 | 25438.648 | 25481.131 | 25419.648 | 25485.851 | 0.55014 | -- | 1956 |
| page_image_150 | 14 | 25418.578 | 25420.810 | 25471.953 | 25357.287 | 25477.636 | 0.55078 | -- | 2085 |
| page_fused_150 | 1 | 26355.902 | 26345.371 | 26382.264 | 26335.971 | 26386.363 | 0.53119 | -- | 1957 |
| page_fused_150 | 4 | 25845.653 | 25850.135 | 25871.569 | 25812.874 | 25873.951 | 0.54168 | -- | 1901 |
| page_fused_150 | 8 | 25910.498 | 25920.178 | 25929.653 | 25880.610 | 25930.706 | 0.54032 | -- | 1967 |
| page_fused_150 | 14 | 25936.868 | 25947.706 | 25966.019 | 25894.845 | 25968.054 | 0.53977 | -- | 2091 |

### 初始未优化 baseline 对照（batch 8 mean）

初始 document baseline 来自同一份 PDF、同一套 8-query/14-page probe，文件为
[`document_embedding_benchmark_before_optimized.json`](results/document_embedding_benchmark_before_optimized.json)。
以下只重新计算最终相对 baseline 的 speedup，没有重新运行 baseline。正值表示
final 更快；INT8 与 INT4 的最终 language/visual 组合已经发生了 causal、mask-free
和第五轮 layout 变化，表格表示累计收益，不把它解释成单一 plugin patch 的收益。

| 表示 | 初始 INT8 | final INT8 causal | speedup | 初始 INT4 | final INT4 causal | speedup | final INT4 gs64 | speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| page_text | 568.104 | 501.612 | 1.133x (-11.7%) | 555.459 | 443.450 | 1.253x (-20.2%) | 503.384 | 1.103x (-9.4%) |
| text_chunk_256 | 568.450 | 500.042 | 1.137x (-12.0%) | 554.420 | 443.483 | 1.250x (-20.0%) | 499.023 | 1.111x (-10.0%) |
| page_image_150 | 68,482.061 | 25,437.627 | 2.692x (-62.9%) | 67,823.817 | 25,069.491 | 2.705x (-63.0%) | 25,448.049 | 2.665x (-62.5%) |
| page_fused_150 | 71,149.129 | 25,827.996 | 2.755x (-63.7%) | 68,941.041 | 25,490.157 | 2.705x (-63.0%) | 25,910.498 | 2.661x (-62.4%) |

初始 baseline 的 batch 8 RSS 为 INT8 `1188/1189/1623/1744 MB`、INT4
`2292/2293/2492/2495 MB`（按 page_text/chunk/image/fused）；第五轮 final
相应值见上面三张完整矩阵。当前 final text batch 8 throughput 为
`27.910`、`31.571`、`27.812 items/s`，image 为 `0.550`、`0.558`、
`0.550 items/s`，顺序为 INT8 causal、INT4 causal、INT4 gs64 causal。

### 当前质量结果

8 个工程 query、14 个页面的内部检索 probe：

| precision | 表示 | Recall@1 | Recall@3 | Recall@5 | MRR |
| --- | --- | ---: | ---: | ---: | ---: |
| `int8_causal_vopt` | page_text | 12.5% | 50.0% | 75.0% | 0.4020 |
| `int8_causal_vopt` | text_chunk_256 | 12.5% | 50.0% | 75.0% | 0.4020 |
| `int8_causal_vopt` | page_image_150 | 25.0% | 75.0% | 87.5% | 0.5218 |
| `int8_causal_vopt` | page_fused_150 | 12.5% | 75.0% | 87.5% | 0.4506 |
| `int4_causal_vopt` | page_text | 12.5% | 50.0% | 75.0% | 0.3996 |
| `int4_causal_vopt` | text_chunk_256 | 12.5% | 50.0% | 75.0% | 0.3996 |
| `int4_causal_vopt` | page_image_150 | 25.0% | 75.0% | 87.5% | 0.5193 |
| `int4_causal_vopt` | page_fused_150 | 12.5% | 75.0% | 75.0% | 0.4464 |
| `int4_gs64_causal_vopt` | page_text | 12.5% | 62.5% | 75.0% | 0.4124 |
| `int4_gs64_causal_vopt` | text_chunk_256 | 12.5% | 62.5% | 75.0% | 0.4124 |
| `int4_gs64_causal_vopt` | page_image_150 | 37.5% | 87.5% | 87.5% | 0.5923 |
| `int4_gs64_causal_vopt` | page_fused_150 | 12.5% | 75.0% | 87.5% | 0.4360 |

对应页面的 cross-representation cosine：

| precision | text vs image mean/min/max | text vs fused mean/min/max | image vs fused mean/min/max |
| --- | --- | --- | --- |
| `int8_causal_vopt` | 0.7041 / 0.4889 / 0.7729 | 0.8373 / 0.5615 / 0.8984 | 0.9099 / 0.8729 / 0.9376 |
| `int4_causal_vopt` | 0.7248 / 0.5188 / 0.7852 | 0.8357 / 0.5911 / 0.8928 | 0.9215 / 0.8839 / 0.9456 |
| `int4_gs64_causal_vopt` | 0.7270 / 0.4931 / 0.7930 | 0.8394 / 0.6179 / 0.9006 | 0.9257 / 0.8977 / 0.9460 |

这些 Recall/MRR 数字只是 8-query/14-page 工程 probe，不能替代真实业务检索集。
第五轮没有观察到 document probe 的结构性质量退化；INT4 是否上线仍应由业务
Recall@K、nDCG 和 Top-K 指标决定。

### Load、token 和内存

| precision | load mean ms | page token mean/min/max | chunk token mean/min/max |
| --- | ---: | --- | --- |
| `int8_causal_vopt` | 7589.058 | 75.07 / 10 / 149 | 75.07 / 10 / 149 |
| `int4_causal_vopt` | 7471.177 | 75.07 / 10 / 149 | 75.07 / 10 / 149 |
| `int4_gs64_causal_vopt` | 7353.288 | 75.07 / 10 / 149 | 75.07 / 10 / 149 |

单次 load 会受 compile cache、GPU 状态和进程生命周期影响，不能作为通用冷启动
SLA；完整 min/max、RSS 和每个 query 的 top pages 都在原始 JSON。

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
    OUTPUT="$EVAL/results/benchmark_gme_round5_final_vopt_20260827.json" \
    PRECISIONS="int8_causal_vopt,int4_causal_vopt,int4_gs64_causal_vopt" \
    "$PY" -u "$EVAL/run_eval.py"
```

第五轮 final 原始矩阵和日志见
[`gme-qwen2-vl-int4-hotspot-optimization/results/r5_final_20260827/`](../gme-qwen2-vl-int4-hotspot-optimization/results/r5_final_20260827/)。
不要用 `results/benchmark.json` 覆盖历史结果。

## 历史结果和中间阶段

以下原有正文记录第四轮 masked/vnomask 阶段和未优化 baseline，保留用于实验
provenance；当前最终结论以上面的第五轮章节为准。

### 历史 masked visual 报告（第五轮之前）

以下原有正文记录 2026-08-25 的 masked visual optimized run 和未优化 ordinary
baseline。它保留用于实验 provenance 和纵向对照；当前结论以上面的第五轮章节为准。

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

当时使用的历史 package 中没有 `int8_causal` 导出，因此第四轮没有对 int8 causal 做外推，也没有重新导出不存在的模型；第五轮 final 已补齐该导出。
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
| `int4_causal` optimized | `gme_language_int4_causal.xml` | `gme_visual_int4.xml` | 第四轮历史优化结果 |

第四轮历史 package precision 配置中没有 `int8_causal`，所以当时 INT8 使用普通模型，
INT4 使用已有的 `int4_causal` 模型。当前第五轮配置见本报告顶部的三种 `_vopt` precision。

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
    PRECISIONS="int8_vnomask,int4_causal_vnomask" \
    OUTPUT="$PWD/results/benchmark_gme_round4_vnomask_int8_int4_causal_20260826.json" \
    /home/chaofan/envs/qwen_env/bin/python3 -u run_eval.py
```

第四轮历史脚本当时使用：

```text
PRECISIONS=int8_vnomask,int4_causal_vnomask
BATCHES=1,4,8,14
REPEATS=3 measured calls per batch, with one warm-up call
```

## 10. 原始结果和边界

- 历史第四轮结果：[results/benchmark_gme_round4_vnomask_int8_int4_causal_20260826.json](results/benchmark_gme_round4_vnomask_int8_int4_causal_20260826.json)
- 历史 masked visual 优化结果：[results/benchmark_gme_opt_int8_int4_causal_20260825.json](results/benchmark_gme_opt_int8_int4_causal_20260825.json)
- 未应用 patch 的已有基线：[results/document_embedding_benchmark_before_optimized.json](results/document_embedding_benchmark_before_optimized.json)
- 旧的 `results/benchmark.json` 保留为历史结果，不作为本报告当前数据源。
- image/fused 阶段拆分：[profile_gme_image_int8_vnomask.json](../gme-qwen2-vl-int4-hotspot-optimization/results/r4_rerun_20260826/profile_gme_image_int8_vnomask.json)、[profile_gme_image_int4_causal_vnomask.json](../gme-qwen2-vl-int4-hotspot-optimization/results/r4_rerun_20260826/profile_gme_image_int4_causal_vnomask.json)
- 当前评测脚本：[run_eval.py](run_eval.py)
- 当前运行说明：[README.md](README.md)
- GME 完整优化 patch：[../gme-qwen2-vl-int4-hotspot-optimization/patches/all_gme_int4_gpu_opts.patch](../gme-qwen2-vl-int4-hotspot-optimization/patches/all_gme_int4_gpu_opts.patch)

第四轮历史结论见本报告上方的第五轮章节和本历史段：在当时的 package 上，`int8_vnomask` 与
`int4_causal_vnomask` 已完成四表示和 batch 1/4/8/14 重测；旧的 plugin/causal
收益结论仍保留在下方历史章节，且普通 INT4 -> causal INT4 的差异不能解释成
plugin-only。当时的 `int8_causal` 尚不存在，也没有为它做外推或重新导出；第五轮已提供
`int8_causal_vopt`。
