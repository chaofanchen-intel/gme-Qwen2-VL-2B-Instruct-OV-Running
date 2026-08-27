# GME-Qwen2-VL-2B-Instruct OpenVINO 部署与性能汇报

**汇报对象**：项目负责人、组内研发与部署同学  
**测试环境**：Intel Core Ultra 7 356H / Intel iGPU / 当前 custom OpenVINO core/plugin

**报告范围**：本文保留 2026-08-01 至 2026-08-03 的历史 `repeat=20` 汇报数据，并在末尾链接第四轮 fresh rerun。

**当前 package**：默认 `int8_vnomask`；低延迟推荐 `int4_causal_vnomask`。`_vnomask` 使用 mask-free visual IR，旧 masked visual 结果只作历史对照。
**数据口径**：历史主表为 `repeat=20`；第四轮 fresh 矩阵为 Python/GPU `repeat=5`，以 mean 为主，完整原始 JSON 单独保存。

---

## 1. 结论先行

### 1.1 项目结论

本项目已完成 GME-Qwen2-VL-2B-Instruct 在 Intel CPU 和 Intel iGPU 上的 OpenVINO 部署，支持：

- text、query、image、fused 四种 embedding 模式；
- Python 参考运行时和 C++ 主运行时；
- FP32、INT8、INT4 多种精度；
- 任意样本数、任意图片分辨率的 batch API；
- CPU/GPU 设备差异化 batch 调度；
- tokenizer、视觉编码、语言解码、池化和归一化全部在 OpenVINO Runtime 内执行。

最终输出固定为 `float32[B,1536]`，每行 L2 归一化，可直接用于 cosine similarity、内积检索和 ANN 索引。

### 1.2 部署建议

| 场景 | 推荐方案 | 原因 |
| --- | --- | --- |
| Intel GPU 生产默认 | `int8_vnomask` + `padded` | 文本精度接近 FP32，视觉路径使用 mask-free IR；batch 后吞吐最好 |
| Intel GPU 低延迟文本 | `int4_causal_vnomask` + `padded` | text 延迟最低，模型体积更小；上线前必须完成业务 Recall@K/nDCG 验证 |
| Intel CPU 生产默认 | `int8_vnomask` + `per_sample` + `streams="auto"` | CPU 上 padded 基本无收益，多 stream 能显著提升 batch 吞吐 |
| Intel CPU 精度优先 | FP32 | 参考级精度，但延迟和内存成本最高 |
| 严格图像精度场景 | Python INT8 或统一 Pillow 预处理后的 C++ | C++ OpenCV 与 Python Pillow 的 resize 存在可测差异 |

**当前建议默认选择：`int8_vnomask`。** `int4_causal_vnomask` 适合作为 GPU 文本和轻量部署的低延迟选项，但不能只按性能上线，必须结合业务检索指标决策。

### 1.3 关键性能结论

- 当前 fresh GPU `int8_vnomask` 短文本 batch=1：约 `43.8–44.1 ms/call`；`int4_causal_vnomask`：约 `32.0–32.3 ms/call`。
- 当前 fresh GPU `int8_vnomask` `448x448` image：`310.5 ms/call`；`int4_causal_vnomask`：`296.4 ms/call`。
- 当前 fresh GPU `int8_vnomask` `1288x728` image：`1850.6 ms/call`；`int4_causal_vnomask`：`1800.3 ms/call`。
- 图片从 256 visual tokens 增加到 1260 visual tokens，GPU INT8 image 延迟约增加 9.9 倍。
- 1024 token 文本相对 32 token 明显超线性增长，不能用短文本结果外推长文本 SLA。
- CPU batch 吞吐主要依靠 `streams="auto"`；GPU batch 吞吐主要依靠 `padded`。

### 1.4 第四轮 fresh rerun（2026-08-26）

当前结果来自 `benchmark/openvino/bin/intel64/Release` 的 matching custom Python
binding/core/plugin，版本为 `2026.3.0-22451-8a17657b995-qwen-ttft-optimizations`。
完整原始矩阵见 [hotspot rerun results](gme-qwen2-vl-int4-hotspot-optimization/results/r4_rerun_20260826/)，
document 四表示和质量结果见 [document REPORT](document_embedding_eval_int8_int4_gpu/REPORT.md)。

| workload | `int8_vnomask` | `int4_causal_vnomask` |
| --- | ---: | ---: |
| text/query, 32 tokens (ms/call) | 44.12 / 43.80 | 32.32 / 31.99 |
| text/query, 1024 tokens (ms/call) | 279.68 / 280.77 | 226.00 / 226.16 |
| image/fused, 448x448 (ms/call) | 310.53 / 306.94 | 296.37 / 296.43 |
| image/fused, 1288x728 (ms/call) | 1850.57 / 1850.86 | 1800.29 / 1795.29 |

在 text=128 tokens 的 batch fresh sweep 中，`padded + streams=1` 的 N=8 为
`32.47 ms/embedding`（INT8）和 `27.64 ms/embedding`（causal INT4）；
448x448 image 的 N=8 为 `284.46` 和 `279.71 ms/embedding`。完整的
`per_sample`/`padded`、`streams=1/auto` 和 N=1/2/4/8/16/32 数据保留在上述 JSON，
本轮主要统计按 mean 而非 p50 解读。

---

## 2. 项目目标与完成范围

### 2.1 目标

将 GME-Qwen2-VL-2B-Instruct 从 Hugging Face checkpoint 转换为可部署的 OpenVINO IR，并在 Intel CPU/GPU 上提供稳定的 embedding API，覆盖：

1. 文本检索：text、query；
2. 多模态检索：image、fused；
3. 低比特量化：INT8、INT4；
4. Python 对照验证与 C++ 生产运行时；
5. 任意 batch、图片尺寸和设备的性能评估。

### 2.2 已完成交付

- OpenVINO tokenizer IR；
- `gme_visual_<precision>.xml`；
- `gme_language_<precision>.xml`；
- Python `GmeOpenVINO` 运行时；
- C++ `gme::GmeQwen2VL` 和 `gme_embed` CLI；
- FP32/量化精度验证；
- batch 正确性验证；
- 完整性能矩阵和本汇报文档。

---

## 3. 方案架构

### 3.1 端到端流程

```text
输入
  ├─ text/query
  │    └─ GME prompt 模板
  │         └─ tokenizer IR
  │              └─ language IR
  │                   └─ last valid token pooling
  │                        └─ L2 normalize
  │                             └─ embedding[B,1536]
  │
  └─ image/fused
       └─ 图片解码与 Qwen2-VL 预处理
            └─ visual IR
                 └─ visual tokens
                      └─ image-pad 注入 language IR
                           └─ last valid token pooling
                                └─ L2 normalize
                                     └─ embedding[B,1536]
```

### 3.2 三个 IR 的职责

| IR | 输入 | 输出 | 作用 |
| --- | --- | --- | --- |
| tokenizer IR | string | `input_ids`、`attention_mask` | tokenizer、模板 token 和截断 |
| visual IR | `pixel_values`、`image_grid_thw` | visual embeddings | Qwen2-VL 视觉编码与 patch merge |
| language IR | `input_ids`、`attention_mask`、`image_embeds` | `embedding[B,1536]` | token embedding、视觉注入、decoder、pooling、归一化 |

GME 不是普通生成模型后面额外接了一个 projection head。最终 1536 维向量来自 GME decoder 的 hidden size，后处理是末有效 token pooling 和 L2 normalization。

### 3.3 为什么保持 visual/language 双 IR

- text/query 不需要执行视觉塔，避免无意义计算；
- visual 和 language 可以独立量化；
- 大图视觉中间张量可以在阶段间释放，降低峰值内存；
- 更符合官方模型的 `pixel_values is not None` 分支；
- 避免单体图在文本场景上承担额外的视觉开销。

---

## 4. 对外能力与 API 语义

### 4.1 四种模式

| 模式 | 输入 | 是否执行 visual IR | 输出 |
| --- | --- | --- | --- |
| text | 文本 | 否 | `[B,1536]` |
| query | 文本 + query instruction | 否 | `[B,1536]` |
| image | 图片 | 是 | `[B,1536]` |
| fused | 文本 + 图片 | 是 | `[B,1536]` |

四种模式输出处于同一检索空间，向量内积可直接作为 cosine similarity。

### 4.2 Batch、`batch_mode` 和 `streams` 的关系

这是最容易混淆的部分：

| 名称 | 控制对象 | 示例 |
| --- | --- | --- |
| `N` | 一次公开 API 调用中包含的样本数，即 batch size | `get_text_embeddings([a,b,c,d])` 的 N=4 |
| `batch_size` | API 分块上限，不是本报告 batch 扫描的 N | 100 条输入、`batch_size=32` 会拆成多个公开调用 |
| `batch_mode` | language IR 如何消费一次调用里的 N 条样本 | `per_sample` 或 `padded` |
| `streams` | OpenVINO plugin 如何调度 infer request | `1` 或 `auto` |

#### `per_sample`

一次 API 调用仍然是一个 batch，但 language IR 内部拆成 N 个真实长度 request：

```text
一次 API 调用：      [A, B, C, D]     N=4
language IR request： [1,Sa] [1,Sb] [1,Sc] [1,Sd]
```

特点：

- 返回 N 行 embedding；
- 去掉 right-padding，按每条样本真实长度计算；
- 结果与逐条调用逐位一致；
- `streams=1` 时 request 主要排队执行；
- `streams="auto"` 时 plugin 可以重叠多个 request，提升吞吐；
- CPU 上通常更合适。

#### `padded`

一次 API 调用也仍然是一个 batch，但 language IR 只接收一次 `[N,S]`：

```text
一次 API 调用：      [A, B, C, D]     N=4
language IR request： [4, max(Sa,Sb,Sc,Sd)]
                      短样本通过 padding 补齐
```

特点：

- 返回 N 行 embedding；
- 一次提交整个 batch，减少 GPU request 提交开销；
- ragged 输入会为短样本补 padding，padding 也会参与计算；
- 结果可能相对逐条调用产生极小数值漂移；
- GPU 上通常更有优势；
- CPU 单个 request 已经能占满核心时，通常没有明显收益。

#### `streams`

`streams` 不是 batch size，也不会改变 N 和输出行数。

- `streams=1`：LATENCY 模式，倾向于让单条 request 获得完整设备资源，单样本延迟最低；
- `streams="auto"`：THROUGHPUT 模式，由 plugin 选择并发度；本机 CPU 约 4 个流；
- 只有 `per_sample` 已经产生 N 个独立 request 时，streams 才能直接提供 request 间并发；
- `padded` 一次只有一个主要 language request，`streams="auto"` 不会把它拆成 N 个并行 request。

### 4.3 N>1 的性能含义

| 组合 | language IR request 数 | CPU 表现 | GPU 表现 |
| --- | ---: | --- | --- |
| `per_sample + streams=1` | N | 单条延迟低，batch 总时间接近 N 次串行 | 通常受提交开销影响，吞吐一般 |
| `per_sample + streams=auto` | N 个可重叠 request | 吞吐最高，单样本延迟变慢 | 通常仍不如 padded |
| `padded + streams=1` | 1 | 通常与串行接近，无明显收益 | 通常最佳，合并提交开销 |
| `padded + streams=auto` | 1 个 padded request | 可能因资源配置变慢 | 接近 padded=1，可能有波动 |

本报告中：

- CPU INT8 text 使用 `per_sample + streams=auto`，最佳约 `172–173 ms/emb`；
- GPU INT8 text 使用 `padded + streams=1`，最佳约 `32.6–34.0 ms/emb`；
- GPU INT4 text 使用 `padded + streams=1`，最佳约 `32.2–34.0 ms/emb`。

visual IR 是例外：图片始终逐图调用，`batch_mode` 只影响后续 language IR。原因是不同图片数和分辨率组合的 visual trace 可能产生错误结果，逐图调用是当前通用且正确的方案。

---

## 5. 正确性与精度结论

### 5.1 FP32 参考一致性

- Python OpenVINO FP32 的 text/query/image/fused 与官方 PyTorch 黄金向量均达到 `cosine=1.000000`；
- C++ FP32 text/query 达到参考级一致；
- C++ image/fused 存在图像预处理差异，主要来自 OpenCV 与 Pillow resize 实现不同。

### 5.2 量化结论

| 设备 | 精度 | text/query | image/fused | 结论 |
| --- | --- | --- | --- | --- |
| CPU | INT8 | 约 0.998 | 约 0.993–0.998 | 默认候选，质量和性能平衡 |
| GPU | INT8 | 约 0.998 | 约 0.989–0.996 | 默认候选，适合生产 |
| CPU/GPU | INT4 | 文本约 0.89–0.95 | 约 0.93–0.97 | 体积小；文本质量需业务验证 |

视觉塔量化时必须排除 rotary 相关计算，否则会造成视觉输出明显崩坏。当前量化流程已包含该保护。

### 5.3 1024 token C++/Python 差异说明

1024 token 的 GPU INT8/INT4 对 GPU 状态更敏感，不能简单归因于“加载了不同模型”：

- 两端使用同一个 OpenVINO runtime 和同一批 language IR；
- CPU 三种精度、GPU FP32 在 1024 上基本一致；
- C++ 使用同步 `InferRequest::infer()`，Python 使用显式 `start_async()/wait()` request pool；
- 长序列量化 attention/memory kernel 对 GPU 频率、资源竞争和 request 提交方式更敏感；
- 交替诊断时 C++/Python 的快慢方向发生过反转，因此单次差值不适合作为稳定回归阈值。

最新独立 C++ GPU 1024 复测结果：

| 精度 | text | query |
| --- | ---: | ---: |
| FP32 | 1619.4 ms | 1623.9 ms |
| INT8 | 574.4 ms | 612.9 ms |
| INT4 | 425.1 ms | 460.1 ms |

正式回归应独占 GPU、固定电源策略、交替运行多轮，并报告中位数和分位数。

---

## 6. 性能结果

### 6.1 N=1 基准延迟

workload：text/query=32 tokens，image/fused=`448x448`、256 visual tokens。

| 设备 | 精度 | Python text | C++ text | Python query | C++ query | Python image | C++ image |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CPU | FP32 | 255.6 | 257.6 | 252.5 | 263.2 | 5211.2 | 5272.9 |
| CPU | INT8 | 88.6 | 89.2 | 89.3 | 98.0 | 2699.0 | 2701.3 |
| CPU | INT4 | 100.4 | 100.1 | 101.0 | 111.9 | 3055.2 | 3049.4 |
| GPU | FP32 | 142.2 | 198.3 | 143.0 | 148.6 | 2151.8 | 2283.0 |
| GPU | INT8 | 42.5 | 41.2 | 41.9 | 43.2 | 509.4 | 498.8 |
| GPU | INT4 | 30.8 | 29.7 | 30.5 | 26.9 | 499.9 | 490.9 |

### 6.2 文本长度趋势

Python 主路径、N=1：

| 设备 | 精度 | 32 tok | 128 tok | 512 tok | 1024 tok |
| --- | --- | ---: | ---: | ---: | ---: |
| CPU | FP32 | 251.1 | 801.1 | 3198.1 | 6796.8 |
| CPU | INT8 | 88.0 | 333.6 | 1462.2 | 3194.7 |
| CPU | INT4 | 98.4 | 386.9 | 1679.0 | 3658.7 |
| GPU | FP32 | 140.9 | 184.3 | 712.7 | 1626.5 |
| GPU | INT8 | 42.6 | 54.3 | 138.3 | 292.9 |
| GPU | INT4 | 31.0 | 44.4 | 143.8 | 279.7 |

结论：1024 相对 32 token，CPU FP32/INT8/INT4 约增加 27.1x/36.3x/37.2x；GPU INT8/INT4 约增加 6.9x/9.0x。长文本必须单独设定 SLA 和长度上限。

### 6.3 图片尺寸趋势

Python GPU image：

| 精度 | 448x448 / 256 tok | 644x476 / 391 tok | 672x672 / 576 tok | 896x896 / 1024 tok | 1288x728 / 1196 tok | logo / 1260 tok |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| INT8 | 515.6 | 845.2 | 1370.8 | 3391.0 | 4562.1 | 5083.8 |
| INT4 | 495.0 | 845.9 | 1373.1 | 3382.8 | 4574.2 | 5106.5 |

图片延迟主要由 visual token 数决定，不是原始像素数或宽高比。`448x448` 是当前 256 token 下界；logo 是 1260 token 的最坏情况，不应作为普通图片 SLA。

### 6.4 Batch 实测结果

以下为主部署路径 Python 的代表性结果，单元格格式为 `ms/call (ms/emb)`。`N` 是一次 API 调用内的样本数；
完整的 Python/C++、CPU/GPU、`per_sample/padded`、`streams=1/auto`、INT8/INT4 矩阵见 [PERFORMANCE_REPORT.md](PERFORMANCE_REPORT.md)。

#### CPU INT8 text（128 tokens）

| batch_mode | streams | N=1 | N=4 | N=8 | N=32 |
| --- | --- | ---: | ---: | ---: | ---: |
| `per_sample` | 1 | 326.3 (326.3) | 1347.4 (336.9) | 2684.0 (335.5) | 10789.9 (337.2) |
| `per_sample` | auto | 552.5 (552.5) | 698.7 (174.7) | 1466.6 (183.3) | 5547.6 (173.4) |
| `padded` | 1 | 368.7 (368.7) | 1360.2 (340.0) | 2721.1 (340.1) | 11233.7 (351.1) |
| `padded` | auto | 554.7 (554.7) | 2198.6 (549.6) | 4395.1 (549.4) | 17557.1 (548.7) |

#### GPU INT8 text（128 tokens）

| batch_mode | streams | N=1 | N=4 | N=8 | N=16 |
| --- | --- | ---: | ---: | ---: | ---: |
| `per_sample` | 1 | 54.5 (54.5) | 212.7 (53.2) | 426.9 (53.4) | 848.6 (53.0) |
| `per_sample` | auto | 57.4 (57.4) | 209.7 (52.4) | 427.6 (53.5) | 969.5 (60.6) |
| `padded` | 1 | 53.7 (53.7) | 134.8 (33.7) | 272.2 (34.0) | 522.5 (32.6) |
| `padded` | auto | 59.7 (59.7) | 134.9 (33.7) | 282.9 (35.4) | 601.1 (37.6) |

#### GPU INT4 text（128 tokens）

| batch_mode | streams | N=1 | N=4 | N=8 | N=16 |
| --- | --- | ---: | ---: | ---: | ---: |
| `per_sample` | 1 | 43.3 (43.3) | 172.5 (43.1) | 344.9 (43.1) | 689.5 (43.1) |
| `per_sample` | auto | 47.1 (47.1) | 168.4 (42.1) | 335.9 (42.0) | 670.0 (41.9) |
| `padded` | 1 | 43.8 (43.8) | 140.7 (35.2) | 260.2 (32.5) | 546.6 (34.2) |
| `padded` | auto | 47.3 (47.3) | 143.5 (35.9) | 265.4 (33.2) | 547.6 (34.2) |

#### GPU INT4 text（1024 tokens）

新增复测仅覆盖 GME Python/GPU/INT4/text，输入固定为 1024 tokens，不包含 CPU、C++ 或 image。

| batch_mode | streams | N=1 | N=4 | N=8 | N=16 | N=32 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `per_sample` | 1 | 500.6 (500.6) | 2064.9 (516.2) | 4110.0 (513.8) | 8239.2 (515.0) | 16379.2 (511.9) |
| `per_sample` | auto | 400.9 (400.9) | 1471.3 (367.8) | 2848.2 (356.0) | 6474.2 (404.6) | 10825.8 (338.3) |
| `padded` | 1 | 283.8 (283.8) | 1161.8 (290.4) | 2337.2 (292.1) | 4654.8 (290.9) | 9416.6 (294.3) |
| `padded` | auto | 292.5 (292.5) | 1172.3 (293.1) | 2350.6 (293.8) | 4689.6 (293.1) | 9624.4 (300.8) |

**1024-token 结论**：GPU Python INT4 使用 `padded + streams=1` 时约 `284–294 ms/embedding`，
而 `per_sample + streams=1` 约 `512–516 ms/embedding`；长序列下 padded 的 request 合并收益约 1.8x。

#### Image（448x448，256 visual tokens）

| 设备/精度 | batch_mode | streams | N=1 | N=4 | N=8 |
| --- | --- | --- | ---: | ---: | ---: |
| CPU INT8 | `per_sample` | auto | 4303.4 (4303.4) | 5918.2 (1479.5) | 11898.4 (1487.3) |
| CPU INT4 | `per_sample` | auto | 5264.8 (5264.8) | 6975.3 (1743.8) | 14085.4 (1760.7) |
| GPU INT8 | `padded` | 1 | 524.1 (524.1) | 1957.1 (489.3) | 3872.3 (484.0) |
| GPU INT4 | `padded` | 1 | 494.6 (494.6) | 1933.3 (483.3) | 3861.7 (482.7) |

**Batch 结论**：

- CPU：`padded` 不带来明显收益；`per_sample + streams="auto"` 将 INT8 text 的最佳单样本成本降至约 `173 ms/emb`，image 降至约 `1477 ms/emb`。
- GPU：`padded` 是主要收益来源；INT8 text 最佳约 `32.6 ms/emb`，INT4 text 最佳约 `32.5 ms/emb`，均明显优于 per-sample 的约 `42–53 ms/emb`。
- image 的 visual 阶段仍然逐图执行，batch 优势主要来自后续 language IR；图片越大，visual token 越多，单个 batch 的收益越容易被视觉计算成本抵消。

---

## 7. 方案边界与风险

### 7.1 当前已知风险

1. **图像预处理差异**：C++ OpenCV 与 Python Pillow 的 resize 不完全一致，影响 image/fused embedding 数值；
2. **GPU 长序列波动**：1024 token 量化路径受 GPU DVFS、并发和 kernel 调度影响，不宜使用单次值作为 SLA；
3. **模型编译成本**：GPU 首次 compile 可能需要数十秒，服务端必须复用模型实例或启用 OpenVINO cache；
4. **INT4 检索质量**：INT4 文本 cosine 明显低于 INT8，必须经过业务数据集验证；
5. **visual 逐图限制**：visual IR 不能简单作为任意图片数的动态 padded batch；
6. **线程安全**：当前 encoder 实例内部复用 infer request，不应未经加锁直接被多线程并发调用；
7. **NPU 未验证**：当前结论只覆盖 CPU/GPU，不能外推 NPU。

### 7.2 不建议的替代方案

- OpenVINO GenAI `TextEmbeddingPipeline` 不能直接加载当前 GME language IR；
- `VLMPipeline`/`LLMPipeline` 面向生成模型，不提供 GME embedding contract；
- Optimum CLI 可以导出 Qwen2-VL 生成模型，但输出是 logits/KV-cache，不是 GME 的 1536 维 embedding；

---

## 8. 上线建议与下一步

### 8.1 上线前必须完成

1. 在真实 query/corpus/image 数据上比较 `int8_vnomask` 与 `int4_causal_vnomask` 的 Recall@K、nDCG、Top-K 命中率；
2. 根据线上文本长度和图片 token 分布重新设置 micro-batch 大小；
3. 在目标并发下测试 P50/P95/P99，而不是只使用本报告平均值；
4. 固定 GPU 电源策略和运行环境，补充长期稳定性测试；
5. 明确 C++ 图像预处理与 Python 参考路径的验收阈值；
6. 配置模型缓存并复用长期存活的 encoder 实例。

### 8.2 推荐初始配置

```text
GPU:
     precision = int8_vnomask
  batch_mode = padded
  streams = 1
  text micro-batch = 8~16
  image micro-batch = 4~8
     low-latency text precision = int4_causal_vnomask

CPU:
     precision = int8_vnomask
  batch_mode = per_sample
  streams = auto
  text micro-batch = 4~8
  image micro-batch = 4
```

INT4 仅在业务检索质量验证通过后作为 GPU 文本低延迟配置启用。

---

## 9. 复现与原始材料

主要脚本：

- `01_reference.py`：官方 PyTorch 黄金向量；
- `02_export.py`：FP32 IR 导出；
- `03_finalize.py`：tokenizer IR 与量化；
- `04_validate_python.py` / `05_validate_cpp.py`：精度验证；
- `07_validate_batch.py`：batch 正确性；
- `08_benchmark_matrix.py`：完整性能矩阵；
- `09_retest_gpu_1024.py`：GPU 1024 长序列诊断。

原始报告与数据：

- [README.md](README.md)：handoff 操作手册与 CLI 参数；
- [GME_TECHNICAL_DETAILS.md](GME_TECHNICAL_DETAILS.md)：完整技术说明、兼容性结论与历史实测结果；
- [PERFORMANCE_REPORT.md](PERFORMANCE_REPORT.md)：完整性能矩阵与异常分析；
- `artifacts/benchmark_matrix_latency_r20.json`；
- `artifacts/benchmark_matrix_text_lengths_r20.json`；
- `artifacts/benchmark_matrix_image_sizes_r20_int8_int4.json`；
- `artifacts/benchmark_matrix_batch_r20_int8.json`；
- `artifacts/benchmark_matrix_batch_r20_int4.json`；
- `artifacts/benchmark_matrix_batch_text1024_r20_int4_python_gpu.json`：GME Python/GPU/INT4/text-only，1024 tokens；
- `artifacts/benchmark_cpp_gpu_1024_retest_r20.json`。

**汇报口径建议**：主汇报使用本文件；需要实现细节、完整表格或复现命令时，再下钻 README 和 PERFORMANCE_REPORT。
