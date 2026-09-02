# GME-Qwen2-VL-2B-Instruct 技术说明与实测结果

本文保留原始运行说明中的技术解释、实验方法、兼容性判断和性能结果。面向需要理解实现边界、复现实验或做部署选型的读者；日常 handoff 操作请先看 [README.md](README.md)，完整性能矩阵请看 [PERFORMANCE_REPORT.md](PERFORMANCE_REPORT.md)。

**当前状态（第九轮）**：package 默认使用 `int8_causal_vopt`，低延迟推荐使用
`int4_causal_vopt`，精度优先 INT4 可选 `int4_gs64_causal_vopt`。`_vopt`
在 mask-free visual 的基础上加入 headfirst 和 rank3；`int8_causal` 也已补齐。
第四轮至第八轮和第五轮 final 数字仅作为历史对照。当前完整
text/query/image/fused/batch 矩阵及 INT8/INT4 baseline 对比见
[`gme-qwen2-vl-int4-hotspot-optimization/docs/16_round9_final_rebenchmark.md`](gme-qwen2-vl-int4-hotspot-optimization/docs/16_round9_final_rebenchmark.md)。

第九轮 public API 的 baseline→round-9 arithmetic-mean 摘要如下；`text/query` 或
`image/fused` 均按该顺序排列，speedup 为 baseline / round-9：

| workload | INT8 baseline → round 9 | INT8 speedup | INT4 baseline → round 9 | INT4 speedup |
| --- | ---: | ---: | ---: | ---: |
| text/query, 1024 tokens | 296.57/296.01 → 253.36/252.94 ms | 1.171x/1.170x | 291.443/280.598 → 216.73/217.05 ms | 1.345x/1.293x |
| image/fused, 448x448 | 476.69/468.07 → 275.23/277.43 ms | 1.732x/1.687x | pre-vopt 296.37/296.43 → 242.23/241.77 ms | 1.224x/1.226x |
| image/fused, 1288x728 | 4654.46/4600.58 → 1728.10/1731.73 ms | 2.693x/2.657x | pre-vopt 1800.29/1795.29 → 1618.10/1623.11 ms | 1.113x/1.106x |

document embedding batch=8 的对应 speedup 为 INT8 `page_text/page_image/page_fused`
的 `1.150x/2.714x/2.774x`，INT4 的 `1.269x/2.796x/2.795x`。原始值和 baseline
来源统一保存在
[`baseline_vs_round9_comparison.json`](gme-qwen2-vl-int4-hotspot-optimization/results/r9_latest/baseline_vs_round9_comparison.json)。

本文对应的脚本编号为连续的 `01` 到 `09`：

- `01_reference.py`：官方 PyTorch golden reference；
- `02_export.py`：FP32 visual/language IR 导出；
- `03_finalize.py`：OpenVINO tokenizer、INT8/INT4 和 package config；
- `04_validate_python.py`：Python OpenVINO 精度；
- `05_validate_cpp.py`：C++ 精度；
- `06_benchmark.py`：固定 workload 延迟；
- `07_validate_batch.py`：batch、API、截断和同形状回归；
- `08_benchmark_matrix.py`：文本长度、图像尺寸和 batch 性能矩阵；
- `09_retest_gpu_1024.py`：GPU 1024-token 交错复测。

## 1. 结论

### 1.1 模型和运行时

GME-Qwen2-VL-2B-Instruct 不是普通 Qwen2-VL 生成模型后面简单接一个 projection head。它是针对多模态检索 embedding 微调过的 checkpoint，官方 `GmeQwen2VL` 将 decoder 最后一个有效 token 的 hidden state 做 L2 normalize 后作为 1536 维向量。

本项目的主要结论：

- Python OpenVINO FP32 在 text、query、image、fused 四种模式上与官方 PyTorch golden 等价；
- C++ text/query 使用相同 tokenizer IR 和 language IR，FP32 下与 golden 等价；
- C++ image/fused 使用 OpenCV，而 Python 参考使用 Pillow，resize 重采样会带来可测的 cosine 差异；
- `int8_causal_vopt` 是当前 near-lossless 默认候选，通常能保持约 `0.99` 级别的检索向量相似度；
- `int4_causal_vopt` 是当前低延迟组合，体积更小，但必须使用业务数据验证 Recall@K、nDCG、Top-K 命中率；
- CPU 和 GPU 的 batch 优化方向不同：CPU 主要靠 streams，GPU 主要靠 padded batch；
- 本文的 CPU/GPU 结论不能外推到未验证的 NPU。

### 1.2 GME 与普通 Qwen2-VL 的区别

| 方面 | Qwen2-VL-2B-Instruct | GME-Qwen2-VL-2B-Instruct |
| --- | --- | --- |
| 主要任务 | 视觉语言生成 | text/image/fused 检索 embedding |
| 顶层使用方式 | 生成 wrapper + logits | `GmeQwen2VL` 直接返回 embedding |
| decoder 输出消费 | vocabulary logits | `last_hidden_state` 的最后有效 token |
| 最终输出 | `[B,S,vocab_size]` logits 或生成文本 | `float32[B,1536]`，L2 normalized |
| 视觉 token 范围 | 由普通 processor 策略决定 | GME 默认 `min_image_tokens=256`、`max_image_tokens=1280` |
| prompt | 对话/生成模板 | GME system/user/assistant 模板，末尾为 `<|endoftext|>` |

两个 checkpoint 的基础网络拓扑相近，均为 1536 hidden size、28 层 decoder、12 个 attention heads、2 个 KV heads 和 Qwen2-VL 视觉塔；关键差异是微调权重和 forward/output contract。不能使用普通 Qwen2-VL checkpoint 替代 GME checkpoint。

## 2. 四种模式的实际执行路径

官方 API 最终都会进入 `embed()`/`get_fused_embeddings()`，但是否执行视觉塔取决于 `images` 是否为实际图片：

| 模式 | images | processor 输出 | visual IR | language IR |
| --- | --- | --- | --- | --- |
| text | `None` | `input_ids`、`attention_mask` | 不调用 | 文本 token embedding + decoder |
| query | `None` | 同 text，instruction 不同 | 不调用 | 文本 token embedding + decoder |
| image | 实际图片 | text tensors + `pixel_values` + `image_grid_thw` | 调用 | 视觉 token 注入后 decoder |
| fused | 实际图片 + 文本 | 同 image | 调用 | 文本和视觉 token 融合后 decoder |

纯文本路径不会传空图片去执行视觉塔。为了保持 language IR 的固定三输入接口，text/query 会传入空的 `image_embeds[0,1536]`，但 prompt 没有 image-pad 位置，因此不会发生视觉替换，也不会运行 visual IR。

最终 embedding 的计算是：

```text
GME prompt template
  -> HF tokenizer with right padding
  -> token embedding
  -> optional visual embedding injection
  -> Qwen2-VL decoder
  -> hidden[batch, attention_mask.sum(-1)-1]
  -> L2 normalize
  -> embedding[batch, 1536]
```

## 3. OpenVINO IR 架构

导出包包含三类主要 IR：

| IR | 输入 | 输出 | 功能 |
| --- | --- | --- | --- |
| `openvino_tokenizer.xml` | string batch | `input_ids`、`attention_mask` | C++ 原生 tokenizer，右填充和 truncation 已烘入 |
| `gme_visual_<precision>.xml` | `pixel_values`、`image_grid_thw` | visual token embeddings | Qwen2-VL visual encoder |
| `gme_language_<precision>.xml` | `input_ids`、`attention_mask`、`image_embeds` | `embedding[B,1536]` | token embedding、视觉注入、decoder、pooling、normalize |

拆分 visual/language 有三个原因：

1. 避免整图转换时的内存峰值；
2. 纯文本不需要运行视觉塔；
3. visual 和 language 可以独立做 INT8/INT4 和混合精度组合。

所有学习参数都来自 GME checkpoint。项目没有新增 projection head；所谓 GME head 是无参数的最后有效 token pooling 和 L2 normalize。

## 4. Batch 语义和 visual 逐图执行

### 4.1 任意 batch 数量

运行时支持任意数量的 text、image 和 fused 样本，不需要针对不同 batch size 重新导出：

```python
encoder.get_text_embeddings(texts, batch_size=32)
encoder.get_fused_embeddings(texts, images, batch_size=8)
```

Python 和 C++ 都会保持输入顺序。`batch_size` 是分块上限，不是模型导出的固定 batch 维度。

### 4.2 language 的两种 batch mode

| 模式 | 执行方式 | 数值性质 | 默认设备 |
| --- | --- | --- | --- |
| `per_sample` | 每条样本一个真实长度的 infer request | batch 与逐条结果 bit-exact | CPU |
| `padded` | 一次 `[B,S]` padded language 调用 | 与官方 batch 行为一致，但 ragged 输入会有微小漂移 | GPU |
| `auto` | 按设备字符串选择 | CPU 选 per-sample，GPU 选 padded | CPU/GPU |

`padded` 的漂移不是 bug：一行 decoder 看到的 padding 数量会随同批次最长样本变化。`per_sample` 会消除这项影响，但 GPU 上多个 request 的提交开销可能更高。

### 4.3 为什么 visual 必须逐图

Qwen2-VL 的 visual forward 将多个图片的 patch 拼成一条序列，通过 `cu_seqlens` 和 block-diagonal attention 保持图片间隔离。数学上逐图运行再拼接是正确的，但导出时有两个限制：

1. `rot_pos_emb` 和视觉 attention mask 在 Transformers 中包含按 `grid_thw` 行数执行的 Python 循环；
2. B=1 trace 得到的 IR 表面上有动态 `[?,3]`，但实际 body 可能拒绝第二个 `image_grid_thw` 行；
3. 即便单独 trace B=2，未覆盖的图片分辨率组合也可能静默输出错误结果。

因此 shipped runtime 采用：

```text
pixel_values[sum(P_i),1176] + image_grid_thw[B,3]
  -> 按 image_grid_thw 切出每张图片的 patch
  -> visual IR 每张图片运行一次
  -> concat visual embeddings[sum(K_i),1536]
  -> language IR 注入 image-pad 位置
```

这里 `P_i` 是第 `i` 张图片的 raw patch 行数，`K_i=P_i/4` 是合并后的视觉 token 数。这样一份 B=1 visual IR 就能支持任意图片数量和任意分辨率，且避免一次执行 `[sum(P_i),sum(P_i)]` 的大 block-diagonal attention。

### 4.4 官方 API 对照

`07_validate_batch.py` 覆盖了以下行为：

| 官方接口/行为 | Python OpenVINO |
| --- | --- |
| `embed(texts, images, is_query, instruction)` | 同名同义的单批原语 |
| `get_text_embeddings` / `get_image_embeddings` / `get_fused_embeddings` | 同名接口 |
| `encode` / `encode_queries` / `encode_corpus` | sentence-transformers 风格 alias |
| `batch_size=32` | 默认值与官方一致 |
| `batch_size=None` | 额外支持，表示不分块 |
| `images=[None]*N` | 作为 text-only 路径处理 |
| texts 比 images 长 | 按配对数量截断，保持官方行为 |
| 超过 `max_length` | Python/C++ tokenizer 截断一致 |
| mixed `None` 图片 | 清晰抛出 `ValueError` |

## 5. 图片处理和视觉 token 数

GME/Qwen2-VL 不把所有图片固定 resize 到 `448x448`，而是通过 `smart_resize` 保持纵横比，并把高宽调整为 28 的倍数：

$$
28 = 14\ \text{patch size} \times 2\ \text{spatial merge size}
$$

对单张图片，设 processor 之后的高宽为 $H\times W$：

| 阶段 | 结果 |
| --- | --- |
| GME resize | 高宽为 28 的倍数，像素范围由 GME 基础配置约束 |
| Qwen2-VL processor | GME 默认视觉 token 范围为 256 到 1280 |
| normalize | RGB、`float32`、CLIP mean/std |
| patchify | 每行 `3*2*14*14=1176` 个 float |
| grid | `image_grid_thw=[1,H/14,W/14]` |
| visual 输出 | `K=(H/14)*(W/14)/4=H*W/784` 个 1536 维 token |

例如 `448x448`：

$$
P=32\times32=1024,\qquad K=1024/4=256
$$

本目录的 `gme_logo.png` 是超宽横幅。它经过 resize 后约为 `5152x588`，processor grid 为 `[1,24,210]`，有 5040 个 raw patch 行和 1260 个 visual token，接近默认上限 1280，因此它是最坏情况 workload，而不是普通图片的代表。

C++ 和 Python 的边界回归还覆盖了：

- EXIF Orientation=6；
- 70 像素 half-factor 的 round-half-to-even；
- 小于 28 像素的图片；
- 超过 200:1 的图片长宽比。

这些情况下 Python 和 C++ 的几何处理已经对齐；仍然存在的 image/fused 差异主要来自 OpenCV 与 Pillow 的 resize 重采样实现不同。

## 6. 为什么不能直接用 OpenVINO GenAI 加载

这里要区分三层：

1. **OpenVINO Runtime**：可以直接运行本项目导出的 IR，但需要应用层编排 prompt、图像处理、visual IR 和 language IR；
2. **OpenVINO GenAI**：提供标准生成模型和标准文本 embedding pipeline，不会自动理解 GME 的自定义 forward contract；
3. **Optimum Intel CLI**：是转换工具，可以转换受支持的生成模型，但不会自动把 GME 转成 embedding contract。

### 6.1 `TextEmbeddingPipeline` 不兼容

`TextEmbeddingPipeline` 假设输入是标准文本 embedding 模型，通常会自行寻找 token 轴并做 pooling/normalize。GME language IR 的结构不同：

- 有三个输入，其中一个是 `image_embeds`；
- 输出已经是最终的 `[B,1536]` embedding；
- pooling 和 normalize 已经在 IR 内完成；
- image/fused 还需要 visual IR、image-pad 展开和视觉 token 注入。

对当前包直接构造 pipeline 时，它首先会寻找标准文件名，例如 `openvino_model.xml`；把 `gme_language_int8.xml` 改名后再次尝试，仍会在 `is_axis_valid(axis, r)` 处失败。这说明问题不是文件名，而是 pipeline 对模型结构的假设不成立。

### 6.2 `VLMPipeline`/`LLMPipeline` 不兼容

`VLMPipeline` 和 `LLMPipeline` 面向自回归生成：需要 logits、KV cache、状态管理和 generation loop。当前 GME IR 是无状态 embedding 图，只有 `embedding` 输出，没有 logits 或生成状态，因此不能直接加载。

如果未来需要 GenAI 风格 API，需要另外实现一个 GME 专用 pipeline，至少处理：

- GME prompt 模板；
- HF 或等价 tokenizer；
- Qwen2-VL image-pad 展开；
- visual IR 与 language IR 编排；
- 已完成 pooling/normalize 的最终向量输出。

这不是现有 `TextEmbeddingPipeline` 的配置项。

### 6.3 Optimum CLI 能做什么

Optimum CLI 可以把相同 checkpoint 转换为标准 Qwen2-VL 生成模型。实际导出结果是：

| 文件 | 输出 |
| --- | --- |
| `openvino_language_model.xml` | `logits[?,?,151936]`，带 KV-cache 的生成图 |
| `openvino_text_embeddings_model.xml` | token embedding 序列 |
| `openvino_vision_embeddings_model.xml` | visual hidden states 和 merger |

它不能替代本项目的 embedding 导出，原因是：

- GME 需要的是 lm_head 之前的 `last_hidden_state` 最后有效 token；
- Optimum 生成图给出的是 vocabulary logits，1536 到 151936 的 lm_head 投影不可逆，不能从 logits 恢复 GME hidden state；
- 生成图有 KV-cache 和 generation loop，而 GME 需要无状态的一次性 embedding forward。

因此 `02_export.py` 的自定义 forward 仍然是获取 GME embedding 的必要步骤。

## 7. Python/C++ API 和部署选项

### 7.1 Python API

```python
from gme_openvino import GmeOpenVINO

enc = GmeOpenVINO(
    "/path/to/gme-qwen2-vl-2b-instruct-openvino",
    device="GPU",
    precision="int8_causal_vopt",
    cache_dir="/var/cache/gme_ov",
)

enc.get_text_embeddings(texts)
enc.get_text_embeddings(queries, instruction="Find an image that matches the given text.")
enc.get_image_embeddings(paths_or_pil_or_urls)
enc.get_fused_embeddings(texts, images)
enc.embed(texts=texts, images=images)
enc.encode(sentences)
enc.encode_queries(queries)
enc.encode_corpus(rows)
```

`streams="auto"` 适合 CPU 批吞吐；GPU 通常使用 `batch_mode="padded"`。`min_image_tokens`、`max_image_tokens` 只改变 processor 图像范围，不需要重新导出。Python language IR 是动态序列维度，`max_length` 可以根据应用需要收缩或放大。

### 7.2 C++ API 和 CLI

```cpp
#include "gme_qwen2vl.hpp"

gme::GmeQwen2VL model(
    "/path/to/gme-qwen2-vl-2b-instruct-openvino", "GPU", "int8");

auto e_text = model.encode_text({"What kind of car is this?"});
auto e_query = model.encode_text(
    {"What kind of car is this?"},
    "Find an image that matches the given text.", true);
auto e_image = model.encode_image({"/path/to/image.png"});
auto e_fused = model.encode_fused(
    {"GME logo"}, {"/path/to/image.png"});
```

C++ 输出是行主序的 float32，`rows*dim` 个元素，`dim=1536`。CLI 的 `--cache-dir` 可以缓存 compiled model；GPU 短生命周期 CLI 应尽量启用缓存，服务端则应复用长期存活的模型实例。

C++ 的 `--min-image-tokens`/`--max-image-tokens` 可以调整图片 token 上下界。`--max-length` 只能收缩导出时烘入 tokenizer IR 的上限；要放大上限，需要用更大的值重新执行 `03_finalize.py`，再重新生成 tokenizer。

## 8. 精度实测

以下数据来自本机 Core Ultra 7 356H、Linux、OpenVINO 2026.2 环境，比较对象是官方 PyTorch FP32 golden；这些是历史精度矩阵。当前第九轮 final precision 组合、baseline 对比和 document embedding 质量结果见 [PERFORMANCE_REPORT.md](PERFORMANCE_REPORT.md)、[第九轮报告](gme-qwen2-vl-int4-hotspot-optimization/docs/16_round9_final_rebenchmark.md) 与 [document_embedding_eval_int8_int4_gpu/REPORT.md](document_embedding_eval_int8_int4_gpu/REPORT.md)。

### 8.1 Python OpenVINO

| 设备 | precision | text | query | image | fused |
| --- | --- | ---: | ---: | ---: | ---: |
| CPU | fp32 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| CPU | int8 | 0.998476 | 0.998361 | 0.992512 | 0.997980 |
| CPU | int4 | 0.908822 | 0.912175 | 0.945496 | 0.962353 |
| CPU | int4_gs64 | 0.921590 | 0.949724 | 0.955428 | 0.967409 |
| CPU | int4_visint8 | 0.908822 | 0.912175 | 0.961537 | 0.957409 |
| GPU | fp32 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| GPU | int8 | 0.997518 | 0.998764 | 0.989255 | 0.995806 |
| GPU | int4 | 0.886320 | 0.911044 | 0.936908 | 0.955260 |
| GPU | int4_gs64 | 0.918643 | 0.950327 | 0.944975 | 0.963901 |
| GPU | int4_visint8 | 0.886320 | 0.911044 | 0.947935 | 0.968358 |

FP32 四种模式均为 `1.0`。上表是历史精度矩阵；第九轮当前 GPU golden cosine
结果如下，完整原始数据见
[`artifacts/openvino_validation_round9_latest.json`](artifacts/openvino_validation_round9_latest.json)：

| precision | text min | query | image | fused |
| --- | ---: | ---: | ---: | ---: |
| `fp32` | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| `int8_causal_vopt` | 0.997899 | 0.998784 | 0.987115 | 0.996531 |
| `int4_causal_vopt` | 0.941730 | 0.955173 | 0.943164 | 0.955472 |
| `int4_gs64_causal_vopt` | 0.944205 | 0.957576 | 0.943657 | 0.958768 |

相对于最初 public baseline 的完整 INT8/INT4 latency reduction 和 speedup 见
[`baseline_vs_round9_comparison.json`](gme-qwen2-vl-int4-hotspot-optimization/results/r9_latest/baseline_vs_round9_comparison.json)。
INT8 是最稳定的低比特方案；INT4 的 text/query 质量下降明显，`int4_gs64` 相对较好。

### 8.2 C++

| 设备 | precision | text | query | image | fused |
| --- | --- | ---: | ---: | ---: | ---: |
| CPU | fp32 | 1.000000 | 1.000000 | 0.975047 | 0.988341 |
| CPU | int8 | 0.998476 | 0.998361 | 0.971103 | 0.985380 |
| CPU | int4 | 0.908822 | 0.912175 | 0.943903 | 0.957391 |
| GPU | fp32 | 1.000000 | 1.000000 | 0.975043 | 0.988341 |
| GPU | int8 | 0.997518 | 0.998764 | 0.967116 | 0.981741 |
| GPU | int4 | 0.886320 | 0.911044 | 0.930794 | 0.948113 |

C++ text/query 与 Python 共用 tokenizer/language IR，因此接近精确；image/fused 的主要剩余差异来自 OpenCV/Pillow resize。

## 9. 延迟实测

### 9.1 固定 workload

固定 workload 为 batch size 1、短文本和仓库中的超宽 `gme_logo.png`。文本模板展开后约 27/30 tokens；图片路径约 1260 visual tokens，接近上限。

**文本/query，warm，ms/次：**

| 设备 | precision | Python text | C++ text | C++ query |
| --- | --- | ---: | ---: | ---: |
| CPU | fp32 | 223.5 | 224.2 | 235.2 |
| CPU | int8 | 78.3 | 78.0 | 85.0 |
| CPU | int4 | 87.9 | 86.9 | 93.9 |
| GPU | fp32 | 174.0 | 172.7 | 168.7 |
| GPU | int8 | 52.9 | 40.8 | 41.4 |
| GPU | int4 | 30.5 | 29.4 | 29.8 |

**图片/fused，warm，ms/次：**

| 实现 | 设备 | precision | image | fused |
| --- | --- | --- | ---: | ---: |
| Python | CPU | fp32 | 35610 | 36056 |
| Python | CPU | int8 | 23394 | 23499 |
| Python | CPU | int4 | 24992 | 24828 |
| C++ | CPU | fp32 | 36552 | 36356 |
| C++ | CPU | int8 | 23441 | 23181 |
| C++ | CPU | int4 | 24956 | 24812 |
| Python | GPU | fp32 | 22325 | 22299 |
| Python | GPU | int8 | 5172 | 5255 |
| Python | GPU | int4 | 5069 | 5165 |
| C++ | GPU | fp32 | 19944 | 19902 |
| C++ | GPU | int8 | 5013 | 5106 |
| C++ | GPU | int4 | 4983 | 5078 |

`gme_logo.png` 的结果是最坏情况基线，不能代表普通图片。`448x448`、256 visual tokens 的 GPU/INT8 image 量级约为 `0.55 s`，而 logo 约 `5.1 s`，相差约 9 倍。

### 9.2 指定 text/query 长度

`06_benchmark.py --sequence-lengths` 的目标是 GME prompt 展开后的有效 token 数。历史 `repeat=5` 结果如下，单位是 batch size 1 的 ms/call，格式为 text/query：

| 实现 | 设备 | precision | 128 tokens | 256 tokens |
| --- | --- | --- | ---: | ---: |
| Python | CPU | fp32 | 795.91 / 797.59 | 1567.99 / 1572.47 |
| Python | CPU | int8 | 327.53 / 327.57 | 685.23 / 688.04 |
| Python | CPU | int4 | 372.39 / 372.82 | 783.39 / 773.17 |
| C++ | CPU | fp32 | 797.37 / 803.70 | 1557.37 / 1556.04 |
| C++ | CPU | int8 | 432.37 / 419.53 | 920.18 / 667.50 |
| C++ | CPU | int4 | 370.79 / 370.06 | 757.32 / 757.78 |
| Python | GPU | fp32 | 236.43 / 355.75 | 332.90 / 329.58 |
| Python | GPU | int8 | 53.33 / 52.96 | 76.42 / 74.92 |
| Python | GPU | int4 | 43.09 / 43.12 | 72.20 / 71.97 |
| C++ | GPU | fp32 | 181.55 / 181.31 | 327.39 / 329.73 |
| C++ | GPU | int8 | 53.02 / 53.08 | 73.42 / 73.56 |
| C++ | GPU | int4 | 42.75 / 43.24 | 70.87 / 70.66 |

短采样的 CPU/C++ INT8 数据有明显噪声；用于选型或回归应使用 `--repeat 20`，并固定设备空闲、电源策略和驱动版本。

### 9.3 Batch 正确性和吞吐

`07_validate_batch.py` 对 `$N\in\{1,2,3,5\}$` 的 ragged text/image/fused 输入做 batch 与逐条比较：

- Python `per_sample` 的 text/image/fused 全部 bit-exact；
- `batch_size=2` 分块不改变结果；
- C++ fused batch 与逐条、`--batch-size 0/1/2` 均 bit-exact；
- Python 官方 API alias 和 `images=[None]*N` 行为通过；
- 超过 `max_length` 的 Python/C++ text 截断一致；
- 同 token length 的不同文本、同 visual grid 的不同图片不会发生行碰撞；
- FP32 Python batch 对官方 PyTorch 的 text/image/fused cosine 均在 `0.99999` 以上；
- C++ text 与 Python bit-exact，fused 的约 `0.998894` 差异来自图像 resize。

吞吐方向：

| 设备 | 推荐方式 | 实测结果 |
| --- | --- | --- |
| CPU historical `int8_vnomask` | `per_sample + streams="auto"` | 历史 text×8 约 `1.81x`，image×4 约 `1.76x` |
| CPU historical `int8_vnomask` | padded | 历史 N=1/4/8 归一化吞吐约 `1.00x/1.00x/0.99x` |
| GPU `int8_causal_vopt` | padded batch | round-9 text=128，N=8 为 `31.23 ms/embedding`；N=4 为 `30.65` |
| GPU `int4_causal_vopt` | padded batch | round-9 text=128，N=8 为 `26.46 ms/embedding`；N=4 为 `27.79` |
| GPU `int4_gs64_causal_vopt` | padded batch | round-9 text=128，N=8 为 `28.28 ms/embedding`；N=4 为 `33.04` |

因此 `batch_mode="auto"` 在 CPU 选择 per-sample，在 GPU 选择 padded。完整的 `N={1,2,4,8,16,32}` 矩阵由 `08_benchmark_matrix.py --suite batch` 生成，结果见 [PERFORMANCE_REPORT.md](PERFORMANCE_REPORT.md)。

## 10. 量化和部署建议

### 10.1 量化结果

模型包中包含：

| precision | language | visual | 说明 |
| --- | --- | --- | --- |
| `fp32` | FP32 | FP32 | 参考精度 |
| `int8_causal_vopt` | INT8、causal | INT8，排除 rotary，mask-free/headfirst/rank3 visual | 当前默认部署候选 |
| `int4_causal_vopt` | INT4 gs128、causal、data-aware | INT4 gs128，排除 rotary，mask-free/headfirst/rank3 visual | 当前低延迟推荐 |
| `int4_gs64_causal_vopt` | INT4 gs64、causal、data-aware | INT4 gs64，排除 rotary，mask-free/headfirst/rank3 visual | 当前精度优先 INT4 |
| `int8` / `int4` | 历史普通语言/视觉组合 | masked visual | 历史对照，仍保留用于复现 |
| `int4_gs64` | INT4 gs64，data-aware | INT4 | 文本质量恢复较好 |
| `int4_visint8` | INT4，data-aware | INT8 | 视觉塔更保守 |

INT4 language 使用 calibration、AWQ 和 scale estimation；校准文本跨长度、语言和领域，并且不使用 golden evaluation texts。视觉塔不需要同样的 data-aware 处理，但必须排除 `rotary_pos_emb` 相关常量，否则视觉输出会崩溃到约 `0.32` cosine。

记录中的 INT4 改进如下：

| 设备 | precision | text | query | image | fused |
| --- | --- | --- | --- | --- | --- |
| CPU | int4 | 0.908822 -> 0.953891 | 0.912175 -> 0.956376 | 0.945496 -> 0.952892 | 0.962353 -> 0.968193 |
| CPU | int4_gs64 | 0.921590 -> 0.967406 | 0.949724 -> 0.956181 | 0.955428 -> 0.957057 | 0.967409 -> 0.973058 |
| GPU | int4 | 0.886320 -> 0.944538 | 0.911044 -> 0.955146 | 0.936908 -> 0.941821 | 0.955260 -> 0.954111 |
| GPU | int4_gs64 | 0.918643 -> 0.949893 | 0.950327 -> 0.957442 | 0.944975 -> 0.943357 | 0.963901 -> 0.959970 |

### 10.2 推荐配置

```text
GPU:
  precision = int8_causal_vopt
  batch_mode = padded
  streams = 1
  text micro-batch = 8~16
  image micro-batch = 4~8
  low-latency text precision = int4_causal_vopt

CPU:
  precision = int8_causal_vopt
  batch_mode = per_sample
  streams = auto
  text micro-batch = 4~8
  image micro-batch = 4
```

INT4 只有在业务数据证明召回质量足够时才建议启用。所有 latency 数值都应在目标图片 token 分布、目标文本长度和目标并发下重新测量。

## 11. 已修复的静默正确性问题

这些修复是当前实现能够交付的前提：

1. Python request pool 不再使用会返回 stale output 的 `AsyncInferQueue`；改为显式 `start_async`/`wait` 并复制输出；
2. OpenVINO request 输出 buffer 在下一次推理前显式复制，避免数组别名；
3. C++ 图像读取使用 `IMREAD_IGNORE_ORIENTATION`，与 Pillow 不自动旋转 EXIF 的行为一致；
4. C++ resize 取整改为 half-to-even，和 Python `round()` 一致；
5. 极小图片和超宽图片按官方逻辑放大或 clamp，而不是无故拒绝；
6. C++ tokenizer 通过 `max_length` 烘入 truncation，修复 Python/C++ 长文本不一致；
7. `03_finalize.py` 按完整 NNCF 配置指纹决定是否重新量化，不会因为旧 `.xml` 存在而静默复用过期配方；
8. Python 图片批量加载使用线程池保留输入顺序，URL 列表不会被逐张网络等待拖慢。

## 12. 方案边界

OpenVINO + 自研 GME 封装是当前 Intel CPU/GPU 的首选方案，因为它能够保留 GME 的自定义 embedding contract，同时支持 tokenizer、视觉塔、语言塔和 NNCF 量化。普通生成模型转换工具或通用 GenAI pipeline 不能替代 `02_export.py` 的自定义 embedding 导出。

上线前仍需：

1. 在真实 query/corpus/image 数据上比较 INT8/INT4 的 Recall@K、nDCG 和 Top-K；
2. 按真实文本长度和图片视觉 token 分布设置 micro-batch；
3. 测试 P50/P95/P99，而不是只使用单次平均值；
4. 固定 GPU 驱动、电源策略和 OpenVINO 版本；
5. 明确 C++ 图像 resize 差异是否满足业务验收阈值；
6. 在目标设备上单独验证，不把 CPU/GPU 结果外推到 NPU。
