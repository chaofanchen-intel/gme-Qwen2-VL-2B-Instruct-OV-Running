# GME-Qwen2-VL-2B-Instruct 在 Intel CPU/GPU 上的 OpenVINO 运行时

用 **OpenVINO** 在 Intel CPU 和 Intel 集显（本机 **Core Ultra 7 356H**，Arc/Xe iGPU）上完整运行
`Alibaba-NLP/gme-Qwen2-VL-2B-Instruct`，覆盖 **text / query(带 instruction) / image / fused(图文)** 四种 embedding。
**主实现是 C++**（`cpp/`，加载 `benchmark/openvino/install`），**Python 作为导出器与参考级精度基准**。截至本文档对应的测试结果，
Python OpenVINO 在四种模式的 FP32 结果与官方参考数值等价；C++ 在 text/query 数值等价，但 image/fused 仍有已量化的图像预处理差异。

---

## 1. 结论（TL;DR）

- ✅ 全部计算在 OpenVINO Runtime：**tokenizer / 视觉编码 / 语言解码 + 池化 + 归一化** 三个 IR。C++ 只做 IR 之外的
  预处理（GME prompt 模板、`<|image_pad|>` 展开、Qwen2-VL 图像预处理）。**推理期不依赖 Python**。
- ✅ **Python OpenVINO 与原始权重一致**：FP32 三个 IR 的组合对官方 PyTorch（从
  `benchmark/models/gme-Qwen2-VL-2B-Instruct` 原始 safetensors 生成的黄金向量）在四种模式均为
  `cosine = 1.000000`（见 §6.1）。导出使用的是 GME 微调后的 checkpoint，而不是未微调的
  `Qwen2-VL-2B-Instruct`。
- ✅ **C++ text/query 逐 token 且数值精确**：FP32=1.0，INT8≈0.998。C++ image/fused 的 FP32 分别为
  `0.975` / `0.988`，当前不应称为与官方图像路径数值等价；差异与 OpenCV 和 Pillow resize 的实现差别一致（见 §6.3）。
- ✅ 量化：**INT8 是当前的部署候选默认值**；**INT4** 体积最小但文本掉点明显。视觉塔 INT8/INT4 必须排除 rotary（见 §7）。
- ✅ **Batch 与官方等价且更强**：四种模式接受任意 $N$、任意图片分辨率组合，输入顺序保留，**无需为不同 $N$ 重新导出**；
  `batch_size` 默认 32（与官方同值）。官方 API 面（含 `embed`/`encode`/`encode_queries`/`encode_corpus`/`prompt_name`/
  `show_progress_bar`/`DataLoader`）逐项对齐，见 §2.0.2.1。视觉塔逐图调用不是妥协——为 $B$=2 导出的 IR 在
  12 种分辨率组合中有 6 组**静默给出错误结果**，逐图才是唯一通用且正确的实现（§2.0.2，实测）。
- ⚙️ 图像时延与图像 token 数强相关（注意力 ∝ patch²）：常见图（`448x448` = 256 token，也是下界）GPU-INT8 约 0.55 s；
  本仓库的 `gme_logo` 是超宽横幅（1260 token，接近上限，5.17 s）→ 属最坏情况。**§6.4 只能当最坏情况与回归基线**，
  选型请用 `09_benchmark_matrix.py` + `PERFORMANCE_REPORT.md`。
- ❓ **Optimum CLI**：能把该 checkpoint 转成 OpenVINO（image-text-to-text），但产物是**生成模型**（language IR 输出
  `logits`+KV-cache），**不能**用作 GME embedding；要 embedding 必须用返回 hidden-state 的自定义导出（= `02_export.py`）。已实测（见 §2.2）。
- ❓ **OpenVINO 能否直接跑（GenAI）**：**不能**直接用 `TextEmbeddingPipeline`/`VLMPipeline` 跑 GME——前者假设标准文本编码器并自行 pooling
  （实测在 `is_axis_valid` 处失败），后者要有状态生成图。OpenVINO **Runtime** 则通过本项目自定义 IR 已经“直接在跑”（见 §2.2）。

---

## 2. 架构

官方 `gme_inference.py` 的计算路径（GME 没有额外可训练 head，“head” 就是 last-token 池化 + L2 归一化）：

```
text : GME prompt 模板 -> tokenizer(右填充) -> token embedding -> GME Qwen2-VL decoder
       -> 取最后有效 token(attention_mask.sum(-1)-1) -> L2 归一化 -> 1536 维
image/fused: 图像 resize + Qwen2-VL 预处理 -> 视觉编码器 -> 视觉 token 注入到 <|image_pad|>
       -> 同一个 GME decoder -> 最后有效 token -> L2 归一化
```

### 2.0 GME-Qwen2-VL-2B-Instruct 与 Qwen2-VL-2B-Instruct 的具体差异

GME 不是在未改动的 `Qwen2-VL-2B-Instruct` 后面简单接一个小的、可训练的 projection head。它以 Qwen2-VL-2B-Instruct 为初始化来源，针对多模态检索目标微调，并以自定义 `GmeQwen2VL` 类把输出语义由“生成下一个 token”改为“返回检索 embedding”。

本地 checkpoint 的静态检查显示两者都拥有 729 个同名权重条目，语言模型主干同为 1536 hidden size、28 层 decoder、12 个 attention heads、2 个 KV heads，视觉端同为 Qwen2-VL ViT/merger 结构。这说明 GME 的基础网络拓扑没有换成另一种模型；关键差别是**同一类网络的微调权重**和**不同的 forward/output contract**。文件被分到不同 safetensors shard 不能证明每个张量是否相同，但 GME 的模型卡将它标为 Qwen2-VL-2B-Instruct 的 embedding fine-tune，不能用原始 Qwen2-VL 权重代替。

| 方面 | 标准 Qwen2-VL-2B-Instruct | GME-Qwen2-VL-2B-Instruct |
| --- | --- | --- |
| 主要任务 | 视觉语言**生成** | text/image/fused **检索 embedding** |
| 顶层 Transformers 类 | `Qwen2VLForConditionalGeneration` | 自定义 `GmeQwen2VL` / `GmeQwen2VLConfig` |
| 语言输出消费方式 | 经生成 wrapper 得到 vocabulary logits，用于 next-token decoding | 直接使用 `Qwen2VLModel.last_hidden_state` |
| 最终输出 | 每个位置的 logits，概念形状 `[B,S,151936]` | 每个样本一个已归一化向量 `[B,1536]` |
| 池化/归一化 | 生成路径不需要 | `hidden[batch, attention_mask.sum(-1)-1]`，再 L2 normalize |
| 视觉 token 数 | 标准 processor 的默认策略 | custom config 固定 GME 约束：`min_image_tokens=256`、`max_image_tokens=1280`；单图输入 |
| 文本模板 | generation/chat template | GME system/user/assistant 模板，末尾显式为 `<|endoftext|>`；默认 instruction 为 `You are a helpful assistant.` |

关于 `lm_head`：两个本地 safetensors index 都没有独立的 `lm_head.*` 条目，因为 Qwen 的 output embedding 与 input embedding tie；但标准 `Qwen2VLForConditionalGeneration` 的生成 forward 仍会计算/暴露 logits。GME custom class 则只实例化 `visual` 和 `Qwen2VLModel`，源码中的 `lm_head` 声明被注释掉，且它的 `forward()` 从 decoder hidden states 直接做 pooling。因此，“没有额外 projection head”不等于“可以把 GME 当作普通生成模型直接运行”；两者最重要的区别是前向图的输出和后处理语义。

### 2.0.1 官方四种模式实际执行的分支

官方公开 API 最终都走 `get_fused_embeddings()` -> `embed()`，但是否运行 visual 由 `images` 是否为 `None` 决定，不是让视觉塔接收空图片：

| 调用模式 | `embed()` 内的 `images` | processor 输出 | `forward()` 是否调用 `self.base.visual(...)` | 语言 decoder 输入 |
| --- | --- | --- | --- | --- |
| text | `None` | `input_ids`, `attention_mask`；**没有** `pixel_values` / `image_grid_thw` | **否** | 仅文本 token embedding |
| query | `None` | 同 text；instruction 改写 system prompt | **否** | 仅文本 token embedding |
| image | 实际图片，文本为 `None` | token、mask、`pixel_values`、`image_grid_thw` | **是** | 视觉 token 注入 image-pad 位置 |
| fused | 实际图片和实际文本 | 同 image | **是** | 文本 token 和视觉 token 的融合序列 |

源码依据是官方 `GmeQwen2VL.forward()` 的条件分支：只有 `pixel_values is not None` 才计算 `image_embeds = self.visual(...)` 并赋给 `input_ids == image_token_id` 的位置。text/query 模式的 processor 也不会生成 `pixel_values`，所以官方不会调用 visual，更不存在“传空值仍跑一遍 visual”。

导出为三个可组合 OpenVINO IR（`ov_models/gme-qwen2-vl-2b-instruct-openvino/`）：

| IR | 输入 → 输出 | 说明 |
| --- | --- | --- |
| `openvino_tokenizer.xml` | string → `input_ids`,`attention_mask` | openvino-tokenizers；右填充；与 HF **逐 token 一致** |
| `gme_visual_<prec>.xml` | `pixel_values`,`image_grid_thw` → 视觉 token 向量 | Qwen2-VL ViT |
| `gme_language_<prec>.xml` | `input_ids`,`attention_mask`,`image_embeds` → `embedding[B,1536]` | token embedding + 视觉注入(masked_scatter) + decoder + last-token 池化 + L2 归一化 |

- 拆 visual / language 两个 IR 是为了避免整图转换时的内存峰值，也与官方 no-image 分支一致。纯文本只调用 `gme_language`；为保持 language IR 固定的三个输入接口，runtime 给 `image_embeds` 传空 `[0,1536]` 张量。它不是“空图片”：不会调用 `gme_visual`，而且 prompt 中没有 image-pad token，因此 `masked_scatter` 没有任何要替换的位置。
- **视觉注入关键点**：HF processor 会按图像网格把单个 `<|image_pad|>` **展开**成 K 个（K = `grid_t*grid_h*grid_w/merge²` = 视觉合并 token 数）。C++ 端在分词前必须同样展开，否则 image_pad 数与视觉输出行数对不上，图像向量会崩（FP32 也会掉到 ~0.29）。

### 2.0.2 Batch 支持、实际并行度与 API 语义

四种模式都接受任意数量的样本并返回 `[B,1536]`，其中 $B$ 是请求内的样本数，**不需要为不同 $B$ 重新导出模型**。

关键设计：一个 GME batch 里的样本彼此完全独立，因此两端运行时都把它派发为 **$B$ 个单样本 infer request**，而不是一次 `[B,S]` 的 padded 大推理。

| 模式 | Python 输入 | C++ 输入 | 执行方式 | `batch_size` 的作用 |
| --- | --- | --- | --- | --- |
| text | `list[str]` | `vector<string>` | 一个 tokenizer call，然后每条一个 language request（按各自真实长度，不含 padding） | 分块上限（默认 **32**，与官方一致）；`None`/`0` 表示不分块。 |
| query | 同 text | 同 text | 同 text，仅 prompt 中的 instruction 不同 | 同 text。 |
| image | `list[path/URL/PIL.Image]` 或官方风格 `DataLoader` | `vector<local_path>` | 每张图一个 visual request，每个样本一个 language request | 同 text；输出顺序与输入一致。 |
| fused | 等长的 text/image 列表 | 等长的 `vector<string>` | 同 image；文本与图片按下标配对 | 同 text；输出顺序与输入一致。 |

这带来一个比 padded batch 更强的性质：**batch 结果与逐条推理逐位相同（bit-identical）**。padded batch 做不到这一点——一行的输出会随同批次里最长样本的长度而轻微漂移。

两端的 `batch_size` 默认值都是官方 `get_fused_embeddings(batch_size=32)` 的 **32**。它只是分块上限：
在 `per_sample` 下**不可能**改变任何一行（`08_validate_batch.py` §F 对 `batch_size ∈ {1,2,5,32,None}` 断言逐位相同）；
在 `padded` 下它会改变“哪些样本一起 padding”，因而与官方一样带来微小漂移。

```python
encoder.get_text_embeddings(texts, batch_size=32)        # 默认值，可省略
encoder.get_fused_embeddings(texts, images, batch_size=8)
encoder.get_text_embeddings(texts, batch_size=None)      # 不分块（本项目扩展）
```

```bash
cpp/build/gme_embed --package $PKG --mode text --batch-size 8 "a" "b" "c"   # 0 = 不分块
```

#### `batch_mode`：哪种更快是设备的属性，不是实现的属性

language IR 支持两种消费 batch 的方式，`batch_mode="auto"`（默认）按设备实测结果自动选：

| | `per_sample`（CPU 默认） | `padded`（GPU 默认） |
| --- | --- | --- |
| 调用方式 | 每条一个 request，按真实长度，无 padding | 一次 `[B,S]` 调用，与官方一致 |
| batch vs 逐条 | **逐位相同** | 约 0.9999（随同批最长样本漂移） |
| CPU（16 核，$N$=4/8） | 基准；配合 `streams` 再快 **1.9x** | **1.00x / 0.99x**——padded 在 CPU 上不带来任何加速 |
| GPU（$N$=4，短文本） | 53–71 ms/emb | **22 ms/emb（快 2.4x）** |

CPU 上单个 request 已占满全部核心，padded 只是多算了 padding；GPU 上则是每次提交的固定开销占主导，合并成一次调用才划算，而且多 stream 在 GPU 上救不回来（实测 0.34–0.41x）。

**CPU 的吞吐来自 `streams`：**

```python
GmeOpenVINO(pkg, precision="int8", streams=1)       # 默认：单 stream 独占全核，单条延迟最低
GmeOpenVINO(pkg, precision="int8", streams="auto")  # 插件自选（16 核上为 4）：批量约 1.8x
GmeOpenVINO(pkg, device="GPU", batch_mode="auto")   # GPU 自动选 padded
```

C++ 对应 `--streams N`（`1` 为默认的 LATENCY，`0` 交给插件自选）与 `--batch-mode auto|per_sample|padded`。多 stream 的代价是单条样本约慢 1.3x、每多一个 stream 约 +0.2 GB；**`streams` 不改变输出，两种设置逐位相同**。

官方 GME 的 `get_fused_embeddings(..., batch_size=32)` 会让 processor 生成图片 batch。本项目现在采用相同的 Qwen2-VL batch 布局：对第 $i$ 张图片，processor 产生 $P_i$ 行 raw patch 和 $K_i=P_i/4$ 个 image token；一个 mixed-resolution batch 将 patch 行沿第 0 维拼接，并以 `image_grid_thw[B,3]` 保存边界：

```text
processor: pixel_values[ΣP_i,1176] + image_grid_thw[B,3]
          -> 对每个 grid_thw[i] 切出 pixel_values[P_i,1176]
          -> gme_visual（每张图一次） -> image_embeds[K_i,1536]
          -> concat image_embeds[ΣK_i,1536]
          -> gme_language(input_ids[B,S], attention_mask[B,S], image_embeds)
          -> embedding[B,1536]
```

每行 `input_ids[i]` 已含恰好 $K_i$ 个 `image_token_id`，因此 language IR 对 `image_embeds` 执行 `masked_scatter` 时仍保持官方的输入顺序和逐图片映射；无需将不同分辨率图片强行 resize 到同一尺寸，也不需要按 `grid_thw` 分桶。

#### 为什么 visual 逐图、而 language 可以真动态 batch

这是本项目被质疑最多的一点，下面的每条结论都有实测支撑，复现脚本是
`experiments/probe_visual_batch.py`（`--only q1/q2/q3/q4` 可分段跑）。

**1）`gme_visual` 的动态 batch 不是"能不能"的问题，而是"官方模型的视觉塔本来就没有 batch 维"。**
Qwen2-VL 把多张图的 patch 拼成一条长序列，图与图之间靠 `cu_seqlens` 形成**块对角注意力**彼此隔离；
每张图的 rotary 查表只依赖自己的 `grid_thw`（`rotary_pos_emb_full[pos_ids]` 的第 $i$ 行与表长无关）。
所以"逐图跑再拼接"与"一次喂多行 `image_grid_thw`"是**同一个数学函数**，没有近似。

**2）trace 会把图片数烘死，这是 OpenVINO 导出的限制，不是模型的限制。**
`transformers` 的 `Qwen2VisionTransformerPretrainedModel.rot_pos_emb` 用 `for t, h, w in grid_thw:` 在
**Python 层**按图片数循环（4.51.3 的 `VisionSdpaAttention` 里 `for i in range(1, len(cu_seqlens))` 构造块对角
mask 同理）。所以 `[?,3]` 只是名义上的动态维。实测按 $B$=1 trace 的 IR 喂 2 行：

```text
[CPU] Reshape node '__module.visual/prim::ListUnpack/Squeeze' ... the shape of
input data (2.3) conflicts with the reshape pattern
```

**3）关键发现：为 $B$=2 单独导出一个 visual IR 也解决不了问题——它在未 trace 过的分辨率组合上会静默给出错误结果。**
用一个刻意按"两张图、grid `[1,32,32]` 与 `[1,30,36]`"trace 出来的 `gme_visual_b2_int8.xml`，与逐图调用逐位比较
（12 种分辨率组合，`--only q2`）：

| 第一张图 grid | 第二张 32×32 | 第二张 40×40 | 第二张 48×48 |
| --- | --- | --- | --- |
| 32×32 | 逐位相同 | 逐位相同 | 逐位相同 |
| 40×40 | **静默错误** (3.2e+01) | **静默错误** (1.5e+01) | **静默错误** (2.8e+01) |
| 48×48 | **静默错误** (1.9e+01) | **静默错误** (1.6e+01) | **静默错误** (2.8e+01) |
| 64×64 | 逐位相同 | 逐位相同 | 逐位相同 |

12 组中 6 组结果错误，**且不抛异常**。也就是说"按 batch 大小重新导出"并不是一个可用的退路：
你需要为**每一种图片分辨率组合**各导出一次，这在真实输入上不可能。而逐图调用一次导出、任意 $N$、任意分辨率通用。

**4）逐图这条路本身是对的，而且与分辨率无关。** 用 fp32 的 $B$=1 visual IR 对 PyTorch 逐分辨率核对（`--only q3`）：

| 图片 grid | 视觉 token | cosine vs PyTorch |
| --- | ---: | ---: |
| 32×32 | 256 | 0.99999982 |
| 40×40 | 400 | 1.00000000 |
| 48×48 | 576 | 1.00000024 |
| 56×56 | 784 | 1.00000012 |
| 64×64 | 1024 | 0.99999982 |
| 70×70 | 1225 | 1.00000024 |
| 52×92（非正方，1288×728） | 1196 | 0.99999988 |
| 34×46（非正方，644×476） | 391 | 1.00000000 |

**5）逐图还更快。** 4.51.3 的批量视觉路径构造一个 `[1,S,S]` 块对角 mask，开销是 $O((\sum_i L_i)^2)$；
逐图是 $O(\sum_i L_i^2)$。实测两张 448×448：逐图 2599 ms，"真 batch" 3209 ms——**逐图快 1.23x**。

**6）官方有没有同样的限制？没有，但原因是它不导出静态图。** 官方 `gme_inference.py` 是 eager PyTorch，
Python 循环每次调用重跑，所以任意图片数都能跑。有意思的是官方那条批量路径**自己也不是逐位可复现的**：
2 图 forward 与 2 次单图 forward 的 max abs diff 为 `2.8e-03`（`--only q4`）——同一个函数，
但 mask 后的 softmax 在不同序列长度上累加顺序不同。换言之，逐图不仅不比官方批量路径差，
它与"批量 IR 正确时的结果"是**逐位相同**的，而官方批量路径与它自己的逐图结果都不逐位相同。

**7）`gme_language` 的动态 batch 是真的**，因为它是标准 `[B,S]` 张量运算，没有任何按 batch 的 Python 循环。
所以 `padded` 模式可用、并且是官方布局。但如 §6.6 所示，在 CPU 上它不带来加速。

**结论**：对外 API 与官方完全一致（任意 $N$、任意分辨率、输入顺序保留）；内部逐图是唯一正确且通用的实现方式，
不是为了绕过导出问题而做的妥协。`ov_models/` 下的 `gme_visual_b2_*` 是上述对照实验的产物，**不是部署产物**，
`gme_openvino_config.json` 的 `precisions` 里不包含它。

Python 图片输入支持本地路径、HTTP(S)、`file://`、base64 data URL、PIL image，以及官方 identity-collate 风格 `DataLoader` 产生的图片 batch。C++ API 的 image 输入是本地路径 `vector<string>`；其功能与 local-path 官方输入一致，但 C++ 不能直接接收 Python 的 PIL object 或 Python DataLoader。HTTP/base64 需要由调用方先下载/解码为本地文件后传入 C++。

#### 2.0.2.1 与官方 `GmeQwen2VL` 的 API 逐项对照

`08_validate_batch.py` §F 逐条断言下表的行为。左列是官方 `gme_inference.py` 的公开面。

| 官方成员 | 本项目 Python | 一致性 |
| --- | --- | --- |
| `embed(texts, images, is_query, instruction)` | 同名同义 | ✅ 单批原语，不分块、不显示进度条 |
| `get_text_embeddings(texts, **kw)` | 同名 | ✅ |
| `get_image_embeddings(images, **kw)` | 同名 | ✅ 接受 path/URL/base64/PIL/`DataLoader` |
| `get_fused_embeddings(texts, images, **kw)` | 同名 | ✅ 两者之一可为 `None` |
| `encode(sentences, prompt_name=None, **kw)` | 同名 | ✅ `prompt_name` 与官方一样被接受并忽略（GME 用 `is_query`/`instruction` 选 prompt） |
| `encode_queries(queries, **kw)` | 同名 | ✅ |
| `encode_corpus(corpus, **kw)` | 同名 | ✅ dict 与 list-of-dict 两种形态，`title` 以 `sep=' '` 拼接后 `strip()`，强制 `is_query=False` |
| `batch_size`（默认 32） | 同默认值 | ✅ 另允许 `None` = 不分块 |
| `show_progress_bar`（默认 True） | 同默认值 | ⚠️ 唯一差异：只有分块数 > 1 时才绘制。单块进度条不携带信息，官方打印它只是因为循环无条件执行 |
| `images` 全为 `None` = 纯文本批 | 同义 | ✅ |
| `images` 混合 `None` 与图片 | 抛 `ValueError` | ⚠️ 官方在这种输入上直接崩（`input_images` 被置 `None` 后又 `.append`）。支持的输入面相同，只是失败方式更清楚 |
| `texts` 比 `images` 长 | 按 `zip` 截断 | ✅ 官方靠这条实现末尾不满的 batch（`none_batch` 是整批 `None`） |
| `max_length=1800` 截断 | 两端一致 | ✅ 见下 |
| 极端尺寸图片（边 < 28px、长宽比 > 200:1） | 与官方同样 `max(factor,…)` 放大 / warning+clamp | ✅ 曾经两端都抛异常，见 §2.0.2.2 第 5 条 |
| `GmeQwen2VL(..., min_image_tokens=256, max_image_tokens=1280, max_length=1800)` | 构造函数同名参数 | ✅ 曾经只能改包配置文件；现两端构造函数都可覆盖（C++ 的 `max_length` 只能收缩，见 §4.3/§4.4/§4.5） |
| 返回值 | `np.ndarray[B,1536]` float32 | ⚠️ 官方返回 `torch.Tensor` float16；本项目全程 float32（精度更高，L2 归一化后可直接内积） |

**`max_length` 截断（曾经的 C++/Python 不一致，已修）**：Python 走 HF processor 的 `truncation=True, max_length=1800`，
而 C++ 的 tokenizer IR 原先是用 `convert_tokenizer(tok)` 转换的，**不做截断**——一段 3000 词的输入在 Python/官方是
1800 token，在 C++ 是 3021 token，两端向量完全不同，且 C++ 会为超长输入付出无上限的 decoder 时间。
现在 `03_finalize.py` 用 `convert_tokenizer(tok, max_length=1800)` 把截断烘进 IR（`max_length` 取自包配置，不是硬编码），
并对"短 + 超长"混合 batch 逐 token 校验与 HF 一致。实测 >1800 token 输入的 C++ vs Python 已是 **bit-exact**
（`08_validate_batch.py` §G）。注意截断发生在 image-pad 展开**之后**，所以超长文本 + 大图仍可能因截断切掉视觉位置而报错——
官方有同样的行为。

#### 2.0.2.2 已修的静默正确性问题

1. **request pool 的行重复（Python，影响 image 与 text batch）**：`_RequestPool` 原先用 `ov.AsyncInferQueue`，
   实测会把 job 0 的输出**重复写进 job 1**：两张 grid 相同（例如都落在 `[1,32,32]`）但内容不同的图片，batch 结果第 1 行
   等于第 0 行，`maxdiff = 6.6e+01`，**不抛异常**，而且是间歇性的（取决于该 compiled model 之前被怎么用过）。
   触发条件恰好是"多个 feed 形状相同"——两张 448×448、两条等长 query，正是最常见的情况。原有回归测试全部使用
   ragged 输入（形状互不相同），因此覆盖不到。现改为显式 `InferRequest` + `start_async`/`wait`（与 C++ 端一直采用的
   方式相同），并新增 `08_validate_batch.py` §I：**刻意非 ragged**、且额外断言 batch 内各行互不相同
   （否则"每行都返回第 0 行"也能通过与单样本串行的比较）。C++ 端从未使用 `AsyncInferQueue`，不受影响。
2. **读取 OpenVINO 输出时的别名**：request 会复用输出缓冲区，`np.asarray(result[port])` **不复制**，
   连续两次推理后先前的数组会指向新数据。本项目的输出读取一律改为显式复制；`experiments/probe_visual_batch.py`
   的注释里记录了这一点，因为它曾让一次对照实验得出相反结论。
3. **C++ 按 EXIF 自动旋转图片（影响 image 与 fused）**：`cv::imread(path, IMREAD_COLOR)` 会依据 JPEG 的
   EXIF orientation flag 旋转图片（OpenCV 头文件明写：*"the EXIF orientation will be taken into account …
   unless the flags IMREAD_IGNORE_ORIENTATION"*），而参考路径 `PIL.Image.open` + `convert("RGB")` **从不**旋转。
   于是任何 `Orientation != 1` 的 JPEG——**手机拍的照片绝大多数如此**——在 C++ 里被旋转 90/180/270 度，
   得到一个与 Python/官方完全不同的 embedding，`grid_h`/`grid_w` 互换连视觉 token 数都不同，**且不报错**。
   实测一张 Orientation=6 的 JPEG：C++ vs Python `cosine=0.917350`。现已改为
   `IMREAD_COLOR | IMREAD_IGNORE_ORIENTATION`。原有图像回归全部用 PIL 生成的无 EXIF PNG，所以覆盖不到。
4. **C++ 的取整不是 Python 的取整（影响 image 与 fused）**：`smart_resize` 里 Python 的 `round()` 是
   **round-half-to-even**，而 C++ 用的 `std::lround` 是 round-half-away-from-zero。两者只在某一边正好落在
   半个 factor 上时分叉，但这类尺寸并不罕见：`70 == 2.5x28` 官方取 `56`，C++ 取 `84`——中间 resize 尺寸不同，
   最终 embedding 就不同（实测 `cosine=0.900773`）。现改用 `std::nearbyint`（默认 FE_TONEAREST 即 half-to-even）。
5. **官方能编码的极端图片被两端拒绝**：官方 `smart_resize` 用 `max(factor, round_by_factor(...))` 把**短于一个
   28px patch 的边放大**，并在长宽比 > 200:1 时 **warning + clamp** 而不是失败。本项目此前
   Python 抛 `ValueError`、C++ 抛 `runtime_error`，于是 10x10 缩略图、1x2000 横幅这类**官方可以编码的输入被拒**。
   现已两端逐语句对齐官方（含 clamp 只缩短长边、因此仍是 28 的倍数）。仅保留一处官方没有的显式报错：
   某一边被 floor 成 0 时官方接着除以 `min(...)` 抛 `ZeroDivisionError`，这里给出可读信息——不拒绝任何官方能编码的输入。

上述 3/4/5 的回归是 `experiments/check_image_edge_cases.py`：第一部分把 `_resize_for_gme` 对官方
`smart_resize` 做 18 组尺寸扫描（纯算术，不加载模型），第二部分用真实编码器跑 EXIF / 半 factor / 超小 / 超宽
四类图片的 C++ vs Python 对照。

### 2.0.3 图片从 $m\times n$ 像素到 1536 维 embedding 的完整过程

设输入 RGB 图像的原始高宽为 $H_0=m$、$W_0=n$。GME/Qwen2-VL 的图片处理不是直接 resize 到固定 `448x448`；它保持接近原始纵横比，并将尺寸调整为 28 的倍数。28 来自：

$$
28 = \text{patch size } 14 \times \text{spatial merge size } 2
$$

下面各维度均描述单张图片；多张图片逐张重复同一流程。

| 阶段 | 操作与状态 | 张量/尺寸 |
| --- | --- | --- |
| 0. 解码 | 本地文件、HTTP(S)、`file://`、base64 data URL 或 PIL image 转为 RGB 8-bit 图像 | `[H_0,W_0,3]`，值域 `0..255` |
| 1. GME `fetch_image` resize | `smart_resize` 保持纵横比近似，$H_1,W_1$ 均为 28 的倍数；像素范围为 $[4\times28^2,16384\times28^2]$ | `[H_1,W_1,3]`；这一步主要规范极小/极大图片 |
| 2. Qwen2-VL processor resize | 再次 `smart_resize`，$H_2,W_2$ 为 28 的倍数；GME 配置范围为 $[256\times28^2,1280\times28^2]$ 像素，即 $[200704,1003520]$ | `[H_2,W_2,3]` |
| 3. normalize | RGB 转 `float32`，逐通道做 $x/255$，再减 CLIP mean 并除 CLIP std | `[H_2,W_2,3]` float32 |
| 4. patchify | 每个空间 patch 是 `14x14`，单图的 temporal patch size 为 2，因此同一图片帧复制两次；每个 raw patch row 展平为 $3\times2\times14\times14=1176$ | `pixel_values[P,1176]` |
| 5. 图像网格 | $g_t=1$，$g_h=H_2/14$，$g_w=W_2/14$；raw patch 行数 $P=g_tg_hg_w$ | `image_grid_thw=[1,g_h,g_w]`，`P=H_2W_2/196` |
| 6. `gme_visual` | Qwen2-VL ViT 编码 raw patch rows，并按 spatial merge $2\times2$ 合并 | `image_embeds[K,1536]` |
| 7. image token 展开 | processor 将 prompt 中一个 `<|image_pad|>` 展开为恰好 $K$ 个 `image_token_id`；每个视觉向量有一个语言序列位置 | $K=P/4=H_2W_2/784$，范围 `256..1280` |
| 8. `gme_language` 注入 | token lookup 得到 `[1,S,1536]`；`masked_scatter` 将 `image_embeds[K,1536]` 写入上述 $K$ 个位置 | 融合的 `inputs_embeds[1,S,1536]` |
| 9. decoder + pooling | 28 层 GME Qwen2-VL decoder 输出 `last_hidden_state[1,S,1536]`；取右填充前最后有效 token，索引为 `attention_mask.sum()-1` | `[1,1536]` |
| 10. normalize | 对最后 token hidden state 做 L2 normalize | 最终 `embedding[1,1536]` |

其中，`S` 是完整 language sequence 的长度：system/user/assistant 模板 token、文本 token、终止 token，以及 $K$ 个 image token 的总和。它不是单纯的用户文本长度。对 image-only，用户文本长度为零但 $S$ 仍包含模板和 $K$ 个视觉位置；对 fused，用户文本 token 追加在视觉 placeholder 后。

举例，`448x448` 图片在第二步后仍为 `448x448`：

$$
g_h=g_w=448/14=32,\quad P=32\times32=1024,\quad K=1024/4=256
$$

所以 visual IR 接收 `pixel_values[1024,1176]`、`image_grid_thw=[1,32,32]`，输出 `image_embeds[256,1536]`，prompt 中也必须有 256 个 image-pad token。对于本仓库的超宽 `gme_logo.png`，第一次 resize 后约为 `5152x588`，第二次 processor resize 后为 `2940x336`，因此：

$$
g_h=24,\quad g_w=210,\quad P=5040,\quad K=1260
$$

这解释了它远慢于常规图片：visual attention 和后续 decoder sequence 都需要处理约 1260 个视觉 token。

### 2.0.4 text、query、image、fused 的端到端流程对照

| 模式 | prompt / processor | visual IR | language IR 输入 | 最终 embedding |
| --- | --- | --- | --- | --- |
| text | 默认 instruction + 用户文本；仅 tokenizer | 不调用 | `input_ids[B,S]`、`attention_mask[B,S]`、空 `image_embeds[0,1536]` | 每条文本一个 `[1536]` |
| query | 用户 instruction + 用户文本；仅 tokenizer | 不调用 | 同 text；instruction 改变 system 段 token | 每条 query 一个 `[1536]` |
| image | 默认 instruction + vision placeholder；processor 同时产生 text/image tensors | 调用，产生 `[K,1536]` | image token 已展开为 $K$ 个位置；注入视觉 token | 每张图一个 `[1536]` |
| fused | 默认或 query instruction + vision placeholder + 文本；processor 同时产生 tensors | 调用，产生 `[K,1536]` | 文本 token 与 $K$ 个视觉 token 在同一 `[1,S,1536]` 序列 | 每个 text-image 对一个 `[1536]` |

`is_query=False` 或 `instruction=None` 时，官方和本项目都回到默认 instruction `You are a helpful assistant.`；只有 `is_query=True` 且传入 instruction 时才使用用户 instruction。所有模式最终都由同一 decoder 的同一 last-token pooling 与 L2 normalize 产生同一 1536 维检索空间的向量。

### 2.1 输出契约：四种输入都输出固定长度 embedding vector

`text`、`query`、`image` 和 `fused` 的最终输出都是处于**同一检索向量空间**的 float32 embedding，而不是 logits、token 序列或分类标签。所有 `gme_language_<prec>.xml` 的输出端口都叫 `embedding`，IR 形状为：

```text
embedding: float32[B, 1536]
```

其中 `B` 是 batch 中的输入样本数量；每一行都是一个 **1536 维、L2 归一化** 的向量，$\\lVert e_i \\rVert_2 \approx 1$。因此，两个输出向量的内积就是 cosine similarity：

$$
\operatorname{similarity}(e_q, e_d) = e_q^T e_d
$$

可直接将 text query 与 image、text 或 fused corpus embedding 做内积/ANN 检索。`query` 并不是另一种 shape：它和 `text` 都输出 `[B,1536]`，区别仅在于 query 会把用户给定的 instruction 写进 GME system prompt，从而改变向量语义。

| 模式 | 每个输入样本的输出 | 说明 |
| --- | --- | --- |
| text | `float32[1536]` | 默认 GME instruction 下的文本 embedding |
| query | `float32[1536]` | 带 query instruction 的文本 embedding |
| image | `float32[1536]` | 单张图片的 embedding |
| fused | `float32[1536]` | 一条文本与一张图片融合后的 embedding |

Python API 返回 `numpy.ndarray[B,1536]`，即使输入只有一个样本也保留 batch 维度；C++ API 返回 `gme::Embedding`，其中 `rows=B`、`dim=1536`、`data` 为行主序的 `float` 数组，长度为 `B*1536`。两端内部都逐样本计算后再沿 batch 维拼接（见 2.0.2），输出契约与顺序不变。

**1536 是模型架构固定的 embedding 维度，不是运行时可调参数。** 当前 checkpoint 的 language hidden size 为 1536，last-token pooling 直接输出该 hidden state，没有额外 projection head。`fp32`、`int8`、`int4`、`int4_gs64` 和 `int4_visint8` 只改变权重存储/计算精度，输出仍是 `float32[B,1536]`。`max_length`、文本 token 长度、`min_image_tokens`、`max_image_tokens`、图像分辨率和 batch size 会影响输入 shape、内存和时延，但**不会**改变 1536 维输出。

若必须使用其他维度，不能只改配置文件或对 IR reshape；需要选用原生维度不同的模型，或训练并部署新的投影层。后者会产生新的 embedding space，不能与现有 1536 维 GME 向量直接做相似度检索，且需要重新进行精度和检索指标验证。

### 2.2 OpenVINO Runtime、OpenVINO GenAI 与 Optimum CLI：是否可以直接运行/转换？

先区分三个层次：

1. **OpenVINO Runtime** 不能直接推理 Hugging Face 的 `safetensors` checkpoint；它需要 OpenVINO IR（XML/BIN）或在应用中先完成转换。当前 C++/Python runtime 已经是“直接用 OpenVINO Runtime 运行”，但运行的是本项目从 GME checkpoint 导出的 IR，不是直接读取 Hugging Face 权重。
2. **OpenVINO GenAI** 是面向标准生成式 LLM/VLM 和标准文本 embedding 模型的高层 runtime，不是任意 Transformers 自定义 `forward()` 的自动适配层。
3. **Optimum Intel CLI** 是转换工具。它可以转换受支持的普通 Qwen2-VL 生成模型，但“转换成功”不等于会得到 GME 所需的 embedding 输出和语义。

#### 本机对 OpenVINO GenAI 的实际探测

本机优化版 OpenVINO `2026.2.1` 安装包含 `openvino_genai`（build `qwen-ttft-llm-bench`），其 Python API 有 `TextEmbeddingPipeline`、`VLMPipeline`、`LLMPipeline` 等。对当前模型包直接构造 pipeline 的**实测结果**：

```text
TextEmbeddingPipeline(package, "CPU")            -> RuntimeError: could not open .../openvino_model.xml
VLMPipeline(package, "CPU")                      -> RuntimeError: could not open .../openvino_language_model.xml
# 把 gme_language_int8.xml 改名成它要的 openvino_model.xml 再喂进去（排除“只是文件名不对”）：
TextEmbeddingPipeline(renamed language IR)       -> RuntimeError: Check 'is_axis_valid(axis, r)' failed  (validation_util.cpp:331)
```

**关键点：改名也不行。** 即使给出它期望的文件名，`TextEmbeddingPipeline` 仍在 `is_axis_valid` 处失败——它假设模型是标准文本编码器、会在某个 token 轴上自己做 pooling，但 GME 的 language IR 有三个输入（含 `image_embeds`）且**已经在图内完成 pooling+normalize**，输出就是 `[B,1536]`，没有可供它 pooling 的 token 轴。这证明不兼容是**结构性的**，不是命名问题：

| GenAI pipeline | 设计目标 | 为什么不能直接替代当前 GME runtime |
| --- | --- | --- |
| `TextEmbeddingPipeline` | 单个标准**文本** embedding IR；pipeline 自己做 CLS/mean/last-token pooling 和 normalize | 当前 GME language IR 的输入还包括 `image_embeds`，并且已经输出最终的 `[B,1536]` embedding。该 pipeline 不处理 GME 的 visual encoder、image-pad 展开或 text-image fused 路径。 |
| `VLMPipeline` | 视觉语言模型的**文本生成**；调用 `.generate()` | 它期望标准 VLM 目录、生成式 language model、logits 以及 KV cache/状态管理。当前 `gme_language` / `gme_visual` 是无状态 embedding 图，只有一个 `embedding` 输出，没有 logits、`ReadValue`/`Assign` KV state 或 generation loop。 |
| `LLMPipeline` | 纯文本生成 | 不处理 GME 图片路径，也不公开本项目需要的末 hidden state embedding。 |

因此，当前包不能且不应该通过 `TextEmbeddingPipeline` 或 `VLMPipeline` 直接加载。可以为 GenAI 单独实现一个新的 GME embedding pipeline，但它需要新增对 visual IR、GME prompt、动态 image-pad 展开和“已在模型内做 pooling/normalize”的支持；这不是现成 pipeline 的配置选项。

#### Optimum CLI：可用于什么，不能替代什么

**已在独立环境 `env-opt`（optimum 2.2.0 / optimum-intel 2.0.0 / transformers 5.0.0 / openvino 2026.2.1）实测。** 注意版本冲突：optimum-intel 需要 `transformers>=5`，而 GME 远程代码要求 `transformers<4.52`，二者不能共存，所以必须用独立环境、并走**标准 Qwen2-VL 路线（不加 `--trust-remote-code`）**。实测命令（在 24G 内存上限、关闭 swap 下）：

```bash
optimum-cli export openvino \
  --model ../benchmark/models/gme-Qwen2-VL-2B-Instruct \
  --task image-text-to-text --weight-format int8 \
  ov-qwen2vl-generation/
# 结果：导出成功（exit 0）
```

**转换成功，但产物不能用作 GME embedding。** 导出目录是一个自回归**生成** VLM，实测 IR 结构为：

| 导出文件 | 输入 | 输出 |
| --- | --- | --- |
| `openvino_language_model.xml` | `attention_mask, position_ids[3,?,?], inputs_embeds[?,?,1536], beam_idx` | **`logits[?,?,151936]`**（有状态 KV-cache 生成图） |
| `openvino_text_embeddings_model.xml` | `input[?,?]` | `inputs_embeds[?,?,1536]` |
| `openvino_vision_embeddings_model.xml` (+ `_merger`) | `hidden_states` | `last_hidden_state[?,1280]` → merger → 视觉 token |

为什么不能直接用于 GME：

* 它的 language IR 输出 **`logits`（词表 151936 维）**，走的是 `beam_idx` + KV-cache 的生成图；而 GME 需要的是 **lm_head 之前的 `last_hidden_state`**，按 `attention_mask.sum(-1)-1` 取末有效 token 再 L2 normalize。**logits 无法反推回该 hidden state**（`lm_head` 是 1536→151936 的不可逆投影），所以拿不到 GME 向量。
* 它是**有状态生成图**（面向 `.generate()`），不是无状态的单次 embedding 前向。
* 要得到 GME embedding，必须用返回 hidden-state 的自定义 forward 重新导出——这正是本项目 `02_export.py` 做的事。

结论：**Optimum CLI 能把该 checkpoint 转成 OpenVINO 上的 Qwen2-VL 聊天/生成模型，但不能产出 GME embedder。** 它可作为“在 Intel 上跑 Qwen2-VL 生成”的现成方案，不是本项目 embedding 包的单命令替代。

### 2.3 当前导出到底做了什么：不是“普通 backbone + 新训练 head”

当前实现**没有**导出未微调的 `Qwen2-VL-2B-Instruct` 再自己接一个训练好的 GME head。两个 IR 的所有参数都来自：

```text
benchmark/models/gme-Qwen2-VL-2B-Instruct
```

也就是 GME 微调后的 checkpoint。GME 没有额外的、可训练的 projection head；所谓 GME “head”是其官方 `GmeQwen2VL.forward()` 中的无参数后处理：末有效 token pooling 和 L2 normalization。当前 `02_export.py` 以同样的计算表达式包装 checkpoint：

```text
image/text input
  -> official tokenizer + GME prompt preprocessing
  -> gme_visual (only when an image is present)
  -> put visual vectors into image_token_id positions
  -> GME checkpoint's token embedding + Qwen2-VL decoder
  -> hidden[:, attention_mask.sum(-1)-1]
  -> L2 normalize
  -> embedding[1536]
```

我们实现的部分只有与官方 forward 等价、且没有新学习参数的逻辑：视觉向量替换 image-pad token 的 `masked_scatter`、right-padding 的末 token 索引，以及 `F.normalize`。这三个步骤同样被导出到 language IR。FP32 在 Python OpenVINO 的四种模式与官方 golden vector 为 `cosine=1.0`，正是该组合与官方 GME forward 一致的验证。

#### `gme_visual` 与 `gme_language` 分别是什么

| IR | 来自 GME checkpoint 的组件 | 输入 | 输出 | 负责的功能 |
| --- | --- | --- | --- | --- |
| `gme_visual_<prec>.xml` | `base.visual`，即 Qwen2-VL ViT | `pixel_values[N,1176]`、`image_grid_thw[M,3]` | `image_embeds[K,1536]` | 将 Qwen2-VL processor 产生的 patch 表示编码、merge 为语言模型可使用的视觉 token。`K` 随图片分辨率变化。 |
| `gme_language_<prec>.xml` | `base.model.embed_tokens` + `base.model`，均为 **GME 微调权重** | `input_ids[B,S]`、`attention_mask[B,S]`、`image_embeds[K,1536]` | `embedding[B,1536]` | 查 token embedding、将每个 `image_token_id` 位置替换为视觉向量、运行全部 decoder、pooling 和 L2 normalize。文本模式传空的 `[0,1536]` 视觉张量。 |

这里的 `S` 和 `K` 是动态的：图片模式中，processor 先把单个 `<|image_pad|>` 展开成正好 `K` 个 image token，language IR 才能执行一对一 `masked_scatter`。例如本项目的超宽 `gme_logo.png` 得到 `K=1260`；这个数字会影响时延，不会改变最终 embedding 的 1536 维度。

#### 为什么保持两个 IR

当前仅保留双 IR 部署：`text/query` 只执行 `gme_language`，而 `image/fused` 执行 `gme_visual` 后把视觉 token 交给 `gme_language`。这与官方 `pixel_values is not None` 分支一致，避免纯文本无意义地执行视觉塔。

双 IR 也便于分别量化 visual 与 language，支持 `int4_visint8` 等混合精度；并且 visual 推理完成后可释放其临时张量，再执行语言 decoder，降低大图的峰值内存。代价是 image/fused 有一次中间 `image_embeds` 传递，但这是当前准确性、资源安全与功能完整性之间更合适的取舍。

---

## 3. 目录结构

```
gme-Qwen2-VL-2B-Instruct-Running/
├─ README.md                    # 本文档
├─ env-ref/                     # Python 参考/导出环境 (transformers==4.51.3, openvino 2026.2, nncf, ov-tokenizers)
├─ env-opt/                     # 独立 optimum 环境 (optimum-intel 2.0, transformers 5.0) — 仅用于 §2.2 Optimum 对照实验
├─ gme_openvino.py              # Python 运行时 GmeOpenVINO（与 C++ 等价，用于对照/调试）
├─ 01_reference.py              # 用官方实现生成 FP32 黄金向量 + 模板/token 记录
├─ 02_export.py                 # 从原始 GME checkpoint 导出 FP32 visual/language IR（分进程限内存）
├─ 03_finalize.py               # 转 tokenizer IR(逐token校验) + NNCF INT8/INT4(含视觉,排除rotary) + 写 config
├─ 04_validate_python.py        # Python OV 精度：全精度 × 全模式 × CPU/GPU
├─ 05_validate_cpp.py           # 驱动 C++ gme_embed，与黄金向量比对
├─ 06_benchmark.py              # Python + C++ 时延基准（fp32/int8/int4 × 模式 × CPU/GPU）
├─ 07_llamacpp_text_check.py    # GGUF/llama.cpp 文本向量对照（备选方案评估）
├─ 08_validate_batch.py         # batch 正确性回归：任意 N、两种 batch_mode、官方 API 对照、截断、同形状行（Python + C++）
├─ 09_benchmark_matrix.py       # 性能矩阵：指定 token 长度 / 常见图片尺寸 / batch 扩展性 -> benchmark_matrix.json
├─ 10_retest_gpu_1024.py        # GPU 1024-token 档位的复测（DVFS/warmup 影响的追查）
├─ PERFORMANCE_REPORT.md        # 性能报告（09 的完整矩阵）
├─ experiments/
│  ├─ probe_visual_batch.py     #   "为什么 visual 逐图" 的实测证据（§2.0.2）
│  └─ check_image_edge_cases.py #   图像解码/smart_resize 边界：EXIF、半 factor 取整、超小、超宽（§2.0.2.2）
├─ cpp/                         # C++ 运行时（主实现）
│  ├─ include/gme_qwen2vl.hpp   #   GmeQwen2VL 类
│  ├─ src/gme_qwen2vl.cpp       #   tokenizer/visual/language IR 编排 + prompt 模板 + image_pad 展开
│  ├─ src/image_preprocess.*    #   Qwen2-VL 图像预处理 (OpenCV: smart_resize+normalize+patchify)
│  ├─ src/main.cpp              #   gme_embed CLI（--mode / --benchmark / --out）
│  ├─ CMakeLists.txt, build.sh  #   对接 benchmark/openvino/install
│  └─ build/gme_embed           #   构建产物
├─ ov_models/gme-qwen2-vl-2b-instruct-openvino/   # 模型包（3类IR × 各精度 + tokenizer IR + 配置 + nncf_fingerprints.json）
└─ artifacts/                   # 黄金向量 + 各项验证/基准 JSON
```

模型包内精度（`gme_openvino_config.json` 的 `precisions`，`default_precision=int8`）：

| 精度键 | language IR | visual IR | 语言/视觉 .bin 体积 |
| --- | --- | --- | --- |
| `fp32` | fp32 | fp32 | 6.2G / 2.7G |
| `int8` | INT8 | INT8(排除rotary) | 1.5G / 668M |
| `int4` | INT4 gs128, data-aware(AWQ+scale estimation) | INT4 gs128(排除rotary), data-free | 922M / 352M |
| `int4_gs64` | INT4 gs64, data-aware(AWQ+scale estimation) | INT4 | 947M / 352M |
| `int4_visint8` | INT4, data-aware(AWQ+scale estimation) | INT8 | 922M / 668M |

体积为实测值（重新压缩后的当前包）。语言塔的两个 INT4 变体现在是 data-aware（见 §7），体积相比纯 data-free
版本略增（AWQ/scale estimation 会让少数层退回 INT8 以保持精度，见下方压缩日志里 `16% (2 / 197)` 的 INT8 占比），
换来文本 cosine 从 ≈0.89–0.92 提升到 ≈0.95–0.97（§7）。

---

## 4. 操作步骤

### 4.0 路径配置与 handoff

所有 Python 脚本都从脚本所在目录定位仓库内的 `ov_models/`、`artifacts/` 和 `cpp/build/`，不依赖启动时的当前目录。
外部依赖通过 `gme_paths.py` 统一管理；默认兼容本仓库原有布局，也可用环境变量覆盖，适合移交到其他项目：

| 环境变量 | 用途 | 默认值 |
| --- | --- | --- |
| `GME_BENCHMARK_ROOT` | benchmark 工作区根目录 | `GME_ROOT/../benchmark` |
| `GME_SOURCE_MODEL` | 原始 GME Hugging Face checkpoint | `$GME_BENCHMARK_ROOT/models/gme-Qwen2-VL-2B-Instruct` |
| `GME_OPENVINO_INSTALL` | OpenVINO 安装目录 | `$GME_BENCHMARK_ROOT/openvino/install` |
| `GME_PACKAGE` | OpenVINO GME 模型包 | `GME_ROOT/ov_models/gme-qwen2-vl-2b-instruct-openvino` |
| `GME_ARTIFACTS` | 黄金向量、日志和 benchmark 产物目录 | `GME_ROOT/artifacts` |
| `GME_CPP_BINARY` | C++ `gme_embed` 可执行文件 | `GME_ROOT/cpp/build/gme_embed` |
| `GME_LLAMA_CPP_ROOT` | llama.cpp 根目录 | `GME_ROOT/../llama.cpp` |
| `GME_LLAMA_EMBEDDING` | llama.cpp embedding 可执行文件 | `$GME_LLAMA_CPP_ROOT/build-openvino/bin/llama-embedding` |
| `GME_GGUF_MODEL` | GGUF 对照模型路径 | `$GME_BENCHMARK_ROOT/models/gme-Qwen2-VL-2B-Instruct-GGUF/...gguf` |

示例：

```bash
cd /path/to/gme-Qwen2-VL-2B-Instruct-Running
export GME_BENCHMARK_ROOT=/path/to/shared/benchmark
export GME_OPENVINO_INSTALL=/path/to/openvino/install
export GME_SOURCE_MODEL=/path/to/gme-Qwen2-VL-2B-Instruct
./env-ref/bin/python 04_validate_python.py
```

如果 GME 已被集成到其他项目，不需要修改脚本源码；只需设置上述外部路径变量，并确保 Python 环境已安装对应依赖。

### 4.1 一次性：导出模型包（Python，`env-ref`）

```bash
cd gme-Qwen2-VL-2B-Instruct-Running
./env-ref/bin/python 01_reference.py     # 仅在需要重建黄金向量时
./env-ref/bin/python 02_export.py --device CPU   # 导出 FP32 visual/language IR
./env-ref/bin/python 03_finalize.py      # tokenizer IR + INT8/INT4（含视觉，排除 rotary）
```

### 4.2 构建 C++ 运行时（主实现）

```bash
cd cpp && bash build.sh      # 内部 source ../../benchmark/openvino/install/setupvars.sh
# 产物: cpp/build/gme_embed；也可 OV_INSTALL=/path/to/openvino/install bash build.sh
```

`build.sh` 自动从该 OpenVINO 安装里找 OpenCV(4.11) 与 `libopenvino_tokenizers.so`（后者路径编入二进制，运行期也可用 `GME_TOKENIZERS_LIB` 覆盖）。

### 4.3 运行（CLI）

```bash
source ../benchmark/openvino/install/setupvars.sh
PKG=ov_models/gme-qwen2-vl-2b-instruct-openvino
LOGO=../benchmark/models/gme-Qwen2-VL-2B-Instruct/images/gme_logo.png

cpp/build/gme_embed --package $PKG --device GPU --precision int8 --mode text  --is-query 0 "What kind of car is this?"
cpp/build/gme_embed --package $PKG --device CPU --precision int8 --mode text  --is-query 1 \
    --instruction "Find an image that matches the given text." "What kind of car is this?"
cpp/build/gme_embed --package $PKG --device GPU --precision int8 --mode image --image $LOGO
cpp/build/gme_embed --package $PKG --device GPU --precision int8 --mode fused --image $LOGO "GME logo"
```

`--precision ∈ {fp32,int8,int4,int4_gs64,int4_visint8}`，`--device ∈ {CPU,GPU}`，`--out FILE` 导出二进制向量。

`--cache-dir DIR` 启用 OpenVINO 编译缓存（`ov::cache_dir`）：同一 package/device/precision/streams 的下一次进程直接加载缓存
blob，不再重新编译。GPU 上编译 INT8 language IR 从零开始要几十秒，而 `gme_embed` 是短生命周期 CLI（每次调用都重新编译），
这是命中率最高的一个旋钮：

```bash
cpp/build/gme_embed --package $PKG --device GPU --precision int8 --cache-dir /var/cache/gme_ov --mode text "..."
# 实测（CPU/int8）：首次编译 6.0s，命中缓存后 2.0s
```

`--min-image-tokens N` / `--max-image-tokens N` 覆盖 package 的 Qwen2-VL processor 图像 token 上下界（默认取
package 自带值，即 256/1280）。这是 image/fused 时延最有效的旋钮——注意力开销随视觉 token 数增长（§6.4/§6.5）——
且纯预处理层面的改动，**不需要重新导出**：visual/language 两个 IR 的序列维度都是动态的，任意 token 数都能跑。

`--max-length N` 收缩 package 的 tokenizer 截断长度（默认取 package 自带值，如 1800）。**只能收缩，不能放大**：
上限是导出时烘进 tokenizer IR 的（`03_finalize.py`），请求更大的值会明确报错，需要用更大的 `--max-length` 重新导出
才能提高上限。收缩时的实现是对已分词好的、右填充的 batch 直接按列裁剪到 N——因为该 tokenizer 的 `truncation_side`
是 `"right"`，这与直接用 `truncation=True, max_length=N` 重新分词逐 token 相同（已用 HF tokenizer 验证）。

```bash
cpp/build/gme_embed --package $PKG --device CPU --precision int8 \
    --min-image-tokens 16 --max-image-tokens 64 --mode image --image $LOGO   # 更小的图，更快
cpp/build/gme_embed --package $PKG --device CPU --precision int8 --max-length 256 --mode text "..."
```

### 4.4 库 API（C++）

```cpp
#include "gme_qwen2vl.hpp"
gme::GmeQwen2VL model("ov_models/gme-qwen2-vl-2b-instruct-openvino", "GPU", "int8");
auto e_text  = model.encode_text({"What kind of car is this?"});                          // [1,1536]
auto e_query = model.encode_text({"..."}, "Find an image that matches the given text.", true);
auto e_image = model.encode_image({"gme_logo.png"});
auto e_fused = model.encode_fused({"GME logo"}, {"gme_logo.png"});
// e_*.data 为行主序 [rows*dim] 的已 L2 归一化 float32
// 此模型的 dim 固定为 1536；e_*.data.size() == e_*.rows * 1536。
```

C++ 的三个 encode 方法都接受可选 `batch_size`（默认 32，`0` 为不分块），与 Python/官方一致：

```cpp
auto e = model.encode_text(texts, /*instruction=*/"", /*is_query=*/false, /*batch_size=*/8);
```

构造函数另接受 `cache_dir`、`min_image_tokens`、`max_image_tokens`、`max_length`（均为可选，语义与 CLI 的同名 flag
一致，见 §4.3）：

```cpp
gme::GmeQwen2VL model(
    "ov_models/gme-qwen2-vl-2b-instruct-openvino", "GPU", "int8",
    /*streams=*/1, /*batch_mode=*/"auto", /*cache_dir=*/"/var/cache/gme_ov",
    /*min_image_tokens=*/16, /*max_image_tokens=*/64, /*max_length=*/256);
```

### 4.5 库 API（Python）

`gme_openvino.py` 的 `GmeOpenVINO` 与官方 `GmeQwen2VL` 同名同义（逐项对照见 §2.0.2.1）：

```python
from gme_openvino import GmeOpenVINO

enc = GmeOpenVINO("ov_models/gme-qwen2-vl-2b-instruct-openvino", device="GPU", precision="int8")

enc.get_text_embeddings(texts)                                    # [N,1536]
enc.get_text_embeddings(texts, instruction="Find an image ...")    # query
enc.get_image_embeddings(paths_or_pil_or_urls)
enc.get_fused_embeddings(texts, images)
enc.embed(texts=texts, images=images)                             # 单批原语（不分块）
enc.encode(sentences), enc.encode_queries(qs), enc.encode_corpus(rows)   # ST/MTEB 风格

# batch 相关：batch_size 默认 32；per_sample 下分块不改变任何一行
enc.get_fused_embeddings(texts, images, batch_size=8, show_progress_bar=False)
GmeOpenVINO(pkg, device="CPU", precision="int8", streams="auto")   # CPU 吞吐
GmeOpenVINO(pkg, device="GPU", precision="int8", batch_mode="padded")

# 编译缓存：同一 device/precision/streams 的下一次构造直接加载缓存 blob
GmeOpenVINO(pkg, device="GPU", precision="int8", cache_dir="/var/cache/gme_ov")

# 覆盖图像 token 上下界（默认取 package 自带的 256/1280）；纯预处理改动，不需要重新导出，
# 因为 visual/language 两个 IR 的序列维度都是动态的
GmeOpenVINO(pkg, device="CPU", precision="int8", min_image_tokens=16, max_image_tokens=64)

# 覆盖 tokenizer 截断长度（默认取 package 自带值，如 1800）；language IR 序列维度全动态，
# 可任意放大或收缩，不像 C++ 端受 tokenizer IR 烘入的上限约束
GmeOpenVINO(pkg, device="CPU", precision="int8", max_length=256)
```

---

## 5. 测试方法

- **黄金基准**：`01_reference.py` 用官方 `gme_inference.py`（原始 safetensors，fp32）对固定输入（5 条文本 + 本地 `gme_logo.png`）生成
  `artifacts/golden_{text,query,image,fused}.npy`，并记录模板串、token id、padding mask（`golden_meta.json`）。所有后端都以它为基准，
  避免“模板/末 token 位置不同导致向量不匹配”的常见误判。
- **精度**：以 cosine（向量已 L2 归一化）对比。
  - `04_validate_python.py`：Python OV，全精度 × {text,query,image,fused} × {CPU,GPU} → `artifacts/openvino_full_validation.json`。FP32 的四个模式均强制断言 `cosine >= 0.9999`。
  - `05_validate_cpp.py`：驱动 C++ `gme_embed --out`，读回二进制向量与黄金比对 → `artifacts/cpp_validation.json`。该脚本报告所有模式；当前只将 C++ FP32 text/query 视为参考级精度，image/fused 的数值差异作为回归指标归档。
- **时延**：`06_benchmark.py`，warm、端到端每次调用（含分词/预处理 + 推理，不含一次性加载/编译），repeat=5，
  单条短文本 + `gme_logo.png` → `artifacts/benchmark.json`。C++ 用进程内 `--benchmark` 一次编译跑完 4 种模式。
- **batch 功能与准确性**：`08_validate_batch.py` 在受限内存的 CPU 进程中运行 9 组检查，结果写入
  `artifacts/batch_validation.json`：

  | 组 | 内容 |
  | --- | --- |
  | A/B | 任意 $N\in\{1,2,3,5\}$ 的 batch vs 逐条：**长度不等的文本 + 五种尺寸的图片**（刻意 ragged），Python 覆盖 text/image/fused |
  | B2 | `padded` 模式相对逐条的漂移（预期存在，记录量级） |
  | C | `streams=1` vs `"auto"` 的数值不变性 |
  | D | FP32、$N$=3 ragged 与官方 PyTorch 对照，两种 `batch_mode` 都要落在 0.9999 以上 |
  | E | C++ batch vs C++ 逐条 vs Python，含 `--streams` 与 `--batch-size ∈ {0,1,2}` |
  | F | 官方 API 面：`embed()`、`batch_size ∈ {1,2,5,32,None}`、三个 alias、全 `None` 图片列表、非法输入的拒绝（§2.0.2.1） |
  | G | 超过 `max_length` 的输入在 C++ 与 Python 的截断一致性（§2.0.2.1） |
  | H | visual IR 的动态图片数：$B$=1 IR 拒绝 2 行；与 $B$=2 IR 的对照（§2.0.2） |
  | I | **同形状但内容不同**的样本不得互相污染——非 ragged，专门覆盖 §2.0.2.2 的 pool 重复行问题 |

  A/B/E 用 ragged 输入暴露对 padding 与统一 grid 的隐含依赖；I 反过来用非 ragged 输入，两者互补。
- **图像解码与 resize 边界**：`experiments/check_image_edge_cases.py`。上面 A–I 全部使用 PIL 生成的、常规尺寸的
  无 EXIF RGB PNG，因此覆盖不到"图片本身长什么样"这一类分叉。该脚本补两部分：
  第一部分把 `_resize_for_gme` 对官方 `smart_resize` 做 18 组尺寸扫描（纯算术、不加载模型，秒级）；
  第二部分用真实编码器跑 EXIF Orientation=6 的 JPEG、正好落在半个 factor 上的 `70x70`、短于一个 patch 的 `10x10`、
  超 200:1 的 `1x2000`，以及作为对照的 `448x448`，做 C++ vs Python 对照。这四类正是 §2.0.2.2 第 3/4/5 条的回归。

  ```bash
  ./env-ref/bin/python experiments/check_image_edge_cases.py --resize-only   # 秒级，纯算术
  ./env-ref/bin/python experiments/check_image_edge_cases.py                 # 含 C++/Python 端到端
  ```
- **性能矩阵**：`09_benchmark_matrix.py`（→ `artifacts/benchmark_matrix.json`，报告见 `PERFORMANCE_REPORT.md`）。
  它补上 `06_benchmark.py` 无法回答的三件事：指定 token 长度（含 1024）的 text/query、**常见分辨率**的 image/fused
  （`06_benchmark.py` 只用 1260 视觉 token 的 `gme_logo.png`，属最坏情况）、以及 batch 扩展性
  （$N$ × `batch_mode` × `streams`，同时报告 ms/call 与 ms/embedding）。
- **备选方案对照**：`07_llamacpp_text_check.py`，用完整 GME 模板 + `--pooling last` 跑 GGUF 文本向量 → `artifacts/llamacpp_validation.json`。

---

## 6. 测试结果（本机 Core Ultra 7 356H，Linux，OpenVINO 2026.2）

### 6.1 Python OpenVINO 权重与数值一致性（FP32 cosine vs 官方 PyTorch 黄金）

| 设备 | text | query | image | fused |
| --- | ---: | ---: | ---: | ---: |
| CPU | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| GPU | 1.000000 | 1.000000 | 1.000000 | 1.000000 |

→ Python OpenVINO 运行时将三个 IR 组合后，在 FP32 下与原始 `benchmark/models/gme-Qwen2-VL-2B-Instruct` 的四种 embedding 模式数值等价。

> §2.0.2.2 重写 request pool 后重跑了本表与 §6.2：FP32 四种模式仍为 `1.000000`（未受影响）；
> CPU 量化档的 `text` 列（5 条文本的最小 cosine）从 `0.998315/0.907792/0.923465` 微移到
> `0.998476/0.908822/0.921590`——同一数量级的量化噪声，非回归。

### 6.2 精度：Python OpenVINO（cosine vs FP32 黄金，`openvino_full_validation.json`）

| 设备 | 精度 | text | query | image | fused |
| --- | --- | ---: | ---: | ---: | ---: |
| CPU | fp32 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| CPU | **int8** | 0.998476 | 0.998361 | 0.992512 | 0.997980 |
| CPU | int4 | 0.908822 | 0.912175 | 0.945496 | 0.962353 |
| CPU | int4_gs64 | 0.921590 | 0.949724 | 0.955428 | 0.967409 |
| CPU | int4_visint8 | 0.908822 | 0.912175 | 0.961537 | 0.957409 |
| GPU | fp32 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| GPU | **int8** | 0.997518 | 0.998764 | 0.989255 | 0.995806 |
| GPU | int4 | 0.886320 | 0.911044 | 0.936908 | 0.955260 |
| GPU | int4_gs64 | 0.918643 | 0.950327 | 0.944975 | 0.963901 |
| GPU | int4_visint8 | 0.886320 | 0.911044 | 0.947935 | 0.968358 |

### 6.3 精度：C++（cosine vs FP32 黄金，`cpp_validation.json`）

| 设备 | 精度 | text | query | image | fused |
| --- | --- | ---: | ---: | ---: | ---: |
| CPU | fp32 | **1.000000** | **1.000000** | 0.975047 | 0.988341 |
| CPU | int8 | 0.998476 | 0.998361 | 0.971103 | 0.985380 |
| CPU | int4 | 0.908822 | 0.912175 | 0.943903 | 0.957391 |
| GPU | fp32 | **1.000000** | **1.000000** | 0.975043 | 0.988341 |
| GPU | int8 | 0.997518 | 0.998764 | 0.967116 | 0.981741 |
| GPU | int4 | 0.886320 | 0.911044 | 0.930794 | 0.948113 |

- 文本/query 与 Python 完全一致（tokenizer IR 逐 token 一致，语言 IR 相同）。
- 图像/图文比 Python 低约 `0.02` / `0.012`。**剩余差异只有一个来源：resize 重采样**——C++ 预处理使用 OpenCV，参考实现使用 Pillow；C++ 已采用缩小时 `INTER_AREA`、放大时 `INTER_CUBIC` 来逼近 Pillow bicubic，但这不是逐像素等价实现。`gme_logo` 的锐利边缘和 1260 个合并视觉 token 使该用例对 resize 差异敏感。
  曾经与它混在一起的另外三项（EXIF 旋转、half-to-even 取整、极端尺寸被拒）已按 §2.0.2.2 修复，不再计入本项。
- **该差异只在某一级 resize 真的重采样时出现。** 两级 `smart_resize` 都是 no-op 的图片（例如 `448x448`，本身就在
  256 token 下界上）C++ vs Python 为 `cosine = 0.999976`；反之从 `10x10` 放大到 `448x448` 这种极端上采样为 `0.941`。
  所以"C++ 图像差 0.02"不是一个常数，而是随重采样比例变化的量。
- 因此，C++ 图像路径的验收含义是“可运行且差异被记录”，不是 FP32 参考级数值等价。对图像检索精度有严格要求时，应使用 Python runtime，或先以 Pillow 等价 resize 替换/验证 C++ 预处理，再提高 C++ image/fused 的验收阈值。

  `experiments/check_image_edge_cases.py` 的实测（GPU/INT8，C++ vs Python），修复前后对照：

  | 用例 | 修复前 | 修复后 | 残余差异来源 |
  | --- | ---: | ---: | --- |
  | `448x448`（两级 resize 均 no-op） | 0.999976 | 0.999976 | 无重采样 |
  | `70x70`（半个 factor 取整） | **0.900773** | 0.967264 | resize 重采样 |
  | `10x10`（短于一个 patch） | **C++ 拒绝** | 0.941486 | resize 重采样（44.8x 上采样） |
  | `1x2000`（超 200:1） | **C++ 拒绝** | 0.984768 | resize 重采样 |
  | `140x224` + EXIF Orientation=6 | **0.917350** | 0.959535 | resize 重采样 |

### 6.4 时延（`benchmark.json`，warm，ms/次，repeat=5）

所有时延行对应同一个固定 workload，均为 batch size 1，且 `benchmark.json.workload` 会由 `06_benchmark.py` 自动记录：

| 模式 | 业务输入 | GME 模板后的有效序列长度 | 图像处理结果 |
| --- | --- | ---: | --- |
| text | `What kind of car is this?`，25 字符 | 27 tokens | 无 |
| query | 同一文本 + `Find an image that matches the given text.`，42 字符 | 30 tokens | 无 |
| image | 无用户文本 | 1282 tokens | `gme_logo.png`: `5164×600` → `5152×588`，`grid_thw=[1,24,210]`，5040 raw patch rows，1260 visual tokens |
| fused | 同一 25 字符文本 + 图像 | 1289 tokens | 与 image 相同 |

这里的 `sequence_tokens` 已包含 system/user/assistant 模板和图像 placeholder 展开，不是裸文本 token 数。文本、instruction、batch size、最终序列长度、图像尺寸和视觉 token 数都会影响时延；特别是注意力计算会随序列长度大致二次增长。因而本节只描述这个 workload，不能直接外推到长文本、大 batch 或普通尺寸图像。

**文本 / query（快）**

| 设备 | 精度 | Python text | C++ text | C++ query |
| --- | --- | ---: | ---: | ---: |
| CPU | fp32 | 223.5 | 224.2 | 235.2 |
| CPU | int8 | 78.3 | 78.0 | 85.0 |
| CPU | int4 | 87.9 | 86.9 | 93.9 |
| GPU | fp32 | 174.0 | 172.7 | 168.7 |
| GPU | int8 | 52.9 | 40.8 | 41.4 |
| GPU | int4 | 30.5 | 29.4 | 29.8 |

**图像 / 图文（随图像 token 数增长；此处 `gme_logo` = 1260 合并 token，接近上限）**

| 实现 | 设备 | 精度 | image | fused |
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

这些结果是 `gme_logo.png` 的端到端测量，包含分词/图像预处理和推理，不含模型加载与编译。该图片产生 1260 个合并视觉 token，接近默认上限 1280；它说明最坏情况下的规模，**不能**作为普通分辨率图片的服务时延承诺。图像时延主要由视觉塔注意力和 token 数决定。

> ⚠️ **本节不能用于选型。** 常见分辨率快近一个数量级：`448x448`（256 视觉 token，也是
> `min_image_tokens` 的下界）在 GPU/INT8 上约 **0.55 s**，而本表的 logo 是 **5.17 s**——约 9x。
> 常见尺寸与指定文本长度的完整矩阵请用 `09_benchmark_matrix.py`，填入 `PERFORMANCE_REPORT.md`；
> 本节保留的意义是**最坏情况**与历史回归基线。（0.55 s 是 `--repeat 1` 的量级校准，不是可引用结果。）

要点：本次短文本基准中，CPU INT8 对 FP32 约快 3 倍；GPU 量化文本为约 29–53 ms/次；C++ 与 Python 文本时延接近，因为二者使用同一 language IR。

### 6.5 指定文本长度的时延对比

`06_benchmark.py --sequence-lengths 128,256` 生成了独立结果
`artifacts/benchmark_sequence_lengths.json`。长度的定义是 **GME 模板展开后的有效、未 padding token 数**，不是原始字符串的字符数或 tokenizer 前的裸文本 token 数。脚本通过实际 tokenizer 反复构造 ASCII filler 并校验 token 数：

| 目标长度 | text filler 字符数 | query filler 字符数 | query instruction 字符数 |
| ---: | ---: | ---: | ---: |
| 128 | 215 | 209 | 42 |
| 256 | 471 | 465 | 42 |

下表为 batch size 1、warm、端到端 `ms/call`，每个单元的格式为 `text / query`，初始 sweep 使用 `repeat=5`：

| 实现 | 设备 | 精度 | 128 tokens | 256 tokens |
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

C++ CPU INT8 的 256-token `repeat=5` text/query 差异明显，不能视作 prompt 类型造成的稳定性能差。对同一输入进行 `repeat=20` 复测得到 `669.05 / 669.13 ms`，说明原始值受短采样噪声影响。用于选型或回归阈值时，应至少使用 `--repeat 20`，并在同一设备空闲、相同电源策略下多次运行。

运行示例：

```bash
# 仅测 text/query；不会执行 image/fused 的长视觉序列。
./env-ref/bin/python 06_benchmark.py --sequence-lengths 128,256 --repeat 20

# 任意正整数。目标是最终有效长度，脚本会校验实际 token 数。
./env-ref/bin/python 06_benchmark.py --sequence-lengths 64,128,256,512 --repeat 20
```

该模式不覆盖 image/fused：图片会引入由 `image_grid_thw` 决定的视觉 token 数。测试图文长度时，应同时固定文本 token 和图像分辨率/视觉 token 数；不能只指定一个总 token 数，因为视觉 token 由图像 processor 动态产生。

更长的档位（512 / 1024 token）与图片尺寸阶梯请用 `09_benchmark_matrix.py --suite text-lengths,image-sizes`；
包内 `max_length=1800`，因此 1024 是留有余量的最大整齐档位，超过 1800 的输入两端都会截断（§2.0.2.1）。

### 6.6 Batch 功能性、准确性与吞吐

`08_validate_batch.py` 用**长度不等的文本**和**五种不同尺寸的图片**（448x448 / 336x336 / 448x336 / 280x280 / 336x448），对 $N\in\{1,2,3,5\}$ 比较 batch 与逐条推理。刻意做成 ragged，是为了让任何对 padding 或统一 grid 的隐含依赖暴露出来。

**任意 $N$ 都不需要重新导出模型，且 `per_sample`（CPU 默认）下 batch 与逐条逐位相同：**

| 后端 | 模式 | $N$ | batch vs sequential |
| --- | --- | ---: | --- |
| Python | text / image / fused | 1, 2, 3, 5 | **bit-exact**（全部 12 组） |
| Python | text（`batch_size=2` 分块） | 1, 2, 3, 5 | **bit-exact** |
| C++ | fused | 3 | **bit-exact** |
| C++ | fused（`--batch-size 0/1/2`） | 3 | **bit-exact** |
| Python | 官方 API 面（`embed`/`batch_size ∈ {1,2,5,32,None}`/三个 alias/全 `None` 图片） | 5 | **bit-exact**（14 项，§2.0.2.1） |
| Python + C++ | 超过 `max_length` 的文本 | 1 | **bit-exact**（截断一致，§2.0.2.1） |
| Python | **同形状不同内容**（4 条等长文本 + 2 张同 grid 图片） | 2, 4 | **bit-exact**，且行间无碰撞（§2.0.2.2） |

对照之下，`padded` 模式（GPU 默认，也是官方的做法）在 INT8、ragged 输入上是 `text 0.99993 / fused 0.99991`——**低于本仓库自己的 0.9999 验收线**。这不是 bug，而是 padded batch 的固有性质：一行的解码会看到多少 padding，取决于同批次里最长的样本。`per_sample` 从根本上消除了这一项。

**`streams` 只影响吞吐，不影响数值**（`streams=1` vs `"auto"`，pool 1 → 4）：text 与 fused 均为 bit-exact。

**与官方 PyTorch（FP32、$N$=3 ragged）对照，两种模式都落在 0.99999 以上：**

| 模式 | text | image | fused |
| --- | ---: | ---: | ---: |
| `per_sample` | 0.99999989 | 0.99999974 | 0.99999990 |
| `padded` | 0.99999989 | 0.99999974 | 0.99999990 |

**C++ 与 Python 对照**：text 在两种模式下都 **bit-exact**——这把两端的 batch 实现钉死在一起。fused 为 0.998894（两种模式完全相同），即 6.5 节记录的 OpenCV-vs-PIL resize 差异；它在两种 batch_mode 下取值一致，说明该差异来自图像预处理而非批处理。

复现命令：

```bash
source ../benchmark/openvino/install/setupvars.sh
systemd-run --user --scope -p MemoryMax=20G -p MemorySwapMax=0 \
  ./env-ref/bin/python 08_validate_batch.py
```

#### 吞吐：CPU 靠 `streams`，GPU 靠 `padded`

CPU（16 核、INT8、warm）上把样本塞进一次 padded 调用**完全没有加速**，真正有效的是多 stream 并发：

| workload | `streams=1` | `streams="auto"`（4 流） | 加速 |
| --- | ---: | ---: | ---: |
| text ×8 | 0.67 s | 0.37 s | **1.81x** |
| image ×4 | 10.23 s | 5.81 s | **1.76x** |

作为对照，同一 language IR 上 padded batch 的归一化吞吐为 $B$=1/4/8 → 1.00x/1.00x/0.99x。代价是单条样本在多 stream 下约慢 1.3x（language 443→578 ms，visual 2221→3350 ms），每多一个 stream 约 +0.2 GB。

GPU（iGPU、INT8、短文本、$N$=4）则相反，per-request 提交开销占主导：

| 方式 | 每次调用 | 每个 embedding |
| --- | ---: | ---: |
| padded $B$=4 | 88.15 ms | **22.04 ms** |
| 4× $B$=1 串行 | 285.94 ms | 71.49 ms |
| 4× $B$=1 异步（4 请求） | 212.80 ms | 53.20 ms |

所以 GPU 上 `batch_mode="auto"` 选 `padded`（快 2.4x），且多 stream 救不回来（0.34–0.41x）。这些都是单机短文本 workload，不是服务级 SLA；在线服务仍建议按最大等待时间形成 micro-batch。

> 以上吞吐数字是 `batch_mode="auto"` 默认值的**依据**，来自若干次一次性测量（$N$ 只到 4/8）。
> 完整的 $N \in \{1,2,4,8,16,32\}$ × `batch_mode` × `streams` × 设备 扫描由
> `09_benchmark_matrix.py --suite batch` 覆盖，表格在 `PERFORMANCE_REPORT.md` §6.4/§6.5。
> 该文档 §6.7 附了一份 checklist：若扫描结论与本节不符，必须同时修改
> `gme_openvino.py` / `gme_qwen2vl.hpp` 中 `auto` 的分支与本节文字。

### 6.7 延迟优化边界与建议

GME-Qwen2-VL-2B-Instruct 是约 2.2B 参数、28 层 decoder 的检索模型。即使最终只输出一个 1536 维向量，仍必须完成所有输入 token 的 decoder forward；它不能与数十到数百 MB 的 encoder-only embedding 模型直接比较单请求时延。当前 GPU INT8 短文本 batch=1 约 41 ms，已经接近该模型和本机 iGPU 的合理范围。

当前已采用且建议保留的优化：

* 默认使用 INT8；它在本机保持约 `0.99+` 的图像/图文余弦，并明显降低内存与 GPU 时间。
* 通过 micro-batch 降低每个向量的成本，但**手段按设备不同**（见 6.6）：GPU 上靠 `batch_mode="padded"`（默认自动选中，约 2.4x），CPU 上靠 `streams="auto"` 多流并发（约 1.8x）——在 CPU 上单纯把样本合成一次 padded 调用是 1.00x，没有收益。
* 限制 text/query 的最终 token 长度；decoder attention 随序列长度增长显著增加。两端构造函数现在都能覆盖
  `max_length`（Python 可放大或收缩，因为 language IR 序列维度全动态；C++ 只能收缩，上限是 tokenizer IR 导出时
  烘入的值，见 §4.3/§4.4）。
* 对图片控制 `max_image_tokens`，并以实际检索指标验证质量；视觉 token 同时增加 visual 和 language 的工作量。两端
  构造函数现在都能覆盖 `min_image_tokens`/`max_image_tokens`，纯预处理层面、不需要重新导出（§4.3/§4.4/§4.5）。
* 对 GPU 使用模型缓存并复用同一个 `GmeOpenVINO`/`GmeQwen2VL` 实例；不要为每个请求重新加载或编译 IR。两端构造函数
  现在都能传 `cache_dir` 启用 `ov::cache_dir`：同一 package/device/precision/streams 的下一次编译直接读缓存 blob。
  这对 `gme_embed` 这类短生命周期 CLI 收益最大——每次调用本来都要重新编译；实测 CPU/int8 首次编译 6.0s，命中缓存
  后 2.0s。**"复用同一实例"说的是进程生命周期，不是并发**：两个类都不是线程安全的（`encode_*`/`get_*`
  内部复用同一批 `ov::InferRequest`），多线程场景要么在同一实例上加锁串行调用，要么每个线程/进程各开一个实例
  （可以共用同一个 `cache_dir`，只有第一个真正编译）。见 `gme_qwen2vl.hpp`/`gme_openvino.py` 类注释里的
  "Not thread-safe" 说明。
* **infer request 按需创建，不按 stream 数预建。** request 持有自己的激活（每 stream 约 +0.2 GB），而 pool 的容量
  取自 `OPTIMAL_NUMBER_OF_INFER_REQUESTS`（实测：LATENCY=1，THROUGHPUT=4）。预先建满会让两类常见负载白付内存：
  纯文本任务从不碰视觉塔，`padded` 模式（GPU 默认）也只发一个 language request。Python 的 `_RequestPool._slots`
  一直是懒增长的；C++ 的 `GmeQwen2VL::Pool` 现在同样懒增长，因此 `--streams 0` 的纯文本任务不再预建 4 个视觉 request。
  这不改变任何数值，只降低峰值内存。
* **Python 图片加载现在并行**（`_load_images`，镜像官方 `DataLoader(num_workers=min(cpu_count//2, 8))`）：一批
  图片交给线程池而不是逐张 `load_image()`。图片是本地路径时磁盘 I/O 通常不是瓶胫，但输入是 URL 列表时，逐张加载
  等价于把每次 HTTP round-trip 串行相加；并行加载后单张图片的路径（`len(images)<=1`）直接跳过线程池，避免纯开销。
  `ThreadPoolExecutor.map` 保证按输入顺序返回，因此不需要额外的排序/对齐逻辑；已用磁盘图片和内存中构造的图片
  两种输入验证过顺序保留与逐张编码结果一致（bit-exact）。
* 选定 micro-batch 与 `max_image_tokens` 之前，先在**目标分辨率与文本长度分布**上跑
  `09_benchmark_matrix.py` 并填写 `PERFORMANCE_REPORT.md`。§6.4/§6.5 的数值分别是 1260 视觉 token 的最坏情况图片
  与 ≤256 token 的短句，不能外推到常见图片和 512/1024 token 长文本（常见尺寸约快一个数量级）。

当前不建议为了短文本延迟强行增加的优化：

* CPU 上 INT4 不保证比 INT8 更快，现有基准已经显示 INT4 可能因解压/反量化开销而变慢。
* 当前 language IR 的 text 路径需要空 `image_embeds[0,1536]`；本机 GPU 插件在尝试将这一输入固定为静态 shape 时出现 input-layout 不兼容。因此尚未引入静态 shape cache，避免增加不稳定的专用分支。若以后导出专用 text-only IR，再单独 benchmark 后决定是否加入。
* 不再维护单体 visual+language IR；它不能改善官方 text/query 分支，且峰值资源更高。

### 6.8 备选方案对照：GGUF + llama.cpp（`llamacpp_validation.json`）

用 GME 完整模板 + `--pooling last` 跑 `gme-Qwen2-VL-2B-Instruct-Q4_K_M.gguf`：5 条文本 cosine 均值 **0.9845**、最小 0.9716，且每条最相似的黄金向量均为自身；去掉模板的对照仅 0.328（证明模板必需）。Q4_K_M 精度明显低于 OV INT8，且仅文本（无 mmproj）、当前 build 的 ggml-openvino 后端在 embedding 路径断言失败需 `-dev none -fa off` 强制 CPU。

---

## 7. 量化与精度建议

- **默认候选 INT8**：Python 路径的文本约 `0.998`、图像约 `0.989–0.993`、图文约 `0.996–0.998`；语言 IR 由 5.8G 降到 1.5G，iGPU 更快。是否可接受仍应由业务 Recall@K/nDCG 决定。
- **INT4 语言塔现在是 data-aware（AWQ + Scale Estimation），不再是纯 data-free 权重压缩**：以前显著影响**文本**（≈0.89–0.92）。
  `03_finalize.py` 的 `_calibration_dataset()` 用 32 条自建的、刻意跨长度/语言/领域的校准文本（不是黄金评测集，避免"在测试集上校准"）过一遍
  GME 官方 prompt 模板，喂给 `nncf.compress_weights(..., dataset=..., awq=True, scale_estimation=True)`。视觉塔的 INT4 已经不差（数据显示图像/图文
  反而略好于文本，last-token 池化对逐 patch 噪声更鲁棒），所以只对语言塔做了 data-aware 校准，视觉塔仍保持 data-free（含 rotary 排除）。

  实测（`04_validate_python.py`，cosine vs 官方 FP32 黄金）：

  | 设备 | 精度 | text（旧→新） | query（旧→新） | image（旧→新） | fused（旧→新） |
  | --- | --- | --- | --- | --- | --- |
  | CPU | int8（未变，data-free） | 0.998476 | 0.998361 | 0.992512 | 0.997980 |
  | CPU | int4 | 0.908822 → **0.953891** | 0.912175 → **0.956376** | 0.945496 → **0.952892** | 0.962353 → **0.968193** |
  | CPU | int4_gs64 | 0.921590 → **0.967406** | 0.949724 → 0.956181 | 0.955428 → 0.957057 | 0.967409 → 0.973058 |
  | GPU | int8（未变，data-free） | 0.997518 | 0.998764 | 0.989255 | 0.995806 |
  | GPU | int4 | 0.886320 → **0.944538** | 0.911044 → **0.955146** | 0.936908 → 0.941821 | 0.955260 → 0.954111 |
  | GPU | int4_gs64 | 0.918643 → **0.949893** | 0.950327 → 0.957442 | 0.944975 → 0.943357 | 0.963901 → 0.959970 |

  text/query 的提升最明显（+0.03～+0.06），这正是问题所在的地方；image/fused 也有小幅提升——即使视觉塔权重完全没变，
  因为 image/fused 的最终 embedding 仍要经过（已改进的）语言塔 decode + pooling。INT8 数值逐位不变，符合预期：`compress()`
  对 INT8 仍是 data-free 调用，重新压缩应产出逐字节相同的产物（已用本次实测确认）。**`int4_gs64` 现在把文本 cosine 带到 ≈0.95–0.97**，
  第一次让 INT4 看起来是可以认真考虑的部署选项，而不只是"能跑、最省体积"；业务上线前仍应在真实 query/corpus 上比 Recall@K/nDCG。
- **视觉塔量化的坑（已修复并内置于 `03_finalize.py`）**：朴素权重压缩会把 ViT 的 `rotary_pos_emb`（`aten::outer` MatMul）也量化，破坏旋转位置频率，使视觉输出崩坏（int8/int4 逐 patch cos 都只有 ~0.32）。用 `ignored_scope(patterns=[".*rotary.*"])` 排除后恢复（int8 图像≈0.99）。语言塔无此问题（其 rotary 用 cos/sin buffer，非可量化 MatMul）。
- 嵌入检索对排序敏感：正式选型请在业务 query/corpus 上比 Recall@K / nDCG，而非只看逐向量 cosine。
- **`compress()` 现在按配置内容判断是否需要重新压缩，不再只看文件是否存在**：`03_finalize.py` 会把每次
  `nncf.compress_weights(...)` 调用的完整参数（含 `mode`/`group_size`/`ignored_scope`/`advanced_parameters`,
  连 nncf 自己填的默认值也算在内）算一个哈希，写进 `nncf_fingerprints.json`。之前只检查目标 `.xml` 是否存在，
  于是改了 `group_size` 却忘了删旧文件时，脚本会静默复用改之前的旧产物——量化配方的改动实际上从未生效，直到有人
  发现 `.xml` 的 mtime 比编辑时间还早。首次在新代码下运行 `03_finalize.py` 因为还没有指纹文件，会把已有的
  fp32/int8/int4/int4_gs64/int4_visint8 全部重新压缩一遍来建立基线——这是预期行为，不是回归：这里的
  `compress_weights` 全部是 data-free 调用（`ratio=1.0`，未传 `dataset=`），重新压缩得到逐字节相同的产物。

---

## 8. 方案比较（回答最初的问题）

| 方案 | Intel CPU | Intel GPU | 低比特 | 图像 | 难度 | 评价 |
| --- | --- | --- | --- | --- | --- | --- |
| **OpenVINO + 自研 GME 封装** | ✅ | ✅(iGPU 已验证) | ✅ INT8 近无损/INT4 可用 | ✅ | 中 | **首选，已落地**（C++ 主实现 + Python 导出/参考） |
| PyTorch + Transformers (+ torch XPU) | ✅ | ⚠️ 需 torch-XPU/新驱动，Xe3 新硅需实测 | ⚠️ 用 TorchAO(非 IPEX) | ✅ | 中 | 适合作参考/兜底；必须 `bf16`（`fp16` 会溢出） |
| sentence-transformers | ✅ | ⚠️ 仅透传底层 PyTorch | ❌ | ✅ | 低 | onnx/openvino 后端**不支持**其自定义多模态模块，等价纯 PyTorch，非加速/量化方案 |
| GGUF + llama.cpp | ✅ 文本(0.985) | ❌ 本机 SYCL 未成功；OV 后端 embedding 断言失败 | ✅ Q4_K_M | ❌ 无 mmproj | 中 | 纯文本、极省内存备选；精度低于 OV INT8 |
| IPEX-LLM | — | — | — | — | — | **已 EOL/归档**，不建议新项目 |

为什么 OpenVINO 最合适：GME 的“head”极简，能与 decoder 合成单输出 IR；Intel CPU/iGPU/NPU 一套 runtime；NNCF 量化成熟；本机 iGPU 已被 OpenVINO 识别为 `['CPU','GPU']` 且实测有效。本文没有对 NPU 执行验证，不应将 CPU/GPU 结果外推为 NPU 兼容性结论。

---

## 9. 复现清单

```bash
# 环境：env-ref（Python 导出/量化/校验）；benchmark/openvino/install（C++ 运行时）
./env-ref/bin/python 02_export.py --device CPU      # 若模型包缺失
./env-ref/bin/python 03_finalize.py                 # tokenizer IR + INT8/INT4
./env-ref/bin/python 04_validate_python.py          # Python 精度  -> openvino_full_validation.json
cd cpp && bash build.sh && cd ..                    # 构建 C++
./env-ref/bin/python 05_validate_cpp.py             # C++ 精度     -> cpp_validation.json
./env-ref/bin/python 06_benchmark.py                # 时延         -> benchmark.json
./env-ref/bin/python 06_benchmark.py --sequence-lengths 128,256 --repeat 20
                                                     # text/query 长度扫描 -> benchmark_sequence_lengths.json
./env-ref/bin/python 07_llamacpp_text_check.py      # GGUF 对照    -> llamacpp_validation.json
./env-ref/bin/python 08_validate_batch.py           # batch 准确性 -> batch_validation.json
./env-ref/bin/python 09_benchmark_matrix.py --repeat 20
                                                     # 性能矩阵     -> benchmark_matrix.json（填 PERFORMANCE_REPORT.md）
./env-ref/bin/python experiments/probe_visual_batch.py
                                                     # §2.0.2 的 visual batch 实测证据（不写 artifact）
./env-ref/bin/python experiments/check_image_edge_cases.py
                                                     # §2.0.2.2 图像解码/resize 边界回归（不写 artifact）
```

`08_validate_batch.py` 与 `09_benchmark_matrix.py` 建议限内存运行：

```bash
systemd-run --user --scope -p MemoryMax=20G -p MemorySwapMax=0 \
  ./env-ref/bin/python 08_validate_batch.py
```

若重新导出，注意 `03_finalize.py` 现在从包配置读 `processor.max_length` 并把截断编入 tokenizer IR；
跳过该步骤会让 C++ 在超长输入上与 Python/官方不一致（§2.0.2.1）。

归档复现前置条件：`env-ref` 中的 Transformers 固定为 `4.51.3`；所有 Python 验证和 C++ CLI 均须先加载
`../benchmark/openvino/install/setupvars.sh`，以使用本仓库优化后的 OpenVINO Runtime 与 GPU 插件。若重新导出或更换
OpenVINO/驱动，必须重新执行 §5 的验证和基准，不能沿用本文数值。

官方模型说明：<https://huggingface.co/Alibaba-NLP/gme-Qwen2-VL-2B-Instruct>
