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
- ⚙️ 图像时延与图像 token 数强相关（注意力 ∝ patch²）：正常图（≈256 token）GPU-INT8 约 0.46 s；本仓库的 `gme_logo`
  是超宽横幅（1260 token，接近上限）→ 属最坏情况（见 §6.4）。

---

## 2. 架构

官方 `gme_inference.py` 的计算路径（GME 没有额外可训练 head，“head” 就是 last-token 池化 + L2 归一化）：

```
text : GME prompt 模板 -> tokenizer(右填充) -> token embedding -> GME Qwen2-VL decoder
       -> 取最后有效 token(attention_mask.sum(-1)-1) -> L2 归一化 -> 1536 维
image/fused: 图像 resize + Qwen2-VL 预处理 -> 视觉编码器 -> 视觉 token 注入到 <|image_pad|>
       -> 同一个 GME decoder -> 最后有效 token -> L2 归一化
```

导出为三个可组合 OpenVINO IR（`ov_models/gme-qwen2-vl-2b-instruct-openvino/`）：

| IR | 输入 → 输出 | 说明 |
| --- | --- | --- |
| `openvino_tokenizer.xml` | string → `input_ids`,`attention_mask` | openvino-tokenizers；右填充；与 HF **逐 token 一致** |
| `gme_visual_<prec>.xml` | `pixel_values`,`image_grid_thw` → 视觉 token 向量 | Qwen2-VL ViT |
| `gme_language_<prec>.xml` | `input_ids`,`attention_mask`,`image_embeds` → `embedding[B,1536]` | token embedding + 视觉注入(masked_scatter) + decoder + last-token 池化 + L2 归一化 |

- 拆 visual / language 两个 IR 是为了避免整图转换时的内存峰值；纯文本给 language IR 传空的 `[0,1536]` 视觉张量（masked_scatter 变空操作）。
- **视觉注入关键点**：HF processor 会按图像网格把单个 `<|image_pad|>` **展开**成 K 个（K = `grid_t*grid_h*grid_w/merge²` = 视觉合并 token 数）。C++ 端在分词前必须同样展开，否则 image_pad 数与视觉输出行数对不上，图像向量会崩（FP32 也会掉到 ~0.29）。

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

Python API 返回 `numpy.ndarray[B,1536]`，即使输入只有一个样本也保留 batch 维度；C++ API 返回 `gme::Embedding`，其中 `rows=B`、`dim=1536`、`data` 为行主序的 `float` 数组，长度为 `B*1536`。当前图像/图文 API 为保持一图一输入语义，内部逐样本计算后再沿 batch 维拼接，但最终输出契约不变。

**1536 是模型架构固定的 embedding 维度，不是运行时可调参数。** 当前 checkpoint 的 language hidden size 为 1536，last-token pooling 直接输出该 hidden state，没有额外 projection head。`fp32`、`int8`、`int4`、`int4_gs64` 和 `int4_visint8` 只改变权重存储/计算精度，输出仍是 `float32[B,1536]`。`max_length`、文本 token 长度、`min_image_tokens`、`max_image_tokens`、图像分辨率和 batch size 会影响输入 shape、内存和时延，但**不会**改变 1536 维输出。

若必须使用其他维度，不能只改配置文件或对 IR reshape；需要选用原生维度不同的模型，或训练并部署新的投影层。后者会产生新的 embedding space，不能与现有 1536 维 GME 向量直接做相似度检索，且需要重新进行精度和检索指标验证。

---

## 3. 目录结构

```
gme-Qwen2-VL-2B-Instruct-Running/
├─ README.md                    # 本文档
├─ env-ref/                     # Python 参考/导出环境 (transformers==4.51.3, openvino 2026.2, nncf, ov-tokenizers)
├─ gme_openvino.py              # Python 运行时 GmeOpenVINO（与 C++ 等价，用于对照/调试）
├─ 01_reference.py              # 用官方实现生成 FP32 黄金向量 + 模板/token 记录
├─ 02_export.py                 # 从原始 GME checkpoint 导出 FP32 visual/language IR（分进程限内存）
├─ 03_finalize.py               # 转 tokenizer IR(逐token校验) + NNCF INT8/INT4(含视觉,排除rotary) + 写 config
├─ 04_validate_python.py        # Python OV 精度：全精度 × 全模式 × CPU/GPU
├─ 05_validate_cpp.py           # 驱动 C++ gme_embed，与黄金向量比对
├─ 06_benchmark.py              # Python + C++ 时延基准（fp32/int8/int4 × 模式 × CPU/GPU）
├─ 07_llamacpp_text_check.py    # GGUF/llama.cpp 文本向量对照（备选方案评估）
├─ cpp/                         # C++ 运行时（主实现）
│  ├─ include/gme_qwen2vl.hpp   #   GmeQwen2VL 类
│  ├─ src/gme_qwen2vl.cpp       #   tokenizer/visual/language IR 编排 + prompt 模板 + image_pad 展开
│  ├─ src/image_preprocess.*    #   Qwen2-VL 图像预处理 (OpenCV: smart_resize+normalize+patchify)
│  ├─ src/main.cpp              #   gme_embed CLI（--mode / --benchmark / --out）
│  ├─ CMakeLists.txt, build.sh  #   对接 benchmark/openvino/install
│  └─ build/gme_embed           #   构建产物
├─ ov_models/gme-qwen2-vl-2b-instruct-openvino/   # 模型包（3类IR × 各精度 + tokenizer IR + 配置）
└─ artifacts/                   # 黄金向量 + 各项验证/基准 JSON
```

模型包内精度（`gme_openvino_config.json` 的 `precisions`，`default_precision=int8`）：

| 精度键 | language IR | visual IR | 语言/视觉 .bin 体积 |
| --- | --- | --- | --- |
| `fp32` | fp32 | fp32 | 5.8G / 2.5G |
| `int8` | INT8 | INT8(排除rotary) | 1.5G / 638M |
| `int4` | INT4 gs128 | INT4 gs128(排除rotary) | 880M / 336M |
| `int4_gs64` | INT4 gs64 | INT4 | 904M / 336M |
| `int4_visint8` | INT4 | INT8 | 880M / 638M |

---

## 4. 操作步骤

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

Python 端等价 API 见 `gme_openvino.py`（`GmeOpenVINO(...).get_text_embeddings / get_image_embeddings / get_fused_embeddings / encode / encode_queries / encode_corpus`）。

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
- **备选方案对照**：`07_llamacpp_text_check.py`，用完整 GME 模板 + `--pooling last` 跑 GGUF 文本向量 → `artifacts/llamacpp_validation.json`。

---

## 6. 测试结果（本机 Core Ultra 7 356H，Linux，OpenVINO 2026.2）

### 6.1 Python OpenVINO 权重与数值一致性（FP32 cosine vs 官方 PyTorch 黄金）

| 设备 | text | query | image | fused |
| --- | ---: | ---: | ---: | ---: |
| CPU | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| GPU | 1.000000 | 1.000000 | 1.000000 | 1.000000 |

→ Python OpenVINO 运行时将三个 IR 组合后，在 FP32 下与原始 `benchmark/models/gme-Qwen2-VL-2B-Instruct` 的四种 embedding 模式数值等价。

### 6.2 精度：Python OpenVINO（cosine vs FP32 黄金，`openvino_full_validation.json`）

| 设备 | 精度 | text | query | image | fused |
| --- | --- | ---: | ---: | ---: | ---: |
| CPU | fp32 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| CPU | **int8** | 0.998315 | 0.998361 | 0.992512 | 0.997980 |
| CPU | int4 | 0.907792 | 0.912175 | 0.945496 | 0.962353 |
| CPU | int4_gs64 | 0.923465 | 0.949724 | 0.955428 | 0.967409 |
| CPU | int4_visint8 | 0.907792 | 0.912175 | 0.961537 | 0.957409 |
| GPU | fp32 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| GPU | **int8** | 0.997518 | 0.998764 | 0.989255 | 0.995806 |
| GPU | int4 | 0.886320 | 0.911044 | 0.936908 | 0.955260 |
| GPU | int4_gs64 | 0.918643 | 0.950327 | 0.944975 | 0.963901 |
| GPU | int4_visint8 | 0.886320 | 0.911044 | 0.947935 | 0.968358 |

### 6.3 精度：C++（cosine vs FP32 黄金，`cpp_validation.json`）

| 设备 | 精度 | text | query | image | fused |
| --- | --- | ---: | ---: | ---: | ---: |
| CPU | fp32 | **1.000000** | **1.000000** | 0.975047 | 0.988341 |
| CPU | int8 | 0.998315 | 0.998361 | 0.971103 | 0.985380 |
| CPU | int4 | 0.907792 | 0.912175 | 0.943903 | 0.957391 |
| GPU | fp32 | **1.000000** | **1.000000** | 0.975043 | 0.988341 |
| GPU | int8 | 0.997518 | 0.998764 | 0.967116 | 0.981741 |
| GPU | int4 | 0.886320 | 0.911044 | 0.930794 | 0.948113 |

- 文本/query 与 Python 完全一致（tokenizer IR 逐 token 一致，语言 IR 相同）。
- 图像/图文比 Python 低约 `0.02` / `0.012`。当前 C++ 预处理使用 OpenCV，参考实现使用 Pillow；C++ 已采用缩小时 `INTER_AREA`、放大时 `INTER_CUBIC` 来逼近 Pillow bicubic，但这不是逐像素等价实现。`gme_logo` 的锐利边缘和 1260 个合并视觉 token 使该用例对 resize 差异敏感。
- 因此，C++ 图像路径的验收含义是“可运行且差异被记录”，不是 FP32 参考级数值等价。对图像检索精度有严格要求时，应使用 Python runtime，或先以 Pillow 等价 resize 替换/验证 C++ 预处理，再提高 C++ image/fused 的验收阈值。

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

这些结果是 `gme_logo.png` 的端到端测量，包含分词/图像预处理和推理，不含模型加载与编译。该图片产生 1260 个合并视觉 token，接近默认上限 1280；它说明最坏情况下的规模，**不能**作为普通分辨率图片的服务时延承诺。图像时延主要由视觉塔注意力和 token 数决定；如需更低时延，应先在目标图片分辨率分布上重跑 `06_benchmark.py`，再评估降低 `max_image_tokens` 的质量影响。

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

### 6.6 备选方案对照：GGUF + llama.cpp（`llamacpp_validation.json`）

用 GME 完整模板 + `--pooling last` 跑 `gme-Qwen2-VL-2B-Instruct-Q4_K_M.gguf`：5 条文本 cosine 均值 **0.9845**、最小 0.9716，且每条最相似的黄金向量均为自身；去掉模板的对照仅 0.328（证明模板必需）。Q4_K_M 精度明显低于 OV INT8，且仅文本（无 mmproj）、当前 build 的 ggml-openvino 后端在 embedding 路径断言失败需 `-dev none -fa off` 强制 CPU。

---

## 7. 量化与精度建议

- **默认候选 INT8**：Python 路径的文本约 `0.998`、图像约 `0.989–0.993`、图文约 `0.996–0.998`；语言 IR 由 5.8G 降到 1.5G，iGPU 更快。是否可接受仍应由业务 Recall@K/nDCG 决定。
- **INT4**：显著影响**文本**（≈0.89–0.92），图像/图文反而略好（last-token 池化对逐 patch 噪声更鲁棒）。`int4_gs64` 把文本恢复到 ≈0.92–0.95。**INT4 = 可跑、最省体积，但检索质量需重新评测后再上线**。
- **视觉塔量化的坑（已修复并内置于 `03_finalize.py`）**：朴素权重压缩会把 ViT 的 `rotary_pos_emb`（`aten::outer` MatMul）也量化，破坏旋转位置频率，使视觉输出崩坏（int8/int4 逐 patch cos 都只有 ~0.32）。用 `ignored_scope(patterns=[".*rotary.*"])` 排除后恢复（int8 图像≈0.99）。语言塔无此问题（其 rotary 用 cos/sin buffer，非可量化 MatMul）。
- 嵌入检索对排序敏感：正式选型请在业务 query/corpus 上比 Recall@K / nDCG，而非只看逐向量 cosine。

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
```

归档复现前置条件：`env-ref` 中的 Transformers 固定为 `4.51.3`；所有 Python 验证和 C++ CLI 均须先加载
`../benchmark/openvino/install/setupvars.sh`，以使用本仓库优化后的 OpenVINO Runtime 与 GPU 插件。若重新导出或更换
OpenVINO/驱动，必须重新执行 §5 的验证和基准，不能沿用本文数值。

官方模型说明：<https://huggingface.co/Alibaba-NLP/gme-Qwen2-VL-2B-Instruct>
