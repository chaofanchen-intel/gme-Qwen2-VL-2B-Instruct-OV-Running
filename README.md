# GME-Qwen2-VL-2B-Instruct OpenVINO

这个目录提供一个可移交的 GME embedding 工作流：从官方 PyTorch 参考向量开始，导出 OpenVINO FP32 IR，生成 tokenizer 和量化模型包，再分别用 Python 与 C++ 运行时做精度、批量和性能验证。

关于 GME 是否支持 Qwen3.5 风格的 PA/SDPA backend 切换，见
[GME_Qwen3VL_Embedding_PA_SDPA_调查.md](./GME_Qwen3VL_Embedding_PA_SDPA_调查.md)。

关于 Intel GPU 上 GME INT4 text/query embedding 的 hotspot 分析、可交付 GPU
plugin patch 和 bit-identical 回归验证，见
[gme-qwen2-vl-int4-hotspot-optimization/README.md](./gme-qwen2-vl-int4-hotspot-optimization/README.md)。

当前交付 package 已包含第四轮 mask-free visual IR。默认精度 key 是
`int8_vnomask`（精度优先）和 `int4_causal_vnomask`（低延迟推荐）；其中
`_vnomask` 只改变 image/fused 使用的 visual IR，纯 text/query 的 language IR
分别仍是普通 INT8 和 causal INT4。当前 fresh GPU 矩阵和 document embedding
结果见 [gme-qwen2-vl-int4-hotspot-optimization/results/README.md](./gme-qwen2-vl-int4-hotspot-optimization/results/README.md)
以及 [document_embedding_eval_int8_int4_gpu/REPORT.md](./document_embedding_eval_int8_int4_gpu/REPORT.md)。

支持四种输入模式：

- `text`：默认 instruction 的文本 embedding；
- `query`：带查询 instruction 的文本 embedding；
- `image`：图片 embedding；
- `fused`：文本与图片融合后的 embedding。

最终输出是 `float32[B, 1536]`，每行已经 L2 normalize，可以直接用内积计算 cosine similarity。

## 设计原则

这里的脚本是**相互独立的 CLI**。不再使用统一的路径配置模块，也不要求设置项目级环境变量。每个脚本都通过自己的参数接收它需要的模型、产物、图片或运行时路径；默认值只用于兼容当前目录布局。

因此，移交给其他项目时推荐显式传路径：脚本可以放在任意工作目录运行，不依赖当前 shell 的 `pwd`。模型包内的 `gme_openvino_config.json` 只描述模型和 processor，不承担外部文件路径配置。

## 文档导航

- [GME_TECHNICAL_DETAILS.md](GME_TECHNICAL_DETAILS.md)：保留完整技术说明，包括四种任务的执行路径、visual/language IR 设计、batch 语义、图像预处理、为什么不能直接使用 OpenVINO GenAI/Optimum、精度结果、性能结果和量化建议；
- [PERFORMANCE_REPORT.md](PERFORMANCE_REPORT.md)：指定文本长度、图片尺寸、设备、精度和 batch 组合的完整性能矩阵；
- [GME_EXECUTIVE_REPORT.md](GME_EXECUTIVE_REPORT.md)：面向交付和部署决策的摘要。

README 本身聚焦于移交时的目录、参数和执行顺序；需要了解“为什么这样实现”或查看历史实测数值时，请进入技术说明文档。

## 架构

```text
GME checkpoint + gme_inference.py
              |
              +-- 01_reference.py
              |       -> golden_text/query/image.npy
              |       -> golden_meta.json
              |
              +-- 02_export.py
                    |       -> gme_language_fp32.xml / gme_language_fp32_causal.xml
                    |       -> gme_visual_fp32.xml / gme_visual_fp32_nomask.xml
              |
              +-- 03_finalize.py
                      -> openvino_tokenizer.xml
                      -> INT8 / INT4 / causal / mask-free IR
                      -> gme_openvino_config.json

Python runtime: gme_openvino.py
  HF processor -> visual IR (image only) -> language IR -> embedding[1536]

C++ runtime: cpp/gme_embed
  native tokenizer + image preprocessing -> visual IR -> language IR -> embedding[1536]
```

`gme_language` 包含 token embedding、视觉 token 注入、decoder、last-valid-token pooling 和 L2 normalize。`gme_visual` 只负责视觉编码。纯文本不会调用视觉 IR。当前 image/fused 默认使用 mask-free visual IR：它去掉了对单图路径恒为全 True 的 rank-3 BOOL attention mask，使视觉 SDPA 走优化 kernel。视觉 IR 按单图调用，这是为了适配 Qwen2-VL 导出图中按 `image_grid_thw` 进行的 Python 控制流；运行时仍然支持任意数量、任意分辨率的图片输入。

## 目录结构

```text
gme-Qwen2-VL-2B-Instruct-Running/
├── 01_reference.py             # 官方 PyTorch 参考向量
├── 02_export.py                # FP32 visual/language IR 导出
├── 03_finalize.py              # tokenizer IR、INT8/INT4、package config
├── 04_validate_python.py       # Python OpenVINO 精度校验
├── 05_validate_cpp.py          # C++ 与 golden 对照
├── 06_benchmark.py             # 固定 workload 时延
├── 07_validate_batch.py        # batch/API/截断/同形状回归
├── 08_benchmark_matrix.py      # token、图片尺寸和 batch 性能矩阵
├── 09_retest_gpu_1024.py       # GPU 1024-token 交错复测
├── gme_openvino.py             # Python OpenVINO runtime
├── cpp/                        # C++ runtime 和 build.sh
├── experiments/                # 边界与 visual batch 诊断
├── ov_models/                  # 导出的模型包
└── artifacts/                  # golden、dump、报告和 benchmark JSON
```

## 环境要求

导出和 Python 校验需要一个包含以下依赖的 Python 环境：

- PyTorch；
- `transformers==4.51.3` 或由 `gme_compat.py` 适配的兼容版本（当前第四轮 package 使用 5.2.0 导出）；
- OpenVINO Runtime；
- NNCF；
- `openvino-tokenizers`；
- NumPy、Pillow、Requests。

仓库现有的 `env-ref/` 可以继续使用。C++ 构建需要一个 OpenVINO 安装目录，其中至少有：

```text
<openvino-install>/setupvars.sh
<openvino-install>/lib/cmake/opencv4/
<openvino-install>/lib/libopenvino.so
```

## 参数和脚本职责

每个脚本都可以用 `--help` 查看完整参数。最重要的路径参数如下：

| 脚本 | 关键参数 |
| --- | --- |
| `01_reference.py` | `--source-model`、`--image`、`--artifacts` |
| `02_export.py` | `--source-model`、`--example-image`、`--output`、`--runtime-module` |
| `03_finalize.py` | `--source-model`、`--package`、`--artifacts` |
| `04_validate_python.py` | `--package`、`--artifacts`、`--source-model`、`--output` |
| `05_validate_cpp.py` | `--package`、`--artifacts`、`--image`、`--cpp-binary`、`--openvino-install` |
| `06_benchmark.py` | 上述运行时路径，加 `--backends`、`--devices`、`--precisions`、`--output` |
| `07_validate_batch.py` | `--package`、`--artifacts`、`--source-model`、`--cpp-binary`、`--openvino-install` |
| `08_benchmark_matrix.py` | `--package`、`--artifacts`、`--logo`、`--cpp-binary`、`--openvino-install` |
| `09_retest_gpu_1024.py` | `--package`、`--workload-source`、`--cpp-binary`、`--openvino-install`、`--output` |

`--source-model` 必须指向 GME 微调后的 checkpoint，而不是普通的 Qwen2-VL 基础模型；目录中需要有官方 `gme_inference.py` 和模型文件。`--runtime-module` 通常指向本目录的 `gme_openvino.py`，导出时会把它复制到模型包中。

## 标准移交流程

先在 shell 中定义路径变量只是为了让命令易读；它们不是项目配置文件，也不会被脚本自动读取。

```bash
RUN=/path/to/gme-Qwen2-VL-2B-Instruct-Running
MODEL=/path/to/gme-Qwen2-VL-2B-Instruct
OV=/path/to/openvino/install
PKG="$RUN/ov_models/gme-qwen2-vl-2b-instruct-openvino"
ART="$RUN/artifacts"
PY=/path/to/python
```

### 1. 生成 golden reference

```bash
"$PY" "$RUN/01_reference.py" \
  --source-model "$MODEL" \
  --image "$MODEL/images/gme_logo.png" \
  --artifacts "$ART"
```

输出：`golden_text.npy`、`golden_query.npy`、`golden_image.npy` 和 `golden_meta.json`。

### 2. 导出 FP32 OpenVINO IR

```bash
"$PY" "$RUN/02_export.py" \
  --source-model "$MODEL" \
  --example-image "$MODEL/images/gme_logo.png" \
  --runtime-module "$RUN/gme_openvino.py" \
  --output "$PKG" \
  --device CPU \
  --stage all
```

`--stage all` 会自动把 language、visual、validate 分成独立子进程，降低导出峰值内存。也可以单独运行 `--stage language`、`--stage visual` 或 `--stage validate`。

### 3. 生成 tokenizer 和量化模型包

```bash
"$PY" "$RUN/03_finalize.py" \
  --source-model "$MODEL" \
  --package "$PKG" \
  --artifacts "$ART"
```

默认生成 `fp32`、`int8`、`int4`、`int4_gs64`、causal 语言 IR、mask-free visual IR 和 `int4_visint8`。package 默认精度是 `int8_vnomask`；`int4_causal_vnomask` 是当前低延迟推荐组合。`--force` 用于忽略 NNCF 指纹并重新量化；正常重复运行会跳过配置未变化的产物。

### 4. Python 精度校验

```bash
"$PY" "$RUN/04_validate_python.py" \
  --package "$PKG" \
  --artifacts "$ART" \
  --source-model "$MODEL" \
  --devices CPU,GPU
```

如果 `golden_fused.npy` 不存在，脚本会用 `--source-model` 生成并缓存它。只测 CPU 时可以使用 `--devices CPU`；只测部分量化精度可以使用 `--precisions int8_vnomask,int4_causal_vnomask`。

### 5. 构建 C++ runtime

```bash
bash "$RUN/cpp/build.sh" \
  --openvino-install "$OV" \
  --build-dir "$RUN/cpp/build" \
  --jobs "$(nproc)"
```

构建脚本也支持 `OV_INSTALL` 环境变量作为备用方式，但移交时推荐使用 `--openvino-install`。构建完成后，可执行文件默认是 `cpp/build/gme_embed`。

### 6. C++ 精度校验

```bash
"$PY" "$RUN/05_validate_cpp.py" \
  --package "$PKG" \
  --artifacts "$ART" \
  --image "$MODEL/images/gme_logo.png" \
  --cpp-binary "$RUN/cpp/build/gme_embed" \
  --openvino-install "$OV" \
  --devices CPU,GPU
```

脚本会为每个设备和精度启动 C++ CLI，并把临时 dump 放在 `artifacts/cpp_dumps/`。C++ image/fused 路径使用 OpenCV，和 Python/Pillow 的 resize 可能有小幅差异；这部分应使用报告中的阈值解释，不能按 text/query 的 FP32 精确标准判断。

### 7. 固定 workload benchmark

```bash
"$PY" "$RUN/06_benchmark.py" \
  --package "$PKG" \
  --artifacts "$ART" \
  --image "$MODEL/images/gme_logo.png" \
  --cpp-binary "$RUN/cpp/build/gme_embed" \
  --openvino-install "$OV" \
  --backends python,cpp \
  --devices CPU,GPU \
  --precisions int8_vnomask,int4_causal_vnomask \
  --repeat 20
```

`--backends python` 可在没有 C++ 构建产物时运行；`--sequence-lengths 128,256,512,1024` 可只测指定的 text/query 长度。报告默认写入 `artifacts/benchmark.json`。

### 8. Batch/API 回归

```bash
"$PY" "$RUN/07_validate_batch.py" \
  --package "$PKG" \
  --artifacts "$ART" \
  --source-model "$MODEL" \
  --image "$MODEL/images/gme_logo.png" \
  --cpp-binary "$RUN/cpp/build/gme_embed" \
  --openvino-install "$OV"
```

该脚本检查任意 batch 数量、`per_sample`/`padded`、streams 不变量、官方 API alias、超长截断、同形状样本不重复，以及 Python/C++ 对照。它会在 `artifacts/` 生成临时 probe 图片，成功后自动删除。

### 9. 性能矩阵和 GPU 复测

```bash
"$PY" "$RUN/08_benchmark_matrix.py" \
  --package "$PKG" \
  --artifacts "$ART" \
  --logo "$MODEL/images/gme_logo.png" \
  --cpp-binary "$RUN/cpp/build/gme_embed" \
  --openvino-install "$OV" \
  --suite latency,text-lengths \
  --devices GPU \
  --precisions int8_vnomask,int4_causal_vnomask \
  --repeat 20

"$PY" "$RUN/09_retest_gpu_1024.py" \
  --package "$PKG" \
  --artifacts "$ART" \
  --workload-source "$ART/benchmark_matrix_text_lengths_r20.json" \
  --cpp-binary "$RUN/cpp/build/gme_embed" \
  --openvino-install "$OV" \
  --device GPU
```

矩阵脚本的 `--suite` 可选 `latency`、`text-lengths`、`image-sizes`、`batch`，建议先缩小 `--devices`、`--precisions`、`--repeat` 验证参数，再运行完整矩阵。GPU 复测脚本会交错启动新的 C++/Python session，专门观察 1024 token 下的稳态时延。

## 直接调用 runtime

### Python

```python
from gme_openvino import GmeOpenVINO

encoder = GmeOpenVINO(
    "/path/to/gme-qwen2-vl-2b-instruct-openvino",
    device="GPU",
    precision="int8_vnomask",
)
text_embeddings = encoder.get_text_embeddings(["a compact electric vehicle"])
query_embeddings = encoder.get_text_embeddings(
    ["What kind of car is this?"],
    instruction="Find an image that matches the given text.",
)
image_embeddings = encoder.get_image_embeddings(["/path/to/image.png"])
fused_embeddings = encoder.get_fused_embeddings(
    ["a compact electric vehicle"], ["/path/to/image.png"]
)
```

Python 图片输入支持本地路径、HTTP(S)、`file://`、base64 data URL 和 PIL Image。`streams` 控制吞吐，`batch_mode` 可选 `auto`、`per_sample`、`padded`；服务端应复用长期存活的 encoder，避免重复编译模型。

### C++

```bash
source "$OV/setupvars.sh"
BIN="$RUN/cpp/build/gme_embed"

"$BIN" --package "$PKG" --device GPU --precision int8_vnomask \
  --mode text --is-query 0 "a compact electric vehicle"

"$BIN" --package "$PKG" --device GPU --precision int8_vnomask \
  --mode image --image "/path/to/image.png"

"$BIN" --package "$PKG" --device GPU --precision int8_vnomask \
  --mode fused --image "/path/to/image.png" "a compact electric vehicle"
```

C++ 输入图片是本地路径；网络图片需要由调用方先下载。使用 `--out output.bin` 可写出 `GEMB` 二进制 embedding dump，使用 `--benchmark` 可测 warm latency。

## 模型包和量化

`03_finalize.py` 写入的 `gme_openvino_config.json` 记录 processor 参数、embedding 维度和每种 precision 使用的 language/visual IR。视觉量化会排除 rotary 相关常量；language INT4 使用 calibration、AWQ 和 scale estimation。量化不会改变输出维度，但会改变检索质量，必须在业务数据上评估 Recall@K、nDCG 或 Top-K 命中率。

建议的初始部署配置：

```text
GPU: precision=int8_vnomask, batch_mode=padded, streams=1
CPU: precision=int8_vnomask, batch_mode=per_sample, streams=auto
GPU text low latency: precision=int4_causal_vnomask, batch_mode=padded, streams=1
```

## 已知限制

1. C++ 使用 OpenCV、Python 使用 Pillow，image/fused 的 resize 结果可能产生可测但有限的 cosine 差异；
2. GPU 长序列时延受 DVFS、驱动和功耗状态影响，单次 benchmark 不能作为 SLA；
3. GPU 首次 compile 可能较慢，应复用 encoder 或配置 OpenVINO compiled-model cache；
4. INT4 的文本 cosine 通常低于 INT8，是否可用必须由业务检索评估决定；
5. 当前验证覆盖 CPU/GPU，不应把结果外推到未验证设备；
6. `GmeOpenVINO` 实例内部复用 infer request，同一个实例不应被多个线程无锁并发调用。

## 诊断实验

两个实验脚本同样是独立 CLI：

```bash
"$PY" "$RUN/experiments/check_image_edge_cases.py" \
  --source-model "$MODEL" --package "$PKG" \
  --cpp-binary "$RUN/cpp/build/gme_embed" --openvino-install "$OV"

"$PY" "$RUN/experiments/probe_visual_batch.py" \
  --source-model "$MODEL" --package "$PKG" \
  --logo "$MODEL/images/gme_logo.png" --only q1 q2
```

前者覆盖 EXIF、极小图片、极端长宽比和 half-factor 取整；后者验证 visual IR 为什么按单图驱动。两者不会修改模型包。