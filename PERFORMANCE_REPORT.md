# GME-Qwen2-VL-2B-Instruct OpenVINO 性能测试报告

**状态：已完成实测（2026-08-03）。** 所有时延数据均来自 `09_benchmark_matrix.py` 的 `repeat=20` 结果；
原始 JSON 保存在 `artifacts/benchmark_matrix_*_r20*.json`。**不要**引用 README §6.4/§6.5 的旧数值来填本表：
那些是另一套 workload（`gme_logo.png` + 27 token 短句），与本报告的常见尺寸/长文本口径不可比。

本报告与 README §6.4–6.6 的关系：README 记录**固定参考 workload** 的历史结果（用于回归对比）；本报告覆盖
**部署选型真正需要的维度**——指定 token 长度的文本、常见分辨率的图片、以及 batch 扩展性。

---

## 1. 为什么需要这份报告

README §6.4 的 image/fused 时延全部基于仓库自带的 `gme_logo.png`。该文件是 `5164x600` 的超宽横幅，
经 Qwen2-VL processor 后得到 `grid=[1,24,210]`、**1260 个视觉 token**，逼近 GME 配置上限 `max_image_tokens=1280`。
视觉注意力随 patch 数约二次增长，因此它测的是**最坏情况**，不能作为常规图片的服务时延依据。

量级差距（GPU/INT8，`--repeat 1` 预热校准，仅用于判断量级，**不得作为结果引用**）：

| 图片 | 视觉 token | image 时延（量级） |
| --- | ---: | ---: |
| `448x448`（本报告基准尺寸） | 256 | ~0.55 s |
| `gme_logo.png`（README §6.4） | 1260 | ~5.2 s |

约 **9x** 差距。同理，README 的 text 基准是 27 token 的短句，而检索场景常见的 512/1024 token 文本会显著更慢。

---

## 2. 环境记录（填表前必须填写）

| 项 | 值 |
| --- | --- |
| 主机 / CPU | Intel Core Ultra 7 356H（16 核） |
| iGPU | Intel Graphics (iGPU)，`ov.Core().available_devices = ['CPU', 'GPU']`，GPU memory property = 30.7 GB |
| 内存 / swap | 30 GiB / 8 GiB；batch scope `MemoryMax=20G` |
| OS / 内核 | Linux 7.0.0-28-generic |
| OpenVINO | 2026.2.1（`benchmark/openvino/install`） |
| GPU 驱动 / compute-runtime | Intel media/compute stack；PCI GPU = Intel device `b0a0`；OpenVINO GPU plugin 可用 |
| Python env | `env-ref`（transformers 4.51.3） |
| 模型包 | `ov_models/gme-qwen2-vl-2b-instruct-openvino` |
| `--repeat` | 20（所有写入表格的性能数字） |
| 电源策略 / governor | `powersave` |
| 测试时间 | 2026-08-01 至 2026-08-03（CST） |

> 采集条件：设备空闲、固定电源策略、无并发负载。跨设备或跨驱动的数值不可混填到同一张表。

---

## 3. 测量口径

- **warm、端到端、每次调用**：包含分词/图像预处理 + 全部 IR 推理；**不含**一次性模型加载与编译。
  每个单元格先跑 1 次 warmup，再取 `--repeat` 次的平均值。
- **两个数字**：`ms/call` 是调用方实际等待的时间；`ms/emb` = `ms/call ÷ N`，是跨 batch 比较**唯一**有意义的指标。
  表格中记为 `ms/call (ms/emb)`。
- **`--repeat` 要求**：探索用 5；**任何写入本报告的数字用 ≥20**。README §6.5 已记录一次反例：
  C++ CPU INT8 在 256 token 上 `repeat=5` 得到 `920 / 668 ms` 的虚假差异，`repeat=20` 复测为 `669 / 669 ms`。
- **序列长度的定义**：`sequence_tokens` 是 **GME 模板展开后的有效未 padding token 数**，包含 system/user/assistant
  模板、`<|endoftext|>`，以及 image/fused 模式下展开的 K 个 image-pad token。它不是原始字符串长度，也不是裸文本 token 数。
- **图片尺寸的定义**：见 §4.2。视觉 token 数 `K = grid_h * grid_w / 4`，是图像时延的主导变量，**不是**像素数。
- 本报告只测时延/吞吐。精度归 `04_validate_python.py` / `05_validate_cpp.py`，batch 正确性归 `08_validate_batch.py`；
  **不要**用性能结论替代精度结论。

---

## 4. 工作负载定义（已确定，无需测量）

### 4.1 text / query

`09_benchmark_matrix.py` 用实际 tokenizer 反复构造 ASCII filler 并**校验**最终 token 数精确命中目标：

| 目标 sequence_tokens | text filler 字符数 | query filler 字符数 | query instruction |
| ---: | ---: | ---: | --- |
| 32 | 23 | 17 | `Find an image that matches the given text.` |
| 128 | 215 | 209 | 同上 |
| 512 | 983 | 977 | 同上 |
| 1024 | 2007 | 2001 | 同上 |

`query` 的 filler 比 `text` 短，是因为 instruction 本身占掉了 system 段的 token；两者的**最终**长度都精确等于目标值。
包内 `max_length=1800`，因此 1024 是留有余量的最大整齐档位；超过 1800 的输入会被截断（两端一致，见 README §2.0.5）。

### 4.2 image / fused

全部尺寸都取 28 的倍数，使 GME `fetch_image` 与 Qwen2-VL processor **都不再缩放**，于是
`grid = (H/14, W/14)`、`K = H*W/784` 完全可预测。图片由脚本生成为确定性梯度图（内容不影响时延，
且避免暗示这些数字与 logo 的最坏情况可比）：

| 输入尺寸 | processor 后 | `grid_thw` | raw patch 行 | 视觉 token K | image 序列 | fused 序列（32 tok 文本） | 代表性 |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `448x448` | 448x448 | `[1,32,32]` | 1024 | **256** | 278 | 290 | **下界**：`min_image_tokens=256`，更小的图会被放大到同样的 256 token，不可能更快 |
| `644x476` | 644x476 | `[1,34,46]` | 1564 | 391 | 413 | 425 | VGA 量级、非正方 |
| `672x672` | 672x672 | `[1,48,48]` | 2304 | 576 | 598 | 610 | 中等 |
| `896x896` | 896x896 | `[1,64,64]` | 4096 | 1024 | 1046 | 1058 | 大图 |
| `1288x728` | 1288x728 | `[1,52,92]` | 4784 | 1196 | 1218 | 1230 | 720p 量级 |
| `gme_logo.png` | 5152x588 | `[1,24,210]` | 5040 | 1260 | 1282 | 1294 | README §6.4 的最坏情况（`--include-logo` 才测） |

> ⚠️ `448x448` 是硬下界。任何 ≤448x448 的输入（`224x224`、`336x336` 等）都会被 processor 放大到 256 视觉 token，
> 时延与 `448x448` 相同。**不要**在报告里为它们单独开行，否则会让读者误以为小图更便宜。

### 4.3 batch、batch_mode 与 streams

这一节的三个词分别处在不同层次，先给出最重要的定义：

| 名称 | 它控制什么 | 它不是什么 |
| --- | --- | --- |
| `N` | 一次公开 API 调用包含多少条样本；这才是 batch size | 不是 IR 的固定 batch 维，也不是 `streams` 数 |
| `batch_mode` | language IR 如何消费这 N 条样本 | 不是“是否 batch”的开关；`per_sample` 和 `padded` 都是 batch |
| `streams` | OpenVINO plugin 同时准备/执行多少个 infer request，以及设备如何调度它们 | 不是输入样本数；增加 streams 不会改变返回行数 |

本报告使用 `N ∈ {1,2,4,8,16,32}` 测 text，使用 `N ∈ {1,2,4,8}` 测 image。
batch 内样本内容相同，这是有意的：唯一的自变量是**一次调用携带多少条样本**，而不是 ragged 程度。
ragged 输入的正确性由 `08_validate_batch.py` 覆盖。

#### N 到底表示什么

例如：

```python
encoder.get_text_embeddings([text_a, text_b, text_c, text_d])
```

这是一次 batch 调用，`N=4`，返回 `[4,1536]`。无论内部选择哪种 `batch_mode`，调用方都只等待一次，
`ms/call` 是这四条样本全部完成的总时间；`ms/emb = ms/call / 4` 才是平均每条样本的成本。
所以 `N=4` 仍然是 batch，即使内部 `per_sample` 把它拆成四个单样本 request。

`batch_size` 是另一个概念：它只是 API 分块上限。例如输入 100 条、`batch_size=32` 会拆成多个公开调用；
本报告 batch 扫描使用 `--batch-size 0`，表示 1 次调用内不再分块，专门观察 N 的影响。

#### `per_sample`：一次 batch，内部 N 个真实长度 request

```text
API call: [sample A, sample B, sample C, sample D]  N=4
            |          |          |          |
language IR: [1,Sa]    [1,Sb]     [1,Sc]     [1,Sd]
```

`per_sample` 先把 right-padding 去掉，再为每条样本建立一个 `[1,S_i]` language request。
因此它的特点是：

- 仍然只有一次 API 调用，返回 N 行 embedding；
- 每条样本按自己的真实长度计算，不为其他样本补 padding；
- batch 结果与逐条调用逐位一致，适合需要稳定行级结果的 CPU 路径；
- `streams=1` 时这些 request 通常串行消耗设备；`streams="auto"` 时 plugin 可以让多个 request 重叠执行。

#### `padded`：一次 batch，一次 `[N,S]` language IR request

```text
API call: [sample A, sample B, sample C, sample D]  N=4
            |          |          |          |
language IR:          [4, S=max(Sa,Sb,Sc,Sd)]
                      shorter rows use padding
```

`padded` 保留 processor 生成的 `[N,S]` 右填充张量，一次把整个 batch 送进 language IR。
它的特点是：

- 仍然是一次 batch API 调用，但 language IR 只提交一个 `[N,S]` request；
- 如果样本长度不同，所有短样本都会补到同批最长长度，padding 也会进入计算图；
- 结果与逐条调用通常会有极小数值漂移，因为同一行是在不同的 padding/批量 shape 下计算；
- GPU 上可以摊薄每次 infer 提交和调度的固定开销，因此通常比 N 个 per-sample request 快；
- CPU 上单个 request 已经能占满核心，合并成 padded request 通常不会带来收益。

本报告的 batch 样本长度相同，所以 padded 的 padding 浪费被刻意消除；真实线上 ragged 输入通常会更偏向
`per_sample`，除非 GPU 的提交开销明显大于额外 padding 计算。

#### `streams` 是什么

`streams` 是 OpenVINO plugin 的执行并发度提示：它决定 compiled model 可以同时拥有多少个可调度的 infer request，
而不是决定一次输入有多少条样本。

- `streams=1`：LATENCY 模式，倾向于让单条 request 获得完整设备资源，单样本延迟最低；N>1 的 `per_sample` 通常近似逐条排队。
- `streams="auto"`：THROUGHPUT 模式，把流数交给 plugin 选择；本机 CPU 实测为约 4 个流，多个 per-sample request 可以重叠，
  但单样本会因为资源共享而变慢。GPU 上 plugin 也可能选择并发 request，但它不能替代 padded 对提交开销的合并。
- `streams` 不改变 N、不改变输出 shape，也不把四条输入变成四次公开 API 调用；它只改变一次调用内部 request 的调度方式。

特别地，`padded` 的一次调用本身只有一个 language request，因此 `streams="auto"` 不会把它拆成 N 个并行 request；
它最多影响 plugin 对这个 request 的资源配置。只有 `per_sample` 已经产生 N 个独立 request 时，streams 才能直接提供 request 间并发。

#### N>1 时四种组合如何表现

| 组合 | language IR 调用数 | 主要效果 | 本报告实测倾向 |
| --- | ---: | --- | --- |
| `per_sample + streams=1` | N | 单条延迟最低；batch 总时间接近 N 次串行推理 | CPU text 约 326–337 ms/emb（int8） |
| `per_sample + streams=auto` | N 个可重叠 request | CPU 吞吐最高，但 N=1 单条更慢 | CPU int8 text 最佳约 173 ms/emb，image 约 1477 ms/emb |
| `padded + streams=1` | 1 | 一次 `[N,S]` 推理，消除 GPU 提交重复 | GPU text 最佳约 32.6–34.0 ms/emb（int8） |
| `padded + streams=auto` | 1 个主要 padded request | 通常接近 padded=1；plugin 自动调度不一定更快 | GPU text 约 33–35 ms/emb（int8），个别 N 会有波动 |

视觉阶段是例外：`image/fused` 的 visual IR 始终逐图调用，`batch_mode` 只影响后面的 language IR；
逐图调用的 visual 并发度来自 `streams`。一次把多张不同分辨率图片喂入未按该组合 trace 的 visual IR，
可能静默产生错误结果，因此不能把 visual 阶段简单理解成 `[N,...]` padded batch。

---

## 5. 复现命令

```bash
source ../benchmark/openvino/install/setupvars.sh
cd gme-Qwen2-VL-2B-Instruct-Running

# 表 6.1 基准时延（4 模式 × 3 精度 × 2 设备 × 2 后端，N=1）
./env-ref/bin/python 09_benchmark_matrix.py --suite latency --repeat 20

# 表 6.2 文本长度扫描（32/128/512/1024 token）
./env-ref/bin/python 09_benchmark_matrix.py --suite text-lengths --repeat 20

# 表 6.3 图片尺寸扫描（本报告实际同时测 int8/int4，并包含 logo）
./env-ref/bin/python 09_benchmark_matrix.py --suite image-sizes --repeat 20 \
  --precisions int8,int4 --include-logo --output artifacts/benchmark_matrix_image_sizes_r20_int8_int4.json

# 表 6.4/6.5 batch 扩展性（分别执行 int8 和 int4；这是最耗时的一档）
./env-ref/bin/python 09_benchmark_matrix.py --suite batch --batch-precision int8 --repeat 20 \
  --output artifacts/benchmark_matrix_batch_r20_int8.json
./env-ref/bin/python 09_benchmark_matrix.py --suite batch --batch-precision int4 --repeat 20 \
  --output artifacts/benchmark_matrix_batch_r20_int4.json

# 全量（耗时最长，建议限内存并分设备跑）
systemd-run --user --scope -p MemoryMax=20G -p MemorySwapMax=0 \
  ./env-ref/bin/python 09_benchmark_matrix.py --repeat 20
```

**先估算耗时。** FP32 + 大图在 CPU 上单次调用可达数十秒，乘以 `--repeat 20` 与 batch N 会迅速失控。建议：
`--suite image-sizes --precisions int8`、batch 档用 `--image-batch-sizes 1,2,4`，并按设备分批运行。
结果全部落在 `artifacts/benchmark_matrix.json`（含 `workloads` 段，可核对每个单元格实际跑的 shape）。

---

## 6. 结果表格（已完成）

单元格格式：`ms/call (ms/emb)`；N=1 时两者相同，可只填一个数。

### 6.1 基准时延：4 种模式 × 精度 × 设备（N=1）

workload：text/query = 32 token；image/fused = `448x448`（256 视觉 token）。

| 后端 | 设备 | 精度 | text | query | image | fused |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Python | CPU | fp32 | 255.6 | 252.5 | 5211.2 | 5354.5 |
| Python | CPU | int8 | 88.6 | 89.3 | 2699.0 | 2688.8 |
| Python | CPU | int4 | 100.4 | 101.0 | 3055.2 | 3041.7 |
| Python | GPU | fp32 | 142.2 | 143.0 | 2151.8 | 1980.6 |
| Python | GPU | int8 | 42.5 | 41.9 | 509.4 | 509.4 |
| Python | GPU | int4 | 30.8 | 30.5 | 499.9 | 500.7 |
| C++ | CPU | fp32 | 257.6 | 263.2 | 5272.9 | 5383.3 |
| C++ | CPU | int8 | 89.2 | 98.0 | 2701.3 | 2684.7 |
| C++ | CPU | int4 | 100.1 | 111.9 | 3049.4 | 3027.0 |
| C++ | GPU | fp32 | 198.3 | 148.6 | 2283.0 | 1930.0 |
| C++ | GPU | int8 | 41.2 | 43.2 | 498.8 | 503.7 |
| C++ | GPU | int4 | 29.7 | 26.9 | 490.9 | 490.3 |

### 6.2 text / query 长度扫描（N=1）

单元格格式 `text / query`。

| 后端 | 设备 | 精度 | 32 tok | 128 tok | 512 tok | 1024 tok |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Python | CPU | fp32 | 251.1 / 251.3 | 801.1 / 815.0 | 3198.1 / 3242.1 | 6796.8 / 6818.6 |
| Python | CPU | int8 | 88.0 / 88.5 | 333.6 / 338.0 | 1462.2 / 1465.2 | 3194.7 / 3184.3 |
| Python | CPU | int4 | 98.4 / 99.0 | 386.9 / 390.6 | 1679.0 / 1688.5 | 3658.7 / 3654.4 |
| Python | GPU | fp32 | 140.9 / 140.5 | 184.3 / 184.2 | 712.7 / 716.3 | 1626.2 / 1625.7¹ |
| Python | GPU | int8 | 42.6 / 42.0 | 54.3 / 54.1 | 138.3 / 137.8 | 291.9 / 291.6¹ |
| Python | GPU | int4 | 31.0 / 30.8 | 44.4 / 43.8 | 143.8 / 145.3 | 279.1 / 279.1¹ |
| C++ | CPU | fp32 | 250.9 / 250.8 | 808.0 / 805.4 | 3227.7 / 3231.5 | 6782.1 / 6781.9 |
| C++ | CPU | int8 | 88.0 / 87.6 | 333.6 / 332.3 | 1451.1 / 1445.9 | 3172.2 / 3192.7 |
| C++ | CPU | int4 | 98.2 / 98.4 | 386.0 / 384.7 | 1682.3 / 1678.3 | 3630.2 / 3641.3 |
| C++ | GPU | fp32 | 154.3 / 149.9 | 183.1 / 180.0 | 735.3 / 717.8 | 1622.7 / 1775.8¹ ² |
| C++ | GPU | int8 | 41.1 / 41.2 | 52.8 / 52.9 | 134.3 / 134.2 | 448.6 / 353.4¹ ² ³ |
| C++ | GPU | int4 | 29.9 / 29.9 | 42.7 / 42.6 | 141.4 / 141.4 | 362.4 / 345.5¹ ² ³ |

**结论**：时延随长度明显超线性，接近 decoder attention 的二次趋势，但不是纯二次函数。
1024 相对 32 token：CPU fp32/int8/int4 约 27.1x/36.3x/37.2x；Python GPU int8/int4 约 6.9x/9.0x。
上表 GPU 1024 tok 一列已经用下方 6.2.1 的多轮复测中位数替换（¹²³ 标注说明各自的取值口径），
GPU int8/int4 的 C++ 数字**不是一个稳定单值**，只是一个跨越很宽分布的中位数——分布本身、成因和为什么
`--warmup` 没能把它变成稳定单值，见 6.2.1。

#### 6.2.1 GPU 1024 token 交替复测：C++/Python 的 2x 差异是 GPU 升频/降频状态，不是实现差异

最早那次单次数据（574.4/612.9 对 292.9/294.3 等）已知不可靠。第一轮交替复测（15-20 轮、无额外 warmup）
怀疑是"进程生命周期导致 GPU 频率漂移"，据此给 C++/Python 两套 benchmark harness 都加了可配置的 `--warmup N`
（C++ 见 `cpp/src/main.cpp` 的 `bench()`，Python 见 `10_retest_gpu_1024.py::timed()`），指望"多跑几次预热"能把
结果收敛到一个稳定值。**加大 warmup 之后重新测了一遍，结果证明这个直觉是错的**：warmup 深度和结果之间不是单调
关系——同一台机器、同一个精度、只改 warmup 次数，1 次 warmup 有时给出 415ms，6 次给出 287ms，16 次给出
674ms；反过来固定 warmup=1 连续起 6 个全新进程，稳定给出 ~287-291ms，固定 warmup=12 连续起 6 个全新进程，
稳定给出 ~671-677ms。也就是说：**GPU 存在一个"boost"和一个"稳态/降频"两档实际存在的物理频率状态**，
warmup 次数只是让某一次测量更可能撞上其中一档，但撞上哪一档还取决于测量开始那一刻 GPU 恰好处于什么状态，
这一点无法单纯靠这个进程自己的 warmup 调用数保证。用纯 Python、完全不涉及 C++ 的实验直接复现了这个现象——
同一个 encoder 实例连续跑 20 次 `get_text_embeddings`（int8, 1024 tok），第 1-3 次稳定在 ~288ms，第 4 次起
稳定跳到 ~675ms 并一直保持到第 20 次；`int4` 同样在第 4 次左右从 ~277ms 跳到 ~508-512ms。**这就是证据本身：
不靠 C++ 也能把"慢了 2x 多"复现出来，所以它不可能是 C++ 实现比 Python 慢的架构性问题。**

在把 `--warmup` 分别调到 int8=12、int4=14、fp32=3（根据上面的实测转折点选的，留了余量）之后重新做了一次完整
交替复测（`int8`/`int4` 各 15 轮、`fp32` 8 轮、`int8 + per_sample` 诊断 6 轮，仍然是 C++ 全新子进程 ↔ Python
全新 `GmeOpenVINO` 实例严格按时间顺序交替），原始数据和统计落盘在
`artifacts/benchmark_gpu_1024_interleaved_retest.json`（旧的、未加大 warmup 的那版数据已被这次运行整体覆盖）。
单元格为 `median [p10, p90] ms (n=轮数)`：

| 精度 | 后端 | text | query |
| --- | --- | ---: | ---: |
| fp32 | C++ | 1622.7 [1616.4, 1626.6] (n=8)⁴ | 1775.8 [1770.1, 1778.8] (n=8)¹ |
| fp32 | Python | 1626.2 [1615.1, 1630.4] (n=8) | 1625.7 [1624.6, 1627.5] (n=8) |
| int8 | C++ | 448.6 [292.1, 520.5] (n=15) | 353.4 [352.6, 354.2] (n=15)¹ |
| int8 | Python | 291.9 [290.6, 293.9] (n=15) | 291.6 [290.7, 293.7] (n=15) |
| int4 | C++ | 362.4 [277.1, 418.6] (n=15) | 345.5 [345.0, 346.0] (n=15)¹ |
| int4 | Python | 279.1 [277.5, 281.1] (n=15) | 279.1 [278.2, 280.4] (n=15) |
| int8 + `per_sample`（诊断） | C++ | 330.8 [290.2, 444.0] (n=6) | 353.9 [352.2, 354.2] (n=6)¹ |
| int8 + `per_sample`（诊断） | Python | 290.4 [289.9, 290.5] (n=6) | 290.6 [289.8, 291.0] (n=6) |

¹ C++ 一次会话里省掉了第二次模型编译，`query` 复用的是 text 标定的 1024-tok prompt 换上 query instruction
（实测 1027 tok，而不是表 6.2 里单独标定到 1024 tok 的 query prompt）；Python 侧仍用各自标定好的 text/query
prompt。但下面会看到，text/query 数值差异的量级和这个 ±3 token 完全对不上，真正的变量是"这是会话里第几个
跑的负载"，见下文。
² C++ `text` 与 `query` 相差很大不是内容差异，而是 `text`（会话里第一个负载）横跨了 boost/稳态两档，`query`
（第二个负载）在这次复测里稳定落在两档之间的一个中间值——本质上仍然是同一个物理现象的不同切片，不是两个
可比的"稳定数字"。
³ 中位数本身具有误导性：C++ int8/int4 的 `text` 不是"围绕 448/362 波动"，而是在 ~290ms 和 ~520/~460ms 两端
来回跳，中位数只是恰好落在中间，不代表任何一次真实调用的典型延迟；使用前务必看 p10/p90 或原始 15 轮数据。
⁴ fp32 `text` 8 轮里有 1 轮是 2978.2 ms 的极端尖峰（其余 7 轮都在 1614-1626 ms），中位数/分位数已经把它稀释掉——
原始数据见 artifact，这一尖峰本身就是"这台机器的 GPU 功耗/温度状态会突然跳变"的又一个例证。

**关键发现，按证据强度排列**：

- **加大 warmup 不能让任何一侧稳定下来。** 单独测 `python_session(int8, warmup=12)` 连续 3 次：一次落在稳态
  (674/672 ms)，两次落在 boost (288-290 ms)；`python_session(int4, warmup=14)` 连续 3 次全部落在 boost
  (276-277 ms)。`int4, warmup=14` 对 8 个全新 C++ 进程也只有 7/8 命中稳态。也就是说 warmup 只能提高撞上某一档
  的概率，不能保证。这与最初"加个 warmup 就能排除干扰"的预期不同，如实记录在这里。
- **Python 在这次正式的 15 轮里全程停在 boost 档**（int8/int4 的 text、query 四组 p10-p90 全部收在 3 ms 以内，
  没有一次落到稳态），而 **C++ 的 15 轮 `text` 横跨了整个区间**（int8: 290/520/473/467/370/428/369/292/297/
  487/528/448/433/452……几乎是连续谱，不是干净的两档）。两边用的是同一个 OpenVINO runtime、同一份 IR、同一个
  `padded` 调用路径；`--batch-mode per_sample` 诊断行再次确认现象与 batch_mode 无关。目前能确定的是：**这与
  "C++ 每轮都是全新 OS 子进程、Python 每轮只是在同一个长期存活进程里重新 `compile_model`"这件事强相关**——但
  完整的驱动/电源管理机制（为什么同一个物理 GPU 在两种调用方式下暴露出的分布形状差这么多）没有能在这台机器上
  用现有工具（缺少 `gt_cur_freq_mhz` 等标准 sysfs 频率节点）定位到底层原因，如实说明这是"强相关但机制未完全
  查明"，不夸大成一个确定的因果链。
- **因此**：表 6.2 里"C++ 比 Python 在 GPU int8/int4 @ 1024 tok 慢 2x"这个印象，主要是「C++ 的测量方式（每次
  独立起进程）比 Python 的测量方式（同一进程里反复编译）更容易被这台机器的 GPU 电源状态采样到较慢的那一档」，
  不是 GME OpenVINO C++/Python 两份实现本身有性能差距——当 C++ 恰好落在 boost 档时（比如 int8 的第 0/7/8 轮，
  290/292/297 ms），和 Python 的 ~292 ms 几乎完全一致。
- **不要把表 6.2 的单个数字当成这两个精度在 GPU 上的稳定延迟**：如果需要给 SLA 或回归阈值定数，要么在同一个
  长期存活的进程里反复调用两侧实现（避免每次重新起进程/编译），要么老老实实报"多轮中位数 + p10/p90"而不是
  一次测量，本节的表格就是这么做的。原始逐轮数据见 `artifacts/benchmark_gpu_1024_interleaved_retest.json`。

### 6.3 image / fused 尺寸扫描（N=1，int8 与 int4）

单元格格式 `image / fused`。

| 后端 | 设备 | 精度 | 448x448 (256) | 644x476 (391) | 672x672 (576) | 896x896 (1024) | 1288x728 (1196) | logo (1260) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Python | CPU | int8 | 2635.5 / 2655.6 | 4513.7 / 4527.7 | 7466.0 / 7422.0 | 18880.5 / 21484.6 | 21434.0 / 21271.0 | 23638.8 / 23524.2 |
| Python | GPU | int8 | 515.6 / 506.0 | 845.2 / 848.9 | 1370.8 / 1376.1 | 3391.0 / 3398.5 | 4562.1 / 4563.4 | 5083.8 / 5084.4 |
| C++ | CPU | int8 | 2676.0 / 2670.0 | 4479.0 / 4507.2 | 7405.2 / 7401.4 | 17808.6 / 16773.5 | 21474.6 / 21303.2 | 23449.4 / 25133.9 |
| C++ | GPU | int8 | 499.4 / 501.8 | 835.1 / 845.4 | 1360.0 / 1401.4 | 3811.2 / 3773.0 | 4538.0 / 4535.3 | 5114.4 / 5118.0 |
| Python | CPU | int4 | 3029.5 / 3015.6 | 4998.3 / 5051.6 | 8180.5 / 8188.8 | 18439.3 / 18511.8 | 23335.4 / 23420.7 | 25216.8 / 25491.2 |
| Python | GPU | int4 | 495.0 / 495.4 | 845.9 / 849.1 | 1373.1 / 1374.6 | 3382.8 / 3386.3 | 4574.2 / 4580.6 | 5106.5 / 5088.8 |
| C++ | CPU | int4 | 3024.2 / 3011.8 | 4991.9 / 5038.5 | 8165.6 / 8160.9 | 18216.4 / 18094.2 | 23325.1 / 23337.1 | 25462.4 / 25308.4 |
| C++ | GPU | int4 | 489.8 / 493.4 | 834.6 / 841.2 | 1359.5 / 1390.9 | 3498.3 / 3487.6 | 4550.6 / 4555.9 | 4982.4 / 4987.6 |

**结论**：时延主要由视觉 token 数驱动，随 K 呈明显超线性增长。GPU int8 Python image 从 256 K 的
515.6 ms 增至 1260 K 的 5083.8 ms，约 9.9x；CPU int8 约 9.0x。`1288x728` 的 1196 K 与 logo 的
1260 K 已处于同一量级，宽高比不是主导因素。把 `max_image_tokens` 降到 512/256 的理论收益需按实际
业务尺寸重测，且必须同时验证 Recall@K/nDCG，不能仅凭时延决定。

### 6.4 batch 扩展性：CPU（text = 128 token，image = 448x448）

单元格为 `ms/call (ms/emb)`；以下先列出部署主路径 Python，C++ 对照见 §6.6，完整原始矩阵见 artifacts。

#### CPU int8 text

| batch_mode | streams | N=1 | N=2 | N=4 | N=8 | N=16 | N=32 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| per_sample | 1 | 326.3 (326.3) | 662.1 (331.0) | 1347.4 (336.9) | 2684.0 (335.5) | 5366.1 (335.4) | 10789.9 (337.2) |
| per_sample | auto | 552.5 (552.5) | 641.4 (320.7) | 698.7 (174.7) | 1466.6 (183.3) | 2951.1 (184.4) | 5547.6 (173.4) |
| padded | 1 | 368.7 (368.7) | 685.0 (342.5) | 1360.2 (340.0) | 2721.1 (340.1) | 5548.8 (346.8) | 11233.7 (351.1) |
| padded | auto | 554.7 (554.7) | 1099.0 (549.5) | 2198.6 (549.6) | 4395.1 (549.4) | 8805.8 (550.4) | 17557.1 (548.7) |

#### CPU int8 image

| batch_mode | streams | N=1 | N=2 | N=4 | N=8 |
| --- | --- | ---: | ---: | ---: | ---: |
| per_sample | 1 | 2608.2 (2608.2) | 5290.6 (2645.3) | 10612.8 (2653.2) | 21227.8 (2653.5) |
| per_sample | auto | 4303.4 (4303.4) | 5162.1 (2581.1) | 5918.2 (1479.5) | 11898.4 (1487.3) |
| padded | 1 | 2790.3 (2790.3) | 5283.3 (2641.6) | 10466.5 (2616.6) | 20973.9 (2621.7) |
| padded | auto | 4320.1 (4320.1) | 6074.1 (3037.0) | 9008.2 (2252.0) | 17698.0 (2212.2) |

#### CPU int4 text

| batch_mode | streams | N=1 | N=2 | N=4 | N=8 | N=16 | N=32 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| per_sample | 1 | 371.1 (371.1) | 783.4 (391.7) | 1574.1 (393.5) | 3195.0 (399.4) | 6425.9 (401.6) | 12855.5 (401.7) |
| per_sample | auto | 704.1 (704.1) | 823.5 (411.8) | 882.2 (220.6) | 1806.8 (225.8) | 3572.9 (223.3) | 6736.3 (210.5) |
| padded | 1 | 427.7 (427.7) | 814.1 (407.0) | 1651.9 (413.0) | 3272.0 (409.0) | 6647.3 (415.5) | 13392.5 (418.5) |
| padded | auto | 706.8 (706.8) | 1390.7 (695.4) | 2798.1 (699.5) | 5597.6 (699.7) | 11218.6 (701.2) | 22421.8 (700.7) |

#### CPU int4 image

| batch_mode | streams | N=1 | N=2 | N=4 | N=8 |
| --- | --- | ---: | ---: | ---: | ---: |
| per_sample | 1 | 3098.5 (3098.5) | 6074.8 (3037.4) | 12260.5 (3065.1) | 24559.2 (3069.9) |
| per_sample | auto | 5264.8 (5264.8) | 6344.1 (3172.1) | 6975.3 (1743.8) | 14085.4 (1760.7) |
| padded | 1 | 3178.2 (3178.2) | 6088.7 (3044.3) | 12167.8 (3041.9) | 24290.9 (3036.4) |
| padded | auto | 5296.9 (5296.9) | 7477.9 (3738.9) | 11088.9 (2772.2) | 21790.0 (2723.8) |

### 6.5 batch 扩展性：GPU（int8，text = 128 token）

| batch_mode | streams | N=1 | N=2 | N=4 | N=8 | N=16 | N=32 | 最佳 ms/emb 相对 N=1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| per_sample | 1 | 54.5 (54.5) | 107.3 (53.6) | 212.7 (53.2) | 426.9 (53.4) | 848.6 (53.0) | 1698.5 (53.1) | 0.97x |
| per_sample | auto | 57.4 (57.4) | 106.8 (53.4) | 209.7 (52.4) | 427.6 (53.5) | 969.5 (60.6) | 1712.9 (53.5) | 0.99x |
| padded | 1 | 53.7 (53.7) | 75.5 (37.7) | 134.8 (33.7) | 272.2 (34.0) | 522.5 (32.6) | 1088.1 (34.0) | 0.61x |
| padded | auto | 59.7 (59.7) | 187.0 (93.5) | 134.9 (33.7) | 282.9 (35.4) | 601.1 (37.6) | 2643.9 (82.6) | 0.57x |

image（`448x448`，N ∈ 1,2,4,8）：

| batch_mode | streams | N=1 | N=2 | N=4 | N=8 | 最佳 ms/emb 相对 N=1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| per_sample | 1 | 516.8 (516.8) | 1024.7 (512.3) | 2027.4 (506.9) | 4053.5 (506.7) | 0.98x |
| per_sample | auto | 510.7 (510.7) | 1107.2 (553.6) | 2148.5 (537.1) | 4514.8 (564.3) | 1.00x |
| padded | 1 | 524.1 (524.1) | 974.6 (487.3) | 1957.1 (489.3) | 3872.3 (484.0) | 0.92x |
| padded | auto | 589.8 (589.8) | 1145.3 (572.6) | 1958.5 (489.6) | 4326.7 (540.8) | 0.83x |

#### GPU int4 text

| batch_mode | streams | N=1 | N=2 | N=4 | N=8 | N=16 | N=32 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| per_sample | 1 | 43.3 (43.3) | 86.5 (43.2) | 172.5 (43.1) | 344.9 (43.1) | 689.5 (43.1) | 1382.7 (43.2) |
| per_sample | auto | 47.1 (47.1) | 85.7 (42.9) | 168.4 (42.1) | 335.9 (42.0) | 670.0 (41.9) | 1371.8 (42.9) |
| padded | 1 | 43.8 (43.8) | 72.7 (36.4) | 140.7 (35.2) | 260.2 (32.5) | 546.6 (34.2) | 1088.7 (34.0) |
| padded | auto | 47.3 (47.3) | 73.7 (36.8) | 143.5 (35.9) | 265.4 (33.2) | 547.6 (34.2) | 1101.1 (34.4) |

#### GPU int4 image

| batch_mode | streams | N=1 | N=2 | N=4 | N=8 |
| --- | --- | ---: | ---: | ---: | ---: |
| per_sample | 1 | 505.2 (505.2) | 1004.0 (502.0) | 1990.8 (497.7) | 3971.4 (496.4) |
| per_sample | auto | 499.7 (499.7) | 994.8 (497.4) | 1987.0 (496.8) | 3950.0 (493.8) |
| padded | 1 | 494.6 (494.6) | 975.4 (487.7) | 1933.3 (483.3) | 3861.7 (482.7) |
| padded | auto | 498.1 (498.1) | 981.1 (490.6) | 1962.5 (490.6) | 3898.7 (487.3) |

### 6.6 C++ 与 Python 的 batch 一致性

| 精度 | 设备 | batch_mode | streams | N | Python ms/emb | C++ ms/emb | 比值 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| int8 | CPU | per_sample | auto | 8 | 183.3 | 179.7 | 0.98 |
| int8 | GPU | padded | 1 | 8 | 34.0 | 34.4 | 1.01 |
| int4 | CPU | per_sample | auto | 8 | 225.8 | 221.9 | 0.98 |
| int4 | GPU | padded | 1 | 8 | 32.5 | 32.2 | 0.99 |

两端使用同一批 IR，text 路径数值 bit-exact（`08_validate_batch.py` §E）。实测 text 时延比值为 0.98–1.01，
说明 Python/C++ 的 batch 编排开销接近；image 路径不作为 bit-exact 时延验收，因为 C++ 使用 OpenCV、
Python 使用 Pillow，图像预处理路径本身不同。

### 6.7 结论与默认值复核

实测支持当前 `batch_mode="auto"` 的设备分支：**CPU → `per_sample`、GPU → `padded`**。

- [x] CPU 上 padded 基本没有收益：int8 `streams=1` text 约 326–337 ms/emb，padded 约 340–351 ms/emb；
  吞吐收益来自 `streams="auto"`，text 最佳约 171.9–173.4 ms/emb，image 最佳约 1476.5–1479.5 ms/emb。
- [x] GPU 上 padded 优势成立：int8 text 最佳约 32.6–34.0 ms/emb，per-sample 约 52.4–53.1 ms/emb；
  int4 text 最佳约 32.2–34.0 ms/emb，per-sample 约 41.8–43.2 ms/emb。
- [x] CPU `streams="auto"` 的吞吐收益成立，但单样本代价明显：int8 text 从约 326–332 ms 增至 553–559 ms，
  int4 text 从约 371–397 ms 增至 704–709 ms；测试 scope 峰值内存约 14 GiB。
- [x] 建议默认 micro-batch：CPU text 使用 `per_sample + streams="auto"`，按延迟预算优先 N=4–8；CPU image 优先 N=4。
  GPU text/image 使用 `padded`，text 优先 N=8–16，image 优先 N=4–8。
- [x] int4 在 GPU 上 text 更快、CPU 上通常比 int8 慢；正式部署仍须结合业务 Recall@K/nDCG，而不是只按时延选精度。

本节 batch 扫描未改变 `gme_openvino.py` / `gme_qwen2vl.hpp` 的 `auto` 分支；README §6.6 的历史结论与本次
完整矩阵一致，但本报告中的数字应优先作为当前 workload 的选型依据。

---

## 7. 已知陷阱

1. **不要把本报告数字与 README §6.4/§6.5 混表**：workload 不同（常见尺寸 vs logo、指定长度 vs 27 token 短句）。
2. **`--repeat 5` 不足以支撑结论**（§3 有实测反例）。
3. **小图不会更快**：≤`448x448` 一律是 256 视觉 token（§4.2）。
4. **`ms/call` 不能用来比较不同 N**，只有 `ms/emb` 可以。
5. **首次 GPU 调用包含时钟爬坡**：warmup 只有 1 次时 GPU 数值会虚高；`--repeat` 越小越明显。
6. **`batch_size` 与 `batch` 是两件事**：`batch_size` 是分块上限（默认 32，官方同值），`N` 是一次请求里的样本数。
   本报告的 batch 档用 `--batch-size 0`（不分块）以测量纯 batch 效应。
7. **不含加载/编译**：GPU 首次编译可达数十秒，服务端必须复用同一个 `GmeOpenVINO` / `gme::GmeQwen2VL` 实例；
   本报告的数字不包含这部分，不能直接当作冷启动 SLA。
8. **单机结果不是服务级 SLA**：线上仍应按最大等待时间形成 micro-batch，并在真实 query/图片分布上重测。
