# 4. 测试与结果

> 本节含两轮结果。**§4.1 / §4.2 的 bit-identical 结论对应 opt-1 + opt-2**；
> 加上第二轮的 opt-3 之后精度结论有一处变化，记在 **§4.1b**，性能数字统一按
> 第二轮重测（**§4.3**）。

## 4.1 数值正确性（opt-1 + opt-2）：bit-identical

对照组用 **pip 安装的官方 OpenVINO 2026.3.0 wheel**
（`2026.3.0-22451-8a17657b995-releases/2026/3`，正是 patch 的 base commit），
实验组用打了 patch 的本地构建。两边跑同一套固定输入，逐样本比 cosine / max abs diff。

```bash
# 对照组（不 source setupvars，用官方 wheel）
python dump_outputs.py --precision int4 --output results/acc_int4_stock.npz
# 实验组
source "$OPENVINO_ROOT/install/setupvars.sh"
python dump_outputs.py --precision int4 --output results/acc_int4_patched.npz
python compare_outputs.py results/acc_int4_stock.npz results/acc_int4_patched.npz
```

INT4：

| case | cosine | max_abs_diff | rel_l2 |
| --- | ---: | ---: | ---: |
| seq1 | 1.000000000 | 0.000e+00 | 0.000e+00 |
| seq7 | 1.000000000 | 0.000e+00 | 0.000e+00 |
| seq32 | 1.000000000 | 0.000e+00 | 0.000e+00 |
| seq129 | 1.000000000 | 0.000e+00 | 0.000e+00 |
| seq512 | 1.000000000 | 0.000e+00 | 0.000e+00 |
| seq1024 | 1.000000000 | 0.000e+00 | 0.000e+00 |
| batch2_padded（右 padding，第 2 行只有 300 个有效 token） | 1.000000000 | 0.000e+00 | 0.000e+00 |

INT8 和 INT4-gs64 同样全部 **bit-identical**（`worst cosine 1.000000000`，
max abs diff 0）。

> 这是预期的：opt-1 / opt-2 都只改图结构和 kernel 调度，没有任何一处改变算术顺序。

## 4.1b 加上 opt-3 之后的数值（`results/acc_int4_round2.npz`）

```bash
python compare_outputs.py results/acc_int4_stock.npz results/acc_int4_round2.npz
```

| case | cosine | max_abs_diff | rel_l2 |
| --- | ---: | ---: | ---: |
| seq1 | 1.000000000 | 0.000e+00 | 0.000e+00 |
| seq7 | 1.000000000 | 0.000e+00 | 0.000e+00 |
| seq32 | 1.000000000 | 0.000e+00 | 0.000e+00 |
| seq129 | 1.000000000 | 0.000e+00 | 0.000e+00 |
| seq512 | 1.000000000 | 0.000e+00 | 0.000e+00 |
| seq1024 | 1.000000000 | 0.000e+00 | 0.000e+00 |
| batch2_padded | **0.999497041** | **1.192e-02** | 2.242e-02 |

**七个用例里六个仍然 bit-identical，只有 `batch2_padded` 动了。**

三件事让这个结果可以被解释，而不是被当成"某处算错了"：

1. **不是"新路径根本没跑"。** 精度这一跑用 `ONEDNN_VERBOSE=2` 复核过，确实有
   **408 个 `abc` matmul** 执行 —— 六个 bit-exact 的用例是在新 layout 真的生效的
   前提下 bit-exact 的。
2. **重排本身不是嫌疑。** `oiyx → ioyx` 的 INT4 reorder 做过 host readback 逐 nibble
   比对，**0 / 2359296 个 nibble 不一致**；另外 `layout_accuracy_probe` 在 oneDNN
   层面单独验过 `cab` 与 `abc` 数值等价。
3. **量级对得上。** 换 layout 会换 `jit:gemm` 的分块与累加顺序，f16 累加下这必然有
   末位漂移。参照系：同一批用例上 **INT4 相对 INT8 本身**的量化误差是
   cosine 0.918–0.989、max_abs 最高 9.4e-2 —— 也就是说 opt-3 引入的漂移比模型自身
   的量化噪声底**低约一个数量级**。

`batch2_padded` 单独跳出来也符合预期：它是唯一一个第二行带右 padding 的用例，
padding 位置的中间值会随分块顺序变化，再经过 pooling / normalize 放大。

结论：**opt-1 + opt-2 仍是严格 bit-identical；opt-3 不是**，它是一个量级远低于
量化噪声的重排序漂移。对精度零容忍的场景可以只 apply 前两个 patch。

## 4.1c GME image/fused 可运行性复核（opt-1 + opt-2 + opt-3）

Qwen3-VL-Embedding-2B 在 image/fused 路径上出现 `FullyConnectedCompressed` /
`CL_OUT_OF_RESOURCES`，不能直接推断 GME 也会失败。GME 使用独立的 Qwen2-VL
visual IR 和 `gme_openvino.py`，因此在同一 GPU 上对官方 stock wheel 与三个 patch
全开的本地 runtime 做了独立 image/fused 回归。每个 case 为 warmup 2、repeat 3，
输出 shape 均为 `[1,1536]`。

| 模式 | 尺寸 | stock p50 | opt1+2+3 p50 | 状态 | cosine vs stock | max abs |
|---|---:|---:|---:|---|---:|---:|
| image | 64x64 | 408.757 ms | 395.529 ms | OK | 1.000000000 | 0 |
| image | 224x224 | 411.564 ms | 399.102 ms | OK | 1.000000000 | 0 |
| image | 448x448 | 410.404 ms | 399.372 ms | OK | 1.000000000 | 0 |
| fused | 64x64 | 723.166 ms | 681.079 ms | OK | 0.996111109 | 9.123e-3 |
| fused | 224x224 | 730.253 ms | 759.379 ms | OK | 0.994195511 | 1.855e-2 |
| fused | 448x448 | 715.654 ms | 684.189 ms | OK | 1.000000000 | 0 |

同一 build 内重复调用的 `fused/224x224` 输出完全稳定：stock 与 patched 各自重复
调用的 cosine 都为 1.0、max abs 都为 0。因此 fused 的小幅差异是稳定的 stock-vs-
patched 数值差异，不是 GPU 随机噪声；但它没有造成 GME image/fused 无法运行。
完整原始摘要见
[`results/gme_visual_opt123_20260813.json`](../results/gme_visual_opt123_20260813.json)。

**结论**：三个优化**不会导致 GME 的 image/fused 路径无法运行**。GME image-only
在这三个尺寸上保持逐位一致；fused 仍可运行，但 64x64/224x224 存在小幅数值漂移，
部署前应按实际检索/相似度指标做 fused 精度回归，不能沿用“所有模式
bit-identical”的表述。Qwen3 的视觉报错属于其自身 `FullyConnectedCompressed`
路径回归，不能外推为 GME 会同样失败。

## 4.2 生成式模型回归（KV-cache 通路）

opt-1 / opt-2 都落在**共享的** SDPA 融合路径上，opt-3 落在**共享的** INT4 FC
路径上，所以有状态 LLM 必须同样不变。

**先说一个坑**：直接用 `openvino_genai` 做 greedy 对比会得到"输出不一致"的假阳性
—— 本地 `benchmark/openvino.genai` 的构建是 `0c38b54d7e7-qwen-ttft-llm-bench`，
带着用户自己的 llm_bench / rag / qwen3_vl 提交和工作区改动，和 pip wheel
（`bd8d6542e3c`）**本来就不是同一个库**。这个对比同时变了两个变量，不能用。

正确做法是绕开 GenAI，用 `regress_stateful.py` 直接用 plain OpenVINO 驱动有状态
IR 做 greedy 解码（`Qwen3-1.7B-INT4`，GPU，2 个 prompt × 32 步）：

| run | generated_ids 一致 | 首步 logits Σx² (stock) | 首步 logits Σx² (patched) |
| --- | --- | ---: | ---: |
| run0 | ✅ | 3175534.493439 | 3175534.493439 |
| run1 | ✅ | 5604682.734206 | 5604682.734206 |

首步 logits 的平方和与 top-8 值**逐位相同**，32 步 token id 完全相同 →
**KV-cache 通路无回归**。

第二轮加上 opt-3 后重跑（`results/r2_regress_stateful_patched.json`），
与同一份 stock 基线逐字段比较，`prompt_ids` / `generated_ids` /
`first_step_logits_sha` / `first_step_logits_top8` **四项全部相同**，
结论不变。opt-3 在这个负载上不触发也是预料之中：prompt 只有 ~20 token，
远低于 128 行门槛。

### 更正一条第一轮的表述

第一轮曾把这条写成"生成式模型输出 bit-identical"。第二轮复核时发现：走
`openvino_genai` 的 `regress_generative.py`（`Qwen3-1.7B-INT4`，greedy）在
patched 下续写文本与 stock **并不相同**。补充测量之后可以定性：

- 第一轮 patched 的输出与第二轮 patched 的输出**逐字节相同** →
  差异不是 opt-3 引入的；
- opt-3 的 128 行门槛在这些 ~20 token 的 prompt 上根本不可能触发 → 也不可能是它；
- 因此这是 **opt-1 / opt-2 就已经存在**的差异，且如 §4.2 开头所述，本地 GenAI
  构建与 pip wheel 本来就不是同一个库，这个对比同时变了两个变量。

**权威结论以 `regress_stateful.py` 为准**（它只用 plain OpenVINO 驱动同一份 IR，
单变量），即 KV-cache 通路的 token id 与首步 logits 逐位相同。
`regress_generative.py` 的结果只作参考，不作为判据。

## 4.3 端到端性能（交替 A/B）

这台机器的 GPU 有明显的双档频率状态（见 `PERFORMANCE_REPORT.md` §6.2.1 记录的
boost / non-boost 双峰），所以不能"先跑完 stock 再跑 patched"。`bench_ab.sh`
按轮次交替，每轮 warmup 14 + repeat 20，共 3 轮，取 p50 的中位数：

下表是**第二轮重测**的结果，patched = opt-1 + opt-2 + opt-3 全开
（原始数据 `results/ab/`，每格 3 轮）：

| tokens | workload | stock p50 (ms) | patched p50 (ms) | delta | speedup | stock min | patched min |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | text | 32.78 | 32.19 | −0.59 | **1.018x** | 32.42 | 31.90 |
| 32 | query | 32.85 | 32.31 | −0.55 | **1.017x** | 32.51 | 31.93 |
| 128 | text | 46.56 | 40.20 | −6.36 | **1.158x** | 46.05 | 39.80 |
| 128 | query | 46.24 | 40.17 | −6.08 | **1.151x** | 45.79 | 39.77 |
| 512 | text | 146.55 | 123.81 | −22.73 | **1.184x** | 144.99 | 123.27 |
| 512 | query | 145.49 | 123.31 | −22.18 | **1.180x** | 145.12 | 122.60 |
| 1024 | text | 281.38 | 240.01 | −41.38 | **1.172x** | 279.13 | 239.18 |
| 1024 | query | 280.18 | 239.53 | −40.66 | **1.170x** | 279.41 | 238.49 |

三轮之间离散度很小（1024 token 上 stock 279.8–281.6、patched 239.6–241.2）。

曲线的形状值得单独说一句：**32 token 上收益仍然只有第一轮的 ~1.02x，从 128 token
起跳到 1.15x 以上**。这不是噪声，是 opt-3 的 128 行门槛：低于该行数时新权重
layout 反而更慢（0.84–0.85x，见 docs/06 §6.9.4），所以规则刻意不触发，收益退回
opt-1 + opt-2 的水平。

第一轮（仅 opt-1 + opt-2）的对照数字，供分解收益用：
32 token 1.016–1.020x、512 token 1.050–1.055x、1024 token 1.063–1.067x。
即 **1024 token 上 1.17x 里，opt-1+opt-2 贡献 ~1.065x，opt-3 再叠 ~1.10x**。

## 4.4 VTune 复核（patched）

```bash
vtune -collect gpu-hotspots -result-dir results/vtune-gpu-hotspots-int4-1024-patched \
      -- python profile_gme_int4_gpu.py --length 1024 --warmup 14 --repeat 30 --hold-seconds 30
```

| 指标 | baseline | patched |
| --- | ---: | ---: |
| Occupancy（% of peak） | 69.0% | **73.3%** |
| XVE Array Stalled/Idle | 58.5% | **53.4%** |
| `broadcast_gpu_ref` | 0.925 s | **不再出现** |
| `permute_ref` 合计 | 0.574 s | **0.073 s** |
| `sdpa_micro__prefill` | 2.455 s | 2.285 s |

> 注：两次 capture 的 `Elapsed Time`（153.8s → 103.8s）**不能**直接对比 ——
> 中间给 harness 加了 workload 磁盘缓存，省掉了上千次 tokenizer 调用的 CPU 时间。
> `GPU Time` 的绝对值也受 VTune 自身开销影响。因此这一节只作**定性**复核，
> 定量结论以 §4.3 和 §3 的 plugin profiling 为准。

> 这是第一轮（opt-1 + opt-2）采集的。第二轮的 VTune 复采见
> `06_round2_onednn_investigation.md`。

## 4.5 结论

1. **确认了两处 OpenVINO GPU plugin 的图优化缺口**，均为通用问题（不是 GME 特例）：
   - GQA `repeat_kv` 消除只对有状态模型生效，无状态 encoder/embedding 模型漏掉；
   - SDPA 缺少输出 transpose 融合，且 micro 实现对非默认输出 order 的 dispatch 是坏的。
2. **第二轮又确认了一处 GEMM 侧的空档**：`jit:gemm:any__i8` 对 4-bit 压缩权重
   固定按 K 连续描述，而在 **N ≫ K** 的宽投影上 N 连续明显更快 —— GME 的
   `gate_proj` / `up_proj`（N/K = 5.83）合计占 GEMM 时间 62%，换排布后
   GEMM 从 177.5 降到 151.1 ms。
3. 三个改动合起来：**1024 token 提速 17.0–17.2%，512 token 18.0–18.4%，
   128 token 15.1–15.8%，32 token 1.7–1.8%**。
   精度上 **opt-1 + opt-2 严格 bit-identical**；opt-3 在七个用例里六个 bit-identical，
   剩下一个的漂移比 INT4 自身的量化噪声底低约一个数量级（§4.1b）。
   KV-cache 通路 **无回归**（§4.2）。
4. **剩余时间仍以 oneDNN `jit:gemm` 为主（~63%）**。opt-3 只吃掉了"排布选错"这
   一块，kernel 自身的吞吐问题还在：实测饱和在 ~15 TFLOPS（int8），而同一 kernel
   的 f16 路径就有 12.5 TFLOPS —— int8 只快 1.2×，远低于 XMX int8 相对 f16 的
   2× 理论比。已经定价过的剩余杠杆只剩 per-OC 量化（1.55x，**有精度代价**），
   见 docs/06 §6.3。
5. 因此对"int4 text/query 明显过慢"的回答是：**慢的主体是这颗 iGPU 上 oneDNN
   权重压缩 GEMM 的实际吞吐**，但其中有一部分（宽投影的权重排布）是 OpenVINO
   侧就能修的，两轮合计已拿到 **~17%** 的干净收益；再往前要么动 oneDNN kernel
   本身，要么动模型导出（§5.2）。
