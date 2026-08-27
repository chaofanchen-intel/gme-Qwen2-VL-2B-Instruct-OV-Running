# 第三轮 GME 性能报告：causal 导出、量化方案与优化点对比

> 本文记录第三轮（2026-08-21）的 language IR 统一重测，属于历史报告。第四轮
> 已新增 mask-free visual IR，当前 package 使用 `int8_vnomask` 和
> `int4_causal_vnomask`；当前 image/fused 和 batch 结果见
> [`../results/r4_rerun_20260826/`](../results/r4_rerun_20260826/) 及
> [`../../document_embedding_eval_int8_int4_gpu/REPORT.md`](../../document_embedding_eval_int8_int4_gpu/REPORT.md)。

## 1. 报告目的

本报告汇总 GME-Qwen2-VL-2B-Instruct 第三轮优化后的 latency，重点回答四个问题：

1. 普通导出和 `causal=true` 导出分别有多快？
2. `gs128`、`gs64`、per-OC、对称量化之间的代价和收益是什么？
3. plugin 侧 opt-1/opt-2/opt-3，以及导出侧 causal，各自贡献了多少 latency？
4. 哪些组合已经实测，哪些组合没有模型产物或缺少独立阶段数据？

本报告将不同来源的数据分开标注：

- **统一重测**：2026-08-21，在当前优化后的 OpenVINO runtime 下，用同一个 `profile_ops.py` 口径重测。
- **阶段 A/B**：前轮已经完成的 stock、opt1+2、opt1+2+3 阶段数据，用于拆分 plugin 优化收益。
- **精度/等价性验证**：来自第三轮 causal 导出验证结果，用于判断性能交换是否改变模型语义。

## 2. 测试环境和口径

| 项目 | 内容 |
|---|---|
| CPU/GPU | Intel Core Ultra 7 356H，32 EU Intel integrated GPU |
| 内存特征 | memory-frequency-limited，实测 DRAM ceiling 约 48 GB/s |
| OpenVINO runtime（第三轮历史） | `2026.3.0-22454-09c4cc4117e-qwen-ttft-optimizations` |
| OpenVINO commit | `09c4cc4117e220cb6771a626d30a45ed027901cd` |
| workload | GME language IR，sequence length 1024 |
| 输入 | `batch=1`，1024 token，全 1 attention mask |
| profiling | `profile_ops.py`，warmup 12，repeat 8 |
| 输出 | GPU profiling wall p50、profiled sum、各 exec_type 和 node |
| 主要优化 | GME opt-1/opt-2/opt-3，causal 为导出侧改动 |

统一重测结果目录为 `results/r3_unified_20260821/`。本次重测不覆盖历史结果文件。

### 2.1 指标定义

- **wall p50**：一次完整 OpenVINO infer 的 host wall time。
- **profiled sum**：plugin profiling counter 中已执行 GPU primitive 的时间总和。
- **host gap**：`wall p50 - profiled sum`，包含 primitive 提交、同步和其他 host/plugin 开销。
- **GEMM**：`jit:gemm:any__i8` 的累计时间。
- **SDPA**：`ocl::sdpa::opt__f16` 的累计时间。
- 所有表格中的 latency 单位均为 ms；百分比定义为 `(before - after) / before`。

## 3. 模型和导出组合

### 3.1 第三轮阶段存在的语言 IR

| precision key | 量化/导出含义 | causal | 是否纳入主比较 |
|---|---|---:|---:|
| `int4` | 默认 gs128，非对称，u4，group size 128 | 否 | 是 |
| `int4_causal` | gs128，非对称，causal SDPA，无显式 S×S mask | 是 | 是 |
| `int4_gs64` | gs64，非对称，u4，group size 64 | 否 | 是 |
| `int4_gs64_causal` | gs64，非对称，causal SDPA | 是 | 是 |
| `int4_per_oc` | per-output-channel scale，非对称 | 否 | 性能备选，不建议采纳 |
| `int4_per_oc_sym` | per-output-channel scale，对称 | 否 | 性能备选，不建议采纳 |
| `int4_sym_causal` | gs128，对称，causal SDPA | 是 | 可选量化对照 |
| `int4_gs64_sym_causal` | gs64，对称，causal SDPA | 是 | 可选量化对照 |
| `int4_v5ctrl` | transformers 5 对照导出，保留显式 BOOL mask | 否 | 失败/反例对照 |
| `int4_gs64_v5ctrl` | gs64 transformers 5 对照导出，保留显式 BOOL mask | 否 | 失败/反例对照 |

**重要缺口：** 当前模型包没有 `int4_per_oc_causal` 或 `int4_per_oc_sym_causal` 文件。
因此 per-OC 只有普通导出性能，不能把 per-OC 普通结果外推成 causal per-OC 结果。

### 3.2 causal 导出改变了什么

普通导出使用 `causal=false` 加显式 `[B,1,S,S]` mask；causal 导出使用
`causal=true`，不再把这张 mask 作为 SDPA 的计算输入。对本 workload，右 padding 和
末位有效 token pooling 保证这两种图在 FP32 上是恒等变换。

causal 路径的收益只应主要出现在 SDPA 和显式 mask 构造，不应改变 GEMM 的工作量。

## 4. 统一重测：第三轮阶段所有可用模型的 latency

以下均为 2026-08-21 统一重测、第三轮优化 runtime、1024 token、GPU、wall p50；不包含第四轮视觉 mask-free 改动。

| 模型 | 导出/量化 | wall p50 | profiled sum | host gap | GEMM | SDPA | DynamicQuantize | RMSNorm | permute |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `int4` | gs128，普通 | **237.77** | 220.56 | 17.21 | 152.15 | 26.04 | 25.12 | 9.31 | 0.88 |
| `int4_causal` | gs128，causal | **223.30** | 205.67 | 17.63 | 152.04 | 11.53 | 26.83 | 9.00 | 0.88 |
| `int4_gs64` | gs64，普通 | **251.85** | 234.68 | 17.17 | 166.02 | 25.82 | 25.50 | 9.35 | 0.88 |
| `int4_gs64_causal` | gs64，causal | **238.06** | 220.71 | 17.35 | 166.55 | 11.62 | 26.98 | 9.29 | 0.88 |
| `int4_sym_causal` | gs128，对称 causal | 215.12 | 198.16 | 16.95 | 145.41 | 11.64 | 25.45 | 9.39 | 0.88 |
| `int4_gs64_sym_causal` | gs64，对称 causal | 228.87 | 211.40 | 17.46 | 157.55 | 11.57 | 26.91 | 9.08 | 0.88 |
| `int4_per_oc` | per-OC，普通 | 222.36 | 204.90 | 17.46 | 135.80 | 25.80 | 25.77 | 9.35 | 0.88 |
| `int4_per_oc_sym` | per-OC，对称，普通 | 215.22 | 198.02 | 17.20 | 129.32 | 25.94 | 25.29 | 9.35 | 0.88 |
| `int4_v5ctrl` | BOOL mask 对照导出 | 297.88 | 273.27 | 24.61 | 151.61 | 未 lower | 26.92 | 9.16 | 9.72 |
| `int4_gs64_v5ctrl` | gs64 BOOL mask 对照导出 | 309.58 | 287.06 | 22.52 | 166.24 | 未 lower | 25.97 | 9.20 | 9.80 |

### 4.1 统一重测的直接结论

1. **causal 是第三轮阶段最有效的无损导出侧优化**：gs128 从 237.77 降到 223.30 ms，gs64 从
   251.85 降到 238.06 ms；GEMM 基本不变，SDPA 约减半。
2. **gs64 的主要代价在 GEMM**：普通 gs64 的 GEMM 166.02 ms，比 gs128 的 152.15 ms
   多 13.87 ms；causal 后这一差距仍然存在。
3. **per-OC 性能很好但精度不达标**：per-OC 普通为 222.36 ms，per-OC-sym 为 215.22 ms，
   但不能只看 latency 决定是否采用。
4. **v5ctrl 不是有效的 causal baseline**：BOOL mask 让 SDPA 没有走优化的 `ocl::sdpa::opt`，
   并重新出现 `broadcast`/`permute`；它只能作为“错误导出会发生什么”的反例。

## 5. causal 导出收益

### 5.1 gs128

| 配置 | wall p50 | SDPA | GEMM |
|---|---:|---:|---:|
| 普通 `int4` | 237.77 | 26.04 | 152.15 |
| causal `int4_causal` | 223.30 | 11.53 | 152.04 |
| 差值 | **−14.47 ms** | **−14.51 ms** | −0.11 ms |
| 加速 | **1.065x（−6.08%）** | **2.26x** | 基本不变 |

### 5.2 gs64

| 配置 | wall p50 | SDPA | GEMM |
|---|---:|---:|---:|
| 普通 `int4_gs64` | 251.85 | 25.82 | 166.02 |
| causal `int4_gs64_causal` | 238.06 | 11.62 | 166.55 |
| 差值 | **−13.79 ms** | **−14.20 ms** | +0.53 ms |
| 加速 | **1.058x（−5.47%）** | **2.22x** | 基本不变 |

### 5.3 causal 的语义验证

第三轮 FP32 等价性检查结果：

| 用例 | cosine | max_abs_diff |
|---|---:|---:|
| seq1 | 1.000000000 | 0.000e+00 |
| seq32 | 1.000000000 | 0.000e+00 |
| seq512 | 1.000000000 | 0.000e+00 |
| seq1024 | 1.000000000 | 0.000e+00 |
| batch2 padded | 1.000000000 | 0.000e+00 |
| batch3 padded | 1.000000000 | 0.000e+00 |

因此 causal 导出是恒等变换；量化后不同 IR 的 cosine 小数差异主要来自重新量化/AWQ
校准抖动，不能用它否定 FP32 已证明的图等价性。

## 6. plugin 优化点的 latency 贡献

### 6.1 gs128：stock → opt1+2 → opt1+2+3 → causal

这张表把 plugin 侧优化和导出侧 causal 分开：

| 阶段 | wall p50 | 相对前一阶段 | 说明 |
|---|---:|---:|---|
| stock 普通导出 | 283.09 | — | 旧轮次 stock reverify |
| opt1+2 普通导出 | 262.97 | **−20.12 ms / −7.11%** | GQA broadcast + attention output permute |
| opt1+2+3 普通导出 | 237.77 | **−25.20 ms / −9.58%** | opt-3 宽 INT4 权重重排 |
| opt1+2+3 + causal | 223.30 | **−14.47 ms / −6.08%** | 导出侧 SDPA causal，无 mask |

累计从 stock 283.09 到 causal 223.30 ms，约 **1.268x，降低 21.13%**。

**opt-1 和 opt-2 的独立贡献：** 当前项目保留的可审计阶段产物是 `opt1+2` 合并阶段，
没有同一构建下的 `opt1-only` 和 `opt2-only` GME plugin binary。因此本报告不把
−20.12 ms 虚假拆给 opt-1/opt-2；算子层面能确认的是：

- stock 的 `broadcast_gpu_ref` 约 10.1 ms，opt1+2 后归零；
- stock 的 `permute_ref` 约 4.9 ms，opt1+2 后约 0.88 ms；
- 二者合计减少约 15 ms，剩余收益来自 primitive 数量和提交路径变化。

### 6.2 gs64：stock → opt1+2+3 → causal

| 阶段 | wall p50 | 相对前一阶段 | 说明 |
|---|---:|---:|---|
| stock 普通导出 | 284.76 | — | 第三轮 gs64 stock profiling |
| opt1+2+3 普通导出 | 251.85 | **−32.91 ms / −11.56%** | 第三轮优化 runtime 统一重测 |
| opt1+2+3 + causal | 238.06 | **−13.79 ms / −5.47%** | 第三轮优化 runtime 统一重测 |

gs64 从 stock 到 causal 约 284.76 → 238.06 ms，约 **1.196x，降低 16.40%**。
注意 gs64 的 opt1+2+3 阶段收益比 gs128 大，不代表 gs64 一定更快；gs64 的量化 GEMM
本身更慢，导致最终 wall 仍比 gs128 causal 高 14.76 ms。

### 6.3 per-OC：stock → plugin optimized

| 模型 | stock wall p50 | optimized wall p50 | 差值 | 加速 |
|---|---:|---:|---:|---:|
| `int4_per_oc` | 268.67 | 222.36 | −46.31 ms | 1.209x / −17.24% |
| `int4_per_oc_sym` | 232.81 | 215.22 | −17.59 ms | 1.082x / −7.56% |

这两组 stock 数字来自已有的 per-OC stock profiling 文件，optimized 数字来自本次
统一重测；它们用于衡量 plugin 优化的方向和量级，不是同一 session 的交错 A/B，
因此不要把最后一位小数的差异解释为稳定收益。这里的模型都是普通导出，不是 causal
导出。当前 package 没有 per-OC causal IR，所以不存在
“per-OC causal latency”实测值。per-OC 只是性能上限/备选，不是推荐生产量化方案。

## 7. 不同量化方案的性能与精度取舍

### 7.1 性能排序（第三轮优化 runtime）

按 wall p50 从快到慢：

1. `int4_per_oc_sym`：215.22 ms
2. `int4_sym_causal`：215.12 ms
3. `int4_per_oc`：222.36 ms
4. `int4_causal`：223.30 ms
5. `int4_gs64_sym_causal`：228.87 ms
6. `int4_gs64_causal`：238.06 ms
7. `int4`：237.77 ms（普通 gs128；与 gs64 causal 接近）
8. `int4_gs64`：251.85 ms

`int4_v5ctrl` 和 `int4_gs64_v5ctrl` 不纳入正常排序，因为它们的 SDPA 没有 lower
到优化 kernel，属于导出失败对照。

### 7.2 质量边界

已有质量结果显示：

| 模型 | text cosine | query cosine | 结论 |
|---|---:|---:|---|
| `int4_causal` | 0.941730 | 0.955173 | 第三轮推荐默认 causal gs128 |
| `int4_gs64_causal` | 0.944205 | 0.957576 | 精度优先 causal 方案 |
| `int4_sym_causal` | 0.926687 | 0.941559 | 性能更快，但 query 明显下降 |
| `int4_gs64_sym_causal` | 0.937980 | 0.943423 | 不建议默认 |
| `int4_per_oc` | 0.826530 | 0.832787 | 性能快，但精度不可接受 |
| `int4_per_oc_sym` | 0.836570 | 0.882145 | 仍不满足默认质量要求 |

causal 的 FP32 图等价已经逐位验证；上表量化 cosine 变化不能归因给 causal 本身，
因为每种 IR 的 AWQ/scale calibration 会产生独立的量化噪声。

## 8. 结论和推荐

### 8.1 第三轮结束时默认推荐（第四轮前）

**第三轮阶段推荐 `int4_causal`：**（第四轮前的 language-only 结论）

- 第三轮优化 runtime 下 wall p50 约 223.3 ms；
- 相比普通 gs128 约再降 14.5 ms；
- SDPA 从约 26.0 ms 降到 11.5 ms；
- GEMM 基本不变；
- FP32 causal 等价性逐位成立；
- query cosine 仍保持在约 0.955。

### 8.2 精度优先

**推荐 `int4_gs64_causal`：**

- wall p50 约 238.1 ms；
- 比普通 gs64 降约 13.8 ms；
- query cosine 约 0.9576，高于 gs128 causal；
- 相比 `int4_causal` 多约 14.8 ms，换取约 0.0024 query cosine。

### 8.3 不建议默认

- `int4_per_oc` / `int4_per_oc_sym`：性能快，但 text/query cosine 显著下降。
- `int4_sym_causal` / `int4_gs64_sym_causal`：性能更快，但 query cosine 下降约 0.014，
  不满足本项目精度优先前提。
- `int4_v5ctrl` / `int4_gs64_v5ctrl`：BOOL mask 导致 SDPA 分解，不能作为有效 causal 对照。

## 9. 数据缺口和未完成项

| 项目 | 状态 | 处理方式 |
|---|---|---|
| per-OC causal | 没有对应 XML/BIN | 明确标记未生成，未做外推 |
| opt-1-only vs opt-2-only | 没有同构建阶段 binary | 以可审计的 opt1+2 合并阶段记录，不拆分虚假单项收益 |
| gs64 的 opt1-only/opt1+2-only | 没有同构建阶段 binary | 使用 stock→opt1+2+3→causal 三段实际数据 |
| causal v5ctrl | 有普通 v5ctrl，对应 causal 是正确导出方案 | v5ctrl 作为失败对照，不纳入正常性能结论 |
| opt-7 在 GME | 现成 plugin opt-7 guard 不命中 GME per-token/预计算 reduction 配置 | GME 本轮不声称有 opt-7 收益；causal 是导出侧收益 |

## 10. 复现命令

统一 profiling：

```bash
source <python-env>/bin/activate
source <openvino-install>/setupvars.sh
cd gme-Qwen2-VL-2B-Instruct-Running/gme-qwen2-vl-int4-hotspot-optimization

for p in int4 int4_causal int4_gs64 int4_gs64_causal \
         int4_sym_causal int4_gs64_sym_causal \
         int4_per_oc int4_per_oc_sym int4_v5ctrl int4_gs64_v5ctrl; do
  python profile_ops.py --precision "$p" --length 1024 \
    --warmup 12 --repeat 8 --output "results/r3_unified_20260821/${p}.json"
done
```

第三轮 causal 的 FP32 等价性验证使用 `check_causal_equivalence.py`，结果文件为
`results/r3_causal_equivalence_fp32.json`。

## 11. 原始结果索引

- 统一重测：`results/r3_unified_20260821/*.json`
- causal/普通第三轮旧复核：`results/r3_final_int4*.json`
- gs128 统一阶段：`results/r2_reverify.json`、`results/opt12_ops_int4_1024.json`、
  `results/r2_final_ops_1024.json`
- gs64 stock/optimized：`results/r3_STOCK_ops_int4_gs64_1024.json`、
  `results/r3_base_ops_int4_gs64_1024.json`
- per-OC stock/optimized：`results/per_oc_ops_1024_stock.json`、
  `results/per_oc_ops_1024_patched.json`、`results/per_oc_sym_ops_1024_stock.json`、
  `results/per_oc_sym_ops_1024_patched.json`
- causal FP32 等价：`results/r3_causal_equivalence_fp32.json`

这份历史报告的核心结论是：**第三轮阶段最值得采用的是 causal 导出；它几乎只优化 SDPA，
不会改变 GEMM latency。gs64 是精度优先选项，per-OC/对称量化是性能上限但有明显精度
代价。plugin 侧 opt-1/opt-2/opt-3 仍然有效，但 opt-1/opt-2 的独立贡献当前只能以
合并阶段观察。**
