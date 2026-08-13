# GME Qwen2-VL-2B-Instruct — GPU INT4 text/query hotspot 分析与优化

针对 `gme-Qwen2-VL-2B-Instruct` 在 Intel iGPU 上 INT4 精度做纯文本
`text` / `query` embedding 时延迟偏高（1024 token ≈ 281 ms）的问题，做了完整的
hotspot 定位、源码优化、数值验证和回归测试。

**所有源码改动基于 OpenVINO 官方 release tag `2026.3.0`（`8a17657b99`），
不掺杂本仓库既有的任何本地提交或工作区改动**，patch 见 [`patches/`](patches/)。

---

## TL;DR

| | |
| --- | --- |
| **结果** | 1024 token **1.170–1.172x**、512 token **1.180–1.184x**、128 token **1.151–1.158x**、32 token **1.017–1.018x** |
| **正确性** | text/语言 IR 结果见 §4；image-only 三尺寸 bit-identical；fused 仍可运行但部分尺寸有小幅稳定漂移，见 `docs/04_validation_and_results.md` §4.1c |
| **回归** | 有状态生成式模型 greedy 解码 token id、首步 logits SHA 与 top-8 逐位相同 |
| **改动** | Intel GPU plugin，7 个文件，+297 −17，图结构/调度 + 权重 layout，无算术改动 |
| **改了什么** | ① GQA `repeat_kv` 的 Broadcast 消除放开到无状态模型；② 新增 SDPA 输出 Transpose 融合 + 修复 `sdpa_gen_micro` 的 dispatch bug；③ **宽 INT4 投影（`N ≥ 4K`）的权重重排成 N 连续**，让 oneDNN `jit:gemm` 走更快的 layout |
| **剩下的大头** | GEMM 仍占 GPU 时间 ~63%。已定价的剩余杠杆只有 per-OC 量化（1.55x，**有精度代价**），见 docs/06 §6.3 |

## 优化前后的 GPU 时间构成（语言 IR，seq=1024，INT4/GPU）

| exec_type | baseline (ms) | opt1+2 (ms) | opt1+2+3 (ms) | Δ 总计 |
| --- | ---: | ---: | ---: | ---: |
| `jit:gemm:any__i8`（oneDNN，所有线性层） | 177.99 | 177.51 | **151.10** | **−26.9** |
| `ocl::sdpa::opt__f16` | 28.03 | 26.05 | 25.90 | −2.1 |
| `dynamic_quantize_gpu_opt__f16` | 26.55 | 25.24 | 25.57 | −1.0 |
| `broadcast_gpu_ref__f16`（GQA repeat_kv） | **10.10** | **0** | **0** | **−10.1** |
| `rms_gpu_bfyx_opt__f16` | 9.14 | 9.42 | 9.38 | +0.2 |
| `permute_ref__f16`（attention transpose） | **4.93** | **0.88** | **0.88** | **−4.1** |
| `ocl::rope::opt___f16` | 4.65 | 4.64 | 4.66 | 0 |
| 其它 | 2.41 | 2.40 | 2.40 | 0 |
| **profiled 合计** | **263.8** | **246.2** | **219.9** | **−43.9** |
| **wall p50** | **283.0** | **264.1** | **238.2** | **−44.8 (−15.8%)** |

opt-3 的 26.4 ms **全部来自 `gate_proj`（63.00 → 49.55）和 `up_proj`（47.43 → 35.24）**，
其余线性层和其余算子不动 —— 这正是该优化的适用范围（见 docs/06 §6.9）。

---

## 文档

| 文档 | 内容 |
| --- | --- |
| [`docs/01_methodology_and_baseline.md`](docs/01_methodology_and_baseline.md) | 被测对象、为什么必须重建干净基线、三套测量工具、理论算力标尺 |
| [`docs/02_hotspot_analysis.md`](docs/02_hotspot_analysis.md) | VTune GPU hotspots、算子级归因、GEMM 为什么只有 15 TFLOPS、两处图融合缺口 |
| [`docs/03_optimizations.md`](docs/03_optimizations.md) | 三个 patch 的问题、改法、为什么安全、逐项收益 |
| [`docs/04_validation_and_results.md`](docs/04_validation_and_results.md) | 数值验证（opt1+2 bit-identical / opt3 的漂移量级）、生成式回归、交替 A/B 性能、VTune 复核 |
| [`docs/05_rejected_and_future_work.md`](docs/05_rejected_and_future_work.md) | 试过且**否掉**的 3 个方案（附实测数字）、后续 5 个方向、与既有 Qwen3.5 patch 的关系 |
| [`docs/06_round2_onednn_investigation.md`](docs/06_round2_onednn_investigation.md) | **第二轮**：oneDNN 单独立项、INT4 GEMM 成本分解 probe、VTune 复采、opt-3 的落地与两次规则修正（§6.9） |
| [`patches/README.md`](patches/README.md) | patch 的 base、应用方式、影响面、上游建议 |
| [`results/qwen3_ab_text/REPORT.md`](results/qwen3_ab_text/REPORT.md) | 三个 GME patch 对 Qwen3-VL-Embedding-2B 的适用性复测：opt-3 text 基本无收益，image/fused `CL_OUT_OF_RESOURCES` 回归 |

## 脚本

| 脚本 | 用途 |
| --- | --- |
| `profile_gme_int4_gpu.py` | 端到端 GME encoder 延迟（含 processor/tokenizer），workload 结果磁盘缓存 |
| `profile_ops.py` | 语言 IR 的 per-op GPU profiling，按 exec_type + node name 双维度聚合 |
| `dump_outputs.py` / `compare_outputs.py` | 固定输入下 dump 输出并做 cosine / max-abs-diff 对比 |
| `regress_stateful.py` | **GenAI-free** 有状态 IR greedy 解码回归（推荐） |
| `regress_generative.py` | 走 `openvino_genai` 的回归；⚠️ 本地 GenAI 构建与 wheel 不同，会有假阳性，见 docs/04 §4.2 |
| `bench_ab.sh` | 交替 A/B 端到端 benchmark（应对 GPU 双档频率） |

## 复现

```bash
cd /path/to/gme-Qwen2-VL-2B-Instruct-Running/gme-qwen2-vl-int4-hotspot-optimization
export HOTSPOT_ROOT="$PWD"
export GME_ROOT="${GME_ROOT:-$(cd "$HOTSPOT_ROOT/.." && pwd)}"
export MODEL_BENCH_ROOT="${MODEL_BENCH_ROOT:-$(cd "$GME_ROOT/.." && pwd)}"
export OPENVINO_ROOT="${OPENVINO_ROOT:-$MODEL_BENCH_ROOT/benchmark/openvino}"
export QWEN_ENV="${QWEN_ENV:-/home/chaofan/envs/qwen_env}"

source /opt/intel/oneapi/setvars.sh                       # VTune
source "$QWEN_ENV/bin/activate"

# --- 基线（官方 2026.3.0 wheel）---
python profile_ops.py --precision int4 --length 1024 --warmup 12 --repeat 8
python dump_outputs.py --precision int4 --output results/acc_int4_stock.npz

# --- 打 patch 并重编 GPU plugin ---
cd "$OPENVINO_ROOT"
git apply "$HOTSPOT_ROOT/patches/all_gme_int4_gpu_opts.patch"
cmake --build build --target openvino_intel_gpu_plugin -j && cmake --install build

# --- patched ---
source "$OPENVINO_ROOT/install/setupvars.sh"
python profile_ops.py --precision int4 --length 1024 --warmup 12 --repeat 8
python dump_outputs.py --precision int4 --output results/acc_int4_round2.npz
python compare_outputs.py results/acc_int4_stock.npz results/acc_int4_round2.npz

# --- 端到端交替 A/B + 回归 ---
./bench_ab.sh                     # 默认 1024 / 512 / 128 / 32
python regress_stateful.py --output results/r2_regress_stateful_patched.json

# --- VTune ---
vtune -collect gpu-hotspots -result-dir results/vtune-gpu-hotspots-int4-1024-patched \
      -- python profile_gme_int4_gpu.py --length 1024 --warmup 14 --repeat 30 --hold-seconds 30
```

> `--hold-seconds` 是必须的：加了 workload 缓存之后进程结束太快，VTune 会报
> "main thread is a zombie thread" 而 attach 失败。

## 目录

```
├── README.md                        本文件
├── docs/                            分析过程 / hotspot / 优化 / 测试结论
├── patches/                         基于官方 2026.3.0 的可交付 patch
├── onednn/                          第二轮的独立 oneDNN probe（脱离 OpenVINO 直接定价 GEMM）
│   ├── *_probe.cpp                  GEMM 成本分解 / u4 重排 / scale 排布 / layout 精度
│   ├── build_probe.sh               链接 oneDNN-ov2026.3 (babb7375ff)
│   └── results/                     probe 输出
├── results/                         profiling 输出、A/B 原始数据、精度对比、VTune 导出
│   ├── README.md                    每个结果文件的口径说明（务必先看这个）
│   └── ab/                          交替 A/B 的逐轮 JSON
├── preexisting-worktree-backup/     开工前 openvino / openvino.genai 工作区改动的备份
└── build-openvino-official/         早期尝试的独立官方源码构建目录（未使用，已 gitignore）
```

本目录现在位于 `gme-Qwen2-VL-2B-Instruct-Running/` 下。Python 脚本会按该布局
自动推导路径；迁移到其他工作区时可通过 `GME_ROOT`、`MODEL_BENCH_ROOT`、
`OPENVINO_ROOT` 和 `MODELS_ROOT` 覆盖默认值。`QWEN_ENV` 用于指定 Python 虚拟环境。

`.gitignore` 排除了 VTune 原始 result 目录（数百 MB、机器相关）。summary 导出为
`results/vtune_gpu_summary{,_patched}.txt`，按 computing-task 的完整指标表导出为
`results/vtune_tasks_{baseline,patched}.tsv`。

## 关于工作区状态

开工时 `benchmark/openvino` 工作区里 apply 着 Qwen3.5 的 5 个 patch，其中
`opt2_permute_lws_f_axis` 和 `opt3_dynamic_quantize_reuse` **正落在 GME 的热点路径上**，
用它做基线会污染结论。处理方式：

1. 先 `git diff` 备份到 `preexisting-worktree-backup/`（**未丢失，可随时恢复**）；
2. `git checkout -- src/` 回到 `2026.3.0`；
3. 重编 GPU plugin，以此为唯一基线。

`benchmark/openvino.genai` **未做任何修改**，其工作区改动也已备份。恢复方式：

```bash
cd benchmark/openvino
git apply ../../gme-qwen2-vl-int4-hotspot-optimization/preexisting-worktree-backup/openvino-worktree-before-gme-work.diff
```
