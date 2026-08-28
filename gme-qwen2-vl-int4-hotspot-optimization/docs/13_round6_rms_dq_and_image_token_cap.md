# 第六轮：语言塔 RMS→DQ 深度融合，以及 image token cap 的时延/精度曲线

第五轮之后，最终 INT4 配置是 `int4_causal_vopt`：语言塔已经有 causal SDPA 和三项 GPU plugin patch，视觉塔已经有 mask-free、headfirst 和 rank3。本轮继续从最终配置重新 profiling，而不是沿用早期热点。

本轮得到两类结论：

1. **语言塔**：先去掉 sibling FC 重复的 DynamicQuantize，再实现 GME `hidden=1536` 的 RMSNorm→per-token symmetric INT8 DynamicQuantize 融合。1024 token 的三轮交错 A/B arithmetic mean 为 **220.28 → 211.63 ms（1.041x）**，且 129/512/1024 和 padded batch 相对 control **逐位相同**。但 128 token 为 46.57 → 47.68 ms，短 query cosine 也下降 0.00626，因此融合只作为 `GPU_DYNAMIC_QUANTIZATION_THRESHOLD=0` 的长序列 opt-in，不改变默认路径。
2. **图像侧**：最终视觉 IR 的剩余大头是 GEMM、SDPA、DQ、QKV permute 和 MVN。对超宽 logo 实测 `max_image_tokens=768`，把实际 visual tokens 从 1248 降到 729，image mean **1930 → 994 ms（1.943x）**，相对 PyTorch FP32 的 image cosine 只从 0.943164 降到 0.941129。MMDocIR `balanced_100` 上 page/layout 编码再快 **1.183x/1.460x**；16 个 close/open retrieval 指标中最大负向变化为 **−0.0110**，最大正向变化为 +0.0200。

---

## 13.0 环境与基线

| 项 | 内容 |
| --- | --- |
| runtime base | OpenVINO `2026.3.0` + `patches/all_gme_int4_gpu_opts.patch` |
| 模型 | `gme-Qwen2-VL-2B-Instruct`，`int4_causal_vopt` |
| 设备 | Intel Panther Lake-P iGPU |
| 结果目录 | `results/r6_20260827/` |
| retrieval | MMDocIR revision `bdcb36ecb3eee73667180ee3fb24fe433f6dd2a4` |

最终语言 IR 的 fresh 1024-token profile：

| exec type | calls | ms | 占比 |
| --- | ---: | ---: | ---: |
| `jit:gemm:any__i8` | 140 | 151.87 | 73.9% |
| `dynamic_quantize_gpu_opt` | 140 | 26.82 | 13.1% |
| `ocl::sdpa::opt` | 28 | 11.51 | 5.6% |
| `rms_gpu_bfyx_opt` | 57 | 9.04 | 4.4% |
| `ocl::rope::opt` | 56 | 4.63 | 2.3% |
| **wall mean** | | **223.6** | |

RMS 和 DQ 合计约 35.9 ms，其中每个 decoder RMS 的 f16 输出被 DQ 立即读回；这是本轮的语言塔目标。

---

## 13.1 先做 DQ 去重

同一 LayerNorm 输出会同时进入 sibling FC。原图为每个 FC 各插一个 DynamicQuantize；属性完全相同时可以共享同一个 DQ 输出。

实测：

| | 原始 | DQ reuse |
| --- | ---: | ---: |
| DQ calls | 140 | **112** |
| DQ ms | 26.82 | **23.16** |
| RMS calls | 57 | 57 |
| wall mean | 223.6 | **222.5** |

调用数每层少一个，和 gate/up 共享 `post_attention_layernorm` 的拓扑完全一致。该优化不改量化公式，只复用同一 activation 上属性相同的 DQ，是默认可采纳项，也是后续 RMS→DQ 融合的前置条件：去重后每个可融合 RMS 只剩一个 DQ user。

---

## 13.2 RMS→DQ 融合

### 13.2.1 为什么旧 opt-7 零命中

第三轮依据当时路径认为 GME per-token DQ 是 asymmetric u8；重新检查最终 package 的 runtime graph 后，56 个可融合节点实际全部是：

- per-token：`group_sizes = [1, 1, UINT64_MAX]`；
- **symmetric i8**，不是 asymmetric u8；
- 无 precomputed reduction；
- 输入/输出 f16，hidden size 1536；
- DQ reuse 后每个 RMS 只有一个 DQ user。

旧 qwen3.5 opt-7 只支持 `group_size=128 + symmetric i8`，所以仍然零命中。真正缺的是 **per-token symmetric i8** 分支。

### 13.2.2 第一个原型为什么数值失败

第一版直接沿用 DQ 的 12 个 sub-group × 128 element 分块做 RMS reduction。性能生效，RMS 57 → 1，但 reduction tree 与原 `rms_gpu_bfyx_opt` 不同，层间累计后数值不可接受：

| case | control vs 第一版 fused cosine |
| --- | ---: |
| seq1 | 0.963440 |
| seq32 | 0.996584 |
| seq512 | 0.993877 |
| seq1024 | 0.996051 |
| padded batch | 0.994795 |

这说明“同一数学公式”不够；必须复刻原 kernel 的元素归属、逐 lane 加法顺序和 SLM reduction tree。

### 13.2.3 采纳实现：复刻原 RMS reduction

最终 fused kernel 限定为 GME 的 `hidden=1536`，使用与 standalone RMS 相同的：

- 256 work-items / 16 sub-groups；
- 每 lane 6 个元素；
- 相同 `native_powr(x, 2)` 累加顺序；
- 相同 sub-group reduce 和 16-slot SLM halving tree；
- normalized value 在进入 DQ min/max 前显式舍入到 f16，保留原 kernel 边界。

修复后：

| case | cosine | max abs | 结论 |
| --- | ---: | ---: | --- |
| seq129 | 1.000000000 | 0 | bit-identical |
| seq512 | 1.000000000 | 0 | bit-identical |
| seq1024 | 1.000000000 | 0 | bit-identical |
| batch2 padded | 1.000000000 | 0 | bit-identical |

seq1/7/32 仍与 default control 不同，原因不是 reduction：默认 DQ 在 activation rows ≤64 时跳过量化，fused node 无法把同一输出动态切换为 f16 pass-through。为保护默认短序列行为，matcher 只有在 compile config 显式设置 `GPU_DYNAMIC_QUANTIZATION_THRESHOLD=0` 时才启用。

### 13.2.4 三轮交错 A/B

control 和 fused 使用保存的两个 GPU plugin 二进制，按 control→fused 交替 3 轮；每轮 warmup=8、repeat=6，表中是 18 个 timed calls 的 arithmetic mean。

| tokens | control | fused | Δ | speedup |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 40.14 | **38.88** | −1.27 | 1.033x |
| 128 | **46.57** | 47.68 | +1.11 | 0.977x |
| 512 | 124.22 | **120.67** | −3.54 | 1.029x |
| 1024 | 220.28 | **211.63** | −8.66 | **1.041x** |

1024-token fused profile 中：

- RMS 57 → **1**，9.04 → 0.16–0.17 ms；
- DQ 保持 112 calls，约 25–26 ms；
- 剩下的 1 个 RMS 是最终输出归一化，不接 FC/DQ；
- 主要其他算子数不变。

### 13.2.5 精度与部署方式

无条件 fused 版本对 PyTorch FP32 golden（该 plugin-only A/B 使用 cap=1280；
显式 cap=768 的候选精度另见 §13.5）：

| precision | text | query | image | fused |
| --- | ---: | ---: | ---: | ---: |
| 第五轮 `int4_causal_vopt` | 0.941730 | 0.955173 | 0.943164 | 0.955472 |
| always-DQ + RMS fusion | 0.941730 | **0.948910** | 0.943164 | 0.955472 |
| `int8_causal_vopt` control | 0.997899 | 0.998784 | 0.987115 | 0.996531 |

query 的 −0.00626 来自短输入原本会绕过 DQ，现在被强制量化；长输入、image 和 fused 不受该问题影响。

因此默认 `GmeOpenVINO(...)` 保持 threshold=64，不融合、短输入输出不变。只对已知以长文档为主的独立实例使用：

```python
encoder = GmeOpenVINO(
    package,
    device="GPU",
    precision="int4_causal_vopt",
    dynamic_quantization_threshold=0,
)
```

GPU plugin 现将已有的 `GPU_DYNAMIC_QUANTIZATION_THRESHOLD` 暴露为 RW property 并纳入 compiled-model cache key；没有新造配置项。

---

## 13.3 最终视觉 IR 的剩余热点

`gme_visual_int4_nomask_headfirst_rank3.xml` fresh profile：

| exec type | L=1024 | 占比 | L=5040 | 占比 |
| --- | ---: | ---: | ---: | ---: |
| `jit:gemm:any__i8` | 78.66 | 47.7% | 383.89 | 26.7% |
| `ocl::sdpa::opt` | 32.99 | 20.0% | **752.40** | **52.3%** |
| DynamicQuantize | 15.16 | 9.2% | 107.37 | 7.5% |
| QKV `permute_ref` | 18.15 | 11.0% | 88.50 | 6.2% |
| MVN / LayerNorm | 9.04 | 5.5% | 49.63 | 3.4% |
| RoPE | 9.17 | 5.6% | 48.78 | 3.4% |
| **wall mean** | **183.1** | | **1436.4** | |

结论：

- L=1024 仍由 GEMM 主导；opt-3 已只加速满足 `N >= 4K` 的 `mlp.fc1`，其余投影盲目放宽已有负收益证据。
- L=5040 已由全注意力 SDPA 主导；它没有 causal 或 block-diagonal 结构可删，继续优化 kernel 常数只能拿小比例收益。
- 32 次 QKV permute 是一次必要的 `[L,3,H,D] -> [3,H,L,D]` 真转置，除非 FC 直接写 head-major output。
- 64 个 LayerNorm 已把 gamma/beta 融进 MVN，随后直接接 DQ。MVN→DQ 是下一项 plugin 候选，理论 traffic 收益 9–50 ms，但需要迁移 MVN fused-op dependencies 和复刻 mean/variance 两次 reduction；本轮不做未经验证的宽改动。

---

## 13.4 `max_image_tokens` Pareto 曲线

使用 public `GmeOpenVINO` API、同一超宽 `gme_logo.png`（5164×600）、INT4 `_vopt`，每档 warmup=2 / repeat=5。cap 是 processor 上限，实际 visual tokens 受 28-pixel factor 和长宽整除约束，不一定等于 cap。

| cap | actual visual tokens | image mean | fused mean | image cos vs FP32 | fused cos vs FP32 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 230 | 348.0 | 348.1 | 0.901687 | 0.892995 |
| 384 | 342 | 478.0 | 477.1 | 0.918228 | 0.923546 |
| 512 | 462 | 623.8 | 627.9 | 0.926113 | 0.933661 |
| **768** | **729** | **993.5** | **996.5** | **0.941129** | **0.950249** |
| 1024 | 930 | 1318.0 | 1314.1 | 0.939915 | 0.951991 |
| 1280 | 1248 | 1930.0 | 1936.5 | 0.943164 | 0.955472 |

相对 cap=1280 embedding，cap=768 的 image/fused cosine 是 0.972791 / 0.984431；相对 PyTorch FP32，image 只降 0.00204、fused 降 0.00522，而 latency 约减半。cap=1024 比 768 慢约 32%，在该样本上的 cosine 收益很小，因此 768 是值得进入 retrieval 评估的 Pareto 点；256/384 的精度损失明显，不建议。

---

## 13.5 MMDocIR `balanced_100` paired retrieval

同一 `int4_causal_vopt`、recommended batch profile、100 queries / 19 documents /
328 pages / 5898 layouts，对比 cap=1280 与 cap=768。

### 13.5.1 编码时间

| stage | cap=1280 | cap=768 | speedup |
| --- | ---: | ---: | ---: |
| query | 3.16 s | 3.02 s | 1.046x（应视为噪声；纯文本不受 cap 影响） |
| page | 293.58 s | **248.22 s** | **1.183x** |
| layout | 680.57 s | **466.10 s** | **1.460x** |
| page + layout | 974.15 s | **714.32 s** | **1.364x** |

layout 里 5477/5898 是纯文本，理论上不受 cap 影响；真正被加速的 421 个 fused
image/table layout 只占 7.1%，所以总 layout 还能达到 1.46x，说明长图的绝对节省很大。

### 13.5.2 Retrieval 指标

| scope | level | metric | cap=1280 | cap=768 | Δ |
| --- | --- | --- | ---: | ---: | ---: |
| close | page | nDCG@5 / @10 | 0.774980 / 0.789285 | **0.777676 / 0.798683** | +0.002696 / +0.009398 |
| close | page | Recall@5 / @10 | 0.884167 / 0.920000 | **0.884167 / 0.940000** | 0 / +0.020000 |
| close | layout | nDCG@5 / @10 | **0.452395 / 0.479695** | 0.448529 / 0.468673 | −0.003867 / −0.011022 |
| close | layout | Recall@5 / @10 | 0.515796 / **0.586418** | **0.523272** / 0.579154 | +0.007477 / −0.007264 |
| open | page | nDCG@5 / @10 | 0.698452 / **0.712068** | **0.700966** / 0.711020 | +0.002514 / −0.001048 |
| open | page | Recall@5 / @10 | 0.778333 / 0.813333 | **0.788333** / 0.813333 | +0.010000 / 0 |
| open | layout | nDCG@5 / @10 | **0.409997 / 0.427506** | 0.400101 / 0.417300 | −0.009896 / −0.010206 |
| open | layout | Recall@5 / @10 | 0.466029 / 0.509965 | **0.468958 / 0.517970** | +0.002930 / +0.008005 |

16 个指标的最大绝对变化是 close page Recall@10 的 +0.020；负向最大是 close layout
nDCG@10 的 −0.0110。page 基本持平或略升，layout nDCG 下降约 0.01，但 Recall 没有一致
下降。`balanced_100` 的二值比例指标最坏抽样标准误约 5pp，因此这些差异都低于子集的
分辨能力：可以排除明显回归，但不能宣称严格等价。

**结论：cap=768 是低时延候选，当前 package 默认仍为 1280。** 改默认前至少补
`balanced_200`，最终质量结论仍应跑 full set；当前使用时需显式传
`max_image_tokens=768`。

显式 cap=768 的 golden 验证为：

| precision | text | query | image | fused |
| --- | ---: | ---: | ---: | ---: |
| `int4_causal_vopt` | 0.941730 | 0.955173 | **0.941129** | **0.950249** |
| `int8_causal_vopt` | 0.997899 | 0.998784 | **0.971184** | **0.979576** |

text/query 不经过 processor resize，完全不变。INT8 image/fused 的下降比 INT4 更显眼，
因为 INT8 原本的量化噪声更低；需要 near-lossless image 的 INT8 使用方应选择 1280 quality 档。

---

## 13.6 产物与复现

新增/修改：

```text
gme-Qwen2-VL-2B-Instruct-Running/
  gme_openvino.py
      新增 dynamic_quantization_threshold 可选参数，仅传给 language compile
  artifacts/openvino_validation_round6_rms_dq.json
       artifacts/openvino_validation_round6_default_final.json
       artifacts/openvino_validation_round6_default_cap768.json

gme-qwen2-vl-int4-hotspot-optimization/
  profile_image_token_caps.py
  results/r6_20260827/
      base_int4_causal_vopt_1024.json
      dq_reuse_int4_causal_vopt_1024.json
      rms_dq_fused_int4_causal_vopt_1024.json
      ab_{control,fused}_{32,128,512,1024}_r{1,2,3}.json
      outputs_{control,rms_dq_exact}_int4_causal_vopt.npz
      visual_vopt_l{1024,5040}.json
      image_token_caps_logo_int4.json

mmdocir_openvino_eval/results/token_cap_balanced_100/
       int4_1280/summary.json
       int4_768/summary.json
```

语言 profile：

```bash
python profile_ops.py --precision int4_causal_vopt --length 1024 \
       --warmup 14 --repeat 10 --output results/r6_20260827/default_1024.json

python profile_ops.py --precision int4_causal_vopt --length 1024 \
       --warmup 14 --repeat 10 --config GPU_DYNAMIC_QUANTIZATION_THRESHOLD=0 \
       --output results/r6_20260827/rms_dq_fused_1024.json
```

image token cap：

```bash
python profile_image_token_caps.py --precision int4_causal_vopt \
       --caps 256,384,512,768,1024,1280 --warmup 2 --repeat 5 \
       --output results/r6_20260827/image_token_caps_logo_int4.json
```

MMDocIR：

```bash
cd ../../mmdocir_openvino_eval
SUBSET_FILE="$PWD/subsets/balanced_100.json" PRECISIONS=int4_causal_vopt \
OUTPUT_DIR="$PWD/results/token_cap_balanced_100/int4_1280" \
./run_campaign.sh --max-image-tokens 1280

SUBSET_FILE="$PWD/subsets/balanced_100.json" PRECISIONS=int4_causal_vopt \
OUTPUT_DIR="$PWD/results/token_cap_balanced_100/int4_768" \
./run_campaign.sh --max-image-tokens 768
```

---

## 13.7 当前建议

- **通用默认**：保留 `int4_causal_vopt`、默认 threshold=64；加入 DQ reuse，短 query 精度和行为不变。
- **长 text/query 专用实例**：设置 `dynamic_quantization_threshold=0`，在 512/1024 token 上得到约 1.028x/1.043x；不要与短 query 共用该实例。
- **image/fused 低时延候选**：显式 cap=768；超宽 logo 约 1.94x，MMDocIR page/layout encoding 1.18x/1.46x，`balanced_100` 未见超过 0.0111 的负向指标变化。改 package 默认前补 balanced_200/full。
- **image/fused quality 档**：保留 cap=1280；cap=256/384 的单图 cosine 损失明显，不建议。
- **下一项 plugin 深挖**：视觉 MVN→DQ 融合，其次才是让 QKV FC 直接写 head-major output。长图主导项仍是全注意力 SDPA，结构性降 token 数的杠杆远大于局部 kernel 调优。
