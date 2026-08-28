# 第八轮：QKV head-major 路径与 INT8 grouped MVN→DQ

本轮同时尝试第七轮留下的两条方向：

1. 让视觉 QKV 投影直接或更高效地到达 `[3,H,L,D]` head-major layout；
2. 把视觉 INT8 的 `MVN + affine` 融入 `group_size=128 + precomputed_reduction` DQ。

所有性能结论统一使用 **arithmetic mean**。每个 A/B 格子交错 3 轮，每轮 warmup=3、
repeat=5，最终对 15 个 raw latency samples 求均值；不使用 p50 作为优化结论。

最终结论：

- QKV direct-write 的普通 rank-4 MatMul 会掉出 compressed oneDNN 路径，否决；
- 采纳 QKV 专用 vectorized transpose kernel：INT4/INT8 四-grid 全部 bit-identical，
  INT4 visual mean **1.005–1.020x**，public image/fused mean **1.004–1.017x**；
- INT8 grouped MVN→DQ 可以做到四-grid bit-identical，但 fused DQ 的 SLM 开销更大，
  visual mean **回归 1.7–3.7%**，已从交付源码回退。

---

## 15.0 环境

| 项 | 内容 |
| --- | --- |
| runtime baseline | OpenVINO `2026.3.0` + round-7 combined patch |
| visual IR | `gme_visual_{int4,int8}_nomask_headfirst_rank3.xml` |
| device | Intel Panther Lake-P iGPU |
| 结果目录 | `results/r8_20260828/` |
| 性能统计 | 每格 3×5 raw samples 的 arithmetic mean |

---

## 15.1 QKV direct-write 可行性

### 15.1.1 当前图

每层 QKV 路径是：

```text
FC/MatMul: [1,L,1280] x [3840,1280]^T -> [1,L,3840]
Add bias
Reshape: [1,L,3840] -> [L,3,16,80]
Transpose: [L,3,16,80] -> [3,16,L,80]
Split/Squeeze -> Q/K/V [16,L,80]
```

目标物理 offset 为：

```text
head-major[qkv,h,l,d] = (((qkv*16+h)*L+l)*80+d)
```

它不能由逻辑 `[1,L,3840]` 的单一 affine stride 描述，因为 channel 轴还要拆成
`[3,16,80]`，且 `L` 插入在 head 和 D 之间。因此不能只给现有 FC dst 换一个 stride。

### 15.1.2 rank-4 batched MatMul probe：否决

在内存图中把 block 0 改成：

```text
A: [1,1,L,1280]
W: [3,16,80,1280]
MatMul(transpose_b=true) -> [3,16,L,80]
```

它能直接替换原 Reshape+Transpose，也能编译运行，但执行类型从 compressed
`jit:gemm:any__i8` 变为 **`jit:gemm:any__f16`**。只改 block 0 时最终 visual output
cosine 已降到 0.95243 / max_abs 24.95，说明 GEMM 分块/量化路径已经改变；单次 mean
171.85→171.27 ms 也没有可信收益。

结论：真正的 direct-write 需要扩展 rank-4 `FullyConnectedCompressed` 和 oneDNN grouped
matmul，不能用普通 OV MatMul 图改写冒充。

---

## 15.2 采纳：QKV 专用 vectorized transpose

既然一次真实转置无法消失，下一步是替换低效的 `permute_ref`。该变换有非常严格且稳定的
结构：input `[L,3,16,80]`，output `[3,16,L,80]`；每个 output `[qkv,h,l,:]`
对应 input `[l,qkv,h,:]` 的连续 80 个 f16。

新增 `permute_qkv_headfirst` kernel：

- 只接受 planar f16 / bfyx；
- internal order 精确为 `[1,3,2,0]`；
- input F=3、Y=16、X=80；output B=3、F=16、X=80；
- 无 fused ops、无 padding、dense pitches；
- 每个 work-item `vload8/vstore8`，每行 10 个 block；
- GWS `{16,L,48}`、LWS `{16,1,1}`，6 个 x lanes 空闲；
- dynamic L 由原 permute dispatch update 机制处理。

曾尝试 16 lanes × 5 values 的 subgroup block read/write，但 Intel block intrinsic 的地址/lane
语义并非简单 row copy，四-grid 输出明显错误，已回退。还测试了 5×`half16`：它逐位
相同，但 L=1024 的 wall mean 比 half8 慢 0.31%，permute mean 14.35 vs 13.98 ms；
L=5040 wall 只在噪声内快 0.11%，permute 本身仍慢 1.51 ms，因此最终采用 half8。

### 15.2.1 正确性

strict control 只不注册新 kernel；两边其余 plugin、MVN fusion、threshold 完全相同。

| precision | grids | 结果 |
| --- | --- | --- |
| INT4 / INT4-gs64 visual | L=1024/2304/2916/5040 | 全部 `max_abs=0` |
| INT8 visual | L=1024/2304/2916/5040 | 全部 `max_abs=0` |
| standard golden | INT8 + INT4 + INT4-gs64，text/query/image/fused | 与 round-7 完全相同 |
| compiled-model cache reload | INT4 L=1024 | `max_abs=0` |

### 15.2.2 INT4 visual arithmetic mean

| grid | control | optimized | speedup | permute mean |
| --- | ---: | ---: | ---: | ---: |
| 1,32,32（L=1024） | 172.63 | **169.50 ms** | **1.019x** | 18.14→13.97 ms |
| 1,18,162（L=2916） | 629.78 | **623.24 ms** | **1.011x** | 49.62→41.86 ms |
| 1,48,48（L=2304） | 453.98 | **445.30 ms** | **1.020x** | 42.98→34.07 ms |
| 1,24,210（L=5040） | 1391.70 | **1384.18 ms** | **1.005x** | 86.99→79.48 ms |

长序列上 SDPA 占比超过 50%，因此即使 permute 省 7–9 ms，wall 比例也只有约 0.5%。

### 15.2.3 Public image/fused arithmetic mean

新版 `profile_gme_image.py` 保存每次 raw samples，并直接输出 mean。

| size | mode | visual mean | e2e mean | e2e speedup |
| --- | --- | ---: | ---: | ---: |
| 448x448 | image | 161.90→**158.53** | 245.28→**241.61** | **1.015x** |
| 448x448 | fused | 162.21→**158.45** | 248.56→**244.45** | **1.017x** |
| 672x672 | image | 442.10→**432.40** | 599.15→**589.68** | **1.016x** |
| 672x672 | fused | 441.57→**433.35** | 601.76→**593.14** | **1.015x** |
| 1288x728 | image | 1272.39→**1265.48** | 1608.17→**1596.99** | **1.007x** |
| 1288x728 | fused | 1272.14→**1265.68** | 1607.74→**1600.67** | **1.004x** |

### 15.2.4 INT8 visual arithmetic mean

QKV transpose 的 tensor 仍是 f16，和权重压缩精度无关，因此同一 kernel 也命中 INT8：

| grid | control | optimized | speedup | permute mean |
| --- | ---: | ---: | ---: | ---: |
| L=1024 | 190.55 | **186.80 ms** | **1.020x** | 18.20→13.97 ms |
| L=5040 | 1472.27 | **1463.74 ms** | **1.006x** | 90.34→81.71 ms |

---

## 15.3 INT8 grouped MVN→DQ：正确但性能否决

### 15.3.1 INT8 DQ 契约

INT8 visual 的 DQ 与 INT4 不同：

```text
input               [1,L,1280] f16
group_sizes          [1,1,128]
quantized output     [1,L,1280] i8
scales               [1,L,10] f16
precomputed sums     [1,L,10] i32
```

每个 compressed FC 使用 source scales 和 precomputed reductions。原 large-GS DQ 是 10 个
SIMD16 subgroup，每个处理一个 128-element group。

### 15.3.2 实现与数值

实验 fused kernel 保留 256-WI MVN reduction tree和三个 f16 affine 边界，然后用 SLM 对
10×128 groups 分别做 abs-max、quantize 和 i32 sum。第一版 reduction 条件误写成
`local_id < 128`，offset=64 时发生越界；修成 `local_id < offset` 后四-grid 全部
bit-identical，MVN 65→2 calls，FC 仍为 130 次 `jit:gemm:any__i8`。

### 15.3.3 Arithmetic mean A/B

| grid | control | grouped fused | speedup | MVN mean | DQ mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| L=1024 | **190.36** | 197.65 ms | **0.963x** | 9.21→0.33 | 16.09→35.65 |
| L=2304 | **496.69** | 508.15 ms | **0.977x** | 21.26→0.75 | 50.27→82.49 |
| L=5040 | **1469.24** | 1494.68 ms | **0.983x** | 48.74→1.61 | 110.57→180.87 |

10×128 SLM max/sum reduction和额外 barriers 的开销显著超过被删 MVN。该实现已从交付源码
完全回退，只保留结果 JSON 和本节否决结论。

---

## 15.4 测量口径修正

`profile_visual_ops.py` 现在用 arithmetic mean 聚合 exec type、node、profiled sum 和 host gap；
仍写 `wall_p50_ms` 仅为历史 JSON 兼容。`profile_gme_image.py` 新增：

- stage/e2e `*_mean_ms`；
- 每次 preprocess/visual/language/end-to-end raw samples；
- 终端主输出使用 mean。

round-6 RMS→DQ 和 round-7 MVN→DQ 主表也已根据保存的 raw/per-round mean 重算，不再用
p50 作为性能结论。

---

## 15.5 最终采纳与后续

采纳：

- `permute_qkv_headfirst`，同时惠及 INT4/INT8；
- mean-based profiling 工具和 raw sample 持久化。

否决：

- 普通 rank-4 MatMul direct-write：掉出 compressed i8 路径且数值变化大；
- INT8 grouped MVN→DQ：bit-identical 但 mean latency 回归 1.7–3.7%；
- subgroup block-copy QKV kernel：数值不正确。

若继续追 direct-write，需要在 `FullyConnectedCompressed`/oneDNN JIT 内增加真正的 blocked
QKV destination，不是图层面的普通 stride 或 rank-4 MatMul。长图的最大剩余结构杠杆仍是
全注意力 SDPA 或减少 visual tokens。

---

## 15.6 产物

```text
gme-Qwen2-VL-2B-Instruct-Running/
  04_validate_python.py
  gme_openvino.py
  artifacts/openvino_validation_round8_qkv.json

gme-qwen2-vl-int4-hotspot-optimization/
  profile_visual_ops.py
  profile_gme_image.py
  docs/15_round8_qkv_permute_and_int8_grouped_mvn.md
  patches/opt6_qkv_headfirst_permute.patch
  patches/all_gme_int4_gpu_opts_round8.patch
  results/r8_20260828/
    qkv_ab_{control,fused}_*_r{1,2,3}.json
    qkv_e2e_{control,fused}_r{1,2,3}.json
    qkv_int8_ab_{control,fused}_*_r{1,2,3}.json
    int8_grouped_ab_{control,fused}_*_r{1,2,3}.json
    int8_grouped_mvn_equivalence.json
```
