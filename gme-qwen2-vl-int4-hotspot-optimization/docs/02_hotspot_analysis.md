# 2. Hotspot 分析

> 本文记录旧 masked visual / language INT4 基线上的 hotspot。第四轮已在导出侧
> 增加 mask-free visual IR；当前 image/fused 结果见 [`docs/11`](11_round4_image_fused_optimization.md)，
> 不要用本文的历史视觉时间作为当前默认性能。

## 2.1 VTune GPU hotspots（定性）

```bash
source /opt/intel/oneapi/setvars.sh
vtune -collect gpu-hotspots -result-dir results/vtune-gpu-hotspots-int4-1024 \
      -- python profile_gme_int4_gpu.py --length 1024 --warmup 14 --repeat 30
vtune -report summary -r results/vtune-gpu-hotspots-int4-1024
```

summary（原始输出见 `results/vtune_gpu_summary.txt`）：

```
GPU Time: 28.138s
XVE Array Stalled/Idle: 58.5% of Elapsed time with GPU busy
Occupancy: 69.0% of peak value
    Hottest GPU Computing Tasks with Low Occupancy
    Computing Task  Total Time  SIMD Width  Peak XVE Threads Occupancy(%)  SIMD Utilization(%)
    gemm_kernel         7.895s          16                          40.0%               100.0%
    gemm_kernel         5.951s          16                          40.0%               100.0%
    gemm_kernel         4.291s          16                          40.0%               100.0%
```

按 computing-task 展开（`-group-by computing-task -format csv`）后的 GPU 时间分布：

| kernel | 总时间 (s) | 占 GPU 时间 | 实例数 | XMX active | 主要 stall |
| --- | ---: | ---: | ---: | ---: | --- |
| `gemm_kernel` ×4 组 | 19.3 | 68.6% | 12232 | 21–33% | SBID 38–47%, Barrier 13–26% |
| `sdpa_micro__prefill` | 2.455 | 8.7% | 2464 | 18.0% | SBID 50.8% |
| `dynamic_quantize_gpu_opt` ×3 组 | 2.328 | 8.3% | 9856 | 15–18% | SBID 60% |
| `broadcast_gpu_ref` | 0.925 | 3.3% | 4928 | 14.6% | SBID 56.3% |
| `rms_gpu_bfyx_opt` | 0.782 | 2.8% | 5016 | 15.8% | SBID 40.9% |
| `permute_ref` ×2 | 0.574 | 2.0% | 4928 | 19–32% | SBID 50–57% |

三个要点：

1. **GEMM 是绝对主体（~69%）**，且 XMX 流水线只有 21–33% 的时间在发射指令，
   stall 的主因是 SBID（内存 scoreboard 依赖），说明它是 **访存/延迟受限**，
   不是算力受限。
2. summary 里那句 `GPU Time, % of Elapsed time: 18.3%` 是**假象** —— 旧 harness
   在计时之外做了上千次 tokenizer 调用（§1.4），跟 GPU 利用率无关。
3. `broadcast_gpu_ref` / `permute_ref` 这两个 **reference kernel**（未优化实现）
   出现在热点里，是典型的"图没融合干净"信号。

## 2.2 算子级归因（定量，`profile_ops.py`）

VTune 只给 kernel 名，给不出"是哪一层的哪个算子"。用 plugin 自带的 profiling
counter 按 exec_type + node name 双维度聚合，得到干净 2026.3.0 基线
（INT4 / GPU / seq=1024，8 次取平均）：

### 按 GPU kernel

| ms/infer | % | calls | exec_type |
| ---: | ---: | ---: | --- |
| 177.99 | 67.5 | 140 | `jit:gemm:any__i8`（oneDNN） |
| 28.03 | 10.6 | 28 | `ocl::sdpa::opt__f16` |
| 26.55 | 10.1 | 140 | `dynamic_quantize_gpu_opt__f16` |
| **10.10** | **3.8** | **59** | **`broadcast_gpu_ref__f16`** |
| 9.14 | 3.5 | 57 | `rms_gpu_bfyx_opt__f16` |
| **4.93** | **1.9** | **56** | **`permute_ref__f16`** |
| 4.65 | 1.8 | 56 | `ocl::rope::opt___f16` |
| 2.41 | 0.9 | ~110 | 其它（eltwise / select / gather / count_nonzero…） |
| **263.8** | | ~590 | profiled 合计（wall 283.0） |

### 按原始算子（层号归一化后）

| ms/infer | % | calls | node | kernel |
| ---: | ---: | ---: | --- | --- |
| 63.03 | 24.1 | 28 | `layers.N.mlp.gate_proj/MatMul` | `jit:gemm:any__i8` |
| 47.41 | 18.1 | 28 | `layers.N.mlp.up_proj/MatMul` | `jit:gemm:any__i8` |
| 42.30 | 16.2 | 28 | `layers.N.mlp.down_proj/MatMul` | `jit:gemm:any__i8` |
| 27.84 | 10.6 | 28 | `layers.N.self_attn/SDPA` | `ocl::sdpa::opt__f16` |
| 12.85 | 4.9 | 28 | `layers.N.self_attn.q_proj/MatMul_with_bias_fused_3FCs` | `jit:gemm:any__i8` |
| 11.96 | 4.6 | 28 | `layers.N.self_attn.o_proj/MatMul` | `jit:gemm:any__i8` |
| **5.16** | **2.0** | 28 | `layers.N.self_attn/aten::expand/Broadcast`（K） | `broadcast_gpu_ref__f16` |
| **4.98** | **1.9** | 28 | `layers.N.self_attn/aten::expand/Broadcast_1`（V） | `broadcast_gpu_ref__f16` |
| 4.57 | 1.7 | 28 | `layers.N.input_layernorm` | `rms_gpu_bfyx_opt__f16` |
| 4.55 | 1.7 | 28 | `layers.N.post_attention_layernorm` | `rms_gpu_bfyx_opt__f16` |
| **3.96** | **1.5** | 28 | `layers.N.self_attn/aten::transpose/Transpose_3`（SDPA 输出） | `permute_ref__f16` |
| 3.79 | 1.4 | 28 | `layers.N.self_attn/RoPE Add` | `ocl::rope::opt___f16` |
| **2.27** | **0.9** | 28 | `layers.N.self_attn/aten::transpose/Transpose_2`（V） | `permute_ref__f16` |
| ~26.5 | 10.1 | 140 | `DynamicQuantize_*`（5 个/层） | `dynamic_quantize_gpu_opt__f16` |

## 2.3 GEMM：为什么 68% 的时间只跑出 15 TFLOPS

`ONEDNN_VERBOSE=1` 显示所有线性层都落到 oneDNN `jit:gemm:any`，activation 动态量化
到 s8、weight 保持 u4：

```
matmul,jit:gemm:any,src:s8::blocked:abc wei:u4::blocked:cab dst:f16::blocked:abc,
  attr-scales:src0:7:f16:1x1536 + wei:7:f16:128x1
  attr-zero-points:wei:7:u8:128x1
  attr-post-ops:eltwise_swish:1+binary_mul:f16:6:abc      <- gate_proj
```

几个直接结论：

**(a) SwiGLU 已经融进 GEMM。** `gate_proj` 的 post-ops 同时挂了 `eltwise_swish` 和
`binary_mul`，即 `silu(gate) * up` 整体是 gate GEMM 的 post-op。这解释了
gate(63.0 ms) 比形状完全相同的 up(47.4 ms) 慢 15.6 ms —— binary_mul 要额外读一遍
up 的 `[1024, 8960]` f16 输出（17.9 MB × 28 层 ≈ 500 MB，按实测 ~50 GB/s 约 10 ms）。
**这里没有可捡的漏。**

**(b) 实测吞吐随 M 饱和在 ~15 TFLOPS。**

| seq | GEMM 时间 (ms) | GFLOP | 有效 TFLOPS |
| ---: | ---: | ---: | ---: |
| 512 | 101.61 | 1341 | 13.2 |
| 1024 | 178.78 | 2682 | 15.0 |
| 2048 | 366.22 | 5364 | 14.6 |

M=1024 已经过了拐点，**不是 batch 太小导致的占用率问题**。

**(c) int8 通路只比 f16 快 1.2×，而 XMX 的 int8 峰值是 f16 的 2×。**

| 配置 | GEMM (ms) | DynamicQuantize (ms) | wall (ms) |
| --- | ---: | ---: | ---: |
| INT4 权重 + s8 激活（默认） | 177.99 | 26.55 | 283.0 |
| INT4 权重 + f16 激活（`DYNAMIC_QUANTIZATION_GROUP_SIZE=0`） | 213.63 | 0 | 287.5 |
| INT4 权重 + s8 激活，group=32 | 220.51 | 97.93 | 392.2 |
| INT8 权重 + s8 激活 | 191.40 | 29.88 | 284.4 |

f16 GEMM 12.5 TFLOPS vs int8 GEMM 15.0 TFLOPS —— 只有 1.2×。**动态量化整套机制
净收益只有 4.5 ms**（GEMM 省 35.6 ms，DynamicQuantize 花掉 26.5 ms，再加上一点
调度开销）。这说明 oneDNN 的 int8 GEMM 离它自己的 roof 还差约 2×，瓶颈在
`jit:gemm` 生成器内部（权重解压 + zero-point 补偿 + SBID stall），不在 OpenVINO 侧。

INT8 权重比 INT4 权重慢 13 ms，恰好等于多出来的权重带宽
（1.30 GB vs 0.65 GB，按 ~50 GB/s ≈ 13 ms），交叉验证了"访存受限"的判断。

**(d) 结论：GEMM 这 178 ms 不是本轮能在 OpenVINO 源码层解决的**，它属于
oneDNN `jit:gemm` 内核生成器的范畴（见 §5.1）。

## 2.4 真正的 OpenVINO 侧漏洞：图没融合干净

去掉 GEMM 之后剩下 **85.8 ms（32.5%）**。逐项排查后找到两处**明确的
plugin 图优化缺口**，合计 **19.0 ms（6.7%）**：

### 缺口 1：GQA `repeat_kv` 的 Broadcast 没有被吃掉（10.10 ms）

IR 里 K/V 是标准的 PyTorch `repeat_kv` 展开：

```
K: [B, 2, S, 128] --Unsqueeze--> [B, 2, 1, S, 128] --Broadcast--> [B, 2, 6, S, 128]
                  --Reshape--> [B, 12, S, 128] --> SDPA
```

GPU plugin 里本来就有 `UnsqueezeBroadcastReshapeSDPAFusion` 专门消除它，但它的
pattern 把 Unsqueeze 的输入**写死成 `KVCache` 节点**：

```cpp
auto input_b_kvcache_m = wrap_type<ov::intel_gpu::op::KVCache>({any_input(), any_input()});
auto unsqueeze_b_m = wrap_type<ov::op::v0::Unsqueeze>({input_b_kvcache_m, axes_const_b_m}, ...);
```

**只有有状态生成式模型才有 `KVCache` 节点。** GME 这种 encoder / embedding 前向是
无状态的，K 直接来自 RoPE、V 直接来自 Transpose，pattern 不匹配 → 融合不触发 →
`broadcast_gpu_ref` 每层实打实把 K 和 V 各放大 6 倍写回显存
（每次读 0.5 MB 写 3.1 MB，56 次共 ~200 MB），而 SDPA 内部本来就支持 GQA 广播。

### 缺口 2：SDPA 输出的 Transpose 没有被吃掉（3.96 ms + 连带）

`op::SDPA` 有 `output_transpose_order` 字段，OCL SDPA 生成器也确实按它重映射目标
stride（`convert_strides("DST", "OUTPUT", extended_output_transpose_order)`）。
但 `TransposeFusion` 里只有 `TransposeSDPAMatcher`（融合**输入** transpose），
**没有对应的输出 transpose matcher** —— 对比之下 MatMul 是有的
（`TransposeMatMulTransposeMatcher`）。

于是标准 attention 收尾的
`SDPA -> Transpose(0,2,1,3) -> Reshape -> o_proj` 里那个 Transpose 只能落到
`permute_ref`，把 `[B, 12, S, 128]` 完整读写一遍。

顺带发现：`Transpose_2`（V 的 `[B,S,2,128] -> [B,2,S,128]`）本来可以被
`TransposeSDPAMatcher` 融进 `input_v_transpose_order`，但因为它和 SDPA 之间隔着
上面那串 Unsqueeze/Broadcast/Reshape 而不相邻，所以也没融上。缺口 1 修好之后
这一段被拆掉，V 的 transpose 变成 SDPA 的直接输入，代价从 2.27 ms 降到 0.88 ms。

### 缺口 3（已评估，收益小）：DynamicQuantize 重复

每层有 5 个 `DynamicQuantize`：qkv 的、o_proj 的、**gate 的、up 的**、down 的。
gate 和 up 消费的是同一个 `post_attention_layernorm` 输出，却各量化了一遍。
实测去重只值 ~1.3 ms（0.5%），且社区已有对应实现思路（见 §5.3），本轮不做。
