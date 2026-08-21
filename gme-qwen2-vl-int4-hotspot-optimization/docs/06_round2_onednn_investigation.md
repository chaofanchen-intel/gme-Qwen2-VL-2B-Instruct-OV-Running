# 6. 第二轮：oneDNN 立项与 INT4 GEMM 成本分解

第一轮（§1–§5）把两个 **图结构** 缺口修掉了，1024 token 从 283 ms 降到 264 ms。
剩下的时间几乎全在 oneDNN 的 INT4 GEMM 里，而第一轮只能给出一句"实测饱和在
~15 TFLOPS，不在 OpenVINO 源码可控范围，需单独立项"——**没有证据说明它为什么慢**。

第二轮就是把这句话拆开：单独立项 oneDNN，用一个可控的 micro-benchmark 把
"INT4 GEMM 慢" 分解成若干个**可单独开关的量化因素**，再判断哪一个能落到
OpenVINO 源码里。

---

## 6.0 第二轮的起点（必须先修正的测量口径）

⚠️ **本目录所有模型侧测量都必须先 `source install/setupvars.sh`**，否则跑的是
site-packages 里的官方 wheel，不是本地 patch 过的构建：

```bash
source /home/chaofan/envs/qwen_env/bin/activate
source /home/chaofan/model_benchmark/benchmark/openvino/install/setupvars.sh
python -c "import openvino; print(openvino.get_version())"
# patched : 2026.3.0-22452-c2638477a62-qwen-ttft-optimizations
# stock   : 2026.3.0-22451-8a17657b995-releases/2026/3
```

第二轮初期有若干次测量踩了这个坑（见 §6.7），结论一律作废后重测。
修正后的基准（1024 token，text/query，p50）：

| 构建 | p50 (ms) | min (ms) |
| --- | ---: | ---: |
| stock 2026.3 wheel | 283.1 / 283.6 | 281.6 |
| 本地 opt-1 + opt-2 | **264.8 / 265.2** | 263.0 |

### 算子分布（patched，1024 token，干净重测）

`results/round2_ops_recheck.json`，wall p50 **263.7 ms**，profiled 合计 247.6 ms：

| exec_type | ms | calls |
| --- | ---: | ---: |
| `jit:gemm:any__i8` | **177.22** | 140 |
| `dynamic_quantize_gpu_opt__f16` | 27.38 | 140 |
| `ocl::sdpa::opt__f16` | 25.83 | 28 |
| `rms_gpu_bfyx_opt__f16` | 9.25 | 57 |
| `ocl::rope::opt___f16` | 4.58 | 56 |
| `permute_ref__f16` | 0.88 | 28 |
| 其它 | ~2.5 | — |

GEMM 占 profiled 时间的 **71.6%**。第一轮的 `broadcast_gpu_ref`（10.1 ms）已归零、
`permute_ref` 从 4.93 ms 降到 0.88 ms，两个 patch 的效果在第二轮复现。

**新增：按线性层拆分**（这是第一轮没有的粒度，后面 §6.3 要用）：

| linear | ms/次推理 | ms/call | 形状 (M×K→N) |
| --- | ---: | ---: | --- |
| `mlp.gate_proj` | 63.05 | 2.252 | 1024×1536 → 8960 |
| `mlp.up_proj` | 47.38 | 1.692 | 1024×1536 → 8960 |
| `mlp.down_proj` | 42.19 | 1.507 | 1024×8960 → 1536 |
| `self_attn.qkv`（融合） | 13.03 | 0.465 | 1024×1536 → 2048 |
| `self_attn.o_proj` | 11.58 | 0.414 | 1024×1536 → 1536 |

MLP 三个 GEMM 合计 152.6 ms = 整体 wall 的 **58%**。任何优化只要打不到
MLP 的 GEMM 上，天花板就很低——这决定了第二轮的全部投入方向。

---

## 6.1 VTune 第二轮 GPU hotspots

```bash
source /opt/intel/oneapi/setvars.sh
source /home/chaofan/model_benchmark/benchmark/openvino/install/setupvars.sh
vtune -collect gpu-hotspots -result-dir results/vtune-r2-gpu-hotspots-patched \
      -- python profile_gme_int4_gpu.py --length 1024 --warmup 14 --repeat 30
```

> 注：必须用激活 venv 后的 `python`。直接launch `python3` 会让 VTune 报
> *"main thread is a zombie thread / Failed to attach"*。

三次采集对比（同一台机器、同一 workload）：

| | 第一轮 stock | 第二轮 stock（复测） | 第二轮 patched |
| --- | ---: | ---: | ---: |
| GPU Time | 28.138 s | 24.718 s | **23.221 s** |
| XVE Array Stalled/Idle | 58.5% | 52.8% | **53.4%** |
| Occupancy（占峰值） | 69.0% | 72.6% | **73.4%** |

按 computing-task 展开（patched，GPU compute 合计 22.56 s，
原始数据 `results/vtune_r2_task_table.txt`）：

| kernel | 时间 (s) | 占比 | 实例 | XMX active | active% | stall% | occ% | 主要 stall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `gemm_kernel` #1 | 5.84 | 25.9% | 2464 | 31.9% | 47.4 | 52.4 | 79.8 | SBID 47%, IFetch 11%, Barrier 10% |
| `gemm_kernel` #2 | 4.50 | 19.9% | 4840 | 33.5% | 52.6 | 46.8 | 75.4 | SBID 46%, Barrier 22%, IFetch 14% |
| `gemm_kernel` #3 | 4.40 | 19.5% | 2464 | 31.9% | 51.3 | 47.8 | 78.4 | SBID 39%, Barrier 14%, IFetch 12% |
| `sdpa_micro__prefill` | 2.28 | 10.1% | 2464 | 25.4% | 38.1 | 57.6 | 58.8 | SBID 60%, Dist/Acc 17%, Send 13% |
| `dynamic_quantize_gpu_opt` #1 | 1.53 | 6.8% | 2376 | 19.3% | 37.8 | 61.8 | 78.2 | SBID 60%, Barrier 35% |
| `gemm_kernel` #4 | 1.21 | 5.4% | 2464 | 29.7% | 51.7 | 45.3 | 75.3 | SBID 42%, Dist/Acc 15% |
| `dynamic_quantize_gpu_opt` #2 | 0.89 | 4.0% | 9856 | 26.9% | 51.5 | 46.3 | 70.7 | SBID 48%, Dist/Acc 20% |
| `rms_gpu_bfyx_opt` | 0.79 | 3.5% | 5016 | 34.5% | 55.5 | 43.3 | 74.9 | SBID 42%, IFetch 15% |
| `rope_opt` | 0.39 | 1.7% | 2464 | 18.7% | 34.7 | 59.9 | 67.4 | SBID 55%, Dist/Acc 16% |
| `gemm_kernel` #5 | 0.23 | 1.0% | 88 | 22.8% | 23.6 | 75.9 | 78.8 | SBID 64%, Send 31% |

**两个结论，和第一轮方向一致但数字更硬：**

1. 五组 `gemm_kernel` 合计 **16.18 s / 22.56 s = 71.7%**，和算子级 profiling 的
   71.6% 完全吻合，交叉验证通过。
2. GEMM 的 **XMX active 只有 29–33%**——DPAS 流水线三分之二时间没在发射。
   Occupancy 却有 75–80%，说明**线程够多、但线程都在等**。stall 主因是
   **SBID（内存 scoreboard 依赖）39–47%**，其次 Barrier 10–22%、
   Instruction Fetch 11–14%。

也就是：**不是算力不够，也不是并行度不够，是依赖链把 XMX 饿死了**。
这与"反量化（unpack u4 + 减 zp + 乘 scale）在关键路径上"的假设一致，
但 VTune 只能到这里——要证实必须能单独开关每个因素。这就是 §6.2 的动机。

---

## 6.2 为什么自己写 probe，而不用 benchdnn

oneDNN 自带 `benchdnn`，本来是这件事的标准工具。**在这台 Xe3 驱动上不可用**：

```bash
cd /home/chaofan/model_benchmark/oneDNN-ov2026.3/build
./tests/benchdnn/benchdnn --matmul --mode=P --engine=gpu ...   # 卡死
```

`--mode=P`（performance）会在 `stream_t::wait()` 里挂住不返回：

* 它的 OpenCL stream profiler 路径挂死；
* 换 threaded execution context 也挂死；
* 没有受支持的绕过开关。

为确认不是用法问题，在本地给 benchdnn 加了一个 `BENCHDNN_NO_GPU_PROFILING=1`
逃生口（`oneDNN-ov2026.3/tests/benchdnn/dnnl_common.cpp:574-585`）——**没有解决**，
挂点在更下面。

> ⚠️ 那处 benchdnn 改动是**本地测量工具的改动，不是交付物**，不进任何 patch。
> probe 取代它之后该改动已无意义，保留仅为记录。

于是改为自己写最小 probe：`onednn/gemm_probe.cpp`。测量方式刻意做成
**和 OpenVINO GPU plugin 驱动这个 primitive 的方式一致**——host 侧 wall clock
包住一批 submission，最后一次 `strm.wait()`，不使用任何 profiling 接口。

```bash
onednn/build_probe.sh                 # 链接 oneDNN-ov2026.3（babb7375ff，与 OV 2026.3 vendored 版本一致）
onednn/gemm_probe 300
```

10 次迭代和 300 次迭代的结果一致到 ~1%，可复现。

**probe 与模型的 primitive 严格对齐**（用 `ONEDNN_VERBOSE=1` 从模型里抄出来的）：

```
attr-scales:src0:7:f16:1x1536 + wei:7:f16:128x1
attr-zero-points:wei:7:u8:128x1
weights format tag: cab            # ← 关键
```

* scale/zp mask 一律 **7**（三个维度全开），不是直觉上的 6；
* 权重 layout 是 **`cab`**（batch==1 时等价于 `acb`），即 **K 连续**；
* `attr-precomputed-reductions` 在模型里只出现 32 次（在 RoPE 那个 matmul 上），
  **MLP 的 GEMM 上没有**，所以 probe 里不加。

---

## 6.3 INT4 GEMM 成本分解（核心结果）

`onednn/results/gemm_decompose_final.txt`。以 GME 实跑的
`A_gme_cab` 为 1.00x 基准；gate/up 形状 28.19 GFLOP。

| case | 含义 | gate 1×1024×1536:1536×8960 | 倍率 |
| --- | --- | ---: | ---: |
| `A_gme_cab` | **GME 实际执行的 primitive** | 1.291 ms / 21.8 T | 1.00x |
| `A_abc` | 只把权重 layout 换成 N 连续 | **1.033 ms / 27.3 T** | **1.25x** |
| `A_any` | 让 oneDNN 自己挑（挑到 `abc`） | 1.033 ms / 27.3 T | 1.25x |
| `A_cab_scK` | 只改 scale/zp layout | 1.289 ms / 21.9 T | 1.00x |
| `A_abc_scK` | abc 权重 + K 连续 scale | 1.034 ms / 27.3 T | 1.25x |
| `A2_cab_swiglu` | 加上 SwiGLU post-op 链 | 1.344 ms / 21.0 T | 0.96x |
| `B_cab_symmetric` | 去掉 grouped zero-point | 1.180 ms / 23.9 T | 1.09x |
| `B_abc_symmetric` | abc + 对称 | 1.008 ms / 28.0 T | 1.28x |
| `C_s8wei_grouped` | 权重换 s8（给 u4 unpack 定价） | 0.963 ms / 29.3 T | 1.34x |
| `D_cab_per_oc` | group-128 → per-output-channel scale | **0.832 ms / 33.9 T** | **1.55x** |
| `D_abc_per_oc` | abc + per-OC | 0.948 ms / 29.7 T | 1.36x |
| `D2_cab_group512` | group 128 → 512 | 1.307 ms / 21.6 T | 0.99x |
| `E_f16src_u4wei` | f16 激活 × u4 权重 | 全部形状 **unsupported** | — |
| `F_f16_dense` | 稠密 f16 参考 | 1.494 ms / 18.9 T | 0.86x |
| `G_s8_dense` | 稠密 s8 参考（**int8 DPAS 天花板**） | 0.777 ms / 36.5 T | 1.66x |

### 读法

1. **权重 layout 是最大的单一可控因素：`cab` → `abc` 白拿 1.25x。**
   而且它**只是 layout**，不动量化方案、不动数值。这是全表唯一
   "免费" 的 1.25x。qkv 那种窄形状（N=2048）几乎没差别（1.01–1.03x），
   收益集中在 N=8960 的 MLP 上——正好是占 58% wall 的那三个 GEMM。

2. **grouped zero-point 只值 9%**（`B_cab_symmetric` 1.09x）。
   第一轮从源码推出的假设是"grouped zp 迫使 `Tb=s8`，是主要开销"——
   **这个假设被实测否掉了**。省下的 9% 还要付非对称量化的精度代价，不值。

3. **scale/zp 的 layout 完全不影响性能**（`A_cab_scK` 1.00x，`A_abc_scK` 1.25x
   与 `A_abc` 同）。这一条在 §6.5 变成了关键的**正确性**结论。

4. **u4 unpack 本身值 1.34x**（`C_s8wei_grouped`），但这是权重从 4 bit 变 8 bit，
   模型体积翻倍，不是可选项。

5. **group-128 → per-OC 值 1.55x**，几乎摸到稠密 int8 天花板（1.66x）。
   但这要求**重新量化**（改量化方案 → 改精度），不能悄悄做，只能作为
   带精度评估的备选（§6.6）。

6. `F_f16_dense` 18.9 T vs `G_s8_dense` 36.5 T：int8 相对 f16 拿到了 1.93x，
   **接近理论 2x**。所以硬件的 int8 通路是健康的——
   第一轮"int8 只比 f16 快 1.2x"的观察是被 INT4 反量化开销污染的结论，
   这里得到修正：**问题在反量化，不在 int8 DPAS**。

### 与模型内实测的差距

把 probe 的单算子时间和模型内的 ms/call 对齐：

| linear | 模型内 ms/call | probe 同形状 | 差距 |
| --- | ---: | ---: | ---: |
| gate_proj | 2.252 | 1.291 | 1.74x |
| up_proj | 1.692 | 1.298 | 1.30x |
| qkv | 0.465 | 0.236 | 1.97x |

模型内**普遍比单算子慢 ~1.3–2.0x**。为找原因逐个排除：

| 假设 | 做法 | 结果 |
| --- | --- | --- |
| GPU 降频 | 全程后台采样 `/sys/class/drm/card1/.../freq0/` | 两边中位数都是 **2450 MHz**，排除 |
| 权重不在 cache | `PROBE_ROTATE=28` 轮换 28 份权重（193 MB） | **无变化**，排除 |
| dispatch 形状特殊 | probe 复现 `1024×1×1536`（M=1、token 在 batch 维） | **无变化**，排除 |
| 少了 submission 流水 | `PROBE_SYNC=1` 每次迭代都 `strm.wait()` | 只慢 **4%**，排除 |
| per-op profiling 的开销 | 干净重测（§6.0） | wall 263.7（开 profiling）vs 264.8（关），**≈0**，排除 |

五条都排除后，**残余 ~1.3–1.5x 差距仍未解释**。已知的线索：

* `gate_proj` 和 `up_proj` **形状完全相同、FLOP 完全相同**，模型内却差 33%
  （2.252 vs 1.692），而 probe 里两者一致（1.291 vs 1.298）。
  probe 测得 SwiGLU post-op 只值 4%，解释不了 33%。
* 怀疑是 **L2/内存子系统在真实依赖链下的表现**（模型里 5 个 GEMM + DQ + RMSNorm
  串成一条链，probe 里只有一个 primitive 在跑），VTune 的 SBID stall 也指向这里。

这一条列为未解决项（§6.6）。**它不影响 §6.4 的结论**：layout 的 1.25x 是在
同一套测量口径下的相对量，模型内应当同比例受益。

---

## 6.4 `cab` → `abc`：可行性已验证

这是第二轮唯一"免费"的大杠杆，所以单独查了它到底为什么没发生、以及能不能做。

### 为什么现在是 `cab`

`src/plugins/intel_gpu/src/graph/impls/onednn/fully_connected_onednn.cpp:112-131`：

```cpp
if (prim_input_size == 3) {
    target_fmt  = dnnl::memory::format_tag::abc;
    weights_fmt = weights_transposed ? dnnl::memory::format_tag::acb
                                     : dnnl::memory::format_tag::abc;
}
```

Qwen2 从 PyTorch 导出后，`nn.Linear` 的权重是 `[N, K]` 行优先，MatMul 是
`x @ W.T` → `transpose_b=true` → `weights_transposed=true` → `acb`（K 连续）。
这里的 format tag **是在描述常量在内存里已有的排布，不是在提要求**——
直接改成 `abc` 而不搬数据会算错。

### 为什么 load 时没有做权重重排

`src/graph/graph_optimizer/post_optimize_weights.cpp:52-64`：

```cpp
if (impl->is_dynamic()) {
    if (p.is_internal_program())                              return;
    if (node.get_preferred_impl_type() == impl_types::onednn)  return;   // ← GME 命中这条
    if (node.type() != fully_connected::type_id())            return;
}
```

GME 的 language tower 是**动态 shape** 编译的，FC 又选了 oneDNN impl，
于是 load-time 权重重排被**设计性地跳过**。上游注释写的理由是
"动态 shape 下选中的权重格式可能在运行时变化"。
对本例这个理由不成立：权重是 `[N, K]` 静态常量，layout 选择和动态的 M 维无关。

另外 `fully_connected_onednn.cpp` 里**根本没有 `get_weights_reorder()`**
（conv / deconv / lstm / gru 都有），所以即使放开上面那个 guard 也没东西可跑。

### 三个前置风险，已逐一实测排除

**(1) oneDNN 能否重排 packed u4？** —— `onednn/u4_reorder_probe.cpp`

```
correctness (hand-checkable shapes):
  tiny 8x4               OK   小形状逐 nibble 对照参考值
  small 128x256          OK
  group-aligned 1536x64  OK
compile-time cost at real GME weight sizes:
  gate/up 1536x8960      OK   0.302 ms  (6.6 MiB)
  down    8960x1536      OK   0.341 ms  (6.6 MiB)
  qkv     1536x2048      OK   0.089 ms  (1.5 MiB)
  o_proj  1536x1536      OK   0.069 ms  (1.1 MiB)
VERDICT: u4 cab->abc reorder is supported and bit-correct.
```

支持，逐 nibble 位正确，且**代价可忽略**：全模型 28 层 × 5 权重
≈ **30 ms 一次性编译期开销**。

**(2) scale/zp 的排布是否与权重 layout 绑定？** —— `onednn/scale_layout_probe.cpp`

这是最危险的一条。plugin 把 scale/zp 传给 oneDNN 时**把 md 拍平成 1-D**
（`fully_connected_onednn.cpp` 的 `get_arguments()`）：

```cpp
desc = onednn::layout_to_memory_desc_flatten(scale_mem->get_layout(), format_tag::a);
```

stride 信息全丢，oneDNN 只能按某个约定顺序解释这块 buffer。
**如果那个约定是从权重 layout 推出来的**，改成 `abc` 就会静默取错 scale ——
而且因为 md 被拍平，从外部完全看不出来；scale/zp 又没有 reorder 钩子。

所以做了一个 host 端参考实现（小形状、值都能被 f16 精确表示），
把 {权重 cab, abc} × {scale n-outer, g-outer} 四种组合全跑一遍：

```
  wei cab + scale n-outer  (== GME today)   MISMATCH   max_rel=6.333
  wei cab + scale g-outer                   MATCH      max_rel=0
  wei abc + scale n-outer                   MISMATCH   max_rel=6.333
  wei abc + scale g-outer                   MATCH      max_rel=0
```

**结论：oneDNN 要的是 `g`-outer（稠密 `[1, K/group, N]`），且与权重 layout
无关。** `cab` 和 `abc` 用同一份 scale buffer 都是 **位精确**（max_rel = 0）。
⇒ 换 layout **完全不需要动 scale/zp 常量**。风险解除。

（副产品：说明 OpenVINO 侧那个 scale 常量本来就是 `g`-outer 排布的。）

**(3) 动态 shape 下重排结果往哪放？** ——
`src/graph/primitive_inst.cpp:2534` 的 `primitive_inst::update_weights()`
已经提供了**运行时**权重重排 + `_reordered_weights_cache` 缓存，
并且明确对 `fully_connected` 生效。所以**不需要碰 `post_optimize_weights`
那个 guard**：只要 FC 的 oneDNN impl 返回非空的 `get_weights_reorder_params()`，
首次执行时就会重排一次并缓存。

### 预期收益

按 §6.3 的倍率外推（qkv / o_proj 基本无收益，忽略）：

| linear | 现在 (ms) | abc 后 (ms) | 省 |
| --- | ---: | ---: | ---: |
| gate_proj | 63.05 | 50.4 | −12.6 |
| up_proj | 47.38 | 37.6 | −9.8 |
| down_proj | 42.19 | ~35.2 | ~−7.0 |
| **合计** | | | **≈ −29 ms** |

263.7 → **≈ 235 ms，约 1.12x**，且**不改任何算术**（位精确可验证）。

### 落地路径（两条，均已探明）

* **A. oneDNN reorder**：`weights_fmt` 改成 `abc` + 新增 `get_weights_reorder()`，
  目标 layout 用 `format::custom` 走 `reorder_onednn`。
  优点是复用已验证的 u4 重排（§6.4-(1)）。
* **B. OCL reorder**：`reorder_weights_int4` 这个 kernel **已经支持 4-bit 权重转置**，
  但方向声明反了——`src/kernel_selector/kernels/reorder/reorder_weights_int4.cpp:135`
  只开了 `ioyx → oiyx`，需要的是镜像的 `oiyx → ioyx`。
  `.cl` 里那个分支是通用 gather（走 `GET_FILTER_INDEX`），加镜像分支约 10 行。

两条都不需要改 oneDNN 本身，**patch 仍然完全落在 OpenVINO 2026.3 之上**。

---

## 6.5 否掉的方案：GatedMLP 融合

`FuseGatedMLP` 会把 gate/up/down + SwiGLU 融成一个 `gated_mlp` 节点。
GME 的 MLP 结构完全匹配，看起来是现成的收益。**实测大幅倒退，已否掉。**

它默认不开，而且在 release 构建里**永远不可能开**：

```cpp
// intel_gpu/include/intel_gpu/runtime/options.inl:102
OV_CONFIG_DEBUG_OPTION(ov::intel_gpu, disable_gated_mlp_fusion, true, "...")
```

`OV_CONFIG_DEBUG_OPTION` 需要 `ENABLE_DEBUG_CAPS=ON`，而
`build/CMakeCache.txt:745` 是 `ENABLE_DEBUG_CAPS:BOOL=OFF`。
为做实验，临时把 `transformations_pipeline.cpp:537` 的 fallback 翻成 `false` 重建。

**融合确实匹配了**（`jit:gemm` 调用 140 → 56，正好少掉 28×3 个 MLP GEMM），
**但融合后的节点拿到的是 reference kernel**：

| exec_type | ms | calls |
| --- | ---: | ---: |
| `ocl:ref:any__f16` | **316.24** | 28 |
| `ocl::sdpa::opt__f16` | 26.09 | 28 |
| `jit:gemm:any__i8` | 24.20 | 56 |
| `generic_eltwise_ref__f16` | 15.12 | 127 |

wall p50 **396.96 ms** vs 264.8 ms —— **慢 1.5x**。

根因在 `src/graph/registry/gated_mlp_impls.cpp`：

```cpp
static const std::vector<std::shared_ptr<ImplementationManager>> impls = {
    OV_GPU_CREATE_INSTANCE_ONEDNN(onednn::GatedMLPImplementationManager, shape_types::static_shape)
};
```

**只注册了 `static_shape` 的 oneDNN impl**。GME 是动态 shape 编译的，
拿不到任何可用 impl，于是退化成 `ocl:ref:any` 参考实现。

⇒ 要让 GatedMLP 对 GME 有意义，得先给它补一个 dynamic-shape impl，
那是比 §6.4 大得多的工程量，收益还不确定。**已改动全部回退**，
`transformations_pipeline.cpp` 恢复到 stock。

---

## 6.6 第二轮结论与未解决项

### 已确立

1. GEMM 占 GPU 时间 **71.6%（算子级）/ 71.7%（VTune）**，两条独立口径互证。
2. GEMM **不是算力受限**：XMX active 仅 29–33%，occupancy 75–80%，
   stall 主因 SBID 39–47% → **依赖/延迟受限**，反量化在关键路径上。
3. 硬件 int8 通路健康：稠密 int8 36.5 T vs 稠密 f16 18.9 T = **1.93x（理论 2x）**。
   修正第一轮"int8 只比 f16 快 1.2x"的结论——那是被 INT4 反量化污染的。
4. **权重 layout `cab` → `abc` = 1.25x，且不改数值**，是唯一免费的大杠杆；
   三个前置风险（u4 可重排、scale 不耦合、动态 shape 有落点）**全部实测排除**。
5. grouped zero-point 只值 9%（否掉第一轮的源码推断）；
   scale/zp layout 对性能零影响；group 128→512 零收益。
6. GatedMLP 融合对 GME **负收益 1.5x**，根因是只注册了 static_shape impl。

### 未解决 / 待办

| 项 | 状态 |
| --- | --- |
| `cab` → `abc` 落地 | **已落地**，见 §6.9。实测收益与 §6.4 预估的 1.12x 吻合，但**适用范围比预估窄得多**（只有 gate/up，且只在 ≥128 行时）。**第三轮补测 down_proj（K/N=5.83 的「长条」方向）：probe 说 1.21x，模型内实测 down 42.09 → 53.58、wall 238.17 → 250.17，大幅负收益，`N≥4K` 保持原样**（docs/09 §9.1.3） |
| 模型内比单算子慢 1.3–1.5x | 五个假设已排除，**仍未解释**；`gate_proj` vs `up_proj` 同形状差 33% 是最好的线索。**第三轮把它锐化了**：probe 对 abc layout 的预测在 qkv（probe 1.00x / 模型 0.42x）、o_proj（1.34x / 0.86x）、down（1.21x / 0.79x）三个形状上**全部同向高估**，只有 gate/up 例外 ⇒ 结论应表述为「单算子 probe 系统性高估 N-contiguous 权重的收益，`N≥4K` 是实测拟合的经验边界而非 probe 推出的物理边界」（docs/09 §9.1.3）。qkv 的 1.95x 缺口已排除 bias（§9.1.4） |
| DQ 去重（gate/up 共享激活） | **已测量**：`qwen3.5_patch/opt3_dynamic_quantize_reuse.patch` 在 GME 上确实生效（DQ 140 → 112 calls、−2.1 ms），与「~1% wall」的估算一致，**收益太小，不重写**（docs/09 §9.4b） |
| per-OC 量化（1.55x） | **已评估并否决**：见 [`08_per_oc_quantization.md`](08_per_oc_quantization.md)。端到端确实能再叠加 ~1.1–1.28x（且与 opt-3 正交），但 text/query cosine 从 ~0.945 掉到 0.83–0.88，精度不达标 |
| GatedMLP dynamic-shape impl | 工程量大、收益未知，不在本轮。第三轮亦未做；注意 oneDNN 的 `gated_mlp/micro_horz` 要求 src 为 f16/bf16/f32，即 gate/up 要退回 f16 激活，须先定价 |
| `E_f16src_u4wei` | oneDNN 在所有形状都 unsupported，无路可走 |

---

## 6.7 第二轮踩过的坑（供复现时避开）

| 坑 | 症状 | 修正 |
| --- | --- | --- |
| **没 source `setupvars.sh`** | 测到的是官方 wheel。表现为"同一份源码漂移 6%"（266→283 ms）、"opt-1/opt-2 的收益消失"、"GatedMLP 实验无效果"、"第一次第二轮 VTune 采集其实是 stock" | 所有模型侧测量前必须 source；stock 结果重命名为 `*_STOCK*` 后重采 |
| benchdnn `--mode=P` 挂死 | 卡在 `stream_t::wait()` | 弃用，自写 probe（§6.2） |
| VTune 报 zombie thread | launch `python3` 时无法 attach | 激活 venv 后用 `python` |
| GPU 频率采样全 0 | 采样窗口落在 kernel 编译期 / GPU 活动结束后 | 后台带时间戳全程采样，再滤非零 |
| `D2_*_group512` 报 unsupported | 其实是 8960 % 512 ≠ 0，probe 自己的 bug | 加显式 guard，打印 `n/a (K%group)` |
| `round2_ops_int4_1024.json` 数字偏高 | wall 318 ms / GEMM 207.8 ms，同构建下与第一轮差 17% | 该次采集被干扰，作废；干净重测 `round2_ops_recheck.json`（263.7 / 177.2）与第一轮吻合 |

---

## 6.8 本轮产物

```
onednn/
  gemm_probe.cpp          INT4 GEMM 成本分解 probe（替代 benchdnn）
                          GEMM_PROBE_SHAPES=msweep 按行数扫 8960x1536（§6.9.4）
                          GEMM_PROBE_CASES=<substr> 只跑匹配的 case
  u4_reorder_probe.cpp    u4 cab->abc 重排可行性 + 位正确性 + 编译期代价
  scale_layout_probe.cpp  scale/zp 排布是否与权重 layout 耦合（对照 host 参考实现）
  layout_accuracy_probe.cpp  同一份逻辑权重物化成 cab / abc，各自与 f64 host
                          参考比 —— 证明换 layout 在 oneDNN 内部数值中性（§6.9.2）
  build_probe.sh          链接 oneDNN-ov2026.3 (babb7375ff)
  results/
    gemm_decompose_final.txt   §6.3 主表
    probe_300iter.txt          300 次迭代复现性
    probe_wtag.txt             layout 扫描
    probe_batched_m1.txt       1024x1x1536 dispatch 形状
    probe_sync.txt             PROBE_SYNC=1 逐次同步
    probe_layout_crossover.txt §6.9.4 行数扫描
    layout_accuracy.txt        §6.9.2 cab/abc 数值中性
results/
  round2_ops_recheck.json          §6.0 干净算子分布（权威）
  round2_patched_recheck.json      patched 基准 264.8 ms
  round2_reverify.json             stock 基准 283.1 ms
  round2_gatedmlp_ops.json         §6.5 GatedMLP 负结果
  r2_abc_{control,forced,heur,any}.json   §6.9.2–§6.9.3 descriptor 级实验
                                   （forced/heur 已作废，见 §6.9.2）
  r2_ncontig_impl.json             §6.9.2 第一版 N>=K 规则（268.6，否决）
  r2_ncontig_wide.json             §6.9.4 N>=4K、无行数门槛（240.5）
  r2_final_ops_1024.json           §6.9.6 最终构建（238.2，权威）
  acc_int4_round2.npz              §6.9.6 最终精度
  r2_regress_{stateful,gen}_patched.json  §6.9.6 回归
  ab/{stock,patched}_len{32,128,512,1024}_r{1,2,3}.json  §6.9.6 端到端 A/B
  vtune-r2-gpu-hotspots-patched/   §6.1 patched 采集
  vtune-r2-gpu-hotspots-STOCK/     §6.1 stock 复测
  vtune_r2_task_table.txt          §6.1 主表
```

每个结果文件的口径说明见 [`../results/README.md`](../results/README.md)。

---

## 6.9 opt-3 落地：宽 FC 的 N-contiguous 权重

§6.4 判定 `cab → abc` 可行且"免费"。落地过程推翻了其中两条推断：
**收益不是全局的**（只有 `N >> K` 的投影受益，其余显著变慢），
**也不是无条件的**（行数少于 ~128 时反而是负收益）。
最终规则因此比预估窄，但收益量级与预估一致。

### 6.9.1 选路：B（OCL reorder）

§6.4 给了两条路。选 **B**：

* 路径 A 要求目标 layout 走 `format::custom` 才会分派到 `reorder_onednn`
  （`primitive_inst.cpp` 里 `impl_type = (out format == custom) ? onednn : ocl`），
  等于把一个 oneDNN 专有 layout 引进 cldnn 的 layout 体系。
* 路径 B 只需要给 `reorder_weights_int4` 补上已有分支的镜像方向。
  `ioyx → oiyx` 那段本来就是走 `GET_FILTER_INDEX` 的通用 gather，
  镜像回来是 26 行 `.cl` + 5 行 kernel selector。

改动落在三个文件：

| 文件 | 改动 |
| --- | --- |
| `impls/onednn/fully_connected_onednn.cpp` | 判定函数 `use_n_contiguous_weights()` + `get_weights_reorder()`；命中时 weights md 改成 `{1,K,N}` + `abc` |
| `kernels/reorder/reorder_weights_int4.cpp` | 注册 `oiyx → ioyx`（`GetSupportedKey` / `SetDefault` / `Validate`） |
| `cl_kernels/reorder_weights_int4.cl` | `oiyx → ioyx` 分支 |

权重实际重排走 `primitive_inst::update_weights()`（§6.4-(3) 已探明的运行时槽位），
因此**动态 shape 下同样生效**，不需要碰 `post_optimize_weights` 的 guard。

判定函数和 md 构造**必须读同一批输入**：`get_weights_reorder()` 会改写权重
layout，如果两处判定不一致，matmul 就会用旧 stride 去读重排后的 buffer。
所以判定**刻意不看权重的 `format`**——那个字段正是被改写的对象。

### 6.9.2 第一版规则 `N >= K` 是错的

第一版按"§6.3 里 qkv/o_proj 也变快了"的实验数据，取 `N >= K`，
结果 wall p50 **268.60 ms，比不改还慢**：

| projection | control | 实验预测 | 实际 |
| --- | ---: | ---: | ---: |
| gate_proj | 63.00 | 51.92 | 49.94 ✅ |
| up_proj | 47.43 | 37.09 | 35.70 ✅ |
| qkv | 13.40 | 8.65 | **33.03** ❌ |
| o_proj | 11.55 | 9.30 | **14.59** ❌ |

gate/up 完全符合预期，qkv/o_proj 反向且幅度巨大。
descriptor 完全相同、`exec_type` 同为 `jit:gemm:any__i8`——
**先前那个"只改 descriptor 不做重排"的实验，在 qkv/o_proj 上的数字不可复现**。
（那种实验会让 kernel 用错误 stride 去读原 buffer，结果无意义，
但当时默认"至少计时是可信的"。这个默认是错的，已作废。）

同时精度也掉了（最差 cos 0.9948）。为定位是 oneDNN 换 kernel 还是自己接错线，
写了 `onednn/layout_accuracy_probe.cpp`：同一份逻辑权重分别物化成 `cab` 和 `abc`，
各自与 f64 host 参考比：

```
o_proj  K1536 N1536   cab 4.876e-4 / abc 4.876e-4
qkv     K1536 N2048   cab 4.876e-4 / abc 4.876e-4
gate/up K1536 N8960   cab 4.876e-4 / abc 4.876e-4
down    K8960 N1536   cab 4.871e-4 / abc 4.871e-4
```

**四位有效数字完全一致 ⇒ 换 layout 在 oneDNN 内部数值中性**，问题在自己这侧。
于是把重排后的 buffer 直接读回 host，和原常量的 CPU 转置逐 nibble 比：

```
layer0 o_proj 1536x1536:  mismatched nibbles 0 / 2359296
```

**重排是位正确的。** 两条排除之后剩下的唯一解释是 layout 选择本身选错了形状——
这也确实是真相（见下）。

### 6.9.3 用 `ONEDNN_VERBOSE` 逐 matmul 定价

不再依赖端到端聚合，直接读 oneDNN 自己报的每次 matmul 耗时
（1024 token，同一构建，规则开/关）：

| src : wei | 节点 | `cab` | `abc` | |
| --- | --- | ---: | ---: | --- |
| `1x1024x1536 : 1x1536x8960` | gate | 2.293 | **1.798** | 1.28x |
| `1024x1x1536 : 1x1536x8960` | up | 1.747 | **1.293** | 1.35x |
| `1024x1x1536 : 1x1536x2048` | qkv | **0.474** | 1.141 | 0.42x |
| `1x1024x1536 : 1x1536x1536` | o_proj | **0.419** | 0.486 | 0.86x |
| `1x1024x8960 : 1x8960x1536` | down | **1.513** | （未启用） | — |

收益只在 `N >> K` 的宽投影上。N/K 比：gate/up = 5.83（赢），
qkv = 1.33（大输），o_proj = 1.00（小输），down = 0.17（§6.3 已知无收益）。
**阈值取 `N >= 4K`**，落在实测最快的赢点和最慢的输点中间。

顺带否掉 `weights_fmt = any`：让 oneDNN 自己挑，它对所有形状都选了带 padding
的 N-contiguous stride，wall **251.4 ms**——比无脑全 `abc`（246.1）还差，
更远不如形状感知的规则。

### 6.9.4 还有一维：行数

`N >= 4K` 在 1024 token 上拿到 1.11x，但交替 A/B 暴露出 **32 token 掉到 0.837x**
（第一轮同一测试是 1.020x）。规则只看权重形状，不看激活行数。
同一形状扫行数（同构建，规则开/关，wall p50）：

| 行数 | 关 | 开 | |
| ---: | ---: | ---: | --- |
| 32 | 40.37 | 48.02 | 0.841x |
| 64 | 42.20 | 49.85 | 0.846x |
| 128 | 50.97 | **47.18** | 1.080x |
| 256 | 78.00 | **67.52** | 1.155x |
| 512 | 143.24 | **128.39** | 1.116x |
| 1024 | 264.78 | **237.88** | 1.113x |

拐点干净地落在 64 和 128 之间，**阈值取 128 行**。
行数取 `input_layout.count() / K`，因为同一个投影会以
`1 x M x K` 和 `M x 1 x K` 两种形式被分派（§6.3 已记录），两者都表示 M 行。

行数不达标时**连重排都不做**，直接用原常量 —— 不额外占显存。

### 6.9.5 最终规则

```
u4/i4 + 3D matmul + weights_transposed + 静态 shape
  且 N >= 4K            （只有宽投影受益）
  且 rows >= 128        （行数太少时反而更慢）
→ 权重重排成 N-innermost（cldnn ioyx），matmul weights md 改成 {1,K,N} + abc
```

GME 上命中的只有 `gate_proj` / `up_proj`，每层 2 个、共 56 个。

**代价**：每个命中的权重多一份副本，8960x1536 的 u4 = 6.56 MiB，
56 个 ≈ **367 MiB 常驻**，外加一次性重排（编译期外，首次执行时）。
这是本优化唯一的实质成本，如果部署侧显存紧张，应当把阈值调高或直接关掉。

### 6.9.6 结果

**算子分布（1024 token，`profile_ops.py`）**

| | wall p50 | GEMM 合计 | gate | up | down | qkv | o_proj |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| opt1+2 | 264.10 | 177.51 | 63.00 | 47.43 | 42.13 | 13.40 | 11.55 |
| opt1+2+3 | **238.21** | **151.10** | **49.55** | **35.24** | 41.96 | 12.89 | 11.45 |

wall **1.109x**，GEMM −26.41 ms，且**全部来自 gate/up**；
SDPA（26.1→25.9）、dynamic quantize（25.5→25.6）等其余算子不动。

**端到端交替 A/B（`bench_ab.sh`，stock wheel vs 本地 patched，3 轮取中位）**

| tokens | workload | stock p50 | patched p50 | delta | speedup |
| ---: | --- | ---: | ---: | ---: | ---: |
| 32 | text | 32.78 | 32.19 | −0.59 | 1.018x |
| 32 | query | 32.85 | 32.31 | −0.55 | 1.017x |
| 128 | text | 46.56 | 40.20 | −6.36 | **1.158x** |
| 128 | query | 46.24 | 40.17 | −6.08 | **1.151x** |
| 512 | text | 146.55 | 123.81 | −22.73 | **1.184x** |
| 512 | query | 145.49 | 123.31 | −22.18 | **1.180x** |
| 1024 | text | 281.38 | 240.01 | −41.38 | **1.172x** |
| 1024 | query | 280.18 | 239.53 | −40.66 | **1.170x** |

32 token 回到第一轮水平（1.018x vs 第一轮 1.020x），无回归。

**精度（`dump_outputs.py` + `compare_outputs.py`，vs stock wheel）**

| case | cosine | max_abs |
| --- | ---: | ---: |
| seq1 / seq7 / seq32 / seq129 / seq512 / seq1024 | 1.000000000 | 0 |
| batch2_padded | 0.999497041 | 1.192e-02 |

七个用例里六个**逐位相同**，只有 batch2_padded 有偏移。
量纲参照：同一套用例下 **INT4 相对 INT8 本身**的误差是
cos 0.918–0.989 / max_abs 最大 9.4e-2 —— 本改动的漂移比量化噪声底**低约一个数量级**。

`ONEDNN_VERBOSE=2` 确认精度跑里确有 408 次 `wei:u4::blocked:abc` matmul，
即该路径**确实被这组用例覆盖**，位相同不是"没走到"。

**KV-cache 通路（`regress_stateful.py`，Qwen3-1.7B-INT4）**

`generated_ids`、`first_step_logits_sha`、`first_step_logits_top8` 三项
与 stock **全部逐位相同**。

> 附带更正一条第一轮的表述：`regress_generative.py` 的贪心续写文本
> （Qwen3-1.7B-INT4）在 patched 下与 stock **并不相同**，这是**第一轮
> opt-1/opt-2 就已经存在**的差异（第一轮产物 `regress_gen_patched.json`
> 与本轮产物逐字相同），opt-3 没有引入任何新的偏移。
> 该模型 prompt 只有 ~20 token，本轮规则的 128 行门槛根本不会触发。
> 贪心解码在近似打平的 logits 上翻转即会整段发散，属预期内的数值敏感，
> 而非 KV-cache 回归——`regress_stateful.py` 的逐位相同才是该通路的结论。

### 6.9.7 交付

`patches/opt3_int4_wide_fc_ncontiguous_weights.patch`（3 files, +118 −0），
与 opt-1 / opt-2 一起在 `2026.3.0` tag 的纯净 worktree 上验证过
单独 apply、顺序 apply、合并 apply 均通过。

## 6.10 Qwen3-VL-Embedding-2B 交叉验证

opt-3 没有沿用 GME 的收益假设，已在 Qwen3-VL-Embedding-2B 上补做验证。Qwen3
language tower 的配置为 `hidden_size=2048`、`intermediate_size=6144`，因此宽投影
比例为 `N/K=3`，低于 opt-3 的 `N >= 4K` 选择条件。它不会重排 Qwen3 language
MLP 的 gate/up 权重，这与 1024-token text 的增量结果一致：

| precision | opt-1 + opt-2 | opt-1 + opt-2 + opt-3 | opt-3 增量 |
|---|---:|---:|---:|
| INT8 | 359.73 ms | 359.34 ms | 1.0011x |
| INT4 | 351.04 ms | 350.76 ms | 1.0008x |

这两行是同一 build 配置下的 warm 7-sample scope check，不替代原有三轮交替
stock-vs-opt1+opt2 A/B；完整输入、错误边界和结果说明见
[`results/qwen3_ab_text/REPORT.md`](../results/qwen3_ab_text/REPORT.md)。

Qwen3 vision tower 为 `hidden_size=1024`、`intermediate_size=4096`，宽度比例虽为
4，但 opt-1 + opt-2 的视觉路径在 224x224 起已经复现
`FullyConnectedCompressed` / `CL_OUT_OF_RESOURCES`。三项组合 build 的 64x64
调用仍可完成，224x224/448x448 仍失败，因此无法把视觉错误或视觉收益单独归因
于 opt-3。交叉验证结论是：**opt-3 对 Qwen3 text 没有实质作用；Qwen3 visual
路径在更早的 opt-1/opt-2 组合上已经不安全，三个 patch 不应直接用于生产。**
