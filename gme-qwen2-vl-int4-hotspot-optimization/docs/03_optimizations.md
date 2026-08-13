# 3. 代码优化

三个改动都在 **Intel GPU plugin**。opt-1 / opt-2 是**纯图结构 / 调度**改动，
opt-3 只改**权重的物理排布**，三者都不碰任何算术（§4 验证）。patch 全部基于官方
tag `2026.3.0`（`8a17657b99`），见 `patches/`。

opt-1 / opt-2 来自第一轮，opt-3 来自第二轮 oneDNN 立项；第二轮完整的取舍过程、
被否掉的中间方案和逐 matmul 计时在 `06_round2_onednn_investigation.md` §6.9，
本节只记最终形态。

---

## opt-1：让 GQA `repeat_kv` 的 Broadcast 融合支持无状态模型

**文件**：`src/plugins/intel_gpu/src/plugin/transformations/unsqueeze_broadcast_reshape_sdpa_fusion.cpp`
**patch**：`patches/opt1_sdpa_gqa_broadcast_stateless.patch`（+13 / −5）

### 问题

`UnsqueezeBroadcastReshapeSDPAFusion` 的 pattern 要求 `Unsqueeze` 的输入是
`ov::intel_gpu::op::KVCache`：

```cpp
auto input_b_kvcache_m = wrap_type<ov::intel_gpu::op::KVCache>({any_input(), any_input()});
auto input_c_kvcache_m = wrap_type<ov::intel_gpu::op::KVCache>({any_input(), any_input()});
```

`KVCache` 只在**有状态生成式模型**里出现。GME 这类 embedding / encoder 前向，
以及任何不带 KV cache 编译的模型，K 直接来自 RoPE、V 直接来自 Transpose，
pattern 不匹配，`Unsqueeze -> Broadcast -> Reshape` 整条链留在 runtime 图里。

### 改动

把这两个 producer 放开成 `any_input()`：

```cpp
// The producer of the GQA repeat_kv expansion is a KVCache only in stateful generative
// models. A stateless graph -- an embedding/encoder forward pass, or any model compiled
// without a KV cache -- feeds the Unsqueeze straight from RoPE / Transpose / FullyConnected,
// ...
auto input_b_kv_producer_m = any_input();
auto input_c_kv_producer_m = any_input();
```

### 为什么安全

producer 的**类型本身不携带任何融合所需的信息**，真正的前提条件全部在 callback
里已经检查过，且一条都没有放松：

| 已有检查 | 作用 |
| --- | --- |
| `unsqueeze_predicate = rank_equals(5) && consumers_count(1)` | 必须是 5D 扩展、且独占消费 |
| `broadcast_predicate` 里 `BroadcastType::BIDIRECTIONAL` | 必须是双向 broadcast |
| `valid_broadcast_target_shape(...) diff_cnt == 1` | 只能在**恰好一个**轴上放大 |
| `target_shape_val_b != target_shape_val_c -> return false` | K 和 V 必须同样的放大规则 |
| `ensure_4d(...)` 找不到 broadcast 轴就 `return std::nullopt` | 拿不到明确的 broadcast 轴就不融合 |
| `reshape_predicate = rank_equals(4)` | 末端必须回到 4D |

融合之后 K/V 以 `num_kv_heads` 送进 SDPA，由 SDPA kernel 内部做 head 广播 ——
这正是这条 pass 本来的设计意图，KV-cache 类 GQA 模型（Qwen/Llama 全系）天天在走
这条路，下游实现是成熟的。

### 效果

- `broadcast_gpu_ref` 从 **10.10 ms → 0**（该 kernel 从热点表里消失）
- SDPA 读的 K/V 小了 6×，自身从 28.03 → 26.14 ms
- 1024 token：283.0 → 269.9 ms（**−13.1 ms**，单独测量时 280.2 → 269.9 = −10.3 ms）

---

## opt-2：把 SDPA 输出的 Transpose 融进 `output_transpose_order`

**文件**：
- `src/plugins/intel_gpu/src/plugin/transformations/transpose_fusion.{cpp,hpp}`（新增 matcher）
- `src/plugins/intel_gpu/src/graph/impls/ocl_v2/sdpa/sdpa_gen_micro.cpp`（修复配套 bug）

**patch**：`patches/opt2_sdpa_output_transpose_fusion.patch`（+165 / −14）

### 2a. 新增 `TransposeSDPATransposeMatcher`

`op::SDPA` 早就有 `output_transpose_order`，OCL SDPA 生成器也确实按它重映射目标
stride：

```cpp
jit.add(convert_strides("DST", "OUTPUT", extended_output_transpose_order));
```

但 `TransposeFusion` 里只有融合**输入** transpose 的 `TransposeSDPAMatcher`，
没有输出侧的 —— 而 MatMul 是两侧都有的
（`TransposeMatMulMatcher` + `TransposeMatMulTransposeMatcher`）。

新 matcher 完全对齐 MatMul 的写法：以输出 `Transpose` 为 match root，把
`Transpose` 的 order 直接写进新建 `op::SDPA` 的 `order_output`，并在
`TransposeFusion` 构造函数里注册在 `TransposeSDPAMatcher` **之前**（更具体的
pattern 优先）：

```cpp
add_matcher<TransposeMatMulTransposeMatcher>(supports_immad);
add_matcher<TransposeMatMulMatcher>(supports_immad);
add_matcher<TransposeSDPATransposeMatcher>();   // <- 新增
add_matcher<TransposeSDPAMatcher>();
```

保守条件：

- SDPA 必须 `consumers_count(1)`（否则拆掉 Transpose 会改变别的消费者看到的布局）；
- `order_output.size()` 必须等于 SDPA 输出 rank；
- `order_output.back() == order_output.size() - 1`，即 **head_size 必须仍是最内维**
  —— kernel 是按 head_size 连续的行去索引目标的，只有保持最内维才能纯靠 stride 表达。

### 2b. 配套修复：`sdpa_gen_micro` 没有按 order 解读输出维度

直接加完 2a 后 GPU 立刻 `clWaitForEvents` 失败。原因是 micro SDPA 生成器
**按位置**读输出 layout 的前两维：

```cpp
const auto batch = out_ps[0] * out_ps[1];          // 假设 [B, H, S, D]
...
wgs.global[1] *= out_ps[1].get_length();           // 当成 num_heads
wgs.global[2] *= out_ps[0].get_length();           // 当成 batch
```

输出 layout 里已经 apply 过 `output_transpose_order`，所以在 order = `(0,2,1,3)`
时 `out_ps[1]` 是 **seq_len(1024)** 而不是 num_heads(12) → 全局 work size 被放大
85 倍 → GPU 挂死。

也就是说：**SDPA 声明支持 `output_transpose_order`，micro 实现只对 stride 生效，
对 dispatch 几何是坏的。** 这是 2026.3 里一个此前无人触发的潜在 bug
（因为此前没有任何 pass 会给 SDPA 设非默认的输出 order）。

修法是走和输入侧完全一样的 order-aware 路径（`sdpa_utils.hpp` 里已有
`get_batch_size(layout, order)` / `get_num_heads(layout, order)`，`sdpa_gen_opt`
早就这么用了）：

```cpp
// [batch, num_heads] of the SDPA result in logical (untransposed) order.
inline std::pair<int64_t, int64_t> micro_get_output_batch_and_num_heads(const kernel_impl_params& params) {
    const auto& out = params.output_layouts[0];
    if (params.is_type<scaled_dot_product_attention>()) {
        const auto desc = params.typed_desc<scaled_dot_product_attention>();
        const auto order = extend_order_in_num_heads_dim(desc->output_transpose_order);
        return {get_batch_size(out, order), get_num_heads(out, order)};
    }
    const auto& out_ps = out.get_partial_shape();
    return {out_ps[0].get_length(), out_ps[1].get_length()};
}
```

替换 3 处使用点（jit 常量里的 `batch`、`get_dispatch_data_func()` 里的
`wgs.global[1]/[2]`、以及 micro-kernel 配置选择里的 `batch`）。`paged_attention`
分支保持原样。

### 效果

- `permute_ref` 从 **4.93 ms（56 次）→ 0.88 ms（28 次）**
  （`Transpose_3` 被吃掉；剩下的 `Transpose_2` 因为 opt-1 拆掉了中间的
  Unsqueeze/Broadcast/Reshape，代价也从 2.27 降到 0.88）
- 1024 token：269.9 → 263.1 ms（**−6.8 ms**）

---

## opt-3：宽 INT4 投影的权重重排成 N 连续

**文件**：
- `src/plugins/intel_gpu/src/graph/impls/onednn/fully_connected_onednn.cpp`（决策 + 描述符 + 挂 reorder）
- `src/plugins/intel_gpu/src/kernel_selector/kernels/reorder/reorder_weights_int4.cpp`（放开 `oiyx → ioyx`）
- `src/plugins/intel_gpu/src/kernel_selector/cl_kernels/reorder_weights_int4.cl`（新增该转换的 kernel 分支）

**patch**：`patches/opt3_int4_wide_fc_ncontiguous_weights.patch`（+118 / −0，纯新增）

### 问题

NNCF group-128 压缩出来的 INT4 权重是一个行主序 `[N, K]` 常量。plugin 把它交给
oneDNN matmul 时，按 3D 权重维 `(1, K, N)` 描述成 **K 连续**（`acb` / `cab`）。
`jit:gemm:any__i8` 在这种排布下对 **N 远大于 K** 的形状明显吃亏 —— GME 的
MLP `gate_proj` / `up_proj` 正好是 `N=8960, K=1536`（N/K = 5.83），两者合计
占 seq=1024 时 GEMM 时间的 62%（110.4 / 177.5 ms）。

第二轮用 `ONEDNN_VERBOSE=2` 做了逐 matmul 定价（同一次 1024 token prefill，
`cab → abc`）：

| 权重 `[K, N]` | N/K | 计时变化 |
| --- | ---: | --- |
| 1536 × 8960（gate / up） | 5.83 | 2.293 → **1.798** ms；1.747 → **1.293** ms |
| 1536 × 2048（qkv） | 1.33 | 0.474 → 1.141 ms ❌ |
| 1536 × 1536（o_proj） | 1.00 | 0.419 → 0.486 ms ❌ |

即：**这个 layout 不是普遍更好的，只在 N ≫ K 时更好**，在方阵形状上是明确的负收益。
第二轮的第一版规则用的是 `N ≥ K`，因此把 qkv/o_proj 也卷了进来，端到端反而变慢 ——
详见 §6.9.3。

### 改动

三件事：

1. `use_n_contiguous_weights()`：在 `fully_connected_onednn` 里判断该权重是否值得换排布；
2. 命中时 `get_matmul_primitive_descriptor()` 把权重描述符从 `acb`/`cab` 换成
   `abc`（dense N 连续）；
3. 同样命中时 `create()` 挂一个 `WeightsReorderParams(oiyx → ioyx)`，由
   `primitive_inst::update_weights()` 在首次执行时把常量转置一次并缓存。

规则本身：

```cpp
if (prim_input_size != 3 || !weights_transposed)   return false;   // 只管这条 matmul 路径
if (dtype != i4 && dtype != u4)                    return false;   // 只管 4-bit 压缩权重
if (weights_layout.data_padding)                   return false;
if (pshape.size() != 2 || pshape.is_dynamic())     return false;
if (N < 4 * K)                                     return false;   // 形状门槛
if (input_layout.is_dynamic())                     return false;
return input_layout.count() / K >= 128;                            // 行数门槛
```

两个常数的来源：

- **`N ≥ 4K`** —— 卡在上表「最慢的正收益（5.83）」和「最快的负收益（1.33）」之间的空档里。
- **`≥ 128 行`** —— 端到端行数扫描（同模型，规则开/关的 wall 比值）：

  | 行数 | 32 | 64 | 128 | 256 | 1024 |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | 加速比 | 0.84x | 0.85x | **1.08x** | **1.16x** | **1.11x** |

  交叉点干净地落在 64 和 128 之间。行数取 `count() / K` 而不是某个固定维，是因为
  同一个投影在图里会以 `1 × M × K` 和 `M × 1 × K` **两种 src 形态**下发（gate /
  o_proj / down 是前者，up / qkv 是后者），两者都表示 M 行。

有一点必须强调：**决策刻意不看权重当前的 format**。`get_weights_reorder()` 正是要
去改这个 format，而 `create()` 之前和之后建的 primitive descriptor 必须一致 ——
否则 matmul 会拿旧 stride 去读重排后的 buffer，读出的是垃圾（这是第二轮排查中一个
真实踩到的坑，见 §6.9.2）。

### 为什么安全

- 走的是 plugin 已有的 `primitive_inst::update_weights()` + `_reordered_weights_cache`
  机制 —— 这是为动态 shape 设计的、在别的 impl 上已经在用的运行时权重重排槽位，
  不新增生命周期管理。
- `oiyx → ioyx` 的 INT4 reorder kernel 用 host readback 逐 nibble 比对过 CPU 转置：
  **0 / 2359296 个 nibble 不一致**。
- scale / zero-point 不受影响：mask 7 的 group 量化参数是 **g-outer** 排列，与权重
  的物理 layout 无关。
- 不命中的形状（含所有 INT8 / FP16 权重、所有窄投影、所有行数不足的调用）走的
  代码路径与 stock 完全一致，**一行也不变**，也不会多分配任何显存。

### 代价

命中的权重会多留一份副本。GME 上是 56 个 `8960 × 1536` u4 权重 × 6.56 MiB
≈ **367 MiB** 常驻显存，外加首次执行时的一次性重排。行数门槛不达标时不重排、
不分配。

### 效果

- `jit:gemm:any__i8` **177.51 → 151.10 ms（−26.4）**，且**全部**来自
  `gate_proj`（63.00 → 49.55）和 `up_proj`（47.43 → 35.24），其余线性层不动；
- 1024 token wall：264.10 → **238.21 ms**（**1.109x**）。

---

## 合并效果（语言 IR，seq=1024，INT4/GPU）

| exec_type | baseline (ms) | opt1+2 (ms) | opt1+2+3 (ms) | Δ 总计 |
| --- | ---: | ---: | ---: | ---: |
| `jit:gemm:any__i8` | 177.99 | 177.51 | **151.10** | **−26.9** |
| `ocl::sdpa::opt__f16` | 28.03 | 26.05 | 25.90 | **−2.1** |
| `dynamic_quantize_gpu_opt__f16` | 26.55 | 25.24 | 25.57 | −1.0 |
| `broadcast_gpu_ref__f16` | **10.10** | **0** | **0** | **−10.1** |
| `rms_gpu_bfyx_opt__f16` | 9.14 | 9.42 | 9.38 | +0.2 |
| `permute_ref__f16` | **4.93** | **0.88** | **0.88** | **−4.1** |
| `ocl::rope::opt___f16` | 4.65 | 4.64 | 4.66 | 0 |
| 其它 | 2.41 | 2.40 | 2.40 | 0 |
| **profiled 合计** | **263.8** | **246.2** | **219.9** | **−43.9** |
| **wall p50** | **283.0** | **264.1** | **238.2** | **−44.8 (−15.8%)** |

> 这里的 wall 是 `profile_ops.py` 在**开着 per-op profiling** 时的语言 IR 时间，
> 比端到端 benchmark 略高（profiling 本身有开销）。端到端交替 A/B 的数字见
> §4。
