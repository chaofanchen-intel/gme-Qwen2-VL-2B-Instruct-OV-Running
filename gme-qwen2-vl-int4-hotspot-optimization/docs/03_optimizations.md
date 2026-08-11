# 3. 代码优化

两个改动都在 **Intel GPU plugin**，都是**纯图结构 / 调度**改动，不碰任何算术，
因此输出 bit-identical（§4 验证）。patch 全部基于官方 tag `2026.3.0`
（`8a17657b99`），见 `patches/`。

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

## 合并效果（语言 IR，seq=1024，INT4/GPU）

| exec_type | baseline (ms) | patched (ms) | Δ |
| --- | ---: | ---: | ---: |
| `jit:gemm:any__i8` | 177.99 | 177.57 | −0.4 |
| `ocl::sdpa::opt__f16` | 28.03 | 26.05 | **−2.0** |
| `dynamic_quantize_gpu_opt__f16` | 26.55 | 25.24 | −1.3 |
| `broadcast_gpu_ref__f16` | **10.10** | **0** | **−10.1** |
| `rms_gpu_bfyx_opt__f16` | 9.14 | 9.42 | +0.3 |
| `permute_ref__f16` | **4.93** | **0.88** | **−4.1** |
| `ocl::rope::opt___f16` | 4.65 | 4.64 | 0 |
| 其它 | 2.41 | 2.40 | 0 |
| **profiled 合计** | **263.8** | **246.2** | **−17.6** |
| **wall** | **283.0** | **263.1** | **−19.9 (−7.0%)** |
