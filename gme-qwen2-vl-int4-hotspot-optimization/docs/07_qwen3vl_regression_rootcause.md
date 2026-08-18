# 07 — Qwen3-VL-Embedding-2B 上 `CL_OUT_OF_RESOURCES` 的根因与修复

Date: 2026-08-17

## 症状

装上三个 GME patch 后，`Qwen3-VL-Embedding-2B` 的 image / fused 路径报：

```text
Check 'shape_type == shape_types::dynamic_shape || node->selected_impl != nullptr' failed at
  src/plugins/intel_gpu/src/graph/graph_optimizer/compile_graph.cpp:54:
[GPU] Failed to select implementation for
  fullyconnectedcompressed:__module.blocks.0.attn.proj/ov_ext::linear/MatMul
  original_type: FullyConnectedCompressed could not set scales primitive attribute

[GPU] clFinish, error code: -5 CL_OUT_OF_RESOURCES
```

单独 apply **opt-2** 即可复现，opt-1 / opt-3 与此无关。复现脚本：
[`probe_qwen3_vision_merger.py`](../probe_qwen3_vision_merger.py)（只跑 vision merger 子模型，
把 tokenizer / language model / GenAI pipeline 排除在外）。

## 定位

`__module.blocks.0.attn.proj` 就是 vision tower 里紧跟 SDPA 的 o_proj。在
`fully_connected_onednn.cpp` 的 impl 创建处打点，得到 224x224（196 patch，对齐到 208 行）时：

```text
blocks.0.attn.qkv   in0 = i8:bfyx:208x1024      <- 正确
blocks.0.attn.proj  in0 = i8:bfyx:16x1024       <- 错，应该是 208x1024
```

`16` 是 head 数。也就是说 **SDPA 的输出 layout 的 batch 轴和 seq 轴被弄反了**，
后面 `Reshape` 的 pattern 是 `[0, 1024]` + `special_zero=true`（dim0 照抄输入 dim0），
于是行数从 208 变成 16，FC 拿到错误形状，oneDNN 建 primitive 失败；同时 SDPA kernel
按错误的 batch/heads 派发 work-group，越界写把 GPU context 打死，才有后面的
`CL_OUT_OF_RESOURCES`。

64x64 之所以“看起来正常”，是因为它的 patch 数恰好等于 head 数（都是 16），
两个轴弄反了也看不出形状差异——但数值仍然是错的。

## 根因：四个独立缺陷

Qwen3-VL 的 vision tower 是**没有 batch 轴的 rank-3 SDPA**
（`[num_heads=16, seq_len, head_size=64]`，见 `openvino_vision_embeddings_merger_model.xml`），
而且 Q/K/V 和输出四个方向上都带 `Transpose([1,0,2])`。GME 的 language tower 是常规 rank-4，
所以这几条在 GME 上全部没被触发。

### 缺陷 1 — 新 matcher 把已经融合好的输入 order 覆盖回默认值

`ov::intel_gpu::op::SDPA` **继承自** `ov::op::v13::ScaledDotProductAttention`
（`intel_gpu/op/sdpa.hpp:15`），而 `WrapType::match_value()` 用的是
`is_castable()`，所以 `wrap_type<ov::op::v13::ScaledDotProductAttention>`
**也会匹配上已经被改写过的 `intel_gpu::op::SDPA`**。

`TransposeFusion` 里两个 matcher 的实际执行顺序由拓扑序决定：SDPA 节点排在它的输出
Transpose 前面，所以

1. `TransposeSDPAMatcher`（root = SDPA）先跑，吃掉 Q/K/V 的 Transpose，
   产出 `op::SDPA`，`order_q/k/v = [1,0,2]`；
2. 轮到输出 Transpose 时，opt-2 的 `TransposeSDPATransposeMatcher` 匹配上
   `Transpose(op::SDPA)`，回调里无条件 `order_q = op::SDPA::default_order(rank)`，
   此时 pattern 里已经没有输入 Transpose 了，于是 `[1,0,2]` 被**默默还原成 `[0,1,2]`**，
   而输入仍然是那个已经去掉 Transpose 的张量。

结果就是 Q/K/V 被按 batch↔seq 互换的方式解释。运行时 dump 可以直接看到这个矛盾状态：

```text
sdpa:...Transpose_3  out=f16:bfyx:16x?x64  order_q=[0,1,2]  order_out=[1,0,2]
                     in0=f16:bfyx:?x16x64   <- 输入已经是转置前的形状，order_q 却是默认值
```

**修复**：回调里先 `ov::as_type_ptr<op::SDPA>(sdpa)`，命中就从既有节点上取
`order_q/k/v`，并把输出 order 与既有 `order_out` **复合**（`new[i] = prev[T[i]]`）
而不是覆盖。同时拒绝 `IndirectSDPA` 和 `kv_compressed` 的 SDPA——六参构造函数带不动
beam table / 量化属性，硬转会把有状态 KV-cache 路径搞坏。

### 缺陷 2 — micro SDPA 的 dispatch 函数没有做 shape canonicalization

`SDPAOptImpl` 建 stage 时传的是 canonicalize 过的（rank-4）params，但
`execute_stage` 交给 dispatch 函数的是 instance 的原始 params，rank-3 模型上它就是 rank 3。
`sdpa_gen_opt.cpp` 的三个 dispatch 函数都会自己再 canonicalize 一次，
`sdpa_gen_micro.cpp` 的**没有**（原来是 `auto params = impl_param;`）。

opt-2 在这里引入了 `micro_get_output_batch_and_num_heads()`，它按 rank-4 的 order 去索引
layout，落在没 canonicalize 的 rank-3 layout 上就读成了 `batch = out_ps[2] = 64`（head_size）、
`heads = out_ps[1] = 16`，正确值是 16 / 1 —— 直接 4 倍超发 work-group，越界写。

**修复**：dispatch lambda 里按 `sdpa_gen_opt` 的写法补上 canonicalization。
这一条同时修掉了 rank-4 动态 shape 下的一个崩溃。

### 缺陷 3 — 静态 shape 推导路径完全忽略 `output_transpose_order`

`scaled_dot_product_attention_inst::calc_output_layout()`（`use_new_shape_infer()` 为 false
时走的旧路径，纯静态图就是这条）只对输入 order 做了 `transpose_shape`，
**输出 order 一行都没应用**：

```cpp
auto q_shape = transpose_shape(input0_layout.get_partial_shape(), desc->input_q_transpose_order);
auto output_shape = q_shape;
output_shape[output_shape.size() - 1] = v_shape[v_shape.size() - 1];
return { layout{output_shape, ...} };      // <- 没有 output_transpose_order
```

新路径 `calc_output_layouts()` → `ov::intel_gpu::op::shape_infer()` 是应用了的。
stock 从来不会产生非默认的 `output_transpose_order`，所以这个不一致一直没人踩到；
opt-2 一旦融合就立刻暴露。表现是**静态 shape 的 rank-4 SDPA 直接 GPU 挂死**
（`probe_sdpa_transpose_variants.py --ranks 4 --variants out`，seq ≥ 32 必挂），
而同样的图在动态 shape 下正常——因为两条 shape 推导路径不一样。

**修复**：`calc_output_layout()` 末尾补上 `output_shape = transpose_shape(output_shape, desc->output_transpose_order);`。

### 缺陷 4 — 只有 micro kernel 认 `output_transpose_order`，`sdpa_opt` 不认

这是 opt-2 原始注释里“Every SDPA kernel already remaps the destination strides from that
order”的说法不成立的地方。输入侧有 `INPUT{0,1,2}_DIMS_ORDER`
（`sdpa_base.cpp` 里 emit，`sdpa_opt.cl` 里 `FUNC(get_input0_index)` 消费），
**输出侧没有对应的 `OUTPUT_DIMS_ORDER`**：`sdpa_opt.cl` 和 `sdpa_ref.cl` 一律用
`OUTPUT_GET_INDEX(b0, b1, seq, head_size)` 这个默认 BHLS 次序写回，
finalization stage 甚至是手写的默认 layout 偏移算术。也就是说对这两个 kernel，
非默认的输出 order 会被**静默忽略**。

而 `SDPAOpt::supports_micro_sdpa()` 里有一条明写的限制：

```cpp
// Known limitation: In vision encoding model of qwen-vl, when the shape of sdpa is 3D and
// num_heads is 1, there is an accuracy issue with sdpa_micro kernel. Therefore, it is
// currently restricted to execute with sdpa_opt kernel.
const bool is_output_rank_4d = desc->output_transpose_order.size() == 4;
if (!is_output_rank_4d) return false;
```

rank-3 SDPA 被**强制**走 `sdpa_opt` —— 于是 Qwen-VL / Qwen3-VL 的 vision tower
在任何情况下都不可能正确执行融合后的输出 order。这是结构性的，不是调参能绕开的。

**修复**：把融合限制在“确定会跑 micro kernel”的 SDPA 上，即要求
`supports_immad` 且输出 rank 为 4。rank-3 直接不融合，退回 stock 行为。

## 修复后的验证

插件 = OpenVINO 2026.3.0 + opt-1 + opt-2（含本次四项修复）+ opt-3，Intel PTL-H iGPU（Xe3）。

| 检查 | 脚本 | 结果 |
|---|---|---|
| 合成 SDPA 变体，rank 3/4 × {无 transpose, 仅输入, 仅输出, 输入+输出} × 静态/动态，GPU vs CPU | [`probe_sdpa_transpose_variants.py`](../probe_sdpa_transpose_variants.py) | 16/16 通过，worst cos = 0.999997（修复前 rank-3 cos≈0.01、rank-4 静态崩溃） |
| Qwen3-VL vision merger，64x64 / 224x224 / 448x448 / 896x896，INT8 + INT4 | [`probe_qwen3_vision_merger.py`](../probe_qwen3_vision_merger.py) | 全部 OK（修复前 224x224 起崩溃） |
| 同上，逐元素对比 stock GPU 输出 | [`check_qwen3_vision_merger_accuracy.py`](../check_qwen3_vision_merger_accuracy.py) | `max_abs_diff = 0`，**与 stock 逐位相同** |
| Qwen3-VL 端到端 embedding，text 32/128/512/1024 + image/fused 五种尺寸 × INT8/INT4，共 28 个 workload | [`smoke_qwen3_vl_embedding.py`](../smoke_qwen3_vl_embedding.py) | 28/28 通过，且与 stock **逐位相同** |
| 融合是否仍在该生效的地方生效（runtime graph 里 permute 计数） | [`check_sdpa_output_transpose_fused.py`](../check_sdpa_output_transpose_fused.py) | 见下表 |
| GME 交替 A/B 收益是否保住 | `bench_ab.sh`（ROUNDS=3, LENGTHS="1024 512 128"） | 见下表 |

### 融合作用范围（runtime graph 里的 permute/transpose 数量）

| 模型 | stock | 修复后 patched | 说明 |
|---|---:|---:|---|
| GME language INT4（rank-4） | 57 | **29** | 28 个 SDPA 输出 transpose 被融合掉，opt-2 照常生效 |
| GME vision INT4 | 194 | 194 | 走 VLSDPA，本来就不在范围内 |
| Qwen3-VL language INT8（rank-4） | 56 | **28** | 也拿到融合，且数值逐位相同 |
| Qwen3-VL vision merger INT8（rank-3） | 48 | **48** | **正确地不融合**，退回 stock |

### GME 收益（交替 A/B，3 轮取中位）

| tokens | text | query |
| ---: | ---: | ---: |
| 128 | 46.63 → 40.07 ms (**1.164x**) | 46.48 → 40.19 ms (**1.156x**) |
| 512 | 147.01 → 123.98 ms (**1.186x**) | 146.06 → 123.57 ms (**1.182x**) |
| 1024 | 280.83 → 239.93 ms (**1.170x**) | 280.13 → 240.13 ms (**1.167x**) |

与 `patches/README.md` 里修复前记录的 1.158x / 1.184x / 1.172x 在噪声范围内一致，
**收益完全保住**。

## 残留风险

融合的前置条件是“这个 SDPA 会跑 micro kernel”，matcher 只能在图变换阶段用
`supports_immad` + 输出 rank 判断，下面几种情况仍会落到 `sdpa_opt` 上而 order 被忽略：

- IGC 太老导致 `query_microkernels_supported()` 为 false；
- `xe3p` 且 `k_head_size <= 64`（`supports_micro_sdpa()` 里的 WA）；
- ARL-H 的 single-token 阶段（`gfx_ver == 12.74` 时 micro 被关掉）；
- micro kernel 因 "Insufficient registers in requested bundle" 建不出来，
  `SDPAOptImpl` 回退到 `regular_multi_tokens`。

前三条是设备属性，可以通过把判据从 `supports_immad` 收紧到完整的
`supports_micro_sdpa()` 前置条件来关掉（需要把 engine/config 传进 `TransposeFusion`）；
第四条是运行期才知道的，图变换阶段无法预判。

要彻底消除这类风险，正解是给 `sdpa_opt.cl` / `sdpa_ref.cl` 补上 `OUTPUT_DIMS_ORDER`
（对称于已有的 `INPUT{0,1,2}_DIMS_ORDER`），涉及 `sdpa_opt.cl` 的两处写回 + `output_pitch`、
finalization stage 的手写偏移算术，以及 `sdpa_ref.cl` 的一处。做完之后不但可以去掉上面的
限制，Qwen-VL 系列的 rank-3 vision tower 也能一并吃到这个融合的收益。这属于新增 kernel
能力，本次没有做。

## 对上游的补充建议

`patches/README.md` 里原有的三条建议之外，缺陷 1、3 是 stock 就存在的通用问题，
可以独立提 PR：

- `WrapType` 匹配派生类型导致 SDPA matcher 互相覆盖，是 `TransposeFusion` 里所有
  SDPA matcher 共有的隐患，不止 opt-2 引入的这一个；
- `calc_output_layout()` 与 `calc_output_layouts()` 对 `output_transpose_order` 的处理不一致，
  是明确的 bug，只是 stock 目前构造不出触发条件；
- `sdpa_gen_micro.cpp` 的 dispatch 函数缺 canonicalization，对 stock 也是有影响的：
  rank-3 SDPA 上 `out_ps[0] * out_ps[1]` 会算出 `16 * seq_len` 而不是 `16`，
  造成 `wgs.global[1]` 超发 seq_len 倍（结果仍然正确，属于纯浪费）。
