# 9. 第三轮：以 gs128/gs64 为主的进一步降延迟

> 本文是第三轮 language-only 优化历史报告；其中“当前”均指 2026-08-21 的第三轮
> 状态。当前部署 precision、baseline 对比和 fresh matrix 见
> [`docs/16`](16_round9_final_rebenchmark.md) 和
> [`results/r9_latest/`](../results/r9_latest/)。

第二轮把 1024 token 从 283.0 ms 降到 238.2 ms（opt-1/2/3），剩下的时间构成里 GEMM 占
63%，而 GEMM 的头部空间在 oneDNN 内部。第三轮的任务是：**在不牺牲精度的前提下继续降
latency，并且以 gs128 / gs64 为主**（`int4_per_oc` 已在 docs/08 因精度否决，只作最后备选）。

本轮最大的一笔收益不在 GPU plugin 里，而在**导出侧**：把 SDPA 从「`causal=false` + 显式
`[B,1,S,S]` mask」改成「`causal=true` 无 mask」，1024 token 上 SDPA 从 **25.99 → 11.62 ms
（2.24x）**，端到端 **238.48 → 222.33 ms（1.073x）**，且经 FP32 逐位比对证明是**恒等变换**。

同时本轮否掉了四个候选，每一个都有实测数字（§9.1.2、§9.1.3、§9.1.4、§9.5）。

---

## 9.0 起点：复位、复现，以及 gs64 的首次算子级基线

### 9.0.1 测量口径

开工时 `benchmark/openvino` 工作区 applied 着 qwen3.5 的 opt3/4/5/6/7（未提交）。三个本地
commit（`c2638477a6`/`32f59cd819`/`09c4cc4117`）**只加 patch 文件和报告、不碰 `src/`**，所以
`git checkout -- src/` 即可回到 `2026.3.0` tag。原工作区已备份到
`preexisting-worktree-backup/openvino-worktree-before-round3.diff`（opt6 的两个 untracked
文件另存 `round3-untracked/`）。

复位后 apply `patches/all_gme_int4_gpu_opts.patch`（7 files, +385 −19）并重编。

> ⚠️ 复现时踩到两个坑，都会让结论作废：
> 1. **CMake 缓存了已删除的源文件**。opt6 的 `move_swish_up_through_reshape.cpp` 被删掉后
>    直接 `cmake --build` 会编译失败，而紧接着的 `cmake --install` 仍会成功并装上**旧的
>    `.so`**。必须先 `cmake -S . -B build` 重新 configure。
> 2. **只重编 `openvino_intel_gpu_plugin` 会让 install 树版本错配**（Python 绑定与 runtime
>    的 git-describe 版本串不一致，`import openvino` 直接抛 `versions_compatible` 失败）。
>    本轮改为整树 `cmake --build build -j16` 后再 install。

构建版本：`2026.3.0-22454-09c4cc4117e-qwen-ttft-optimizations`。

**这个构建与第二轮交付构建逐位相同**——`compare_outputs.py results/acc_int4_round2.npz
results/r3_acc_int4_patched.npz` 七个用例全部 `cosine 1.000000000 / max_abs 0`。所以第三轮
的所有对比都建立在第二轮已发表的同一基线上。

### 9.0.2 gs128 基线复现

| | 本轮实测 | docs/06 §6.9.6 已发表 |
| --- | ---: | ---: |
| wall p50 | 239.68 | 238.21 |
| GEMM 合计 | 151.90 | 151.10 |
| gate_proj | 49.85 | 49.55 |
| up_proj | 35.47 | 35.24 |
| down_proj | 42.23 | 41.96 |
| qkv | 13.03 | 12.89 |
| o_proj | 11.32 | 11.45 |
| SDPA | 25.92 | 25.90 |
| dynamic_quantize | 26.83 | 25.57 |
| host（wall − profiled） | 18.03 | 18.3 |

复现，误差在这台机器的正常波动内。

### 9.0.3 gs64：本项目第一次拿到算子级基线

docs/04 §4.1 只说 gs64「性能代价可忽略」，从未给过算子分布。本轮补上，并且顺带发现
**这句话只在 stock 上成立**：

| | gs128 patched | gs64 patched | gs64 stock |
| --- | ---: | ---: | ---: |
| wall p50 | **239.68** | **254.19** | 284.76 |
| GEMM 合计 | 151.90 | 166.09 | 181.37 |
| gate_proj | 49.85 | 55.57 | 61.76 |
| up_proj | 35.47 | 40.68 | 49.05 |
| down_proj | 42.23 | 44.04 | 43.98 |
| qkv | 13.03 | 14.91 | 15.21 |
| o_proj | 11.32 | 10.90 | 11.37 |
| broadcast_gpu_ref | 0 | **0** | 10.09 |
| permute_ref | 0.88 | **0.88** | 4.92 |

三个结论：

1. **opt-1 / opt-2 在 gs64 上照常生效**（broadcast 归零、permute 0.88），量化分组不影响
   这两个图结构优化。
2. **gs64 在 stock 上确实几乎免费**（284.76 vs gs128 stock 283.0，+1.8 ms），但在 patched
   上要贵 **14.5 ms**。原因单一：**opt-3 在 gs64 上只兑现了 −14.6 ms，而在 gs128 上是
   −25.1 ms**（gate −6.19 / up −8.37 vs gate −13.15 / up −11.96）。
3. 也就是说 **opt-3 的 N-contiguous 权重重排和 gs64 更细的 scale 分组互相抵消**。这个现象
   在 §9.1 的 probe 里得到独立复现（`Q_gs64_abc` 1.251 ms vs `A_abc` 1.033 ms），不是
   模型侧的偶然。

> 对使用者的直接含义：**gs64 换来的 text cosine +0.0054 要付 14.5 ms（约 6%）**，
> 而不是 docs/04 暗示的「可忽略」。这个取舍现在有数字了。

---

## 9.1 oneDNN probe round-3：先定价，再决定动不动手

复用 `onednn/gemm_probe.cpp`（链接 `oneDNN-ov2026.3` babb7375ff，与 OV 2026.3 vendored 版本
一致），新增 `src_pr`（precomputed reductions）和 `bias` 两个开关、gs64 / 分组对称 / bias 共
9 个 case，并加了 `GEMM_PROBE_SHAPES=gme` 只跑五个真实投影（case 表已有 25 条，12 列全跑
太慢）。

先复核基线：`A_gme_cab` gate **1.288 ms / 21.9 T**，docs/06 §6.3 发表值 1.291 / 21.8 —— probe
本身可复现。

完整表 `onednn/results/r3_pricing.txt`。以 `A_gme_cab` 为 1.00x：

| case | gate | up | down | qkv | o_proj |
| --- | ---: | ---: | ---: | ---: | ---: |
| `A_gme_cab`（模型现状） | 1.00x | 1.00x | 1.00x | 1.00x | 1.00x |
| `A_abc`（opt-3 的 layout） | **1.25x** | **1.26x** | **1.21x** | 1.00x | 1.34x |
| `B_cab_symmetric`（gs128 对称） | 1.09x | 1.10x | 1.09x | **1.20x** | 1.08x |
| `B_abc_symmetric`（gs128 对称 + abc） | **1.28x** | **1.30x** | 1.25x | 1.07x | 1.35x |
| `C_s8wei_grouped`（给 u4 unpack 定价） | 1.33x | 1.35x | 1.30x | 1.11x | 1.45x |
| `G_s8_dense`（int8 DPAS 天花板） | 1.68x | 1.69x | 1.68x | 1.29x | 1.67x |
| `P_prB_cab`（precomputed reductions） | **0.001x** | 0.001x | 0.001x | 0.001x | 0.001x |
| `Q_gs64_cab` | 0.94x | 0.95x | 0.93x | 0.93x | 1.02x |
| `Q_gs64_abc` | 1.03x | 1.04x | 1.00x | 0.88x | 1.15x |
| `R_sym_gs64_cab` | **1.11x** | 1.11x | 1.10x | 1.06x | 1.11x |
| `R_sym_gs64_abc` | 1.04x | 1.04x | 1.03x | 0.92x | 1.18x |
| `U_cab_bias`（qkv 的 bias） | 0.99x | 1.00x | 0.99x | 0.94x | 0.99x |

### 9.1.1 读法

1. **对称量化是 gs128 之上唯一还剩的 GEMM 杠杆**：`B_abc_symmetric` 1.28x 是全表除
   `C`/`G`（都要改权重位宽）之外最快的一格，且相对 opt-3 已有的 `A_abc` 还有 1.028x；
   窄投影（仍是 cab）能吃到 1.08–1.20x。落地见 §9.4。
2. **gs64 的代价被独立确认**：`Q_gs64_cab` 0.93–0.95x（比 gs128 慢），而 `Q_gs64_abc`
   相对 `A_abc` 是 1.033/1.251 = **0.83x**——gs64 把 abc 的好处吃掉了六分之五。
   与 §9.0.3 模型内实测方向一致。
3. **gs64 的最优组合是「对称 + cab」而不是「非对称 + abc」**（`R_sym_gs64_cab` 1.11x >
   `Q_gs64_abc` 1.03x），也就是说**如果采用 gs64-sym，opt-3 的 `N≥4K` 规则对它应该关掉**。
   这是本轮发现的一个规则级结论。

### 9.1.2 否决：precomputed reductions（1000x 更慢）

思路本来很干净：grouped weight zero-point 迫使 GEMM 在 kernel 内对 src 按 K-group 求和
（`oneDNN doc/programming_model/attributes_quantization.md` 的 Precomputed Reductions 一节给
了公式），而 `dynamic_quantize` kernel 本来就把整张激活读了一遍，顺手多输出一份 s32 部分和
几乎不加访存。`src/gpu/intel/gemm/jit/pd.cpp:369` 也确实有 `has_prB` 分支。

**实测 1484 ms（对照 1.288 ms）。** `ONEDNN_VERBOSE=1` 给出根因：

```
matmul,ocl:ref:any,...,attr-precomputed-reductions:src0:7:s32:1x128,...,1467.31
```

primitive 建得起来，但 **`jit:gemm` 拒绝这个属性组合，整个 matmul 掉到 `ocl:ref:any`
参考实现**。也就是这个属性在本 oneDNN 版本上对我们的配置是「API 支持、实现不支持」。
`Q_gs64_prB_cab` 同样掉 ref。**否决，不再投入**；要救得改 oneDNN 的 jit:gemm dispatch，
而它的上限只有 `B_cab_symmetric` 的 1.09x —— 对称量化能直接拿到，不必绕这条路。

### 9.1.3 否决：把 opt-3 扩到 down_proj（probe 说赢 1.21x，模型里输 11.5 ms）

probe 里 `A_abc` 对 down（K=8960, N=1536）是 **1.210 → 0.999 ms = 1.21x**，而第二轮从来没
在模型里量过 down 的 abc（docs/06 §6.9.3 该格写的是「（未启用）」），docs/06 §6.4 当年还
估过 down 能省 ~7 ms。所以这看起来是第二轮漏掉的一笔。

`N/K` 比上，gate/up = 5.83（赢）、down = 0.17、qkv = 1.33（输）、o_proj = 1.00（输）。
一个自然的假设是：**真正的判据不是「N 远大于 K」而是「两个方向上足够长条」**，近方形才输。
按这个假设临时在 `fully_connected_onednn.cpp` 里加了 env 开关
（`GME_R3_TALL` / `GME_R3_RECT_RATIO`，沿用 docs/05 §5.1 的 `GME_OPT_*` 做法，一次编译测两臂）。

交替 3 轮 A/B（同一构建、同一会话）：

| | wall p50 | GEMM 合计 | down_proj |
| --- | ---: | ---: | ---: |
| `GME_R3_TALL=0`（现状规则） | **238.17** | **151.61** | **42.09** |
| `GME_R3_TALL=1`（加上 down） | 250.17 | 163.78 | 53.58 |

三轮 wall 分别 250.0 / 250.2 / 250.2，**方差极小，不是噪声**：down_proj 从 42.09 涨到
53.58，**+11.5 ms**。假设被否掉，`N≥4K` 保持原样，实验代码已回退。

> 这一条的价值超出 down_proj 本身：**probe 对 abc layout 的预测，只在 gate/up 上成立。**
> qkv（probe 1.00x / 模型 0.42x）、o_proj（probe 1.34x / 模型 0.86x）、down（probe 1.21x /
> 模型 0.79x）三个形状全部同向高估。docs/06 §6.6 里「模型内比单算子慢 1.3–1.5x」那条未解决项
> 现在有了更锋利的表述：**单算子 probe 系统性高估 N-contiguous 权重的收益**，因此
> opt-3 的 `N≥4K` 门槛应当被理解为「实测拟合出来的经验边界」，而不是「probe 推出来的物理边界」。

### 9.1.4 否决：qkv 的 1.95x model-vs-probe 缺口不是 bias

docs/06 §6.3 记录 qkv 模型内 0.465 ms/call vs probe 0.236，是全表最大的未解释缺口。qkv 是
唯一带 bias 的融合 3-FC，所以先给 bias 定价：`U_cab_bias` 相对 `A_gme_cab` 是
**0.99–1.00x**（qkv 列 0.94x，即 +7% 而不是 +95%）。**bias 解释不了这个缺口**，
该未解决项保持开放。

---

## 9.2 SDPA causal：本轮最大的一笔（−16.2 ms，且是恒等变换）

### 9.2.1 根因：不是 GME 写错了，是 tracing 下 transformers 不允许跳过 mask

IR 里 28 个 `ScaledDotProductAttention` 全是 `causal="false"` 且带一个 `[B,1,S,S]` 显式
mask。根因在 `transformers/masking_utils.py::_ignore_causal_mask_sdpa()`，它的第一个条件
就是 `not is_tracing(padding_mask)`：

```python
if (
    not is_tracing(padding_mask)          # <-- torch.jit.trace 下永远 False
    and (query_length == 1 or kv_length == query_length)
    and (padding_mask is None or padding_mask.all())
):
    return True
```

`ov.convert_model()` 走的正是 `torch.jit.trace`，所以 `is_tracing()` 恒为真，
`create_causal_mask()` 只能老老实实物化一张 S×S。这是 transformers 刻意的保护
（避免把 `is_causal` 烧进导出图，pytorch/pytorch#108108），**不是 bug**——但在 GME 这个
场景里，把它烧进去恰好是我们想要的。

### 9.2.2 为什么对 GME 是恒等变换

三条同时成立：

1. **kernel 真的会跳过上三角。** `sdpa_micro.cl:226` 在 `IS_CAUSAL` 下把 K 循环上界收成
   `causal_k = min(k, wg_j0 + ugemm_kq_wg_tile_n)`，是**不进循环**，不是算完再 mask。
2. **padding mask 对被读取的输出是冗余的。** 包是**右 padding**（`gme_openvino.py` 显式
   `raise` 掉非 right），池化取**末位有效 token**。右 padding 下 padding 位置的 index 严格
   大于任何被读取 token 的 index，causal 掩码已经把它们排除了。
3. **position_ids 不需要 mask。** 原本 `get_rope_index()` 在无图像时走
   `attention_mask.cumsum(-1) - 1`；右 padding 下有效区等于 `arange`，所以显式传
   `arange(S)` 与原来在有效区上逐位相同，还顺手把 cumsum 从图里去掉。

### 9.2.3 改法

`02_export.py`：`GmeLanguageEmbed(base, causal_attention=True)` 时传
`attention_mask=None` + 显式 `position_ids`，并在 trace 期间用
`suppress_causal_mask_materialization()` 把 `create_causal_mask` 打成返回 `None`。
mask 仍留在图输入上（它还要选池化下标）。新增 `--causal-sdpa` 与 `--language-suffix`，
产物是独立文件，老 IR 一律不动。

### 9.2.4 两个必须绕开的 plugin 限制（各花了一轮测量才定位）

朴素改法跑出来是**灾难**，两个原因互相独立：

| 症状 | 根因 | 处理 |
| --- | --- | --- |
| v5 导出的**对照** IR（保留 mask）wall 295.5 ms，SDPA 掉出 `ocl::sdpa::opt`，出现 `softmax_gpu_bf` 22.42 ms + `jit:gemm:any__f16` 32.99 ms | transformers 5 的 mask 是 **BOOL**（v4 是 additive FP32）。GPU plugin 的 SDPA 不接受布尔 mask，于是把 SDPA **分解**成显式 MatMul+Softmax | 与本轮目标无关（causal 路径根本没有 mask）。但它是**对照组必须存在**的理由，见 §9.2.5 |
| causal IR wall **427.96 ms**，出现 `undef` 40.91 ms / 577 calls | `attention_mask is None` 会让 `use_gqa_in_sdpa()` 返回 True → transformers 跳过 `repeat_kv` 改传 `enable_gqa=True` → 导出成 **rank-5** SDPA（Q `[B,2,6,S,128]`，K/V `[B,2,1,S,128]`）。**plugin 不 lower rank-5 的通用 SDPA**，同样分解 | 在 trace 期间把 `use_gqa_in_sdpa` 打成 `False`，保住 `repeat_kv` → SDPA 维持 rank-4。由此产生的 Unsqueeze/Broadcast/Reshape **正是 opt-1 已经消除的那一串**，所以零代价 |

修好后 IR 的样子：28 个 SDPA 全是 `causal="true"`、rank-4、**没有 mask 输入端口**、
`Select` 从 1 降到 0；模型输入仍是 `input_ids / attention_mask / image_embeds` 三个，
**运行时 `gme_openvino.py` 无需任何改动**。

### 9.2.5 精度：为什么必须用 FP32 逐位比对，而不能只看量化后的 cosine

先看 `04_validate_python.py`（vs PyTorch FP32 golden，GPU）：

| precision | text | query | image | fused |
| --- | ---: | ---: | ---: | ---: |
| `int4`（已交付，v4 导出） | 0.944538 | 0.955146 | 0.943653 | 0.952074 |
| `int4_v5ctrl`（v5 导出，保留 mask） | **0.950983** | 0.955428 | 0.941987 | 0.950910 |
| `int4_causal`（v5 导出，causal） | 0.941730 | 0.955173 | 0.945703 | 0.954420 |
| `int4_gs64`（已交付） | 0.949893 | 0.957442 | 0.942268 | 0.960488 |
| `int4_gs64_v5ctrl` | **0.937888** | 0.956787 | 0.941846 | 0.953098 |
| `int4_gs64_causal` | 0.944205 | 0.957576 | 0.941856 | 0.959374 |

如果只看 `int4` vs `int4_causal`，text 掉了 0.0028，会得出「causal 伤精度」的结论。
**但对照组证明这是重量化噪声**：`_v5ctrl` 和 causal 只差 mask，text 却分别 **+0.0064**
（gs128）和 **−0.0120**（gs64），**方向相反、幅度更大**。`text_cos_min` 是「所有样本里最差
的一个」，而每次换源 IR 都要重跑 AWQ + scale estimation（30 条校准文本）搜出一套不同的权重
—— 这个搜索的 run-to-run 抖动量级（±0.01）本身就比待测效应（0.003）大好几倍。

所以真正能回答问题的是**去掉量化这一层**，直接比两个 **FP32** IR
（`check_causal_equivalence.py`，CPU，fp32）：

| case | 有效长度 | cosine | max_abs_diff |
| --- | --- | ---: | ---: |
| seq1 | [1] | 1.000000000 | 0.000e+00 |
| seq32 | [32] | 1.000000000 | 0.000e+00 |
| seq512 | [512] | 1.000000000 | 0.000e+00 |
| seq1024 | [1024] | 1.000000000 | 0.000e+00 |
| **batch2_padded** | [1024, 300] | **1.000000000** | **0.000e+00** |
| **batch3_padded** | [512, 129, 7] | **1.000000000** | **0.000e+00** |

**六个用例全部逐位相同，包括两个不等长右 padding 批**——正是 §9.2.2 第 2 条推理唯一可能
出错的地方。causal 改造是**恒等变换**，已证。

> 附带结论：`text_cos_min` 这个指标在本项目的样本量下**分辨不了 0.01 以内的差异**。
> docs/08 里 per-OC 掉到 0.83~0.88 的量级（0.06~0.11）远在噪声之上，那个结论不受影响；
> 但今后任何「重新量化后 cosine 动了千分之几」的对比，都必须先给出同源对照组。

### 9.2.6 性能

交替 3 轮 A/B（同一构建、同一会话，`profile_ops.py --length 1024`）：

| | int4 | int4_causal | int4_gs64 | int4_gs64_causal |
| --- | ---: | ---: | ---: | ---: |
| `jit:gemm:any__i8` | 152.45 | 152.81 | 166.52 | 167.13 |
| `ocl::sdpa::opt__f16` | **25.99** | **11.62** | **25.98** | **11.58** |
| `dynamic_quantize_gpu_opt` | 27.04 | 25.49 | 25.39 | 25.08 |
| `rms_gpu_bfyx_opt` | 9.34 | 9.39 | 9.41 | 9.37 |
| `permute_ref` | 0.88 | 0.88 | 0.88 | 0.88 |
| profiled 合计 | 222.09 | 205.83 | 235.95 | 219.38 |
| host（wall − profiled） | 16.39 | 16.50 | 17.00 | 16.59 |
| **wall p50** | **238.48** | **222.33** | **252.94** | **235.97** |
| 三轮原始 wall | 238.5/238.0/239.0 | 222.3/223.5/221.4 | 253.5/252.9/251.3 | 236.6/236.0/235.6 |

- **SDPA 2.24x**（25.99 → 11.62），GEMM 完全不动（152.45 → 152.81），符合预期。
- gs128 **1.073x（−16.15 ms）**、gs64 **1.072x（−16.97 ms）**。
- opt-1/opt-2 在新 IR 上照常生效（broadcast 0、permute 0.88）。

长度扫描：

| tokens | 基线 wall | causal wall | speedup | 基线 SDPA | causal SDPA |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 50.29 | 46.93 | 1.072x | 0.65 | 0.44 |
| 512 | 128.94 | 124.05 | 1.039x | 7.26 | 3.21 |
| 1024 | 238.48 | 222.33 | **1.073x** | 25.99 | 11.62 |

SDPA 的绝对收益随 S² 增长（0.21 / 4.05 / 14.37 ms），但 128 token 上 wall 却省了 3.36 ms
——那里省的主要不是 SDPA，而是**被删掉的 mask 构造算子本身的调度开销**（`Select` +
若干 eltwise，28 层各一份，小张量上完全由 launch 开销主导）。512 那一行 1.039x 低于两侧，
且只跑了 2 轮，按本项目口径只作方向性参考。

### 9.2.7 这一笔不需要改 OpenVINO 一行代码

Phase 2 全部改动在 `02_export.py` + 重新量化，**GPU plugin 零改动**，因此它与 opt-1/2/3
以及任何 plugin 侧后续工作完全正交、可直接叠加。

---

## 9.3 顺带修好的：导出链在 transformers 5 下已经跑不起来

这不是本轮目标，但不修就无法重新导出，且它是一个**已经存在于仓库里的隐性故障**：
唯一可用的环境 `/home/chaofan/envs/qwen_env` 是 transformers **5.0.0**，而现有 IR 是
transformers 4.x 时代导出的。v5 的 API 变动让 `02_export.py` 直接 ImportError。

修的都是版本兼容，不改任何数值语义：

| 位置 | v5 的变化 | 处理 |
| --- | --- | --- |
| `AutoModelForVision2Seq` | 已删除 | `try/except` 回落到 `AutoModelForImageTextToText` |
| checkpoint 自带的 `gme_inference.py`（`fetch_image`，以及 `01_reference.py`/`04_validate_python.py` 的 golden）也 import 老名字 | 同上 | 把新类 **alias 回 `transformers.AutoModelForVision2Seq`**，不改 vendored 上游文件 |
| `base.model.embed_tokens` | 下移到 `base.model.language_model` | 改用两版都有的 `base.model.get_input_embeddings()` |
| `base.visual` | 下移到 `base.model.visual` | `getattr(base, "visual", None) or base.model.visual` |
| `base.config.hidden_size` | 下移到 `config.text_config` | `getattr(...) or base.config.get_text_config().hidden_size` |

**同时必须知道：v5 导出的 IR 不能直接替换现有 IR。** §9.2.4 第一行已经量到，v5 的
BOOL mask 会让 SDPA 在 GPU plugin 上被分解，wall 从 238 涨到 295。也就是说
**「只是把导出脚本跑一遍」这件事本身就是一次 24% 的性能回归**——只有叠加 causal
（把 mask 整个去掉）才不仅避开这个坑，还净赚 16 ms。这一条务必写进任何重新导出的
操作手册里。

### 9.3.1 另一个陷阱：NNCF 的 fingerprint 不覆盖源 IR 内容

`03_finalize.py::_fingerprint()` 的 payload 是 `repr((src_xml.name, sorted(kw.items())))`
—— **只有源 IR 的文件名，没有内容**。所以重新导出同名 FP32 IR 之后，已有的 INT4 产物
看起来仍然「config unchanged, skip」，实际却是用旧源 IR 压出来的。本轮因此白测了一轮
（拿 rank-5 那版 causal IR 的 INT4 去跑）。

`03_finalize.py::compress()` 里加了防护：**源 IR 比产物新就强制重压**，不再依赖记得加 `--force`。
（该防护对已交付产物永不触发——它们的源 IR 比自己旧。）

---

## 9.4 保持分组的对称量化（gs128-sym / gs64-sym）：收益真实，但精度代价更大

docs/08 测的是 **per-OC 对称**（`group_size=-1` + `INT4_SYM`），把「分组变粗」和「去掉
zero-point」两个因素**一起**换了，结论是精度不达标。但 §9.1 的 probe 指出，
**只去掉 zero-point、保持 K 分组** 是另一格：`B_abc_symmetric` 1.28x、
`R_sym_gs64_cab` 1.11x。这个组合此前**从未测过**，而 docs/08 的精度崩塌按其自身分析
来自分组变粗，不来自对称性。

`03_finalize.py --round3-extras` 从**同一个 causal 源 IR** 压出 gs128-sym / gs64-sym，
因此与 §9.2 的 `int4_causal` / `int4_gs64_causal` 之间**只差 `mode` 一个变量**——这是
§9.2.5 之后唯一可信的对比方式。

### 9.4.1 性能：−8.5 ms，几乎全部来自 qkv

交替 3 轮 A/B，1024 token：

| | int4_causal | int4_sym_causal | int4_gs64_causal | int4_gs64_sym_causal |
| --- | ---: | ---: | ---: | ---: |
| GEMM 合计 | 152.84 | **144.77** | 166.44 | **157.62** |
| gate_proj | 50.07 | 49.91 | 55.66 | 55.84 |
| up_proj | 35.62 | 35.03 | 40.77 | 41.75 |
| down_proj | 42.09 | 40.30 | 43.80 | 40.01 |
| **qkv** | 13.25 | **7.96** | 15.24 | **9.38** |
| o_proj | 11.79 | 11.35 | 11.08 | 10.64 |
| **wall p50** | 222.92 | **214.47** | 237.18 | **227.47** |
| 三轮原始 wall | 222.9/223.1/222.0 | 213.6/215.7/214.5 | 237.2/237.2/235.3 | 227.5/227.8/226.0 |

gs128 **1.039x（−8.45 ms）**、gs64 **1.043x（−9.71 ms）**。收益结构与 probe 完全吻合：
**qkv 一项就占 5.29 ms（1.66x）**，因为它是最窄的投影、zero-point 补偿占比最高
（probe `B_cab_symmetric` 的 qkv 列正是全表最高的 1.20x）；gate/up 已经在 abc layout 上，
probe 预测只剩 1.025x，实测基本不动。

### 9.4.2 精度：query cosine 掉 0.014，是噪声底的 20 倍

| precision | text | query | image | fused |
| --- | ---: | ---: | ---: | ---: |
| `int4_causal`（非对称，同源） | 0.941730 | **0.955173** | 0.945703 | 0.954420 |
| `int4_sym_causal` | 0.926687 | **0.941559** | 0.935076 | 0.944311 |
| `int4_gs64_causal`（非对称，同源） | 0.944205 | **0.957576** | 0.941856 | 0.959374 |
| `int4_gs64_sym_causal` | 0.937980 | **0.943423** | 0.935988 | 0.966638 |

这里**必须看 query 而不是 text**。§9.2.5 已经证明 `text_cos_min`（所有样本里最差的一个）
在重量化之间抖动 ±0.01；而 `query` 是单样本 cosine，在本轮所有非对称变体上分别是
0.955146 / 0.955428 / 0.955173（gs128 族）和 0.957442 / 0.956787 / 0.957576（gs64 族），
**极差只有 0.0007**。对称化让 query 掉 **−0.0136（gs128）/ −0.0142（gs64）**，
是这个噪声底的 **20 倍**，且两个分组、四个模式（text/query/image 一致下降）方向一致
——这是系统性退化，不是抖动。

### 9.4.3 结论：不采纳为默认

量纲对比很清楚：**gs64 的存在就是为了用 14.5 ms 换 query +0.0023**（§9.0.3、§9.1）。
在这样一个「精度优先」的偏好序下，用 query −0.014 去换 8.5 ms 是反向的交易——
代价是 gs64 收益的 6 倍，而省下的时间只有 gs64 花掉的 0.6 倍。

更直接的是一条**支配关系**：

| 配置 | wall | query cosine |
| --- | ---: | ---: |
| `int4_causal` | **222.92** | **0.9552** |
| `int4_gs64_sym_causal` | 227.47 | 0.9434 |

**`int4_causal` 同时更快且更准**，所以 gs64-sym 这一格可以直接排除。gs128-sym（214.47 /
0.9416）确实在 Pareto 前沿上，但它的精度已经低于**第三轮默认 gs128 非对称**（0.9552），
按本轮「不牺牲精度」的前提不予采纳。**已量化、已评估、作为可选项记录，不设为默认。**

---

## 9.4b Phase 3：qwen3.5 的 opt3 / opt7 在 GME 上的实测

docs/05 §5.2 (4) 估过「RMSNorm + DynamicQuantize 融合 ≈ −7 ms」、§5.1.4 估过
「DQ 去重 ≈ −1.3 ms」，而 `benchmark/openvino/qwen3.5_patch/` 里 opt7 / opt3 正是这两件事的
现成实现。按用户拍板的顺序——**先测量，再决定要不要自己重写**——把
opt3+opt4+opt5+opt6+opt7 整套叠在 GME opt1/2/3 之上（opt6 要 opt4 的 conv1d 文件做 context、
opt7 要 opt3+opt6 在先，所以必须整套叠，单独叠 opt7 会冲突）。

1024 token 实测：

| | GME opt1/2/3 | + qwen3.5 opt3…opt7 |
| --- | ---: | ---: |
| `dynamic_quantize_gpu_opt` calls | 140 | **112** |
| `dynamic_quantize_gpu_opt` ms | 27.04 | **24.94** |
| `rms_gpu_bfyx_opt` calls | 57 | **57** |
| `rms_gpu_bfyx_opt` ms | 9.34 | **9.27** |
| wall p50（int4） | 238.48 | 237.97 |
| wall p50（int4_causal） | 222.92 | 220.88 |

### 结论 1：opt3（DQ 去重）生效，但只值约 1%

DQ 调用数 **140 → 112**（每层 5 个降到 4 个，正是 gate/up 共享 `post_attention_layernorm`
那一次去重），kernel 时间 −2.1 ms。wall 上 −0.5~2.0 ms，处于这台机器会话间漂移的边缘。
量级与 docs/05 §5.1.4 的 ~1.3 ms 估计一致，**结论也一致：收益太小**。它是外部 patch、
可直接叠加，本轮**不重写**，只记录实测值。

### 结论 2：opt7（RMS→DQ 融合）在 GME 上**完全不生效**，那 7 ms 拿不到

`rms_gpu_bfyx_opt` 的调用数和时间**一点没动**（57 → 57，9.34 → 9.27），即**零次融合**。
根因在 `prepare_primitive_fusing.cpp::fuse_rms_into_dynamic_quantize()` 的两道 guard，
而 GME 会分别撞上其中一道：

| GME 的 DQ 配置 | 撞上的 guard | 出处 |
| --- | --- | --- |
| **per-token（默认）** | `attrs.quantization_type != Symmetric → continue`。per-token 时 `adj_group_size == UINT64_MAX`，`dynamic_quantize_fully_connected.cpp:138` 把它设成 **Asymmetric** | 该 pass 的注释自己写了「excludes the per-token mode the plugin picks by default on systolic hardware」 |
| **强制 `DYNAMIC_QUANTIZATION_GROUP_SIZE=128`** | `attrs.precomputed_reduction → continue`。group≠UINT64_MAX 且有静态权重 zp 时，`dynamic_quantize_fully_connected.cpp:85` 会把 `precomputed_reduction` 打开 | 实测 group=128 下 rms 仍是 57 calls / 9.20 ms |

而且 group=128 这条路本身也不通：实测 wall **220.88 → 239.49**，GEMM 153.56 → 167.70
（docs/05 §5.1.3 测过 group=32 是 392 ms，group=128 这一格本轮补齐，同样是负收益）。

顺带一个交叉印证：group=128 时 plugin 自己会启用 `precomputed_reduction` 喂给 oneDNN，
而 §9.1.2 的 probe 已经证明这个属性在本 oneDNN 版本上会把 matmul 打到 `ocl:ref:any`
或至少显著变慢——两条独立证据指向同一结论：**precomputed reductions 这条路在这套
软件栈上不成立**。

**所以 docs/05 §5.2 (4) 的「RMSNorm + DynamicQuantize 融合 ≈ −7 ms」现在的状态是：
方向仍然合理（RMS 写 3.1 MB、DQ 立刻读回，一来一回 ~350 MB），但现成实现拿不到，
需要给 `DynamicQuantizeKernelOpt` 的 FUSE_RMS 分支补上 per-token / 非对称形态。**
这是一件有界但非平凡的工作，本轮不做，列为下一轮首选。

---

## 9.5 本轮结论

### 采纳

| 改动 | 收益（1024 token） | 性质 |
| --- | --- | --- |
| **SDPA causal 导出**（§9.2） | gs128 **1.073x（−16.15 ms）**、gs64 **1.072x（−16.97 ms）**；SDPA 单项 **2.24x** | 导出侧，**FP32 逐位相同**（含不等长右 padding 批），**plugin 零改动** |

### 可选（实测有效但不设默认）

| 改动 | 收益 | 为什么不设默认 |
| --- | --- | --- |
| qwen3.5 `opt3`（DQ 去重，§9.4b） | DQ 140→112 calls、−2.1 ms（~1% wall） | 外部 patch、可叠加；收益在会话漂移边缘，本轮不重写 |
| 分组对称量化（§9.4） | gs128 **1.039x（−8.45 ms）**、gs64 1.043x | query cosine −0.014，是噪声底的 20 倍；与「精度优先」前提冲突 |

### 否决（都有实测数字）

| 候选 | 实测 | 结论 |
| --- | --- | --- |
| precomputed reductions（§9.1.2） | 1484 ms vs 1.288 ms | `jit:gemm` 拒绝该属性组合，整个 matmul 掉 `ocl:ref:any`。**否** |
| opt-3 扩到 down_proj（§9.1.3） | down 42.09 → 53.58，wall 238.17 → 250.17 | probe 高估，模型内 **+11.5 ms** 负收益。**否**，`N≥4K` 保持原样 |
| bias 解释 qkv 缺口（§9.1.4） | `U_cab_bias` 0.94–1.00x | 不是 bias。未解决项保持开放 |
| qwen3.5 `opt7`（RMS→DQ 融合，§9.4b） | rms 57→**57** calls，零次融合 | per-token 走 Asymmetric、group=128 走 precomputed_reduction，两道 guard 各挡一次 |
| `DYNAMIC_QUANTIZATION_GROUP_SIZE=128`（§9.4b） | wall 220.88 → 239.49 | 补齐 docs/05 §5.1.3 缺的这一格，同样负收益 |
| 直接重跑现有导出脚本（§9.3） | wall 238 → 295 | v5 的 BOOL mask 让 SDPA 被分解。**必须配合 causal 一起做** |

### 第三轮结束时推荐配置（第四轮前）

| 场景 | 精度 key | wall（1024） | query cosine |
| --- | --- | ---: | ---: |
| 第三轮默认 / 均衡 | **`int4_causal`** | **222.3** | 0.9552 |
| 精度优先 | **`int4_gs64_causal`** | 236.0 | **0.9576** |
| 对照：本轮之前的默认 | `int4` | 238.5 | 0.9551 |
| 对照：本轮之前的精度档 | `int4_gs64` | 252.9 | 0.9574 |

也就是：**同精度下 gs128 从 238.5 → 222.3（1.073x）、gs64 从 252.9 → 236.0（1.072x）**；
相对第一轮最初的 stock gs128（283.0）累计 **1.273x**。

### 仍然开放（下一轮的起点，按预估收益排序）

| 项 | 状态 |
| --- | --- |
| **RMS→DQ 融合**（~7 ms） | 方向成立（RMS 写 3.1 MB / DQ 立刻读回，56 次共 ~350 MB），但现成实现拿不到。需给 `DynamicQuantizeKernelOpt` 的 FUSE_RMS 分支补 **per-token + 非对称** 形态（§9.4b）。**本轮结束时的首选** |
| GEMM 的 ~69%（gate 50.1 / up 35.6 / down 42.1 / qkv 13.3 / o_proj 11.8） | causal 之后 GEMM 占比从 63% 升到 **69%**。距稠密 int8 天花板（`G_s8_dense` 1.68x）仍有近 2x，probe 已排除 layout（1.25x 已吃掉）和 zero-point（1.09x，精度不划算），**剩下的是 u4 反量化本身**（`C_s8wei_grouped` 1.33x），属 oneDNN `jit:gemm` 生成器 |
| 模型内比单算子慢 1.3–2.0x | §9.1.3 把它锐化成「**probe 系统性高估 abc layout**，只有 gate/up 例外」；qkv 的 1.95x 已排除 bias |
| gs64 的 14.5 ms 代价 | §9.0.3 首次量化，根因是 opt-3 在 gs64 上只兑现 −14.6 而非 −25.1 ms。§9.1 指出若采用 gs64 应配「对称 + cab」而非「非对称 + abc」，即 **opt-3 的 `N≥4K` 规则对 gs64 可能该关掉**——本轮未单独验证 |
| host 提交开销 | causal 后从 18.0 降到 **16.4 ms**，随 primitive 数下降，佐证 §9.2.6 中 128 token 的收益来自算子数而非 SDPA |
| `text_cos_min` 的噪声 | ±0.01，分辨不了千分级差异（§9.2.5）。今后精度对比应以 **query cosine**（极差 0.0007）为主，或提高样本量 |

---

## 9.6 本轮产物

```
gme-Qwen2-VL-2B-Instruct-Running/
  gme_compat.py                       新增：transformers 5 的名字兼容（01/02/04 共用）
  01_reference.py                     golden 已存在时跳过（--force 才重算），避免作废历史 cosine
  02_export.py                        --causal-sdpa / --language-suffix；--stage all 把语言 IR 导两遍
   03_finalize.py                      第三轮默认产出 causal 两档；--round3-extras 才产出对照组和分组对称；
                                      compress() 加「源比产物新就强制重压」防护
  ov_models/.../gme_language_fp32_causal.xml      causal=true、rank-4、无 mask
  ov_models/.../gme_language_fp32_v5ctrl.xml      对照：v5 导出但保留 mask
  ov_models/.../gme_language_int4[_gs64]_causal.xml          §9.2 采纳
  ov_models/.../gme_language_int4[_gs64]_v5ctrl.xml          §9.2.5 对照组
  ov_models/.../gme_language_int4[_gs64]_sym_causal.xml      §9.4 分组对称（可选，不设默认）
  artifacts/openvino_validation_round3_causal.json
  artifacts/openvino_validation_round3_v5ctrl.json
  artifacts/openvino_validation_round3_sym.json

gme-qwen2-vl-int4-hotspot-optimization/
  check_causal_equivalence.py         FP32 逐位等价性检查（含不等长右 padding 批）
  onednn/gemm_probe.cpp               新增 src_pr / bias 开关、gs64 / 分组对称 / bias 共 9 个 case
                                      GEMM_PROBE_SHAPES=gme 只跑五个真实投影
  onednn/results/r3_pricing.txt       §9.1 定价主表
  profile_ops.py / dump_outputs.py    IR 字典自动并入 gme_openvino_config.json 里声明的精度
  results/r3_base_ops_int4_1024.json          §9.0.2 gs128 基线
  results/r3_base_ops_int4_gs64_1024.json     §9.0.3 gs64 patched（首次）
  results/r3_STOCK_ops_int4_gs64_1024.json    §9.0.3 gs64 stock
  results/r3_acc_int4[_gs64]_patched.npz      §9.0.1 自洽性参考
  results/r3_causal_equivalence_fp32.json     §9.2.5
  results/r3_final_int4[_causal,_gs64_causal]_r*.json  §9.5 交付构建上的最终复核
  preexisting-worktree-backup/openvino-worktree-before-round3.diff   §9.0.1
  preexisting-worktree-backup/round3-untracked/                      opt6 的两个 untracked 文件
```

**GPU plugin 交付面：本轮没有新增任何 plugin 改动。** `benchmark/openvino` 现在的
`src/` 只 apply 了 `patches/all_gme_int4_gpu_opts.patch`（8 files, +390 −19），
§9.1.3 的 env 开关实验和 §9.4b 的 qwen3.5 五个 patch 都已回退，回退后复核：

| precision | 回退后 wall | 本轮测得 |
| --- | ---: | ---: |
| `int4` | 239.03 | 238.48 |
| `int4_causal` | 222.72 | 222.33 / 222.92 |
| `int4_gs64_causal` | 236.80 | 235.97 / 237.18 |

## 9.7 复现

```bash
cd gme-Qwen2-VL-2B-Instruct-Running

# 1+2. 完整流水线：01 -> 02 -> 03 可重现第三轮 language-only 历史包。
#      第四轮当前包还需要 mask-free visual IR；见 docs/11 的复现命令。
python 01_reference.py                 # 已有 golden 时自动跳过（--force 才会重算）
python 02_export.py                    # --stage all 是默认
python 03_finalize.py                  # 产出第三轮 language IR 的 int4_causal / int4_gs64_causal

# 复现 docs/09 的分析（对照组 + 分组对称，仅分析用，约 +3.7 GB / +20 min）
python 02_export.py --stage language --language-suffix _v5ctrl
python 03_finalize.py --round3-extras

# 3. 证明 causal 是恒等变换（需要上面的 _v5ctrl 对照）
cd gme-qwen2-vl-int4-hotspot-optimization
python check_causal_equivalence.py --output results/r3_causal_equivalence_fp32.json

# 4. 性能（务必先 source setupvars.sh，见 §9.0.1）
source "$OPENVINO_ROOT/install/setupvars.sh"
for r in 1 2 3; do for p in int4 int4_causal; do
  python profile_ops.py --precision $p --length 1024 --warmup 12 --repeat 8 \
         --output /tmp/ab_${p}_r${r}.json
done; done

# 5. 精度（vs PyTorch FP32 golden）
python ../04_validate_python.py --precisions int4,int4_causal,int4_gs64,int4_gs64_causal \
       --devices GPU --output ../artifacts/openvino_validation_round3_causal.json

# 6. oneDNN 定价表
onednn/build_probe.sh
GEMM_PROBE_SHAPES=gme onednn/gemm_probe 30
```
