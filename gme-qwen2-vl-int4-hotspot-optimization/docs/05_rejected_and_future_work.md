# 5. 试过但否掉的方案，以及后续方向

> 本文的候选排序以第三轮结束时的语言塔为边界。第四轮已落地视觉塔
> mask-free SDPA（见 [`docs/11`](11_round4_image_fused_optimization.md)）：
> `int8_vnomask` 和 `int4_causal_vnomask` 是当前 image/fused precision，且本轮
> 没有新增 plugin patch。下文保留历史否决数据；涉及当前默认的表述均按第三轮
> 历史上下文理解。

## 5.1 已试且**否掉**的方案（都有实测数据）

为了不重复踩坑，这里记录本轮实际编译、跑通、然后否掉的候选。测试方法是在
plugin 里临时加 `GME_OPT_*` 环境变量开关，一次编译测完所有组合，最终交付的
patch 里这些开关已全部移除。

基准（同一次会话内的 baseline）：**280.2 ms**。

| 候选 | 结果 | 结论 |
| --- | ---: | --- |
| `GME_OPT_SDPA_BCAST=1`（opt-1） | **269.9 ms** | ✅ 采纳 |
| `GME_OPT_HFUSE2=1`：允许 immad 设备上把 gate/up 两个 FC 横向融合 | **308.0 ms** | ❌ 否 |
| `GME_OPT_GATED_MLP=1`：打开 `FuseGatedMLP` | **484.4 ms** | ❌ 否 |

### 5.1.1 为什么 gate/up 横向融合反而更慢

`FullyConnectedHorizontalFusion` 默认 `min_num_fcs_to_fuse = 3`，只有在
`fuse_mlp_swiglu`（要求 `!use_onednn && !supports_immad && EU >= 128`）时才降到 2。
本机是 immad 设备，所以 q/k/v（3 个）融了、gate/up（2 个）没融。

强行降到 2 之后：

| exec_type | baseline | HFUSE2 | Δ |
| --- | ---: | ---: | ---: |
| `jit:gemm:any__i8` | 177.87 | 167.99 | **−9.9** |
| `dynamic_quantize_gpu_opt` | 25.75 | 24.49 | −1.3 |
| `swiglu_gpu_opt__f16` | 0 | **41.25** | **+41.3** |
| wall | 280.2 | **308.0** | +27.8 |

GEMM 确实变快了（`[1536→17920]` 一次比两次 `[1536→8960]` 好），DynamicQuantize 也
少了一个。但**代价是 SwiGLU 从 oneDNN post-op 掉出来变成独立 kernel**：
原本 `silu(gate)*up` 是 gate GEMM 的 `eltwise_swish + binary_mul` post-op（零成本），
融合后必须先算出 `[1024, 17920]` 再由 `swiglu_gpu_opt` 读 36 MB / 写 18 MB
（≈37 GB/s，接近这台机器的实测显存带宽上限）。净亏 27.8 ms。

> 若要救活这条路，得让 oneDNN GEMM 直接输出 SwiGLU 后的结果（即把 VariadicSplit +
> swiglu 也做成 post-op），那是 oneDNN 侧的工作。源码里 `min_num_fcs_to_fuse` 上方
> 那句注释 "Onednn gemms are to be handled in a different way (TBD)" 说的就是这件事。

### 5.1.2 为什么 `FuseGatedMLP` 打开会灾难性变慢

`transformations_pipeline.cpp` 里：

```cpp
const bool disable_gated_mlp_fusion = GPU_DEBUG_VALUE_OR(config.get_disable_gated_mlp_fusion(), true);
```

`OV_CONFIG_DEBUG_OPTION` + release 构建（`ENABLE_DEBUG_CAPS=OFF`）→ 这个 pass 在
官方 release 里是**无条件关闭**的。强行打开后 `GatedMLP` 算子在本机没有优化实现，
落到 `ocl:ref:any__f16`，单这一项就 361.6 ms。**默认关闭是对的。**

### 5.1.3 动态量化相关

| 配置 | GEMM | DynamicQuantize | wall |
| --- | ---: | ---: | ---: |
| 默认（per-token，group=1536） | 177.99 | 26.55 | 283.0 |
| `DYNAMIC_QUANTIZATION_GROUP_SIZE=0`（关掉） | 213.63 | 0 | 287.5 |
| `DYNAMIC_QUANTIZATION_GROUP_SIZE=32` | 220.51 | 97.93 | 392.2 |

默认配置是三者里最好的，但**整套动态量化机制净收益只有 4.5 ms**。不要在这上面
再花时间调 group size。

### 5.1.4 DynamicQuantize 去重（gate/up 共享）

每层 5 个 DQ，其中 gate 和 up 消费同一个 `post_attention_layernorm` 输出却各量化
一遍。从 5.1.1 的数据里可以读出去重的价值：**约 1.3 ms（0.5%）**。收益太小，
且 `qwen3.5_patch/opt3_dynamic_quantize_reuse.patch` 已经有一份现成实现，
本轮不重复做。

### 5.1.5 第二轮否掉的方案

完整过程在 `06_round2_onednn_investigation.md`，这里只列结论，避免重复踩坑：

| 候选 | 结果 | 结论 |
| --- | ---: | --- |
| 权重描述符交给 oneDNN 自己选（`weights_fmt = any`） | 251.4 ms | ❌ 否。oneDNN 的启发式对**每个**形状都挑了 padded 的 N 连续 stride，比直接强制 `abc`（246.1 ms）还慢 |
| 只按 `N ≥ K` 决定是否换排布 | 端到端**变慢** | ❌ 否。逐 matmul 定价显示 qkv（N/K=1.33）0.474 → 1.141 ms、o_proj（N/K=1.00）0.419 → 0.486 ms 都是负收益。门槛必须抬到 `N ≥ 4K`（§6.9.3） |
| 不看行数、一律换排布 | 32/64 token 上 0.84–0.85x | ❌ 否。必须叠加 ≥128 行的门槛（§6.9.4） |
| per-OC（per-output-channel）量化替代 group-128 | **1.55x** | ⚠️ 挂起。收益是所有候选里最大的，但**要重新导出模型且有精度代价**，不属于"plugin 侧无损优化"这个范畴（§6.3） |

顺带更正一条第一轮的排查结论：第一轮曾把一次"描述符换成 `abc` 后 qkv/o_proj
也变快"的实验当成证据。第二轮复现失败 —— 那次实验只换了描述符、没有真的重排
buffer，kernel 是拿新 stride 读旧数据，**计时不可信**。§5.1 上面几条不受影响
（它们都是 wall 级别的 A/B），但那一条已作废。

---

## 5.2 后续方向（按预估收益排序）

> **第三轮状态更新**（见 [`09_round3_further_latency_optimization.md`](09_round3_further_latency_optimization.md)）：
>
> | 本节的方向 | 第三轮结论 |
> | --- | --- |
> | (1) oneDNN `jit:gemm` 的 int8 压缩 GEMM（~2×） | **仍然开放，且现在是唯一的大头**（causal 之后 GEMM 占比 63% → 69%）。probe 已把它收窄到「u4 反量化本身」：layout 的 1.25x 已由 opt-3 吃掉，zero-point 只值 1.09x 且精度不划算（§9.4），`C_s8wei_grouped` 1.33x vs 稠密 int8 1.68x 就是剩下的空间 |
> | (2) SDPA 用 `causal=true` 而不是显式 mask（~13 ms / 5%） | **已落地，实测比预估更好：SDPA 25.99 → 11.62 ms（2.24x），端到端 −16.2 ms（1.073x）**，且 FP32 逐位相同。见 §9.2。落地时踩到两个 plugin 限制（BOOL mask、rank-5 enable_gqa），处理方式见 §9.2.4 |
> | (3) 对称 INT4 量化 | **已测：不采纳。** 保持 K 分组的对称量化（本节说的正是这个）实测 −8.45 ms（1.039x），但 query cosine −0.014，是噪声底的 20 倍。见 §9.4。另 per-OC 对称见 docs/08 |
> | (4) RMSNorm + DynamicQuantize 融合（~7 ms） | **方向成立但现成实现拿不到。** `qwen3.5_patch/opt7` 在 GME 上**零次融合**（rms 57 → 57 calls），因为 GME 的 per-token DQ 是 Asymmetric、而强制 group=128 又会打开 `precomputed_reduction`，两道 guard 各挡一次。需给 `DynamicQuantizeKernelOpt` 的 FUSE_RMS 分支补 per-token/非对称形态。见 §9.4b。**这是下一轮的首选** |
> | (5) host 侧提交开销（~18 ms） | causal 之后从 18.0 降到 **16.4 ms**，确认它随 primitive 数下降而不是固定开销。128 token 上 causal 的收益几乎全部来自这一项 |
>
> §5.1.4 的 DQ 去重也已实测：`qwen3.5_patch/opt3` 确实生效（DQ 140 → 112 calls，−2.1 ms），
> 量级与本节 ~1.3 ms 的估计一致，**结论同样是收益太小**，本轮不重写。

> 第二轮已经把 (1) 里"权重排布选错"这一块吃掉了（opt-3，GEMM 177.5 → 151.1 ms）。
> 下面 (1) 的证据链是第一轮采集的，**结论仍然成立**：kernel 自身的吞吐问题和
> 排布问题是两回事，换完排布后 GEMM 仍占 GPU 时间 ~63%。

### (1) oneDNN `jit:gemm` 的 int8 权重压缩 GEMM —— 预估空间 ~2×，约 80 ms

这是**唯一一个数量级上的大头**。证据链：

- 占 GPU 时间 68%（177.6 ms / 2682 GFLOP = **15.0 TFLOPS**）；
- 同一 kernel 的 f16 路径是 12.5 TFLOPS，int8 只快 **1.2×**，而 XMX 上 int8 峰值
  是 f16 的 **2×** —— 说明 int8 路径浪费了将近一半；
- VTune：XMX 流水线只有 21–33% 的时间在发射，stall 主因 SBID 38–47%
  （访存 scoreboard 依赖）+ Barrier 13–26%；
- 吞吐在 M=1024 已饱和（512/1024/2048 → 13.2/15.0/14.6 TFLOPS），不是 batch 不够。

怀疑方向：u4 权重在 GEMM 内部的解压 + per-128-group zero-point 补偿占掉了本该给
XMX 的 ALU 周期。验证手段是 `benchdnn --matmul` 扫 `wei:u4 + zp` vs `wei:s8 无 zp`
的同形状对比。这属于 oneDNN 而不是 OpenVINO，建议单独立项。

### (2) 模型侧：SDPA 用 `causal=true` 而不是显式 mask —— 预估 ~13 ms（5%）

IR 里 SDPA 是 `causal="false"` + 一个由 `masked_fill/Select` 构造的显式
`[B,1,S,S]` mask。这意味着 **kernel 老老实实算了整个 S×S，包括被 mask 掉的上三角**，
纯浪费一半注意力计算（当前 26.1 ms）。

对 batch=1 且 attention_mask 全 1 的场景（GME 的绝大多数 text/query 调用），
导出时把 SDPA 设成 `is_causal=True` 并去掉 mask 输入，kernel 就能跳过上三角。
这是 `02_export.py` 侧的改动，不是 OpenVINO 侧。

代价：padded batch（`batch_mode="padded"`，右 padding）仍然需要显式 mask。
可行做法是导出两条路径，或者对 padded batch 走 `per_sample`。

### (3) 模型侧：对称 INT4 量化 —— 收益待测

当前 IR 是**非对称** INT4（`attr-zero-points:wei:7:u8:128x1`）。对称量化能让
oneDNN 省掉 zero-point 补偿。`benchmark/models/` 下已经有
`Qwen3-1.7B-INT4-g128sym` 之类的对照，可以先在别的模型上量化这个收益，
再决定要不要重新导出 GME。

### (4) RMSNorm + DynamicQuantize 融合 —— 预估 ~7 ms（2.6%）

每层 2 个 DQ 的输入直接来自 RMS 输出：RMS 写 `[1024,1536]` f16（3.1 MB），
DQ 立刻读回来。融合掉这一来一回，56 次共省 ~347 MB，按实测 ~50 GB/s ≈ 7 ms。
需要新增一个融合 primitive 或把 DQ 做成 RMS 的 post-op，工作量中等。

### (5) host 侧提交开销 —— ~18 ms（7.7%）

wall 238.2 ms vs plugin profiled 219.9 ms，差 18.3 ms，全图约 590 个 primitive。
这一项是**绝对值几乎不变、占比随 GPU 时间下降而上升**的固定开销：第一轮是
263.1 vs 246.2（17 ms / 6.5%）。减少 primitive 数量（上面每一条融合都在做这件事）
或降低 per-primitive 提交成本都能吃到一部分。

---

## 5.3 关于 `openvino.genai`

**本轮没有修改 `openvino.genai`，也不应该改。** GME 的 Python 推理链路
（`gme_openvino.py`）直接用 `ov.Core().compile_model()` 驱动自定义导出的 IR，
完全不经过 GenAI。`benchmark/openvino.genai` 工作区里现存的两处改动
（`rag/embedding_pipeline.cpp`、`visual_language/qwen3_vl/classes.cpp`）是用户
既有工作，已备份在 `preexisting-worktree-backup/genai-worktree-before-gme-work.diff`，
本轮未触碰。

如果后续要走 GenAI 的 `TextEmbeddingPipeline`，三个 patch 一样生效 ——
它们都在 GPU plugin 层，对上层 API 透明。

## 5.4 关于既有 Qwen3.5 patch 的关系

`benchmark/openvino/qwen3.5_patch/` 里的 5 个 patch 与本轮**互不冲突**，但有两处
交叠需要注意：

| Qwen3.5 patch | 与本轮的关系 |
| --- | --- |
| `opt2_permute_lws_f_axis` | 改 `permute_kernel_ref` 的 LWS。本轮 opt-2 是把 permute **整个消除**，两者叠加时 opt-2 先生效，opt2 的收益变得可忽略。实测在 GME 上 Qwen3.5-opt2 反而让 permute 从 4.93 变 6.23 ms，GME 场景下建议不要叠加 |
| `opt3_dynamic_quantize_reuse` | 做的正是 §5.1.4 的 DQ 去重，与本轮正交，可叠加，值 ~1.3 ms。**注意与本轮的 opt-3 同名但完全无关**，本轮 opt-3 是权重排布 |
| `opt1` / `opt4` / `opt5` | 针对 SSM / conv1d / 动态 activation，GME 图里没有这些算子，不生效也无害 |
