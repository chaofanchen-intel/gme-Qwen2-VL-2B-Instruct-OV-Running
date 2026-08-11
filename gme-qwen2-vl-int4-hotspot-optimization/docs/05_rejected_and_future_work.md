# 5. 试过但否掉的方案，以及后续方向

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

---

## 5.2 后续方向（按预估收益排序）

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

### (5) host 侧提交开销 —— ~17 ms（6.5%）

wall 263.1 ms vs plugin profiled 246.2 ms，差 17 ms，全图约 590 个 primitive。
减少 primitive 数量（上面每一条融合都在做这件事）或降低 per-primitive 提交成本
都能吃到一部分。

---

## 5.3 关于 `openvino.genai`

**本轮没有修改 `openvino.genai`，也不应该改。** GME 的 Python 推理链路
（`gme_openvino.py`）直接用 `ov.Core().compile_model()` 驱动自定义导出的 IR，
完全不经过 GenAI。`benchmark/openvino.genai` 工作区里现存的两处改动
（`rag/embedding_pipeline.cpp`、`visual_language/qwen3_vl/classes.cpp`）是用户
既有工作，已备份在 `preexisting-worktree-backup/genai-worktree-before-gme-work.diff`，
本轮未触碰。

如果后续要走 GenAI 的 `TextEmbeddingPipeline`，两个 patch 一样生效 ——
它们在 GPU plugin 层，对上层 API 透明。

## 5.4 关于既有 Qwen3.5 patch 的关系

`benchmark/openvino/qwen3.5_patch/` 里的 5 个 patch 与本轮**互不冲突**，但有两处
交叠需要注意：

| Qwen3.5 patch | 与本轮的关系 |
| --- | --- |
| `opt2_permute_lws_f_axis` | 改 `permute_kernel_ref` 的 LWS。本轮 opt-2 是把 permute **整个消除**，两者叠加时 opt-2 先生效，opt2 的收益变得可忽略。实测在 GME 上 Qwen3.5-opt2 反而让 permute 从 4.93 变 6.23 ms，GME 场景下建议不要叠加 |
| `opt3_dynamic_quantize_reuse` | 做的正是 §5.1.4 的 DQ 去重，与本轮正交，可叠加，值 ~1.3 ms |
| `opt1` / `opt4` / `opt5` | 针对 SSM / conv1d / 动态 activation，GME 图里没有这些算子，不生效也无害 |
