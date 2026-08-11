# 1. 方法论与基线

## 1.1 问题陈述

`gme-Qwen2-VL-2B-Instruct` 在 Intel iGPU 上以 INT4 精度做纯文本 `text` / `query`
embedding 时，**1024 token 单样本延迟约 280 ms**，明显偏慢。本轮工作的目标是：

1. 定位真实热点（而不是猜测）；
2. 在 **OpenVINO 官方 2026.3 release** 源码基础上做可交付的优化；
3. 保证数值无损；
4. 把过程、结论、patch 全部沉淀成文档。

## 1.2 被测对象

| 项 | 值 |
| --- | --- |
| 模型包 | `$GME_ROOT/ov_models/gme-qwen2-vl-2b-instruct-openvino` |
| 语言 IR | `gme_language_int4.xml`（NNCF INT4，group size 128，**非对称**，带 zero point） |
| 结构 | Qwen2-VL 语言塔，28 层，hidden 1536，intermediate 8960，12 heads / 2 KV heads（GQA），head_dim 128 |
| 设备 | Intel Panther Lake-P iGPU（device id `0xb0a0`），32 XVE，2.45 GHz，OpenCL 3.0 NEO，driver 26.22.38646.6 |
| 主机 | 16 逻辑核，30 GiB 内存，Ubuntu 24.04 |
| 推理精度 | GPU 默认（f16）。`gme_openvino.py` 只在 `precision == "fp32"` 时才强制 f32，INT4 走设备默认 |
| batch 模式 | `padded`（GPU 上的默认，1 次 `[B, S]` 调用） |

**关键点：这条链路完全不经过 `openvino.genai`。** `gme_openvino.py` 直接用
`ov.Core().compile_model()` 驱动一个自定义导出的 IR，没有 KV cache、没有
`LLMPipeline`、没有 `Text2SpeechPipeline` 之类。因此本轮所有源码改动都落在
**OpenVINO GPU plugin**，`openvino.genai` 未做任何修改（见 §5.4 的说明）。

## 1.3 基线口径：为什么必须重新编译

`$OPENVINO_ROOT` 的 HEAD 是 `c2638477a6`（只新增了 `qwen3.5_patch/` 目录，源码
与 tag `2026.3.0` = `8a17657b99` 完全一致），但**工作区里还 apply 着 Qwen3.5 的 5 个
patch**：

```
 M src/plugins/intel_gpu/src/graph/dynamic_quantize.cpp
 M src/plugins/intel_gpu/src/graph/impls/ocl_v2/paged_causal_conv1d_ref.{cl,cpp}
 M src/plugins/intel_gpu/src/graph/layout_optimizer.cpp
 M src/plugins/intel_gpu/src/kernel_selector/cl_kernels/activation_opt.cl
 M src/plugins/intel_gpu/src/kernel_selector/kernels/activation/activation_kernel_opt.cpp
 M src/plugins/intel_gpu/src/kernel_selector/kernels/permute/permute_kernel_ref.cpp
 M src/plugins/intel_gpu/src/plugin/transformations/dynamic_quantize_fully_connected.cpp
```

其中 `opt2_permute_lws_f_axis`（permute）和 `opt3_dynamic_quantize_reuse`
（DynamicQuantize 复用）**正好落在 GME 的热点路径上**，所以它们不是"与 gme 无关"，
用它们做基线会污染结论。

处理方式：

1. `git diff` 备份工作区到 `preexisting-worktree-backup/`（原样保留，未丢失）；
2. `git checkout -- src/` 把源码恢复到 `2026.3.0`；
3. 重新编译 `openvino_intel_gpu_plugin` 并 install；
4. 之后所有 baseline / patched 数字都以这个干净的 2026.3.0 为准。

实测差异（1024 token，INT4）：带 Qwen3.5 patch 的构建 279.7 ms，干净 2026.3.0
283.0 ms。同时 `DynamicQuantize` 调用数从 112（4/层）变回 140（5/层），
`permute_ref` 从 6.23 ms 变成 4.93 ms —— 证明这两个 patch 确实在这条路径上生效，
必须剔除。

## 1.4 三套测量工具

| 工具 | 作用 | 特点 |
| --- | --- | --- |
| `profile_gme_int4_gpu.py` | 端到端 GME encoder（含 processor/tokenizer） | 用户视角的真实延迟 |
| `profile_ops.py` | 只喂语言 IR，开 `ov::enable_profiling`，按 **exec_type**（真实 GPU kernel）和 **node name**（原始算子）双维度聚合 | 把 GPU 时间直接归因到源码算子，比 VTune 更好定位 |
| VTune `hotspots` / `gpu-hotspots` | XVE 占用率、stall 原因、XMX 利用率、work-group 尺寸 | 回答"为什么这个 kernel 慢" |

`profile_gme_int4_gpu.py` 里构造"恰好 N 个 token"的 workload 原本要做 ~N 次
tokenizer 调用，这部分 CPU 时间会严重稀释 VTune 的 GPU 占比（旧的 capture 里
`GPU Time / Elapsed` 只有 18.3%，其实全是这个搜索循环造成的假象）。已加磁盘缓存
（`results/workloads.json`）。

## 1.5 基线数字

端到端（`profile_gme_int4_gpu.py`，GPU / INT4 / padded / streams=1，warmup 14，repeat 20）：

| tokens | text p50 (ms) | query p50 (ms) |
| ---: | ---: | ---: |
| 32 | 见 §4 表 | 见 §4 表 |
| 512 | ~145.5 | ~145.4 |
| 1024 | ~281.1 | ~281.7 |

语言 IR（`profile_ops.py`，seq=1024，warmup 12，repeat 8）：
**wall 283.0 ms，plugin 侧 profiled 合计 263.8 ms**（差值 ~19 ms 是 host 端提交/
同步开销，全图约 590 个 primitive）。

## 1.6 理论算力标尺

一次 1024-token 前向的 GEMM 计算量：

| 层内算子 | 形状 (M×K×N) | GFLOP @1024 tok |
| --- | --- | ---: |
| q/k/v (horizontal fused) | 1024×1536×2048 | 6.44 |
| o_proj | 1024×1536×1536 | 4.83 |
| gate_proj | 1024×1536×8960 | 28.18 |
| up_proj | 1024×1536×8960 | 28.18 |
| down_proj | 1024×8960×1536 | 28.18 |
| **单层合计** | | **95.8** |
| **28 层合计** | | **2682** |

注意力（QKᵀ + PV，含被 causal mask 浪费的上三角）：单层 6.44 GFLOP，28 层 180 GFLOP。

以此为标尺，后面所有 kernel 效率都可以换算成 TFLOPS。
