# Patches — GME Qwen2-VL INT4 GPU 优化

## Base

所有 patch 均基于 **OpenVINO 官方 release tag `2026.3.0`**：

```
commit 8a17657b995fd3b4a52f8484acfcf2bb61214623 (tag: 2026.3.0)
```

**不包含**本仓库工作区里既有的任何本地改动（`qwen3.5_patch` 的 5 个 patch、
`llm_bench` 相关提交等）。生成方式：

```bash
cd "$OPENVINO_ROOT"
git diff 8a17657b99 -- src/ > "$HOTSPOT_ROOT/patches/all_gme_int4_gpu_opts.patch"
```

已在 `git worktree add --detach <tmp> 2026.3.0` 出来的**纯净 tag checkout** 上验证过
`git apply --check` 通过（单独 apply、顺序 apply、合并 apply 三种方式都验证）。

## 文件

| 文件 | 内容 | 改动量 |
| --- | --- | --- |
| `opt1_sdpa_gqa_broadcast_stateless.patch` | GQA `repeat_kv` 的 `Unsqueeze→Broadcast→Reshape` 融合支持无状态模型 | 1 file, +13 −5 |
| `opt2_sdpa_output_transpose_fusion.patch` | 新增 `TransposeSDPATransposeMatcher`；修复 `sdpa_gen_micro` 对非默认 `output_transpose_order` 的 dispatch bug | 3 files, +165 −14 |
| `all_gme_int4_gpu_opts.patch` | 以上两者合并 | 4 files, +182 −14 |

两个 patch 相互独立，可单独 apply；但 **opt-2 单独用时依赖它自带的
`sdpa_gen_micro` 修复**（否则一旦融合触发就会 GPU 挂死），该修复已包含在 opt-2 内。

## 应用

```bash
cd "$OPENVINO_ROOT"
git checkout 2026.3.0            # 或任何 src/ 与该 tag 一致的 checkout
git apply "$HOTSPOT_ROOT/patches/all_gme_int4_gpu_opts.patch"

cmake --build <build-dir> --target openvino_intel_gpu_plugin -j
cmake --install <build-dir>
```

只需重编 GPU plugin，不需要全量重编。

## 影响面

| 项 | 结论 |
| --- | --- |
| 数值 | **bit-identical**（INT4 / INT8 / INT4-gs64，含 padded batch，max abs diff = 0） |
| 有状态生成式模型 | 无回归（`Qwen3-1.7B-INT4` greedy 32 步，token id + 首步 logits 逐位相同） |
| 适用范围 | 通用。opt-1 惠及任何**无状态 GQA** 模型（embedding / encoder / reranker / 不带 KV cache 编译的模型）；opt-2 惠及任何以 `SDPA → Transpose → Reshape → o_proj` 收尾的 attention 块 |
| 平台 | 仅 Intel GPU plugin；CPU / NPU 不受影响 |
| 依赖 | 无。不需要 `ENABLE_DEBUG_CAPS`，不引入新的 config option 或环境变量 |

## 收益（Intel Panther Lake-P iGPU，gme-Qwen2-VL-2B-Instruct，INT4，GPU）

| tokens | text | query |
| ---: | ---: | ---: |
| 32 | 32.90 → 32.26 ms (1.020x) | 32.88 → 32.34 ms (1.016x) |
| 512 | 146.16 → 139.25 ms (1.050x) | 145.84 → 138.21 ms (1.055x) |
| 1024 | 281.10 → 264.52 ms (**1.063x**) | 281.74 → 264.07 ms (**1.067x**) |

详见 `../docs/04_validation_and_results.md`。

## 上游建议

## Qwen3-VL-Embedding-2B 适用性复测

这两个 patch 另外在 `Qwen3-VL-Embedding-2B` 上做了 A/B。结果不能直接沿用 GME 的
收益结论：

| 路径 | Qwen3 结果 |
|---|---|
| INT8 text 512 | 0.997x，基本无收益 |
| INT8 text 1024 | 1.020x，小幅收益 |
| INT4 text 512 | 0.991x，小幅回归 |
| INT4 text 1024 | 1.013x，小幅收益 |
| image/fused | optimized plugin 在 64x64 后触发 `CL_OUT_OF_RESOURCES`，出现 `FullyConnectedCompressed` 资源/实现选择失败 |

因此 `all_gme_int4_gpu_opts.patch` **不应直接应用到 Qwen3-VL-Embedding-2B 生产环境**。
完整 stock/optimized JSON 和复测说明见
[`results/qwen3_ab_text/REPORT.md`](../results/qwen3_ab_text/REPORT.md)。

两处都是通用缺陷，建议分别提 PR：

1. **opt-1** — `UnsqueezeBroadcastReshapeSDPAFusion` 的 pattern 把 K/V producer 限定
   为 `KVCache`，导致无状态 GQA 模型漏掉 `repeat_kv` 消除。融合的全部前提条件在
   callback 里已有检查，producer 类型不携带信息，可安全放开。
2. **opt-2** — 两件事：(a) `TransposeFusion` 缺少 SDPA 输出侧的 matcher（MatMul 两侧
   都有）；(b) `sdpa_gen_micro` 按位置读输出 layout 的 batch/num_heads，对非默认
   `output_transpose_order` 是错的 —— 这是个**已存在但此前无人触发**的 bug，
   (a) 一旦合入就会立刻暴露它。建议 (b) 可以先独立合入。
