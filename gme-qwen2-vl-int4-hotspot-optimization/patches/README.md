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
| `opt2_sdpa_output_transpose_fusion.patch` | 新增 `TransposeSDPATransposeMatcher`；修复 `sdpa_gen_micro` 的 dispatch 与 canonicalization；修复 `calc_output_layout()` 漏用 `output_transpose_order` | 4 files, +257 −11 |
| `opt3_int4_wide_fc_ncontiguous_weights.patch` | 宽 INT4 投影（`N ≥ 4K` 且 ≥128 行）的权重重排成 N 连续，matmul 描述符改用 `abc` | 3 files, +115 −3 |
| `all_gme_int4_gpu_opts.patch` | 以上三者合并 | 8 files, +385 −19 |

三个 patch 相互独立，可单独 apply；但 **opt-2 单独用时依赖它自带的运行时修复**
（否则一旦融合触发就会 GPU 挂死），这些修复已包含在 opt-2 内。

`superseded/` 下是 2026-08-13 版本的 opt-2 / 合并 patch，它们在 Qwen3-VL-Embedding-2B
的 vision 路径上会导致 `CL_OUT_OF_RESOURCES`，**不要再使用**；根因和修复见
[`../docs/07_qwen3vl_regression_rootcause.md`](../docs/07_qwen3vl_regression_rootcause.md)。

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
| 数值（opt-1 + opt-2） | **bit-identical**（INT4 / INT8 / INT4-gs64，含 padded batch，max abs diff = 0） |
| 数值（加上 opt-3） | 七个用例里六个仍 bit-identical；padded batch cos 0.99950 / max_abs 1.19e-2。opt-3 换的是权重 layout 而非算术，`layout_accuracy_probe` 证明 oneDNN 侧 `cab`/`abc` 数值中性，重排本身逐 nibble 位正确；残留漂移来自 gemm 分块顺序，比 INT4 自身的量化噪声底（cos 0.918–0.989）低约一个数量级 |
| 有状态生成式模型 | 无回归（`Qwen3-1.7B-INT4` greedy 32 步，token id + 首步 logits SHA + top-8 逐位相同） |
| 其他模型（`Qwen3-VL-Embedding-2B`） | 无影响。text / image / fused 共 28 个 workload（INT8 + INT4）与 stock **逐位相同**，见 [`../docs/07_qwen3vl_regression_rootcause.md`](../docs/07_qwen3vl_regression_rootcause.md) |
| 适用范围 | opt-1 惠及任何**无状态 GQA** 模型；opt-2 惠及以 `SDPA → Transpose → Reshape → o_proj` 收尾、且 **SDPA 输出为 rank-4** 的 attention 块（rank-3 的 headless attention——Qwen-VL 系列 vision tower——被 `supports_micro_sdpa()` 强制走 `sdpa_opt`，该 kernel 不认 `output_transpose_order`，故 matcher 主动跳过）；opt-3 惠及任何 **group 量化 INT4、`N ≥ 4K` 的宽投影**（典型是 MLP 的 gate/up），且只在 ≥128 行的 prefill 型负载上生效 |
| 显存 | **opt-3 会给命中的权重多留一份副本**。GME 上是 56 × 6.56 MiB ≈ **367 MiB** 常驻。行数不达标时不重排、不额外分配 |
| 平台 | 仅 Intel GPU plugin；CPU / NPU 不受影响 |
| 依赖 | 无。不需要 `ENABLE_DEBUG_CAPS`，不引入新的 config option 或环境变量 |

## 收益（Intel Panther Lake-P iGPU，gme-Qwen2-VL-2B-Instruct，INT4，GPU）

`bench_ab.sh` 交替 A/B，stock 2026.3 wheel vs 三个 patch 全开，3 轮取中位：

| tokens | text | query |
| ---: | ---: | ---: |
| 32 | 32.78 → 32.19 ms (1.018x) | 32.85 → 32.31 ms (1.017x) |
| 128 | 46.56 → 40.20 ms (**1.158x**) | 46.24 → 40.17 ms (**1.151x**) |
| 512 | 146.55 → 123.81 ms (**1.184x**) | 145.49 → 123.31 ms (**1.180x**) |
| 1024 | 281.38 → 240.01 ms (**1.172x**) | 280.18 → 239.53 ms (**1.170x**) |

32 token 上收益仍只有第一轮的 ~1.02x：opt-3 的 128 行门槛在那里不触发，
这是**刻意的**——低于该行数时新 layout 反而更慢（docs/06 §6.9.4）。

2026-08-17 的 Qwen3-VL 修复（docs/07）之后复测，收益在噪声范围内不变：
128 → 1.164x / 1.156x，512 → 1.186x / 1.182x，1024 → 1.170x / 1.167x。

详见 `../docs/04_validation_and_results.md` 与 `../docs/06_round2_onednn_investigation.md` §6.9。

## Qwen3-VL-Embedding-2B 适用性复测

三个 patch 都在 `Qwen3-VL-Embedding-2B` 上做了适用性复核。opt-1 / opt-2 的原始
交替 A/B，以及 opt-3 的增量测量和三项组合的视觉 smoke test，见
[`results/qwen3_ab_text/REPORT.md`](../results/qwen3_ab_text/REPORT.md)。结果不能直接
沿用 GME 的收益结论：

| 路径 | Qwen3 结果 |
|---|---|
| INT8 text 512 | 0.997x，基本无收益 |
| INT8 text 1024 | 1.020x，小幅收益 |
| INT4 text 512 | 0.991x，小幅回归 |
| INT4 text 1024 | 1.013x，小幅收益 |
| image/fused（2026-08-13 版 opt-2） | optimized plugin 在 64x64 后触发 `CL_OUT_OF_RESOURCES` / `FullyConnectedCompressed` 实现选择失败 |
| opt-3 增量（INT8 text 1024） | 359.73 → 359.34 ms，1.0011x，基本无收益 |
| opt-3 增量（INT4 text 1024） | 351.04 → 350.76 ms，1.0008x，基本无收益 |

opt-3 对 Qwen3 language text 不触发：`hidden=2048`、`intermediate=6144`，
`N/K=3 < 4`，不满足 `N >= 4K`。

**2026-08-17 更新**：上表 image/fused 那一行的崩溃已定位并修复，根因是 opt-2 的
matcher 会覆盖已融合的输入 transpose order，加上三处 rank-3 / 静态 shape 路径的
既有缺陷；详见 [`../docs/07_qwen3vl_regression_rootcause.md`](../docs/07_qwen3vl_regression_rootcause.md)。
修复后 `Qwen3-VL-Embedding-2B` 的 text / image / fused 共 28 个 workload
（INT8 + INT4）与 stock **逐位相同**——vision tower 是 rank-3 SDPA，matcher 现在会
主动跳过它。也就是说这三个 patch 现在对 Qwen3-VL **既不损害也不加速**（text 侧的
±2% 属于测量噪声），可以安全共存于同一个 plugin 里。

完整 stock/optimized JSON 和三项复测说明见
[`results/qwen3_ab_text/REPORT.md`](../results/qwen3_ab_text/REPORT.md)。

## 上游建议

前两处都是通用缺陷，建议分别提 PR：

1. **opt-1** — `UnsqueezeBroadcastReshapeSDPAFusion` 的 pattern 把 K/V producer 限定
   为 `KVCache`，导致无状态 GQA 模型漏掉 `repeat_kv` 消除。融合的全部前提条件在
   callback 里已有检查，producer 类型不携带信息，可安全放开。
2. **opt-2** — 两件事：(a) `TransposeFusion` 缺少 SDPA 输出侧的 matcher（MatMul 两侧
   都有）；(b) `sdpa_gen_micro` 按位置读输出 layout 的 batch/num_heads，对非默认
   `output_transpose_order` 是错的 —— 这是个**已存在但此前无人触发**的 bug，
   (a) 一旦合入就会立刻暴露它。建议 (b) 可以先独立合入。
3. **opt-3** — 提 PR 前需要先谈清两点，不建议按现在的形态直接上游：
   - **阈值是实测拟合的，不是推导出来的。** `N ≥ 4K` 和 `≥128 行` 两个常数来自
     Panther Lake iGPU 上的逐 matmul 计时（docs/06 §6.9.3–§6.9.4），换 GPU 代次
     或换 driver 后不保证还在同一位置。上游更合适的形态是把它做成
     `jit:gemm` 侧的 layout 偏好查询，而不是 plugin 侧硬编码常数。
   - **附带的 `oiyx → ioyx` INT4 reorder 是通用能力，可以先独立合入。**
     `reorder_weights_int4` 此前不支持这条转换路径；补上它（kernel + selector
     的三处）本身不改变任何现有行为，已用 host readback 验证逐 nibble 位正确。
