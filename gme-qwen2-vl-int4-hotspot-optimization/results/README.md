# results/ 内容说明

## 性能 profiling（`profile_ops.py`，语言 IR，INT4/GPU，seq=1024）

| 文件 | 构建 | wall (ms) |
| --- | --- | ---: |
| `base_ops_int4_1024.json` | **干净 2026.3.0**（基线） | 283.0 |
| `opt12_ops_int4_1024.json` | opt-1 + opt-2（含实验期残留的 inert 环境变量钩子） | 263.1 |
| `final_ops_int4_1024.json` | **最终交付 patch 精确对应的构建** | 263.2 |
| `ops_int4_1024_dirty_qwen35_build.json` | ⚠️ 开工时的**脏**构建（带 Qwen3.5 5 个 patch），仅作对照，**不是基线** | 279.7 |
| `scale_512.json` / `scale_2048.json` | opt-1 下的序列长度扫描，用于算 GEMM 有效 TFLOPS | 145.2 / 611.4 |
| `r2_final_ops_1024.json` | **第二轮最终交付构建**（opt-1 + opt-2 + opt-3） | **238.2** |

### 第二轮过程数据（`docs/06` §6.9）

| 文件 | 是什么 | wall (ms) | 可信度 |
| --- | --- | ---: | --- |
| `round2_ops_recheck.json` | 第二轮的干净算子分布（§6.0，权威基线） | 263.7 | ✅ |
| `round2_ops_int4_1024.json` | 同构建但被干扰的一次采集 | 318.2 | ❌ **作废**，见 §6.7 |
| `round2_gatedmlp_ops.json` | `FuseGatedMLP` 负结果复测 | 397.0 | ✅ |
| `r2_abc_control.json` | opt-3 实验的对照组（opt1+2） | 264.1 | ✅ |
| `r2_abc_forced.json` | **只改 descriptor**、无条件全 `abc` | 246.1 | ⚠️ 该实验让 kernel 用新 stride 读旧 buffer，**结果无效** |
| `r2_abc_heur.json` | 同上，加形状启发式 | 236.2 | ⚠️ 同上，**已作废** |
| `r2_abc_any.json` | `weights_fmt = any`，交给 oneDNN 自选 | 251.4 | ✅ 否决依据 |
| `r2_ncontig_impl.json` | 真重排 + 第一版 `N ≥ K` 规则 | 268.6 | ✅ 否决依据（比不改还慢） |
| `r2_ncontig_wide.json` | 真重排 + `N ≥ 4K`，**尚无行数门槛** | 240.5 | ✅ 中间态 |

`round2_{reverify,patched_recheck,verify_patched}.json` 是端到端 harness
（`profile_gme_int4_gpu.py`）的中间基准，口径与 `ab/` 相同。

## 端到端 A/B（`bench_ab.sh`）

`ab/{stock,patched}_len{32,128,512,1024}_r{1,2,3}.json` — 交替 3 轮，每轮
warmup 14 + repeat 20。**这批是第二轮重测的**，`patched` = 三个 patch 全开。
汇总表见 `../docs/04_validation_and_results.md` §4.3。

> `len128` 是第二轮新加的：opt-3 的行数门槛就落在 64–128 之间，
> 需要一个门槛刚好触发的点。

## 数值验证（`dump_outputs.py` / `compare_outputs.py`）

固定 7 组输入（seq 1/7/32/129/512/1024 + 一组右 padding 的 batch=2）下的
embedding 输出，stock = 官方 2026.3.0 wheel。

| 文件 | 对应构建 | 结果 |
| --- | --- | --- |
| `acc_{int4,int8,int4_gs64}_{stock,patched}.npz` | opt-1 + opt-2 | 三种精度全部 **bit-identical**（max abs diff = 0） |
| `acc_int4_round2.npz` | **opt-1 + opt-2 + opt-3（最终）** | 7 个用例中 6 个 bit-identical；`batch2_padded` cos 0.999497 / max_abs 1.19e-2，见 §4.1b |
| `acc_int4_ncontig.npz` | 第一版 `N ≥ K` 规则（已否决） | 最差 cos 0.9948，仅作记录 |
| `acc_int4_ncontig_wide.npz` | `N ≥ 4K`、无行数门槛的中间态 | 仅作记录 |

## 生成式回归

| 文件 | 说明 |
| --- | --- |
| `regress_stateful_{stock,patched}.json` | **GenAI-free**，plain OpenVINO 驱动有状态 IR greedy 解码。token id 与首步 logits 逐位相同 → 无回归（第一轮） |
| `r2_regress_stateful_patched.json` | 同上，第二轮加 opt-3 后重跑。与 `regress_stateful_stock.json` 逐字段比较，`prompt_ids` / `generated_ids` / `first_step_logits_sha` / `first_step_logits_top8` **四项全同** |
| `regress_gen_{stock,patched}.json`、`r2_regress_gen_patched.json` | ⚠️ 走 `openvino_genai`。两次运行的 GenAI 库本身不同（本地构建 vs wheel），文本差异**不能归因于本轮 patch**。两份 patched 结果**逐字节相同** → 也证明 opt-3 没有引入新偏移。保留仅为记录该陷阱，判据以 `regress_stateful` 为准 |

## VTune

原始 result 目录被 `.gitignore` 排除（数百 MB、机器相关），已导出：

| 文件 | 内容 |
| --- | --- |
| `vtune_gpu_summary.txt` | 第一轮 baseline 的 gpu-hotspots summary |
| `vtune_gpu_summary_patched.txt` | 第一轮 patched 的 gpu-hotspots summary |
| `vtune_tasks_baseline.tsv` | baseline 按 computing-task 的完整指标表（占用率/stall/XMX/带宽…） |
| `vtune_tasks_patched.tsv` | patched 同上 |
| `profile_run_vtune_patched.json` | VTune capture 时那次 run 的延迟数字 |
| `vtune_r2_summary.txt` / `vtune_r2_task_table.txt` / `vtune_r2_tasks.csv` | **第二轮** patched 采集（`docs/06` §6.1 主表） |
| `vtune_r2_summary_STOCK.txt` / `vtune_r2_task_table_STOCK.txt` / `vtune_r2_tasks_STOCK.csv` | 第二轮 stock 复测 |
| `vtune_r2_collect{,_patched}.log` | 两次采集的命令行日志 |

> 两次 capture 的 `Elapsed Time` 不可直接比较：中间给 harness 加了 workload 磁盘缓存。
> VTune 结论只作定性使用，定量以 `ab/` 和 `*_ops_*.json` 为准。

带 `-bad` / `-STOCK` 后缀的 result 目录是**踩坑记录**：`-bad` 是没 source
`setupvars.sh` 时采到的（测的其实是官方 wheel），保留是为了对得上 §6.7 的坑表，
**不要拿来作结论**。

## 跨模型

| 路径 | 内容 |
| --- | --- |
| `qwen3_ab_text/`、`qwen3_{stock,optimized}_{text,detailed}/` | opt-1 / opt-2 在 `Qwen3-VL-Embedding-2B` 上的适用性 A/B，结论见 `qwen3_ab_text/REPORT.md`（**opt-3 未在该模型上复测**） |
| `cross_model_per_call/` | 跨模型的 per-call 延迟采样 |

## 其它（本轮之前留下的）

`profile_run.json`、`tuning_{padded,per_sample}_{1,auto}.json`、`gpu-dump-int4-32/`
是早期 batch_mode / streams 调参的产物，与本轮结论无关，保留备查。
