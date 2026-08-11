# results/ 内容说明

## 性能 profiling（`profile_ops.py`，语言 IR，INT4/GPU，seq=1024）

| 文件 | 构建 | wall (ms) |
| --- | --- | ---: |
| `base_ops_int4_1024.json` | **干净 2026.3.0**（基线） | 283.0 |
| `opt12_ops_int4_1024.json` | opt-1 + opt-2（含实验期残留的 inert 环境变量钩子） | 263.1 |
| `final_ops_int4_1024.json` | **最终交付 patch 精确对应的构建** | 263.2 |
| `ops_int4_1024_dirty_qwen35_build.json` | ⚠️ 开工时的**脏**构建（带 Qwen3.5 5 个 patch），仅作对照，**不是基线** | 279.7 |
| `scale_512.json` / `scale_2048.json` | opt-1 下的序列长度扫描，用于算 GEMM 有效 TFLOPS | 145.2 / 611.4 |

## 端到端 A/B（`bench_ab.sh`）

`ab/{stock,patched}_len{32,512,1024}_r{1,2,3}.json` — 交替 3 轮，每轮 warmup 14 +
repeat 20。汇总表见 `../docs/04_validation_and_results.md` §4.3。

## 数值验证（`dump_outputs.py` / `compare_outputs.py`）

`acc_{int4,int8,int4_gs64}_{stock,patched}.npz` — 固定 7 组输入（seq 1/7/32/129/512/1024
+ 一组右 padding 的 batch=2）下的 embedding 输出。stock = 官方 2026.3.0 wheel。
三种精度全部 **bit-identical**（max abs diff = 0）。

## 生成式回归

| 文件 | 说明 |
| --- | --- |
| `regress_stateful_{stock,patched}.json` | **GenAI-free**，plain OpenVINO 驱动有状态 IR greedy 解码。token id 与首步 logits 逐位相同 → 无回归 |
| `regress_gen_{stock,patched}.json` | ⚠️ 走 `openvino_genai`。两次运行的 GenAI 库本身不同（本地构建 vs wheel），输出差异**不能归因于本轮 patch**，保留仅为记录该陷阱 |

## VTune

原始 result 目录被 `.gitignore` 排除（数百 MB、机器相关），已导出：

| 文件 | 内容 |
| --- | --- |
| `vtune_gpu_summary.txt` | baseline 的 gpu-hotspots summary |
| `vtune_gpu_summary_patched.txt` | patched 的 gpu-hotspots summary |
| `vtune_tasks_baseline.tsv` | baseline 按 computing-task 的完整指标表（占用率/stall/XMX/带宽…） |
| `vtune_tasks_patched.tsv` | patched 同上 |
| `profile_run_vtune_patched.json` | VTune capture 时那次 run 的延迟数字 |

> 两次 capture 的 `Elapsed Time` 不可直接比较：中间给 harness 加了 workload 磁盘缓存。
> VTune 结论只作定性使用，定量以 `ab/` 和 `*_ops_*.json` 为准。

## 其它（本轮之前留下的）

`profile_run.json`、`tuning_{padded,per_sample}_{1,auto}.json`、`gpu-dump-int4-32/`
是早期 batch_mode / streams 调参的产物，与本轮结论无关，保留备查。
