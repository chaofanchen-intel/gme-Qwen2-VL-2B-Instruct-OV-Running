# GME hotspot results

## 当前第十轮 Python-only latest package/runtime rerun：`r10_latest/`

这是当前 package 的 GPU/Python 重测，使用同源 custom OpenVINO
core/plugin/Python binding：`2026.3.0-22455-2080527e54b-qwen-ttft-optimizations`。
正式 precision 为 `int8_causal_vopt`、`int4_causal_vopt` 和
`int4_gs64_causal_vopt`，package 默认是 `int8_causal_vopt`。text/query 使用
32/128/512/1024 effective tokens，image/fused 使用 448x448、672x672 和
1288x728，batch 使用 text/query N=1/2/4/8/16/32、image/fused N=1/2/4/8，
本次 formal batch 为 `padded`、`streams=1`，每个 cell repeat=5。visual
`GPU_DYNAMIC_QUANTIZATION_THRESHOLD=0` 只传给 visual IR。

| 文件 | 内容 |
| --- | --- |
| `benchmark_matrix_python_vdq0_formal.json` | 三个 formal precision 的 text/query 与 image/fused 多尺寸矩阵 |
| `benchmark_batch_int8_causal_vopt_vdq0.json` | INT8 causal vopt 的四模式 batch 矩阵 |
| `benchmark_batch_int4_causal_vopt_vdq0.json` | INT4 causal vopt 的四模式 batch 矩阵 |
| `benchmark_batch_int4_gs64_causal_vopt_vdq0.json` | gs64 causal vopt 的四模式 batch 矩阵 |
| `profile_gme_image_int8_causal_vopt_vdq0.json` / `profile_gme_image_int4_causal_vopt_vdq0.json` | 最新 image/fused stage split |
| `baseline_vs_r10_python_comparison.json` | 最初 baseline 到 round-10 的 Python-only arithmetic-mean 对比 |

最初未优化 stock 对照不重新运行，直接使用统一 comparison artifact：
[`baseline_vs_r10_python_comparison.json`](baseline_vs_r10_python_comparison.json)。其来源包括
[`../../artifacts/benchmark_matrix_fresh_r1.json`](../../artifacts/benchmark_matrix_fresh_r1.json)
（初始 INT8 text/image/batch 矩阵）、`base_ops_int4_1024.json`（初始 INT4
language profiling）和 `r4_20260826/vis_base_int4_*.json`（初始 INT4 visual
profiling）。document 四表示的初始 INT8/INT4 baseline 见
[`../../document_embedding_eval_int8_int4_gpu/results/document_embedding_benchmark_before_optimized.json`](../../document_embedding_eval_int8_int4_gpu/results/document_embedding_benchmark_before_optimized.json)。

### Public API baseline -> round-10（arithmetic mean）

| workload | INT8 baseline -> round 10 | INT8 speedup | INT4 baseline -> round 10 | INT4 speedup |
| --- | ---: | ---: | ---: | ---: |
| text/query, 1024 tokens | 296.57/296.01 -> 267.31/266.85 ms | 1.109x/1.109x | 291.443/280.598 -> 228.18/228.96 ms | 1.277x/1.226x |
| image/fused, 448x448 | 476.69/468.07 -> 285.68/284.25 ms | 1.669x/1.647x | pre-vopt 296.37/296.43 -> 248.36/248.38 ms | 1.193x/1.193x |
| image/fused, 1288x728 | 4654.46/4600.58 -> 1759.95/1742.62 ms | 2.645x/2.640x | pre-vopt 1800.29/1795.29 -> 1647.54/1639.14 ms | 1.093x/1.095x |

N=8、128-token、GPU `padded + streams=1` 的 text/query ms/embedding 为
`31.63/33.62`（INT8）、`27.18/29.29`（INT4）、`28.70/32.23`（gs64）。
INT4 image 的 baseline 是最早保存的完整 arithmetic-mean `pre-vopt` 矩阵；绝对最初
masked INT4 image/fused 记录只有 p50，不混入本表。

document embedding 的当前结果见
[`../../document_embedding_eval_int8_int4_gpu/results/benchmark_gme_r10_int4_vopt_vdq0_20260907.json`](../../document_embedding_eval_int8_int4_gpu/results/benchmark_gme_r10_int4_vopt_vdq0_20260907.json)
（fresh INT4）以及 round-9 threshold=0 reference（INT8/gs64）。batch=8 相对初始
document baseline 的 INT4 speedup 为 `page_text/page_image/page_fused = 1.275x/2.806x/2.796x`；
完整四表示对照见 comparison artifact。

MMDocIR `balanced_200` 的当前 FP32/INT8/INT4 close/open 质量结果见
[`../../mmdocir_openvino_eval/REPORT.md`](../../mmdocir_openvino_eval/REPORT.md)。INT4
使用 `ONEDNN_PRIMITIVE_CACHE_CAPACITY=0` 绕过死锁，本轮不使用 MMDocIR 的 INT4
运行时间或 throughput 与 FP32/INT8 做比较。

第五轮 final 的原始结果仍保留在
[`r5_final_20260827/`](r5_final_20260827/)，但只作为历史 provenance，不再作为当前结果。

## 历史第四轮 fresh rerun：`r4_rerun_20260826/`

这是第五轮 final 之前的历史 GPU/Python 重测，使用与 package 同源的 custom
OpenVINO core、GPU plugin 和 Python binding：
`2026.3.0-22451-8a17657b995-qwen-ttft-optimizations`。本轮使用
`int8_vnomask` 和 `int4_causal_vnomask`，即 mask-free visual IR；没有使用官方
wheel runtime。text/query 使用 32/128/512/1024 effective tokens，image/fused
使用 448x448、644x476、672x672、896x896、1288x728 和 logo，batch 使用
N=1/2/4/8/16/32（image 到 N=8），并覆盖 `per_sample`/`padded` 与
`streams=1/auto`。

| 文件 | 内容 |
| --- | --- |
| `benchmark_text_image_matrix.json` | 两个当前 precision 的 text/query 长度和 image/fused 尺寸矩阵，Python/GPU，repeat=5 |
| `benchmark_batch_int8_vnomask.json` | `int8_vnomask` 的完整 text/image batch 矩阵 |
| `benchmark_batch_int4_causal_vnomask.json` | `int4_causal_vnomask` 的完整 text/image batch 矩阵 |
| `profile_gme_image_int8_vnomask.json` | `int8_vnomask` 的 preprocess/visual/language/end-to-end stage split，p50 |
| `profile_gme_image_int4_causal_vnomask.json` | `int4_causal_vnomask` 的 preprocess/visual/language/end-to-end stage split，p50 |
| `benchmark_*.log` | 对应命令的原始 stdout/stderr |
| `bench_images/` | 矩阵使用的确定性 synthetic 图片 |

fresh 矩阵的主要结果应以 **mean ms/call** 和 `ms/embedding` 为主，原始 JSON
保留所有 cell；当前 document embedding 的四表示、batch 1/4/8/14 和质量结果
见 `../../document_embedding_eval_int8_int4_gpu/results/` 下的当前 rerun 文件及其
`REPORT.md`。

`gme_visual_opt123_20260813.json` records the stock-vs-opt1+opt2+opt3 GME
image/fused smoke and correctness comparison. The patched GME visual path remains
runnable; image-only is bit-identical for the tested sizes, while some fused cases
have stable small numerical drift and require deployment-level retrieval validation.

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

## 第三轮（docs/09）

所有 `r3_*` 文件都产自**同一个交付构建**（`2026.3.0-22454-09c4cc4117e`，只 apply
`patches/all_gme_int4_gpu_opts.patch`），且该构建已用 `compare_outputs.py` 证明与第二轮
交付构建逐位相同（`acc_int4_round2.npz` vs `r3_acc_int4_patched.npz` 全 1.000000000）。

| 路径 | 口径 |
| --- | --- |
| `r3_base_ops_int4_1024.json` | gs128 基线复现（wall 239.68 / GEMM 151.90），对应 docs/09 §9.0.2 |
| `r3_base_ops_int4_gs64_1024.json` | **gs64 patched，本项目首次算子级基线**（wall 254.19 / GEMM 166.09），§9.0.3 |
| `r3_STOCK_ops_int4_gs64_1024.json` | gs64 **stock wheel**（无任何 GME patch，wall 284.76）。⚠️ 这份是「忘记 source setupvars.sh」意外采到的，但口径干净、可用作 gs64 的 stock 对照，§9.0.3 |
| `r3_acc_int4_patched.npz`、`r3_acc_int4_gs64_patched.npz` | 两个精度的自洽性参考 npz（后续任何改动都对这两个比） |
| `r3_causal_equivalence_fp32.json` | **causal 改造的 FP32 逐位等价性**：6 个用例（含 `[1024,300]`、`[512,129,7]` 两个不等长右 padding 批）全部 `max_abs_diff = 0`，§9.2.5 |
| `r3_final_int4_r*.json`、`r3_final_int4_causal_r*.json`、`r3_final_int4_gs64_causal_r*.json` | 回退所有实验代码后在交付构建上的最终复核（239.03 / 222.72 / 236.80），§9.6 |
| `../onednn/results/r3_pricing.txt` | oneDNN 定价主表 round-3（新增 prB / gs64 / 分组对称 / bias 共 9 个 case），§9.1 |
| `../../artifacts/openvino_validation_round3_{causal,v5ctrl,sym}.json` | 三组精度评估（vs PyTorch FP32 golden）。**必须三份一起读**：`v5ctrl` 是同源对照组，用来证明 `text_cos_min` 的 ±0.01 抖动来自重量化而非 causal 改造，§9.2.5 |

> ⚠️ 第三轮学到的两条口径教训：
> 1. **`text_cos_min` 分辨不了千分级差异**（同源重量化之间抖 ±0.01）。精度对比应以
>    **query cosine** 为主（同族极差仅 0.0007），或提高样本量。
> 2. **`nncf_fingerprints.json` 只哈希源 IR 的文件名、不含内容**，重新导出同名 FP32 IR
>    后已有 INT4 产物会被误判为「未变化」。`03_finalize.py::compress()` 已加「源比产物新就强制
>    重压」的防护（对已交付产物永不触发）。

## 其它（本轮之前留下的）

`profile_run.json`、`tuning_{padded,per_sample}_{1,auto}.json`、`gpu-dump-int4-32/`
是早期 batch_mode / streams 调参的产物，与本轮结论无关，保留备查。

## 第四轮：image / fused（`r4_20260826/`，docs/11）

与 `r3_*` 同一个交付构建，**本轮同样没有新增任何 plugin 改动**；所有变化都在导出侧的
视觉 IR 上。开工前用 `profile_ops.py --precision int4_causal --length 1024` 复核到
wall p50 221.4 ms，与 docs/09 发布的 222.3 ms 一致。

文件名里的 `1_32_32` 是 `image_grid_thw` 的 `t,h,w`。**视觉塔的序列长度是 merge 前的
patch 行数 `L = t*h*w`，不是视觉 token 数 `K = L/4`**——这是本轮最容易读错的口径。

| 路径 | 口径 |
| --- | --- |
| `vis_base_int4_*.json` | 优化前的视觉 IR（`gme_visual_int4.xml`，带 rank-3 BOOL mask）。六个 grid，L = 1024…5040 |
| `vis_nomaskA_int4_*.json` | **采纳方案**：mask-free 视觉 IR（`gme_visual_int4_nomask.xml`）。与 `vis_base_*` 对比即 docs/11 §11.3.6 的主表 |
| `vis_nomaskAB_int4_*.json` | §11.5 **否决**方案：在 mask-free 之上再把 SDPA 输入 Transpose 融进 primitive（`gme_visual_int4_nomask_qkvfused.xml`） |
| `vis_nomaskA_s1_*.json`、`vis_nomaskAB_s1_*.json` | 同样两个配置在**更早一个 session** 采的数据（`s1` = session 1）。IR 内容与上面两组逐位相同（重导出经 `cmp` 验证），保留是为了给 §11.5 的交叉点提供跨 session 佐证 |
| `ab/vis_{A,AB}_*_r{1,2,3}.json` | §11.5 的**三轮交错 A/B**（A 与 A+B 交替，应对本机 GPU 双档频率）。三轮极差 < 1%，是「A+B 在大图上是真回退」这一结论的依据 |
| `e2e_int4_causal.json`、`e2e_int4_causal_vnomask.json` | `profile_gme_image.py` 的端到端四段拆分（preprocess / visual / language / end-to-end），image 与 fused 各五个尺寸，§11.4 |
| `r4_vision_equivalence_fp32.json` | **mask-free 的 FP32 逐位等价性**：5 个 grid（含 trace 时没见过的长宽比）全部 `max_abs_diff = 0`，§11.3.3 |
| `r4_vision_batch_contract.json` | **单图契约**：三张不同尺寸的图走 batch 路径时，每张图的视觉输出与单独跑逐位相同，§11.3.4 |
| `../../artifacts/openvino_validation_round4_vision.json` | 八个 precision 的精度评估（vs PyTorch FP32 golden），masked / mask-free 成对，§11.6.1 |

> ⚠️ 第四轮学到的两条口径教训：
> 1. **image/fused 的 `language` 段不能紧跟着视觉推理测**。在这块与 CPU 共享 TDP 的 32 EU
>    核显上，一次 300–4000 ms 的重视觉负载会把 GPU 压进低频档，紧接着的语言推理被拖慢
>    2–3 倍（S=281 上 79.6 → 238.5 ms）。`e2e_*.json` 的 `language_p50_ms` 列**只在
>    mask-free 那一份可信**；优化前那一份要对照 `profile_ops.py` 的独立测量读，§11.1.1。
> 2. **验证「单图契约」要比视觉 IR 的输出，不能比最终 embedding**。后者会混进语言 IR
>    `padded` batch mode 本身的行间漂移（实测可达 1e-2），那是 `GmeOpenVINO` docstring
>    早就写明的既有性质，与视觉塔无关，§11.3.4 的口径说明。

补充（回答"mask 真的可以去掉吗"）：`r4_mask_necessity.json` 由
`check_vision_mask_necessity.py` 产出，是 docs/11 §11.2.2–§11.2.4 的原始数据。它同时覆盖
两件事：(a) 三种输入形态（单图 `t=1` / 多行 grid / `t>1`）在**两个 IR** 上的实际行为——
多行 grid 两者都抛异常，`t>1` 时**带 mask 的 IR 输出全 NaN**；(b) 在**官方 PyTorch fp32**
上量化「打包多图 vs 逐图」= row cosine 0.999999702（归约噪声），而「真的跨图串味」=
row cosine 0.058744（比噪声底大 4399 倍）。后者说明这类误用是响的，不是哑的。

## 第五轮：视觉塔布局 + int8_causal（`r5_20260827/`，docs/12）

与 `r4_*` 同一个 runtime，**本轮同样没有新增 plugin 改动**。

| 路径 | 口径 |
| --- | --- |
| `lang_int8_r{1,2}.json`、`lang_int8_causal_r{1,2}.json` | §12.1 的交错两轮：`int8` 277.7 → `int8_causal` **262.5 ms**。补的是一个**漏做**（`03_finalize.py` 只把非 causal 的 FP32 压成 INT8），不是被否掉过的方案 |
| `vis__*.json`、`vis__unbind_*.json`、`vis__headfirst_*.json`、`vis__headfirst_rank3_*.json` | §12.2.5 四种 Q/K/V 布局在三个 grid 上的对比（文件名里空的那段是第四轮的 `_nomask` 基线） |
| `ab_*_r{1,2}.json` | §12.3.2 **失败的第一次尝试**：per-Linear rank-3。opt-3 命中了（GEMM −60.2 ms），但 64 次 `generic_eltwise_ref` 拷贝（+89.2 ms）反噬，净 +22.6 ms。保留是因为它是"为什么必须整塔改一次"的证据 |
| `ab2_*_r{1,2}.json` | §12.3.4 **采纳的做法**：整塔一次 rank-3。收益完全落在 `mlp.fc1`（−28~32%），其余投影 ±0.5 ms 内不动 |
| `r5_headfirst_equivalence_fp32.json`、`r5_rank3_equivalence_fp32.json` | §12.4.1 两个变体各 5 个 grid，全部 `max_abs_diff = 0` |
| `e2e_int4_causal_vopt.json` | §12.5 端到端四段拆分 |
| `../../artifacts/openvino_validation_round5.json` | §12.4.2 六个 precision 的精度评估 |

> ⚠️ 第五轮的一条口径教训：**图上"看起来免费"的 Unsqueeze/Squeeze 不一定免费**。
> §12.3.2 里每个投影外面包一对，本以为是 metadata-only，实际被 lower 成 64 次
> `generic_eltwise_ref` 拷贝，把一笔 −60 ms 的真实 GEMM 收益变成了净 +22.6 ms。
> 判断依据只能是 profiling 里的 `calls_by_exec_type`，不能靠"reshape 应该是免费的"这种直觉。
