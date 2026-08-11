# 4. 测试与结果

## 4.1 数值正确性：bit-identical

对照组用 **pip 安装的官方 OpenVINO 2026.3.0 wheel**
（`2026.3.0-22451-8a17657b995-releases/2026/3`，正是 patch 的 base commit），
实验组用打了 patch 的本地构建。两边跑同一套固定输入，逐样本比 cosine / max abs diff。

```bash
# 对照组（不 source setupvars，用官方 wheel）
python dump_outputs.py --precision int4 --output results/acc_int4_stock.npz
# 实验组
source "$OPENVINO_ROOT/install/setupvars.sh"
python dump_outputs.py --precision int4 --output results/acc_int4_patched.npz
python compare_outputs.py results/acc_int4_stock.npz results/acc_int4_patched.npz
```

INT4：

| case | cosine | max_abs_diff | rel_l2 |
| --- | ---: | ---: | ---: |
| seq1 | 1.000000000 | 0.000e+00 | 0.000e+00 |
| seq7 | 1.000000000 | 0.000e+00 | 0.000e+00 |
| seq32 | 1.000000000 | 0.000e+00 | 0.000e+00 |
| seq129 | 1.000000000 | 0.000e+00 | 0.000e+00 |
| seq512 | 1.000000000 | 0.000e+00 | 0.000e+00 |
| seq1024 | 1.000000000 | 0.000e+00 | 0.000e+00 |
| batch2_padded（右 padding，第 2 行只有 300 个有效 token） | 1.000000000 | 0.000e+00 | 0.000e+00 |

INT8 和 INT4-gs64 同样全部 **bit-identical**（`worst cosine 1.000000000`，
max abs diff 0）。

> 这是预期的：两个改动都只改数据布局和 kernel 调度，没有任何一处改变算术顺序。

## 4.2 生成式模型回归（KV-cache 通路）

两个 patch 都落在**共享的** SDPA 融合路径上，所以有状态 LLM 必须同样不变。

**先说一个坑**：直接用 `openvino_genai` 做 greedy 对比会得到"输出不一致"的假阳性
—— 本地 `benchmark/openvino.genai` 的构建是 `0c38b54d7e7-qwen-ttft-llm-bench`，
带着用户自己的 llm_bench / rag / qwen3_vl 提交和工作区改动，和 pip wheel
（`bd8d6542e3c`）**本来就不是同一个库**。这个对比同时变了两个变量，不能用。

正确做法是绕开 GenAI，用 `regress_stateful.py` 直接用 plain OpenVINO 驱动有状态
IR 做 greedy 解码（`Qwen3-1.7B-INT4`，GPU，2 个 prompt × 32 步）：

| run | generated_ids 一致 | 首步 logits Σx² (stock) | 首步 logits Σx² (patched) |
| --- | --- | ---: | ---: |
| run0 | ✅ | 3175534.493439 | 3175534.493439 |
| run1 | ✅ | 5604682.734206 | 5604682.734206 |

首步 logits 的平方和与 top-8 值**逐位相同**，32 步 token id 完全相同 →
**KV-cache 通路无回归**。

## 4.3 端到端性能（交替 A/B）

这台机器的 GPU 有明显的双档频率状态（见 `PERFORMANCE_REPORT.md` §6.2.1 记录的
boost / non-boost 双峰），所以不能"先跑完 stock 再跑 patched"。`bench_ab.sh`
按轮次交替，每轮 warmup 14 + repeat 20，共 3 轮，取 p50 的中位数：

| tokens | workload | stock p50 (ms) | patched p50 (ms) | delta | speedup | stock best | patched best |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | text | 32.90 | 32.26 | −0.65 | **1.020x** | 32.85 | 32.17 |
| 32 | query | 32.88 | 32.34 | −0.53 | **1.016x** | 32.76 | 32.21 |
| 512 | text | 146.16 | 139.25 | −6.90 | **1.050x** | 145.69 | 138.77 |
| 512 | query | 145.84 | 138.21 | −7.64 | **1.055x** | 145.38 | 138.00 |
| 1024 | text | 281.10 | 264.52 | −16.57 | **1.063x** | 280.78 | 263.98 |
| 1024 | query | 281.74 | 264.07 | −17.67 | **1.067x** | 281.56 | 262.81 |

三轮之间离散度很小（1024 token 上 stock 280.8–283.1、patched 262.8–265.3），
提速随序列长度单调上升，符合"省掉的是 O(S) 的显存搬运"这一机理。

## 4.4 VTune 复核（patched）

```bash
vtune -collect gpu-hotspots -result-dir results/vtune-gpu-hotspots-int4-1024-patched \
      -- python profile_gme_int4_gpu.py --length 1024 --warmup 14 --repeat 30 --hold-seconds 30
```

| 指标 | baseline | patched |
| --- | ---: | ---: |
| Occupancy（% of peak） | 69.0% | **73.3%** |
| XVE Array Stalled/Idle | 58.5% | **53.4%** |
| `broadcast_gpu_ref` | 0.925 s | **不再出现** |
| `permute_ref` 合计 | 0.574 s | **0.073 s** |
| `sdpa_micro__prefill` | 2.455 s | 2.285 s |

> 注：两次 capture 的 `Elapsed Time`（153.8s → 103.8s）**不能**直接对比 ——
> 中间给 harness 加了 workload 磁盘缓存，省掉了上千次 tokenizer 调用的 CPU 时间。
> `GPU Time` 的绝对值也受 VTune 自身开销影响。因此这一节只作**定性**复核，
> 定量结论以 §4.3 和 §3 的 plugin profiling 为准。

## 4.5 结论

1. **确认了两处 OpenVINO GPU plugin 的图优化缺口**，均为通用问题（不是 GME 特例）：
   - GQA `repeat_kv` 消除只对有状态模型生效，无状态 encoder/embedding 模型漏掉；
   - SDPA 缺少输出 transpose 融合，且 micro 实现对非默认输出 order 的 dispatch 是坏的。
2. 修复后 **1024 token 的 text/query 提速 6.3–6.7%，512 token 5.0–5.5%，
   32 token 1.6–2.0%**，输出 **bit-identical**，生成式模型 **无回归**。
3. **剩余 68% 的时间在 oneDNN `jit:gemm` 里**，实测饱和在 ~15 TFLOPS（int8），
   而同一 kernel 的 f16 路径就有 12.5 TFLOPS —— int8 只快 1.2×，远低于 XMX
   int8 相对 f16 的 2× 理论比。这部分不在 OpenVINO 源码可控范围内，需要单独立项
   （见 §5.1）。
4. 因此对"int4 text/query 明显过慢"的回答是：**慢的主体不是 OpenVINO 的图或
   kernel 选择，而是这颗 iGPU 上 oneDNN 权重压缩 GEMM 的实际吞吐**；OpenVINO 侧
   能拿到的干净收益本轮已经拿到 ~7%，进一步要么动 oneDNN，要么动模型导出（§5.2）。
