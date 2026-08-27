# 第五轮：视觉塔的算子级布局优化，以及补上漏掉的 int8_causal

第四轮（docs/11）把视觉塔的 rank-3 BOOL mask 删掉，image/fused 端到端快了 1.79–2.50x。
本轮回答两个后续问题：

1. **为什么有 `int4_causal` 却没有 `int8_causal`？是无效还是漏了？**
   —— **漏了。** 而且漏在默认精度上：`default_precision` 是 int8 系，一直没吃到 causal 收益。
   补上之后 1024 token 从 277.7 → **262.5 ms（1.058x）**。
2. **mask 之外，视觉塔还有什么可优化？**
   —— 有两笔，都在**导出侧**，都**FP32 逐位相同**，合计再快 **1.09–1.17x**：
   - **`headfirst`**：把「一个 permute + 三个 transpose」合并成**一个 permute**，
     `permute_ref` 从 128 calls 降到 **32 calls**；
   - **`rank3`**：让 ViT 的投影拿到 `[1, L, C]` 而不是 `[L, C]`，
     **解锁 plugin 的 opt-3 INT4 权重重排**（它的第一道 guard 就是 `prim_input_size != 3`），
     `mlp.fc1` 单项快 **28–32%**。

累计效果（视觉 IR，INT4，wall p50，历史阶段 profiling）：

| grid | L | 第四轮之前 | 第四轮 mask-free | **本轮 `_vopt`** | 累计 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1,32,32（448x448） | 1024 | 326.7 | 213.1 | **181.8** | **1.80x** |
| 1,48,48（672x672） | 2304 | 1074.7 | 544.0 | **478.9** | **2.24x** |
| 1,24,210（logo） | 5040 | 4444.4 | 1575.0 | **1439.8** | **3.09x** |

端到端 image（448x448）**563.6 → 256.6 ms（2.20x）**，`1288x728` **4494.4 → 1677.6 ms（2.68x）**。

**本轮同样没有任何 plugin 改动**——`rank3` 是让既有的 opt-3 能命中，不是改 opt-3。

---

## 12.0 起点

| 项目 | 内容 |
| --- | --- |
| runtime | `2026.3.0-22451-8a17657b995-qwen-ttft-optimizations`，只 apply `patches/all_gme_int4_gpu_opts.patch` |
| 基线 | docs/11 的 `_nomask` 视觉 IR + `gme_language_int4_causal.xml` |
| 结果目录 | `results/r5_20260827/` |

---

## 12.1 `int8_causal`：是漏做，不是无效

### 12.1.1 检查结论

`03_finalize.py` 里 INT8 那一行压的是 `lang_fp32`，也就是**非 causal 的**
`gme_language_fp32.xml`；而 causal 的 FP32 源（`gme_language_fp32_causal.xml`）
只进了 INT4 的循环（`round3_variants`）。所以包里从来没有 `gme_language_int8_causal.xml`。

**没有任何实测把它否掉过**——docs/09/10 的否决项列表里也没有它。就是漏了。

而且漏得不巧：当前 `default_precision` 是 int8 系，**默认路径一直没吃到 causal 的收益**。

### 12.1.2 为什么它本来就该有收益

causal 的收益机制与权重精度**正交**：它删掉的是显式 `[B,1,S,S]` mask，让 SDPA 走
`ocl::sdpa::opt` 并跳过上三角。**SDPA 无论权重是 INT4 还是 INT8 都在 f16 下执行**，
所以省下的那一笔不随权重精度变化。docs/09 在 INT4 上测到 SDPA 25.99 → 11.62 ms。

### 12.1.3 实测（1024 token，交错两轮）

| 轮次 | `int8` | `int8_causal` |
| --- | ---: | ---: |
| r1 | 277.3 | 263.0 |
| r2 | 278.1 | 262.0 |
| **均值** | **277.7** | **262.5** |

**−15.2 ms / 1.058x**，与 INT4 上 causal 的 −14.5 ms 量级一致，符合"与权重精度正交"的预期。

### 12.1.4 精度

| precision | text | query | image | fused |
| --- | ---: | ---: | ---: | ---: |
| `int8` | 0.997518 | 0.998764 | 0.989255 | 0.995806 |
| `int8_causal` | **0.997899** | **0.998784** | 0.988600 | 0.995388 |

text/query **略有提升**（+0.0004 / +0.00002），image/fused 在 ±0.0005 内。
docs/09 §9.2.5 已证明 causal 在 FP32 上是恒等变换，这里的小幅波动是重量化噪声。

---

## 12.2 视觉塔的 Q/K/V 布局：一个 permute 顶四个

### 12.2.1 hotspot：mask 修完之后剩下什么

docs/11 的 `_nomask` 视觉 IR，L=1024，按 node 归并：

| calls | ms | % | node |
| ---: | ---: | ---: | --- |
| 32 | 33.81 | 17.1% | SDPA |
| 32 | 33.08 | 16.7% | `mlp.fc1` MatMul |
| 32 | 23.78 | 12.0% | `mlp.fc2` MatMul |
| 32 | 22.52 | 11.4% | `attn.qkv` MatMul |
| **96** | **20.07** | **10.1%** | `attn/aten::transpose` → `permute_ref` |
| **32** | **18.29** | **9.2%** | `attn/aten::permute` → `permute_ref` |
| 130 | 15.10 | 7.6% | DynamicQuantize |
| 32 | 9.67 | 4.9% | `attn.proj` MatMul |

**`permute_ref` 合计 38.4 ms（19.4%），是最大的非 GEMM、非 SDPA 开销。**

### 12.2.2 这 128 次拷贝是怎么来的

upstream 的 `VisionAttention.forward` 走**两趟**数据搬运：

```python
q, k, v = qkv(x).reshape(L, 3, H, D).permute(1, 0, 2, 3).unbind(0)  # 第一趟：拆 QKV
...
q = q.transpose(0, 1).unsqueeze(0)   # 第二趟：[L,H,D] -> [1,H,L,D]，q/k/v 各一次
```

第一趟 32 次（18.29 ms），第二趟 96 次（20.07 ms）。而**目标布局其实一步可达**：

```python
q, k, v = qkv(x).reshape(L, 3, H, D).permute(1, 2, 0, 3).unbind(0)   # -> [3, H, L, D]
```

`permute(1,2,0,3)` 直接落到 `[3,H,L,D]`，三个 `transpose` 全部消失，只剩 `unsqueeze`（免费）。

### 12.2.3 这和 docs/11 §11.5 被否决的那个**不是**一回事

§11.5 的 `fuse_input_transposes` 也删掉了那 96 次拷贝，但**做法是让 SDPA 去读 strided 输入**，
于是把 O(L) 的拷贝换成了 O(L²) 的低效读取，L=5040 上净 **+103.8 ms**，被否决。

本轮的 `headfirst` **仍然显式做转置**，SDPA 拿到的还是连续的 `[1,H,L,D]`；
省掉的只是那趟**冗余的第二次搬运**。实测印证了这个区别：

| | `permute_ref` | **SDPA** |
| --- | ---: | ---: |
| L=1024，upstream → headfirst | 39.27 → **18.38** | 34.72 → **33.24**（不变） |
| L=5040，upstream → headfirst | 183.92 → **86.21** | 747.50 → **748.50**（不变） |

**SDPA 完全不动**——这正是 §11.5 做不到的。

### 12.2.4 rope 要跟着改，但算术不变

`apply_rotary_pos_emb_vision` 内部写死 `cos.unsqueeze(-2)`，假设 seq-first `[L,H,D]`。
换到 `[H,L,D]` 之后改成 `cos.unsqueeze(0)` 广播到 head 维即可：`rotate_half` 只动最后一维，
乘法是逐元素的，**数值完全一致**。实测 rope 本身也没有变慢（8.92 → 9.04 ms），
仍然走 `ocl::rope::opt`。

### 12.2.5 结果（三种布局，同一 runtime）

| grid | L | upstream | `unbind` | **`headfirst`** |
| --- | ---: | ---: | ---: | ---: |
| 1,32,32 | 1024 | 213.1 | 210.8 | **193.2** |
| 1,48,48 | 2304 | 544.0 | 535.2 | **498.7** |
| 1,24,210 | 5040 | 1575.0 | 1552.4 | **1474.4** |

`unbind`（把 `permute(1,0,2,3).unbind(0)` 写成等价的 `unbind(1)`，少一个图节点）
只有 1–1.5% 的小收益，说明**代价在数据搬运本身，不在图节点数**。
`headfirst` 才是真正把一趟搬运整个去掉，**1.07–1.10x，且不随 L 翻盘**。

---

## 12.3 让视觉塔够得着 opt-3：rank-3 投影

### 12.3.1 为什么 opt-3 一直没在视觉塔生效

第二轮的 opt-3（宽 INT4 投影权重重排成 N 连续，docs/06 §6.9）在语言塔上把
`gate_proj` 63.00 → 49.55、`up_proj` 47.43 → 35.24。它的 guard 第一行是：

```cpp
if (prim_input_size != 3 || !weights_transposed)
    return false;
```

**视觉塔的 Linear 输入是 rank-2 `[L, 1280]`**（Qwen2-VL 的 ViT 没有 batch 维），
语言塔是 rank-3 `[B, S, H]`。所以整个视觉塔**仅仅因为 rank 是 2 就被排除在外**。

> docs/11 §11.8 写的是"opt-3 的 `N≥4K` 规则在视觉塔维度上不触发"，**这个说法是错的**。
> 真正的 guard 是 `N >= 4*K`（不是 N≥4096），而 `mlp.fc1` 是 K=1280 / N=5120 = 恰好 4K，
> **能过**。挡住它的是前面那道 rank 检查。

按 `N >= 4*K` 逐层核对，视觉塔里**只有 `mlp.fc1` 合格**：

| 投影 | K | N | 4K | 合格 |
| --- | ---: | ---: | ---: | :---: |
| `attn.qkv` | 1280 | 3840 | 5120 | 否 |
| `attn.proj` | 1280 | 1280 | 5120 | 否 |
| **`mlp.fc1`** | 1280 | **5120** | 5120 | **是** |
| `mlp.fc2` | 5120 | 1280 | 20480 | 否 |
| `merger.mlp.0` | 5120 | 5120 | 20480 | 否 |

所以这是一个**范围明确**的实验，不是普适加速。

### 12.3.2 第一次尝试：每个投影各加一对 Unsqueeze/Squeeze —— 失败

最直接的做法是 `self.fc1(x.unsqueeze(0))...squeeze(0)`。结果：

| exec_type | headfirst | +per-Linear rank3 | Δ |
| --- | ---: | ---: | ---: |
| `jit:gemm:any__i8` | 417.64 | **357.44** | **−60.20** |
| └ 其中 `mlp.fc1` | 141.19 | **102.04** | **−39.16** |
| **`generic_eltwise_ref__f16`** | 0.00 (0 calls) | **89.18 (64 calls)** | **+89.18** |
| **wall p50** | 1482.7 | 1505.3 | **+22.6** |

**opt-3 确实命中了，而且很值钱（−60 ms）**，但那 64 次 Unsqueeze/Squeeze 没有被优化掉，
反而变成了 64 次 `generic_eltwise_ref` 拷贝（+89 ms），把收益全吃了还倒亏。

### 12.3.3 改法：整塔只做一次维度变换

维度变换应该**在塔的入口做一次、出口做一次**，而不是每个投影做一次：

```python
hidden_states = hidden_states.unsqueeze(0)      # patch_embed 之后，一次
for blk in self.blocks:                          # 整个 block 栈都跑 [1, L, C]
    hidden_states = blk(...)
hidden_states = hidden_states.squeeze(0)         # merger 之前，一次
```

`VisionMlp` 和残差加法本来就与形状无关，`VisionAttention` 只需把 `shape[0]` 改成
`shape[-2]`、把 attention 输出直接 reshape 成 `[1, L, C]`（省掉 proj 前的那个 Unsqueeze）。
merger 的第一步是 `view(-1, hidden_size)`，会把前导轴自然抹掉。
**净增两个 metadata-only reshape。**

### 12.3.4 结果

| exec_type | headfirst | **+整塔 rank3** | Δ |
| --- | ---: | ---: | ---: |
| L=1024，`jit:gemm:any__i8` | 90.04 | **79.41** | **−10.63** |
| L=1024，└ `mlp.fc1` | 32.99 | **22.54** | **−10.45（−32%）** |
| L=1024，└ `mlp.fc2` | 23.68 | 23.76 | +0.08 |
| L=1024，└ `attn.qkv` | 22.52 | 22.35 | −0.16 |
| L=1024，└ `attn.proj` | 9.73 | 9.61 | −0.11 |
| L=1024，SDPA | 33.37 | 33.47 | +0.10 |
| L=1024，`permute_ref` | 18.17 | 18.00 | −0.16 |
| L=5040，`jit:gemm:any__i8` | 417.65 | **377.52** | **−40.13** |
| L=5040，└ `mlp.fc1` | 141.45 | **101.90** | **−39.55（−28%）** |
| **`generic_eltwise_ref__f16`** | 0 | **0** | **0** |

**收益完全集中在 `mlp.fc1`，其余投影和其余算子在 ±0.5 ms 内不动**——
和 §12.3.1 按 `N >= 4K` 推出的"只有 fc1 合格"完全吻合。这是这条优化正确命中的最好证据。

| grid | L | headfirst | **+rank3** | Δ |
| --- | ---: | ---: | ---: | ---: |
| 1,32,32 | 1024 | 192.3 | **181.8** | −10.5 |
| 1,48,48 | 2304 | 497.0 | **478.9** | −18.1 |
| 1,24,210 | 5040 | 1475.7 | **1439.8** | −35.9 |

---

## 12.4 等价性与精度

### 12.4.1 FP32 逐位相同

`check_vision_equivalence.py`，两个新变体各跑 5 个 grid（含 trace 没见过的长宽比）：

| 候选 | 5 个 grid 全部 | worst max_abs_diff |
| --- | --- | ---: |
| `gme_visual_fp32_nomask_headfirst.xml` | cos 1.000000000 / row_cos_min 1.000000000 | **0.000e+00** |
| `gme_visual_fp32_nomask_headfirst_rank3.xml` | cos 1.000000000 / row_cos_min 1.000000000 | **0.000e+00** |

两笔改动都只是**重排算子和插入 metadata-only reshape**，没有改任何算术。

### 12.4.2 量化后精度（vs PyTorch FP32 golden，GPU）

| precision | 视觉塔 | text | query | image | fused |
| --- | --- | ---: | ---: | ---: | ---: |
| `int4_causal` | docs/11 之前 | 0.941730 | 0.955173 | 0.945703 | 0.954420 |
| **`int4_causal_vopt`** | 本轮 | 0.941730 | 0.955173 | 0.943164 | 0.955472 |
| `int4_gs64_causal_vopt` | 本轮 | 0.944205 | 0.957576 | 0.943657 | 0.958768 |
| `int8` | 原始 | 0.997518 | 0.998764 | 0.989255 | 0.995806 |
| `int8_causal` | 原始 | 0.997899 | 0.998784 | 0.988600 | 0.995388 |
| **`int8_causal_vopt`** | 本轮 | 0.997899 | 0.998784 | 0.987115 | 0.996531 |

两个值得注意的点：

1. **`int4_causal_vopt` 的 image/fused 与 docs/11 的 `int4_causal_vnomask` 完全相同**
   （0.943164 / 0.955472）。本轮改的是图的拓扑，**权重张量本身一个都没动**，
   NNCF 是按权重张量压缩的，所以连量化结果都一模一样。这是"纯重排"最硬的旁证。
2. text/query 在同一语言 IR 下**逐位相同**，符合预期。

---

## 12.5 端到端

`profile_gme_image.py`，GPU，p50，`int4_causal_vopt`：

| mode | size | K | docs/11 之前 | docs/11 | **本轮** | 累计 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| image | 448x448 | 256 | 563.6 | 288.7 | **256.6** | **2.20x** |
| fused | 448x448 | 256 | 547.9 | 290.2 | **259.8** | **2.11x** |
| image | 644x476 | 391 | 831.7 | 464.6 | **424.5** | **1.96x** |
| image | 672x672 | 576 | 1408.9 | 691.0 | **629.9** | **2.24x** |
| image | 896x896 | 1024 | 2975.2 | 1452.4 | **1351.7** | **2.20x** |
| image | 1288x728 | 1196 | 4494.4 | 1798.3 | **1677.6** | **2.68x** |
| fused | 1288x728 | 1196 | 4479.9 | 1789.6 | **1658.8** | **2.70x** |

分段看 `448x448`：preprocess 14.1 / visual **169.1** / language 72.5。
视觉塔已经从最初占 55% 降到 66%——绝对值降了但占比升了，因为 language 段
（曾被降频拖到 238.5 ms，docs/11 §11.1.1）现在也回到了正常的 72.5 ms。

---

## 12.5.1 当前 final public benchmark（2026-08-27）

上面的 stage-split `e2e_int4_causal_vopt.json` 是第五轮分析口径的 p50；当前最终
部署性能以 `08_benchmark_matrix.py` 的 public API mean 为主，并且同时覆盖三个
正式 precision。完整原始结果位于 [`results/r5_final_20260827/`](../results/r5_final_20260827/)。

| workload | `int8_causal_vopt` | `int4_causal_vopt` | `int4_gs64_causal_vopt` |
| --- | ---: | ---: | ---: |
| text/query, 32 tokens (mean ms/call) | 43.75 / 43.37 | 32.47 / 32.07 | 36.64 / 36.27 |
| text/query, 128 tokens (mean ms/call) | 54.13 / 53.53 | 40.30 / 39.78 | 43.42 / 42.92 |
| text/query, 512 tokens (mean ms/call) | 128.60 / 127.97 | 119.25 / 119.01 | 139.35 / 138.66 |
| text/query, 1024 tokens (mean ms/call) | 260.95 / 262.29 | 225.47 / 225.54 | 240.39 / 239.83 |
| image/fused, 448x448 (mean ms/call) | 285.13 / 284.54 | 262.76 / 257.84 | 269.60 / 268.09 |
| image/fused, 644x476 (mean ms/call) | 446.63 / 447.00 | 426.25 / 425.61 | 440.13 / 439.11 |
| image/fused, 672x672 (mean ms/call) | 675.19 / 672.58 | 643.63 / 640.44 | 649.76 / 693.67 |
| image/fused, 896x896 (mean ms/call) | 1404.58 / 1422.48 | 1343.65 / 1349.16 | 1386.19 / 1387.92 |
| image/fused, 1288x728 (mean ms/call) | 1750.35 / 1757.35 | 1688.70 / 1688.79 | 1707.59 / 1711.78 |
| image/fused, logo (1260 visual tokens, mean ms/call) | 2050.55 / 2049.64 | 1999.73 / 1996.27 | 2017.99 / 2020.06 |

batch=128 tokens、`padded + streams=1` 的 N=8 为：

| precision | text mean ms/call (ms/embedding) | image mean ms/call (ms/embedding) |
| --- | ---: | ---: |
| `int8_causal_vopt` | 255.22 (31.90) | 2084.37 (260.55) |
| `int4_causal_vopt` | 220.27 (27.53) | 1982.36 (247.79) |
| `int4_gs64_causal_vopt` | 234.76 (29.35) | 2027.38 (253.42) |

### 12.5.2 最初 stock 到第五轮 final 的累计收益

最初 stock INT8 的同 workload 数据来自
[`../../artifacts/benchmark_matrix_fresh_r1.json`](../../artifacts/benchmark_matrix_fresh_r1.json)，
无 GME plugin patch、普通 INT8 visual/language、repeat=1；第五轮 final 为
repeat=5，因此这不是严格交错 A/B，只用于保留初始基线和显示累计方向。

| workload | 初始 stock INT8 | 第五轮 `int8_causal_vopt` | reduction | speedup |
| --- | ---: | ---: | ---: | ---: |
| text/query, 32 tokens | 51.55 / 49.41 | 43.75 / 43.37 | 15.1% / 12.2% | 1.178x / 1.139x |
| text/query, 128 tokens | 64.07 / 62.74 | 54.13 / 53.53 | 15.5% / 14.7% | 1.184x / 1.172x |
| text/query, 512 tokens | 144.65 / 139.36 | 128.60 / 127.97 | 11.1% / 8.2% | 1.125x / 1.089x |
| text/query, 1024 tokens | 296.57 / 296.01 | 260.95 / 262.29 | 12.0% / 11.4% | 1.137x / 1.129x |
| image/fused, 448x448 | 476.69 / 468.07 | 285.13 / 284.54 | 40.2% / 39.2% | 1.672x / 1.645x |
| image/fused, 1288x728 | 4654.46 / 4600.58 | 1750.35 / 1757.35 | 62.4% / 61.8% | 2.659x / 2.618x |

最初未优化 INT4 的 document embedding batch baseline 也保留在
`document_embedding_benchmark_before_optimized.json`；它与第五轮最终三种
precision 的四表示对照见 `document_embedding_eval_int8_int4_gpu/REPORT.md`。

## 12.6 推荐配置

| 场景 | precision key | 1024-token text | image 448x448 | image 1288x728 |
| --- | --- | ---: | ---: | ---: |
| **默认 / near-lossless** | **`int8_causal_vopt`** | **260.95** | **285.13** | **1750.35** |
| **低时延** | **`int4_causal_vopt`** | **225.47** | **262.76** | **1688.70** |
| 精度优先 INT4 | `int4_gs64_causal_vopt` | 240.39 | 269.60 | 1707.59 |
| 对照：本轮之前 | `int4_causal_vnomask` | 221.4 | 288.7 | 1798.3 |
| 对照：第四轮之前 | `int4_causal` | 221.4 | 563.6 | 4494.4 |

`03_finalize.py` 的 `default_precision` 已从 `int8_vnomask` 改为 **`int8_causal_vopt`**。

`_vopt` = mask-free + `headfirst` + `rank3`，是
`gme_visual_*_nomask_headfirst_rank3.xml` 的短别名；描述性长名 key 仍然注册着，
用于 A/B。

---

## 12.7 本轮产物

```
gme-Qwen2-VL-2B-Instruct-Running/
  02_export.py        unmasked_vision_attention() 新增 qkv_layout / rank3_linear
                      --vision-qkv-layout {upstream,unbind,headfirst}
                      --vision-rank3-linear
  03_finalize.py      新增 gme_language_int8_causal.xml（§12.1 补漏）
                      压缩三个布局变体；新增 _vopt 短别名；default_precision -> int8_causal_vopt
  ov_models/.../gme_language_int8_causal.xml                    §12.1 采纳
  ov_models/.../gme_visual_{int4,int8}_nomask_headfirst.xml     §12.2 中间阶段
  ov_models/.../gme_visual_{int4,int8}_nomask_headfirst_rank3.xml  §12.3 采纳（_vopt）
  ov_models/.../gme_visual_int4_nomask_unbind.xml               §12.2.5 对照
  artifacts/openvino_validation_round5.json                     §12.4.2

gme-qwen2-vl-int4-hotspot-optimization/
  results/r5_20260827/   本轮全部原始结果
```

**GPU plugin 交付面：本轮没有新增任何 plugin 改动。**

---

## 12.8 仍然开放

| 项 | 现状 | 备注 |
| --- | --- | --- |
| **视觉塔剩下的 32 次 permute（18.0 ms / L=1024）** | `[L,3,H,D] → [3,H,L,D]` 的 QKV 拆分 | 导出侧已到极限：GEMM 只能产出 `[L, N]`，把 L 换到倒数第二轴必然是一次真实转置。要再省需要 plugin 侧让 qkv 直接写成 head-major |
| **`mlp.fc2` / `attn.qkv` / `attn.proj` 的 GEMM** | 都不满足 `N >= 4K`，opt-3 不适用 | docs/09 §9.1.3 已实测把 opt-3 放宽到 `down_proj`（语言塔同类形状）是 **−11.5 ms 负收益**，不宜盲目放宽阈值 |
| **视觉塔 SDPA（L=5040 上 748 ms，占 52%）** | 已是 `ocl::sdpa::opt`，且是全注意力，没有 causal 那样的一半可省 | 唯一结构性杠杆仍是降 `max_image_tokens`（docs/11 §11.6.3） |
| **RMS→DQ 融合（语言塔，~7 ms）** | docs/09 §9.4b 的首选项，仍未做 | 需要给 `DynamicQuantizeKernelOpt` 的 FUSE_RMS 分支补 per-token + 非对称形态，属 plugin 改动 |
| **`max_image_tokens` 的时延/精度对照表** | 仍未测 | 这是 image 路径上收益最大且唯一未量化的一格；必须同时验 Recall@K / nDCG |
| **docs/11 §11.8 的一处错误已修正** | 原文说 opt-3 因 `N≥4K` 不触发 | 实际是因 `prim_input_size != 3`（rank-2）；见 §12.3.1 |

---

## 12.9 复现

```bash
cd gme-Qwen2-VL-2B-Instruct-Running

# 1. 导出：默认的 _nomask（docs/11）+ 本轮两个变体
python 02_export.py --stage visual --unmasked-vision
python 02_export.py --stage visual --unmasked-vision --vision-qkv-layout headfirst
python 02_export.py --stage visual --unmasked-vision --vision-qkv-layout headfirst \
                    --vision-rank3-linear
python 03_finalize.py            # 产出 int8_causal、三个布局变体，注册 _vopt

# 2. 等价性（FP32、CPU、逐位）
cd gme-qwen2-vl-int4-hotspot-optimization
python check_vision_equivalence.py --candidate gme_visual_fp32_nomask_headfirst.xml \
       --output results/r5_20260827/r5_headfirst_equivalence_fp32.json
python check_vision_equivalence.py --candidate gme_visual_fp32_nomask_headfirst_rank3.xml \
       --output results/r5_20260827/r5_rank3_equivalence_fp32.json

# 3. int8_causal（§12.1，交错两轮）
source "$OPENVINO_ROOT/install/setupvars.sh"
for r in 1 2; do for p in int8 int8_causal; do
  python profile_ops.py --precision $p --length 1024 --warmup 8 --repeat 6 \
         --output results/r5_20260827/lang_${p}_r${r}.json
done; done

# 4. 视觉布局 A/B（§12.2、§12.3）
for g in 1,32,32 1,48,48 1,24,210; do
  for ir in gme_visual_int4_nomask gme_visual_int4_nomask_unbind \
            gme_visual_int4_nomask_headfirst gme_visual_int4_nomask_headfirst_rank3; do
    python profile_visual_ops.py --ir $ir.xml --grid $g --warmup 3 --repeat 5 \
           --output "results/r5_20260827/vis_${ir##*nomask}_${g//,/_}.json"
  done
done

# 5. 端到端 + 精度
python profile_gme_image.py --precision int4_causal_vopt --warmup 3 --repeat 5 \
       --output results/r5_20260827/e2e_int4_causal_vopt.json
python ../04_validate_python.py --devices GPU \
       --precisions int8,int8_causal,int8_causal_vopt,int4_causal,int4_causal_vopt,int4_gs64_causal_vopt \
       --output ../artifacts/openvino_validation_round5.json
```

---

## 12.10 一句话总结

**`int8_causal` 只是漏做，补上值 15.2 ms 且落在默认精度上；视觉塔在 mask 之外还有两笔纯
重排的收益——把四趟 Q/K/V 搬运并成一趟（`headfirst`），以及给 ViT 投影加一个 batch 维
让既有的 opt-3 权重重排够得着 `mlp.fc1`（`rank3`）。两笔都 FP32 逐位相同、都不需要动 plugin，
视觉 IR 再快 1.09–1.17x，端到端 image 累计 2.20–2.70x。**
