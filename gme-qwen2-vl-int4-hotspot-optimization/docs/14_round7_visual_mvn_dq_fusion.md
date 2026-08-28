# 第七轮：视觉 LayerNorm（MVN + affine）融合进 DynamicQuantize

第六轮最终视觉 IR 的剩余算子中，LayerNorm 对应的 `mvn_gpu_bfyx_opt` 在 L=1024
耗时 9.04 ms，在 L=5040 耗时 49.63 ms。runtime graph 显示每个 Transformer block
的 LayerNorm 已经变成单个 MVN，并把 gamma/beta 吸收为两个 fused eltwise，随后立即接
per-token symmetric INT8 DynamicQuantize：

```text
raw f16 -> MVN -> Multiply(gamma) -> Add(beta) -> DynamicQuantize -> compressed FC
```

本轮把这条链合并为一个 DynamicQuantize kernel，消除 MVN 写出并由 DQ 读回的 f16
中间张量。实现严格限定 GME 视觉塔的 hidden=1280，不改变默认 threshold 配置；调用方
通过 visual-only threshold=0 显式启用。

最终结果（arithmetic mean）：INT4 视觉 IR **1.036–1.053x**，public image/fused 端到端
**1.027–1.039x**；MVN 65→2 calls。四个 grid、真实 public image/fused、standard golden
validation 和 compiled-model cache reload 全部 `max_abs=0`。INT8 visual 是
group=128 + precomputed-reduction DQ，不在本 patch 的适用范围内。

---

## 14.0 起点

| 项 | 内容 |
| --- | --- |
| runtime | OpenVINO `2026.3.0` + round-6 combined patch |
| 模型 | `gme_visual_int4_nomask_headfirst_rank3.xml` |
| 设备 | Intel Panther Lake-P iGPU |
| 结果目录 | `results/r7_20260828/` |
| 增量 patch | `patches/opt5_visual_mvn_dq_fusion.patch` |

第六轮最终视觉 profile：

| exec type | L=1024 | L=5040 |
| --- | ---: | ---: |
| GEMM | 78.66 ms | 383.89 ms |
| SDPA | 32.99 ms | 752.40 ms |
| DynamicQuantize | 15.16 ms | 107.37 ms |
| QKV permute | 18.15 ms | 88.50 ms |
| **MVN / LayerNorm** | **9.04 ms** | **49.63 ms** |
| RoPE | 9.17 ms | 48.78 ms |
| wall mean | 183.1 ms | 1436.4 ms |

---

## 14.1 真实图结构

在 `prepare_primitive_fusing` 结束时，视觉 block 的 MVN 具有三个 dependencies：

1. raw activation；
2. gamma constant；
3. beta constant。

并且恰有两个 fused descriptors：

| fused op | mode | inputs | external dependency |
| --- | --- | --- | --- |
| 0 | `eltwise_mode::prod` | `ORIGINAL(0), EXTERNAL(1)` | gamma |
| 1 | `eltwise_mode::sum` | `INTERNAL(0), EXTERNAL(2)` | beta |

所有 input/output/gamma/beta 都是 planar f16；输入逻辑形状是 `[1, L, 1280]`，物理
canonical layout 为 `B=1, F=L, Y=1280, X=1`。DQ 是 per-token symmetric i8，
`group_sizes=[1,1,UINT64_MAX]`，无 precomputed reduction。

matcher 不依赖 friendly name，只接受上述结构，并额外要求：

- threshold 显式为 0；
- MVN 只有一个 user，DQ 没有其他 fused primitive；
- last-axis-only variance normalization；
- hidden 精确为 1280；
- gamma/beta 是无 padding 的 f16 constant，各 1280 elements；
- device `max_work_group_size >= 256`；
- per-token symmetric i8、identity scale order。

任何 guard 不满足都保留原图。

---

## 14.2 kernel 实现

fused kernel 使用和 `mvn_gpu_bfyx_opt` 相同的 dispatch：

```text
GWS = [256, 1, L]
LWS = [256, 1, 1]
items per work-item = 5
index = row_offset + local_id + item * 256
```

每个 token row：

1. 三次读取 raw activation，分别用于 mean、variance、normalized output；
2. mean 使用 `work_group_reduce_add(sum) / 1280`；
3. variance 使用相同 `fma(diff, diff, acc)`、`work_group_reduce_add` 和 epsilon mode；
4. `(x - mean) * inv_std` 先舍入到 f16；
5. `normalized * gamma` 再舍入到 f16；
6. `scaled + beta` 再舍入到 f16；
7. 对 shifted f16 做 abs-max，沿用原 DQ 的 f16 scale 和 `convert_char_rte`。

这保留了原 MVN 输出、两个 fused eltwise 输出和 DQ 输入之间的三个 f16 边界。

### 14.2.1 第一次失败与根因

第一版给整个 DynamicQuantize kernel 无条件保留 `REQD_SUB_GROUP_SIZE(16)`。它的
workgroup 大小、元素归属、公式看似都与 MVN 一致，但 `work_group_reduce_add` 的内部
加法树改变；误差前十几层通常被 i8 rounding 掩盖，随后跨过量化边界并层间放大：

| grid | 最终 visual output cosine | max abs |
| --- | ---: | ---: |
| 1,32,32 | 0.99549 | 17.35 |
| 1,18,162 | 0.97056 | 28.26 |
| 1,48,48 | 0.96716 | 24.66 |
| 1,24,210 | 0.96251 | 28.99 |

额外输出定位表明 block 0 的 qkv/fc1 都逐位相同，第一处微小差异直到 block 14 qkv
才出现。这排除了 gamma/beta 索引错误，指向 reduction tree。

修复是仅在 `FUSE_MVN` 分支取消 `REQD_SUB_GROUP_SIZE(16)`，让编译器采用与 standalone
MVN 相同的 subgroup geometry；RMS 和普通 DQ 分支仍保留原 SIMD16 要求。修复后所有
测试逐位相同。

### 14.2.2 这条逐位相同依赖编译器启发式，不是构造上的保证

上面的修复方式必须记清楚它的性质：**它没有把 subgroup 几何钉死，而是把选择权交还给了
IGC**，指望 IGC 给融合 kernel 选到和 standalone `mvn_gpu_bfyx_opt` 一样的 SIMD 宽度，
从而让两边 `work_group_reduce_add` 的内部加法树一致。核对过 standalone MVN 确实是
`reqd_work_group_size(LWS,1,1)`、**没有** `REQD_SUB_GROUP_SIZE`、同样用
`work_group_reduce_add` / `work_group_broadcast`，所以今天两者对齐。

但 IGC 的 SIMD8/16/32 选择是按**寄存器压力**决定的启发式，而融合 kernel 的 body 比
standalone MVN 重（多了 `shifted[5]`、gamma/beta 两路加载和量化循环）。因此这条逐位
相同在下列情况下可能**静默失效**：

- IGC / GPU 驱动升级；
- 换到寄存器预算不同的 GPU；
- 今后再修改 `FUSE_MVN` 分支的 kernel body。

**对比 opt-4 的 RMS 融合：那边是构造上确定的**——它保留 `REQD_SUB_GROUP_SIZE(SIMD)`，
并手写复刻了 `sub_group_reduce_add` + 16-slot SLM halving tree（§13.2.3）。两个融合的
健壮性等级不一样，这一点在扩展或移植时必须知道。

> **交付要求：`check_visual_mvn_dq_equivalence.py` 是驱动/runtime 变更后的必过闸门。**
> 任何 OpenVINO 版本升级、GPU 驱动升级或换设备之后，必须重跑它并确认四个 grid 全部
> `max_abs=0`，才能继续使用 `visual_dynamic_quantization_threshold=0`。若要彻底消除这个
> 依赖，正确做法是像 RMS 分支那样显式 pin 住 subgroup size 并手写 reduction tree。

---

## 14.3 覆盖与正确性

### 14.3.1 算子覆盖

L=1024 的实际 profiling calls：

| | control | fused |
| --- | ---: | ---: |
| MVN | 65 | **2** |
| DynamicQuantize | 130 | 130 |
| compressed FC | 130 | 130 |

剩余两个 MVN 是：

- `visual.blocks.0.norm1`：首层特殊拓扑；
- `visual.merger.ln_q`：merger 路径不直接接目标 DQ。

继续覆盖它们的上限约 0.3 ms，不值得扩大 matcher。

### 14.3.2 strict control/fused

control 与 fused 来自同一源码构建；control 只关闭 MVN matcher。两者都设置
`GPU_DYNAMIC_QUANTIZATION_THRESHOLD=0`，因此差异只来自本融合。

| grid | raw patches L | merged K | max abs | exact |
| --- | ---: | ---: | ---: | :---: |
| 1,32,32 | 1024 | 256 | 0 | yes |
| 1,18,162 | 2916 | 729 | 0 | yes |
| 1,48,48 | 2304 | 576 | 0 | yes |
| 1,24,210 | 5040 | 1260 | 0 | yes |

真实 public API、显式 cap=768 的 `image` 和 `fused` embedding 也都是 `max_abs=0`。

### 14.3.2b `threshold=0` 的逐位相同结论适用范围（复核补充）

上表比的是"同一 threshold=0 下，融合 on/off"，隔离得很干净。但**用户实际感知到的**是
"设不设 `visual_dynamic_quantization_threshold=0`"，也就是 `default(64)` vs `0`——这两者
之间除了融合，还差一个 **DQ 阈值旁路**。用出厂二进制补测三档（`default` / `threshold=1`
/ `threshold=0`，其中 `1` 关旁路但也关融合，用于隔离）：

| grid | L | `default` vs `t=1`（旁路差异） | **`t=1` vs `t=0`（纯融合差异）** | DQ calls default / t1 / t0 |
| --- | ---: | ---: | ---: | --- |
| 1,16,16 | 256 | 1.500e+00 | **0.000e+00** | 128 / 130 / 130 |
| 1,32,32 | 1024 | 0.000e+00 | **0.000e+00** | 128 / 130 / 130 |

两个结论：

1. **融合本身在 L=256 上也是精确的**，比 §14.3.2 只测 L≥1024 的覆盖更强；
2. L=256 上 `default` 与 `t=0` 的差异**全部来自阈值旁路**（default 少执行 2 个 DQ），
   与本轮的 MVN 融合无关，是语言侧 §13.2.3 已记录的同一机制。

因此**"设 `visual_dynamic_quantization_threshold=0` 与默认逐位相同"这句话是有条件的**：
它成立的前提是包默认 `min_image_tokens=256`，即 `L = 4K ≥ 1024`，旁路本就不会触发。
`min_image_tokens` 是 `GmeOpenVINO` 的**公开构造参数**，把它调到 256 以下会让
`L < 1024`，此时 threshold=0 的输出与默认不再逐位相同。这与语言侧
`dynamic_quantization_threshold=0` 的警告是同一类，只是视觉侧被默认的 token 下界挡住了。

### 14.3.3 compiled-model cache

threshold=0 的 fused visual model 写入临时 OpenVINO cache，再由新的 `ov.Core` 导入；
cache reload 输出 `max_abs=0`。新增 primitive 字段、三输入 kernel 和 dispatch update 的
序列化路径已覆盖。

---

## 14.4 性能

等 `balanced_200` FP32 MMDocIR benchmark 结束、GPU 空闲后，用保存的 strict
control/fused plugin 二进制做三轮交错 A/B。两边都设置 threshold=0，control 只关闭
MVN matcher；每轮 warmup=3 / repeat=5，表中是三轮共 15 个 raw wall samples 的
**arithmetic mean**。

| grid | control | fused | speedup |
| --- | ---: | ---: | ---: |
| 1,32,32（L=1024） | 182.27 ms | **173.07 ms** | **1.053x** |
| 1,18,162（L=2916，cap=768 logo） | 659.15 ms | **630.25 ms** | **1.046x** |
| 1,48,48（L=2304） | 477.59 ms | **453.91 ms** | **1.052x** |
| 1,24,210（L=5040） | 1440.56 ms | **1390.51 ms** | **1.036x** |

算子归并结果：

| grid | MVN control → fused | DQ control → fused | profiled sum |
| --- | ---: | ---: | ---: |
| L=1024 | 9.43（65 calls）→ 0.31 ms（2 calls） | 15.20 → 17.52 ms | 166.55 → 156.95 ms |
| L=2916 | 27.50 → 0.92 ms | 59.52 → 57.14 ms | 646.94 → 618.56 ms |
| L=2304 | 21.70 → 0.73 ms | 47.01 → 44.51 ms | 463.81 → 440.99 ms |
| L=5040 | 48.75 → 1.59 ms | 104.39 → 102.11 ms | 1424.17 → 1378.99 ms |

fused DQ 内增加 mean/variance 和 affine，但省掉 f16 中间写回/读回以及 63 次 kernel
提交，净收益接近 standalone MVN 的原开销。

### 14.4.1 public image/fused 端到端

同样三轮交错，显式 visual threshold=0：

| size | mode | visual | end-to-end | e2e speedup |
| --- | --- | ---: | ---: | ---: |
| 448x448 | image | 166.75 → **158.60** | 249.99 → **241.57** | **1.035x** |
| 448x448 | fused | 167.02 → **158.14** | 252.36 → **244.20** | **1.033x** |
| 672x672 | image | 456.58 → **433.79** | 613.90 → **590.66** | **1.039x** |
| 672x672 | fused | 454.78 → **433.80** | 614.80 → **594.48** | **1.034x** |
| 1288x728 | image | 1308.59 → **1264.45** | 1648.04 → **1604.32** | **1.027x** |
| 1288x728 | fused | 1311.47 → **1264.65** | 1645.10 → **1599.25** | **1.029x** |

### 14.4.2 INT8 为什么没有收益

`gme_visual_int8_nomask_headfirst_rank3.xml` 的 DQ 不是 per-token：它是
`group_size=128 + symmetric i8 + precomputed_reduction`。63 个 MVN 候选在拓扑、affine、
layout、axes guard 都通过，但被 DQ attrs guard 正确跳过。三轮 A/B 为：

| grid | control | nominal fused | 结论 |
| --- | ---: | ---: | --- |
| L=1024 | 189.59 | 191.19 ms | 0.992x，见下 |
| L=5040 | 1467.63 | 1465.82 ms | 1.001x，噪声（Welch t=1.2，不显著） |

两边实际都是 65 MVN calls。

> **口径补充：L=1024 那格不是噪声。** 按保存的 15 个 raw sample 复算，control 与
> nominal fused 的 Welch t = **−4.1**，统计上显著。既然两边的融合都没有命中、执行的是
> 同一套 kernel，这 0.6% 只能是 **A/B harness 本身的槽位系统偏差**（每轮固定
> control 先、fused 后）。它给出了本项目 A/B 的**可信度下限：约 0.6%**，所以
> docs/15 里 1.004x–1.006x 那几格必须结合这个底噪来读。
>
> 方向上它是保守的：这个 A/A 对照显示**第二个槽位偏慢**，而所有被采纳的优化都跑在第二
> 个槽位并测得更快，因此那些收益若有偏差也是被低估而非高估。要优化 INT8 必须另行实现 group=128 量化与 precomputed sum，
不能直接放宽本 patch 的 per-token guard。

---

## 14.5 使用与复现

Python public API 独立控制 visual threshold，不改变短 text/query：

```python
encoder = GmeOpenVINO(
    package,
    device="GPU",
    precision="int4_causal_vopt",
    visual_dynamic_quantization_threshold=0,
)
```

端到端 profiling：

```bash
python profile_gme_image.py --precision int4_causal_vopt \
       --visual-dq-threshold 0 --warmup 3 --repeat 6 \
       --output results/r7_20260828/e2e_mvn_dq_fused.json
```

逐位等价性：

```bash
python check_visual_mvn_dq_equivalence.py \
       --output results/r7_20260828/mvn_dq_equivalence.json
```

plugin patch：先应用 round-6 combined，再应用：

```bash
git apply patches/opt5_visual_mvn_dq_fusion.patch
```

已在 clean `2026.3.0` + `all_gme_int4_gpu_opts_round6.patch` 上通过
`git apply --check`。

---

## 14.6 产物

```text
gme-Qwen2-VL-2B-Instruct-Running/
  gme_openvino.py
         visual_dynamic_quantization_threshold 参数
  04_validate_python.py
         --visual-dq-threshold
  artifacts/openvino_validation_round7_mvn_dq.json

gme-qwen2-vl-int4-hotspot-optimization/
  check_visual_mvn_dq_equivalence.py
  profile_gme_image.py
         --visual-dq-threshold
  patches/opt5_visual_mvn_dq_fusion.patch
  patches/all_gme_int4_gpu_opts_round7.patch
  results/r7_20260828/
         mvn_dq_equivalence_final.json
         mvn_dq_equivalence_int8.json
         ab_{control,fused}_*_r{1,2,3}.json
         e2e_{control,fused}_r{1,2,3}.json
         int8_ab_{control,fused}_*_r{1,2,3}.json
```
