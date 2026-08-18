# 8. per-OC INT4 量化：性能基线与精度评估

docs/06 §6.3 的 oneDNN `gemm_probe` 把 `D_cab_per_oc`（group-128 → per-output-channel
scale）测到 **1.55x**，是所有已定价候选里最大的一个，但因为"需要重新量化、有精度
代价"被挂起（§6.6 未解决项）。本节把它做成两个真正可跑的量化方案，补齐端到端性能
基线和精度评估，**结论是精度代价太大，不建议采纳**——但过程本身澄清了 probe 数字
的构成，也顺带证明了"换量化方案"这条路径本身的收益天花板比"换权重 layout"
（opt-3）高得多。

## 8.1 复核 probe：1.55x 其实是两个因素叠加

重新读 `onednn/gemm_probe.cpp` 的 case 定义（`wei_zp` 字段）：

```cpp
{"A_gme_cab",       dt::s8, dt::u4, /*grouped*/true,  128, /*wei_zp*/true,  ...},  // 当前 gme_language_int4.xml
{"B_cab_symmetric", ...,          /*grouped*/true,  128, /*wei_zp*/false, ...},  // 1.09x：只去掉 zero-point
{"D_cab_per_oc",    ...,          /*grouped*/false, 0,   /*wei_zp*/false, ...},  // 1.55x：per-OC + 对称 一起换
```

`D_cab_per_oc` 的 1.55x 是「per-output-channel scale」和「对称量化（去掉
zero-point）」两个因素**一起**换出来的。只把 group_size 改成 `-1`、量化模式仍保持
当前的 `INT4_ASYM` 的组合，probe 里从未单独测过——oneDNN 仍要做 per-channel 的
zero-point 补偿，理论上收益应该落在 1.09x（纯对称，`B_cab_symmetric`）到 1.55x
之间。因此本节做两个变体，各自对应一个量化因子：

| 精度 key | `mode` | `group_size` | 对标 probe case | 目的 |
| --- | --- | --- | --- | --- |
| `int4_per_oc` | `INT4_ASYM`（不变） | `-1` | 无（新组合，此前未测） | 只拿掉 K 分组 |
| `int4_per_oc_sym` | `INT4_SYM`（新） | `-1` | `D_cab_per_oc` | 直接对标 1.55x |

## 8.2 量化配置与产物

`03_finalize.py`，复用现有的 `compress()` helper 和 `lang_calibration`
（30 条校准文本，AWQ + scale estimation，与现有 `gme_language_int4.xml` /
`gme_language_int4_gs64.xml` 完全一致的校准流程，只改 `mode`/`group_size`）：

```python
compress(lang_fp32, package / "gme_language_int4_per_oc.xml", ...,
         mode=nncf.CompressWeightsMode.INT4_ASYM, group_size=-1, ratio=1.0,
         dataset=lang_calibration, awq=True, scale_estimation=True, subset_size=30)
compress(lang_fp32, package / "gme_language_int4_per_oc_sym.xml", ...,
         mode=nncf.CompressWeightsMode.INT4_SYM, group_size=-1, ratio=1.0,
         dataset=lang_calibration, awq=True, scale_estimation=True, subset_size=30)
```

两次 `nncf.compress_weights()` 都跑通，AWQ + scale estimation 在 `INT4_SYM` 模式下
没有报错或异常行为。产物大小（`.bin`）：

| 精度 | 大小 | 相对 gs128 |
| --- | ---: | --- |
| `gme_language_int4.xml`（gs128, ASYM，当前默认） | 922 MB | 基准 |
| `gme_language_int4_gs64.xml`（gs64, ASYM） | 947 MB | +25 MB（更细分组，更多 scale/zp） |
| `gme_language_int4_per_oc.xml`（per-OC, ASYM） | **898 MB** | −24 MB |
| `gme_language_int4_per_oc_sym.xml`（per-OC, SYM） | **898 MB** | −24 MB |

per-OC 两个变体比 gs128 更小，因为每个输出通道只有一个 scale（+ ASYM 情形下一个
zero-point），比每 128 个 K 元素一个 scale/zp 的元数据少——这与 §6.3 里
"u4 权重从 4 bit 变 8 bit是主要开销、grouping 只是次要开销"的结论方向一致。

两个新精度已注册进 `gme_openvino_config.json` 的 `precisions` 字典（视觉塔沿用
`gme_visual_int4.xml`，和 `int4_gs64` 一样只换语言塔），`dump_outputs.py` /
`profile_ops.py` 的 `IR` 字典、`profile_gme_int4_gpu.py`（新增 `--precision`
参数）、`bench_ab.sh`（新增 `PRECISION` 环境变量）已全部接入。

## 8.3 算子级基线（1024 token，单次 profile_ops.py，`--warmup 12 --repeat 8`）

**patched（opt1+2+3，当前交付构建）**，GEMM 按线性层拆分：

| linear | int4 (gs128)\* | int4_per_oc | int4_per_oc_sym |
| --- | ---: | ---: | ---: |
| gate_proj | 49.55 | 46.79 | 46.22 |
| up_proj | 35.24 | 32.29 | 31.98 |
| down_proj | 41.96 | 36.72 | 34.89 |
| qkv | 12.89 | 10.68 | 7.09 |
| o_proj | 11.45 | 8.50 | 8.71 |
| **GEMM 合计** | **151.10** | **134.97** | **128.90** |
| wall p50 | 238.21 | 221.00 | 217.04 |

\* gs128 列取自 `results/r2_final_ops_1024.json`（docs/06 §6.9.6，同一 patched 构建）。

GEMM 合计相对 gs128 分别快 **1.12x**（per_oc）、**1.17x**（per_oc_sym）——比 opt-3
单独吃到的 1.17x GEMM 收益（177.5→151.1）量级相当，是完全独立的另一条杠杆。所有
线性层普遍变快，不像 opt-3 只对 gate/up 生效。

stock wheel（无任何 OpenVINO 补丁）下的单次算子分布出现了一个数值不稳定的点
（`int4_per_oc` 的 `down_proj` 单次测到 54.72 ms，明显高于 `int4_per_oc_sym` 同条件
下的 35.03 ms，且高于 patched 构建下两者都在 35\~37 ms），怀疑是这台机器已知的
GPU 双档频率状态（§4.3 开头提到的问题）在单次、小 `repeat=8` 采样下命中了非
boost 档；这正是项目本身在 §4.3 里放弃"先测完 stock 再测 patched"、改用交替 A/B
的原因。**因此本节端到端结论只采信下面 §8.4 的交替 A/B 数据，算子级数字仅作
GEMM-占比方向性参考。**

## 8.4 端到端交替 A/B（`bench_ab.sh`，3 轮取中位，与 §4.3 同方法论）

`PRECISION=int4_per_oc ./bench_ab.sh` / `PRECISION=int4_per_oc_sym ./bench_ab.sh`：

| tokens | 精度 | stock p50 (text) | patched p50 (text) | speedup（同精度, stock→patched） |
| ---: | --- | ---: | ---: | ---: |
| 1024 | int4_per_oc | 239.80 | 225.94 | 1.061x |
| 1024 | int4_per_oc_sym | 235.88 | 218.94 | 1.077x |
| 512 | int4_per_oc | 125.30 | 108.76 | 1.152x |
| 512 | int4_per_oc_sym | 115.66 | 108.26 | 1.068x |
| 128 | int4_per_oc | 42.90 | 38.27 | 1.121x |
| 128 | int4_per_oc_sym | 40.10 | 37.77 | 1.062x |
| 32 | int4_per_oc | 30.39 | 29.84 | 1.018x |
| 32 | int4_per_oc_sym | 29.23 | 28.59 | 1.022x |

query workload 数字与 text 几乎一致（差 <1ms），从略。

**与当前默认量化方案 `int4`（gs128）横向对比**（gs128 数字取自
`results/ab/`，即 §4.3 已发表的表）：

| tokens | 对比维度 | int4 (gs128) | int4_per_oc | int4_per_oc_sym | per_oc vs gs128 | per_oc_sym vs gs128 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1024 | stock only（纯换量化，无 plugin 补丁） | 281.38 | 239.80 | 235.88 | **1.173x** | **1.193x** |
| 1024 | patched only（opt1+2+3 上叠加新量化） | 240.01 | 225.94 | 218.94 | **1.062x** | **1.096x** |
| 1024 | 全叠加（新量化 + opt1+2+3 vs 原始 stock gs128） | 281.38 | 225.94 | 218.94 | **1.245x** | **1.285x** |
| 512 | stock only | 146.55 | 125.30 | 115.66 | 1.170x | 1.267x |
| 512 | patched only | 123.81 | 108.76 | 108.26 | 1.138x | 1.143x |
| 512 | 全叠加 | 146.55 | 108.76 | 108.26 | 1.348x | 1.354x |
| 128 | stock only | 46.56 | 42.90 | 40.10 | 1.085x | 1.161x |
| 128 | patched only | 40.20 | 38.27 | 37.77 | 1.050x | 1.064x |
| 128 | 全叠加 | 46.56 | 38.27 | 37.77 | 1.217x | 1.233x |
| 32 | stock only | 32.78 | 30.39 | 29.23 | 1.079x | 1.121x |
| 32 | patched only | 32.19 | 29.84 | 28.59 | 1.079x | 1.126x |
| 32 | 全叠加 | 32.78 | 29.84 | 28.59 | 1.099x | 1.146x |

**关键发现：只换量化方案（stock only 那一行），不改一行 OpenVINO 源码，在
128\~1024 token 上就能拿到 1.09x\~1.27x，量级上和三个 plugin patch 合起来的收益
（opt1+2+3 相对原始 stock 是 1.15x\~1.18x）相当甚至更大。** 两条杠杆彼此独立、
可以叠加：`int4_per_oc_sym` + opt1+2+3 在 1024 token 上把总收益拉到 **1.285x**，
512 token 到 **1.354x**，远超本项目此前任何单一改动。

`int4_per_oc_sym` 在几乎所有格子上都比 `int4_per_oc`（保持非对称）快，方向和
probe 的"对称量化额外值一截"的结论一致（`B_cab_symmetric` 1.09x vs
`A_gme_cab` 1.00x）。

## 8.5 精度评估

### (a) 语义精度 vs PyTorch FP32 golden（`04_validate_python.py`，GPU，权威）

```
python 04_validate_python.py --precisions int4,int4_gs64,int4_per_oc,int4_per_oc_sym \
       --devices GPU --output artifacts/openvino_full_validation_per_oc.json
```

| 精度 | text cosine | query cosine | image cosine | fused cosine |
| --- | ---: | ---: | ---: | ---: |
| `int4`（gs128，当前默认） | 0.944538 | 0.955146 | 0.941821 | 0.954111 |
| `int4_gs64`（精度恢复档） | 0.949893 | 0.957442 | 0.943357 | 0.959970 |
| `int4_per_oc` | **0.826530** | **0.832787** | 0.891442 | 0.923546 |
| `int4_per_oc_sym` | **0.836570** | 0.882145 | 0.907891 | 0.930688 |

**这是本节最重要的数字**：per-OC 的两个变体在 text/query 上的 cosine 从 gs128 的
~0.945/0.955 掉到 **0.83~0.88**，跌幅比 gs128→gs64 的精度恢复反过来还大得多，
量级上也远超 opt-3 引入的漂移（§4.1b：opt-3 的漂移比量化噪声底低一个数量级；这里
是量化方案本身的噪声底直接抬高了一个数量级）。image/fused 掉得较少
（0.94→0.89~0.93），但同样是全面变差，没有一个模式变好。

`text_cos_min` 是所有校准/评测样本里最差的一个（脚本口径，见 `04_validate_python.py`
docstring），所以这不是个别样本的离群点，是系统性下降。

### (b) 自洽性：opt-1/2/3 在新量化方案下是否引入额外漂移（`dump_outputs.py` + `compare_outputs.py`，方法论对齐 §4.1b）

stock wheel vs patched（opt1+2+3）构建，同一个量化方案：

| 用例 | int4_per_oc cosine | int4_per_oc_sym cosine |
| --- | ---: | ---: |
| seq1 / seq7 / seq32 / seq129 / seq512 / seq1024 | 1.000000000 | 1.000000000 |
| batch2_padded | **1.000000000** | **1.000000000** |

**两个新精度下全部七个用例都逐位相同**，包括 gs128 上唯一有漂移的
`batch2_padded`（§4.1b：gs128 上 cosine 0.9995、max_abs 1.192e-2）。也就是说
opt-1/2/3 补丁本身在这两种新量化方案下没有引入任何额外数值漂移——8.5(a) 里
的精度损失是量化方案本身（NNCF 侧）造成的，与本项目交付的 OpenVINO plugin 改动
无关。

## 8.6 结论

1. **单独换量化方案（per-OC）比换权重 layout（opt-3）杠杆更大**：stock only 就有
   1.09x\~1.27x，且与 opt-3 完全正交、可叠加，1024 token 全叠加拿到 1.285x
   （`int4_per_oc_sym`），是本项目目前测到的最大端到端收益。
2. **但精度代价远超可接受范围**：text/query cosine 从 ~0.945/0.955 掉到
   0.83~0.88，跌幅是 gs128→gs64 精度恢复档改善量的一个数量级以上，也是 opt-3
   引入漂移量级的一个数量级以上。对 GME 这种检索/embedding 模型，text/query
   cosine 掉到 0.83~0.88 大概率会在下游检索排序上造成肉眼可见的质量下降——
   这正是 docs/03_finalize.py 里当初专门给语言塔加 AWQ + scale estimation
   的原因（数据无关 INT4 对 text 的伤害就已经不可接受，见该文件注释），per-OC
   分组比 128-分组更粗只会让这个问题更严重。
3. **不建议把 `int4_per_oc` / `int4_per_oc_sym` 作为语言塔量化方案采纳**，
   无论对称与否。docs/06 §6.6 里"per-OC 量化（1.55x，需精度评估）"这条待办
   现在有了明确结论：**已评估，精度不达标，维持挂起/否决状态**。
4. 两个变体之间：`int4_per_oc_sym` 在性能和 image/fused 精度上都略优于保持
   非对称的 `int4_per_oc`，text/query 精度互有胜负、都不达标。**如果未来要
   继续在这个方向上找收益，方向应该是"用更多/更有代表性的校准数据 + 更细致
   的逐层混合精度（`ratio<1.0`，敏感层退回 gs128 或 INT8）"，而不是简单粗暴
   地把所有层都换成 per-OC**——这是后续方向，本轮不做。
5. `int4_gs64` 仍然是唯一验证过的"精度恢复"选项（cosine 全面优于 gs128，
   性能代价可忽略，docs/04 §4.1），在精度敏感场景应继续用它而不是 gs128 或
   per-OC。

**补充**：在 `document_embedding_eval_int8_int4_gpu/`（14 页工程 PDF、8 条人工
query 的检索质量 probe）上复测了同样的两个精度，性能结论方向一致（batch 8
文本再快 7.6%\~10.8%），但**检索 Recall/MRR 没有表现出这里测到的绝对精度下降
**——这是意料之中的（Recall/MRR 测的是同一量化方案内部的相对排序，cosine-vs-
FP32-golden 测的是绝对误差，两者不矛盾，样本量也差两个数量级），详见
[`../document_embedding_eval_int8_int4_gpu/REPORT_per_oc.md`](../document_embedding_eval_int8_int4_gpu/REPORT_per_oc.md)
§3。**不改变本节"不建议采纳"的结论**，只是记录一个值得注意的现象。

## 8.7 产物

```
ov_models/gme-qwen2-vl-2b-instruct-openvino/
  gme_language_int4_per_oc.xml/.bin       group_size=-1, INT4_ASYM, AWQ+SE
  gme_language_int4_per_oc_sym.xml/.bin   group_size=-1, INT4_SYM,  AWQ+SE
  gme_openvino_config.json                新增 int4_per_oc / int4_per_oc_sym 两个 precision key
  nncf_fingerprints.json                  两个新 key 的量化配置指纹

gme-qwen2-vl-int4-hotspot-optimization/
  results/per_oc_ops_1024_{stock,patched}.json        §8.3
  results/per_oc_sym_ops_1024_{stock,patched}.json    §8.3
  results/ab/{stock,patched}_int4_per_oc[_sym]_len*_r*.json   §8.4（新文件名格式，
                                                        旧的 int4(gs128) 结果文件名不变）
  results/acc_int4_per_oc[_sym]_{stock,patched}.npz   §8.5(b)

gme-Qwen2-VL-2B-Instruct-Running/
  artifacts/openvino_full_validation_per_oc.json      §8.5(a)
```

代码改动：`03_finalize.py`（新增两个 `compress()` 调用 + `precisions` 字典两条）、
`dump_outputs.py` / `profile_ops.py`（`IR` 字典各加两条）、
`profile_gme_int4_gpu.py`（新增 `--precision` 参数，默认 `int4` 保持向后兼容）、
`bench_ab.sh`（新增 `PRECISION` 环境变量，输出文件名加入精度字段，旧格式文件
仍可被汇总脚本正确解析）。
