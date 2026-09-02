# 第四轮：image / fused 路径的 hotspot 分析与优化（历史）

> 第五轮及之后又加入了 `headfirst`、`rank3`、DQ reuse、RMS/MVN fusion 和 QKV
> transpose。因此本文的 `_vnomask` 结果是第四轮历史中间态；当前 final precision、
> 初始 stock 对照和完整性能见 [`docs/16`](16_round9_final_rebenchmark.md) 及
> [`results/r9_latest/`](../results/r9_latest/)。

前三轮全部围绕**语言 IR** 的纯文本 `text` / `query` 路径：1024 token 从 stock 283.0 ms
降到 `int4_causal` 的 222.3 ms（累计 1.273x，docs/06、docs/09、docs/10）。

但 `image` / `fused` 两个模式没有跟着变快。PERFORMANCE_REPORT.md §6.3 里 GPU INT4 的
`448x448` 仍然是 495 ms，`1288x728` 是 4574 ms，并且随视觉 token 数**超线性**增长。
本轮的任务就是把这条路径拆开、定位、优化。

**结论先说：**本轮在**导出侧**找到并修掉了一个和第三轮 causal 完全同类、但影响大得多的问题——
Qwen2-VL 视觉塔的 SDPA 带着一张 **rank-3 BOOL `[1, L, L]` mask**，导致 GPU plugin
**完全不 lower 优化 SDPA kernel**，32 层注意力全部退化成
`MatMul + Select + Softmax + MatMul` 加一堆 permute。而这张 mask 对本包**恒为全 True**。
删掉它之后：

| workload | 之前 | 之后 | 加速 |
| --- | ---: | ---: | ---: |
| 视觉 IR，`448x448`（L=1024） | 326.7 ms | **213.0 ms** | **1.53x** |
| 视觉 IR，`1288x728`（L=4784） | 4034.7 ms | **1457.8 ms** | **2.77x** |
| 视觉 IR，logo（L=5040） | 4444.4 ms | **1573.0 ms** | **2.83x** |
| 端到端 image，`448x448` | 563.6 ms | **288.7 ms** | **1.95x** |
| 端到端 image，`1288x728` | 4494.4 ms | **1798.3 ms** | **2.50x** |
| 端到端 fused，`1288x728` | 4479.9 ms | **1789.6 ms** | **2.50x** |

FP32 上**逐位相同**（5 个 grid，`max_abs_diff = 0.0`），**没有任何 plugin 改动**。

顺便回答两个必然会被问到的设计问题（§11.2.6、§11.6.3，都有实测）：
**这张 mask 在 transformers 5.2.0 上游已经不存在了**（两个分支都传 `attention_mask=None`），
本包 IR 里的它是 transformers 4 时代的残留，删它是**追上上游**而不是自创；
而本包逐图调用（不像官方那样打包）是有依据的选择——**打包在本后端上更慢**
（N=4 时 1.46x，N=5 时 1.62x），因为没有 FlashAttention varlen 就得算完 O((ΣL)²) 再丢掉。
代价是 image 路径的 batch 吞吐收益很小（N=8 仅 3.5%，而 text 是 4.4x），
但那是「一张 448x448 就已经打满 32 EU」决定的（并发实测 0 收益），与逐图无关。
**这条饱和结论是本机性质**：§11.2.7 给出模型和翻盘条件——换一台空闲算力更多的设备，
打包确实可能反超逐图，但那时更该用的是并发或真 batch 维，打包始终夹在中间。

同时本轮否掉了一个看起来很对、小图上也确实有效的候选（§11.5）：把 SDPA 的输入 Transpose
融进 primitive。它在 L=2304 上 −31.8 ms，但在 L=5040 上 **+103.8 ms**，因为它用一笔 O(L)
的拷贝换来了 O(L²) 的 strided 读取。本包 `max_image_tokens=1280` 正好落在它变坏的一侧，
所以**不设默认**。

---

## 11.0 起点：口径与测量工具

### 11.0.1 环境

| 项目 | 内容 |
| --- | --- |
| CPU/GPU | Intel Core Ultra 7 356H，32 EU Intel integrated GPU |
| OpenVINO runtime | `2026.3.0-22451-8a17657b995-qwen-ttft-optimizations` |
| plugin 改动 | 仅 `patches/all_gme_int4_gpu_opts.patch`（8 files, +390 −19），**本轮未新增** |
| transformers | 5.2.0 |
| 语言 IR | `gme_language_int4_causal.xml`（docs/09 采纳的默认） |
| 复核基线 | 1024 token text = 221.4 ms，与 docs/09 的 222.3 ms 一致 |

开工前先复核 plugin 状态：`profile_ops.py --precision int4_causal --length 1024` 得到
wall p50 **221.4 ms**、SDPA 11.73 ms、`broadcast_gpu_ref__f16` 为 0，与 docs/09/10 发布的
数字一致，说明 install 树确实带着第二轮的 patch，本轮所有对比都建立在同一基线上。

### 11.0.2 本轮新增的两个测量工具

之前的工具只覆盖语言 IR，image/fused 只有一个端到端数字，不可归因。本轮补上：

| 脚本 | 作用 |
| --- | --- |
| `profile_visual_ops.py` | 视觉 IR 的 per-op GPU profiling，`profile_ops.py` 的视觉侧对应物。直接喂合成 `pixel_values` + `image_grid_thw`，不经 PIL/processor，所以每一毫秒都能归到算子上 |
| `profile_gme_image.py` | 端到端 image/fused，拆成 `preprocess` / `visual` / `language` / `end_to_end` 四段 |

### 11.0.3 一个必须先说清楚的量：L 和 K 不是一回事

这是本轮最重要的口径问题，之前的报告一直只提 K：

- **K**：merge 之后的**视觉 token 数**，语言 IR 看到的就是这个，`448x448` → K=256；
- **L**：merge 之前的**原始 patch 行数**，视觉塔内部的序列长度，**L = 4K**。

Qwen2-VL 的 `spatial_merge_size=2`，所以视觉塔实际是在一条 **4 倍长**的序列上做**全**注意力。
`448x448` 的视觉塔序列是 1024，不是 256；`1288x728` 是 4784，不是 1196。
**image 时延的超线性增长几乎全部来自这里**，而不是语言 IR。

---

## 11.1 阶段拆分：时间到底在哪

`profile_gme_image.py`，`int4_causal`（即优化前），GPU，p50：

| mode | size | K | S | preprocess | **visual** | language | end-to-end |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| image | 448x448 | 256 | 281 | 14.2 | **309.1** | 238.5 | 563.6 |
| image | 644x476 | 391 | 416 | 20.0 | **655.8** | 155.7 | 831.7 |
| image | 672x672 | 576 | 601 | 27.3 | **1037.7** | 341.5 | 1408.9 |
| image | 896x896 | 1024 | 1049 | 45.3 | **2685.7** | 240.9 | 2975.2 |
| image | 1288x728 | 1196 | 1221 | 57.9 | **4141.1** | 280.9 | 4494.4 |

三个直接结论：

1. **视觉 IR 是绝对大头**：从 55% 到 92%，并且占比随分辨率单调上升。
2. **CPU 预处理不是问题**：14–58 ms，占比 < 3%，不值得优化。
3. **语言段的数字不可信**，见下。

### 11.1.1 language 段这一列为什么是错的（以及它揭示了什么）

上表 language 列在 S=281 时是 238.5 ms，在 S=1049 时反而是 240.9 ms——不单调，显然有问题。
单独测语言 IR（`profile_ops.py`，独立进程，同一个 `gme_language_int4_causal.xml`）：

| S | 单独测 | 上表（跟在视觉之后） | 差 |
| ---: | ---: | ---: | ---: |
| 281 | **79.6** | 238.5 | +158.9 |
| 601 | **137.0** | 341.5 | +204.5 |
| 1049 | **232.3** | 240.9 | +8.6 |

所以语言 IR 本身是正常单调的；上表那一列是**被前面那一次很重的视觉推理污染**了。
在这块 32 EU、与 CPU 共享 TDP 的核显上，一次 300–4000 ms 的重负载会把 GPU 压进低频档，
紧接着的语言推理就跑在降频状态下。

这不是测量瑕疵，而是**真实的端到端效应**：优化前的用户确实要吃这笔。它也解释了为什么
§11.4 的端到端加速比视觉段本身的加速更大——视觉段变轻之后，后面那次语言推理也回到了正常速度。
但归因必须说清楚：**这一笔不是"语言模型变快了"，而是"不再被拖慢了"。**

---

## 11.2 hotspot：视觉塔里发生了什么

`profile_visual_ops.py --precision int4 --grid 1,32,32`（L=1024，K=256）：

| exec_type | ms | % | calls |
| --- | ---: | ---: | ---: |
| `jit:gemm:any__i8`（权重线性层） | 89.75 | 30.1 | 130 |
| **`jit:gemm:any__f16`** | **67.44** | **22.6** | **65** |
| **`softmax_gpu_bf__f16`** | **54.28** | **18.2** | **32** |
| **`permute_ref__f16`** | **48.26** | **16.2** | **192** |
| `dynamic_quantize_gpu_opt__f16` | 15.03 | 5.0 | 130 |
| `ocl::rope::opt___f16` | 10.02 | 3.4 | 64 |
| `mvn_gpu_bfyx_opt__f16` | 9.26 | 3.1 | 65 |
| **`select_gpu_ref__u8`** | **3.03** | **1.0** | **32** |
| profiled 合计 | 298.56 | | |
| host gap | 28.10 | | |
| **wall p50** | **326.66** | | |

**`ocl::sdpa::opt__f16` 一次都没有出现。**

加粗的四项就是被拆开的注意力：每层一个 `Select`（mask）、两个 f16 MatMul（QKᵀ 和 PV）、
一个 Softmax，外加围绕它们的 permute。合计
`67.44 + 54.28 + 48.26 + 3.03 = 173.0 ms`，占 profiled 的 **58%**。

对照语言 IR：28 层、S=1024、causal，SDPA 只要 **11.73 ms**。视觉塔 32 层、L=1024、
16 头 × 80 维、全注意力，按同样的 kernel 估算应在 20–25 ms 量级。**差了 7 倍**。

### 11.2.1 根因：一张 rank-3 BOOL mask

顺着 SDPA 的第 4 个输入往上走（`gme_visual_fp32.xml`）：

```
SDPA.in[3]  [1,?,?]  Type: char        <- Reshape 'SliceAssign_1144/Reshape_4'
  <- ScatterNDUpdate 'SliceAssign_1144/ScatterNDUpdate'
    <- Broadcast '__module.visual.blocks.0.attn/aten::zeros/Broadcast'  [1,?,?] char
```

即 **transformers 4** 时代 `VisionAttention` 的这段（Qwen2-VL 的 packed 视觉注意力）：

```python
attention_mask = torch.zeros([1, seq_length, seq_length], dtype=torch.bool)
for i in range(1, len(cu_seqlens)):
    attention_mask[..., cu_seqlens[i-1]:cu_seqlens[i], cu_seqlens[i-1]:cu_seqlens[i]] = True
```

> 这段代码在 **transformers 5.2.0 里已经不存在了**：两个分支（FlashAttention varlen /
> 其它后端按 `cu_seqlens` 切分）都传 `attention_mask=None`，实测 `torch.zeros`、`full(`
> 均不出现。本包的 IR 导出于 2026-08-05，是 transformers 4 的产物，所以这张 mask 被
> 烘进了图里。详见 §11.2.6——**§11.3 删掉它，方向上是追上上游而不是自创。**

两个后果：

1. **类型和 rank 都不对**。GPU plugin 的 SDPA fusion 不接受 rank-3 的 BOOL mask，于是
   整个 SDPA 被分解。这与 docs/10 §4.1 记录的语言侧 `_v5ctrl` 对照组是**完全同一个失败模式**
   ——那次也是 BOOL mask 让 SDPA 没走 `ocl::sdpa::opt`，并重新长出 `broadcast`/`permute`。
2. **这张 mask 对本包恒为全 True**。Qwen2-VL 视觉塔是 *packed* 而非 batched：多张图的 patch
   拼成一条序列，用 `cu_seqlens` 限制注意力只在各自块内。而本包**一次只喂一张图**，
   `cu_seqlens = [0, L]`，块对角 mask 就是全 1。

### 11.2.2 "一次只喂一张图"不是假设，是既有事实

这一点必须站得住，否则整个优化不成立。**注意：官方 GME 不是这么做的**——
`gme_inference.py:61` 是 `self.base.visual(pixel_values, grid_thw=image_grid_thw)`，
一次把**整批**图片的 patch 拼起来喂进视觉塔，靠 `cu_seqlens` 块对角 mask 隔开。
所以"mask 恒为全 True"是**本包的性质，不是 Qwen2-VL 的性质**，必须单独论证。

论证分三步，全部实测（见 §11.2.3）：

1. **打包多图 ≡ 逐图**。块对角注意力等价于各块独立注意力，且 rotary 表只依赖各图自己的
   grid。在**官方 PyTorch FP32** 上实测：打包 3 张不同尺寸的图 vs 逐图，
   row cosine min **0.999999702**（差异 1.37e-2 / 相对 1.24e-4，是 fp32 归约顺序噪声）。
   所以 `_encode_image_batch` 拆成逐图调用**没有改变语义**——这是本包早就在做的事，
   不是本轮引入的。
2. **本包运行时确实逐图调用**。`gme_openvino.py:_encode_image_batch` 显式把多图 batch
   拆成每图一个 visual request，注释写明了理由。
3. **导出的 IR 根本接不了多行 grid**。`rot_pos_emb` 在 Python 里 `for t, h, w in grid_thw`
   循环，trace 把图片张数烘进了图里（导出时就打 `TracerWarning`）。
   **原来那个带 mask 的 IR 同样接不了**——实测两个 IR 都在同一个节点
   `__module.visual/prim::ListUnpack/Squeeze` 抛异常。

所以删掉 mask **没有丢掉任何本来能用的能力**：那张 mask 在导出后的图里只可能是全 True，
唯一能让它有意义的输入形态（多块 `cu_seqlens`）在两个 IR 上都是不可达的。

### 11.2.3 完整情况枚举（这张表是本节的结论）

| 输入形态 | 官方 HF PyTorch | 本包**带 mask** 的 IR | 本包 **mask-free** IR |
| --- | --- | --- | --- |
| 单图，`t=1`，1 行 grid（**GME 图像路径唯一会产生的形态**） | mask 恒全 True | 正常 | **与带 mask 逐位相同**（§11.3.3） |
| 多图一次调用，n 行 grid | mask 块对角，**有意义** | **抛异常**（`prim::ListUnpack/Squeeze`） | **抛异常**（同一节点） |
| 视频，`t>1`，1 行 grid（`cu_seqlens` 有多块） | mask 块对角，**有意义** | **输出全 NaN**（128×1536 全部） | 有限值，但语义是跨帧稠密注意力 |
| 多图 batch（公开 API `get_image_embeddings([...])`） | 打包一次调用 | 运行时拆成逐图 → 正常 | 运行时拆成逐图 → **正常** |

两点需要强调：

- **GME 的 processor 永远只产生 `t=1`**。实测 `448x448 → [[1,32,32]]`、
  `672x448 → [[1,48,32]]`；三张图一起进 processor 得到 **3 行**（每行 `t=1`），而不是
  一行 `t=3`。GME 是图像 embedding 模型，`fetch_image` 不走视频路径，
  `preprocessor_config.json` 里的 `temporal_patch_size=2` 只是内部把单图复制填满时间维，
  报出来的 grid `t` 仍是 1。所以表中第三行对本包是**不可达状态**。
- **带 mask 的 IR 在 `t>1` 上本来就是坏的**（输出 128×1536 全为 NaN）。也就是说 mask-free 并没有
  把一个"能用的视频路径"换掉；那条路径在导出后从来没有工作过。

### 11.2.4 万一真的误用，错误是响的还是哑的？

这是采纳前必须回答的问题。在官方 PyTorch 上强行对**打包的多图序列**关掉 mask
（即让图片之间真的互相注意），与逐图结果对比：

| 对比 | max_abs | 相对 | row cosine min |
| --- | ---: | ---: | ---: |
| 打包**带** mask vs 逐图（fp32 归约噪声） | 1.371e-02 | 1.24e-04 | **0.999999702** |
| 打包**不带** mask vs 逐图（真的串味） | 6.030e+01 | 5.45e-01 | **0.058744** |

真实的跨图串味比噪声底大 **4399 倍**，row cosine 直接掉到 0.059。
**这类错误是响的，不是哑的**——任何一次精度检查都会立刻抓到，不会悄悄劣化召回。

### 11.2.5 加了一道 trace 期守卫

上面三条论证里最脆弱的一条是"IR 只支持单图"，因为它依赖 `make_example` 传进来的样例
**恰好只有一张图**。如果将来有人把导出样例改成多图，或者接了视频路径，
mask-free 分支就会**静默**产出一个跨块串味的 IR。

所以 `unmasked_vision_attention()` 里加了一道 trace 期检查（Python 层，不进图）：
`cu_seqlens` 的长度必须是 2（= 恰好一个块）。多图或 `t>1` 都会长度 ≥ 3 而直接报错。
实测：

| trace 样例 | 结果 |
| --- | --- |
| 单图 `t=1` | 正常导出 |
| 视频 `t=2`，1 行 | **守卫报错** |
| 两张图，2 行 | **守卫报错** |

也就是说：这个优化不可能因为将来改导出样例而悄悄失效——它会在导出时就失败。

### 11.2.6 为什么本包逐图调用，而不是像官方一样打包？

§11.2.2 指出官方是打包调用，这自然引出一个设计问题：**本包为什么不照做？是有意的还是当初
没设计好？** 三条理由，都可实测：

**(1) 在这套后端上，打包比逐图更慢——而且随 N 越来越慢。**

打包的注意力是 O((ΣL)²)，然后用 mask 把非对角块**算完再丢掉**；逐图是 O(ΣL²)，不做无用功。
在 mask-free 视觉 IR 上按「相同总 patch 数」实测（448x448 每张 L=1024）：

| N 张 448x448 | 逐图（N 次调用） | 打包（一次 L=N×1024） | 打包/逐图 |
| ---: | ---: | ---: | ---: |
| 1 | 197.7 | 197.7 | 1.00x |
| 2 | 395.5 | 467.8 | **1.18x 更慢** |
| 4 | 790.9 | 1156.1 | **1.46x 更慢** |
| 5 | 988.7 | 1597.7 | **1.62x 更慢** |

所以打包不是"本包放弃了的优化"，而是**本包避开了的劣化**。

**(2) 官方能打包得起，靠的是 FlashAttention varlen——这条路在 OpenVINO/iGPU 上不存在。**

只有 FA2 的 `cu_seq_lens` varlen kernel 才会**真正跳过**非对角块，让打包不浪费算力。
查 transformers 5.2.0 的 `VisionAttention.forward`：

| 分支 | 做法 |
| --- | --- |
| `is_flash_attention_requested()` | `attention_mask=None` + `cu_seq_lens_q/k` varlen（真跳过，需要 CUDA + FA2） |
| 其它所有后端（sdpa / eager） | 按 `cu_seqlens` **`torch.split` 成逐图**，且 `attention_mask=None` |

**transformers 5.2.0 里已经完全没有那张 dense BOOL mask 了**（实测 `torch.zeros`/`full` 均不
出现，两个分支都传 `attention_mask=None`）。也就是说：

- **上游自己在所有非 FlashAttention 后端上就是逐图**——本包的做法与**当前**上游一致，不是偏离；
- 本包 IR 里那张 dense BOOL mask 是 **transformers 4 时代的产物**（IR 导出于 2026-08-05），
  上游后来已经删掉了。§11.3 删 mask 这件事，**方向上是在追上上游，而不是自创一套**。

**(3) 打包会让"一次导出支持任意 N"变成不可能。**

`rot_pos_emb` 在 Python 里遍历 `grid_thw` 的行，trace 会把图片张数烘成常量（§11.2.2 第 3 点，
两个 IR 都因此拒绝多行 grid）。要走打包路线，就得**每个 batch size 各导一份 IR**，
而且尺寸组合不同还不能复用。逐图调用是让**一份导出服务任意 N、任意尺寸混合**的前提。

**结论**：这是一个有依据的设计选择，不是缺陷。它同时拿到了「和官方 FA2 路径同样不浪费算力」
和「一份 IR 服务任意 N」两件事，代价是放弃了一个在本后端上本来就是负收益的打包路径。
`gme_openvino.py` 的类 docstring 早就写了这个理由（"a shared sequence costs O((sum L)^2)
masked attention instead of O(sum L^2)"），本轮第一次把它量化成上面那张表。

关于 §11.2.6 中"上游非 FA 后端就是逐图"这一点，条件可以精确写出来：
`is_flash_attention_requested()` 的实现就是 `"flash" in config._attn_implementation`。
所以 **`sdpa` / `eager` / `flex_attention` 以及任何名字里不含 `flash` 的实现，全部走
`else` 分支的 `torch.split` 逐图路径**。只有 FA2/FA3 系列走 varlen。
（限 transformers 5.2.0 实测；4.x 是 dense mask。）

### 11.2.7 换一台算力更强的机器，逐图还会赢吗？——会翻盘，但赢家不是打包

§11.6.3 说"一张图就把设备打满了"，这是**本机性质**，不能外推。本机实测的饱和证据：
对 8 张 448x448 用不同并发配置跑同一个视觉 IR，

| 配置 | N=8 总时间 | ms/图 | vs 顺序 |
| --- | ---: | ---: | ---: |
| LATENCY，streams=1，顺序 | 1590.5 | 198.8 | 1.00x |
| streams=2，async | 1614.1 | 201.8 | 0.99x |
| streams=4，async | 1608.6 | 201.1 | 0.99x |
| THROUGHPUT hint，async | 1578.1 | 197.3 | 1.01x |

**并发收益为 0**——这块 32 EU 核显确实被一张图打满了。

那么在一台空闲算力更多的设备上呢？把它模型化。L=1024 时 profiled 197.8 ms 中
SDPA 占 **17.1%**（`a`），其余 **82.9%**（`g`）。打包让注意力变成 N 倍、其余不变；
设逐图在该设备上只能达到利用率 `f`：

```
packed/split ≈ (g + N·a) · f
```

先用本机（`f = 1.0`）校验：

| N | 模型预测 | §11.2.6 实测 |
| ---: | ---: | ---: |
| 2 | 1.17 | 1.18 |
| 4 | 1.51 | 1.46 |
| 5 | 1.68 | 1.62 |

吻合，于是可以反推**打包要划算，设备得空闲到什么程度**：

| N | 打包多做的工作 | 逐图利用率需低于 |
| ---: | ---: | ---: |
| 2 | 1.17x | **85%** |
| 4 | 1.51x | **66%** |
| 8 | 2.20x | **46%** |
| 16 | 3.56x | **28%** |

**所以结论要分两半说：**

1. **"打包更慢"不是普遍真理。** 在一块大 dGPU 上，单张图的 1024 行 ViT 很可能填不满设备；
   只要逐图利用率掉到 66% 以下（N=4），dense 打包就会反超。本机是 ~100%，所以打包输；
   换机器这个结论**确实会翻盘**。
2. **但即使翻盘，正确答案也不是打包。** 打包是拿 N 倍的注意力算力去换占用率；而
   **并发（streams/async）能拿到同样的占用率，且完全不多做功**。也就是说在那台设备上
   `并发 > 打包 > 串行逐图`，打包永远夹在中间。本机因为已经饱和，三者才并列。

**只有配上 varlen kernel（FA2 的 `cu_seq_lens`，真正跳过非对角块）时，打包才既有占用率
又不浪费算力**——那正是官方走打包路线的原因，也是 OpenVINO/iGPU 上拿不到的东西。

**迁移到更强设备时的正确动作顺序**（都不需要重新导出）：

1. 先调 `streams` / 用 `AsyncInferQueue`——零代码改动、零额外算力，本机 0 收益纯粹是因为已饱和；
2. 如果图片尺寸**统一**，可以考虑导一份带真正 batch 维 `[N, H, L, D]` 的视觉 IR：
   占用率和打包一样好，算力却和逐图一样省——**严格优于打包**。Qwen2-VL 之所以打包而不是
   加 batch 维，只是为了在一次调用里容纳**不同尺寸**；
3. 只有当后端提供 varlen/block-sparse 注意力时，打包才值得考虑。

本轮没有第二台设备可测，所以上表是**模型外推**（已在本机校验），不是实测值——
换机器时应重跑 §11.2.6 的 packed/split 表和上面的并发表再下结论。

---

## 11.3 优化：mask-free 视觉 SDPA（采纳）

### 11.3.1 改法

`02_export.py` 新增 `unmasked_vision_attention()` 上下文管理器和 `--unmasked-vision`，
与第三轮 `suppress_causal_mask_materialization()` / `--causal-sdpa` 完全对称。trace 期间把
`VisionAttention.forward` 换成一版：对整条序列**只调用一次** SDPA，`attn_mask=None`、
`is_causal=False`、rank-4 Q/K/V。产物写到独立文件 `gme_visual_fp32_nomask.xml`，
原 IR 保留，两者可在同一 session 里 A/B。

两个必须绕开的坑：

- **不能直接重跑现有导出脚本。** transformers 5 的 `VisionAttention` 走的是
  `torch.split(..., lengths.tolist(), dim=2)` 分块路径，`.tolist()` 会把本次 trace 的 L
  变成常量塞进图里，**动态序列维直接没了**。所以必须显式替换 forward，而不是指望上游默认行为。
  （这与 docs/09 §9.3 "直接重跑现有导出脚本 wall 238 → 295" 是同一类教训。）
- **不要显式传 `scale=`。** `VisionAttention.scaling = head_dim ** -0.5` 恰好就是 SDPA 的
  默认 `1/sqrt(q.size(-1))`，显式传会导出成 **5 输入** SDPA（Q, K, V, 标量 0 mask, scale），
  而 plugin 的 fusion 匹配的是 3 输入形态。改回默认后是干净的 3 输入 SDPA。

### 11.3.2 顺手修掉的一个真 bug：transformers 5 下视觉 IR 输出接错了

`GmeVisualEncoder.forward` 原来直接 `return self.visual(...)`。transformers 4 返回的就是
merge 后的 `[K, 1536]` 张量，但 transformers 5 返回
`BaseModelOutputWithPooling(last_hidden_state=[L, 1280], pooler_output=[K, 1536])`。
照原样 trace 会导出一个**双输出** IR，而且 **output(0) 是没 merge 的 1280 维隐状态**——
`gme_openvino.py` 恰恰读的是 `visual.output(0)`。

也就是说：在 transformers 5 下重新导出视觉 IR，会**静默**产出一个喂给语言 IR 的张量既错宽度
又错语义的包。本轮把 `GmeVisualEncoder` 改成显式取 `pooler_output`，两个 transformers 大版本
下导出签名一致。这个 bug 与本轮的性能改动无关，但会绊倒任何一次重新导出。

### 11.3.3 等价性：FP32 逐位相同

`check_vision_equivalence.py`（CPU，两个 FP32 IR，同一批输入）：

| 用例 | grid | K | cosine | row_cos_min | max_abs_diff | 逐位相同 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 极小 | 1,2,2 | 1 | 1.000000000 | 1.000000000 | 0.000e+00 | 是 |
| 448x448 | 1,32,32 | 256 | 1.000000000 | 1.000000000 | 0.000e+00 | 是 |
| 非正方 | 1,34,46 | 391 | 1.000000000 | 1.000000000 | 0.000e+00 | 是 |
| 672x672 | 1,48,48 | 576 | 1.000000000 | 1.000000000 | 0.000e+00 | 是 |
| logo 长宽比 | 1,24,210 | 1260 | 1.000000000 | 1.000000000 | 0.000e+00 | 是 |

用 row cosine 而不只是整体 cosine，是因为语言 IR 把这 K 行当成 K 个独立 token 消费，
整体 cosine 可能掩盖单独一行的问题。这些 grid 都不是 trace 时用的那个，正好检验动态维
有没有被烘成常量。

### 11.3.4 多图 batch 仍然完全可用

这一点容易被 §11.2 的"一次只喂一张图"误读，所以单独说明：**公开 API 的 batch 模式不受影响，
用法与官方 GME 完全一致。**

```python
enc.get_image_embeddings([img1, img2, img3])          # 任意 N，任意尺寸混合
enc.get_fused_embeddings([t1, t2, t3], [i1, i2, i3])
```

"一次只喂一张图"说的是**视觉 IR 这一层**的调用粒度，不是公开 API 的粒度。
`_encode_image_batch` 收到 N 张图后拆成 N 个 visual request，再把结果拼回去做一次
`[B, S]` 的语言调用。§11.2.2 第 1 步已经在官方 PyTorch 上验证了这个拆分与官方的打包调用
等价（row cosine 0.9999997）。

顺带一提：这个拆分本身还**比官方更快**——打包序列的注意力是 O((ΣL)²)，逐图是 O(ΣL²)。

### 11.3.5 单图契约仍然成立

mask 删掉之后，如果真有人把多张图的 patch 一次喂进去，注意力会跨图串味。
`check_vision_batch_contract.py` 直接验证运行时守住了单图契约：三张**不同尺寸**的图
（448x448 / 672x448 / 448x672，K = 256/384/384）走 batch 路径，每张图的**视觉 IR 输出**
必须和它单独跑时逐位相同。

| precision | 视觉 IR | row_cos_min | max_abs | 逐位相同 |
| --- | --- | ---: | ---: | ---: |
| `int4_causal` | `gme_visual_int4.xml` | 1.000000000 | 0.000e+00 | 是 |
| `int4_causal_vnomask` | `gme_visual_int4_nomask.xml` | 1.000000000 | 0.000e+00 | **是** |
| `int8` | `gme_visual_int8.xml` | 1.000000000 | 0.000e+00 | 是 |
| `int8_vnomask` | `gme_visual_int8_nomask.xml` | 1.000000000 | 0.000e+00 | **是** |

> **口径说明（一次测错，记下来）：** 这个检查最初是拿**最终 embedding** 比的，结果
> `int4_causal` 显示 batch 与单图差 `9.9e-3`，看上去像 mask-free 反而更稳。追下去发现
> 差异**完全在语言 IR**：`padded` batch mode 下行会随 batch 组成轻微漂移，这是
> `GmeOpenVINO` docstring 早就写明的既有性质，与视觉塔无关，恰好在 vnomask 那组输入上为 0。
> 视觉 IR 本身两组都是逐位相同。所以检查改成只比视觉 IR 的输出——那才是这次改动可能破坏的东西。

### 11.3.6 算子构成：mask 删掉之后

L=1024（K=256），同一 grid：

| exec_type | 优化前 | 优化后 | Δ | calls 前/后 |
| --- | ---: | ---: | ---: | ---: |
| `jit:gemm:any__i8` | 89.75 | 90.18 | +0.4 | 130/130 |
| `jit:gemm:any__f16` | **67.44** | **0.01** | **−67.4** | 65/1 |
| `softmax_gpu_bf__f16` | **54.28** | **0.00** | **−54.3** | 32/0 |
| `permute_ref__f16` | 48.26 | 38.36 | −9.9 | 192/128 |
| **`ocl::sdpa::opt__f16`** | **0.00** | **33.82** | **+33.8** | 0/**32** |
| `select_gpu_ref__u8` | **3.03** | **0.00** | **−3.0** | 32/0 |
| `dynamic_quantize_gpu_opt__f16` | 15.03 | 15.10 | +0.1 | 130/130 |
| `ocl::rope::opt___f16` | 10.02 | 9.52 | −0.5 | 64/64 |
| `mvn_gpu_bfyx_opt__f16` | 9.26 | 9.22 | 0 | 65/65 |
| profiled 合计 | 298.56 | 197.82 | −100.7 | |
| host gap | 28.10 | 15.17 | −12.9 | |
| **wall p50** | **326.66** | **212.99** | **−113.7（1.53x）** | |

注意力从 `67.44 + 54.28 + 3.03 = 124.75 ms` 变成一个 **33.82 ms** 的 `ocl::sdpa::opt`，
**权重 GEMM 完全没动**（89.75 → 90.18）——和第三轮 causal 一样，这笔收益与任何 GEMM 侧
后续优化正交。host gap 从 28.1 降到 15.2，来自 primitive 数量下降。

### 11.3.7 随分辨率的收益

`profile_visual_ops.py`，视觉 IR wall p50：

| grid | L | K | 优化前 | 优化后 | 加速 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1,32,32（448x448） | 1024 | 256 | 326.7 | **213.0** | **1.53x** |
| 1,34,46（644x476） | 1564 | 391 | 668.4 | **350.6** | **1.91x** |
| 1,48,48（672x672） | 2304 | 576 | 1074.7 | **543.1** | **1.98x** |
| 1,64,64（896x896） | 4096 | 1024 | 2708.1 | **1168.2** | **2.32x** |
| 1,52,92（1288x728） | 4784 | 1196 | 4034.7 | **1457.8** | **2.77x** |
| 1,24,210（logo） | 5040 | 1260 | 4444.4 | **1573.0** | **2.83x** |

加速随 L 单调上升，正是 O(L²) 项被换掉的特征：优化前那部分是"分解的注意力 + 显式 mask"，
优化后是一个 flash-attention 式的融合 kernel。

---

## 11.4 端到端 image / fused

`profile_gme_image.py`，GPU，p50，`int4_causal` vs `int4_causal_vnomask`
（**只有视觉 IR 不同**，语言 IR 都是 `gme_language_int4_causal.xml`）：

| mode | size | K | 视觉段 前→后 | 端到端 前→后 | 加速 |
| --- | --- | ---: | ---: | ---: | ---: |
| image | 448x448 | 256 | 309.1 → **201.3** | 563.6 → **288.7** | **1.95x** |
| fused | 448x448 | 256 | 307.4 → **201.1** | 547.9 → **290.2** | **1.89x** |
| image | 644x476 | 391 | 655.8 → **338.4** | 831.7 → **464.6** | **1.79x** |
| fused | 644x476 | 391 | 653.4 → **338.0** | 832.0 → **467.4** | **1.78x** |
| image | 672x672 | 576 | 1037.7 → **529.3** | 1408.9 → **691.0** | **2.04x** |
| fused | 672x672 | 576 | 1039.8 → **529.8** | 1418.8 → **694.0** | **2.04x** |
| image | 896x896 | 1024 | 2685.7 → **1158.4** | 2975.2 → **1452.4** | **2.05x** |
| fused | 896x896 | 1024 | 2680.2 → **1165.3** | 2990.1 → **1459.8** | **2.05x** |
| image | 1288x728 | 1196 | 4141.1 → **1456.9** | 4494.4 → **1798.3** | **2.50x** |
| fused | 1288x728 | 1196 | 4142.8 → **1453.7** | 4479.9 → **1789.6** | **2.50x** |

`image` 和 `fused` 几乎完全一致，符合预期：两者走同一个 `_encode_image_batch`，只差
fused 多 24 个文本 token。

端到端加速普遍**高于**视觉段自身的加速（例如 448x448 是 1.95x vs 1.54x）。差额来自
§11.1.1 那一笔：优化后的语言段回到了 72.6 ms（单独测 79.6 ms），而优化前是 238.5 ms。
**这不是语言模型变快了，是它不再被前一次重负载拖到低频档。**

### 11.4.1 第四轮历史 fresh stage split（2026-08-26）

上表是历史 stage-split A/B。为给当前 package 一个可直接复核的阶段口径，
`profile_gme_image.py` 又在同一套 custom core/plugin/Python binding 上运行了
两种 vnomask precision（warmup=3、repeat=5）。下表是 p50，单位为 ms；
它与 §11.6.4 的 public API mean 矩阵是不同统计口径，不能混算。

| mode | size | K | `int8_vnomask` pre / visual / language / e2e | `int4_causal_vnomask` pre / visual / language / e2e |
| --- | --- | ---: | ---: | ---: |
| image | 448x448 | 256 | 13.910 / 199.152 / 95.751 / 306.910 | 13.932 / 199.241 / 72.153 / 285.901 |
| fused | 448x448 | 256 | 14.034 / 198.770 / 98.834 / 309.748 | 14.003 / 199.503 / 75.390 / 289.660 |
| image | 644x476 | 391 | 18.837 / 338.976 / 120.892 / 484.721 | 19.025 / 338.887 / 107.377 / 466.933 |
| fused | 644x476 | 391 | 18.844 / 337.943 / 123.595 / 489.606 | 18.863 / 339.393 / 110.811 / 469.933 |
| image | 672x672 | 576 | 26.510 / 528.106 / 183.425 / 719.559 | 25.479 / 530.430 / 133.534 / 690.998 |
| fused | 672x672 | 576 | 26.314 / 529.987 / 179.306 / 725.751 | 25.831 / 530.080 / 137.014 / 693.851 |
| image | 896x896 | 1024 | 46.444 / 1151.871 / 370.346 / 1506.826 | 41.508 / 1158.582 / 239.387 / 1441.793 |
| fused | 896x896 | 1024 | 50.024 / 1150.413 / 378.664 / 1517.464 | 44.680 / 1158.818 / 243.768 / 1449.996 |
| image | 1288x728 | 1196 | 55.185 / 1443.638 / 379.597 / 1838.345 | 50.647 / 1455.570 / 283.165 / 1797.777 |
| fused | 1288x728 | 1196 | 48.104 / 1443.767 / 387.423 / 1837.780 | 51.398 / 1459.675 / 285.095 / 1799.941 |

原始 stage-split 文件为
[`profile_gme_image_int8_vnomask.json`](../results/r4_rerun_20260826/profile_gme_image_int8_vnomask.json)
和
[`profile_gme_image_int4_causal_vnomask.json`](../results/r4_rerun_20260826/profile_gme_image_int4_causal_vnomask.json)。
第四轮历史 fresh stage 数据显示 mask-free visual 的 visual 段在两种 precision 上都约
为 199/339/529/1155/1450 ms；两者主要差异来自 language 段的 INT8/causal INT4
选择，而不是 visual mask-free 图之间的语义差异。

---

## 11.5 否决：把 SDPA 输入 Transpose 融进 primitive（有实测数字）

### 11.5.1 想法与实现

§11.3.6 之后剩下的最大非 GEMM 项是 `permute_ref__f16`：128 calls / 38.4 ms。按 node 归并
是每层 4 个——`qkv` 的 `permute(1,0,2,3)` 一个（32 次，18.2 ms），Q/K/V 各一个
`transpose(0,1)`（96 次，19.6 ms）；输出那个 transpose 已经被 plugin 的 opt-2 吸收了。

plugin 里 `TransposeSDPAMatcher` 本可以把这三个吸收进 SDPA 的 input order，但它要求
Transpose **直接**喂给 SDPA。上游写的是 `q.transpose(0,1).unsqueeze(0)`，图上是
`Transpose(rank 3) → Unsqueeze → SDPA`，匹配不上。改成 `q.unsqueeze(0).transpose(1,2)`
——**同一个张量**，只是算子顺序变成 `Unsqueeze → Transpose(rank 4) → SDPA`——就能匹配。

实现为 `--fuse-vision-input-transposes`，产物 `gme_visual_fp32_nomask_qkvfused.xml`。
效果确实如预期：`permute_ref__f16` 从 128 calls / 38.4 ms 降到 **32 calls / 18.4 ms**。

### 11.5.2 但它在大图上是负收益

三轮交错 A/B（每轮 warmup 3 / repeat 5，A 与 A+B 交替，应对本机 GPU 双档频率）：

| grid | L | A（仅 mask-free） | A+B（再融 transpose） | Δ |
| --- | ---: | ---: | ---: | ---: |
| 1,48,48 | 2304 | 542.9 | **509.2** | **−33.7** |
| 1,64,64 | 4096 | 1168.6 | 1162.6 | −6.0 |
| 1,24,210 | 5040 | **1577.5** | 1681.3 | **+103.8** |

三轮之间极差 < 1%，不是噪声。按 exec_type 拆开，机制非常清楚：

| grid | L | `permute_ref` Δ | `ocl::sdpa::opt` Δ | 净 |
| --- | ---: | ---: | ---: | ---: |
| 1,48,48 | 2304 | −44.8 | +12.0 | **−31.8** |
| 1,64,64 | 4096 | −79.0 | +79.6 | −6.0 |
| 1,24,210 | 5040 | −97.5 | +206.8 | **+103.8** |

**它拿一笔 O(L) 的拷贝，换了 O(L²) 的 strided 读取。** 省掉的 permute 随 L 线性增长
（−44.8 → −79.0 → −97.5），而 SDPA kernel 因为要按 input order 跨步读 Q/K/V，
每次注意力的代价上升，这部分随 L² 增长（+12.0 → +79.6 → +206.8）。交叉点在 L ≈ 4100。

### 11.5.3 结论：不设默认

本包 `max_image_tokens=1280`，即 L 最大 5120，**最坏情况正好落在它变坏的一侧**。
所以默认导出**不**开这个开关；它作为分析产物保留在
`--fuse-vision-input-transposes` / `03_finalize.py --round4-extras` 之后，
精度 key 为 `int4_causal_vnomask_qkvfused`。

如果部署场景把 `max_image_tokens` 压到 ≤ 1024（L ≤ 4096），它是一笔干净的额外收益
（K=256 上 −18.7 ms，K=576 上 −33.7 ms），可以按业务尺寸自行开启。

---

## 11.6 精度

### 11.6.1 vs PyTorch FP32 golden（`04_validate_python.py`，GPU）

| precision | 视觉 IR | text | query | image | fused |
| --- | --- | ---: | ---: | ---: | ---: |
| `int4` | masked | 0.944538 | 0.955146 | 0.943653 | 0.952074 |
| `int4_vnomask` | mask-free | 0.944538 | 0.955146 | 0.943769 | 0.951327 |
| `int4_causal` | masked | 0.941730 | 0.955173 | 0.945703 | 0.954420 |
| `int4_causal_vnomask` | mask-free | 0.941730 | 0.955173 | 0.943164 | 0.955472 |
| `int4_gs64_causal` | masked | 0.944205 | 0.957576 | 0.941856 | 0.959374 |
| `int4_gs64_causal_vnomask` | mask-free | 0.944205 | 0.957576 | 0.943657 | 0.958768 |
| `int8` | masked | 0.997518 | 0.998764 | 0.989255 | 0.995806 |
| `int8_vnomask` | mask-free | 0.997518 | 0.998764 | 0.987343 | 0.996379 |

`text` / `query` **逐位相同**（语言 IR 没变），这本身就是一个有用的自洽性检查。

`image` / `fused` 的差在 ±0.002 以内，**且符号两边都有**（int4_causal：image −0.0025 但
fused +0.0011；int8：image −0.0019 但 fused +0.0006）。这正是 NNCF 对两个**不同排列**的
源图各自独立跑一遍权重压缩产生的量化噪声——docs/09 §9.2.5 已经量化过这类噪声的量级
（`text_cos_min` ±0.01），比这里的差大 5 倍。

真正的等价性论据是 §11.3.3 的 **FP32 逐位相同**；量化后这点双向小差不能用来否定它，
也不构成精度回退。

### 11.6.2 第四轮阶段推荐（第五轮前）

| 场景 | precision key | 1024-token text | image 448x448 | image 1288x728 |
| --- | --- | ---: | ---: | ---: |
| 第四轮 package 默认 / 精度优先 | **`int8_vnomask`** | — | — | — |
| 第四轮低延迟 / 均衡 | **`int4_causal_vnomask`** | 221.4 | **288.7** | **1798.3** |
| 第四轮更高精度的 INT4 | `int4_gs64_causal_vnomask` | 236.0 | — | — |
| 对照：本轮之前的默认 | `int4_causal` | 221.4 | 563.6 | 4494.4 |

`_vnomask` 系列与它们同名的基准 key **只差视觉塔**，语言 IR 完全相同——这正是让 §11.4
的 A/B 可归因的原因，也意味着 docs/09/10 的所有文本结论**原样成立**。

### 11.6.4 第四轮历史 fresh rerun：text/query、image/fused 和 batch

为避免把历史 stage-split 的 p50 数字误当成当前 package 的完整矩阵，2026-08-26
又使用同一套 custom core/plugin/Python binding 对第四轮阶段 precision 做了独立重测。
原始结果见 [`results/r4_rerun_20260826/`](../results/r4_rerun_20260826/)：

- Python/GPU，`repeat=5`，warm end-to-end；主要统计使用 mean；
- text/query：32、128、512、1024 effective tokens；
- image/fused：448x448、644x476、672x672、896x896、1288x728，以及 1260 visual-token logo；
- batch：text N=1/2/4/8/16/32，image N=1/2/4/8，覆盖 `per_sample`/`padded` 和 `streams=1/auto`；
- batch suite 的 `batch-precision` 每次只能指定一个 precision，因此 INT8/INT4 分别落在两个 JSON 文件中。

text/query fresh mean（ms/call，N=1）：

| precision | 32 text/query | 128 text/query | 512 text/query | 1024 text/query |
| --- | ---: | ---: | ---: | ---: |
| `int8_vnomask` | 44.12 / 43.80 | 54.18 / 54.15 | 135.56 / 135.63 | 279.68 / 280.77 |
| `int4_causal_vnomask` | 32.32 / 31.99 | 41.86 / 41.65 | 119.36 / 119.35 | 226.00 / 226.16 |

image/fused fresh mean（ms/call，N=1）：

| precision | 448x448 / 256K | 644x476 / 391K | 672x672 / 576K | 896x896 / 1024K | 1288x728 / 1196K | logo / 1260K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `int8_vnomask` | 310.53 / 306.94 | 486.84 / 484.11 | 729.97 / 728.56 | 1523.18 / 1521.81 | 1850.57 / 1850.86 | 2176.51 / 2170.73 |
| `int4_causal_vnomask` | 296.37 / 296.43 | 470.46 / 468.91 | 699.25 / 694.68 | 1449.80 / 1449.04 | 1800.29 / 1795.29 | 2110.59 / 2114.97 |

batch fresh mean（Python/GPU，text=128 tokens，image=448x448；单元格为
`ms/call (ms/embedding)`）：

| precision / mode / streams | N=1 | N=2 | N=4 | N=8 | N=16 | N=32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `int8_vnomask` / per_sample / 1 | 54.70 (54.70) | 105.27 (52.64) | 204.46 (51.11) | 402.41 (50.30) | 798.05 (49.88) | 1584.76 (49.52) |
| `int8_vnomask` / padded / 1 | 54.71 (54.71) | 73.12 (36.56) | 131.42 (32.85) | 259.72 (32.47) | 486.51 (30.41) | 1007.78 (31.49) |
| `int4_causal_vnomask` / per_sample / 1 | 40.43 (40.43) | 75.74 (37.87) | 152.07 (38.02) | 287.73 (35.97) | 570.85 (35.68) | 1135.49 (35.48) |
| `int4_causal_vnomask` / padded / 1 | 40.12 (40.12) | 60.82 (30.41) | 119.24 (29.81) | 221.12 (27.64) | 470.79 (29.42) | 890.09 (27.82) |

对应的 image batch mean（只到 N=8）：

| precision / mode / streams | N=1 | N=2 | N=4 | N=8 |
| --- | ---: | ---: | ---: | ---: |
| `int8_vnomask` / per_sample / 1 | 305.54 (305.54) | 605.65 (302.83) | 1199.44 (299.86) | 2382.24 (297.78) |
| `int8_vnomask` / padded / 1 | 306.81 (306.81) | 579.71 (289.86) | 1146.85 (286.71) | 2275.64 (284.46) |
| `int4_causal_vnomask` / per_sample / 1 | 286.99 (286.99) | 574.05 (287.03) | 1143.63 (285.91) | 2263.17 (282.90) |
| `int4_causal_vnomask` / padded / 1 | 288.68 (288.68) | 555.45 (277.72) | 1112.75 (278.19) | 2237.67 (279.71) |

完整的 `streams=auto` 和所有 batch cell 保留在两个 `benchmark_batch_*.json` 中。
在这次 fresh workload 上，GPU `padded` 仍是 text 的主路径；image 的视觉塔逐图执行，
所以 batch 主要摊薄后续 language/request 开销，不能期待 text 那样的吞吐增益。

### 11.6.3 batch 能带来多少性能？（image 上很少，text 上很多）

这是 §11.2.6 的直接推论，但容易被误解成"逐图调用让 batch 失去了意义"，所以单独量化。
公开 API，`int4_causal_vnomask`，GPU，448x448 图 / 32-token 文本：

| N | image ms/call | image **ms/emb** | text ms/call | text **ms/emb** |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 283.8 | **283.8** | 31.4 | **31.4** |
| 2 | 546.1 | 273.0 | 33.8 | 16.9 |
| 4 | 1088.8 | 272.2 | 41.4 | 10.3 |
| 8 | 2191.5 | **273.9** | 57.4 | **7.2** |

- **text：batch 非常有效，N=8 是 4.4x。** 短序列单条跑不满设备，合并成一次 `[B,S]` 调用
  正好补上（这也是 `batch_mode="padded"` 在 GPU 上是默认的原因）。
- **image：batch 基本无效，N=8 只有 3.5%。**

**关键点：image 的 batch 收益低不是因为逐图调用，而是因为一张图就已经把设备打满了。**
`min_image_tokens=256` 是硬下界，任何图片都至少 K=256 / **L=1024**，而 L=1024 的视觉塔
本身就有约 198 ms 的实打实工作量（INT4 GEMM 90 ms + SDPA 34 ms + …）。设备没有空闲算力
留给第二张图去填，所以并发度提不上去；§11.2.6 也说明了打包只会更慢（N=8 量级已是 1.6x+）。

所以 image 路径想快，杠杆**不在 batch**，而在这三处，按收益排序：

1. **本轮的 mask-free 视觉塔**：1.79–2.50x，无损，已落地；
2. **`max_image_tokens`**：时延随 L 超线性，降分辨率是唯一的结构性杠杆——但必须同时验证
   Recall@K / nDCG，不能只看时延（§11.8）；
3. **视觉塔的 INT4 GEMM**（现在占 46%）：与语言侧同一个 oneDNN 问题，尚未定价（§11.8）。

`batch_size` 参数在 image 路径上仍然有意义，但它是**内存/编排**旋钮（分块避免峰值内存），
不是吞吐旋钮。

---

## 11.7 本轮产物

```
gme-Qwen2-VL-2B-Instruct-Running/
  02_export.py                        unmasked_vision_attention() 上下文管理器
                                      --unmasked-vision / --visual-suffix
                                      --fuse-vision-input-transposes（§11.5，分析用）
                                      GmeVisualEncoder 显式取 pooler_output（§11.3.2 的 bug 修复）
                                      --stage all 现在把视觉 IR 也导两遍
  03_finalize.py                      压缩 mask-free 视觉 IR；--round4-extras 产出 qkvfused 对照
                                      新增 precision key：int4_vnomask / int4_causal_vnomask /
                                      int4_gs64_causal_vnomask / int8_vnomask
  ov_models/.../gme_visual_fp32_nomask.xml            §11.3 采纳（FP32 源）
  ov_models/.../gme_visual_int4_nomask.xml            §11.3 采纳
  ov_models/.../gme_visual_int8_nomask.xml            §11.3 采纳
  ov_models/.../gme_visual_fp32_nomask_qkvfused.xml   §11.5 否决项对照
  ov_models/.../gme_visual_int4_nomask_qkvfused.xml   §11.5 否决项对照
  artifacts/openvino_validation_round4_vision.json    §11.6.1

gme-qwen2-vl-int4-hotspot-optimization/
  profile_visual_ops.py               新增：视觉 IR 的 per-op GPU profiling（--ir 可直接指定 IR）
  profile_gme_image.py                新增：image/fused 四段拆分
  check_vision_equivalence.py         新增：FP32 逐位等价性（§11.3.3）
  check_vision_batch_contract.py      新增：单图契约（§11.3.5）
  check_vision_mask_necessity.py      新增：mask 必要性的完整枚举 + PyTorch 等价性（§11.2.2–11.2.5）
  results/r4_20260826/                历史第四轮分析原始结果（stage split / 等价性）
  results/r4_rerun_20260826/          第四轮 vnomask fresh text/image/batch 原始结果
```

**GPU plugin 交付面：本轮没有新增任何 plugin 改动。** 与第三轮一样，收益全在导出侧。

---

## 11.8 仍然开放（下一轮的起点）

| 项 | 现状 | 预估 |
| --- | --- | --- |
| **视觉塔的 `jit:gemm:any__i8`** | mask 修掉后占比从 30% 升到 **46%**（L=1024 上 90.2 ms / 130 calls） | ~~opt-3 的 `N≥4K` 规则在视觉塔维度上不触发~~ —— **这条已被 docs/12 §12.3.1 证伪**：真正的 guard 是 `N >= 4*K`，`mlp.fc1`（K=1280 / N=5120）恰好合格；挡住它的是 `prim_input_size != 3`（视觉塔的 Linear 输入是 rank-2）。docs/12 用整塔 rank-3 解锁了它，fc1 快 28–32% |
| **剩下的 32 个 `permute_ref`（18.4 ms）** | `qkv` 的 `permute(1,0,2,3)`，§11.5 的方案不碰它 | 可考虑改成在 head 维上切三份，避免物化整个置换张量；需重新验证数值 |
| **`ocl::sdpa::opt` 本身（L=1024 上 33.8 ms，L=5040 上 747 ms）** | 已是优化 kernel，但视觉塔是全注意力，没有 causal 那样的一半可省 | 唯一的结构性杠杆是降 `max_image_tokens`；需同时验证 Recall@K/nDCG，不能只看时延 |
| **§11.5 的 L 依赖** | 交叉点实测在 L ≈ 4100 | 若 plugin 能让 SDPA 对 strided 输入不掉性能，A+B 就能全区间为正；属 plugin/kernel 侧 |
| **`int8_vnomask` 的 image/fused 时延** | 已在 §11.6.4 完成 fresh matrix：448x448 image 310.53 ms，1288x728 image 1850.57 ms | 后续只需按目标业务图片分布复测，不再把该项视为数据缺口 |
| **§11.5 的 `qkvfused` 与 docs/12 的 `headfirst` 之别** | §11.5 删掉 96 次转置的办法是让 SDPA 读 strided（O(L²) 代价，大图翻盘）；docs/12 §12.2 改成把四趟搬运并成一趟，SDPA 仍读连续内存，**全尺寸正收益** | 已落地，见 docs/12 |
| **§11.2.7 的跨设备外推** | 只在本机（f≈1.0）校验过模型，**没有第二台设备的实测** | 若迁到 dGPU/更强 iGPU，应重跑 §11.2.6 的 packed/split 表与 §11.2.7 的并发表；若逐图利用率 < 66%（N=4），需重新评估「并发 vs 真 batch 维视觉 IR」 |
| **真 batch 维的视觉 IR** | 未实现。尺寸统一时它严格优于打包（占用率同、算力省） | 需要绕开 `rot_pos_emb` 的 Python 循环；只对「同尺寸批量」场景有意义，属跨设备迁移时的备选（§11.2.7） |
| GPU 降频对相邻推理的影响（§11.1.1） | 已量化（S=281 上 79.6 → 238.5 ms） | 这解释了历史 image/fused 数字为何普遍偏高；重测 PERFORMANCE_REPORT §6.3 时应按本轮口径分段 |

---

## 11.9 复现

下面第 1--3 步重建的是历史第四轮分析产物目录 `results/r4_20260826/`；当前
第四轮 fresh 性能矩阵和阶段拆分已经保存在 `results/r4_rerun_20260826/`，无需覆盖历史结果。

```bash
cd gme-Qwen2-VL-2B-Instruct-Running

# 1. 导出 mask-free 视觉 IR（--stage all 现在默认也会导它）
python 02_export.py --stage visual --unmasked-vision
python 03_finalize.py                       # 产出 int4/int8 的 _nomask 与 _vnomask precision key

# 2. 证明是恒等变换（FP32、CPU、逐位）
cd gme-qwen2-vl-int4-hotspot-optimization
python check_vision_equivalence.py --output results/r4_20260826/r4_vision_equivalence_fp32.json
python check_vision_batch_contract.py \
       --precisions int4_causal,int4_causal_vnomask,int8,int8_vnomask \
       --output results/r4_20260826/r4_vision_batch_contract.json
# mask 到底有没有用（§11.2.2–11.2.4）：三种输入形态 + PyTorch 打包/逐图/串味对比
python check_vision_mask_necessity.py --output results/r4_20260826/r4_mask_necessity.json

# 3. 视觉 IR 的算子级 A/B（务必先 source setupvars.sh）
source "$OPENVINO_ROOT/install/setupvars.sh"
for g in 1,32,32 1,34,46 1,48,48 1,64,64 1,52,92 1,24,210; do
  for ir in gme_visual_int4.xml gme_visual_int4_nomask.xml; do
    python profile_visual_ops.py --ir $ir --grid $g --warmup 3 --repeat 4 \
           --output "results/r4_20260826/vis_${ir%.xml}_${g//,/_}.json"
  done
done

# 4. 端到端 image/fused
for p in int4_causal int4_causal_vnomask; do
  python profile_gme_image.py --precision $p --warmup 3 --repeat 5 \
         --output "results/r4_20260826/e2e_${p}.json"
done

# 5. 精度
python ../04_validate_python.py --devices GPU \
       --precisions int4,int4_causal,int4_vnomask,int4_causal_vnomask,int8,int8_vnomask \
       --output ../artifacts/openvino_validation_round4_vision.json

# 6. §11.5 的否决项（可选，+0.4 GB）
python ../02_export.py --stage visual --unmasked-vision --fuse-vision-input-transposes
python ../03_finalize.py --round4-extras
```

---

## 11.10 一句话总结

**image/fused 的时延一直不在语言 IR 上，而在视觉塔；视觉塔慢是因为一张对本包恒为全 True 的
rank-3 BOOL mask 让 32 层 SDPA 全部退化成手工分解的 MatMul+Softmax。删掉它是 FP32 逐位恒等
变换，不需要动 plugin，端到端 image/fused 快 1.79–2.50x，且收益随分辨率上升。**

而 image 路径的吞吐不要指望 batch（N=8 仅 3.5%）：`min_image_tokens=256` 让每张图至少
L=1024，单图就已经打满这块 32 EU 核显。想再快只有降 `max_image_tokens`，或者去啃视觉塔
那 46% 的 INT4 GEMM。
