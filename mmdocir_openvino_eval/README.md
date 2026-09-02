# MMDocIR OpenVINO Retrieval Evaluation

本目录用于评估 GME-2B OpenVINO FP32/INT8/INT4 在 MMDocIR Evaluation set 上的真实检索质量和性能。代码将数据、模型 adapter、指标、缓存和报告解耦，后续接入 Qwen3-VL-Embedding-2B 时不需要修改 MMDocIR 数据或指标逻辑。

## 固定评测口径

- Dataset: `MMDocIR/MMDocIR_Evaluation_Dataset`，313 documents / 1,658 queries / 20,395 pages / 170,338 layouts。
- Page input: `image_binary`，调用 image embedding。
- Layout input: `image`/`table` 调用 fused embedding (`vlm_text` + `image_binary`)；其他类型调用 text embedding (`text`)。
- Query input: text embedding，`is_query=True`。默认 instruction 与导出包保持一致；可用 `--query-instruction` 显式设置。
- Visual DynamicQuantize: 用 `--visual-dq-threshold 0`（或 campaign 的
  `VISUAL_DQ_THRESHOLD=0`）启用视觉 MVN-to-DQ 融合；该选项只作用于 visual IR，
  不改变 query/text language IR。该值会写入 backend metadata 和 embedding cache
  fingerprint，不能与未启用融合的 cache 混用。
- Matching: L2-normalized embedding 的 cosine similarity。
- Close-domain: query 只搜索所属 document 的 page/layout。
- Open-domain: query 搜索全部 313 documents 的 page/layout。
- Page metrics: macro Recall@5/10、binary nDCG@5/10，与官方 `metric_eval.py` 一致。
- Layout Recall: top-k candidate bbox 与 ground-truth bbox 的相交面积之和 / ground-truth 总面积，与官方实现一致。候选重叠时数值理论上可能超过 1，不做静默截断。
- Layout nDCG: 官方代码没有实现该项。本目录以候选框和真值框的相交面积作为 graded relevance，在相同候选 scope 内计算 nDCG，并在结果中明确标识该定义。

bbox 计算遵循官方代码：四元组前两项为起点、后两项为终点进行相交面积计算。官方 README 称其为 `[x1,y1,x2,y2]`，代码注释又称 `[top,left,bottom,right]`；两种命名不影响同序坐标的面积运算。

## 环境

必须让 Python binding、OpenVINO core 与 GPU plugin 来自同一份 `benchmark/openvino` custom build：

```bash
ROOT=/home/chaofan/model_benchmark
OV=$ROOT/benchmark/openvino
GME=$ROOT/gme-Qwen2-VL-2B-Instruct-Running
PYTHON=$HOME/envs/qwen_env/bin/python3

cd $GME/mmdocir_openvino_eval
$PYTHON -m pip install -r requirements.txt
export PYTHONPATH="$OV/bin/intel64/Release/python:$GME/mmdocir_openvino_eval:$GME"
export LD_LIBRARY_PATH="$OV/bin/intel64/Release:$OV/temp/Linux_x86_64/tbb/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OPENVINO_LIB_PATHS="$OV/bin/intel64/Release"
export OPENVINO_PLUGIN_PATH="$OV/bin/intel64/Release"
```

结果 `system.openvino_module` 必须指向 `benchmark/openvino/bin/intel64/Release/python/openvino`，否则结果不能与本项目优化 backend 对应。

## 下载数据

只下载评测必需的 annotation 和两份 parquet，不下载 PDF/RAR 辅助文件：

```bash
$PYTHON download_dataset.py
```

默认位置是 `/home/chaofan/datasets/MMDocIR_Evaluation_Dataset`。下载器固定 Hugging Face revision，并写入 `manifest.json` 记录 revision 与文件字节数。当前两份 parquet 合计约 3.78 GiB。

## 样本数量与推荐子集

CLI 提供两类限制：

- `--max-samples N`（等价于 `--max-queries N`）：取 annotation 顺序中的前 N 条 query。适合验证代码和显存，不适合质量结论；例如前 12 条只覆盖 2/10 domains。
- `--subset-file FILE`：按固定 global query ID 选择，并自动编码这些 query 所属文档。适合三精度公平对比，manifest 路径、SHA256、query IDs 和 domain 数量都会写入 summary/cache fingerprint。

本目录提供两个固定子集：

| 子集 | Queries | Domains | Documents | Pages | Layouts | batch=1 三精度估算 | 用途 |
|---|---:|---:|---:|---:|---:|---:|---|
| `balanced_100` | 100 | 10 x 10 | 19 | 328 | 5,898 | 约 2.6 小时 | 快速预筛 |
| `balanced_200` | 200 | 10 x 20 | 34 | 607 | 7,956 | 约 4 小时（旧统一 batch=1 估算） | **推荐的周末初步对比** |
| full | 1,658 | 10（不平衡） | 313 | 20,395 | 170,338 | 约 4.4 天 | 最终结论 |

推荐 `balanced_200`：相比 100-query，二值比例指标在最坏 $p=0.5$ 情况下的抽样标准误由约 5.0pp 降至 3.5pp，而预计总耗时只增加约 1.4 小时。它适合识别明显或系统性的量化退化，但不能证明小于数个百分点的等价性，最终发布结论仍需 full set。

可在不加载模型的情况下检查选择：

```bash
SUBSET_FILE="$PWD/subsets/balanced_200.json" ./run_campaign.sh --dry-run
MAX_SAMPLES=20 ./run_campaign.sh --dry-run
```

如需重建 manifest：

```bash
$PYTHON make_balanced_subset.py --queries-per-domain 20 --output subsets/balanced_200.json
```

## 首轮 smoke test

先以一个 document、batch=1 验证真实 schema、四条输入路径和 close/open 指标：

```bash
$PYTHON run_benchmark.py \
  --precisions fp32,int8_causal_vopt,int4_causal_vopt \
  --batch-size 1 \
  --max-documents 1
```

## 推荐子集：三精度整体运行

```bash
SUBSET_FILE="$PWD/subsets/balanced_200.json" \
  VISUAL_DQ_THRESHOLD=0 ONEDNN_PRIMITIVE_CACHE_CAPACITY=0 ./run_campaign.sh
```

默认使用实测的 `recommended` batch profile；设置 `VISUAL_DQ_THRESHOLD=0` 时，输出为
`results/balanced_200_recommended_vdq0/summary.json`，包含 FP32 baseline 及
INT8/INT4 自动 delta。该命令一次完成 close-domain 和 subset open-domain；后者只在所选
34 documents 中开放检索，不等同于完整 313-document open-domain。

### 当前 balanced_200 质量结果

`balanced_200` 已于 2026-09-02 完成三档质量评测：200 queries、10 个 domain（每个 20 条）、
34 documents、607 pages、7,956 layouts。结果文件为
[`results/balanced_200_recommended_vdq0/combined/summary.json`](results/balanced_200_recommended_vdq0/combined/summary.json)，
详细分析见 [`REPORT.md`](REPORT.md) 和
[`round-9 report`](../gme-qwen2-vl-int4-hotspot-optimization/docs/16_round9_final_rebenchmark.md)。

下表为 `page/layout Recall@5` 和 `page/layout nDCG@5`，数值均为同一 FP32 baseline
下的 close/open 结果：

| precision | scope | page R@5 | page nDCG@5 | layout R@5 | layout nDCG@5 |
|---|---|---:|---:|---:|---:|
| FP32 | close | 0.8650 | 0.7699 | 0.5995 | 0.4853 |
| INT8 | close | 0.8663 | 0.7671 | 0.5824 | 0.4728 |
| INT4 | close | 0.8529 | 0.7580 | 0.5661 | 0.4657 |
| FP32 | open | 0.7571 | 0.6678 | 0.5208 | 0.4232 |
| INT8 | open | 0.7513 | 0.6633 | 0.5267 | 0.4208 |
| INT4 | open | 0.7425 | 0.6519 | 0.4825 | 0.3992 |

相对 FP32，INT8 的 page 指标最大下降约 `0.58` percentage points，INT4 的 page R@5
下降 `1.21/1.46` points（close/open），layout R@5 下降 `3.35/3.83` points。
因此当前质量结论是 INT8 更接近 FP32，INT4 仍需业务检索指标确认后再部署。

本轮不报告 balanced_200 的 precision 间耗时或 throughput 对比：INT4 为绕过 oneDNN
primitive-cache 死锁而使用了 `ONEDNN_PRIMITIVE_CACHE_CAPACITY=0`，运行时间显著增加，
与 FP32/INT8 的 elapsed time、throughput、ETA 不具备可比性。时间字段仅作为运行记录保留。

## 推荐子集：三精度分别运行

分别运行便于单独保存日志、失败后按 precision 恢复：

```bash
SUBSET="$PWD/subsets/balanced_200.json"

SUBSET_FILE="$SUBSET" PRECISIONS=fp32 \
  VISUAL_DQ_THRESHOLD=0 \
  OUTPUT_DIR="$PWD/results/balanced_200_recommended_vdq0/fp32" ./run_campaign.sh

SUBSET_FILE="$SUBSET" PRECISIONS=int8_causal_vopt \
  VISUAL_DQ_THRESHOLD=0 \
  OUTPUT_DIR="$PWD/results/balanced_200_recommended_vdq0/int8_causal_vopt" ./run_campaign.sh

SUBSET_FILE="$SUBSET" PRECISIONS=int4_causal_vopt \
  VISUAL_DQ_THRESHOLD=0 ONEDNN_PRIMITIVE_CACHE_CAPACITY=0 \
  OUTPUT_DIR="$PWD/results/balanced_200_recommended_vdq0/int4_causal_vopt" ./run_campaign.sh

# 利用上述 cache 生成同一 summary 中的 FP32/INT8/INT4 delta：
SUBSET_FILE="$SUBSET" PRECISIONS=fp32,int8_causal_vopt,int4_causal_vopt \
  VISUAL_DQ_THRESHOLD=0 ONEDNN_PRIMITIVE_CACHE_CAPACITY=0 \
  OUTPUT_DIR="$PWD/results/balanced_200_recommended_vdq0/combined" ./run_campaign.sh
```

## 周末自动套件

[run_weekend_suite.sh](run_weekend_suite.sh) 已符合“分别跑三精度 + 生成整体对比”的要求。设置
`VISUAL_DQ_THRESHOLD=0` 和 `ONEDNN_PRIMITIVE_CACHE_CAPACITY=0` 后运行 `balanced_200` 和实测 `recommended` profile，顺序为
FP32 -> INT8 -> INT4 -> combined；combined 会复用前三步的 page/layout cache：

```bash
VISUAL_DQ_THRESHOLD=0 ONEDNN_PRIMITIVE_CACHE_CAPACITY=0 ./run_weekend_suite.sh
```

立即放到后台：

```bash
nohup ./run_weekend_suite.sh > weekend_suite.out 2>&1 &
echo $! > weekend_suite.pid
```

使用 systemd 安排周六 00:30 启动（系统支持 user systemd 时）：

```bash
systemd-run --user --collect \
  --unit=mmdocir-balanced-200 \
  --on-calendar='Sat 00:30' \
  "$PWD/run_weekend_suite.sh"

systemctl --user status mmdocir-balanced-200.timer
journalctl --user -u mmdocir-balanced-200.service -f
```

## 全量评测：整体运行

```bash
# INT4 需要 capacity=0 绕过 oneDNN primitive-cache 死锁；本轮不比较耗时。
ONEDNN_PRIMITIVE_CACHE_CAPACITY=0 ./run_campaign.sh
```

脚本固定加载本仓库 `benchmark/openvino` custom build，并按 `close,open` 顺序评测。编码索引对两个 scope 完全复用。可通过环境变量覆盖：

```bash
BATCH_PROFILE=uniform BATCH_SIZE=4 \
  VISUAL_DQ_THRESHOLD=0 ONEDNN_PRIMITIVE_CACHE_CAPACITY=0 \
  PRECISIONS=int8_causal_vopt,int4_causal_vopt ./run_campaign.sh
```

`run_benchmark.py` 也可直接接收 `--visual-dq-threshold 0`。campaign wrapper 会把该
值加入输出目录名（例如 `_vdq0`）；backend metadata、package identity 和执行 batch
共同参与 page/layout cache fingerprint，因此不同 threshold 配置会生成独立 cache。

## 全量评测：三精度分别运行

```bash
PRECISIONS=fp32 OUTPUT_DIR="$PWD/results/full_recommended/fp32" ./run_campaign.sh
PRECISIONS=int8_causal_vopt OUTPUT_DIR="$PWD/results/full_recommended/int8_causal_vopt" ./run_campaign.sh
ONEDNN_PRIMITIVE_CACHE_CAPACITY=0 PRECISIONS=int4_causal_vopt \
  OUTPUT_DIR="$PWD/results/full_recommended/int4_causal_vopt" ./run_campaign.sh

ONEDNN_PRIMITIVE_CACHE_CAPACITY=0 PRECISIONS=fp32,int8_causal_vopt,int4_causal_vopt \
  OUTPUT_DIR="$PWD/results/full_recommended/combined" ./run_campaign.sh
```

也可让自动套件切换到 full set：

```bash
VISUAL_DQ_THRESHOLD=0 ONEDNN_PRIMITIVE_CACHE_CAPACITY=0 FULL_DATASET=1 ./run_weekend_suite.sh
```

中断后执行同一命令会复用已经完整发布的 precision/page/layout cache。正在写入的向量先使用 `.tmp` 文件，未完成时不会被当作有效 cache。

完整向量按配置指纹缓存在 `cache/`。完成的 page/layout cache 会复用；不同 precision、分路径 batch、image token 配置或数据选择不会互相污染。每个 precision 完成后都会 checkpoint 当前 output directory 的 `summary.json`，逐 query 排名写入 `per_query_*.jsonl`。

FP32 是质量 baseline。最终报告应同时给出 INT8/INT4 相对 FP32 的 Recall/nDCG 绝对差与相对差，不能只与 A100 表格比较。

## Batch 选择（历史配置；当前不作 INT4 时间比较）

下面的 batch 选择和 throughput 数字来自 workaround 之前的历史测量，只用于说明
`recommended` profile 的配置来源。当前 balanced_200 的 INT4 运行使用
`ONEDNN_PRIMITIVE_CACHE_CAPACITY=0`，因此本轮不以这些时间或 throughput 比较 INT4、
INT8 和 FP32。

GPU 的最优 batch 取决于输入路径，不能对 query、整页图像和 layout 强制使用同一个值。当前默认 `--batch-profile recommended` 按 precision 自动选择：

| Precision | Query | Page image | Text/equation layout | Image/table fused layout |
|---|---:|---:|---:|---:|
| FP32 | **8** | **1** | **1** | **1** |
| INT8 optimized | **8** | **1** | **4** | **1** |
| INT4 optimized | **8** | **1** | **4** | **1** |

依据如下：

- 历史 200 条 `balanced_200` query、每档 3 次：batch 1/2/4/8 的 query throughput 分别为 FP32 `5.83/6.09/7.83/8.16`，INT8 `18.62/29.05/49.32/65.60`，INT4 `20.10/35.03/55.63/73.05 queries/s`，因此原始 `recommended` query 配置选 8。
- 首份真实文档（23 pages、138 text layouts、15 fused layouts）：FP32 batch=2 相比 1，page throughput `0.159->0.121/s`、text `3.53->2.94/s`、fused `0.112->0.0467/s`，全部退化，因此 FP32 document 路径选 1。
- 历史 INT8 text layout batch 1/2/4/8 为 `15.63/23.28/26.23/22.26 layouts/s`；历史 INT4 为 `17.72/23.54/26.20/24.00 layouts/s`，因此原始量化 text layout 配置选 4。
- INT8/INT4 page 在 batch=1/2 基本持平，但 fused batch=2 分别下降约 6%/9%，视觉路径选择更稳且显存更低的 1。
- 所有实测 profile 的 page/layout Recall 和 nDCG 与统一 batch=1 完全一致。

默认命令无需传 batch 参数：

```bash
./run_campaign.sh
./run_weekend_suite.sh
```

需要复现实验或手工覆盖时：

```bash
# 所有路径强制同一 batch
BATCH_PROFILE=uniform BATCH_SIZE=2 ./run_campaign.sh

# 分路径覆盖；未指定项仍取 recommended profile
./run_campaign.sh \
  --query-batch-size 8 \
  --page-batch-size 1 \
  --text-layout-batch-size 4 \
  --fused-layout-batch-size 1

# 一文档统一 batch sweep（默认 1,2,4,8）
./run_batch_sweep.sh
```

吞吐比较使用：

- `encoding.page.pages_per_second`
- `encoding.layout.modalities.text.model_only_layouts_per_second`
- `encoding.layout.modalities.fused.model_only_layouts_per_second`
- `encoding.query.average_per_query_ms`

所有分路径 batch 都是 cache fingerprint 的一部分，因此 batch sweep 会重新编码。不要用 cache-hit run 的 wall time 比较吞吐。实测原始结果见 `results/batch_sweep/`；200-query 重复结果见 `results/batch_sweep/query_200.json`。

每种输入同时记录包含首调用的端到端结果和 `steady_state_*` 字段。后者排除该模态第一个 batch，适合比较稳定运行时性能；前者保留实际一次性离线索引任务会承担的冷启动成本。

## INT4 挂死：根因与绕过

`ONEDNN_PRIMITIVE_CACHE_CAPACITY=0` 可绕过这个挂死，但会显著增加 layout 阶段时间；
本报告不使用 workaround 前后的倍数作为 precision 间性能结论。

**症状**：INT4 在 layout 阶段停住，一个 GPU plugin 线程 100% 占一个核，永不返回。2026-08-29
挂了 4 天 11 小时，2026-09-02 复现两次，都在 layout 开始后约 10 分钟。

**根因**：卡在 `std::_Rb_tree_increment` 的 `while (x = x->_M_left)` 里——红黑树的 left 链上
有一个 96 节点的环。节点内容是垃圾（`_M_color` 读出 949495552，合法值只有 0/1；`parent=0xb1`），
是已被释放又被 jemalloc 复用的内存。调用方反汇编对应 oneDNN 的：

```cpp
// thirdparty/onednn_gpu/src/common/primitive_attr_quant.cpp:84
size_t quant_entries_t::get_hash() const {
    for (const auto &e : entries_) {          // std::map<int, quant_entry_t>
        seed = hash_combine(seed, e.first);
        seed = hash_combine(seed, e.second.get_hash());
    }
}
```

即 `primitive_attr_t` 的 scales / zero-points 表——**INT4 非对称权重反量化必走的路径**，
经 `get_attr_hash()` → `std::hash<primitive_hashing::key_t>` 被调用。而这个 key 只存裸指针：

```cpp
// thirdparty/onednn_gpu/src/common/primitive_hashing.hpp:53
mutable const primitive_attr_t *attr_;
```

`std::hash<key_t>::operator()` 不是 `noexcept`，libstdc++ 因此缓存哈希值、rehash 不重算——
所以哈希**只在调用方传入的 probe key 上计算**，悬垂的是 cldnn 侧传进去的 attr。cldnn 把它存在
`program_node::onednn_attrs`（`std::shared_ptr<dnnl::primitive_attr>`）里，而动态 shape 的
impl 构建跑在一个独立的异步编译线程上（`program::make_compilation_context()` 里
`_num_async_build_threads = 1`），与推理线程并发访问同一个 shared_ptr。

哈希是在 `get_or_add()` 持有**全局 rw_mutex 读锁**时算的，所以这个线程一旦转不出来，所有其他
oneDNN 使用者都会永久阻塞——这解释了主线程为什么停在 `ov::InferRequest::wait()`。

**为什么 capacity=0 有效**：`lru_cache_t::get()` 和 `get_or_add()` 在 `capacity_ == 0` 时
在任何 map 操作之前就返回，`std::hash<key_t>` 根本不会被调用，那段代码不可达。

**排除掉的方案**：

* 调大 capacity 无效——哈希在每次 probe 时都会算，与是否淘汰无关。
* `disable_async_compilation`（真正对症）在本 build 不可用：`ENABLE_DEBUG_CAPS=OFF`，
  `OV_CONFIG_DEBUG_OPTION` 被整个编译掉了。要用得重编 OpenVINO。
* `compilation_num_threads=1` 无效——异步编译上下文本来就固定 1 个线程，竞争发生在它和推理
  线程之间。

**已验证**：balanced_200 INT4 全量 7956 layouts 用 `capacity=0` 跑完，26 分 02 秒，无挂死
（越过了上次卡死的 6684）。质量指标排序正常（close page nDCG@10：FP32 0.7968 > INT8 0.7932
> INT4 0.7802）。同一份 153-layout 输入，`capacity=0` 与默认配置产出的 embedding **逐位相同**
（max abs diff = 0.0），所以**数值完全不受影响，只影响吞吐**。

**性能代价与可比性**：同一份 153-layout 输入，默认 26.6s vs `capacity=0` 46.8s，慢 1.76 倍；
balanced_200 layout 阶段 1562.5s，而挂死那次在默认配置下实测约 11 layouts/s（外推约 726s）。
**因此：开了这个旋钮的 run 与没开的 run 之间，吞吐/延迟数字不可比。** 要横向比较三精度性能，
三者必须用同一个设置重跑。`summary.json` 里 `system.onednn_primitive_cache_capacity` 以及
每个阶段 performance 字典里的同名字段会记下实际取值（`null` = 该阶段用的 oneDNN 默认值，或
产出于这个字段存在之前）；注意 `cache_hit: true` 的阶段报的是**当初产出它的那次 run** 的吞吐。

## 挂死检测与断点续跑

2026-08-29 的 `balanced_200_recommended_vdq0` INT4 run 在 layout 阶段卡死了 4 天 11 小时：
GPU plugin 的一个工作线程在 oneDNN primitive cache 的 key 哈希里遍历一块已被释放的
`std::map`（`quant_entries_t::get_hash()`，INT4 非对称反量化的 scales/zero-points 表），
红黑树的 `_M_left` 链上有一个 96 节点的环，`std::_Rb_tree_increment` 永远退不出来。
这是上游 OpenVINO/oneDNN 的并发缺陷，概率性触发，本仓库无法修复；能修的是**它当时完全
不可见**——阶段开始后不打印任何东西，挂死和“跑得慢”长得一模一样，而且重启要从 0 重编码。

因此编码阶段现在有三层保护，默认全部开启：

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--progress-interval-seconds` | `30` | 阶段内心跳，打印 `已完成/总数`、已用时、ETA；`0` 关闭 |
| `--call-timeout-seconds` | `1800` | 单次 model call 超时看门狗；`0` 关闭 |
| `--checkpoint-interval-seconds` | `120` | 部分向量 cache 的落盘间隔；`0` 表示每批都落盘 |
| `--resume` | 关闭 | 从 checkpoint 继续，而不是重编码 |

看门狗只对**单次调用**计时（阶段本身可以跑几小时，单次调用超过几分钟就已经不正常了；
实测最慢的一次是 FP32 fused layout 的 56 秒）。超时后它 dump 所有 Python 线程栈并以
**exit code 86** 结束进程——卡死的 OpenVINO 调用无法从 Python 取消，只能杀进程。栈里
真正卡住的是原生 plugin 线程，要看它得用：

```bash
gdb -p <pid> -batch -ex 'thread apply all bt'
```

断点续跑：崩溃/被杀之后 `cache/` 里会留下 `*.npy.tmp` 和 `*.progress.json`，加 `--resume`
即可从 watermark 继续：

```bash
PRECISIONS=int4_causal_vopt VISUAL_DQ_THRESHOLD=0 ./run_campaign.sh --resume
```

watermark 是**已完成的最长前缀**而不是完成计数——layout 阶段的 text 队列和 fused 队列各自
独立 flush，位置不按顺序完成，所以 watermark 最多比真实完成数落后各一个 batch。已验证续跑
产物与不中断跑完的产物**逐位相同**。

两点注意：

* 续跑后 `summary.json` 里该阶段的 `resumed_from` 会记下起点，吞吐/延迟统计**只覆盖续跑
  部分**。要拿干净的性能数字就别用 `--resume`，重跑。
* 每次启动会删除没有 checkpoint 的孤儿 `*.npy.tmp`（这类文件之前累积了 4 个、1 GB 稀疏
  文件）；有 checkpoint 的会保留并提示可以 `--resume`。已完整落盘的 `.npy` cache 优先级
  高于部分 cache。

## 输出指标

`summary.json` 记录：模型加载、每个 precision 的实际 `execution_batches`、OpenVINO 路径、数据 revision/selection、query latency、page throughput、text/fused layout throughput、向量大小、进程 RSS、OpenVINO GPU memory statistics、close/open page/layout Recall/nDCG，以及 CPU cosine score/top-k 的 mean/p50/p95/min/max。资源峰值是在每个 inference call 前后采样到的 observed peak，调用内部瞬时峰值可能更高。

当前 Python 环境的 PyTorch 是 CUDA build，`torch.xpu` 不可用，因此 GPU-resident match latency 标记为 unavailable；CPU match latency仍完整记录。embedding 编码本身使用 OpenVINO GPU，不受此限制。

## 接入新模型

实现 `mmdocir_eval.backend.EmbeddingBackend` 的四个方法（也支持不继承基类但具有相同属性/方法的对象），然后用 `--backend package.module:factory` 加载。adapter 负责模型特有的 prompt 和 preprocessing；数据范围、cache、cosine matching 与指标保持不变。factory 只会收到其签名声明的参数；声明 `**kwargs` 时则收到完整内置参数集。

## 验证

```bash
PYTHONPATH=. $PYTHON -m unittest discover -s tests -v
$PYTHON -m py_compile run_benchmark.py download_dataset.py mmdocir_eval/*.py
```

官方参考实现固定在 `/home/chaofan/datasets/MMDocIR_official`，本次核对 revision 为 `394d84dbda393a1abbdddddc265d60b74f52dcc3`。

本机 batch=1 smoke 外推的三精度串行编码时间约 4.4 天（FP32 约 85 小时、INT8 约 11 小时、INT4 约 10 小时）。这是 workaround 之前的历史容量规划估算，不是当前 INT4 的耗时依据；full 质量评测仍需单独安排长时间 GPU 任务。