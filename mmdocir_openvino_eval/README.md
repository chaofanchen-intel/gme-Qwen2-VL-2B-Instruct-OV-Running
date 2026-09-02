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
  VISUAL_DQ_THRESHOLD=0 ./run_campaign.sh
```

默认使用实测的 `recommended` batch profile；设置 `VISUAL_DQ_THRESHOLD=0` 时，输出为
`results/balanced_200_recommended_vdq0/summary.json`，包含 FP32 baseline 及
INT8/INT4 自动 delta。该命令一次完成 close-domain 和 subset open-domain；后者只在所选
34 documents 中开放检索，不等同于完整 313-document open-domain。

## 推荐子集：三精度分别运行

分别运行便于单独保存日志、失败后按 precision 恢复：

```bash
SUBSET="$PWD/subsets/balanced_200.json"

SUBSET_FILE="$SUBSET" PRECISIONS=fp32 \
  OUTPUT_DIR="$PWD/results/balanced_200_recommended/fp32" ./run_campaign.sh

SUBSET_FILE="$SUBSET" PRECISIONS=int8_causal_vopt \
  OUTPUT_DIR="$PWD/results/balanced_200_recommended/int8_causal_vopt" ./run_campaign.sh

SUBSET_FILE="$SUBSET" PRECISIONS=int4_causal_vopt \
  OUTPUT_DIR="$PWD/results/balanced_200_recommended/int4_causal_vopt" ./run_campaign.sh

# 利用上述 cache 生成同一 summary 中的 FP32/INT8/INT4 delta：
SUBSET_FILE="$SUBSET" PRECISIONS=fp32,int8_causal_vopt,int4_causal_vopt \
  VISUAL_DQ_THRESHOLD=0 \
  OUTPUT_DIR="$PWD/results/balanced_200_recommended_vdq0/combined" ./run_campaign.sh
```

## 周末自动套件

[run_weekend_suite.sh](run_weekend_suite.sh) 已符合“分别跑三精度 + 生成整体对比”的要求。设置
`VISUAL_DQ_THRESHOLD=0` 后运行 `balanced_200` 和实测 `recommended` profile，顺序为
FP32 -> INT8 -> INT4 -> combined；combined 会复用前三步的 page/layout cache：

```bash
VISUAL_DQ_THRESHOLD=0 ./run_weekend_suite.sh
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
./run_campaign.sh
```

脚本固定加载本仓库 `benchmark/openvino` custom build，并按 `close,open` 顺序评测。编码索引对两个 scope 完全复用。可通过环境变量覆盖：

```bash
BATCH_PROFILE=uniform BATCH_SIZE=4 \
  VISUAL_DQ_THRESHOLD=0 \
  PRECISIONS=int8_causal_vopt,int4_causal_vopt ./run_campaign.sh
```

`run_benchmark.py` 也可直接接收 `--visual-dq-threshold 0`。campaign wrapper 会把该
值加入输出目录名（例如 `_vdq0`）；backend metadata、package identity 和执行 batch
共同参与 page/layout cache fingerprint，因此不同 threshold 配置会生成独立 cache。

## 全量评测：三精度分别运行

```bash
PRECISIONS=fp32 OUTPUT_DIR="$PWD/results/full_recommended/fp32" ./run_campaign.sh
PRECISIONS=int8_causal_vopt OUTPUT_DIR="$PWD/results/full_recommended/int8_causal_vopt" ./run_campaign.sh
PRECISIONS=int4_causal_vopt OUTPUT_DIR="$PWD/results/full_recommended/int4_causal_vopt" ./run_campaign.sh

PRECISIONS=fp32,int8_causal_vopt,int4_causal_vopt \
  OUTPUT_DIR="$PWD/results/full_recommended/combined" ./run_campaign.sh
```

也可让自动套件切换到 full set：

```bash
VISUAL_DQ_THRESHOLD=0 FULL_DATASET=1 ./run_weekend_suite.sh
```

中断后执行同一命令会复用已经完整发布的 precision/page/layout cache。正在写入的向量先使用 `.tmp` 文件，未完成时不会被当作有效 cache。

完整向量按配置指纹缓存在 `cache/`。完成的 page/layout cache 会复用；不同 precision、分路径 batch、image token 配置或数据选择不会互相污染。每个 precision 完成后都会 checkpoint 当前 output directory 的 `summary.json`，逐 query 排名写入 `per_query_*.jsonl`。

FP32 是质量 baseline。最终报告应同时给出 INT8/INT4 相对 FP32 的 Recall/nDCG 绝对差与相对差，不能只与 A100 表格比较。

## Batch 选择（已实测）

GPU 的最优 batch 取决于输入路径，不能对 query、整页图像和 layout 强制使用同一个值。当前默认 `--batch-profile recommended` 按 precision 自动选择：

| Precision | Query | Page image | Text/equation layout | Image/table fused layout |
|---|---:|---:|---:|---:|
| FP32 | **8** | **1** | **1** | **1** |
| INT8 optimized | **8** | **1** | **4** | **1** |
| INT4 optimized | **8** | **1** | **4** | **1** |

依据如下：

- 200 条 `balanced_200` query、每档 3 次：batch 1/2/4/8 的 query throughput 分别为 FP32 `5.83/6.09/7.83/8.16`，INT8 `18.62/29.05/49.32/65.60`，INT4 `20.10/35.03/55.63/73.05 queries/s`，因此 query 选 8。
- 首份真实文档（23 pages、138 text layouts、15 fused layouts）：FP32 batch=2 相比 1，page throughput `0.159->0.121/s`、text `3.53->2.94/s`、fused `0.112->0.0467/s`，全部退化，因此 FP32 document 路径选 1。
- INT8 text layout batch 1/2/4/8 为 `15.63/23.28/26.23/22.26 layouts/s`；INT4 为 `17.72/23.54/26.20/24.00 layouts/s`，因此量化 text layout 选 4。
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

本机 batch=1 smoke 外推的三精度串行编码时间约 4.4 天（FP32 约 85 小时、INT8 约 11 小时、INT4 约 10 小时）。这是容量规划估算，不是全量实测值；先运行推荐的 `balanced_200`，再用其实际吞吐重估 full campaign。