# GME-Qwen2-VL-2B-Instruct / Qwen3-VL-Embedding-2B 的 PA 与 SDPA 调查

日期：2026-08-09

## 结论先行

`GME-Qwen2-VL-2B-Instruct` 和 `Qwen3-VL-Embedding-2B` 都不能像
`Qwen3.5-2B` 生成模型那样，通过 `ATTENTION_BACKEND=PA` / `ATTENTION_BACKEND=SDPA`
在同一条 GenAI VLM 生成 pipeline 上切换两种 backend。

因此：

- 两个 embedding 模型没有一个“尚未测试的 PA/SDPA backend”可以补跑；
- 不存在可成立的“PA 是否比 SDPA 更快”的模型内 A/B 结论；
- 不应把这两个模型当前 OpenVINO graph 中的 `ScaledDotProductAttention`，等同为
  Qwen3.5 continuous-batching 的 SDPA backend；
- 也不应把 OpenVINO GPU plugin 对 SDPA 算子的内部 kernel 选择，称为 GenAI 的 PA/SDPA
  backend 切换。

这不是缺少测试，而是两套 embedding runtime 的 API、模型图和 backend ownership 都不同。

## 1. Qwen3.5 的 PA/SDPA 到底是什么

在 OpenVINO GenAI 2026.3 中，`ATTENTION_BACKEND` 属于生成式 `VLMPipeline` 的
backend 选择。相关代码位于：

```text
benchmark/openvino.genai/src/cpp/src/utils.cpp
benchmark/openvino.genai/src/cpp/src/visual_language/pipeline.cpp
benchmark/openvino.genai/src/cpp/src/visual_language/pipeline_base.cpp
benchmark/openvino.genai/src/cpp/src/visual_language/continuous_batching_adapter.hpp
```

关键逻辑：

1. `utils::extract_attention_backend()` 默认把 backend 设为 `PA`；
2. `VLMPipeline` 构造时，如果 backend 是 PA，会尝试创建
   `VLMContinuousBatchingAdapter`；
3. `VLMContinuousBatchingAdapter` 内部使用 `ContinuousBatchingPipeline`，并明确设置
   `set_attention_backend(PA_BACKEND)`；
4. 显式传入 `ATTENTION_BACKEND=SDPA` 时，不创建 PA continuous-batching adapter，走
   普通 `VLMPipelineImpl`；
5. PA 初始化失败时，官方代码会警告并 fallback 到 SDPA；
6. scheduler、prefix caching、decode KV cache 等生成式状态管理都属于这条路径。

官方默认语义因此是：

```text
生成式 VLMPipeline：默认尝试 PA；显式 ATTENTION_BACKEND=SDPA 才走 SDPA
```

注意：`VLMBackend` 的成员默认值在 `pipeline_base.cpp` 中是 SDPA，但普通
`VLMPipeline` 构造会先通过 `extract_attention_backend()` 默认选择 PA，并在适用时
创建 PA adapter。不要只看 `VLMBackend` 构造函数就断言整个 VLMPipeline 默认 SDPA。

## 2. GME-Qwen2-VL-2B-Instruct

### 2.1 实际运行入口

本机 GME runtime 是：

```text
gme-Qwen2-VL-2B-Instruct-Running/gme_openvino.py
```

`GmeOpenVINO.__init__()` 直接执行：

```python
self.language = self.core.compile_model(self.model_dir / language_model, device, compile_config)
self.visual = self.core.compile_model(self.model_dir / visual_model, device, compile_config)
```

text/query 的调用链是：

```text
GmeOpenVINO.get_text_embeddings()
  -> GmeOpenVINO._encode()
  -> GmeOpenVINO.embed()
  -> GmeOpenVINO._encode_text_batch()
  -> GmeOpenVINO._language_embedding()
  -> OpenVINO compiled language IR
```

这条路径：

- 不创建 `openvino_genai.VLMPipeline`；
- 不创建 `VLMContinuousBatchingAdapter`；
- 不创建 `ContinuousBatchingPipeline`；
- 不解析 `ATTENTION_BACKEND`；
- 只是将已经导出的 language/visual IR 交给 OpenVINO Runtime compile/infer。

因此对 GME Python runtime 设置：

```bash
export ATTENTION_BACKEND=PA
export ATTENTION_BACKEND=SDPA
```

不会产生 Qwen3.5 式的 backend 切换。该环境变量没有被 `gme_openvino.py` 读取，也
没有进入 GenAI `VLMPipeline` 构造。

### 2.2 GME IR 证据

GME language IR：

```text
gme_language_fp32.xml
gme_language_int8.xml
gme_language_int4.xml
gme_language_int4_gs64.xml
```

每份图中都有 28 个：

```xml
type="ScaledDotProductAttention"
```

没有：

```text
PagedAttentionExtension
```

所以 GME 导出图是一个无状态、固定模型图语义的 SDPA 算子图；它不是包含可由
GenAI continuous batching 转换为 PA 的生成式 KV-cache 图。

### 2.3 GME 的实际性能结论

现有 GPU INT4 text/query steady-state 基线：

| Effective tokens | text | query |
|---:|---:|---:|
| 32 | 约 29.8 ms | 约 28.7 ms |
| 128 | 约 43.6 ms | 约 42.3 ms |
| 512 | 约 140.4 ms | 约 140.7 ms |
| 1024 | 约 272.9 ms | 约 272.5 ms |

这些数字是 GME 自己的 OpenVINO IR runtime，不是 PA/SDPA A/B。

### 2.4 GME 结论

```text
GME-Qwen2-VL-2B-Instruct：没有可切换的 GenAI PA/SDPA backend。
当前 graph 是 ScaledDotProductAttention；没有 PagedAttentionExtension。
因此不补跑 PA/SDPA 性能表，也不能声称 PA 更快或 SDPA 更快。
```

GME 可以继续优化 OpenVINO GPU plugin 的 SDPA/GEMM/kernel 以及 wrapper 的 batch、
stream、cache、shape 策略，但那是 OpenVINO Runtime/kernel 优化，不是
Qwen3.5 风格的 attention backend 选择。

## 3. Qwen3-VL-Embedding-2B

### 3.1 实际运行入口

Qwen3 embedding 使用：

```text
openvino_genai.EmbeddingPipeline
```

实现位于：

```text
benchmark/openvino.genai/src/cpp/src/rag/embedding_pipeline.cpp
```

`EmbeddingPipelineImpl::init_multimodal()` 直接读取并编译：

```cpp
std::shared_ptr<ov::Model> language_model =
    utils::singleton_core().read_model(models_path / "openvino_language_model.xml");
language_model = utils::apply_postprocessing(language_model, m_config);
m_compiled_language_model =
    utils::singleton_core().compile_model(language_model, device, properties_copy);
```

调用链是：

```text
EmbeddingPipeline.embed()
  -> EmbeddingPipelineImpl::multimodal_embed()
  -> InputsEmbedder::get_inputs_embeds()
  -> compiled openvino_language_model.xml
```

这条 embedding pipeline：

- 不调用 `VLMPipeline`；
- 不创建 `VLMContinuousBatchingAdapter`；
- 不创建 `ContinuousBatchingPipeline`；
- 不调用 `extract_attention_backend()`；
- 不把 `ATTENTION_BACKEND` 当作 embedding config。

即使把 `ATTENTION_BACKEND=PA` 放进 Python/C++ properties，当前
`EmbeddingPipeline` 也只是把普通 plugin properties 传给 `compile_model()`，不会把
这份 embedding graph 转换成 Qwen3.5 那种 PA continuous-batching graph。

### 3.2 Qwen3-VL-Embedding IR 证据

本机 INT8/INT4 language IR：

```text
benchmark/models/Qwen3-VL-Embedding-2B-INT8/openvino_language_model.xml
benchmark/models/Qwen3-VL-Embedding-2B-INT4/openvino_language_model.xml
```

每份 language graph 都包含 28 个：

```xml
type="ScaledDotProductAttention"
```

没有：

```text
PagedAttentionExtension
```

vision merger graph 同样使用 `ScaledDotProductAttention`。这说明 Qwen3 embedding
模型是已导出的 stateless embedding graph，不是生成式 PA graph。

### 3.3 Qwen3 embedding 现有性能结果

当前 native 2048 维 GPU baseline：

| Precision | text 32 tokens | image 448x448 | fused 448x448 |
|---|---:|---:|---:|
| INT8 | 53.1 ms | 266.0 ms | 253.6 ms |
| INT4 | 29.1 ms | 255.0 ms | 262.1 ms |

这些是 `EmbeddingPipeline` 直接编译 embedding IR 的结果，不是 PA/SDPA A/B。

### 3.4 Qwen3-VL-Embedding 结论

```text
Qwen3-VL-Embedding-2B：没有可切换的 GenAI PA/SDPA backend。
当前 language/vision graph 是 ScaledDotProductAttention；没有 PagedAttentionExtension。
因此没有“缺失 backend”可补测，也不能声称 PA 比 SDPA 更快。
```

它可以使用 OpenVINO GPU plugin 对 `ScaledDotProductAttention` 的内部实现，但那属于
plugin kernel selection；不等于 GenAI 的 `ATTENTION_BACKEND=PA/SDPA`。

### 3.5 运行时 probe

在当前 patched 2026.3 `EmbeddingPipeline` 上尝试：

```python
EmbeddingPipeline(model_dir, "GPU", ATTENTION_BACKEND="PA")
EmbeddingPipeline(model_dir, "GPU", ATTENTION_BACKEND="SDPA")
```

两者都不会进入 PA/SDPA pipeline，而是在 `compile_model()` 的 GPU plugin property
检查处失败：

```text
Option not found: ATTENTION_BACKEND
```

这进一步证明 `ATTENTION_BACKEND` 不是当前 embedding pipeline 接受的 runtime property。
异常随后尝试 text-only fallback，但 embedding model 目录没有 `openvino_model.xml`，
所以最终显示 `EmbeddingPipeline initialization failed`；这不是 PA/SDPA A/B 结果，也
不是模型 text/image/fused 正常调用的失败。正常不传该属性时，Qwen3 embedding 的
patched benchmark 已验证 text/image/fused 和 batch 路径。

## 4. 为什么不能直接把 Qwen3.5 的测试复制过来

Qwen3.5 的 PA/SDPA 对比通常测的是生成式流程：

```text
prompt prefill
  -> attention backend
  -> KV cache
  -> decode / generated tokens
  -> TTFT / TPOT
```

GME 和 Qwen3-VL-Embedding 测的是：

```text
text/image preprocessing
  -> stateless embedding language/vision IR
  -> pooled single vector
```

embedding graph 没有生成 decode、没有连续 batch scheduler、没有生成 KV cache 生命周期，
所以没有相同的 backend switch point。把 `ATTENTION_BACKEND` 字符串塞进 embedding
调用只能测试“这个属性是否被 OpenVINO plugin 忽略”，不能得到有意义的 PA/SDPA 性能对比。

## 5. 当前建议

### GME

继续使用现有 GPU INT4 benchmark 作为 embedding latency 基线；优化方向应是：

- GPU plugin 的 SDPA/GEMM kernel selection；
- INT4 compressed fully-connected/GEMM kernel；
- OpenVINO compiled-model cache；
- GME language batch mode、streams 和 dynamic shape；
- tokenizer/preprocessing 与 infer 的端到端 overlap。

这些优化应标注为 GME/OpenVINO Runtime 优化，而不是 PA 优化。

### Qwen3-VL-Embedding-2B

继续使用 `EmbeddingPipeline` 的 text/image/fused benchmark；优化方向应是：

- embedding language/vision IR 的 SDPA/GEMM kernel；
- text/image batch extra-input 组装；
- native 2048 维输出和 MRL 后处理；
- 图像预处理与视觉 token 数控制。

不要为它新增不存在的 PA/SDPA latency 表。

## 6. 证据文件

- GME runtime：[gme_openvino.py](../gme-Qwen2-VL-2B-Instruct-Running/gme_openvino.py)
- GME benchmark：[08_benchmark_matrix.py](../gme-Qwen2-VL-2B-Instruct-Running/08_benchmark_matrix.py)
- Qwen3 embedding runtime：[embedding_pipeline.cpp](../openvino.genai/src/cpp/src/rag/embedding_pipeline.cpp)
- GenAI backend parser：[utils.cpp](../openvino.genai/src/cpp/src/utils.cpp)
- GenAI VLM backend selection：[pipeline.cpp](../openvino.genai/src/cpp/src/visual_language/pipeline.cpp)
- Qwen3-VL-Embedding benchmark：[Qwen3-VL-Embedding-2B-Running/benchmark_results/BENCHMARK_RESULTS.md](../Qwen3-VL-Embedding-2B-Running/benchmark_results/BENCHMARK_RESULTS.md)
- Qwen3.5 PA/SDPA reference：[Qwen3.5_iGPU_TTFT_Optimization_Report_EN.md](Qwen3.5_iGPU_TTFT_Optimization_Report_EN.md)
