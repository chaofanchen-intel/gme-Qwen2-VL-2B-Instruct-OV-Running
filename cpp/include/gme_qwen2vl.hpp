// GME-Qwen2-VL-2B-Instruct embedding runtime on OpenVINO (C++).
//
// The learned computation lives in three OpenVINO IRs produced by the Python
// exporter (04/06):
//   openvino_tokenizer.xml  string  -> input_ids, attention_mask   (openvino-tokenizers)
//   gme_visual_<prec>.xml    pixels -> image-token embeddings       (Qwen2-VL ViT)
//   gme_language_<prec>.xml  ids+mask+image_embeds -> L2-normalized last-token embedding
//
// This library adds only the preprocessing the IRs do not contain: GME prompt
// templating (done here), and Qwen2-VL image preprocessing (OpenCV: smart_resize
// + rescale + normalize + patchify). Tokenization runs natively inside OpenVINO,
// so no Python is required at inference time.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <openvino/openvino.hpp>

namespace gme {

// Row-major [rows, dim] float32 embeddings (already L2-normalized by the IR).
struct Embedding {
    std::vector<float> data;
    std::size_t rows = 0;
    std::size_t dim = 0;
    const float* row(std::size_t i) const { return data.data() + i * dim; }
};

// Official GME's get_fused_embeddings() default, mirrored by gme_openvino.py.
// Chunking bounds peak memory; pass 0 to encode the whole input as one chunk.
// Under batch_mode "per_sample" the chunk size cannot change a row at all; under
// "padded" it changes which samples pad together, exactly as it does upstream.
inline constexpr std::size_t kDefaultBatchSize = 32;

class GmeQwen2VL {
public:
    // package_dir: the exported gme-qwen2-vl-2b-instruct-openvino directory.
    // device:      OpenVINO device for the neural IRs ("CPU", "GPU", ...).
    // precision:   "fp32" | "int8" | "int4" | "int4_gs64" | "int4_visint8".
    // streams:     execution streams per IR. 1 (default) is the LATENCY hint: one
    //              request gets every core, best for a single sample. 0 lets the
    //              plugin pick (THROUGHPUT), or pass an explicit count; batches then
    //              overlap across streams (~1.8x on 16 cores) while a lone sample is
    //              ~1.3x slower. Output is identical either way.
    // batch_mode:  how the language IR consumes a batch. "per_sample" issues one
    //              request per sample at its true length, making a batch
    //              bit-identical to the same items encoded one by one; "padded"
    //              issues one [B,S] request the way the official model does.
    //              "auto" (default) picks per device, where each is faster:
    //              per_sample on CPU (a padded call is no faster there, 1.00x,
    //              and per_sample also drops the padding work), padded on GPU
    //              (per-submission overhead dominates: 2.4x faster per embedding).
    GmeQwen2VL(const std::string& package_dir,
               const std::string& device = "CPU",
               const std::string& precision = "int8",
               int streams = 1,
               const std::string& batch_mode = "auto");

    // Text-only, batched. instruction is used only when is_query is true and it is
    // non-empty; otherwise the default GME instruction is applied. Inputs longer
    // than the package's max_length (1800) are truncated inside the tokenizer IR,
    // matching HF `truncation=True` and therefore the official model.
    Embedding encode_text(const std::vector<std::string>& texts,
                          const std::string& instruction = "",
                          bool is_query = true,
                          std::size_t batch_size = kDefaultBatchSize);

    // Image-only batch API: returns one embedding per image, in input order.
    // Any batch size works with no per-batch-size export: the visual IR runs once
    // per image and the language IR once per sample, both dispatched concurrently
    // over the request pool. See run_visual/run_language for why that is exact.
    Embedding encode_image(const std::vector<std::string>& image_paths,
                           const std::string& instruction = "",
                           bool is_query = true,
                           std::size_t batch_size = kDefaultBatchSize);

    // Fused batch API: paired element-wise (texts.size() == image_paths.size()).
    Embedding encode_fused(const std::vector<std::string>& texts,
                           const std::vector<std::string>& image_paths,
                           const std::string& instruction = "",
                           bool is_query = true,
                           std::size_t batch_size = kDefaultBatchSize);

    std::size_t dim() const { return embedding_dim_; }
    const std::string& device() const { return device_; }
    const std::string& precision() const { return precision_; }
    const std::string& batch_mode() const { return batch_mode_; }

private:
    // has_image expands the <|image_pad|> placeholder to n_image_pad copies
    // (== visual merged-token count), mirroring the HF processor.
    std::string build_prompt(const std::string* text, bool has_image,
                             const std::string& instruction, bool is_query,
                             std::int64_t n_image_pad = 0) const;

    // Tokenize prompts -> (input_ids, attention_mask) int64 tensors via the OV tokenizer.
    void tokenize(const std::vector<std::string>& prompts,
                  ov::Tensor& input_ids, ov::Tensor& attention_mask);

    // Run the visual IR once per image. Qwen2-VL's vision tower has no batch
    // dimension: images become one patch sequence whose attention is block-diagonal
    // per image, so per-image invocation is bit-identical to passing several
    // image_grid_thw rows at once, and cheaper -- a shared sequence costs
    // O((sum L)^2) attention rather than O(sum L^2). It is also the only thing a
    // traced IR accepts, since rot_pos_emb loops over the grid rows in Python and
    // tracing bakes that count in.
    struct VisualResult {
        std::vector<ov::Tensor> embeds;           // one [K_i, dim] tensor per image
        std::vector<std::int64_t> merged_tokens;  // K_i image-pad tokens per input image
    };
    VisualResult run_visual(const std::vector<std::string>& image_paths);

    // Run the language IR over the batch, per batch_mode_. image_embeds is null
    // for text-only, else one [K_i, dim] tensor per prompt.
    Embedding run_language(const std::vector<std::string>& prompts,
                           const std::vector<ov::Tensor>* image_embeds);

    // Encode [begin, end) of a batch as one chunk; the entry points loop over
    // chunks of batch_size and concatenate, as official GME does.
    Embedding encode_chunk(const std::vector<std::string>* texts,
                           const std::vector<std::string>* image_paths,
                           const std::string& instruction, bool is_query,
                           std::size_t begin, std::size_t end);

    // Split [0, count) into batch_size chunks, encode each via encode_chunk, and
    // concatenate in input order. batch_size 0 means one chunk.
    Embedding encode_chunked(const std::vector<std::string>* texts,
                             const std::vector<std::string>* image_paths,
                             const std::string& instruction, bool is_query,
                             std::size_t count, std::size_t batch_size);

    // Dispatch `jobs` independent inferences over the pool, `feed` filling a
    // request's inputs and `sink` consuming its output, both by job index.
    static void run_pool(std::vector<ov::InferRequest>& pool, std::size_t jobs,
                         const std::function<void(std::size_t, ov::InferRequest&)>& feed,
                         const std::function<void(std::size_t, ov::InferRequest&)>& sink);

    ov::Core core_;
    ov::InferRequest tokenizer_req_;
    std::vector<ov::InferRequest> language_pool_;
    std::vector<ov::InferRequest> visual_pool_;
    std::string device_;
    std::string precision_;
    std::string batch_mode_;  // resolved: "per_sample" or "padded"
    std::size_t embedding_dim_ = 1536;
};

}  // namespace gme
