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

class GmeQwen2VL {
public:
    // package_dir: the exported gme-qwen2-vl-2b-instruct-openvino directory.
    // device:      OpenVINO device for the neural IRs ("CPU", "GPU", ...).
    // precision:   "fp32" | "int8" | "int4" | "int4_gs64" | "int4_visint8".
    GmeQwen2VL(const std::string& package_dir,
               const std::string& device = "CPU",
               const std::string& precision = "int8");

    // Text-only, batched. instruction is used only when is_query is true and it is
    // non-empty; otherwise the default GME instruction is applied.
    Embedding encode_text(const std::vector<std::string>& texts,
                          const std::string& instruction = "",
                          bool is_query = true);

    // Image-only, one embedding per image (evaluated one image at a time, matching
    // the official one-image-per-sample contract).
    Embedding encode_image(const std::vector<std::string>& image_paths,
                           const std::string& instruction = "",
                           bool is_query = true);

    // Fused text+image, paired element-wise (texts.size() == image_paths.size()).
    Embedding encode_fused(const std::vector<std::string>& texts,
                           const std::vector<std::string>& image_paths,
                           const std::string& instruction = "",
                           bool is_query = true);

    std::size_t dim() const { return embedding_dim_; }
    const std::string& device() const { return device_; }
    const std::string& precision() const { return precision_; }

private:
    // has_image expands the <|image_pad|> placeholder to n_image_pad copies
    // (== visual merged-token count), mirroring the HF processor.
    std::string build_prompt(const std::string* text, bool has_image,
                             const std::string& instruction, bool is_query,
                             std::int64_t n_image_pad = 0) const;

    // Tokenize prompts -> (input_ids, attention_mask) int64 tensors via the OV tokenizer.
    void tokenize(const std::vector<std::string>& prompts,
                  ov::Tensor& input_ids, ov::Tensor& attention_mask);

    // Run the visual IR on preprocessed image tensors; returns [n_merged_tokens, dim].
    struct VisualResult {
        ov::Tensor embeds;
        std::int64_t merged_tokens;  // == embeds rows == number of <|image_pad|> to emit
    };
    VisualResult run_visual(const std::string& image_path);

    // Run the language IR; image_embeds may be an empty [0, dim] tensor for text-only.
    Embedding run_language(const std::vector<std::string>& prompts, const ov::Tensor& image_embeds);

    ov::Core core_;
    ov::InferRequest tokenizer_req_;
    ov::InferRequest language_req_;
    ov::InferRequest visual_req_;
    std::string device_;
    std::string precision_;
    std::size_t embedding_dim_ = 1536;
};

}  // namespace gme
