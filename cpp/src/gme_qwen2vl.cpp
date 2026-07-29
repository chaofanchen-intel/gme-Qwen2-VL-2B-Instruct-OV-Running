#include "gme_qwen2vl.hpp"

#include <cstdlib>
#include <map>
#include <stdexcept>

#include <openvino/runtime/intel_gpu/properties.hpp>

#include "image_preprocess.hpp"

namespace gme {

namespace {

const std::string DEFAULT_INSTRUCTION = "You are a helpful assistant.";

// precision -> (language IR, visual IR); mirrors gme_openvino_config.json.
const std::map<std::string, std::pair<std::string, std::string>> kPrecisionMap = {
    {"fp32", {"gme_language_fp32.xml", "gme_visual_fp32.xml"}},
    {"int8", {"gme_language_int8.xml", "gme_visual_int8.xml"}},
    {"int4", {"gme_language_int4.xml", "gme_visual_int4.xml"}},
    {"int4_gs64", {"gme_language_int4_gs64.xml", "gme_visual_int4.xml"}},
    {"int4_visint8", {"gme_language_int4.xml", "gme_visual_int8.xml"}},
};

std::string tokenizers_library() {
    if (const char* env = std::getenv("GME_TOKENIZERS_LIB")) return env;
#ifdef GME_OV_TOKENIZERS_LIB
    return GME_OV_TOKENIZERS_LIB;
#else
    return "libopenvino_tokenizers.so";  // resolved via LD_LIBRARY_PATH (setupvars.sh)
#endif
}

// Find a request output tensor whose name contains `needle`.
ov::Tensor output_by_substr(ov::InferRequest& req, const ov::CompiledModel& model, const char* needle) {
    for (const auto& out : model.outputs()) {
        for (const auto& name : out.get_names()) {
            if (name.find(needle) != std::string::npos) return req.get_tensor(out);
        }
    }
    throw std::runtime_error(std::string("tokenizer output not found: ") + needle);
}

}  // namespace

GmeQwen2VL::GmeQwen2VL(const std::string& package_dir, const std::string& device,
                       const std::string& precision)
    : device_(device), precision_(precision) {
    auto it = kPrecisionMap.find(precision);
    if (it == kPrecisionMap.end()) {
        std::string keys;
        for (auto& kv : kPrecisionMap) keys += kv.first + " ";
        throw std::runtime_error("unknown precision '" + precision + "'; choose from: " + keys);
    }
    const std::string language_xml = package_dir + "/" + it->second.first;
    const std::string visual_xml = package_dir + "/" + it->second.second;
    const std::string tokenizer_xml = package_dir + "/openvino_tokenizer.xml";
    // The tokenizer IR uses custom ops from openvino-tokenizers.
    core_.add_extension(tokenizers_library());

    ov::AnyMap fp32_math;
    if (precision == "fp32") fp32_math = {ov::hint::inference_precision(ov::element::f32)};

    // Tokenizer always on CPU (string ops); neural IRs on the requested device.
    tokenizer_req_ = core_.compile_model(tokenizer_xml, "CPU").create_infer_request();
    language_req_ = core_.compile_model(language_xml, device, fp32_math).create_infer_request();
    visual_req_ = core_.compile_model(visual_xml, device, fp32_math).create_infer_request();
}

std::string GmeQwen2VL::build_prompt(const std::string* text, bool has_image,
                                     const std::string& instruction, bool is_query,
                                     std::int64_t n_image_pad) const {
    const std::string inst = (is_query && !instruction.empty()) ? instruction : DEFAULT_INSTRUCTION;
    std::string content;
    if (has_image) {
        // Mirror the HF processor: expand <|image_pad|> to one token per visual
        // merged patch so the language IR's masked_scatter has a slot per embed.
        content = "<|vision_start|>";
        std::string pads;
        pads.reserve(static_cast<std::size_t>(n_image_pad) * 13);
        for (std::int64_t i = 0; i < n_image_pad; ++i) pads += "<|image_pad|>";
        content += pads + "<|vision_end|>";
    }
    if (text) content += *text;
    return "<|im_start|>system\n" + inst + "<|im_end|>\n" +
           "<|im_start|>user\n" + content + "<|im_end|>\n" +
           "<|im_start|>assistant\n<|endoftext|>";
}

void GmeQwen2VL::tokenize(const std::vector<std::string>& prompts, ov::Tensor& input_ids,
                          ov::Tensor& attention_mask) {
    ov::Tensor strings(ov::element::string, ov::Shape{prompts.size()});
    auto* sdata = strings.data<std::string>();
    for (std::size_t i = 0; i < prompts.size(); ++i) sdata[i] = prompts[i];
    tokenizer_req_.set_input_tensor(strings);
    tokenizer_req_.infer();

    const auto model = tokenizer_req_.get_compiled_model();
    ov::Tensor ids = output_by_substr(tokenizer_req_, model, "input_ids");
    ov::Tensor mask = output_by_substr(tokenizer_req_, model, "attention_mask");

    // Copy into owned int64 tensors (language IR expects i64 [B,S]).
    auto to_i64 = [](const ov::Tensor& src) {
        ov::Tensor dst(ov::element::i64, src.get_shape());
        auto* d = dst.data<int64_t>();
        const std::size_t n = src.get_size();
        if (src.get_element_type() == ov::element::i64) {
            std::copy_n(src.data<int64_t>(), n, d);
        } else if (src.get_element_type() == ov::element::i32) {
            const auto* s = src.data<int32_t>();
            for (std::size_t i = 0; i < n; ++i) d[i] = s[i];
        } else {
            throw std::runtime_error("unexpected tokenizer output dtype");
        }
        return dst;
    };
    input_ids = to_i64(ids);
    attention_mask = to_i64(mask);
}

GmeQwen2VL::VisualResult GmeQwen2VL::run_visual(const std::vector<std::string>& image_paths) {
    if (image_paths.empty()) throw std::runtime_error("run_visual: empty image batch");
    std::vector<float> concatenated_embeds;
    std::vector<std::int64_t> merged_tokens;
    merged_tokens.reserve(image_paths.size());
    for (const auto& image_path : image_paths) {
        detail::ImageTensors image = detail::preprocess_image(image_path);
        const std::size_t patches = static_cast<std::size_t>(image.num_patches());
        const std::size_t row_len = image.pixel_values.size() / patches;
        ov::Tensor pixels(ov::element::f32, ov::Shape{patches, row_len});
        std::copy(image.pixel_values.begin(), image.pixel_values.end(), pixels.data<float>());
        ov::Tensor grid(ov::element::i64, ov::Shape{1, 3});
        auto* grid_data = grid.data<int64_t>();
        grid_data[0] = image.grid_t;
        grid_data[1] = image.grid_h;
        grid_data[2] = image.grid_w;
        visual_req_.set_tensor("pixel_values", pixels);
        visual_req_.set_tensor("image_grid_thw", grid);
        visual_req_.infer();
        ov::Tensor output = visual_req_.get_output_tensor(0);
        concatenated_embeds.insert(
            concatenated_embeds.end(), output.data<float>(), output.data<float>() + output.get_size()
        );
        merged_tokens.push_back(static_cast<std::int64_t>(output.get_shape()[0]));
    }
    ov::Tensor out(ov::element::f32, ov::Shape{concatenated_embeds.size() / embedding_dim_, embedding_dim_});
    std::copy(concatenated_embeds.begin(), concatenated_embeds.end(), out.data<float>());
    return {out, merged_tokens};
}

Embedding GmeQwen2VL::run_language(const std::vector<std::string>& prompts,
                                   const ov::Tensor& image_embeds) {
    ov::Tensor input_ids, attention_mask;
    tokenize(prompts, input_ids, attention_mask);

    language_req_.set_tensor("input_ids", input_ids);
    language_req_.set_tensor("attention_mask", attention_mask);
    language_req_.set_tensor("image_embeds", image_embeds);
    language_req_.infer();

    ov::Tensor out = language_req_.get_output_tensor(0);
    const auto shape = out.get_shape();  // [B, dim]
    Embedding emb;
    emb.rows = shape[0];
    emb.dim = shape[1];
    emb.data.resize(out.get_size());
    std::copy_n(out.data<float>(), out.get_size(), emb.data.data());
    return emb;
}

Embedding GmeQwen2VL::encode_text(const std::vector<std::string>& texts,
                                  const std::string& instruction, bool is_query) {
    if (texts.empty()) throw std::runtime_error("encode_text: empty input");
    std::vector<std::string> prompts;
    prompts.reserve(texts.size());
    for (const auto& t : texts) prompts.push_back(build_prompt(&t, false, instruction, is_query));
    ov::Tensor empty(ov::element::f32, ov::Shape{0, embedding_dim_});  // no image tokens
    return run_language(prompts, empty);
}

Embedding GmeQwen2VL::encode_image(const std::vector<std::string>& image_paths,
                                   const std::string& instruction, bool is_query) {
    if (image_paths.empty()) throw std::runtime_error("encode_image: empty input");
    VisualResult visual = run_visual(image_paths);
    std::vector<std::string> prompts;
    prompts.reserve(image_paths.size());
    for (const auto token_count : visual.merged_tokens) {
        prompts.push_back(build_prompt(nullptr, true, instruction, is_query, token_count));
    }
    return run_language(prompts, visual.embeds);
}

Embedding GmeQwen2VL::encode_fused(const std::vector<std::string>& texts,
                                   const std::vector<std::string>& image_paths,
                                   const std::string& instruction, bool is_query) {
    if (texts.size() != image_paths.size() || texts.empty()) {
        throw std::runtime_error("encode_fused: texts and images must be non-empty and equal length");
    }
    VisualResult visual = run_visual(image_paths);
    std::vector<std::string> prompts;
    prompts.reserve(texts.size());
    for (std::size_t index = 0; index < texts.size(); ++index) {
        prompts.push_back(build_prompt(&texts[index], true, instruction, is_query, visual.merged_tokens[index]));
    }
    return run_language(prompts, visual.embeds);
}

}  // namespace gme
