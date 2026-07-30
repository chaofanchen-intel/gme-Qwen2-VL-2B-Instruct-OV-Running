#include "gme_qwen2vl.hpp"

#include <algorithm>
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

// LATENCY gives one request every core; THROUGHPUT overlaps requests instead.
ov::AnyMap stream_config(int streams) {
    if (streams == 1) return {ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)};
    if (streams < 0) throw std::runtime_error("streams must be >= 0 (0 lets the plugin choose)");
    ov::AnyMap config{ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)};
    if (streams > 1) config.emplace(ov::num_streams(streams));
    return config;
}

// One request per execution stream, so a batch can actually overlap.
std::vector<ov::InferRequest> make_pool(ov::CompiledModel model) {
    const auto optimal = model.get_property(ov::optimal_number_of_infer_requests);
    const std::size_t count = std::max<std::size_t>(1, optimal);
    std::vector<ov::InferRequest> pool;
    pool.reserve(count);
    for (std::size_t i = 0; i < count; ++i) pool.push_back(model.create_infer_request());
    return pool;
}

}  // namespace

GmeQwen2VL::GmeQwen2VL(const std::string& package_dir, const std::string& device,
                       const std::string& precision, int streams,
                       const std::string& batch_mode)
    : device_(device), precision_(precision) {
    // Per-sample drops padding work and overlaps across CPU streams; a padded call
    // amortizes the GPU's per-submission overhead. See the header for measurements.
    if (batch_mode == "auto") {
        batch_mode_ = device.rfind("CPU", 0) == 0 ? "per_sample" : "padded";
    } else if (batch_mode == "per_sample" || batch_mode == "padded") {
        batch_mode_ = batch_mode;
    } else {
        throw std::runtime_error("batch_mode must be auto, per_sample, or padded; got '" +
                                 batch_mode + "'");
    }
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

    ov::AnyMap compile_config = stream_config(streams);
    if (precision == "fp32") compile_config.emplace(ov::hint::inference_precision(ov::element::f32));

    // Tokenizer always on CPU (string ops); neural IRs on the requested device.
    tokenizer_req_ = core_.compile_model(tokenizer_xml, "CPU").create_infer_request();
    language_pool_ = make_pool(core_.compile_model(language_xml, device, compile_config));
    visual_pool_ = make_pool(core_.compile_model(visual_xml, device, compile_config));
}

void GmeQwen2VL::run_pool(std::vector<ov::InferRequest>& pool, std::size_t jobs,
                          const std::function<void(std::size_t, ov::InferRequest&)>& feed,
                          const std::function<void(std::size_t, ov::InferRequest&)>& sink) {
    if (jobs == 0) return;
    const std::size_t slots = std::min(pool.size(), jobs);
    std::vector<std::size_t> in_flight(slots, 0);
    std::vector<bool> busy(slots, false);
    // Collect the job currently occupying `slot`, freeing it for the next one.
    auto harvest = [&](std::size_t slot) {
        if (!busy[slot]) return;
        pool[slot].wait();
        sink(in_flight[slot], pool[slot]);
        busy[slot] = false;
    };
    try {
        for (std::size_t job = 0; job < jobs; ++job) {
            const std::size_t slot = job % slots;
            harvest(slot);
            feed(job, pool[slot]);
            pool[slot].start_async();
            in_flight[slot] = job;
            busy[slot] = true;
        }
        for (std::size_t slot = 0; slot < slots; ++slot) harvest(slot);
    } catch (...) {
        // Requests still in flight hold pointers into the caller's tensors, which
        // unwinding is about to destroy. Drain before letting the throw escape.
        for (std::size_t slot = 0; slot < slots; ++slot) {
            if (busy[slot]) pool[slot].wait();
        }
        throw;
    }
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
    const std::size_t count = image_paths.size();

    // Preprocessing is CPU-side and independent per image; do it up front so the
    // pool dispatch below is pure inference.
    std::vector<ov::Tensor> pixels(count), grids(count);
    for (std::size_t index = 0; index < count; ++index) {
        detail::ImageTensors image = detail::preprocess_image(image_paths[index]);
        const std::size_t patches = static_cast<std::size_t>(image.num_patches());
        const std::size_t row_len = image.pixel_values.size() / patches;
        pixels[index] = ov::Tensor(ov::element::f32, ov::Shape{patches, row_len});
        std::copy(image.pixel_values.begin(), image.pixel_values.end(), pixels[index].data<float>());
        grids[index] = ov::Tensor(ov::element::i64, ov::Shape{1, 3});
        auto* grid_data = grids[index].data<int64_t>();
        grid_data[0] = image.grid_t;
        grid_data[1] = image.grid_h;
        grid_data[2] = image.grid_w;
    }

    VisualResult result;
    result.embeds.resize(count);
    result.merged_tokens.resize(count);
    run_pool(
        visual_pool_, count,
        [&](std::size_t job, ov::InferRequest& request) {
            request.set_tensor("pixel_values", pixels[job]);
            request.set_tensor("image_grid_thw", grids[job]);
        },
        [&](std::size_t job, ov::InferRequest& request) {
            // The request reuses its output buffer for the next job: copy now.
            ov::Tensor output = request.get_output_tensor(0);
            result.embeds[job] = ov::Tensor(output.get_element_type(), output.get_shape());
            std::copy_n(output.data<float>(), output.get_size(), result.embeds[job].data<float>());
            result.merged_tokens[job] = static_cast<std::int64_t>(output.get_shape()[0]);
        }
    );
    return result;
}

Embedding GmeQwen2VL::run_language(const std::vector<std::string>& prompts,
                                   const std::vector<ov::Tensor>* image_embeds) {
    ov::Tensor input_ids, attention_mask;
    tokenize(prompts, input_ids, attention_mask);
    const auto padded = input_ids.get_shape();  // [B, S], right-padded
    const std::size_t rows = padded[0], stride = padded[1];
    if (image_embeds && image_embeds->size() != rows) {
        throw std::runtime_error("run_language: one image_embeds tensor is required per prompt");
    }
    const ov::Tensor no_images(ov::element::f32, ov::Shape{0, embedding_dim_});

    Embedding emb;
    emb.rows = rows;
    emb.dim = embedding_dim_;
    emb.data.resize(rows * embedding_dim_);

    if (batch_mode_ == "padded") {
        // One [B,S] call, exactly the official layout: every sample's visual tokens
        // are concatenated and masked_scatter distributes them across the flattened
        // batch, one image-pad slot per row.
        ov::Tensor all_images = no_images;
        if (image_embeds) {
            std::size_t total = 0;
            for (const auto& tensor : *image_embeds) total += tensor.get_shape()[0];
            all_images = ov::Tensor(ov::element::f32, ov::Shape{total, embedding_dim_});
            float* cursor = all_images.data<float>();
            for (const auto& tensor : *image_embeds) {
                cursor = std::copy_n(tensor.data<float>(), tensor.get_size(), cursor);
            }
        }
        ov::InferRequest& request = language_pool_.front();
        request.set_tensor("input_ids", input_ids);
        request.set_tensor("attention_mask", attention_mask);
        request.set_tensor("image_embeds", all_images);
        request.infer();
        const ov::Tensor out = request.get_output_tensor(0);  // [B, dim]
        if (out.get_size() != emb.data.size()) {
            throw std::runtime_error("run_language: unexpected padded output shape");
        }
        std::copy_n(out.data<float>(), out.get_size(), emb.data.data());
        return emb;
    }

    // Slice each row down to its true length: right padding means the valid tokens
    // are a prefix, so a row copies straight out without repacking.
    std::vector<ov::Tensor> ids(rows), masks(rows);
    const auto* ids_data = input_ids.data<int64_t>();
    const auto* mask_data = attention_mask.data<int64_t>();
    for (std::size_t row = 0; row < rows; ++row) {
        std::size_t length = 0;
        while (length < stride && mask_data[row * stride + length] != 0) ++length;
        const ov::Shape shape{1, length};
        ids[row] = ov::Tensor(ov::element::i64, shape);
        masks[row] = ov::Tensor(ov::element::i64, shape);
        std::copy_n(ids_data + row * stride, length, ids[row].data<int64_t>());
        std::copy_n(mask_data + row * stride, length, masks[row].data<int64_t>());
    }

    run_pool(
        language_pool_, rows,
        [&](std::size_t job, ov::InferRequest& request) {
            request.set_tensor("input_ids", ids[job]);
            request.set_tensor("attention_mask", masks[job]);
            request.set_tensor("image_embeds", image_embeds ? (*image_embeds)[job] : no_images);
        },
        [&](std::size_t job, ov::InferRequest& request) {
            ov::Tensor out = request.get_output_tensor(0);  // [1, dim]
            if (out.get_size() != embedding_dim_) {
                throw std::runtime_error("run_language: unexpected embedding width");
            }
            std::copy_n(out.data<float>(), embedding_dim_, emb.data.data() + job * embedding_dim_);
        }
    );
    return emb;
}

Embedding GmeQwen2VL::encode_text(const std::vector<std::string>& texts,
                                  const std::string& instruction, bool is_query) {
    if (texts.empty()) throw std::runtime_error("encode_text: empty input");
    std::vector<std::string> prompts;
    prompts.reserve(texts.size());
    for (const auto& t : texts) prompts.push_back(build_prompt(&t, false, instruction, is_query));
    return run_language(prompts, nullptr);  // no image tokens
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
    return run_language(prompts, &visual.embeds);
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
    return run_language(prompts, &visual.embeds);
}

}  // namespace gme
