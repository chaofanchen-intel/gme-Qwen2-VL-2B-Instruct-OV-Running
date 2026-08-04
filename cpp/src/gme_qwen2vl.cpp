#include "gme_qwen2vl.hpp"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iterator>
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

// Extract one integer field from gme_openvino_config.json's "processor" object
// (e.g. "max_length": 1800). The project has no JSON library and needs exactly
// this one number to validate a max_length override against what the tokenizer
// IR actually baked in; the file is our own export product (03_finalize.py), not
// external input, so a plain substring scan is enough -- it does not need to
// parse the rest of the document.
std::int64_t read_processor_config_int(const std::string& package_dir, const std::string& key) {
    std::ifstream file(package_dir + "/gme_openvino_config.json");
    if (!file) {
        throw std::runtime_error("cannot open gme_openvino_config.json in " + package_dir);
    }
    const std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    const std::string needle = "\"" + key + "\"";
    const std::size_t key_pos = text.find(needle);
    if (key_pos == std::string::npos) {
        throw std::runtime_error("gme_openvino_config.json has no field '" + key + "'");
    }
    const std::size_t colon = text.find(':', key_pos + needle.size());
    if (colon == std::string::npos) {
        throw std::runtime_error("malformed gme_openvino_config.json near '" + key + "'");
    }
    return std::stoll(text.substr(colon + 1));
}

}  // namespace

GmeQwen2VL::Pool::Pool(ov::CompiledModel model)
    : model_(std::move(model)),
      capacity_(std::max<std::size_t>(1, model_.get_property(ov::optimal_number_of_infer_requests))) {}

void GmeQwen2VL::Pool::run(std::size_t jobs,
                           const std::function<void(std::size_t, ov::InferRequest&)>& feed,
                           const std::function<void(std::size_t, ov::InferRequest&)>& sink) {
    if (jobs == 0) return;
    const std::size_t slots = std::min(capacity_, jobs);
    while (requests_.size() < slots) requests_.push_back(model_.create_infer_request());

    std::vector<std::size_t> in_flight(slots, 0);
    std::vector<bool> busy(slots, false);
    // Collect the job currently occupying `slot`, freeing it for the next one.
    auto harvest = [&](std::size_t slot) {
        if (!busy[slot]) return;
        requests_[slot].wait();
        sink(in_flight[slot], requests_[slot]);
        busy[slot] = false;
    };
    try {
        for (std::size_t job = 0; job < jobs; ++job) {
            const std::size_t slot = job % slots;
            harvest(slot);
            feed(job, requests_[slot]);
            requests_[slot].start_async();
            in_flight[slot] = job;
            busy[slot] = true;
        }
        for (std::size_t slot = 0; slot < slots; ++slot) harvest(slot);
    } catch (...) {
        // Requests still in flight hold pointers into the caller's tensors, which
        // unwinding is about to destroy. Drain before letting the throw escape.
        for (std::size_t slot = 0; slot < slots; ++slot) {
            if (busy[slot]) requests_[slot].wait();
        }
        throw;
    }
}

GmeQwen2VL::GmeQwen2VL(const std::string& package_dir, const std::string& device,
                       const std::string& precision, int streams,
                       const std::string& batch_mode, const std::string& cache_dir,
                       int min_image_tokens, int max_image_tokens, std::size_t max_length)
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

    if (min_image_tokens < 0 || max_image_tokens < 0) {
        throw std::runtime_error("min_image_tokens/max_image_tokens must be >= 0 (0 = package default)");
    }
    // 0 keeps the package default (README/gme_openvino_config.json: 256/1280).
    proc_min_pixels_ = (min_image_tokens > 0)
        ? static_cast<long>(min_image_tokens) * detail::IMAGE_FACTOR * detail::IMAGE_FACTOR
        : detail::PROC_MIN_PIXELS;
    proc_max_pixels_ = (max_image_tokens > 0)
        ? static_cast<long>(max_image_tokens) * detail::IMAGE_FACTOR * detail::IMAGE_FACTOR
        : detail::PROC_MAX_PIXELS;
    if (proc_min_pixels_ > proc_max_pixels_) {
        throw std::runtime_error(
            "min_image_tokens must be <= max_image_tokens (got " +
            std::to_string(min_image_tokens) + " > " + std::to_string(max_image_tokens) + ")");
    }

    // The tokenizer IR bakes its truncation ceiling in at export time
    // (03_finalize.py); a runtime request above that ceiling cannot be honored
    // without re-exporting, so fail clearly instead of silently doing less than
    // asked. 0 (no override) skips this check entirely: max_length_ stays 0,
    // meaning run_language applies no extra client-side truncation.
    if (max_length > 0) {
        const std::int64_t baked = read_processor_config_int(package_dir, "max_length");
        if (static_cast<std::int64_t>(max_length) > baked) {
            throw std::runtime_error(
                "requested max_length=" + std::to_string(max_length) +
                " exceeds this package's baked tokenizer limit of " + std::to_string(baked) +
                "; re-export with 03_finalize.py using a larger --max-length to raise it");
        }
        max_length_ = max_length;
    }

    // The tokenizer IR uses custom ops from openvino-tokenizers.
    core_.add_extension(tokenizers_library());
    if (!cache_dir.empty()) core_.set_property(ov::cache_dir(cache_dir));

    ov::AnyMap compile_config = stream_config(streams);
    if (precision == "fp32") compile_config.emplace(ov::hint::inference_precision(ov::element::f32));

    // Tokenizer always on CPU (string ops); neural IRs on the requested device.
    tokenizer_req_ = core_.compile_model(tokenizer_xml, "CPU").create_infer_request();
    language_pool_ = Pool(core_.compile_model(language_xml, device, compile_config));
    visual_pool_ = Pool(core_.compile_model(visual_xml, device, compile_config));
}

std::string GmeQwen2VL::build_prompt(const std::string* text, bool has_image,
                                     const std::optional<std::string>& instruction, bool is_query,
                                     std::int64_t n_image_pad) const {
    // has_value(), not "non-empty": an explicit "" is a real instruction, matching
    // official/Python (`instruction is not None`). Only a truly absent instruction
    // (nullopt, the default) falls back to DEFAULT_INSTRUCTION.
    const std::string& inst = (is_query && instruction.has_value()) ? *instruction : DEFAULT_INSTRUCTION;
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
        detail::ImageTensors image =
            detail::preprocess_image(image_paths[index], proc_min_pixels_, proc_max_pixels_);
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
    visual_pool_.run(
        count,
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
    if (max_length_ && input_ids.get_shape()[1] > max_length_) {
        // Column-slicing a right-padded, already-truncated-at-1800 batch down to
        // max_length_ columns is column-for-column identical to re-tokenizing with
        // truncation=True, max_length=max_length_ -- verified against the HF
        // tokenizer, whose truncation_side is "right" for this model. See the
        // constructor parameter comment in the header for the full argument.
        auto slice_columns = [&](const ov::Tensor& src) {
            const auto shape = src.get_shape();
            ov::Tensor dst(ov::element::i64, ov::Shape{shape[0], max_length_});
            const auto* s = src.data<int64_t>();
            auto* d = dst.data<int64_t>();
            for (std::size_t row = 0; row < shape[0]; ++row) {
                std::copy_n(s + row * shape[1], max_length_, d + row * max_length_);
            }
            return dst;
        };
        input_ids = slice_columns(input_ids);
        attention_mask = slice_columns(attention_mask);
    }
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
        // One job, so the pool creates and reuses exactly one request here.
        language_pool_.run(
            1,
            [&](std::size_t, ov::InferRequest& request) {
                request.set_tensor("input_ids", input_ids);
                request.set_tensor("attention_mask", attention_mask);
                request.set_tensor("image_embeds", all_images);
            },
            [&](std::size_t, ov::InferRequest& request) {
                const ov::Tensor out = request.get_output_tensor(0);  // [B, dim]
                if (out.get_size() != emb.data.size()) {
                    throw std::runtime_error("run_language: unexpected padded output shape");
                }
                std::copy_n(out.data<float>(), out.get_size(), emb.data.data());
            }
        );
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

    language_pool_.run(
        rows,
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

Embedding GmeQwen2VL::encode_chunk(const std::vector<std::string>* texts,
                                   const std::vector<std::string>* image_paths,
                                   const std::optional<std::string>& instruction, bool is_query,
                                   std::size_t begin, std::size_t end) {
    const std::size_t count = end - begin;
    std::vector<std::string> prompts;
    prompts.reserve(count);

    if (image_paths == nullptr) {  // text-only: no visual pass, no image tokens
        for (std::size_t i = begin; i < end; ++i) {
            prompts.push_back(build_prompt(&(*texts)[i], false, instruction, is_query));
        }
        return run_language(prompts, nullptr);
    }

    VisualResult visual =
        run_visual(std::vector<std::string>(image_paths->begin() + begin, image_paths->begin() + end));
    for (std::size_t i = 0; i < count; ++i) {
        // The image-pad count must equal this image's visual rows, so it comes from
        // the visual output rather than being predicted from the file.
        const std::string* text = texts ? &(*texts)[begin + i] : nullptr;
        prompts.push_back(build_prompt(text, true, instruction, is_query, visual.merged_tokens[i]));
    }
    return run_language(prompts, &visual.embeds);
}

Embedding GmeQwen2VL::encode_chunked(const std::vector<std::string>* texts,
                                     const std::vector<std::string>* image_paths,
                                     const std::optional<std::string>& instruction, bool is_query,
                                     std::size_t count, std::size_t batch_size) {
    const std::size_t step = batch_size ? batch_size : count;
    if (step == count) return encode_chunk(texts, image_paths, instruction, is_query, 0, count);

    Embedding all;
    all.dim = embedding_dim_;
    all.rows = count;
    all.data.reserve(count * embedding_dim_);
    for (std::size_t begin = 0; begin < count; begin += step) {
        Embedding chunk = encode_chunk(texts, image_paths, instruction, is_query, begin,
                                       std::min(begin + step, count));
        all.data.insert(all.data.end(), chunk.data.begin(), chunk.data.end());
    }
    if (all.data.size() != count * embedding_dim_) {
        throw std::runtime_error("encode_chunked: chunk rows do not sum to the input size");
    }
    return all;
}

Embedding GmeQwen2VL::encode_text(const std::vector<std::string>& texts,
                                  const std::optional<std::string>& instruction, bool is_query,
                                  std::size_t batch_size) {
    if (texts.empty()) throw std::runtime_error("encode_text: empty input");
    return encode_chunked(&texts, nullptr, instruction, is_query, texts.size(), batch_size);
}

Embedding GmeQwen2VL::encode_image(const std::vector<std::string>& image_paths,
                                   const std::optional<std::string>& instruction, bool is_query,
                                   std::size_t batch_size) {
    if (image_paths.empty()) throw std::runtime_error("encode_image: empty input");
    return encode_chunked(nullptr, &image_paths, instruction, is_query, image_paths.size(), batch_size);
}

Embedding GmeQwen2VL::encode_fused(const std::vector<std::string>& texts,
                                   const std::vector<std::string>& image_paths,
                                   const std::optional<std::string>& instruction, bool is_query,
                                   std::size_t batch_size) {
    if (texts.size() != image_paths.size() || texts.empty()) {
        throw std::runtime_error("encode_fused: texts and images must be non-empty and equal length");
    }
    return encode_chunked(&texts, &image_paths, instruction, is_query, texts.size(), batch_size);
}

}  // namespace gme
