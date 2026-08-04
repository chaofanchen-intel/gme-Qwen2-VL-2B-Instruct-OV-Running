// gme_embed: CLI for the GME-Qwen2-VL OpenVINO C++ runtime.
//
//   gme_embed --package DIR [--device CPU] [--precision int8] [--streams N]
//             [--batch-mode auto|per_sample|padded] [--batch-size N]
//             [--cache-dir DIR] [--min-image-tokens N] [--max-image-tokens N]
//             [--max-length N]
//             --mode text|image|fused [--instruction STR] [--is-query 0|1]
//             [--image PATH ...] [--out FILE] [TEXT ...]
//   gme_embed --package DIR --benchmark [--benchmark-modes text,query] [--batch N]
//             [--warmup N]
//
// --cache-dir enables OpenVINO's compiled-model cache: a second process with the
// same package/device/precision/streams loads a cached blob instead of
// recompiling. Worth setting for any short-lived invocation on GPU, where
// compiling the INT8 language IR from scratch is tens of seconds.
// --min-image-tokens/--max-image-tokens override the package's Qwen2-VL processor
// image-token bounds (package default otherwise); the single most effective
// image/fused latency knob, since attention cost grows with the visual token
// count. --max-length shrinks the package's tokenizer truncation length (package
// default otherwise); it can only shrink, not grow, since the ceiling is baked
// into the tokenizer IR at export time.
//
// --warmup N (default 1) runs the benchmarked call N times before starting the
// timed loop, discarding those iterations. On GPU, a fresh process's first few
// inferences can land on whatever DVFS frequency the device happened to be idling
// at; a single warmup call does not reliably force it to ramp to a steady clock
// before quantized (fast) kernels finish, which showed up as a several-hundred-ms,
// discretely-banded latency spread on 1024-token GPU int8/int4 benchmarks. A
// longer warmup keeps the GPU busy long enough to reach a stable frequency before
// the timed region starts; fp32 calls are long enough on their own that this
// mattered far less there.
//
// Any batch size works with no per-batch-size export; see gme_qwen2vl.hpp.
// --streams 1 (default) runs one request on every core: lowest single-sample
// latency. --streams 0 lets the plugin choose, overlapping a batch's samples
// across streams (~1.8x on 16 cores); results are identical either way.
// --batch-mode auto (default) decodes per sample on CPU and as one padded [B,S]
// call on GPU, which is where each is faster.
// --batch-size N chunks a large input into groups of N (default 32, as official
// GME); 0 encodes everything as one chunk.
// --benchmark --batch N repeats the given input N times so a run measures batch
// throughput; the reported avg_ms is per call, and ms/embedding is avg_ms / N.
//
// --out writes a binary blob {"GEMB", int32 rows, int32 dim, float32[rows*dim]}
// consumed by 05_validate_cpp.py to check cosine against the golden embeddings.
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "gme_qwen2vl.hpp"

namespace {

[[noreturn]] void usage(const char* argv0) {
    std::cerr << "usage: " << argv0
              << " --package DIR [--device CPU] [--precision int8]"
                 " --mode text|image|fused [--instruction STR] [--is-query 0|1]"
                 " [--streams N] [--batch-mode auto|per_sample|padded] [--batch-size N]"
                 " [--cache-dir DIR] [--min-image-tokens N] [--max-image-tokens N]"
                 " [--max-length N]\n"
                 " [--image PATH ...] [--out FILE] [TEXT ...]\n"
                 "       "
              << argv0
              << " --package DIR --benchmark [--benchmark-modes text,query,image,fused]"
                 " [--repeat N] [--warmup N] [--batch N] [--instruction STR] [--image PATH] TEXT\n";
    std::exit(2);
}

bool contains_mode(const std::string& modes, const std::string& wanted) {
    std::size_t start = 0;
    while (start <= modes.size()) {
        const std::size_t end = modes.find(',', start);
        const std::string value = modes.substr(start, end - start);
        if (value == wanted) return true;
        if (end == std::string::npos) return false;
        start = end + 1;
    }
    return false;
}

void dump(const std::string& path, const gme::Embedding& e) {
    std::ofstream f(path, std::ios::binary);
    const char magic[4] = {'G', 'E', 'M', 'B'};
    int32_t rows = static_cast<int32_t>(e.rows);
    int32_t dim = static_cast<int32_t>(e.dim);
    f.write(magic, 4);
    f.write(reinterpret_cast<const char*>(&rows), sizeof rows);
    f.write(reinterpret_cast<const char*>(&dim), sizeof dim);
    f.write(reinterpret_cast<const char*>(e.data.data()),
            static_cast<std::streamsize>(e.data.size() * sizeof(float)));
}

}  // namespace

int main(int argc, char** argv) {
    std::string package, device = "CPU", precision = "int8", mode, instruction, out;
    std::string batch_mode = "auto", cache_dir;
    std::string benchmark_modes = "text,query,image,fused";
    // Tracks whether --instruction was actually given, so an omitted flag reaches
    // the library as nullopt (fall back to the default GME instruction) rather
    // than as an empty string (a real, if unusual, instruction) -- see the
    // encode_text/encode_image/encode_fused doc comments in gme_qwen2vl.hpp.
    bool instruction_given = false;
    bool is_query = true, benchmark = false;
    int repeat = 1, streams = 1, bench_batch = 1, warmup = 1;
    int min_image_tokens = 0, max_image_tokens = 0;
    std::size_t batch_size = gme::kDefaultBatchSize, max_length = 0;
    std::vector<std::string> texts, images;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* name) -> std::string {
            if (i + 1 >= argc) { std::cerr << name << " needs a value\n"; usage(argv[0]); }
            return argv[++i];
        };
        if (a == "--package") package = next("--package");
        else if (a == "--device") device = next("--device");
        else if (a == "--precision") precision = next("--precision");
        else if (a == "--mode") mode = next("--mode");
        else if (a == "--instruction") { instruction = next("--instruction"); instruction_given = true; }
        else if (a == "--is-query") is_query = next("--is-query") != "0";
        else if (a == "--image") images.push_back(next("--image"));
        else if (a == "--out") out = next("--out");
        else if (a == "--repeat") repeat = std::stoi(next("--repeat"));
        else if (a == "--warmup") warmup = std::stoi(next("--warmup"));
        else if (a == "--streams") streams = std::stoi(next("--streams"));
        else if (a == "--batch-mode") batch_mode = next("--batch-mode");
        else if (a == "--batch-size") batch_size = static_cast<std::size_t>(std::stoul(next("--batch-size")));
        else if (a == "--cache-dir") cache_dir = next("--cache-dir");
        else if (a == "--min-image-tokens") min_image_tokens = std::stoi(next("--min-image-tokens"));
        else if (a == "--max-image-tokens") max_image_tokens = std::stoi(next("--max-image-tokens"));
        else if (a == "--max-length") max_length = static_cast<std::size_t>(std::stoul(next("--max-length")));
        else if (a == "--batch") bench_batch = std::stoi(next("--batch"));
        else if (a == "--benchmark") benchmark = true;
        else if (a == "--benchmark-modes") benchmark_modes = next("--benchmark-modes");
        else if (a == "-h" || a == "--help") usage(argv[0]);
        else texts.push_back(a);
    }
    if (package.empty() || (!benchmark && mode.empty())) usage(argv[0]);

    const std::optional<std::string> instruction_opt =
        instruction_given ? std::optional<std::string>(instruction) : std::nullopt;

    try {
        gme::GmeQwen2VL model(package, device, precision, streams, batch_mode, cache_dir,
                              min_image_tokens, max_image_tokens, max_length);
        std::cerr << "[gme_embed] device=" << model.device() << " precision=" << model.precision()
                  << " dim=" << model.dim() << " streams=" << streams
                  << " batch_mode=" << model.batch_mode() << " batch_size=" << batch_size << "\n";

        if (benchmark) {
            // One process, selected modes: amortizes model compile. Image is only
            // required when image/fused is selected, enabling text-only sweeps.
            const bool run_text = contains_mode(benchmark_modes, "text");
            const bool run_query = contains_mode(benchmark_modes, "query");
            const bool run_image = contains_mode(benchmark_modes, "image");
            const bool run_fused = contains_mode(benchmark_modes, "fused");
            if (!run_text && !run_query && !run_image && !run_fused) {
                std::cerr << "benchmark-modes must include text, query, image, or fused\n";
                return 2;
            }
            if ((run_text || run_query || run_fused) && texts.empty()) {
                std::cerr << "selected benchmark modes need one TEXT\n";
                return 2;
            }
            if ((run_image || run_fused) && images.empty()) {
                std::cerr << "selected image/fused benchmark modes need one --image\n";
                return 2;
            }
            if (bench_batch < 1) { std::cerr << "--batch must be >= 1\n"; return 2; }
            // Repeat the given sample --batch times. Every row is then identical, which
            // is exactly what a batch-scaling measurement wants: the only variable is
            // how many samples share a call, not how ragged they are.
            const std::vector<std::string> bench_texts(
                static_cast<std::size_t>(bench_batch), texts.empty() ? std::string() : texts[0]);
            const std::vector<std::string> bench_images(
                static_cast<std::size_t>(bench_batch), images.empty() ? std::string() : images[0]);
            if (warmup < 1) { std::cerr << "--warmup must be >= 1\n"; return 2; }
            auto bench = [&](const char* name, const std::function<void()>& fn) {
                for (int w = 0; w < warmup; ++w) fn();  // let the GPU clock ramp before timing
                auto t0 = std::chrono::steady_clock::now();
                for (int r = 0; r < repeat; ++r) fn();
                auto t1 = std::chrono::steady_clock::now();
                double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / repeat;
                std::cout << "[bench] device=" << device << " precision=" << precision
                          << " mode=" << name << " batch=" << bench_batch
                          << " streams=" << streams << " batch_mode=" << model.batch_mode()
                          << " avg_ms=" << ms << " ms_per_embedding=" << ms / bench_batch << "\n";
            };
            if (run_text) bench("text", [&] { model.encode_text(bench_texts, std::nullopt, false, batch_size); });
            if (run_query) bench("query", [&] { model.encode_text(bench_texts, instruction_opt, true, batch_size); });
            if (run_image) bench("image", [&] { model.encode_image(bench_images, std::nullopt, true, batch_size); });
            if (run_fused) bench("fused", [&] { model.encode_fused(bench_texts, bench_images, std::nullopt, true, batch_size); });
            return 0;
        }

        gme::Embedding emb;
        if (mode == "text") {
            emb = model.encode_text(texts, instruction_opt, is_query, batch_size);
        } else if (mode == "image") {
            emb = model.encode_image(images, instruction_opt, is_query, batch_size);
        } else if (mode == "fused") {
            emb = model.encode_fused(texts, images, instruction_opt, is_query, batch_size);
        } else {
            usage(argv[0]);
        }

        std::cerr << "[gme_embed] rows=" << emb.rows << " dim=" << emb.dim << "\n";
        for (std::size_t r = 0; r < emb.rows; ++r) {
            const float* v = emb.row(r);
            double norm = 0;
            for (std::size_t d = 0; d < emb.dim; ++d) norm += double(v[d]) * v[d];
            std::cout << "row " << r << " l2=" << norm << " [" << v[0] << ", " << v[1]
                      << ", " << v[2] << ", ...]\n";
        }
        if (!out.empty()) { dump(out, emb); std::cerr << "[gme_embed] wrote " << out << "\n"; }
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
