// gme_embed: CLI for the GME-Qwen2-VL OpenVINO C++ runtime.
//
//   gme_embed --package DIR [--device CPU] [--precision int8]
//             --mode text|image|fused [--instruction STR] [--is-query 0|1]
//             [--image PATH ...] [--out FILE] [TEXT ...]
//   gme_embed --package DIR --benchmark [--benchmark-modes text,query]
//
// --out writes a binary blob {"GEMB", int32 rows, int32 dim, float32[rows*dim]}
// consumed by 08_validate_cpp.py to check cosine against the golden embeddings.
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include "gme_qwen2vl.hpp"

namespace {

[[noreturn]] void usage(const char* argv0) {
    std::cerr << "usage: " << argv0
              << " --package DIR [--device CPU] [--precision int8]"
                 " --mode text|image|fused [--instruction STR] [--is-query 0|1]"
                 " [--image PATH ...] [--out FILE] [TEXT ...]\n"
                 "       "
              << argv0
              << " --package DIR --benchmark [--benchmark-modes text,query,image,fused]"
                 " [--repeat N] [--instruction STR] [--image PATH] TEXT\n";
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
    std::string benchmark_modes = "text,query,image,fused";
    bool is_query = true, benchmark = false;
    int repeat = 1;
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
        else if (a == "--instruction") instruction = next("--instruction");
        else if (a == "--is-query") is_query = next("--is-query") != "0";
        else if (a == "--image") images.push_back(next("--image"));
        else if (a == "--out") out = next("--out");
        else if (a == "--repeat") repeat = std::stoi(next("--repeat"));
        else if (a == "--benchmark") benchmark = true;
        else if (a == "--benchmark-modes") benchmark_modes = next("--benchmark-modes");
        else if (a == "-h" || a == "--help") usage(argv[0]);
        else texts.push_back(a);
    }
    if (package.empty() || (!benchmark && mode.empty())) usage(argv[0]);

    try {
          gme::GmeQwen2VL model(package, device, precision);
        std::cerr << "[gme_embed] device=" << model.device() << " precision=" << model.precision()
              << " dim=" << model.dim() << "\n";

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
            auto bench = [&](const char* name, const std::function<void()>& fn) {
                fn();  // warmup
                auto t0 = std::chrono::steady_clock::now();
                for (int r = 0; r < repeat; ++r) fn();
                auto t1 = std::chrono::steady_clock::now();
                double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / repeat;
                std::cout << "[bench] device=" << device << " precision=" << precision
                          << " mode=" << name << " avg_ms=" << ms << "\n";
            };
            if (run_text) bench("text", [&] { model.encode_text({texts[0]}, "", false); });
            if (run_query) bench("query", [&] { model.encode_text({texts[0]}, instruction, true); });
            if (run_image) bench("image", [&] { model.encode_image({images[0]}); });
            if (run_fused) bench("fused", [&] { model.encode_fused({texts[0]}, {images[0]}); });
            return 0;
        }

        gme::Embedding emb;
        if (mode == "text") {
            emb = model.encode_text(texts, instruction, is_query);
        } else if (mode == "image") {
            emb = model.encode_image(images, instruction, is_query);
        } else if (mode == "fused") {
            emb = model.encode_fused(texts, images, instruction, is_query);
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
