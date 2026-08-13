// Can oneDNN reorder a *packed u4* weight tensor from K-contiguous (`cab`, what the
// OpenVINO GPU plugin hands to matmul today) to N-contiguous (`abc`, which the probe
// measured 1.25x faster on the GME MLP shape)?
//
// This is the single go/no-go question for the `cab` -> `abc` weight-layout change:
//   * if oneDNN can do the reorder, the plugin change is "create the pd with
//     format_tag::any + add a get_weights_reorder()", reusing the existing and tested
//     WeightsReorderParamsOneDNN machinery, and the u4 nibble repacking is oneDNN's
//     problem rather than ours;
//   * if it cannot, the transpose has to be open-coded on the host over packed
//     nibbles inside a GPU-plugin transformation pass, which is far more invasive.
//
// Checks, in order: (1) does the reorder pd even build, (2) is the result bit-correct
// on a tiny shape we can verify by hand, (3) what does it cost at real GME size --
// it runs once per weight at model-compile time, so it only has to be tolerable.
#include <oneapi/dnnl/dnnl.hpp>

#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

using namespace dnnl;
using dt = memory::data_type;
using tag = memory::format_tag;

namespace {

double now_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

// u4 is packed two elements per byte, low nibble first.
uint8_t get_nibble(const uint8_t *p, size_t i) {
    return (i & 1) ? (p[i / 2] >> 4) : (p[i / 2] & 0xf);
}
void set_nibble(uint8_t *p, size_t i, uint8_t v) {
    if (i & 1)
        p[i / 2] = static_cast<uint8_t>((p[i / 2] & 0x0f) | ((v & 0xf) << 4));
    else
        p[i / 2] = static_cast<uint8_t>((p[i / 2] & 0xf0) | (v & 0xf));
}

// Reference logical value of weight element (k, n); kept deliberately non-symmetric in
// k and n so a transposed-vs-untransposed mixup cannot pass by accident.
uint8_t ref_val(int k, int n) { return static_cast<uint8_t>((k * 3 + n * 7) & 0xf); }

bool check(const std::string &label, memory::dim K, memory::dim N, engine &eng, stream &strm,
           bool verify) {
    // Logical dims [1, K, N] for both, differing only in physical layout.
    memory::desc src_md({1, K, N}, dt::u4, tag::cab);  // K-contiguous: what GME runs
    memory::desc dst_md({1, K, N}, dt::u4, tag::abc);  // N-contiguous: what we want

    reorder::primitive_desc rpd;
    try {
        rpd = reorder::primitive_desc(eng, src_md, eng, dst_md);
    } catch (const error &e) {
        printf("  %-22s reorder pd FAILED: %s\n", label.c_str(), e.what());
        return false;
    }

    memory src(src_md, eng), dst(dst_md, eng);

    {
        auto *p = static_cast<uint8_t *>(src.map_data());
        std::memset(p, 0, src_md.get_size());
        // cab strides for (1, K, N): element (k, n) lives at flat index n*K + k.
        for (memory::dim k = 0; k < K; ++k)
            for (memory::dim n = 0; n < N; ++n) set_nibble(p, n * K + k, ref_val(k, n));
        src.unmap_data(p);
    }
    {
        auto *p = static_cast<uint8_t *>(dst.map_data());
        std::memset(p, 0xAA, dst_md.get_size());  // poison, so a no-op reorder is visible
        dst.unmap_data(p);
    }

    auto r = reorder(rpd);
    r.execute(strm, src, dst);
    strm.wait();

    if (verify) {
        auto *p = static_cast<uint8_t *>(dst.map_data());
        long bad = 0;
        // abc strides for (1, K, N): element (k, n) lives at flat index k*N + n.
        for (memory::dim k = 0; k < K; ++k)
            for (memory::dim n = 0; n < N; ++n) {
                uint8_t got = get_nibble(p, k * N + n);
                if (got != ref_val(k, n)) {
                    if (bad < 6)
                        printf("    mismatch (k=%ld n=%ld): got %u want %u\n", (long)k, (long)n,
                               got, ref_val(k, n));
                    ++bad;
                }
            }
        dst.unmap_data(p);
        if (bad) {
            printf("  %-22s CONTENT WRONG (%ld/%ld nibbles)\n", label.c_str(), bad, (long)(K * N));
            return false;
        }
    }

    // Compile-time cost: median of a handful of runs.
    std::vector<double> t;
    for (int i = 0; i < 7; ++i) {
        double t0 = now_ms();
        r.execute(strm, src, dst);
        strm.wait();
        t.push_back(now_ms() - t0);
    }
    std::sort(t.begin(), t.end());
    printf("  %-22s OK  K=%-5ld N=%-5ld  %8.3f ms  (%.1f MiB)\n", label.c_str(), (long)K, (long)N,
           t[t.size() / 2], K * N / 2.0 / (1 << 20));
    return true;
}

}  // namespace

int main() {
    engine eng(engine::kind::gpu, 0);
    stream strm(eng);

    printf("oneDNN %s -- u4 cab->abc reorder feasibility\n\n", dnnl_version()->hash);

    bool ok = true;
    printf("correctness (hand-checkable shapes):\n");
    ok &= check("tiny 8x4", 8, 4, eng, strm, true);
    ok &= check("small 128x256", 128, 256, eng, strm, true);
    ok &= check("group-aligned 1536x64", 1536, 64, eng, strm, true);

    printf("\ncompile-time cost at real GME weight sizes:\n");
    check("gate/up 1536x8960", 1536, 8960, eng, strm, false);
    check("down    8960x1536", 8960, 1536, eng, strm, false);
    check("qkv     1536x2048", 1536, 2048, eng, strm, false);
    check("o_proj  1536x1536", 1536, 1536, eng, strm, false);

    printf("\n%s\n", ok ? "VERDICT: u4 cab->abc reorder is supported and bit-correct."
                        : "VERDICT: u4 cab->abc reorder is NOT usable as-is.");
    return ok ? 0 : 1;
}
