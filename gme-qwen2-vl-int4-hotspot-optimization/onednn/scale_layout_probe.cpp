// Is oneDNN's expected layout of the *decompression scale / zero-point* buffers coupled
// to the physical layout of the u4 weight tensor?
//
// Why this decides the `cab` -> `abc` weight-layout optimization:
//
// The OpenVINO GPU plugin does not describe the scale/zp tensors properly.  In
// fully_connected_onednn.cpp::get_arguments() it passes them as a *flattened 1-D* memory
// desc:
//
//     desc = onednn::layout_to_memory_desc_flatten(scale_mem->get_layout(), format_tag::a);
//
// so all stride information is thrown away and oneDNN has to fall back on some canonical
// element order for a grouped, mask=7 scale tensor.  The GME model is numerically correct
// today with K-contiguous (`cab`) weights, which means that canonical order happens to
// agree with how NNCF laid the scale constant out ([N, K/group], i.e. n-outer/g-inner).
//
// If that canonical order is derived from the weight layout, flipping the weights to
// `abc` silently changes which scale each weight element gets -- a correctness bug with
// no test to catch it, because the flat md makes it un-diagnosable from the outside.  And
// unlike the weights, there is no reorder hook for scale/zp inputs, so a fix would have
// to transpose those constants in a transformation pass.
//
// So: compute a host reference dequant-GEMM, then run all four combinations of
// {weights cab, weights abc} x {scale order n-outer, scale order g-outer} and see which
// ones reproduce it.  Small shapes, exact-representable values, so "matches" is
// unambiguous rather than a tolerance judgement.
#include <oneapi/dnnl/dnnl.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace dnnl;
using dt = memory::data_type;
using tag = memory::format_tag;

namespace {

// Problem is deliberately tiny and non-square so an index mixup cannot cancel out.
constexpr int M = 4;
constexpr int K = 256;
constexpr int N = 8;
constexpr int GROUP = 128;
constexpr int G = K / GROUP;  // 2 groups

constexpr int WEI_MASK_KN = 7;

uint8_t get_nibble(const uint8_t *p, size_t i) {
    return (i & 1) ? (p[i / 2] >> 4) : (p[i / 2] & 0xf);
}
void set_nibble(uint8_t *p, size_t i, uint8_t v) {
    if (i & 1)
        p[i / 2] = static_cast<uint8_t>((p[i / 2] & 0x0f) | ((v & 0xf) << 4));
    else
        p[i / 2] = static_cast<uint8_t>((p[i / 2] & 0xf0) | (v & 0xf));
}

// Deterministic, no RNG, and every value is exactly representable in f16 so the
// comparison against the host reference can be exact.
uint8_t wei_val(int k, int n) { return static_cast<uint8_t>((k * 5 + n * 3) & 0xf); }
int8_t src_val(int m, int k) { return static_cast<int8_t>(((m * 7 + k) % 9) - 4); }
// Scales are powers of two -> f16-exact, and distinct per (group, n) so a g<->n swap
// changes the result.
float scale_val(int g, int n) { return std::ldexp(1.0f, -(((g * 3 + n) % 5) + 1)); }
uint8_t zp_val(int g, int n) { return static_cast<uint8_t>(((g * 2 + n) % 7) + 1); }
float src_scale_val(int m) { return std::ldexp(1.0f, -(m % 3)); }

// Host reference: dequantize asymmetric grouped u4 weights, then plain f32 GEMM.
std::vector<float> reference() {
    std::vector<float> out(M * N, 0.0f);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                int g = k / GROUP;
                float w = (static_cast<float>(wei_val(k, n)) - static_cast<float>(zp_val(g, n))) *
                          scale_val(g, n);
                acc += static_cast<float>(src_val(m, k)) * src_scale_val(m) * w;
            }
            out[m * N + n] = acc;
        }
    return out;
}

enum class ScaleOrder {
    n_outer,  // index = n*G + g  -- how NNCF/OpenVINO lays out a [N, K/group] constant
    g_outer,  // index = g*N + n  -- dense [1, K/group, N] in `abc`
};

struct Result {
    bool built = false;
    bool matches = false;
    double max_rel = 0.0;
};

Result run(engine &eng, stream &strm, tag wtag, ScaleOrder so, const std::vector<float> &ref) {
    Result r;

    // Logical dims are [1, K, N] regardless of layout; only the tag differs.
    memory::desc src_md({1, M, K}, dt::s8, tag::abc);
    memory::desc wei_md({1, K, N}, dt::u4, wtag);
    memory::desc dst_md({1, M, N}, dt::f16, tag::abc);

    primitive_attr attr;
    attr.set_scales(DNNL_ARG_WEIGHTS, WEI_MASK_KN, {GROUP, 1}, dt::f16);
    attr.set_zero_points(DNNL_ARG_WEIGHTS, WEI_MASK_KN, {GROUP, 1}, dt::u8);
    attr.set_scales(DNNL_ARG_SRC_0, WEI_MASK_KN, {1, K}, dt::f16);

    matmul::primitive_desc pd;
    try {
        pd = matmul::primitive_desc(eng, src_md, wei_md, dst_md, attr);
    } catch (const error &e) {
        return r;
    }
    r.built = true;

    memory src(src_md, eng), wei(wei_md, eng), dst(dst_md, eng);
    // Exactly what the plugin does: flat 1-D descs, all strides discarded.
    memory::desc sc_md({G * N}, dt::f16, tag::a);
    memory::desc zp_md({G * N}, dt::u8, tag::a);
    memory::desc ssc_md({M}, dt::f16, tag::a);
    memory sc(sc_md, eng), zp(zp_md, eng), ssc(ssc_md, eng);

    {
        auto *p = static_cast<int8_t *>(src.map_data());
        for (int m = 0; m < M; ++m)
            for (int k = 0; k < K; ++k) p[m * K + k] = src_val(m, k);
        src.unmap_data(p);
    }
    {
        auto *p = static_cast<uint8_t *>(wei.map_data());
        std::memset(p, 0, wei_md.get_size());
        const bool k_contig = (wtag == tag::cab || wtag == tag::acb);
        for (int k = 0; k < K; ++k)
            for (int n = 0; n < N; ++n)
                set_nibble(p, k_contig ? (size_t)n * K + k : (size_t)k * N + n, wei_val(k, n));
        wei.unmap_data(p);
    }
    {
        auto *p = static_cast<uint16_t *>(sc.map_data());
        auto *q = static_cast<uint8_t *>(zp.map_data());
        for (int g = 0; g < G; ++g)
            for (int n = 0; n < N; ++n) {
                size_t i = (so == ScaleOrder::n_outer) ? (size_t)n * G + g : (size_t)g * N + n;
                float f = scale_val(g, n);
                // f16 bit pattern for a power of two, built by hand to avoid needing _Float16.
                int e = static_cast<int>(std::log2(f));
                p[i] = static_cast<uint16_t>((15 + e) << 10);
                q[i] = zp_val(g, n);
            }
        sc.unmap_data(p);
        zp.unmap_data(q);
    }
    {
        auto *p = static_cast<uint16_t *>(ssc.map_data());
        for (int m = 0; m < M; ++m) {
            int e = static_cast<int>(std::log2(src_scale_val(m)));
            p[m] = static_cast<uint16_t>((15 + e) << 10);
        }
        ssc.unmap_data(p);
    }

    matmul(pd).execute(strm, {{DNNL_ARG_SRC, src},
                              {DNNL_ARG_WEIGHTS, wei},
                              {DNNL_ARG_DST, dst},
                              {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, sc},
                              {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS, zp},
                              {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_0, ssc}});
    strm.wait();

    {
        auto *p = static_cast<uint16_t *>(dst.map_data());
        for (int i = 0; i < M * N; ++i) {
            // decode f16 -> float
            uint16_t h = p[i];
            int sign = (h >> 15) & 1, exp = (h >> 10) & 0x1f, man = h & 0x3ff;
            float v;
            if (exp == 0)
                v = std::ldexp(static_cast<float>(man), -24);
            else if (exp == 31)
                v = INFINITY;
            else
                v = std::ldexp(static_cast<float>(man | 0x400), exp - 25);
            if (sign) v = -v;
            double denom = std::max(1.0, std::abs((double)ref[i]));
            r.max_rel = std::max(r.max_rel, std::abs((double)v - ref[i]) / denom);
        }
        dst.unmap_data(p);
    }
    // f16 accumulation over K=256 leaves real rounding error; 2% is loose enough to accept
    // a correct kernel and far tighter than any index permutation could sneak through.
    r.matches = r.max_rel < 2e-2;
    return r;
}

}  // namespace

int main() {
    engine eng(engine::kind::gpu, 0);
    stream strm(eng);
    auto ref = reference();

    printf("oneDNN %s -- scale/zp layout coupling to weight layout\n", dnnl_version()->hash);
    printf("M=%d K=%d N=%d group=%d (G=%d), scale/zp passed as flat 1-D md (as the plugin does)\n\n",
           M, K, N, GROUP, G);

    struct Combo {
        const char *name;
        tag wtag;
        ScaleOrder so;
    } combos[] = {
        {"wei cab + scale n-outer  (== GME today)", tag::cab, ScaleOrder::n_outer},
        {"wei cab + scale g-outer", tag::cab, ScaleOrder::g_outer},
        {"wei abc + scale n-outer", tag::abc, ScaleOrder::n_outer},
        {"wei abc + scale g-outer", tag::abc, ScaleOrder::g_outer},
    };

    for (auto &c : combos) {
        Result r = run(eng, strm, c.wtag, c.so, ref);
        if (!r.built)
            printf("  %-40s  pd unsupported\n", c.name);
        else
            printf("  %-40s  %s   max_rel=%.4g\n", c.name, r.matches ? "MATCH   " : "MISMATCH",
                   r.max_rel);
    }

    printf("\nRead: if `wei cab + scale n-outer` matches and `wei abc + scale n-outer` does\n"
           "not, the scale order IS coupled to the weight layout and the abc change needs\n"
           "the scale/zp constants transposed too.\n");
    return 0;
}
