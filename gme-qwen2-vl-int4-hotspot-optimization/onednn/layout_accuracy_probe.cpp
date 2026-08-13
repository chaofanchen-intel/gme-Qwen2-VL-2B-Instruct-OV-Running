// Does switching the u4 weight layout from K-contiguous (`cab`, what GME runs today) to
// N-contiguous (`abc`) change the *numerics* of the jit:gemm kernel, or only its speed?
//
// Why this matters: the in-model change (weights physically transposed by a new
// get_weights_reorder(), matmul md switched to `abc`) is a pure layout change and should be
// bit-exact.  It is not -- end-to-end embeddings move by up to ~1e-2 relative.  There are
// only two possible causes:
//
//   (a) the plumbing is wrong somewhere (reorder, descriptor, or scale/zp interpretation),
//       in which case the same weights fed through `cab` and `abc` disagree here too; or
//   (b) oneDNN simply dispatches a different gemm kernel for `abc`, and that kernel
//       accumulates differently -- a numerical difference, not an error.
//
// So: build one reference set of weights/scales/zero-points, materialize it in both layouts,
// run both, and compare each against an f64 host reference computed from the same logical
// values.  If both sit at the same error level, it is (b).  If `abc` is far off, it is (a),
// and the per-shape error pattern says where to look.
//
// Shapes are the real GME projections; the scale/zp order is g-outer, which is what
// scale_layout_probe.cpp showed oneDNN expects regardless of weight layout.
#include <oneapi/dnnl/dnnl.hpp>

#include <algorithm>
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

constexpr int GROUP = 128;
constexpr int WEI_MASK_KN = 7;

void set_nibble(uint8_t *p, size_t i, uint8_t v) {
    if (i & 1)
        p[i / 2] = static_cast<uint8_t>((p[i / 2] & 0x0f) | ((v & 0xf) << 4));
    else
        p[i / 2] = static_cast<uint8_t>((p[i / 2] & 0xf0) | (v & 0xf));
}

// Deterministic pseudo-random values with the same statistics an NNCF-compressed tensor has:
// full 4-bit range for weights, zero-points near the middle, small positive scales.
uint8_t wei_val(int k, int n) { return static_cast<uint8_t>((k * 7919 + n * 104729) % 16); }
int8_t src_val(int m, int k) { return static_cast<int8_t>(((m * 31337 + k * 6151) % 255) - 127); }
uint8_t zp_val(int g, int n) { return static_cast<uint8_t>(((g * 13 + n * 7) % 5) + 6); }
// Powers of two so the scale itself is f16-exact and cannot contribute rounding of its own.
float scale_val(int g, int n) { return std::ldexp(1.0f, -(((g * 3 + n * 5) % 4) + 6)); }
float src_scale_val(int m) { return std::ldexp(1.0f, -((m % 3) + 2)); }

uint16_t f32_to_f16_pow2(float f) {
    int e = static_cast<int>(std::lround(std::log2(f)));
    return static_cast<uint16_t>((15 + e) << 10);
}

float f16_to_f32(uint16_t h) {
    int sign = (h >> 15) & 1, exp = (h >> 10) & 0x1f, man = h & 0x3ff;
    float v;
    if (exp == 0)
        v = std::ldexp(static_cast<float>(man), -24);
    else if (exp == 31)
        v = man ? NAN : INFINITY;
    else
        v = std::ldexp(static_cast<float>(man | 0x400), exp - 25);
    return sign ? -v : v;
}

struct Shape {
    const char *name;
    int m, k, n;
};

struct Run {
    bool built = false;
    double max_rel = 0.0;   // vs the f64 host reference
    double rel_l2 = 0.0;
    std::string kernel;
};

// Only a slice of rows is checked: M=1024 x N=8960 in f64 on the host is slow and the error
// statistics are already stable over a few hundred rows.
constexpr int CHECK_ROWS = 64;

std::vector<double> reference(const Shape &s) {
    std::vector<double> out(static_cast<size_t>(CHECK_ROWS) * s.n, 0.0);
    for (int m = 0; m < CHECK_ROWS; ++m)
        for (int n = 0; n < s.n; ++n) {
            double acc = 0.0;
            for (int k = 0; k < s.k; ++k) {
                int g = k / GROUP;
                double w = (static_cast<double>(wei_val(k, n)) - zp_val(g, n)) * scale_val(g, n);
                acc += static_cast<double>(src_val(m, k)) * src_scale_val(m) * w;
            }
            out[static_cast<size_t>(m) * s.n + n] = acc;
        }
    return out;
}

Run run(engine &eng, stream &strm, const Shape &s, tag wtag, const std::vector<double> &ref) {
    Run r;
    const int G = s.k / GROUP;

    memory::desc src_md({1, s.m, s.k}, dt::s8, tag::abc);
    memory::desc wei_md({1, s.k, s.n}, dt::u4, wtag);
    memory::desc dst_md({1, s.m, s.n}, dt::f16, tag::abc);

    primitive_attr attr;
    attr.set_scales(DNNL_ARG_WEIGHTS, WEI_MASK_KN, {GROUP, 1}, dt::f16);
    attr.set_zero_points(DNNL_ARG_WEIGHTS, WEI_MASK_KN, {GROUP, 1}, dt::u8);
    attr.set_scales(DNNL_ARG_SRC_0, WEI_MASK_KN, {1, s.k}, dt::f16);

    matmul::primitive_desc pd;
    try {
        pd = matmul::primitive_desc(eng, src_md, wei_md, dst_md, attr);
    } catch (const error &) {
        return r;
    }
    r.built = true;
    r.kernel = pd.impl_info_str();

    memory src(src_md, eng), wei(wei_md, eng), dst(dst_md, eng);
    memory::desc sc_md({G * s.n}, dt::f16, tag::a);
    memory::desc zp_md({G * s.n}, dt::u8, tag::a);
    memory::desc ssc_md({s.m}, dt::f16, tag::a);
    memory sc(sc_md, eng), zp(zp_md, eng), ssc(ssc_md, eng);

    {
        auto *p = static_cast<int8_t *>(src.map_data());
        for (int m = 0; m < s.m; ++m)
            for (int k = 0; k < s.k; ++k) p[static_cast<size_t>(m) * s.k + k] = src_val(m, k);
        src.unmap_data(p);
    }
    {
        // Identical logical contents, laid out per the tag under test.  This is the host-side
        // equivalent of what the new get_weights_reorder() does on the GPU.
        auto *p = static_cast<uint8_t *>(wei.map_data());
        std::memset(p, 0, wei_md.get_size());
        const bool k_contig = (wtag == tag::cab || wtag == tag::acb);
        for (int k = 0; k < s.k; ++k)
            for (int n = 0; n < s.n; ++n)
                set_nibble(p, k_contig ? static_cast<size_t>(n) * s.k + k
                                       : static_cast<size_t>(k) * s.n + n,
                           wei_val(k, n));
        wei.unmap_data(p);
    }
    {
        auto *p = static_cast<uint16_t *>(sc.map_data());
        auto *q = static_cast<uint8_t *>(zp.map_data());
        for (int g = 0; g < G; ++g)
            for (int n = 0; n < s.n; ++n) {
                size_t i = static_cast<size_t>(g) * s.n + n;  // g-outer
                p[i] = f32_to_f16_pow2(scale_val(g, n));
                q[i] = zp_val(g, n);
            }
        sc.unmap_data(p);
        zp.unmap_data(q);
    }
    {
        auto *p = static_cast<uint16_t *>(ssc.map_data());
        for (int m = 0; m < s.m; ++m) p[m] = f32_to_f16_pow2(src_scale_val(m));
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
        double num = 0.0, den = 0.0;
        for (int m = 0; m < CHECK_ROWS; ++m)
            for (int n = 0; n < s.n; ++n) {
                size_t i = static_cast<size_t>(m) * s.n + n;
                double got = f16_to_f32(p[i]);
                double want = ref[i];
                double d = got - want;
                num += d * d;
                den += want * want;
                double scale = std::max(1e-3, std::abs(want));
                r.max_rel = std::max(r.max_rel, std::abs(d) / scale);
            }
        r.rel_l2 = std::sqrt(num / den);
        dst.unmap_data(p);
    }
    return r;
}

}  // namespace

int main() {
    engine eng(engine::kind::gpu, 0);
    stream strm(eng);

    printf("oneDNN %s -- does the cab->abc weight layout change gemm numerics?\n", dnnl_version()->hash);
    printf("group=%d, asymmetric u4 weights, s8 src with per-token scales, g-outer scale/zp\n", GROUP);
    printf("error is measured against an f64 host reference over the first %d rows\n\n", CHECK_ROWS);

    Shape shapes[] = {
        {"o_proj  K1536 N1536", 1024, 1536, 1536},
        {"qkv     K1536 N2048", 1024, 1536, 2048},
        {"gate/up K1536 N8960", 1024, 1536, 8960},
        {"down    K8960 N1536", 1024, 8960, 1536},
    };

    printf("%-22s %-6s %14s %12s  %s\n", "shape", "layout", "max_rel", "rel_l2", "kernel");
    for (auto &s : shapes) {
        auto ref = reference(s);
        for (auto wtag : {tag::cab, tag::abc}) {
            Run r = run(eng, strm, s, wtag, ref);
            const char *lname = (wtag == tag::cab) ? "cab" : "abc";
            if (!r.built)
                printf("%-22s %-6s   pd unsupported\n", s.name, lname);
            else
                printf("%-22s %-6s %14.4g %12.4g  %s\n", s.name, lname, r.max_rel, r.rel_l2,
                       r.kernel.c_str());
        }
    }

    printf("\nRead: comparable error for both layouts => the layout swap is numerically neutral\n"
           "and any end-to-end difference is kernel-dispatch noise.  A much larger error for\n"
           "`abc` => the weights or the scale/zero-point indexing is being misinterpreted.\n");
    return 0;
}
