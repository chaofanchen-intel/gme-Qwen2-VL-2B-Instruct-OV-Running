// Standalone oneDNN matmul micro-benchmark for the GME INT4 weight-compressed GEMM.
//
// Why not benchdnn: on this Xe3 driver benchdnn's --mode=P wedges inside
// stream_t::wait() (see docs/06_onednn_investigation.md).  Both its OpenCL stream
// profiler and its threaded execution context hang, and neither has a supported
// escape hatch.  Everything measured here is a plain host-side wall clock around a
// batch of submissions followed by a single stream wait, which is exactly how the
// OpenVINO GPU plugin drives the same primitive.
//
// Build: see build_probe.sh
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <numeric>
#include <string>
#include <vector>

#include "oneapi/dnnl/dnnl.hpp"

using namespace dnnl;
using dt = memory::data_type;
using tag = memory::format_tag;

namespace {

struct Case {
    std::string name;
    dt src_dt, wei_dt;
    bool wei_scales_grouped;  // per-(group_k x 1) weight scales
    int wei_scale_group;      // K group size when grouped; 0 = per-output-channel
    bool wei_zp;              // asymmetric weights
    bool src_scales;          // per-token dynamic activation scales
    bool swiglu;              // eltwise_swish + binary_mul post-op chain
    tag wtag;                 // weight layout; `any` lets oneDNN choose
    // Layout of the grouped scale/zp tensors, which are [1, K/group, N].  `abc` is
    // N-contiguous, `cab` is K-contiguous.  This has to be varied independently of
    // wtag: comparing a cab-weight case against an abc-weight case while holding the
    // scales N-contiguous would credit the weight layout for a scale-layout effect.
    tag sctag = tag::abc;
};

struct Shape {
    std::string name;
    memory::dim m, k, n;
    // The model does not always present the token count as M.  up_proj and qkv are
    // dispatched as `1024x1x1536`, i.e. batch=1024 / M=1 with broadcast weights,
    // while gate/down/o_proj get the expected `1x1024x1536`.  batched_m1 reproduces
    // the former so the two layouts can be priced against each other.
    bool batched_m1 = false;
};

// Mask conventions for a 3D matmul (batch, M, K) x (batch, K, N):
//   weights argument dims are (batch, K, N); K is dim 1, N is dim 2.
// Mask 7 (all three dims) is what the OpenVINO plugin emits -- see the
// `attr-scales:src0:7:...+wei:7:...` line in the model's ONEDNN_VERBOSE output.
constexpr int WEI_MASK_N = 1 << 2;  // per output channel
constexpr int WEI_MASK_KN = 7;      // grouped over K, per N
constexpr int SRC_MASK_MK = 7;      // grouped over K, per M

double now_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

// Buffer contents do not change which kernel runs or what it costs, but leaving GPU
// memory uninitialized can seed f16 buffers with NaN/denormal bit patterns, so every
// buffer is zeroed once before the timed loop.
void zero(memory &m) {
    if (!m) return;
    size_t bytes = m.get_desc().get_size();
    void *p = m.map_data();
    if (p) std::memset(p, 0, bytes);
    m.unmap_data(p);
}

}  // namespace

int main(int argc, char **argv) {
    const int warmup = 20;
    const int iters = argc > 1 ? std::atoi(argv[1]) : 50;

    engine eng(engine::kind::gpu, 0);
    stream strm(eng);

    // All five GME linear layers.  `down` was missing from the first sweep, which is
    // exactly the one that turned out to REGRESS under abc (K=8960, N=1536), so the
    // sweep below deliberately brackets it in both K and N to find where the
    // cab/abc crossover actually sits rather than special-casing one shape.
    std::vector<Shape> shapes = {
        {"gate 1x1024x1536:1536x8960", 1024, 1536, 8960, false},
        {"up   1024x1x1536:1536x8960", 1024, 1536, 8960, true},
        {"down 1x1024x8960:8960x1536", 1024, 8960, 1536, false},
        {"qkv  1024x1x1536:1536x2048", 1024, 1536, 2048, true},
        {"oprj 1x1024x1536:1536x1536", 1024, 1536, 1536, false},
        // crossover sweep: hold K, vary N
        {"K1536 N1024", 1024, 1536, 1024, false},
        {"K1536 N2048", 1024, 1536, 2048, false},
        {"K1536 N4096", 1024, 1536, 4096, false},
        {"K8960 N1024", 1024, 8960, 1024, false},
        {"K8960 N2048", 1024, 8960, 2048, false},
        {"K8960 N4096", 1024, 8960, 4096, false},
        {"K8960 N8960", 1024, 8960, 8960, false},
    };

    // The in-model rule only rewrites the layout of the wide gate/up weights (N=8960,
    // K=1536), and end-to-end that helped at 512/1024 tokens but hurt at 32.  Since the
    // weights are identical in all three runs, the token count must be what flips the
    // preference, so GEMM_PROBE_SHAPES=msweep prices exactly that shape at a range of
    // token counts, in both of the src forms the model dispatches.
    if (const char *sel = getenv("GEMM_PROBE_SHAPES")) {
        if (std::string(sel) == "msweep") {
            shapes.clear();
            for (memory::dim m : {16, 32, 64, 128, 256, 512, 1024}) {
                shapes.push_back({"M" + std::to_string(m) + " 1xMxK", m, 1536, 8960, false});
                shapes.push_back({"M" + std::to_string(m) + " Mx1xK", m, 1536, 8960, true});
            }
        }
    }

    std::vector<Case> cases = {
        // A/A_abc/A_any are the SAME problem, differing only in weight layout.
        // The OpenVINO GPU plugin reorders INT4 weights to `cab` at compile time, so
        // A_cab is what GME actually executes; the other two are the comparison.
        {"A_gme_cab",         dt::s8,  dt::u4,  true, 128, true,  true,  false, tag::cab},
        {"A_abc",             dt::s8,  dt::u4,  true, 128, true,  true,  false, tag::abc},
        {"A_any",             dt::s8,  dt::u4,  true, 128, true,  true,  false, tag::any},
        // Same problem again, now varying the scale/zp layout instead.
        {"A_cab_scK",         dt::s8,  dt::u4,  true, 128, true,  true,  false, tag::cab, tag::cab},
        {"A_abc_scK",         dt::s8,  dt::u4,  true, 128, true,  true,  false, tag::abc, tag::cab},
        // Same as A_cab plus the SwiGLU post-op chain the gate projection carries.
        {"A2_cab_swiglu",     dt::s8,  dt::u4,  true, 128, true,  true,  true,  tag::cab},
        // Symmetric weights: removes the zero-point compensation entirely.
        {"B_cab_symmetric",   dt::s8,  dt::u4,  true, 128, false, true,  false, tag::cab},
        {"B_abc_symmetric",   dt::s8,  dt::u4,  true, 128, false, true,  false, tag::abc},
        // s8 weights, same grouped scales: prices the u4 unpack.
        {"C_s8wei_grouped",   dt::s8,  dt::s8,  true, 128, false, true,  false, tag::cab},
        // u4 weights, per-output-channel scales: removes the K-loop grouping.
        {"D_cab_per_oc",      dt::s8,  dt::u4,  false, 0,  false, true,  false, tag::cab},
        {"D_abc_per_oc",      dt::s8,  dt::u4,  false, 0,  false, true,  false, tag::abc},
        // Coarser groups, to price the grouping itself.
        {"D2_cab_group512",   dt::s8,  dt::u4,  true, 512, true,  true,  false, tag::cab},
        // f16 activations against the real u4 weights.
        {"E_f16src_u4wei",    dt::f16, dt::u4,  true, 128, true,  false, false, tag::cab},
        // Dense references: the f16 and int8 ceilings.
        {"F_f16_dense",       dt::f16, dt::f16, false, 0,  false, false, false, tag::any},
        {"G_s8_dense",        dt::s8,  dt::s8,  false, 0,  false, false, false, tag::any},
    };

    printf("%-22s", "case");
    for (auto &s : shapes) printf("%28s", s.name.c_str());
    printf("\n%s\n", std::string(22 + 28 * shapes.size(), '-').c_str());

    std::map<std::string, double> baseline;

    // A sweep only needs the two layouts being compared; GEMM_PROBE_CASES keeps the run
    // short without changing what any individual cell measures.
    const char *case_filter = getenv("GEMM_PROBE_CASES");

    for (auto &c : cases) {
        if (case_filter && std::string(case_filter).find(c.name) == std::string::npos)
            continue;
        printf("%-22s", c.name.c_str());
        fflush(stdout);
        for (auto &s : shapes) {
            // batched_m1 puts the token count in the batch dim and leaves M=1, with
            // the weights broadcast across the batch -- exactly what the model does
            // for up_proj and qkv.
            memory::dims src_dims = s.batched_m1 ? memory::dims{s.m, 1, s.k}
                                                 : memory::dims{1, s.m, s.k};
            memory::dims wei_dims = {1, s.k, s.n};
            memory::dims dst_dims = s.batched_m1 ? memory::dims{s.m, 1, s.n}
                                                 : memory::dims{1, s.m, s.n};

            // A grouped scale/zp only makes sense when the group divides K evenly;
            // K=8960 is not a multiple of 512, so that combination is skipped rather
            // than silently reported as an unsupported kernel.
            if (c.wei_scales_grouped && s.k % c.wei_scale_group != 0) {
                printf("%28s", "n/a (K%group)");
                fflush(stdout);
                continue;
            }

            auto src_md = memory::desc(src_dims, c.src_dt, tag::abc);
            auto wei_md = memory::desc(wei_dims, c.wei_dt, c.wtag);
            auto dst_md = memory::desc(dst_dims, dt::f16, tag::abc);

            primitive_attr attr;
            attr.set_scratchpad_mode(scratchpad_mode::user);
            if (c.src_scales)
                attr.set_scales(DNNL_ARG_SRC, SRC_MASK_MK, {1, s.k}, dt::f16);
            if (c.wei_scales_grouped)
                attr.set_scales(DNNL_ARG_WEIGHTS, WEI_MASK_KN,
                                {c.wei_scale_group, 1}, dt::f16);
            else if (c.wei_dt == dt::u4 || c.wei_dt == dt::s8)
                attr.set_scales(DNNL_ARG_WEIGHTS, WEI_MASK_N, {}, dt::f16);
            if (c.wei_zp)
                attr.set_zero_points(DNNL_ARG_WEIGHTS, WEI_MASK_KN,
                                     {c.wei_scale_group, 1}, dt::u8);
            if (c.swiglu) {
                post_ops po;
                po.append_eltwise(algorithm::eltwise_swish, 1.0f, 0.0f);
                po.append_binary(algorithm::binary_mul, dst_md);
                attr.set_post_ops(po);
            }

            matmul::primitive_desc pd;
            try {
                pd = matmul::primitive_desc(eng, src_md, wei_md, dst_md, attr);
            } catch (const dnnl::error &e) {
                printf("%28s", "unsupported");
                fflush(stdout);
                if (getenv("PROBE_EXPLAIN")) fprintf(stderr, "\n  %s: %s\n", c.name.c_str(), e.what());
                continue;
            }

            matmul prim(pd);

            auto src = memory(pd.src_desc(), eng);
            auto wei = memory(pd.weights_desc(), eng);
            auto dst = memory(pd.dst_desc(), eng);
            auto scratch = memory(pd.scratchpad_desc(), eng);

            std::unordered_map<int, memory> args = {
                {DNNL_ARG_SRC, src}, {DNNL_ARG_WEIGHTS, wei},
                {DNNL_ARG_DST, dst}, {DNNL_ARG_SCRATCHPAD, scratch}};

            memory src_sc, wei_sc, wei_zp, mul_src;
            if (c.src_scales) {
                memory::dims sc_dims = s.batched_m1 ? memory::dims{s.m, 1, 1}
                                                    : memory::dims{1, s.m, 1};
                src_sc = memory({sc_dims, dt::f16, tag::abc}, eng);
                args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC] = src_sc;
            }
            if (c.wei_scales_grouped) {
                wei_sc = memory({{1, s.k / c.wei_scale_group, s.n}, dt::f16, c.sctag}, eng);
                args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS] = wei_sc;
            } else if (c.wei_dt == dt::u4 || c.wei_dt == dt::s8) {
                wei_sc = memory({{1, 1, s.n}, dt::f16, tag::abc}, eng);
                args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS] = wei_sc;
            }
            if (c.wei_zp) {
                wei_zp = memory({{1, s.k / c.wei_scale_group, s.n}, dt::u8, c.sctag}, eng);
                args[DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS] = wei_zp;
            }
            if (c.swiglu) {
                mul_src = memory(dst_md, eng);
                args[DNNL_ARG_ATTR_MULTIPLE_POST_OP(1) | DNNL_ARG_SRC_1] = mul_src;
            }

            for (auto &kv : args) zero(const_cast<memory &>(kv.second));

            // PROBE_ROTATE=N cycles the timed loop over N distinct weight buffers
            // instead of hammering one.  Reusing a single buffer leaves the weights
            // resident in GPU cache, which the model never gets: it walks 28 layers
            // of distinct weights (~655 MB) per forward, so every GEMM sees its
            // weights cold.  Rotating reproduces that streaming behaviour.
            std::vector<memory> wei_rot;
            int rotate = getenv("PROBE_ROTATE") ? std::atoi(getenv("PROBE_ROTATE")) : 1;
            if (rotate > 1) {
                for (int r = 0; r < rotate; r++) {
                    memory w(pd.weights_desc(), eng);
                    zero(w);
                    wei_rot.push_back(w);
                }
            }

            for (int i = 0; i < warmup; i++) prim.execute(strm, args);
            strm.wait();

            // By default the whole batch is submitted then waited on once, so
            // successive kernels pipeline and dispatch latency is hidden.  In the
            // model every GEMM depends on the op before it, so the queue drains
            // between kernels.  PROBE_SYNC=1 waits per iteration to reproduce that.
            const bool sync_each = getenv("PROBE_SYNC") && getenv("PROBE_SYNC")[0] == '1';
            double t0 = now_ms();
            for (int i = 0; i < iters; i++) {
                if (rotate > 1) args[DNNL_ARG_WEIGHTS] = wei_rot[i % rotate];
                prim.execute(strm, args);
                if (sync_each) strm.wait();
            }
            strm.wait();
            double ms = (now_ms() - t0) / iters;

            double gflop = 2.0 * s.m * s.k * s.n / 1e9;
            double tflops = gflop / (ms / 1e3) / 1e3;

            if (c.name == "A_gme_cab") baseline[s.name] = ms;
            double ratio = baseline.count(s.name) ? baseline[s.name] / ms : 0.0;

            printf("%12.3fms%7.1fT%7.2fx", ms, tflops, ratio);
            fflush(stdout);
        }
        printf("\n");
    }
    printf("\n(ratio is speedup vs A_gme_cab for the same shape)\n");
    return 0;
}
