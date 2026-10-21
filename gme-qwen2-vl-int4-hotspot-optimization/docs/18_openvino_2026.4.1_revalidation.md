# GME patch sweep on OpenVINO 2026.4.1

## Base and protocol

- OpenVINO source base: release tag `2026.4.1`, commit
  `e213a14725769a6a2bf86fc86650aa78a9c05d3e`.
- Model/package: GME Qwen2-VL-2B, `int4_causal_vopt`, GPU.
- Text/query workloads: 1024 effective tokens after GME prompt templating.
- Visual workloads: 448×448 and the 1260-visual-token logo; image and fused modes.
- One warm-up followed by five measured calls per cell; values are end-to-end
  arithmetic means in ms/call.
- Driver: `08_benchmark_matrix.py`, Python backend only.
- Patch stages were applied cumulatively in opt-1 → opt-2 → opt-3 → opt-4 →
  opt-5 → opt-6 order. The plugin was rebuilt after each stage.
- Runtime bindings were OpenVINO 2026.4.0 / GenAI 2026.4.0.0 from the
  pre-existing local install; the loaded GPU plugin was built from the
  2026.4.1 source base. The OpenVINO `install/` directory was not modified.

## Default-threshold performance

Text/query cells use the 1024-token workload. Visual cells are image/fused,
respectively. `opt-5`'s MVN-to-DQ fusion is opt-in and is not enabled in this
default-threshold table.

| Cumulative patch state | Text / query | 448×448 image / fused | Logo image / fused |
|---|---:|---:|---:|
| Pristine 2026.4.1 | 252.96 / 252.68 | 266.86 / 296.23 | 1913.34 / 1933.85 |
| opt-1 | 241.40 / 240.93 | 264.00 / 300.30 | 1899.75 / 1924.03 |
| opt-1 + opt-2 | 240.82 / 240.58 | 261.07 / 290.79 | 1904.92 / 1930.45 |
| opt-1 + opt-2 + opt-3 | 218.57 / 218.85 | 245.92 / 277.11 | 1846.77 / 1848.35 |
| + opt-4 | 218.10 / 219.21 | 245.00 / 275.97 | 1858.56 / 1852.13 |
| + opt-5 (default threshold) | 217.83 / 218.32 | 244.68 / 274.63 | 1856.40 / 1854.70 |
| + opt-6 | 218.47 / 218.86 | 242.22 / 273.04 | 1845.83 / 1857.42 |

The largest default-path change was opt-3: relative to pristine, the
opt-1/2/3 stack reduced 1024-token text/query latency by about 13.6%/13.4%.
The full round-8 stack reduced 448×448 image/fused latency by about 9.2%/7.8%.
Later default-threshold changes are small and should be treated as within-run
measurement noise unless repeated in an interleaved A/B test.

## Opt-in threshold checks

On the round-8 stack, five-call means were:

| Mode | Default threshold | Threshold 0 | Change |
|---|---:|---:|---:|
| 1024-token text / query | 216.71 / 217.01 | 208.70 / 208.95 | -3.7% / -3.7% |
| 448×448 image / fused | 242.22 / 273.04 | 235.03 / 234.45 | -3.0% / -14.1% |
| Logo image / fused | 1845.83 / 1857.42 | 1804.59 / 1818.08 | -2.2% / -2.1% |

These threshold comparisons use the same round-8 source and isolate the
documented opt-in path; they are not a pristine-versus-patched comparison.

## Patch application and build notes

All active individual and cumulative patch files applied cleanly to a clean
2026.4.1 source worktree. The full GPU plugin linked for every measured stage.
Opt-6 adds a C++ source file: when reusing an existing build directory, rerun
CMake configuration after applying it so the generated source list includes the
new file. A build attempted without that reconfiguration failed at link time
with an undefined QKV-kernel vtable; reconfiguration followed by a full link
and GPU test succeeded.

Raw matrix results are in this directory:

- `baseline.json`
- `opt1.json`
- `opt1_opt2.json`
- `opt1_opt2_opt3.json`
- `opt1_opt2_opt3_opt4.json`
- `opt1_opt2_opt3_opt4_opt5.json`
- `opt1_opt2_opt3_opt4_opt5_opt6.json`
- `opt1_opt2_opt3_opt4_opt5_opt6_visual_dq0.json`
- `round8_language_dq_threshold_ab.json`

Historical correctness reports and performance tables remain tied to their
original runtime versions; this sweep measures performance and does not redo
numeric-equivalence validation.
