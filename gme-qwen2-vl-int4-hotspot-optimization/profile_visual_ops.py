#!/usr/bin/env python3
"""Per-operation GPU profiling of the GME visual IR (Qwen2-VL vision tower).

The text-side counterpart is ``profile_ops.py``. This one feeds the visual IR
directly -- synthetic ``pixel_values`` plus an ``image_grid_thw`` row -- so a run
costs no PIL/processor time and every millisecond is attributable to an OpenVINO
op rather than to the harness.

The vision tower's sequence length is the *raw patch* count ``L = t*h*w``, not the
merged visual-token count ``K = L/4`` the language IR sees. Attention inside the
tower is therefore O(L^2) over a sequence 4x longer than the one the language
model gets, which is why image latency grows super-linearly with resolution.
``--grid`` takes ``t,h,w``; ``--tokens`` is the friendlier way in and picks a
square-ish grid with ``K`` merged tokens.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import time
from pathlib import Path

import numpy as np
import openvino as ov
from openvino import properties as props
from openvino.properties import hint

from paths import GME_PACKAGE

HERE = Path(__file__).resolve().parent
PACKAGE = GME_PACKAGE

# Qwen2-VL vision: 14x14 patches, temporal patch size 2, 3 channels
# -> one patch row is 3 * 2 * 14 * 14 = 1176 floats.
PATCH_ROW = 1176
MERGE = 2  # spatial_merge_size; K = L / MERGE**2

IR = {
    "int4": "gme_visual_int4.xml",
    "int8": "gme_visual_int8.xml",
    "fp32": "gme_visual_fp32.xml",
}


def _config_precisions() -> dict[str, str]:
    """Visual IRs declared in the package config, so new precisions need no edit here."""
    cfg_path = PACKAGE / "gme_openvino_config.json"
    if not cfg_path.exists():
        return {}
    cfg = json.loads(cfg_path.read_text())
    return {key: entry["visual"]
            for key, entry in (cfg.get("precisions") or {}).items()
            if entry.get("visual") and (PACKAGE / entry["visual"]).exists()}


IR = {**_config_precisions(), **IR}


def grid_for_tokens(tokens: int) -> tuple[int, int, int]:
    """Square-ish ``(t, h, w)`` whose merged token count is exactly `tokens`.

    ``h`` and ``w`` must both be multiples of ``MERGE`` for the merger to divide
    evenly, so the search walks divisors of ``4 * tokens`` in that lattice.
    """
    patches = tokens * MERGE * MERGE
    best = None
    for h in range(MERGE, int(math.isqrt(patches)) + 1, MERGE):
        if patches % h:
            continue
        w = patches // h
        if w % MERGE:
            continue
        score = abs(math.log(w / h))
        if best is None or score < best[0]:
            best = (score, h, w)
    if best is None:
        raise SystemExit(
            f"no (h, w) with h,w multiples of {MERGE} gives {tokens} merged tokens; "
            "pass --grid t,h,w explicitly"
        )
    return 1, best[1], best[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", default="int4_causal_vopt", choices=sorted(IR))
    ap.add_argument("--ir", default=None,
                    help="visual IR filename inside the package, overriding --precision. "
                         "Lets an intermediate/analysis-only IR be profiled without "
                         "registering a precision key for it.")
    ap.add_argument("--device", default="GPU")
    ap.add_argument("--tokens", type=int, default=256,
                    help="merged visual tokens K (patch rows L = 4K)")
    ap.add_argument("--grid", default=None, help="explicit t,h,w overriding --tokens")
    ap.add_argument("--warmup", type=int, default=6)
    ap.add_argument("--repeat", type=int, default=8)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--config", action="append", default=[],
                    help="extra plugin property KEY=VALUE (repeatable)")
    args = ap.parse_args()

    if args.grid:
        t, h, w = (int(v) for v in args.grid.split(","))
    else:
        t, h, w = grid_for_tokens(args.tokens)
    patches = t * h * w
    tokens = patches // (MERGE * MERGE)

    core = ov.Core()
    cfg = {hint.performance_mode: hint.PerformanceMode.LATENCY,
           props.enable_profiling: True}
    for item in args.config:
        key, _, value = item.partition("=")
        cfg[key] = value

    ir_name = args.ir or IR[args.precision]
    t0 = time.perf_counter()
    compiled = core.compile_model(PACKAGE / ir_name, args.device, cfg)
    compile_s = time.perf_counter() - t0

    # Content does not affect latency; a fixed seed keeps runs comparable and avoids
    # denormals that would.
    rng = np.random.default_rng(0)
    feed = {
        "pixel_values": rng.standard_normal((patches, PATCH_ROW), dtype=np.float32),
        "image_grid_thw": np.array([[t, h, w]], dtype=np.int64),
    }

    request = compiled.create_infer_request()
    for _ in range(args.warmup):
        request.infer(feed)

    samples = []
    by_type: dict[str, list[float]] = collections.defaultdict(list)
    by_node: dict[str, list[float]] = collections.defaultdict(list)
    type_calls: collections.Counter[str] = collections.Counter()
    for _ in range(args.repeat):
        started = time.perf_counter()
        request.infer(feed)
        samples.append((time.perf_counter() - started) * 1000)
        run_type: dict[str, float] = collections.defaultdict(float)
        run_node: dict[str, float] = collections.defaultdict(float)
        calls: collections.Counter[str] = collections.Counter()
        for info in request.profiling_info:
            if info.status != info.Status.EXECUTED:
                continue
            ms = info.real_time.total_seconds() * 1000
            run_type[info.exec_type] += ms
            run_node[f"{info.node_name}  [{info.exec_type}]"] += ms
            calls[info.exec_type] += 1
        for key, value in run_type.items():
            by_type[key].append(value)
        for key, value in run_node.items():
            by_node[key].append(value)
        type_calls.update(calls)

    arithmetic_mean = lambda values: float(np.mean(values))
    type_ms = {k: arithmetic_mean(v) for k, v in by_type.items()}
    node_ms = {k: arithmetic_mean(v) for k, v in by_node.items()}
    profiled = sum(type_ms.values())
    wall_mean = float(np.mean(samples))
    wall_p50 = float(np.percentile(samples, 50))

    print(f"ir={ir_name} device={args.device} grid={t},{h},{w} "
          f"patches(L)={patches} tokens(K)={tokens} compile={compile_s:.1f}s "
            f"wall_mean={wall_mean:.1f}ms "
          f"profiled_mean_sum={profiled:.1f}ms host_gap_mean={wall_mean - profiled:.1f}ms")

    print(f"\n{'ms/infer':>9} {'%':>6} {'calls':>7}  exec_type")
    for key, value in sorted(type_ms.items(), key=lambda kv: -kv[1]):
        calls = type_calls[key] / args.repeat
        print(f"{value:9.2f} {100 * value / profiled:6.1f} {calls:7.0f}  {key}")

    print(f"\n{'ms/infer':>9} {'%':>6}  node (top {args.top})")
    for key, value in sorted(node_ms.items(), key=lambda kv: -kv[1])[: args.top]:
        print(f"{value:9.2f} {100 * value / profiled:6.1f}  {key}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "precision": args.precision,
            "device": args.device,
            "ir": ir_name,
            "grid": [t, h, w],
            "patches": patches,
            "tokens": tokens,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "compile_s": round(compile_s, 3),
            "wall_mean_ms": round(wall_mean, 3),
            "wall_p50_ms": round(wall_p50, 3),
            "profiled_sum_ms": round(profiled, 3),
            "host_gap_ms": round(wall_mean - profiled, 3),
            "samples_ms": [round(s, 3) for s in samples],
            "by_exec_type": {k: round(v, 4) for k, v in sorted(type_ms.items(), key=lambda kv: -kv[1])},
            "calls_by_exec_type": {k: round(v / args.repeat, 2) for k, v in type_calls.items()},
            "by_node": {k: round(v, 4) for k, v in sorted(node_ms.items(), key=lambda kv: -kv[1])},
        }, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
