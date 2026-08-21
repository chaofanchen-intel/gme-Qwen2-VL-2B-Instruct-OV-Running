#!/usr/bin/env python3
"""Per-operation GPU profiling of the GME language IR.

Feeds the language IR directly (no tokenizer/processor) so a run is cheap and the
numbers are attributable to OpenVINO ops rather than to the harness. Reports the
plugin's own profiling counters grouped by execution type (the GPU kernel that
actually ran) and by original op name.
"""
from __future__ import annotations

import argparse
import collections
import json
import time
from pathlib import Path

import numpy as np
import openvino as ov
from openvino import properties as props
from openvino.properties import hint

from paths import GME_PACKAGE

HERE = Path(__file__).resolve().parent
PACKAGE = GME_PACKAGE
IR = {
    "int4": "gme_language_int4.xml",
    "int4_gs64": "gme_language_int4_gs64.xml",
    "int4_per_oc": "gme_language_int4_per_oc.xml",
    "int4_per_oc_sym": "gme_language_int4_per_oc_sym.xml",
    "int8": "gme_language_int8.xml",
    "fp32": "gme_language_fp32.xml",
}


def _config_precisions() -> dict[str, str]:
    """Language IRs declared in the package config, so new precisions need no edit here."""
    cfg_path = PACKAGE / "gme_openvino_config.json"
    if not cfg_path.exists():
        return {}
    cfg = __import__("json").loads(cfg_path.read_text())
    return {key: entry["language"]
            for key, entry in (cfg.get("precisions") or {}).items()
            if (PACKAGE / entry["language"]).exists()}


# Config-declared precisions extend the table above; the hard-coded entries win on conflict
# so the historical keys keep their exact meaning.
IR = {**_config_precisions(), **IR}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", default="int4", choices=sorted(IR))
    ap.add_argument("--device", default="GPU")
    ap.add_argument("--length", type=int, default=1024)
    ap.add_argument("--warmup", type=int, default=14)
    ap.add_argument("--repeat", type=int, default=10)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--config", action="append", default=[],
                    help="extra plugin property KEY=VALUE (repeatable)")
    args = ap.parse_args()

    core = ov.Core()
    cfg = {hint.performance_mode: hint.PerformanceMode.LATENCY,
           props.enable_profiling: True}
    for item in args.config:
        key, _, value = item.partition("=")
        cfg[key] = value

    t0 = time.perf_counter()
    compiled = core.compile_model(PACKAGE / IR[args.precision], args.device, cfg)
    compile_s = time.perf_counter() - t0

    seq = args.length
    feed = {
        "input_ids": np.full((1, seq), 9707, dtype=np.int64),
        "attention_mask": np.ones((1, seq), dtype=np.int64),
        "image_embeds": np.empty((0, 1536), dtype=np.float32),
    }
    request = compiled.create_infer_request()
    for _ in range(args.warmup):
        request.infer(feed)

    wall = []
    by_type: dict[str, list] = collections.defaultdict(lambda: [0.0, 0])
    by_name: dict[str, list] = collections.defaultdict(lambda: [0.0, 0, ""])
    for _ in range(args.repeat):
        started = time.perf_counter()
        request.infer(feed)
        wall.append((time.perf_counter() - started) * 1000)
        for info in request.profiling_info:
            if info.status == info.status.NOT_RUN:
                continue
            ms = info.real_time.total_seconds() * 1000
            by_type[info.exec_type][0] += ms
            by_type[info.exec_type][1] += 1
            by_name[info.node_name][0] += ms
            by_name[info.node_name][1] += 1
            by_name[info.node_name][2] = info.exec_type

    n = args.repeat
    types = sorted(((v[0] / n, v[1] / n, k) for k, v in by_type.items()), reverse=True)
    names = sorted(((v[0] / n, v[1] / n, k, v[2]) for k, v in by_name.items()), reverse=True)
    total = sum(t[0] for t in types)

    print(f"precision={args.precision} device={args.device} seq={seq} "
          f"compile={compile_s:.1f}s wall_mean={np.mean(wall):.1f}ms "
          f"wall_p50={np.percentile(wall, 50):.1f}ms profiled_sum={total:.1f}ms")
    print(f"\n{'ms/infer':>9} {'%':>6} {'calls':>7}  exec_type")
    for ms, calls, name in types[: args.top]:
        print(f"{ms:9.2f} {100 * ms / total:6.1f} {calls:7.0f}  {name}")
    print(f"\n{'ms/infer':>9} {'%':>6} {'calls':>7}  node (top {args.top})")
    for ms, calls, name, etype in names[: args.top]:
        print(f"{ms:9.2f} {100 * ms / total:6.1f} {calls:7.0f}  {name}  [{etype}]")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "precision": args.precision, "device": args.device, "length": seq,
            "config": args.config,
            "wall_mean_ms": round(float(np.mean(wall)), 3),
            "wall_p50_ms": round(float(np.percentile(wall, 50)), 3),
            "profiled_sum_ms": round(total, 3),
            "by_exec_type": [{"exec_type": k, "ms": round(m, 3), "calls": c}
                             for m, c, k in types],
            "by_node": [{"node": k, "exec_type": e, "ms": round(m, 3), "calls": c}
                        for m, c, k, e in names],
        }, indent=2) + "\n", encoding="utf-8")
        print(f"\n{args.output}")


if __name__ == "__main__":
    main()
