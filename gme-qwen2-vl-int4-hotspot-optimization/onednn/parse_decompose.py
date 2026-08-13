#!/usr/bin/env python3
"""Turn a bench_gemm_decompose.sh log into a readable table.

benchdnn's `perf,...` line is comma separated but the problem-descriptor field is a
whole command line, so positional indexing from the left is fragile.  The numeric
columns are the last five, so they are taken from the right instead.
"""
from __future__ import annotations

import argparse
import collections
from pathlib import Path

# gate/up and down are each executed twice per layer over 28 layers -- see
# docs/02_hotspot_analysis.md for how the model's GEMM time is distributed.
GFLOP = {"gate_up": 28.1857, "down": 28.1857}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", type=Path, nargs="?",
                    default=Path(__file__).resolve().parent / "results" / "gemm_decompose.log")
    args = ap.parse_args()

    rows: dict[str, dict[str, tuple[float, float]]] = collections.defaultdict(dict)
    order: list[str] = []
    label = shape = None
    for line in args.log.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("### "):
            parts = line.split()
            label, shape = parts[1], parts[2].lstrip("[")
            if label not in order:
                order.append(label)
        elif line.startswith("perf,gpu,") and label:
            f = line.split(",")
            # ...,Gops,max_ctime,min_time,min_Gflops,avg_time,avg_Gflops
            try:
                min_ms, min_gf = float(f[-4]), float(f[-3])
            except ValueError:
                continue
            rows[label][shape] = (min_ms, min_gf)
        elif "(FAILED)" in line and label:
            rows[label][shape] = (float("nan"), float("nan"))

    shapes = sorted({s for v in rows.values() for s in v})
    print(f"{'case':<26}" + "".join(f"{s:>26}" for s in shapes))
    print("-" * (26 + 26 * len(shapes)))
    base = rows.get("A_gme_actual", {})
    for label in order:
        cells = ""
        for s in shapes:
            if s not in rows.get(label, {}):
                cells += f"{'-':>26}"
                continue
            ms, gf = rows[label][s]
            ratio = base[s][0] / ms if s in base and ms == ms and ms > 0 else float("nan")
            cells += f"{ms:>10.3f}ms{gf / 1000:>7.1f}T{ratio:>7.2f}x"
        print(f"{label:<26}{cells}")
    print("\n(ratio is speedup vs A_gme_actual for the same shape)")


if __name__ == "__main__":
    main()
