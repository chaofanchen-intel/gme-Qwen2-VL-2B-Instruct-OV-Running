#!/usr/bin/env python
"""Interleaved GPU 1024-token retest: cpp/python sessions alternate in strict time
order, each session a fresh model compile + warmup + repeat, forcing both backends
into the same GPU power/thermal regime before timing so the comparison is apples
to apples.

Motivated by PERFORMANCE_REPORT.md 6.2: a single-run matrix showed C++ int8/int4
at GPU 1024 tok roughly 2x slower than Python, while fp32 and every other length
matched closely. The first pass at this (see git history / the JSON this file
used to write) assumed a cold-GPU-clock-ramp story and "fixed" it by adding a
single extra warmup call -- that made things *worse* and inconsistent. Bisecting
warmup depth (1, 3, 6, 8, 10, 12, 16) on the same config showed the latency is not
noisy but bimodal in a way that tracks call count, not warmup depth per se: a
fresh compiled model runs its first ~3 calls at a fast "boost" clock (~288ms for
int8 @ 1024 tok text), then deterministically drops to a slower steady-state
("throttled") clock (~675ms) from the 4th call onward and stays there. This
reproduces identically in plain Python with no C++ involved at all -- 20
back-to-back get_text_embeddings() calls on one encoder instance land at ~288ms
for calls 1-3 and ~675ms for every call after -- so it is a GPU/SoC power
behavior, not a backend implementation difference. The prior "process lifecycle"
theory (C++ = fresh subprocess each round vs Python = one long-lived process) was
a red herring: it correlated with which regime a given session happened to land
in, but the cause is call count within a warm model, not process identity.

Because the boost window is short and a real deployment will almost never sustain
work exclusively inside it, the reproducible fixed point to benchmark is the
steady-state regime -- so WARMUP below is deliberately large enough (per
precision) to run past the transition before timing starts, for both backends.

        ./env-ref/bin/python 09_retest_gpu_1024.py \
            --openvino-install /path/to/openvino/install
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import statistics
import subprocess
import time
from pathlib import Path

from gme_openvino import GmeOpenVINO

HERE = Path(__file__).resolve().parent
DEFAULT_PACKAGE = HERE / "ov_models" / "gme-qwen2-vl-2b-instruct-openvino"
DEFAULT_ARTIFACTS = HERE / "artifacts"
DEFAULT_WORKLOAD_SOURCE = DEFAULT_ARTIFACTS / "benchmark_matrix_text_lengths_r20.json"
DEFAULT_CPP_BINARY = HERE / "cpp" / "build" / "gme_embed"
DEFAULT_OPENVINO_INSTALL = HERE.parent / "benchmark" / "openvino" / "install"

INSTRUCTION = "Find an image that matches the given text."

# Calls needed to run past the boost->steady-state transition. Measured by hand:
# inside one continuous process, int8/int4 flip after the 3rd call and never flip
# back across 20 further calls. But each FRESH PROCESS launch samples whatever
# ambient GPU power state happens to exist at that instant -- which is not a pure
# function of this process's own warmup count -- so no fixed warmup count forces
# steady state 100% of the time across independent launches: warmup=10 held
# steady state in 6/6 int8 launches but only 5/6 int4 launches; warmup=14 held
# 7/8 int4 launches. This is as far as warmup alone gets; the remaining residual
# is left to the multi-round median/p10/p90 in the summary rather than assumed
# away. fp32 calls are already ~1.7s, long enough that the transition happens
# within a single call.
STEADY_WARMUP = {"fp32": 3, "int8": 12, "int4": 14}

# (precision, session count, repeat per session, batch_mode override or None)
PLAN = [
    ("fp32", 8, 2, None),
    ("int8", 15, 5, None),
    ("int4", 15, 5, None),
    ("int8", 6, 5, "per_sample"),  # diagnostic: does forcing per_sample on GPU change the picture?
]


def load_workload(workload_source: Path) -> tuple[str, str]:
    data = json.loads(workload_source.read_text())
    entry = data["workloads"]["texts"]["1024"]
    return entry["text"]["raw_text"], entry["query"]["raw_text"]


def timed(call, repeat: int, warmup: int) -> float:
    for _ in range(warmup):  # run past the boost->steady-state transition before timing
        call()
    start = time.perf_counter()
    for _ in range(repeat):
        call()
    return (time.perf_counter() - start) / repeat * 1000


def cpp_session(
    package: Path,
    binary: Path,
    setup: Path,
    device: str,
    instruction: str,
    text: str,
    precision: str,
    repeat: int,
    warmup: int,
    batch_mode: str | None,
) -> dict[str, float]:
    """One fresh subprocess: compiles the language+visual IR, then benches text and
    query modes back to back. Both modes reuse the same (text-calibrated) prompt
    string -- main.cpp's --benchmark-modes always feeds texts[0] to every mode -- so
    the "query" number here uses a ~1024-tok prompt under the query instruction
    rather than the separately-calibrated query prompt the main matrix uses. That is
    fine for this diagnostic (it measures session-to-session variance, not an exact
    1024-token reproduction), but it means this artifact's cpp "query" figure is not
    directly comparable token-for-token to PERFORMANCE_REPORT.md's table.
    """
    args = ["--device", device, "--precision", precision, "--benchmark",
            "--benchmark-modes", "text,query", "--repeat", str(repeat),
            "--warmup", str(warmup), "--instruction", instruction]
    if batch_mode:
        args += ["--batch-mode", batch_mode]
    args.append(text)
    command = [str(binary), "--package", str(package), *args]
    shell = (
        f"set +u; source {shlex.quote(str(setup))} >/dev/null 2>&1; "
        f"exec {shlex.join(command)}"
    )
    result = subprocess.run(["bash", "-c", shell], capture_output=True, text=True, timeout=600)
    rows: dict[str, float] = {}
    for match in re.finditer(
        r"mode=(\w+) batch=(\d+) streams=(\S+) batch_mode=(\S+) avg_ms=([\d.eE+-]+) ms_per_embedding=([\d.eE+-]+)",
        result.stdout,
    ):
        rows[match.group(1)] = float(match.group(5))
    if "text" not in rows or "query" not in rows:
        raise RuntimeError(f"cpp session produced no timing: stderr={result.stderr[-500:]}")
    return rows


def python_session(
    package: Path,
    device: str,
    instruction: str,
    text: str,
    query: str,
    precision: str,
    repeat: int,
    warmup: int,
    batch_mode: str | None,
) -> dict[str, float]:
    encoder = GmeOpenVINO(package, device=device, precision=precision, batch_mode=batch_mode or "auto")
    return {
        "text": timed(lambda: encoder.get_text_embeddings(
            [text], is_query=False, show_progress_bar=False), repeat, warmup),
        "query": timed(lambda: encoder.get_text_embeddings(
            [query], instruction=instruction, show_progress_bar=False), repeat, warmup),
    }


def summarize(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    n = len(ordered)

    def pct(p: float) -> float:
        idx = min(n - 1, max(0, round(p * (n - 1))))
        return ordered[idx]

    return {
        "n": n,
        "median": round(statistics.median(ordered), 2),
        "p10": round(pct(0.10), 2),
        "p90": round(pct(0.90), 2),
        "min": round(ordered[0], 2),
        "max": round(ordered[-1], 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE,
                        help="exported OpenVINO package directory")
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS,
                        help="directory for the default workload and output")
    parser.add_argument("--workload-source", type=Path, default=DEFAULT_WORKLOAD_SOURCE,
                        help="benchmark-matrix JSON containing the 1024-token workload")
    parser.add_argument("--cpp-binary", type=Path, default=DEFAULT_CPP_BINARY,
                        help="gme_embed executable")
    parser.add_argument("--openvino-install", type=Path, default=DEFAULT_OPENVINO_INSTALL,
                        help="OpenVINO installation containing setupvars.sh")
    parser.add_argument("--device", default="GPU", help="device to retest (default: GPU)")
    parser.add_argument("--instruction", default=INSTRUCTION, help="query instruction")
    parser.add_argument("--output", type=Path,
                        help="report path (default: <artifacts>/benchmark_gpu_1024_interleaved_retest.json)")
    args = parser.parse_args()

    package = args.package.expanduser().resolve()
    artifacts = args.artifacts.expanduser().resolve()
    workload_source = args.workload_source.expanduser().resolve()
    binary = args.cpp_binary.expanduser().resolve()
    setup = args.openvino_install.expanduser().resolve() / "setupvars.sh"
    if not package.is_dir():
        raise SystemExit(f"package directory not found: {package}")
    if not workload_source.is_file():
        raise SystemExit(f"workload source not found: {workload_source}")
    if not binary.is_file():
        raise SystemExit(f"C++ binary not found: {binary}")
    if not setup.is_file():
        raise SystemExit(f"OpenVINO setupvars.sh not found: {setup}")
    artifacts.mkdir(parents=True, exist_ok=True)

    text, query = load_workload(workload_source)
    sessions: list[dict] = []
    session_id = 0
    total_plan_sessions = sum(count for _, count, _, _ in PLAN)
    started = time.perf_counter()

    for precision, count, repeat, batch_mode in PLAN:
        label = f"{precision}" + (f"+{batch_mode}" if batch_mode else "")
        warmup = STEADY_WARMUP[precision]
        for round_idx in range(count):
            for backend in ("cpp", "python"):
                session_id += 1
                t0 = time.perf_counter()
                if backend == "cpp":
                    ms = cpp_session(
                        package, binary, setup, args.device, args.instruction,
                        text, precision, repeat, warmup, batch_mode,
                    )
                else:
                    ms = python_session(
                        package, args.device, args.instruction,
                        text, query, precision, repeat, warmup, batch_mode,
                    )
                elapsed = time.perf_counter() - t0
                sessions.append({
                    "session_id": session_id,
                    "plan_label": label,
                    "precision": precision,
                    "batch_mode": batch_mode or "auto",
                    "backend": backend,
                    "round": round_idx,
                    "repeat": repeat,
                    "ms_per_call": ms,
                    "session_wall_seconds": round(elapsed, 2),
                })
                print(f"[{session_id}/{total_plan_sessions * 2}] {label} {backend} round={round_idx} "
                      f"text={ms['text']:.1f}ms query={ms['query']:.1f}ms wall={elapsed:.1f}s "
                      f"total_elapsed={time.perf_counter() - started:.0f}s", flush=True)

    summary: dict[str, dict] = {}
    for precision, count, repeat, batch_mode in PLAN:
        label = f"{precision}" + (f"+{batch_mode}" if batch_mode else "")
        for backend in ("cpp", "python"):
            rows = [s for s in sessions if s["plan_label"] == label and s["backend"] == backend]
            for mode in ("text", "query"):
                summary[f"{label}/{backend}/{mode}"] = summarize([row["ms_per_call"][mode] for row in rows])

    output = (args.output or artifacts / "benchmark_gpu_1024_interleaved_retest.json").expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "device": args.device,
        "sequence_tokens": 1024,
        "workload_source": str(workload_source),
        "note": "Both backends are warmed up past the boost->steady-state GPU clock "
                "transition (see module docstring) before timing starts, using "
                "STEADY_WARMUP calls per precision -- this is what makes the two "
                "backends comparable; a single warmup call (the previous methodology) "
                "left the measured regime up to chance. cpp sessions reuse the "
                "text-calibrated 1024-tok prompt for both text and query benches "
                "(main.cpp feeds one positional TEXT to every --benchmark-modes entry); "
                "python sessions use the separately-calibrated text/query prompts.",
        "steady_warmup": STEADY_WARMUP,
        "plan": [{"precision": p, "sessions": c, "repeat": r, "warmup": STEADY_WARMUP[p], "batch_mode": bm or "auto"}
                 for p, c, r, bm in PLAN],
        "sessions": sessions,
        "summary": summary,
    }, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
