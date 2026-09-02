"""Liveness guards for the long encode phases.

A wedged OpenVINO call cannot be cancelled from Python. When the GPU plugin
spins inside the driver, the calling thread parks in ``InferRequest::wait()``
with the GIL released, so nothing short of killing the process recovers -- an
int4 run once burned four and a half days that way, indistinguishable from a
run that was merely slow. These helpers therefore do not attempt recovery.
They make a stalled phase *look* stalled: a heartbeat while it advances, and a
bounded, loud abort when it stops.
"""
from __future__ import annotations

import faulthandler
import os
import sys
import threading
import time
from contextlib import contextmanager
from typing import Iterator

# Distinct from any exit code the interpreter itself produces, so a suite script
# can tell "this run wedged" apart from "this run raised".
HANG_EXIT_CODE = 86


class Watchdog:
    """Abort the process when a single guarded call stops making progress.

    The deadline is armed per call rather than per phase: a phase legitimately
    runs for hours, while any one model call that exceeds a few minutes is
    already pathological.
    """

    def __init__(self, timeout_seconds: float, poll_seconds: float = 5.0) -> None:
        self.timeout_seconds = timeout_seconds
        self.poll_seconds = poll_seconds
        self._lock = threading.Lock()
        self._deadline: float | None = None
        self._label = ""
        self._thread: threading.Thread | None = None

    @property
    def enabled(self) -> bool:
        return self.timeout_seconds > 0

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._watch, name="encode-watchdog", daemon=True)
        self._thread.start()

    @contextmanager
    def guard(self, label: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        self.start()
        with self._lock:
            self._deadline = time.monotonic() + self.timeout_seconds
            self._label = label
        try:
            yield
        finally:
            with self._lock:
                self._deadline = None
                self._label = ""

    def _watch(self) -> None:
        while True:
            time.sleep(self.poll_seconds)
            with self._lock:
                deadline, label = self._deadline, self._label
            if deadline is not None and time.monotonic() >= deadline:
                self._abort(label)

    def _abort(self, label: str) -> None:
        sys.stdout.flush()
        print(
            f"\nWATCHDOG: {label} made no progress for {self.timeout_seconds:.0f}s; aborting.\n"
            "A wedged OpenVINO call cannot be cancelled from Python, so the process is being\n"
            "killed rather than left to spin. Python stacks for every thread follow, but the\n"
            "stuck work itself lives in a native plugin thread -- inspect that with:\n"
            "    gdb -p <pid> -batch -ex 'thread apply all bt'",
            file=sys.stderr,
            flush=True,
        )
        faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
        sys.stderr.flush()
        # _exit, not sys.exit: the main thread is blocked in C and would never
        # observe an exception raised here.
        os._exit(HANG_EXIT_CODE)


def format_duration(seconds: float) -> str:
    if seconds != seconds or seconds in (float("inf"), float("-inf")):  # NaN/inf
        return "unknown"
    seconds = int(max(0.0, seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, second = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{second:02d}s"
    return f"{second}s"


class ProgressReporter:
    """Emit a periodic heartbeat so a phase's silence is bounded."""

    def __init__(self, label: str, total: int, interval_seconds: float = 30.0) -> None:
        self.label = label
        self.total = total
        self.interval_seconds = interval_seconds
        self.started = time.monotonic()
        self._last_emit = self.started
        self._emitted = False

    @property
    def enabled(self) -> bool:
        return self.interval_seconds > 0 and self.total > 0

    def update(self, done: int) -> None:
        if not self.enabled or done <= 0:
            return
        now = time.monotonic()
        if now - self._last_emit < self.interval_seconds:
            return
        self._emit(done, now)

    def close(self, done: int) -> None:
        """Print a closing line only for phases that already printed a heartbeat."""
        if self._emitted:
            self._emit(done, time.monotonic())

    def _emit(self, done: int, now: float) -> None:
        elapsed = now - self.started
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = (self.total - done) / rate if rate > 0 else float("inf")
        percent = 100.0 * done / self.total if self.total else 0.0
        print(
            f"{self.label} {done}/{self.total} ({percent:.1f}%) "
            f"elapsed {format_duration(elapsed)} eta {format_duration(eta)}",
            flush=True,
        )
        self._last_emit = now
        self._emitted = True
