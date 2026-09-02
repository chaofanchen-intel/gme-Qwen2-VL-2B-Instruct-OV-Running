import io
import unittest
from contextlib import redirect_stdout

from mmdocir_eval.progress import ProgressReporter, Watchdog, format_duration


class WatchdogTest(unittest.TestCase):
    def test_zero_timeout_never_arms(self) -> None:
        watchdog = Watchdog(0)
        self.assertFalse(watchdog.enabled)
        with watchdog.guard("call"):
            pass
        # No thread is started, so a disabled watchdog cannot abort anything.
        self.assertIsNone(watchdog._thread)

    def test_guard_disarms_after_the_call_returns(self) -> None:
        watchdog = Watchdog(600)
        with watchdog.guard("page batch at position 0"):
            self.assertIsNotNone(watchdog._deadline)
            self.assertEqual(watchdog._label, "page batch at position 0")
        self.assertIsNone(watchdog._deadline)

    def test_guard_disarms_when_the_call_raises(self) -> None:
        watchdog = Watchdog(600)
        with self.assertRaises(ValueError):
            with watchdog.guard("call"):
                raise ValueError("boom")
        self.assertIsNone(watchdog._deadline)


class ProgressReporterTest(unittest.TestCase):
    def test_interval_throttles_heartbeats(self) -> None:
        reporter = ProgressReporter("[int4]", 100, interval_seconds=3600)
        output = io.StringIO()
        with redirect_stdout(output):
            reporter.update(10)
            reporter.close(10)
        self.assertEqual(output.getvalue(), "")

    def test_emits_counts_and_eta(self) -> None:
        reporter = ProgressReporter("[int4] layouts", 100, interval_seconds=60)
        reporter.started -= 120  # pretend two minutes of work have already elapsed
        reporter._last_emit = reporter.started
        output = io.StringIO()
        with redirect_stdout(output):
            reporter.update(25)
        line = output.getvalue()
        self.assertIn("[int4] layouts 25/100 (25.0%)", line)
        self.assertIn("eta", line)

    def test_disabled_for_empty_phases(self) -> None:
        self.assertFalse(ProgressReporter("[fp32]", 0, interval_seconds=1).enabled)


class FormatDurationTest(unittest.TestCase):
    def test_scales_units(self) -> None:
        self.assertEqual(format_duration(9), "9s")
        self.assertEqual(format_duration(125), "2m05s")
        self.assertEqual(format_duration(7325), "2h02m")

    def test_unknown_for_infinite_eta(self) -> None:
        self.assertEqual(format_duration(float("inf")), "unknown")


if __name__ == "__main__":
    unittest.main()
