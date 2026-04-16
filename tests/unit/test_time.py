import unittest
from unittest.mock import patch

import time
from xarp.time import utc_ts, _MONO_TO_UTC_MS


class TestUtcTs(unittest.TestCase):

    def test_close_to_wall_clock(self):
        """Must be within 1 s of time.time_ns()-based UTC ms."""
        delta = abs(utc_ts() - time.time_ns() // 1_000_000)
        self.assertLess(delta, 1_000, f"Drift too large: {delta} ms")

    def test_result_is_in_milliseconds_not_seconds(self):
        """Value should be ~1.7 trillion (ms), not ~1.7 billion (s) or higher (µs)."""
        ts = utc_ts()
        self.assertGreater(ts, 1_000_000_000_000)  # > 1 trillion  → ms
        self.assertLess(ts, 100_000_000_000_000)  # < 100 trillion → not µs

    def test_monotonically_non_decreasing(self):
        """Rapid successive calls must never go backwards."""
        samples = [utc_ts() for _ in range(200)]
        for a, b in zip(samples, samples[1:]):
            self.assertLessEqual(a, b, "Timestamp went backwards")

    def test_elapsed_matches_sleep(self):
        """Elapsed ms over a known sleep must match within 20 ms."""
        t1 = utc_ts()
        time.sleep(0.1)
        t2 = utc_ts()
        self.assertAlmostEqual(t2 - t1, 100, delta=20)

    def test_arithmetic(self):
        """Result must equal monotonic_ns // 1e6 + _MONO_TO_UTC_MS."""
        fake_mono_ns = 1_234_567_890_000
        with patch("time.monotonic_ns", return_value=fake_mono_ns):
            result = utc_ts()
        self.assertEqual(result, fake_mono_ns // 1_000_000 + _MONO_TO_UTC_MS)

    def test_does_not_call_time_time_ns_at_call_site(self):
        """time.time_ns() must only run at module load, not on every call."""
        with patch("time.time_ns") as mock_time_ns:
            utc_ts()
            mock_time_ns.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
