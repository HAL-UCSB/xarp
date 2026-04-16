import time

# Compute offset once at startup: UTC epoch of the monotonic clock's zero point
_MONO_TO_UTC_MS = (time.time_ns() - time.monotonic_ns()) // 1_000_000

def utc_ts() -> int:
    """Monotonic UTC timestamp in milliseconds."""
    return time.monotonic_ns() // 1_000_000 + _MONO_TO_UTC_MS