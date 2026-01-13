import time


def utc_ts() -> int:
    return time.time_ns() // 1_000_000
