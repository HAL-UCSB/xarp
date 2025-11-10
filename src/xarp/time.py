from datetime import datetime, timezone


def utc_ts():
    return int(datetime.now(timezone.utc).timestamp())