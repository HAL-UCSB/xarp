import timeit
import csv
import os
from datetime import datetime
from itertools import combinations

import numpy as np

from xarp.express import SyncXR
from xarp.server import show_qrcode_link, run
from xarp.settings import settings

# -------------------- Grid + Benchmark config (override via params) --------------------
DEFAULT = {
    # grid:
    "grid_modes": ("stream", ),  # "grid_modes": ("stream", "single"),
    "grid_modalities_universe": ("head", "hands" ), #"grid_modalities_universe": ("head", "hands", "image"),
    "grid_min_k": 1,
    "grid_max_k": None,

    # per benchmark:
    "runs": 3,
    "n": 1000,
    "warmup": 10,

    # output:
    "csv_path": None,

    # NEW: job slicing (1-based, inclusive)
    "start_job": 1,     # run from this job index onward
    "end_job": None,    # optional: stop after this job index (inclusive)
}


# -------------------- Minimal sufficient metrics (computed from frame times) --------------------
# 1) FPS mean (total-time)
# 2) p50 frame time (median)
# 3) tail ratio = p99/p50
# 4) hitch rate = P(frame_time > 2*p50)
def minimal_metrics_from_dt(dt_s: np.ndarray) -> dict:
    ms = dt_s * 1e3
    p50 = float(np.percentile(ms, 50))
    p99 = float(np.percentile(ms, 99))
    tail = (p99 / p50) if p50 > 0 else float("inf")
    hitch2 = float(np.mean(ms > 2.0 * p50))
    return {
        "p50_ms": p50,
        "tail_p99_p50": float(tail),
        "hitch2_rate": hitch2,
    }


def ci95_mean(x: np.ndarray) -> tuple[float, float]:
    # Normal approx; fine for >= ~10 runs, dependency-free.
    m = float(np.mean(x))
    if len(x) < 2:
        return (m, 0.0)
    s = float(np.std(x, ddof=1))
    half = 1.96 * s / np.sqrt(len(x))
    return (m, half)


def _default_csv_path() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"xarp_grid_bench_min_{ts}.csv"


# -------------------- XARP runners --------------------
def _validate_single_calls(xarp: SyncXR, modalities):
    missing = [m for m in modalities if not callable(getattr(xarp, m, None))]
    if missing:
        raise AttributeError(f"SyncXR missing callable methods: {missing}")


def _single_step(xarp: SyncXR, modalities):
    for m in modalities:
        getattr(xarp, m)()


def _stream_factory(xarp: SyncXR, modalities):
    kwargs = {m: True for m in modalities}
    return xarp.sense(**kwargs)


def _run_single_once(xarp: SyncXR, modalities, n: int, warmup: int):
    _validate_single_calls(xarp, modalities)

    for _ in range(warmup):
        _single_step(xarp, modalities)

    dts = np.empty(n, dtype=np.float64)
    t0 = timeit.default_timer()
    prev = t0
    for i in range(n):
        _single_step(xarp, modalities)
        now = timeit.default_timer()
        dts[i] = now - prev
        prev = now
    t1 = timeit.default_timer()
    fps_total = n / (t1 - t0) if (t1 - t0) > 0 else float("inf")
    return dts, float(fps_total)


def _run_stream_once(xarp: SyncXR, modalities, n: int, warmup: int):
    stream = _stream_factory(xarp, modalities)
    try:
        it = iter(stream)
        for _ in range(warmup):
            next(it)

        dts = np.empty(n, dtype=np.float64)
        t0 = timeit.default_timer()
        prev = t0
        for i in range(n):
            next(it)
            now = timeit.default_timer()
            dts[i] = now - prev
            prev = now
        t1 = timeit.default_timer()
        fps_total = n / (t1 - t0) if (t1 - t0) > 0 else float("inf")
        return dts, float(fps_total)
    finally:
        try:
            stream.close()
        except Exception:
            pass


def _bench_combo_minimal(xarp: SyncXR, mode: str, modalities: tuple[str, ...], n: int, warmup: int, runs: int):
    run_fps = np.empty(runs, dtype=np.float64)
    run_p50 = np.empty(runs, dtype=np.float64)
    run_tail = np.empty(runs, dtype=np.float64)
    run_h2 = np.empty(runs, dtype=np.float64)

    for r in range(runs):
        if mode == "single":
            dts, fps = _run_single_once(xarp, modalities, n=n, warmup=warmup)
        elif mode == "stream":
            dts, fps = _run_stream_once(xarp, modalities, n=n, warmup=warmup)
        else:
            raise ValueError("mode must be 'single' or 'stream'")

        m = minimal_metrics_from_dt(dts)
        run_fps[r] = fps
        run_p50[r] = m["p50_ms"]
        run_tail[r] = m["tail_p99_p50"]
        run_h2[r] = m["hitch2_rate"]

    fps_mean, fps_ci = ci95_mean(run_fps)
    p50_mean, p50_ci = ci95_mean(run_p50)
    tail_mean, tail_ci = ci95_mean(run_tail)
    h2_mean, h2_ci = ci95_mean(run_h2)

    return {
        "fps_mean": float(fps_mean),
        "fps_ci95_half": float(fps_ci),

        "p50_ms_mean": float(p50_mean),
        "p50_ms_ci95_half": float(p50_ci),

        "tail_p99_p50_mean": float(tail_mean),
        "tail_p99_p50_ci95_half": float(tail_ci),

        "hitch2_rate_mean": float(h2_mean),
        "hitch2_rate_ci95_half": float(h2_ci),
    }





# -------------------- Grid search driver (run() entrypoint) --------------------
def grid_bench_minimal(xarp: SyncXR, params=None, n: int = DEFAULT["n"]):
    params = params or {}

    modes = tuple(params.get("grid_modes", DEFAULT["grid_modes"]))
    universe = tuple(params.get("grid_modalities_universe", DEFAULT["grid_modalities_universe"]))
    min_k = int(params.get("grid_min_k", DEFAULT["grid_min_k"]))
    max_k = params.get("grid_max_k", DEFAULT["grid_max_k"])
    max_k = int(max_k) if max_k is not None else len(universe)

    n = int(params.get("n", n))
    warmup = int(params.get("warmup", DEFAULT["warmup"]))
    runs = int(params.get("runs", DEFAULT["runs"]))

    csv_path = params.get("csv_path", DEFAULT["csv_path"]) or _default_csv_path()
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)

    # NEW: job range controls (1-based, inclusive)
    start_job = int(params.get("start_job", DEFAULT["start_job"]))
    end_job = params.get("end_job", DEFAULT["end_job"])
    end_job = int(end_job) if end_job is not None else None

    if start_job < 1:
        raise ValueError(f"start_job must be >= 1 (got {start_job})")

    # build grid
    combos = []
    for k in range(min_k, max_k + 1):
        combos.extend(combinations(universe, k))

    total_jobs = len(modes) * len(combos)

    if start_job > total_jobs:
        raise ValueError(f"start_job={start_job} exceeds total_jobs={total_jobs}")

    if end_job is not None:
        if end_job < start_job:
            raise ValueError(f"end_job must be >= start_job (got end_job={end_job}, start_job={start_job})")
        if end_job > total_jobs:
            raise ValueError(f"end_job={end_job} exceeds total_jobs={total_jobs}")

    session_id = datetime.now().isoformat(timespec="seconds")

    fieldnames = [
        "session_id",
        "timestamp_iso",
        "mode",
        "modalities",
        "k",
        "runs",
        "n_per_run",
        "warmup_per_run",
        "fps_mean",
        "fps_ci95_half",
        "p50_ms_mean",
        "p50_ms_ci95_half",
        "tail_p99_p50_mean",
        "tail_p99_p50_ci95_half",
        "hitch2_rate_mean",
        "hitch2_rate_ci95_half",
    ]

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        job = 0
        for mode in modes:
            if mode not in ("stream", "single"):
                raise ValueError(f"Invalid mode in grid_modes: {mode}")

            for modalities in combos:
                job += 1

                # NEW: skip / stop logic
                if job < start_job:
                    continue
                if end_job is not None and job > end_job:
                    break

                label = "+".join(modalities)
                print(f"[{job:03d}/{total_jobs}] {mode:6s}  {label}  (k={len(modalities)})")

                s = _bench_combo_minimal(xarp, mode, modalities, n=n, warmup=warmup, runs=runs)

                row = {
                    "session_id": session_id,
                    "timestamp_iso": datetime.now().isoformat(timespec="seconds"),
                    "mode": mode,
                    "modalities": label,
                    "k": len(modalities),
                    "runs": runs,
                    "n_per_run": n,
                    "warmup_per_run": warmup,
                    **s,
                }
                w.writerow(row)

                print(
                    f"  fps={s['fps_mean']:.2f}±{s['fps_ci95_half']:.2f}  "
                    f"p50={s['p50_ms_mean']:.3f}±{s['p50_ms_ci95_half']:.3f}ms  "
                    f"tail={s['tail_p99_p50_mean']:.2f}±{s['tail_p99_p50_ci95_half']:.2f}  "
                    f"h2={100*s['hitch2_rate_mean']:.2f}%±{100*s['hitch2_rate_ci95_half']:.2f}%"
                )

            # NEW: if end_job hit inside combos loop, break outer loop too
            if end_job is not None and job >= end_job:
                break

    print("\n" + "-" * 78)
    print(f"Grid finished. CSV written to: {csv_path}")
    print(f"Universe={universe}  modes={modes}  combos={len(combos)}  jobs={total_jobs}")
    print(f"Per job: runs={runs} n/run={n} warmup/run={warmup}")
    if start_job != 1 or end_job is not None:
        print(f"Job slice: start_job={start_job} end_job={end_job}")



if __name__ == "__main__":
    # settings.host = "127.0.0.1"
    show_qrcode_link()
    run(grid_bench_minimal)