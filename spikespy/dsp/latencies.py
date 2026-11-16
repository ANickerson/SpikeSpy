from __future__ import annotations

from typing import Dict, Tuple
import numpy as np


def compute_latency_histogram(
    latencies_ms: np.ndarray,
    bins: int = 30,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute normalized latency histogram and bin edges."""
    arr = np.asarray(latencies_ms, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    counts, edges = np.histogram(arr, bins=max(1, int(bins)))
    max_c = int(np.max(counts)) if counts.size > 0 else 0
    if max_c > 0:
        norm = counts.astype(np.float64) / float(max_c)
    else:
        norm = counts.astype(np.float64)
    return norm, edges.astype(np.float64)


def compute_isi_and_firing_rate(
    event_times_s: np.ndarray,
    idx_arr: np.ndarray,
    total_stims: int,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Compute Instantaneous ISI, Firing Rate (Hz), and Cumulative Spike Count per sweep."""
    ev_mag = np.asarray(event_times_s, dtype=np.float64).ravel()
    idx = np.asarray(idx_arr, dtype=np.int64).ravel()
    y_stims = np.arange(int(total_stims), dtype=np.float64)

    isi = np.full(int(total_stims), np.nan, dtype=np.float64)
    rate = np.full(int(total_stims), np.nan, dtype=np.float64)
    cum = np.zeros(int(total_stims), dtype=np.float64)

    prev_t = np.nan
    running = 0
    n_idx = min(int(total_stims), idx.size)
    for s in range(n_idx):
        ev_i = int(idx[s])
        if 0 <= ev_i < ev_mag.size:
            t = float(ev_mag[ev_i])
            running += 1
            if not np.isnan(prev_t) and t > prev_t:
                dt = t - prev_t
                isi[s] = dt
                rate[s] = 1.0 / dt
            prev_t = t
        cum[s] = float(running)
    if n_idx < int(total_stims):
        cum[n_idx:] = float(running)

    return {
        "Instantaneous ISI": (isi, y_stims),
        "Firing Rate (Hz)": (rate, y_stims),
        "Cumulative Spike Count": (cum, y_stims),
    }


__all__ = [
    "compute_latency_histogram",
    "compute_isi_and_firing_rate",
]
