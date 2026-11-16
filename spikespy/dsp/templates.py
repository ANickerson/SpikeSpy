from __future__ import annotations

from typing import Dict, Optional, Tuple
import numpy as np
from scipy.signal import fftconvolve, find_peaks

from spikespy.processing import MultiChannelTemplateMatcher


def compute_single_channel_template_conv(
    trace: np.ndarray,
    template: np.ndarray,
    peak_height_sigma: float = 2.0,
    min_distance: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute normalized FFT template convolution and peak indices for a 1D trace."""
    tr = np.asarray(trace, dtype=np.float64).ravel()
    tmpl = np.asarray(template, dtype=np.float64).ravel()
    if tr.size == 0 or tmpl.size == 0:
        empty_f = np.empty(0, dtype=np.float64)
        empty_i = np.empty(0, dtype=np.int64)
        return empty_f, empty_i, empty_f

    tmpl_centered = tmpl - np.mean(tmpl)
    norm = float(np.linalg.norm(tmpl_centered))
    if norm > 0:
        tmpl_norm = tmpl_centered / norm
    else:
        tmpl_norm = tmpl_centered

    conv = fftconvolve(tr - np.mean(tr), tmpl_norm[::-1], mode="same")
    std_conv = float(np.std(conv))
    conv_norm = conv / (std_conv if std_conv > 0 else 1.0)
    peaks, _ = find_peaks(conv_norm, height=peak_height_sigma, distance=max(1, int(min_distance)))
    return conv_norm, peaks.astype(np.int64), conv_norm[peaks].astype(np.float64)


def compute_multi_channel_sweep_conv(
    channel_traces: Dict[int, np.ndarray],
    templates: Dict[int, np.ndarray],
    weights: Optional[Dict[int, float]] = None,
    polarity: str = "positive",
    min_distance: int = 10,
    prominence_std: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute composite multi-channel template convolution score and extract candidate peaks."""
    if not channel_traces or not templates:
        empty_f = np.empty(0, dtype=np.float64)
        empty_i = np.empty(0, dtype=np.int64)
        return empty_f, empty_i, empty_f

    composite: Optional[np.ndarray] = None
    total_w = 0.0
    for ch, tmpl in templates.items():
        if ch not in channel_traces:
            continue
        raw = np.asarray(channel_traces[ch], dtype=np.float64).ravel()
        t = np.asarray(tmpl, dtype=np.float64).ravel()
        if raw.size == 0 or t.size == 0:
            continue
        w = float(weights.get(ch, 1.0)) if weights is not None else 1.0
        t_c = t - np.mean(t)
        t_norm = float(np.linalg.norm(t_c))
        if t_norm > 0:
            t_c = t_c / t_norm
        c = fftconvolve(raw - np.mean(raw), t_c[::-1], mode="same")
        if composite is None:
            composite = w * c
        else:
            n = min(composite.size, c.size)
            composite = composite[:n] + w * c[:n]
        total_w += w

    if composite is None or composite.size == 0:
        empty_f = np.empty(0, dtype=np.float64)
        empty_i = np.empty(0, dtype=np.int64)
        return empty_f, empty_i, empty_f

    if total_w > 0:
        composite = composite / total_w

    search_sig = -composite if polarity == "negative" else composite
    prom = max(float(np.std(search_sig)) * float(prominence_std), 1e-12)
    peaks, _ = find_peaks(search_sig, distance=max(1, int(min_distance)), prominence=prom)
    return composite, peaks.astype(np.int64), composite[peaks].astype(np.float64)


__all__ = [
    "MultiChannelTemplateMatcher",
    "compute_single_channel_template_conv",
    "compute_multi_channel_sweep_conv",
]
