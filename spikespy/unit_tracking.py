"""Shared spike-tracking logic for live tracking and offline gap filling.

This module deliberately has no Qt dependencies so the same code paths can run
on the GUI thread (ad-hoc tracking) and on the live orchestrator workers.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import quantities as pq

from .dsp.detectors import SpikeDetectorRegistry


def tracking_params(
    sg: Any,
    overrides: Optional[Dict[str, Any]] = None,
    waveform_units: Any = None,
) -> Tuple[str, Dict[str, Any]]:
    """Return ``(algorithm, merged_params)`` for a unit.

    ``threshold`` is stored on units in mV.  When the waveform is a bare numpy
    array whose physical units are known, convert it so threshold detection is
    correct for live data in uV as well as offline data in mV.
    """
    algorithm = str(getattr(sg, "tracking_algorithm", "basic") or "basic")
    detector = SpikeDetectorRegistry.get(algorithm)
    params: Dict[str, Any] = dict(detector.default_params())
    params.update(
        {
            "window_ms": float(getattr(sg, "tracking_window_ms", 5.0)),
            "threshold": float(getattr(sg, "tracking_threshold_mv", 0.01)),
            "max_skip": int(getattr(sg, "tracking_max_skips", 1)),
            "track_limit": int(
                getattr(sg, "track_limit", getattr(sg, "tracking_track_limit", 0))
            ),
        }
    )
    params.update(dict(getattr(sg, "tracking_params", {}) or {}))
    if overrides:
        params.update(overrides)

    if waveform_units is not None:
        try:
            params["threshold"] = float(
                (float(params["threshold"]) * pq.mV).rescale(waveform_units).magnitude
            )
        except Exception:
            pass
    return algorithm, params


def detect_sample(
    sg: Any,
    waveform: Any,
    sampling_rate: float,
    expected_sample: Optional[int],
    params: Optional[Dict[str, Any]] = None,
    algorithm: Optional[str] = None,
    template_matcher: Any = None,
    multi_channel_getter=None,
) -> Optional[int]:
    """Detect one spike sample around ``expected_sample`` for a unit."""
    if waveform is None:
        return None
    algo = str(algorithm or getattr(sg, "tracking_algorithm", "basic") or "basic")
    merged = params if params is not None else tracking_params(sg)[1]
    detector = SpikeDetectorRegistry.get(algo)
    try:
        hit = detector.detect_sweep(
            sweep_wf=waveform,
            sampling_rate=sampling_rate,
            expected_idx=expected_sample,
            params=merged,
            template_matcher=template_matcher,
            multi_channel_getter=multi_channel_getter,
        )
    except Exception:
        return None
    return None if hit is None else int(hit)


def sample_map_from_sg(
    sg: Any,
    n_stims: int,
) -> Dict[int, int]:
    """Return ``{stim_index: sample}`` for all currently annotated spikes."""
    out: Dict[int, int] = {}
    idx_arr = getattr(sg, "idx_arr", None)
    if idx_arr is not None:
        for i, item in enumerate(idx_arr[:n_stims]):
            if item is None:
                continue
            sample = item[0] if isinstance(item, (tuple, list)) else item
            try:
                out[int(i)] = int(sample)
            except (TypeError, ValueError):
                continue
    return out


def nearest_previous_sample(
    stimno: int,
    samples: Dict[int, int],
    max_skips: int,
) -> Optional[int]:
    """Most recent annotated sample within ``max_skips`` sweeps, else None."""
    first = max(0, int(stimno) - 1 - int(max_skips))
    for prev in range(int(stimno) - 1, first - 1, -1):
        sample = samples.get(prev)
        if sample is not None:
            return int(sample)
    return None


def track_range(
    erp: Any,
    sampling_rate: float,
    sg: Any,
    *,
    start_stim: int = 0,
    end_stim: Optional[int] = None,
    existing: Optional[Dict[int, int]] = None,
    only_missing: bool = True,
    seed_stim: Optional[int] = None,
    seed_sample: Optional[int] = None,
    params: Optional[Dict[str, Any]] = None,
    algorithm: Optional[str] = None,
    template_matcher: Any = None,
    multi_channel_getter=None,
    waveform_units: Any = None,
) -> Dict[int, int]:
    """Fill missing spike samples between sweeps using the unit's algorithm.

    Existing annotated spikes act as anchors.  Tracking advances causally from
    each anchor and stops when more than ``max_skip`` consecutive sweeps miss or
    when ``track_limit`` detections have been added in this run.  The next
    existing anchor then starts a fresh segment, which is what makes this fill
    gaps rather than simply track forward from the current sweep.

    Returns only the newly detected ``{stim_index: sample}`` entries.
    """
    arr = getattr(erp, "magnitude", erp)
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    n_stims = int(arr.shape[0])
    if n_stims == 0:
        return {}

    sr = float(
        sampling_rate.rescale(pq.Hz).magnitude
        if hasattr(sampling_rate, "rescale")
        else sampling_rate
    )
    if sr <= 0:
        return {}

    start = max(0, int(start_stim))
    end = n_stims - 1 if end_stim is None else min(n_stims - 1, int(end_stim))
    if end < start:
        return {}

    algo, merged = tracking_params(
        sg, overrides=params, waveform_units=waveform_units
    )
    if algorithm:
        algo = str(algorithm)
    max_skip = max(0, int(merged.get("max_skip", 1)))
    track_limit = max(0, int(merged.get("track_limit", 0)))

    anchors: Dict[int, int] = {}
    if existing:
        for stim, sample in existing.items():
            if sample is not None and start <= int(stim) <= end:
                anchors[int(stim)] = int(sample)
    if seed_stim is not None and seed_sample is not None:
        anchors[int(seed_stim)] = int(seed_sample)

    if only_missing is False and seed_stim is None and anchors:
        seed_stim = min(anchors)
        seed_sample = anchors[seed_stim]
        anchors = {int(seed_stim): int(seed_sample)}

    if not anchors:
        return {}

    detected: Dict[int, int] = {}
    tracked_total = 0
    sorted_anchors = sorted(anchors)

    for ai, anchor_stim in enumerate(sorted_anchors):
        if anchor_stim > end:
            break
        next_anchor = sorted_anchors[ai + 1] if ai + 1 < len(sorted_anchors) else end + 1
        lo = max(anchor_stim + 1, start)
        hi = min(next_anchor, end + 1)
        prev_sample = anchors[anchor_stim]
        misses = 0

        for stim in range(lo, hi):
            if stim in anchors:
                prev_sample = anchors[stim]
                misses = 0
                continue
            if track_limit > 0 and tracked_total >= track_limit:
                return detected
            if prev_sample is None:
                break

            hit = detect_sample(
                sg,
                arr[stim],
                sr,
                prev_sample,
                params=merged,
                algorithm=algo,
                template_matcher=template_matcher,
                multi_channel_getter=multi_channel_getter,
            )
            if hit is not None:
                detected[stim] = hit
                prev_sample = hit
                misses = 0
                tracked_total += 1
            else:
                misses += 1
                if misses > max_skip:
                    break

    return detected
