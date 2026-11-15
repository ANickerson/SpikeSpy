import logging
from typing import Any, Callable, Dict, Optional

import neo
import numpy as np
import quantities as pq

from spikespy.dsp.detectors import SpikeDetectorRegistry

logger = logging.getLogger("Tracking")


def track_basic(
    raw_data: neo.AnalogSignal,
    stimulus_events: neo.Event,
    starting_time: pq.s,
    threshold=0.1 * pq.mV,
    window=0.01 * pq.s,
    max_skip=1,
    track_limit=0,
    algorithm: str = "basic",
    params: Optional[Dict[str, Any]] = None,
):
    """
    basic tracking as per APTrack plugin

    given an identified time & threshold, look at track below within a given window for a threshold crossing, repeat until end
    """
    start_idx = stimulus_events.as_array().searchsorted(starting_time)
    offset = starting_time - stimulus_events[start_idx - 1]

    traced_times = []
    skipped = 0

    if algorithm != "basic":
        detector = SpikeDetectorRegistry.get(algorithm)
        sr = float(raw_data.sampling_rate.rescale(pq.Hz).magnitude)
        win_ms = float(window.rescale(pq.ms).magnitude if hasattr(window, "rescale") else window * 1000.0)
        thresh_val = float(threshold.rescale(pq.mV).magnitude if hasattr(threshold, "rescale") else threshold)
        merged_params = {
            **detector.default_params(),
            "window_ms": win_ms,
            "threshold": thresh_val,
            "max_skip": int(max_skip),
            "track_limit": int(track_limit),
            **(params or {}),
        }
        for idx, x in enumerate(stimulus_events[start_idx:]):
            if (track_limit > 0) and (track_limit < idx):
                break
            timeslice = raw_data.time_slice(x + offset - (window / 2), x + offset + (window / 2))
            if timeslice is None or len(timeslice) == 0:
                skipped += 1
                if skipped > max_skip:
                    break
                continue
            arr = timeslice.as_array()[:, 0] if timeslice.ndim > 1 else np.asarray(timeslice).ravel()
            hit = detector.detect_sweep(
                arr * timeslice.units if hasattr(timeslice, "units") else arr,
                fs_hz=sr,
                expected_idx=len(arr) // 2,
                params=merged_params,
            )
            if hit is not None and 0 <= hit < len(timeslice.times):
                skipped = 0
                max_ts = timeslice.times[int(hit)]
                traced_times.append(max_ts.rescale("s"))
                offset = max_ts - x
            else:
                skipped += 1
            if skipped > max_skip:
                break
        return neo.Event(np.array(traced_times) * pq.s)

    flip = False
    if threshold < 0:
        flip = True
        threshold = -threshold

    for idx, x in enumerate(stimulus_events[start_idx:]):
        if (track_limit > 0) & (track_limit < idx):
            logger.info("Stopping tracking as hit the track_limit")
            break

        timeslice = raw_data.time_slice(
            x + offset - (window / 2), x + offset + (window / 2)
        )
        if flip:
            timeslice = timeslice * -1
        max_idx = np.argmax(timeslice)
        max_val = timeslice[max_idx]

        if max_val >= threshold and threshold >= 0:
            skipped = 0
            max_ts = timeslice.times[max_idx]
            traced_times.append(max_ts.rescale("s"))
            offset = max_ts - x
        else:
            skipped += 1
            logger.info("max < threshold")
        if skipped > max_skip:
            logger.info("Stopping tracking as max skips reached")
            break

    return neo.Event(np.array(traced_times) * pq.s)


def track_single_sweep(
    raw_data,
    stim_time: pq.Quantity,
    prev_latency: pq.Quantity,
    threshold: pq.Quantity,
    window: pq.Quantity,
) -> Optional[pq.Quantity]:
    """
    Given a single stimulus time stim_time and prev_latency (offset from stimulus),
    slices raw_data.time_slice(stim_time + prev_latency - window/2, stim_time + prev_latency + window/2),
    checks for threshold crossing (handling negative thresholds with flip = True just like track_basic),
    and returns the absolute spike timestamp max_ts.rescale("s") if max_val >= abs(threshold),
    or None if no spike crossed threshold.
    """
    if not hasattr(stim_time, "rescale"):
        stim_time = stim_time * pq.s
    if not hasattr(prev_latency, "rescale"):
        prev_latency = prev_latency * pq.s
    if not hasattr(window, "rescale"):
        window = window * pq.s

    flip = False
    thresh = threshold
    if thresh < 0:
        flip = True
        thresh = -thresh

    start_time = stim_time + prev_latency - (window / 2)
    end_time = stim_time + prev_latency + (window / 2)

    try:
        timeslice = raw_data.time_slice(start_time, end_time)
    except Exception as e:
        logger.warning(
            f"Error slicing raw_data between {start_time} and {end_time}: {e}"
        )
        return None

    if timeslice is None or len(timeslice) == 0:
        return None

    ts_data = timeslice
    if flip:
        ts_data = ts_data * -1

    if hasattr(ts_data, "as_array"):
        arr = ts_data.as_array()
    else:
        arr = np.asarray(ts_data)

    max_idx = np.argmax(arr[:, 0] if arr.ndim > 1 else arr)
    max_val = ts_data[max_idx]
    if hasattr(max_val, "__len__") and len(max_val) == 1:
        max_val = max_val[0]

    if hasattr(thresh, "units") and not hasattr(max_val, "units"):
        thresh_comp = thresh.magnitude
    elif not hasattr(thresh, "units") and hasattr(max_val, "units"):
        thresh_comp = thresh * max_val.units
    else:
        thresh_comp = thresh

    if max_val >= thresh_comp:
        max_ts = timeslice.times[max_idx]
        return max_ts.rescale("s")

    return None


def track_sweep_waveform(
    waveform,
    sampling_rate,
    prev_latency,
    threshold=0.01,
    window_ms=5.0,
    algorithm: str = "basic",
    params: Optional[Dict[str, Any]] = None,
    multi_channel_getter: Optional[Callable[[int], np.ndarray]] = None,
    template_matcher: Any = None,
) -> Optional[int]:
    """Track a spike peak in a single sweep waveform given previous spike latency.
    Delegates to ``SpikeDetectorRegistry.get(algorithm)``.
    Returns detected sample index (integer) relative to sweep start, or None.
    """
    detector = SpikeDetectorRegistry.get(algorithm)
    merged: Dict[str, Any] = {
        **detector.default_params(),
        "threshold": threshold,
        "window_ms": window_ms,
        **(params or {}),
    }
    return detector.detect_sweep(
        sweep_wf=waveform,
        sampling_rate=sampling_rate,
        prev_latency=prev_latency,
        params=merged,
        multi_channel_getter=multi_channel_getter,
        template_matcher=template_matcher,
    )
