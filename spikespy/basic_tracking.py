import logging
from typing import Optional

import neo
import numpy as np
import quantities as pq

logger = logging.getLogger("Tracking")


def track_basic(
    raw_data: neo.AnalogSignal,
    stimulus_events: neo.Event,
    starting_time: pq.s,
    threshold=0.1 * pq.mV,
    window=0.01 * pq.s,
    max_skip=1,
    track_limit=0,
):
    """
    basic tracking as per APTrack plugin

    given an identified time & threshold, look at track below within a given window for a threshold crossing, repeat until end
    """

    start_idx = stimulus_events.as_array().searchsorted(starting_time)

    offset = starting_time - stimulus_events[start_idx - 1]

    traced_times = []
    skipped = 0

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
        print(threshold)
        print(max_val)

        if max_val >= threshold and threshold >= 0:
            skipped = 0
            max_ts = timeslice.times[
                max_idx
            ]  # ((max_idx + t_start) / raw_data.sampling_rate) + raw_data.t_start
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
) -> Optional[int]:
    """Track a spike peak in a single sweep waveform given previous spike latency.
    Returns detected sample index (integer) relative to sweep start, or None.
    """
    from scipy.signal import find_peaks

    if hasattr(waveform, "magnitude"):
        arr_data = waveform.magnitude
    else:
        arr_data = waveform
    arr = np.asarray(arr_data, dtype=np.float32).flatten()

    sr = float(sampling_rate.rescale(pq.Hz).magnitude if hasattr(sampling_rate, "rescale") else sampling_rate)
    prev_s = float(prev_latency.rescale(pq.s).magnitude if hasattr(prev_latency, "rescale") else prev_latency)
    win_s = float((window_ms * pq.ms).rescale(pq.s).magnitude if not hasattr(window_ms, "rescale") else window_ms.rescale(pq.s).magnitude)
    thresh_val = float(threshold.magnitude if hasattr(threshold, "magnitude") else threshold)

    # If waveform has physical units and threshold is provided as mV or unitless mV,
    # convert appropriately to match waveform.units
    if hasattr(waveform, "units"):
        wf_units_str = str(waveform.units).lower()
        if hasattr(threshold, "rescale"):
            thresh_val = float(threshold.rescale(waveform.units).magnitude)
        elif "uv" in wf_units_str or "µv" in wf_units_str:
            # threshold is in mV, waveform is in uV: 1 mV = 1000 uV
            thresh_val = thresh_val * 1000.0
        elif "v" in wf_units_str and "mv" not in wf_units_str:
            # waveform in V: 1 mV = 0.001 V
            thresh_val = thresh_val * 0.001

    if thresh_val < 0:
        flip = True
        thresh_comp = -thresh_val
    else:
        flip = False
        thresh_comp = thresh_val

    prev_sample = int(round(prev_s * sr))
    half_win = max(1, int(round((win_s / 2.0) * sr)))
    start_s = max(0, prev_sample - half_win)
    end_s = min(len(arr), prev_sample + half_win + 1)
    if end_s <= start_s:
        return None

    snip = -arr[start_s:end_s] if flip else arr[start_s:end_s]
    peaks, _ = find_peaks(snip, height=thresh_comp)

    if len(peaks) > 0:
        target = prev_sample - start_s
        chosen_peak_offset = int(peaks[np.argmin(np.abs(peaks - target))])
        return start_s + chosen_peak_offset
    elif len(snip) > 0 and np.max(snip) >= thresh_comp:
        chosen_peak_offset = int(np.argmax(snip))
        return start_s + chosen_peak_offset

    return None



def test_track_basic():
    fname = r"data/test2.h5"
    data = neo.NixIO(fname, "ro").read_block().segments[0]
    stimulus_events = data.events[0]

    raw_data = data.analogsignals[0]
    starting_time = data.events[1][10]  # a tracked unit
    window = 0.02 * pq.s
    threshold = raw_data[raw_data.time_index(starting_time)][0] * 0.8  # 0.1 * pq.mV

    evt2 = track_basic(
        raw_data,
        stimulus_events,
        starting_time=starting_time,
        window=window,
        threshold=threshold,
    )


if __name__ == "__main__":
    print(test_track_basic())
