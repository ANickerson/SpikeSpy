from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import quantities as pq


def compute_unit_footprint_from_state(
    state: Any,
    sg: Any = None,
    last_n: int = 50,
    window_ms: float = 15.0,
    align_peaks: bool = True,
    include_latest_sweep: bool = False,
) -> Optional[Dict[str, Any]]:
    """Pure-NumPy spatiotemporal unit footprint computation with zero Qt dependencies."""
    if state is None:
        return None
    if sg is None:
        if hasattr(state, "spike_groups") and 0 <= getattr(state, "cur_spike_group", 0) < len(state.spike_groups):
            sg = state.spike_groups[state.cur_spike_group]
        else:
            return None

    sr = float(
        state.sampling_rate
        if getattr(state, "sampling_rate", None) is not None
        else (
            state.analog_signal.sampling_rate.rescale(pq.Hz).magnitude
            if getattr(state, "analog_signal", None) is not None
            else 30000.0
        )
    )
    if sr <= 0:
        return None

    channels = state.get_available_channels() if hasattr(state, "get_available_channels") else []
    if not channels:
        return None

    ref_chan = (
        getattr(sg, "channel", "")
        or getattr(getattr(state, "analog_signal", None), "name", "")
        or channels[0]
    )
    if ref_chan not in channels:
        ref_chan = channels[0]

    valid_sweeps: List[int] = []
    valid_samples: List[int] = []

    if getattr(sg, "idx_arr", None) is not None and len(sg.idx_arr) > 0:
        idx_arr = sg.idx_arr
        for s_idx in range(len(idx_arr) - 1, -1, -1):
            a = idx_arr[s_idx]
            if a is not None and len(a) > 0 and a[0] is not None:
                valid_sweeps.append(s_idx)
                valid_samples.append(int(a[0]))
                if len(valid_sweeps) >= last_n:
                    break
        valid_sweeps.reverse()
        valid_samples.reverse()
    elif getattr(sg, "event", None) is not None and len(sg.event) > 0 and getattr(state, "event_signal", None) is not None:
        sw_arr, sa_arr = sg.get_valid_sweep_indices_and_samples(
            state.event_signal, sr, state.window_size
        )
        valid_sweeps = list(sw_arr)
        valid_samples = list(sa_arr)

    if not valid_sweeps:
        return None

    if len(valid_sweeps) > last_n:
        valid_sweeps = valid_sweeps[-last_n:]
        valid_samples = valid_samples[-last_n:]

    window_samples = max(8, int(round((window_ms / 1000.0) * sr)))
    if window_samples % 2 != 0:
        window_samples += 1
    half_win = window_samples // 2
    t_ms = (np.arange(window_samples) - half_win) * (1000.0 / sr)

    n_ch = len(channels)
    n_events = len(valid_sweeps)

    chan_erps: Dict[str, Optional[np.ndarray]] = {}
    for ch in channels:
        erp = state.get_channel_erp(ch)
        if erp is not None:
            erp_mag = erp.magnitude if hasattr(erp, "magnitude") else erp
            chan_erps[ch] = erp_mag
        else:
            chan_erps[ch] = None

    if ref_chan not in chan_erps or chan_erps[ref_chan] is None:
        return None

    ref_erp = chan_erps[ref_chan]
    snippets = np.zeros((n_ch, n_events, window_samples), dtype=np.float32)
    aligned_centers: List[int] = []

    for ev_i, (s_idx, s_samp) in enumerate(zip(valid_sweeps, valid_samples)):
        if s_idx >= ref_erp.shape[0]:
            continue

        ref_trace = ref_erp[s_idx]
        n_tot_pts = len(ref_trace)

        center_samp = s_samp
        if align_peaks:
            jitter_search_w = max(4, int(0.001 * sr))
            search_start = max(0, s_samp - jitter_search_w)
            search_end = min(n_tot_pts, s_samp + jitter_search_w + 1)
            if search_end > search_start:
                sub_win = ref_trace[search_start:search_end]
                peak_idx = int(np.argmax(np.abs(sub_win)))
                center_samp = search_start + peak_idx

        aligned_centers.append(center_samp)
        w_start = center_samp - half_win
        w_end = center_samp + half_win

        for ch_i, ch in enumerate(channels):
            c_erp = chan_erps[ch]
            if c_erp is not None and s_idx < c_erp.shape[0]:
                tr = c_erp[s_idx]
                pts_avail = len(tr)
                src_s = max(0, w_start)
                src_e = min(pts_avail, w_end)
                dst_s = max(0, -w_start)
                dst_e = dst_s + (src_e - src_s)
                if src_e > src_s and dst_e <= window_samples:
                    snippets[ch_i, ev_i, dst_s:dst_e] = tr[src_s:src_e]

    mean_matrix = np.mean(snippets, axis=1)
    std_matrix = np.std(snippets, axis=1)

    base_samples = max(3, int(0.12 * window_samples))
    for ch_i in range(n_ch):
        baseline = np.mean(mean_matrix[ch_i, :base_samples])
        mean_matrix[ch_i, :] -= baseline
        for ev_i in range(n_events):
            snippets[ch_i, ev_i, :] -= np.mean(snippets[ch_i, ev_i, :base_samples])

    ptp_amp = np.zeros(n_ch, dtype=np.float32)
    lag_ms = np.zeros(n_ch, dtype=np.float32)
    snr = np.zeros(n_ch, dtype=np.float32)

    ref_ch_idx = channels.index(ref_chan)
    ref_peak_idx = int(np.argmax(np.abs(mean_matrix[ref_ch_idx])))

    for ch_i in range(n_ch):
        ch_mean = mean_matrix[ch_i]
        ptp_amp[ch_i] = float(np.max(ch_mean) - np.min(ch_mean))

        ch_peak_idx = int(np.argmax(np.abs(ch_mean)))
        lag_ms[ch_i] = float((ch_peak_idx - ref_peak_idx) * (1000.0 / sr))

        mad = float(np.median(np.abs(ch_mean)))
        bg_std = (mad / 0.6745) if mad > 1e-9 else float(np.std(ch_mean) or 1.0)
        if bg_std < 1e-9:
            bg_std = 1.0
        snr[ch_i] = float(np.max(np.abs(ch_mean)) / bg_std)

    max_ptp = float(np.max(ptp_amp)) if len(ptp_amp) > 0 else 0.0
    detected_channels = [
        channels[i]
        for i in range(n_ch)
        if ptp_amp[i] >= 0.25 * max_ptp and snr[i] >= 2.0
    ]

    latest_sweep_wf: Optional[np.ndarray] = None
    if include_latest_sweep and aligned_centers:
        mean_center = int(round(float(np.median(aligned_centers))))
        stim_curr = int(getattr(state, "stimno", 0))
        latest_sweep_wf = np.zeros((n_ch, window_samples), dtype=np.float32)
        w_start = mean_center - half_win
        w_end = mean_center + half_win
        for ch_i, ch in enumerate(channels):
            c_erp = chan_erps[ch]
            if c_erp is not None and 0 <= stim_curr < c_erp.shape[0]:
                tr = c_erp[stim_curr]
                pts_avail = len(tr)
                src_s = max(0, w_start)
                src_e = min(pts_avail, w_end)
                dst_s = max(0, -w_start)
                dst_e = dst_s + (src_e - src_s)
                if src_e > src_s and dst_e <= window_samples:
                    snip = np.asarray(tr[src_s:src_e], dtype=np.float32)
                    latest_sweep_wf[ch_i, dst_s:dst_e] = snip - float(np.mean(snip[: min(base_samples, len(snip))]))

    return {
        "channels": channels,
        "ref_channel": ref_chan,
        "t_ms": t_ms,
        "time_ms": t_ms,
        "snippets": snippets,
        "mean_matrix": mean_matrix,
        "mean_waveforms": mean_matrix,
        "std_waveforms": std_matrix,
        "ptp_amp": ptp_amp,
        "peak_amps": ptp_amp,
        "ptp": ptp_amp,
        "lag_ms": lag_ms,
        "lags_ms": lag_ms,
        "snr": snr,
        "detected_channels": detected_channels,
        "n_events": len(valid_sweeps),
        "n_events_used": len(valid_sweeps),
        "latest_sweep_waveforms": latest_sweep_wf,
    }


def auto_detect_channels_from_footprint(
    fp: Optional[Dict[str, Any]],
) -> Optional[Tuple[str, List[str]]]:
    """Identify primary channel and neighbor channels from a footprint result dict."""
    if fp is None or len(fp.get("channels", [])) == 0:
        return None

    best_idx = int(np.argmax(fp["ptp_amp"]))
    primary_chan = fp["channels"][best_idx]
    max_amp = fp["ptp_amp"][best_idx]

    neighbors: List[str] = []
    for i, ch in enumerate(fp["channels"]):
        if ch == primary_chan:
            continue
        if max_amp > 0 and fp["ptp_amp"][i] >= 0.30 * max_amp and fp["snr"][i] >= 2.5:
            neighbors.append(ch)
    return primary_chan, neighbors


__all__ = [
    "compute_unit_footprint_from_state",
    "auto_detect_channels_from_footprint",
]
