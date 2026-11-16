from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union
import numpy as np
import quantities as pq
from scipy.signal import find_peaks


@dataclass
class DetectorParamSpec:
    """Declarative specification for a configurable detector parameter."""

    name: str
    label: str
    param_type: str  # "float", "int", "bool", "choice"
    default: Any
    min_val: float = 0.0
    max_val: float = 1000.0
    step: float = 1.0
    decimals: int = 2
    choices: Optional[List[str]] = None
    tooltip: str = ""


@dataclass
class CandidatePeaks:
    """Detected candidate spike peaks on a sweep."""

    sample_indices: np.ndarray
    scores: np.ndarray
    latencies_s: Optional[np.ndarray] = None

    def __init__(
        self,
        sample_indices: Optional[np.ndarray] = None,
        scores: Optional[np.ndarray] = None,
        latencies_s: Optional[np.ndarray] = None,
        *,
        indices: Optional[np.ndarray] = None,
        amplitudes: Optional[np.ndarray] = None,
    ):
        self.sample_indices = np.asarray(
            indices if sample_indices is None and indices is not None else (sample_indices if sample_indices is not None else []),
            dtype=np.int64,
        )
        self.scores = np.asarray(
            amplitudes if scores is None and amplitudes is not None else (scores if scores is not None else []),
            dtype=np.float64,
        )
        self.latencies_s = latencies_s

    @property
    def indices(self) -> np.ndarray:
        return self.sample_indices

    @property
    def amplitudes(self) -> np.ndarray:
        return self.scores


def _normalize_detect_inputs(
    trace_1d: Any = None,
    fs_hz: Any = None,
    expected_idx: Optional[int] = None,
    params: Optional[Dict[str, Any]] = None,
    *,
    sweep_wf: Any = None,
    sampling_rate: Any = None,
    prev_latency: Any = None,
) -> Tuple[np.ndarray, Any, float, Optional[int], Dict[str, Any]]:
    """Normalize positional/keyword inputs across ViewerState, workers, and direct DSP callers."""
    wf = sweep_wf if sweep_wf is not None else trace_1d
    sr_raw = sampling_rate if sampling_rate is not None else fs_hz
    sr = float(sr_raw.rescale(pq.Hz).magnitude if hasattr(sr_raw, "rescale") else (sr_raw or 30000.0))

    if expected_idx is None and prev_latency is not None:
        prev_s = float(prev_latency.rescale(pq.s).magnitude if hasattr(prev_latency, "rescale") else prev_latency)
        expected_idx = int(round(prev_s * sr))

    arr_data = wf.magnitude if hasattr(wf, "magnitude") else wf
    arr = np.asarray(arr_data, dtype=np.float64).ravel() if arr_data is not None else np.empty(0, dtype=np.float64)
    return arr, wf, sr, expected_idx, dict(params or {})


def _convert_threshold_units(wf: Any, threshold: Any) -> float:
    """Convert threshold (possibly in mV or Quantity) to match waveform physical units if present."""
    thresh_val = float(threshold.magnitude if hasattr(threshold, "magnitude") else threshold)
    if hasattr(wf, "units"):
        wf_units_str = str(wf.units).lower()
        if hasattr(threshold, "rescale"):
            thresh_val = float(threshold.rescale(wf.units).magnitude)
        elif "uv" in wf_units_str or "µv" in wf_units_str:
            thresh_val = thresh_val * 1000.0
        elif "v" in wf_units_str and "mv" not in wf_units_str:
            thresh_val = thresh_val * 0.001
    return thresh_val


class SpikeDetectorPlugin:
    """Base class for pluggable spike detection and tracking algorithms."""

    name: str = "base"
    display_name: str = "Base Detector"
    description: str = ""

    def param_schema(self) -> List[DetectorParamSpec]:
        schema = getattr(type(self), "param_schema", None)
        if isinstance(schema, (list, tuple)):
            return list(schema)
        return []

    def default_params(self) -> Dict[str, Any]:
        schema = self.param_schema() if callable(getattr(self, "param_schema", None)) else getattr(self, "param_schema", [])
        return {spec.name: spec.default for spec in schema}

    def merge_params(self, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        merged = self.default_params()
        if params:
            merged.update({k: v for k, v in params.items() if v is not None})
        return merged

    def detect_sweep(
        self,
        trace_1d: Any = None,
        fs_hz: Any = None,
        expected_idx: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
        multi_channel_getter: Optional[Callable[[int], np.ndarray]] = None,
        template_matcher: Any = None,
        *,
        sweep_wf: Any = None,
        sampling_rate: Any = None,
        prev_latency: Any = None,
    ) -> Optional[int]:
        """Detect a single spike sample index on a sweep near ``expected_idx`` / ``prev_latency``."""
        if type(self).track_sweep is not SpikeDetectorPlugin.track_sweep:
            return self.track_sweep(
                trace_1d,
                fs_hz,
                expected_idx,
                params=params,
                multi_channel_getter=multi_channel_getter,
                template_matcher=template_matcher,
                sweep_wf=sweep_wf,
                sampling_rate=sampling_rate,
                prev_latency=prev_latency,
            )
        raise NotImplementedError

    def track_sweep(
        self,
        trace_1d: Any = None,
        fs_hz: Any = None,
        expected_idx: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Optional[int]:
        """Alias for detect_sweep."""
        return self.detect_sweep(
            trace_1d, fs_hz=fs_hz, expected_idx=expected_idx, params=params, **kwargs
        )

    def find_candidates(
        self,
        trace_1d: Any,
        fs_hz: Any,
        params: Optional[Dict[str, Any]] = None,
        min_idx: int = 0,
        max_idx: Optional[int] = None,
        multi_channel_getter: Optional[Callable[[int], np.ndarray]] = None,
        template_matcher: Any = None,
    ) -> CandidatePeaks:
        """Find all candidate spike peaks in ``trace_1d[min_idx:max_idx]``."""
        raise NotImplementedError

    def detect_batch(
        self,
        erp_2d: np.ndarray,
        fs_hz: float,
        start_stim: int,
        start_sample_idx: int,
        params: Optional[Dict[str, Any]] = None,
        existing_mask: Optional[np.ndarray] = None,
        multi_channel_erp_getter: Optional[Callable[[int, int], np.ndarray]] = None,
        template_matcher: Any = None,
    ) -> Dict[int, int]:
        """Track a spike across multiple sweeps starting from ``(start_stim, start_sample_idx)``."""
        p = self.merge_params(params)
        max_skip = int(p.get("max_skip", 1))
        track_limit = int(p.get("track_limit", 0))

        n_stims = erp_2d.shape[0]
        if start_stim < 0 or start_stim >= n_stims:
            return {}

        end_stim = n_stims if track_limit <= 0 else min(n_stims, start_stim + track_limit + 1)
        cur_idx = int(start_sample_idx)
        cur_skip = 0
        tracked: Dict[int, int] = {int(start_stim): cur_idx}

        for stim_idx in range(start_stim + 1, end_stim):
            if existing_mask is not None and stim_idx < len(existing_mask) and existing_mask[stim_idx]:
                continue
            mc_getter = None
            if multi_channel_erp_getter is not None:
                mc_getter = lambda ch, s=stim_idx: multi_channel_erp_getter(ch, s)

            hit = self.detect_sweep(
                erp_2d[stim_idx],
                fs_hz=fs_hz,
                expected_idx=cur_idx,
                params=p,
                multi_channel_getter=mc_getter,
                template_matcher=template_matcher,
            )
            if hit is not None:
                cur_idx = int(hit)
                tracked[int(stim_idx)] = cur_idx
                cur_skip = 0
            else:
                cur_skip += 1
                if cur_skip > max_skip:
                    break
        return tracked


class BasicWindowThresholdDetector(SpikeDetectorPlugin):
    """Windowed threshold-crossing peak detector."""

    name = "basic"
    display_name = "Windowed Threshold (Basic)"
    description = "Finds the closest peak exceeding threshold within a latency window around the previous spike."

    def param_schema(self) -> List[DetectorParamSpec]:
        return [
            DetectorParamSpec(
                name="window_ms",
                label="Window Size (ms)",
                param_type="float",
                default=5.0,
                min_val=0.1,
                max_val=500.0,
                step=0.5,
                decimals=1,
            ),
            DetectorParamSpec(
                name="threshold",
                label="Threshold (mV)",
                param_type="float",
                default=0.01,
                min_val=-1e6,
                max_val=1e6,
                step=0.01,
                decimals=4,
            ),
            DetectorParamSpec(
                name="max_skip",
                label="Max Consecutive Misses",
                param_type="int",
                default=1,
                min_val=0,
                max_val=1000,
                step=1,
                decimals=0,
            ),
            DetectorParamSpec(
                name="track_limit",
                label="Track Limit (sweeps)",
                param_type="int",
                default=0,
                min_val=0,
                max_val=100000,
                step=5,
                decimals=0,
            ),
        ]

    def detect_sweep(
        self,
        trace_1d: Any = None,
        fs_hz: Any = None,
        expected_idx: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
        multi_channel_getter: Optional[Callable[[int], np.ndarray]] = None,
        template_matcher: Any = None,
        *,
        sweep_wf: Any = None,
        sampling_rate: Any = None,
        prev_latency: Any = None,
    ) -> Optional[int]:
        arr, wf_obj, sr, exp_idx, raw_p = _normalize_detect_inputs(
            trace_1d, fs_hz, expected_idx, params,
            sweep_wf=sweep_wf, sampling_rate=sampling_rate, prev_latency=prev_latency,
        )
        n = arr.size
        if n == 0 or sr <= 0:
            return None

        p = self.merge_params(raw_p)
        win_raw = p.get("window_ms", p.get("window_size", 5.0))
        win_s = float(
            (win_raw * pq.ms).rescale(pq.s).magnitude
            if not hasattr(win_raw, "rescale")
            else win_raw.rescale(pq.s).magnitude
        )
        thresh_val = _convert_threshold_units(wf_obj, p.get("threshold", 0.01))

        if thresh_val < 0:
            flip = True
            thresh_comp = -thresh_val
        else:
            flip = False
            thresh_comp = thresh_val

        half_win = max(1, int(round((win_s / 2.0) * sr)))
        if exp_idx is None:
            start_s = 0
            end_s = n
            target = n // 2
        else:
            prev_sample = int(exp_idx)
            start_s = max(0, prev_sample - half_win)
            end_s = min(n, prev_sample + half_win + 1)
            target = prev_sample - start_s

        if end_s <= start_s:
            return None

        snip = -arr[start_s:end_s] if flip else arr[start_s:end_s]
        peaks, _ = find_peaks(snip, height=thresh_comp)
        if len(peaks) > 0:
            chosen_offset = int(peaks[np.argmin(np.abs(peaks - target))])
            return start_s + chosen_offset
        elif len(snip) > 0 and float(np.max(snip)) >= thresh_comp:
            chosen_offset = int(np.argmax(snip))
            return start_s + chosen_offset
        return None

    def find_candidates(
        self,
        trace_1d: Any,
        fs_hz: Any,
        params: Optional[Dict[str, Any]] = None,
        min_idx: int = 0,
        max_idx: Optional[int] = None,
        multi_channel_getter: Optional[Callable[[int], np.ndarray]] = None,
        template_matcher: Any = None,
    ) -> CandidatePeaks:
        arr, wf_obj, sr, _, raw_p = _normalize_detect_inputs(trace_1d, fs_hz, None, params)
        n = arr.size
        if n == 0 or sr <= 0:
            return CandidatePeaks(np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64))

        s0 = max(0, int(min_idx))
        s1 = n if max_idx is None else min(n, int(max_idx))
        if s1 <= s0:
            return CandidatePeaks(np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64))

        p = self.merge_params(raw_p)
        thresh_val = _convert_threshold_units(wf_obj, p.get("threshold", 0.01))
        snip = -arr[s0:s1] if thresh_val < 0 else arr[s0:s1]
        peaks, _ = find_peaks(snip, height=abs(thresh_val))
        abs_idx = (peaks + s0).astype(np.int64)
        scores = arr[abs_idx].astype(np.float64)
        latencies = abs_idx.astype(np.float64) / sr
        return CandidatePeaks(sample_indices=abs_idx, scores=scores, latencies_s=latencies)


class MultiChannelTemplateDetector(SpikeDetectorPlugin):
    """SNR-weighted multi-channel FFT template matching detector."""

    name = "template"
    display_name = "Template Match (Multi-Ch)"
    description = "Matches multi-channel spike waveform templates using SNR-weighted normalized cross-correlation."

    def param_schema(self) -> List[DetectorParamSpec]:
        return [
            DetectorParamSpec(
                name="window_ms",
                label="Search Window (ms)",
                param_type="float",
                default=5.0,
                min_val=0.2,
                max_val=500.0,
                step=0.5,
                decimals=1,
            ),
            DetectorParamSpec(
                name="corr_threshold",
                label="Correlation Threshold",
                param_type="float",
                default=0.5,
                min_val=0.05,
                max_val=5.0,
                step=0.05,
                decimals=2,
            ),
            DetectorParamSpec(
                name="max_skip",
                label="Max Consecutive Misses",
                param_type="int",
                default=1,
                min_val=0,
                max_val=1000,
                step=1,
                decimals=0,
            ),
            DetectorParamSpec(
                name="track_limit",
                label="Track Limit (sweeps)",
                param_type="int",
                default=0,
                min_val=0,
                max_val=100000,
                step=5,
                decimals=0,
            ),
        ]

    def detect_sweep(
        self,
        trace_1d: Any = None,
        fs_hz: Any = None,
        expected_idx: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
        multi_channel_getter: Optional[Callable[[int], np.ndarray]] = None,
        template_matcher: Any = None,
        *,
        sweep_wf: Any = None,
        sampling_rate: Any = None,
        prev_latency: Any = None,
    ) -> Optional[int]:
        arr, wf_obj, sr, exp_idx, raw_p = _normalize_detect_inputs(
            trace_1d, fs_hz, expected_idx, params,
            sweep_wf=sweep_wf, sampling_rate=sampling_rate, prev_latency=prev_latency,
        )
        p = self.merge_params(raw_p)
        window_ms = float(p.get("window_ms", p.get("window_size", 5.0)))
        corr_threshold = float(p.get("corr_threshold", 0.5))
        half_win = max(1, int(round((window_ms / 1000.0) * sr / 2.0)))

        if template_matcher is not None and getattr(template_matcher, "_initialized", False):
            old_thresh = getattr(template_matcher, "threshold", corr_threshold)
            try:
                template_matcher.threshold = corr_threshold
                getter = multi_channel_getter if multi_channel_getter is not None else (lambda _ch: arr)
                res = template_matcher.match_single_stimulus(
                    getter,
                    expected_idx=exp_idx,
                    search_radius_samples=half_win if exp_idx is not None else None,
                )
                if res is not None:
                    return int(res[0])
                return None
            finally:
                template_matcher.threshold = old_thresh

        return BasicWindowThresholdDetector().detect_sweep(
            wf_obj, fs_hz=sr, expected_idx=exp_idx, params=p
        )

    def find_candidates(
        self,
        trace_1d: Any,
        fs_hz: Any,
        params: Optional[Dict[str, Any]] = None,
        min_idx: int = 0,
        max_idx: Optional[int] = None,
        multi_channel_getter: Optional[Callable[[int], np.ndarray]] = None,
        template_matcher: Any = None,
    ) -> CandidatePeaks:
        arr, wf_obj, sr, _, raw_p = _normalize_detect_inputs(trace_1d, fs_hz, None, params)
        n = arr.size
        if n == 0 or sr <= 0:
            return CandidatePeaks(np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64))

        p = self.merge_params(raw_p)
        corr_threshold = float(p.get("corr_threshold", 0.5))
        s0 = max(0, int(min_idx))
        s1 = n if max_idx is None else min(n, int(max_idx))

        if template_matcher is not None and getattr(template_matcher, "_initialized", False):
            getter = multi_channel_getter if multi_channel_getter is not None else (lambda _ch: arr)
            composite = template_matcher.compute_composite_score(getter)
            if composite.size > 0 and s1 > s0:
                sub = composite[s0:s1]
                min_dist = max(1, int(0.001 * sr))
                peaks, _ = find_peaks(sub, height=corr_threshold, distance=min_dist)
                abs_idx = (peaks + s0).astype(np.int64)
                scores = composite[abs_idx].astype(np.float64)
                latencies = abs_idx.astype(np.float64) / sr
                return CandidatePeaks(sample_indices=abs_idx, scores=scores, latencies_s=latencies)

        return BasicWindowThresholdDetector().find_candidates(
            wf_obj, fs_hz=sr, params=p, min_idx=min_idx, max_idx=max_idx
        )


class ZScoreThresholdDetector(SpikeDetectorPlugin):
    """Robust MAD / STD Z-Score automatic threshold detector."""

    name = "zscore"
    display_name = "Auto Z-Score (MAD/STD)"
    description = "Estimates background noise via Median Absolute Deviation (MAD) or STD and detects peaks above Z * sigma."

    def param_schema(self) -> List[DetectorParamSpec]:
        return [
            DetectorParamSpec(
                name="z_thresh",
                label="Z-Score Threshold",
                param_type="float",
                default=4.5,
                min_val=1.0,
                max_val=50.0,
                step=0.5,
                decimals=2,
            ),
            DetectorParamSpec(
                name="polarity",
                label="Spike Polarity",
                param_type="choice",
                default="auto",
                choices=["auto", "positive", "negative", "both"],
            ),
            DetectorParamSpec(
                name="min_distance_ms",
                label="Refractory Period (ms)",
                param_type="float",
                default=1.5,
                min_val=0.1,
                max_val=100.0,
                step=0.2,
                decimals=1,
            ),
            DetectorParamSpec(
                name="use_mad",
                label="Use Robust MAD Sigma",
                param_type="bool",
                default=True,
            ),
        ]

    @staticmethod
    def _estimate_sigma(arr: np.ndarray, use_mad: bool = True) -> Tuple[float, float]:
        if arr.size == 0:
            return 0.0, 1.0
        med = float(np.median(arr))
        if use_mad:
            mad = float(np.median(np.abs(arr - med)))
            sigma = mad / 0.6745 if mad > 0 else float(np.std(arr))
        else:
            sigma = float(np.std(arr))
        return med, max(sigma, 1e-12)

    def _prepare_signal(self, arr: np.ndarray, polarity: str, med: float) -> np.ndarray:
        centered = arr - med
        if polarity == "negative":
            return -centered
        if polarity == "both":
            return np.abs(centered)
        if polarity == "auto":
            if float(np.max(-centered)) > float(np.max(centered)) * 1.15:
                return -centered
            return centered
        return centered

    def detect_sweep(
        self,
        trace_1d: Any = None,
        fs_hz: Any = None,
        expected_idx: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
        multi_channel_getter: Optional[Callable[[int], np.ndarray]] = None,
        template_matcher: Any = None,
        *,
        sweep_wf: Any = None,
        sampling_rate: Any = None,
        prev_latency: Any = None,
    ) -> Optional[int]:
        arr, _, sr, exp_idx, raw_p = _normalize_detect_inputs(
            trace_1d, fs_hz, expected_idx, params,
            sweep_wf=sweep_wf, sampling_rate=sampling_rate, prev_latency=prev_latency,
        )
        n = arr.size
        if n == 0 or sr <= 0:
            return None

        p = self.merge_params(raw_p)
        z_thresh = float(p.get("z_thresh", 4.5))
        polarity = str(p.get("polarity", "auto"))
        use_mad = bool(p.get("use_mad", True))
        window_ms = float(p.get("window_ms", p.get("window_size", 5.0)))
        min_dist = max(1, int(round(float(p.get("min_distance_ms", 1.5)) * sr / 1000.0)))

        med, sigma = self._estimate_sigma(arr, use_mad=use_mad)
        sig = self._prepare_signal(arr, polarity, med)

        half_win = max(1, int(round((window_ms / 1000.0) * sr / 2.0)))
        if exp_idx is None:
            win_start, win_end = 0, n
            center = n // 2
        else:
            c = int(exp_idx)
            win_start = max(0, c - half_win)
            win_end = min(n, c + half_win + 1)
            center = c - win_start

        if win_end <= win_start:
            return None

        w = sig[win_start:win_end]
        height = z_thresh * sigma
        peaks, _ = find_peaks(w, height=height, distance=min_dist)
        if len(peaks) == 0:
            best = int(np.argmax(w))
            if w[best] >= height:
                return int(win_start + best)
            return None

        closest = int(peaks[np.argmin(np.abs(peaks - center))])
        return int(win_start + closest)

    def find_candidates(
        self,
        trace_1d: Any,
        fs_hz: Any,
        params: Optional[Dict[str, Any]] = None,
        min_idx: int = 0,
        max_idx: Optional[int] = None,
        multi_channel_getter: Optional[Callable[[int], np.ndarray]] = None,
        template_matcher: Any = None,
    ) -> CandidatePeaks:
        arr, _, sr, _, raw_p = _normalize_detect_inputs(trace_1d, fs_hz, None, params)
        n = arr.size
        if n == 0 or sr <= 0:
            return CandidatePeaks(np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64))

        p = self.merge_params(raw_p)
        z_thresh = float(p.get("z_thresh", 4.5))
        polarity = str(p.get("polarity", "auto"))
        use_mad = bool(p.get("use_mad", True))
        min_dist = max(1, int(round(float(p.get("min_distance_ms", 1.5)) * sr / 1000.0)))

        s0 = max(0, int(min_idx))
        s1 = n if max_idx is None else min(n, int(max_idx))
        if s1 <= s0:
            return CandidatePeaks(np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64))

        med, sigma = self._estimate_sigma(arr[s0:s1], use_mad=use_mad)
        sig = self._prepare_signal(arr[s0:s1], polarity, med)
        height = z_thresh * sigma
        peaks, _ = find_peaks(sig, height=height, distance=min_dist)
        abs_idx = (peaks + s0).astype(np.int64)
        scores = (sig[peaks] / sigma).astype(np.float64)
        latencies = abs_idx.astype(np.float64) / sr
        return CandidatePeaks(sample_indices=abs_idx, scores=scores, latencies_s=latencies)


class WaveletDenoiseDetector(SpikeDetectorPlugin):
    """Wavelet-denoised spike peak detector using VisuShrink universal thresholding."""

    name = "wavelet"
    display_name = "Wavelet Denoised Peak"
    description = "Denoises the trace using discrete wavelet shrinkage before finding peaks."

    def param_schema(self) -> List[DetectorParamSpec]:
        return [
            DetectorParamSpec(
                name="threshold_sigma",
                label="Noise Sigma Multiplier",
                param_type="float",
                default=3.5,
                min_val=0.5,
                max_val=50.0,
                step=0.5,
                decimals=2,
            ),
            DetectorParamSpec(
                name="wavelet",
                label="Wavelet Basis",
                param_type="choice",
                default="sym7",
                choices=["sym7", "db4", "haar"],
            ),
            DetectorParamSpec(
                name="level",
                label="Decomposition Level",
                param_type="int",
                default=5,
                min_val=1,
                max_val=10,
                step=1,
                decimals=0,
            ),
        ]

    @staticmethod
    def _denoise(arr: np.ndarray, wavelet: str = "sym7", level: int = 5) -> np.ndarray:
        if arr.size < 8:
            return arr.copy()
        try:
            import pywt

            max_lvl = pywt.dwt_max_level(arr.size, pywt.Wavelet(wavelet).dec_len)
            lvl = max(1, min(int(level), max_lvl)) if max_lvl > 0 else 1
            coeffs = pywt.wavedec(arr, wavelet, level=lvl)
            detail = coeffs[-1]
            sigma = float(np.median(np.abs(detail - np.median(detail)))) / 0.6745
            thresh = sigma * np.sqrt(2.0 * np.log(max(arr.size, 2)))
            new_coeffs = [coeffs[0]] + [pywt.threshold(c, thresh, mode="soft") for c in coeffs[1:]]
            rec = pywt.waverec(new_coeffs, wavelet)
            return np.asarray(rec[: arr.size], dtype=np.float64)
        except Exception:
            return arr.copy()

    def detect_sweep(
        self,
        trace_1d: Any = None,
        fs_hz: Any = None,
        expected_idx: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
        multi_channel_getter: Optional[Callable[[int], np.ndarray]] = None,
        template_matcher: Any = None,
        *,
        sweep_wf: Any = None,
        sampling_rate: Any = None,
        prev_latency: Any = None,
    ) -> Optional[int]:
        arr, _, sr, exp_idx, raw_p = _normalize_detect_inputs(
            trace_1d, fs_hz, expected_idx, params,
            sweep_wf=sweep_wf, sampling_rate=sampling_rate, prev_latency=prev_latency,
        )
        n = arr.size
        if n == 0 or sr <= 0:
            return None

        p = self.merge_params(raw_p)
        window_ms = float(p.get("window_ms", p.get("window_size", 5.0)))
        thresh_sigma = float(p.get("threshold_sigma", 3.5))
        wavelet = str(p.get("wavelet", "sym7"))
        level = int(p.get("level", 5))

        denoised = self._denoise(arr, wavelet=wavelet, level=level)
        _, sigma = ZScoreThresholdDetector._estimate_sigma(arr, use_mad=True)

        half_win = max(1, int(round((window_ms / 1000.0) * sr / 2.0)))
        if exp_idx is None:
            win_start, win_end = 0, n
            center = n // 2
        else:
            c = int(exp_idx)
            win_start = max(0, c - half_win)
            win_end = min(n, c + half_win + 1)
            center = c - win_start

        if win_end <= win_start:
            return None

        w = denoised[win_start:win_end]
        raw_thresh = p.get("threshold")
        if raw_thresh is not None and float(raw_thresh.magnitude if hasattr(raw_thresh, "magnitude") else raw_thresh) < 0:
            w = -w
        height = thresh_sigma * sigma
        peaks, _ = find_peaks(w, height=height)
        if len(peaks) == 0:
            best = int(np.argmax(w))
            if w[best] >= height:
                return int(win_start + best)
            return None

        closest = int(peaks[np.argmin(np.abs(peaks - center))])
        return int(win_start + closest)

    def find_candidates(
        self,
        trace_1d: Any,
        fs_hz: Any,
        params: Optional[Dict[str, Any]] = None,
        min_idx: int = 0,
        max_idx: Optional[int] = None,
        multi_channel_getter: Optional[Callable[[int], np.ndarray]] = None,
        template_matcher: Any = None,
    ) -> CandidatePeaks:
        arr, _, sr, _, raw_p = _normalize_detect_inputs(trace_1d, fs_hz, None, params)
        n = arr.size
        if n == 0 or sr <= 0:
            return CandidatePeaks(np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64))

        p = self.merge_params(raw_p)
        thresh_sigma = float(p.get("threshold_sigma", 3.5))
        wavelet = str(p.get("wavelet", "sym7"))
        level = int(p.get("level", 5))

        s0 = max(0, int(min_idx))
        s1 = n if max_idx is None else min(n, int(max_idx))
        if s1 <= s0:
            return CandidatePeaks(np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64))

        sub = arr[s0:s1]
        denoised = self._denoise(sub, wavelet=wavelet, level=level)
        _, sigma = ZScoreThresholdDetector._estimate_sigma(sub, use_mad=True)
        height = thresh_sigma * sigma
        min_dist = max(1, int(round(0.001 * sr)))
        peaks, _ = find_peaks(denoised, height=height, distance=min_dist)
        abs_idx = (peaks + s0).astype(np.int64)
        scores = denoised[peaks].astype(np.float64)
        latencies = abs_idx.astype(np.float64) / sr
        return CandidatePeaks(sample_indices=abs_idx, scores=scores, latencies_s=latencies)


class SpikeDetectorRegistry:
    """Central registry for pluggable spike detection and tracking algorithms."""

    _detectors: Dict[str, SpikeDetectorPlugin] = {}

    @classmethod
    def register(
        cls, plugin: Union[SpikeDetectorPlugin, Type[SpikeDetectorPlugin]]
    ) -> SpikeDetectorPlugin:
        instance = plugin() if isinstance(plugin, type) else plugin
        plugin_key = getattr(instance, "name", None)
        if not plugin_key or plugin_key == "base":
            plugin_key = getattr(instance, "key", None) or plugin_key
            if plugin_key:
                instance.name = plugin_key
        if not plugin_key:
            raise ValueError("SpikeDetectorPlugin must define a non-empty 'name'.")
        cls._detectors[plugin_key] = instance
        return instance

    @classmethod
    def unregister(cls, name: str) -> None:
        cls._detectors.pop(name, None)

    @classmethod
    def get(cls, name: str, strict: bool = False) -> SpikeDetectorPlugin:
        if name in cls._detectors:
            return cls._detectors[name]
        if strict:
            raise KeyError(f"Unknown spike detector plugin: {name!r}")
        return cls._detectors.get("basic") or BasicWindowThresholdDetector()

    @classmethod
    def list_detectors(cls) -> List[SpikeDetectorPlugin]:
        return list(cls._detectors.values())

    @classmethod
    def names(cls) -> List[str]:
        return list(cls._detectors.keys())

    @classmethod
    def list_keys(cls) -> List[str]:
        return list(cls._detectors.keys())

    @classmethod
    def detect_sweep(
        cls,
        algorithm: str,
        trace_1d: Any = None,
        fs_hz: Any = None,
        expected_idx: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Optional[int]:
        detector = cls.get(algorithm)
        return detector.detect_sweep(
            trace_1d, fs_hz=fs_hz, expected_idx=expected_idx, params=params, **kwargs
        )

    @classmethod
    def find_candidates(
        cls,
        algorithm: str,
        trace_1d: Any,
        fs_hz: Any,
        params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> CandidatePeaks:
        detector = cls.get(algorithm)
        return detector.find_candidates(trace_1d, fs_hz=fs_hz, params=params, **kwargs)

    @classmethod
    def detect_batch(
        cls,
        algorithm: str,
        erp_2d: np.ndarray,
        fs_hz: float,
        start_stim: int,
        start_sample_idx: int,
        params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[int, int]:
        detector = cls.get(algorithm)
        return detector.detect_batch(
            erp_2d,
            fs_hz=fs_hz,
            start_stim=start_stim,
            start_sample_idx=start_sample_idx,
            params=params,
            **kwargs,
        )


def register_spike_detector(
    plugin_or_cls: Union[SpikeDetectorPlugin, Type[SpikeDetectorPlugin]]
) -> Union[SpikeDetectorPlugin, Type[SpikeDetectorPlugin]]:
    """Decorator / helper to register a SpikeDetectorPlugin class or instance."""
    SpikeDetectorRegistry.register(plugin_or_cls)
    return plugin_or_cls


# Register built-in detectors
SpikeDetectorRegistry.register(BasicWindowThresholdDetector)
SpikeDetectorRegistry.register(MultiChannelTemplateDetector)
SpikeDetectorRegistry.register(ZScoreThresholdDetector)
SpikeDetectorRegistry.register(WaveletDenoiseDetector)


def select_spikes_in_polygon(
    erp_2d: np.ndarray,
    fs_hz: float,
    polygon_xy: Sequence[Tuple[float, float]],
    start_offset_ms: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Find peak sample indices inside a 2D (latency_ms, stim_idx) polygon corridor."""
    from matplotlib.path import Path as MplPath

    verts = np.asarray(polygon_xy, dtype=np.float64)
    if verts.ndim != 2 or verts.shape[0] < 3 or erp_2d.ndim != 2 or fs_hz <= 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    poly = MplPath(verts)
    n_stims, n_samples = erp_2d.shape
    min_y = max(0, int(np.floor(np.min(verts[:, 1]))))
    max_y = min(n_stims - 1, int(np.ceil(np.max(verts[:, 1]))))
    if max_y < min_y:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    min_x_ms = float(np.min(verts[:, 0])) - float(start_offset_ms)
    max_x_ms = float(np.max(verts[:, 0])) - float(start_offset_ms)
    s_min = max(0, int(np.floor((min_x_ms / 1000.0) * fs_hz)))
    s_max = min(n_samples, int(np.ceil((max_x_ms / 1000.0) * fs_hz)) + 1)
    if s_max <= s_min:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    sample_grid = np.arange(s_min, s_max, dtype=np.int64)
    x_ms_grid = (sample_grid.astype(np.float64) / float(fs_hz)) * 1000.0 + float(start_offset_ms)

    out_stims: List[int] = []
    out_samples: List[int] = []
    for stim_idx in range(min_y, max_y + 1):
        pts = np.column_stack((x_ms_grid, np.full_like(x_ms_grid, float(stim_idx))))
        inside = poly.contains_points(pts)
        if not np.any(inside):
            continue
        valid_samples = sample_grid[inside]
        sub_trace = erp_2d[stim_idx, valid_samples]
        if sub_trace.size == 0:
            continue
        best_local = int(np.argmax(sub_trace))
        out_stims.append(int(stim_idx))
        out_samples.append(int(valid_samples[best_local]))

    return np.asarray(out_stims, dtype=np.int64), np.asarray(out_samples, dtype=np.int64)
