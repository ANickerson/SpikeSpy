"""Pure candidate-unit detection for the microneurography search phase.

This module deliberately has no Qt dependencies and never imports
``ViewerState``.  It consumes stimulus-locked ERP arrays that are already on
the caller's side (a snapshot), finds latency bands that recur across sweeps,
scores the evidence, and returns ``CandidateProposal`` objects.  Identity
matching, persistence, and the UI live in the caller.

Peak extraction is delegated to :class:`spikespy.dsp.detectors.SpikeDetectorRegistry`
so candidate search sits on the same DSP stack as the rest of the application.
Optional multi-channel template refinement uses the Qt-free parts of
:class:`spikespy.processing.MultiChannelTemplateMatcher`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
from scipy.signal import find_peaks

from .dsp.detectors import SpikeDetectorRegistry

logger = logging.getLogger(__name__)


def _registry_peak_finder(
    algorithm: str, trace: np.ndarray, fs_hz: float, params: Dict[str, Any]
):
    return SpikeDetectorRegistry.find_candidates(
        algorithm, trace, fs_hz=fs_hz, params=params
    )


@dataclass
class CandidateScanConfig:
    """Tunable values for candidate search.  All times are milliseconds."""

    blank_ms: float = 1.0
    min_peak_distance_ms: float = 0.5
    min_peak_snr: float = 3.0
    tolerance_ms: float = 1.0
    merge_tolerance_ms: float = 1.0
    min_hits: int = 3
    max_jitter_ms: float = 2.0
    max_candidates: int = 8
    max_per_channel: int = 3
    live_scan_window_sweeps: int = 40
    recency_window_sweeps: int = 10
    scan_interval_ms: int = 2000
    review_quiet_ms: int = 2000
    reactivation_score_margin: float = 0.15
    accepted_overlap_suppress: float = 0.6
    max_scan_sweeps: int = 20000
    algorithm: str = "zscore"
    detector_params: Dict[str, Any] = field(default_factory=dict)
    enabled_algorithms: List[str] = field(default_factory=list)
    scan_channels: List[str] = field(default_factory=list)
    template_refine: bool = True
    template_corr_threshold: float = 0.35
    max_template_refine_sweeps: int = 2000


@dataclass
class CandidateEvidence:
    """Quantified support behind one proposal."""

    hit_count: int = 0
    recent_hit_count: int = 0
    jitter_ms: float = 0.0
    hit_rate: float = 0.0
    median_snr: float = 0.0
    waveform_score: float = 0.5
    footprint_score: float = 0.5
    recency_score: float = 0.0
    distinctiveness: float = 0.0


@dataclass
class CandidateProposal:
    """A latency band proposed by :func:`detect_candidates`."""

    channel: str
    latency_center_samples: float
    latency_center_s: float
    latency_tolerance_samples: float
    score: float
    samples_by_sweep: Dict[int, int] = field(default_factory=dict)
    hits: int = 0
    last_seen_stim: int = -1
    first_seen_stim: int = -1
    polarity_fraction: float = 0.5
    waveform_correlation: float = 0.0
    snr: float = 0.0
    neighbor_support: float = 0.0
    algorithm: str = "zscore"
    detector_params: Dict[str, Any] = field(default_factory=dict)
    template_refined: bool = False
    evidence: CandidateEvidence = field(default_factory=CandidateEvidence)
    threshold_signal: float = 0.0
    signal_units: str = ""
    neighbor_channels: List[str] = field(default_factory=list)
    _mean_waveform: Optional[np.ndarray] = field(
        default=None, repr=False, compare=False
    )

    @property
    def center_samples(self) -> float:
        return float(self.latency_center_samples)

    @property
    def tolerance_samples(self) -> float:
        return float(self.latency_tolerance_samples)


@dataclass
class _PeakRow:
    indices: np.ndarray
    scores: np.ndarray
    amplitudes: np.ndarray
    sigma: float


def _as_float_array(value: Any) -> np.ndarray:
    arr = getattr(value, "magnitude", value)
    return np.asarray(arr, dtype=np.float64)


def _robust_sigma(arr: np.ndarray) -> float:
    if arr.size == 0:
        return 1e-12
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    if mad > 0:
        return max(mad / 0.6745, 1e-12)
    return max(float(np.std(arr)), 1e-12)


def _samples_for_ms(ms: float, fs_hz: float) -> int:
    return max(1, int(round(float(ms) * float(fs_hz) / 1000.0)))


def _window_px(ms: float, fs_hz: float) -> int:
    return max(1, int(np.ceil(float(ms) * float(fs_hz) / 1000.0)))


def _detector_params(
    config: CandidateScanConfig,
    sigma: float,
    algorithm_override: Optional[str] = None,
) -> Dict[str, Any]:
    algorithm = str(algorithm_override or config.algorithm or "zscore")
    if algorithm == "zscore":
        params: Dict[str, Any] = {
            "polarity": "both",
            "z_thresh": float(config.min_peak_snr),
            "use_mad": True,
            "min_distance_ms": float(config.min_peak_distance_ms),
        }
    elif algorithm == "wavelet":
        params = {
            "threshold_sigma": float(config.min_peak_snr),
        }
    else:
        threshold = float(config.min_peak_snr) * float(sigma)
        params = {
            "threshold": threshold,
            "window_ms": max(1.0, float(config.tolerance_ms) * 2.0),
        }

    overrides = getattr(config, "detector_params", None)
    if isinstance(overrides, Mapping) and overrides:
        for key, value in overrides.items():
            if value is not None:
                params[key] = value
        if algorithm == "zscore":
            if "z_thresh" not in overrides and config.min_peak_snr is not None:
                params["z_thresh"] = float(config.min_peak_snr)
            if (
                "min_distance_ms" not in overrides
                and config.min_peak_distance_ms is not None
            ):
                params["min_distance_ms"] = float(config.min_peak_distance_ms)
    return params


def _dedupe_peaks(
    indices: np.ndarray,
    scores: np.ndarray,
    amplitudes: np.ndarray,
    min_distance: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if indices.size == 0:
        return indices, scores, amplitudes
    order = np.argsort(indices)
    indices = indices[order]
    scores = scores[order]
    amplitudes = amplitudes[order]
    keep: List[int] = []
    for i in range(indices.size):
        if not keep or (indices[i] - indices[keep[-1]]) >= max(1, min_distance):
            keep.append(i)
        elif abs(scores[i]) > abs(scores[keep[-1]]):
            keep[-1] = i
    keep_arr = np.asarray(keep, dtype=np.int64)
    return indices[keep_arr], scores[keep_arr], amplitudes[keep_arr]


def _extract_peak_row_for_algorithm(
    sweep: np.ndarray,
    fs_hz: float,
    config: CandidateScanConfig,
    peak_finder: Callable[..., Any],
    min_idx: int,
    algorithm_name: str,
    sigma: float,
) -> _PeakRow:
    empty = np.empty(0, dtype=np.float64)
    algorithm = str(algorithm_name or "zscore")
    if algorithm == "template":
        # A template algorithm has no template at proposal time.
        algorithm = "zscore"

    params = _detector_params(config, sigma, algorithm_override=algorithm)
    min_distance = _samples_for_ms(config.min_peak_distance_ms, fs_hz)
    work = np.asarray(sweep, dtype=np.float64).copy()
    if min_idx > 0:
        work[: min(int(min_idx), work.size)] = float(np.median(work))

    try:
        if algorithm == "zscore":
            cand = peak_finder(algorithm, work, fs_hz, params)
            raw_idx = np.asarray(cand.sample_indices, dtype=np.int64)
            raw_scores = np.asarray(cand.scores, dtype=np.float64)
            mask = (raw_idx >= min_idx) & (raw_idx < sweep.size)
            idx = raw_idx[mask]
            scores = (
                raw_scores[mask]
                if raw_scores.shape == raw_idx.shape
                else np.abs(_as_float_array(sweep)[np.clip(idx, 0, sweep.size - 1)])
            )
            amp = _as_float_array(sweep)[np.clip(idx, 0, sweep.size - 1)] - float(
                np.median(sweep)
            )
            return _PeakRow(*_dedupe_peaks(idx, scores, amp, min_distance), sigma)
    except Exception:  # noqa: BLE001 - detector failures must not kill a scan
        logger.debug("Candidate peak finder failed", exc_info=True)
        return _PeakRow(empty.astype(np.int64), empty, empty, sigma)

    # basic / wavelet / custom detectors do not consistently expose polarity in
    # find_candidates, so run one pass on the trace and one on the inverted
    # trace, then merge.
    try:
        forwards = peak_finder(algorithm, work, fs_hz, params)
        backwards = peak_finder(algorithm, -work, fs_hz, params)
    except Exception:  # noqa: BLE001
        logger.debug("Candidate peak finder failed", exc_info=True)
        return _PeakRow(empty.astype(np.int64), empty, empty, sigma)

    idx_parts: List[np.ndarray] = []
    score_parts: List[np.ndarray] = []
    amp_parts: List[np.ndarray] = []
    for cand, _flip in ((forwards, 1.0), (backwards, -1.0)):
        raw_idx = np.asarray(cand.sample_indices, dtype=np.int64)
        mask = (raw_idx >= min_idx) & (raw_idx < sweep.size)
        idx = raw_idx[mask]
        if idx.size == 0:
            continue
        signed = _as_float_array(sweep)[idx] - float(np.median(sweep))
        raw_scores = np.asarray(cand.scores, dtype=np.float64)
        if raw_scores.shape == raw_idx.shape:
            raw_scores = raw_scores[mask]
        else:
            raw_scores = np.abs(signed)
        if algorithm == "basic":
            snr = np.abs(signed) / max(sigma, 1e-12)
        else:
            snr = np.abs(raw_scores) / max(sigma, 1e-12)
        idx_parts.append(idx)
        score_parts.append(snr)
        amp_parts.append(signed)

    if not idx_parts:
        return _PeakRow(empty.astype(np.int64), empty, empty, sigma)
    return _PeakRow(
        *_dedupe_peaks(
            np.concatenate(idx_parts),
            np.concatenate(score_parts),
            np.concatenate(amp_parts),
            min_distance,
        ),
        sigma,
    )


def _extract_peak_row(
    sweep: np.ndarray,
    fs_hz: float,
    config: CandidateScanConfig,
    peak_finder: Callable[..., Any],
    min_idx: int,
) -> _PeakRow:
    """Find peaks on one sweep using the configured detector(s)."""
    empty = np.empty(0, dtype=np.float64)
    sigma = _robust_sigma(sweep)
    if sweep.size == 0 or fs_hz <= 0:
        return _PeakRow(empty.astype(np.int64), empty, empty, sigma)

    enabled = [
        str(a).strip()
        for a in (getattr(config, "enabled_algorithms", None) or [])
        if str(a).strip()
    ]
    if len(enabled) > 1:
        rows = [
            _extract_peak_row_for_algorithm(
                sweep, fs_hz, config, peak_finder, min_idx, algo, sigma
            )
            for algo in enabled
        ]
        idx_parts = [r.indices for r in rows if r.indices.size > 0]
        if not idx_parts:
            return _PeakRow(empty.astype(np.int64), empty, empty, sigma)
        score_parts = [r.scores for r in rows if r.indices.size > 0]
        amp_parts = [r.amplitudes for r in rows if r.indices.size > 0]
        min_distance = _samples_for_ms(config.min_peak_distance_ms, fs_hz)
        return _PeakRow(
            *_dedupe_peaks(
                np.concatenate(idx_parts),
                np.concatenate(score_parts),
                np.concatenate(amp_parts),
                min_distance,
            ),
            sigma,
        )

    return _extract_peak_row_for_algorithm(
        sweep,
        fs_hz,
        config,
        peak_finder,
        min_idx,
        str(config.algorithm or "zscore"),
        sigma,
    )


def _nearest_peak(
    row: _PeakRow, center: float, tolerance: int
) -> Optional[Tuple[int, float, float]]:
    if row.indices.size == 0:
        return None
    diffs = np.abs(row.indices.astype(np.float64) - float(center))
    best = int(np.argmin(diffs))
    if diffs[best] > tolerance:
        return None
    return int(row.indices[best]), float(row.scores[best]), float(row.amplitudes[best])


def _snippet(arr: np.ndarray, sample: int, pre: int, post: int) -> np.ndarray:
    n = arr.size
    out = np.zeros(pre + post, dtype=np.float64)
    start = int(sample) - pre
    end = int(sample) + post
    src_start = max(0, start)
    src_end = min(n, end)
    if src_start < src_end:
        dst_start = src_start - start
        out[dst_start : dst_start + (src_end - src_start)] = arr[src_start:src_end]
    return out


def _normalize(snippet: np.ndarray) -> np.ndarray:
    snippet = np.asarray(snippet, dtype=np.float64)
    std = float(np.std(snippet))
    if std <= 1e-12:
        return snippet - float(np.mean(snippet))
    return (snippet - float(np.mean(snippet))) / std


def _mean_snippet_2d(
    arr2d: np.ndarray,
    locals: Sequence[int],
    samples: Sequence[int],
    fs_hz: float,
) -> Optional[np.ndarray]:
    if len(samples) == 0:
        return None
    pre = _window_px(1.5, fs_hz)
    post = _window_px(2.5, fs_hz)
    snippets = []
    for local, sample in zip(locals, samples):
        if 0 <= int(local) < arr2d.shape[0]:
            snippets.append(_normalize(_snippet(arr2d[int(local)], sample, pre, post)))
    if not snippets:
        return None
    return _normalize(np.median(np.vstack(snippets), axis=0))


def _corr_snippets(snippets: Sequence[np.ndarray]) -> float:
    if not snippets:
        return 0.0
    template = _normalize(np.median(np.vstack(snippets), axis=0))
    denom = float(np.linalg.norm(template))
    if denom <= 1e-12:
        return 0.0
    corrs = []
    for snip in snippets:
        d = float(np.linalg.norm(snip))
        if d <= 1e-12:
            continue
        corrs.append(float(np.dot(snip, template)) / (d * denom))
    return float(np.clip(np.median(corrs), 0.0, 1.0)) if corrs else 0.0


def _snippet_correlation(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return 0.0
    n = min(a.size, b.size)
    if n == 0:
        return 0.0
    a = a[:n]
    b = b[:n]
    da = float(np.linalg.norm(a))
    db = float(np.linalg.norm(b))
    if da <= 1e-12 or db <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / (da * db))


def _waveform_score(
    arr2d: np.ndarray,
    locals: Sequence[int],
    samples: Sequence[int],
    fs_hz: float,
) -> float:
    if len(samples) < 3:
        return 0.5
    pre = _window_px(1.5, fs_hz)
    post = _window_px(2.5, fs_hz)
    snippets = []
    for local, sample in zip(locals, samples):
        if 0 <= int(local) < arr2d.shape[0]:
            snippets.append(_normalize(_snippet(arr2d[int(local)], sample, pre, post)))
    return _corr_snippets(snippets)


def _support_maps(
    rows: Sequence[_PeakRow],
    n_samples: int,
    tolerance: int,
    recent_rows: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (support, recent_support) counting distinct sweeps with a nearby peak."""
    n_rows = len(rows)
    if n_rows == 0 or n_samples <= 0:
        return np.zeros(max(n_samples, 0)), np.zeros(max(n_samples, 0))
    mask = np.zeros((n_rows, n_samples), dtype=np.float32)
    for i, row in enumerate(rows):
        idx = row.indices
        idx = idx[(idx >= 0) & (idx < n_samples)]
        mask[i, idx] = 1.0
    width = max(1, 2 * int(tolerance) + 1)
    box = np.ones(width, dtype=np.float32)
    counts = np.apply_along_axis(lambda r: np.convolve(r, box, mode="same"), 1, mask)
    present = counts > 0.5
    support = present.sum(axis=0).astype(np.float64)
    recent = present[max(0, n_rows - int(recent_rows)) :].sum(axis=0).astype(np.float64)
    return support, recent


def _normalize_existing_units(existing_units: Any) -> Dict[str, List[Dict[int, int]]]:
    """Accept several convenient shapes for accepted-unit sample maps."""
    out: Dict[str, List[Dict[int, int]]] = {}
    if not existing_units:
        return out
    if not isinstance(existing_units, Mapping):
        return out
    for channel, value in existing_units.items():
        maps: List[Dict[int, int]] = []
        if isinstance(value, Mapping):
            if (
                value
                and all(isinstance(k, int) for k in value.keys())
                and all(not isinstance(v, Mapping) for v in value.values())
            ):
                maps.append({int(k): int(v) for k, v in value.items() if v is not None})
            else:
                for item in value.values():
                    if isinstance(item, Mapping):
                        maps.append(
                            {int(k): int(v) for k, v in item.items() if v is not None}
                        )
        else:
            for item in value:
                if isinstance(item, Mapping):
                    maps.append(
                        {int(k): int(v) for k, v in item.items() if v is not None}
                    )
        if maps:
            out[str(channel)] = maps
    return out


def _suppressed_by_accepted(
    channel: str,
    samples_by_sweep: Mapping[int, int],
    accepted: Dict[str, List[Dict[int, int]]],
    tolerance: int,
    config: CandidateScanConfig,
) -> bool:
    maps = accepted.get(channel, [])
    if not maps or not samples_by_sweep:
        return False
    hits = list(samples_by_sweep.items())
    for unit_map in maps:
        overlap = 0
        for stim, sample in hits:
            unit_sample = unit_map.get(int(stim))
            if (
                unit_sample is not None
                and abs(int(unit_sample) - int(sample)) <= tolerance
            ):
                overlap += 1
        if overlap / float(len(hits)) > float(config.accepted_overlap_suppress):
            return True
    return False


def _build_template(
    proposal: CandidateProposal,
    arrays: Dict[str, np.ndarray],
    scan_start: int,
    fs_hz: float,
) -> Optional[Tuple[Any, List[str]]]:
    """Build a normalized multi-channel template from the proposal's hits."""
    if not proposal.samples_by_sweep:
        return None
    try:
        from .processing import MultiChannelTemplateMatcher
    except Exception:  # noqa: BLE001 - processing must remain optional-safe
        return None

    pre = _window_px(2.0, fs_hz)
    post = _window_px(4.0, fs_hz)
    channels = [proposal.channel] + [
        ch for ch in sorted(arrays.keys()) if ch != proposal.channel
    ][:2]

    templates: Dict[str, np.ndarray] = {}
    weights: Dict[str, float] = {}
    ptps: Dict[str, float] = {}
    for ch in channels:
        arr = arrays.get(ch)
        if arr is None or arr.size == 0:
            continue
        snippets = []
        for global_sweep, sample in proposal.samples_by_sweep.items():
            local = int(global_sweep) - int(scan_start)
            if 0 <= local < arr.shape[0]:
                snippets.append(_snippet(arr[local], sample, pre, post))
        if not snippets:
            continue
        mean_snip = np.mean(np.vstack(snippets), axis=0)
        if float(np.std(mean_snip)) <= 1e-12:
            continue
        templates[ch] = _normalize(mean_snip).astype(np.float32)
        ptps[ch] = max(float(np.ptp(mean_snip)), 1e-9)
    if not templates:
        return None
    total_ptp = sum(ptps.values())
    for ch in templates:
        weights[ch] = ptps[ch] / total_ptp if total_ptp > 0 else 1.0 / len(templates)

    matcher = MultiChannelTemplateMatcher()
    matcher.templates = templates
    matcher.weights = weights
    matcher.threshold = 0.0
    matcher._initialized = True
    return matcher, list(templates.keys())


def _refine_proposal(
    proposal: CandidateProposal,
    arrays: Dict[str, np.ndarray],
    local_sweeps: range,
    scan_start: int,
    fs_hz: float,
    config: CandidateScanConfig,
) -> CandidateProposal:
    built = _build_template(proposal, arrays, scan_start, fs_hz)
    if built is None:
        return proposal
    matcher, channels = built
    threshold = float(config.template_corr_threshold)
    matcher.threshold = threshold
    tolerance = max(1, int(round(proposal.latency_tolerance_samples)))
    center = int(round(proposal.latency_center_samples))
    pre = _window_px(2.0, fs_hz)
    post = _window_px(4.0, fs_hz)
    # scipy's ``fftconvolve(..., mode="same")`` shifts a template whose peak is
    # at ``pre`` by ``(L - 1) // 2 - pre`` samples relative to the waveform.
    conv_shift = (pre + post - 1) // 2 - pre

    refine_start = max(0, len(local_sweeps) - int(config.max_template_refine_sweeps))
    refined: Dict[int, int] = {}
    refined_scores: Dict[int, float] = {}
    for local in local_sweeps[refine_start:]:
        global_stim = int(scan_start) + int(local)

        def getter(ch, _local=local):
            arr = arrays.get(ch)
            if arr is None or _local < 0 or _local >= arr.shape[0]:
                return None
            return arr[_local]

        result = matcher.match_single_stimulus(
            getter, expected_idx=center, search_radius_samples=tolerance
        )
        if result is not None and float(result[1]) >= threshold:
            sample = int(result[0]) + int(conv_shift)
            refined[global_stim] = int(
                min(max(sample, 0), int(arrays[proposal.channel].shape[1]) - 1)
            )
            refined_scores[global_stim] = float(result[1])
        elif global_stim in proposal.samples_by_sweep:
            refined[global_stim] = proposal.samples_by_sweep[global_stim]

    if not refined:
        return proposal
    proposal.samples_by_sweep = refined
    proposal.template_refined = True
    proposal.neighbor_channels = [ch for ch in channels if ch != proposal.channel]
    if refined_scores:
        proposal.evidence.waveform_score = max(
            proposal.evidence.waveform_score,
            float(np.clip(np.median(list(refined_scores.values())), 0.0, 1.0)),
        )
    return proposal


def _score_proposal(
    hits: int,
    jitter_ms: float,
    median_snr: float,
    waveform_score: float,
    recency_score: float,
    footprint_score: float,
    distinctiveness: float,
    config: CandidateScanConfig,
) -> float:
    jitter_score = 1.0 / (
        1.0 + float(jitter_ms) / max(float(config.tolerance_ms), 1e-6)
    )
    snr_score = min(1.0, float(median_snr) / 8.0)
    return float(
        0.30 * min(1.0, hits / 8.0)
        + 0.20 * jitter_score
        + 0.15 * float(waveform_score)
        + 0.10 * snr_score
        + 0.10 * float(recency_score)
        + 0.10 * float(footprint_score)
        + 0.05 * float(distinctiveness)
    )


def detect_candidates(
    erp_by_channel: Dict[str, np.ndarray],
    sampling_rate_hz: float,
    scan_start_stim: int,
    scan_end_stim: Optional[int],
    existing_units: Optional[Dict[str, Any]] = None,
    candidate_memory: Optional[Any] = None,
    config: Optional[CandidateScanConfig] = None,
    peak_finder: Optional[Callable[..., Any]] = None,
    waveform_units: Any = None,
) -> List[CandidateProposal]:
    """Detect recurring latency bands in stimulus-locked ERP arrays.

    ``erp_by_channel`` maps channel name to a ``n_sweeps x n_samples`` array.
    ``scan_start_stim`` is the global stimulus index of row 0; rows past
    ``scan_end_stim`` are ignored.  Accepted-unit suppression uses
    ``existing_units`` as ``{channel: [ {stim: sample}, ... ]}`` (a single map
    per channel is accepted as well).
    """
    config = config or CandidateScanConfig()
    peak_finder = peak_finder or _registry_peak_finder
    fs_hz = float(sampling_rate_hz or 0.0)
    if fs_hz <= 0:
        return []

    start = max(0, int(scan_start_stim))
    end = int(scan_end_stim) if scan_end_stim is not None else None

    candidate_memory = candidate_memory or []
    accepted = _normalize_existing_units(existing_units)

    all_arrays: Dict[str, np.ndarray] = {}
    peak_rows: Dict[str, List[_PeakRow]] = {}
    blank_samples = _samples_for_ms(config.blank_ms, fs_hz)
    tolerance = max(1, int(round(float(config.tolerance_ms) * fs_hz / 1000.0)))

    for channel, erp in erp_by_channel.items():
        arr = _as_float_array(erp)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
            continue
        channel = str(channel)
        n_avail = arr.shape[0]
        lo = 0
        hi = n_avail if end is None else min(n_avail, int(end) - start + 1)
        hi = min(hi, n_avail)
        if hi <= lo:
            continue
        rows = []
        for local in range(lo, hi):
            rows.append(
                _extract_peak_row(arr[local], fs_hz, config, peak_finder, blank_samples)
            )
        all_arrays[channel] = arr[lo:hi]
        peak_rows[channel] = rows

    if not peak_rows:
        return []

    proposals: List[CandidateProposal] = []
    recency = max(1, int(config.recency_window_sweeps))
    merge_tol = max(1, int(round(float(config.merge_tolerance_ms) * fs_hz / 1000.0)))

    for channel, rows in peak_rows.items():
        arr = all_arrays[channel]
        n_samples = int(arr.shape[1])
        support, recent_support = _support_maps(rows, n_samples, tolerance, recency)

        peaks, _ = find_peaks(
            support,
            height=float(config.min_hits),
            distance=max(1, merge_tol),
        )
        for center_idx in np.asarray(peaks, dtype=np.int64):
            center_idx = int(center_idx)
            if center_idx < blank_samples:
                continue
            samples_by_sweep: Dict[int, int] = {}
            snrs: List[float] = []
            amps: List[float] = []
            local_seen: List[int] = []
            for local, row in enumerate(rows):
                nearest = _nearest_peak(row, center_idx, tolerance)
                if nearest is None:
                    continue
                sample, snr, amp = nearest
                global_stim = int(start) + int(local)
                samples_by_sweep[global_stim] = int(sample)
                snrs.append(float(snr))
                amps.append(float(amp))
                local_seen.append(local)

            hits = len(samples_by_sweep)
            if hits < int(config.min_hits):
                continue
            values = np.asarray(list(samples_by_sweep.values()), dtype=np.float64)
            latencies_ms = values / fs_hz * 1000.0
            if latencies_ms.size >= 2:
                med = float(np.median(latencies_ms))
                jitter_ms = 1.4826 * float(np.median(np.abs(latencies_ms - med)))
            else:
                jitter_ms = 0.0
            if jitter_ms > float(config.max_jitter_ms):
                continue
            median_snr = float(np.median(snrs)) if snrs else 0.0
            if median_snr < float(config.min_peak_snr):
                continue

            last_local = max(local_seen)
            first_local = min(local_seen)
            recent_hits = int(
                recent_support[min(max(center_idx, 0), max(n_samples - 1, 0))]
            )
            recency_score = min(1.0, recent_hits / float(recency))
            center_global = int(start) + last_local
            center_sample = int(round(float(np.median(values))))

            if _suppressed_by_accepted(
                channel, samples_by_sweep, accepted, tolerance, config
            ):
                continue

            polarity = 0.5
            if amps:
                pos = sum(1 for a in amps if a >= 0)
                polarity = max(pos, len(amps) - pos) / float(len(amps))

            waveform_score = _waveform_score(arr, local_seen, values.tolist(), fs_hz)

            footprint = 0.5
            other_channels = [ch for ch in peak_rows if ch != channel]
            if other_channels:
                matched = 0
                for local, sample in zip(local_seen, values):
                    for other in other_channels:
                        other_rows = peak_rows[other]
                        if local >= len(other_rows):
                            continue
                        if (
                            _nearest_peak(other_rows[local], sample, tolerance)
                            is not None
                        ):
                            matched += 1
                            break
                footprint = matched / float(hits)

            lo_win = max(0, center_idx - merge_tol)
            hi_win = min(n_samples, center_idx + merge_tol + 1)
            outside = (
                np.concatenate([support[:lo_win], support[hi_win:]])
                if support.size
                else np.empty(0)
            )
            outside_best = float(np.max(outside)) if outside.size else 0.0
            distinctiveness = float(
                np.clip(
                    (float(support[center_idx]) - outside_best)
                    / max(float(support[center_idx]), 1.0),
                    0.0,
                    1.0,
                )
            )

            evidence = CandidateEvidence(
                hit_count=hits,
                recent_hit_count=recent_hits,
                jitter_ms=float(jitter_ms),
                hit_rate=hits / float(len(rows)),
                median_snr=median_snr,
                waveform_score=float(np.clip(waveform_score, 0.0, 1.0)),
                footprint_score=float(np.clip(footprint, 0.0, 1.0)),
                recency_score=float(np.clip(recency_score, 0.0, 1.0)),
                distinctiveness=distinctiveness,
            )
            score = _score_proposal(
                hits,
                jitter_ms,
                median_snr,
                evidence.waveform_score,
                evidence.recency_score,
                evidence.footprint_score,
                distinctiveness,
                config,
            )
            sigma = float(np.median([r.sigma for r in rows])) if rows else 0.0
            detector_params = _detector_params(config, sigma)
            threshold_signal = float(
                detector_params.get("threshold")
                or (float(config.min_peak_snr) * float(sigma))
            )
            proposals.append(
                CandidateProposal(
                    channel=channel,
                    latency_center_samples=float(center_sample),
                    latency_center_s=float(center_sample) / fs_hz,
                    latency_tolerance_samples=float(tolerance),
                    score=score,
                    samples_by_sweep=samples_by_sweep,
                    hits=hits,
                    last_seen_stim=int(center_global),
                    first_seen_stim=int(start) + int(first_local),
                    polarity_fraction=float(polarity),
                    waveform_correlation=float(evidence.waveform_score),
                    snr=median_snr,
                    neighbor_support=float(evidence.footprint_score),
                    algorithm=str(config.algorithm or "zscore"),
                    detector_params=detector_params,
                    evidence=evidence,
                    threshold_signal=threshold_signal,
                    signal_units=str(waveform_units or ""),
                    _mean_waveform=_mean_snippet_2d(
                        arr, local_seen, values.tolist(), fs_hz
                    ),
                )
            )

    if not proposals:
        return []

    # Optional second pass: rebuild a multi-channel template from the best
    # hits and re-detect each sweep around the band center.
    if config.template_refine:
        proposals.sort(key=lambda p: p.score, reverse=True)
        refine_budget = max(1, int(config.max_candidates))
        for i, proposal in enumerate(proposals[:refine_budget]):
            if proposal.hits < 3:
                continue
            channel_rows = peak_rows[proposal.channel]
            local_sweeps = range(len(channel_rows))
            try:
                proposals[i] = _refine_proposal(
                    proposal,
                    all_arrays,
                    local_sweeps,
                    start,
                    fs_hz,
                    config,
                )
            except Exception:  # noqa: BLE001 - refinement is best-effort
                logger.debug("Candidate template refinement failed", exc_info=True)
            proposals[i].hits = len(proposals[i].samples_by_sweep)
            values = np.asarray(
                list(proposals[i].samples_by_sweep.values()), dtype=np.float64
            )
            if values.size:
                latencies_ms = values / fs_hz * 1000.0
                med = float(np.median(latencies_ms))
                proposals[i].evidence.hit_count = int(values.size)
                proposals[i].evidence.jitter_ms = 1.4826 * float(
                    np.median(np.abs(latencies_ms - med))
                )
            proposals[i].score = _score_proposal(
                proposals[i].hits,
                proposals[i].evidence.jitter_ms,
                proposals[i].snr,
                proposals[i].evidence.waveform_score,
                proposals[i].evidence.recency_score,
                proposals[i].evidence.footprint_score,
                proposals[i].evidence.distinctiveness,
                config,
            )

    # Non-max suppression.
    proposals.sort(key=lambda p: (p.score, p.hits), reverse=True)
    kept: List[CandidateProposal] = []
    per_channel: Dict[str, int] = {}
    for proposal in proposals:
        if len(kept) >= int(config.max_candidates):
            break
        if per_channel.get(proposal.channel, 0) >= int(config.max_per_channel):
            continue
        suppress = False
        for other in kept:
            if other.channel != proposal.channel:
                continue
            if abs(
                other.latency_center_samples - proposal.latency_center_samples
            ) <= max(merge_tol, proposal.latency_tolerance_samples):
                corr = abs(
                    _snippet_correlation(other._mean_waveform, proposal._mean_waveform)
                )
                # High waveform similarity at the same latency is a duplicate;
                # genuinely different waveforms may coexist in one band.
                if corr >= 0.5:
                    suppress = True
                    break
        if suppress:
            continue
        kept.append(proposal)
        per_channel[proposal.channel] = per_channel.get(proposal.channel, 0) + 1
    return kept
