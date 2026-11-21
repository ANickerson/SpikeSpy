import json
import logging
import sys
import tempfile
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
import time
from tempfile import tempdir
from typing import Any, Dict, List, Optional, Tuple, Union

import neo
import numpy as np
import quantities as pq
from neo import Event
from neo.io import NixIO, NixIOFr
from PySide6.QtCore import QObject, Qt, QTimer, Signal, Slot
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QInputDialog,
    QDialog,
    QFormLayout,
    QLabel,
    QLineEdit,
    QProgressDialog,
    QSpinBox,
    QDialogButtonBox,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from functools import lru_cache
from neo.core.objectlist import ObjectList

from .APTrack_experiment_import import (
    process_folder as open_aptrack,
    APTrackRecording,
    ChannelRoleSpec,
    TypeID,
    get_default_channel_roles,
    normalize_channel_roles,
    preprocess_segment_with_roles,
)
from .NeoOpenEphyisIO import open_ephys_to_neo
from .io_manager import (
    APTrackDialog,
    FileLoadWorker,
    expand_multichannel_segment,
    export_units_csv,
    export_units_npz,
    import_units_csv,
    load_file,
    open_matlab_to_neo,
    prompt_for_neo_file,
    save_state_to_nix,
)


from .processing import (
    BoundedERPCache,
    LiveERPBuffer,
    DiskRowStore,
    clear_live_cache_dir,
    MultiAnalogSignal,
    MultiChannelTemplateMatcher,
    create_erp_signals,
)
from . import basic_tracking
from . import unit_tracking
from .candidate_detection import (
    CandidateEvidence,
    CandidateProposal,
    CandidateScanConfig,
)
from .live_orchestrator import BatchingMode, LiveSweepItem
from .tracking_worker import LiveTrackingManager

lru_numpy_memmap = BoundedERPCache

logger = logging.getLogger(__name__)

UNIT_COLORS: List[Tuple[int, int, int]] = [
    (102, 194, 165),
    (252, 141, 98),
    (141, 160, 203),
    (231, 138, 195),
    (166, 216, 84),
    (255, 217, 47),
    (229, 196, 148),
    (179, 179, 179),
]


@dataclass
class tracked_neuron_unit:
    """A UI class for tracked neurons.
    Stores the current state of neurons tracked and provides high-performance vectorized queries.
    """

    channel: str = ""
    neighbor_channels: List[str] = field(default_factory=list)
    fibre_type: str = ""
    notes: str = ""
    _idx_arr: List[Any] = None
    _event: Event = field(default=None, repr=False)
    auto_track: bool = False
    tracking_window_ms: float = 5.0
    tracking_threshold_mv: float = 0.01
    tracking_max_skips: int = 1
    tracking_channel: str = ""
    tracking_algorithm: str = "basic"
    tracking_params: dict = field(default_factory=dict)
    _cached_window: Optional[Tuple[int, int]] = None
    _cached_window_len: int = -1
    _consecutive_misses: int = 0
    _tracked_count: int = 0
    _template_matcher: Any = None
    pre_offset_s: float = 0.0

    def __init__(
        self,
        event: Event = None,
        channel: str = "",
        neighbor_channels: List[str] = None,
        fibre_type: str = "",
        notes: str = "",
        idx_arr: List[Any] = None,
        auto_track: bool = False,
        tracking_window_ms: float = 5.0,
        tracking_threshold_mv: float = 0.01,
        tracking_max_skips: int = 1,
        tracking_channel: str = "",
        tracking_algorithm: str = "basic",
        tracking_params: dict = None,
        pre_offset_s: float = 0.0,
    ):
        self._cached_window = None
        self._cached_window_len = -1
        self._consecutive_misses = 0
        self._tracked_count = 0
        self._template_matcher = None
        self.pre_offset_s = float(pre_offset_s or 0.0)
        self.tracking_params = (
            dict(tracking_params) if tracking_params is not None else {}
        )
        self.event = event if event is not None else Event()
        self.channel = channel
        self.neighbor_channels = (
            list(neighbor_channels) if neighbor_channels is not None else []
        )
        self.fibre_type = fibre_type
        self.notes = notes
        self._idx_arr = idx_arr

        ann = getattr(self.event, "annotations", {})
        self.auto_track = bool(ann.get("auto_track", auto_track))
        self.tracking_window_ms = float(
            ann.get("tracking_window_ms", tracking_window_ms)
        )
        self.tracking_threshold_mv = float(
            ann.get("tracking_threshold_mv", tracking_threshold_mv)
        )
        self.tracking_max_skips = int(ann.get("tracking_max_skips", tracking_max_skips))
        self.tracking_channel = str(ann.get("tracking_channel", tracking_channel))
        self.tracking_algorithm = str(ann.get("tracking_algorithm", tracking_algorithm))
        if "tracking_params" in ann and isinstance(ann["tracking_params"], dict):
            self.tracking_params.update(ann["tracking_params"])

        # Ensure annotations dictionary has these synced
        if hasattr(self.event, "annotations"):
            self.event.annotations["auto_track"] = self.auto_track
            self.event.annotations["tracking_window_ms"] = self.tracking_window_ms
            self.event.annotations["tracking_threshold_mv"] = self.tracking_threshold_mv
            self.event.annotations["tracking_max_skips"] = self.tracking_max_skips
            self.event.annotations["tracking_channel"] = self.tracking_channel
            self.event.annotations["tracking_algorithm"] = self.tracking_algorithm
            self.event.annotations["tracking_params"] = self.tracking_params

    @property
    def event(self):
        return self._event

    @event.setter
    def event(self, value):
        self._event = value
        self._cached_window = None
        self._cached_window_len = -1

    @event.deleter
    def event(self):
        self._event = None
        self._cached_window = None
        self._cached_window_len = -1

    @property
    def idx_arr(self):
        return self._idx_arr

    @idx_arr.setter
    def idx_arr(self, value):
        self._idx_arr = value
        self._cached_window = None
        self._cached_window_len = -1

    @idx_arr.deleter
    def idx_arr(self):
        self._idx_arr = None
        self._cached_window = None
        self._cached_window_len = -1

    @property
    def window_size(self):
        return self.tracking_window_ms

    @window_size.setter
    def window_size(self, val):
        self.tracking_window_ms = float(val)

    @property
    def threshold(self):
        return self.tracking_threshold_mv

    @threshold.setter
    def threshold(self, val):
        self.tracking_threshold_mv = float(val)

    @property
    def max_skip(self):
        return self.tracking_max_skips

    @max_skip.setter
    def max_skip(self, val):
        self.tracking_max_skips = int(val)

    @property
    def track_limit(self):
        return getattr(self, "tracking_track_limit", 0)

    @track_limit.setter
    def track_limit(self, val):
        self.tracking_track_limit = int(val)

    def get_number_of_events(self, event_signal=None, window_size=None, pre_offset_s=None):
        if self.event is not None and len(self.event) > 0:
            if (
                event_signal is not None
                and len(event_signal) > 0
                and window_size is not None
            ):
                if pre_offset_s is None:
                    pre_offset_s = getattr(self, "pre_offset_s", 0.0)
                pre_offset_s = float(pre_offset_s or 0.0)
                ev_s = self.event.rescale(pq.s).magnitude
                stim_s = event_signal.rescale(pq.s).magnitude
                max_lat_s = float(
                    window_size.rescale(pq.s).magnitude
                    if hasattr(window_size, "rescale")
                    else float(window_size)
                )
                stim_starts = stim_s - pre_offset_s
                idx_sweeps = np.searchsorted(stim_starts, ev_s, side="right") - 1
                valid_sw = (idx_sweeps >= 0) & (idx_sweeps < len(stim_s))
                dt = ev_s[valid_sw] - stim_s[idx_sweeps[valid_sw]]
                in_win = (dt >= -pre_offset_s - 1e-9) & (dt <= max_lat_s + 1e-9)
                return int(len(np.unique(idx_sweeps[valid_sw][in_win])))
            return len(self.event)
        if self._idx_arr is not None:
            return len([x for x in self._idx_arr if x is not None])
        return 0

    def get_window(self, event_signal=None, sampling_rate=None, window_size=None):
        if (
            self._cached_window is not None
            and self.event is not None
            and len(self.event) == self._cached_window_len
        ):
            return self._cached_window

        win = None
        if (
            event_signal is not None
            and sampling_rate is not None
            and self.event is not None
            and len(self.event) > 0
        ):
            sweeps, samples = self.get_valid_sweep_indices_and_samples(
                event_signal, sampling_rate, window_size
            )
            if len(samples) > 0:
                win = (int(np.min(samples)), int(np.max(samples)))

        if win is None and self._idx_arr is not None:
            idx_arr_filt = [
                (x[0] if isinstance(x, (tuple, list)) else x)
                for x in self._idx_arr
                if x is not None
            ]
            if len(idx_arr_filt) > 0:
                win = (int(min(idx_arr_filt)), int(max(idx_arr_filt)))

        if win is not None and self.event is not None:
            self._cached_window = win
            self._cached_window_len = len(self.event)

        return win

    def get_sweep_sample_latencies(
        self,
        event_signal: neo.Event,
        sampling_rate: float,
        window_size: pq.Quantity = None,
        pre_offset_s: float = None,
    ) -> np.ndarray:
        """Returns a 1D float64 array of length len(event_signal) containing the spike latency
        in samples for each sweep (np.nan where no valid spike exists within [-pre_offset_s, window_size]).
        """
        if event_signal is None or len(event_signal) == 0:
            return np.empty(0, dtype=np.float64)

        if pre_offset_s is None:
            pre_offset_s = getattr(self, "pre_offset_s", 0.0)
        pre_offset_s = float(pre_offset_s or 0.0)

        n_sweeps = len(event_signal)
        lat_samples = np.full(n_sweeps, np.nan, dtype=np.float64)

        if self.event is None or len(self.event) == 0:
            return lat_samples

        ev_s = (
            self.event.rescale(pq.s).magnitude
            if hasattr(self.event, "rescale")
            else np.asarray(self.event, dtype=np.float64)
        )
        stim_s = (
            event_signal.rescale(pq.s).magnitude
            if hasattr(event_signal, "rescale")
            else np.asarray(event_signal, dtype=np.float64)
        )

        max_lat_s = (
            float(window_size.rescale(pq.s).magnitude)
            if (window_size is not None and hasattr(window_size, "rescale"))
            else (float(window_size) if window_size is not None else float("inf"))
        )
        sr = float(
            sampling_rate.rescale(pq.Hz).magnitude
            if hasattr(sampling_rate, "rescale")
            else float(sampling_rate)
        )

        stim_starts = stim_s - pre_offset_s
        idx_sweeps = np.searchsorted(stim_starts, ev_s, side="right") - 1
        valid_mask = (idx_sweeps >= 0) & (idx_sweeps < n_sweeps)
        if not np.any(valid_mask):
            return lat_samples

        sw = idx_sweeps[valid_mask]
        dt = ev_s[valid_mask] - stim_s[sw]
        in_win = (dt >= -pre_offset_s - 1e-9) & (dt <= max_lat_s + 1e-9)
        sw_win = sw[in_win]
        dt_win = dt[in_win]
        samp = np.round((dt_win + pre_offset_s) * sr)

        for s_idx, sample_val in zip(sw_win, samp):
            if np.isnan(lat_samples[s_idx]):
                lat_samples[s_idx] = sample_val

        return lat_samples

    def get_valid_sweep_indices_and_samples(
        self,
        event_signal: neo.Event,
        sampling_rate: float,
        window_size: pq.Quantity = None,
        pre_offset_s: float = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (sweep_indices, sample_latencies) as 1D int64 arrays for all sweeps
        that have a valid spike within [-pre_offset_s, window_size].
        """
        lat_samples = self.get_sweep_sample_latencies(
            event_signal, sampling_rate, window_size, pre_offset_s=pre_offset_s
        )
        if len(lat_samples) == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

        valid_mask = ~np.isnan(lat_samples)
        sweeps = np.where(valid_mask)[0].astype(np.int64)
        samples = lat_samples[valid_mask].astype(np.int64)
        return sweeps, samples

    def get_sweep_sample(
        self,
        stimno: int,
        event_signal: neo.Event,
        sampling_rate: float,
        window_size: pq.Quantity = None,
        pre_offset_s: float = None,
    ) -> Optional[int]:
        """Fast O(log N) lookup for a single stimno using np.searchsorted."""
        if (
            event_signal is None
            or stimno < 0
            or stimno >= len(event_signal)
            or self.event is None
            or len(self.event) == 0
        ):
            return None

        if pre_offset_s is None:
            pre_offset_s = getattr(self, "pre_offset_s", 0.0)
        pre_offset_s = float(pre_offset_s or 0.0)

        ev_s = (
            self.event.rescale(pq.s).magnitude
            if hasattr(self.event, "rescale")
            else np.asarray(self.event, dtype=np.float64)
        )
        stim_ts = float(
            event_signal[stimno].rescale(pq.s).magnitude
            if hasattr(event_signal[stimno], "rescale")
            else float(event_signal[stimno])
        )
        max_lat_s = (
            float(window_size.rescale(pq.s).magnitude)
            if (window_size is not None and hasattr(window_size, "rescale"))
            else (float(window_size) if window_size is not None else float("inf"))
        )
        sr = float(
            sampling_rate.rescale(pq.Hz).magnitude
            if hasattr(sampling_rate, "rescale")
            else float(sampling_rate)
        )

        idx = np.searchsorted(ev_s, stim_ts - pre_offset_s)
        if idx < len(ev_s):
            dt = ev_s[idx] - stim_ts
            if -pre_offset_s - 1e-9 <= dt <= max_lat_s + 1e-9:
                return int(np.round((dt + pre_offset_s) * sr))

        return None

    def get_idx_arr(self, event_signal, signal_chan):
        p = [None for _ in range(len(event_signal))]
        if self.event is None or len(self.event) == 0:
            return p
        sr = float(signal_chan.sampling_rate.rescale(pq.Hz).magnitude)
        sweeps, samples = self.get_valid_sweep_indices_and_samples(
            event_signal, sr, pre_offset_s=getattr(self, "pre_offset_s", 0.0)
        )
        for sw, sa in zip(sweeps, samples):
            p[sw] = (int(sa), 1)
        return p

    def get_latencies(
        self, event_signal: pq.UnitTime, pre_offset_s: float = None
    ) -> pq.UnitTime:
        """Returns the latencies of the events in the spikegroup in ms.
        If the event is not in the spikegroup, returns np.nan.
        """
        if self.event is None or len(self.event.times) == 0:
            return np.ones(len(event_signal)) * np.nan * pq.ms
        if len(event_signal) == 0:
            return np.empty(0) * pq.ms

        if pre_offset_s is None:
            pre_offset_s = getattr(self, "pre_offset_s", 0.0)
        pre_offset_s = float(pre_offset_s or 0.0)

        stim_starts = (
            np.asarray(event_signal) * event_signal.units
            - (pre_offset_s * pq.s)
        )
        next_spikes = np.searchsorted(self.event.times, stim_starts).magnitude
        next_spikes = np.clip(next_spikes, 0, len(self.event.times) - 1)
        spike_times = self.event.times[next_spikes] - (
            np.asarray(event_signal) * event_signal.units
        )
        spike_times = spike_times.rescale(pq.ms)
        spike_times[spike_times < (-pre_offset_s * 1000.0 * pq.ms - (1e-6 * pq.ms))] = np.nan * pq.ms
        return spike_times

    def get_sweeps(self, event_signal: pq.UnitTime) -> pq.UnitTime:
        """Returns the sweep numbers/event numbers in the spikegroup."""
        if self.event is None or len(self.event.times) == 0 or len(event_signal) == 0:
            return np.nan
        sweeps = np.searchsorted(event_signal, self.event.times).magnitude
        return sweeps

    def get_tracking_health(
        self,
        event_signal=None,
        current_stimno: Optional[int] = None,
        window_sweeps: int = 20,
    ) -> Dict[str, Any]:
        """Compute tracking health metrics for this unit up to ``current_stimno``."""
        n_events = len(event_signal) if event_signal is not None else 0
        if current_stimno is None:
            if n_events > 0:
                current_stimno = n_events - 1
            elif self._idx_arr:
                current_stimno = len(self._idx_arr) - 1
            else:
                current_stimno = 0
        else:
            current_stimno = int(current_stimno)
            if n_events > 0:
                current_stimno = max(0, min(current_stimno, n_events - 1))

        # Collect sweep -> latency_ms (if computable) or presence from idx_arr and/or event
        sweep_latencies_ms: Dict[int, Optional[float]] = {}
        if self._idx_arr is not None:
            sr_hz: Optional[float] = None
            if (
                event_signal is not None
                and self.event is not None
                and len(self.event) > 0
                and n_events > 0
            ):
                # idx_arr only stores sample indices; latency_ms is derived from event timestamps when present.
                sr_hz = None
            for sw, item in enumerate(self._idx_arr):
                if item is None:
                    continue
                sample = item[0] if isinstance(item, (tuple, list)) else item
                if sample is not None:
                    sweep_latencies_ms[int(sw)] = None

        if (
            event_signal is not None
            and n_events > 0
            and self.event is not None
            and len(self.event) > 0
        ):
            ev_s = (
                self.event.rescale(pq.s).magnitude
                if hasattr(self.event, "rescale")
                else np.asarray(self.event, dtype=np.float64)
            )
            stim_s = (
                event_signal.rescale(pq.s).magnitude
                if hasattr(event_signal, "rescale")
                else np.asarray(event_signal, dtype=np.float64)
            )
            ev_s = np.asarray(ev_s, dtype=np.float64).ravel()
            stim_s = np.asarray(stim_s, dtype=np.float64).ravel()
            pre_offset_s = float(getattr(self, "pre_offset_s", 0.0) or 0.0)
            stim_starts = stim_s - pre_offset_s
            idx_sweeps = np.searchsorted(stim_starts, ev_s, side="right") - 1
            valid_mask = (idx_sweeps >= 0) & (idx_sweeps < len(stim_s))
            for sw, t_ev in zip(idx_sweeps[valid_mask], ev_s[valid_mask]):
                sw_i = int(sw)
                dt_ms = float((t_ev - stim_s[sw_i]) * 1000.0)
                if dt_ms >= -pre_offset_s * 1000.0 - 1e-6 and sw_i not in sweep_latencies_ms:
                    sweep_latencies_ms[sw_i] = dt_ms
                elif dt_ms >= 0.0 and sweep_latencies_ms.get(sw_i) is None:
                    sweep_latencies_ms[sw_i] = dt_ms

        total_spikes = len(sweep_latencies_ms)
        if total_spikes == 0 and self.event is not None and len(self.event) > 0 and n_events == 0:
            total_spikes = len(self.event)

        if n_events == 0 and not self._idx_arr:
            recent_window = 0
        elif current_stimno < 0:
            recent_window = 0
        else:
            recent_window = min(max(1, int(window_sweeps)), current_stimno + 1)

        if recent_window > 0:
            win_start = max(0, current_stimno - recent_window + 1)
            recent_hits = sum(
                1 for sw in range(win_start, current_stimno + 1) if sw in sweep_latencies_ms
            )
            hit_rate = float(recent_hits) / float(recent_window)
        else:
            recent_hits = 0
            hit_rate = 0.0

        # Trailing consecutive misses up to current_stimno
        if recent_window > 0:
            consecutive_misses = 0
            for sw in range(current_stimno, -1, -1):
                if sw in sweep_latencies_ms:
                    break
                consecutive_misses += 1
        else:
            consecutive_misses = 0

        self._consecutive_misses = int(consecutive_misses)

        # Latencies in ms for first and latest detected spikes (up to current_stimno if sweeps exist)
        ordered_sweeps = sorted(
            sw for sw in sweep_latencies_ms.keys()
            if (recent_window == 0 or sw <= current_stimno)
        )
        if not ordered_sweeps:
            ordered_sweeps = sorted(sweep_latencies_ms.keys())

        lat_values = [
            sweep_latencies_ms[sw]
            for sw in ordered_sweeps
            if sweep_latencies_ms[sw] is not None
        ]
        if lat_values:
            first_latency_ms = float(lat_values[0])
            last_latency_ms: Optional[float] = float(lat_values[-1])
            latency_shift_ms = float(last_latency_ms - first_latency_ms)
        else:
            last_latency_ms = None
            latency_shift_ms = 0.0

        max_skips = max(1, int(getattr(self, "tracking_max_skips", 1)))
        if total_spikes == 0:
            status = "empty"
        elif not bool(getattr(self, "auto_track", False)):
            status = "manual"
        elif consecutive_misses == 0:
            status = "locked"
        elif consecutive_misses <= max_skips:
            status = "skipping"
        else:
            status = "lost"

        return {
            "total_spikes": int(total_spikes),
            "recent_hits": int(recent_hits),
            "recent_window": int(recent_window),
            "hit_rate": float(hit_rate),
            "consecutive_misses": int(consecutive_misses),
            "last_latency_ms": last_latency_ms,
            "latency_shift_ms": float(latency_shift_ms),
            "status": status,
        }


@dataclass
class CandidateUnit:
    """A provisional candidate offered by the search phase.

    ``unit`` is a normal :class:`tracked_neuron_unit` payload (event, idx_arr,
    channel, tracking parameters) that is never part of ``spike_groups`` until
    the user accepts the candidate.
    """

    candidate_id: int
    unit: tracked_neuron_unit
    algorithm: str = "zscore"
    detector_params: Dict[str, Any] = field(default_factory=dict)
    template_refined: bool = False
    name: str = "candidate_0"
    name_edited: bool = False
    channel: str = ""
    neighbor_channels: List[str] = field(default_factory=list)
    score: float = 0.0
    evidence: Any = field(default_factory=CandidateEvidence)
    latency_center_samples: float = 0.0
    latency_tolerance_samples: float = 0.0
    status: str = "pending"
    score_at_discard: Optional[float] = None
    channel_at_discard: Optional[str] = None
    first_seen_stim: int = 0
    last_seen_stim: int = 0
    created_at_s: float = 0.0
    notes: str = ""
    notes_edited: bool = False
    tracking_window_ms: float = 5.0
    tracking_threshold_mv: float = 0.01
    tracking_max_skips: int = 3
    threshold_signal: float = 0.0
    signal_units: str = ""

    def __post_init__(self):
        if not self.name:
            self.name = f"candidate_{self.candidate_id}"
        if not self.channel:
            self.channel = getattr(self.unit, "channel", "") or ""
        if not self.neighbor_channels:
            self.neighbor_channels = list(
                getattr(self.unit, "neighbor_channels", []) or []
            )
        if self.evidence is None:
            self.evidence = CandidateEvidence()
        if not self.created_at_s:
            self.created_at_s = time.time()

    @property
    def hit_count(self) -> int:
        return int(getattr(self.evidence, "hit_count", 0) or 0)

    @property
    def recent_hit_count(self) -> int:
        return int(getattr(self.evidence, "recent_hit_count", 0) or 0)


class ViewerState(QObject):
    onLoadNewFile = Signal()
    onFileLoaded = Signal(str, str)
    onUnitChange = Signal(int)
    onUnitGroupChange = Signal()
    onActiveUnitChanged = Signal(int)
    onUnitListChanged = Signal()
    onUnitParamsChanged = Signal(int)
    onUnitSpikeChanged = Signal(int, int)
    onStimNoChange = Signal(int)
    onNewSweep = Signal(int)
    # Live UI refresh period; sweeps arriving within one period trigger a single redraw.
    # Set to 0 to emit per sweep (synchronous).
    live_frame_interval_ms: int = 20
    onNewSpikeEvents = Signal(int)
    onStreamStateChanged = Signal(bool)
    onEventsUpdated = Signal()
    onUnitAnnotated = Signal(dict)
    onChannelChange = Signal(str)
    onUIRefreshRateChange = Signal(int)
    onChannelGroupsChanged = Signal()
    onSharedChannelSetChanged = Signal(str)
    onHideViewTopBarsChanged = Signal(bool)
    onFollowLiveChanged = Signal(bool)
    onFollowLiveEventsChanged = Signal(bool)
    onCandidatesChanged = Signal()
    onCandidateSelectionChanged = Signal(int)
    onCandidateScanStateChanged = Signal(bool)
    onCandidateConfigChanged = Signal()
    onSharedVisualSettingsChanged = Signal(str, dict)
    onProtocolMarkersChanged = Signal()
    onLiveConfigChanged = Signal()
    onChannelRolesChanged = Signal()

    @staticmethod
    def _default_visual_settings() -> Dict[str, Any]:
        return {
            "x_range": None,
            "y_range": None,
            "scale_mode": "std",
            "color_scale": (0.5, 2.0),
            "trace_scale": 1.0,
            "show_track_window": True,
            "show_template": True,
        }

    def __init__(self, parent=None, segment=None, spike_groups=None):
        super().__init__(parent)

        self._file_load_worker = None
        self._file_load_progress = None
        self._file_load_cancelled = False
        self._loaded_file_path: Optional[str] = None
        self._loaded_file_type: Optional[str] = None
        self._last_aptrack_config: Optional[Dict[str, Any]] = None

        self.segment: neo.Segment = segment
        self.spike_groups: List[tracked_neuron_unit] = spike_groups or []
        self.stimno = 0
        self.cur_spike_group = 0
        self.trace_count = 3
        self.analog_signal: neo.AnalogSignal = None
        self.sampling_rate = None
        self.event_signal: neo.Event = None
        # Growable backing store for live stimulus timestamps. Events are views into
        # this buffer, so growth must reallocate instead of resizing in place.
        self._stim_times_buf = np.empty(0, dtype=np.float64)
        self._stim_times_count = 0
        self._stim_times_event = None
        self.window_size = 0.5 * pq.s
        self.undo_queue = []
        self.MAX_UNDO = 10
        self.title = ""
        self.ui_refresh_rate_ms: int = 33  # ~30 FPS

        # Channel roles & preprocessing configuration
        self.channel_roles: List[Dict[str, Any]] = [
            r.to_dict() for r in get_default_channel_roles()
        ]
        self._channel_roles_customized: bool = False

        # Channel groups storage: name -> list of channel names
        self.channel_groups: dict = {}

        # Shared channel sets for coordinated multi-panel viewing
        self.shared_channel_sets: dict = {
            "Shared Group A": {
                "channels": [],
                "group_name": "",
                "unit_idx": None,
                "multichannel": True,
                "visual_settings": self._default_visual_settings(),
            },
            "Shared Group B": {
                "channels": [],
                "group_name": "",
                "unit_idx": None,
                "multichannel": True,
                "visual_settings": self._default_visual_settings(),
            },
        }
        self.protocol_markers: List[Dict[str, Any]] = []

        # Live streaming state attributes
        self.is_streaming: bool = False
        self.follow_live: bool = True
        self.auto_track_enabled: bool = False
        self.auto_switch_unit_channel: bool = False
        self.hide_view_top_bars: bool = False
        self._live_erp_buffer: Optional[LiveERPBuffer] = None
        # Store live sweeps in memory-mapped temp files instead of RAM
        self.live_disk_cache: bool = True
        self.live_config: Dict[str, Any] = {
            "host": "127.0.0.1",
            "http_port": 37497,
            "data_port": 5556,
            "event_port": 5557,
            "stream_mode": "zmq_raw",
            "rec_chan": 0,
            "active_channels": "0-3",
            "trigger_source": "ttl",
            "adc_trigger_chan": 1,
            "ttl_threshold_v": 2.5,
            "ttl_refractory_ms": 100.0,
            "window_ms": 500.0,
            "pre_ms": 0.0,
            "online_filter": True,
            "filter_low_hz": 500.0,
            "filter_high_hz": 7000.0,
            "auto_align_clock": True,
            "sync_annotations": True,
            "live_disk_cache": bool(getattr(self, "live_disk_cache", True)),
            "ui_refresh_rate_ms": int(getattr(self, "ui_refresh_rate_ms", 33)),
            "channel_roles": [dict(r) for r in self.channel_roles],
        }
        self._live_frame_timer = None
        self._live_frame_pending_idx = None
        self._last_live_emitted_stimno = None
        self._live_raw_stores: dict = {}
        self._live_aux_erp_buffers: dict = {}
        self._live_channel_signals: dict = {}
        self.active_channel_names: List[str] = []
        self.focus_channels: List[str] = []

        self.tracking_manager = LiveTrackingManager(self)
        self.tracking_manager.spike_tracked.connect(self._on_async_spike_tracked)

        # Orchestrator running view-registered processors on new live sweeps.
        # Created lazily by `live_orchestrator` when a view first registers.
        self._live_orchestrator = None

        # Candidate-unit search store (separate from spike_groups until accepted).
        self.candidate_units: Dict[int, CandidateUnit] = {}
        self.active_candidate_id: Optional[int] = None
        self.candidate_config: CandidateScanConfig = CandidateScanConfig()
        self.auto_scan_live: bool = False
        self._candidate_scan_manager = None
        self._candidate_scan_generation = 0
        self._candidate_last_scan_stim = -1
        self._candidate_last_interaction = 0.0
        self._candidate_id_counter = 0
        self._candidate_rescan_discarded = False
        self._candidate_scan_truncated = False
        self._candidate_preferred_center: Dict[int, float] = {}
        self.onStreamStateChanged.connect(self._update_candidate_auto_scan)

        from spikespy.ui_diagnostics import AdaptiveFrameScheduler, UIWatchdog

        self.frame_scheduler = AdaptiveFrameScheduler(
            self, base_interval_ms=self.ui_refresh_rate_ms
        )
        self.ui_watchdog = UIWatchdog(
            self, heartbeat_interval_ms=100, stall_threshold_ms=750.0
        )

    @property
    def live_orchestrator(self):
        """LiveOrchestrator executing view-registered functions on new sweeps.

        Views register processing functions here instead of connecting to
        onNewSweep with per-view throttles; the orchestrator applies LIFO
        (latest wins) and numpy-friendly batching.
        """
        from .live_orchestrator import LiveOrchestrator

        if self._live_orchestrator is None:
            self._live_orchestrator = LiveOrchestrator(self)
        return self._live_orchestrator

    @property
    def pre_offset_s(self) -> float:
        """Pre-stimulus window duration in seconds."""
        return getattr(self, "_pre_offset_s", 0.0)

    @pre_offset_s.setter
    def pre_offset_s(self, val: float):
        self._pre_offset_s = max(0.0, float(val or 0.0))
        if hasattr(self, "live_config") and isinstance(self.live_config, dict):
            self.live_config["pre_ms"] = self._pre_offset_s * 1000.0
        if hasattr(self, "spike_groups") and self.spike_groups:
            for sg in self.spike_groups:
                sg.pre_offset_s = self._pre_offset_s

    @property
    def pre_stim_ms(self) -> float:
        """Pre-stimulus window duration in milliseconds."""
        return self.pre_offset_s * 1000.0

    @pre_stim_ms.setter
    def pre_stim_ms(self, val: float):
        self.pre_offset_s = float(val or 0.0) / 1000.0

    @Slot(int, int, int)
    def _on_async_spike_tracked(self, unit_idx: int, detected_sample: int, stimno: int):
        self.setUnit(
            detected_sample,
            stimno=stimno,
            unit_idx=unit_idx,
            save_undo=not getattr(self, "is_streaming", False),
        )

    def set_ui_refresh_rate_ms(self, interval_ms: int):
        self.ui_refresh_rate_ms = max(5, int(interval_ms))
        if hasattr(self, "live_config") and isinstance(self.live_config, dict):
            self.live_config["ui_refresh_rate_ms"] = self.ui_refresh_rate_ms
        if hasattr(self, "frame_scheduler") and self.frame_scheduler is not None:
            self.frame_scheduler.set_base_interval_ms(self.ui_refresh_rate_ms)
        self.onUIRefreshRateChange.emit(self.ui_refresh_rate_ms)

    def get_ui_refresh_rate_ms(self) -> int:
        return getattr(self, "ui_refresh_rate_ms", 33)

    def get_live_config(self) -> Dict[str, Any]:
        self.live_config["live_disk_cache"] = bool(
            getattr(self, "live_disk_cache", True)
        )
        self.live_config["ui_refresh_rate_ms"] = int(
            getattr(self, "ui_refresh_rate_ms", 33)
        )
        return dict(self.live_config)

    def update_live_config(self, **kwargs) -> bool:
        int_keys = {
            "http_port",
            "data_port",
            "event_port",
            "rec_chan",
            "adc_trigger_chan",
            "ui_refresh_rate_ms",
        }
        float_keys = {
            "ttl_threshold_v",
            "ttl_refractory_ms",
            "window_ms",
            "pre_ms",
            "filter_low_hz",
            "filter_high_hz",
        }
        bool_keys = {
            "online_filter",
            "auto_align_clock",
            "sync_annotations",
            "live_disk_cache",
        }
        str_keys = {
            "host",
            "stream_mode",
            "active_channels",
            "trigger_source",
        }

        changed = False
        refresh_rate_updated = False
        roles_updated = False
        for k, v in kwargs.items():
            if k == "channel_roles":
                coerced = [r.to_dict() for r in normalize_channel_roles(v)]
            elif k in int_keys:
                coerced = int(v)
            elif k in float_keys:
                coerced = float(v)
            elif k in bool_keys:
                coerced = bool(v)
            elif k in str_keys:
                coerced = str(v)
            else:
                coerced = v

            if k == "ui_refresh_rate_ms":
                coerced = max(5, int(coerced))
                cur = int(getattr(self, "ui_refresh_rate_ms", 33))
            elif k == "live_disk_cache":
                cur = bool(getattr(self, "live_disk_cache", True))
            else:
                cur = self.live_config.get(k)

            if cur != coerced or k not in self.live_config:
                self.live_config[k] = coerced
                changed = True
                if k == "live_disk_cache":
                    self.live_disk_cache = bool(coerced)
                elif k == "ui_refresh_rate_ms":
                    refresh_rate_updated = True
                elif k == "channel_roles":
                    self.channel_roles = [dict(r) for r in coerced]
                    self._channel_roles_customized = True
                    self.live_config["channel_roles_customized"] = True
                    roles_updated = True

        if refresh_rate_updated:
            self.set_ui_refresh_rate_ms(int(self.live_config["ui_refresh_rate_ms"]))

        if roles_updated:
            self.onChannelRolesChanged.emit()

        if changed:
            self.onLiveConfigChanged.emit()
        return changed

    def get_channel_roles(self) -> List[Dict[str, Any]]:
        return [dict(r) for r in getattr(self, "channel_roles", [])]

    def _sync_roles_to_live_config(self, norm_roles: List[ChannelRoleSpec]):
        for role in norm_roles:
            if not role.enabled:
                continue
            if role.purpose == "probe":
                self.live_config["online_filter"] = bool(role.bandpass)
                self.live_config["filter_low_hz"] = float(role.bandpass_low)
                self.live_config["filter_high_hz"] = float(role.bandpass_high)
            elif role.purpose == "stim_trigger":
                if role.ttl_threshold is not None and role.ttl_threshold > 0:
                    self.live_config["ttl_threshold_v"] = float(role.ttl_threshold)
                self.live_config["ttl_refractory_ms"] = float(role.ttl_refractory_ms)

    def set_channel_roles(
        self,
        roles,
        apply_preprocess: bool = False,
        reload_if_aptrack: bool = False,
    ) -> bool:
        norm_roles = normalize_channel_roles(roles)
        role_dicts = [r.to_dict() for r in norm_roles]
        self.channel_roles = role_dicts
        self._channel_roles_customized = True
        self.live_config["channel_roles"] = [dict(r) for r in role_dicts]
        self.live_config["channel_roles_customized"] = True
        self._sync_roles_to_live_config(norm_roles)

        self.onChannelRolesChanged.emit()
        self.onLiveConfigChanged.emit()

        if (
            reload_if_aptrack
            and getattr(self, "_loaded_file_type", None) == "APTrack"
            and getattr(self, "_loaded_file_path", None)
            and Path(self._loaded_file_path).exists()
            and not self.is_streaming
        ):
            cfg = dict(self._last_aptrack_config or {"record": 1})
            cfg["roles"] = [dict(r) for r in role_dicts]
            self.loadFile(
                self._loaded_file_path,
                type="APTrack",
                blocking=True,
                aptrack_config=cfg,
            )
            return True

        if apply_preprocess and self.segment is not None and not self.is_streaming:
            self.apply_channel_roles_preprocessing()

        return True

    def _select_role_signals(
        self,
        seg: neo.Segment,
        norm_roles: List[ChannelRoleSpec],
    ):
        sigs = list(getattr(seg, "analogsignals", []) or [])
        new_analog = None
        if sigs:
            # Prefer rd.0.bp, then rd.0, then probe role (.bp if bandpass else raw), then current if still present, else first
            for preferred in ("rd.0.bp", "rd.0"):
                new_analog = next((s for s in sigs if getattr(s, "name", "") == preferred), None)
                if new_analog is not None:
                    break
            if new_analog is None:
                for r in norm_roles:
                    if r.enabled and r.purpose == "probe":
                        cand_names = [f"{r.name}.bp", r.name] if r.bandpass else [r.name]
                        for cname in cand_names:
                            new_analog = next(
                                (s for s in sigs if getattr(s, "name", "") == cname), None
                            )
                            if new_analog is not None:
                                break
                    if new_analog is not None:
                        break
            if new_analog is None and self.analog_signal in sigs:
                new_analog = self.analog_signal
            if new_analog is None:
                new_analog = sigs[0]

        evts = list(getattr(seg, "events", []) or [])
        non_unit_evts = [
            e
            for e in evts
            if not (getattr(e, "name", "") or "").startswith("unit")
            and e.annotations.get("type") != "unit"
        ]
        new_event = None
        if non_unit_evts:
            for r in norm_roles:
                if r.enabled and r.purpose == "stim_trigger":
                    new_event = next(
                        (e for e in non_unit_evts if getattr(e, "name", "") == r.name),
                        None,
                    )
                    if new_event is not None:
                        break
            if new_event is None:
                new_event = next(
                    (e for e in non_unit_evts if getattr(e, "name", "") == "env.stim"),
                    None,
                )
            if (
                new_event is None
                and self.event_signal in non_unit_evts
                and len(self.event_signal) > 0
            ):
                new_event = self.event_signal
            if new_event is None:
                new_event = next(
                    (
                        e
                        for e in non_unit_evts
                        if e.annotations.get("type_id") == TypeID.TTL.value
                    ),
                    None,
                )
            if new_event is None:
                new_event = non_unit_evts[0]
        elif evts:
            new_event = evts[0]

        return new_analog, new_event

    def apply_channel_roles_preprocessing(self, roles=None) -> bool:
        if roles is not None:
            norm_roles = normalize_channel_roles(roles)
            role_dicts = [r.to_dict() for r in norm_roles]
            self.channel_roles = role_dicts
            self.live_config["channel_roles"] = [dict(r) for r in role_dicts]
            self._sync_roles_to_live_config(norm_roles)
            self.onChannelRolesChanged.emit()
            self.onLiveConfigChanged.emit()
        else:
            norm_roles = normalize_channel_roles(self.channel_roles)

        if self.segment is None:
            return False

        if self.is_streaming or self.has_live_session():
            # Handle live session buffers and non-MultiAnalogSignal channels gracefully
            has_regular_signals = any(
                not isinstance(s, MultiAnalogSignal)
                for s in (self.segment.analogsignals or [])
            )
            if not self.is_streaming and has_regular_signals:
                # Apply standard preprocessing only to non-MultiAnalogSignal signals if present
                regular_seg = neo.Segment()
                regular_seg.analogsignals = [
                    s
                    for s in self.segment.analogsignals
                    if not isinstance(s, MultiAnalogSignal)
                ]
                regular_seg.events = list(self.segment.events)
                preprocess_segment_with_roles(regular_seg, norm_roles, in_place=True)
                multi_sigs = [
                    s
                    for s in self.segment.analogsignals
                    if isinstance(s, MultiAnalogSignal)
                ]
                self.segment.analogsignals = multi_sigs + list(regular_seg.analogsignals)
                self.segment.events = list(regular_seg.events)

            # Map/alias live channel buffers (e.g. CH0 / CH1 -> rd.0)
            for role in norm_roles:
                if not role.enabled or role.probe_type != TypeID.ANALOG:
                    continue
                src_candidates = [role.source]
                m = None
                import re

                m = re.fullmatch(r"ch(\d+)", role.source.strip(), re.IGNORECASE)
                if m and int(m.group(1)) >= 1:
                    src_candidates.append(f"CH{int(m.group(1)) - 1}")
                matched_key = next(
                    (c for c in src_candidates if c in self._live_channel_signals),
                    None,
                )
                if matched_key is None:
                    matched_key = next(
                        (c for c in src_candidates if c in self._live_aux_erp_buffers),
                        None,
                    )
                if matched_key is not None and role.name and role.name != matched_key:
                    if matched_key in self._live_channel_signals:
                        sig = self._live_channel_signals[matched_key]
                        self._live_channel_signals[role.name] = sig
                        sig.annotations.setdefault("raw_channel_name", matched_key)
                        sig.annotations["channel_purpose"] = role.purpose
                    if matched_key in self._live_aux_erp_buffers:
                        self._live_aux_erp_buffers[role.name] = (
                            self._live_aux_erp_buffers[matched_key]
                        )
                    if (
                        role.name == "rd.0"
                        and matched_key in self._live_channel_signals
                    ):
                        self.analog_signal = self._live_channel_signals[role.name]
                        if role.name in self._live_aux_erp_buffers:
                            self._live_erp_buffer = self._live_aux_erp_buffers[role.name]

            self.segment.annotations["channel_roles"] = [dict(r) for r in self.channel_roles]
            self.onChannelGroupsChanged.emit()
            self.onLoadNewFile.emit()
            self.onEventsUpdated.emit()
            return True

        preprocess_segment_with_roles(self.segment, norm_roles, in_place=True)
        self.segment.annotations["channel_roles"] = [dict(r) for r in self.channel_roles]

        new_analog, new_event = self._select_role_signals(self.segment, norm_roles)
        self.set_data(
            analog_signal=new_analog,
            event_signal=new_event,
            spike_groups=self.spike_groups,
        )
        self.onChannelGroupsChanged.emit()
        self.onEventsUpdated.emit()
        return True

    def reload_with_channel_roles(
        self, roles=None, record_no: Optional[int] = None
    ) -> bool:
        if roles is not None:
            norm_roles = normalize_channel_roles(roles)
            role_dicts = [r.to_dict() for r in norm_roles]
            self.channel_roles = role_dicts
            self.live_config["channel_roles"] = [dict(r) for r in role_dicts]
            self._sync_roles_to_live_config(norm_roles)
            self.onChannelRolesChanged.emit()
            self.onLiveConfigChanged.emit()

        if record_no is not None:
            if self._last_aptrack_config is None:
                self._last_aptrack_config = {}
            self._last_aptrack_config["record"] = int(record_no)

        if (
            getattr(self, "_loaded_file_type", None) == "APTrack"
            and getattr(self, "_loaded_file_path", None)
            and Path(self._loaded_file_path).exists()
            and not self.is_streaming
        ):
            preserved_groups = list(self.spike_groups or [])
            cfg = dict(self._last_aptrack_config or {"record": 1})
            cfg["roles"] = [dict(r) for r in self.channel_roles]
            self.loadFile(
                self._loaded_file_path,
                type="APTrack",
                blocking=True,
                aptrack_config=cfg,
            )
            if preserved_groups:
                existing_names = {
                    getattr(sg.event, "name", None) for sg in (self.spike_groups or [])
                }
                merged = list(self.spike_groups or [])
                has_non_empty_loaded = any(
                    len(getattr(sg, "event", [])) > 0 for sg in merged
                )
                if not has_non_empty_loaded:
                    merged = preserved_groups
                else:
                    for sg in preserved_groups:
                        sg_name = getattr(getattr(sg, "event", None), "name", None)
                        if sg_name and sg_name not in existing_names and len(sg.event) > 0:
                            merged.append(sg)
                            existing_names.add(sg_name)
                if merged is not self.spike_groups:
                    self.set_data(spike_groups=merged)
            return True

        return self.apply_channel_roles_preprocessing()

    def get_effective_refresh_rate_ms(
        self, view_name: str = None, weight: str = "normal"
    ) -> int:
        if hasattr(self, "frame_scheduler") and self.frame_scheduler is not None:
            return self.frame_scheduler.get_effective_interval_ms(
                view_name=view_name, weight=weight
            )
        return getattr(self, "ui_refresh_rate_ms", 33)

    def add_channel_group(self, name: str, channels: List[str]):
        """Add or update a named channel group."""
        if not name or not channels:
            return
        cleaned = [str(c).strip() for c in channels if str(c).strip()]
        self.channel_groups[str(name).strip()] = cleaned
        self.onChannelGroupsChanged.emit()

    def remove_channel_group(self, name: str):
        """Remove a named channel group."""
        if name in self.channel_groups:
            del self.channel_groups[name]
            self.onChannelGroupsChanged.emit()

    def get_channel_group(self, name: str) -> List[str]:
        """Return the channels in a named group."""
        return list(self.channel_groups.get(name, []))

    def get_all_channel_groups(self) -> dict:
        """Return all user-defined and unit-derived channel groups."""
        groups = dict(self.channel_groups)
        for i, sg in enumerate(self.spike_groups):
            u_chans = self.get_unit_channels(i)
            if u_chans:
                u_name = getattr(sg.event, "name", f"unit_{i}") or f"unit_{i}"
                groups[f"Unit {i} ({u_name})"] = u_chans
        return groups

    def get_unit_channels(self, unit_idx: int) -> List[str]:
        """Return the primary and neighbor channels associated with the specified unit."""
        if unit_idx < 0 or unit_idx >= len(self.spike_groups):
            return []
        sg = self.spike_groups[unit_idx]
        primary = getattr(sg, "channel", "")
        if (
            not primary
            and hasattr(sg, "event")
            and getattr(sg.event, "annotations", None)
        ):
            primary = sg.event.annotations.get("channel", "")
        neighbors = getattr(sg, "neighbor_channels", []) or []
        if (
            not neighbors
            and hasattr(sg, "event")
            and getattr(sg.event, "annotations", None)
        ):
            neighbors = sg.event.annotations.get("neighbor_channels", []) or []

        chans = []
        if primary:
            chans.append(str(primary))
        for c in neighbors:
            c_str = str(c).strip()
            if c_str and c_str not in chans:
                chans.append(c_str)
        if (
            not chans
            and self.analog_signal is not None
            and getattr(self.analog_signal, "name", "")
        ):
            chans.append(self.analog_signal.name)
        return chans

    def get_unit_detected_channels(
        self,
        unit_idx: int,
        last_n: int = 50,
        window_ms: float = 15.0,
    ) -> List[str]:
        """Return dynamically detected channels for a unit based on its spatiotemporal footprint (amplitude & SNR)."""
        if unit_idx < 0 or unit_idx >= len(self.spike_groups):
            return []
        sg = self.spike_groups[unit_idx]
        try:
            from .dsp.footprint import (
                compute_unit_footprint_from_state,
                auto_detect_channels_from_footprint,
            )

            fp = compute_unit_footprint_from_state(
                self,
                sg,
                last_n=last_n,
                window_ms=window_ms,
                align_peaks=True,
                include_latest_sweep=False,
            )
            if fp is not None:
                res = auto_detect_channels_from_footprint(fp)
                if res is not None:
                    primary, neighbors = res
                    detected = [primary] + [c for c in neighbors if c != primary]
                    if detected:
                        return detected
                det_list = fp.get("detected_channels", [])
                if det_list:
                    return list(det_list)
        except Exception:
            pass
        return self.get_unit_channels(unit_idx)

    def auto_assign_unit_detected_channels(self, unit_idx: int) -> List[str]:
        """Detects the unit's active footprint channels and assigns them as its configured primary + neighbor channels."""
        detected = self.get_unit_detected_channels(unit_idx)
        if detected:
            self.set_unit_channels(unit_idx, detected)
        return detected

    def emit_unit_params_changed(self, unit_idx: int = -1):
        """Emit onUnitParamsChanged for a specific unit or -1 for all units."""
        self.onUnitParamsChanged.emit(unit_idx)

    def set_unit_channels(self, unit_idx: int, channels: List[str]):
        """Set the primary and neighbor channels for a unit."""
        if unit_idx < 0 or unit_idx >= len(self.spike_groups) or not channels:
            return
        sg = self.spike_groups[unit_idx]
        cleaned = [str(c).strip() for c in channels if str(c).strip()]
        if not cleaned:
            return
        primary = cleaned[0]
        neighbors = [c for c in cleaned[1:] if c != primary]
        sg.channel = primary
        sg.neighbor_channels = neighbors
        if hasattr(sg, "event") and sg.event is not None:
            if not hasattr(sg.event, "annotations"):
                sg.event.annotations = {}
            sg.event.annotations["channel"] = primary
            sg.event.annotations["neighbor_channels"] = neighbors
        self.onUnitGroupChange.emit()
        self.onUnitParamsChanged.emit(unit_idx)
        self.onChannelGroupsChanged.emit()

    def shift_channels(self, channels: List[str], step: int = 1) -> List[str]:
        """
        Shifts a list of channels forward or backward along available channels in the recording.
        Preserves relative offsets and channel group size.
        """
        avail = self.get_available_channels()
        if not avail or not channels:
            return list(channels)

        chan_to_idx = {name: i for i, name in enumerate(avail)}
        valid_indices = [chan_to_idx[c] for c in channels if c in chan_to_idx]
        if not valid_indices:
            return list(channels)

        min_idx = min(valid_indices)
        max_idx = max(valid_indices)
        span = max_idx - min_idx

        new_min = min_idx + step
        if new_min < 0:
            new_min = 0
        if new_min + span >= len(avail):
            new_min = max(0, len(avail) - 1 - span)

        shift_delta = new_min - min_idx
        new_channels = []
        for c in channels:
            if c in chan_to_idx:
                old_i = chan_to_idx[c]
                new_i = max(0, min(len(avail) - 1, old_i + shift_delta))
                new_channels.append(avail[new_i])
            else:
                new_channels.append(c)

        return new_channels

    def shift_unit_channels(self, unit_idx: int, step: int = 1) -> List[str]:
        """Shifts the entire channel group of a unit along available recording channels by step."""
        if unit_idx < 0 or unit_idx >= len(self.spike_groups):
            return []
        current = self.get_unit_channels(unit_idx)
        shifted = self.shift_channels(current, step=step)
        if shifted:
            self.set_unit_channels(unit_idx, shifted)
        return shifted

    def get_unit_color(self, unit_idx: int) -> Tuple[int, int, int]:
        """Return consistent RGB tuple for a unit index."""
        try:
            idx = int(unit_idx)
        except (TypeError, ValueError):
            return UNIT_COLORS[0]
        if idx < 0:
            return UNIT_COLORS[0]
        return UNIT_COLORS[idx % len(UNIT_COLORS)]

    def get_unit_hex_color(self, unit_idx: int) -> str:
        """Return consistent #rrggbb hex color string for a unit index."""
        r, g, b = self.get_unit_color(unit_idx)
        return f"#{r:02x}{g:02x}{b:02x}"

    def get_unit_health(self, unit_idx: Optional[int] = None) -> Dict[str, Any]:
        """Return tracking health metrics for ``unit_idx`` (defaults to ``cur_spike_group``)."""
        if unit_idx is None:
            unit_idx = self.cur_spike_group
        sg = self.getUnitGroup(unit_idx)
        if sg is None:
            return {
                "total_spikes": 0,
                "recent_hits": 0,
                "recent_window": 0,
                "hit_rate": 0.0,
                "consecutive_misses": 0,
                "last_latency_ms": None,
                "latency_shift_ms": 0.0,
                "status": "empty",
            }
        return sg.get_tracking_health(self.event_signal, self.stimno)

    def get_shared_channel_set(self, set_name: str = "Shared Group A") -> dict:
        """Returns the dictionary configuration for a shared channel set."""
        if not set_name:
            set_name = "Shared Group A"
        if set_name not in self.shared_channel_sets:
            self.shared_channel_sets[set_name] = {
                "channels": [],
                "group_name": "",
                "unit_idx": None,
                "multichannel": True,
                "visual_settings": self._default_visual_settings(),
            }
        entry = self.shared_channel_sets[set_name]
        if "visual_settings" not in entry or not isinstance(entry["visual_settings"], dict):
            entry["visual_settings"] = self._default_visual_settings()
        s = dict(entry)
        s["visual_settings"] = dict(entry["visual_settings"])
        if not s.get("channels"):
            avail = self.get_available_channels()
            s["channels"] = avail[:4] if len(avail) >= 4 else avail
        return s

    def set_shared_channels(
        self,
        set_name: str,
        channels: List[str],
        group_name: str = "",
        unit_idx: Optional[int] = None,
        multichannel: bool = True,
    ):
        """Updates the channel configuration for a shared viewer set and notifies all listening viewers."""
        if not set_name:
            set_name = "Shared Group A"
        cleaned = [str(c).strip() for c in channels if str(c).strip()]
        existing_vis = (
            dict(self.shared_channel_sets.get(set_name, {}).get("visual_settings") or {})
            or self._default_visual_settings()
        )
        self.shared_channel_sets[set_name] = {
            "channels": cleaned,
            "group_name": group_name,
            "unit_idx": unit_idx,
            "multichannel": multichannel,
            "visual_settings": existing_vis,
        }
        self.onSharedChannelSetChanged.emit(set_name)

    def get_shared_visual_settings(self, set_name: str = "Shared Group A") -> Dict[str, Any]:
        """Return visual settings dictionary for the given shared channel set."""
        if not set_name:
            set_name = "Shared Group A"
        s = self.get_shared_channel_set(set_name)
        vis = dict(self._default_visual_settings())
        vis.update(s.get("visual_settings") or {})
        return vis

    def update_shared_visual_settings(self, set_name: str, **kwargs) -> Dict[str, Any]:
        """Update visual settings for ``set_name`` and emit ``onSharedVisualSettingsChanged``."""
        if not set_name:
            set_name = "Shared Group A"
        self.get_shared_channel_set(set_name)
        vis = self.shared_channel_sets[set_name].setdefault(
            "visual_settings", self._default_visual_settings()
        )
        vis.update(kwargs)
        updated = dict(vis)
        self.onSharedVisualSettingsChanged.emit(set_name, updated)
        return updated

    def add_protocol_marker(
        self,
        label: str,
        stimno: Optional[int] = None,
        timestamp_s: Optional[float] = None,
        category: str = "protocol",
    ) -> Dict[str, Any]:
        """Record a protocol marker at ``stimno`` / ``timestamp_s`` and sync ``log.other``."""
        if stimno is None:
            stimno = int(getattr(self, "stimno", 0) or 0)
        else:
            stimno = int(stimno)

        if timestamp_s is None:
            if (
                self.event_signal is not None
                and len(self.event_signal) > 0
                and 0 <= stimno < len(self.event_signal)
            ):
                ev_t = self.event_signal[stimno]
                timestamp_s = float(
                    ev_t.rescale(pq.s).magnitude if hasattr(ev_t, "rescale") else ev_t
                )
            else:
                timestamp_s = 0.0
        else:
            timestamp_s = float(timestamp_s)

        marker = {
            "stimno": int(stimno),
            "timestamp_s": float(timestamp_s),
            "label": str(label),
            "category": str(category),
        }
        self.protocol_markers.append(marker)

        if self.segment is not None:
            existing_idx = None
            for i, ev in enumerate(self.segment.events):
                if getattr(ev, "name", None) == "log.other":
                    existing_idx = i
                    break
            if existing_idx is not None:
                ev_other = self.segment.events[existing_idx]
                old_times = (
                    np.asarray(ev_other.rescale(pq.s).magnitude, dtype=np.float64).ravel()
                    if hasattr(ev_other, "rescale")
                    else np.asarray(ev_other, dtype=np.float64).ravel()
                )
                old_msgs = list(
                    (getattr(ev_other, "array_annotations", None) or {}).get(
                        "message", [""] * len(old_times)
                    )
                )
                new_times = np.append(old_times, float(timestamp_s)) * pq.s
                new_msgs = np.array([str(m) for m in old_msgs] + [str(label)])
                updated_ev = neo.Event(
                    new_times,
                    name="log.other",
                    units=pq.s,
                    array_annotations={"message": new_msgs},
                )
                self.segment.events[existing_idx] = updated_ev
            else:
                new_ev = neo.Event(
                    np.array([float(timestamp_s)], dtype=np.float64) * pq.s,
                    name="log.other",
                    units=pq.s,
                    array_annotations={"message": np.array([str(label)])},
                )
                self.segment.events.append(new_ev)

        self.onProtocolMarkersChanged.emit()
        self.onEventsUpdated.emit()
        return dict(marker)

    def get_protocol_markers(self) -> List[Dict[str, Any]]:
        """Return protocol markers, populating from ``segment`` ``log.other`` if empty."""
        if not self.protocol_markers and self.segment is not None:
            ev_other = next(
                (e for e in getattr(self.segment, "events", []) if getattr(e, "name", None) == "log.other"),
                None,
            )
            if ev_other is not None and len(ev_other) > 0:
                times = (
                    np.asarray(ev_other.rescale(pq.s).magnitude, dtype=np.float64).ravel()
                    if hasattr(ev_other, "rescale")
                    else np.asarray(ev_other, dtype=np.float64).ravel()
                )
                msgs = list(
                    (getattr(ev_other, "array_annotations", None) or {}).get(
                        "message", [""] * len(times)
                    )
                )
                stim_times = None
                if self.event_signal is not None and len(self.event_signal) > 0:
                    stim_times = (
                        np.asarray(self.event_signal.rescale(pq.s).magnitude, dtype=np.float64).ravel()
                        if hasattr(self.event_signal, "rescale")
                        else np.asarray(self.event_signal, dtype=np.float64).ravel()
                    )
                for idx, t in enumerate(times):
                    msg = str(msgs[idx]) if idx < len(msgs) else ""
                    if stim_times is not None and len(stim_times) > 0:
                        sw = int(np.clip(np.searchsorted(stim_times, t, side="right") - 1, 0, len(stim_times) - 1))
                    else:
                        sw = 0
                    self.protocol_markers.append(
                        {
                            "stimno": sw,
                            "timestamp_s": float(t),
                            "label": msg,
                            "category": "protocol",
                        }
                    )
        return [dict(m) for m in self.protocol_markers]

    def shift_shared_channels(self, set_name: str, step: int = 1) -> List[str]:
        """Shifts the active channels of a shared channel set forward or backward along recording channels."""
        current_set = self.get_shared_channel_set(set_name)
        current_channels = current_set.get("channels", [])
        if not current_channels:
            return []
        shifted = self.shift_channels(current_channels, step=step)
        if shifted:
            self.set_shared_channels(
                set_name=set_name,
                channels=shifted,
                group_name=f"Shifted ({len(shifted)} ch)",
                unit_idx=current_set.get("unit_idx"),
                multichannel=current_set.get("multichannel", True),
            )
        return shifted

    def cycle_shared_group(self, set_name: str, direction: int = 1) -> str:
        """Cycles the shared channel set to the next or previous saved channel group."""
        all_groups = self.get_all_channel_groups()
        if not all_groups:
            return ""
        group_names = list(all_groups.keys())
        current_set = self.get_shared_channel_set(set_name)
        current_group = current_set.get("group_name", "")

        curr_i = (
            group_names.index(current_group) if current_group in group_names else -1
        )
        next_i = (curr_i + direction) % len(group_names)
        target_name = group_names[next_i]
        target_chans = all_groups[target_name]
        self.set_shared_channels(
            set_name=set_name,
            channels=target_chans,
            group_name=target_name,
            unit_idx=current_set.get("unit_idx"),
            multichannel=current_set.get("multichannel", True),
        )
        return target_name

    def auto_create_channel_groups(
        self, group_size: int = 4, prefix: str = "Group"
    ) -> dict:
        """Splits all available channels into consecutive groups of group_size."""
        avail = self.get_available_channels()
        if not avail or group_size <= 0:
            return {}
        created = {}
        for idx, start_i in enumerate(range(0, len(avail), group_size)):
            chunk = avail[start_i : start_i + group_size]
            g_name = f"{prefix} {idx + 1} ({len(chunk)} ch: {chunk[0]}..{chunk[-1]})"
            self.add_channel_group(g_name, chunk)
            created[g_name] = chunk
        return created

    def set_hide_view_top_bars(self, hide: bool):
        hide = bool(hide)
        if getattr(self, "hide_view_top_bars", False) != hide:
            self.hide_view_top_bars = hide
            self.onHideViewTopBarsChanged.emit(self.hide_view_top_bars)

    def get_hide_view_top_bars(self) -> bool:
        return getattr(self, "hide_view_top_bars", False)

    def get_multichannel_erp(
        self, channel_names: List[str] = None, combine_mode: str = "mean"
    ) -> Optional[np.ndarray]:
        """
        Retrieves and integrates ERP across multiple channels.
        combine_mode can be 'mean', 'sum', or 'stack'.
        """
        if not channel_names:
            erp = self.get_erp()
            return erp.magnitude if hasattr(erp, "magnitude") else erp

        erps = []
        for ch in channel_names:
            e = self.get_channel_erp(ch)
            if e is not None and len(e) > 0:
                e_mag = e.magnitude if hasattr(e, "magnitude") else np.asarray(e)
                erps.append(e_mag)

        if not erps:
            erp = self.get_erp()
            return erp.magnitude if hasattr(erp, "magnitude") else erp

        min_sweeps = min(e.shape[0] for e in erps)
        min_samples = min(e.shape[1] for e in erps)
        aligned = [e[:min_sweeps, :min_samples] for e in erps]

        if combine_mode == "mean":
            return np.mean(aligned, axis=0)
        elif combine_mode == "sum":
            return np.sum(aligned, axis=0)
        elif combine_mode == "stack":
            return np.stack(aligned, axis=0)
        return aligned[0]

    def get_available_channels(self) -> List[str]:
        if (
            self.is_streaming or getattr(self, "_live_aux_erp_buffers", None)
        ) and self._live_aux_erp_buffers:
            channels = []
            if self.analog_signal is not None and getattr(
                self.analog_signal, "name", None
            ):
                channels.append(self.analog_signal.name)
            for k in self._live_aux_erp_buffers.keys():
                if k not in channels:
                    channels.append(k)
            return channels
        elif self.segment is not None and len(self.segment.analogsignals) > 0:
            return [sig.name for sig in self.segment.analogsignals]
        else:
            return (
                [self.analog_signal.name]
                if self.analog_signal is not None
                and getattr(self.analog_signal, "name", None)
                else []
            )

    def get_channel_erp(self, channel_name: str = None):
        if channel_name is None or (
            self.analog_signal is not None and channel_name == self.analog_signal.name
        ):
            return self.get_erp()
        if (
            hasattr(self, "_live_aux_erp_buffers")
            and self._live_aux_erp_buffers
            and channel_name in self._live_aux_erp_buffers
        ):
            buf = self._live_aux_erp_buffers[channel_name]
            return (
                buf.data
                if len(buf) > 0
                else np.zeros((0, buf.window_samples)) * buf.units
            )
        if self.segment is not None:
            for sig in self.segment.analogsignals:
                if sig.name == channel_name:
                    return self.get_erp(signal=sig, event_signal=self.event_signal)
        return None

    def get_channel_signal(self, channel_name: str = None):
        if channel_name is None:
            return self.analog_signal
        if (
            hasattr(self, "_live_channel_signals")
            and self._live_channel_signals
            and channel_name in self._live_channel_signals
        ):
            return self._live_channel_signals[channel_name]
        if self.segment is not None:
            for sig in self.segment.analogsignals:
                if sig.name == channel_name:
                    return sig
        return None

    def save_undo(self, func):
        self.undo_queue.append(func)
        if len(self.undo_queue) > self.MAX_UNDO:
            self.undo_queue.pop(0)

    def undo(self):
        if len(self.undo_queue) == 0:
            return
        func = self.undo_queue.pop()
        func()

    @property
    def analog_signal_erp(self):
        return self.get_erp()

    @Slot(int)
    def setUnitGroup(self, unitid: int):
        if unitid > len(self.spike_groups):
            return
        if self.active_candidate_id is not None:
            self.active_candidate_id = None
            self.onCandidateSelectionChanged.emit(-1)
        self.cur_spike_group = unitid
        if 0 <= unitid < len(self.spike_groups):
            sg = self.spike_groups[unitid]
            if (
                getattr(sg, "channel", "") != ""
                and getattr(self, "auto_switch_unit_channel", False)
                and sg.channel != getattr(self.analog_signal, "name", "")
            ):
                self.set_active_channel(sg.channel)
        self.onUnitGroupChange.emit()
        self.onActiveUnitChanged.emit(self.cur_spike_group)

    @Slot(int, int, int)
    @Slot(int, int)
    @Slot(int)
    @Slot()
    def setUnit(
        self,
        latency,
        stimno=None,
        unit_idx: Optional[int] = None,
        unitid: Optional[int] = None,
        save_undo: bool = True,
    ):
        if unitid is not None:
            unit_idx = unitid
        if unit_idx is None:
            unit_idx = self.cur_spike_group
        if unit_idx < 0 or unit_idx >= len(self.spike_groups):
            return
        if stimno is None:
            stimno = self.stimno
        if (
            self.event_signal is None
            or len(self.event_signal) == 0
            or stimno < 0
            or stimno >= len(self.event_signal)
        ):
            return

        sg = self.spike_groups[unit_idx]
        evt = sg.event

        if save_undo:

            def undo(
                x=unit_idx,
                evt=sg.event.copy(),
                stimno=stimno,
                self=self,
            ):
                evt.sort()
                self.spike_groups[x].event = evt
                self.update_idx_arrs()
                self.onUnitChange.emit(stimno)
                self.onNewSpikeEvents.emit(x)

            self.save_undo(undo)

        sr_val = (
            float(self.sampling_rate.rescale(pq.Hz).magnitude)
            if hasattr(self.sampling_rate, "rescale")
            else float(self.sampling_rate or (self.analog_signal.sampling_rate.rescale(pq.Hz).magnitude if self.analog_signal is not None and hasattr(self.analog_signal.sampling_rate, "rescale") else 30000.0))
        )
        pre_samples = int(round(self.pre_offset_s * sr_val)) if self.pre_offset_s > 0 else 0

        if len(evt) > 0:
            sweep_start = self.event_signal[stimno] - (self.pre_offset_s * pq.s)
            sweep_end = (
                (self.event_signal[stimno + 1] - (self.pre_offset_s * pq.s))
                if stimno < len(self.event_signal) - 1
                else (self.event_signal[stimno] + self.window_size)
            )
            to_keep = (evt < sweep_start) | (evt >= sweep_end)
            sg.event = evt[to_keep]
            evt = evt[to_keep]

        if sg.idx_arr is None:
            sg.idx_arr = []
        while len(sg.idx_arr) <= stimno:
            sg.idx_arr.append(None)

        if latency is None:
            sg.idx_arr[stimno] = None
        else:
            if latency < 0 and latency not in (-1, -2):
                latency = 0
            if (
                latency == -1
            ):  # if latency is -1, we want to find the previous spike and use that as the latency
                prv_lat = np.nan
                for attempts in range(3):
                    prev_event_time = self.event_signal.times[stimno - 1 - attempts]
                    prv_lat = sg.get_latencies(
                        np.array([prev_event_time]) * prev_event_time.units
                    )[0].rescale("s")
                    if prv_lat != np.nan:
                        latency = int(
                            (prv_lat * sr_val).magnitude if hasattr(prv_lat, "magnitude") else (prv_lat * sr_val)
                        ) + pre_samples
                        break
            elif (
                latency == -2
            ):  # if latency is -2, we want to find the next spike and use that as the latency
                next_event_time = self.event_signal.times[stimno + 1]
                next_lat = sg.get_latencies(
                    np.array([next_event_time]) * next_event_time.units
                )[0].rescale("s")
                if next_lat != np.nan:
                    latency = int(
                        (next_lat * sr_val).magnitude if hasattr(next_lat, "magnitude") else (next_lat * sr_val)
                    ) + pre_samples

            sg.idx_arr[stimno] = (
                latency,
                1,
            )
            rel_samples = latency - pre_samples
            evt2 = evt.merge(
                Event(
                    (((rel_samples / sr_val) * pq.s).rescale(pq.s))
                    + self.event_signal[[stimno]],
                )
            )
            evt2.name = evt.name
            sg.event = evt2

            # When annotating a spike on a unit, if sg.channel == "" and self.analog_signal is not None:
            if (
                getattr(sg, "channel", "") == ""
                and self.analog_signal is not None
                and getattr(self.analog_signal, "name", "")
            ):
                sg.channel = self.analog_signal.name
                sg.event.annotations["channel"] = self.analog_signal.name

        sg.event.sort()  # TODO: not sure if this breaks the array annotations.

        # Emit onUnitAnnotated with unit details
        lat_samples = int(latency) if latency is not None and latency >= 0 else None
        lat_ms = (
            float((latency - pre_samples) / sr_val * 1000.0)
            if lat_samples is not None
            else None
        )
        ts_s = (
            float(
                (
                    self.event_signal[stimno]
                    + (((latency - pre_samples) / self.analog_signal.sampling_rate) * pq.s if not hasattr(self.analog_signal.sampling_rate, "units") else ((latency - pre_samples) / self.analog_signal.sampling_rate))
                )
                .rescale(pq.s)
                .magnitude
            )
            if lat_samples is not None
            and self.analog_signal is not None
            and self.event_signal is not None
            and stimno < len(self.event_signal)
            else None
        )
        self.onUnitAnnotated.emit(
            {
                "unit": sg.event.name,
                "unit_idx": int(unit_idx),
                "stimno": int(stimno),
                "latency_samples": lat_samples,
                "latency_ms": lat_ms,
                "timestamp_s": ts_s,
                "fibre_type": str(sg.fibre_type),
                "automated": not save_undo,
            }
        )

        if latency is not None:
            latest_idx = self.get_latest_event_idx()
            if latest_idx is None or stimno >= latest_idx:
                sg._consecutive_misses = 0

        self.onUnitChange.emit(stimno)
        self.onNewSpikeEvents.emit(unit_idx)
        self.onUnitSpikeChanged.emit(unit_idx, stimno)

    def _ensure_template_matcher(self, unit_idx: int, rebuild: bool = False):
        """Build (or return) the multi-channel template matcher for a unit."""
        if unit_idx < 0 or unit_idx >= len(self.spike_groups):
            return None
        sg = self.spike_groups[unit_idx]
        matcher = getattr(sg, "_template_matcher", None)
        if not rebuild and matcher is not None and getattr(matcher, "templates", None):
            return matcher

        matcher = MultiChannelTemplateMatcher()
        ch = (
            getattr(sg, "tracking_channel", "")
            or getattr(sg, "channel", "")
            or (self.analog_signal.name if self.analog_signal is not None else "")
        )
        channels = self.get_unit_channels(unit_idx) or ([ch] if ch else [])
        if channels and matcher.build_template(self, sg, channels=channels):
            sg._template_matcher = matcher
            return matcher
        sg._template_matcher = None
        return None

    def rebuild_unit_template(self, unit_idx: int) -> bool:
        """Force a fresh convolution template for a unit (used by UI actions)."""
        matcher = self._ensure_template_matcher(unit_idx, rebuild=True)
        if hasattr(self, "onUnitParamsChanged"):
            self.onUnitParamsChanged.emit(int(unit_idx))
        return matcher is not None

    def apply_tracked_samples(
        self,
        unit_idx: int,
        samples: Dict[int, int],
        save_undo: bool = True,
    ) -> int:
        """Apply a batch of ``{stim_index: sample}`` detections to one unit.

        This is the single write path for live batches and ad-hoc gap filling:
        it updates ``idx_arr`` and the event list together, emits the scoped
        unit signals once, and keeps the per-spike undo stack to one entry.
        Returns the number of samples applied.
        """
        if (
            not samples
            or self.event_signal is None
            or len(self.event_signal) == 0
            or unit_idx < 0
            or unit_idx >= len(self.spike_groups)
        ):
            return 0

        n_sweeps = len(self.event_signal)
        keys = sorted(
            int(k) for k, v in samples.items() if v is not None and 0 <= int(k) < n_sweeps
        )
        if not keys:
            return 0

        sg = self.spike_groups[unit_idx]
        sr = float(
            self.sampling_rate
            if self.sampling_rate is not None
            else (
                self.analog_signal.sampling_rate.rescale(pq.Hz).magnitude
                if self.analog_signal is not None
                else 30000.0
            )
        )

        if save_undo:
            old_evt = sg.event.copy() if sg.event is not None else Event()
            old_idx = list(sg.idx_arr) if sg.idx_arr is not None else None

            def undo(x=unit_idx, evt=old_evt, idx_arr=old_idx, stimno=self.stimno):
                self.spike_groups[x].event = evt
                self.spike_groups[x].idx_arr = idx_arr
                self.onUnitChange.emit(stimno)
                self.onNewSpikeEvents.emit(x)

            self.save_undo(undo)

        if sg.idx_arr is None:
            sg.idx_arr = []
        while len(sg.idx_arr) <= keys[-1]:
            sg.idx_arr.append(None)
        for stim in keys:
            sg.idx_arr[stim] = (int(samples[stim]), 1)

        stim_s = np.asarray(
            self.event_signal.rescale(pq.s).magnitude
            if hasattr(self.event_signal, "rescale")
            else self.event_signal,
            dtype=np.float64,
        ).ravel()

        ev = sg.event if sg.event is not None else Event()
        ev_s = np.asarray(
            ev.rescale(pq.s).magnitude if hasattr(ev, "rescale") else ev,
            dtype=np.float64,
        ).ravel()

        pre_offset_s = float(getattr(self, "pre_offset_s", 0.0) or 0.0)
        pre_samples = int(round(pre_offset_s * sr)) if pre_offset_s > 0 else 0

        if ev_s.size > 0:
            stim_starts = stim_s - pre_offset_s
            ev_stim_idx = np.searchsorted(stim_starts, ev_s, side="right") - 1
            remove = np.zeros(len(stim_s), dtype=bool)
            remove[keys] = True
            keep = np.ones(ev_s.size, dtype=bool)
            valid = (ev_stim_idx >= 0) & (ev_stim_idx < len(stim_s))
            keep[valid] = ~remove[ev_stim_idx[valid]]
            keep_times = ev_s[keep]
        else:
            keep_times = np.empty(0, dtype=np.float64)

        new_times = stim_s[keys] + (
            (np.asarray([samples[k] for k in keys], dtype=np.float64) - pre_samples) / sr
        )
        all_times = np.sort(np.concatenate([keep_times, new_times]))
        new_event = Event(
            all_times * pq.s,
            name=getattr(ev, "name", f"unit_{unit_idx}"),
            units=pq.s,
            **getattr(ev, "annotations", {}),
        )
        sg.event = new_event
        sg._tracked_count = getattr(sg, "_tracked_count", 0) + len(keys)

        if self.segment is not None:
            for i, seg_ev in enumerate(self.segment.events):
                if getattr(seg_ev, "name", None) == getattr(new_event, "name", None):
                    self.segment.events[i] = new_event
                    break

        if (
            getattr(sg, "channel", "") == ""
            and self.analog_signal is not None
            and getattr(self.analog_signal, "name", "")
        ):
            sg.channel = self.analog_signal.name
            sg.event.annotations["channel"] = self.analog_signal.name

        self.onUnitChange.emit(self.stimno)
        self.onNewSpikeEvents.emit(unit_idx)
        self.onUnitSpikeChanged.emit(unit_idx, self.stimno)
        return len(keys)

    def track_unit_range(
        self,
        unit_idx: int,
        start_stim: Optional[int] = None,
        end_stim: Optional[int] = None,
        only_missing: bool = True,
        apply: bool = True,
        save_undo: bool = True,
        seed_stim: Optional[int] = None,
        seed_sample: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[int, int]:
        """Run ad-hoc tracking for one unit and fill missing sweep samples.

        Existing spikes anchor the search; tracking advances causally from each
        anchor until it misses too many sweeps or exhausts ``track_limit``.  The
        next existing spike then restarts tracking, so gaps between sparse
        annotations are filled without re-deriving present spikes.
        """
        if (
            self.event_signal is None
            or len(self.event_signal) == 0
            or unit_idx < 0
            or unit_idx >= len(self.spike_groups)
        ):
            return {}

        n_sweeps = len(self.event_signal)
        start = 0 if start_stim is None else max(0, min(int(start_stim), n_sweeps - 1))
        end = (
            n_sweeps - 1
            if end_stim is None
            else max(0, min(int(end_stim), n_sweeps - 1))
        )
        if end < start:
            return {}

        sg = self.spike_groups[unit_idx]
        ch = (
            getattr(sg, "tracking_channel", "")
            or getattr(sg, "channel", "")
            or (self.analog_signal.name if self.analog_signal is not None else "")
        )
        ch_erp = self.get_channel_erp(ch) if ch else self.get_erp()
        if ch_erp is None:
            ch_erp = self.get_erp()
        if ch_erp is None or len(ch_erp) == 0:
            return {}

        sr = float(
            self.sampling_rate
            if self.sampling_rate is not None
            else (
                self.analog_signal.sampling_rate.rescale(pq.Hz).magnitude
                if self.analog_signal is not None
                else 30000.0
            )
        )
        existing = unit_tracking.sample_map_from_sg(sg, n_sweeps)

        # If filling only part of the recording, seed from the nearest earlier
        # annotated spike so tracking can continue into the requested range.
        if seed_stim is None and start > 0:
            for prev in range(start - 1, -1, -1):
                if existing.get(prev) is not None:
                    seed_stim = prev
                    seed_sample = existing[prev]
                    break

        algo = str(getattr(sg, "tracking_algorithm", "basic") or "basic")
        matcher = sg._template_matcher
        if algo == "template":
            matcher = self._ensure_template_matcher(unit_idx)

        units = getattr(ch_erp, "units", None)
        if units is None and self.analog_signal is not None:
            units = getattr(self.analog_signal, "units", None)

        detections = unit_tracking.track_range(
            ch_erp,
            sr,
            sg,
            start_stim=start,
            end_stim=end,
            existing=existing,
            only_missing=only_missing,
            seed_stim=seed_stim,
            seed_sample=seed_sample,
            params=params,
            template_matcher=matcher,
            waveform_units=units,
        )
        if apply and detections:
            self.apply_tracked_samples(unit_idx, detections, save_undo=save_undo)
        return detections

    def ensure_live_tracking_processor(self):
        """Register the state-owned live tracking processor exactly once.

        Keeping this on ``ViewerState`` means live tracking works even when no
        TrackingView is open, and avoids the previous duplicate paths where a
        view registration and the tracking manager could both run.
        """
        orch = self.live_orchestrator
        owner = "viewerstate_live_tracking"
        if not orch.has_registration(owner, "auto_track"):
            orch.register(
                owner,
                "auto_track",
                self._process_live_tracking_batch,
                self._on_live_tracking_result,
                batching=BatchingMode.BATCH,
            )
        return orch

    def _process_live_tracking_batch(self, batch):
        """Runs on an orchestrator worker for newly arrived sweeps."""
        detections = []
        if self.event_signal is None or not batch.items:
            return detections

        units = [
            (i, sg)
            for i, sg in enumerate(self.spike_groups)
            if getattr(sg, "auto_track", False)
        ]
        if not units:
            return detections

        sr = float(
            self.sampling_rate
            if self.sampling_rate is not None
            else (
                self.analog_signal.sampling_rate.rescale(pq.Hz).magnitude
                if self.analog_signal is not None
                else 30000.0
            )
        )
        win_s = (
            float(self.window_size.rescale(pq.s).magnitude)
            if hasattr(self.window_size, "rescale")
            else float(self.window_size)
        )
        window_samples = max(1, int(round(win_s * sr)))
        units_obj = getattr(self.analog_signal, "units", None)

        existing_maps = {
            ui: unit_tracking.sample_map_from_sg(sg, len(self.event_signal))
            for ui, sg in units
        }
        local_samples: Dict[int, Dict[int, int]] = {ui: {} for ui, _ in units}
        tracked_counts: Dict[int, int] = {ui: 0 for ui, _ in units}

        tracked_stims_by_unit: Dict[int, set] = {ui: set() for ui, _ in units}
        for item in batch.items:
            stimno = int(item.stim_index)
            if stimno < 0 or stimno >= len(self.event_signal):
                continue
            for unit_idx, sg in units:
                if local_samples[unit_idx].get(stimno) is not None:
                    continue
                track_limit = int(getattr(sg, "track_limit", 0))
                if track_limit > 0 and tracked_counts[unit_idx] >= track_limit:
                    continue

                existing = existing_maps[unit_idx].get(stimno)
                if existing is not None:
                    local_samples[unit_idx][stimno] = int(existing)
                    continue

                max_skip = int(getattr(sg, "tracking_max_skips", 1))
                prev_sample = unit_tracking.nearest_previous_sample(
                    stimno, local_samples[unit_idx], max_skip
                )
                if prev_sample is None:
                    prev_sample = unit_tracking.nearest_previous_sample(
                        stimno, existing_maps[unit_idx], max_skip
                    )
                if prev_sample is None:
                    continue

                ch = (
                    getattr(sg, "tracking_channel", "")
                    or getattr(sg, "channel", "")
                    or (self.analog_signal.name if self.analog_signal is not None else "")
                )
                row = item.erp_row(ch, window_samples) if ch else None
                if row is None:
                    channels = list(item.channel_sweeps.keys())
                    if channels:
                        row = item.erp_row(channels[0], window_samples)
                if row is None or row.size == 0:
                    continue

                algo, params = unit_tracking.tracking_params(
                    sg, waveform_units=units_obj
                )
                matcher = getattr(sg, "_template_matcher", None)
                getter = None
                if algo == "template" and matcher is not None:
                    getter = lambda c, it=item: it.erp_row(c, window_samples)

                hit = unit_tracking.detect_sample(
                    sg,
                    row,
                    sr,
                    prev_sample,
                    params=params,
                    algorithm=algo,
                    template_matcher=matcher,
                    multi_channel_getter=getter,
                )
                tracked_stims_by_unit[unit_idx].add(stimno)
                if hit is not None:
                    detections.append((unit_idx, int(hit), stimno))
                    local_samples[unit_idx][stimno] = int(hit)
                    tracked_counts[unit_idx] += 1

        # Update _consecutive_misses for live tracked units when sweeps were evaluated
        latest_batch_stim = max(
            (int(it.stim_index) for it in batch.items if 0 <= int(it.stim_index) < len(self.event_signal)),
            default=None,
        )
        if latest_batch_stim is not None:
            for unit_idx, sg in units:
                if latest_batch_stim in local_samples[unit_idx]:
                    sg._consecutive_misses = 0
                elif latest_batch_stim in tracked_stims_by_unit[unit_idx]:
                    sg._consecutive_misses = int(getattr(sg, "_consecutive_misses", 0)) + 1

        return detections

    def _on_live_tracking_result(self, detections, error):
        """Apply orchestrator detections on the GUI thread."""
        if error:
            return
        if not detections:
            for sg in self.spike_groups:
                if getattr(sg, "auto_track", False):
                    sg._consecutive_misses = int(getattr(sg, "_consecutive_misses", 0)) + 1
            return
        by_unit: Dict[int, Dict[int, int]] = {}
        for unit_idx, sample, stimno in detections:
            by_unit.setdefault(int(unit_idx), {})[int(stimno)] = int(sample)
        for unit_idx, samples in by_unit.items():
            self.apply_tracked_samples(unit_idx, samples, save_undo=False)
            if 0 <= unit_idx < len(self.spike_groups):
                self.spike_groups[unit_idx]._consecutive_misses = 0
        for ui, sg in enumerate(self.spike_groups):
            if getattr(sg, "auto_track", False) and ui not in by_unit:
                sg._consecutive_misses = int(getattr(sg, "_consecutive_misses", 0)) + 1

    @Slot(Event)
    def updateUnit(self, event, merge=False, unitid=None):
        if unitid is None:
            unitid = self.cur_spike_group
        if unitid < 0 or unitid >= len(self.spike_groups):
            return

        def undo(
            x=unitid,
            evt=self.spike_groups[unitid].event,
            stimno=self.stimno,
        ):
            self.spike_groups[x].event = evt
            self.update_idx_arrs()
            self.onUnitChange.emit(stimno)

        self.save_undo(undo)

        old_evt = self.spike_groups[unitid].event.rescale(pq.s)
        new_event = event.rescale(pq.s)
        if merge:
            # update the existing events with the new ones, removing ones within 0.5s of the other

            newEvents = []
            new_evt_idxs = self.event_signal.rescale(pq.s).searchsorted(new_event) - 1
            old_evt_idxs = self.event_signal.rescale(pq.s).searchsorted(old_evt) - 1

            for i, t in enumerate(self.event_signal):
                time_gap = 1 * pq.s  # TODO get from state

                for x in [*new_event[new_evt_idxs == i], *old_evt[old_evt_idxs == i]]:
                    if x <= t + time_gap and x >= t:
                        newEvents.append(x)
                        break

            event = Event(
                np.array(newEvents) * pq.s, **old_evt.annotations, name=old_evt.name
            )
            # TODO: this does not maintain array_annotations.
        self.spike_groups[unitid].event = event
        del self.spike_groups[unitid].idx_arr
        self.update_idx_arrs()
        self.onUnitGroupChange.emit()
        self.onUnitListChanged.emit()

    def track_stimulus_all_units(self, stimno: int) -> List[dict]:
        """Track all auto_track enabled units on their respective channels for stimulus index stimno."""
        results = []
        if self.event_signal is None or stimno < 0 or stimno >= len(self.event_signal):
            return results

        sr = float(
            self.sampling_rate
            if self.sampling_rate is not None
            else (
                self.analog_signal.sampling_rate.rescale(pq.Hz).magnitude
                if self.analog_signal is not None
                else 30000.0
            )
        )

        for unit_idx, sg in enumerate(self.spike_groups):
            if not getattr(sg, "auto_track", False):
                continue

            ch = (
                getattr(sg, "tracking_channel", "")
                or getattr(sg, "channel", "")
                or (self.analog_signal.name if self.analog_signal else "")
            )
            ch_erp = self.get_channel_erp(ch) if ch else self.get_erp()
            if ch_erp is None:
                ch_erp = self.get_erp()

            if ch_erp is None or stimno >= len(ch_erp):
                continue

            # Check track_limit
            track_limit = int(
                getattr(sg, "track_limit", getattr(sg, "tracking_track_limit", 0))
            )
            tracked_count = getattr(sg, "_tracked_count", 0)
            if track_limit > 0 and tracked_count >= track_limit:
                continue

            # Check if unit already has an annotated spike on this sweep
            existing_sample = sg.get_sweep_sample(
                stimno, self.event_signal, sr, self.window_size
            )
            if existing_sample is not None:
                continue

            # Find most recent spike latency in prior sweeps within tracking_max_skips
            max_skips = int(
                getattr(sg, "max_skip", getattr(sg, "tracking_max_skips", 1))
            )
            prev_latency = None
            min_stim = max(0, stimno - 1 - max_skips)
            for prev_s in range(stimno - 1, min_stim - 1, -1):
                prev_sample = sg.get_sweep_sample(
                    prev_s, self.event_signal, sr, self.window_size
                )
                if prev_sample is not None:
                    prev_latency = (prev_sample / sr) * pq.s
                    break
                elif (
                    getattr(sg, "idx_arr", None) is not None
                    and prev_s < len(sg.idx_arr)
                    and sg.idx_arr[prev_s] is not None
                ):
                    prev_samp = sg.idx_arr[prev_s][0]
                    prev_latency = (prev_samp / sr) * pq.s
                    break

            if prev_latency is None:
                continue

            sweep_wf = np.asarray(
                ch_erp[stimno].magnitude
                if hasattr(ch_erp[stimno], "magnitude")
                else ch_erp[stimno],
                dtype=np.float32,
            )
            thresh_mv = float(
                getattr(sg, "threshold", getattr(sg, "tracking_threshold_mv", 0.01))
            )
            win_ms = float(
                getattr(sg, "window_size", getattr(sg, "tracking_window_ms", 5.0))
            )
            algorithm = getattr(sg, "tracking_algorithm", "basic") or "basic"

            detected_sample = None

            # 1. Template matching path
            if algorithm == "template":
                matcher = getattr(sg, "_template_matcher", None)
                if matcher is None:
                    matcher = MultiChannelTemplateMatcher()
                    u_chans = self.get_unit_channels(unit_idx)
                    if not u_chans:
                        u_chans = [ch]
                    if matcher.build_template(self, sg, channels=u_chans):
                        sg._template_matcher = matcher
                    else:
                        matcher = None

                if matcher is not None and matcher.templates:
                    u_chans = self.get_unit_channels(unit_idx) or [ch]
                    scores = matcher.score_sweep(self, stimno, channels=u_chans)
                    if scores is not None and len(scores) > 0:
                        prev_s_val = float(
                            prev_latency.rescale(pq.s).magnitude
                            if hasattr(prev_latency, "rescale")
                            else prev_latency
                        )
                        prev_samp = int(round(prev_s_val * sr))
                        half_w = max(1, int(round((win_ms / 1000.0 / 2.0) * sr)))
                        s_start = max(0, prev_samp - half_w)
                        s_end = min(len(scores), prev_samp + half_w + 1)
                        if s_end > s_start:
                            snip_scores = scores[s_start:s_end]
                            if (
                                len(snip_scores) > 0
                                and np.max(snip_scores) >= thresh_mv
                            ):
                                detected_sample = s_start + int(np.argmax(snip_scores))

            # 2. Check SpikeDetectorRegistry for registered algorithms
            if detected_sample is None and algorithm != "template":
                try:
                    from .dsp.detectors import SpikeDetectorRegistry

                    detector = SpikeDetectorRegistry.get(algorithm)
                except Exception:
                    detector = None

                if detector is not None:
                    params = {
                        **detector.default_params(),
                        "window_ms": float(win_ms),
                        "threshold": float(thresh_mv),
                        "max_skip": int(max_skips),
                        "track_limit": int(track_limit),
                        **getattr(sg, "tracking_params", {}),
                    }
                    detected_sample = detector.detect_sweep(
                        sweep_wf=sweep_wf,
                        sampling_rate=sr,
                        prev_latency=prev_latency,
                        params=params,
                    )

            # 3. Fallback to basic tracking
            if detected_sample is None and algorithm not in ("template",):
                detected_sample = basic_tracking.track_sweep_waveform(
                    sweep_wf,
                    sr,
                    prev_latency,
                    threshold=thresh_mv,
                    window_ms=win_ms,
                )

            consecutive_misses = getattr(sg, "_consecutive_misses", 0)
            if detected_sample is not None:
                sg._consecutive_misses = 0
                sg._tracked_count = getattr(sg, "_tracked_count", 0) + 1
                self.setUnit(
                    detected_sample, stimno=stimno, unit_idx=unit_idx, save_undo=False
                )
                results.append(
                    {
                        "unit_idx": unit_idx,
                        "unit": sg.event.name,
                        "stimno": stimno,
                        "channel": ch,
                        "sample": detected_sample,
                        "latency_ms": (detected_sample / sr) * 1000.0,
                    }
                )
            else:
                sg._consecutive_misses = consecutive_misses + 1
                if sg._consecutive_misses > max_skips:
                    # Exceeded max skip consecutive misses
                    pass

        return results

    def stimno_offset_to_event(self, stimno, offset):
        """
        convert arrays of stimno and offset to Event
        """
        sr = self.sampling_rate if self.sampling_rate is not None else self.analog_signal.sampling_rate
        sr_val = float(sr.rescale(pq.Hz).magnitude) if hasattr(sr, "rescale") else float(sr)
        pre_offset = getattr(self, "pre_offset_s", 0.0)
        pre_samples = int(round(pre_offset * sr_val))
        t = self.event_signal[stimno] + (((offset - pre_samples) / sr_val) * pq.s)
        return Event(t)

    def get_latest_event_idx(self) -> Optional[int]:
        """Returns the index of the latest stimulus sweep / event."""
        if self.event_signal is not None and len(self.event_signal) > 0:
            return len(self.event_signal) - 1
        if self._live_erp_buffer is not None and self._live_erp_buffer.n_sweeps > 0:
            return self._live_erp_buffer.n_sweeps - 1
        return 0 if self.event_signal is not None else None

    def is_following_live_events(self) -> bool:
        """
        Determines if following live new events (i.e. is the idx being kept in sync with latest).
        """
        if not getattr(self, "follow_live", True):
            return False
        latest = self.get_latest_event_idx()
        if latest is None:
            return bool(self.follow_live)
        return self.stimno == latest

    def set_follow_live(self, follow: bool):
        """Enables or disables following live new events. When enabled, syncs to latest index."""
        self.follow_live = bool(follow)
        if self.follow_live:
            latest = self.get_latest_event_idx()
            if latest is not None and self.stimno != latest:
                self.setStimNo(latest, user_set=False)
        self.onFollowLiveChanged.emit(self.follow_live)
        self.onFollowLiveEventsChanged.emit(self.follow_live)

    def set_follow_live_events(self, enable: bool):
        self.set_follow_live(enable)

    @Slot(int)
    def setStimNo(self, stimno: int, user_set: bool = True):
        # If the user manually sets (e.g. with arrow keys), stop tracing the event
        if user_set and getattr(self, "follow_live", False):
            self.follow_live = False
            self.onFollowLiveChanged.emit(False)
            self.onFollowLiveEventsChanged.emit(False)

        if self.event_signal is None or len(self.event_signal) == 0:
            if self.stimno != 0:
                self.stimno = 0
                self.onStimNoChange.emit(0)
            return
        new_stim = max(min(stimno, len(self.event_signal) - 1), 0)
        if self.stimno == new_stim:
            return
        self.stimno = new_stim
        self.onStimNoChange.emit(self.stimno)

    @Slot(str, str)
    def loadFile(self, fname, type="h5", blocking=True, aptrack_config=None, **kwargs):
        """Load a recording file.

        With the default ``blocking=True`` the file is read synchronously (used
        by tests and headless scripts). The GUI passes ``blocking=False`` so the
        file is read on a background thread while a cancelable progress dialog
        keeps the UI responsive; the data is applied on the GUI thread when the
        read completes.
        """
        if not fname:
            return
        if type is None:
            type = "h5"

        app = QApplication.instance()
        if type == "APTrack" and aptrack_config is None and app is not None:
            dialog = APTrackDialog()
            accepted = dialog.exec_() if hasattr(dialog, "exec_") else dialog.exec()
            if not accepted:
                return
            aptrack_config = dialog.get_config()

        if blocking or app is None:
            try:
                result = load_file(fname, type, aptrack_config=aptrack_config)
            except Exception:
                import traceback

                traceback.print_exc()
                self._show_load_error(fname, type)
                return
            self._apply_loaded_file(fname, type, *result)
            return

        self._start_async_load(fname, type, aptrack_config)

    def _apply_loaded_file(
        self, fname, type, data, analog_signal, event_signal, spike_groups
    ):
        self._loaded_file_path = fname
        self._loaded_file_type = type
        self.title = fname
        self.segment = data
        self.cur_spike_group = 0
        self.event_signal = event_signal
        self.analog_signal = analog_signal
        if data is not None and hasattr(data, "annotations"):
            if "aptrack_config" in data.annotations:
                self._last_aptrack_config = dict(data.annotations["aptrack_config"])
            if "channel_roles" in data.annotations:
                self.channel_roles = [
                    r.to_dict()
                    for r in normalize_channel_roles(data.annotations["channel_roles"])
                ]
                self.live_config["channel_roles"] = [dict(r) for r in self.channel_roles]
                self.onChannelRolesChanged.emit()
        self.restore_candidate_store(
            data.annotations.get("spikespy_candidates"), event_signal=event_signal
        )
        self.set_data(analog_signal, event_signal, spike_groups=spike_groups)
        self.onFileLoaded.emit(fname, type)

    def _show_load_error(self, fname, type):
        app = QApplication.instance()
        if app and app.platformName() != "offscreen":
            qmb = QMessageBox(
                QMessageBox.Critical,
                "Sorry :(",
                f'Unable to load "{fname}" using {type}',
            )
            qmb.setModal(True)
            qmb.exec()

    def _start_async_load(self, fname, type, aptrack_config=None):
        app = QApplication.instance()
        worker = getattr(self, "_file_load_worker", None)
        if worker is not None and worker.isRunning():
            QMessageBox.warning(
                app.activeWindow() if app else None,
                "Still loading",
                "A file is already being loaded.\nPlease wait for it to finish.",
            )
            return

        self._file_load_cancelled = False
        self._file_load_progress = self._make_load_progress(fname)
        worker = FileLoadWorker(fname, type, aptrack_config=aptrack_config, parent=self)
        self._file_load_worker = worker
        worker.stage.connect(self._on_load_stage)
        worker.loaded.connect(self._on_async_file_loaded)
        worker.failed.connect(self._on_async_file_failed)
        worker.start()

    def _make_load_progress(self, fname):
        app = QApplication.instance()
        if app is None or app.platformName() == "offscreen":
            return None
        dlg = QProgressDialog(
            f"Loading {Path(fname).name}...", "Cancel", 0, 0, app.activeWindow()
        )
        dlg.setWindowTitle("Loading")
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setMinimumDuration(0)
        dlg.setAutoClose(False)
        dlg.setAutoReset(False)
        dlg.canceled.connect(self._on_load_cancelled)
        dlg.show()
        return dlg

    def _on_load_stage(self, msg):
        dlg = getattr(self, "_file_load_progress", None)
        if dlg is not None:
            dlg.setLabelText(msg)

    def _on_load_cancelled(self):
        self._file_load_cancelled = True

    def _finish_load_ui(self):
        dlg = getattr(self, "_file_load_progress", None)
        self._file_load_progress = None
        if dlg is not None:
            # QProgressDialog.reset()/close() emit canceled(); disconnect first
            # so closing the dialog programmatically is not read as a cancel.
            try:
                dlg.canceled.disconnect(self._on_load_cancelled)
            except (RuntimeError, TypeError):
                pass
            dlg.reset()
            dlg.close()
            dlg.deleteLater()
        self._file_load_worker = None

    def _on_async_file_loaded(
        self, fname, type, data, analog_signal, event_signal, spike_groups
    ):
        self._finish_load_ui()
        if getattr(self, "_file_load_cancelled", False):
            return
        self._apply_loaded_file(fname, type, data, analog_signal, event_signal, spike_groups)

    def _on_async_file_failed(self, fname, type, error):
        self._finish_load_ui()
        if getattr(self, "_file_load_cancelled", False):
            return
        self._show_load_error(fname, type)

    def removeUnitGroup(self, unit):
        spike_groups = self.spike_groups
        self.spike_groups = [
            x for i, x in enumerate(spike_groups) if x.event.name != unit.event.name
        ]
        self.update_idx_arrs()
        self.save_undo(
            lambda: self.set_data(
                spike_groups=spike_groups,
            )
        )
        self.onUnitGroupChange.emit()
        self.onUnitListChanged.emit()

    @Slot()
    def addUnitGroup(self):
        evt = Event()
        nom = f"unit"
        x = len(self.spike_groups)
        while f"{nom}_{x}" in [sg.event.name for sg in self.spike_groups]:
            x += 1
        evt.name = f"{nom}_{x}"
        evt.annotations["type"] = "unit"
        evt.annotations["unit"] = len(self.spike_groups)
        evt.annotations["notes"] = ""
        evt.annotations["fibre_type"] = ""
        self.spike_groups.append(tracked_neuron_unit(event=evt))
        self.update_idx_arrs()
        self.onUnitGroupChange.emit()
        self.onUnitListChanged.emit()

    def seed_unit_at(
        self,
        sample_idx: int,
        stimno: Optional[int] = None,
        channel: Optional[str] = None,
        auto_track: bool = True,
        window_ms: float = 5.0,
    ) -> int:
        """Create a new unit seeded at ``(stimno, sample_idx)`` and optionally enable auto-tracking."""
        if stimno is None:
            stimno = int(getattr(self, "stimno", 0) or 0)
        else:
            stimno = int(stimno)
        sample_idx = int(sample_idx)

        primary_ch = channel or (
            getattr(self.analog_signal, "name", "") if self.analog_signal is not None else ""
        )
        avail = self.get_available_channels()
        if not primary_ch and avail:
            primary_ch = avail[0]

        neighbors: List[str] = []
        if avail and primary_ch in avail:
            idx = avail.index(primary_ch)
            candidates: List[Tuple[int, int, str]] = []
            for j, name in enumerate(avail):
                if name == primary_ch:
                    continue
                candidates.append((abs(j - idx), j, name))
            candidates.sort(key=lambda item: (item[0], item[1]))
            neighbors = [name for _, _, name in candidates[:3]]
        elif avail:
            neighbors = [c for c in avail if c != primary_ch][:3]

        # Estimate threshold in mV from waveform amplitude around (stimno, sample_idx)
        thresh_mv = 0.01
        ch_erp = self.get_channel_erp(primary_ch) if primary_ch else self.get_erp()
        if ch_erp is None:
            ch_erp = self.get_erp()
        if ch_erp is not None and 0 <= stimno < len(ch_erp):
            row = ch_erp[stimno]
            if hasattr(row, "rescale"):
                try:
                    row_mv = row.rescale(pq.mV).magnitude
                except Exception:
                    row_mv = row.magnitude
            elif hasattr(row, "magnitude"):
                row_mv = row.magnitude
                sig = self.get_channel_signal(primary_ch) if primary_ch else self.analog_signal
                u = getattr(ch_erp, "units", None) or getattr(sig, "units", None)
                if u is not None:
                    try:
                        row_mv = (np.asarray(row_mv) * u).rescale(pq.mV).magnitude
                    except Exception:
                        pass
            else:
                row_mv = np.asarray(row)
                sig = self.get_channel_signal(primary_ch) if primary_ch else self.analog_signal
                u = getattr(sig, "units", None)
                if u is not None:
                    try:
                        row_mv = (row_mv * u).rescale(pq.mV).magnitude
                    except Exception:
                        pass
            row_arr = np.asarray(row_mv, dtype=np.float64).ravel()
            if row_arr.size > 0:
                clamped_idx = max(0, min(sample_idx, row_arr.size - 1))
                val_in_mv = float(row_arr[clamped_idx])
                thresh_mv = max(0.005, 0.75 * abs(val_in_mv))

        self.addUnitGroup()
        new_idx = len(self.spike_groups) - 1
        sg = self.spike_groups[new_idx]
        sg.channel = str(primary_ch or "")
        sg.neighbor_channels = list(neighbors)
        sg.tracking_threshold_mv = float(thresh_mv)
        sg.tracking_window_ms = float(window_ms)
        sg.auto_track = bool(auto_track)
        if getattr(sg, "event", None) is not None:
            sg.event.annotations["channel"] = sg.channel
            sg.event.annotations["neighbor_channels"] = list(neighbors)
            sg.event.annotations["tracking_threshold_mv"] = float(thresh_mv)
            sg.event.annotations["tracking_window_ms"] = float(window_ms)
            sg.event.annotations["auto_track"] = bool(auto_track)

        self.setUnit(sample_idx, stimno=stimno, unit_idx=new_idx)
        self.setUnitGroup(new_idx)
        self.onUnitParamsChanged.emit(new_idx)
        if auto_track and getattr(self, "is_streaming", False):
            self.ensure_live_tracking_processor()
        return new_idx

    def reanchor_unit_at(
        self,
        unit_idx: int,
        sample_idx: int,
        stimno: Optional[int] = None,
        enable_auto_track: bool = True,
    ) -> bool:
        """Re-anchor ``unit_idx`` at ``(stimno, sample_idx)`` and reset miss counters."""
        if unit_idx < 0 or unit_idx >= len(self.spike_groups):
            return False
        if stimno is None:
            stimno = int(getattr(self, "stimno", 0) or 0)
        else:
            stimno = int(stimno)
        if (
            self.event_signal is None
            or len(self.event_signal) == 0
            or stimno < 0
            or stimno >= len(self.event_signal)
        ):
            return False

        sg = self.spike_groups[unit_idx]
        self.setUnit(int(sample_idx), stimno=stimno, unit_idx=unit_idx)
        sg._consecutive_misses = 0
        if enable_auto_track:
            sg.auto_track = True
            if getattr(sg, "event", None) is not None:
                sg.event.annotations["auto_track"] = True
        self.onUnitParamsChanged.emit(unit_idx)
        if getattr(sg, "auto_track", False) and getattr(self, "is_streaming", False):
            self.ensure_live_tracking_processor()
        return True

    def mergeUnitGroups(self, keep_idx: int, remove_idx: int, save_undo: bool = True):
        """Merge unit ``remove_idx`` into unit ``keep_idx``.

        Combines both units' spike times into a single sorted event kept by
        ``keep_idx`` (its name, channel and annotations win; conflicting
        empty metadata is filled from ``remove_idx``), removes ``remove_idx``
        from the unit list, and supports undo.  Returns the new index of the
        merged unit or -1 on failure.
        """
        n = len(self.spike_groups)
        if (
            not (0 <= keep_idx < n)
            or not (0 <= remove_idx < n)
            or keep_idx == remove_idx
        ):
            return -1
        keep_sg = self.spike_groups[keep_idx]
        rm_sg = self.spike_groups[remove_idx]

        def _times(ev):
            if ev is None or len(ev) == 0:
                return np.empty(0, dtype=np.float64)
            return np.asarray(ev.rescale(pq.s).magnitude, dtype=np.float64).ravel()

        keep_ev = keep_sg.event
        merged = np.unique(np.concatenate([_times(keep_ev), _times(rm_sg.event)]))

        anns = dict(getattr(keep_ev, "annotations", {}) or {})
        anns.pop("name", None)
        rm_anns = dict(getattr(rm_sg.event, "annotations", {}) or {})
        # Fill blanks from the removed unit so its metadata is not lost.
        keep_notes = str(anns.get("notes", "") or getattr(keep_sg, "notes", "") or "")
        rm_notes = str(rm_anns.get("notes", "") or getattr(rm_sg, "notes", "") or "")
        if keep_notes and rm_notes and rm_notes != keep_notes:
            anns["notes"] = f"{keep_notes}; {rm_notes}"
        elif rm_notes:
            anns["notes"] = rm_notes
        keep_ft = str(
            anns.get("fibre_type", "") or getattr(keep_sg, "fibre_type", "") or ""
        )
        rm_ft = str(
            rm_anns.get("fibre_type", "") or getattr(rm_sg, "fibre_type", "") or ""
        )
        if not keep_ft and rm_ft:
            anns["fibre_type"] = rm_ft
        new_event = Event(
            merged * pq.s,
            name=getattr(keep_ev, "name", f"unit_{keep_idx}"),
            units=pq.s,
            **anns,
        )

        old_groups = list(self.spike_groups)
        old_keep_ev = keep_ev
        old_keep_idx_arr = (
            list(keep_sg.idx_arr) if keep_sg.idx_arr is not None else None
        )
        old_rm_ev = rm_sg.event
        old_rm_idx_arr = list(rm_sg.idx_arr) if rm_sg.idx_arr is not None else None
        old_cur = int(self.cur_spike_group)

        keep_sg.event = new_event
        keep_sg._template_matcher = None
        keep_sg._tracked_count = (
            getattr(keep_sg, "_tracked_count", 0) + getattr(rm_sg, "_tracked_count", 0)
        )
        self.spike_groups = [sg for i, sg in enumerate(old_groups) if i != remove_idx]
        new_keep_idx = keep_idx - 1 if remove_idx < keep_idx else keep_idx

        if self.segment is not None:
            for i, seg_ev in enumerate(self.segment.events):
                if getattr(seg_ev, "name", None) == getattr(new_event, "name", None):
                    self.segment.events[i] = new_event
                    break
            self.segment.events = [
                ev for ev in self.segment.events if ev is not old_rm_ev
            ]

        def undo(
            self=self,
            groups=old_groups,
            keep_sg=keep_sg,
            keep_ev=old_keep_ev,
            keep_idx_arr=old_keep_idx_arr,
            rm_sg=rm_sg,
            rm_ev=old_rm_ev,
            rm_idx_arr=old_rm_idx_arr,
            old_cur=old_cur,
        ):
            self.spike_groups = groups
            keep_sg.event = keep_ev
            keep_sg.idx_arr = keep_idx_arr
            rm_sg.event = rm_ev
            rm_sg.idx_arr = rm_idx_arr
            self.cur_spike_group = old_cur
            self.update_idx_arrs()
            self.onUnitChange.emit(self.stimno)
            self.onUnitGroupChange.emit()
            self.onUnitListChanged.emit()

        if save_undo:
            self.save_undo(undo)

        self.update_idx_arrs()
        self.cur_spike_group = max(0, min(new_keep_idx, len(self.spike_groups) - 1))
        self.onUnitChange.emit(self.stimno)
        self.onNewSpikeEvents.emit(new_keep_idx)
        self.onUnitSpikeChanged.emit(new_keep_idx, self.stimno)
        self.onUnitGroupChange.emit()
        self.onUnitListChanged.emit()
        self.setUnitGroup(new_keep_idx)
        return new_keep_idx

    # ------------------------------------------------------------------
    # Candidate-unit search and review
    # ------------------------------------------------------------------

    def _candidate_sampling_rate(self) -> float:
        sr = self.sampling_rate
        if sr is None and self.analog_signal is not None:
            sr = self.analog_signal.sampling_rate
        if sr is None and self.segment is not None:
            for sig in getattr(self.segment, "analogsignals", []) or []:
                sr = getattr(sig, "sampling_rate", None)
                if sr is not None:
                    break
        if hasattr(sr, "rescale"):
            return float(sr.rescale(pq.Hz).magnitude)
        return float(sr or 30000.0)

    def _candidate_active_channel(self) -> str:
        primary = getattr(self.analog_signal, "name", "") or ""
        if primary:
            return primary
        names = list(getattr(self, "active_channel_names", None) or [])
        return names[0] if names else ""

    def _candidate_scan_channels(self) -> List[str]:
        configured = [
            str(c).strip()
            for c in (getattr(self.candidate_config, "scan_channels", None) or [])
            if str(c).strip()
        ]
        if configured:
            avail = self.get_available_channels()
            if avail:
                avail_set = set(avail)
                return [c for c in configured if c in avail_set]
            return configured
        names = [str(c) for c in (getattr(self, "active_channel_names", None) or []) if c]
        primary = getattr(self.analog_signal, "name", "") or ""
        if primary and primary not in names:
            names.insert(0, primary)
        if not names and primary:
            names = [primary]
        return names

    def _candidate_merge_tolerance_samples(self) -> int:
        sr = self._candidate_sampling_rate()
        return max(
            1,
            int(
                round(
                    float(self.candidate_config.merge_tolerance_ms) * sr / 1000.0
                )
            ),
        )

    def _candidate_now(self) -> float:
        return float(time.monotonic())

    def _touch_candidate_review(self) -> None:
        self._candidate_last_interaction = self._candidate_now()

    def get_active_candidate(self) -> Optional[CandidateUnit]:
        if self.active_candidate_id is None:
            return None
        return self.candidate_units.get(self.active_candidate_id)

    def get_visible_candidates(self) -> List[CandidateUnit]:
        pending = [c for c in self.candidate_units.values() if c.status == "pending"]
        pending.sort(key=lambda c: (-float(c.score), int(c.candidate_id)))
        return pending[: int(self.candidate_config.max_candidates)]

    def _pending_candidate_count(self) -> int:
        return sum(1 for c in self.candidate_units.values() if c.status == "pending")

    def set_active_candidate(self, candidate_id: Optional[int]) -> None:
        if candidate_id is not None and candidate_id not in self.candidate_units:
            candidate_id = None
        changed = self.active_candidate_id != candidate_id
        self.active_candidate_id = candidate_id
        self._touch_candidate_review()

        candidate = (
            self.candidate_units.get(candidate_id)
            if candidate_id is not None
            else None
        )
        if candidate is not None:
            target_ch = str(getattr(candidate, "channel", "") or "")
            cur_ch = getattr(self.analog_signal, "name", "") or ""
            if target_ch and target_ch != cur_ch:
                avail = self.get_available_channels()
                if not avail or target_ch in avail:
                    self.set_active_channel(target_ch)

            idx_arr = getattr(getattr(candidate, "unit", None), "idx_arr", None) or []
            n_events = len(self.event_signal) if self.event_signal is not None else 0
            if n_events > 0 and idx_arr:
                cur_stim = int(getattr(self, "stimno", 0) or 0)
                has_at_cur = (
                    0 <= cur_stim < len(idx_arr) and idx_arr[cur_stim] is not None
                )
                if not has_at_cur:
                    detected_stims = [
                        i
                        for i, item in enumerate(idx_arr)
                        if item is not None and 0 <= i < n_events
                    ]
                    if detected_stims:
                        nearest = min(detected_stims, key=lambda s: abs(s - cur_stim))
                        self.setStimNo(nearest)

        if changed:
            self.onCandidateSelectionChanged.emit(
                -1 if candidate_id is None else int(candidate_id)
            )

    def clear_candidates(self, emit: bool = True) -> None:
        self.candidate_units = {}
        self.active_candidate_id = None
        self._candidate_id_counter = 0
        self._candidate_scan_truncated = False
        self._candidate_last_scan_stim = -1
        self._candidate_last_interaction = 0.0
        self._candidate_rescan_discarded = False
        if self._candidate_scan_manager is not None:
            self._candidate_scan_manager.shutdown()
        if emit:
            self.onCandidatesChanged.emit()
            self.onCandidateSelectionChanged.emit(-1)

    def _next_candidate_id(self) -> int:
        self._candidate_id_counter += 1
        while self._candidate_id_counter in self.candidate_units:
            self._candidate_id_counter += 1
        return self._candidate_id_counter

    def _next_candidate_name(self) -> str:
        used = {c.name for c in self.candidate_units.values()}
        used.update(getattr(sg.event, "name", "") for sg in self.spike_groups)
        index = len(used)
        while f"candidate_{index}" in used:
            index += 1
        return f"candidate_{index}"

    def _next_unit_name(self) -> str:
        used = {getattr(sg.event, "name", "") for sg in self.spike_groups}
        used.update(c.name for c in self.candidate_units.values())
        index = len(self.spike_groups)
        while f"unit_{index}" in used:
            index += 1
        return f"unit_{index}"

    def _candidate_timestamp_s(self, stim: int, sample: int) -> float:
        sr = self._candidate_sampling_rate()
        base = None
        if (
            self.event_signal is not None
            and 0 <= int(stim) < len(self.event_signal)
        ):
            base = self.event_signal[int(stim)]
        if base is None:
            return float(sample) / sr
        try:
            return float((base + (float(sample) / sr) * pq.s).rescale(pq.s).magnitude)
        except Exception:
            return float(sample) / sr

    def _candidate_event(
        self,
        name: str,
        detections: Dict[int, int],
        channel: str,
        neighbors: List[str],
        candidate_id: Optional[int] = None,
    ) -> Event:
        times = [
            self._candidate_timestamp_s(stim, sample)
            for stim, sample in sorted(detections.items())
        ]
        evt = Event(np.asarray(times, dtype=np.float64) * pq.s, name=name, units=pq.s)
        evt.annotations["type"] = "candidate"
        if candidate_id is not None:
            evt.annotations["candidate_id"] = int(candidate_id)
        if channel:
            evt.annotations["channel"] = channel
        if neighbors:
            evt.annotations["neighbor_channels"] = list(neighbors)
        return evt

    def _candidate_unit_payload(
        self,
        name: str,
        detections: Dict[int, int],
        channel: str,
        neighbors: List[str],
        candidate_id: Optional[int] = None,
        algorithm: str = "zscore",
        detector_params: Optional[Dict[str, Any]] = None,
        notes: str = "",
    ) -> tracked_neuron_unit:
        evt = self._candidate_event(
            name, detections, channel, neighbors, candidate_id=candidate_id
        )
        n_sweeps = (
            len(self.event_signal) if self.event_signal is not None else 0
        )
        idx_arr: List[Any] = [None for _ in range(n_sweeps)]
        for stim, sample in sorted(detections.items()):
            if 0 <= int(stim) < n_sweeps:
                idx_arr[int(stim)] = (int(sample), 1)
        return tracked_neuron_unit(
            event=evt,
            channel=channel,
            neighbor_channels=list(neighbors),
            notes=notes,
            idx_arr=idx_arr,
            tracking_algorithm=algorithm,
            tracking_params=dict(detector_params or {}),
        )

    def _candidate_threshold_mv(self, candidate: CandidateUnit) -> float:
        raw = float(
            candidate.threshold_signal
            or candidate.detector_params.get("threshold", 0.0)
            or 0.0
        )
        if raw <= 0:
            return 0.01
        units = candidate.signal_units or str(
            getattr(self.analog_signal, "units", pq.mV)
        )
        try:
            return float((pq.Quantity(raw, units)).rescale(pq.mV).magnitude)
        except Exception:
            lowered = str(units or "").lower()
            if "uv" in lowered or "µv" in lowered:
                return raw / 1000.0
            return raw

    def _candidate_tracking_params(self, candidate: CandidateUnit) -> Dict[str, Any]:
        params = {
            k: v
            for k, v in dict(candidate.detector_params or {}).items()
            if k
            not in (
                "window_ms",
                "max_skip",
                "track_limit",
                "threshold",
            )
        }
        if candidate.algorithm == "zscore":
            params.setdefault("polarity", "both")
            params.setdefault("z_thresh", float(self.candidate_config.min_peak_snr))
            params.setdefault("use_mad", True)
        return params

    def _candidate_tracking_window_ms(self, candidate: CandidateUnit) -> float:
        jitter = float(getattr(candidate.evidence, "jitter_ms", 0.0) or 0.0)
        return float(min(10.0, max(2.0, 4.0 * jitter + 0.5)))

    def _candidate_from_proposal(
        self, proposal: CandidateProposal, status: str = "pending"
    ) -> CandidateUnit:
        candidate_id = self._next_candidate_id()
        name = self._next_candidate_name()
        unit = self._candidate_unit_payload(
            name,
            dict(proposal.samples_by_sweep),
            proposal.channel,
            list(proposal.neighbor_channels),
            candidate_id=candidate_id,
            algorithm=proposal.algorithm,
            detector_params=proposal.detector_params,
        )
        candidate = CandidateUnit(
            candidate_id=candidate_id,
            unit=unit,
            algorithm=proposal.algorithm,
            detector_params=dict(proposal.detector_params or {}),
            template_refined=bool(proposal.template_refined),
            name=name,
            channel=proposal.channel,
            neighbor_channels=list(proposal.neighbor_channels),
            score=float(proposal.score),
            evidence=proposal.evidence,
            latency_center_samples=float(proposal.latency_center_samples),
            latency_tolerance_samples=float(proposal.latency_tolerance_samples),
            status=status,
            first_seen_stim=int(proposal.first_seen_stim),
            last_seen_stim=int(proposal.last_seen_stim),
            threshold_signal=float(proposal.threshold_signal),
            signal_units=str(proposal.signal_units),
        )
        candidate.tracking_window_ms = self._candidate_tracking_window_ms(candidate)
        candidate.tracking_threshold_mv = self._candidate_threshold_mv(candidate)
        self.candidate_units[candidate_id] = candidate
        return candidate

    def _match_candidate(self, proposal: CandidateProposal) -> Optional[CandidateUnit]:
        best: Optional[CandidateUnit] = None
        best_distance = float("inf")
        for candidate in self.candidate_units.values():
            if candidate.channel != proposal.channel:
                continue
            tolerance = max(
                float(candidate.latency_tolerance_samples),
                float(proposal.latency_tolerance_samples),
                float(self._candidate_merge_tolerance_samples()),
            )
            distance = abs(
                float(candidate.latency_center_samples)
                - float(proposal.latency_center_samples)
            )
            if distance <= tolerance and distance < best_distance:
                best = candidate
                best_distance = distance
        return best

    def _update_candidate_from_proposal(
        self,
        candidate: CandidateUnit,
        proposal: CandidateProposal,
        scan_start: int,
        scan_end: int,
    ) -> None:
        old_detections: Dict[int, int] = {}
        idx_arr = candidate.unit.idx_arr or []
        for stim, item in enumerate(idx_arr):
            if item is None:
                continue
            sample = item[0] if isinstance(item, (tuple, list)) else item
            try:
                old_detections[int(stim)] = int(sample)
            except (TypeError, ValueError):
                continue
        merged = {
            stim: sample
            for stim, sample in old_detections.items()
            if not (int(scan_start) <= stim <= int(scan_end))
        }
        merged.update(
            {int(s): int(v) for s, v in proposal.samples_by_sweep.items()}
        )
        if merged:
            center = float(np.median(np.asarray(list(merged.values()), dtype=float)))
            latencies_ms = np.asarray(list(merged.values()), dtype=float) / (
                self._candidate_sampling_rate()
            ) * 1000.0
            jitter = (
                1.4826 * float(np.median(np.abs(latencies_ms - np.median(latencies_ms))))
                if latencies_ms.size >= 2
                else 0.0
            )
        else:
            center = float(proposal.latency_center_samples)
            jitter = float(getattr(proposal.evidence, "jitter_ms", 0.0) or 0.0)

        candidate.unit = self._candidate_unit_payload(
            candidate.name,
            merged,
            proposal.channel,
            list(proposal.neighbor_channels),
            candidate_id=candidate.candidate_id,
            algorithm=proposal.algorithm,
            detector_params=proposal.detector_params,
            notes=candidate.notes,
        )
        candidate.channel = proposal.channel
        candidate.neighbor_channels = list(proposal.neighbor_channels)
        candidate.algorithm = proposal.algorithm
        candidate.detector_params = dict(proposal.detector_params or {})
        candidate.template_refined = bool(proposal.template_refined)
        candidate.score = float(proposal.score)
        candidate.evidence = proposal.evidence
        candidate.evidence.hit_count = len(merged)
        candidate.evidence.jitter_ms = jitter
        candidate.latency_center_samples = center
        candidate.latency_tolerance_samples = float(
            proposal.latency_tolerance_samples
        )
        candidate.first_seen_stim = min(
            [int(stim) for stim in merged] + [int(candidate.first_seen_stim)]
        )
        candidate.last_seen_stim = max(
            [int(stim) for stim in merged] + [int(candidate.last_seen_stim)]
        )
        candidate.threshold_signal = float(proposal.threshold_signal)
        candidate.signal_units = str(proposal.signal_units)
        candidate.tracking_window_ms = self._candidate_tracking_window_ms(candidate)
        candidate.tracking_threshold_mv = self._candidate_threshold_mv(candidate)

        if not merged:
            candidate.status = "dormant"
            return

        if candidate.status == "pending":
            if int(proposal.evidence.recent_hit_count) < 1:
                candidate.status = "dormant"
        elif candidate.status == "dormant":
            if (
                float(proposal.score)
                >= float(candidate.score_at_discard or 0.0)
                or int(proposal.evidence.recent_hit_count) >= 1
            ):
                candidate.status = "pending"
        elif candidate.status == "discarded":
            channel_changed = bool(candidate.channel_at_discard) and (
                candidate.channel_at_discard != self._candidate_active_channel()
            )
            strong_enough = float(proposal.score) > float(
                candidate.score_at_discard or 0.0
            ) + float(self.candidate_config.reactivation_score_margin)
            if self._candidate_rescan_discarded and strong_enough:
                candidate.status = "pending"
                candidate.score_at_discard = None
                candidate.channel_at_discard = None
            elif channel_changed and strong_enough:
                candidate.status = "pending"
                candidate.score_at_discard = None
                candidate.channel_at_discard = None

    def _dormant_stale_candidates(self, scan_end: int, matched_ids: set) -> None:
        recency = int(self.candidate_config.recency_window_sweeps)
        cutoff = int(scan_end) - recency
        for candidate in self.candidate_units.values():
            if candidate.candidate_id in matched_ids:
                continue
            if candidate.status == "pending" and int(candidate.last_seen_stim) < cutoff:
                candidate.status = "dormant"
                if self.active_candidate_id == candidate.candidate_id:
                    self.active_candidate_id = None
                    self.onCandidateSelectionChanged.emit(-1)

    def _apply_candidate_scan_result(
        self,
        generation: int,
        proposals: Any,
        scan_start: int,
        scan_end: int,
        error: str = "",
    ) -> None:
        if error:
            logger.warning("Candidate scan failed: %s", error)
            return
        if int(generation) != int(self._candidate_scan_generation):
            return
        matched_ids: set = set()
        for proposal in proposals or []:
            candidate = self._match_candidate(proposal)
            if candidate is None:
                if self._pending_candidate_count() >= int(
                    self.candidate_config.max_candidates
                ):
                    pending = [
                        c
                        for c in self.candidate_units.values()
                        if c.status == "pending"
                        and c.candidate_id != self.active_candidate_id
                    ]
                    if pending:
                        lowest = min(pending, key=lambda c: float(c.score))
                        if float(proposal.score) > float(lowest.score):
                            lowest.status = "dormant"
                            candidate = self._candidate_from_proposal(proposal)
                        else:
                            candidate = self._candidate_from_proposal(
                                proposal, status="dormant"
                            )
                    else:
                        candidate = self._candidate_from_proposal(
                            proposal, status="dormant"
                        )
                else:
                    recent = int(getattr(proposal.evidence, "recent_hit_count", 0))
                    candidate = self._candidate_from_proposal(
                        proposal, status="pending" if recent >= 1 else "dormant"
                    )
            else:
                self._update_candidate_from_proposal(
                    candidate, proposal, scan_start, scan_end
                )
            matched_ids.add(candidate.candidate_id)
        self._dormant_stale_candidates(scan_end, matched_ids)
        self._candidate_last_scan_stim = int(scan_end)
        if self._candidate_rescan_discarded:
            self._candidate_rescan_discarded = False
        self.onCandidatesChanged.emit()

    def _build_candidate_scan_snapshot(self, scope: str = "auto") -> Optional[Dict[str, Any]]:
        n_sweeps = len(self.event_signal) if self.event_signal is not None else 0
        if n_sweeps <= 0:
            return None
        cfg = self.candidate_config
        live = str(scope) in ("auto", "live", "recent")
        if live:
            requested_start = max(0, n_sweeps - int(cfg.live_scan_window_sweeps))
            requested_end = n_sweeps - 1
            self._candidate_scan_truncated = False
        else:
            requested_start = 0
            requested_end = n_sweeps - 1
            if n_sweeps > int(cfg.max_scan_sweeps):
                requested_end = int(cfg.max_scan_sweeps) - 1
                self._candidate_scan_truncated = True
            else:
                self._candidate_scan_truncated = False

        channels = self._candidate_scan_channels()
        arrays: Dict[str, np.ndarray] = {}
        offsets: Dict[str, int] = {}
        for channel in channels:
            erp = self.get_channel_erp(channel)
            if erp is None:
                continue
            arr = getattr(erp, "magnitude", erp)
            arr = np.asarray(arr, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
                continue
            arrays[channel] = arr
            offsets[channel] = max(0, n_sweeps - int(arr.shape[0]))
        if not arrays:
            return None

        start = max(requested_start, max(offsets.values()))
        end = min(requested_end, n_sweeps - 1)
        if end < start:
            return None

        erp_by_channel: Dict[str, np.ndarray] = {}
        for channel, arr in arrays.items():
            local_start = start - offsets[channel]
            local_end = min(arr.shape[0], end - offsets[channel] + 1)
            if local_start < 0:
                local_start = 0
            if local_end > local_start:
                erp_by_channel[channel] = arr[local_start:local_end]

        if not erp_by_channel:
            return None

        existing_units: Dict[str, List[Dict[int, int]]] = {}
        for sg in self.spike_groups:
            channel = (
                getattr(sg, "channel", "")
                or getattr(sg, "tracking_channel", "")
                or self._candidate_active_channel()
            )
            samples: Dict[int, int] = {}
            for stim, item in enumerate(getattr(sg, "idx_arr", None) or []):
                if item is None or stim >= n_sweeps:
                    continue
                sample = item[0] if isinstance(item, (tuple, list)) else item
                try:
                    samples[int(stim)] = int(sample)
                except (TypeError, ValueError):
                    continue
            if samples:
                existing_units.setdefault(channel, []).append(samples)

        candidate_memory = [
            {
                "id": int(c.candidate_id),
                "channel": c.channel,
                "center": float(c.latency_center_samples),
                "tolerance": float(c.latency_tolerance_samples),
                "status": c.status,
                "score": float(c.score),
            }
            for c in self.candidate_units.values()
        ]

        units = (
            getattr(self.analog_signal, "units", None)
            if self.analog_signal is not None
            else None
        )
        return {
            "erp_by_channel": erp_by_channel,
            "sampling_rate_hz": self._candidate_sampling_rate(),
            "scan_start_stim": int(start),
            "scan_end_stim": int(end),
            "existing_units": existing_units,
            "candidate_memory": candidate_memory,
            "config": replace(self.candidate_config),
            "waveform_units": units,
            "truncated": bool(self._candidate_scan_truncated),
        }

    def _ensure_candidate_scan_manager(self):
        manager = self._candidate_scan_manager
        if manager is not None:
            return manager
        from .candidate_worker import CandidateScanManager

        manager = CandidateScanManager(
            on_result=self._apply_candidate_scan_result,
            tick_cb=lambda: self.request_candidate_scan(
                synchronous=False, scope="live"
            ),
            parent=self,
        )
        manager.scanStateChanged.connect(self.onCandidateScanStateChanged.emit)
        self._candidate_scan_manager = manager
        return manager

    def _candidate_scan_gate(self, scope: str) -> bool:
        if self._candidate_scan_manager is not None and self._candidate_scan_manager.busy:
            return False
        if str(scope) not in ("live", "auto", "recent"):
            return True
        if self.event_signal is None or len(self.event_signal) == 0:
            return False
        latest = len(self.event_signal) - 1
        if latest <= self._candidate_last_scan_stim:
            return False
        quiet_ms = float(self.candidate_config.review_quiet_ms)
        if (self._candidate_now() - self._candidate_last_interaction) * 1000.0 < quiet_ms:
            return False
        if self._pending_candidate_count() >= int(
            self.candidate_config.max_candidates
        ):
            return False
        return True

    def request_candidate_scan(self, synchronous: bool = False, scope: str = "auto") -> bool:
        """Run a candidate scan.  ``scope`` is 'live', 'offline', or 'auto'."""
        scope = str(scope or "auto")
        if scope == "auto":
            scope = "live" if self.is_streaming else "offline"
        if scope in ("live", "recent") and not self.is_streaming:
            scope = "live"
        if not synchronous and not self._candidate_scan_gate(scope):
            return False
        snapshot = self._build_candidate_scan_snapshot(scope)
        if snapshot is None:
            return False
        manager = self._ensure_candidate_scan_manager()
        # Set the expected generation before the synchronous path runs so its
        # inline result is not rejected as stale.
        self._candidate_scan_generation = int(manager.generation) + 1
        generation = manager.request(snapshot, scope=scope, synchronous=synchronous)
        if generation is None:
            return False
        self._candidate_scan_generation = int(generation)
        self._candidate_last_scan_stim = int(snapshot["scan_end_stim"])
        return True

    def update_candidate_config(self, **kwargs: Any) -> CandidateScanConfig:
        """Update fields on ``self.candidate_config`` and notify listeners."""
        cfg = self.candidate_config
        scan_interval_updated = False
        for key, value in kwargs.items():
            if not hasattr(cfg, key):
                continue
            if key == "detector_params":
                if isinstance(value, dict):
                    cfg.detector_params = dict(value)
                elif value is None:
                    cfg.detector_params = {}
                continue
            if key == "enabled_algorithms":
                cfg.enabled_algorithms = [str(a) for a in (value or []) if str(a).strip()]
                continue
            if key == "scan_channels":
                seen = set()
                cleaned: List[str] = []
                for ch in value or []:
                    s = str(ch).strip()
                    if s and s not in seen:
                        seen.add(s)
                        cleaned.append(s)
                cfg.scan_channels = cleaned
                continue
            setattr(cfg, key, value)
            if key == "scan_interval_ms":
                scan_interval_updated = True
        if scan_interval_updated and self.auto_scan_live and self.is_streaming:
            self._update_candidate_auto_scan()
        self.onCandidateConfigChanged.emit()
        return cfg

    def set_auto_scan_live(self, enabled: bool) -> None:
        self.auto_scan_live = bool(enabled)
        self._update_candidate_auto_scan()

    def _update_candidate_auto_scan(self, *args) -> None:
        manager = self._candidate_scan_manager
        if manager is None:
            if not self.auto_scan_live:
                return
            manager = self._ensure_candidate_scan_manager()
        if self.auto_scan_live and self.is_streaming:
            manager.start_auto(self.candidate_config.scan_interval_ms)
        else:
            manager.stop_auto()

    def accept_candidate(self, candidate_id: int) -> int:
        candidate = self.candidate_units.get(int(candidate_id))
        if candidate is None or candidate.status == "discarded":
            return -1
        detections: Dict[int, int] = {}
        for stim, item in enumerate(candidate.unit.idx_arr or []):
            if item is None:
                continue
            sample = item[0] if isinstance(item, (tuple, list)) else item
            try:
                detections[int(stim)] = int(sample)
            except (TypeError, ValueError):
                continue
        if not detections:
            return -1

        name = candidate.name if candidate.name_edited else self._next_unit_name()
        used = {getattr(sg.event, "name", "") for sg in self.spike_groups}
        used.update(
            c.name
            for c in self.candidate_units.values()
            if c.candidate_id != candidate.candidate_id
        )
        if name in used:
            base = name
            suffix = 1
            while name in used:
                name = f"{base}_{suffix}"
                suffix += 1
        unit = self._candidate_unit_payload(
            name,
            detections,
            candidate.channel,
            list(candidate.neighbor_channels),
            candidate_id=None,
            algorithm=candidate.algorithm,
            detector_params=self._candidate_tracking_params(candidate),
        )
        evt = unit.event
        evt.annotations["type"] = "unit"
        evt.annotations["fibre_type"] = ""
        evt.annotations["notes"] = candidate.notes if candidate.notes_edited else ""
        evt.annotations["auto_track"] = True
        evt.annotations["tracking_window_ms"] = float(candidate.tracking_window_ms)
        evt.annotations["tracking_threshold_mv"] = float(
            candidate.tracking_threshold_mv
        )
        evt.annotations["tracking_max_skips"] = 3
        evt.annotations["tracking_algorithm"] = candidate.algorithm
        evt.annotations["tracking_params"] = self._candidate_tracking_params(candidate)
        evt.annotations["candidate_source"] = "search"
        evt.annotations["candidate_score"] = float(candidate.score)
        evt.annotations["candidate_first_seen_stim"] = int(candidate.first_seen_stim)

        tracked = tracked_neuron_unit(
            event=evt,
            channel=candidate.channel,
            neighbor_channels=list(candidate.neighbor_channels),
            notes=str(evt.annotations.get("notes", "")),
            idx_arr=[None for _ in range((len(self.event_signal) if self.event_signal is not None else 0))],
            auto_track=True,
            tracking_window_ms=float(candidate.tracking_window_ms),
            tracking_threshold_mv=float(candidate.tracking_threshold_mv),
            tracking_max_skips=3,
            tracking_algorithm=str(candidate.algorithm),
            tracking_params=self._candidate_tracking_params(candidate),
        )
        exact_idx: List[Any] = [None for _ in range((len(self.event_signal) if self.event_signal is not None else 0))]
        for stim, sample in detections.items():
            if 0 <= int(stim) < len(exact_idx):
                exact_idx[int(stim)] = (int(sample), 1)
        tracked._idx_arr = exact_idx

        old_cur = int(self.cur_spike_group)
        old_candidate = candidate
        self.spike_groups.append(tracked)
        new_idx = len(self.spike_groups) - 1
        self.update_idx_arrs()
        tracked._idx_arr = exact_idx

        if candidate.template_refined:
            try:
                self.rebuild_unit_template(new_idx)
            except Exception:
                logger.debug("Could not initialise accepted candidate template", exc_info=True)

        del self.candidate_units[int(candidate_id)]
        self.active_candidate_id = None

        def undo_accept(
            self=self,
            tracked=tracked,
            name=name,
            candidate=old_candidate,
            cid=int(candidate_id),
            old_cur=old_cur,
        ):
            self.spike_groups = [
                sg for sg in self.spike_groups if sg is not tracked and sg.event.name != name
            ]
            self.candidate_units[cid] = candidate
            self.active_candidate_id = cid
            self.cur_spike_group = max(0, min(old_cur, len(self.spike_groups) - 1))
            self.update_idx_arrs()
            self.onUnitGroupChange.emit()
            self.onUnitListChanged.emit()
            self.onCandidatesChanged.emit()
            self.onCandidateSelectionChanged.emit(cid)

        self.save_undo(undo_accept)
        self.cur_spike_group = new_idx
        self.onUnitGroupChange.emit()
        self.onUnitListChanged.emit()
        self.onNewSpikeEvents.emit(new_idx)
        self.onCandidatesChanged.emit()
        self.onCandidateSelectionChanged.emit(-1)
        self.setUnitGroup(new_idx)
        return new_idx

    def discard_candidate(self, candidate_id: int) -> bool:
        candidate = self.candidate_units.get(int(candidate_id))
        if candidate is None:
            return False
        previous_status = candidate.status
        previous_score = candidate.score_at_discard
        previous_channel = candidate.channel_at_discard
        candidate.status = "discarded"
        candidate.score_at_discard = float(candidate.score)
        candidate.channel_at_discard = candidate.channel
        was_active = self.active_candidate_id == candidate.candidate_id
        if was_active:
            self.active_candidate_id = None
        self._touch_candidate_review()

        def undo_discard(
            self=self,
            candidate=candidate,
            previous_status=previous_status,
            previous_score=previous_score,
            previous_channel=previous_channel,
            was_active=was_active,
        ):
            candidate.status = previous_status
            candidate.score_at_discard = previous_score
            candidate.channel_at_discard = previous_channel
            if was_active:
                self.active_candidate_id = candidate.candidate_id
            self.onCandidatesChanged.emit()
            self.onCandidateSelectionChanged.emit(
                candidate.candidate_id if was_active else -1
            )

        self.save_undo(undo_discard)
        self.onCandidatesChanged.emit()
        self.onCandidateSelectionChanged.emit(-1)
        return True

    def reactivate_candidate(self, candidate_id: int) -> bool:
        candidate = self.candidate_units.get(int(candidate_id))
        if candidate is None:
            return False
        candidate.status = "pending"
        candidate.score_at_discard = None
        candidate.channel_at_discard = None
        self.onCandidatesChanged.emit()
        return True

    def export_candidate_store(self) -> Dict[str, Any]:
        sr = self._candidate_sampling_rate()
        candidates = []
        for candidate in sorted(
            self.candidate_units.values(), key=lambda c: int(c.candidate_id)
        ):
            detections = []
            for stim, item in enumerate(candidate.unit.idx_arr or []):
                if item is None:
                    continue
                sample = item[0] if isinstance(item, (tuple, list)) else item
                try:
                    sample = int(sample)
                except (TypeError, ValueError):
                    continue
                detections.append(
                    [int(stim), sample, self._candidate_timestamp_s(stim, sample)]
                )
            evidence = candidate.evidence or CandidateEvidence()
            candidates.append(
                {
                    "id": int(candidate.candidate_id),
                    "status": candidate.status,
                    "name": candidate.name,
                    "name_edited": bool(candidate.name_edited),
                    "channel": candidate.channel,
                    "neighbors": list(candidate.neighbor_channels),
                    "score": float(candidate.score),
                    "score_at_discard": candidate.score_at_discard,
                    "channel_at_discard": candidate.channel_at_discard,
                    "latency_center_s": float(candidate.latency_center_samples) / sr,
                    "latency_tolerance_s": float(candidate.latency_tolerance_samples)
                    / sr,
                    "algorithm": candidate.algorithm,
                    "detector_params": dict(candidate.detector_params or {}),
                    "template_refined": bool(candidate.template_refined),
                    "features": {
                        "hits": int(getattr(evidence, "hit_count", 0)),
                        "recent_hits": int(getattr(evidence, "recent_hit_count", 0)),
                        "jitter_ms": float(getattr(evidence, "jitter_ms", 0.0)),
                        "snr": float(getattr(evidence, "median_snr", 0.0)),
                        "hit_rate": float(getattr(evidence, "hit_rate", 0.0)),
                        "waveform_score": float(
                            getattr(evidence, "waveform_score", 0.5)
                        ),
                        "footprint_score": float(
                            getattr(evidence, "footprint_score", 0.5)
                        ),
                        "recency_score": float(
                            getattr(evidence, "recency_score", 0.0)
                        ),
                        "distinctiveness": float(
                            getattr(evidence, "distinctiveness", 0.0)
                        ),
                    },
                    "tracking": {
                        "window_ms": float(candidate.tracking_window_ms),
                        "threshold_mv": float(candidate.tracking_threshold_mv),
                        "max_skips": int(candidate.tracking_max_skips),
                    },
                    "detections": detections,
                    "first_seen_stim": int(candidate.first_seen_stim),
                    "last_seen_stim": int(candidate.last_seen_stim),
                    "notes": candidate.notes,
                    "notes_edited": bool(candidate.notes_edited),
                    "threshold_signal": float(candidate.threshold_signal),
                    "signal_units": candidate.signal_units,
                }
            )
        return {"version": 1, "candidates": candidates}

    def restore_candidate_store(
        self, payload: Any, event_signal: Optional[Event] = None
    ) -> bool:
        """Restore candidates from a saved payload (dict or JSON string)."""
        if payload is None or payload == "" or payload == {}:
            self.clear_candidates(emit=True)
            return True
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except (TypeError, ValueError):
                logger.warning("Ignoring malformed candidate annotation")
                self.clear_candidates(emit=True)
                return False
        if not isinstance(payload, dict):
            self.clear_candidates(emit=True)
            return False
        version = payload.get("version", 0)
        if int(version or 0) != 1:
            logger.warning("Ignoring unknown candidate store version %r", version)
            self.clear_candidates(emit=True)
            return False

        ev = event_signal if event_signal is not None else self.event_signal
        n_sweeps = len(ev) if ev is not None else 0
        self.candidate_units = {}
        self.active_candidate_id = None
        self._candidate_id_counter = 0
        for item in payload.get("candidates", []) or []:
            try:
                cid = int(item.get("id", 0))
            except (TypeError, ValueError):
                continue
            detections: Dict[int, int] = {}
            stored_times: Dict[int, float] = {}
            for det in item.get("detections", []) or []:
                if not isinstance(det, (list, tuple)) or len(det) < 2:
                    continue
                try:
                    stim, sample = int(det[0]), int(det[1])
                except (TypeError, ValueError):
                    continue
                if n_sweeps and not (0 <= stim < n_sweeps):
                    continue
                if sample < 0:
                    continue
                detections[stim] = sample
                if len(det) >= 3:
                    try:
                        stored_times[stim] = float(det[2])
                    except (TypeError, ValueError):
                        pass
            if cid <= 0:
                cid = self._next_candidate_id()
            else:
                self._candidate_id_counter = max(self._candidate_id_counter, cid)
            status = str(item.get("status", "pending") or "pending")
            if status not in ("pending", "dormant", "discarded"):
                status = "dormant"
            if not detections:
                status = "dormant"
            name = str(item.get("name", "") or f"candidate_{cid}")
            used = {getattr(sg.event, "name", "") for sg in self.spike_groups}
            used.update(c.name for c in self.candidate_units.values())
            if name in used:
                base = name
                suffix = 1
                while name in used:
                    name = f"{base}_{suffix}"
                    suffix += 1
            channel = str(item.get("channel", "") or "")
            neighbors = [str(c) for c in (item.get("neighbors") or [])]
            algorithm = str(item.get("algorithm", "zscore") or "zscore")
            detector_params = dict(item.get("detector_params") or {})
            unit = self._candidate_unit_payload(
                name,
                detections,
                channel,
                neighbors,
                candidate_id=cid,
                algorithm=algorithm,
                detector_params=detector_params,
                notes=str(item.get("notes", "") or ""),
            )
            if n_sweeps == 0 and detections:
                times = [
                    stored_times.get(stim, sample / self._candidate_sampling_rate())
                    for stim, sample in sorted(detections.items())
                ]
                evt = Event(np.asarray(times, dtype=float) * pq.s, name=name, units=pq.s)
                evt.annotations.update(dict(unit.event.annotations))
                unit.event = evt
                unit.idx_arr = [None for _ in range(max(detections) + 1)]
                for stim, sample in detections.items():
                    unit.idx_arr[int(stim)] = (int(sample), 1)
            features = dict(item.get("features") or {})
            evidence = CandidateEvidence(
                hit_count=int(features.get("hits", len(detections)) or 0),
                recent_hit_count=int(features.get("recent_hits", 0) or 0),
                jitter_ms=float(features.get("jitter_ms", 0.0) or 0.0),
                hit_rate=float(features.get("hit_rate", 0.0) or 0.0),
                median_snr=float(features.get("snr", 0.0) or 0.0),
                waveform_score=float(features.get("waveform_score", 0.5) or 0.5),
                footprint_score=float(features.get("footprint_score", 0.5) or 0.5),
                recency_score=float(features.get("recency_score", 0.0) or 0.0),
                distinctiveness=float(features.get("distinctiveness", 0.0) or 0.0),
            )
            sr = self._candidate_sampling_rate()
            candidate = CandidateUnit(
                candidate_id=cid,
                unit=unit,
                algorithm=algorithm,
                detector_params=detector_params,
                template_refined=bool(item.get("template_refined", False)),
                name=name,
                name_edited=bool(item.get("name_edited", False)),
                channel=channel,
                neighbor_channels=neighbors,
                score=float(item.get("score", 0.0) or 0.0),
                evidence=evidence,
                latency_center_samples=float(item.get("latency_center_s", 0.0) or 0.0)
                * sr,
                latency_tolerance_samples=float(
                    item.get("latency_tolerance_s", 0.0) or 0.0
                )
                * sr,
                status=status,
                score_at_discard=item.get("score_at_discard"),
                channel_at_discard=item.get("channel_at_discard"),
                first_seen_stim=int(item.get("first_seen_stim", 0) or 0),
                last_seen_stim=int(item.get("last_seen_stim", 0) or 0),
                notes=str(item.get("notes", "") or ""),
                notes_edited=bool(item.get("notes_edited", False)),
                threshold_signal=float(item.get("threshold_signal", 0.0) or 0.0),
                signal_units=str(item.get("signal_units", "") or ""),
            )
            tracking = dict(item.get("tracking") or {})
            candidate.tracking_window_ms = float(
                tracking.get("window_ms", 5.0) or 5.0
            )
            candidate.tracking_threshold_mv = float(
                tracking.get("threshold_mv", 0.01) or 0.01
            )
            candidate.tracking_max_skips = int(tracking.get("max_skips", 3) or 3)
            unit.tracking_algorithm = algorithm
            unit.tracking_window_ms = candidate.tracking_window_ms
            unit.tracking_threshold_mv = candidate.tracking_threshold_mv
            unit.tracking_max_skips = candidate.tracking_max_skips
            if not candidate.latency_center_samples and detections:
                candidate.latency_center_samples = float(
                    np.median(list(detections.values()))
                )
            if not candidate.latency_tolerance_samples:
                candidate.latency_tolerance_samples = float(
                    max(1, int(self._candidate_merge_tolerance_samples()))
                )
            self.candidate_units[cid] = candidate
        self.onCandidatesChanged.emit()
        self.onCandidateSelectionChanged.emit(-1)
        return True

    def update_idx_arrs(self):
        if self.event_signal is None or len(self.event_signal) == 0:
            for sg in self.spike_groups:
                sg.idx_arr = []
            return

        sr = float(
            self.sampling_rate
            if self.sampling_rate is not None
            else (
                self.analog_signal.sampling_rate.rescale(pq.Hz).magnitude
                if self.analog_signal is not None
                else 30000.0
            )
        )

        n_sweeps = len(self.event_signal)
        for sg in self.spike_groups:
            sg.pre_offset_s = self.pre_offset_s
            p = [None for _ in range(n_sweeps)]
            if sg.event is not None and len(sg.event) > 0:
                sweeps, samples = sg.get_valid_sweep_indices_and_samples(
                    self.event_signal, sr, self.window_size, pre_offset_s=self.pre_offset_s
                )
                for sw, sa in zip(sweeps, samples):
                    p[sw] = (int(sa), 1)
            sg.idx_arr = p

    def get_erp(
        self,
        signal: neo.AnalogSignal = None,
        event_signal: neo.Event = None,
        channel=0,
        window_size=None,
    ):
        if window_size is None:
            window_size = self.window_size

        if (
            self._live_erp_buffer is not None
            and (signal is None or signal is self.analog_signal)
            and (event_signal is None or event_signal is self.event_signal)
        ):
            return self._live_erp_buffer.data

        if (
            self.segment is None
            or self.event_signal is None
            or len(self.event_signal) == 0
        ):
            num_samples = int(
                self.window_size.rescale(pq.s).magnitude * (self.sampling_rate or 30000)
            )
            units = (
                self.analog_signal.units if self.analog_signal is not None else pq.mV
            )
            return np.zeros((1, num_samples)) * units

        if signal is None:
            signal = self.analog_signal
        if event_signal is None:
            event_signal = self.event_signal

        try:
            signal_idx = next(
                i
                for i, x in enumerate(self.segment.analogsignals)
                if x.name == signal.name
            )
            events_idx = next(
                i
                for i, x in enumerate(self.segment.events)
                if x.name == event_signal.name
            )
        except StopIteration:
            num_samples = int(
                self.window_size.rescale(pq.s).magnitude * (self.sampling_rate or 30000)
            )
            units = (
                self.analog_signal.units if self.analog_signal is not None else pq.mV
            )
            return np.zeros((1, num_samples)) * units

        return self._get_erp(
            signal_idx, events_idx, channel, float(window_size.rescale(pq.second))
        )

    @lru_numpy_memmap()
    def _get_erp(
        self, signal_idx=None, event_signal_idx=None, channel=0, window_size=0.5
    ):
        signal = self.segment.analogsignals[signal_idx]
        event_signal = self.segment.events[event_signal_idx]
        erp = create_erp_signals(signal, event_signal, 0 * pq.s, window_size * pq.s)

        # s = int(np.array(signal.sampling_rate.base) // ((1 / window_size)))
        # erp = create_erp(
        #     signal.rescale("mV").as_array()[:, channel],
        #     (event_signal.as_array() - signal.t_start.base)
        #     * np.array(signal.sampling_rate, dtype=int),
        #     0,
        #     s,
        # )
        return erp

    _peaks = None

    def get_peaks(self):
        if self._peaks is not None:
            return self._peaks

        state_ref = self

        class _LazyPeaksList(list):
            def _compute_idx(self, idx: int):
                from scipy.signal import find_peaks

                dpts = state_ref.get_erp()
                if dpts is None or idx < 0 or idx >= len(dpts):
                    return np.empty(0, dtype=int)
                d = dpts[idx]
                if hasattr(d, "magnitude"):
                    d = d.magnitude
                d = np.asarray(d).flatten()
                pts, _ = find_peaks(d)
                pts_down, _ = find_peaks(-1 * d)
                return np.sort(np.hstack([pts, pts_down]).flatten())

            def __getitem__(self, index):
                if isinstance(index, int):
                    if index < 0:
                        index += len(self)
                    while len(self) <= index:
                        super().append(None)
                    val = super().__getitem__(index)
                    if val is None:
                        val = self._compute_idx(index)
                        super().__setitem__(index, val)
                    return val
                return super().__getitem__(index)

        erp = self.get_erp()
        n_sweeps = len(erp) if erp is not None else 0
        self._peaks = _LazyPeaksList([None] * n_sweeps)
        return self._peaks

    def set_window_size(self, window_size):
        self.window_size = window_size * pq.ms
        self._get_erp.cache_clear()
        self._notify_dataset_loaded("window")
        self.onLoadNewFile.emit()

    def get_unit_suggestions(self):
        # This will return a list of event suggestions provided by all the activated UnitSuggestion providers
        # This allows for custom algorithms to be implemented
        # The first one in the list will be the selected default.
        pass

    def set_data(
        self,
        analog_signal=None,
        event_signal=None,
        spike_groups=None,
    ):
        self._get_erp.cache_clear()
        self._peaks = None
        if analog_signal is not None:
            self.analog_signal = analog_signal
        if event_signal is not None:
            self.event_signal = event_signal
        if spike_groups is not None:
            self.spike_groups = spike_groups

        # ensure essentials are populated on state
        if self.spike_groups is None:
            spike_groups = [tracked_neuron_unit(event=Event())]
        if self.analog_signal is None:
            analog_signal = neo.AnalogSignal([], pq.mV, sampling_rate=1 * pq.Hz)
        if self.event_signal is None:
            event_signal = neo.Event()

        self._sync_stim_times_buffer()
        self.sampling_rate = int(np.array(self.analog_signal.sampling_rate.magnitude))
        if len(self.spike_groups) == 0:
            self.spike_groups.append(tracked_neuron_unit(event=Event()))

        self.update_idx_arrs()
        self._notify_dataset_loaded("load")
        self.onLoadNewFile.emit()
        self.onUnitListChanged.emit()
        self.undo_queue = []

    def getUnitGroup(self, unit=None):
        if unit is None:
            unit = self.cur_spike_group
        if (
            unit is None
            or not self.spike_groups
            or unit < 0
            or unit >= len(self.spike_groups)
        ):
            return None
        return self.spike_groups[unit]

    def set_segment(self, data: neo.Segment):
        if data is not None:
            data = expand_multichannel_segment(data)

        kwargs = {}
        cur_evt = [
            i for i, x in enumerate(data.events) if id(x) == id(self.event_signal)
        ]
        if len(cur_evt) == 0:
            event_signal = data.events[0]
            kwargs["event_signal"] = event_signal
        else:
            event_signal = data.events[cur_evt[0]]

        cur_analogsig = [
            i
            for i, x in enumerate(data.analogsignals)
            if id(x) == id(self.analog_signal)
        ]
        if len(cur_analogsig) == 0:
            analog_signal = data.analogsignals[0]
            kwargs["analog_signal"] = analog_signal
        else:
            analog_signal = data.analogsignals[cur_analogsig[0]]

        spike_event_groups = []

        for x in data.events:
            if not x.name.startswith("unit"):
                continue

            spike_event_groups.append(
                tracked_neuron_unit(
                    event=x,
                    channel=x.annotations.get("channel", ""),
                    neighbor_channels=list(x.annotations.get("neighbor_channels", [])),
                )
            )

        cur_sel = [
            i
            for i, x in enumerate(spike_event_groups)
            if id(x.event) == id(self.spike_groups[self.cur_spike_group])
        ]
        if len(cur_sel) == 0:
            self.cur_spike_group = 0
        else:
            self.cur_spike_group = cur_sel[0]

        self.segment = data
        self.event_signal = event_signal
        self.analog_signal = analog_signal
        self.restore_candidate_store(
            data.annotations.get("spikespy_candidates"), event_signal=event_signal
        )

        self.set_data(analog_signal, event_signal, spike_groups=spike_event_groups)

    def has_live_session(self) -> bool:
        """Returns True if there is an active or stopped live session with recorded sweeps or buffers."""
        return bool(
            getattr(self, "_live_erp_buffer", None) is not None
            or getattr(self, "_live_aux_erp_buffers", None)
            or (
                self.segment is not None
                and getattr(self.segment, "name", "") == "OpenEphys Live"
            )
        )

    def resume_live_session(self):
        """
        Resumes an existing live session without wiping recorded sweeps or spike annotations.
        """
        if not self.has_live_session():
            return
        self.is_streaming = True
        self._last_live_emitted_stimno = None
        self._sync_stim_times_buffer()
        if hasattr(self, "frame_scheduler") and self.frame_scheduler is not None:
            self.frame_scheduler.reset_metrics()
        if hasattr(self, "ui_watchdog") and self.ui_watchdog is not None:
            self.ui_watchdog.start()
        self.onStreamStateChanged.emit(True)

    def clear_live_session(self):
        """
        Explicitly clears and releases the current live session history and buffers.
        """
        self.stop_live_session()
        self._release_live_disk_cache()
        self._live_erp_buffer = None
        self._live_aux_erp_buffers = {}
        self._live_channel_signals = {}
        self.segment = None
        self.clear_candidates(emit=False)
        self.analog_signal = None
        self.event_signal = None
        self.spike_groups = []
        self.cur_spike_group = 0
        self.stimno = 0
        self._peaks = None
        self._get_erp.cache_clear()
        self.onLoadNewFile.emit()

    def start_live_session(
        self,
        sampling_rate: float = 30000.0,
        window_size_ms: float = 500.0,
        channel_name: str = "rd.0",
        units=pq.uV,
        aux_channels: List[str] = None,
        channels: List[Union[int, str]] = None,
        resume: bool = False,
        pre_ms: Optional[float] = None,
        pre_stim_ms: Optional[float] = None,
    ):
        """
        Initializes a fresh live streaming session for real-time visualization and annotation.
        If resume is True and a live session already exists, continues recording without wiping history.
        """
        if resume and self.has_live_session():
            self.resume_live_session()
            return

        chosen_pre_ms = (
            pre_ms
            if pre_ms is not None
            else (pre_stim_ms if pre_stim_ms is not None else None)
        )
        if chosen_pre_ms is None and hasattr(self, "live_config") and isinstance(self.live_config, dict):
            chosen_pre_ms = self.live_config.get("pre_ms", 0.0)
        self.pre_offset_s = max(0.0, float(chosen_pre_ms or 0.0) / 1000.0)

        if channels:
            ch_names = []
            for idx, c in enumerate(channels):
                if isinstance(c, int):
                    ch_names.append(
                        channel_name
                        if (idx == 0 and channel_name != "rd.0")
                        else f"CH{c}"
                    )
                else:
                    ch_names.append(str(c))
            channel_name = ch_names[0]
            self.active_channel_names = list(ch_names)
        else:
            self.active_channel_names = [channel_name]
            if aux_channels:
                for ch in aux_channels:
                    ch_str = str(ch)
                    if ch_str not in self.active_channel_names:
                        self.active_channel_names.append(ch_str)

        self._release_live_disk_cache()
        self.segment = neo.Segment(name="OpenEphys Live")
        self.clear_candidates(emit=False)
        self._live_channel_signals = {}
        self._live_aux_erp_buffers = {}
        total_window_ms = float(window_size_ms) + (self.pre_offset_s * 1000.0)
        window_samples = int(round(sampling_rate * total_window_ms / 1000.0))

        for ch in self.active_channel_names:
            sig = MultiAnalogSignal()
            sig.name = ch
            sig.sampling_rate = sampling_rate * pq.Hz
            sig.units = units
            self._live_channel_signals[ch] = sig
            self._live_aux_erp_buffers[ch] = LiveERPBuffer(
                disk_backed=self.live_disk_cache,
                window_samples=window_samples,
                units=units,
            )
            self.segment.analogsignals.append(sig)

        self.analog_signal = self._live_channel_signals[channel_name]
        self._live_erp_buffer = self._live_aux_erp_buffers[channel_name]

        self.event_signal = neo.Event(
            np.array([], dtype=float) * pq.s,
            name="env.stim",
            units=pq.s,
        )
        self.segment.events.append(self.event_signal)
        self._sync_stim_times_buffer()

        if self.spike_groups and len(self.spike_groups) > 0:
            for sg in self.spike_groups:
                sg.pre_offset_s = self.pre_offset_s
                sg.event = neo.Event(
                    np.array([], dtype=float) * pq.s,
                    name=sg.event.name,
                    units=pq.s,
                    **sg.event.annotations,
                )
                sg.idx_arr = []
                if not any(
                    e is sg.event or getattr(e, "name", "") == sg.event.name
                    for e in self.segment.events
                ):
                    self.segment.events.append(sg.event)
        else:
            evt = neo.Event(
                np.array([], dtype=float) * pq.s,
                name="unit_0",
                units=pq.s,
            )
            evt.annotations["type"] = "unit"
            evt.annotations["unit"] = 0
            evt.annotations["notes"] = ""
            evt.annotations["fibre_type"] = ""
            self.spike_groups = [
                tracked_neuron_unit(event=evt, idx_arr=[], pre_offset_s=self.pre_offset_s)
            ]
            self.segment.events.append(evt)

        self.cur_spike_group = 0
        self.stimno = 0
        self._last_live_emitted_stimno = None
        self.is_streaming = True
        self.sampling_rate = int(sampling_rate)
        self.window_size = (window_size_ms / 1000.0) * pq.s
        self._peaks = None
        self._get_erp.cache_clear()
        self._notify_dataset_loaded("live_start")
        if hasattr(self, "frame_scheduler") and self.frame_scheduler is not None:
            self.frame_scheduler.reset_metrics()
        if hasattr(self, "ui_watchdog") and self.ui_watchdog is not None:
            self.ui_watchdog.start()
        self.onLoadNewFile.emit()
        self.onStreamStateChanged.emit(True)

    def stop_live_session(self):
        """
        Stops the live streaming session. Preserves all recorded sweeps,
        channel signals, and spike annotations for viewing, editing, saving, or resuming.
        """
        self.flush_live_frame()
        self.is_streaming = False
        if hasattr(self, "ui_watchdog") and self.ui_watchdog is not None:
            self.ui_watchdog.stop()
        self.update_idx_arrs()
        self.onStreamStateChanged.emit(False)
        self.onEventsUpdated.emit()
        if self.stimno is not None:
            self.onStimNoChange.emit(self.stimno)

    def _reactivate_candidates_for_channel(self, channel: str) -> None:
        if not channel or not self.candidate_units:
            return
        revived = False
        for candidate in self.candidate_units.values():
            if (
                candidate.status == "discarded"
                and candidate.channel == channel
                and float(candidate.score) >= 0.5
            ):
                candidate.status = "pending"
                candidate.score_at_discard = None
                candidate.channel_at_discard = None
                revived = True
        if revived:
            self.onCandidatesChanged.emit()

    def set_active_channel(self, channel_name_or_sig: Union[str, neo.AnalogSignal]):
        """
        Switch the active displayed channel in real time during live streaming or offline mode.
        """
        if channel_name_or_sig is None:
            return

        if isinstance(channel_name_or_sig, str):
            ch_name = channel_name_or_sig
            if (
                hasattr(self, "_live_aux_erp_buffers")
                and self._live_aux_erp_buffers
                and ch_name in self._live_aux_erp_buffers
            ):
                if (
                    hasattr(self, "_live_channel_signals")
                    and ch_name in self._live_channel_signals
                ):
                    self.analog_signal = self._live_channel_signals[ch_name]
                elif self.analog_signal is not None:
                    self.analog_signal.name = ch_name
                self._live_erp_buffer = self._live_aux_erp_buffers[ch_name]
                self._peaks = None
                self._get_erp.cache_clear()
                self.update_idx_arrs()
                self.onChannelChange.emit(ch_name)
                self._reactivate_candidates_for_channel(ch_name)
                self._notify_dataset_loaded("channel")
                self.onLoadNewFile.emit()
                if self.event_signal is not None and len(self.event_signal) > 0:
                    self.onStimNoChange.emit(self.stimno)
                self.onEventsUpdated.emit()
                return

            # Offline mode by channel name
            found_sig = None
            if self.segment is not None and getattr(
                self.segment, "analogsignals", None
            ):
                for x in self.segment.analogsignals:
                    if getattr(x, "name", "") == ch_name:
                        found_sig = x
                        break
            if found_sig is not None:
                self.set_data(analog_signal=found_sig)
                self.onChannelChange.emit(found_sig.name)
                self._reactivate_candidates_for_channel(found_sig.name)
            else:
                self.onChannelChange.emit(ch_name)
        else:
            sig = channel_name_or_sig
            sig_name = getattr(sig, "name", "")
            if (
                hasattr(self, "_live_aux_erp_buffers")
                and self._live_aux_erp_buffers
                and sig_name in self._live_aux_erp_buffers
            ):
                self.analog_signal = sig
                self._live_erp_buffer = self._live_aux_erp_buffers[sig_name]
                self._peaks = None
                self._get_erp.cache_clear()
                self.update_idx_arrs()
                self.onChannelChange.emit(sig_name)
                self._reactivate_candidates_for_channel(sig_name)
                self._notify_dataset_loaded("channel")
                self.onLoadNewFile.emit()
                if self.event_signal is not None and len(self.event_signal) > 0:
                    self.onStimNoChange.emit(self.stimno)
                self.onEventsUpdated.emit()
            else:
                self.set_data(analog_signal=sig)
                self.onChannelChange.emit(sig_name)
                self._reactivate_candidates_for_channel(sig_name)

    def _schedule_live_frame(self, stim_idx: int):
        """
        Record that new sweeps arrived and emit onNewSweep/onStimNoChange/onEventsUpdated
        at most once per live_frame_interval_ms, however many sweeps arrived meanwhile.
        """
        self._live_frame_pending_idx = stim_idx
        if self.live_frame_interval_ms <= 0:
            self._emit_live_frame()
            return
        if self._live_frame_timer is None:
            self._live_frame_timer = QTimer(self)
            self._live_frame_timer.setSingleShot(True)
            self._live_frame_timer.timeout.connect(self._emit_live_frame)
        if not self._live_frame_timer.isActive():
            self._live_frame_timer.start(self.live_frame_interval_ms)

    def _emit_live_frame(self):
        idx = self._live_frame_pending_idx
        if idx is None:
            return
        self._live_frame_pending_idx = None
        self.onNewSweep.emit(idx)
        if self.follow_live:
            if getattr(self, "_last_live_emitted_stimno", None) != self.stimno:
                self._last_live_emitted_stimno = self.stimno
                self.onStimNoChange.emit(self.stimno)
        self.onEventsUpdated.emit()
        if (
            self.live_frame_interval_ms <= 0
            and not getattr(self, "_in_live_frame_process_events", False)
            and QApplication.instance() is not None
        ):
            self._in_live_frame_process_events = True
            try:
                QApplication.processEvents()
            finally:
                self._in_live_frame_process_events = False

    def flush_live_frame(self):
        """Emit any pending coalesced live update immediately."""
        if self._live_frame_timer is not None:
            self._live_frame_timer.stop()
        self._emit_live_frame()

    def _store_raw_sweep(self, ch_name: str, wave) -> np.ndarray:
        """Return a float32 1D array for a raw sweep, disk-backed when live_disk_cache is on."""
        arr = np.asarray(getattr(wave, "magnitude", wave), dtype=np.float32).ravel()
        if not self.live_disk_cache:
            return arr
        store = self._live_raw_stores.get(ch_name)
        if store is None:
            store = DiskRowStore(len(arr), np.float32)
            self._live_raw_stores[ch_name] = store
        if store.row_len != len(arr):
            return arr.copy()  # variable-length sweep: keep in RAM
        return store.append(arr)

    def _release_live_disk_cache(self):
        """Drop references to the previous session's disk stores and remove stale cache files."""
        for store in getattr(self, "_live_raw_stores", {}).values():
            store.close()
        self._live_raw_stores = {}
        clear_live_cache_dir()

    def _sync_stim_times_buffer(self):
        """Rebuild the growable stimulus buffer from the current event_signal."""
        ev = self.event_signal
        self._stim_times_event = ev
        if ev is None:
            self._stim_times_buf = np.empty(0, dtype=np.float64)
            self._stim_times_count = 0
            return
        arr = ev.as_array() if hasattr(ev, "as_array") else np.asarray(ev)
        arr = np.ascontiguousarray(arr, dtype=np.float64).ravel()
        self._stim_times_buf = arr.copy()
        self._stim_times_count = arr.size

    def _grow_stim_times_buffer(self, min_capacity: int):
        """Double the stimulus buffer capacity, preserving live Event views on the old one."""
        cap = self._stim_times_buf.size if self._stim_times_buf is not None else 0
        new_cap = max(16, cap * 2, int(min_capacity))
        new_buf = np.empty(new_cap, dtype=np.float64)
        if cap > 0:
            new_buf[: self._stim_times_count] = self._stim_times_buf[
                : self._stim_times_count
            ]
        self._stim_times_buf = new_buf

    def append_sweep(
        self,
        stim_timestamp: Union[float, pq.Quantity],
        sweep_waveform: np.ndarray,
        pre_offset_s: float = 0.0,
        spike_latencies_s: dict = None,
        aux_sweeps: dict = None,
        emit_signals: bool = True,
    ) -> int:
        """
        Append a new stimulus sweep to the live session.
        Supports 1D (window_samples,) and 2D (n_channels, window_samples) waveforms.
        """
        if isinstance(stim_timestamp, (int, float)):
            stim_ts_q = stim_timestamp * pq.s
        else:
            stim_ts_q = stim_timestamp.rescale(pq.s)

        # Append stim_timestamp to self.event_signal (updating self.segment.events[0] as well)
        ev = self.event_signal
        if ev is None:
            ev = neo.Event(
                np.array([], dtype=float) * pq.s, name="env.stim", units=pq.s
            )
            self.event_signal = ev
        if self._stim_times_event is not ev or self._stim_times_count != len(ev):
            self._sync_stim_times_buffer()
        new_stim_idx = self._stim_times_count
        if new_stim_idx >= self._stim_times_buf.size:
            self._grow_stim_times_buffer(new_stim_idx + 1)
        self._stim_times_buf[new_stim_idx] = float(stim_ts_q.magnitude)
        self._stim_times_count = new_stim_idx + 1
        new_event = neo.Event(
            self._stim_times_buf[: new_stim_idx + 1],
            name=ev.name,
            units=pq.s,
            **ev.annotations,
        )
        if self.segment is not None and len(self.segment.events) > 0:
            for idx_ev, seg_ev in enumerate(self.segment.events):
                if seg_ev.name == ev.name or id(seg_ev) == id(ev):
                    self.segment.events[idx_ev] = new_event
                    break
        self.event_signal = new_event
        self._stim_times_event = new_event

        units = self.analog_signal.units if self.analog_signal is not None else pq.uV
        sr = (
            self.sampling_rate * pq.Hz
            if not hasattr(self.sampling_rate, "rescale")
            else self.sampling_rate
        )

        sr_val = (
            float(self.sampling_rate.rescale(pq.Hz).magnitude)
            if hasattr(self.sampling_rate, "rescale")
            else float(self.sampling_rate or 30000.0)
        )
        is_2d = hasattr(sweep_waveform, "ndim") and sweep_waveform.ndim == 2
        pre_idx = int(round(pre_offset_s * sr_val)) if pre_offset_s > 0 else 0

        new_channels_added = False
        if is_2d:
            n_ch = sweep_waveform.shape[0]
            for ch_i in range(n_ch):
                ch_name = (
                    self.active_channel_names[ch_i]
                    if hasattr(self, "active_channel_names")
                    and ch_i < len(self.active_channel_names)
                    else f"CH{ch_i}"
                )
                ch_wave = sweep_waveform[ch_i]

                if (
                    hasattr(self, "_live_aux_erp_buffers")
                    and ch_name not in self._live_aux_erp_buffers
                ):
                    w_samples = (
                        self._live_erp_buffer.window_samples
                        if self._live_erp_buffer is not None
                        else int(
                            sr_val
                            * float(
                                self.window_size.rescale(pq.s).magnitude
                                if hasattr(self.window_size, "magnitude")
                                else self.window_size
                            )
                        )
                    )
                    self._live_aux_erp_buffers[ch_name] = LiveERPBuffer(
                        disk_backed=self.live_disk_cache,
                        window_samples=w_samples,
                        units=units,
                    )
                    new_sig = MultiAnalogSignal()
                    new_sig.name = ch_name
                    new_sig.sampling_rate = sr
                    new_sig.units = units
                    if hasattr(self, "_live_channel_signals"):
                        self._live_channel_signals[ch_name] = new_sig
                    if self.segment is not None and hasattr(
                        self.segment, "analogsignals"
                    ):
                        if new_sig not in self.segment.analogsignals:
                            self.segment.analogsignals.append(new_sig)
                    if (
                        hasattr(self, "active_channel_names")
                        and ch_name not in self.active_channel_names
                    ):
                        self.active_channel_names.append(ch_name)
                    new_channels_added = True

                sig_slice = neo.AnalogSignal(
                    self._store_raw_sweep(ch_name, ch_wave),
                    units=units,
                    sampling_rate=sr,
                    t_start=stim_ts_q - (pre_offset_s * pq.s),
                    name=ch_name,
                )
                if (
                    hasattr(self, "_live_channel_signals")
                    and ch_name in self._live_channel_signals
                ):
                    self._live_channel_signals[ch_name].append_signal(sig_slice)
                elif ch_name == getattr(self.analog_signal, "name", "") and isinstance(
                    self.analog_signal, MultiAnalogSignal
                ):
                    self.analog_signal.append_signal(sig_slice)

                if (
                    hasattr(self, "_live_aux_erp_buffers")
                    and ch_name in self._live_aux_erp_buffers
                ):
                    aux_buf = self._live_aux_erp_buffers[ch_name]
                    aux_buf.append_sweep(ch_wave)
                elif (
                    ch_name == getattr(self.analog_signal, "name", "")
                    and self._live_erp_buffer is not None
                ):
                    self._live_erp_buffer.append_sweep(ch_wave)

            primary_waveform = (
                sweep_waveform[0]
                if (
                    not hasattr(self, "active_channel_names")
                    or self.analog_signal is None
                    or self.analog_signal.name not in self.active_channel_names
                )
                else sweep_waveform[
                    self.active_channel_names.index(self.analog_signal.name)
                ]
            )
        else:
            # 1D waveform
            sig_slice = neo.AnalogSignal(
                self._store_raw_sweep(
                    getattr(self.analog_signal, "name", "rd.0"), sweep_waveform
                ),
                units=units,
                sampling_rate=sr,
                t_start=stim_ts_q - (pre_offset_s * pq.s),
                name=self.analog_signal.name
                if self.analog_signal is not None
                else "rd.0",
            )
            if (
                hasattr(self, "_live_channel_signals")
                and getattr(self.analog_signal, "name", "")
                in self._live_channel_signals
            ):
                self._live_channel_signals[self.analog_signal.name].append_signal(
                    sig_slice
                )
            elif isinstance(self.analog_signal, MultiAnalogSignal):
                self.analog_signal.append_signal(sig_slice)

            # Stimulus-locked ERP buffer stores the full sweep waveform
            cur_ch_name = getattr(self.analog_signal, "name", "")
            if (
                hasattr(self, "_live_aux_erp_buffers")
                and cur_ch_name in self._live_aux_erp_buffers
            ):
                aux_buf = self._live_aux_erp_buffers[cur_ch_name]
                aux_buf.append_sweep(sweep_waveform)
            elif self._live_erp_buffer is not None:
                self._live_erp_buffer.append_sweep(sweep_waveform)

            # Handle aux sweeps if any
            if aux_sweeps:
                for aux_ch, aux_wave in aux_sweeps.items():
                    if aux_ch == cur_ch_name:
                        continue  # Do not double-append to primary
                    if hasattr(self, "_live_aux_erp_buffers"):
                        if aux_ch not in self._live_aux_erp_buffers:
                            w_samples = (
                                self._live_erp_buffer.window_samples
                                if self._live_erp_buffer is not None
                                else int(
                                    sr_val
                                    * float(
                                        self.window_size.rescale(pq.s).magnitude
                                        if hasattr(self.window_size, "magnitude")
                                        else self.window_size
                                    )
                                )
                            )
                            self._live_aux_erp_buffers[aux_ch] = LiveERPBuffer(
                                disk_backed=self.live_disk_cache,
                                window_samples=w_samples,
                                units=units,
                            )
                            new_sig = MultiAnalogSignal()
                            new_sig.name = aux_ch
                            new_sig.sampling_rate = sr
                            new_sig.units = units
                            if hasattr(self, "_live_channel_signals"):
                                self._live_channel_signals[aux_ch] = new_sig
                            if self.segment is not None and hasattr(
                                self.segment, "analogsignals"
                            ):
                                if new_sig not in self.segment.analogsignals:
                                    self.segment.analogsignals.append(new_sig)
                            if (
                                hasattr(self, "active_channel_names")
                                and aux_ch not in self.active_channel_names
                            ):
                                self.active_channel_names.append(aux_ch)
                            new_channels_added = True

                        aux_buf = self._live_aux_erp_buffers[aux_ch]
                        aux_buf.append_sweep(aux_wave)
                    if (
                        hasattr(self, "_live_channel_signals")
                        and aux_ch in self._live_channel_signals
                    ):
                        aux_sig = neo.AnalogSignal(
                            self._store_raw_sweep(aux_ch, aux_wave),
                            units=units,
                            sampling_rate=sr,
                            t_start=stim_ts_q - (pre_offset_s * pq.s),
                            name=aux_ch,
                        )
                        self._live_channel_signals[aux_ch].append_signal(aux_sig)

            primary_waveform = sweep_waveform

        if new_channels_added and hasattr(self, "onChannelGroupsChanged"):
            self.onChannelGroupsChanged.emit()

        # Maintain lazy peak slot (computed on demand in get_peaks()[idx] only when snap navigation is used)
        if self._peaks is not None:
            self._peaks.append(None)

        # Appends None to sg.idx_arr for each sg in self.spike_groups
        for sg in self.spike_groups:
            if sg.idx_arr is None:
                sg.idx_arr = []
            while len(sg.idx_arr) < new_stim_idx:
                sg.idx_arr.append(None)
            sg.idx_arr.append(None)

        # If spike_latencies_s is provided, records those spikes in corresponding tracked_neuron_unit
        if spike_latencies_s:
            for key, lat in spike_latencies_s.items():
                sg = None
                sg_idx = None
                if isinstance(key, int) and 0 <= key < len(self.spike_groups):
                    sg = self.spike_groups[key]
                    sg_idx = key
                else:
                    unit_name_str = str(key)
                    for idx_u, u in enumerate(self.spike_groups):
                        if u.event.name == unit_name_str:
                            sg = u
                            sg_idx = idx_u
                            break
                    if sg is None:
                        evt = neo.Event(
                            np.array([], dtype=float) * pq.s,
                            name=unit_name_str,
                            units=pq.s,
                        )
                        primary_ch = getattr(self.analog_signal, "name", "CH0")
                        neighbors = [
                            c
                            for c in getattr(self, "active_channel_names", [])
                            if c != primary_ch
                        ]
                        evt.annotations["type"] = "unit"
                        evt.annotations["unit"] = len(self.spike_groups)
                        evt.annotations["notes"] = ""
                        evt.annotations["fibre_type"] = "C-fibre"
                        evt.annotations["channel"] = primary_ch
                        evt.annotations["neighbor_channels"] = neighbors
                        sg = tracked_neuron_unit(
                            event=evt,
                            channel=primary_ch,
                            neighbor_channels=neighbors,
                            fibre_type="C-fibre",
                            notes="",
                            idx_arr=[None] * (new_stim_idx + 1),
                            pre_offset_s=self.pre_offset_s,
                        )
                        self.spike_groups.append(sg)
                        if self.segment is not None:
                            self.segment.events.append(evt)
                        sg_idx = len(self.spike_groups) - 1
                if sg is not None and lat is not None:
                    lat_qty = (
                        lat * pq.s if not hasattr(lat, "rescale") else lat.rescale(pq.s)
                    )
                    spike_time = stim_ts_q + lat_qty
                    lat_samples = int(
                        np.round(
                            (
                                lat_qty * (self.sampling_rate * pq.Hz)
                            ).simplified.magnitude
                        )
                    )
                    sample_idx = pre_idx + lat_samples
                    sg.idx_arr[new_stim_idx] = (sample_idx, 1)
                    new_ev_times = (
                        np.append(sg.event.as_array(), spike_time.magnitude) * pq.s
                    )
                    sg.event = neo.Event(
                        new_ev_times,
                        name=sg.event.name,
                        units=pq.s,
                        **sg.event.annotations,
                    )
                    sg.event.sort()
                    if self.segment is not None:
                        for seg_i, seg_ev in enumerate(self.segment.events):
                            if seg_ev.name == sg.event.name:
                                self.segment.events[seg_i] = sg.event
                                break

        # If self.follow_live is True, updates self.stimno = new_stim_idx
        if self.follow_live:
            self.stimno = new_stim_idx

        # Live multi-unit auto-tracking (asynchronous on background thread pool).
        # Register the state-owned processor first so it is the single live
        # tracking path regardless of which views are open.
        wants_tracking = getattr(self, "auto_track_enabled", False) or any(
            getattr(sg, "auto_track", False) for sg in self.spike_groups
        )
        if wants_tracking:
            try:
                self.ensure_live_tracking_processor()
            except Exception:
                logger.exception("Unable to register live tracking processor")

        # Avoid race if live_orchestrator already has a tracking function registered
        orch = getattr(self, "_live_orchestrator", None)
        has_orch_tracker = False
        if orch is not None:
            if getattr(orch, "_tracking_fn", None) is not None:
                has_orch_tracker = True
            elif hasattr(orch, "_registrations"):
                with orch._cond:
                    has_orch_tracker = any(
                        reg.name == "auto_track" for reg in orch._registrations.values()
                    )

        if not has_orch_tracker:
            if wants_tracking:
                self.tracking_manager.track_stimulus_async(self, new_stim_idx)

        # Feed the raw sweep to view-registered orchestrator processors.
        self._submit_live_item(
            new_stim_idx,
            stim_ts_q,
            sweep_waveform,
            aux_sweeps,
            spike_latencies_s,
            pre_offset_s,
            is_2d,
            sr_val,
        )

        if emit_signals:
            # Coalesce UI notifications: views redraw once per frame, not once per sweep
            self._schedule_live_frame(new_stim_idx)

        return new_stim_idx

    def _submit_live_item(
        self,
        stim_idx: int,
        stim_ts_q,
        sweep_waveform,
        aux_sweeps,
        spike_latencies_s,
        pre_offset_s: float,
        is_2d: bool,
        sr_val: float,
    ):
        """Submit one ingested sweep to the live orchestrator (no-op if no view registered)."""
        orch = self._live_orchestrator
        if orch is None:
            return

        channel_sweeps = {}
        if is_2d and hasattr(sweep_waveform, "shape") and sweep_waveform.ndim == 2:
            names = list(getattr(self, "active_channel_names", None) or [])
            for ch_i in range(sweep_waveform.shape[0]):
                ch_name = names[ch_i] if ch_i < len(names) else f"CH{ch_i}"
                channel_sweeps[ch_name] = sweep_waveform[ch_i]
        elif sweep_waveform is not None:
            ch_name = getattr(self.analog_signal, "name", "") or "CH0"
            channel_sweeps[ch_name] = sweep_waveform
        for aux_ch, aux_wave in (aux_sweeps or {}).items():
            channel_sweeps.setdefault(str(aux_ch), aux_wave)

        ts_s = float(getattr(stim_ts_q, "magnitude", stim_ts_q))

        orch.submit(
            LiveSweepItem(
                stim_index=stim_idx,
                stim_timestamp_s=ts_s,
                channel_sweeps=channel_sweeps,
                spike_latencies_s=spike_latencies_s,
                pre_offset_s=pre_offset_s or 0.0,
                sampling_rate=sr_val,
            ),
            synchronous=self.live_frame_interval_ms <= 0,
        )

    def _notify_dataset_loaded(self, reason: str):
        """Submit a static dataset (re)load event to the live orchestrator.

        This keeps static file processing in the same pipeline as live
        sweeps (single point of view processing); it is a no-op when no view
        has registered with the orchestrator yet.
        """
        orch = self._live_orchestrator
        if orch is None:
            return
        from .live_orchestrator import LiveReloadItem

        sr = (
            float(np.array(self.sampling_rate.magnitude))
            if self.sampling_rate is not None
            and hasattr(self.sampling_rate, "magnitude")
            else None
        )
        n_sweeps = len(self.event_signal) if self.event_signal is not None else 0
        orch.submit(
            LiveReloadItem(
                reason=reason,
                n_sweeps=n_sweeps,
                sampling_rate=sr,
                segment_name=getattr(self.segment, "name", "") or "",
            ),
            synchronous=self.live_frame_interval_ms <= 0,
        )

    def append_unit_spikes(
        self,
        unit_name: str,
        spike_timestamps_s: Union[np.ndarray, pq.Quantity],
        create_if_missing: bool = True,
    ):
        """
        Append spike timestamps to a unit, updating its event and idx_arr.
        """
        unit_idx = None
        sg = None
        for i, u in enumerate(self.spike_groups):
            if u.event.name == unit_name:
                unit_idx = i
                sg = u
                break

        if sg is None:
            if create_if_missing:
                evt = neo.Event(
                    np.array([], dtype=float) * pq.s,
                    name=unit_name,
                    units=pq.s,
                )
                primary_ch = getattr(self.analog_signal, "name", "CH0")
                neighbors = [
                    c
                    for c in getattr(self, "active_channel_names", [])
                    if c != primary_ch
                ]
                evt.annotations["type"] = "unit"
                evt.annotations["unit"] = len(self.spike_groups)
                evt.annotations["notes"] = ""
                evt.annotations["fibre_type"] = "C-fibre"
                evt.annotations["channel"] = primary_ch
                evt.annotations["neighbor_channels"] = neighbors
                sg = tracked_neuron_unit(
                    event=evt,
                    channel=primary_ch,
                    neighbor_channels=neighbors,
                    fibre_type="C-fibre",
                    notes="",
                    idx_arr=[None]
                    * (len(self.event_signal) if self.event_signal is not None else 0),
                )
                self.spike_groups.append(sg)
                if self.segment is not None:
                    self.segment.events.append(evt)
                unit_idx = len(self.spike_groups) - 1
            else:
                return

        if not hasattr(spike_timestamps_s, "rescale"):
            spikes_q = np.atleast_1d(spike_timestamps_s) * pq.s
        else:
            spikes_q = np.atleast_1d(spike_timestamps_s).rescale(pq.s)

        spikes_mag = (
            spikes_q.magnitude
            if hasattr(spikes_q, "magnitude")
            else np.asarray(spikes_q)
        )
        ev_mag = (
            sg.event.as_array()
            if hasattr(sg.event, "as_array")
            else np.asarray(sg.event)
        )

        if len(sg.event) == 0:
            combined_mag = np.asarray(spikes_mag, dtype=np.float64)
        elif len(spikes_mag) == 0:
            combined_mag = np.asarray(ev_mag, dtype=np.float64)
        else:
            combined_mag = np.concatenate(
                [
                    np.asarray(ev_mag, dtype=np.float64),
                    np.asarray(spikes_mag, dtype=np.float64),
                ]
            )

        if combined_mag.size > 1 and not np.all(combined_mag[1:] >= combined_mag[:-1]):
            unique_mag = np.unique(combined_mag)
        elif combined_mag.size > 1:
            keep = np.empty(combined_mag.size, dtype=bool)
            keep[0] = True
            np.not_equal(combined_mag[1:], combined_mag[:-1], out=keep[1:])
            unique_mag = combined_mag[keep]
        else:
            unique_mag = combined_mag

        sg.event = neo.Event(
            unique_mag * pq.s,
            name=sg.event.name,
            units=pq.s,
            **sg.event.annotations,
        )
        sg.event.sort()

        if self.segment is not None:
            for seg_i, seg_ev in enumerate(self.segment.events):
                if seg_ev.name == sg.event.name:
                    self.segment.events[seg_i] = sg.event
                    break

        if self.event_signal is not None and len(self.event_signal) > 0:
            if sg.idx_arr is None:
                sg.idx_arr = []
            while len(sg.idx_arr) < len(self.event_signal):
                sg.idx_arr.append(None)

            stim_mag = self.event_signal.rescale(pq.s).magnitude
            spikes_mag_s = spikes_q.rescale(pq.s).magnitude
            idx = np.searchsorted(stim_mag, spikes_mag_s, side="right") - 1
            valid = (idx >= 0) & (idx < len(stim_mag))
            if np.any(valid):
                clipped = np.clip(idx, 0, len(stim_mag) - 1)
                lat_s = spikes_mag_s - stim_mag[clipped]
                window_s = (
                    float(self.window_size.rescale(pq.s).magnitude)
                    if hasattr(self.window_size, "rescale")
                    else float(self.window_size)
                )
                win_mask = valid & (lat_s >= 0) & (lat_s <= window_s)
                if np.any(win_mask):
                    sr_src = (
                        self.sampling_rate
                        if self.sampling_rate is not None
                        else (
                            self.analog_signal.sampling_rate
                            if self.analog_signal is not None
                            else 30000
                        )
                    )
                    sr_val = (
                        float(sr_src.rescale(pq.Hz).magnitude)
                        if hasattr(sr_src, "rescale")
                        else float(sr_src)
                    )
                    lat_samples = np.round(lat_s[win_mask] * sr_val)
                    for i, lat_val in zip(idx[win_mask], lat_samples):
                        sg.idx_arr[int(i)] = (int(lat_val), 1)

        self.onNewSpikeEvents.emit(unit_idx)
        self.onEventsUpdated.emit()


class APPredictor:
    def __init__(self, state: ViewerState):
        self.state = state
        self.state.onLoadNewFile.connect(self.update)
        self.state.onUnitGroupChange.connect(self.update)
        self.state.onUnitChange.connect(self.update)

    def update(self):
        # return an event signal with the predicted action potentials
        pass
