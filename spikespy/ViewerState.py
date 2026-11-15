import logging
import sys
import tempfile
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
import time
from tempfile import tempdir
from turtle import update
from typing import Any, List, Optional, Tuple, Union

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
    TypeID,
)
from .NeoOpenEphyisIO import open_ephys_to_neo
from .io_manager import (
    APTrackDialog,
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
from .live_orchestrator import LiveSweepItem
from .tracking_worker import LiveTrackingManager

lru_numpy_memmap = BoundedERPCache


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
    event: Event = None
    auto_track: bool = False
    tracking_window_ms: float = 5.0
    tracking_threshold_mv: float = 0.01
    tracking_max_skips: int = 1
    tracking_channel: str = ""
    tracking_algorithm: str = "basic"

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
    ):
        self.event = event if event is not None else Event()
        self.channel = channel
        self.neighbor_channels = list(neighbor_channels) if neighbor_channels is not None else []
        self.fibre_type = fibre_type
        self.notes = notes
        self._idx_arr = idx_arr

        ann = getattr(self.event, "annotations", {})
        self.auto_track = bool(ann.get("auto_track", auto_track))
        self.tracking_window_ms = float(ann.get("tracking_window_ms", tracking_window_ms))
        self.tracking_threshold_mv = float(ann.get("tracking_threshold_mv", tracking_threshold_mv))
        self.tracking_max_skips = int(ann.get("tracking_max_skips", tracking_max_skips))
        self.tracking_channel = str(ann.get("tracking_channel", tracking_channel))
        self.tracking_algorithm = str(ann.get("tracking_algorithm", tracking_algorithm))

        # Ensure annotations dictionary has these synced
        if hasattr(self.event, "annotations"):
            self.event.annotations["auto_track"] = self.auto_track
            self.event.annotations["tracking_window_ms"] = self.tracking_window_ms
            self.event.annotations["tracking_threshold_mv"] = self.tracking_threshold_mv
            self.event.annotations["tracking_max_skips"] = self.tracking_max_skips
            self.event.annotations["tracking_channel"] = self.tracking_channel
            self.event.annotations["tracking_algorithm"] = self.tracking_algorithm

    @property
    def idx_arr(self):
        return self._idx_arr

    @idx_arr.setter
    def idx_arr(self, value):
        self._idx_arr = value

    @idx_arr.deleter
    def idx_arr(self):
        self._idx_arr = None

    def get_number_of_events(self, event_signal=None, window_size=None):
        if self.event is not None and len(self.event) > 0:
            if event_signal is not None and len(event_signal) > 0 and window_size is not None:
                ev_s = self.event.rescale(pq.s).magnitude
                stim_s = event_signal.rescale(pq.s).magnitude
                max_lat_s = float(
                    window_size.rescale(pq.s).magnitude
                    if hasattr(window_size, "rescale")
                    else float(window_size)
                )
                idx_sweeps = np.searchsorted(stim_s, ev_s, side="right") - 1
                valid_sw = (idx_sweeps >= 0) & (idx_sweeps < len(stim_s))
                dt = ev_s[valid_sw] - stim_s[idx_sweeps[valid_sw]]
                in_win = (dt >= 0) & (dt <= max_lat_s)
                return int(len(np.unique(idx_sweeps[valid_sw][in_win])))
            return len(self.event)
        if self._idx_arr is not None:
            return len([x for x in self._idx_arr if x is not None])
        return 0

    def get_window(self, event_signal=None, sampling_rate=None, window_size=None):
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
                return (int(np.min(samples)), int(np.max(samples)))

        if self._idx_arr is not None:
            idx_arr_filt = [
                (x[0] if isinstance(x, (tuple, list)) else x)
                for x in self._idx_arr
                if x is not None
            ]
            if len(idx_arr_filt) > 0:
                return (int(min(idx_arr_filt)), int(max(idx_arr_filt)))
        return None

    def get_sweep_sample_latencies(
        self,
        event_signal: neo.Event,
        sampling_rate: float,
        window_size: pq.Quantity = None,
    ) -> np.ndarray:
        """Returns a 1D float64 array of length len(event_signal) containing the spike latency
        in samples for each sweep (np.nan where no valid spike exists within [0, window_size]).
        """
        if event_signal is None or len(event_signal) == 0:
            return np.empty(0, dtype=np.float64)

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

        idx_sweeps = np.searchsorted(stim_s, ev_s, side="right") - 1
        valid_mask = (idx_sweeps >= 0) & (idx_sweeps < n_sweeps)
        if not np.any(valid_mask):
            return lat_samples

        sw = idx_sweeps[valid_mask]
        dt = ev_s[valid_mask] - stim_s[sw]
        in_win = (dt >= 0) & (dt <= max_lat_s)
        sw_win = sw[in_win]
        dt_win = dt[in_win]
        samp = np.round(dt_win * sr)

        for s_idx, sample_val in zip(sw_win, samp):
            if np.isnan(lat_samples[s_idx]):
                lat_samples[s_idx] = sample_val

        return lat_samples

    def get_valid_sweep_indices_and_samples(
        self,
        event_signal: neo.Event,
        sampling_rate: float,
        window_size: pq.Quantity = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (sweep_indices, sample_latencies) as 1D int64 arrays for all sweeps
        that have a valid spike within [0, window_size].
        """
        lat_samples = self.get_sweep_sample_latencies(event_signal, sampling_rate, window_size)
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

        idx = np.searchsorted(ev_s, stim_ts)
        if idx < len(ev_s):
            dt = ev_s[idx] - stim_ts
            if 0 <= dt <= max_lat_s:
                return int(np.round(dt * sr))

        return None

    def get_idx_arr(self, event_signal, signal_chan):
        p = [None for _ in range(len(event_signal))]
        if self.event is None or len(self.event) == 0:
            return p
        sr = float(signal_chan.sampling_rate.rescale(pq.Hz).magnitude)
        sweeps, samples = self.get_valid_sweep_indices_and_samples(event_signal, sr)
        for sw, sa in zip(sweeps, samples):
            p[sw] = (int(sa), 1)
        return p

    def get_latencies(self, event_signal: pq.UnitTime) -> pq.UnitTime:
        """Returns the latencies of the events in the spikegroup in ms.
        If the event is not in the spikegroup, returns np.nan.
        """
        if self.event is None or len(self.event.times) == 0:
            return np.ones(len(event_signal)) * np.nan * pq.ms
        if len(event_signal) == 0:
            return np.empty(0) * pq.ms

        next_spikes = np.searchsorted(self.event.times, event_signal).magnitude
        next_spikes = np.clip(next_spikes, 0, len(self.event.times) - 1)
        spike_times = self.event.times[next_spikes] - (
            np.asarray(event_signal) * event_signal.units
        )
        spike_times = spike_times.rescale(pq.ms)
        spike_times[spike_times < (0 * pq.s)] = np.nan * pq.ms
        return spike_times

    def get_sweeps(self, event_signal: pq.UnitTime) -> pq.UnitTime:
        """Returns the sweep numbers/event numbers in the spikegroup."""
        if self.event is None or len(self.event.times) == 0 or len(event_signal) == 0:
            return np.nan
        sweeps = np.searchsorted(event_signal, self.event.times).magnitude
        return sweeps


class ViewerState(QObject):
    onLoadNewFile = Signal()
    onUnitChange = Signal(int)
    onUnitGroupChange = Signal()
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

    def __init__(self, parent=None, segment=None, spike_groups=None):
        super().__init__(parent)

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

        # Channel groups storage: name -> list of channel names
        self.channel_groups: dict = {}

        # Shared channel sets for coordinated multi-panel viewing
        self.shared_channel_sets: dict = {
            "Shared Group A": {"channels": [], "group_name": "", "unit_idx": None, "multichannel": True},
            "Shared Group B": {"channels": [], "group_name": "", "unit_idx": None, "multichannel": True},
        }

        # Live streaming state attributes
        self.is_streaming: bool = False
        self.follow_live: bool = True
        self.auto_track_enabled: bool = False
        self.auto_switch_unit_channel: bool = False
        self._live_erp_buffer: Optional[LiveERPBuffer] = None
        # Store live sweeps in memory-mapped temp files instead of RAM
        self.live_disk_cache: bool = True
        self._live_frame_timer = None
        self._live_frame_pending_idx = None
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

    @Slot(int, int, int)
    def _on_async_spike_tracked(self, unit_idx: int, detected_sample: int, stimno: int):
        self.setUnit(detected_sample, stimno=stimno, unit_idx=unit_idx)

    def set_ui_refresh_rate_ms(self, interval_ms: int):
        self.ui_refresh_rate_ms = max(5, int(interval_ms))
        self.onUIRefreshRateChange.emit(self.ui_refresh_rate_ms)

    def get_ui_refresh_rate_ms(self) -> int:
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
        if not primary and hasattr(sg, "event") and getattr(sg.event, "annotations", None):
            primary = sg.event.annotations.get("channel", "")
        neighbors = getattr(sg, "neighbor_channels", []) or []
        if not neighbors and hasattr(sg, "event") and getattr(sg.event, "annotations", None):
            neighbors = sg.event.annotations.get("neighbor_channels", []) or []

        chans = []
        if primary:
            chans.append(str(primary))
        for c in neighbors:
            c_str = str(c).strip()
            if c_str and c_str not in chans:
                chans.append(c_str)
        if not chans and self.analog_signal is not None and getattr(self.analog_signal, "name", ""):
            chans.append(self.analog_signal.name)
        return chans

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
            }
        s = dict(self.shared_channel_sets[set_name])
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
        self.shared_channel_sets[set_name] = {
            "channels": cleaned,
            "group_name": group_name,
            "unit_idx": unit_idx,
            "multichannel": multichannel,
        }
        self.onSharedChannelSetChanged.emit(set_name)

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

        curr_i = group_names.index(current_group) if current_group in group_names else -1
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

    def auto_create_channel_groups(self, group_size: int = 4, prefix: str = "Group") -> dict:
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
        if self.is_streaming and self._live_aux_erp_buffers:
            channels = []
            if self.analog_signal is not None and getattr(self.analog_signal, "name", None):
                channels.append(self.analog_signal.name)
            for k in self._live_aux_erp_buffers.keys():
                if k not in channels:
                    channels.append(k)
            return channels
        elif self.segment is not None and len(self.segment.analogsignals) > 0:
            return [sig.name for sig in self.segment.analogsignals]
        else:
            return [self.analog_signal.name] if self.analog_signal is not None and getattr(self.analog_signal, "name", None) else []

    def get_channel_erp(self, channel_name: str = None):
        if channel_name is None or (self.analog_signal is not None and channel_name == self.analog_signal.name):
            return self.get_erp()
        if hasattr(self, "_live_aux_erp_buffers") and self._live_aux_erp_buffers and channel_name in self._live_aux_erp_buffers:
            buf = self._live_aux_erp_buffers[channel_name]
            return buf.data if len(buf) > 0 else np.zeros((0, buf.window_samples)) * buf.units
        if self.segment is not None:
            for sig in self.segment.analogsignals:
                if sig.name == channel_name:
                    return self.get_erp(signal=sig, event_signal=self.event_signal)
        return None

    def get_channel_signal(self, channel_name: str = None):
        if channel_name is None:
            return self.analog_signal
        if getattr(self, "is_streaming", False) and hasattr(self, "_live_channel_signals") and channel_name in self._live_channel_signals:
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

    @Slot(int, int, int)
    @Slot(int, int)
    @Slot(int)
    @Slot()
    def setUnit(self, latency, stimno=None, unit_idx: Optional[int] = None, unitid: Optional[int] = None):
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

        if len(evt) > 0:
            to_keep = (evt < self.event_signal[stimno]) | (
                (evt > self.event_signal[stimno + 1])
                if stimno < len(self.event_signal) - 1
                else False
            )  # Prevent crash on final index
            # TODO: we should remove by index, and also use purely the timestamps not idx_arr
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
                    prev_event_time = self.event_signal.times[
                        stimno - 1 - attempts
                    ]
                    prv_lat = sg.get_latencies(
                        np.array([prev_event_time]) * prev_event_time.units
                    )[0].rescale("s")
                    if prv_lat != np.nan:
                        latency = int(
                            (prv_lat * self.analog_signal.sampling_rate).magnitude
                        )
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
                        (next_lat * self.analog_signal.sampling_rate).magnitude
                    )

            sg.idx_arr[stimno] = (
                latency,
                1,
            )  # TODO: fix
            evt2 = evt.merge(
                Event(
                    ((latency / self.analog_signal.sampling_rate).rescale(pq.s))
                    + self.event_signal[[stimno]],
                )
            )
            evt2.name = evt.name
            sg.event = evt2

            # When annotating a spike on a unit, if sg.channel == "" and self.analog_signal is not None:
            if getattr(sg, "channel", "") == "" and self.analog_signal is not None and getattr(self.analog_signal, "name", ""):
                sg.channel = self.analog_signal.name
                sg.event.annotations["channel"] = self.analog_signal.name

        sg.event.sort()  # TODO: not sure if this breaks the array annotations.

        # Emit onUnitAnnotated with unit details
        lat_samples = int(latency) if latency is not None and latency >= 0 else None
        lat_ms = (
            float((latency / self.analog_signal.sampling_rate).rescale(pq.ms).magnitude)
            if lat_samples is not None and self.analog_signal is not None
            else None
        )
        ts_s = (
            float(
                (
                    self.event_signal[stimno]
                    + (latency / self.analog_signal.sampling_rate)
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
        self.onUnitAnnotated.emit({
            "unit": sg.event.name,
            "unit_idx": int(unit_idx),
            "stimno": int(stimno),
            "latency_samples": lat_samples,
            "latency_ms": lat_ms,
            "timestamp_s": ts_s,
            "fibre_type": str(sg.fibre_type),
        })

        self.onUnitChange.emit(stimno)
        self.onNewSpikeEvents.emit(unit_idx)

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

            ch = getattr(sg, "tracking_channel", "") or getattr(sg, "channel", "") or (self.analog_signal.name if self.analog_signal else "")
            ch_erp = self.get_channel_erp(ch) if ch else self.get_erp()
            if ch_erp is None:
                ch_erp = self.get_erp()

            if ch_erp is None or stimno >= len(ch_erp):
                continue

            # Check if unit already has an annotated spike on this sweep
            existing_sample = sg.get_sweep_sample(stimno, self.event_signal, sr, self.window_size)
            if existing_sample is not None:
                continue

            # Find most recent spike latency in prior sweeps within tracking_max_skips
            max_skips = getattr(sg, "tracking_max_skips", 1)
            prev_latency = None
            min_stim = max(0, stimno - 1 - max_skips)
            for prev_s in range(stimno - 1, min_stim - 1, -1):
                prev_sample = sg.get_sweep_sample(prev_s, self.event_signal, sr, self.window_size)
                if prev_sample is not None:
                    prev_latency = (prev_sample / sr) * pq.s
                    break
                elif getattr(sg, "idx_arr", None) is not None and prev_s < len(sg.idx_arr) and sg.idx_arr[prev_s] is not None:
                    prev_samp = sg.idx_arr[prev_s][0]
                    prev_latency = (prev_samp / sr) * pq.s
                    break

            if prev_latency is None:
                continue

            sweep_wf = np.asarray(ch_erp[stimno].magnitude if hasattr(ch_erp[stimno], "magnitude") else ch_erp[stimno], dtype=np.float32)
            thresh_mv = getattr(sg, "tracking_threshold_mv", 0.01)
            win_ms = getattr(sg, "tracking_window_ms", 5.0)

            detected_sample = basic_tracking.track_sweep_waveform(
                sweep_wf,
                sr,
                prev_latency,
                threshold=thresh_mv,
                window_ms=win_ms,
            )

            if detected_sample is not None:
                self.setUnit(detected_sample, stimno=stimno, unit_idx=unit_idx)
                results.append({
                    "unit_idx": unit_idx,
                    "unit": sg.event.name,
                    "stimno": stimno,
                    "channel": ch,
                    "sample": detected_sample,
                    "latency_ms": (detected_sample / sr) * 1000.0,
                })

        return results

    def stimno_offset_to_event(self, stimno, offset):
        """
        convert arrays of stimno and offset to Event
        """
        t = self.event_signal[stimno] + (offset / self.analog_signal.sampling_rate)
        return Event(t)

    @Slot(int)
    def setStimNo(self, stimno: int):
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
    def loadFile(self, fname, type="h5", **kwargs):
        try:
            data, analog_signal, event_signal, spike_groups = load_file(fname, type)
        except:
            import traceback

            traceback.print_exc()
            if QApplication.instance() and QApplication.instance().platformName() != "offscreen":
                qmb = QMessageBox(
                    QMessageBox.Critical,
                    "Sorry :(",
                    f'Unable to load "{fname}" using {type}',
                )
                qmb.setModal(True)
                qmb.exec()

            return
        self.title = fname
        self.segment = data
        self.cur_spike_group = 0
        self.set_data(analog_signal, event_signal, spike_groups=spike_groups)

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
            p = [None for _ in range(n_sweeps)]
            if sg.event is not None and len(sg.event) > 0:
                sweeps, samples = sg.get_valid_sweep_indices_and_samples(
                    self.event_signal, sr, self.window_size
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
            if len(self._live_erp_buffer) > 0:
                return self._live_erp_buffer.data
            else:
                num_samples = int(
                    self.window_size.rescale(pq.s).magnitude
                    * (self.sampling_rate or 30000)
                )
                units = (
                    self.analog_signal.units
                    if self.analog_signal is not None
                    else pq.mV
                )
                return np.zeros((0, num_samples)) * units

        if (
            self.segment is None
            or self.event_signal is None
            or len(self.event_signal) == 0
        ):
            num_samples = int(
                self.window_size.rescale(pq.s).magnitude
                * (self.sampling_rate or 30000)
            )
            units = (
                self.analog_signal.units
                if self.analog_signal is not None
                else pq.mV
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
                self.window_size.rescale(pq.s).magnitude
                * (self.sampling_rate or 30000)
            )
            units = (
                self.analog_signal.units
                if self.analog_signal is not None
                else pq.mV
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
        t = time.time()
        dpts = self.get_erp()
        n_sweeps = len(dpts) if dpts is not None else 0
        if self._peaks is not None:
            from scipy.signal import find_peaks

            for idx_p in range(len(self._peaks)):
                if self._peaks[idx_p] is None and idx_p < n_sweeps:
                    d = dpts[idx_p]
                    if hasattr(d, "magnitude"):
                        d = d.magnitude
                    d = np.asarray(d).flatten()
                    pts, _ = find_peaks(d)
                    pts_down, _ = find_peaks(-1 * d)
                    self._peaks[idx_p] = np.sort(np.hstack([pts, pts_down]).flatten())
            return self._peaks
        from scipy.signal import find_peaks

        all_pts = []
        for d in dpts:
            pts, _ = find_peaks(d)
            pts_down, _ = find_peaks(-1 * d)
            pts = np.sort(np.hstack([pts, pts_down]).flatten())
            all_pts.append(pts)
        self._peaks = all_pts
        elapsed = time.time() - t
        print(f"Finding peaks took {elapsed:.2f} seconds")
        return all_pts

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
        self.undo_queue = []

    def getUnitGroup(self, unit=None):
        if unit is None:
            unit = self.cur_spike_group
        if unit is None or not self.spike_groups or unit < 0 or unit >= len(self.spike_groups):
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

        self.set_data(analog_signal, event_signal, spike_groups=spike_event_groups)

    def start_live_session(
        self,
        sampling_rate: float = 30000.0,
        window_size_ms: float = 500.0,
        channel_name: str = "rd.0",
        units=pq.uV,
        aux_channels: List[str] = None,
        channels: List[Union[int, str]] = None,
    ):
        """
        Initializes a fresh live streaming session for real-time visualization and annotation.
        """
        if channels:
            ch_names = []
            for idx, c in enumerate(channels):
                if isinstance(c, int):
                    ch_names.append(channel_name if (idx == 0 and channel_name != "rd.0") else f"CH{c}")
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
        self._live_channel_signals = {}
        self._live_aux_erp_buffers = {}
        window_samples = int(sampling_rate * window_size_ms / 1000.0)

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
                sg.event = neo.Event(
                    np.array([], dtype=float) * pq.s,
                    name=sg.event.name,
                    units=pq.s,
                    **sg.event.annotations,
                )
                sg.idx_arr = []
                if not any(e is sg.event or getattr(e, "name", "") == sg.event.name for e in self.segment.events):
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
            self.spike_groups = [tracked_neuron_unit(event=evt, idx_arr=[])]
            self.segment.events.append(evt)

        self.cur_spike_group = 0
        self.stimno = 0
        self.is_streaming = True
        self.sampling_rate = int(sampling_rate)
        self.window_size = (window_size_ms / 1000.0) * pq.s
        self._peaks = []
        self._get_erp.cache_clear()
        self._notify_dataset_loaded("live_start")
        self.onLoadNewFile.emit()
        self.onStreamStateChanged.emit(True)

    def stop_live_session(self):
        """
        Stops the live streaming session.
        """
        self.flush_live_frame()
        self.is_streaming = False
        self.onStreamStateChanged.emit(False)

    def set_active_channel(self, channel_name_or_sig: Union[str, neo.AnalogSignal]):
        """
        Switch the active displayed channel in real time during live streaming or offline mode.
        """
        if channel_name_or_sig is None:
            return

        if isinstance(channel_name_or_sig, str):
            ch_name = channel_name_or_sig
            if getattr(self, "is_streaming", False) and hasattr(self, "_live_aux_erp_buffers") and ch_name in self._live_aux_erp_buffers:
                if hasattr(self, "_live_channel_signals") and ch_name in self._live_channel_signals:
                    self.analog_signal = self._live_channel_signals[ch_name]
                elif self.analog_signal is not None:
                    self.analog_signal.name = ch_name
                self._live_erp_buffer = self._live_aux_erp_buffers[ch_name]
                self._peaks = None
                self._get_erp.cache_clear()
                self.update_idx_arrs()
                self.onChannelChange.emit(ch_name)
                self._notify_dataset_loaded("channel")
                self.onLoadNewFile.emit()
                if self.event_signal is not None and len(self.event_signal) > 0:
                    self.onStimNoChange.emit(self.stimno)
                self.onEventsUpdated.emit()
                return

            # Offline mode by channel name
            found_sig = None
            if self.segment is not None and getattr(self.segment, "analogsignals", None):
                for x in self.segment.analogsignals:
                    if getattr(x, "name", "") == ch_name:
                        found_sig = x
                        break
            if found_sig is not None:
                self.set_data(analog_signal=found_sig)
                self.onChannelChange.emit(found_sig.name)
            else:
                self.onChannelChange.emit(ch_name)
        else:
            sig = channel_name_or_sig
            sig_name = getattr(sig, "name", "")
            if getattr(self, "is_streaming", False) and hasattr(self, "_live_aux_erp_buffers") and sig_name in self._live_aux_erp_buffers:
                self.analog_signal = sig
                self._live_erp_buffer = self._live_aux_erp_buffers[sig_name]
                self._peaks = None
                self._get_erp.cache_clear()
                self.update_idx_arrs()
                self.onChannelChange.emit(sig_name)
                self._notify_dataset_loaded("channel")
                self.onLoadNewFile.emit()
                if self.event_signal is not None and len(self.event_signal) > 0:
                    self.onStimNoChange.emit(self.stimno)
                self.onEventsUpdated.emit()
            else:
                self.set_data(analog_signal=sig)
                self.onChannelChange.emit(sig_name)

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
            self.onStimNoChange.emit(self.stimno)
        self.onEventsUpdated.emit()

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
            new_buf[: self._stim_times_count] = self._stim_times_buf[: self._stim_times_count]
        self._stim_times_buf = new_buf

    def append_sweep(
        self,
        stim_timestamp: Union[float, pq.Quantity],
        sweep_waveform: np.ndarray,
        pre_offset_s: float = 0.0,
        spike_latencies_s: dict = None,
        aux_sweeps: dict = None,
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
            ev = neo.Event(np.array([], dtype=float) * pq.s, name="env.stim", units=pq.s)
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
                    if hasattr(self, "active_channel_names") and ch_i < len(self.active_channel_names)
                    else f"CH{ch_i}"
                )
                ch_wave = sweep_waveform[ch_i]

                if hasattr(self, "_live_aux_erp_buffers") and ch_name not in self._live_aux_erp_buffers:
                    w_samples = (
                        self._live_erp_buffer.window_samples
                        if self._live_erp_buffer is not None
                        else int(sr_val * float(self.window_size.rescale(pq.s).magnitude if hasattr(self.window_size, "magnitude") else self.window_size))
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
                    if self.segment is not None and hasattr(self.segment, "analogsignals"):
                        if new_sig not in self.segment.analogsignals:
                            self.segment.analogsignals.append(new_sig)
                    if hasattr(self, "active_channel_names") and ch_name not in self.active_channel_names:
                        self.active_channel_names.append(ch_name)
                    new_channels_added = True

                sig_slice = neo.AnalogSignal(
                    self._store_raw_sweep(ch_name, ch_wave),
                    units=units,
                    sampling_rate=sr,
                    t_start=stim_ts_q - (pre_offset_s * pq.s),
                    name=ch_name,
                )
                if hasattr(self, "_live_channel_signals") and ch_name in self._live_channel_signals:
                    self._live_channel_signals[ch_name].append_signal(sig_slice)
                elif ch_name == getattr(self.analog_signal, "name", "") and isinstance(self.analog_signal, MultiAnalogSignal):
                    self.analog_signal.append_signal(sig_slice)

                if hasattr(self, "_live_aux_erp_buffers") and ch_name in self._live_aux_erp_buffers:
                    aux_buf = self._live_aux_erp_buffers[ch_name]
                    erp_wave = ch_wave[pre_idx : pre_idx + aux_buf.window_samples]
                    aux_buf.append_sweep(erp_wave)
                elif ch_name == getattr(self.analog_signal, "name", "") and self._live_erp_buffer is not None:
                    erp_wave = ch_wave[pre_idx : pre_idx + self._live_erp_buffer.window_samples]
                    self._live_erp_buffer.append_sweep(erp_wave)

            primary_waveform = (
                sweep_waveform[0]
                if (not hasattr(self, "active_channel_names") or self.analog_signal is None or self.analog_signal.name not in self.active_channel_names)
                else sweep_waveform[self.active_channel_names.index(self.analog_signal.name)]
            )
        else:
            # 1D waveform
            sig_slice = neo.AnalogSignal(
                self._store_raw_sweep(getattr(self.analog_signal, "name", "rd.0"), sweep_waveform),
                units=units,
                sampling_rate=sr,
                t_start=stim_ts_q - (pre_offset_s * pq.s),
                name=self.analog_signal.name if self.analog_signal is not None else "rd.0",
            )
            if hasattr(self, "_live_channel_signals") and getattr(self.analog_signal, "name", "") in self._live_channel_signals:
                self._live_channel_signals[self.analog_signal.name].append_signal(sig_slice)
            elif isinstance(self.analog_signal, MultiAnalogSignal):
                self.analog_signal.append_signal(sig_slice)

            # Stimulus-locked ERP buffer starts at t=0 (offset by pre_idx)
            cur_ch_name = getattr(self.analog_signal, "name", "")
            if hasattr(self, "_live_aux_erp_buffers") and cur_ch_name in self._live_aux_erp_buffers:
                aux_buf = self._live_aux_erp_buffers[cur_ch_name]
                erp_wave = sweep_waveform[pre_idx : pre_idx + aux_buf.window_samples]
                aux_buf.append_sweep(erp_wave)
            elif self._live_erp_buffer is not None:
                erp_wave = sweep_waveform[pre_idx : pre_idx + self._live_erp_buffer.window_samples]
                self._live_erp_buffer.append_sweep(erp_wave)

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
                                else int(sr_val * float(self.window_size.rescale(pq.s).magnitude if hasattr(self.window_size, "magnitude") else self.window_size))
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
                            if self.segment is not None and hasattr(self.segment, "analogsignals"):
                                if new_sig not in self.segment.analogsignals:
                                    self.segment.analogsignals.append(new_sig)
                            if hasattr(self, "active_channel_names") and aux_ch not in self.active_channel_names:
                                self.active_channel_names.append(aux_ch)
                            new_channels_added = True

                        aux_buf = self._live_aux_erp_buffers[aux_ch]
                        erp_aux = aux_wave[pre_idx : pre_idx + aux_buf.window_samples]
                        aux_buf.append_sweep(erp_aux)
                    if hasattr(self, "_live_channel_signals") and aux_ch in self._live_channel_signals:
                        aux_sig = neo.AnalogSignal(
                            self._store_raw_sweep(aux_ch, aux_wave),
                            units=units,
                            sampling_rate=sr,
                            t_start=stim_ts_q - (pre_offset_s * pq.s),
                            name=aux_ch,
                        )
                        self._live_channel_signals[aux_ch].append_signal(aux_sig)

            primary_waveform = sweep_waveform[pre_idx :]

        if new_channels_added and hasattr(self, "onChannelGroupsChanged"):
            self.onChannelGroupsChanged.emit()

        # Defer peak finding until requested by get_peaks() to avoid GUI thread work per sweep
        if self._peaks is None:
            self._peaks = []
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
                    for idx_u, u in enumerate(self.spike_groups):
                        if u.event.name == str(key):
                            sg = u
                            sg_idx = idx_u
                            break
                if sg is not None and lat is not None:
                    lat_qty = (
                        lat * pq.s if not hasattr(lat, "rescale") else lat.rescale(pq.s)
                    )
                    spike_time = stim_ts_q + lat_qty
                    lat_samples = int(
                        np.round(
                            (lat_qty * (self.sampling_rate * pq.Hz)).simplified.magnitude
                        )
                    )
                    sg.idx_arr[new_stim_idx] = (lat_samples, 1)
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

        # Live multi-unit auto-tracking (asynchronous on background thread pool)
        if getattr(self, "auto_track_enabled", False) or any(getattr(sg, "auto_track", False) for sg in self.spike_groups):
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
            if self.sampling_rate is not None and hasattr(self.sampling_rate, "magnitude")
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
                neighbors = [c for c in getattr(self, "active_channel_names", []) if c != primary_ch]
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
                    * (
                        len(self.event_signal)
                        if self.event_signal is not None
                        else 0
                    ),
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
            spikes_q.magnitude if hasattr(spikes_q, "magnitude") else np.asarray(spikes_q)
        )
        ev_mag = (
            sg.event.as_array() if hasattr(sg.event, "as_array") else np.asarray(sg.event)
        )

        if len(sg.event) == 0:
            combined_mag = np.asarray(spikes_mag, dtype=np.float64)
        elif len(spikes_mag) == 0:
            combined_mag = np.asarray(ev_mag, dtype=np.float64)
        else:
            combined_mag = np.concatenate(
                [np.asarray(ev_mag, dtype=np.float64), np.asarray(spikes_mag, dtype=np.float64)]
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
