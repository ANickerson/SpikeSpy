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
from PySide6.QtCore import QObject, Qt, Signal, Slot
from PySide6.QtWidgets import (
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
    MultiAnalogSignal,
    MultiChannelTemplateMatcher,
    create_erp_signals,
)

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

    def __init__(
        self,
        event: Event = None,
        channel: str = "",
        neighbor_channels: List[str] = None,
        fibre_type: str = "",
        notes: str = "",
        idx_arr: List[Any] = None,
    ):
        self.event = event if event is not None else Event()
        self.channel = channel
        self.neighbor_channels = list(neighbor_channels) if neighbor_channels is not None else []
        self.fibre_type = fibre_type
        self.notes = notes
        self._idx_arr = idx_arr

    @property
    def idx_arr(self):
        return self._idx_arr

    @idx_arr.setter
    def idx_arr(self, value):
        self._idx_arr = value

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
    onNewSpikeEvents = Signal(int)
    onStreamStateChanged = Signal(bool)
    onEventsUpdated = Signal()
    onUnitAnnotated = Signal(dict)
    onChannelChange = Signal(str)

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
        self.window_size = 0.5 * pq.s
        self.undo_queue = []
        self.MAX_UNDO = 10
        self.title = ""

        # Live streaming state attributes
        self.is_streaming: bool = False
        self.follow_live: bool = True
        self.auto_track_enabled: bool = False
        self.auto_switch_unit_channel: bool = False
        self._live_erp_buffer: Optional[LiveERPBuffer] = None
        self._live_aux_erp_buffers: dict = {}
        self._live_channel_signals: dict = {}
        self.active_channel_names: List[str] = []
        self.focus_channels: List[str] = []

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

    @Slot(int, int)
    def setUnit(self, latency, stimno=None):
        if stimno is None:
            stimno = self.stimno
        evt = self.spike_groups[self.cur_spike_group].event

        def undo(
            x=self.cur_spike_group,
            evt=self.spike_groups[self.cur_spike_group].event.copy(),
            stimno=self.stimno,
            self=self,
        ):
            evt.sort()
            self.spike_groups[self.cur_spike_group].event = evt
            self.update_idx_arrs()
            self.onUnitChange.emit(stimno)

        self.save_undo(undo)

        if len(evt) > 0:
            to_keep = (evt < self.event_signal[stimno]) | (
                (evt > self.event_signal[stimno + 1])
                if stimno < len(self.event_signal) - 1
                else False
            )  # Prevent crash on final index
            # TODO: we should remove by index, and also use purely the timestamps not idx_arr
            self.spike_groups[self.cur_spike_group].event = evt[to_keep]
            evt = evt[to_keep]

        sg = self.spike_groups[self.cur_spike_group]
        if sg.idx_arr is None:
            sg.idx_arr = []
        while len(sg.idx_arr) <= stimno:
            sg.idx_arr.append(None)

        if latency is None:
            self.spike_groups[self.cur_spike_group].idx_arr[stimno] = None

        else:
            if (
                latency == -1
            ):  # if latency is -1, we want to find the previous spike and use that as the latency
                prv_lat = np.nan
                for attempts in range(3):
                    prev_event_time = self.event_signal.times[
                        self.stimno - 1 - attempts
                    ]
                    sg = self.getUnitGroup()
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
                next_event_time = self.event_signal.times[self.stimno + 1]
                sg = self.getUnitGroup()
                next_lat = sg.get_latencies(
                    np.array([next_event_time]) * next_event_time.units
                )[0].rescale("s")
                if next_lat != np.nan:
                    latency = int(
                        (next_lat * self.analog_signal.sampling_rate).magnitude
                    )

            self.spike_groups[self.cur_spike_group].idx_arr[stimno] = (
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
            self.spike_groups[self.cur_spike_group].event = evt2

            # When annotating a spike on a unit, if sg.channel == "" and self.analog_signal is not None:
            if getattr(sg, "channel", "") == "" and self.analog_signal is not None and getattr(self.analog_signal, "name", ""):
                sg.channel = self.analog_signal.name
                sg.event.annotations["channel"] = self.analog_signal.name

        self.spike_groups[
            self.cur_spike_group
        ].event.sort()  # TODO: not sure if this breaks the array annotations.

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
            "unit": self.spike_groups[self.cur_spike_group].event.name,
            "unit_idx": int(self.cur_spike_group),
            "stimno": int(stimno),
            "latency_samples": lat_samples,
            "latency_ms": lat_ms,
            "timestamp_s": ts_s,
            "fibre_type": str(self.spike_groups[self.cur_spike_group].fibre_type),
        })

        self.onUnitChange.emit(stimno)

    @Slot(Event)
    def updateUnit(self, event, merge=False):
        def undo(
            x=self.cur_spike_group,
            evt=self.spike_groups[self.cur_spike_group].event,
            stimno=self.stimno,
        ):
            self.spike_groups[self.cur_spike_group].event = evt
            self.update_idx_arrs()
            self.onUnitChange.emit(stimno)

        self.save_undo(undo)

        old_evt = self.spike_groups[self.cur_spike_group].event.rescale(pq.s)
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
        self.spike_groups[self.cur_spike_group].event = event
        del self.spike_groups[self.cur_spike_group].idx_arr
        self.update_idx_arrs()
        self.onUnitGroupChange.emit()

    def stimno_offset_to_event(self, stimno, offset):
        """
        convert arrays of stimno and offset to Event
        """
        t = self.event_signal[stimno] + (offset / self.analog_signal.sampling_rate)
        return Event(t)

    @Slot(int)
    def setStimNo(self, stimno: int):
        if self.event_signal is None or len(self.event_signal) == 0:
            self.stimno = 0
            return
        self.stimno = max(min(stimno, len(self.event_signal) - 1), 0)
        self.onStimNoChange.emit(self.stimno)

    @Slot(str, str)
    def loadFile(self, fname, type="h5", **kwargs):
        try:
            data, analog_signal, event_signal, spike_groups = load_file(fname, type)
        except:
            import traceback

            traceback.print_exc()
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
        if self._peaks is not None:
            return self._peaks
        from scipy.signal import find_peaks

        dpts = self.get_erp()
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

        self.sampling_rate = int(np.array(self.analog_signal.sampling_rate.magnitude))
        if len(self.spike_groups) == 0:
            self.spike_groups.append(tracked_neuron_unit(event=Event()))

        self.update_idx_arrs()
        self.onLoadNewFile.emit()
        self.undo_queue = []

    def getUnitGroup(self, unit=None):
        if unit is None:
            unit = self.cur_spike_group
        if unit is None:
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

        if self.spike_groups and len(self.spike_groups) > 0:
            for sg in self.spike_groups:
                sg.event = neo.Event(
                    np.array([], dtype=float) * pq.s,
                    name=sg.event.name,
                    units=pq.s,
                    **sg.event.annotations,
                )
                sg.idx_arr = []
                if sg.event not in self.segment.events:
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
        self.onLoadNewFile.emit()
        self.onStreamStateChanged.emit(True)

    def stop_live_session(self):
        """
        Stops the live streaming session.
        """
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
                self.onLoadNewFile.emit()
                if self.event_signal is not None and len(self.event_signal) > 0:
                    self.onStimNoChange.emit(self.stimno)
                self.onEventsUpdated.emit()
            else:
                self.set_data(analog_signal=sig)
                self.onChannelChange.emit(sig_name)

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
        new_times = np.append(self.event_signal.as_array(), stim_ts_q.magnitude) * pq.s
        new_event = neo.Event(
            new_times,
            name=self.event_signal.name,
            units=pq.s,
            **self.event_signal.annotations,
        )
        if self.segment is not None and len(self.segment.events) > 0:
            for idx_ev, ev in enumerate(self.segment.events):
                if ev.name == self.event_signal.name or id(ev) == id(self.event_signal):
                    self.segment.events[idx_ev] = new_event
                    break
        self.event_signal = new_event
        new_stim_idx = len(self.event_signal) - 1

        units = self.analog_signal.units if self.analog_signal is not None else pq.uV
        sr = (
            self.sampling_rate * pq.Hz
            if not hasattr(self.sampling_rate, "rescale")
            else self.sampling_rate
        )

        is_2d = hasattr(sweep_waveform, "ndim") and sweep_waveform.ndim == 2

        if is_2d:
            n_ch = sweep_waveform.shape[0]
            for ch_i in range(n_ch):
                ch_name = (
                    self.active_channel_names[ch_i]
                    if hasattr(self, "active_channel_names") and ch_i < len(self.active_channel_names)
                    else f"CH{ch_i}"
                )
                ch_wave = sweep_waveform[ch_i]
                sig_slice = neo.AnalogSignal(
                    ch_wave,
                    units=units,
                    sampling_rate=sr,
                    t_start=stim_ts_q + (pre_offset_s * pq.s),
                    name=ch_name,
                )
                if hasattr(self, "_live_channel_signals") and ch_name in self._live_channel_signals:
                    self._live_channel_signals[ch_name].append_signal(sig_slice)
                elif ch_name == getattr(self.analog_signal, "name", "") and isinstance(self.analog_signal, MultiAnalogSignal):
                    self.analog_signal.append_signal(sig_slice)

                if hasattr(self, "_live_aux_erp_buffers") and ch_name in self._live_aux_erp_buffers:
                    aux_buf = self._live_aux_erp_buffers[ch_name]
                    aux_buf.append_sweep(ch_wave[: aux_buf.window_samples])
                elif ch_name == getattr(self.analog_signal, "name", "") and self._live_erp_buffer is not None:
                    self._live_erp_buffer.append_sweep(ch_wave[: self._live_erp_buffer.window_samples])

            primary_waveform = (
                sweep_waveform[0]
                if (not hasattr(self, "active_channel_names") or self.analog_signal is None or self.analog_signal.name not in self.active_channel_names)
                else sweep_waveform[self.active_channel_names.index(self.analog_signal.name)]
            )
        else:
            # 1D waveform
            sig_slice = neo.AnalogSignal(
                sweep_waveform,
                units=units,
                sampling_rate=sr,
                t_start=stim_ts_q + (pre_offset_s * pq.s),
                name=self.analog_signal.name if self.analog_signal is not None else "rd.0",
            )
            if hasattr(self, "_live_channel_signals") and getattr(self.analog_signal, "name", "") in self._live_channel_signals:
                self._live_channel_signals[self.analog_signal.name].append_signal(sig_slice)
            elif isinstance(self.analog_signal, MultiAnalogSignal):
                self.analog_signal.append_signal(sig_slice)

            # Appends the [0, window_samples] portion to primary buffer
            cur_ch_name = getattr(self.analog_signal, "name", "")
            if hasattr(self, "_live_aux_erp_buffers") and cur_ch_name in self._live_aux_erp_buffers:
                self._live_aux_erp_buffers[cur_ch_name].append_sweep(
                    sweep_waveform[: self._live_aux_erp_buffers[cur_ch_name].window_samples]
                )
            elif self._live_erp_buffer is not None:
                self._live_erp_buffer.append_sweep(
                    sweep_waveform[: self._live_erp_buffer.window_samples]
                )

            # Handle aux sweeps if any
            if aux_sweeps and hasattr(self, "_live_aux_erp_buffers") and self._live_aux_erp_buffers:
                for aux_ch, aux_wave in aux_sweeps.items():
                    if aux_ch == cur_ch_name:
                        continue  # Do not double-append to primary
                    if aux_ch in self._live_aux_erp_buffers:
                        aux_buf = self._live_aux_erp_buffers[aux_ch]
                        aux_buf.append_sweep(aux_wave[: aux_buf.window_samples])
                    if hasattr(self, "_live_channel_signals") and aux_ch in self._live_channel_signals:
                        aux_sig = neo.AnalogSignal(
                            aux_wave,
                            units=units,
                            sampling_rate=sr,
                            t_start=stim_ts_q + (pre_offset_s * pq.s),
                            name=aux_ch,
                        )
                        self._live_channel_signals[aux_ch].append_signal(aux_sig)

            primary_waveform = sweep_waveform

        # Incrementally computes peaks (find_peaks(d) and find_peaks(-d)) for this single new sweep
        if self._peaks is None:
            self._peaks = []
        from scipy.signal import find_peaks

        d = primary_waveform
        if hasattr(d, "magnitude"):
            d = d.magnitude
        d = np.asarray(d).flatten()
        pts, _ = find_peaks(d)
        pts_down, _ = find_peaks(-1 * d)
        pts = np.sort(np.hstack([pts, pts_down]).flatten())
        self._peaks.append(pts)

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

        # Emits signals
        self.onNewSweep.emit(new_stim_idx)
        if self.follow_live:
            self.onStimNoChange.emit(self.stimno)
        self.onEventsUpdated.emit()

        return new_stim_idx

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
                evt.annotations["type"] = "unit"
                evt.annotations["unit"] = len(self.spike_groups)
                evt.annotations["notes"] = ""
                evt.annotations["fibre_type"] = ""
                sg = tracked_neuron_unit(
                    event=evt,
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
            combined_mag = spikes_mag
        else:
            combined_mag = np.concatenate([ev_mag, spikes_mag])

        unique_times = np.unique(combined_mag) * pq.s
        sg.event = neo.Event(
            unique_times,
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

            for spike in spikes_q:
                i = (
                    int(
                        np.searchsorted(
                            self.event_signal.rescale("s"), spike, side="right"
                        ).base
                    )
                    - 1
                )
                if 0 <= i < len(self.event_signal):
                    lat_s = spike - self.event_signal[i]
                    if 0 * pq.s <= lat_s <= self.window_size:
                        sr_val = (
                            self.sampling_rate
                            if self.sampling_rate is not None
                            else (
                                self.analog_signal.sampling_rate.magnitude
                                if self.analog_signal is not None
                                else 30000
                            )
                        )
                        lat_samples = int(
                            np.round((lat_s * (sr_val * pq.Hz)).simplified.magnitude)
                        )
                        sg.idx_arr[i] = (lat_samples, 1)

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
