"""Real-Time Open Ephys ZMQ Stream Receiver & ERP Slicer for SpikeSpy.

Supports two streaming modes:
1. Standard Open Ephys v1.0+ ZMQ Interface + Event Broadcaster (`mode="zmq_raw"`)
   - Data SUB on port 5556 (continuous data multi-channel float/int16 stream)
   - Event SUB on port 5557 (TTL pulses and Spike events)
   - Heartbeat REQ on port 5557/5558 (Open Ephys ZMQ requirement)
   - RingBuffer + stimulus trigger detection (digital TTL or analog ADC)
   - Slices sweeps [stim_sample - pre_samples : stim_sample + window_samples]
2. Pre-Sliced ERP Stream Mode (`mode="sliced_erp"`)
   - Direct stimulus-locked ERP sweeps from custom SpikeSpyStreamer C++ plugin
"""

from __future__ import annotations

import json
import logging
import re
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import zmq
from PySide6.QtCore import QObject, QThread, Signal, Slot

try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
    def njit(*args, **kwargs):
        def wrapper(func):
            return func
        return wrapper if args else wrapper

from spikespy.processing import OnlineBandpassFilter, OnlineTTLDetector

logger = logging.getLogger(__name__)


def _has_ch0_source(roles: Optional[List[Any]]) -> bool:
    """Return True if any role explicitly references 0-based 'CH0'."""
    if not roles:
        return False
    for role in roles:
        src = role.get("source") if isinstance(role, dict) else getattr(role, "source", None)
        if src is None:
            continue
        if re.fullmatch(r"CH\s*0+", str(src).strip().upper()):
            return True
    return False


def _parse_channel_source_index(
    source: Any,
    has_ch0: bool = False,
    n_channels: Optional[int] = None,
) -> Optional[int]:
    """Parse a channel source string or integer into a 0-based channel index.

    Rules:
    - int or digit string (e.g. 0, "0", "16") -> int(source)
    - "CH0" -> 0
    - "CH<k>" -> k if has_ch0 else max(0, k - 1)
    - "ADC<k>" -> 16 + k - 1 (standard Open Ephys 16 headstage + 8 ADC),
      falling back to k on compact streams where n_channels <= 16 and k < n_channels.
    """
    if source is None:
        return None
    if isinstance(source, (int, np.integer)):
        idx = int(source)
        return idx if idx >= 0 else None

    s = str(source).strip()
    if not s:
        return None
    if s.isdigit():
        return int(s)

    upper = s.upper()
    m_ch = re.fullmatch(r"CH\s*(\d+)", upper)
    if m_ch:
        k = int(m_ch.group(1))
        if k == 0:
            return 0
        return k if has_ch0 else max(0, k - 1)

    m_adc = re.fullmatch(r"ADC\s*(\d+)", upper)
    if m_adc:
        k = int(m_adc.group(1))
        std_idx = 16 + (k - 1 if k >= 1 else 0)
        if n_channels is not None and n_channels <= 16 and std_idx >= n_channels:
            if k < n_channels:
                return k
            return None
        return std_idx

    return None


def resolve_source_to_index(
    source_str: Any,
    n_channels: int = 64,
    has_ch0: bool = False,
) -> Optional[int]:
    """Resolve a channel source string (e.g. '0', 'CH1', 'CH0', 'ADC4') to a 0-based index."""
    return _parse_channel_source_index(
        source_str,
        has_ch0=has_ch0,
        n_channels=n_channels,
    )


if _HAS_NUMBA:
    @njit(cache=True)
    def _ring_buffer_slice_core(buffer, write_idx, capacity, n_channels,
                                 buf_start_idx, requested_len, channel):
        """Numba-accelerated ring buffer slice for single channel or all channels."""
        if channel >= 0:
            # Single channel
            if buf_start_idx + requested_len <= capacity:
                return buffer[channel, buf_start_idx:buf_start_idx + requested_len].copy()
            else:
                first_part = capacity - buf_start_idx
                second_part = requested_len - first_part
                out = np.empty(requested_len, dtype=np.float32)
                out[:first_part] = buffer[channel, buf_start_idx:]
                out[first_part:] = buffer[channel, :second_part]
                return out
        else:
            # All channels
            if buf_start_idx + requested_len <= capacity:
                return buffer[:, buf_start_idx:buf_start_idx + requested_len].copy()
            else:
                first_part = capacity - buf_start_idx
                second_part = requested_len - first_part
                out = np.empty((n_channels, requested_len), dtype=np.float32)
                out[:, :first_part] = buffer[:, buf_start_idx:]
                out[:, first_part:] = buffer[:, :second_part]
                return out

    @njit(cache=True)
    def _ring_buffer_slice_channels_core(buffer, write_idx, capacity, n_channels,
                                          buf_start_idx, requested_len, rows):
        """Numba-accelerated ring buffer slice for specific channels."""
        n_rows = rows.shape[0]
        if buf_start_idx + requested_len <= capacity:
            out = np.empty((n_rows, requested_len), dtype=np.float32)
            for i in range(n_rows):
                out[i, :] = buffer[rows[i], buf_start_idx:buf_start_idx + requested_len]
            return out
        else:
            first_part = capacity - buf_start_idx
            second_part = requested_len - first_part
            out = np.empty((n_rows, requested_len), dtype=np.float32)
            for i in range(n_rows):
                out[i, :first_part] = buffer[rows[i], buf_start_idx:]
                out[i, first_part:] = buffer[rows[i], :second_part]
            return out
else:
    def _ring_buffer_slice_core(buffer, write_idx, capacity, n_channels,
                                 buf_start_idx, requested_len, channel):
        """Fallback pure Python implementation."""
        if channel >= 0:
            if buf_start_idx + requested_len <= capacity:
                return buffer[channel, buf_start_idx:buf_start_idx + requested_len].copy()
            else:
                first_part = capacity - buf_start_idx
                second_part = requested_len - first_part
                out = np.empty(requested_len, dtype=np.float32)
                out[:first_part] = buffer[channel, buf_start_idx:]
                out[first_part:] = buffer[channel, :second_part]
                return out
        else:
            if buf_start_idx + requested_len <= capacity:
                return buffer[:, buf_start_idx:buf_start_idx + requested_len].copy()
            else:
                first_part = capacity - buf_start_idx
                second_part = requested_len - first_part
                out = np.empty((n_channels, requested_len), dtype=np.float32)
                out[:, :first_part] = buffer[:, buf_start_idx:]
                out[:, first_part:] = buffer[:, :second_part]
                return out

    def _ring_buffer_slice_channels_core(buffer, write_idx, capacity, n_channels,
                                          buf_start_idx, requested_len, rows):
        """Fallback pure Python implementation."""
        n_rows = rows.shape[0]
        if buf_start_idx + requested_len <= capacity:
            out = np.empty((n_rows, requested_len), dtype=np.float32)
            for i in range(n_rows):
                out[i, :] = buffer[rows[i], buf_start_idx:buf_start_idx + requested_len]
            return out
        else:
            first_part = capacity - buf_start_idx
            second_part = requested_len - first_part
            out = np.empty((n_rows, requested_len), dtype=np.float32)
            for i in range(n_rows):
                out[i, :first_part] = buffer[rows[i], buf_start_idx:]
                out[i, first_part:] = buffer[rows[i], :second_part]
            return out


class RingBuffer:
    """Thread-safe circular ring buffer for multi-channel continuous streaming data."""

    def __init__(self, n_channels: int, capacity_samples: int):
        self.n_channels = n_channels
        self.capacity = capacity_samples
        self.buffer = np.zeros((n_channels, capacity_samples), dtype=np.float32)
        self.write_idx = 0
        self.total_samples_written = 0
        self.first_sample_index: Optional[int] = None
        self._latest_sample: Optional[int] = None
        self.lock = threading.Lock()

    @property
    def latest_sample(self) -> Optional[int]:
        """Sample index of the most recently written sample (None when empty)."""
        return self._latest_sample

    @property
    def earliest_sample(self) -> Optional[int]:
        """Oldest sample index still retained in the ring buffer (None when empty)."""
        if self._latest_sample is None or self.first_sample_index is None:
            return None
        if self.total_samples_written >= self.capacity:
            return self._latest_sample - self.capacity + 1
        return self.first_sample_index

    def clear(self):
        """Discard all buffered samples without reallocating the backing array."""
        with self.lock:
            self.write_idx = 0
            self.total_samples_written = 0
            self.first_sample_index = None
            self._latest_sample = None

    def append(self, samples: np.ndarray, start_sample_index: int):
        """Append samples of shape (n_channels, n_samples) or (n_samples, n_channels)."""
        if samples.ndim == 1:
            samples = samples[np.newaxis, :]
        elif samples.shape[0] != self.n_channels and samples.shape[1] == self.n_channels:
            samples = samples.T

        n_ch, n_samples = samples.shape
        if n_samples == 0:
            return

        with self.lock:
            if n_ch > self.n_channels:
                new_buf = np.zeros((n_ch, self.capacity), dtype=np.float32)
                new_buf[: self.n_channels, :] = self.buffer
                self.buffer = new_buf
                self.n_channels = n_ch

            if self.first_sample_index is None:
                self.first_sample_index = int(start_sample_index)

            if n_samples >= self.capacity:
                samples = samples[:, -self.capacity :]
                start_sample_index += (n_samples - self.capacity)
                n_samples = self.capacity

            w = self.write_idx
            cap = self.capacity

            if w + n_samples <= cap:
                self.buffer[:n_ch, w : w + n_samples] = samples
            else:
                first_chunk = cap - w
                second_chunk = n_samples - first_chunk
                self.buffer[:n_ch, w:cap] = samples[:, :first_chunk]
                self.buffer[:n_ch, :second_chunk] = samples[:, first_chunk:]

            self.write_idx = (w + n_samples) % cap
            self.total_samples_written += n_samples
            self._latest_sample = int(start_sample_index + n_samples - 1)

    def slice(
        self, start_sample_index: int, end_sample_index: int, channel: Optional[int] = None
    ) -> Optional[np.ndarray]:
        """Extract a contiguous slice [start_sample_index, end_sample_index) by sample counter.

        Returns None if requested range is not yet fully in buffer or has been overwritten.
        """
        requested_len = end_sample_index - start_sample_index
        if requested_len <= 0:
            return np.empty((0,) if channel is not None else (self.n_channels, 0), dtype=np.float32)

        with self.lock:
            latest_sample = self.latest_sample
            if latest_sample is None:
                return None
            earliest_sample = self.earliest_sample

            # Check if data is within available window
            if end_sample_index - 1 > latest_sample or start_sample_index < earliest_sample:
                return None

            # Calculate index position in ring buffer
            # Distance from latest_sample:
            dist_from_latest_start = latest_sample - start_sample_index
            buf_start_idx = (self.write_idx - 1 - dist_from_latest_start) % self.capacity

            ch = channel if channel is not None else -1
            return _ring_buffer_slice_core(
                self.buffer, self.write_idx, self.capacity, self.n_channels,
                buf_start_idx, requested_len, ch
            )

    def slice_channels(
        self, start_sample_index: int, end_sample_index: int, channels: List[int]
    ) -> Optional[np.ndarray]:
        """Extract selected channel rows for [start_sample_index, end_sample_index).

        Allocates only ``(len(channels), n_samples)`` and copies just those rows.
        Returns None if the requested range is not fully available anymore.
        """
        requested_len = end_sample_index - start_sample_index
        rows = np.asarray(channels, dtype=np.intp)
        if requested_len <= 0:
            return np.empty((len(rows), 0), dtype=np.float32)
        if rows.size == 0:
            return np.empty((0, requested_len), dtype=np.float32)

        with self.lock:
            latest_sample = self.latest_sample
            if latest_sample is None:
                return None
            earliest_sample = self.earliest_sample
            if end_sample_index - 1 > latest_sample or start_sample_index < earliest_sample:
                return None
            if int(rows.min()) < 0 or int(rows.max()) >= self.buffer.shape[0]:
                return None

            dist_from_latest_start = latest_sample - start_sample_index
            buf_start_idx = (self.write_idx - 1 - dist_from_latest_start) % self.capacity

            return _ring_buffer_slice_channels_core(
                self.buffer, self.write_idx, self.capacity, self.n_channels,
                buf_start_idx, requested_len, rows
            )



@dataclass
class PendingSweep:
    stim_sample: int
    stim_timestamp_s: float
    window_samples: int
    pre_samples: int
    rec_chan: int
    aux_chans: List[int] = field(default_factory=list)
    spike_latencies: Dict[str, float] = field(default_factory=dict)


class OpenEphysStreamWorker(QThread):
    """Real-Time ZMQ Stream Worker for Open Ephys v1.0+.

    Manages SUB sockets for data and events, handles heartbeats, maintains
    circular ring buffers, detects TTL pulses, slices sweeps, and emits Qt signals.
    """

    sweepReady = Signal(dict)
    sweepsReady = Signal(list)
    spikeEventsReady = Signal(str, object)
    protocolEventReady = Signal(str, float, dict)
    statsUpdated = Signal(dict)
    connectionStatusChanged = Signal(bool, str)
    debugEventEmitted = Signal(dict)

    def __init__(
        self,
        host: str = "127.0.0.1",
        data_port: int = 5556,
        event_port: int = 5557,
        heartbeat_port: int = 5557,
        mode: str = "zmq_raw",  # "zmq_raw" or "sliced_erp"
        rec_channel: int = 0,
        active_channels: Optional[List[int]] = None,
        trigger_source: str = "ttl",  # "ttl" or "analog"
        analog_trigger_channel: int = 1,
        window_ms: float = 500.0,
        pre_stim_ms: float = 50.0,
        apply_filter: bool = True,
        filter_low: float = 500.0,
        filter_high: float = 7000.0,
        sample_rate: float = 30000.0,
        ttl_threshold_v: float = 2.5,
        ttl_refractory_ms: float = 100.0,
        channel_roles: Optional[List[Any]] = None,
        parent: Optional[QObject] = None,
    ):
        super().__init__(parent)
        self.host = host
        self.data_port = data_port
        self.event_port = event_port
        self.heartbeat_port = heartbeat_port
        self.mode = mode
        self.rec_channel = rec_channel
        self.active_channels: List[int] = (
            sorted(list(set(active_channels))) if active_channels else [self.rec_channel]
        )
        if self.rec_channel not in self.active_channels:
            self.active_channels.append(self.rec_channel)
            self.active_channels.sort()
        self.trigger_source = trigger_source
        self.analog_trigger_channel = analog_trigger_channel
        self.window_ms = window_ms
        self.pre_stim_ms = pre_stim_ms
        self.apply_filter = apply_filter
        self.filter_low = float(filter_low)
        self.filter_high = float(filter_high)
        self.sample_rate = sample_rate
        self.ttl_threshold_v = float(ttl_threshold_v)
        self.ttl_refractory_ms = float(ttl_refractory_ms)
        self.auto_align_clock = True

        self._running = False
        self._pending_sweeps: List[PendingSweep] = []
        self._pending_lock = threading.Lock()
        self._lock = threading.RLock()

        # Batched sweep delivery to the GUI (flush at most every ~20 ms or at cap)
        self._pending_payloads: List[Dict[str, Any]] = []
        self._last_sweep_flush = 0.0
        self._sweep_flush_interval_s = 0.02
        self._sweep_batch_max = 32

        # Ring buffer: default 16 channels, 15 seconds at sample_rate
        cap = int(sample_rate * 15.0)
        self.ring_buffer = RingBuffer(n_channels=16, capacity_samples=cap)
        self.bandpass_filter = OnlineBandpassFilter(
            lowcut=self.filter_low,
            highcut=self.filter_high,
            sampling_rate=sample_rate,
        )
        self._channel_filters: Dict[int, OnlineBandpassFilter] = {}
        self._pending_block_start: Optional[int] = None
        self._pending_block_len: int = 0
        self._pending_block_time: float = 0.0
        self._pending_block_chans: Dict[int, np.ndarray] = {}
        self._seen_single_channels: set = set()
        self._expected_single_ch_count: int = 0
        self.analog_ttl_detector = OnlineTTLDetector(
            threshold=self.ttl_threshold_v,
            refractory_s=self.ttl_refractory_ms / 1000.0,
        )
        self.ttl_detector = self.analog_ttl_detector

        # Channel purpose / role mappings
        self.channel_roles: List[Dict[str, Any]] = []
        self.channel_role_names: Dict[int, str] = {}
        self.filtered_channels: Optional[Set[int]] = None
        self.unfiltered_channels: Set[int] = set()
        self._aux_ttl_detectors: Dict[str, Tuple[int, OnlineTTLDetector, Dict[str, Any]]] = {}
        self._channel_filters: Dict[int, OnlineBandpassFilter] = {}
        self._channel_filter_bands: Dict[int, Tuple[float, float]] = {}
        self._role_analog_channels: List[int] = []
        self._channel_analog_roles: Dict[int, List[Dict[str, Any]]] = {}
        self._stream_n_channels: Optional[int] = None

        # Metrics
        self._sweeps_count = 0
        self._spikes_count = 0
        self._packets_count = 0
        self._last_stim_timestamp: Optional[float] = None
        self._stim_intervals: List[float] = []

        # Connection monitoring & heartbeat state
        self._plugin_connected = False
        self._last_heartbeat_reply = 0.0
        self._last_data_received = 0.0
        self._plugin_info: Dict[str, Any] = {}

        if channel_roles is not None:
            self.set_channel_roles(channel_roles)

    @property
    def pre_ms(self) -> float:
        return self.pre_stim_ms

    @pre_ms.setter
    def pre_ms(self, value: float):
        self.pre_stim_ms = float(value)

    @property
    def use_analog_trigger(self) -> bool:
        return self.trigger_source == "analog"

    @use_analog_trigger.setter
    def use_analog_trigger(self, enabled: bool):
        self.trigger_source = "analog" if enabled else "ttl"

    @property
    def adc_trigger_chan(self) -> int:
        return self.analog_trigger_channel

    @adc_trigger_chan.setter
    def adc_trigger_chan(self, chan: int):
        self.analog_trigger_channel = int(chan)

    def stop(self):
        """Signal thread to stop and wait."""
        self._running = False
        self.wait(2000)

    def set_rec_channel(self, chan: int):
        self.rec_channel = chan
        if self.rec_channel not in self.active_channels:
            self.active_channels.append(self.rec_channel)
            self.active_channels.sort()

    def set_active_channels(self, channels: List[int], primary_channel: Optional[int] = None):
        prev_count = len(self.active_channels)
        new_active = sorted(list(set(channels))) if channels else [self.rec_channel]
        if primary_channel is not None:
            self.rec_channel = primary_channel
        if self.rec_channel not in new_active:
            new_active.append(self.rec_channel)
            new_active.sort()
        self.active_channels = new_active
        if len(self.active_channels) != prev_count:
            with self._lock:
                self.bandpass_filter.reset()

    def _reset_pending_channel_block(self):
        self._pending_block_start = None
        self._pending_block_len = 0
        self._pending_block_time = 0.0
        self._pending_block_chans.clear()
        self._seen_single_channels.clear()
        self._expected_single_ch_count = 0

    def _update_sample_rate(self, sr: float):
        sr = float(sr)
        if sr <= 0 or abs(sr - self.sample_rate) <= 1e-6:
            return
        self.sample_rate = sr
        with self._lock:
            self.bandpass_filter = OnlineBandpassFilter(
                lowcut=self.filter_low,
                highcut=self.filter_high,
                sampling_rate=self.sample_rate,
            )
            self._channel_filters.clear()

    def _flush_pending_channel_block(self):
        if not self._pending_block_chans or self._pending_block_start is None:
            return
        n_samples = self._pending_block_len
        if n_samples <= 0:
            self._pending_block_chans.clear()
            self._pending_block_start = None
            self._pending_block_len = 0
            return
        max_ch = max(
            max(self._pending_block_chans.keys()),
            max(self._seen_single_channels, default=0),
        )
        n_ch = max_ch + 1
        block = np.zeros((n_ch, n_samples), dtype=np.float32)
        for ch_i, arr in self._pending_block_chans.items():
            m = min(len(arr), n_samples)
            block[ch_i, :m] = arr[:m]
        start_samp = self._pending_block_start
        self._pending_block_chans.clear()
        self._pending_block_start = None
        self._pending_block_len = 0
        self.ring_buffer.append(block, start_samp)

    def _ingest_single_channel_chunk(
        self, ch_idx: int, ch_data: np.ndarray, start_sample: int
    ):
        """Ingest a single-channel continuous block (standard Open Ephys ZMQ Interface format)."""
        if ch_idx < 0 or ch_data.size == 0:
            return

        # Analog TTL detection must run on unfiltered samples
        if self.trigger_source == "analog" and ch_idx == self.analog_trigger_channel:
            with self._lock:
                triggers = self.analog_ttl_detector.process_chunk(ch_data, start_sample)
            for t_sample in triggers:
                self.queue_sweep_trigger(t_sample, t_sample / self.sample_rate)

        if self.apply_filter:
            with self._lock:
                filt = self._channel_filters.get(ch_idx)
                if filt is None:
                    filt = OnlineBandpassFilter(
                        lowcut=self.filter_low,
                        highcut=self.filter_high,
                        sampling_rate=self.sample_rate,
                    )
                    self._channel_filters[ch_idx] = filt
                ch_data = filt.process(ch_data)

        if (
            self._pending_block_start is not None
            and start_sample != self._pending_block_start
        ):
            self._expected_single_ch_count = max(
                self._expected_single_ch_count, len(self._pending_block_chans)
            )
            self._flush_pending_channel_block()

        if self._pending_block_start is None:
            self._pending_block_start = start_sample
            self._pending_block_len = len(ch_data)
            self._pending_block_time = time.time()

        self._pending_block_chans[ch_idx] = ch_data
        if len(ch_data) > self._pending_block_len:
            self._pending_block_len = len(ch_data)
        self._seen_single_channels.add(ch_idx)

        if (
            self._expected_single_ch_count > 0
            and len(self._pending_block_chans) >= self._expected_single_ch_count
        ):
            self._flush_pending_channel_block()

    def set_filtering(self, enabled: bool):
        if enabled != self.apply_filter:
            # Never mix filtered and unfiltered samples inside one pending sweep.
            self._reset_pending_channel_block()
            self.ring_buffer.clear()
            with self._pending_lock:
                self._pending_sweeps.clear()
        self.apply_filter = enabled
        if not enabled:
            with self._lock:
                self.bandpass_filter.reset()
                for filt in self._channel_filters.values():
                    filt.reset()
                self._channel_filters.clear()

    def set_filter_band(self, low_hz: float, high_hz: float):
        low_hz = float(low_hz)
        high_hz = float(high_hz)
        if low_hz <= 0 or high_hz <= low_hz:
            raise ValueError(
                f"Invalid filter band: low_hz={low_hz}, high_hz={high_hz}"
            )
        with self._lock:
            if (
                self.filter_low == low_hz
                and self.filter_high == high_hz
                and self.bandpass_filter is not None
                and self.bandpass_filter.sample_rate == self.sample_rate
            ):
                return
            self.filter_low = low_hz
            self.filter_high = high_hz
            self.bandpass_filter = OnlineBandpassFilter(
                lowcut=self.filter_low,
                highcut=self.filter_high,
                sampling_rate=self.sample_rate,
            )
            self._channel_filters.clear()
            for ch_idx in list(self._channel_filter_bands.keys()):
                self._channel_filter_bands[ch_idx] = (low_hz, high_hz)
                self._channel_filters[ch_idx] = OnlineBandpassFilter(
                    lowcut=low_hz,
                    highcut=high_hz,
                    sampling_rate=self.sample_rate,
                )

    def set_ttl_params(
        self,
        threshold: Optional[float] = None,
        refractory_ms: Optional[float] = None,
    ):
        with self._lock:
            if threshold is not None:
                self.ttl_threshold_v = float(threshold)
                self.ttl_detector.threshold = float(threshold)
            if refractory_ms is not None:
                self.ttl_refractory_ms = float(refractory_ms)
                self.ttl_detector.refractory_s = float(refractory_ms) / 1000.0

    @staticmethod
    def _normalize_role_entry(role: Any) -> Dict[str, Any]:
        """Normalize a role dict or ChannelRoleSpec-like object into a canonical dict."""
        if isinstance(role, dict):
            getter = role.get
        else:
            getter = lambda k, default=None: getattr(role, k, default)

        raw_src = getter("source", None)
        if raw_src is None:
            raw_src = getter("source_channel", None)
        if raw_src is None:
            raw_src = getter("channel", "")
        source = str(raw_src if raw_src is not None else "").strip()

        raw_name = getter("name", None)
        if raw_name is None:
            raw_name = getter("target_name", None)
        if raw_name is None:
            raw_name = getter("target", None)
        name = str(raw_name if raw_name is not None else source).strip()

        purpose = str(getter("purpose", "") or getter("role", "") or "").strip()

        raw_pt = getter("probe_type", None) or getter("type", None)
        if hasattr(raw_pt, "value"):
            raw_pt = raw_pt.value
        probe_type = str(raw_pt or "").strip().upper()
        if probe_type in ("ANALOG", "AS"):
            probe_type = "AS"
        elif probe_type == "TTL":
            probe_type = "TTL"
        elif probe_type in ("EVT", "EVENTS"):
            probe_type = "EVT"

        mode = str(getter("mode", "") or "").strip().lower()
        if not probe_type:
            if mode == "ttl":
                probe_type = "TTL"
            else:
                probe_type = "AS"

        raw_bp = getter("bandpass", None)
        if mode == "analog_bp":
            bandpass = True
        elif mode in ("analog", "ttl"):
            bandpass = False
        elif raw_bp is not None:
            bandpass = bool(raw_bp)
        else:
            bandpass = (purpose == "probe" and probe_type == "AS")

        bp_low = getter("bandpass_low", None)
        bp_high = getter("bandpass_high", None)
        bandpass_low = float(bp_low) if bp_low is not None else 500.0
        bandpass_high = float(bp_high) if bp_high is not None else 7000.0

        raw_thresh = getter("ttl_threshold", None)
        if raw_thresh is None:
            raw_thresh = getter("ttl_threshold_v", None)
        ttl_threshold = float(raw_thresh) if raw_thresh is not None else None

        raw_refrac = getter("ttl_refractory_ms", None)
        ttl_refractory_ms = float(raw_refrac) if raw_refrac is not None else 100.0

        raw_min_w = getter("ttl_min_width_ms", None)
        ttl_min_width_ms = float(raw_min_w) if raw_min_w is not None else 0.4

        raw_is_adc = getter("is_adc", None)
        is_adc = (
            bool(raw_is_adc)
            if raw_is_adc is not None
            else source.upper().startswith("ADC")
        )

        raw_enabled = getter("enabled", None)
        enabled = True if raw_enabled is None else bool(raw_enabled)

        raw_ch_idx = getter("channel_index", None)
        channel_index = int(raw_ch_idx) if raw_ch_idx is not None else None

        norm: Dict[str, Any] = {
            "source": source,
            "name": name,
            "probe_type": probe_type,
            "purpose": purpose,
            "bandpass": bandpass,
            "bandpass_low": bandpass_low,
            "bandpass_high": bandpass_high,
            "ttl_threshold": ttl_threshold,
            "ttl_refractory_ms": ttl_refractory_ms,
            "ttl_min_width_ms": ttl_min_width_ms,
            "is_adc": is_adc,
            "enabled": enabled,
        }
        if channel_index is not None:
            norm["channel_index"] = channel_index
        trig_src_override = getter("trigger_source", None)
        if trig_src_override is not None:
            norm["trigger_source"] = str(trig_src_override).strip().lower()
        return norm

    def set_channel_roles(self, roles: Optional[List[Any]]) -> None:
        """Configure channel purpose/role mapping and per-channel preprocessing.

        Thread-safe and reloadable while streaming.
        """
        with self._lock:
            if not roles:
                self.channel_roles = []
                self.channel_role_names = {}
                self.filtered_channels = None
                self.unfiltered_channels = set()
                self._aux_ttl_detectors = {}
                self._channel_filters = {}
                self._channel_filter_bands = {}
                self._role_analog_channels = []
                self._channel_analog_roles = {}
                return

            self.channel_roles = [self._normalize_role_entry(r) for r in roles]
            self._rebuild_role_mappings_locked()

    def _rebuild_role_mappings_locked(self) -> None:
        """Resolve role indices, filters, and TTL detectors under self._lock."""
        roles = self.channel_roles
        if not roles:
            return

        n_channels = self._stream_n_channels if self._stream_n_channels is not None else 64
        has_ch0 = _has_ch0_source(roles)

        new_role_names: Dict[int, str] = {}
        new_filtered: Set[int] = set()
        new_unfiltered: Set[int] = set()
        new_analog_channels: List[int] = []
        new_channel_analog_roles: Dict[int, List[Dict[str, Any]]] = {}
        new_filter_bands: Dict[int, Tuple[float, float]] = {}
        new_aux_ttl: Dict[str, Tuple[int, OnlineTTLDetector, Dict[str, Any]]] = {}

        primary_probe_role: Optional[Tuple[int, Dict[str, Any]]] = None
        stim_trigger_role: Optional[Tuple[int, Dict[str, Any]]] = None

        for role in roles:
            if not role.get("enabled", True):
                continue

            if role.get("channel_index") is not None:
                ch_idx: Optional[int] = int(role["channel_index"])
            else:
                ch_idx = _parse_channel_source_index(
                    role.get("source", ""),
                    has_ch0=has_ch0,
                    n_channels=n_channels,
                )

            if ch_idx is None or ch_idx < 0:
                continue

            role_name = str(role.get("name") or role.get("source") or f"CH{ch_idx}")
            probe_type = str(role.get("probe_type", "AS")).upper()
            purpose = str(role.get("purpose", "")).lower()

            if probe_type == "AS":
                new_role_names[ch_idx] = role_name
                if ch_idx not in new_analog_channels:
                    new_analog_channels.append(ch_idx)
                new_channel_analog_roles.setdefault(ch_idx, []).append(dict(role))
                if role.get("bandpass", False):
                    new_filtered.add(ch_idx)
                    low_hz = float(role.get("bandpass_low", self.filter_low))
                    high_hz = float(role.get("bandpass_high", self.filter_high))
                    if 0 < low_hz < high_hz:
                        new_filter_bands[ch_idx] = (low_hz, high_hz)
                    else:
                        new_filter_bands[ch_idx] = (self.filter_low, self.filter_high)
                    if primary_probe_role is None and purpose == "probe":
                        primary_probe_role = (ch_idx, role)
                else:
                    new_unfiltered.add(ch_idx)
                    if primary_probe_role is None and purpose == "probe":
                        primary_probe_role = (ch_idx, role)
            elif probe_type == "TTL":
                new_role_names.setdefault(ch_idx, role_name)
                if ch_idx not in new_filtered:
                    new_unfiltered.add(ch_idx)

                is_stim_trig = (purpose == "stim_trigger") or (
                    role_name == "env.stim" and probe_type == "TTL"
                )
                if is_stim_trig:
                    stim_trigger_role = (ch_idx, role)
                else:
                    raw_thresh = role.get("ttl_threshold")
                    eff_thresh = (
                        float(raw_thresh)
                        if (raw_thresh is not None and float(raw_thresh) > 0.0)
                        else self.ttl_threshold_v
                    )
                    refrac_ms = float(role.get("ttl_refractory_ms", self.ttl_refractory_ms))

                    existing = self._aux_ttl_detectors.get(role_name)
                    if existing is not None and existing[0] == ch_idx:
                        det = existing[1]
                        det.threshold = eff_thresh
                        det.refractory_s = refrac_ms / 1000.0
                    else:
                        det = OnlineTTLDetector(
                            threshold=eff_thresh,
                            refractory_s=refrac_ms / 1000.0,
                        )
                    new_aux_ttl[role_name] = (ch_idx, det, dict(role))

        new_unfiltered -= new_filtered

        self.channel_role_names = new_role_names
        self.filtered_channels = new_filtered
        self.unfiltered_channels = new_unfiltered
        self._role_analog_channels = new_analog_channels
        self._channel_analog_roles = new_channel_analog_roles
        self._aux_ttl_detectors = new_aux_ttl

        # Rebuild or preserve per-channel OnlineBandpassFilter instances
        new_filters: Dict[int, OnlineBandpassFilter] = {}
        sr = float(self.sample_rate or 30000.0)
        for ch_idx in new_filtered:
            low_hz, high_hz = new_filter_bands.get(
                ch_idx, (self.filter_low, self.filter_high)
            )
            old_filt = self._channel_filters.get(ch_idx)
            old_band = self._channel_filter_bands.get(ch_idx)
            if (
                old_filt is not None
                and old_band == (low_hz, high_hz)
                and getattr(old_filt, "sample_rate", None) == sr
            ):
                new_filters[ch_idx] = old_filt
            else:
                new_filters[ch_idx] = OnlineBandpassFilter(
                    lowcut=low_hz,
                    highcut=high_hz,
                    sampling_rate=sr,
                )
        self._channel_filters = new_filters
        self._channel_filter_bands = new_filter_bands

        if primary_probe_role is not None:
            probe_ch, probe_role = primary_probe_role
            if probe_role.get("bandpass", False):
                p_low = float(probe_role.get("bandpass_low", self.filter_low))
                p_high = float(probe_role.get("bandpass_high", self.filter_high))
                if 0 < p_low < p_high and (
                    p_low != self.filter_low or p_high != self.filter_high
                ):
                    self.filter_low = p_low
                    self.filter_high = p_high
                    self.bandpass_filter = OnlineBandpassFilter(
                        lowcut=self.filter_low,
                        highcut=self.filter_high,
                        sampling_rate=sr,
                    )

        if stim_trigger_role is not None:
            trig_ch, trig_role = stim_trigger_role
            if "trigger_source" in trig_role:
                self.trigger_source = trig_role["trigger_source"]
            if self.trigger_source == "analog" or trig_role.get("is_adc", False):
                self.analog_trigger_channel = trig_ch
            raw_thresh = trig_role.get("ttl_threshold")
            eff_thresh = (
                float(raw_thresh)
                if (raw_thresh is not None and float(raw_thresh) > 0.0)
                else None
            )
            refrac_ms = float(trig_role.get("ttl_refractory_ms", self.ttl_refractory_ms))
            self.set_ttl_params(threshold=eff_thresh, refractory_ms=refrac_ms)

    def set_window_params(
        self,
        window_ms: Optional[float] = None,
        pre_ms: Optional[float] = None,
    ):
        if window_ms is not None:
            self.window_ms = float(window_ms)
        if pre_ms is not None:
            self.pre_stim_ms = float(pre_ms)

    def _queue_sweep_payload(self, payload: Dict[str, Any]):
        """Accumulate a sweep payload and flush in batches to reduce GUI wakeups."""
        self._sweeps_count += 1
        self._pending_payloads.append(payload)
        self._flush_sweep_payloads()

    def _flush_sweep_payloads(self, force: bool = False):
        """Emit accumulated sweep payloads respecting the batch interval/cap."""
        if not self._pending_payloads:
            return
        now = time.time()
        if (
            not force
            and len(self._pending_payloads) < self._sweep_batch_max
            and (now - self._last_sweep_flush) < self._sweep_flush_interval_s
        ):
            return
        payloads = self._pending_payloads
        self._pending_payloads = []
        self._last_sweep_flush = now
        self.sweepsReady.emit(payloads)

    def force_trigger(self) -> bool:
        """Manually force a sweep cut at current head to test slicing and rendering."""
        latest_samp = self.ring_buffer.latest_sample
        if latest_samp is None:
            return False
        window_samples = int((self.window_ms / 1000.0) * self.sample_rate)
        stim_sample = max(0, latest_samp - window_samples)
        self.queue_sweep_trigger(stim_sample, stim_sample / self.sample_rate, source_info="MANUAL_FORCE")
        return True

    def get_recent_continuous(
        self, channel: int = 0, duration_s: float = 1.0
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Return (time_axis_s, samples_uV) for channel over the most recent duration_s."""
        latest = self.ring_buffer.latest_sample
        if latest is None:
            return None
        n_samples = int(duration_s * self.sample_rate)
        if n_samples <= 0:
            return None
        start_samp = max(0, latest - n_samples)
        if start_samp >= latest:
            return None
        ch = max(0, min(channel, self.ring_buffer.n_channels - 1))
        matrix = self.ring_buffer.slice_channels(start_samp, latest, [ch])
        if matrix is None or matrix.size == 0:
            return None
        sig = matrix[0]
        t = np.linspace(-duration_s, 0.0, len(sig), endpoint=False)
        return t, sig

    def run(self):
        """Thread execution loop."""
        self._running = True
        context = zmq.Context()
        data_sub: Optional[zmq.Socket] = None
        event_sub: Optional[zmq.Socket] = None
        heartbeat_req: Optional[zmq.Socket] = None
        poller = zmq.Poller()

        try:
            # 1. Setup Data Socket
            data_sub = context.socket(zmq.SUB)
            data_sub.setsockopt_string(zmq.SUBSCRIBE, "")
            data_sub.setsockopt(zmq.RCVTIMEO, 500)
            data_sub.setsockopt(zmq.LINGER, 0)
            data_url = f"tcp://{self.host}:{self.data_port}"
            data_sub.connect(data_url)
            poller.register(data_sub, zmq.POLLIN)

            # 2. Setup Event Socket (if zmq_raw mode)
            if self.mode == "zmq_raw":
                if self.event_port and self.event_port != self.data_port:
                    event_sub = context.socket(zmq.SUB)
                    event_sub.setsockopt_string(zmq.SUBSCRIBE, "")
                    event_sub.setsockopt(zmq.RCVTIMEO, 500)
                    event_sub.setsockopt(zmq.LINGER, 0)
                    event_url = f"tcp://{self.host}:{self.event_port}"
                    event_sub.connect(event_url)
                    poller.register(event_sub, zmq.POLLIN)

            # 3. Setup Heartbeat REQ socket
            # Both Open Ephys ZMQ Interface ("zmq_raw") and SpikeSpyStreamer ("sliced_erp")
            # listen for REQ heartbeats on heartbeat_port / event_port / data_port + 1.
            hb_target_port = None
            if self.heartbeat_port is None:
                hb_target_port = self.event_port or (self.data_port + 1)
            elif self.heartbeat_port > 0:
                hb_target_port = self.heartbeat_port

            def _create_heartbeat_socket() -> Optional[zmq.Socket]:
                if hb_target_port is None or hb_target_port == self.data_port:
                    return None
                sock = context.socket(zmq.REQ)
                sock.setsockopt(zmq.RCVTIMEO, 500)
                sock.setsockopt(zmq.SNDTIMEO, 500)
                sock.setsockopt(zmq.LINGER, 0)
                sock.connect(f"tcp://{self.host}:{hb_target_port}")
                poller.register(sock, zmq.POLLIN)
                return sock

            def _reset_heartbeat_socket() -> Optional[zmq.Socket]:
                nonlocal heartbeat_req, heartbeat_pending
                heartbeat_pending = False
                if heartbeat_req is not None:
                    try:
                        poller.unregister(heartbeat_req)
                    except Exception:
                        pass
                    try:
                        heartbeat_req.close(linger=0)
                    except Exception:
                        pass
                try:
                    heartbeat_req = _create_heartbeat_socket()
                except Exception:
                    heartbeat_req = None
                return heartbeat_req

            heartbeat_req = _create_heartbeat_socket()
            default_label = "SpikeSpy Streamer" if self.mode == "sliced_erp" else "Open Ephys ZMQ"

            self._plugin_connected = False
            self._last_heartbeat_reply = 0.0
            self._last_data_received = 0.0
            self.connectionStatusChanged.emit(False, f"Connecting to {self.host}:{self.data_port}...")
            logger.info("OpenEphysStreamWorker connecting to %s (mode=%s)", self.host, self.mode)

            last_heartbeat_time = 0.0
            last_stats_time = time.time()
            heartbeat_pending = False

            while self._running:
                now = time.time()

                # Heartbeat every 1.2 seconds; reset REQ socket if a reply times out
                # so we recover cleanly if Open Ephys starts or restarts after us.
                if heartbeat_req is not None:
                    if heartbeat_pending and (now - last_heartbeat_time >= 2.5):
                        _reset_heartbeat_socket()
                    if (
                        heartbeat_req is not None
                        and not heartbeat_pending
                        and (now - last_heartbeat_time >= 1.2)
                    ):
                        try:
                            hb_msg = json.dumps(
                                {
                                    "application": "SpikeSpy",
                                    "client": "SpikeSpy",
                                    "uuid": "spikespy-oe-addin",
                                    "type": "heartbeat",
                                    "time": now,
                                }
                            ).encode("utf-8")
                            heartbeat_req.send(hb_msg, zmq.NOBLOCK)
                            heartbeat_pending = True
                            last_heartbeat_time = now
                        except Exception as e:
                            logger.debug("Heartbeat error: %s", e)
                            _reset_heartbeat_socket()

                # Poll sockets
                socks = dict(poller.poll(20))

                if heartbeat_req is not None and heartbeat_req in socks:
                    try:
                        reply_raw = heartbeat_req.recv(zmq.NOBLOCK)
                        heartbeat_pending = False
                        self._last_heartbeat_reply = now
                        plugin_label = default_label
                        if reply_raw:
                            try:
                                parsed = json.loads(reply_raw.decode("utf-8"))
                                if isinstance(parsed, dict):
                                    self._plugin_info = parsed
                                    plugin_label = self._plugin_info.get("plugin", plugin_label)
                            except Exception:
                                pass
                        if not self._plugin_connected:
                            self._plugin_connected = True
                            self.connectionStatusChanged.emit(
                                True, f"Connected to {plugin_label} ({self.host}:{self.data_port})"
                            )
                    except Exception:
                        heartbeat_pending = False

                if data_sub in socks and socks[data_sub] == zmq.POLLIN:
                    self._handle_data_socket(data_sub)

                if event_sub is not None and event_sub in socks and socks[event_sub] == zmq.POLLIN:
                    self._handle_event_socket(event_sub)

                # Flush any stalled single-channel block (e.g. 1-ch stream or pause)
                if (
                    self._pending_block_start is not None
                    and (now - self._pending_block_time) >= 0.05
                ):
                    self._flush_pending_channel_block()

                # Check pending sweeps
                self._check_pending_sweeps()
                self._flush_sweep_payloads()

                # Connection timeout check: if connected but no activity for > 3.5s
                last_act = max(self._last_heartbeat_reply, self._last_data_received)
                if self._plugin_connected and last_act > 0 and (now - last_act > 3.5):
                    self._plugin_connected = False
                    self.connectionStatusChanged.emit(
                        False, f"Waiting for {default_label} on {self.host}:{self.data_port}..."
                    )

                # Emit stats every 1 second
                if now - last_stats_time >= 1.0:
                    avg_freq = 0.0
                    if len(self._stim_intervals) > 0:
                        avg_freq = 1.0 / np.mean(self._stim_intervals[-10:])
                    self.statsUpdated.emit(
                        {
                            "sample_rate": self.sample_rate,
                            "sweeps_received": self._sweeps_count,
                            "spikes_received": self._spikes_count,
                            "packets_received": self._packets_count,
                            "last_stim_freq_hz": float(avg_freq),
                            "plugin_connected": self._plugin_connected,
                            "last_ping_s": (now - last_act) if last_act > 0 else -1.0,
                            "plugin_info": self._plugin_info,
                        }
                    )
                    last_stats_time = now

        except Exception as e:
            logger.exception("Error in stream worker: %s", e)
            self.connectionStatusChanged.emit(False, str(e))
        finally:
            self._flush_sweep_payloads(force=True)
            if data_sub is not None:
                data_sub.close(0)
            if event_sub is not None:
                event_sub.close(0)
            if heartbeat_req is not None:
                heartbeat_req.close(0)
            context.term()
            self._plugin_connected = False
            self.connectionStatusChanged.emit(False, "Disconnected")

    def _handle_data_socket(self, sock: zmq.Socket):
        """Handle incoming packets from the data socket."""
        while self._running:
            try:
                msg_parts = sock.recv_multipart(zmq.NOBLOCK)
            except zmq.ZMQError:
                break

            if not msg_parts:
                continue

            self._packets_count += 1
            self._last_data_received = time.time()
            if not self._plugin_connected:
                self._plugin_connected = True
                self.connectionStatusChanged.emit(
                    True, f"Receiving live stream on {self.host}:{self.data_port}"
                )

            # Check if pre-sliced ERP packet
            if self.mode == "sliced_erp" or msg_parts[0].startswith(b"ERP"):
                self._parse_sliced_erp_packet(msg_parts)
                continue

            if len(msg_parts) > 0 and msg_parts[0].startswith(b"{"):
                try:
                    hdr = json.loads(msg_parts[0].decode("utf-8"))
                    if hdr.get("type") == "erp_sweep":
                        self._parse_sliced_erp_packet(msg_parts)
                        continue
                except Exception:
                    pass

            # Standard Open Ephys ZMQ Interface packet:
            # Typically multipart: [header_json_or_envelope, raw_data_buffer]
            self._parse_raw_zmq_data_packet(msg_parts)

    def _populate_role_aliases(self, aux_sweeps: Dict[str, Any], channel_idx: int, ch_waveform: np.ndarray):
        """Populate semantic role names and .bp aliases for a channel waveform."""
        with self._lock:
            roles_for_ch = list(self._channel_analog_roles.get(channel_idx, []))
            mapped_name = self.channel_role_names.get(channel_idx)
        if roles_for_ch:
            for role in roles_for_ch:
                rname = str(role.get("name") or "").strip()
                if not rname:
                    continue
                aux_sweeps[rname] = ch_waveform
                if role.get("bandpass", False) and not rname.endswith(".bp"):
                    aux_sweeps[f"{rname}.bp"] = ch_waveform
        elif mapped_name:
            aux_sweeps[mapped_name] = ch_waveform

    def _parse_sliced_erp_packet(self, parts: List[bytes]):
        """Parse pre-sliced ERP packet from SpikeSpyStreamer C++ plugin or mock server."""
        try:
            # Header is either JSON string in parts[0] or parts[1]
            header = None
            raw_data = None

            for p in parts:
                if p.startswith(b"{") and p.endswith(b"}"):
                    try:
                        header = json.loads(p.decode("utf-8"))
                    except Exception:
                        pass
                elif len(p) >= 8:  # potential audio buffer
                    raw_data = p

            if header is None and parts[0].startswith(b"ERP:"):
                # "ERP:<json_len>" followed by json and buffer
                header_str = parts[0][4:].decode("utf-8")
                header = json.loads(header_str)

            if header is None:
                return

            sample_rate = float(header.get("sample_rate", self.sample_rate))
            self.sample_rate = sample_rate
            pre_offset_s = float(header.get("pre_offset_s", self.pre_stim_ms / 1000.0))
            stim_ts = float(header.get("stim_timestamp_s", time.time()))
            spike_latencies = header.get("spike_latencies_s", {})

            # Parse aux_sweeps or 2D waveforms
            aux_sweeps: Dict[str, Any] = {}
            raw_aux = header.get("aux_sweeps", None)
            raw_waveforms = header.get("waveforms", None)

            if isinstance(raw_aux, dict):
                for k, v in raw_aux.items():
                    aux_sweeps[str(k)] = np.array(v, dtype=np.float32) if isinstance(v, list) else v
            elif isinstance(raw_waveforms, dict):
                for k, v in raw_waveforms.items():
                    aux_sweeps[str(k)] = np.array(v, dtype=np.float32) if isinstance(v, list) else v
            elif isinstance(raw_waveforms, list):
                wf_mat = np.array(raw_waveforms, dtype=np.float32)
                if wf_mat.ndim == 2:
                    if self._stream_n_channels != wf_mat.shape[0]:
                        with self._lock:
                            self._stream_n_channels = int(wf_mat.shape[0])
                            if self.channel_roles:
                                self._rebuild_role_mappings_locked()
                    for ch_idx in range(wf_mat.shape[0]):
                        ch_wf = wf_mat[ch_idx]
                        aux_sweeps[f"CH{ch_idx}"] = ch_wf
                        self._populate_role_aliases(aux_sweeps, ch_idx, ch_wf)

            if self.channel_roles:
                with self._lock:
                    role_names_snap = dict(self.channel_role_names)
                    roles_snap = {k: list(v) for k, v in self._channel_analog_roles.items()}
                for ch_idx, rname in role_names_snap.items():
                    ch_key = f"CH{ch_idx}"
                    if ch_key in aux_sweeps:
                        wf_val = aux_sweeps[ch_key]
                        if ch_idx in roles_snap:
                            for role in roles_snap[ch_idx]:
                                rn = str(role.get("name") or "").strip()
                                if rn:
                                    aux_sweeps.setdefault(rn, wf_val)
                                    if role.get("bandpass", False) and not rn.endswith(".bp"):
                                        aux_sweeps.setdefault(f"{rn}.bp", wf_val)
                        else:
                            aux_sweeps.setdefault(rname, wf_val)

            # Waveform data
            waveform = None
            if "waveform" in header and isinstance(header["waveform"], list):
                waveform = np.array(header["waveform"], dtype=np.float32)
            elif raw_data is not None:
                dtype = np.float32 if header.get("dtype", "float32") == "float32" else np.int16
                waveform = np.frombuffer(raw_data, dtype=dtype)
                if header.get("dtype") == "int16" and "bit_volts" in header:
                    waveform = waveform.astype(np.float32) * float(header["bit_volts"])

            if waveform is not None:
                if self.apply_filter:
                    with self._lock:
                        should_filter_primary = (
                            self.filtered_channels is None
                            or self.rec_channel in self.filtered_channels
                        )
                        if should_filter_primary:
                            filt = self._channel_filters.get(
                                self.rec_channel, self.bandpass_filter
                            )
                            waveform = filt.process(waveform)

                if self.channel_roles:
                    self._populate_role_aliases(aux_sweeps, self.rec_channel, waveform)

                # Record stim interval
                if self._last_stim_timestamp is not None:
                    dt = stim_ts - self._last_stim_timestamp
                    if 0.001 < dt < 30.0:
                        self._stim_intervals.append(dt)
                self._last_stim_timestamp = stim_ts

                payload = {
                    "stim_timestamp_s": stim_ts,
                    "waveform": waveform,
                    "pre_offset_s": pre_offset_s,
                    "spike_latencies_s": spike_latencies,
                    "aux_sweeps": aux_sweeps,
                    "channel_waveforms": aux_sweeps,
                    "sample_rate": sample_rate,
                }
                self._queue_sweep_payload(payload)
        except Exception as e:
            logger.error("Failed to parse sliced ERP packet: %s", e)

    def _parse_raw_zmq_data_packet(self, parts: List[bytes]):
        """Parse continuous multi-channel raw stream or events from Open Ephys ZMQ Interface."""
        try:
            # Open Ephys ZMQ Interface protocol sends:
            # Case A (3 frames): [Envelope (b'DATA\0' / b'EVENT\0' or JSON envelope), Header JSON, Binary Buffer]
            # Case B (2 frames): [Header JSON, Binary Buffer] or [Topic, Header JSON]
            # Case C (1 frame):  [Header JSON]
            json_parts: List[Tuple[int, Dict[str, Any]]] = []
            for i, p in enumerate(parts):
                cleaned_p = p.strip(b"\x00 \r\n\t")
                if cleaned_p.startswith(b"{") and cleaned_p.endswith(b"}"):
                    try:
                        parsed = json.loads(cleaned_p.decode("utf-8"))
                        if isinstance(parsed, dict):
                            json_parts.append((i, parsed))
                    except Exception:
                        pass

            if not json_parts:
                if len(parts) == 1 and parts[0].strip(b"\x00 \r\n\t").startswith(b"{"):
                    self._handle_event_json(parts[0])
                return

            envelope = ""
            if len(json_parts) >= 2 and json_parts[0][0] == 0:
                # Both frame 0 (envelope JSON) and frame 1 (header JSON) are JSON dicts
                env_dict = json_parts[0][1]
                hdr_idx, hdr_dict = json_parts[1]
                envelope = str(env_dict.get("type", "")).lower()
                meta = {**env_dict, **hdr_dict}
                buf = parts[hdr_idx + 1] if (hdr_idx + 1 < len(parts)) else None
            else:
                meta_idx, meta = json_parts[0]
                if meta_idx > 0:
                    envelope = parts[0].decode("utf-8", errors="ignore").strip("\x00 \r\n\t").lower()
                buf = parts[meta_idx + 1] if (meta_idx + 1 < len(parts)) else None

            if len(parts) == 1:
                self._handle_event_json(parts[0])
                return

            # Flatten Open Ephys ZMQ Interface nested "content" object while keeping raw meta
            content = meta.get("content")
            fields = {**meta, **content} if isinstance(content, dict) else meta

            msg_type = (
                envelope
                or str(meta.get("type", fields.get("type", fields.get("stream", "continuous"))))
            ).lower()

            # Check if this packet is an Event or Spike broadcast on the data socket
            if (
                "event" in msg_type
                or str(meta.get("type", "")).lower() in ("event", "ttl")
                or str(fields.get("type", "")).lower() == "ttl"
                or "line" in fields
                or "event_line" in fields
                or (isinstance(fields.get("data"), dict) and "line" in fields["data"])
            ):
                line = fields.get("line", fields.get("event_line"))
                state = fields.get("state", fields.get("event_state"))
                if (line is None or state is None) and isinstance(fields.get("data"), dict):
                    if line is None:
                        line = fields["data"].get("line")
                    if state is None:
                        state = fields["data"].get("state")

                if (line is None or state is None) and buf is not None and len(buf) >= 2:
                    line = int(buf[0])
                    state = int(buf[1])

                if line is not None or state is not None:
                    raw_sn = fields.get("sample_num")
                    if raw_sn is None:
                        raw_sn = fields.get("sample_number")
                    if raw_sn is None:
                        raw_sn = fields.get("sampleIndex")
                    sample_num = int(raw_sn) if raw_sn is not None else 0

                    if raw_sn is not None and self.sample_rate:
                        ts = float(sample_num) / float(self.sample_rate)
                    else:
                        raw_ts = fields.get("timestamp")
                        if raw_ts is not None:
                            ts = float(raw_ts)
                            if ts > 1e11:
                                ts /= 1000.0
                        else:
                            ts = time.time()

                    is_rising = False
                    if isinstance(state, (int, float, bool)):
                        is_rising = (int(state) == 1)
                    elif isinstance(state, str):
                        is_rising = state.strip().lower() in ("1", "true", "high", "rising", "on")

                    if not is_rising:
                        action_status = f"IGNORED (state={state})"
                    elif self.trigger_source != "ttl":
                        action_status = f"IGNORED (trigger_source={self.trigger_source})"
                    else:
                        action_status = "TRIGGER QUEUED"

                    latest_buf = self.ring_buffer.latest_sample
                    if hasattr(self, "debugEventEmitted"):
                        self.debugEventEmitted.emit({
                            "timestamp": time.time(),
                            "topic": envelope or "DATA_SOCK_EVENT",
                            "type": fields.get("type", "ttl"),
                            "line": line,
                            "state": state,
                            "sample_number": sample_num,
                            "buffer_latest_sample": latest_buf,
                            "clock_skew": (sample_num - latest_buf) if latest_buf is not None else 0,
                            "status": action_status,
                            "raw_meta": meta,
                        })

                    if is_rising and self.trigger_source == "ttl":
                        self.queue_sweep_trigger(sample_num, ts, source_info=f"TTL_Line{line if line is not None else 1}")
                else:
                    self._handle_event_json(json.dumps(fields).encode("utf-8"))
                return

            if "spike" in msg_type or str(meta.get("type", "")).lower() == "spike":
                self._handle_event_json(json.dumps(fields).encode("utf-8"))
                return

            # Continuous data handling
            if buf is None:
                return

            sr = float(fields.get("sample_rate", self.sample_rate))
            sr_changed = (sr != self.sample_rate)
            if sr > 0:
                self._update_sample_rate(sr)

            raw_start = fields.get("sample_number")
            if raw_start is None:
                raw_start = fields.get("start_sample")
            if raw_start is None:
                raw_start = fields.get("sample_num")
            if raw_start is None:
                raw_start = fields.get("sampleIndex")
            if raw_start is not None:
                start_sample = int(raw_start)
            else:
                latest_buf = self.ring_buffer.latest_sample
                start_sample = (latest_buf + 1) if latest_buf is not None else 0

            dtype_str = str(fields.get("dtype", "float32")).lower()
            dt = np.float32 if "float" in dtype_str else np.int16
            itemsize = np.dtype(dt).itemsize

            channel_num = fields.get(
                "channel_num", fields.get("channel_index", fields.get("channel"))
            )
            has_channel_count = any(
                k in fields for k in ("channel_count", "channels", "n_channels")
            )
            n_chans = int(
                fields.get(
                    "channel_count",
                    fields.get("channels", fields.get("n_channels", 1)),
                )
            )
            n_samples = int(
                fields.get(
                    "num_samples",
                    fields.get(
                        "n_samples",
                        fields.get(
                            "sample_count",
                            len(buf) // (itemsize * max(1, n_chans)),
                        ),
                    ),
                )
            )
            n_real_samples = int(fields.get("n_real_samples", n_samples))

            arr = np.frombuffer(buf, dtype=dt)

            # Standard Open Ephys ZMQ Interface: one channel per message ("channel_num")
            if channel_num is not None and (not has_channel_count or n_chans == 1):
                if arr.size < n_real_samples or n_real_samples <= 0:
                    return
                ch_data = arr[:n_real_samples]
                if dt == np.int16:
                    ch_data = ch_data.astype(np.float32)
                    if "bit_volts" in fields:
                        ch_data *= float(fields["bit_volts"])
                else:
                    ch_data = np.array(ch_data, dtype=np.float32, copy=True)
                self._ingest_single_channel_chunk(int(channel_num), ch_data, start_sample)
                return

            if arr.size == n_chans * n_samples:
                data = arr.reshape((n_chans, n_samples))
                transposed = False
            elif arr.size == n_samples * n_chans:
                data = arr.reshape((n_samples, n_chans))
                transposed = True
            elif n_chans == 1 and arr.size == n_samples:
                data = arr.reshape((1, n_samples))
                transposed = False
            else:
                return

            # Convert int16 to microvolts if needed (single allocation, then in-place scale)
            if dt == np.int16 and "bit_volts" in fields:
                data = data.astype(np.float32)
                data *= float(fields["bit_volts"])
            elif dt == np.int16:
                data = data.astype(np.float32)
            if transposed:
                data = data.T

            n_stream_ch = int(data.shape[0])
            if self._stream_n_channels != n_stream_ch or sr_changed:
                with self._lock:
                    self._stream_n_channels = n_stream_ch
                    if self.channel_roles:
                        self._rebuild_role_mappings_locked()

            # Check analog TTL trigger if configured, using unfiltered samples
            if self.trigger_source == "analog" and 0 <= self.analog_trigger_channel < n_stream_ch:
                adc_data = data[self.analog_trigger_channel, :]
                with self._lock:
                    triggers = self.analog_ttl_detector.process_chunk(
                        adc_data, start_sample, sample_rate=self.sample_rate
                    )
                for t_sample in triggers:
                    self.queue_sweep_trigger(t_sample, t_sample / self.sample_rate)

            # Check secondary TTL roles (e.g. rec.button, env.stimVolt) using unfiltered samples
            aux_ttl_events: List[Tuple[str, float, Dict[str, Any]]] = []
            with self._lock:
                aux_ttl_items = list(self._aux_ttl_detectors.items())
            if aux_ttl_items:
                for role_name, (ch_idx, det, role_meta) in aux_ttl_items:
                    if 0 <= ch_idx < n_stream_ch:
                        ttl_chunk = data[ch_idx, :]
                        with self._lock:
                            detected = det.process_chunk(
                                ttl_chunk, start_sample, sample_rate=self.sample_rate
                            )
                        for trig_ev in detected:
                            t_sample = int(trig_ev)
                            payload = {
                                "name": role_name,
                                "purpose": role_meta.get("purpose", ""),
                                "sample_number": t_sample,
                                "source": role_meta.get("source", f"CH{ch_idx}"),
                            }
                            if isinstance(trig_ev, dict) or hasattr(trig_ev, "get"):
                                if "duration_s" in trig_ev:
                                    payload["duration_s"] = float(trig_ev["duration_s"])
                                if "max_val" in trig_ev:
                                    payload["max_val"] = float(trig_ev["max_val"])
                            aux_ttl_events.append(
                                ("ttl_role", t_sample / self.sample_rate, payload)
                            )

            for ev_kind, ev_ts, ev_payload in aux_ttl_events:
                self.protocolEventReady.emit(ev_kind, ev_ts, ev_payload)

            # Filter the continuous stream selectively based on channel_roles
            if self.apply_filter:
                with self._lock:
                    if self.channel_roles and self.filtered_channels is not None:
                        if self.filtered_channels:
                            data = data.astype(np.float32, copy=True)
                            for ch_idx in sorted(self.filtered_channels):
                                if 0 <= ch_idx < n_stream_ch:
                                    filt = self._channel_filters.get(ch_idx)
                                    if filt is None:
                                        low_hz, high_hz = self._channel_filter_bands.get(
                                            ch_idx, (self.filter_low, self.filter_high)
                                        )
                                        filt = OnlineBandpassFilter(
                                            lowcut=low_hz,
                                            highcut=high_hz,
                                            sampling_rate=self.sample_rate,
                                        )
                                        self._channel_filters[ch_idx] = filt
                                    data[ch_idx, :] = filt.process(data[ch_idx, :])
                    else:
                        data = self.bandpass_filter.process(data)

            # Append to circular ring buffer (RingBuffer handles channel expansion thread-safely)
            self.ring_buffer.append(data, start_sample)

        except Exception as e:
            logger.debug("Error parsing raw zmq data packet: %s", e)

    def _handle_event_socket(self, sock: zmq.Socket):
        """Handle events arriving on event socket."""
        while self._running:
            try:
                msg = sock.recv_multipart(zmq.NOBLOCK)
            except zmq.ZMQError:
                break

            for part in msg:
                cleaned_part = part.strip(b"\x00 \r\n\t")
                if cleaned_part.startswith(b"{") and cleaned_part.endswith(b"}"):
                    self._handle_event_json(cleaned_part)

    def _handle_event_json(self, raw_json: bytes):
        """Process event JSON (TTL events, Spikes, APTrack protocol events)."""
        try:
            cleaned = raw_json.strip(b"\x00 \r\n\t")
            evt = json.loads(cleaned.decode("utf-8"))
        except Exception:
            return

        evt_type = str(evt.get("type", evt.get("event_type", ""))).lower()

        line = evt.get("line")
        state = evt.get("state")
        if isinstance(evt.get("data"), dict):
            if line is None:
                line = evt["data"].get("line")
            if state is None:
                state = evt["data"].get("state")
        if state is None:
            state = 1

        # 1. TTL Trigger Event (Digital Line)
        if evt_type in ("ttl", "event") or line is not None or "line" in evt:
            sample_num = int(
                evt.get(
                    "sample_number",
                    evt.get("sample_num", evt.get("sampleIndex", int(time.time() * self.sample_rate))),
                )
            )
            timestamp_s = float(evt.get("timestamp", sample_num / self.sample_rate if self.sample_rate else time.time()))

            is_rising = False
            if isinstance(state, (int, float, bool)):
                is_rising = (int(state) == 1)
            elif isinstance(state, str):
                is_rising = state.strip().lower() in ("1", "true", "high", "rising", "on")

            action_status = "TRIGGER QUEUED" if is_rising else f"IGNORED (state={state})"
            latest_buf = self.ring_buffer.latest_sample
            if hasattr(self, "debugEventEmitted"):
                self.debugEventEmitted.emit({
                    "timestamp": time.time(),
                    "topic": "EVENT_SOCK",
                    "type": evt_type or "ttl",
                    "line": line if line is not None else 1,
                    "state": state,
                    "sample_number": sample_num,
                    "buffer_latest_sample": latest_buf,
                    "clock_skew": (sample_num - latest_buf) if latest_buf is not None else 0,
                    "status": action_status,
                    "raw_meta": evt,
                })

            if is_rising and self.trigger_source == "ttl":
                self.queue_sweep_trigger(sample_num, timestamp_s, source_info=f"TTL_Line{line if line is not None else 1}")

        # 2. Spike Event
        elif evt_type == "spike" or "electrode" in evt:
            electrode = evt.get("electrode", 0)
            unit_name = evt.get("unit_name", f"unit_ch{electrode}")
            sample_num = int(evt.get("sample_number", evt.get("sample_num", int(time.time() * self.sample_rate))))
            spike_ts = float(evt.get("timestamp", sample_num / self.sample_rate))
            self._spikes_count += 1

            matched_pending = False
            with self._pending_lock:
                for sweep in self._pending_sweeps:
                    start_s = sweep.stim_sample - sweep.pre_samples
                    end_s = sweep.stim_sample + sweep.window_samples
                    if start_s <= sample_num <= end_s:
                        matched_pending = True
                        if unit_name not in sweep.spike_latencies:
                            sweep.spike_latencies[unit_name] = (sample_num - sweep.stim_sample) / self.sample_rate

            if not matched_pending:
                self.spikeEventsReady.emit(unit_name, np.array([spike_ts]))

        # 3. Protocol / MessageCenter text event (e.g. APTrack stimulus protocol, setStimVoltage)
        elif evt_type in ("message", "text", "protocol") or "text" in evt:
            text = evt.get("text", "")
            ts = float(evt.get("timestamp", time.time()))
            self.protocolEventReady.emit(evt_type, ts, evt)
            if "spikeSampleNumber" in text:
                try:
                    parts = text.split(":")
                    samp = int(parts[1].strip())
                    self.spikeEventsReady.emit("APTrack_unit", np.array([samp / self.sample_rate]))
                except Exception:
                    pass

    def queue_sweep_trigger(
        self, stim_sample: int, stim_timestamp_s: float, source_info: str = "TTL"
    ):
        """Queue a pending sweep to be sliced when future samples have arrived in the ring buffer."""
        window_samples = int((self.window_ms / 1000.0) * self.sample_rate)
        pre_samples = int((self.pre_stim_ms / 1000.0) * self.sample_rate)

        latest_samp = self.ring_buffer.latest_sample
        earliest_samp = self.ring_buffer.earliest_sample

        adjusted_stim_sample = stim_sample
        skew = (stim_sample - latest_samp) if latest_samp is not None else 0

        # Auto-align clock if sample is wildly desynchronized (e.g. 0-based relative clock vs hardware clock)
        if getattr(self, "auto_align_clock", True) and latest_samp is not None:
            if earliest_samp is not None and (stim_sample < earliest_samp or stim_sample > latest_samp + int(self.sample_rate * 2.0)):
                adjusted_stim_sample = max(0, latest_samp - int(0.01 * self.sample_rate))
                logger.info(
                    "Auto-aligning trigger sample: original %d -> adjusted %d (latest %d, skew %d)",
                    stim_sample,
                    adjusted_stim_sample,
                    latest_samp,
                    skew,
                )

        # Record stim interval
        if self._last_stim_timestamp is not None:
            dt = stim_timestamp_s - self._last_stim_timestamp
            if 0.001 < dt < 30.0:
                self._stim_intervals.append(dt)
        self._last_stim_timestamp = stim_timestamp_s

        with self._lock:
            aux_chans = list(self.active_channels)
            for ch_idx in self._role_analog_channels:
                if ch_idx not in aux_chans:
                    aux_chans.append(ch_idx)

        pending = PendingSweep(
            stim_sample=adjusted_stim_sample,
            stim_timestamp_s=stim_timestamp_s,
            window_samples=window_samples,
            pre_samples=pre_samples,
            rec_chan=self.rec_channel,
            aux_chans=aux_chans,
        )

        with self._pending_lock:
            if len(self._pending_sweeps) > 2000:
                self._pending_sweeps.pop(0)
            self._pending_sweeps.append(pending)

    def _check_pending_sweeps(self):
        """Check if pending sweeps can be sliced from the ring buffer."""
        with self._pending_lock:
            if not self._pending_sweeps:
                return
            sweeps_snapshot = list(self._pending_sweeps)

        latest_samp = self.ring_buffer.latest_sample
        if latest_samp is None:
            return
        earliest_samp = self.ring_buffer.earliest_sample
        if earliest_samp is None:
            earliest_samp = latest_samp
        n_avail = self.ring_buffer.n_channels

        ready_items = []
        finished_ids = set()

        with self._lock:
            role_analog_extra = list(self._role_analog_channels)

        for sweep in sweeps_snapshot:
            start_samp = sweep.stim_sample - sweep.pre_samples
            end_samp = sweep.stim_sample + sweep.window_samples

            # Cheap check first: do not copy data that cannot be complete yet.
            if end_samp > latest_samp:
                if start_samp < earliest_samp:
                    logger.warning(
                        "Sweep at sample %d dropped (before earliest_samp %d)",
                        sweep.stim_sample,
                        earliest_samp,
                    )
                    finished_ids.add(id(sweep))
                    if hasattr(self, "debugEventEmitted"):
                        self.debugEventEmitted.emit({
                            "timestamp": time.time(),
                            "topic": "SWEEP_DROP",
                            "type": "drop",
                            "line": "--",
                            "state": "--",
                            "sample_number": sweep.stim_sample,
                            "buffer_latest_sample": latest_samp,
                            "clock_skew": sweep.stim_sample - latest_samp,
                            "status": f"DROPPED: start_samp {start_samp} < earliest_samp {earliest_samp}",
                        })
                continue

            if start_samp < earliest_samp:
                logger.warning(
                    "Sweep at sample %d dropped (before earliest_samp %d)",
                    sweep.stim_sample,
                    earliest_samp,
                )
                finished_ids.add(id(sweep))
                if hasattr(self, "debugEventEmitted"):
                    self.debugEventEmitted.emit({
                        "timestamp": time.time(),
                        "topic": "SWEEP_DROP",
                        "type": "drop",
                        "line": "--",
                        "state": "--",
                        "sample_number": sweep.stim_sample,
                        "buffer_latest_sample": latest_samp,
                        "clock_skew": sweep.stim_sample - latest_samp,
                        "status": f"DROPPED: start_samp {start_samp} < earliest_samp {earliest_samp}",
                    })
                continue

            target_chans = list(sweep.aux_chans) if sweep.aux_chans else list(self.active_channels)
            for ch_idx in role_analog_extra:
                if ch_idx not in target_chans:
                    target_chans.append(ch_idx)
            if sweep.rec_chan not in target_chans:
                target_chans = sorted(list(set(target_chans + [sweep.rec_chan])))

            valid_chans = [ch for ch in target_chans if 0 <= ch < n_avail]
            if not valid_chans:
                valid_chans = [min(sweep.rec_chan, n_avail - 1)]

            active_matrix = self.ring_buffer.slice_channels(start_samp, end_samp, valid_chans)
            if active_matrix is None:
                logger.warning(
                    "Sweep at sample %d dropped (before earliest_samp %d)",
                    sweep.stim_sample,
                    self.ring_buffer.earliest_sample,
                )
                finished_ids.add(id(sweep))
                continue

            ready_items.append((sweep, valid_chans, active_matrix))
            finished_ids.add(id(sweep))

        # Update pending list by identity, never by dataclass equality.
        with self._pending_lock:
            self._pending_sweeps = [
                s for s in self._pending_sweeps if id(s) not in finished_ids
            ]

        if hasattr(self, "debugEventEmitted"):
            for sweep, valid_chans, _ in ready_items:
                self.debugEventEmitted.emit({
                    "timestamp": time.time(),
                    "topic": "SWEEP_READY",
                    "type": "slice",
                    "line": "--",
                    "state": "--",
                    "sample_number": sweep.stim_sample,
                    "buffer_latest_sample": latest_samp,
                    "clock_skew": sweep.stim_sample - latest_samp,
                    "status": f"SWEEP SLICED ({len(valid_chans)} channels)",
                })

        # Process ready sweeps outside the lock; data is already filtered on ingest.
        for sweep, valid_chans, active_matrix in ready_items:
            if sweep.rec_chan in valid_chans:
                rec_idx = valid_chans.index(sweep.rec_chan)
                primary_waveform = active_matrix[rec_idx]
            else:
                primary_waveform = active_matrix[0]

            aux_sweeps: Dict[str, Any] = {}
            for ch, ch_waveform in zip(valid_chans, active_matrix):
                aux_sweeps[f"CH{ch}"] = ch_waveform
                self._populate_role_aliases(aux_sweeps, ch, ch_waveform)

            payload = {
                "stim_timestamp_s": sweep.stim_timestamp_s,
                "waveform": primary_waveform,
                "pre_offset_s": sweep.pre_samples / self.sample_rate,
                "spike_latencies_s": sweep.spike_latencies,
                "aux_sweeps": aux_sweeps,
                "channel_waveforms": aux_sweeps,
                "sample_rate": self.sample_rate,
            }
            self._queue_sweep_payload(payload)
