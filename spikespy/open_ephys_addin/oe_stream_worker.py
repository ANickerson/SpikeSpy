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
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import zmq
from PySide6.QtCore import QObject, QThread, Signal, Slot
from scipy import signal

from spikespy.processing import OnlineBandpassFilter, OnlineTTLDetector

logger = logging.getLogger(__name__)


class RingBuffer:
    """Thread-safe circular ring buffer for multi-channel continuous streaming data."""

    def __init__(self, n_channels: int, capacity_samples: int):
        self.n_channels = n_channels
        self.capacity = capacity_samples
        self.buffer = np.zeros((n_channels, capacity_samples), dtype=np.float32)
        self.timestamps = np.zeros(capacity_samples, dtype=np.int64)
        self.write_idx = 0
        self.total_samples_written = 0
        self.first_sample_index: Optional[int] = None
        self.lock = threading.Lock()

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
                self.timestamps[w : w + n_samples] = np.arange(
                    start_sample_index, start_sample_index + n_samples, dtype=np.int64
                )
            else:
                first_chunk = cap - w
                second_chunk = n_samples - first_chunk
                self.buffer[:n_ch, w:cap] = samples[:, :first_chunk]
                self.timestamps[w:cap] = np.arange(
                    start_sample_index, start_sample_index + first_chunk, dtype=np.int64
                )
                self.buffer[:n_ch, :second_chunk] = samples[:, first_chunk:]
                self.timestamps[:second_chunk] = np.arange(
                    start_sample_index + first_chunk,
                    start_sample_index + n_samples,
                    dtype=np.int64,
                )

            self.write_idx = (w + n_samples) % cap
            self.total_samples_written += n_samples

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
            if self.total_samples_written == 0 or self.first_sample_index is None:
                return None

            latest_sample = int(
                self.timestamps[(self.write_idx - 1) % self.capacity]
            )
            earliest_sample = (
                latest_sample - self.capacity + 1
                if self.total_samples_written >= self.capacity
                else self.first_sample_index
            )

            # Check if data is within available window
            if end_sample_index - 1 > latest_sample or start_sample_index < earliest_sample:
                return None

            # Calculate index position in ring buffer
            # Distance from latest_sample:
            dist_from_latest_start = latest_sample - start_sample_index
            buf_start_idx = (self.write_idx - 1 - dist_from_latest_start) % self.capacity

            if buf_start_idx + requested_len <= self.capacity:
                if channel is None:
                    return self.buffer[:, buf_start_idx : buf_start_idx + requested_len].copy()
                return self.buffer[channel, buf_start_idx : buf_start_idx + requested_len].copy()
            else:
                first_part = self.capacity - buf_start_idx
                second_part = requested_len - first_part
                if channel is None:
                    out = np.empty((self.n_channels, requested_len), dtype=np.float32)
                    out[:, :first_part] = self.buffer[:, buf_start_idx:]
                    out[:, first_part:] = self.buffer[:, :second_part]
                    return out
                out = np.empty(requested_len, dtype=np.float32)
                out[:first_part] = self.buffer[channel, buf_start_idx:]
                out[first_part:] = self.buffer[channel, :second_part]
                return out



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
    spikeEventsReady = Signal(str, object)
    protocolEventReady = Signal(str, float, dict)
    statsUpdated = Signal(dict)
    connectionStatusChanged = Signal(bool, str)

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
        self.filter_low = filter_low
        self.filter_high = filter_high
        self.sample_rate = sample_rate

        self._running = False
        self._pending_sweeps: List[PendingSweep] = []
        self._pending_lock = threading.Lock()

        # Ring buffer: default 16 channels, 15 seconds at sample_rate
        cap = int(sample_rate * 15.0)
        self.ring_buffer = RingBuffer(n_channels=16, capacity_samples=cap)
        self.bandpass_filter = OnlineBandpassFilter(
            lowcut=filter_low, highcut=filter_high, sampling_rate=sample_rate
        )
        self.analog_ttl_detector = OnlineTTLDetector()

        # Metrics
        self._sweeps_count = 0
        self._spikes_count = 0
        self._packets_count = 0
        self._last_stim_timestamp: Optional[float] = None
        self._stim_intervals: List[float] = []

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
            self.bandpass_filter.reset()

    def set_filtering(self, enabled: bool):
        self.apply_filter = enabled
        if not enabled:
            self.bandpass_filter.reset()

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

            # 2. Setup Event Socket (if different port or zmq_raw mode)
            if self.mode == "zmq_raw":
                event_sub = context.socket(zmq.SUB)
                event_sub.setsockopt_string(zmq.SUBSCRIBE, "")
                event_sub.setsockopt(zmq.RCVTIMEO, 500)
                event_sub.setsockopt(zmq.LINGER, 0)
                event_url = f"tcp://{self.host}:{self.event_port}"
                event_sub.connect(event_url)
                poller.register(event_sub, zmq.POLLIN)

                # Heartbeat REQ socket for Open Ephys ZMQ Interface (only if separate port)
                if self.heartbeat_port and self.heartbeat_port != self.event_port and self.heartbeat_port != self.data_port:
                    heartbeat_req = context.socket(zmq.REQ)
                    heartbeat_req.setsockopt(zmq.RCVTIMEO, 500)
                    heartbeat_req.setsockopt(zmq.SNDTIMEO, 500)
                    heartbeat_req.setsockopt(zmq.LINGER, 0)
                    heartbeat_req.connect(f"tcp://{self.host}:{self.heartbeat_port}")

            self.connectionStatusChanged.emit(True, f"Connected to {self.host}:{self.data_port}")
            logger.info("OpenEphysStreamWorker connected to %s (mode=%s)", self.host, self.mode)

            last_heartbeat_time = 0.0
            last_stats_time = time.time()

            while self._running:
                now = time.time()

                # Heartbeat every 2.0 seconds
                if self.mode == "zmq_raw" and heartbeat_req is not None:
                    if now - last_heartbeat_time >= 2.0:
                        try:
                            hb_msg = json.dumps(
                                {
                                    "application": "SpikeSpy",
                                    "uuid": "spikespy-oe-addin",
                                    "type": "heartbeat",
                                }
                            ).encode("utf-8")
                            heartbeat_req.send(hb_msg, zmq.NOBLOCK)
                            # Non-blocking receive or ignore reply
                            poller_hb = zmq.Poller()
                            poller_hb.register(heartbeat_req, zmq.POLLIN)
                            if poller_hb.poll(50):
                                heartbeat_req.recv()
                            else:
                                try:
                                    poller_hb.unregister(heartbeat_req)
                                    heartbeat_req.close(linger=0)
                                    heartbeat_req = ctx.socket(zmq.REQ)
                                    heartbeat_req.connect(f"tcp://{self.host}:{self.event_port}")
                                except Exception:
                                    pass
                        except Exception as e:
                            logger.debug("Heartbeat error: %s", e)
                            try:
                                heartbeat_req.close(linger=0)
                                heartbeat_req = ctx.socket(zmq.REQ)
                                heartbeat_req.connect(f"tcp://{self.host}:{self.event_port}")
                            except Exception:
                                pass
                        last_heartbeat_time = now

                # Poll sockets
                socks = dict(poller.poll(20))

                if data_sub in socks and socks[data_sub] == zmq.POLLIN:
                    self._handle_data_socket(data_sub)

                if event_sub is not None and event_sub in socks and socks[event_sub] == zmq.POLLIN:
                    self._handle_event_socket(event_sub)

                # Check pending sweeps
                self._check_pending_sweeps()

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
                        }
                    )
                    last_stats_time = now

        except Exception as e:
            logger.exception("Error in stream worker: %s", e)
            self.connectionStatusChanged.emit(False, str(e))
        finally:
            if data_sub is not None:
                data_sub.close(0)
            if event_sub is not None:
                event_sub.close(0)
            if heartbeat_req is not None:
                heartbeat_req.close(0)
            context.term()
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
                    for ch_idx in range(wf_mat.shape[0]):
                        aux_sweeps[f"CH{ch_idx}"] = wf_mat[ch_idx]

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
                    waveform = self.bandpass_filter.process(waveform)

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
                    "sample_rate": sample_rate,
                }
                self._sweeps_count += 1
                self.sweepReady.emit(payload)
        except Exception as e:
            logger.error("Failed to parse sliced ERP packet: %s", e)

    def _parse_raw_zmq_data_packet(self, parts: List[bytes]):
        """Parse continuous multi-channel raw stream from Open Ephys ZMQ Interface."""
        try:
            # Open Ephys ZMQ Interface protocol:
            # Frame 0: JSON header {
            #   "stream": "continuous", "channel_count": N, "sample_rate": 30000.0,
            #   "sample_number": S, "num_samples": M, "type": "continuous"
            # }
            # Frame 1: Raw binary buffer (float32 or int16 interleaved samples)
            if len(parts) >= 2 and parts[0].startswith(b"{"):
                meta = json.loads(parts[0].decode("utf-8"))
                buf = parts[1]
                n_chans = int(meta.get("channel_count", meta.get("channels", 1)))
                n_samples = int(meta.get("num_samples", meta.get("sample_count", len(buf) // (4 * n_chans))))
                start_sample = int(meta.get("sample_number", meta.get("start_sample", 0)))
                sr = float(meta.get("sample_rate", self.sample_rate))
                self.sample_rate = sr

                dtype_str = meta.get("dtype", "float32")
                dt = np.float32 if "float" in dtype_str else np.int16
                arr = np.frombuffer(buf, dtype=dt)

                if arr.size == n_chans * n_samples:
                    data = arr.reshape((n_chans, n_samples))
                elif arr.size == n_samples * n_chans:
                    data = arr.reshape((n_samples, n_chans)).T
                else:
                    return

                # Convert int16 to microvolts if needed
                if dt == np.int16 and "bit_volts" in meta:
                    bv = float(meta["bit_volts"])
                    data = data.astype(np.float32) * bv

                # Append to circular ring buffer (RingBuffer handles channel expansion thread-safely)
                self.ring_buffer.append(data, start_sample)

                # Check analog TTL trigger if configured
                if self.trigger_source == "analog" and self.analog_trigger_channel < data.shape[0]:
                    adc_data = data[self.analog_trigger_channel, :]
                    triggers = self.analog_ttl_detector.process_chunk(adc_data, start_sample)
                    for t_sample in triggers:
                        self.queue_sweep_trigger(t_sample, t_sample / self.sample_rate)

            elif len(parts) == 1 and parts[0].startswith(b"{"):
                # Single json event frame
                self._handle_event_json(parts[0])
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
                if part.startswith(b"{") and part.endswith(b"}"):
                    self._handle_event_json(part)

    def _handle_event_json(self, raw_json: bytes):
        """Process event JSON (TTL events, Spikes, APTrack protocol events)."""
        try:
            evt = json.loads(raw_json.decode("utf-8"))
        except Exception:
            return

        evt_type = evt.get("type", "").lower()

        # 1. TTL Trigger Event (Digital Line)
        if evt_type == "ttl" or "line" in evt:
            state = evt.get("state", 1)
            line = evt.get("line", 1)
            sample_num = evt.get("sample_number", evt.get("sample_num", int(time.time() * self.sample_rate)))
            timestamp_s = evt.get("timestamp", sample_num / self.sample_rate)

            # Check if rising edge
            if state == 1 and self.trigger_source == "ttl":
                self.queue_sweep_trigger(sample_num, timestamp_s)

        # 2. Spike Event
        elif evt_type == "spike" or "electrode" in evt:
            electrode = evt.get("electrode", 0)
            unit_name = evt.get("unit_name", f"unit_ch{electrode}")
            sample_num = evt.get("sample_number", evt.get("sample_num", int(time.time() * self.sample_rate)))
            spike_ts = float(evt.get("timestamp", sample_num / self.sample_rate))
            self._spikes_count += 1
            self.spikeEventsReady.emit(unit_name, np.array([spike_ts]))

            # If there is a pending sweep covering sample_num, record spike latency
            with self._pending_lock:
                for sweep in self._pending_sweeps:
                    start_s = sweep.stim_sample - sweep.pre_samples
                    end_s = sweep.stim_sample + sweep.window_samples
                    if start_s <= sample_num <= end_s:
                        if unit_name not in sweep.spike_latencies:
                            sweep.spike_latencies[unit_name] = (sample_num - sweep.stim_sample) / self.sample_rate

        # 3. Protocol / MessageCenter text event (e.g. APTrack stimulus protocol, setStimVoltage)
        elif evt_type in ("message", "text", "protocol") or "text" in evt:
            text = evt.get("text", "")
            ts = float(evt.get("timestamp", time.time()))
            self.protocolEventReady.emit(evt_type, ts, evt)
            # Check for embedded spike/voltage messages:
            # "spikeSampleNumber: 12345"
            if "spikeSampleNumber" in text:
                try:
                    parts = text.split(":")
                    samp = int(parts[1].strip())
                    self.spikeEventsReady.emit("APTrack_unit", np.array([samp / self.sample_rate]))
                except Exception:
                    pass

    def queue_sweep_trigger(self, stim_sample: int, stim_timestamp_s: float):
        """Queue a pending sweep to be sliced when future samples have arrived in the ring buffer."""
        # Calculate samples needed
        window_samples = int((self.window_ms / 1000.0) * self.sample_rate)
        pre_samples = int((self.pre_stim_ms / 1000.0) * self.sample_rate)

        # Record stim interval
        if self._last_stim_timestamp is not None:
            dt = stim_timestamp_s - self._last_stim_timestamp
            if 0.001 < dt < 30.0:
                self._stim_intervals.append(dt)
        self._last_stim_timestamp = stim_timestamp_s

        pending = PendingSweep(
            stim_sample=stim_sample,
            stim_timestamp_s=stim_timestamp_s,
            window_samples=window_samples,
            pre_samples=pre_samples,
            rec_chan=self.rec_channel,
            aux_chans=list(self.active_channels),
        )

        with self._pending_lock:
            # Bound pending queue to prevent unbounded memory growth in extreme stalls
            if len(self._pending_sweeps) > 2000:
                self._pending_sweeps.pop(0)
            self._pending_sweeps.append(pending)

    def _check_pending_sweeps(self):
        """Check if pending sweeps can be sliced from the ring buffer."""
        with self._pending_lock:
            if not self._pending_sweeps:
                return
            sweeps_snapshot = list(self._pending_sweeps)

        ready_items = []
        remaining = []

        for sweep in sweeps_snapshot:
            start_samp = sweep.stim_sample - sweep.pre_samples
            end_samp = sweep.stim_sample + sweep.window_samples

            sliced_all = self.ring_buffer.slice(start_samp, end_samp, channel=None)
            if sliced_all is not None:
                ready_items.append((sweep, sliced_all))
            else:
                # Check if sweep has expired (overwritten or before first_sample_index)
                if self.ring_buffer.total_samples_written == 0 or self.ring_buffer.first_sample_index is None:
                    remaining.append(sweep)
                    continue
                latest_samp = int(
                    self.ring_buffer.timestamps[(self.ring_buffer.write_idx - 1) % self.ring_buffer.capacity]
                )
                earliest_samp = (
                    latest_samp - self.ring_buffer.capacity + 1
                    if self.ring_buffer.total_samples_written >= self.ring_buffer.capacity
                    else self.ring_buffer.first_sample_index
                )
                if start_samp >= earliest_samp:
                    remaining.append(sweep)
                else:
                    logger.warning("Sweep at sample %d dropped (before earliest_samp %d)", sweep.stim_sample, earliest_samp)

        # Update pending list inside lock
        with self._pending_lock:
            ready_set = set(id(s) for s, _ in ready_items)
            self._pending_sweeps = [
                s for s in self._pending_sweeps
                if id(s) not in ready_set and (s in remaining or s not in sweeps_snapshot)
            ]

        # Process and emit ready sweeps outside the lock
        for sweep, sliced_all in ready_items:
            target_chans = sweep.aux_chans if sweep.aux_chans else list(self.active_channels)
            if sweep.rec_chan not in target_chans:
                target_chans = sorted(list(set(target_chans + [sweep.rec_chan])))

            valid_chans = [ch for ch in target_chans if 0 <= ch < sliced_all.shape[0]]
            if valid_chans:
                active_matrix = sliced_all[valid_chans, :].copy()
            else:
                valid_chans = [min(sweep.rec_chan, sliced_all.shape[0] - 1)]
                active_matrix = sliced_all[valid_chans, :].copy()

            if self.apply_filter and hasattr(self, "bandpass_filter") and hasattr(self.bandpass_filter, "sos"):
                try:
                    active_matrix = signal.sosfiltfilt(self.bandpass_filter.sos, active_matrix, axis=-1).astype(np.float32)
                except Exception:
                    pass

            if sweep.rec_chan in valid_chans:
                rec_idx = valid_chans.index(sweep.rec_chan)
                primary_waveform = active_matrix[rec_idx]
            else:
                primary_waveform = active_matrix[0]

            aux_sweeps = {
                f"CH{ch}": ch_waveform
                for ch, ch_waveform in zip(valid_chans, active_matrix)
            }

            payload = {
                "stim_timestamp_s": sweep.stim_timestamp_s,
                "waveform": primary_waveform,
                "pre_offset_s": sweep.pre_samples / self.sample_rate,
                "spike_latencies_s": sweep.spike_latencies,
                "aux_sweeps": aux_sweeps,
                "channel_waveforms": aux_sweeps,
                "sample_rate": self.sample_rate,
            }
            self._sweeps_count += 1
            self.sweepReady.emit(payload)
