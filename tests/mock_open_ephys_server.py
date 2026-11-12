"""Mock Open Ephys v1.0+ ZMQ Publisher and HTTP Control Server for SpikeSpy.

Generates realistic synthetic microneurography continuous streaming, digital TTL triggers,
upstream spike events, and serves the Open Ephys GUI v1.0+ HTTP REST API.
Supports dynamic stimulation frequency control and multiple simulated units
with distinct neurophysiological characteristics.
"""

from __future__ import annotations

import json
import logging
import math
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import zmq
from scipy import signal

logger = logging.getLogger(__name__)


# Default realistic simulated unit neurophysiological characteristics
DEFAULT_UNIT_CONFIGS: List[Dict[str, Any]] = [
    {
        "unit_id": 0,
        "name": "unit_0",
        "fibre_type": "Fast A-fiber",
        "latency_ms": 12.0,
        "jitter_ms": 0.5,
        "amplitude_uv": 65.0,
        "duration_ms": 0.8,
        "channels": [0, 1],
        "channel_scales": [1.0, 0.75],
        "electrode": 0,
    },
    {
        "unit_id": 1,
        "name": "unit_1",
        "fibre_type": "Medium A-delta fiber",
        "latency_ms": 32.0,
        "jitter_ms": 2.0,
        "amplitude_uv": 45.0,
        "duration_ms": 1.4,
        "channels": [2, 3],
        "channel_scales": [1.0, 0.75],
        "electrode": 2,
    },
    {
        "unit_id": 2,
        "name": "unit_2",
        "fibre_type": "Slow C-fiber",
        "latency_ms": 80.0,
        "jitter_ms": 6.0,
        "amplitude_uv": 25.0,
        "duration_ms": 2.2,
        "channels": [4, 5],
        "channel_scales": [1.0, 0.75],
        "electrode": 4,
    },
]


class MockOpenEphysHTTPHandler(BaseHTTPRequestHandler):
    """Handles Open Ephys GUI v1.0+ REST API requests."""

    server: "MockOpenEphysServerHTTPServer"  # type hint for parent server

    def log_message(self, format, *args):
        # Suppress noisy HTTP request logging in test outputs
        pass

    def _send_json(self, status_code: int, data: Any):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed_path = self.path.split("?")[0]
        if parsed_path == "/api/status":
            self._send_json(200, {"mode": self.server.oe_status})
        elif parsed_path == "/api/processors":
            self._send_json(200, self.server.processors)
        elif parsed_path == "/api/recording":
            self._send_json(200, {"parent_directory": "/tmp/oe_recordings", "recording_number": 1})
        elif parsed_path == "/api/stim_freq":
            freq = self.server.parent_server.stim_freq_hz if self.server.parent_server else 5.0
            self._send_json(200, {"stim_freq_hz": freq})
        else:
            self._send_json(404, {"error": "Not found"})

    def do_PUT(self):
        content_length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else ""
        try:
            body = json.loads(raw_body) if raw_body else {}
        except Exception:
            body = {"text": raw_body}

        parsed_path = self.path.split("?")[0]

        if parsed_path == "/api/status":
            new_mode = body.get("mode", "IDLE").upper()
            self.server.oe_status = new_mode
            self._send_json(200, {"mode": self.server.oe_status})

        elif parsed_path == "/api/stim_freq":
            freq = float(body.get("stim_freq_hz", body.get("freq", 5.0)))
            if self.server.parent_server:
                self.server.parent_server.set_stim_freq(freq)
            self._send_json(200, {"status": "ok", "stim_freq_hz": freq})

        elif parsed_path == "/api/message":
            text = body.get("text", "")
            self.server.received_messages.append({"timestamp": time.time(), "text": text})
            logger.info("Mock Open Ephys received message: %s", text)
            # Check for stimulation frequency update commands in message
            if "setStimFreq" in text or "setStimFrequency" in text:
                try:
                    parts = text.split(":")
                    if len(parts) >= 2:
                        freq_val = float(parts[1].strip().split()[0])
                        if self.server.parent_server:
                            self.server.parent_server.set_stim_freq(freq_val)
                except Exception as ex:
                    logger.warning("Could not parse frequency from message: %s", ex)
            self._send_json(200, {"status": "Message sent"})

        elif parsed_path.startswith("/api/processors/") and parsed_path.endswith("/config"):
            parts = parsed_path.strip("/").split("/")
            try:
                proc_id = int(parts[2])
            except (ValueError, IndexError):
                proc_id = -1
            config_text = body.get("text", "")
            self.server.received_processor_configs.append(
                {"timestamp": time.time(), "processor_id": proc_id, "config": config_text}
            )
            logger.info("Mock Open Ephys processor %d received config: %s", proc_id, config_text)
            self._send_json(200, {"status": "ok", "processor_id": proc_id, "applied": config_text})

        else:
            self._send_json(404, {"error": "Not found"})


class MockOpenEphysServerHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True

    def __init__(self, server_address, RequestHandlerClass, parent_server: Optional["MockOpenEphysServer"] = None):
        super().__init__(server_address, RequestHandlerClass)
        self.parent_server = parent_server
        self.oe_status = "ACQUIRE"
        self.received_messages: List[Dict[str, Any]] = []
        self.received_processor_configs: List[Dict[str, Any]] = []
        self.processors: List[Dict[str, Any]] = [
            {"id": 100, "name": "Rhythm FPGA", "type": "SOURCE"},
            {"id": 101, "name": "Bandpass Filter", "type": "FILTER"},
            {"id": 102, "name": "Spike Detector", "type": "SINK"},
            {"id": 103, "name": "SpikeSpyStreamer", "type": "SINK"},
            {"id": 104, "name": "Event Broadcaster", "type": "SINK"},
        ]


class MockOpenEphysServer:
    """Mock Open Ephys Server.

    Runs an HTTP REST server and a ZMQ PUB thread generating realistic continuous
    data (30 kHz continuous stream with microneurography noise and evoked action potentials)
    and TTL pulse events.

    Supports dynamic stimulation frequency control and multiple simulated units
    with distinct neurophysiological characteristics.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        data_port: int = 5556,
        event_port: int = 5557,
        http_port: int = 37497,
        n_channels: int = 4,
        sample_rate: float = 30000.0,
        stim_freq_hz: float = 5.0,
        adc_channel: Optional[int] = None,
        num_units: Optional[int] = None,
        unit_configs: Optional[List[Dict[str, Any]]] = None,
    ):
        self.host = host
        self.data_port = data_port
        self.event_port = event_port
        self.http_port = http_port
        self.sample_rate = sample_rate
        self.adc_channel = adc_channel

        # Stimulation frequency (0.5 to 50 Hz, default 5.0 Hz)
        self._stim_freq_hz = max(0.1, min(100.0, float(stim_freq_hz)))

        # Unit configuration
        self._is_legacy_mode = (num_units is None and unit_configs is None)
        self._units: List[Dict[str, Any]] = []

        if not self._is_legacy_mode:
            if unit_configs is not None:
                self._units = [dict(u) for u in unit_configs]
            else:
                n_u = max(1, min(len(DEFAULT_UNIT_CONFIGS), int(num_units or 1)))
                self._units = [dict(u) for u in DEFAULT_UNIT_CONFIGS[:n_u]]

            # Ensure channel count accommodates all configured units
            max_ch = max(max(u.get("channels", [0])) for u in self._units)
            self.n_channels = max(n_channels, max_ch + 1)
        else:
            self.n_channels = n_channels

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._httpd: Optional[MockOpenEphysServerHTTPServer] = None
        self._http_thread: Optional[threading.Thread] = None

        # Simulation state
        self._sample_counter = 0
        self._stim_count = 0
        self._base_latency_s = 0.025  # 25 ms base latency for legacy C-fibre mode

        # Precompute unit waveforms
        self._precompute_waveforms()

    @property
    def stim_freq_hz(self) -> float:
        return self._stim_freq_hz

    @stim_freq_hz.setter
    def stim_freq_hz(self, val: float):
        self.set_stim_freq(val)

    def set_stim_freq(self, freq_hz: float):
        """Dynamically update stimulation frequency while running."""
        self._stim_freq_hz = max(0.1, min(100.0, float(freq_hz)))

    def set_stim_frequency(self, freq_hz: float):
        self.set_stim_freq(freq_hz)

    def get_stim_freq(self) -> float:
        return self._stim_freq_hz

    @property
    def inter_sweep_interval_s(self) -> float:
        return 1.0 / self._stim_freq_hz

    @property
    def units(self) -> List[Dict[str, Any]]:
        return self._units

    def _precompute_waveforms(self):
        """Precompute synthetic action potential waveforms for configured units."""
        for u in self._units:
            duration_ms = float(u.get("duration_ms", 1.2))
            amp_uv = float(u.get("amplitude_uv", 50.0))
            wf, peak_offset = self._generate_unit_action_potential(
                duration_ms=duration_ms,
                amplitude_uv=amp_uv,
                sample_rate=self.sample_rate,
            )
            u["waveform"] = wf
            u["peak_offset"] = peak_offset

    def start(self):
        """Start both HTTP and ZMQ simulation servers."""
        if self._running:
            return

        self._running = True

        # 1. Start HTTP Server
        self._httpd = MockOpenEphysServerHTTPServer(
            (self.host, self.http_port), MockOpenEphysHTTPHandler, parent_server=self
        )
        self._http_thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._http_thread.start()

        # 2. Start ZMQ Publisher Thread
        self._thread = threading.Thread(target=self._run_zmq, daemon=True)
        self._thread.start()

    def is_running(self) -> bool:
        return self._running

    def stop(self):
        """Stop servers and cleanup."""
        self._running = False
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None

        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    @property
    def received_messages(self) -> List[Dict[str, Any]]:
        return self._httpd.received_messages if self._httpd else []

    @property
    def received_processor_configs(self) -> List[Dict[str, Any]]:
        return self._httpd.received_processor_configs if self._httpd else []

    def _generate_unit_action_potential(
        self,
        duration_ms: float,
        amplitude_uv: float,
        sample_rate: float,
    ) -> Tuple[np.ndarray, int]:
        """Generate an extracellular action potential tailored to fiber characteristics.

        Returns (waveform, peak_index) where peak_index is the sample index of the primary positive peak.
        """
        n_pts = max(12, int((duration_ms / 1000.0) * sample_rate))
        peak_idx = int(0.38 * n_pts)
        t_ms = (np.arange(n_pts) - peak_idx) / (sample_rate / 1000.0)

        w_scale = duration_ms / 1.5
        p_main = amplitude_uv * np.exp(-((t_ms) ** 2) / (0.05 * w_scale))
        p_pre = -(0.22 * amplitude_uv) * np.exp(-((t_ms + 0.22 * w_scale) ** 2) / (0.03 * w_scale))
        p_post = -(0.32 * amplitude_uv) * np.exp(-((t_ms - 0.32 * w_scale) ** 2) / (0.12 * w_scale))
        ap = (p_main + p_pre + p_post).astype(np.float32)

        # Smooth cosine taper on edges
        n_taper = max(2, min(8, n_pts // 5))
        taper = np.ones(n_pts, dtype=np.float32)
        taper[:n_taper] = np.sin(np.linspace(0, np.pi / 2, n_taper)) ** 2
        taper[-n_taper:] = np.cos(np.linspace(0, np.pi / 2, n_taper)) ** 2
        ap *= taper

        actual_peak = int(np.argmax(ap))
        return ap, actual_peak

    def _generate_action_potential(self, sample_rate: float) -> Tuple[np.ndarray, int]:
        """Generate a realistic asymmetric triphasic extracellular C-fibre action potential (legacy mode)."""
        n_pts = int(0.002 * sample_rate)  # 2.0 ms duration (60 samples at 30 kHz)
        peak_idx = int(0.4 * n_pts)       # sample 24
        t_ms = (np.arange(n_pts) - peak_idx) / (sample_rate / 1000.0)

        p_main = 100.0 * np.exp(-((t_ms) ** 2) / 0.08)
        p_pre = -22.0 * np.exp(-((t_ms + 0.35) ** 2) / 0.05)
        p_post = -32.0 * np.exp(-((t_ms - 0.45) ** 2) / 0.18)
        ap = (p_main + p_pre + p_post).astype(np.float32)

        n_taper_start = min(6, n_pts // 5)
        n_taper_end = min(10, n_pts // 3)
        taper = np.ones(n_pts, dtype=np.float32)
        taper[:n_taper_start] = np.sin(np.linspace(0, np.pi / 2, n_taper_start)) ** 2
        taper[-n_taper_end:] = np.cos(np.linspace(0, np.pi / 2, n_taper_end)) ** 2
        ap *= taper

        actual_peak = int(np.argmax(ap))
        return ap, actual_peak

    def _generate_stim_artifact(self, sample_rate: float) -> np.ndarray:
        """Generate a realistic electrical stimulus artifact (sharp transient + capacitive decay)."""
        n_art = int(0.0012 * sample_rate)  # 1.2 ms duration (36 samples at 30 kHz)
        t_art = np.arange(n_art) / (sample_rate / 1000.0)  # ms
        art = (110.0 * np.exp(-t_art / 0.10) - 40.0 * np.exp(-t_art / 0.35)).astype(np.float32)
        n_taper = min(8, n_art // 4)
        taper = np.ones(n_art, dtype=np.float32)
        taper[-n_taper:] = np.cos(np.linspace(0, np.pi / 2, n_taper)) ** 2
        art *= taper
        return art

    def _run_zmq(self):
        """ZMQ publishing loop for continuous microneurography streaming."""
        context = zmq.Context()
        data_pub = context.socket(zmq.PUB)
        data_pub.setsockopt(zmq.LINGER, 0)
        data_pub.bind(f"tcp://{self.host}:{self.data_port}")

        event_pub = context.socket(zmq.PUB)
        event_pub.setsockopt(zmq.LINGER, 0)
        event_pub.bind(f"tcp://{self.host}:{self.event_port}")

        chunk_size = 600  # 20 ms at 30 kHz (50 packets/sec)
        chunk_dt = chunk_size / self.sample_rate

        last_stim_freq = self._stim_freq_hz
        samples_between_stims = max(1, int(self.sample_rate / self._stim_freq_hz))
        next_stim_sample = samples_between_stims

        legacy_spike_wf, legacy_peak_offset = self._generate_action_potential(self.sample_rate)
        stim_art_wf = self._generate_stim_artifact(self.sample_rate)

        # Overlay accumulator buffer to prevent edge truncation across chunk boundaries (1 second buffer)
        overlay_len = max(chunk_size * 10, int(self.sample_rate * 1.0))
        overlay_buf = np.zeros((self.n_channels, overlay_len), dtype=np.float32)

        # Legacy channel scaling factors
        chan_delays = [int(round(c * (0.000133 * self.sample_rate))) for c in range(self.n_channels)]
        chan_scales = [float(0.65 ** c) for c in range(self.n_channels)]
        art_scales = [float(0.92 ** c) for c in range(self.n_channels)]

        # Continuous colored noise filter states (IIR AR(1) state per channel)
        noise_coeffs = []
        noise_zi = []
        for c in range(self.n_channels):
            if c == 0:
                # Primary channel: clean electrode thermal / 1/f noise (RMS ~7.5 uV)
                a = 0.88
                sigma = 7.0
            elif c == 1:
                # Neighbor 1: higher impedance with slow biological baseline drift (RMS ~10.5 uV)
                a = 0.94
                sigma = 9.0
            elif c == 2:
                # Neighbor 2: clean low-noise contact (RMS ~5.0 uV) with subtle 50 Hz hum
                a = 0.78
                sigma = 4.2
            elif c == 3:
                # Neighbor 3: broadband hash contact (RMS ~8.5 uV)
                a = 0.65
                sigma = 8.5
            else:
                a = 0.85
                sigma = 7.0
            b = [sigma * np.sqrt(1.0 - a ** 2)]
            zi = signal.lfilter_zi(b, [1.0, -a]) * 0.0
            noise_coeffs.append((b, [1.0, -a]))
            noise_zi.append(zi)

        try:
            while self._running:
                t0 = time.time()
                cur_sample = self._sample_counter
                self._sample_counter += chunk_size

                # Dynamic frequency adjustment: update interval and next stim if frequency changed
                if self._stim_freq_hz != last_stim_freq:
                    last_stim_freq = self._stim_freq_hz
                    samples_between_stims = max(1, int(self.sample_rate / last_stim_freq))
                    if next_stim_sample - cur_sample > samples_between_stims:
                        next_stim_sample = cur_sample + samples_between_stims

                # 1. Generate distinct background noise per channel
                raw_chunk = np.zeros((self.n_channels, chunk_size), dtype=np.float32)
                t_chunk = (cur_sample + np.arange(chunk_size)) / self.sample_rate

                for c in range(self.n_channels):
                    b, a = noise_coeffs[c]
                    w = np.random.normal(0.0, 1.0, chunk_size).astype(np.float64)
                    filtered_noise, noise_zi[c] = signal.lfilter(b, a, w, zi=noise_zi[c])
                    ch_noise = filtered_noise.astype(np.float32)

                    # Distinct noise profile characteristics:
                    if c == 1:
                        # Baseline slow drift (~0.7 Hz, ~8 uV)
                        ch_noise += (8.0 * np.sin(2 * np.pi * 0.7 * t_chunk)).astype(np.float32)
                    elif c == 2:
                        # Subtle 50 Hz mains hum harmonic (~3.0 uV)
                        ch_noise += (3.0 * np.sin(2 * np.pi * 50.0 * t_chunk)).astype(np.float32)
                    elif c == 3:
                        # Sparse miniature background asynchronous spikes in hash
                        if np.random.rand() < 0.25:
                            mini_pos = np.random.randint(0, max(1, chunk_size - 10))
                            ch_noise[mini_pos : mini_pos + 6] += np.random.uniform(4.0, 7.0)

                    raw_chunk[c] = ch_noise

                # 2. Check if a stimulus falls within this chunk
                while next_stim_sample < cur_sample + chunk_size:
                    stim_sample = next_stim_sample
                    stim_timestamp = stim_sample / self.sample_rate
                    self._stim_count += 1
                    n = self._stim_count

                    # 2a. Inject Stimulus Artifact across all channels (at stim_sample)
                    stim_offset_in_buf = stim_sample - cur_sample
                    if 0 <= stim_offset_in_buf < overlay_len - len(stim_art_wf):
                        for c in range(self.n_channels):
                            overlay_buf[c, stim_offset_in_buf : stim_offset_in_buf + len(stim_art_wf)] += (
                                stim_art_wf * art_scales[c]
                            )

                    # 2b. If ADC trigger channel is configured, place 5V pulse on it
                    if self.adc_channel is not None and 0 <= self.adc_channel < self.n_channels:
                        pulse_len = min(60, overlay_len - stim_offset_in_buf)
                        if stim_offset_in_buf >= 0 and pulse_len > 0:
                            overlay_buf[self.adc_channel, stim_offset_in_buf : stim_offset_in_buf + pulse_len] = 5.0

                    if not self._is_legacy_mode and self._units:
                        # 2c. Multi-Unit Mode: Inject each simulated unit waveform and emit its spike event
                        unit_spk_events = []
                        for u in self._units:
                            u_id = int(u.get("unit_id", 0))
                            base_lat_ms = float(u.get("latency_ms", 12.0))
                            jitter_ms = float(u.get("jitter_ms", 0.5))

                            # Organic deterministic-like micro-jitter
                            j_norm = math.sin(n * (2.1 + 0.7 * u_id) + u_id * 1.5)
                            jitter_s = (jitter_ms / 1000.0) * j_norm

                            # C-fiber activity-dependent slowing drift
                            ads_drift = 0.0
                            if base_lat_ms >= 70.0:
                                ads_drift = 0.006 * (1.0 - math.exp(-n / 25.0))

                            evoked_lat_s = (base_lat_ms / 1000.0) + ads_drift + jitter_s
                            peak_sample = stim_sample + int(round(evoked_lat_s * self.sample_rate))

                            wf = u["waveform"]
                            p_offset = u["peak_offset"]
                            channels = u.get("channels", [0])
                            scales = u.get("channel_scales", [1.0] * len(channels))

                            for ch_idx, (ch, scale) in enumerate(zip(channels, scales)):
                                if ch >= self.n_channels:
                                    continue
                                ch_delay = int(round(ch_idx * (0.000133 * self.sample_rate)))
                                ch_peak_sample = peak_sample + ch_delay
                                ch_start_sample = ch_peak_sample - p_offset
                                ch_offset_in_buf = ch_start_sample - cur_sample
                                if 0 <= ch_offset_in_buf < overlay_len - len(wf):
                                    overlay_buf[ch, ch_offset_in_buf : ch_offset_in_buf + len(wf)] += wf * scale

                            unit_spk_events.append(
                                {
                                    "type": "spike",
                                    "electrode": u.get("electrode", channels[0]),
                                    "unit_name": u.get("name", f"unit_{u_id}"),
                                    "sample_number": peak_sample,
                                    "timestamp": peak_sample / self.sample_rate,
                                }
                            )

                        # 2d. Publish Digital TTL Event via Event socket
                        ttl_event = {
                            "type": "ttl",
                            "state": 1,
                            "line": 1,
                            "sample_number": stim_sample,
                            "timestamp": stim_timestamp,
                        }
                        event_pub.send_string(json.dumps(ttl_event))

                        # 2e. Publish upstream spike events for each unit
                        for spk_event in unit_spk_events:
                            event_pub.send_string(json.dumps(spk_event))

                    else:
                        # Legacy Single-Unit Mode (C-fiber at 25 ms with slowing)
                        ads_drift = 0.016 * (1.0 - math.exp(-n / 25.0)) + (n * 0.00004)
                        jitter = 0.00003 * math.sin(n * 2.3)
                        evoked_latency_s = self._base_latency_s + ads_drift + jitter
                        peak_sample = stim_sample + int(round(evoked_latency_s * self.sample_rate))

                        spk_len = len(legacy_spike_wf)
                        for c in range(self.n_channels):
                            ch_peak_sample = peak_sample + chan_delays[c]
                            ch_start_sample = ch_peak_sample - legacy_peak_offset
                            ch_offset_in_buf = ch_start_sample - cur_sample
                            if 0 <= ch_offset_in_buf < overlay_len - spk_len:
                                overlay_buf[c, ch_offset_in_buf : ch_offset_in_buf + spk_len] += (
                                    legacy_spike_wf * chan_scales[c]
                                )

                        # 2d. Publish Digital TTL Event via Event socket
                        ttl_event = {
                            "type": "ttl",
                            "state": 1,
                            "line": 1,
                            "sample_number": stim_sample,
                            "timestamp": stim_timestamp,
                        }
                        event_pub.send_string(json.dumps(ttl_event))

                        # 2e. Publish upstream Spike Event
                        spk_event = {
                            "type": "spike",
                            "electrode": 0,
                            "unit_name": "unit_0",
                            "sample_number": peak_sample,
                            "timestamp": peak_sample / self.sample_rate,
                        }
                        event_pub.send_string(json.dumps(spk_event))

                    # Schedule next stimulus based on current stim_freq_hz
                    samples_between_stims = max(1, int(self.sample_rate / self._stim_freq_hz))
                    next_stim_sample += samples_between_stims

                # 3. Add overlay buffer to current chunk and advance overlay buffer
                raw_chunk += overlay_buf[:, :chunk_size]
                overlay_buf = np.roll(overlay_buf, -chunk_size, axis=1)
                overlay_buf[:, -chunk_size:] = 0.0

                # 4. Publish Open Ephys continuous data stream frame
                header = {
                    "stream": "continuous",
                    "channel_count": self.n_channels,
                    "sample_rate": self.sample_rate,
                    "sample_number": cur_sample,
                    "num_samples": chunk_size,
                    "dtype": "float32",
                }
                data_pub.send_multipart([json.dumps(header).encode("utf-8"), raw_chunk.tobytes()])

                # Sleep to maintain real-time rate
                elapsed = time.time() - t0
                sleep_time = chunk_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except Exception as e:
            logger.debug("Mock ZMQ publisher stopped: %s", e)
        finally:
            data_pub.close()
            event_pub.close()
            context.term()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Starting Mock Open Ephys Server on data=5556, event=5557, http=37497...")
    server = MockOpenEphysServer(num_units=3)
    server.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop()
        print("Server stopped.")
