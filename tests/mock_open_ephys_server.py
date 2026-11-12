"""Mock Open Ephys v1.0+ ZMQ Publisher and HTTP Control Server for SpikeSpy.

Generates realistic synthetic microneurography continuous streaming, digital TTL triggers,
upstream spike events, and serves the Open Ephys GUI v1.0+ HTTP REST API.
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

        elif parsed_path == "/api/message":
            text = body.get("text", "")
            self.server.received_messages.append({"timestamp": time.time(), "text": text})
            logger.info("Mock Open Ephys received message: %s", text)
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

    def __init__(self, server_address, RequestHandlerClass):
        super().__init__(server_address, RequestHandlerClass)
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
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        data_port: int = 5556,
        event_port: int = 5557,
        http_port: int = 37497,
        n_channels: int = 4,
        sample_rate: float = 30000.0,
        stim_freq_hz: float = 2.0,
        adc_channel: Optional[int] = None,
    ):
        self.host = host
        self.data_port = data_port
        self.event_port = event_port
        self.http_port = http_port
        self.n_channels = n_channels
        self.sample_rate = sample_rate
        self.stim_freq_hz = stim_freq_hz
        self.adc_channel = adc_channel

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._httpd: Optional[MockOpenEphysServerHTTPServer] = None
        self._http_thread: Optional[threading.Thread] = None

        # Simulation state
        self._sample_counter = 0
        self._stim_count = 0
        self._base_latency_s = 0.025  # 25 ms base latency for microneurography C-fibre

    def start(self):
        """Start both HTTP and ZMQ simulation servers."""
        if self._running:
            return

        self._running = True

        # 1. Start HTTP Server
        self._httpd = MockOpenEphysServerHTTPServer((self.host, self.http_port), MockOpenEphysHTTPHandler)
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

    def _generate_action_potential(self, sample_rate: float) -> Tuple[np.ndarray, int]:
        """Generate a realistic asymmetric triphasic extracellular C-fibre action potential.

        Returns (waveform, peak_index) where peak_index is the sample index of the primary positive peak.
        """
        n_pts = int(0.002 * sample_rate)  # 2.0 ms duration (60 samples at 30 kHz)
        peak_idx = int(0.4 * n_pts)       # sample 24
        t_ms = (np.arange(n_pts) - peak_idx) / (sample_rate / 1000.0)

        # Realistic biophysical extracellular waveform components (uV):
        # 1. Sharp depolarizing positive peak (aligned with SpikeSpy tracking and heatmap)
        p_main = 100.0 * np.exp(-((t_ms) ** 2) / 0.08)
        # 2. Initial negative capacitive approach dip
        p_pre = -22.0 * np.exp(-((t_ms + 0.35) ** 2) / 0.05)
        # 3. Repolarizing undershoot / after-hyperpolarization
        p_post = -32.0 * np.exp(-((t_ms - 0.45) ** 2) / 0.18)
        ap = (p_main + p_pre + p_post).astype(np.float32)

        # Smooth cosine taper on the extreme edges
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

        samples_between_stims = int(self.sample_rate / self.stim_freq_hz)
        next_stim_sample = samples_between_stims

        spike_wf, peak_offset = self._generate_action_potential(self.sample_rate)
        stim_art_wf = self._generate_stim_artifact(self.sample_rate)

        # Overlay accumulator buffer to prevent edge truncation across chunk boundaries
        overlay_len = max(chunk_size * 6, int(self.sample_rate * 0.5))
        overlay_buf = np.zeros((self.n_channels, overlay_len), dtype=np.float32)

        # Multi-channel propagation delay and distance attenuation profiles:
        # Nearby channels record the same event offset by small conduction delay and amplitude decay
        # (e.g. conduction velocity ~1 m/s, ~133 um electrode spacing = ~0.133 ms / ~4 samples delay per contact)
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

                    # Activity-dependent slowing (ADS):
                    # Repetitive C-fibre stimulation causes progressive latency slowing
                    # with rapid early phase, asymptotic plateau, plus organic micro-jitter
                    ads_drift = 0.016 * (1.0 - math.exp(-n / 25.0)) + (n * 0.00004)
                    jitter = 0.00003 * math.sin(n * 2.3)
                    evoked_latency_s = self._base_latency_s + ads_drift + jitter
                    peak_sample = stim_sample + int(round(evoked_latency_s * self.sample_rate))

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

                    # 2c. Inject Evoked Action Potential across all channels
                    # Primary channel (CH0) peak is at peak_sample (start is peak_sample - peak_offset)
                    # Nearby channels have propagation delay chan_delays[c] and amplitude decay chan_scales[c]
                    spk_len = len(spike_wf)
                    for c in range(self.n_channels):
                        ch_peak_sample = peak_sample + chan_delays[c]
                        ch_start_sample = ch_peak_sample - peak_offset
                        ch_offset_in_buf = ch_start_sample - cur_sample
                        if 0 <= ch_offset_in_buf < overlay_len - spk_len:
                            overlay_buf[c, ch_offset_in_buf : ch_offset_in_buf + spk_len] += (
                                spike_wf * chan_scales[c]
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

                    # 2e. Publish upstream Spike Event (aligned with primary peak on CH0)
                    spk_event = {
                        "type": "spike",
                        "electrode": 0,
                        "unit_name": "unit_0",
                        "sample_number": peak_sample,
                        "timestamp": peak_sample / self.sample_rate,
                    }
                    event_pub.send_string(json.dumps(spk_event))

                    # Schedule next stimulus
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
    server = MockOpenEphysServer()
    server.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop()
        print("Server stopped.")
