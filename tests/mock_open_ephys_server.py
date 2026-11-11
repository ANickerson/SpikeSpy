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
from typing import Any, Dict, List, Optional

import numpy as np
import zmq

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
    ):
        self.host = host
        self.data_port = data_port
        self.event_port = event_port
        self.http_port = http_port
        self.n_channels = n_channels
        self.sample_rate = sample_rate
        self.stim_freq_hz = stim_freq_hz

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._httpd: Optional[MockOpenEphysServerHTTPServer] = None
        self._http_thread: Optional[threading.Thread] = None

        # Simulation state
        self._sample_counter = 0
        self._stim_count = 0
        self._base_latency_s = 0.025  # 25 ms base latency for microneurography C-fibre
        self._drift_s = 0.0

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

    def _generate_action_potential(self, n_pts: int = 60) -> np.ndarray:
        """Generate a realistic triphasic extracellular action potential waveform (~2 ms)."""
        t = np.linspace(-1.0, 1.0, n_pts)
        # Ricker/Hermite-like triphasic spike
        ap = -np.exp(-3 * t**2) * (1 - 4 * t**2) * 80.0  # ~80 uV amplitude
        return ap.astype(np.float32)

    def _run_zmq(self):
        """ZMQ publishing loop."""
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

        spike_wf = self._generate_action_potential(int(0.002 * self.sample_rate))  # 2ms spike
        spike_len = len(spike_wf)

        try:
            while self._running:
                t0 = time.time()
                cur_sample = self._sample_counter
                self._sample_counter += chunk_size

                # Continuous background recording noise (band-limited white noise)
                raw_chunk = np.random.normal(0.0, 8.0, (self.n_channels, chunk_size)).astype(np.float32)

                # Check if stimulus falls within this chunk
                if next_stim_sample >= cur_sample and next_stim_sample < cur_sample + chunk_size:
                    stim_offset_in_chunk = next_stim_sample - cur_sample
                    stim_sample = next_stim_sample
                    stim_timestamp = stim_sample / self.sample_rate
                    self._stim_count += 1

                    # 1. Analog TTL pulse on channel 1 (simulating DS4 / ADC input)
                    if self.n_channels > 1:
                        pulse_len = min(60, chunk_size - stim_offset_in_chunk)
                        raw_chunk[1, stim_offset_in_chunk : stim_offset_in_chunk + pulse_len] = 5.0  # 5V pulse

                    # 2. Activity-dependent slowing: progressive latency drift (microneurography)
                    self._drift_s += 0.0003  # +0.3 ms latency drift per stimulus
                    evoked_latency_s = self._base_latency_s + self._drift_s
                    evoked_sample = stim_sample + int(evoked_latency_s * self.sample_rate)

                    # 3. Publish Digital TTL Event via Event socket (Event Broadcaster JSON format)
                    ttl_event = {
                        "type": "ttl",
                        "state": 1,
                        "line": 1,
                        "sample_number": stim_sample,
                        "timestamp": stim_timestamp,
                    }
                    event_pub.send_string(json.dumps(ttl_event))

                    # 4. Schedule next stimulus
                    next_stim_sample += samples_between_stims

                # Check if an evoked spike falls in this chunk
                # Re-calculate spike location from last stim
                if self._stim_count > 0:
                    last_stim_sample = next_stim_sample - samples_between_stims
                    evoked_sample = last_stim_sample + int((self._base_latency_s + self._drift_s) * self.sample_rate)
                    if evoked_sample >= cur_sample and evoked_sample < cur_sample + chunk_size:
                        spk_offset = evoked_sample - cur_sample
                        valid_len = min(spike_len, chunk_size - spk_offset)
                        # Add evoked spike to recording channel 0
                        raw_chunk[0, spk_offset : spk_offset + valid_len] += spike_wf[:valid_len]

                        # Also emit upstream Spike event on event socket
                        spk_event = {
                            "type": "spike",
                            "electrode": 0,
                            "unit_name": "unit_0",
                            "sample_number": evoked_sample,
                            "timestamp": evoked_sample / self.sample_rate,
                        }
                        event_pub.send_string(json.dumps(spk_event))

                # Publish Open Ephys continuous data stream frame:
                # Frame 0: JSON header
                # Frame 1: Binary data buffer (interleaved float32 or planar)
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
