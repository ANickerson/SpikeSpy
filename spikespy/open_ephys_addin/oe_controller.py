"""Open Ephys Controller for SpikeSpy.

Provides bidirectional HTTP communication with the Open Ephys GUI v1.0+ HTTP API
(default: http://127.0.0.1:37497).
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

from PySide6.QtCore import QObject, Signal, Slot

logger = logging.getLogger(__name__)


class OpenEphysController(QObject):
    """Bidirectional Open Ephys v1.0+ HTTP Controller.

    Communicates with the built-in HTTP server in Open Ephys GUI v1.0+
    (default: http://127.0.0.1:37497) using standard library urllib and json.
    """

    statusChanged = Signal(str)
    messageSent = Signal(str, bool)
    configUpdated = Signal(int, str, bool)
    errorOccurred = Signal(str)

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 37497,
        parent: Optional[QObject] = None,
        timeout: float = 2.0,
    ) -> None:
        super().__init__(parent)
        self.host = host
        self.port = port
        self.timeout = timeout
        self.target_processor_id: Optional[int] = None
        self._last_status: str = "DISCONNECTED"

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def _http_request(
        self,
        method: str,
        path: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> tuple[int, Any]:
        """Perform an HTTP request to the Open Ephys server."""
        url = f"{self.base_url}{path}"
        data = None
        headers = {
            "User-Agent": "SpikeSpy-Addin/1.0",
            "Accept": "application/json",
        }
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            if self.host in ("127.0.0.1", "localhost", "::1"):
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                resp_ctx = opener.open(req, timeout=self.timeout)
            else:
                resp_ctx = urllib.request.urlopen(req, timeout=self.timeout)
            with resp_ctx as resp:
                status_code = resp.status
                body = resp.read().decode("utf-8")
                if body:
                    try:
                        return status_code, json.loads(body)
                    except json.JSONDecodeError:
                        return status_code, body
                return status_code, {}
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8") if e.fp else ""
            logger.warning("HTTPError %d for %s %s: %s", e.code, method, url, err_body)
            return e.code, {"error": str(e), "body": err_body}
        except Exception as e:
            logger.debug("Request failed for %s %s: %s", method, url, e)
            return -1, {"error": str(e)}

    def get_status(self) -> str:
        """Get the current acquisition status of Open Ephys GUI.

        Returns:
            "IDLE", "ACQUIRE", "RECORD", or "DISCONNECTED" on error.
        """
        code, data = self._http_request("GET", "/api/status")
        if code == 200 and isinstance(data, dict):
            status = data.get("mode", data.get("status", "IDLE")).upper()
            if status != self._last_status:
                self._last_status = status
                self.statusChanged.emit(status)
            return status
        elif code == 200 and isinstance(data, str):
            status = data.strip().strip('"').upper()
            if status != self._last_status:
                self._last_status = status
                self.statusChanged.emit(status)
            return status
        else:
            if self._last_status != "DISCONNECTED":
                self._last_status = "DISCONNECTED"
                self.statusChanged.emit("DISCONNECTED")
            return "DISCONNECTED"

    def set_status(self, mode: str) -> bool:
        """Set the acquisition mode of Open Ephys GUI.

        Args:
            mode: "ACQUIRE", "RECORD", or "IDLE"

        Returns:
            True if successful, False otherwise.
        """
        valid_modes = {"ACQUIRE", "RECORD", "IDLE"}
        mode_upper = mode.upper()
        if mode_upper not in valid_modes:
            logger.error("Invalid status mode requested: %s", mode)
            return False

        code, resp = self._http_request("PUT", "/api/status", {"mode": mode_upper})
        success = code in (200, 204)
        if success:
            self._last_status = mode_upper
            self.statusChanged.emit(mode_upper)
        else:
            self.errorOccurred.emit(f"Failed to set status to {mode}: {resp}")
        return success

    def get_processors(self) -> List[Dict[str, Any]]:
        """Get the list of active processors in the signal chain."""
        code, data = self._http_request("GET", "/api/processors")
        if code == 200:
            if isinstance(data, list):
                return data
            if isinstance(data, dict) and "processors" in data:
                return data["processors"]
            return [data] if isinstance(data, dict) else []
        self.errorOccurred.emit(f"Failed to get processors: HTTP {code}")
        return []

    def send_message(self, text: str) -> bool:
        """Send a text message broadcast via Open Ephys MessageCenter.

        This broadcasts the message to the signal chain and logs it to
        MessageCenter/text.npy during recording.

        Args:
            text: Message string (or serialized JSON string)

        Returns:
            True if successful, False otherwise.
        """
        code, resp = self._http_request("PUT", "/api/message", {"text": text})
        success = code in (200, 204)
        self.messageSent.emit(text, success)
        if not success:
            self.errorOccurred.emit(f"Failed to send message: HTTP {code}")
        return success

    def send_processor_config(self, processor_id: int, config_text: str) -> Dict[str, Any]:
        """Send a direct configuration update to a specific processor.

        Args:
            processor_id: Open Ephys processor ID (from get_processors)
            config_text: Configuration string or serialized JSON

        Returns:
            Response dict with response status or error.
        """
        code, resp = self._http_request(
            "PUT",
            f"/api/processors/{processor_id}/config",
            {"text": config_text},
        )
        success = code in (200, 204)
        self.configUpdated.emit(processor_id, config_text, success)
        if isinstance(resp, dict):
            return resp
        return {"status": "ok" if success else "error", "response": resp}

    @Slot(dict)
    def sync_unit_annotation(self, annotation_info: Dict[str, Any]) -> bool:
        """Slot to synchronize a unit annotation event to Open Ephys.

        Compatible with APTrack / SpikeSpy event parsing schema.
        Payload format sent to Open Ephys MessageCenter / processor:
        {
            "source": "SpikeSpy",
            "type": "unit_annotation",
            "spikeGroup": unit_id_or_name,
            "stimno": stimno,
            "spikeSampleLatency": latency_samples,
            "latency_ms": latency_ms,
            "fibre_type": fibre_type
        }

        Also sends to target_processor_id if set.
        """
        unit = annotation_info.get("unit", annotation_info.get("unit_id", 0))
        stimno = annotation_info.get("stimno", 0)
        latency_samples = annotation_info.get(
            "latency_samples",
            annotation_info.get("latency", 0),
        )
        latency_ms = annotation_info.get("latency_ms", 0.0)
        fibre_type = annotation_info.get("fibre_type", "")
        channel = annotation_info.get("channel", "")
        neighbor_channels = annotation_info.get("neighbor_channels", [])

        payload = {
            "source": "SpikeSpy",
            "type": "unit_annotation",
            "spikeGroup": unit,
            "stimno": stimno,
            "spikeSampleLatency": latency_samples,
            "latency_ms": latency_ms,
            "fibre_type": fibre_type,
            "channel": channel,
            "neighbor_channels": neighbor_channels,
        }
        msg = json.dumps(payload)
        logger.info("Syncing unit annotation to Open Ephys: %s", msg)

        # 1. Broadcast via MessageCenter
        ok = self.send_message(msg)

        # 2. If a specific processor target is configured (e.g. SpikeSpyStreamer or APTrack), forward config
        if self.target_processor_id is not None:
            self.send_processor_config(self.target_processor_id, msg)

        return ok
