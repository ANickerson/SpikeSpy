"""Tests for Open Ephys plugin bidirectional connection status and heartbeat detection."""

import time
import pytest
from PySide6.QtCore import QCoreApplication

from spikespy.ViewerState import ViewerState
from spikespy.open_ephys_addin import (
    OpenEphysController,
    OpenEphysLiveView,
    OpenEphysStreamWorker,
)
from tests.mock_open_ephys_server import MockOpenEphysServer


def test_controller_processor_discovery_and_ping():
    """Verify OpenEphysController can discover SpikeSpyStreamer processor and ping it."""
    http_port = 37510
    data_port = 5572
    event_port = 5573
    server = MockOpenEphysServer(
        host="127.0.0.1",
        data_port=data_port,
        event_port=event_port,
        http_port=http_port,
        stim_freq_hz=5.0,
    )
    server.start()
    try:
        controller = OpenEphysController(host="127.0.0.1", port=http_port)
        status = controller.get_status()
        assert status in ("ACQUIRE", "RECORD", "IDLE")

        # Discover SpikeSpyStreamer processor
        proc_id = controller.find_spikespy_streamer_processor()
        assert proc_id == 103

        # Ping processor
        success, resp = controller.ping_processor(proc_id)
        assert success is True
        assert resp.get("status") == "ok"
        assert resp.get("plugin") == "SpikeSpyStreamer"
    finally:
        server.stop()


def test_stream_worker_heartbeat_connection_status(qtbot):
    """Verify OpenEphysStreamWorker handles heartbeat REQ/REP and updates connection status."""
    http_port = 37511
    data_port = 5574
    event_port = 5575
    heartbeat_port = 5576

    server = MockOpenEphysServer(
        host="127.0.0.1",
        data_port=data_port,
        event_port=event_port,
        http_port=http_port,
        heartbeat_port=heartbeat_port,
        stim_freq_hz=5.0,
    )
    server.start()

    worker = OpenEphysStreamWorker(
        host="127.0.0.1",
        data_port=data_port,
        event_port=event_port,
        heartbeat_port=heartbeat_port,
        mode="sliced_erp",
    )

    statuses = []
    worker.connectionStatusChanged.connect(lambda conn, msg: statuses.append((conn, msg)))

    stats_list = []
    worker.statsUpdated.connect(lambda stats: stats_list.append(stats))

    worker.start()
    try:
        # Wait for worker to connect via heartbeat (heartbeat sent within 1.2s)
        qtbot.waitUntil(lambda: any(conn for conn, _ in statuses), timeout=4000)
        assert any(conn for conn, _ in statuses)

        # Verify stats updated with plugin_connected = True
        qtbot.waitUntil(lambda: any(s.get("plugin_connected") for s in stats_list), timeout=3000)
        last_stat = [s for s in stats_list if s.get("plugin_connected")][-1]
        assert last_stat["plugin_connected"] is True
        assert last_stat.get("plugin_info") is not None
        assert last_stat["plugin_info"].get("plugin") == "SpikeSpyStreamer"
    finally:
        worker.stop()
        server.stop()


def test_oe_live_view_status_badges(qtbot):
    """Verify OpenEphysLiveView updates stream, GUI, and plugin connection badges."""
    state = ViewerState()
    view = OpenEphysLiveView(state=state)
    qtbot.addWidget(view)

    # 1. Initial disconnected badges
    assert "○ Disconnected" in view.lbl_status.text()
    assert "○ Disconnected" in view.lbl_oe_gui_status.text()

    # 2. Connection status changed when streaming
    view._is_streaming = True
    view._on_connection_status_changed(True, "Connected to SpikeSpy Streamer")
    assert "● Connected" in view.lbl_status.text()
    assert "#2b7a2b" in view.lbl_status.styleSheet()

    # Transition to waiting
    view._on_connection_status_changed(False, "Waiting for stream...")
    assert "◌ Waiting" in view.lbl_status.text()
    assert "#d9822b" in view.lbl_status.styleSheet()

    # Stop stream
    view.stop_stream()
    assert "○ Stopped" in view.lbl_status.text()

    # 3. Open Ephys GUI status changed
    view._on_oe_status_changed("ACQUIRE")
    assert "● ACQUIRE" in view.lbl_oe_gui_status.text()
    assert "#2b7a2b" in view.lbl_oe_gui_status.styleSheet()

    view._on_oe_status_changed("IDLE")
    assert "◌ IDLE" in view.lbl_oe_gui_status.text()
    assert "#d9822b" in view.lbl_oe_gui_status.styleSheet()

    view._on_oe_status_changed("DISCONNECTED")
    assert "○ Disconnected" in view.lbl_oe_gui_status.text()

    # 4. Plugin detection via statsUpdated
    view._on_stats_updated({
        "sweeps_received": 10,
        "spikes_received": 5,
        "last_stim_freq_hz": 4.0,
        "plugin_connected": True,
        "plugin_info": {"plugin": "SpikeSpy Streamer", "sweeps_sent": 10},
    })
    assert "● SpikeSpy Streamer" in view.lbl_plugin_detected.text()
    assert "10 sweeps sent" in view.lbl_plugin_detected.text()
    assert "#2b7a2b" in view.lbl_plugin_detected.styleSheet()
