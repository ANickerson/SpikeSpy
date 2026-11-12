"""Dedicated test suite for MockOpenEphysServer demo live streaming with dynamic stimulation frequency
and simulated multi-unit characteristics.
"""

from __future__ import annotations

import json
import time
import urllib.request
import numpy as np
import pytest
import quantities as pq

from spikespy.ViewerState import ViewerState
from spikespy.open_ephys_addin import OpenEphysLiveView
from tests.mock_open_ephys_server import DEFAULT_UNIT_CONFIGS, MockOpenEphysServer


def test_mock_server_frequency_and_multi_unit_config():
    """Verify MockOpenEphysServer supports frequency ranges (0.5 to 50 Hz) and multi-unit presets."""
    # 1. Frequency configuration & inter-sweep interval calculation
    for freq in [0.5, 1.0, 5.0, 10.0, 25.0, 50.0]:
        server = MockOpenEphysServer(stim_freq_hz=freq, num_units=1)
        assert server.stim_freq_hz == pytest.approx(freq)
        assert server.inter_sweep_interval_s == pytest.approx(1.0 / freq)

    # 2. 1 Unit configuration (Fast A-fiber)
    s1 = MockOpenEphysServer(num_units=1)
    assert len(s1.units) == 1
    u0 = s1.units[0]
    assert u0["name"] == "unit_0"
    assert "Fast A-fiber" in u0["fibre_type"]
    assert u0["latency_ms"] == pytest.approx(12.0)
    assert u0["jitter_ms"] == pytest.approx(0.5)
    assert u0["amplitude_uv"] == pytest.approx(65.0)
    assert u0["channels"] == [0, 1]
    assert s1.n_channels >= 2

    # 3. 2 Units configuration (Fast A-fiber + Medium A-delta)
    s2 = MockOpenEphysServer(num_units=2)
    assert len(s2.units) == 2
    u1 = s2.units[1]
    assert u1["name"] == "unit_1"
    assert "A-delta" in u1["fibre_type"]
    assert u1["latency_ms"] == pytest.approx(32.0)
    assert u1["jitter_ms"] == pytest.approx(2.0)
    assert u1["amplitude_uv"] == pytest.approx(45.0)
    assert u1["channels"] == [2, 3]
    assert s2.n_channels >= 4

    # 4. 3 Units configuration (A-fiber, A-delta, and Slow C-fiber)
    s3 = MockOpenEphysServer(num_units=3)
    assert len(s3.units) == 3
    u2 = s3.units[2]
    assert u2["name"] == "unit_2"
    assert "C-fiber" in u2["fibre_type"]
    assert u2["latency_ms"] == pytest.approx(80.0)
    assert u2["jitter_ms"] == pytest.approx(6.0)
    assert u2["amplitude_uv"] == pytest.approx(25.0)
    assert u2["channels"] == [4, 5]
    assert s3.n_channels >= 6

    # 5. Custom unit configuration
    custom_units = [
        {
            "unit_id": 99,
            "name": "custom_unit",
            "fibre_type": "Special sensory",
            "latency_ms": 18.5,
            "jitter_ms": 1.0,
            "amplitude_uv": 55.0,
            "duration_ms": 1.0,
            "channels": [0, 2],
            "channel_scales": [1.0, 0.5],
            "electrode": 0,
        }
    ]
    s_custom = MockOpenEphysServer(unit_configs=custom_units)
    assert len(s_custom.units) == 1
    assert s_custom.units[0]["name"] == "custom_unit"
    assert s_custom.units[0]["latency_ms"] == pytest.approx(18.5)


def test_dynamic_frequency_control():
    """Verify dynamic update of stimulation frequency via Python API, HTTP PUT, and messages."""
    server = MockOpenEphysServer(
        host="127.0.0.1",
        data_port=5646,
        event_port=5647,
        http_port=37546,
        stim_freq_hz=4.0,
        num_units=1,
    )
    server.start()

    try:
        assert server.stim_freq_hz == pytest.approx(4.0)
        assert server.inter_sweep_interval_s == pytest.approx(0.25)

        # 1. Update frequency via direct method
        server.set_stim_freq(15.0)
        assert server.stim_freq_hz == pytest.approx(15.0)
        assert server.inter_sweep_interval_s == pytest.approx(1.0 / 15.0)

        # Property setter
        server.stim_freq_hz = 8.0
        assert server.stim_freq_hz == pytest.approx(8.0)

        # 2. Update frequency via HTTP endpoint /api/stim_freq
        url = "http://127.0.0.1:37546/api/stim_freq"
        req = urllib.request.Request(
            url,
            data=json.dumps({"stim_freq_hz": 20.0}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="PUT",
        )
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            assert data.get("status") == "ok"
            assert data.get("stim_freq_hz") == 20.0
        assert server.stim_freq_hz == pytest.approx(20.0)

        # Check GET /api/stim_freq
        with urllib.request.urlopen(url, timeout=3.0) as resp:
            get_data = json.loads(resp.read().decode("utf-8"))
            assert get_data.get("stim_freq_hz") == pytest.approx(20.0)

        # 3. Update frequency via Open Ephys MessageCenter string command
        msg_url = "http://127.0.0.1:37546/api/message"
        req_msg = urllib.request.Request(
            msg_url,
            data=json.dumps({"text": "setStimFreq: 12.5"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="PUT",
        )
        with urllib.request.urlopen(req_msg, timeout=3.0) as resp:
            pass
        assert server.stim_freq_hz == pytest.approx(12.5)

    finally:
        server.stop()


def test_multi_unit_live_sweeps_and_annotations(qtbot):
    """Verify live stream slices sweeps with waveforms and spike annotations for all 3 simulated units."""
    server = MockOpenEphysServer(
        host="127.0.0.1",
        data_port=5656,
        event_port=5657,
        http_port=37556,
        sample_rate=30000.0,
        stim_freq_hz=12.0,
        num_units=3,
        n_channels=6,
    )
    server.start()

    try:
        state = ViewerState()
        live_view = OpenEphysLiveView(state=state)
        qtbot.addWidget(live_view)

        live_view.spin_data_port.setValue(5656)
        live_view.spin_event_port.setValue(5657)
        live_view.spin_http_port.setValue(37556)
        live_view.spin_window_ms.setValue(150.0)
        live_view.spin_pre_ms.setValue(10.0)
        live_view.edit_active_channels.setText("0-5")
        live_view.spin_rec_chan.setValue(0)
        live_view.chk_online_filter.setChecked(False)  # Raw continuous signals

        # Initialize state spike groups matching the 3 units
        live_view.combo_demo_units.setCurrentIndex(2)  # 3 units
        # Trigger live stream
        live_view.start_stream()

        # Wait for at least 3 sweeps to arrive
        qtbot.waitUntil(lambda: len(state.event_signal) >= 3, timeout=5000)

        assert state.get_erp().shape[0] >= 3

        # Verify all 3 units exist in state spike groups
        unit_names = [sg.event.name for sg in state.spike_groups]
        assert "unit_0" in unit_names
        assert "unit_1" in unit_names
        assert "unit_2" in unit_names

        # Map units by name
        unit_map = {sg.event.name: sg for sg in state.spike_groups}

        # Verify spike latencies recorded on sweep 0
        sg0 = unit_map["unit_0"]
        sg1 = unit_map["unit_1"]
        sg2 = unit_map["unit_2"]

        assert len(sg0.idx_arr) >= 1 and sg0.idx_arr[0] is not None
        assert len(sg1.idx_arr) >= 1 and sg1.idx_arr[0] is not None
        assert len(sg2.idx_arr) >= 1 and sg2.idx_arr[0] is not None

        lat0_samples = sg0.idx_arr[0][0]
        lat1_samples = sg1.idx_arr[0][0]
        lat2_samples = sg2.idx_arr[0][0]

        sr = 30000.0
        lat0_ms = (lat0_samples / sr) * 1000.0
        lat1_ms = (lat1_samples / sr) * 1000.0
        lat2_ms = (lat2_samples / sr) * 1000.0

        # Unit 0 (Fast A-fiber): ~12 ms (±0.5 ms jitter)
        assert 11.0 <= lat0_ms <= 13.0, f"Unit 0 latency unexpected: {lat0_ms} ms"

        # Unit 1 (Medium A-delta fiber): ~32 ms (±2.0 ms jitter)
        assert 29.0 <= lat1_ms <= 35.0, f"Unit 1 latency unexpected: {lat1_ms} ms"

        # Unit 2 (Slow C-fiber): ~80 ms (±6.0 ms jitter)
        assert 72.0 <= lat2_ms <= 88.0, f"Unit 2 latency unexpected: {lat2_ms} ms"

        # Verify multi-channel waveforms:
        # CH0 should have Unit 0 peak (~65 uV)
        erp_ch0 = state.get_erp()
        sw0_ch0 = erp_ch0[0].magnitude if hasattr(erp_ch0[0], "magnitude") else erp_ch0[0]
        peak_idx0 = int(round(lat0_samples))
        # Search window around Unit 0
        w0 = sw0_ch0[max(0, peak_idx0 - 15) : min(len(sw0_ch0), peak_idx0 + 15)]
        assert np.max(w0) > 40.0, f"Expected Unit 0 peak on CH0 > 40 uV, got {np.max(w0)}"

        # CH2 should have Unit 1 peak (~45 uV)
        assert "CH2" in state._live_aux_erp_buffers
        erp_ch2 = state._live_aux_erp_buffers["CH2"].data
        sw0_ch2 = erp_ch2[0].magnitude if hasattr(erp_ch2[0], "magnitude") else erp_ch2[0]
        peak_idx1 = int(round(lat1_samples))
        w1 = sw0_ch2[max(0, peak_idx1 - 25) : min(len(sw0_ch2), peak_idx1 + 25)]
        assert np.max(w1) > 25.0, f"Expected Unit 1 peak on CH2 > 25 uV, got {np.max(w1)}"

        # CH4 should have Unit 2 peak (~25 uV)
        assert "CH4" in state._live_aux_erp_buffers
        erp_ch4 = state._live_aux_erp_buffers["CH4"].data
        sw0_ch4 = erp_ch4[0].magnitude if hasattr(erp_ch4[0], "magnitude") else erp_ch4[0]
        peak_idx2 = int(round(lat2_samples))
        w2 = sw0_ch4[max(0, peak_idx2 - 35) : min(len(sw0_ch4), peak_idx2 + 35)]
        assert np.max(w2) > 12.0, f"Expected Unit 2 peak on CH4 > 12 uV, got {np.max(w2)}"

        live_view.stop_stream()
    finally:
        server.stop()


def test_demo_stream_ui_controls(qtbot):
    """Verify built-in demo UI controls: stimulation frequency, demo units combo, dynamic updates."""
    state = ViewerState()
    live_view = OpenEphysLiveView(state=state)
    qtbot.addWidget(live_view)

    # 1. Check UI control defaults and ranges
    assert hasattr(live_view, "spin_demo_stim_freq")
    assert hasattr(live_view, "combo_demo_units")
    assert live_view.spin_demo_stim_freq.minimum() == 0.5
    assert live_view.spin_demo_stim_freq.maximum() == 50.0
    assert live_view.spin_demo_stim_freq.value() == 5.0
    assert live_view.combo_demo_units.count() == 3

    # 2. Select 2 Units and set frequency to 8.0 Hz
    live_view.combo_demo_units.setCurrentIndex(1)  # 2 Units
    live_view.spin_demo_stim_freq.setValue(8.0)

    # 3. Start demo stream
    live_view.start_demo_stream()
    assert live_view._demo_server_process is not None
    assert live_view._demo_server_process.is_running()
    assert live_view._demo_server_process.stim_freq_hz == pytest.approx(8.0)
    assert len(live_view._demo_server_process.units) == 2

    # Check state was automatically initialized with 2 units
    assert len(state.spike_groups) == 2
    assert state.spike_groups[0].event.name == "unit_0"
    assert state.spike_groups[1].event.name == "unit_1"

    # Wait for sweeps
    qtbot.waitUntil(lambda: len(state.event_signal) >= 2, timeout=5000)
    assert state.get_erp().shape[0] >= 2

    # 4. Test dynamic frequency change while running
    live_view.spin_demo_stim_freq.setValue(16.0)
    assert live_view._demo_server_process.stim_freq_hz == pytest.approx(16.0)

    # 5. Stop demo stream
    live_view.start_demo_stream()  # Toggles off
    assert live_view._demo_server_process is None
    assert live_view._is_streaming is False
