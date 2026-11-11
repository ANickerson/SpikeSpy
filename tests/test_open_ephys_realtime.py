"""End-to-end integration tests for the Real-Time Open Ephys v1.0+ Interface in SpikeSpy."""

import json
import time
import numpy as np
import pytest
import quantities as pq

from spikespy.ViewerState import ViewerState
from spikespy.processing import (
    LiveERPBuffer,
    MultiAnalogSignal,
    OnlineBandpassFilter,
    OnlineTTLDetector,
)
from spikespy.basic_tracking import track_single_sweep
from spikespy.MultiTraceView import MultiTraceView
from spikespy.SingleTraceView import SingleTraceView
from spikespy.UnitView import UnitView
from spikespy.TrackingView import TrackingView
from spikespy.SpikeGroupTable import SpikeGroupTableView
from spikespy.open_ephys_addin import (
    OpenEphysController,
    OpenEphysLiveView,
    OpenEphysStreamWorker,
)
from tests.mock_open_ephys_server import MockOpenEphysServer


def test_live_erp_buffer_and_multi_analog_signal():
    """Verify O(1) amortized LiveERPBuffer and dynamic MultiAnalogSignal appending."""
    buf = LiveERPBuffer(window_samples=500, units=pq.uV, initial_capacity=4)
    assert len(buf) == 0

    for i in range(10):
        sweep = np.ones(500, dtype=np.float32) * i
        buf.append_sweep(sweep)

    assert len(buf) == 10
    assert buf.data.shape == (10, 500)
    assert np.allclose(buf.data[9].magnitude, 9.0)

    mas = MultiAnalogSignal()
    assert len(mas) == 0
    for i in range(3):
        import neo

        sig = neo.AnalogSignal(
            np.arange(100, dtype=float) + i,
            units=pq.uV,
            sampling_rate=1000 * pq.Hz,
            t_start=(i * 1.0) * pq.s,
        )
        mas.append_signal(sig)

    assert len(mas) == 3
    sliced = mas.time_slice(1.01 * pq.s, 1.05 * pq.s)
    assert len(sliced) == 40


def test_online_bandpass_and_ttl_detector():
    """Verify stateful causal bandpass filter and analog rising-edge TTL detector."""
    filt = OnlineBandpassFilter(500.0, 7000.0, sample_rate=30000.0)
    t = np.arange(3000) / 30000.0
    # 1000 Hz tone (in band) + 10 Hz drift (out of band)
    sig = np.sin(2 * np.pi * 1000 * t) * 50.0 + np.sin(2 * np.pi * 10 * t) * 500.0
    out1 = filt.process(sig[:1500])
    out2 = filt.process(sig[1500:])
    assert out1.shape == (1500,)
    assert out2.shape == (1500,)
    # Ensure low-frequency 500 uV drift is strongly attenuated
    assert np.max(np.abs(out2)) < 100.0

    det = OnlineTTLDetector(threshold=2.5, min_width_samples=5, refractory_samples=1000)
    ttl_chunk = np.zeros(2000, dtype=np.float32)
    ttl_chunk[200:250] = 5.0
    ttl_chunk[500:550] = 5.0  # within 1000-sample refractory period -> ignored
    ttl_chunk[1500:1550] = 5.0  # after refractory -> detected
    events = det.process_chunk(ttl_chunk, start_sample=0, sample_rate=30000.0)
    assert len(events) == 2
    assert events[0]["sample_index"] == 200
    assert events[1]["sample_index"] == 1500


def test_viewer_state_live_sweeps_and_auto_tracking(qtbot):
    """Verify ViewerState live session initialization, sweep appending, and live views."""
    state = ViewerState()
    state.start_live_session(sampling_rate=10000.0, window_size_ms=100.0, units=pq.uV)
    assert state.is_streaming is True
    assert state.get_erp().shape == (0, 1000)

    mt_view = MultiTraceView(state=state)
    st_view = SingleTraceView(state=state)
    u_view = UnitView(state=state)
    tr_view = TrackingView(state=state)
    tbl_view = SpikeGroupTableView(state=state)
    qtbot.addWidget(mt_view)
    qtbot.addWidget(st_view)
    qtbot.addWidget(u_view)
    qtbot.addWidget(tr_view)
    qtbot.addWidget(tbl_view)

    # Enable live auto-tracking on TrackingView with a 50 uV threshold (= 0.05 mV)
    tr_view.qsb_threshold.setValue(0.05)
    tr_view.qsb_window_size.setValue(10.0)
    tr_view.qcb_auto_track.setChecked(True)

    # Sweep 0: spike at sample 250 (25 ms latency), manually annotated to seed tracking
    w0 = np.zeros(1000, dtype=np.float32)
    w0[250] = 100.0  # 100 uV = 0.1 mV peak
    idx0 = state.append_sweep(1.0, w0)
    assert idx0 == 0
    state.setUnit(250, stimno=0)
    assert state.getUnitGroup().idx_arr[0][0] == 250

    # Sweep 1: spike drifts to sample 260 (26 ms latency) -> should be auto-tracked!
    w1 = np.zeros(1000, dtype=np.float32)
    w1[260] = 100.0
    idx1 = state.append_sweep(2.0, w1)
    assert idx1 == 1
    assert state.getUnitGroup().idx_arr[1] is not None
    assert state.getUnitGroup().idx_arr[1][0] == 260

    state.stop_live_session()
    assert state.is_streaming is False


def test_open_ephys_mock_server_and_bidirectional_sync(qtbot):
    """Verify live ZMQ streaming and bidirectional HTTP sync against MockOpenEphysServer."""
    server = MockOpenEphysServer(
        host="127.0.0.1",
        data_port=5586,
        event_port=5587,
        http_port=37505,
        sample_rate=30000.0,
        stim_freq_hz=10.0,
    )
    server.start()

    try:
        state = ViewerState()
        live_view = OpenEphysLiveView(state=state)
        qtbot.addWidget(live_view)

        live_view.spin_data_port.setValue(5586)
        live_view.spin_event_port.setValue(5587)
        live_view.spin_http_port.setValue(37505)
        live_view.spin_window_ms.setValue(100.0)
        live_view.spin_pre_ms.setValue(10.0)
        live_view.spin_target_proc.setValue(103)

        # 1. Verify HTTP controller status & processor query
        live_view.controller.host = "127.0.0.1"
        live_view.controller.port = 37505
        assert live_view.controller.get_status() == "ACQUIRE"
        procs = live_view.controller.get_processors()
        assert any(p["name"] == "SpikeSpyStreamer" for p in procs)

        # 2. Start live stream and wait for at least 2 sweeps to be sliced and delivered
        live_view.start_stream()
        qtbot.waitUntil(lambda: len(state.event_signal) >= 2, timeout=4000)

        assert state.get_erp().shape[0] >= 2
        assert state.get_erp().shape[1] == int(30000 * 0.1)

        # 3. Annotate a unit in SpikeSpy and verify bidirectional sync to Open Ephys HTTP server
        state.setUnit(750, stimno=0)
        assert len(server.received_messages) >= 1
        last_msg = json.loads(server.received_messages[-1]["text"])
        assert last_msg["source"] == "SpikeSpy"
        assert last_msg["spikeSampleLatency"] == 750
        assert "channel" in last_msg
        assert "neighbor_channels" in last_msg
        assert len(server.received_processor_configs) >= 1
        assert server.received_processor_configs[-1]["processor_id"] == 103

        live_view.stop_stream()
    finally:
        server.stop()


def test_multichannel_streaming_and_channel_switching(qtbot):
    """Verify multi-channel streaming, aux sweep buffering, and live channel switching."""
    server = MockOpenEphysServer(
        host="127.0.0.1",
        data_port=5596,
        event_port=5597,
        http_port=37515,
        n_channels=4,
        sample_rate=30000.0,
        stim_freq_hz=10.0,
    )
    server.start()

    try:
        state = ViewerState()
        live_view = OpenEphysLiveView(state=state)
        qtbot.addWidget(live_view)

        live_view.spin_data_port.setValue(5596)
        live_view.spin_event_port.setValue(5597)
        live_view.spin_http_port.setValue(37515)
        live_view.spin_window_ms.setValue(100.0)
        live_view.spin_pre_ms.setValue(10.0)
        live_view.edit_active_channels.setText("0-3")
        live_view.spin_rec_chan.setValue(0)

        live_view.start_stream()
        qtbot.waitUntil(lambda: len(state.event_signal) >= 2, timeout=4000)

        # Ensure aux buffers were populated for CH0, CH1, CH2, CH3
        assert hasattr(state, "_live_aux_erp_buffers")
        assert "CH1" in state._live_aux_erp_buffers
        assert len(state._live_aux_erp_buffers["CH1"]) >= 2

        # Switch active displayed channel to CH1
        live_view.spin_rec_chan.setValue(1)
        assert state.analog_signal.name == "CH1"
        assert state.get_erp().shape[0] >= 2

        # Switch back to CH0
        live_view.spin_rec_chan.setValue(0)
        assert state.analog_signal.name == "CH0"
        assert state.get_erp().shape[0] >= 2

        live_view.stop_stream()
    finally:
        server.stop()

