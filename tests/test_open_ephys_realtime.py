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


def test_realistic_microneurography_and_event_marker_alignment(qtbot):
    """Verify realistic microneurography trace features:
    1. Action potential with positive peak aligned with event markers in ERP.
    2. Activity-dependent slowing: spike latency increases progressively over sweeps.
    3. Multi-channel footprint: nearby channels CH1, CH2, CH3 have the same spike offset by conduction delay.
    4. Distinct noise profiles across channels.
    """
    server = MockOpenEphysServer(
        host="127.0.0.1",
        data_port=5606,
        event_port=5607,
        http_port=37525,
        n_channels=4,
        sample_rate=30000.0,
        stim_freq_hz=10.0,
    )
    server.start()

    try:
        state = ViewerState()
        live_view = OpenEphysLiveView(state=state)
        qtbot.addWidget(live_view)

        live_view.spin_data_port.setValue(5606)
        live_view.spin_event_port.setValue(5607)
        live_view.spin_http_port.setValue(37525)
        live_view.spin_window_ms.setValue(100.0)
        live_view.spin_pre_ms.setValue(10.0)
        live_view.edit_active_channels.setText("0-3")
        live_view.spin_rec_chan.setValue(0)

        live_view.start_stream()
        # Collect at least 4 sweeps to measure slowing
        qtbot.waitUntil(lambda: len(state.event_signal) >= 4, timeout=5000)

        erp_ch0 = state.get_erp()
        assert erp_ch0.shape[0] >= 4

        # 1. Action potential morphology: clear positive peak (> 40 uV)
        sweep0 = erp_ch0[0].magnitude if hasattr(erp_ch0[0], "magnitude") else erp_ch0[0]
        # Peak should be around 25 ms (750 samples at 30 kHz)
        search_window = sweep0[650:850]
        peak_sample_sweep0 = 650 + int(np.argmax(search_window))
        assert sweep0[peak_sample_sweep0] > 30.0, f"Expected positive peak > 30 uV, got {sweep0[peak_sample_sweep0]}"

        # 2. Event marker alignment: verify unit_0 marker matches peak_sample_sweep0!
        ug = state.getUnitGroup()
        assert ug is not None
        assert ug.idx_arr is not None and len(ug.idx_arr) >= 1
        marked_lat = ug.idx_arr[0][0]
        # Must be aligned within 5 samples (~0.16 ms, accounting for causal filter group delay)
        assert abs(marked_lat - peak_sample_sweep0) <= 5, (
            f"Event marker at sample {marked_lat} does not align with peak at {peak_sample_sweep0}!"
        )

        # 3. Activity-dependent slowing: sweep 3 peak should have shifted to the right (higher latency)
        sweep3 = erp_ch0[3].magnitude if hasattr(erp_ch0[3], "magnitude") else erp_ch0[3]
        search_window3 = sweep3[650:950]
        peak_sample_sweep3 = 650 + int(np.argmax(search_window3))
        assert peak_sample_sweep3 >= peak_sample_sweep0, (
            f"Expected slowing: sweep 3 ({peak_sample_sweep3}) should be >= sweep 0 ({peak_sample_sweep0})"
        )

        # 4. Multi-channel footprint: CH1 should have the same spike offset by propagation delay
        erp_ch1 = state._live_aux_erp_buffers["CH1"].data
        sweep0_ch1 = erp_ch1[0].magnitude if hasattr(erp_ch1[0], "magnitude") else erp_ch1[0]
        w_start = max(0, peak_sample_sweep0 - 5)
        w_end = min(len(sweep0_ch1), peak_sample_sweep0 + 25)
        peak_ch1 = w_start + int(np.argmax(sweep0_ch1[w_start:w_end]))
        # Conduction delay: CH1 is delayed relative to CH0 by ~4 samples (+0.133 ms)
        assert peak_ch1 >= peak_sample_sweep0

        # Spatial attenuation on sweep 0:
        erp_ch2 = state._live_aux_erp_buffers["CH2"].data
        sweep0_ch2 = erp_ch2[0].magnitude if hasattr(erp_ch2[0], "magnitude") else erp_ch2[0]
        peak_ch2 = w_start + int(np.argmax(sweep0_ch2[w_start:w_end]))
        assert abs(peak_ch2 - peak_ch1) <= 10
        assert peak_ch2 >= peak_sample_sweep0
        # More distant channel CH2 shows clear spatial attenuation relative to primary CH0
        assert sweep0[peak_sample_sweep0] > sweep0_ch2[peak_ch2]

        # 5. Distinct noise profiles: CH0, CH2 (clean low-noise), and CH3 (broadband hash) have distinct noise profiles
        # Measure noise in baseline period (samples 100:600)
        std_ch0 = np.std(sweep0[100:600])
        erp_ch2 = state._live_aux_erp_buffers["CH2"].data
        sweep0_ch2 = erp_ch2[0].magnitude if hasattr(erp_ch2[0], "magnitude") else erp_ch2[0]
        std_ch2 = np.std(sweep0_ch2[100:600])
        erp_ch3 = state._live_aux_erp_buffers["CH3"].data
        sweep0_ch3 = erp_ch3[0].magnitude if hasattr(erp_ch3[0], "magnitude") else erp_ch3[0]
        std_ch3 = np.std(sweep0_ch3[100:600])

        # CH2 is a low-noise contact (significantly lower RMS than CH0)
        assert std_ch2 < std_ch0 - 0.5
        # CH3 is a broadband hash contact (significantly higher RMS than CH0)
        assert std_ch3 > std_ch0 + 0.5

        live_view.stop_stream()
    finally:
        server.stop()


def test_open_ephys_heartbeat_and_3part_packets(qtbot):
    """Verify heartbeat is delivered on listening port and 3-part OE frames are parsed."""
    import threading
    import time
    import zmq

    ctx = zmq.Context()
    # Mock listening REP socket on 5592
    rep_sock = ctx.socket(zmq.REP)
    rep_sock.setsockopt(zmq.RCVTIMEO, 1000)
    rep_sock.setsockopt(zmq.LINGER, 0)
    rep_sock.bind("tcp://127.0.0.1:5592")

    # Mock data PUB socket on 5591
    pub_sock = ctx.socket(zmq.PUB)
    pub_sock.setsockopt(zmq.LINGER, 0)
    pub_sock.bind("tcp://127.0.0.1:5591")

    received_heartbeats = []
    stop_event = threading.Event()

    def rep_listener():
        rep_poller = zmq.Poller()
        rep_poller.register(rep_sock, zmq.POLLIN)
        while not stop_event.is_set():
            socks = dict(rep_poller.poll(50))
            if rep_sock in socks:
                try:
                    msg = rep_sock.recv_string(zmq.NOBLOCK)
                    received_heartbeats.append(msg)
                    rep_sock.send_string('{"status": "ok"}')
                except Exception:
                    break

    t = threading.Thread(target=rep_listener, daemon=True)
    t.start()

    from spikespy.open_ephys_addin.oe_stream_worker import OpenEphysStreamWorker
    worker = OpenEphysStreamWorker(
        host="127.0.0.1",
        data_port=5591,
        event_port=5592,
        heartbeat_port=5592,
        mode="zmq_raw",
        apply_filter=False,
        window_ms=100.0,
        pre_stim_ms=10.0,
        sample_rate=30000.0,
    )

    sweeps = []
    worker.sweepsReady.connect(sweeps.extend)
    worker.start()

    try:
        # Wait for heartbeat to arrive at REP socket (proves connection established)
        qtbot.waitUntil(lambda: len(received_heartbeats) > 0, timeout=3000)
        hb_data = json.loads(received_heartbeats[0])
        assert hb_data.get("type") == "heartbeat"
        assert hb_data.get("application") == "SpikeSpy"

        # Give SUB socket a brief moment to complete subscription handshake
        import time
        time.sleep(0.1)

        # Now test 3-part frame delivery: [b"data", header_json, buffer]
        hdr = json.dumps({
            "stream": "continuous",
            "channel_count": 1,
            "sample_rate": 30000.0,
            "sample_num": 0,
            "num_samples": 6000,
            "dtype": "float32",
        }).encode("utf-8")
        data_chunk = np.ones((1, 6000), dtype=np.float32).tobytes()

        # Send 3-part data frame
        pub_sock.send_multipart([b"data", hdr, data_chunk])

        # Also send a TTL event trigger on data socket
        evt_hdr = json.dumps({
            "stream": "events",
            "type": "ttl",
            "sample_num": 1500,
        }).encode("utf-8")
        evt_buf = bytes([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])  # line 1, state 1
        pub_sock.send_multipart([b"event", evt_hdr, evt_buf])

        # Send second continuous chunk so pending sweep can complete
        hdr2 = json.dumps({
            "stream": "continuous",
            "channel_count": 1,
            "sample_rate": 30000.0,
            "sample_num": 6000,
            "num_samples": 6000,
            "dtype": "float32",
        }).encode("utf-8")
        pub_sock.send_multipart([b"data", hdr2, data_chunk])

        qtbot.waitUntil(lambda: len(sweeps) > 0, timeout=3000)
        assert len(sweeps) >= 1
        assert sweeps[0]["stim_timestamp_s"] == 1500 / 30000.0

    finally:
        stop_event.set()
        worker.stop()
        t.join(timeout=1.0)
        rep_sock.close(linger=0)
        pub_sock.close(linger=0)
        ctx.term()


def test_live_view_zmq_raw_mode_sends_heartbeat_to_oe_zmq_interface(qtbot):
    """Regression test: OpenEphysLiveView in Standard OE ZMQ mode must send heartbeats
    to the listening/event port so Open Ephys ZMQ Interface reports SpikeSpy as connected.
    """
    import threading
    import zmq

    ctx = zmq.Context()
    rep_sock = ctx.socket(zmq.REP)
    rep_sock.setsockopt(zmq.RCVTIMEO, 1000)
    rep_sock.setsockopt(zmq.LINGER, 0)
    rep_sock.bind("tcp://127.0.0.1:5617")

    received_heartbeats = []
    stop_event = threading.Event()

    def rep_listener():
        rep_poller = zmq.Poller()
        rep_poller.register(rep_sock, zmq.POLLIN)
        while not stop_event.is_set():
            socks = dict(rep_poller.poll(50))
            if rep_sock in socks:
                try:
                    msg = rep_sock.recv_string(zmq.NOBLOCK)
                    received_heartbeats.append(msg)
                    rep_sock.send_string("OK")
                except Exception:
                    break

    t = threading.Thread(target=rep_listener, daemon=True)
    t.start()

    state = ViewerState()
    live_view = OpenEphysLiveView(state=state)
    qtbot.addWidget(live_view)

    # Mode 0 = Standard OE ZMQ + Event Broadcaster ("zmq_raw")
    live_view.combo_stream_mode.setCurrentIndex(0)
    live_view.spin_data_port.setValue(5616)
    live_view.spin_event_port.setValue(5617)
    live_view.spin_http_port.setValue(37535)

    try:
        live_view.start_stream()
        assert live_view.worker is not None
        assert live_view.worker.mode == "zmq_raw"
        assert live_view.worker.heartbeat_port == 5617

        qtbot.waitUntil(lambda: len(received_heartbeats) > 0, timeout=3000)
        hb_data = json.loads(received_heartbeats[0])
        assert hb_data.get("type") == "heartbeat"
        assert hb_data.get("application") == "SpikeSpy"
        assert "uuid" in hb_data
    finally:
        live_view.stop_stream()
        stop_event.set()
        t.join(timeout=1.0)
        rep_sock.close(linger=0)
        ctx.term()


def test_oe_zmq_interface_v1_per_channel_nested_content_and_ttl(qtbot):
    """Regression test matching real Open Ephys ZMQ Interface v1.0 wire format:
    - 3-part DATA frames: [b'DATA\\x00', {"message_num": ..., "type": "data", "content": {"channel_num": k, "num_samples": ..., "sample_num": ..., "sample_rate": ...}}, float32 bytes]
    - 3-part EVENT frames: [b'EVENT\\x00', {"message_num": ..., "type": "event", "content": {"type": 3, "sample_num": ...}, "timestamp": ms_epoch}, 10-byte TTL payload]
    Verifies all channels populate distinct rows and rising TTL edges slice sweeps accurately.
    """
    worker = OpenEphysStreamWorker(
        host="127.0.0.1",
        data_port=5626,
        event_port=5627,
        heartbeat_port=5627,
        mode="zmq_raw",
        rec_channel=0,
        active_channels=[0, 1, 2, 3],
        apply_filter=False,
        window_ms=100.0,
        pre_stim_ms=10.0,
        sample_rate=30000.0,
    )

    sweeps = []
    debug_events = []
    worker.sweepsReady.connect(sweeps.extend)
    worker.debugEventEmitted.connect(debug_events.append)

    base_sample = 77098240
    block_len = 640
    n_ch = 4

    # Feed 10 blocks (~213 ms) of 4-channel single-channel packets, with a distinct DC level per channel
    for b_idx in range(10):
        start_s = base_sample + b_idx * block_len
        for ch in range(n_ch):
            hdr = json.dumps({
                "message_num": 1000 + b_idx * n_ch + ch,
                "type": "data",
                "content": {
                    "stream": "Data-B",
                    "channel_num": ch,
                    "channel_name": f"CH{ch + 1}",
                    "num_samples": block_len,
                    "sample_num": start_s,
                    "sample_rate": 30000.0,
                },
                "data_size": block_len * 4,
            }).encode("utf-8")
            payload = np.full(block_len, float(ch + 1) * 10.0, dtype=np.float32).tobytes()
            worker._parse_raw_zmq_data_packet([b"DATA\x00", hdr, payload])

        # Send rising + falling TTL event during block 2 (at base_sample + 1000)
        if b_idx == 2:
            stim_sample = base_sample + 1000
            evt_rise_hdr = json.dumps({
                "message_num": 2000,
                "type": "event",
                "content": {
                    "stream": "Data-B",
                    "source_node": 111,
                    "type": 3,
                    "sample_num": stim_sample,
                },
                "data_size": 10,
                "timestamp": 1795293815117,
            }).encode("utf-8")
            worker._parse_raw_zmq_data_packet([
                b"EVENT\x00",
                evt_rise_hdr,
                b"\x00\x01\x01\x00\x00\x00\x00\x00\x00\x00",
            ])

            evt_fall_hdr = json.dumps({
                "message_num": 2001,
                "type": "event",
                "content": {
                    "stream": "Data-B",
                    "source_node": 111,
                    "type": 3,
                    "sample_num": stim_sample + 300,
                },
                "data_size": 10,
                "timestamp": 1795293815127,
            }).encode("utf-8")
            worker._parse_raw_zmq_data_packet([
                b"EVENT\x00",
                evt_fall_hdr,
                b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
            ])

    worker._check_pending_sweeps()
    worker._flush_sweep_payloads(force=True)

    # RingBuffer latest_sample must track the real sample_num, not 0-based
    assert worker.ring_buffer.latest_sample == base_sample + 10 * block_len - 1

    # Exactly 1 sweep should have been sliced from the rising edge (falling edge ignored)
    assert len(sweeps) == 1
    sw = sweeps[0]
    assert np.isclose(sw["stim_timestamp_s"], (base_sample + 1000) / 30000.0)
    assert sw["waveform"].shape == (3300,)  # 10 ms pre (300) + 100 ms window (3000)

    # Every channel CH0..CH3 must contain its own distinct signal (not all shoved into CH0)
    for ch in range(n_ch):
        ch_wf = sw["aux_sweeps"][f"CH{ch}"]
        assert ch_wf.shape == (3300,)
        assert np.allclose(ch_wf, float(ch + 1) * 10.0)

    statuses = [e["status"] for e in debug_events]
    assert any("TRIGGER QUEUED" in s for s in statuses)
    assert any("IGNORED (state=0)" in s for s in statuses)
    assert any("SWEEP SLICED" in s for s in statuses)




