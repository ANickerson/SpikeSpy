import json
import time
import numpy as np
import quantities as pq
import pytest
from PySide6.QtWidgets import QApplication

from spikespy.ViewerState import ViewerState
from spikespy.ui_diagnostics import UIWatchdog, AdaptiveFrameScheduler
from spikespy.open_ephys_addin.oe_live_view import OpenEphysLiveView
from spikespy.open_ephys_addin.oe_stream_worker import OpenEphysStreamWorker
from spikespy.SingleTracePGView import SingleTracePGView
from spikespy.MultiTracePGView import MultiTracePGView
from spikespy.MultiTraceFixedPGView import MultiTraceFixedPGView
from spikespy.UnitPGView import UnitPGView
from spikespy.UnitFootprintPGView import UnitFootprintPGView
from spikespy.SpikeGroupTable import SpikeGroupTableView
from spikespy.TrackingView import TrackingView


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_ui_watchdog_detects_main_thread_stall(qapp):
    """Verify UIWatchdog detects an artificial main-thread block and captures Python stack frames."""
    watchdog = UIWatchdog(
        stall_threshold_ms=120.0,
        heartbeat_interval_ms=25,
        log_to_stderr=False,
    )
    watchdog.start()
    try:
        qapp.processEvents()
        # Artificially block the main thread longer than stall_threshold_ms
        time.sleep(0.22)
        qapp.processEvents()

        assert len(watchdog.reports) >= 1
        report = watchdog.reports[-1]
        assert report.duration_ms >= 120.0
        assert "test_ui_watchdog_detects_main_thread_stall" in report.stack_trace
    finally:
        watchdog.stop()


def test_adaptive_frame_scheduler_scales_intervals_under_strain():
    """Verify AdaptiveFrameScheduler scales effective intervals and enforces idle gap under heavy render load."""
    sched = AdaptiveFrameScheduler(base_interval_ms=33, min_idle_gap_ms=12, max_interval_ms=400)

    # Normal low-cost rendering: no strain
    sched.record_render_time("SingleTracePGView", 2.0, weight="primary")
    sched.record_render_time("MultiTracePGView", 3.0, weight="normal")
    assert sched.strain_level < 0.25
    assert not sched.should_skip_nonessential_work()
    assert sched.get_effective_interval_ms(weight="primary") == 33

    # Simulate heavy multi-window rendering (e.g. views taking ~22-28ms each)
    for _ in range(6):
        sched.record_render_time("SingleTraceView", 22.0, weight="heavy")
        sched.record_render_time("MultiTraceView", 25.0, weight="heavy")
        sched.record_render_time("UnitFootprintView", 28.0, weight="heavy")
        sched.update_loop_lag(65.0)

    assert sched.strain_level >= 0.8
    assert sched.should_skip_nonessential_work()

    primary_iv = sched.get_effective_interval_ms(weight="primary")
    normal_iv = sched.get_effective_interval_ms(weight="normal")
    heavy_iv = sched.get_effective_interval_ms(weight="heavy")

    # Heavy views back off more aggressively than normal, which back off more than primary
    assert heavy_iv > normal_iv >= primary_iv
    assert heavy_iv >= 60

    summary = sched.get_diagnostics_summary()
    assert "SingleTraceView" in summary["views"]
    assert "MultiTraceView" in summary["views"]


def test_stream_worker_coalesces_spike_events_within_pending_sweep():
    """Verify spike events falling inside an active pending sweep do not emit redundant immediate signals."""
    worker = OpenEphysStreamWorker(window_ms=40.0, pre_stim_ms=10.0)
    worker.sample_rate = 10000.0
    emitted_batches = []
    worker.spikeEventsReady.connect(lambda unit, evts: emitted_batches.append((unit, evts)))

    # Queue a stimulus trigger at sample 1000 (sweep window: [900, 1400])
    worker.queue_sweep_trigger(1000, 0.1)
    assert len(worker._pending_sweeps) == 1

    # Spike at sample 1120 falls inside the pending sweep [900, 1400]:
    # should be attached to the pending sweep and NOT emitted immediately via spikeEventsReady
    raw_in_sweep = json.dumps({
        "type": "spike",
        "sample_number": 1120,
        "unit_name": "unit_0",
        "electrode": 0,
    }).encode("utf-8")
    worker._handle_event_json(raw_in_sweep)
    assert len(emitted_batches) == 0
    assert "unit_0" in worker._pending_sweeps[0].spike_latencies

    # Spike at sample 5000 is outside any pending sweep: should emit spikeEventsReady
    raw_out_sweep = json.dumps({
        "type": "spike",
        "sample_number": 5000,
        "unit_name": "unit_0",
        "electrode": 0,
    }).encode("utf-8")
    worker._handle_event_json(raw_out_sweep)
    assert len(emitted_batches) == 1


def test_multi_window_50hz_live_streaming_responsive(qapp):
    """
    Stress-test 50 Hz live streaming with 8 simultaneous views open.
    Verifies sweep batch coalescing, responsive event loop (no freeze), and bounded render counts.
    """
    state = ViewerState()
    state.start_live_session(
        sampling_rate=10000.0,
        window_size_ms=50.0,
        channel_name="CH0",
        aux_channels=["CH0", "CH1"],
    )

    live_view = OpenEphysLiveView(state=state)
    v_single = SingleTracePGView(state=state)
    v_multi = MultiTracePGView(state=state)
    v_fixed = MultiTraceFixedPGView(state=state)
    v_unit = UnitPGView(state=state)
    v_fp = UnitFootprintPGView(state=state)
    v_table = SpikeGroupTableView(state=state)
    v_track = TrackingView(state=state)

    views = [live_view, v_single, v_multi, v_fixed, v_unit, v_fp, v_table, v_track]
    for v in views:
        v.show()
    qapp.processEvents()

    sr = 10000.0
    n_samples = 600
    t_vec = np.arange(n_samples) / sr
    n_sweeps = 40  # 40 sweeps at 50 Hz

    t0 = time.perf_counter()
    for i in range(n_sweeps):
        ch0 = (np.sin(2 * np.pi * 60.0 * t_vec) * 0.05).astype(np.float32)
        ch1 = (np.cos(2 * np.pi * 60.0 * t_vec) * 0.03).astype(np.float32)
        # Inject primary spike at 12ms
        ch0[118:123] = np.array([-0.2, -0.8, -1.5, -0.6, 0.3], dtype=np.float32)
        ch1[120:125] = np.array([-0.1, -0.5, -0.9, -0.3, 0.2], dtype=np.float32)

        payload = {
            "stim_timestamp_s": i * 0.020,
            "waveform": ch0,
            "pre_offset_s": 0.010,
            "spike_latencies_s": {"unit_0": 0.012, "unit_1": 0.018},
            "aux_sweeps": {"CH0": ch0, "CH1": ch1},
            "sample_rate": sr,
        }
        live_view._on_sweep_ready(payload)

        # Simulate realistic bursty GUI event processing every few sweeps
        if i % 4 == 3:
            qapp.processEvents()

    # Drain any remaining queued sweeps
    live_view._drain_sweep_queue()
    qapp.processEvents()
    elapsed_s = time.perf_counter() - t0

    assert len(state.event_signal) == n_sweeps
    assert state.stimno == n_sweeps - 1
    # Processing 40 multi-channel sweeps across 8 open windows should complete quickly (< 4s)
    assert elapsed_s < 4.0, f"Multi-window 50Hz streaming took too long: {elapsed_s:.2f}s"

    state.stop_live_session()
    for v in views:
        v.close()
