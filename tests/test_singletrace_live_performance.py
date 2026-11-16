"""Performance and stability guards for SingleTracePGView during live streaming.

These tests reproduce the demo scenario that used to make the UI unresponsive
(and eventually crash) when a Single Trace view was opened on top of a long live
recording: the old implementation built template convolution for every sweep in
the session and called ``MultiAnalogSignal.time_slice`` once per spike.
"""

from __future__ import annotations

import logging
import time

import neo
import numpy as np
import pytest
import quantities as pq
from PySide6.QtWidgets import QApplication

from spikespy.processing import MultiAnalogSignal
from spikespy.SingleTracePGView import SingleTracePGView
from spikespy.ViewerState import ViewerState, tracked_neuron_unit
from spikespy.viewer_worker import (
    TaskPriority,
    TaskQueue,
    TaskType,
    ViewerTask,
    ViewerWorker,
)

SAMPLING_RATE = 50000.0
WINDOW_MS = 20.0
WINDOW_SAMPLES = int(SAMPLING_RATE * WINDOW_MS / 1000.0)
CHANNELS = [f"CH{i}" for i in range(6)]
HISTORY_SWEEPS = 2000


@pytest.fixture(scope="module")
def long_live_state():
    """A live session already holding thousands of sweeps, like demo mode."""
    state = ViewerState()
    state.live_frame_interval_ms = 0
    state.start_live_session(
        sampling_rate=SAMPLING_RATE,
        window_size_ms=WINDOW_MS,
        channel_name=CHANNELS[0],
        units=pq.uV,
        aux_channels=CHANNELS,
    )
    state.spike_groups = [
        tracked_neuron_unit(
            event=neo.Event(times=[] * pq.s, name="unit_0"),
            channel="CH0",
            idx_arr=[],
        )
    ]
    state.cur_spike_group = 0

    rng = np.random.default_rng(0)
    for i in range(HISTORY_SWEEPS):
        wave = (rng.standard_normal(WINDOW_SAMPLES) * 10.0).astype(np.float32)
        aux = {
            ch: (rng.standard_normal(WINDOW_SAMPLES) * 8.0).astype(np.float32)
            for ch in CHANNELS[1:]
        }
        state.append_sweep(
            stim_timestamp=i * 0.02,
            sweep_waveform=wave,
            spike_latencies_s={0: 0.012},
            aux_sweeps=aux,
        )
    yield state
    state.stop_live_session()


def _show(view, qtbot):
    qtbot.addWidget(view)
    view.show()
    qtbot.waitExposed(view)
    QApplication.processEvents()


def _process_offline(payload, viewer_id):
    """Run the worker method without touching the live worker thread."""
    worker = ViewerWorker(TaskQueue())
    task = ViewerTask(
        task_type=TaskType.SINGLE_TRACE_RENDER,
        priority=TaskPriority.HIGH,
        viewer_id=viewer_id,
        payload=payload,
    )
    return worker._process_single_trace(task)


def test_singletrace_show_on_long_history_does_not_walk_full_history(
    qtbot, monkeypatch, long_live_state
):
    """Opening Single Trace must not re-slice/convert every historical sweep."""
    calls = {"count": 0}
    original = MultiAnalogSignal.time_slice

    def counting_time_slice(self, start, stop):
        calls["count"] += 1
        return original(self, start, stop)

    monkeypatch.setattr(MultiAnalogSignal, "time_slice", counting_time_slice)

    started = time.monotonic()
    view = SingleTracePGView(state=long_live_state)
    _show(view, qtbot)
    qtbot.wait(150)
    elapsed = time.monotonic() - started

    assert calls["count"] == 0, (
        f"SingleTrace re-sliced the live signal {calls['count']} times; "
        "template convolution must only use the current sweep"
    )
    assert (
        elapsed < 5.0
    ), f"Showing SingleTrace took {elapsed:.2f}s on {HISTORY_SWEEPS} sweeps"


def test_singletrace_render_payload_is_one_sweep(qtbot, monkeypatch, long_live_state):
    """The background render payload must carry only the active sweep."""
    view = SingleTracePGView(state=long_live_state)
    _show(view, qtbot)

    captured = {}
    manager = view._task_manager

    def capture(viewer_id, task_type, payload, priority=TaskPriority.NORMAL):
        captured.setdefault(task_type, payload)
        return -1

    monkeypatch.setattr(manager, "submit_render", capture)
    view._schedule_render(long_live_state.stimno)

    payload = captured[TaskType.SINGLE_TRACE_RENDER]
    primary = np.asarray(payload["primary_erp"])
    assert primary.ndim == 1
    assert primary.size == WINDOW_SAMPLES
    for overlay in payload["overlay_erps"].values():
        assert np.asarray(overlay).ndim == 1


def test_singletrace_worker_result_is_a_single_trace(qtbot, long_live_state):
    """The worker must return a 1-D decimated trace, not the whole ERP matrix."""
    view = SingleTracePGView(state=long_live_state)
    _show(view, qtbot)

    payload = {
        "primary_erp": long_live_state.get_erp()[long_live_state.stimno],
        "overlay_erps": {},
        "spike_group": view.get_target_unit(),
        "event_signal": long_live_state.event_signal,
        "sampling_rate": long_live_state.sampling_rate,
        "stimno": long_live_state.stimno,
        "window_size": long_live_state.window_size,
        "channel_spacing": 50.0,
        "conv_channels": ["CH0"],
        "templates": {},
    }
    result = _process_offline(payload, view._viewer_id)

    assert result["primary_trace"]["y"].ndim == 1
    assert result["primary_trace"]["y"].size <= WINDOW_SAMPLES


def test_singletrace_convolution_cache_is_per_sweep_and_bounded(qtbot, long_live_state):
    """Convolution is computed for the visible sweep only and cached in small numbers."""
    view = SingleTracePGView(state=long_live_state)
    _show(view, qtbot)

    stimno = long_live_state.stimno
    conv_row = view.conv[stimno]

    assert conv_row is not None
    assert conv_row.shape == (WINDOW_SAMPLES,)
    assert len(view.conv._cache) <= 32
    assert len(view.conv._cache) < HISTORY_SWEEPS


def test_singletrace_histogram_step_payload_is_valid(qtbot, long_live_state):
    """Worker histogram payload (bin edges) must plot without raising."""
    view = SingleTracePGView(state=long_live_state)
    _show(view, qtbot)

    bins = np.array([0.0, 0.002, 0.004, 0.006, 0.008])
    counts = np.array([1.0, 3.0, 2.0, 1.0])
    view._on_histogram_result({"bins": bins, "counts": counts}, "")

    assert view.histogram_curve.xData.size == view.histogram_curve.yData.size + 1

    # Older payloads that returned right edges are also accepted.
    view._on_histogram_result({"bins": bins[1:], "counts": counts}, "")


def test_singletrace_convolution_is_computed_in_worker(qtbot, long_live_state):
    """Template convolution for the displayed sweep must happen off the UI thread."""
    view = SingleTracePGView(state=long_live_state)
    _show(view, qtbot)

    # Let the worker produce templates, then build the same payload the view
    # submits during a live refresh.
    qtbot.waitUntil(lambda: bool(view._templates), timeout=3000)
    sr = view._sampling_rate_hz()
    erp = long_live_state.get_erp()
    sweep = erp[long_live_state.stimno]
    channels = view._get_conv_channels()
    payload = {
        "primary_erp": sweep,
        "overlay_erps": {},
        "conv_sweeps": {ch: sweep for ch in channels},
        "spike_group": view.get_target_unit(),
        "event_signal": long_live_state.event_signal,
        "sampling_rate": sr,
        "stimno": long_live_state.stimno,
        "window_size": long_live_state.window_size,
        "channel_spacing": 50.0,
        "conv_channels": channels,
        "templates": view._templates,
    }
    result = _process_offline(payload, view._viewer_id)

    conv = result["convolution"]
    assert conv is not None
    assert conv["conv"].shape == (WINDOW_SAMPLES,)
    assert len(conv["peaks"]) <= 100
    assert result["cur_latency"] is not None
    assert result["prev_latency"] is None or isinstance(result["prev_latency"], float)
    assert result["next_latency"] is None or isinstance(result["next_latency"], float)


def test_live_stimno_change_does_not_run_plot_updates_on_gui_thread(
    qtbot, monkeypatch, long_live_state
):
    """At live refresh rates onStimNoChange must not re-render the whole trace."""
    view = SingleTracePGView(state=long_live_state)
    _show(view, qtbot)
    qtbot.wait(100)

    update_calls = {"count": 0}
    conv_calls = {"count": 0}
    original_update = view.update_figure
    original_conv = view._compute_sweep_conv

    def counting_update(*args, **kwargs):
        update_calls["count"] += 1
        return original_update(*args, **kwargs)

    def counting_conv(*args, **kwargs):
        conv_calls["count"] += 1
        return original_conv(*args, **kwargs)

    monkeypatch.setattr(view, "update_figure", counting_update)
    monkeypatch.setattr(view, "_compute_sweep_conv", counting_conv)

    rng = np.random.default_rng(2)
    for i in range(HISTORY_SWEEPS, HISTORY_SWEEPS + 16):
        wave = (rng.standard_normal(WINDOW_SAMPLES) * 10.0).astype(np.float32)
        aux = {
            ch: (rng.standard_normal(WINDOW_SAMPLES) * 8.0).astype(np.float32)
            for ch in CHANNELS[1:]
        }
        long_live_state.append_sweep(
            stim_timestamp=i * 0.02,
            sweep_waveform=wave,
            spike_latencies_s={0: 0.012},
            aux_sweeps=aux,
        )
        QApplication.processEvents()

    assert update_calls["count"] == 0, (
        "onStimNoChange triggered synchronous update_figure calls during live streaming"
    )
    assert conv_calls["count"] == 0, (
        "template convolution ran on the UI thread during live streaming"
    )


def test_live_refresh_reuses_graphics_items(qtbot, long_live_state):
    """Live refreshes must not create/remove scene items every frame."""
    view = SingleTracePGView(state=long_live_state)
    _show(view, qtbot)

    # Warm up so overlay/conv items exist before measuring.
    long_live_state.flush_live_frame()
    QApplication.processEvents()
    qtbot.wait(200)
    QApplication.processEvents()

    main_items = len(view.main_plot.items)
    top_items = len(view.top_plot.items)

    rng = np.random.default_rng(3)
    for i in range(HISTORY_SWEEPS + 20, HISTORY_SWEEPS + 50):
        wave = (rng.standard_normal(WINDOW_SAMPLES) * 10.0).astype(np.float32)
        aux = {
            ch: (rng.standard_normal(WINDOW_SAMPLES) * 8.0).astype(np.float32)
            for ch in CHANNELS[1:]
        }
        long_live_state.append_sweep(
            stim_timestamp=i * 0.02,
            sweep_waveform=wave,
            spike_latencies_s={0: 0.012},
            aux_sweeps=aux,
        )
        QApplication.processEvents()

    qtbot.wait(300)
    QApplication.processEvents()

    assert len(view.main_plot.items) == main_items
    assert len(view.top_plot.items) == top_items


def test_live_signal_handling_stays_responsive_at_16hz(qtbot, long_live_state):
    """80 live frames (5s at 16Hz) must be handled quickly by the GUI thread."""
    from spikespy.live_orchestrator import LiveSweepItem

    view = SingleTracePGView(state=long_live_state)
    _show(view, qtbot)
    qtbot.wait(100)

    started = time.perf_counter()
    for i in range(80):
        long_live_state.onStimNoChange.emit(long_live_state.stimno)
        # New sweeps are delivered to views through the live orchestrator.
        long_live_state.live_orchestrator.submit(
            LiveSweepItem(
                stim_index=long_live_state.stimno,
                stim_timestamp_s=i * 0.0625,
                channel_sweeps={"CH0": np.zeros(WINDOW_SAMPLES, dtype=np.float32)},
                sampling_rate=SAMPLING_RATE,
            ),
            synchronous=True,
        )
        QApplication.processEvents()
    elapsed = time.perf_counter() - started

    assert elapsed < 1.5, (
        f"80 live refresh notifications took {elapsed:.2f}s on the UI thread"
    )


def test_streaming_with_multitrace_and_singletrace_open(qtbot, long_live_state):
    """Adding Single Trace after a Multi Trace during streaming must stay stable."""
    from spikespy.MultiTracePGView import MultiTracePGView

    records = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = _Capture()
    worker_logger = logging.getLogger("spikespy.viewer_worker")
    worker_logger.addHandler(handler)

    try:
        multi = MultiTracePGView(state=long_live_state)
        _show(multi, qtbot)
        qtbot.wait(100)

        single = SingleTracePGView(state=long_live_state)
        _show(single, qtbot)

        rng = np.random.default_rng(1)
        for i in range(HISTORY_SWEEPS, HISTORY_SWEEPS + 50):
            wave = (rng.standard_normal(WINDOW_SAMPLES) * 10.0).astype(np.float32)
            aux = {
                ch: (rng.standard_normal(WINDOW_SAMPLES) * 8.0).astype(np.float32)
                for ch in CHANNELS[1:]
            }
            long_live_state.append_sweep(
                stim_timestamp=i * 0.02,
                sweep_waveform=wave,
                spike_latencies_s={0: 0.012},
                aux_sweeps=aux,
            )
            QApplication.processEvents()

        qtbot.wait(300)
        QApplication.processEvents()
    finally:
        worker_logger.removeHandler(handler)

    assert multi.isVisible()
    assert single.isVisible()
    errors = [r.getMessage() for r in records if r.levelno >= logging.ERROR]
    assert not errors, f"Viewer worker errors while streaming: {errors}"
