"""Integration tests for unit tracking, gap filling, and parameter overlays."""

import neo
import numpy as np
import pytest
import quantities as pq
from PySide6.QtCore import Qt

from spikespy.SingleTracePGView import SingleTracePGView
from spikespy.SpikeGroupTable import SpikeGroupTableView
from spikespy.ViewerState import ViewerState, tracked_neuron_unit
from spikespy.live_orchestrator import LiveSweepBatch, LiveSweepItem


def _make_state(peaks, *, units=pq.mV, amplitude=1.0, sr=10000):
    state = ViewerState()
    n_sweeps = len(peaks)
    n_samples = 1000
    times = np.arange(n_sweeps) * 0.1 * pq.s
    event = neo.Event(times, name="stim", units=pq.s)

    data = np.zeros((n_sweeps, n_samples))
    for i, peak in enumerate(peaks):
        if peak is not None:
            data[i, int(peak)] = amplitude
    sig = neo.AnalogSignal(
        data.flatten(), units=units, sampling_rate=sr * pq.Hz, name="CH0"
    )

    seg = neo.Segment("tracking")
    seg.analogsignals.append(sig)
    seg.events.append(event)
    state.segment = seg

    evt = neo.Event(np.array([], dtype=float) * pq.s, name="unit_0", units=pq.s)
    evt.annotations["channel"] = "CH0"
    sg = tracked_neuron_unit(
        event=evt,
        channel="CH0",
        tracking_window_ms=2.0,
        tracking_threshold_mv=0.5,
        tracking_max_skips=1,
    )
    state.set_data(analog_signal=sig, event_signal=event, spike_groups=[sg])
    return state


def test_track_unit_range_fills_gaps_between_existing_anchors():
    state = _make_state([100, 110, 120, 130, 140])
    state.setUnit(100, stimno=0, unit_idx=0)
    state.setUnit(140, stimno=4, unit_idx=0)
    state.update_idx_arrs()

    added = state.track_unit_range(0, only_missing=True)

    assert added == {1: 110, 2: 120, 3: 130}
    assert state.spike_groups[0].idx_arr[4][0] == 140
    assert state.spike_groups[0].idx_arr[2][0] == 120


def test_track_unit_range_stops_after_max_skips():
    state = _make_state([100, None, None, None, None])
    state.setUnit(100, stimno=0, unit_idx=0)
    state.update_idx_arrs()

    added = state.track_unit_range(0, only_missing=True)

    assert added == {}


def test_live_batch_keeps_continuity_across_sweeps():
    state = ViewerState()
    state.start_live_session(
        sampling_rate=10000.0,
        window_size_ms=100.0,
        units=pq.uV,
        channel_name="CH0",
    )
    try:
        waves = []
        for peak in (100, 105, 110):
            w = np.zeros(1000, dtype=np.float32)
            w[peak] = 1000.0
            waves.append(w)
            state.append_sweep(float(len(waves) - 1) * 0.1, w)

        sg = state.spike_groups[0]
        sg.tracking_window_ms = 2.0
        sg.tracking_threshold_mv = 0.5
        sg.tracking_max_skips = 1
        state.setUnit(100, stimno=0, unit_idx=0)
        sg.auto_track = True

        items = [
            LiveSweepItem(
                stim_index=i,
                stim_timestamp_s=i * 0.1,
                channel_sweeps={"CH0": waves[i]},
                sampling_rate=10000.0,
            )
            for i in (1, 2)
        ]
        detections = state._process_live_tracking_batch(LiveSweepBatch(items))

        assert (0, 105, 1) in detections
        assert (0, 110, 2) in detections
    finally:
        state.stop_live_session()


def test_table_tracking_params_emit_and_invalidate_template(qtbot):
    state = _make_state([100, 110])
    state.setUnit(100, stimno=0, unit_idx=0)
    state.spike_groups[0]._template_matcher = object()

    changes = []
    state.onUnitParamsChanged.connect(changes.append)
    view = SpikeGroupTableView(state=state)
    qtbot.addWidget(view)

    model = view.spike_tablemodel
    model.setData(model.index(0, 9), 7.5, Qt.EditRole)

    assert state.spike_groups[0].tracking_window_ms == pytest.approx(7.5)
    assert state.spike_groups[0]._template_matcher is None
    assert 0 in changes


def test_single_trace_tracking_window_and_template_overlay(qtbot):
    state = _make_state([100, 110])
    state.setUnit(100, stimno=0, unit_idx=0)
    view = SingleTracePGView(state=state)
    qtbot.addWidget(view)
    view.channel_bar.bind_to_unit(0)

    view.track_window_spin.setValue(12.0)
    assert state.spike_groups[0].tracking_window_ms == pytest.approx(12.0)
    assert state.spike_groups[0].event.annotations["tracking_window_ms"] == pytest.approx(
        12.0
    )
    assert view.track_window_region.isVisible()

    view._templates = {"CH0": np.array([-1.0, 0.0, 2.0, -0.5], dtype=np.float32)}
    view._update_template_curve(state.spike_groups[0])
    x_data, y_data = view.template_curve.getData()
    assert x_data is not None and len(x_data) == 4
    assert np.min(y_data) >= 0.0 and np.max(y_data) <= 1.0
    assert view.template_curve.isVisible()
