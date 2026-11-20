"""Unit tests for shared visuals, unit palette, tracking health, protocol markers, and unit seeding."""

import neo
import numpy as np
import pytest
import quantities as pq

from spikespy.ViewerState import UNIT_COLORS, ViewerState, tracked_neuron_unit
from spikespy.container_view_state import ContainerViewerState
from spikespy.live_orchestrator import LiveSweepBatch, LiveSweepItem


def _make_multichannel_state(n_sweeps=10, n_samples=500, sr=10000.0, n_channels=6, units=pq.mV):
    state = ViewerState()
    seg = neo.Segment(name="TestSegment")
    times = np.arange(n_sweeps) * 0.2 * pq.s
    event_sig = neo.Event(times, name="stim", units=pq.s)
    seg.events.append(event_sig)

    for ch_idx in range(n_channels):
        arr = np.zeros((n_sweeps, n_samples), dtype=np.float32)
        arr[:, 150] = 0.2 * (ch_idx + 1)
        sig = neo.AnalogSignal(
            arr.flatten(),
            units=units,
            sampling_rate=sr * pq.Hz,
            name=f"CH{ch_idx}",
        )
        seg.analogsignals.append(sig)

    state.segment = seg
    state.set_data(
        analog_signal=seg.analogsignals[0],
        event_signal=event_sig,
        spike_groups=[],
    )
    return state


def test_unit_colors_and_hex_helpers():
    state = ViewerState()
    assert len(UNIT_COLORS) == 8
    assert state.get_unit_color(0) == (102, 194, 165)
    assert state.get_unit_color(1) == (252, 141, 98)
    assert state.get_unit_color(8) == UNIT_COLORS[0]
    assert state.get_unit_color(-1) == (102, 194, 165)
    assert state.get_unit_hex_color(0) == "#66c2a5"
    assert state.get_unit_hex_color(1) == "#fc8d62"
    assert state.get_unit_hex_color(-5) == "#66c2a5"


def test_tracking_health_statuses():
    n_sweeps = 10
    times = np.arange(n_sweeps) * 0.2 * pq.s
    stim = neo.Event(times, name="stim", units=pq.s)

    # 1. Empty
    sg_empty = tracked_neuron_unit(event=neo.Event(np.array([]) * pq.s, units=pq.s))
    h_empty = sg_empty.get_tracking_health(stim, current_stimno=5)
    assert h_empty["status"] == "empty"
    assert h_empty["total_spikes"] == 0
    assert h_empty["recent_window"] == 6
    assert h_empty["hit_rate"] == 0.0
    assert h_empty["last_latency_ms"] is None
    assert h_empty["latency_shift_ms"] == 0.0

    # 2. Manual (auto_track=False, has spikes)
    ev_manual = neo.Event(np.array([times[0].magnitude, times[2].magnitude]) * pq.s + 0.01 * pq.s, units=pq.s)
    sg_manual = tracked_neuron_unit(event=ev_manual, auto_track=False)
    h_manual = sg_manual.get_tracking_health(stim, current_stimno=4)
    assert h_manual["status"] == "manual"
    assert h_manual["total_spikes"] == 2
    assert h_manual["recent_hits"] == 2
    assert h_manual["recent_window"] == 5
    assert h_manual["hit_rate"] == pytest.approx(0.4)
    assert h_manual["consecutive_misses"] == 2
    assert h_manual["last_latency_ms"] == pytest.approx(10.0)
    assert h_manual["latency_shift_ms"] == pytest.approx(0.0)

    # 3. Locked (auto_track=True, consecutive_misses == 0, with latency shift)
    spike_times = np.array([times[0].magnitude + 0.010, times[1].magnitude + 0.012, times[2].magnitude + 0.015]) * pq.s
    sg_locked = tracked_neuron_unit(event=neo.Event(spike_times, units=pq.s), auto_track=True, tracking_max_skips=2)
    h_locked = sg_locked.get_tracking_health(stim, current_stimno=2, window_sweeps=20)
    assert h_locked["status"] == "locked"
    assert h_locked["consecutive_misses"] == 0
    assert sg_locked._consecutive_misses == 0
    assert h_locked["recent_hits"] == 3
    assert h_locked["recent_window"] == 3
    assert h_locked["hit_rate"] == pytest.approx(1.0)
    assert h_locked["last_latency_ms"] == pytest.approx(15.0)
    assert h_locked["latency_shift_ms"] == pytest.approx(5.0)

    # 4. Skipping (1 <= consecutive_misses <= max(1, tracking_max_skips))
    h_skip = sg_locked.get_tracking_health(stim, current_stimno=4)
    assert h_skip["status"] == "skipping"
    assert h_skip["consecutive_misses"] == 2
    assert sg_locked._consecutive_misses == 2

    # 5. Lost (consecutive_misses > max(1, tracking_max_skips))
    h_lost = sg_locked.get_tracking_health(stim, current_stimno=5)
    assert h_lost["status"] == "lost"
    assert h_lost["consecutive_misses"] == 3
    assert sg_locked._consecutive_misses == 3


def test_viewer_state_get_unit_health_and_live_miss_updates():
    state = ViewerState()
    state.start_live_session(sampling_rate=10000.0, window_size_ms=100.0, units=pq.uV, channel_name="CH0")
    try:
        w0 = np.zeros(1000, dtype=np.float32)
        w0[150] = 1000.0
        state.append_sweep(0.0, w0)

        w1 = np.zeros(1000, dtype=np.float32)  # no spike -> miss
        state.append_sweep(0.2, w1)

        sg = state.spike_groups[0]
        sg.tracking_window_ms = 2.0
        sg.tracking_threshold_mv = 0.5
        sg.tracking_max_skips = 1
        state.setUnit(150, stimno=0, unit_idx=0)
        sg.auto_track = True

        batch = LiveSweepBatch([
            LiveSweepItem(
                stim_index=1,
                stim_timestamp_s=0.2,
                channel_sweeps={"CH0": w1},
                sampling_rate=10000.0,
            )
        ])
        detections = state._process_live_tracking_batch(batch)
        assert detections == []
        assert sg._consecutive_misses == 1

        # Calling _on_live_tracking_result with empty detections increments miss; with detection resets to 0
        state._on_live_tracking_result([], None)
        assert sg._consecutive_misses == 2

        w2 = np.zeros(1000, dtype=np.float32)
        w2[150] = 1000.0
        state.append_sweep(0.4, w2)
        state._on_live_tracking_result([(0, 150, 2)], None)
        assert sg._consecutive_misses == 0

        state.stimno = 2
        health = state.get_unit_health(0)
        assert health["status"] == "locked"
        assert health["total_spikes"] == 2
    finally:
        state.stop_live_session()


def test_seed_unit_at_and_reanchor_unit_at():
    state = _make_multichannel_state(n_sweeps=5, n_samples=500, sr=10000.0, n_channels=6, units=pq.mV)
    # Channel CH2 has peak 0.6 mV at sample 150
    params_signals = []
    state.onUnitParamsChanged.connect(params_signals.append)

    new_idx = state.seed_unit_at(sample_idx=150, stimno=1, channel="CH2", auto_track=True, window_ms=7.5)
    assert new_idx == len(state.spike_groups) - 1
    assert state.cur_spike_group == new_idx

    sg = state.spike_groups[new_idx]
    assert sg.channel == "CH2"
    assert len(sg.neighbor_channels) == 3
    assert set(sg.neighbor_channels) == {"CH1", "CH3", "CH0"}
    assert sg.tracking_window_ms == pytest.approx(7.5)
    assert sg.tracking_threshold_mv == pytest.approx(0.75 * 0.6, rel=1e-3)
    assert sg.auto_track is True
    assert sg.idx_arr[1][0] == 150
    assert new_idx in params_signals

    # Tiny signal clamps threshold to >= 0.005 mV
    idx_tiny = state.seed_unit_at(sample_idx=10, stimno=0, channel="CH0", auto_track=False)
    assert state.spike_groups[idx_tiny].tracking_threshold_mv >= 0.005
    assert state.spike_groups[idx_tiny].auto_track is False

    # Reanchor restores auto_track, sets spike, resets _consecutive_misses, and emits onUnitParamsChanged
    sg_re = state.spike_groups[idx_tiny]
    sg_re._consecutive_misses = 4
    params_signals.clear()
    ok = state.reanchor_unit_at(idx_tiny, sample_idx=155, stimno=2, enable_auto_track=True)
    assert ok is True
    assert sg_re.idx_arr[2][0] == 155
    assert sg_re._consecutive_misses == 0
    assert sg_re.auto_track is True
    assert idx_tiny in params_signals
    assert state.reanchor_unit_at(999, sample_idx=10) is False


def test_shared_visual_settings_and_container_visual_settings():
    state = ViewerState()
    defaults = state.get_shared_visual_settings("Shared Group A")
    assert defaults["scale_mode"] == "std"
    assert defaults["color_scale"] == (0.5, 2.0)
    assert defaults["trace_scale"] == 1.0
    assert defaults["show_track_window"] is True
    assert defaults["show_template"] is True
    assert defaults["x_range"] is None
    assert defaults["y_range"] is None

    received = []
    state.onSharedVisualSettingsChanged.connect(lambda name, d: received.append((name, d)))
    updated = state.update_shared_visual_settings(
        "Shared Group A",
        scale_mode="mV",
        color_scale=(-1.0, 1.0),
        trace_scale=2.5,
        show_template=False,
    )
    assert updated["scale_mode"] == "mV"
    assert updated["color_scale"] == (-1.0, 1.0)
    assert updated["trace_scale"] == 2.5
    assert updated["show_template"] is False
    assert state.get_shared_visual_settings("Shared Group A")["scale_mode"] == "mV"
    assert state.get_shared_visual_settings("Shared Group B")["scale_mode"] == "std"
    assert len(received) == 1
    assert received[0][0] == "Shared Group A"
    assert received[0][1]["scale_mode"] == "mV"

    # Preserved across set_shared_channels
    state.set_shared_channels("Shared Group A", ["CH0", "CH1"])
    assert state.get_shared_visual_settings("Shared Group A")["scale_mode"] == "mV"

    # ContainerViewerState independent visual settings
    c_state = ContainerViewerState(state)
    c_defaults = c_state.get_visual_settings()
    assert c_defaults["scale_mode"] == "std"
    c_received = []
    c_state.onContainerVisualSettingsChanged.connect(c_received.append)
    c_updated = c_state.update_visual_settings(scale_mode="mV", x_range=(0.0, 50.0))
    assert c_updated["scale_mode"] == "mV"
    assert c_updated["x_range"] == (0.0, 50.0)
    assert c_state.get_visual_settings()["x_range"] == (0.0, 50.0)
    assert len(c_received) == 1
    assert c_received[0]["x_range"] == (0.0, 50.0)


def test_protocol_markers_and_log_other_sync():
    state = _make_multichannel_state(n_sweeps=5)
    state.stimno = 2

    marker_signals = []
    events_signals = []
    state.onProtocolMarkersChanged.connect(lambda: marker_signals.append(True))
    state.onEventsUpdated.connect(lambda: events_signals.append(True))

    m1 = state.add_protocol_marker("Capsaicin applied")
    assert m1 == {
        "stimno": 2,
        "timestamp_s": pytest.approx(0.4),
        "label": "Capsaicin applied",
        "category": "protocol",
    }
    m2 = state.add_protocol_marker("Washout", stimno=4, category="intervention")
    assert m2["stimno"] == 4
    assert m2["timestamp_s"] == pytest.approx(0.8)
    assert m2["category"] == "intervention"

    assert len(marker_signals) == 2
    assert len(events_signals) == 2

    markers = state.get_protocol_markers()
    assert len(markers) == 2
    assert markers[0]["label"] == "Capsaicin applied"
    assert markers[1]["label"] == "Washout"

    ev_other = next(e for e in state.segment.events if e.name == "log.other")
    assert len(ev_other) == 2
    assert list(ev_other.array_annotations["message"]) == ["Capsaicin applied", "Washout"]

    # New ViewerState sharing the segment populates protocol_markers lazily from log.other
    state2 = ViewerState(segment=state.segment)
    state2.event_signal = state.event_signal
    loaded_markers = state2.get_protocol_markers()
    assert len(loaded_markers) == 2
    assert loaded_markers[0]["label"] == "Capsaicin applied"
    assert loaded_markers[0]["stimno"] == 2
    assert loaded_markers[1]["label"] == "Washout"
    assert loaded_markers[1]["stimno"] == 4
