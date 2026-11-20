import os

import neo
import numpy as np
import pytest
import quantities as pq
from PySide6.QtCore import Qt

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from spikespy.ActiveUnitInfoView import ActiveUnitInfoView
from spikespy.ViewerState import ViewerState, tracked_neuron_unit


@pytest.fixture()
def two_unit_state():
    state = ViewerState()
    sr = 10000.0 * pq.Hz
    n_sweeps = 10
    n_samples = 2500
    stim_times = np.arange(n_sweeps) * 0.25 * pq.s
    event_sig = neo.Event(stim_times, name="stim", units=pq.s)

    seg = neo.Segment("test_seg")
    seg.events.append(event_sig)

    for ch_idx in range(4):
        w = np.zeros((n_sweeps, n_samples), dtype=np.float32)
        if ch_idx == 0:
            # Spike at sample 100..109 (10.0 ms -> 10.9 ms)
            for s in range(n_sweeps):
                w[s, 100 + s] = 2.0
        elif ch_idx == 1:
            for s in range(n_sweeps):
                w[s, 150] = 1.5
        sig = neo.AnalogSignal(
            w.flatten(), units=pq.mV, sampling_rate=sr, name=f"CH{ch_idx}"
        )
        seg.analogsignals.append(sig)

    # Unit 0: has spikes on sweeps 0, 1, 2, 4 (gap at sweep 3)
    u0_times = np.array([
        stim_times[0].magnitude + 0.0100,
        stim_times[1].magnitude + 0.0101,
        stim_times[2].magnitude + 0.0102,
        stim_times[4].magnitude + 0.0104,
    ]) * pq.s
    evt0 = neo.Event(u0_times, name="unit_alpha", units=pq.s)
    evt0.annotations["channel"] = "CH0"
    evt0.annotations["neighbor_channels"] = ["CH1"]
    evt0.annotations["fibre_type"] = "CM"
    evt0.annotations["notes"] = "initial note"
    u0 = tracked_neuron_unit(
        event=evt0,
        channel="CH0",
        neighbor_channels=["CH1"],
        fibre_type="CM",
        notes="initial note",
        tracking_window_ms=5.0,
        tracking_threshold_mv=0.5,
        tracking_max_skips=2,
    )

    # Unit 1: empty unit initially
    evt1 = neo.Event(np.array([], dtype=float) * pq.s, name="unit_beta", units=pq.s)
    evt1.annotations["channel"] = "CH1"
    u1 = tracked_neuron_unit(
        event=evt1,
        channel="CH1",
        neighbor_channels=["CH2"],
        tracking_window_ms=4.0,
        tracking_threshold_mv=0.25,
        tracking_max_skips=1,
    )

    state.segment = seg
    state.set_data(
        analog_signal=seg.analogsignals[0],
        event_signal=event_sig,
        spike_groups=[u0, u1],
    )
    return state


def test_creation_and_binding_to_two_units(qtbot, two_unit_state):
    view = ActiveUnitInfoView(state=two_unit_state)
    qtbot.addWidget(view)

    assert view.get_target_unit_idx() == 0
    assert view.edit_name.text() == "unit_alpha"
    assert view.combo_fibre_type.currentText() == "CM"
    assert view.edit_notes.text() == "initial note"
    assert view.combo_primary_channel.currentData() == "CH0"
    assert "CH1" in view.lbl_neighbors.text()
    assert view.spin_window_ms.value() == pytest.approx(5.0)
    assert view.spin_threshold_mv.value() == pytest.approx(0.5)
    assert view.spin_max_skips.value() == 2
    assert view.lbl_health_badge.text() == "MANUAL"
    assert "N=4" in view.lbl_metrics.text()
    # At stimno=0, ViewerState.get_unit_health reports latency up to sweep 0 (10.0 ms);
    # advancing stimno to sweep 4 updates latest latency to 10.4 ms and shift +0.4 ms.
    assert "Lat: 10.0 ms" in view.lbl_metrics.text()
    two_unit_state.setStimNo(4)
    view.sync_from_unit()
    assert "Lat: 10.4 ms" in view.lbl_metrics.text()
    assert "4/5 (80%)" in view.lbl_metrics.text()

    # Switching active unit updates UI automatically
    two_unit_state.setUnitGroup(1)
    assert view.get_target_unit_idx() == 1
    assert view.edit_name.text() == "unit_beta"
    assert view.combo_primary_channel.currentData() == "CH1"
    assert view.lbl_health_badge.text() == "EMPTY"

    # Pinning via set_target_unit overrides global cur_spike_group
    view.set_target_unit(0)
    assert view.get_target_unit_idx() == 0
    assert view.edit_name.text() == "unit_alpha"


def test_editing_metadata_channels_and_tracking_params(qtbot, two_unit_state):
    view = ActiveUnitInfoView(state=two_unit_state)
    qtbot.addWidget(view)
    sg = two_unit_state.spike_groups[0]

    # Identity edits
    view.edit_name.setText("c_fibre_1")
    view.edit_name.editingFinished.emit()
    view.combo_fibre_type.setCurrentText("CMiHi")
    view.edit_notes.setText("updated note")
    view.edit_notes.editingFinished.emit()

    assert sg.event.name == "c_fibre_1"
    assert sg.fibre_type == "CMiHi"
    assert sg.event.annotations["fibre_type"] == "CMiHi"
    assert sg.notes == "updated note"
    assert sg.event.annotations["notes"] == "updated note"

    # Primary channel change
    ch2_idx = view.combo_primary_channel.findData("CH2")
    assert ch2_idx >= 0
    view.combo_primary_channel.setCurrentIndex(ch2_idx)
    assert sg.channel == "CH2"
    assert sg.event.annotations["channel"] == "CH2"

    # Tracking parameters & algorithm
    view.spin_window_ms.setValue(8.5)
    view.spin_threshold_mv.setValue(0.75)
    view.spin_max_skips.setValue(4)
    view.spin_track_limit.setValue(25)
    z_idx = view.combo_algorithm.findData("zscore")
    assert z_idx >= 0
    view.combo_algorithm.setCurrentIndex(z_idx)
    view.chk_auto_track.setChecked(True)
    assert "z_thresh" in view.extra_param_controls
    view.extra_param_controls["z_thresh"].setValue(4.25)

    assert sg.tracking_window_ms == pytest.approx(8.5)
    assert sg.tracking_threshold_mv == pytest.approx(0.75)
    assert sg.tracking_max_skips == 4
    assert sg.track_limit == 25
    assert sg.tracking_algorithm == "zscore"
    assert sg.auto_track is True
    assert sg.tracking_params["z_thresh"] == pytest.approx(4.25)
    assert sg.event.annotations["tracking_window_ms"] == pytest.approx(8.5)
    assert sg.event.annotations["tracking_threshold_mv"] == pytest.approx(0.75)
    assert sg.event.annotations["tracking_max_skips"] == 4
    assert sg.event.annotations["track_limit"] == 25
    assert sg.event.annotations["tracking_algorithm"] == "zscore"
    assert sg.event.annotations["auto_track"] is True


def test_health_badge_transitions(qtbot, two_unit_state):
    view = ActiveUnitInfoView(state=two_unit_state)
    qtbot.addWidget(view)

    # Unit 1 starts empty and manual
    two_unit_state.setUnitGroup(1)
    assert view.lbl_health_badge.text() == "EMPTY"

    # Unit 0 has spikes and auto_track=False -> MANUAL
    two_unit_state.setUnitGroup(0)
    assert view.lbl_health_badge.text() == "MANUAL"

    sg0 = two_unit_state.spike_groups[0]
    sg0.tracking_max_skips = 2
    sg0._consecutive_misses = 0
    view.chk_auto_track.setChecked(True)
    assert view.lbl_health_badge.text() == "LOCKED"

    # Skipping (1 <= consecutive_misses <= max_skips)
    sg0._consecutive_misses = 2
    two_unit_state.onUnitChange.emit(0)
    assert view.lbl_health_badge.text() == "SKIPPING (2)"

    # Lost (consecutive_misses > max_skips)
    sg0._consecutive_misses = 3
    two_unit_state.onUnitChange.emit(0)
    assert view.lbl_health_badge.text() == "LOST (3 misses)"


def test_quick_actions_and_channel_shift(qtbot, two_unit_state):
    view = ActiveUnitInfoView(state=two_unit_state)
    qtbot.addWidget(view)

    sg0 = two_unit_state.spike_groups[0]
    assert sg0.channel == "CH0"
    assert sg0.neighbor_channels == ["CH1"]

    # Shift +1 -> CH1, neighbors [CH2]
    view.btn_shift_next.click()
    assert sg0.channel == "CH1"
    assert sg0.neighbor_channels == ["CH2"]

    # Shift -1 -> back to CH0, neighbors [CH1]
    view.btn_shift_prev.click()
    assert sg0.channel == "CH0"
    assert sg0.neighbor_channels == ["CH1"]

    # Auto (80%) threshold button
    view.btn_auto_threshold.click()
    assert view.spin_threshold_mv.value() == pytest.approx(2.0 * 0.8, rel=1e-3)

    # Fill Missing fills sweep 3 (between sweeps 2 and 4) and tracks onwards
    sg0.tracking_threshold_mv = 0.5
    sg0.tracking_window_ms = 2.0
    view.sync_from_unit()
    view.btn_fill_missing.click()
    assert sg0.idx_arr[3] is not None
    assert sg0.idx_arr[3][0] == 103
    assert sg0.idx_arr[9] is not None
    assert sg0.idx_arr[9][0] == 109

    # Track Forward starting from stimno=4
    two_unit_state.setStimNo(4)
    view.btn_track_now.click()
    assert sg0.idx_arr[5] is not None
    assert sg0.idx_arr[5][0] == 105

    # Rebuild Template constructs a matcher for unit 0
    sg0._template_matcher = None
    view.btn_rebuild_template.click()
    assert sg0._template_matcher is not None


def test_settings_persistence_and_new_unit(qtbot, two_unit_state):
    view = ActiveUnitInfoView(state=two_unit_state, compact=True)
    qtbot.addWidget(view)

    # Pin to unit 1 via combo_unit
    u1_idx = view.combo_unit.findData(1)
    assert u1_idx >= 0
    view.combo_unit.setCurrentIndex(u1_idx)

    settings = view.get_settings()
    assert settings["pinned_unit_idx"] == 1
    assert settings["compact"] is True

    view2 = ActiveUnitInfoView(state=two_unit_state)
    qtbot.addWidget(view2)
    view2.set_settings(settings)
    assert view2.get_target_unit_idx() == 1
    assert view2.edit_name.text() == "unit_beta"

    # + New Unit button creates a 3rd unit
    initial_count = len(two_unit_state.spike_groups)
    view2.btn_new_unit.click()
    assert len(two_unit_state.spike_groups) == initial_count + 1
    assert view2.get_target_unit_idx() == initial_count
