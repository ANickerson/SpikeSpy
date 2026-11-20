import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import neo
import numpy as np
import pytest
import quantities as pq
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QMenu

from spikespy.SpikeGroupTable import SET2_COLORS, SpikeGroupTableModel, SpikeGroupTableView
from spikespy.ViewerState import ViewerState, tracked_neuron_unit


def _make_state_with_units(n_sweeps: int = 5, n_units: int = 2) -> ViewerState:
    state = ViewerState()
    sr = 10000.0 * pq.Hz
    n_samples = 1000
    stim_times = np.arange(n_sweeps) * 0.1 * pq.s
    event_sig = neo.Event(stim_times, name="stim", units=pq.s)

    seg = neo.Segment("live_ux_test")
    seg.events.append(event_sig)

    w = np.zeros((n_sweeps, n_samples), dtype=np.float32)
    for s in range(n_sweeps):
        w[s, 120] = 2.0
        w[s, 150] = 1.5
    sig = neo.AnalogSignal(w.flatten(), units=pq.mV, sampling_rate=sr, name="CH0")
    seg.analogsignals.append(sig)

    groups = []
    for i in range(n_units):
        lat_s = 0.012 if i == 0 else 0.015
        times = np.array([stim_times[0].magnitude + lat_s]) * pq.s
        evt = neo.Event(times, name=f"unit_{i}", units=pq.s)
        evt.annotations["channel"] = "CH0"
        evt.annotations["neighbor_channels"] = ["CH1"] if i == 0 else []
        sg = tracked_neuron_unit(
            event=evt,
            channel="CH0",
            neighbor_channels=["CH1"] if i == 0 else [],
            tracking_window_ms=5.0,
            tracking_threshold_mv=0.5,
            tracking_max_skips=2,
        )
        groups.append(sg)

    state.segment = seg
    state.set_data(analog_signal=sig, event_signal=event_sig, spike_groups=groups)
    return state


def _make_candidate_state() -> ViewerState:
    state = ViewerState()
    sr = 10000.0 * pq.Hz
    n_sweeps = 12
    n_samples = 1000
    stim_times = np.arange(n_sweeps) * 0.1 * pq.s
    event_sig = neo.Event(stim_times, name="stim", units=pq.s)

    rng = np.random.default_rng(7)
    w = rng.normal(0.0, 0.02, size=(n_sweeps, n_samples)).astype(np.float32)
    for s in range(n_sweeps):
        w[s, 120] = 1.5
        w[s, 119] = 0.5
        w[s, 121] = -0.4
    sig = neo.AnalogSignal(w.flatten(), units=pq.mV, sampling_rate=sr, name="CH0")

    seg = neo.Segment("cand_seg")
    seg.events.append(event_sig)
    seg.analogsignals.append(sig)
    state.segment = seg

    evt0 = neo.Event(np.array([], dtype=float) * pq.s, name="unit_0", units=pq.s)
    sg0 = tracked_neuron_unit(event=evt0, channel="CH0")
    state.set_data(analog_signal=sig, event_signal=event_sig, spike_groups=[sg0])
    state.request_candidate_scan(synchronous=True, scope="offline")
    return state


def test_unit_rows_display_qualitative_color_swatch(qtbot):
    state = _make_state_with_units(n_units=10)
    view = SpikeGroupTableView(state=state)
    qtbot.addWidget(view)
    model = view.spike_tablemodel
    model.update()

    for idx in range(10):
        dec = model.data(model.index(idx, 0), Qt.DecorationRole)
        assert isinstance(dec, QColor)
        expected_rgb = SET2_COLORS[idx % len(SET2_COLORS)]
        assert (dec.red(), dec.green(), dec.blue()) == expected_rgb

    # Column 1 should not return a swatch
    assert model.data(model.index(0, 1), Qt.DecorationRole) is None


def test_tracking_health_status_badges_colors_and_tooltip(qtbot):
    state = _make_state_with_units(n_units=2)
    view = SpikeGroupTableView(state=state)
    qtbot.addWidget(view)
    model = view.spike_tablemodel

    sg0 = state.spike_groups[0]
    sg0.tracking_max_skips = 2
    idx_track = model.index(0, 8)

    # auto_track=False -> empty display & no foreground override
    sg0.auto_track = False
    assert model.data(idx_track, Qt.DisplayRole) == ""
    assert model.data(idx_track, Qt.ForegroundRole) is None
    assert model.data(idx_track, Qt.CheckStateRole) == Qt.Unchecked

    # auto_track=True, misses=0 -> Locked (green)
    sg0.auto_track = True
    sg0._consecutive_misses = 0
    assert "Locked" in model.data(idx_track, Qt.DisplayRole)
    assert model.data(idx_track, Qt.ForegroundRole) == QColor("#2b7a2b")
    assert model.data(idx_track, Qt.CheckStateRole) == Qt.Checked

    # 1 <= misses <= max_skips -> Skip (amber)
    sg0._consecutive_misses = 1
    assert "Skip" in model.data(idx_track, Qt.DisplayRole)
    assert model.data(idx_track, Qt.ForegroundRole) == QColor("#b86b11")

    sg0._consecutive_misses = 2
    assert "Skip" in model.data(idx_track, Qt.DisplayRole)
    assert model.data(idx_track, Qt.ForegroundRole) == QColor("#b86b11")

    # misses > max_skips -> Lost (red)
    sg0._consecutive_misses = 3
    assert "Lost" in model.data(idx_track, Qt.DisplayRole)
    assert model.data(idx_track, Qt.ForegroundRole) == QColor("#c9302c")

    # Rich tooltip on unit row
    tip = model.data(model.index(0, 0), Qt.ToolTipRole)
    assert "unit_0" in tip
    assert "CH0" in tip
    assert "CH1" in tip
    assert "total spikes: 1" in tip
    assert "auto-track: ON" in tip
    assert "consecutive misses: 3" in tip
    assert "5.00 ms" in tip
    assert "0.5 mV" in tip


def test_tracking_health_uses_sg_get_tracking_health_hook():
    state = _make_state_with_units(n_units=1)
    sg = state.spike_groups[0]
    sg.auto_track = True

    class _CustomUnit:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def get_tracking_health(self):
            return {
                "status": "lost",
                "auto_track": True,
                "consecutive_misses": 5,
                "max_skips": 1,
            }

    model = SpikeGroupTableModel(lambda: [_CustomUnit(sg)], state=state)
    idx_track = model.index(0, 8)
    assert "Lost" in model.data(idx_track, Qt.DisplayRole)
    assert model.data(idx_track, Qt.ForegroundRole) == QColor("#c9302c")


def test_candidate_triage_buttons_visibility_accept_and_discard(qtbot):
    # Without candidates, candidate triage buttons are hidden/disabled
    plain_state = _make_state_with_units(n_units=1)
    plain_view = SpikeGroupTableView(state=plain_state)
    qtbot.addWidget(plain_view)
    plain_view.show()
    assert not plain_view.acceptCandidateButton.isVisible()
    assert not plain_view.discardCandidateButton.isVisible()
    assert not plain_view.acceptCandidateButton.isEnabled()
    assert not plain_view.discardCandidateButton.isEnabled()

    # With a visible candidate
    state = _make_candidate_state()
    candidates = state.get_visible_candidates()
    assert len(candidates) >= 1
    view = SpikeGroupTableView(state=state)
    qtbot.addWidget(view)
    view.show()
    model = view.spike_tablemodel
    model.update()

    # Candidate exists, unit row selected -> buttons visible (because candidates exist) but disabled
    unit_row = model.row_for_item(("unit", 0))
    view.tbl.selectRow(unit_row)
    assert view.acceptCandidateButton.isVisible()
    assert view.discardCandidateButton.isVisible()
    assert not view.acceptCandidateButton.isEnabled()
    assert not view.discardCandidateButton.isEnabled()

    # Select candidate row -> buttons visible and enabled
    cand_id = int(candidates[0].candidate_id)
    cand_row = model.row_for_item(("candidate", cand_id))
    assert cand_row is not None
    view.tbl.selectRow(cand_row)
    assert view.acceptCandidateButton.isVisible()
    assert view.discardCandidateButton.isVisible()
    assert view.acceptCandidateButton.isEnabled()
    assert view.discardCandidateButton.isEnabled()

    # Click Accept Candidate -> converts candidate into a real unit
    n_before = len(state.spike_groups)
    n_cands_before = len(state.get_visible_candidates())
    view.acceptCandidateButton.click()
    assert len(state.spike_groups) == n_before + 1
    assert len(state.get_visible_candidates()) == n_cands_before - 1
    # Accept remaining candidates so triage buttons hide once no candidates remain
    while state.get_visible_candidates():
        rem_id = int(state.get_visible_candidates()[0].candidate_id)
        rem_row = view.spike_tablemodel.row_for_item(("candidate", rem_id))
        view.tbl.selectRow(rem_row)
        view.acceptCandidateButton.click()
    assert state.get_visible_candidates() == []
    assert not view.acceptCandidateButton.isVisible()
    assert not view.discardCandidateButton.isVisible()

    # Test Discard Candidate on another candidate state
    state2 = _make_candidate_state()
    cands2 = state2.get_visible_candidates()
    assert len(cands2) >= 1
    view2 = SpikeGroupTableView(state=state2)
    qtbot.addWidget(view2)
    view2.show()
    cand2_id = int(cands2[0].candidate_id)
    cand2_row = view2.spike_tablemodel.row_for_item(("candidate", cand2_id))
    view2.tbl.selectRow(cand2_row)
    n_before2 = len(state2.spike_groups)
    n_cands2_before = len(cands2)
    view2.discardCandidateButton.click()
    assert len(state2.spike_groups) == n_before2
    assert state2.candidate_units[cand2_id].status == "discarded"
    assert len(state2.get_visible_candidates()) == n_cands2_before - 1
    while state2.get_visible_candidates():
        rem_id = int(state2.get_visible_candidates()[0].candidate_id)
        rem_row = view2.spike_tablemodel.row_for_item(("candidate", rem_id))
        view2.tbl.selectRow(rem_row)
        view2.discardCandidateButton.click()
    assert state2.get_visible_candidates() == []
    assert not view2.acceptCandidateButton.isVisible()
    assert not view2.discardCandidateButton.isVisible()


def test_context_menu_reanchor_and_open_active_unit_info(qtbot, monkeypatch):
    state = _make_state_with_units(n_sweeps=5, n_units=1)
    # Put a prominent peak at sweep 2, sample 250
    w = np.zeros((5, 1000), dtype=np.float32)
    for s in range(5):
        w[s, 120] = 1.0
    w[2, 250] = 5.0
    sig = neo.AnalogSignal(
        w.flatten(), units=pq.mV, sampling_rate=10000.0 * pq.Hz, name="CH0"
    )
    state.segment.analogsignals[0] = sig
    state.set_data(
        analog_signal=sig,
        event_signal=state.event_signal,
        spike_groups=state.spike_groups,
    )
    state.spike_groups[0]._consecutive_misses = 4
    state.setStimNo(2)

    class _FakeTopWindow:
        def __init__(self):
            self.opened = []

        def newWindow(self, key, **kwargs):
            self.opened.append((key, kwargs))

    view = SpikeGroupTableView(state=state)
    qtbot.addWidget(view)
    fake_top = _FakeTopWindow()
    monkeypatch.setattr(view, "window", lambda: fake_top)

    captured_actions = []

    class _FakeMenu(QMenu):
        def exec(self, *args, **kwargs):
            for act in self.actions():
                captured_actions.append(act.text())
                if act.menu() is not None:
                    for sub in act.menu().actions():
                        captured_actions.append(sub.text())
            return None

        exec_ = exec

    import spikespy.SpikeGroupTable as sgt_mod

    monkeypatch.setattr(sgt_mod, "QMenu", _FakeMenu)
    rect = view.tbl.visualRect(view.spike_tablemodel.index(0, 0))
    view.on_table_context_menu(rect.center())

    assert "Re-Anchor Active Unit at Current Sweep" in captured_actions
    assert "Open Active Unit Info" in captured_actions

    # Trigger re-anchor directly and verify it resets misses and sets spike on sweep 2
    sample = view._reanchor_unit_at_current_sweep(0)
    assert sample is not None
    assert state.spike_groups[0]._consecutive_misses == 0
    assert state.spike_groups[0].idx_arr[2] is not None

    # Trigger Open Active Unit Info
    view._open_active_unit_info(0)
    assert ("ActiveUnitInfo", {}) in fake_top.opened
