"""Candidate-unit search: detection, review lifecycle, views, and persistence."""

import json
import os

import neo
import numpy as np
import pytest
import quantities as pq
from PySide6.QtCore import Qt

from spikespy.SpikeGroupTable import SpikeGroupTableView
from spikespy.ViewerState import ViewerState
from spikespy.candidate_detection import CandidateScanConfig, detect_candidates
from spikespy.io_manager import load_file, save_state_to_nix


def _make_state(peaks, *, amplitudes=None, n_sweeps=None, n_samples=1000, units=pq.mV):
    state = ViewerState()
    if n_sweeps is None:
        n_sweeps = len(peaks)
    times = np.arange(n_sweeps) * 0.1 * pq.s
    event = neo.Event(times, name="stim", units=pq.s)
    data = np.zeros((n_sweeps, n_samples))
    for i in range(n_sweeps):
        peak = peaks[i] if i < len(peaks) else None
        if peak is None:
            continue
        amp = 1.0 if amplitudes is None else float(amplitudes[i])
        data[i, int(peak)] = amp
    sig = neo.AnalogSignal(
        data.flatten(), units=units, sampling_rate=10000 * pq.Hz, name="CH0"
    )
    seg = neo.Segment("candidate")
    seg.analogsignals.append(sig)
    seg.events.append(event)
    state.segment = seg
    state.window_size = 20 * pq.ms
    state.set_data(analog_signal=sig, event_signal=event, spike_groups=[])
    return state


def _scan(state, scope="offline"):
    assert state.request_candidate_scan(synchronous=True, scope=scope)


# ---------------------------------------------------------------------------
# Pure detector
# ---------------------------------------------------------------------------


def test_detector_finds_recurring_band_once_and_suppresses_accepted_unit():
    erp = np.zeros((20, 300), dtype=np.float32)
    erp[:, 120] = 1.0
    config = CandidateScanConfig(template_refine=False)
    proposals = detect_candidates({"CH0": erp}, 10000.0, 0, 19, config=config)
    assert len(proposals) == 1
    proposal = proposals[0]
    assert proposal.channel == "CH0"
    assert int(round(proposal.latency_center_samples)) == 120
    assert proposal.hits == 20
    assert proposal.samples_by_sweep[7] == 120

    existing = {"CH0": [{i: 120 for i in range(20)}]}
    assert (
        detect_candidates(
            {"CH0": erp}, 10000.0, 0, 19, existing_units=existing, config=config
        )
        == []
    )


def test_detector_can_propose_negative_bands_and_merges_across_channels():
    erp = np.zeros((20, 300), dtype=np.float32)
    erp[:, 150] = -1.0
    inverted = -erp
    config = CandidateScanConfig(template_refine=False)
    proposals = detect_candidates(
        {"CH0": erp, "CH1": inverted}, 10000.0, 0, 19, config=config
    )
    assert len(proposals) == 2
    assert {p.channel for p in proposals} == {"CH0", "CH1"}
    for proposal in proposals:
        assert int(round(proposal.latency_center_samples)) == 150
        assert proposal.polarity_fraction >= 0.9
        assert proposal.neighbor_support >= 0.9


def test_template_refinement_recovers_weak_sweeps_for_a_strong_band():
    rng = np.random.default_rng(7)
    n_sweeps, n_samples = 20, 300
    erp = rng.standard_normal((n_sweeps, n_samples)).astype(np.float32) * 0.01
    for i in range(n_sweeps):
        erp[i, 120] += 1.0 if i % 2 == 0 else 0.02
    base = CandidateScanConfig(template_refine=False, min_peak_snr=4.0)
    without = detect_candidates({"CH0": erp}, 10000.0, 0, n_sweeps - 1, config=base)
    assert without and without[0].hits < n_sweeps

    refined_config = CandidateScanConfig(
        template_refine=True,
        template_corr_threshold=0.3,
        min_peak_snr=4.0,
    )
    refined = detect_candidates(
        {"CH0": erp}, 10000.0, 0, n_sweeps - 1, config=refined_config
    )
    assert refined and refined[0].hits > without[0].hits
    assert refined[0].template_refined
    assert refined[0].hits <= n_sweeps


# ---------------------------------------------------------------------------
# ViewerState lifecycle
# ---------------------------------------------------------------------------


def test_scan_creates_one_pending_candidate_and_accept_matches_preview():
    state = _make_state([120] * 20)
    _scan(state)
    candidates = state.get_visible_candidates()
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.status == "pending"
    assert candidate.hit_count == 20
    assert candidate.name.startswith("candidate_")
    preview_idx = [x[0] for x in candidate.unit.idx_arr if x is not None]
    assert set(preview_idx) == {120}

    state.set_active_candidate(candidate.candidate_id)
    assert state.get_active_candidate() is candidate

    new_idx = state.accept_candidate(candidate.candidate_id)
    assert new_idx == 1
    assert state.candidate_units == {}
    assert state.active_candidate_id is None
    accepted = state.spike_groups[new_idx]
    assert accepted.event.name.startswith("unit_")
    assert accepted.auto_track is True
    assert accepted.channel == "CH0"
    assert set(x[0] for x in accepted.idx_arr if x is not None) == {120}
    assert accepted.event.annotations["type"] == "unit"
    assert accepted.event.annotations["candidate_source"] == "search"

    state.undo()
    assert state.candidate_units
    assert len(state.spike_groups) == 1
    assert state.get_active_candidate().candidate_id == candidate.candidate_id


def test_new_candidate_fingerprint_merges_posterior_detections():
    state = _make_state([120] * 10)
    _scan(state)
    candidate = state.get_visible_candidates()[0]
    candidate_id = candidate.candidate_id

    # Extend the recording with ten more sweeps sharing the same band.
    times = (
        np.concatenate([state.event_signal.times.magnitude, np.arange(10, 20) * 0.1])
        * pq.s
    )
    state.event_signal = neo.Event(times, name="stim", units=pq.s)
    data = np.zeros((20, 1000))
    data[:, 120] = 1.0
    state.analog_signal = neo.AnalogSignal(
        data.flatten(), units=pq.mV, sampling_rate=10000 * pq.Hz, name="CH0"
    )
    state._get_erp.cache_clear()
    state.segment.analogsignals[0] = state.analog_signal
    state.segment.events[0] = state.event_signal
    state.update_idx_arrs()

    _scan(state)
    assert list(state.candidate_units) == [candidate_id]
    assert state.candidate_units[candidate_id].hit_count == 20


def test_discard_hides_candidate_and_scan_does_not_revive_it():
    state = _make_state([120] * 20)
    _scan(state)
    candidate = state.get_visible_candidates()[0]
    state.set_active_candidate(candidate.candidate_id)
    assert state.discard_candidate(candidate.candidate_id)
    assert candidate.status == "discarded"
    assert state.get_visible_candidates() == []
    assert state.active_candidate_id is None

    _scan(state)
    assert candidate.status == "discarded"
    assert state.get_visible_candidates() == []

    state.undo()
    assert candidate.status == "pending"
    assert state.get_visible_candidates() == [candidate]


def test_explicit_rescan_revives_a_discarded_candidate_only_when_stronger():
    state = _make_state([120] * 20)
    _scan(state)
    candidate = state.get_visible_candidates()[0]
    state.discard_candidate(candidate.candidate_id)
    candidate.score_at_discard = 0.99
    _scan(state)
    assert candidate.status == "discarded"

    candidate.score_at_discard = -1.0
    state._candidate_rescan_discarded = True
    _scan(state)
    assert candidate.status == "pending"
    assert state.get_visible_candidates() == [candidate]


def test_active_channel_change_can_reactivate_a_strong_discarded_candidate():
    state = _make_state([120] * 20)
    _scan(state)
    candidate = state.get_visible_candidates()[0]
    state.discard_candidate(candidate.candidate_id)
    state._reactivate_candidates_for_channel("CH0")
    assert candidate.status == "pending"
    assert candidate.candidate_id in [
        c.candidate_id for c in state.get_visible_candidates()
    ]


def test_stale_band_is_stored_dormant_not_visible():
    peaks = [120] * 5 + [None] * 35
    state = _make_state(peaks, n_sweeps=40)
    _scan(state)
    assert state.get_visible_candidates() == []
    dormant = [c for c in state.candidate_units.values() if c.status == "dormant"]
    assert len(dormant) == 1


def test_async_scan_manager_delivers_results(qtbot):
    state = _make_state([120] * 20)
    started = state.request_candidate_scan(synchronous=False, scope="offline")
    assert started
    qtbot.waitUntil(lambda: len(state.get_visible_candidates()) == 1, timeout=5000)
    assert int(round(state.get_visible_candidates()[0].latency_center_samples)) == 120


def test_live_scan_uses_recent_window_and_auto_timer(qtbot):
    state = ViewerState()
    state.start_live_session(
        sampling_rate=10000.0, window_size_ms=20.0, units=pq.mV, channel_name="CH0"
    )
    try:
        for i in range(30):
            wave = np.zeros(200)
            wave[120] = 1000.0
            state.append_sweep(i * 0.1, wave)
        state.candidate_config.live_scan_window_sweeps = 10
        snapshot = state._build_candidate_scan_snapshot("live")
        assert snapshot["scan_start_stim"] == 20
        assert snapshot["scan_end_stim"] == 29

        _scan(state, scope="live")
        assert len(state.get_visible_candidates()) == 1

        state.set_auto_scan_live(True)
        assert state._candidate_scan_manager is not None
        assert state._candidate_scan_manager._timer.isActive()
        state.set_auto_scan_live(False)
        assert not state._candidate_scan_manager._timer.isActive()
    finally:
        state.set_auto_scan_live(False)
        state.stop_live_session()


def test_live_gate_requires_new_sweeps_and_quiet_review():
    state = _make_state([120] * 20)
    state.is_streaming = True
    try:
        state._candidate_last_scan_stim = len(state.event_signal) - 1
        assert not state._candidate_scan_gate("live")
        state._candidate_last_scan_stim = -1
        state._candidate_last_interaction = float("inf")
        assert not state._candidate_scan_gate("live")
        state._candidate_last_interaction = state._candidate_now() - 100.0
        assert state._candidate_scan_gate("live")
    finally:
        state.is_streaming = False


def test_accepted_unit_suppresses_candidate_on_next_scan():
    state = _make_state([120] * 20)
    _scan(state)
    candidate = state.get_visible_candidates()[0]
    state.accept_candidate(candidate.candidate_id)
    _scan(state)
    assert state.get_visible_candidates() == []


# ---------------------------------------------------------------------------
# Table routing
# ---------------------------------------------------------------------------


def test_table_routes_candidate_rows_and_add_after_candidates(qtbot):
    state = _make_state([120] * 20)
    _scan(state)
    view = SpikeGroupTableView(state=state)
    qtbot.addWidget(view)
    model = view.spike_tablemodel
    model.update()

    assert model.rowCount() == 2
    assert model.row_to_item(0) == ("unit", 0)
    item = model.row_to_item(1)
    assert item is not None and item[0] == "candidate"
    candidate_id = item[1]

    assert model.data(model.index(1, 0), Qt.DisplayRole).startswith("candidate_")
    assert "candidate" in model.data(model.index(1, 4), Qt.DisplayRole)
    assert model.data(model.index(1, 3), Qt.DisplayRole) == 20

    view.tbl.selectRow(1)
    assert state.active_candidate_id == candidate_id

    model.setData(model.index(1, 5), "checked by hand", Qt.EditRole)
    candidate = state.candidate_units[candidate_id]
    assert candidate.notes == "checked by hand"
    assert candidate.notes_edited

    view.add_row_and_select()
    assert len(state.spike_groups) == 2
    current = view.spike_tablemodel.row_to_item(view.tbl.currentIndex().row())
    assert current == ("unit", 1)

    view.tbl.selectRow(view.spike_tablemodel.row_for_item(("candidate", candidate_id)))
    assert view._accept_selected_candidate()
    assert len(state.spike_groups) == 3
    assert state.candidate_units == {}


def test_table_edit_candidate_tracking_and_name(qtbot):
    state = _make_state([120] * 20)
    _scan(state)
    view = SpikeGroupTableView(state=state)
    qtbot.addWidget(view)
    model = view.spike_tablemodel
    model.update()
    candidate = state.get_visible_candidates()[0]

    model.setData(model.index(1, 9), 7.5, Qt.EditRole)
    assert candidate.tracking_window_ms == pytest.approx(7.5)
    model.setData(model.index(1, 10), 0.25, Qt.EditRole)
    assert candidate.tracking_threshold_mv == pytest.approx(0.25)
    model.setData(model.index(1, 0), "candidate_alpha", Qt.EditRole)
    assert candidate.name == "candidate_alpha"
    assert candidate.name_edited

    new_idx = state.accept_candidate(candidate.candidate_id)
    assert state.spike_groups[new_idx].event.name == "candidate_alpha"
    assert state.spike_groups[new_idx].tracking_window_ms == pytest.approx(7.5)
    assert state.spike_groups[new_idx].tracking_threshold_mv == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# View overlays
# ---------------------------------------------------------------------------


def test_single_trace_and_fixed_and_multitrace_show_active_candidate(qtbot):
    from spikespy.MultiTraceFixedPGView import MultiTraceFixedPGView
    from spikespy.MultiTracePGView import MultiTracePGView
    from spikespy.SingleTracePGView import SingleTracePGView

    state = _make_state([120] * 20)
    _scan(state)
    candidate = state.get_visible_candidates()[0]

    single = SingleTracePGView(state=state)
    qtbot.addWidget(single)
    single.channel_bar.bind_to_unit(0)
    state.set_active_candidate(candidate.candidate_id)
    single.update_figure(force=True)
    assert single.candidate_line.isVisible()
    assert single.candidate_line.value() == pytest.approx(0.012)
    assert single.candidate_marker.isVisible()

    fixed = MultiTraceFixedPGView(state=state)
    qtbot.addWidget(fixed)
    fixed.update_figure(force=True)
    assert fixed.candidate_scatter.isVisible()
    assert fixed.candidate_band.isVisible()

    multi = MultiTracePGView(state=state)
    qtbot.addWidget(multi)
    multi.plot_spikegroups()
    assert multi.channel_panels[0].candidate_scatter.isVisible()
    assert multi.channel_panels[0].candidate_band.isVisible()

    state.set_active_candidate(None)
    single.update_figure(force=True)
    fixed.update_figure(force=True)
    multi.plot_spikegroups()
    assert not single.candidate_line.isVisible()
    assert not fixed.candidate_scatter.isVisible()
    assert not multi.channel_panels[0].candidate_scatter.isVisible()


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_candidate_store_roundtrips_through_json_and_nix(tmp_path):
    state = _make_state([120] * 20)
    _scan(state)
    candidate = state.get_visible_candidates()[0]
    candidate.notes = "saved note"
    candidate.notes_edited = True
    state.discard_candidate(candidate.candidate_id)

    payload = state.export_candidate_store()
    assert payload["version"] == 1
    assert payload["candidates"][0]["detections"]
    assert json.loads(json.dumps(payload))["candidates"][0]["name"] == candidate.name

    restored = ViewerState()
    restored.event_signal = state.event_signal
    restored.analog_signal = state.analog_signal
    restored.restore_candidate_store(json.dumps(payload))
    restored_candidate = restored.candidate_units[candidate.candidate_id]
    assert restored_candidate.status == "discarded"
    assert restored_candidate.name == candidate.name
    assert restored_candidate.notes == "saved note"
    assert int(round(restored_candidate.latency_center_samples)) == 120
    assert restored_candidate.hit_count == 20

    path = os.path.join(str(tmp_path), "candidates.h5")
    save_state_to_nix(
        path,
        state.spike_groups,
        state.segment,
        candidates=payload,
    )
    data, analog, event, groups = load_file(path, "h5")
    assert "spikespy_candidates" in data.annotations
    assert all(not getattr(g.event, "name", "").startswith("candidate") for g in groups)

    loaded = ViewerState()
    loaded.loadFile(path, "h5")
    assert candidate.name in [c.name for c in loaded.candidate_units.values()]
    loaded_candidate = loaded.candidate_units[candidate.candidate_id]
    assert loaded_candidate.status == "discarded"
    assert int(round(loaded_candidate.latency_center_samples)) == 120
