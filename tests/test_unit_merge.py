import neo
import numpy as np
import pytest
import quantities as pq

from spikespy.SpikeGroupTable import SpikeGroupTableView
from spikespy.ViewerState import ViewerState, tracked_neuron_unit


def _make_state(n_units_notes=("alpha notes", "beta notes")):
    state = ViewerState()
    fs = 10000.0
    n_samples = int(fs * 2.0)
    sig = neo.AnalogSignal(
        np.zeros((n_samples, 1)) * pq.mV,
        t_start=0 * pq.s,
        sampling_rate=fs * pq.Hz,
        name="CH0",
    )
    stim_times = np.arange(0.1, 2.0, 0.2)
    ev = neo.Event(stim_times * pq.s, name="Trig")
    seg = neo.Segment("seg")
    seg.analogsignals.append(sig)
    seg.events.append(ev)
    state.segment = seg

    u0 = neo.Event((stim_times + 0.025) * pq.s, name="unit_0")
    u0.annotations["notes"] = n_units_notes[0]
    u0.annotations["fibre_type"] = "CM"
    u1 = neo.Event((stim_times[:3] + 0.026) * pq.s, name="unit_1")
    u1.annotations["notes"] = n_units_notes[1]

    sg0 = tracked_neuron_unit(event=u0, channel="CH0")
    sg1 = tracked_neuron_unit(event=u1, channel="CH1")
    state.set_data(analog_signal=sig, event_signal=ev, spike_groups=[sg0, sg1])
    return state


def test_merge_combines_events_and_removes_source():
    state = _make_state()
    t0 = state.spike_groups[0].event.rescale(pq.s).magnitude
    t1 = state.spike_groups[1].event.rescale(pq.s).magnitude

    new_idx = state.mergeUnitGroups(0, 1)

    assert new_idx == 0
    assert len(state.spike_groups) == 1
    merged = state.spike_groups[0]
    expect = np.sort(np.unique(np.concatenate([t0, t1])))
    got = merged.event.rescale(pq.s).magnitude
    assert len(got) == len(t0) + len(t1)
    assert np.allclose(got, expect)
    assert merged.event.name == "unit_0"
    assert merged.event.annotations["notes"] == "alpha notes; beta notes"
    assert merged.event.annotations["fibre_type"] == "CM"
    # idx_arr is one entry per sweep (last spike in a sweep wins); the 13
    # merged spikes all land inside the 10 stimulated sweeps
    assert sum(1 for x in merged.idx_arr if x is not None) == 10
    # current unit points at the merged unit
    assert state.getUnitGroup() is merged


def test_merge_fills_blank_metadata_from_removed_unit():
    state = _make_state(n_units_notes=("", "beta notes"))
    state.spike_groups[0].event.annotations["fibre_type"] = ""
    state.spike_groups[1].event.annotations["fibre_type"] = "CMH"

    state.mergeUnitGroups(0, 1)

    merged = state.spike_groups[0]
    assert merged.event.annotations["notes"] == "beta notes"
    assert merged.event.annotations["fibre_type"] == "CMH"


def test_merge_concatenates_different_notes():
    state = _make_state()

    state.mergeUnitGroups(0, 1)

    merged = state.spike_groups[0]
    assert merged.event.annotations["notes"] == "alpha notes; beta notes"


def test_merge_dedupes_identical_times():
    state = _make_state()
    shared = 0.35
    state.spike_groups[0].event = neo.Event(
        np.array([0.125, shared]) * pq.s, name="unit_0"
    )
    state.spike_groups[1].event = neo.Event(np.array([shared]) * pq.s, name="unit_1")

    state.mergeUnitGroups(0, 1)

    got = state.spike_groups[0].event.rescale(pq.s).magnitude
    assert len(got) == 2
    assert np.allclose(np.sort(got), [0.125, shared])


def test_merge_undo_restores_both_units():
    state = _make_state()
    old_groups = list(state.spike_groups)
    old_ev0 = state.spike_groups[0].event
    old_ev1 = state.spike_groups[1].event

    state.mergeUnitGroups(0, 1)
    state.undo()

    assert len(state.spike_groups) == 2
    assert state.spike_groups[0] is old_groups[0]
    assert state.spike_groups[1] is old_groups[1]
    assert state.spike_groups[0].event is old_ev0
    assert state.spike_groups[1].event is old_ev1
    assert len(state.spike_groups[0].event) == len(old_ev0)
    assert len(state.spike_groups[1].event) == len(old_ev1)


def test_merge_shifts_keep_index_when_earlier_unit_removed():
    state = _make_state()

    new_idx = state.mergeUnitGroups(1, 0)

    assert new_idx == 0
    assert len(state.spike_groups) == 1
    assert state.spike_groups[0].event.name == "unit_1"
    assert state.cur_spike_group == 0


def test_merge_rejects_invalid_indices():
    state = _make_state()
    n_before = len(state.spike_groups)

    assert state.mergeUnitGroups(0, 0) == -1
    assert state.mergeUnitGroups(0, 99) == -1
    assert state.mergeUnitGroups(-1, 0) == -1
    assert len(state.spike_groups) == n_before


def test_merge_updates_segment_events():
    state = _make_state()
    keep_ev = state.spike_groups[0].event
    rm_ev = state.spike_groups[1].event
    state.segment.events.extend([keep_ev, rm_ev])

    state.mergeUnitGroups(0, 1)

    seg_events = [
        x for x in state.segment.events if getattr(x, "name", None) == "unit_0"
    ]
    assert len(seg_events) == 1
    assert seg_events[0] is state.spike_groups[0].event
    assert all(x is not rm_ev for x in state.segment.events)


def test_table_merge_helpers(qtbot):
    state = _make_state()
    u2 = tracked_neuron_unit(
        event=neo.Event(np.array([0.5, 0.7]) * pq.s, name="unit_2"), channel="CH0"
    )
    state.spike_groups.append(u2)

    view = SpikeGroupTableView(state=state)
    qtbot.addWidget(view)

    targets = view._merge_targets(1)
    assert [t[0] for t in targets] == [0, 2]
    assert "unit_0" in targets[0][1] and "unit_2" in targets[1][1]
    assert view._merge_targets(0)[0][0] == 1
    assert not view._merge_targets(0)[0][0] == 0

    view._merge_units(0, 2)
    assert len(state.spike_groups) == 2
    assert state.spike_groups[0].event.name == "unit_0"
    assert len(state.spike_groups[0].event) == 12
