import os

import neo
import numpy as np
import pytest
import quantities as pq
from PySide6.QtWidgets import QApplication

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from spikespy.ActiveUnitInfoView import ActiveUnitInfoView
from spikespy.ChannelGroupBar import ChannelGroupBar
from spikespy.SpikeGroupTable import SpikeGroupTableView
from spikespy.SpikeSpy import MdiView, _has_pg, window_options
from spikespy.ViewerState import ViewerState, tracked_neuron_unit
from spikespy.open_ephys_addin.oe_live_view import OpenEphysLiveView


@pytest.fixture
def cockpit_state():
    state = ViewerState()
    seg = neo.Segment(name="CockpitSeg")
    sr = 10000.0 * pq.Hz
    n_sweeps = 6
    n_samples = 500
    stim_times = np.arange(n_sweeps) * 0.2 * pq.s
    event_sig = neo.Event(stim_times, name="stim", units=pq.s)
    seg.events.append(event_sig)

    for ch in range(4):
        w = np.zeros((n_sweeps, n_samples), dtype=np.float32)
        w[:, 120] = 2.0 / (ch + 1)
        sig = neo.AnalogSignal(w.flatten(), units=pq.mV, sampling_rate=sr, name=f"CH{ch}")
        seg.analogsignals.append(sig)

    u0_times = np.array([stim_times[i].magnitude + 0.012 for i in range(n_sweeps)]) * pq.s
    evt0 = neo.Event(u0_times, name="unit_0", units=pq.s)
    evt0.annotations["channel"] = "CH0"
    evt0.annotations["neighbor_channels"] = ["CH1"]
    u0 = tracked_neuron_unit(event=evt0, channel="CH0", neighbor_channels=["CH1"])

    state.segment = seg
    state.set_data(
        analog_signal=seg.analogsignals[0],
        event_signal=event_sig,
        spike_groups=[u0],
    )
    return state


def test_active_unit_info_registered_and_singleton_open(qtbot, cockpit_state):
    assert "ActiveUnitInfo" in window_options
    assert window_options["ActiveUnitInfo"] is ActiveUnitInfoView

    mdi = MdiView(state=cockpit_state, check_updates=False)
    qtbot.addWidget(mdi)

    # Toolbar action exists
    assert hasattr(mdi, "act_unit_info")

    # Opening via convenience method opens a dock and returns ActiveUnitInfoView
    w1 = mdi.open_or_show_active_unit_info()
    assert isinstance(w1, ActiveUnitInfoView)

    # Calling again reuses the existing dock rather than duplicating
    w2 = mdi.open_or_show_active_unit_info()
    assert w2 is w1
    info_docks = [
        dw for dw in mdi.dock_manager.dockWidgets() if isinstance(dw.widget(), ActiveUnitInfoView)
    ]
    assert len(info_docks) == 1

    # Closing and re-opening still reuses or opens cleanly without duplicating
    info_docks[0].closeDockWidget()
    w3 = mdi.open_or_show_active_unit_info()
    assert isinstance(w3, ActiveUnitInfoView)


def test_live_cockpit_pill_and_follow_live_updates(qtbot, cockpit_state):
    mdi = MdiView(state=cockpit_state, check_updates=False)
    qtbot.addWidget(mdi)

    assert hasattr(mdi, "lbl_live_status_pill")
    assert hasattr(mdi, "btn_follow_live_pill")
    assert hasattr(mdi, "btn_add_marker")

    # Initial offline state with loaded sweeps (follow_live=True)
    assert "Offline" in mdi.lbl_live_status_pill.text()
    assert mdi.btn_follow_live_pill.isChecked() is True
    assert "Following Live" in mdi.btn_follow_live_pill.text()

    # Streaming active + following live -> green LIVE pill
    cockpit_state.is_streaming = True
    cockpit_state.onStreamStateChanged.emit(True)
    assert "LIVE" in mdi.lbl_live_status_pill.text()
    assert f"Sweep #{cockpit_state.stimno}" in mdi.lbl_live_status_pill.text()

    # Emit new sweep -> pill reflects updated sweep index
    cockpit_state.stimno = 5
    cockpit_state.onNewSweep.emit(5)
    assert "Sweep #5" in mdi.lbl_live_status_pill.text()

    # Toggle follow_live off via cockpit button -> PAUSED pill & amber button text
    mdi.btn_follow_live_pill.click()
    assert cockpit_state.follow_live is False
    assert mdi.btn_follow_live_pill.isChecked() is False
    assert "PAUSED" in mdi.lbl_live_status_pill.text()
    assert "History" in mdi.btn_follow_live_pill.text()

    # Toggle back on via cockpit button -> LIVE pill & checked
    mdi.btn_follow_live_pill.click()
    assert cockpit_state.follow_live is True
    assert mdi.btn_follow_live_pill.isChecked() is True
    assert "LIVE" in mdi.lbl_live_status_pill.text()

    # Empty state shows Sweep #0 offline
    empty_state = ViewerState()
    mdi_empty = MdiView(state=empty_state, check_updates=False)
    qtbot.addWidget(mdi_empty)
    assert "Offline | Sweep #0" in mdi_empty.lbl_live_status_pill.text()


def test_protocol_marker_quick_and_prompt(qtbot, cockpit_state):
    mdi = MdiView(state=cockpit_state, check_updates=False)
    qtbot.addWidget(mdi)

    cockpit_state.setStimNo(3)
    m1 = mdi.add_quick_protocol_marker("2 Hz Tetanus Start")
    assert m1 is not None
    assert m1["label"] == "2 Hz Tetanus Start"
    assert m1["stimno"] == 3
    assert m1["category"] == "protocol"
    assert len(cockpit_state.protocol_markers) == 1

    # In offscreen mode prompt_add_protocol_marker adds "Marker" at current stimno without modal dialog
    cockpit_state.setStimNo(4)
    m2 = mdi.prompt_add_protocol_marker()
    assert m2 is not None
    assert m2["label"] == "Marker"
    assert m2["stimno"] == 4
    assert len(cockpit_state.protocol_markers) == 2

    # Preset menu actions exist on btn_add_marker
    menu_labels = [a.text() for a in mdi.btn_add_marker.menu().actions() if not a.isSeparator()]
    assert "Baseline Start" in menu_labels
    assert "2 Hz Tetanus Start" in menu_labels
    assert "Recovery" in menu_labels


def test_live_search_and_tracking_layouts(qtbot, cockpit_state):
    mdi = MdiView(state=cockpit_state, check_updates=False)
    qtbot.addWidget(mdi)

    # 1. Live Search Layout (Electrode Hunting)
    mdi.apply_default_layout("Live Search Layout (Electrode Hunting)")
    widgets_search = [dw.widget() for dw in mdi.dock_manager.dockWidgets()]
    types_search = [type(w) for w in widgets_search]

    assert OpenEphysLiveView in types_search
    assert SpikeGroupTableView in types_search
    assert cockpit_state.auto_scan_live is True
    if _has_pg:
        from spikespy.MultiTracePGView import MultiTracePGView
        from spikespy.SingleTracePGView import SingleTracePGView

        assert MultiTracePGView in types_search
        assert SingleTracePGView in types_search

    # Clean up candidate auto-scan timer before switching layout
    cockpit_state.set_auto_scan_live(False)

    # 2. Live Tracking Layout (Protocol Execution)
    mdi.apply_default_layout("Live Tracking Layout (Protocol Execution)")
    widgets_track = [dw.widget() for dw in mdi.dock_manager.dockWidgets()]
    types_track = [type(w) for w in widgets_track]

    assert ActiveUnitInfoView in types_track
    assert SpikeGroupTableView in types_track
    if _has_pg:
        from spikespy.MultiTracePGView import MultiTracePGView
        from spikespy.SingleTracePGView import SingleTracePGView
        from spikespy.UnitPGView import UnitPGView

        assert MultiTracePGView in types_track
        assert SingleTracePGView in types_track
        assert UnitPGView in types_track

    # Views with channel_bar are bound to Shared Group A
    shared_bound = [
        w for w in widgets_track if hasattr(w, "channel_bar") and w.channel_bar is not None
    ]
    assert len(shared_bound) >= 3
    for w in shared_bound:
        mode, val = w.channel_bar.get_channel_mode()
        assert mode == ChannelGroupBar.MODE_SHARED_SET
        assert val == "Shared Group A"


def test_backspace_safety_guard_during_live_follow(qtbot, cockpit_state):
    mdi = MdiView(state=cockpit_state, check_updates=False)
    qtbot.addWidget(mdi)

    cockpit_state.stimno = 2
    initial_count = len(cockpit_state.spike_groups[0].event)
    assert initial_count == 6

    # 1. While streaming AND following live -> Backspace must NOT delete spike
    cockpit_state.is_streaming = True
    cockpit_state.follow_live = True
    mdi._on_backspace_pressed()
    assert len(cockpit_state.spike_groups[0].event) == initial_count

    # 2. While streaming BUT paused (follow_live=False) -> Backspace deletes spike
    cockpit_state.follow_live = False
    mdi._on_backspace_pressed()
    assert len(cockpit_state.spike_groups[0].event) == initial_count - 1

    # 3. When offline (is_streaming=False) even if follow_live=True -> Backspace deletes spike
    cockpit_state.is_streaming = False
    cockpit_state.follow_live = True
    cockpit_state.stimno = 1
    mdi._on_backspace_pressed()
    assert len(cockpit_state.spike_groups[0].event) == initial_count - 2
