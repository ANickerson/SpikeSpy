import numpy as np
import pytest
import quantities as pq
import neo
from PySide6.QtCore import Qt, QPoint
from PySide6.QtWidgets import QApplication

from spikespy.ViewerState import ViewerState, tracked_neuron_unit
from spikespy.ChannelGroupBar import ChannelGroupBar, ChannelGroupDialog
from spikespy.MultiTraceView import MultiTraceView
from spikespy.SingleTraceView import SingleTraceView
from spikespy.MultiTraceFixedView import MultiTraceFixedView
from spikespy.UnitView import UnitView
from spikespy.UnitFootprintView import UnitFootprintView
from spikespy.SpikeGroupTable import SpikeGroupTableView
from spikespy.SharedChannelPanel import SharedChannelPanel
from spikespy.SpikeSpy import MdiView


@pytest.fixture
def mock_multichannel_state():
    """Create a ViewerState with 8 channels (CH0 to CH7) and two units."""
    state = ViewerState()
    seg = neo.Segment(name="Synthetic Multichannel 8Ch")
    sr = 10000.0 * pq.Hz
    n_samples = 500
    n_sweeps = 6
    times = np.arange(n_sweeps) * 0.1 * pq.s
    event_sig = neo.Event(times, name="stim", units=pq.s)
    seg.events.append(event_sig)

    # Create 8 channels
    # CH0-CH3: Tetrode 1 (Unit 0 has spikes on CH0-CH3 at sample 100)
    # CH4-CH7: Tetrode 2 (Unit 1 has spikes on CH4-CH7 at sample 250)
    for i in range(8):
        chan_name = f"CH{i}"
        w = np.zeros((n_sweeps, n_samples), dtype=np.float32)
        if i < 4:
            # Unit 0 peak
            w[:, 100] = 50.0 - (i * 10.0)
        else:
            # Unit 1 peak
            w[:, 250] = 60.0 - ((i - 4) * 10.0)
        sig = neo.AnalogSignal(w.flatten(), units=pq.uV, sampling_rate=sr, name=chan_name)
        seg.analogsignals.append(sig)

    # Unit 0: CH0 (primary), neighbors CH1, CH2, CH3
    evt0 = neo.Event(times + (100 / 10000.0) * pq.s, name="unit_0", units=pq.s)
    evt0.annotations["channel"] = "CH0"
    evt0.annotations["neighbor_channels"] = ["CH1", "CH2", "CH3"]
    sg0 = tracked_neuron_unit(
        event=evt0,
        channel="CH0",
        neighbor_channels=["CH1", "CH2", "CH3"],
        idx_arr=[(100, 1) for _ in range(n_sweeps)],
    )

    # Unit 1: CH4 (primary), neighbors CH5, CH6, CH7
    evt1 = neo.Event(times + (250 / 10000.0) * pq.s, name="unit_1", units=pq.s)
    evt1.annotations["channel"] = "CH4"
    evt1.annotations["neighbor_channels"] = ["CH5", "CH6", "CH7"]
    sg1 = tracked_neuron_unit(
        event=evt1,
        channel="CH4",
        neighbor_channels=["CH5", "CH6", "CH7"],
        idx_arr=[(250, 1) for _ in range(n_sweeps)],
    )

    state.segment = seg
    state.set_data(analog_signal=seg.analogsignals[0], event_signal=event_sig, spike_groups=[sg0, sg1])
    return state


def test_viewer_state_channel_groups_and_shifting(mock_multichannel_state):
    state = mock_multichannel_state

    # 1. Preset channel groups
    state.add_channel_group("Tetrode1", ["CH0", "CH1", "CH2", "CH3"])
    state.add_channel_group("Tetrode2", ["CH4", "CH5", "CH6", "CH7"])

    assert state.get_channel_group("Tetrode1") == ["CH0", "CH1", "CH2", "CH3"]
    assert state.get_channel_group("Tetrode2") == ["CH4", "CH5", "CH6", "CH7"]
    all_groups = state.get_all_channel_groups()
    assert "Tetrode1" in all_groups and "Tetrode2" in all_groups

    # 2. Unit channels getter / setter
    assert state.get_unit_channels(0) == ["CH0", "CH1", "CH2", "CH3"]
    assert state.get_unit_channels(1) == ["CH4", "CH5", "CH6", "CH7"]

    state.set_unit_channels(0, ["CH1", "CH2"])
    assert state.spike_groups[0].channel == "CH1"
    assert state.spike_groups[0].neighbor_channels == ["CH2"]
    assert state.spike_groups[0].event.annotations["channel"] == "CH1"
    assert state.spike_groups[0].event.annotations["neighbor_channels"] == ["CH2"]

    # 3. Channel shifting
    shifted = state.shift_channels(["CH0", "CH1", "CH2", "CH3"], step=4)
    assert shifted == ["CH4", "CH5", "CH6", "CH7"]

    # Shift backwards
    shifted_back = state.shift_channels(shifted, step=-4)
    assert shifted_back == ["CH0", "CH1", "CH2", "CH3"]

    # Clamping at boundary
    shifted_clamp = state.shift_channels(["CH6", "CH7"], step=4)
    assert shifted_clamp == ["CH6", "CH7"]  # Cannot shift past CH7

    # Shift unit channels directly
    state.shift_unit_channels(0, step=2)
    assert state.get_unit_channels(0) == ["CH3", "CH4"]

    # 4. Multi-channel ERP calculation
    erp_mean = state.get_multichannel_erp(["CH0", "CH1", "CH2", "CH3"], combine_mode="mean")
    assert erp_mean is not None
    assert erp_mean.shape == state.analog_signal_erp.shape

    erp_stack = state.get_multichannel_erp(["CH0", "CH1"], combine_mode="stack")
    assert erp_stack is not None
    assert erp_stack.shape == (2, state.analog_signal_erp.shape[0], state.analog_signal_erp.shape[1])

    erp_sum = state.get_multichannel_erp(["CH0", "CH1"], combine_mode="sum")
    assert erp_sum is not None
    assert erp_sum.shape == state.analog_signal_erp.shape


def test_channel_group_dialog(qtbot, mock_multichannel_state):
    state = mock_multichannel_state
    dlg = ChannelGroupDialog(state=state, initial_channels=["CH0", "CH1"], unit_idx=0)
    qtbot.addWidget(dlg)

    # Initial selection
    assert dlg.get_selected_channels() == ["CH0", "CH1"]

    # Test Tetrode select
    dlg.list_widget.setCurrentRow(0)
    dlg._select_tetrode()
    assert dlg.get_selected_channels() == ["CH0", "CH1", "CH2", "CH3"]

    # Test shifting buttons
    dlg._shift_selection(4)
    assert dlg.get_selected_channels() == ["CH4", "CH5", "CH6", "CH7"]

    dlg._shift_selection(-1)
    assert dlg.get_selected_channels() == ["CH3", "CH4", "CH5", "CH6"]

    # Apply to unit
    dlg._apply_to_unit()
    assert state.get_unit_channels(0) == ["CH3", "CH4", "CH5", "CH6"]


def test_channel_group_bar_controls(qtbot, mock_multichannel_state):
    state = mock_multichannel_state
    bar = ChannelGroupBar(state=state)
    qtbot.addWidget(bar)

    # Default: Follow Active Unit, Follow Global Channel
    assert not bar.is_unit_locked()
    assert bar.get_target_unit_idx() == 0  # Matches state.cur_spike_group
    assert bar.get_primary_channel() == "CH0"
    assert bar.is_multichannel() is False

    # Bind to Unit 1
    bar.bind_to_unit(1)
    assert bar.is_unit_locked()
    assert bar.get_target_unit_idx() == 1
    assert "Unit 1" in bar.combo_unit.currentText()

    # Selecting Bound Unit Channels
    bar.set_channel_mode_unit()
    assert bar.get_primary_channel() == "CH4"
    assert bar.get_selected_channels() == ["CH4", "CH5", "CH6", "CH7"]
    assert bar.is_multichannel() is True

    # Shift group
    bar.shift_group(step=-1)
    # Unit 1 channels shifted backward by 1 -> CH3, CH4, CH5, CH6
    assert state.get_unit_channels(1) == ["CH3", "CH4", "CH5", "CH6"]

    # Test settings persistence
    settings = bar.get_settings()
    assert settings["unit_binding"] == 1
    assert settings["multichannel"] is True

    # New bar restoring settings
    bar2 = ChannelGroupBar(state=state)
    qtbot.addWidget(bar2)
    bar2.set_settings(settings)
    assert bar2.get_target_unit_idx() == 1
    assert bar2.is_multichannel() is True


def test_independent_views_and_simultaneous_multi_window(qtbot, mock_multichannel_state):
    state = mock_multichannel_state

    # Window 1: MultiTraceView bound to Unit 0 on channels CH0, CH1, CH2, CH3
    win1 = MultiTraceView(state=state)
    qtbot.addWidget(win1)
    win1.channel_bar.bind_to_unit(0)
    win1.channel_bar.bind_to_channels(["CH0", "CH1", "CH2", "CH3"])
    win1.channel_bar.chk_multichannel.setChecked(True)

    # Window 2: MultiTraceView bound to Unit 1 on channels CH4, CH5, CH6, CH7
    win2 = MultiTraceView(state=state)
    qtbot.addWidget(win2)
    win2.channel_bar.bind_to_unit(1)
    win2.channel_bar.bind_to_channels(["CH4", "CH5", "CH6", "CH7"])
    win2.channel_bar.chk_multichannel.setChecked(True)

    # Window 3: SingleTraceView bound to Unit 1
    win3 = SingleTraceView(state=state)
    qtbot.addWidget(win3)
    win3.channel_bar.bind_to_unit(1)
    win3.channel_bar.bind_to_channels(["CH4", "CH5", "CH6", "CH7"])

    # Window 4: MultiTraceFixedView bound to Unit 0
    win4 = MultiTraceFixedView(state=state)
    qtbot.addWidget(win4)
    win4.channel_bar.bind_to_unit(0)

    # Window 5: UnitView bound to Unit 1
    win5 = UnitView(state=state)
    qtbot.addWidget(win5)
    win5.channel_bar.bind_to_unit(1)

    # Window 6: UnitFootprintView bound to Unit 0
    win6 = UnitFootprintView(state=state)
    qtbot.addWidget(win6)
    win6.channel_bar.bind_to_unit(0)

    # 1. Verify independent target units and channels
    assert win1.channel_bar.get_target_unit_idx() == 0
    assert win2.channel_bar.get_target_unit_idx() == 1
    assert win1.channel_bar.get_selected_channels() == ["CH0", "CH1", "CH2", "CH3"]
    assert win2.channel_bar.get_selected_channels() == ["CH4", "CH5", "CH6", "CH7"]

    # 2. Verify window title reflects unit and channels
    title1 = win1.windowTitle()
    title2 = win2.windowTitle()
    assert "Unit 0" in title1
    assert "CH0" in title1
    assert "Unit 1" in title2
    assert "CH4" in title2

    # 3. Test independent unit spike annotation:
    # Set unit spike via setUnit on state targeting Unit 1 specifically without altering Unit 0
    init_u0_len = len(state.spike_groups[0].event)
    init_u1_len = len(state.spike_groups[1].event)

    # Global active unit is still 0
    assert state.cur_spike_group == 0

    # Annotate on Unit 1 using unitid=1
    new_latency = 260
    state.setUnit(latency=new_latency, stimno=2, unitid=1)

    # Global active unit is STILL 0
    assert state.cur_spike_group == 0
    # Unit 1 updated its spike latency at stimno 2 from 250 to 260
    assert state.spike_groups[1].idx_arr[2] == (260, 1)
    # Unit 0 is untouched at sample 100
    assert state.spike_groups[0].idx_arr[2] == (100, 1)

    # In win1, bound unit (Unit 0) is rendered in red with priority z-order
    p1 = win1.plot_spikegroups()
    assert 0 in win1.Patches
    assert win1.Patches[0].get_markeredgecolor() == "red"

    # In win2, bound unit (Unit 1) is rendered in red with priority z-order
    p2 = win2.plot_spikegroups()
    assert 1 in win2.Patches
    assert win2.Patches[1].get_markeredgecolor() == "red"

    # When all units are displayed in win1, bound unit 0 is highlighted red while unit 1 uses colormap
    win1.includeAllUnitsCheckBox.setChecked(True)
    win1.plot_spikegroups()
    assert win1.Patches[0].get_markeredgecolor() == "red"
    assert win1.Patches[1].get_markeredgecolor() != "red"

    # SingleTraceView overlays CH5, CH6, CH7 for Unit 1
    win3.chk_multichannel.setChecked(True)
    overlay = win3.get_overlay_channels()
    assert "CH5" in overlay and "CH6" in overlay and "CH7" in overlay
    assert "CH4" not in overlay  # CH4 is primary


def test_spikegroup_table_channel_shifting(mock_multichannel_state):
    state = mock_multichannel_state

    # Test shift unit group by +1
    orig_chans = state.get_unit_channels(0)
    state.shift_unit_channels(0, step=1)
    assert state.get_unit_channels(0) == ["CH1", "CH2", "CH3", "CH4"]

    # Shift back by -1
    state.shift_unit_channels(0, step=-1)
    assert state.get_unit_channels(0) == orig_chans


def test_spikespy_main_window_multichannel_windows(qtbot, mock_multichannel_state):
    state = mock_multichannel_state
    spy = MdiView(state=state, check_updates=False)
    qtbot.addWidget(spy)

    # Open a new MultiTraceView pinned to Unit 1 with channels CH4-CH7
    w = spy.newWindow(
        k="MultiTrace",
        target_unit=1,
        channels=["CH4", "CH5", "CH6", "CH7"],
    )
    assert w is not None
    assert hasattr(w, "channel_bar")
    assert w.channel_bar.get_target_unit_idx() == 1
    assert w.channel_bar.get_selected_channels() == ["CH4", "CH5", "CH6", "CH7"]

    # Open Dual Unit View
    spy.open_dual_unit_view()
    # There should now be multiple docked widgets
    dock_widgets = spy.dock_manager.dockWidgetsMap()
    assert len(dock_widgets) >= 2


def test_shared_channel_sets_and_auto_split(mock_multichannel_state):
    state = mock_multichannel_state

    # 1. auto_create_channel_groups
    created = state.auto_create_channel_groups(group_size=4)
    assert len(created) == 2
    groups = list(created.values())
    assert groups[0] == ["CH0", "CH1", "CH2", "CH3"]
    assert groups[1] == ["CH4", "CH5", "CH6", "CH7"]

    # 2. set_shared_channels
    state.set_shared_channels("Shared Group A", ["CH0", "CH1", "CH2", "CH3"], group_name="Tetrode 1")
    s_a = state.get_shared_channel_set("Shared Group A")
    assert s_a["channels"] == ["CH0", "CH1", "CH2", "CH3"]
    assert s_a["group_name"] == "Tetrode 1"

    # 3. shift_shared_channels
    shifted = state.shift_shared_channels("Shared Group A", step=2)
    assert shifted == ["CH2", "CH3", "CH4", "CH5"]
    assert state.get_shared_channel_set("Shared Group A")["channels"] == ["CH2", "CH3", "CH4", "CH5"]

    shifted_back = state.shift_shared_channels("Shared Group A", step=-2)
    assert shifted_back == ["CH0", "CH1", "CH2", "CH3"]
    assert state.get_shared_channel_set("Shared Group A")["channels"] == ["CH0", "CH1", "CH2", "CH3"]

    # 4. cycle_shared_group
    next_group = state.cycle_shared_group("Shared Group A", direction=1)
    assert next_group != ""
    assert len(state.get_shared_channel_set("Shared Group A")["channels"]) > 0


def test_shared_channel_panel_and_viewer_sync(qtbot, mock_multichannel_state):
    state = mock_multichannel_state

    # Setup 3 viewers linked to Shared Group A
    win_multi = MultiTraceView(state=state)
    qtbot.addWidget(win_multi)
    win_multi.channel_bar.bind_to_shared_set("Shared Group A")

    win_single = SingleTraceView(state=state)
    qtbot.addWidget(win_single)
    win_single.channel_bar.bind_to_shared_set("Shared Group A")

    win_unit = UnitView(state=state)
    qtbot.addWidget(win_unit)
    win_unit.channel_bar.bind_to_shared_set("Shared Group A")

    # Controller panel
    panel = SharedChannelPanel(state=state)
    qtbot.addWidget(panel)
    panel.show()

    # Initial channels check
    assert win_multi.channel_bar.get_selected_channels() == ["CH0", "CH1", "CH2", "CH3"]
    assert win_single.channel_bar.get_selected_channels() == ["CH0", "CH1", "CH2", "CH3"]
    assert win_unit.channel_bar.get_selected_channels() == ["CH0", "CH1", "CH2", "CH3"]

    # Auto-split tool creates groups of 4
    panel.spin_split.setValue(4)
    panel.btn_create_groups.click()
    assert len(panel._chip_buttons) >= 2

    # Steppers: shift by +1
    panel.btn_next_1.click()
    expected_p1 = ["CH1", "CH2", "CH3", "CH4"]
    assert win_multi.channel_bar.get_selected_channels() == expected_p1
    assert win_single.channel_bar.get_selected_channels() == expected_p1
    assert win_unit.channel_bar.get_selected_channels() == expected_p1

    # Steppers: shift back by -1
    panel.btn_prev_1.click()
    expected_orig = ["CH0", "CH1", "CH2", "CH3"]
    assert win_multi.channel_bar.get_selected_channels() == expected_orig
    assert win_single.channel_bar.get_selected_channels() == expected_orig
    assert win_unit.channel_bar.get_selected_channels() == expected_orig

    # One-click group chip switching: click the second chip
    chips = list(panel._chip_buttons.values())
    if len(chips) >= 2:
        chips[1].click()
        chans_chip = state.get_shared_channel_set("Shared Group A")["channels"]
        assert win_multi.channel_bar.get_selected_channels() == chans_chip
        assert win_single.channel_bar.get_selected_channels() == chans_chip
        assert win_unit.channel_bar.get_selected_channels() == chans_chip

    # Shift by group size
    curr_len = len(win_multi.channel_bar.get_selected_channels())
    panel.btn_shift_span.click()
    assert len(win_multi.channel_bar.get_selected_channels()) == curr_len

    # Test link all open viewers
    spy = MdiView(state=state, check_updates=False)
    qtbot.addWidget(spy)
    w_m = spy.newWindow("MultiTrace")
    w_s = spy.newWindow("SingleTraceView")
    p_ctrl = spy.open_or_show_shared_channel_panel()
    p_ctrl.link_all_open_viewers()
    assert w_m.channel_bar.get_channel_mode()[0] == ChannelGroupBar.MODE_SHARED_SET
    assert w_s.channel_bar.get_channel_mode()[0] == ChannelGroupBar.MODE_SHARED_SET


def test_layout_presets_and_named_perspectives(qtbot, mock_multichannel_state):
    state = mock_multichannel_state
    spy = MdiView(state=state, check_updates=False)
    qtbot.addWidget(spy)

    # 1. Shared Channel Group Suite
    spy.apply_default_layout("Shared Channel Group Suite")
    dock_widgets = spy.dock_manager.dockWidgets()
    assert len(dock_widgets) >= 4
    widget_types = [type(dw.widget()) for dw in dock_widgets]
    assert MultiTraceView in widget_types
    assert SingleTraceView in widget_types
    assert UnitView in widget_types
    assert SharedChannelPanel in widget_types

    # Viewers in Shared Channel Group Suite are bound to Shared Group A
    for dw in dock_widgets:
        inner = dw.widget()
        if hasattr(inner, "channel_bar"):
            mode, val = inner.channel_bar.get_channel_mode()
            assert mode == ChannelGroupBar.MODE_SHARED_SET
            assert val == "Shared Group A"

    # 2. Default Analysis Layout
    spy.apply_default_layout("Default Analysis Layout")
    dock_widgets2 = spy.dock_manager.dockWidgets()
    assert len(dock_widgets2) >= 4
    widget_types2 = [type(dw.widget()) for dw in dock_widgets2]
    assert MultiTraceView in widget_types2
    assert SingleTraceView in widget_types2
    assert UnitView in widget_types2
    assert SpikeGroupTableView in widget_types2

    # 3. Dual Unit Compare
    spy.apply_default_layout("Dual Unit Compare")
    dock_widgets3 = spy.dock_manager.dockWidgets()
    multi_views = []
    for dw in dock_widgets3:
        w = dw.widget()
        if isinstance(w, MultiTraceView):
            multi_views.append(w)
        elif hasattr(w, "get_child_views") and w.get_child_views():
            c0 = w.get_child_views()[0]
            if isinstance(c0, MultiTraceView):
                multi_views.append(w)  # Channel group container represents the view
    assert len(multi_views) >= 2
    assert multi_views[0].channel_bar.get_target_unit_idx() == 0
    assert multi_views[1].channel_bar.get_target_unit_idx() == 1

    # 4. Arbitrary Named Perspective Save / Load
    spy.save_perspective_as("MyNamedPerspective")
    assert "MyNamedPerspective" in spy.dock_manager.perspectiveNames()

    spy.clear_dock_area()
    assert len(spy.dock_manager.dockWidgets()) == 0

    ok = spy.load_perspective("MyNamedPerspective")
    assert ok is True
    assert len(spy.dock_manager.dockWidgets()) >= 2

