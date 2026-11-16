import os
import pytest
import numpy as np
import quantities as pq
import neo
from PySide6.QtCore import Qt, QPointF
from PySide6.QtWidgets import QApplication

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from spikespy.ViewerState import ViewerState, tracked_neuron_unit
from spikespy.MultiTracePGView import MultiTracePGView
from spikespy.ChannelGroupContainer import ChannelGroupContainerView
from spikespy.ChannelGroupBar import ChannelGroupBar


def _create_multichannel_state() -> ViewerState:
    state = ViewerState()
    seg = neo.Segment(name="Multichannel 8Ch Test")
    sr = 10000.0 * pq.Hz
    n_samples = 400
    n_sweeps = 5
    times = np.arange(n_sweeps) * 0.1 * pq.s
    event_sig = neo.Event(times, name="stim", units=pq.s)
    seg.events.append(event_sig)

    for i in range(8):
        chan_name = f"CH{i}"
        w = np.zeros((n_sweeps, n_samples), dtype=np.float32)
        w[:, 50 + i * 20] = 40.0 + i * 5.0
        sig = neo.AnalogSignal(w.flatten(), units=pq.uV, sampling_rate=sr, name=chan_name)
        seg.analogsignals.append(sig)

    # Unit 0 on CH0 with neighbors CH1, CH2, CH3
    evt0 = neo.Event(times + (50 / 10000.0) * pq.s, name="unit_0", units=pq.s)
    evt0.annotations["channel"] = "CH0"
    evt0.annotations["neighbor_channels"] = ["CH1", "CH2", "CH3"]
    u0 = tracked_neuron_unit(event=evt0, channel="CH0", neighbor_channels=["CH1", "CH2", "CH3"])

    # Unit 1 on CH4 with neighbors CH5, CH6, CH7
    evt1 = neo.Event(times + (130 / 10000.0) * pq.s, name="unit_1", units=pq.s)
    evt1.annotations["channel"] = "CH4"
    evt1.annotations["neighbor_channels"] = ["CH5", "CH6", "CH7"]
    u1 = tracked_neuron_unit(event=evt1, channel="CH4", neighbor_channels=["CH5", "CH6", "CH7"])

    state.segment = seg
    state.set_data(analog_signal=seg.analogsignals[0], event_signal=event_sig, spike_groups=[u0, u1])
    return state


@pytest.fixture
def multichannel_8ch_state():
    """Create a ViewerState with 8 channels (CH0 to CH7) and two tracked units."""
    return _create_multichannel_state()


def test_multitrace_pg_add_remove_channels_and_slots(multichannel_8ch_state):
    """Test adding, removing, and clearing channel slots in MultiTracePGView."""
    app = QApplication.instance() or QApplication([])

    view = MultiTracePGView(state=multichannel_8ch_state)
    view.show()
    app.processEvents()

    # Initial state: 1 channel panel (CH0)
    assert len(view.channel_panels) == 1
    assert view.channel_panels[0].channel_name == "CH0"
    assert "[1/1 in group] CH0" in view.channel_panels[0].plot.titleLabel.text

    # 1. Add CH1
    view.add_channel_to_display("CH1")
    app.processEvents()
    assert len(view.channel_panels) == 2
    assert view.displayed_channels == ["CH0", "CH1"]
    assert view.channel_panels[0].channel_name == "CH0"
    assert view.channel_panels[1].channel_name == "CH1"
    assert "[1/2 in group]" in view.channel_panels[0].plot.titleLabel.text
    assert "[2/2 in group] CH1" in view.channel_panels[1].plot.titleLabel.text

    # 2. Add an empty slot
    view.add_empty_slot()
    app.processEvents()
    assert len(view.channel_panels) == 3
    assert view.displayed_channels == ["CH0", "CH1", None]
    assert view.channel_panels[2].channel_name is None
    assert "[3/3 in group] (Empty)" in view.channel_panels[2].plot.titleLabel.text
    assert view.channel_panels[2].empty_text_item.isVisible() is True
    assert view.channel_panels[2].has_data is False

    # 3. Add CH2 - should fill the existing empty slot
    view.add_channel_to_display("CH2")
    app.processEvents()
    assert len(view.channel_panels) == 3
    assert view.displayed_channels == ["CH0", "CH1", "CH2"]
    assert view.channel_panels[2].channel_name == "CH2"
    assert "[3/3 in group] CH2" in view.channel_panels[2].plot.titleLabel.text
    assert view.channel_panels[2].has_data is True

    # 4. Clear slot 1 to empty (CH1 -> None)
    view.clear_slot_to_empty(1)
    app.processEvents()
    assert view.displayed_channels == ["CH0", None, "CH2"]
    assert view.channel_panels[1].channel_name is None
    assert "[2/3 in group] (Empty)" in view.channel_panels[1].plot.titleLabel.text

    # 5. Remove empty screens
    view._remove_empty_screens()
    app.processEvents()
    assert view.displayed_channels == ["CH0", "CH2"]
    assert len(view.channel_panels) == 2

    # 6. Remove CH2 completely
    view.remove_channel_from_display("CH2", keep_empty_slot=False)
    app.processEvents()
    assert view.displayed_channels == ["CH0"]
    assert len(view.channel_panels) == 1

    view.close()
    app.processEvents()


def test_multitrace_pg_group_size_n_and_presets(multichannel_8ch_state):
    """Test setting group size N and using presets (1-Ch, 2-Ch, 4-Ch Tetrode, 8-Ch)."""
    app = QApplication.instance() or QApplication([])

    view = MultiTracePGView(state=multichannel_8ch_state)
    view.show()
    app.processEvents()

    # Preset 4-Ch (Tetrode)
    view.set_group_size_n(4)
    app.processEvents()
    assert view.group_size_n == 4
    assert len(view.channel_panels) == 4
    # CH0 was present, next slots auto-padded with remaining unit/probe channels or empty
    assert view.channel_panels[0].channel_name == "CH0"
    for i in range(4):
        assert f"[{i+1}/4 in group]" in view.channel_panels[i].plot.titleLabel.text

    # Preset 8-Ch
    view.set_group_size_n(8)
    app.processEvents()
    assert view.group_size_n == 8
    assert len(view.channel_panels) == 8
    for i in range(8):
        assert f"[{i+1}/8 in group]" in view.channel_panels[i].plot.titleLabel.text

    # Preset 1-Ch
    view.set_group_size_n(1)
    app.processEvents()
    assert view.group_size_n == 1
    assert len(view.channel_panels) == 1

    view.close()
    app.processEvents()


def test_multitrace_pg_channel_group_sync(multichannel_8ch_state):
    """Test that modifying displayed channels updates the linked ViewerState channel group."""
    app = QApplication.instance() or QApplication([])

    view = MultiTracePGView(state=multichannel_8ch_state)
    view.show()
    app.processEvents()

    # Unit 0 initially has primary CH0 and neighbor channels CH1, CH2, CH3
    u0 = multichannel_8ch_state.spike_groups[0]
    assert u0.neighbor_channels == ["CH1", "CH2", "CH3"]

    # Bind view's channel bar to Unit 0
    view.channel_bar.bind_to_unit(0)
    app.processEvents()

    # Expand to 4 channels
    view.add_channel_to_display("CH1")
    view.add_channel_to_display("CH2")
    view.add_channel_to_display("CH3")
    app.processEvents()

    # Add CH4
    view.add_channel_to_display("CH4")
    app.processEvents()

    # Unit 0 neighbor channels should now include CH4
    assert "CH4" in u0.neighbor_channels or "CH4" == u0.channel

    # Test named group sync
    multichannel_8ch_state.add_channel_group("CustomGroup", ["CH0", "CH1"])
    view.channel_bar.bind_to_group("CustomGroup")
    app.processEvents()

    view.add_channel_to_display("CH5")
    app.processEvents()
    assert "CH5" in multichannel_8ch_state.channel_groups["CustomGroup"]

    view.close()
    app.processEvents()


def test_multitrace_pg_retains_empty_screens_on_switching(multichannel_8ch_state):
    """Test that switching channel group retains N screens, keeping missing channels as empty screens."""
    app = QApplication.instance() or QApplication([])

    # Group A has 4 channels
    multichannel_8ch_state.add_channel_group("GroupA", ["CH0", "CH1", "CH2", "CH3"])
    # Group B has only 2 channels
    multichannel_8ch_state.add_channel_group("GroupB", ["CH4", "CH5"])

    view = MultiTracePGView(state=multichannel_8ch_state)
    view.show()
    app.processEvents()

    # Set target group size N = 4
    view.set_group_size_n(4)
    view.channel_bar.bind_to_group("GroupA")
    app.processEvents()

    assert len(view.channel_panels) == 4
    assert view.displayed_channels == ["CH0", "CH1", "CH2", "CH3"]

    # Switch to Group B (which only has 2 channels)
    view.channel_bar.bind_to_group("GroupB")
    app.processEvents()

    # MultiTracePGView MUST retain N = 4 screens!
    assert view.group_size_n == 4
    assert len(view.channel_panels) == 4
    assert view.displayed_channels[0] == "CH4"
    assert view.displayed_channels[1] == "CH5"
    assert view.displayed_channels[2] is None
    assert view.displayed_channels[3] is None

    # Verify labels
    assert "[1/4 in group] CH4" in view.channel_panels[0].plot.titleLabel.text
    assert "[2/4 in group] CH5" in view.channel_panels[1].plot.titleLabel.text
    assert "[3/4 in group] (Empty)" in view.channel_panels[2].plot.titleLabel.text
    assert "[4/4 in group] (Empty)" in view.channel_panels[3].plot.titleLabel.text

    # Slots 2 and 3 should show empty watermarks
    assert view.channel_panels[2].empty_text_item.isVisible() is True
    assert view.channel_panels[3].empty_text_item.isVisible() is True

    view.close()
    app.processEvents()


def test_multitrace_pg_missing_unrecorded_channel(multichannel_8ch_state):
    """Test displaying an unrecorded/missing channel as (No data) without crashing."""
    app = QApplication.instance() or QApplication([])

    view = MultiTracePGView(state=multichannel_8ch_state)
    view.show()
    app.processEvents()

    # Add a channel name that is not in the recording
    view.add_channel_to_display("CH99_Unrecorded")
    app.processEvents()

    assert len(view.channel_panels) == 2
    panel_unrec = view.channel_panels[1]
    assert panel_unrec.channel_name == "CH99_Unrecorded"
    assert panel_unrec.has_data is False
    assert "[2/2 in group] CH99_Unrecorded (No data)" in panel_unrec.plot.titleLabel.text
    assert "No data: CH99_Unrecorded" in panel_unrec.empty_text_item.toPlainText()

    view.close()
    app.processEvents()


def test_multitrace_pg_multi_channel_views_linked_axes_and_interactions(qtbot, multichannel_8ch_state):
    """Test that all channel plots link X and Y axes, and curstim/spikegroups update across panels."""
    app = QApplication.instance() or QApplication([])
    view = MultiTracePGView(state=multichannel_8ch_state)
    qtbot.addWidget(view)
    view.show()

    view.set_group_size_n(2)
    view.add_channel_to_display("CH1")
    qtbot.wait(50)

    assert len(view.channel_panels) == 2
    p0 = view.channel_panels[0]
    p1 = view.channel_panels[1]

    # Verify X and Y linking
    assert p1.plot.getViewBox().linkedView(0) == p0.plot.getViewBox()  # X link
    assert p1.plot.getViewBox().linkedView(1) == p0.plot.getViewBox()  # Y link

    # Stimulus cursor line updates across both panels
    multichannel_8ch_state.setStimNo(3)
    qtbot.wait(50)
    assert view.channel_panels[0].stim_line.value() == 3
    assert view.channel_panels[1].stim_line.value() == 3

    # Contrast update across both panels
    view.lowerSpinBox.setValue(0.2)
    view.upperSpinBox.setValue(3.0)
    view.updateAll()
    app.processEvents()
    assert view.channel_panels[0].img_item.levels is not None
    assert view.channel_panels[1].img_item.levels is not None

    # Tracked unit scatter renders on active panels
    p0_scatter_pts = view.channel_panels[0].tracked_unit_scatter.getData()
    p1_scatter_pts = view.channel_panels[1].tracked_unit_scatter.getData()
    assert len(p0_scatter_pts[0]) == 5
    assert len(p1_scatter_pts[0]) == 5


def test_multitrace_pg_settings_persistence(multichannel_8ch_state):
    """Test saving and restoring group_size_n and displayed_channels via get_settings and set_settings."""
    app = QApplication.instance() or QApplication([])

    view = MultiTracePGView(state=multichannel_8ch_state)
    view.show()
    app.processEvents()

    view.set_group_size_n(3)
    view.displayed_channels = ["CH0", "CH3", None]
    view.setup_figure()
    app.processEvents()

    settings = view.get_settings()
    assert settings["group_size_n"] == 3
    assert settings["displayed_channels"] == ["CH0", "CH3", None]

    # Create new view and restore settings
    view2 = MultiTracePGView(state=multichannel_8ch_state)
    view2.show()
    app.processEvents()

    view2.set_settings(settings)
    app.processEvents()

    assert view2.group_size_n == 3
    assert view2.displayed_channels == ["CH0", "CH3", None]
    assert len(view2.channel_panels) == 3
    assert view2.channel_panels[0].channel_name == "CH0"
    assert view2.channel_panels[1].channel_name == "CH3"
    assert view2.channel_panels[2].channel_name is None

    view.close()
    view2.close()
    app.processEvents()


def test_multitrace_pg_summary_line_graphs_channel_group_linking(multichannel_8ch_state):
    """Test that summary line graphs display per-channel stats with N-in-group labels and star for primary."""
    app = QApplication.instance() or QApplication([])

    view = MultiTracePGView(state=multichannel_8ch_state)
    view.show()
    app.processEvents()

    # Set up 2 channels: CH0 and CH1
    view.set_group_size_n(2)
    view.displayed_channels = ["CH0", "CH1"]
    view.setup_figure()
    app.processEvents()

    # Check summary panels
    assert hasattr(view, "summary_panels")
    assert "CH0" in view.summary_panels
    assert "CH1" in view.summary_panels

    sp0 = view.summary_panels["CH0"]
    sp1 = view.summary_panels["CH1"]

    # Labels contain [1/2 in group] and [2/2 in group]
    assert "[1/2 in group] CH0" in sp0.label
    assert "[2/2 in group] CH1" in sp1.label
    # Primary channel CH0 has star
    assert "★" in sp0.label

    # Summary curves have valid row mean data
    x0, y0 = sp0.curve.getData()
    assert len(x0) == 5
    assert len(y0) == 5

    view.close()
    app.processEvents()


def test_multitrace_pg_summary_line_graphs_toggle_mean_max_and_stim_cursor(multichannel_8ch_state):
    """Test toggling Mean/Max updates summary graph titles and data, and stim_line updates."""
    app = QApplication.instance() or QApplication([])

    view = MultiTracePGView(state=multichannel_8ch_state)
    view.show()
    app.processEvents()

    view.set_group_size_n(2)
    view.displayed_channels = ["CH0", "CH1"]
    view.setup_figure()
    app.processEvents()

    # Initially in Mean mode
    sp0 = view.summary_panels["CH0"]
    assert "(Mean)" in sp0.label
    mean_val = float(sp0.curve.getData()[0][0])

    # Toggle to Max
    view.toggleMaxMean()
    app.processEvents()

    sp0 = view.summary_panels["CH0"]
    assert "(Max)" in sp0.label
    max_val = float(sp0.curve.getData()[0][0])
    # Max should be greater than or equal to mean
    assert max_val >= mean_val

    # Stim line cursor sync
    multichannel_8ch_state.setStimNo(3)
    app.processEvents()
    for sp in view.summary_panels.values():
        assert sp.stim_line.value() == 3

    view.close()
    app.processEvents()


def test_multitrace_pg_summary_line_graphs_retains_empty_screens_and_settings(multichannel_8ch_state):
    """Test that switching groups preserves empty summary screens and settings persistence."""
    app = QApplication.instance() or QApplication([])

    multichannel_8ch_state.add_channel_group("GroupSmall", ["CH0"])
    view = MultiTracePGView(state=multichannel_8ch_state)
    view.show()
    app.processEvents()

    # Set target group size N = 3 and bind to GroupSmall (1 channel)
    view.set_group_size_n(3)
    view.channel_bar.bind_to_group("GroupSmall")
    app.processEvents()

    assert view.group_size_n == 3
    assert view.displayed_channels == ["CH0", None, None]

    # Summary panels must maintain 3 slots (1 channel + 2 empty screens)
    assert "__empty_slot_1__" in view.summary_panels
    assert "__empty_slot_2__" in view.summary_panels

    sp_empty1 = view.summary_panels["__empty_slot_1__"]
    sp_empty2 = view.summary_panels["__empty_slot_2__"]

    assert "[2/3 in group] (Empty)" in sp_empty1.label
    assert "[3/3 in group] (Empty)" in sp_empty2.label
    assert sp_empty1.empty_text_item.isVisible() is True
    assert sp_empty2.empty_text_item.isVisible() is True

    # Test settings persistence
    settings = view.get_settings()
    assert settings["group_size_n"] == 3
    assert "right_plots" in settings
    assert "show_max" in settings

    view.close()
    app.processEvents()
