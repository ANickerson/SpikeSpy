import os
import pytest
import numpy as np
import quantities as pq
import neo
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication
import PySide6QtAds as QtAds

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from spikespy.ViewerState import ViewerState, tracked_neuron_unit
from spikespy.ChannelGroupContainer import ChannelGroupContainerView
from spikespy.ChannelGroupBar import ChannelGroupBar
from spikespy.SingleTracePGView import SingleTracePGView
from spikespy.MultiTracePGView import MultiTracePGView
from spikespy.UnitPGView import UnitPGView
from spikespy.MultiTraceView import MultiTraceView


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

    for i in range(8):
        chan_name = f"CH{i}"
        w = np.zeros((n_sweeps, n_samples), dtype=np.float32)
        if i < 4:
            w[:, 100] = 50.0 - (i * 10.0)
        else:
            w[:, 250] = 60.0 - ((i - 4) * 10.0)
        sig = neo.AnalogSignal(w.flatten(), units=pq.uV, sampling_rate=sr, name=chan_name)
        seg.analogsignals.append(sig)

    evt0 = neo.Event(times + (100 / 10000.0) * pq.s, name="unit_0", units=pq.s)
    evt0.annotations["channel"] = "CH0"
    evt0.annotations["neighbor_channels"] = ["CH1", "CH2", "CH3"]
    u0 = tracked_neuron_unit(event=evt0, channel="CH0", neighbor_channels=["CH1", "CH2", "CH3"])

    evt1 = neo.Event(times + (250 / 10000.0) * pq.s, name="unit_1", units=pq.s)
    evt1.annotations["channel"] = "CH4"
    evt1.annotations["neighbor_channels"] = ["CH5", "CH6", "CH7"]
    u1 = tracked_neuron_unit(event=evt1, channel="CH4", neighbor_channels=["CH5", "CH6", "CH7"])

    state.segment = seg
    state.set_data(analog_signal=seg.analogsignals[0], event_signal=event_sig, spike_groups=[u0, u1])
    return state


def test_singletrace_pg_respects_group_unit(mock_multichannel_state):
    """Verify SingleTracePGView respects the group's bound unit for display, snapping, and editing."""
    app = QApplication.instance() or QApplication([])

    container = ChannelGroupContainerView(state=mock_multichannel_state, name="Test Group")

    # Add SingleTracePGView
    st_pg = SingleTracePGView(parent=container, state=mock_multichannel_state)
    container.add_view(st_pg, title="SingleTracePG")

    # Child's individual channel selector is deferred/hidden
    assert hasattr(st_pg, "channel_bar")
    assert st_pg.channel_bar.isHidden()

    # Ensure global active unit is Unit 0
    mock_multichannel_state.cur_spike_group = 0

    # Bind container to Unit 1
    container.channel_bar.bind_to_unit(1)
    app.processEvents()

    # Verify SingleTracePGView target unit is Unit 1, not global Unit 0
    assert st_pg.get_target_unit_idx() == 1
    bound_sg = st_pg.get_target_unit()
    assert bound_sg is not None
    assert getattr(bound_sg.event, "name", "") == "unit_1"

    # Click / set unit in SingleTracePGView and verify Unit 1 is modified, NOT Unit 0
    initial_u0_len = len([x for x in mock_multichannel_state.spike_groups[0].idx_arr if x is not None])
    st_pg.set_cur_pos(250.0)
    app.processEvents()

    # Unit 1 should have its spike set around sample 250
    u1_idx_arr = mock_multichannel_state.spike_groups[1].idx_arr
    assert u1_idx_arr[mock_multichannel_state.stimno] is not None
    assert abs(u1_idx_arr[mock_multichannel_state.stimno][0] - 250) <= 2

    # Unit 0 must NOT have been modified
    final_u0_len = len([x for x in mock_multichannel_state.spike_groups[0].idx_arr if x is not None])
    assert final_u0_len == initial_u0_len

    container.close()
    app.processEvents()


def test_channel_group_container_nested_advanced_window_management(mock_multichannel_state):
    """Verify ChannelGroupContainer embeds QtAds CDockManager and supports complex arrangements."""
    app = QApplication.instance() or QApplication([])

    container = ChannelGroupContainerView(state=mock_multichannel_state, name="Docking Group")

    # 1. Verify CDockManager is initialized
    assert hasattr(container, "dock_manager")
    assert isinstance(container.dock_manager, QtAds.CDockManager)

    # 2. Add multiple sub-views
    v1 = container.add_view_by_key("MultiTracePG")
    v2 = container.add_view_by_key("SingleTracePG")
    v3 = container.add_view_by_key("UnitPG")
    assert len(container.get_child_views()) == 3

    # 3. Test complex layout arrangements
    container.arrange_layout("horizontal")
    assert container._current_layout == "horizontal"
    assert container._current_orientation == "horizontal"

    container.arrange_layout("vertical")
    assert container._current_layout == "vertical"
    assert container._current_orientation == "vertical"

    container.arrange_layout("grid")
    assert container._current_layout == "grid"

    container.arrange_layout("tabbed")
    assert container._current_layout == "tabbed"

    # 4. Toggle orientation button
    container._toggle_orientation()
    assert container._current_orientation in ("horizontal", "vertical")

    # 5. Remove sub-view and verify dock manager updates
    container.remove_view(v2)
    assert v2 not in container.get_child_views()
    assert len(container.get_child_views()) == 2

    # 6. Test nested container with its own independent dock manager
    nested_container = ChannelGroupContainerView(state=mock_multichannel_state, name="Nested Group")
    container.add_view(nested_container, title="Nested Group")
    assert nested_container in container.get_child_views()
    assert isinstance(nested_container.dock_manager, QtAds.CDockManager)

    # Adding views to nested container
    nv1 = nested_container.add_view_by_key("SingleTracePG")
    assert nv1 in nested_container.get_child_views()
    assert nv1 not in container.get_child_views()

    # Propagate unit from parent to nested container
    container.channel_bar.bind_to_unit(0)
    app.processEvents()
    assert nested_container.channel_bar.get_bound_unit() == 0
    assert nv1.get_target_unit_idx() == 0

    container.close()
    app.processEvents()


def test_channel_group_container_follow_live_inputs(mock_multichannel_state):
    """Verify ChannelGroupContainer follow live inputs option tracks incoming streaming data."""
    app = QApplication.instance() or QApplication([])

    container = ChannelGroupContainerView(state=mock_multichannel_state, name="Live Group")
    st_pg = SingleTracePGView(parent=container, state=mock_multichannel_state)
    container.add_view(st_pg, title="SingleTracePG")

    # Follow live input checkbox exists and defaults to False
    assert hasattr(container, "chk_follow_live")
    assert container.is_following_live() is False

    # Enable Follow Live Input
    container.set_follow_live(True)
    app.processEvents()
    assert container.is_following_live() is True
    assert container.channel_bar.get_channel_mode()[0] == ChannelGroupBar.MODE_FOLLOW_LIVE

    # Set up mock live buffers on ViewerState
    from spikespy.processing import LiveERPBuffer

    buf_ch2 = LiveERPBuffer(window_samples=500, initial_capacity=10)
    sweep_data = np.zeros(500, dtype=np.float32)
    sweep_data[150] = 45.0
    buf_ch2.append_sweep(sweep_data)

    mock_multichannel_state._live_aux_erp_buffers = {"CH2": buf_ch2}
    mock_multichannel_state.analog_signal.name = "CH1"

    # Trigger new sweep arrival
    mock_multichannel_state.onNewSweep.emit(0)
    app.processEvents()

    # Container and child views should now follow the live streaming channels
    selected_chans = container.channel_bar.get_selected_channels()
    assert "CH1" in selected_chans

    # If multichannel is enabled, both CH1 and CH2 should be included
    container.channel_bar.chk_multichannel.setChecked(True)
    app.processEvents()
    multi_live_chans = container.channel_bar.get_selected_channels()
    assert "CH1" in multi_live_chans
    assert "CH2" in multi_live_chans

    # Verify child view also has MODE_FOLLOW_LIVE synchronized
    assert st_pg.channel_bar.get_channel_mode()[0] == ChannelGroupBar.MODE_FOLLOW_LIVE

    # Disable Follow Live Input
    container.set_follow_live(False)
    app.processEvents()
    assert container.is_following_live() is False
    assert container.channel_bar.get_channel_mode()[0] != ChannelGroupBar.MODE_FOLLOW_LIVE

    container.close()
    app.processEvents()


def test_channel_group_container_settings_persistence(mock_multichannel_state):
    """Verify layout and follow_live settings are saved and restored."""
    app = QApplication.instance() or QApplication([])

    container = ChannelGroupContainerView(state=mock_multichannel_state, name="Persist Group")
    container.set_follow_live(True)
    container.arrange_layout("grid")
    container.add_view_by_key("MultiTracePG")

    settings = container.get_settings()
    assert settings.get("follow_live") is True
    assert settings.get("layout") == "grid"

    # Restore in new container
    new_container = ChannelGroupContainerView(state=mock_multichannel_state, name="Restored Group")
    new_container.set_settings(settings)
    assert new_container.is_following_live() is True
    assert new_container._current_layout == "grid"

    container.close()
    new_container.close()
    app.processEvents()


def test_multitrace_pg_single_channel_heatmap_respects_group(mock_multichannel_state):
    """Verify MultiTracePGView displays a heatmap of only one channel respecting its channel group."""
    app = QApplication.instance() or QApplication([])

    container = ChannelGroupContainerView(state=mock_multichannel_state, name="Heatmap Group")
    mt_pg = MultiTracePGView(parent=container, state=mock_multichannel_state)
    container.add_view(mt_pg, title="MultiTracePG")

    # 1. Child's channel bar is deferred / hidden
    assert hasattr(mt_pg, "channel_bar")
    assert mt_pg.channel_bar.isHidden()

    # 2. Bind container to single channel CH4
    container.channel_bar.bind_to_channel("CH4")
    app.processEvents()

    # mt_pg should resolve primary channel as CH4
    assert mt_pg.channel_bar.get_primary_channel() == "CH4"
    erp = mt_pg.get_erp_signal()
    assert erp is not None
    erp_mag = erp.magnitude if hasattr(erp, "magnitude") else erp
    # CH4 has peak at sample 250 (value 60.0), whereas CH0 has peak at sample 100
    assert np.argmax(erp_mag[0]) == 250
    assert erp_mag[0, 250] == pytest.approx(60.0)
    assert erp_mag[0, 100] == pytest.approx(0.0)

    # 3. Bind container to a channel group with multiple channels (e.g. ["CH4", "CH5", "CH6"])
    # and turn on Multi-Channel in container
    mock_multichannel_state.add_channel_group("Group_456", ["CH4", "CH5", "CH6"])
    container.channel_bar.bind_to_group("Group_456")
    container.channel_bar.chk_multichannel.setChecked(True)
    app.processEvents()

    # MultiTracePGView must show a heatmap ONLY of one channel (primary channel CH4), not multi-channel blended
    primary_ch = mt_pg.channel_bar.get_primary_channel()
    assert primary_ch == "CH4"
    erp_group = mt_pg.get_erp_signal()
    assert erp_group.shape == (6, 5000)
    erp_group_mag = erp_group.magnitude if hasattr(erp_group, "magnitude") else erp_group
    assert np.argmax(erp_group_mag[0]) == 250
    assert erp_group_mag[0, 250] == pytest.approx(60.0)

    # 4. Switch container back to CH0
    container.channel_bar.bind_to_channel("CH0")
    app.processEvents()
    assert mt_pg.channel_bar.get_primary_channel() == "CH0"
    erp_ch0 = mt_pg.get_erp_signal()
    erp_ch0_mag = erp_ch0.magnitude if hasattr(erp_ch0, "magnitude") else erp_ch0
    assert np.argmax(erp_ch0_mag[0]) == 100
    assert erp_ch0_mag[0, 100] == pytest.approx(50.0)

    container.close()
    app.processEvents()


def test_multitrace_pg_tracked_unit_strictly_shows_bound_unit(mock_multichannel_state):
    """Verify MultiTracePGView tracked unit only displays the bound unit and not all units."""
    app = QApplication.instance() or QApplication([])

    # Add a 3rd unit (unit_2) with distinct latency
    times = mock_multichannel_state.event_signal.times
    evt2 = neo.Event(times + (400 / 10000.0) * pq.s, name="unit_2", units=pq.s)
    evt2.annotations["channel"] = "CH6"
    u2 = tracked_neuron_unit(event=evt2, channel="CH6")
    mock_multichannel_state.spike_groups.append(u2)
    mock_multichannel_state.update_idx_arrs()

    assert len(mock_multichannel_state.spike_groups) == 3

    container = ChannelGroupContainerView(state=mock_multichannel_state, name="Units Test Group")
    mt_pg = MultiTracePGView(parent=container, state=mock_multichannel_state)
    container.add_view(mt_pg, title="MultiTracePG")

    # Global active unit is Unit 0
    mock_multichannel_state.cur_spike_group = 0

    # 1. Bind container to Unit 1
    container.channel_bar.bind_to_unit(1)
    app.processEvents()

    assert mt_pg.get_target_unit_idx() == 1
    assert mt_pg.includeUnitCheckbox.isChecked() is True
    assert mt_pg.includeAllUnitsCheckBox.isChecked() is False

    # Check scatter data for tracked unit
    tracked_data = mt_pg.tracked_unit_scatter.getData()
    x_pts, y_pts = tracked_data[0], tracked_data[1]
    assert len(x_pts) == 6  # 6 sweeps
    # Unit 1's spikes are all around sample 250
    assert np.all(x_pts == 250)
    # Unit 0's spikes (sample 100) and Unit 2's spikes (sample 400) must NOT be in tracked_unit_scatter
    assert not np.any(x_pts == 100)
    assert not np.any(x_pts == 400)
    assert len(mt_pg.other_unit_scatters) == 0

    # 2. Switch container to Unit 2
    container.channel_bar.bind_to_unit(2)
    app.processEvents()

    assert mt_pg.get_target_unit_idx() == 2
    tracked_data2 = mt_pg.tracked_unit_scatter.getData()
    x_pts2, y_pts2 = tracked_data2[0], tracked_data2[1]
    assert len(x_pts2) == 6
    # Unit 2's spikes are all at sample 400
    assert np.all(x_pts2 == 400)
    assert not np.any(x_pts2 == 100)
    assert not np.any(x_pts2 == 250)

    # 3. Enable "Others" checkbox
    mt_pg.includeAllUnitsCheckBox.setChecked(True)
    app.processEvents()

    # Now other_unit_scatters should contain the other 2 units (Unit 0 and Unit 1)
    assert len(mt_pg.other_unit_scatters) == 2
    # But tracked_unit_scatter still strictly contains only Unit 2
    tracked_data_still_u2 = mt_pg.tracked_unit_scatter.getData()
    assert np.all(tracked_data_still_u2[0] == 400)

    container.close()
    app.processEvents()

