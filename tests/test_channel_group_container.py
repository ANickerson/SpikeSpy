import os
import pytest
import numpy as np
import quantities as pq
import neo
from PySide6 import QtCore
from PySide6.QtWidgets import QApplication

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from spikespy.ViewerState import ViewerState, tracked_neuron_unit
from spikespy.ChannelGroupContainer import ChannelGroupContainerView, SubViewCard
from spikespy.MultiTraceView import MultiTraceView
from spikespy.SingleTraceView import SingleTraceView
from spikespy.UnitView import UnitView
from spikespy.ChannelGroupBar import ChannelGroupBar


@pytest.fixture
def mock_state():
    state = ViewerState()
    seg = neo.Segment(name="Test Segment")
    sr = 10000.0 * pq.Hz
    n_samples = 500
    n_sweeps = 5
    times = np.arange(n_sweeps) * 0.1 * pq.s
    event_sig = neo.Event(times, name="stim", units=pq.s)
    seg.events.append(event_sig)

    for i in range(4):
        ch_name = f"CH{i}"
        w = np.zeros((n_sweeps, n_samples))
        w[:, 100] = 50.0 / (i + 1)
        sig = neo.AnalogSignal(w.flatten(), units=pq.uV, sampling_rate=sr, name=ch_name)
        seg.analogsignals.append(sig)

    evt0 = neo.Event(times + 0.01 * pq.s, name="unit_0", units=pq.s)
    evt0.annotations["channel"] = "CH0"
    evt0.annotations["neighbor_channels"] = ["CH1"]
    u0 = tracked_neuron_unit(event=evt0, channel="CH0", neighbor_channels=["CH1"])

    evt1 = neo.Event(times + 0.02 * pq.s, name="unit_1", units=pq.s)
    evt1.annotations["channel"] = "CH2"
    evt1.annotations["neighbor_channels"] = ["CH3"]
    u1 = tracked_neuron_unit(event=evt1, channel="CH2", neighbor_channels=["CH3"])

    state.segment = seg
    state.set_data(analog_signal=seg.analogsignals[0], event_signal=event_sig, spike_groups=[u0, u1])
    return state


def test_channel_group_container_basics(mock_state):
    app = QApplication.instance() or QApplication([])

    container = ChannelGroupContainerView(state=mock_state)
    assert container.channel_bar is not None
    assert not container.channel_bar.isHidden()

    # Add MultiTraceView child
    mt_view = MultiTraceView(parent=container, state=mock_state)
    container.add_view(mt_view, title="MultiTrace")

    # Verify child's own channel selector is hidden and deferred to container
    assert hasattr(mt_view, "channel_bar")
    assert mt_view.channel_bar.isHidden()
    assert mt_view in container.get_child_views()

    # Add SingleTraceView child
    st_view = SingleTraceView(parent=container, state=mock_state)
    container.add_view(st_view, title="SingleTrace")
    assert st_view.channel_bar.isHidden()
    assert len(container.get_child_views()) == 2

    # Propagate unit binding through container
    container.channel_bar.bind_to_unit(1)
    app.processEvents()

    assert mt_view.channel_bar.get_bound_unit() == 1
    assert st_view.channel_bar.get_bound_unit() == 1

    # Propagate explicit channel selection through container
    container.channel_bar.bind_to_channels(["CH1", "CH2"])
    app.processEvents()
    assert mt_view.channel_bar.get_selected_channels() == ["CH1", "CH2"]
    assert st_view.channel_bar.get_selected_channels() == ["CH1", "CH2"]

    # Propagate multi-channel toggle
    container.channel_bar.chk_multichannel.setChecked(True)
    app.processEvents()
    assert mt_view.channel_bar.is_multichannel() is True
    assert st_view.channel_bar.is_multichannel() is True

    # Test closing / removing a child view
    container.remove_view(mt_view)
    assert mt_view not in container.get_child_views()
    assert len(container.get_child_views()) == 1

    # Clean up
    container.close()
    app.processEvents()


def test_channel_group_container_nested(mock_state):
    app = QApplication.instance() or QApplication([])

    parent_container = ChannelGroupContainerView(state=mock_state, name="Parent Group")
    child_container = ChannelGroupContainerView(state=mock_state, name="Child Group")

    parent_container.add_view(child_container, title="Nested Group")
    assert child_container in parent_container.get_child_views()

    # Add view inside nested container
    u_view = UnitView(parent=child_container, state=mock_state)
    child_container.add_view(u_view, title="UnitView")
    assert u_view.channel_bar.isHidden()

    # Parent syncs to child container and in turn to nested views
    parent_container.channel_bar.bind_to_unit(0)
    app.processEvents()

    assert child_container.channel_bar.get_bound_unit() == 0
    assert u_view.channel_bar.get_bound_unit() == 0

    parent_container.close()
    app.processEvents()


def test_channel_group_container_settings_serialization(mock_state):
    app = QApplication.instance() or QApplication([])

    container = ChannelGroupContainerView(state=mock_state)
    container.channel_bar.bind_to_unit(1)
    container.add_view(MultiTraceView(parent=container, state=mock_state), title="MultiTrace")

    settings = container.get_settings()
    assert "channel_bar" in settings
    assert "children" in settings
    assert len(settings["children"]) == 1

    # Restore settings in a new container
    new_container = ChannelGroupContainerView(state=mock_state)
    new_container.set_settings(settings)
    assert new_container.channel_bar.get_bound_unit() == 1

    container.close()
    new_container.close()
    app.processEvents()
