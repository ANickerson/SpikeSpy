import os
import pytest
import numpy as np
import quantities as pq
import neo
from PySide6.QtWidgets import QApplication
import PySide6QtAds as QtAds

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from spikespy.ViewerState import ViewerState, tracked_neuron_unit
from spikespy.SpikeSpy import MdiView
from spikespy.open_ephys_addin.oe_live_view import OpenEphysLiveView
from spikespy.ChannelGroupContainer import ChannelGroupContainerView
from spikespy.MultiTraceView import MultiTraceView


@pytest.fixture
def mock_mdi_state():
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

    state.segment = seg
    state.set_data(analog_signal=seg.analogsignals[0], event_signal=event_sig, spike_groups=[u0])
    return state


def test_open_ephys_live_view_singleton(mock_mdi_state):
    app = QApplication.instance() or QApplication([])

    mdi = MdiView(state=mock_mdi_state)
    mdi.show()
    app.processEvents()

    # 1. Open OpenEphysLive
    live1 = mdi.newWindow("OpenEphysLive")
    assert isinstance(live1, OpenEphysLiveView)
    assert mdi._open_ephys_live_dock is not None
    assert not mdi._open_ephys_live_dock.isClosed()

    # 2. Close the live dock widget
    dock = mdi._open_ephys_live_dock
    dock.closeDockWidget()
    app.processEvents()

    # Should be closed/hidden, but the C++ object and Python instance must NOT be deleted
    assert dock.isClosed()
    assert not dock.isVisible()

    # 3. Open OpenEphysLive again
    live2 = mdi.newWindow("OpenEphysLive")
    app.processEvents()

    # Must be the exact same instance!
    assert live2 is live1
    assert not dock.isClosed()
    assert dock.isVisible()

    mdi.close()
    app.processEvents()


def test_normal_windows_open_close(mock_mdi_state):
    app = QApplication.instance() or QApplication([])

    mdi = MdiView(state=mock_mdi_state)
    mdi.show()
    app.processEvents()

    # Open standard window without group wrapping
    w1 = mdi.newWindow("MultiTrace", wrap_in_channel_group=False)
    assert isinstance(w1, MultiTraceView)
    assert len(mdi.dock_manager.dockWidgets()) >= 2  # settings dock + w1

    # Find dock for w1 and close it
    dw1 = next(dw for dw in mdi.dock_manager.dockWidgets() if dw.widget() is w1)
    dw1.closeDockWidget()
    app.processEvents()
    assert dw1 not in mdi.dock_manager.dockWidgets()

    mdi.close()
    app.processEvents()


def test_channel_group_container_default_for_channel_specific_views(mock_mdi_state):
    app = QApplication.instance() or QApplication([])

    mdi = MdiView(state=mock_mdi_state)
    mdi.show()
    app.processEvents()

    # When created with target_unit, it should wrap in ChannelGroupContainerView by default
    w = mdi.newWindow("MultiTrace", target_unit=0)
    assert isinstance(w, ChannelGroupContainerView)
    assert w.channel_bar.get_bound_unit() == 0

    # Child view inside container
    child = w.get_child_views()[0]
    assert isinstance(child, MultiTraceView)
    assert child.channel_bar.isHidden()

    # Explicitly requesting container window
    c_win = mdi.newWindow("ChannelGroupContainer")
    assert isinstance(c_win, ChannelGroupContainerView)

    mdi.close()
    app.processEvents()
