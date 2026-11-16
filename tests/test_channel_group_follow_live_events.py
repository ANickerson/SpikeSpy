import numpy as np
import quantities as pq
import pytest
from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QKeyEvent, QKeySequence
from PySide6.QtWidgets import QApplication

from spikespy.ChannelGroupBar import ChannelGroupBar
from spikespy.ChannelGroupContainer import ChannelGroupContainerView
from spikespy.SpikeSpy import MdiView
from spikespy.ViewSettingsPanel import ViewSettingsPanel
from spikespy.ViewerState import ViewerState, tracked_neuron_unit
import neo


@pytest.fixture
def test_state():
    state = ViewerState()
    n_samples = 5000
    n_channels = 8
    sig_data = np.random.randn(n_samples, n_channels).astype(np.float32)
    analog_sig = neo.AnalogSignal(
        sig_data,
        units="mV",
        sampling_rate=10000 * pq.Hz,
        name="sig",
    )
    # 10 stimulus events
    evt_times = np.linspace(0.1, 0.45, 10) * pq.s
    event_sig = neo.Event(times=evt_times, name="stim")

    sg = tracked_neuron_unit(
        event=neo.Event(times=evt_times + 0.005 * pq.s, name="unit_0"),
        channel="sig_0",
        idx_arr=[(50, 1)] * 10,
    )
    state.set_data(analog_sig, event_sig, [sg])
    # By default, latest stim is 9
    state.setStimNo(9, user_set=False)
    state.set_follow_live(True)
    return state


def test_channel_group_determines_following_live_events(test_state):
    app = QApplication.instance() or QApplication([])

    container = ChannelGroupContainerView(state=test_state)
    container.show()
    app.processEvents()

    # Initial state: at latest index (9), follow_live is True
    assert test_state.is_following_live_events() is True
    assert container.is_following_live_events() is True
    assert container.channel_bar.is_following_live_events() is True
    assert container.chk_follow_events.isChecked() is True
    assert container.channel_bar.chk_follow_events.isChecked() is True

    container.close()
    app.processEvents()


def test_manual_arrow_navigation_stops_tracing(test_state):
    app = QApplication.instance() or QApplication([])

    container = ChannelGroupContainerView(state=test_state)
    container.show()
    app.processEvents()

    assert container.is_following_live_events() is True

    # User manually navigates with arrow keys (e.g. Up to go to previous event)
    # Shortcuts target the active view's state, which is this container's local
    # overlay rather than the shared global index.
    container.view_state.setStimNo(container.view_state.stimno - 1, user_set=True)
    app.processEvents()

    # Tracing should now be stopped!
    assert container.view_state.stimno == 8
    assert container.is_following_live_events() is False
    assert container.channel_bar.is_following_live_events() is False
    assert container.chk_follow_events.isChecked() is False
    assert container.channel_bar.chk_follow_events.isChecked() is False

    # Simulate arrival of a new live sweep
    # When follow_live is False, stimno must remain at 8 and NOT follow the new sweep
    if hasattr(test_state, "_emit_live_frame"):
        test_state._live_frame_pending_idx = 10
        test_state._emit_live_frame()
    assert container.view_state.stimno == 8
    assert test_state.stimno == 9

    container.close()
    app.processEvents()


def test_toggle_follow_live_events_and_shortcut(test_state):
    app = QApplication.instance() or QApplication([])

    container = ChannelGroupContainerView(state=test_state)
    container.show()
    app.processEvents()

    # Move manually
    container.view_state.setStimNo(5, user_set=True)
    app.processEvents()
    assert container.is_following_live_events() is False

    # Toggle via checkbox
    container.chk_follow_events.setChecked(True)
    app.processEvents()

    # Should now be following live events and synced to latest index (9)
    assert container.is_following_live_events() is True
    assert container.view_state.stimno == 9
    assert container.chk_follow_events.isChecked() is True
    assert container.channel_bar.chk_follow_events.isChecked() is True

    # Toggle off via toggle_follow_live_events
    container.toggle_follow_live_events()
    app.processEvents()
    assert container.is_following_live_events() is False
    assert container.chk_follow_events.isChecked() is False

    # Toggle back on via channel_bar.toggle_follow_live_events
    container.channel_bar.toggle_follow_live_events()
    app.processEvents()
    assert container.is_following_live_events() is True
    assert container.chk_follow_events.isChecked() is True

    container.close()
    app.processEvents()


def test_view_settings_panel_channel_selection_options(test_state):
    app = QApplication.instance() or QApplication([])

    panel = ViewSettingsPanel(state=test_state)
    panel.show()
    app.processEvents()

    container = ChannelGroupContainerView(state=test_state)
    container.show()
    app.processEvents()

    panel.setActiveView(container)
    app.processEvents()

    # 1. Verify channel selection options exist in ViewSettingsPanel
    assert hasattr(panel, "combo_unit")
    assert hasattr(panel, "combo_channel")
    assert hasattr(panel, "chk_multichannel")
    assert hasattr(panel, "btn_shift_prev")
    assert hasattr(panel, "btn_shift_next")
    assert hasattr(panel, "btn_group_menu")
    assert hasattr(panel, "chk_follow_events")
    assert panel.grp_unit_selector.isVisible()

    # Check channels populated
    assert panel.combo_channel.count() > 0

    # 2. Test Follow Live Events checkbox in ViewSettingsPanel
    assert panel.is_following_live_events() is True
    assert panel.chk_follow_events.isChecked() is True

    # Manually move stimno
    container.view_state.setStimNo(3, user_set=True)
    app.processEvents()
    assert panel.is_following_live_events() is False
    assert panel.chk_follow_events.isChecked() is False

    # Toggle via panel shortcut method / checkbox
    panel.toggle_follow_live_events()
    app.processEvents()
    assert panel.is_following_live_events() is True
    assert panel.chk_follow_events.isChecked() is True
    assert container.view_state.stimno == 9
    assert container.is_following_live_events() is True

    # 3. Test multi-channel toggle from ViewSettingsPanel
    panel.chk_multichannel.setChecked(True)
    app.processEvents()
    assert container.channel_bar.is_multichannel() is True

    panel.chk_multichannel.setChecked(False)
    app.processEvents()
    assert container.channel_bar.is_multichannel() is False

    # 4. Test channel selection from ViewSettingsPanel
    avail = test_state.get_available_channels()
    if len(avail) >= 2:
        ch_target = avail[1]
        for i in range(panel.combo_channel.count()):
            d = panel.combo_channel.itemData(i)
            if isinstance(d, (tuple, list)) and len(d) == 2 and d[0] == ChannelGroupBar.MODE_SINGLE and d[1] == ch_target:
                panel.combo_channel.setCurrentIndex(i)
                break
        app.processEvents()
        assert container.channel_bar.get_primary_channel() == ch_target

    # 5. Test channel shifting from ViewSettingsPanel
    panel._shift_channel_group(1)
    app.processEvents()
    assert container.channel_bar.get_selected_channels() != []

    panel.close()
    container.close()
    app.processEvents()


def test_two_channel_groups_follow_events_independently(test_state):
    app = QApplication.instance() or QApplication([])

    follower = ChannelGroupContainerView(state=test_state, name="Follower")
    frozen = ChannelGroupContainerView(state=test_state, name="Frozen")
    follower.show()
    frozen.show()
    app.processEvents()

    follower.view_state.setStimNo(2, user_set=False)
    frozen.view_state.setStimNo(5, user_set=True)
    assert frozen.is_following_live_events() is False

    test_state._live_frame_pending_idx = 9
    test_state._emit_live_frame()
    app.processEvents()

    assert follower.view_state.stimno == 9
    assert follower.is_following_live_events() is True
    assert frozen.view_state.stimno == 5
    assert frozen.is_following_live_events() is False
    assert frozen.chk_follow_events.isChecked() is False

    frozen.close()
    follower.close()
    app.processEvents()


def test_child_views_use_their_container_local_stim_index(test_state):
    app = QApplication.instance() or QApplication([])

    first = ChannelGroupContainerView(state=test_state, name="First")
    second = ChannelGroupContainerView(state=test_state, name="Second")
    first.show()
    second.show()

    first_view = first.add_view_by_key("SingleTracePG")
    second_view = second.add_view_by_key("SingleTracePG")
    app.processEvents()

    assert first_view.state is first.view_state
    assert second_view.state is second.view_state

    second.view_state.setStimNo(3, user_set=True)
    app.processEvents()
    assert first_view.state.stimno == test_state.stimno
    assert second_view.state.stimno == 3

    second.close()
    first.close()
    app.processEvents()


def test_oe_live_pane_and_view_settings_follow_checkbox_stay_in_sync(test_state):
    """Flipping follow mode in either pane must update the other, not fight it."""
    from spikespy.open_ephys_addin.oe_live_view import OpenEphysLiveView

    app = QApplication.instance() or QApplication([])

    panel = ViewSettingsPanel(state=test_state)
    panel.show()
    oe_view = OpenEphysLiveView(state=test_state)
    oe_view.show()
    app.processEvents()

    test_state.set_follow_live(True)
    app.processEvents()
    assert oe_view.chk_follow_live.isChecked() is True

    # Unchecking in the View Settings pane must stick (demo mode used to
    # overwrite it on every incoming sweep).
    panel.chk_follow_events.setChecked(False)
    app.processEvents()
    assert test_state.follow_live is False
    assert oe_view.chk_follow_live.isChecked() is False
    assert panel.chk_follow_events.isChecked() is False

    # Re-checking in the Open Ephys pane must update the panel too.
    oe_view.chk_follow_live.setChecked(True)
    app.processEvents()
    assert test_state.follow_live is True
    assert panel.chk_follow_events.isChecked() is True
    assert test_state.stimno == test_state.get_latest_event_idx()

    oe_view.close()
    panel.close()
    app.processEvents()


def test_mdiview_ctrl_l_shortcut_toggles_tracing(test_state):
    app = QApplication.instance() or QApplication([])

    win = MdiView(state=test_state)
    win.show()
    app.processEvents()

    assert test_state.is_following_live_events() is True

    # Simulate Ctrl+L shortcut
    win.shortcut_toggle_live_events.activated.emit()
    app.processEvents()
    assert test_state.is_following_live_events() is False

    # Simulate Ctrl+L shortcut again
    win.shortcut_toggle_live_events.activated.emit()
    app.processEvents()
    assert test_state.is_following_live_events() is True
    assert test_state.stimno == 9

    win.close()
    app.processEvents()
