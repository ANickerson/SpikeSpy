import os
import neo
import numpy as np
import pytest
import quantities as pq
from PySide6.QtWidgets import QApplication

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from spikespy.MultiTracePGView import MultiTracePGView
from spikespy.MultiTraceView import MultiTraceView
from spikespy.SingleTraceView import SingleTraceView
from spikespy.SpikeSpy import MdiView
from spikespy.ViewSettingsPanel import ViewSettingsPanel
from spikespy.ViewerState import ViewerState, tracked_neuron_unit
from spikespy.helpers import notify_view_selected
from spikespy.open_ephys_addin import OpenEphysLiveView, OpenEphysStreamWorker


@pytest.fixture
def sample_state():
    state = ViewerState()
    seg = neo.Segment(name="TestSegment")
    sr = 10000.0 * pq.Hz
    n_samples = 500
    n_sweeps = 5
    times = np.arange(n_sweeps) * 0.1 * pq.s
    event_sig = neo.Event(times, name="stim", units=pq.s)
    seg.events.append(event_sig)

    for i in range(4):
        w = np.zeros((n_sweeps, n_samples), dtype=np.float32)
        w[:, 100] = 50.0 / (i + 1)
        sig = neo.AnalogSignal(
            w.flatten(), units=pq.uV, sampling_rate=sr, name=f"CH{i}"
        )
        seg.analogsignals.append(sig)

    evt0 = neo.Event(times + 0.01 * pq.s, name="unit_0", units=pq.s)
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


def test_pagination_buttons_and_all_page(qtbot, sample_state):
    panel = ViewSettingsPanel(state=sample_state)
    qtbot.addWidget(panel)
    panel.show()

    mt_view = MultiTracePGView(state=sample_state)
    qtbot.addWidget(mt_view)
    mt_view.show()

    panel.setActiveView(mt_view)
    QApplication.processEvents()

    # Default page is "view"; only grp_multitrace is visible
    assert panel.current_page == "view"
    assert panel.btn_page_view.isChecked() is True
    assert panel.grp_multitrace.isVisible() is True
    assert panel.grp_unit_selector.isVisible() is False
    assert panel.grp_active_unit_info.isVisible() is False
    assert panel.grp_navigation.isVisible() is False
    assert panel.grp_streaming.isVisible() is False

    # Channels page
    panel.btn_page_channels.click()
    QApplication.processEvents()
    assert panel.current_page == "channels"
    assert panel.grp_multitrace.isVisible() is False
    assert panel.grp_unit_selector.isVisible() is True
    assert panel.grp_active_unit_info.isVisible() is False
    assert panel.grp_navigation.isVisible() is False
    assert panel.grp_streaming.isVisible() is False

    # Unit page
    panel.btn_page_unit.click()
    QApplication.processEvents()
    assert panel.current_page == "unit"
    assert panel.grp_multitrace.isVisible() is False
    assert panel.grp_unit_selector.isVisible() is False
    assert panel.grp_active_unit_info.isVisible() is True
    assert panel.grp_navigation.isVisible() is False
    assert panel.grp_streaming.isVisible() is False

    # Nav page
    panel.btn_page_nav.click()
    QApplication.processEvents()
    assert panel.current_page == "nav"
    assert panel.grp_multitrace.isVisible() is False
    assert panel.grp_unit_selector.isVisible() is False
    assert panel.grp_active_unit_info.isVisible() is False
    assert panel.grp_navigation.isVisible() is True
    assert panel.grp_streaming.isVisible() is False

    # Stream page
    panel.btn_page_stream.click()
    QApplication.processEvents()
    assert panel.current_page == "stream"
    assert panel.grp_multitrace.isVisible() is False
    assert panel.grp_unit_selector.isVisible() is False
    assert panel.grp_active_unit_info.isVisible() is False
    assert panel.grp_navigation.isVisible() is False
    assert panel.grp_streaming.isVisible() is True

    # View page again
    panel.btn_page_view.click()
    QApplication.processEvents()
    assert panel.current_page == "view"
    assert panel.grp_multitrace.isVisible() is True
    assert panel.grp_unit_selector.isVisible() is False
    assert panel.grp_active_unit_info.isVisible() is False
    assert panel.grp_navigation.isVisible() is False
    assert panel.grp_streaming.isVisible() is False

    # All page shows all applicable groups simultaneously
    panel.btn_page_all.click()
    QApplication.processEvents()
    assert panel.current_page == "all"
    assert panel.grp_multitrace.isVisible() is True
    assert panel.grp_unit_selector.isVisible() is True
    assert panel.grp_active_unit_info.isVisible() is True
    assert panel.grp_navigation.isVisible() is True
    assert panel.grp_streaming.isVisible() is True


def test_pinned_settings_panel_and_persistence(qtbot, sample_state):
    win = MdiView(state=sample_state)
    qtbot.addWidget(win)
    win.show()

    v1 = win.newWindow("MultiTrace")
    v2 = win.newWindow("SingleTraceView")
    QApplication.processEvents()

    panel = win.settings_panel
    panel.refresh_target_views()

    # Pin via pin_to_view(v1)
    panel.pin_to_view(v1)
    assert panel.is_pinned is True
    assert panel.pinned_view is v1
    assert panel.active_view is v1
    assert panel.btn_pin_view.isChecked() is True
    assert panel.combo_target_view.currentData() is v1

    # Selecting v2 via setActiveView or notify_view_selected does NOT switch while pinned
    panel.setActiveView(v2)
    assert panel.active_view is v1
    notify_view_selected(v2, ensure_panel_visible=False)
    assert panel.active_view is v1

    # Unpin restores focus-following
    panel.unpin_view()
    assert panel.is_pinned is False
    assert panel.pinned_view is None
    assert panel.btn_pin_view.isChecked() is False
    assert panel.combo_target_view.currentIndex() == 0

    panel.setActiveView(v2)
    assert panel.active_view is v2

    # Pin via btn_pin_view toggle
    panel.btn_pin_view.setChecked(True)
    assert panel.is_pinned is True
    assert panel.pinned_view is v2
    panel.btn_pin_view.setChecked(False)
    assert panel.is_pinned is False
    assert panel.pinned_view is None

    # Pin via combo_target_view selection
    v1_idx = -1
    for i in range(panel.combo_target_view.count()):
        if panel.combo_target_view.itemData(i) is v1:
            v1_idx = i
            break
    assert v1_idx > 0
    panel.combo_target_view.setCurrentIndex(v1_idx)
    assert panel.is_pinned is True
    assert panel.pinned_view is v1

    # Persist page and pin state via get_settings() / set_settings()
    panel.set_settings_page("nav")
    saved = panel.get_settings()
    assert saved["current_page"] == "nav"
    assert saved["is_pinned"] is True

    # Reset panel state and restore from saved settings
    panel.unpin_view()
    panel.set_settings_page("view")
    panel.setActiveView(v2)
    assert panel.current_page == "view"
    assert panel.is_pinned is False

    panel.set_settings(saved)
    assert panel.current_page == "nav"
    assert panel.is_pinned is True
    assert panel.pinned_view is v1
    assert panel.active_view is v1

    win.close()
    QApplication.processEvents()


def test_multiple_split_settings_panels_in_mdi(qtbot, sample_state):
    win = MdiView(state=sample_state)
    qtbot.addWidget(win)
    win.show()

    mt = win.newWindow("MultiTrace")
    st = win.newWindow("SingleTraceView")
    QApplication.processEvents()

    pinned_panel = win.new_settings_panel(target_view=mt, pinned=True, page="view")
    channels_panel = win.new_settings_panel(page="channels")
    QApplication.processEvents()

    assert pinned_panel.is_pinned is True
    assert pinned_panel.pinned_view is mt
    assert pinned_panel.current_page == "view"

    assert channels_panel.is_pinned is False
    assert channels_panel.current_page == "channels"

    # Select st; main & unpinned channels panel follow st, pinned panel stays on mt
    notify_view_selected(st, ensure_panel_visible=False)
    QApplication.processEvents()

    assert win.settings_panel.active_view is st
    assert channels_panel.active_view is st
    assert channels_panel.current_page == "channels"
    assert pinned_panel.active_view is mt
    assert pinned_panel.pinned_view is mt

    win.close()
    QApplication.processEvents()


def test_streaming_settings_integration(qtbot, sample_state):
    panel = ViewSettingsPanel(state=sample_state)
    live_view = OpenEphysLiveView(state=sample_state)
    qtbot.addWidget(panel)
    qtbot.addWidget(live_view)
    panel.show()
    live_view.show()

    worker = OpenEphysStreamWorker(
        rec_channel=0,
        active_channels=[0, 1, 2, 3],
        window_ms=500.0,
        pre_stim_ms=0.0,
        filter_low=500.0,
        filter_high=7000.0,
    )
    live_view.worker = worker

    panel.set_settings_page("stream")
    QApplication.processEvents()
    assert panel.grp_streaming.isVisible() is True

    # 1. Change controls in ViewSettingsPanel -> updates state, OpenEphysLiveView, and worker
    panel.spin_stream_rec_chan.setValue(2)
    panel.edit_stream_active_chans.setText("1-3, 5")
    panel.edit_stream_active_chans.editingFinished.emit()
    panel.spin_stream_window_ms.setValue(750.0)
    panel.spin_stream_pre_ms.setValue(25.0)

    analog_idx = panel.combo_stream_trig_source.findData("analog")
    assert analog_idx >= 0
    panel.combo_stream_trig_source.setCurrentIndex(analog_idx)

    panel.spin_stream_ttl_thresh.setValue(3.25)
    panel.spin_stream_filter_low.setValue(300.0)
    panel.spin_stream_filter_high.setValue(5000.0)
    panel.chk_stream_auto_align.setChecked(False)
    panel.chk_stream_disk_cache.setChecked(False)
    panel.spin_stream_refresh_ms.setValue(50)
    QApplication.processEvents()

    cfg = sample_state.get_live_config()
    assert cfg["rec_chan"] == 2
    assert cfg["active_channels"] == "1-3, 5"
    assert cfg["window_ms"] == 750.0
    assert cfg["pre_ms"] == 25.0
    assert cfg["trigger_source"] == "analog"
    assert cfg["ttl_threshold_v"] == pytest.approx(3.25)
    assert cfg["filter_low_hz"] == pytest.approx(300.0)
    assert cfg["filter_high_hz"] == pytest.approx(5000.0)
    assert cfg["auto_align_clock"] is False
    assert cfg["live_disk_cache"] is False
    assert cfg["ui_refresh_rate_ms"] == 50
    assert sample_state.live_disk_cache is False
    assert sample_state.get_ui_refresh_rate_ms() == 50

    # OpenEphysLiveView widgets updated
    assert live_view.spin_rec_chan.value() == 2
    assert live_view.edit_active_channels.text() == "1-3, 5"
    assert live_view.spin_window_ms.value() == pytest.approx(750.0)
    assert live_view.spin_pre_ms.value() == pytest.approx(25.0)
    assert live_view.combo_trigger_source.currentIndex() == 1
    assert "300" in live_view.chk_online_filter.text()
    assert "5000" in live_view.chk_online_filter.text()

    # Attached OpenEphysStreamWorker updated
    assert worker.rec_channel == 2
    assert worker.active_channels == [1, 2, 3, 5]
    assert worker.window_ms == pytest.approx(750.0)
    assert worker.pre_stim_ms == pytest.approx(25.0)
    assert worker.trigger_source == "analog"
    assert worker.use_analog_trigger is True
    assert worker.ttl_threshold_v == pytest.approx(3.25)
    assert worker.ttl_detector.threshold == pytest.approx(3.25)
    assert worker.filter_low == pytest.approx(300.0)
    assert worker.filter_high == pytest.approx(5000.0)
    assert worker.auto_align_clock is False

    # 2. Change controls in OpenEphysLiveView -> updates ViewSettingsPanel widgets
    live_view.spin_rec_chan.setValue(3)
    live_view.edit_active_channels.setText("0, 2-4")
    live_view.edit_active_channels.editingFinished.emit()
    live_view.spin_window_ms.setValue(400.0)
    live_view.spin_pre_ms.setValue(10.0)
    live_view.combo_trigger_source.setCurrentIndex(0)  # TTL
    live_view.chk_online_filter.setChecked(False)
    live_view.edit_host.setText("192.168.1.50")
    live_view.edit_host.editingFinished.emit()
    live_view.spin_data_port.setValue(6000)
    QApplication.processEvents()

    assert panel.spin_stream_rec_chan.value() == 3
    assert panel.edit_stream_active_chans.text() == "0, 2-4"
    assert panel.spin_stream_window_ms.value() == pytest.approx(400.0)
    assert panel.spin_stream_pre_ms.value() == pytest.approx(10.0)
    assert panel.combo_stream_trig_source.currentData() == "ttl"
    assert panel.chk_stream_filter.isChecked() is False
    assert panel.edit_stream_host.text() == "192.168.1.50"
    assert panel.spin_stream_data_port.value() == 6000

    live_view.worker = None
