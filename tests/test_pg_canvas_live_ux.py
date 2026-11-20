import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from types import SimpleNamespace
import numpy as np
import neo
import quantities as pq
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from spikespy.ViewerState import ViewerState, tracked_neuron_unit
from spikespy.container_view_state import ContainerViewerState
from spikespy.SingleTracePGView import SingleTracePGView
from spikespy.MultiTracePGView import MultiTracePGView


@pytest.fixture
def multichannel_uV_state():
    state = ViewerState()
    seg = neo.Segment(name="LiveUXTest")
    sr = 10000.0 * pq.Hz
    n_samples = 500
    n_sweeps = 6
    times = np.arange(n_sweeps) * 0.05 * pq.s
    event_sig = neo.Event(times, name="stim", units=pq.s)
    seg.events.append(event_sig)

    for ch_idx in range(2):
        ch_name = f"CH{ch_idx}"
        w = np.zeros((n_sweeps, n_samples), dtype=np.float32)
        if ch_idx == 0:
            w[:, 120] = 80.0  # 80 uV = 0.08 mV peak at sample 120
            w[:, 260] = 120.0  # 120 uV = 0.12 mV peak at sample 260
        else:
            w[:, 260] = 150.0
        sig = neo.AnalogSignal(w.flatten(), units=pq.uV, sampling_rate=sr, name=ch_name)
        seg.analogsignals.append(sig)

    evt0 = neo.Event(times + (120 / 10000.0) * pq.s, name="unit_0", units=pq.s)
    evt0.annotations["channel"] = "CH0"
    u0 = tracked_neuron_unit(
        event=evt0,
        channel="CH0",
        tracking_threshold_mv=0.05,
        tracking_window_ms=4.0,
    )

    state.segment = seg
    state.window_size = 0.05 * pq.s
    state.set_data(
        analog_signal=seg.analogsignals[0],
        event_signal=event_sig,
        spike_groups=[u0],
    )
    return state


def test_singletrace_pg_threshold_line_and_spinbox(qtbot, multichannel_uV_state):
    view = SingleTracePGView(state=multichannel_uV_state)
    qtbot.addWidget(view)
    view.show()
    QApplication.processEvents()

    sg = multichannel_uV_state.spike_groups[0]
    view._update_tracking_controls(sg)

    # 0.05 mV in uV signal display units == 50.0 uV
    assert view.threshold_line.isVisible() is True
    assert view.track_threshold_spin.value() == pytest.approx(0.05)
    assert float(view.threshold_line.value()) == pytest.approx(50.0)

    emitted_params = []
    multichannel_uV_state.onUnitParamsChanged.connect(lambda idx: emitted_params.append(idx))

    # Drag threshold_line to 75 uV -> 0.075 mV
    view.threshold_line.setValue(75.0)
    view._on_threshold_line_dragged()
    assert sg.tracking_threshold_mv == pytest.approx(0.075)
    assert sg.event.annotations["tracking_threshold_mv"] == pytest.approx(0.075)
    assert view.track_threshold_spin.value() == pytest.approx(0.075)
    assert emitted_params and emitted_params[-1] == 0

    # Edit spinbox to 0.09 mV -> threshold_line updates to 90 uV
    view.track_threshold_spin.setValue(0.09)
    assert sg.tracking_threshold_mv == pytest.approx(0.09)
    assert sg.event.annotations["tracking_threshold_mv"] == pytest.approx(0.09)
    assert float(view.threshold_line.value()) == pytest.approx(90.0)

    # Hiding search window hides threshold line as well
    view.show_track_window_check.setChecked(False)
    assert view.threshold_line.isVisible() is False


def test_live_paused_banner_visibility_and_return_to_live(qtbot, multichannel_uV_state):
    st_view = SingleTracePGView(state=multichannel_uV_state)
    mt_view = MultiTracePGView(state=multichannel_uV_state)
    qtbot.addWidget(st_view)
    qtbot.addWidget(mt_view)
    st_view.show()
    mt_view.show()
    QApplication.processEvents()

    assert st_view.live_paused_banner.isVisible() is False
    assert mt_view.live_paused_banner.isVisible() is False

    # Navigating to a historical sweep sets follow_live=False and displays both banners
    multichannel_uV_state.setStimNo(2, user_set=True)
    QApplication.processEvents()
    assert multichannel_uV_state.follow_live is False
    assert st_view.live_paused_banner.isVisible() is True
    assert mt_view.live_paused_banner.isVisible() is True
    assert "Sweep #2" in st_view.lbl_live_paused.text()
    assert "Sweep #2" in mt_view.lbl_live_paused.text()

    # Clicking Return to Live restores follow_live=True and hides banners
    st_view.btn_return_live.click()
    QApplication.processEvents()
    assert multichannel_uV_state.follow_live is True
    assert st_view.live_paused_banner.isVisible() is False
    assert mt_view.live_paused_banner.isVisible() is False

    # Pause via mt_view then click mt_view's Return to Live button
    multichannel_uV_state.set_follow_live(False)
    QApplication.processEvents()
    assert mt_view.live_paused_banner.isVisible() is True
    mt_view.btn_return_live.click()
    QApplication.processEvents()
    assert multichannel_uV_state.follow_live is True
    assert mt_view.live_paused_banner.isVisible() is False
    assert st_view.live_paused_banner.isVisible() is False


def test_seed_new_unit_at_pos_singletrace_and_multitrace(qtbot, multichannel_uV_state):
    st_view = SingleTracePGView(state=multichannel_uV_state)
    mt_view = MultiTracePGView(state=multichannel_uV_state)
    qtbot.addWidget(st_view)
    qtbot.addWidget(mt_view)
    st_view.show()
    mt_view.show()
    QApplication.processEvents()

    # Real ViewerState.seed_unit_at path via SingleTracePGView: snaps to peak at 260
    initial_count = len(multichannel_uV_state.spike_groups)
    new_idx = st_view.seed_new_unit_at_pos(257.0, channel_name="CH0")
    assert new_idx == initial_count
    assert len(multichannel_uV_state.spike_groups) == initial_count + 1
    seeded_sg = multichannel_uV_state.spike_groups[new_idx]
    assert seeded_sg.channel == "CH0"
    assert seeded_sg.auto_track is True
    assert seeded_sg.idx_arr[multichannel_uV_state.stimno][0] == 260

    # Shift+Click event path on SingleTracePGView
    rect = st_view.main_plot.vb.sceneBoundingRect()
    center = rect.center()
    view_pos = st_view.main_plot.vb.mapSceneToView(center)
    sr = 10000.0
    expected_sample = int(round(view_pos.x() * sr))
    fake_ev = SimpleNamespace(
        button=lambda: Qt.LeftButton,
        modifiers=lambda: Qt.ShiftModifier,
        scenePos=lambda: center,
        accept=lambda: None,
    )
    st_view._on_plot_clicked(fake_ev)
    assert len(multichannel_uV_state.spike_groups) == initial_count + 2
    click_sg = multichannel_uV_state.spike_groups[-1]
    assert click_sg.channel == "CH0"
    assert click_sg.auto_track is True
    assert abs(click_sg.idx_arr[multichannel_uV_state.stimno][0] - expected_sample) <= max(
        5, int(st_view.select_local_maxima_width)
    )

    # MultiTracePGView seed_new_unit_at_pos snaps to peak on sweep 3, channel CH1
    mt_idx = mt_view.seed_new_unit_at_pos(256.0, stimno=3, channel_name="CH1")
    assert mt_idx == initial_count + 2
    assert len(multichannel_uV_state.spike_groups) == initial_count + 3
    mt_sg = multichannel_uV_state.spike_groups[mt_idx]
    assert mt_sg.channel == "CH1"
    assert mt_sg.auto_track is True
    assert mt_sg.idx_arr[3][0] == 260

    # Shift+Click on MultiTracePGView panel
    mt_panel = mt_view.channel_panels[0]
    mt_rect = mt_panel.plot.vb.sceneBoundingRect()
    mt_center = mt_rect.center()
    mt_fake_ev = SimpleNamespace(
        button=lambda: Qt.LeftButton,
        modifiers=lambda: Qt.ShiftModifier,
        scenePos=lambda: mt_center,
        accept=lambda: None,
    )
    mt_view._on_plot_clicked(mt_fake_ev)
    assert len(multichannel_uV_state.spike_groups) == initial_count + 4
    assert multichannel_uV_state.spike_groups[-1].auto_track is True

    # Re-anchor active unit (unit 0) at current sweep using real ViewerState.reanchor_unit_at
    multichannel_uV_state.setUnitGroup(0)
    multichannel_uV_state.setStimNo(2, user_set=True)
    mt_view._reanchor_active_unit_at_current_sweep(mt_panel)
    assert multichannel_uV_state.spike_groups[0].idx_arr[2] is not None


def test_multitrace_pg_protocol_markers_overlay(qtbot, multichannel_uV_state):
    multichannel_uV_state.add_protocol_marker("Baseline", stimno=1)
    multichannel_uV_state.add_protocol_marker("2 Hz Stim", stimno=4)

    mt_view = MultiTracePGView(state=multichannel_uV_state)
    qtbot.addWidget(mt_view)
    mt_view.show()
    QApplication.processEvents()

    # 2 markers * (1 InfiniteLine + 1 TextItem badge) = 4 items
    assert len(mt_view.protocol_marker_items) == 4

    # Updating protocol markers via real ViewerState method emits onProtocolMarkersChanged
    multichannel_uV_state.add_protocol_marker("Recovery", stimno=5)
    QApplication.processEvents()
    assert len(mt_view.protocol_marker_items) == 6


def test_apply_shared_visual_settings_sync(qtbot, multichannel_uV_state):
    container_state = ContainerViewerState(multichannel_uV_state, parent=multichannel_uV_state)

    mt_view = MultiTracePGView(state=multichannel_uV_state)
    st_container_view = SingleTracePGView(state=container_state)
    st_shared_view = SingleTracePGView(state=multichannel_uV_state)
    qtbot.addWidget(mt_view)
    qtbot.addWidget(st_container_view)
    qtbot.addWidget(st_shared_view)
    mt_view.show()
    st_container_view.show()
    st_shared_view.show()
    QApplication.processEvents()

    # Direct apply_shared_visual_settings on MultiTracePGView
    mt_view.apply_shared_visual_settings(
        {
            "scale_mode": "mV",
            "color_scale": (0.25, 3.5),
            "trace_scale": 2.25,
            "x_range": (50.0, 300.0),
        }
    )
    assert mt_view.scale_mode == "mV"
    assert mt_view.scale_mode_combo.currentText() == "mV"
    assert mt_view.lowerSpinBox.value() == pytest.approx(0.25)
    assert mt_view.upperSpinBox.value() == pytest.approx(3.5)
    assert mt_view.scale == pytest.approx(2.25)
    assert mt_view.scale_spinner.value() == pytest.approx(2.25)
    xr = mt_view.main_plot.viewRange()[0]
    assert xr[0] == pytest.approx(50.0, abs=1e-6)
    assert xr[1] == pytest.approx(300.0, abs=1e-6)

    # ContainerViewerState.update_visual_settings emits onContainerVisualSettingsChanged -> syncs SingleTracePGView
    container_state.update_visual_settings(
        show_track_window=False,
        show_template=False,
        x_range=(0.01, 0.04),
    )
    QApplication.processEvents()
    assert st_container_view.show_track_window_check.isChecked() is False
    assert st_container_view.show_template_check.isChecked() is False
    assert st_container_view.threshold_line.isVisible() is False
    st_xr = st_container_view.main_plot.viewRange()[0]
    assert st_xr[0] == pytest.approx(0.01, abs=1e-6)
    assert st_xr[1] == pytest.approx(0.04, abs=1e-6)

    # Named shared set broadcast only applies to views bound to that shared set
    st_shared_view.channel_bar.bind_to_shared_set("Shared Group A")
    st_shared_view.apply_shared_visual_settings(
        {"show_track_window": False, "show_template": False}
    )
    multichannel_uV_state.update_shared_visual_settings(
        "Shared Group B",
        show_track_window=True,
        show_template=True,
    )
    QApplication.processEvents()
    assert st_shared_view.show_track_window_check.isChecked() is False

    multichannel_uV_state.update_shared_visual_settings(
        "Shared Group A",
        show_track_window=True,
        show_template=True,
    )
    QApplication.processEvents()
    assert st_shared_view.show_track_window_check.isChecked() is True
    assert st_shared_view.show_template_check.isChecked() is True
