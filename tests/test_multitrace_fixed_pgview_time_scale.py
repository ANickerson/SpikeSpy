import numpy as np
import pytest
import quantities as pq
from PySide6.QtCore import Qt, QPointF
from PySide6.QtWidgets import QApplication

from spikespy.MultiTraceFixedPGView import MultiTraceFixedPGView
from spikespy.ViewSettingsPanel import ViewSettingsPanel
from spikespy.ViewerState import ViewerState


class DummyPGSceneClickEvent:
    def __init__(self, scene_pos, button=Qt.LeftButton):
        self._pos = scene_pos
        self._button = button

    def button(self):
        return self._button

    def scenePos(self):
        return self._pos


@pytest.fixture()
def live_streaming_state():
    state = ViewerState()
    state.start_live_session(sampling_rate=10000.0, window_size_ms=100.0, units=pq.uV)
    yield state
    state.stop_live_session()


def test_multitrace_fixed_time_scale_toggle_and_settings(qtbot, synthetic_state):
    """Test switching MultiTraceFixedPGView between Sweep and Time mode, and get/set_settings."""
    view = MultiTraceFixedPGView(state=synthetic_state)
    qtbot.addWidget(view)
    view.show()
    qtbot.waitExposed(view)

    # Default mode is sweep
    assert view.settings.get("y_axis_mode", "sweep") == "sweep"
    assert view.y_axis_combo.currentIndex() == 0
    assert view.pre_spinner.isVisible()
    assert not view.time_window_spinner.isVisible()

    # Switch to Time mode via set_settings
    view.set_settings({"y_axis_mode": "time", "time_window_s": 4.0})
    assert view.settings["y_axis_mode"] == "time"
    assert view.settings["time_window_s"] == 4.0
    assert view.y_axis_combo.currentIndex() == 1
    assert not view.pre_spinner.isVisible()
    assert view.time_window_spinner.isVisible()
    assert np.isclose(view.time_window_spinner.value(), 4.0)

    # Check axis label and settings round-trip
    assert view.main_plot.getAxis("left").labelText == "Time"
    settings = view.get_settings()
    assert settings["y_axis_mode"] == "time"
    assert settings["time_scale"] is True
    assert np.isclose(settings["time_window_s"], 4.0)
    assert np.isclose(settings["n_seconds"], 4.0)

    # Switch back via combo box
    view.y_axis_combo.setCurrentIndex(0)
    assert view.settings["y_axis_mode"] == "sweep"
    assert view.pre_spinner.isVisible()
    assert not view.time_window_spinner.isVisible()
    assert view.main_plot.getAxis("left").labelText == "Stimulus Sweep"


def test_multitrace_fixed_time_scale_sweeps_and_positions(qtbot, live_streaming_state):
    """Test that in Time mode, traces are placed at physical time offsets from the active stimulus."""
    view = MultiTraceFixedPGView(state=live_streaming_state)
    qtbot.addWidget(view)
    view.show()
    qtbot.waitExposed(view)

    # Append sweeps at known time intervals: 0.0s, 1.0s, 2.5s, 3.0s, 6.0s
    timestamps = [0.0, 1.0, 2.5, 3.0, 6.0]
    for t in timestamps:
        w = np.zeros(1000, dtype=np.float32)
        live_streaming_state.append_sweep(t, w)

    # Select stimulus #3 (t = 3.0s)
    live_streaming_state.setStimNo(3)

    # Enable Time mode with window of 2.5s (should include t = 3.0s, 2.5s, 1.0s; but not 0.0s since 3.0 - 2.5 = 0.5 > 0.0)
    view.set_settings({"y_axis_mode": "time", "time_window_s": 2.5})
    view.update_figure(force=True)

    # Verify displayed sweeps
    # Stimuli within [3.0 - 2.5 = 0.5, 3.0] are indices 1 (1.0s), 2 (2.5s), 3 (3.0s)
    assert view._displayed_stim_indices == [1, 2, 3]

    # Baselines relative to t_selected = 3.0:
    # index 1: 1.0 - 3.0 = -2.0
    # index 2: 2.5 - 3.0 = -0.5
    # index 3: 3.0 - 3.0 = 0.0
    expected_baselines = [-2.0, -0.5, 0.0]
    assert np.allclose(view._displayed_baselines, expected_baselines)

    # Active curve should be at baseline 0.0
    assert view.active_curve.isVisible()
    active_y = view.active_curve.yData
    # Mean of baseline 0 with 0 signal should be 0.0
    assert np.isclose(np.mean(active_y), 0.0)

    # Top of the Y-range should accommodate -2.5 (the window)
    y_range = view.main_plot.viewRange()[1]
    assert y_range[0] <= -2.5
    assert y_range[1] >= 0.0


def test_multitrace_fixed_time_scale_click_interaction(qtbot, live_streaming_state):
    """Test clicking on traces in Time mode selects the closest stimulus in time."""
    view = MultiTraceFixedPGView(state=live_streaming_state)
    qtbot.addWidget(view)
    view.show()
    qtbot.waitExposed(view)

    # Append sweeps at 0s, 2s, 4s, 5s
    for t in [0.0, 2.0, 4.0, 5.0]:
        w = np.zeros(1000, dtype=np.float32)
        live_streaming_state.append_sweep(t, w)

    live_streaming_state.setStimNo(3)  # t = 5.0s
    view.set_settings({"y_axis_mode": "time", "time_window_s": 6.0})
    view.update_figure(force=True)

    # At t_selected = 5.0s:
    # stimulus 0 (0.0s) -> y = -5.0
    # stimulus 1 (2.0s) -> y = -3.0
    # stimulus 2 (4.0s) -> y = -1.0
    # stimulus 3 (5.0s) -> y = 0.0

    # Click near stimulus 1 (y = -3.1)
    scene_pt = view.main_plot.vb.mapViewToScene(QPointF(0.01, -3.1))
    ev = DummyPGSceneClickEvent(scene_pt, button=Qt.LeftButton)
    view._on_click(ev)
    assert live_streaming_state.stimno == 1

    # Click out of bounds (y = -50.0) - should not change stimno
    scene_pt_oob = view.main_plot.vb.mapViewToScene(QPointF(0.01, -50.0))
    ev_oob = DummyPGSceneClickEvent(scene_pt_oob, button=Qt.LeftButton)
    view._on_click(ev_oob)
    assert live_streaming_state.stimno == 1


def test_view_settings_panel_fixed_time_scale_sync(qtbot, synthetic_state):
    """Test that ViewSettingsPanel correctly reads and controls MultiTraceFixedPGView's time scale."""
    view = MultiTraceFixedPGView(state=synthetic_state)
    panel = ViewSettingsPanel(state=synthetic_state)
    qtbot.addWidget(view)
    qtbot.addWidget(panel)
    view.show()
    panel.show()
    qtbot.waitExposed(view)
    qtbot.waitExposed(panel)

    panel.setActiveView(view)
    QApplication.processEvents()

    # Initial mode in panel should be Sweep (index 0)
    assert panel.combo_fix_axis_mode.currentIndex() == 0

    # Change mode via ViewSettingsPanel
    panel.combo_fix_axis_mode.setCurrentIndex(1)  # Time
    QApplication.processEvents()
    assert view.settings["y_axis_mode"] == "time"

    # Change time window via ViewSettingsPanel
    panel.spin_fix_time_window.setValue(15.0)
    QApplication.processEvents()
    assert np.isclose(view.settings["time_window_s"], 15.0)
    assert np.isclose(view.time_window_spinner.value(), 15.0)
