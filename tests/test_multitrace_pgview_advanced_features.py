import numpy as np
import pytest
import quantities as pq
from PySide6.QtCore import Qt, QPointF
from PySide6.QtWidgets import QApplication

from spikespy.MultiTracePGView import MultiTracePGView
from spikespy.ViewerState import ViewerState, tracked_neuron_unit
import neo


class DummyPGSceneClickEvent:
    def __init__(self, scene_pos, button=Qt.LeftButton):
        self._pos = scene_pos
        self._button = button

    def button(self):
        return self._button

    def scenePos(self):
        return self._pos


@pytest.fixture()
def multichannel_state():
    state = ViewerState()
    sampling_rate = 10000 * pq.Hz
    n_samples = 30000

    sig1 = neo.AnalogSignal(np.random.randn(n_samples) * 0.1, units=pq.mV, sampling_rate=sampling_rate, name="CH1")
    sig2 = neo.AnalogSignal(np.random.randn(n_samples) * 0.1, units=pq.mV, sampling_rate=sampling_rate, name="CH2")

    ev_times = np.arange(0.1, 2.5, 0.2) * pq.s  # 12 stimuli: 0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3
    ev = neo.Event(ev_times, name="stim")

    seg = neo.Segment("multich_seg")
    seg.analogsignals.extend([sig1, sig2])
    seg.events.append(ev)
    state.segment = seg

    # Unit 0 at latency ~15 ms (150 samples)
    u0_times = ev_times + 0.015 * pq.s
    u0_ev = neo.Event(u0_times, name="unit_0")
    sg0 = tracked_neuron_unit(event=u0_ev)

    # Unit 1 at latency ~40 ms (400 samples)
    u1_times = ev_times + 0.040 * pq.s
    u1_ev = neo.Event(u1_times, name="unit_1")
    sg1 = tracked_neuron_unit(event=u1_ev)

    state.set_data(analog_signal=sig1, event_signal=ev, spike_groups=[sg0, sg1])
    return state


def test_multitrace_pgview_fixed_height_sweep_mode(qtbot, synthetic_state):
    """Test fixed height toggle, pre/post sweep spinners, and automatic tracking in sweep mode."""
    view = MultiTracePGView(state=synthetic_state)
    qtbot.addWidget(view)
    view.show()
    qtbot.waitExposed(view)

    # Default is not fixed height
    assert view.fixed_height is False
    assert view.y_axis_mode == "sweep"

    # Enable fixed height
    view.set_settings({"fixed_height": True, "n_lines_pre": 3, "n_lines_post": 3})
    assert view.fixed_height is True
    assert view.n_lines_pre == 3
    assert view.n_lines_post == 3

    # Select stimulus #5
    synthetic_state.setStimNo(5)
    view.update_ylim(5)
    y_range = view.main_plot.viewRange()[1]

    # Pre = 3, Post = 3, center = 5: range should span roughly [5 - 3, 5 + 3]
    start = max(5 - 3, 0)
    stop = min(start + 3 + 3 + 1, len(synthetic_state.event_signal))
    assert np.isclose(y_range[0], start - 0.5)
    assert np.isclose(y_range[1], stop - 0.5)

    # Settings round-trip
    settings = view.get_settings()
    assert settings["fixed_height"] is True
    assert settings["n_lines_pre"] == 3
    assert settings["n_lines_post"] == 3


def test_multitrace_pgview_fixed_height_time_mode_and_click(qtbot, synthetic_state):
    """Test fixed height in time mode with physical time Y-axis and click-to-select timestamp."""
    view = MultiTracePGView(state=synthetic_state)
    qtbot.addWidget(view)
    view.show()
    qtbot.waitExposed(view)

    # Switch to Time mode with fixed height window of 1.0 s
    view.set_settings({
        "fixed_height": True,
        "y_axis_mode": "time",
        "time_window_s": 0.8,
        "post_seconds": 0.2,
    })
    view.setup_figure()

    assert view.y_axis_mode == "time"
    assert view.main_plot.getAxis("left").labelText == "Time"

    # Current stim is 0 (t = 0.1s)
    # Move to stimulus #4 (t = 0.9s)
    synthetic_state.setStimNo(4)
    view.update_ylim(4)

    y_range = view.main_plot.viewRange()[1]
    # Y-range top should cover -0.8s
    assert y_range[0] <= -0.8
    assert y_range[1] >= 0.2

    # Click interaction in Time mode:
    # At t_selected = 0.9s: stimulus #3 was at 0.7s (offset = -0.2s)
    scene_pt = view.main_plot.vb.mapViewToScene(QPointF(50.0, -0.2))
    ev = DummyPGSceneClickEvent(scene_pt, button=Qt.LeftButton)
    view._on_plot_clicked(ev)
    assert synthetic_state.stimno == 3


def test_multitrace_pgview_line_rendering_upgrades(qtbot, synthetic_state):
    """Test that lines mode has scale multiplier, purple active curve, red spike snippets, and latency dots."""
    view = MultiTracePGView(state=synthetic_state)
    qtbot.addWidget(view)
    view.show()
    qtbot.waitExposed(view)

    view.mode = "lines"
    view.scale = 2.5
    view.scale_mode = "std"
    synthetic_state.setStimNo(4)
    view.setup_figure()

    panel = view.channel_panels[0]
    assert panel.lines_curve.isVisible()
    assert panel.active_sweep_curve.isVisible()
    assert panel.highlight_segments.isVisible()
    assert panel.points_scatter.isVisible()

    # Active sweep curve should contain waveform points
    active_y = panel.active_sweep_curve.yData
    assert active_y is not None and len(active_y) > 0

    # Highlight segments should contain spike snippet coordinates
    hl_y = panel.highlight_segments.yData
    assert hl_y is not None and len(hl_y) > 0

    # Points scatter should contain the spike latency marker
    scat_pts = panel.points_scatter.points()
    assert len(scat_pts) > 0


def test_multitrace_pgview_detach_x_axis(qtbot, multichannel_state):
    """Test detaching X-axis globally and per-channel so channels can zoom independently."""
    view = MultiTracePGView(state=multichannel_state)
    qtbot.addWidget(view)
    view.show()
    qtbot.waitExposed(view)

    # Set 2 channels
    view.set_group_size_n(2)
    assert len(view.channel_panels) == 2

    p0 = view.channel_panels[0]
    p1 = view.channel_panels[1]

    # By default, channels are X-linked
    assert view.link_x_axis is True
    assert p1.plot.getViewBox().linkedView(0) is p0.plot.getViewBox()

    # Detach channel 1 independently
    p1.x_linked = False
    view._update_axis_linking()
    assert p1.plot.getViewBox().linkedView(0) is None

    # Zoom channel 1 independently to [100, 200]
    p1.plot.setXRange(100, 200, padding=0)
    p0.plot.setXRange(300, 600, padding=0)

    r0 = p0.plot.viewRange()[0]
    r1 = p1.plot.viewRange()[0]
    assert np.isclose(r0[0], 300) and np.isclose(r0[1], 600)
    assert np.isclose(r1[0], 100) and np.isclose(r1[1], 200)

    # Test global Link X toggle
    view.set_settings({"link_x_axis": False})
    assert view.link_x_axis is False
    assert p1.plot.getViewBox().linkedView(0) is None


def test_multitrace_pgview_pin_to_unit_latency_and_window_groups(qtbot, multichannel_state):
    """Test pinning channel screens to specific unit latencies and window groups."""
    view = MultiTracePGView(state=multichannel_state)
    qtbot.addWidget(view)
    view.show()
    qtbot.waitExposed(view)

    view.set_group_size_n(2)
    p0 = view.channel_panels[0]
    p1 = view.channel_panels[1]

    # Pin channel 0 to Unit 0 (latency ~15 ms = 150 samples)
    # Pin channel 1 to Unit 1 (latency ~40 ms = 400 samples)
    view._pin_channel_to_unit(0, 0)
    p0.window_width_ms = 10.0  # 10 ms = 100 samples (range: [100, 200])

    view._pin_channel_to_unit(1, 1)
    p1.window_width_ms = 20.0  # 20 ms = 200 samples (range: [300, 500])

    view._apply_pinned_unit_windows()

    r0 = p0.plot.viewRange()[0]
    r1 = p1.plot.viewRange()[0]

    # Channel 0 centered around sample ~150
    center_0 = (r0[0] + r0[1]) / 2.0
    assert np.isclose(center_0, 150.0, atol=2.0)
    assert np.isclose(r0[1] - r0[0], 100.0, atol=2.0)

    # Channel 1 centered around sample ~400
    center_1 = (r1[0] + r1[1]) / 2.0
    assert np.isclose(center_1, 400.0, atol=2.0)
    assert np.isclose(r1[1] - r1[0], 200.0, atol=2.0)

    # Test Window Group assignment
    view._set_channel_window_group(0, "Group A")
    view._set_channel_window_group(1, "Group A")
    view._update_axis_linking()
    # Pinned channels remain independent, but unpinning links them within Group A
    view._pin_channel_to_unit(0, None)
    view._pin_channel_to_unit(1, None)
    assert p1.plot.getViewBox().linkedView(0) is p0.plot.getViewBox()

    # Settings round-trip test
    settings = view.get_settings()
    assert "pinned_units" in settings
    assert "window_groups" in settings
    assert "window_widths" in settings

    view.set_settings(settings)
