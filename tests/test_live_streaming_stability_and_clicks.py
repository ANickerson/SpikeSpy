"""Test suite verifying live streaming stability under high stimulation frequency
and aggressive UI click / zoom interactions across all PyQtGraph and Matplotlib views.
"""

from __future__ import annotations

import time
import numpy as np
import pytest
import quantities as pq
from PySide6.QtCore import Qt, QPoint, QPointF
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication

import neo

from spikespy.ViewerState import ViewerState, tracked_neuron_unit
from spikespy.MultiTraceView import MultiTraceView
from spikespy.SingleTraceView import SingleTraceView
from spikespy.MultiTraceFixedView import MultiTraceFixedView
from spikespy.UnitFootprintView import UnitFootprintView
from spikespy.MultiTracePGView import MultiTracePGView
from spikespy.SingleTracePGView import SingleTracePGView
from spikespy.MultiTraceFixedPGView import MultiTraceFixedPGView
from spikespy.UnitPGView import UnitPGView
from spikespy.UnitFootprintPGView import UnitFootprintPGView
from spikespy.ChannelGroupContainer import ChannelGroupContainerView
from spikespy.tracking_worker import LiveTrackingManager


class DummyClickEvent:
    """Mock for Matplotlib click events."""
    def __init__(self, xdata=None, ydata=None, inaxes=None, button=1):
        self.xdata = xdata
        self.ydata = ydata
        self.inaxes = inaxes
        self.button = button


class DummyPGSceneClickEvent:
    """Mock for PyQtGraph scene mouse click events."""
    def __init__(self, scene_pos: QPointF, button=Qt.LeftButton):
        self._pos = scene_pos
        self._button = button
        self._accepted = False

    def scenePos(self):
        return self._pos

    def button(self):
        return self._button

    def accept(self):
        self._accepted = True


@pytest.fixture()
def populated_state():
    """Returns a ViewerState preloaded with 20 sweeps, 2 units, and 2 channels."""
    state = ViewerState()
    state.start_live_session(sampling_rate=10000.0, window_size_ms=50.0, units=pq.uV)

    # Add 2 tracked units
    unit0 = tracked_neuron_unit(event=neo.Event(times=[] * pq.s, name="unit_0"), channel="CH0", idx_arr=[])
    unit1 = tracked_neuron_unit(event=neo.Event(times=[] * pq.s, name="unit_1"), channel="CH1", idx_arr=[])
    state.spike_groups = [unit0, unit1]

    for i in range(20):
        t_sec = i * 0.05
        # 500 samples per sweep
        w_ch0 = (np.sin(np.linspace(0, 4 * np.pi, 500)) * 40.0).astype(np.float32)
        w_ch1 = (np.cos(np.linspace(0, 4 * np.pi, 500)) * 30.0).astype(np.float32)
        state.append_sweep(t_sec, w_ch0, aux_sweeps={"CH1": w_ch1})
        state.setUnit(0, i, 120)  # unit 0 at sample 120
        state.setUnit(1, i, 280)  # unit 1 at sample 280

    yield state
    state.stop_live_session()


def test_rapid_streaming_tracking_manager_no_starvation(populated_state):
    """Verify that rapid sweep updates (simulating 50-100 Hz) don't starve LiveTrackingManager."""
    mgr = LiveTrackingManager()
    populated_state.spike_groups[0].auto_track = True

    # Simulate rapid sweeps arriving at ~100 Hz
    for i in range(20, 80):
        t_sec = i * 0.01
        w = (np.random.randn(500) * 10.0).astype(np.float32)
        populated_state.append_sweep(t_sec, w)
        mgr.track_stimulus_async(populated_state, i)

    # In-flight task count must be bounded by throttling (<= 12)
    assert mgr._active_task_count <= 12


def test_multitrace_view_click_edge_cases(qtbot, populated_state):
    """Verify MultiTraceView (Matplotlib) handles clicks outside axes, None, NaN, and negative coordinates."""
    view = MultiTraceView(state=populated_state)
    qtbot.addWidget(view)
    view.show()
    qtbot.waitExposed(view)

    # 1. Normal click inside axes
    ev_normal = DummyClickEvent(xdata=150.0, ydata=5.0, inaxes=view.ax)
    view.view_clicked(ev_normal)
    assert populated_state.stimno == 5

    # 2. Click outside axes (inaxes is None or different)
    ev_outside = DummyClickEvent(xdata=None, ydata=None, inaxes=None)
    view.view_clicked(ev_outside)  # Must not raise TypeError: type NoneType doesn't define __round__

    # 3. Click with NaN
    ev_nan = DummyClickEvent(xdata=np.nan, ydata=np.nan, inaxes=view.ax)
    view.view_clicked(ev_nan)

    # 4. Out of bounds ydata
    ev_neg = DummyClickEvent(xdata=100.0, ydata=-10.0, inaxes=view.ax)
    view.view_clicked(ev_neg)
    ev_overflow = DummyClickEvent(xdata=100.0, ydata=9999.0, inaxes=view.ax)
    view.view_clicked(ev_overflow)

    # 5. Right click (should be ignored)
    ev_right = DummyClickEvent(xdata=150.0, ydata=2.0, inaxes=view.ax, button=3)
    view.view_clicked(ev_right)


def test_singletrace_view_click_edge_cases(qtbot, populated_state):
    """Verify SingleTraceView (Matplotlib) handles out-of-bounds, negative, and None coordinates."""
    view = SingleTraceView(state=populated_state)
    qtbot.addWidget(view)
    view.show()
    qtbot.waitExposed(view)

    # 1. Normal click
    ev_normal = DummyClickEvent(xdata=0.015, ydata=10.0, inaxes=view.ax)
    view.view_clicked(ev_normal)

    # 2. Click outside axes / None
    ev_none = DummyClickEvent(xdata=None, ydata=None, inaxes=None)
    view.view_clicked(ev_none)

    # 3. Click negative xdata
    ev_neg = DummyClickEvent(xdata=-0.05, ydata=0.0, inaxes=view.ax)
    view.view_clicked(ev_neg)

    # 4. Click far beyond duration
    ev_large = DummyClickEvent(xdata=999.0, ydata=0.0, inaxes=view.ax)
    view.view_clicked(ev_large)


def test_unit_footprint_view_toolbar_zoom_during_streaming(qtbot, populated_state):
    """Verify that UnitFootprintView does not crash when live sweeps arrive during an active zoom/pan."""
    view = UnitFootprintView(state=populated_state)
    qtbot.addWidget(view)
    view.show()
    qtbot.waitExposed(view)

    # Simulate user activating Zoom mode on the matplotlib toolbar
    view.matplotlib_toolbar.mode = "zoom rect"

    # Rapid sweeps arrive
    for i in range(20, 25):
        w = (np.random.randn(500) * 10.0).astype(np.float32)
        populated_state.append_sweep(i * 0.05, w)
        # Calling _do_update_plot directly or via timer should safely defer clearing
        view._do_update_plot()

    # Disengage toolbar mode
    view.matplotlib_toolbar.mode = ""
    view._do_update_plot()
    assert len(view.figure.axes) > 0


def test_unit_footprint_pg_view_no_signal_leak_and_click(qtbot, populated_state):
    """Verify UnitFootprintPGView render ticks do not cause errors and handles click edge cases."""
    view = UnitFootprintPGView(state=populated_state)
    qtbot.addWidget(view)
    view.show()
    qtbot.waitExposed(view)

    # Trigger multiple render cycles - verify stable execution without accumulating connections or crashing
    for _ in range(5):
        view.render_views()
        view._on_render_timer_tick()

    # Simulate clicks
    ev_valid = DummyPGSceneClickEvent(QPointF(50.0, 50.0), button=Qt.LeftButton)
    view._on_scene_clicked(ev_valid)

    ev_right = DummyPGSceneClickEvent(QPointF(50.0, 50.0), button=Qt.RightButton)
    view._on_scene_clicked(ev_right)


def test_multitrace_pg_view_click_edge_cases(qtbot, populated_state):
    """Verify MultiTracePGView handles clicks with NaN, negative, out of bounds, and during selector mode."""
    view = MultiTracePGView(state=populated_state)
    qtbot.addWidget(view)
    view.show()
    qtbot.waitExposed(view)

    # 1. Normal click
    scene_pt = view.main_plot.vb.mapViewToScene(QPointF(0.02, 7.0))
    ev_normal = DummyPGSceneClickEvent(scene_pt, button=Qt.LeftButton)
    view._on_plot_clicked(ev_normal)
    assert populated_state.stimno == 7

    # 2. Click with out of bounds y
    scene_pt_oob = view.main_plot.vb.mapViewToScene(QPointF(0.02, 9999.0))
    ev_oob = DummyPGSceneClickEvent(scene_pt_oob, button=Qt.LeftButton)
    view._on_plot_clicked(ev_oob)
    # Stimno should remain 7
    assert populated_state.stimno == 7

    # 3. Selector mode clicks
    view.selector_active = True
    view.selector_points = []
    view._on_plot_clicked(ev_normal)
    assert len(view.selector_points) == 1
    # Right-click undo in selector mode
    ev_right = DummyPGSceneClickEvent(scene_pt, button=Qt.RightButton)
    view._on_plot_clicked(ev_right)
    assert len(view.selector_points) == 0


def test_singletrace_pg_view_click_edge_cases(qtbot, populated_state):
    """Verify SingleTracePGView handles negative coordinate clicks without negative latency bugs."""
    view = SingleTracePGView(state=populated_state)
    qtbot.addWidget(view)
    view.show()
    qtbot.waitExposed(view)

    # 1. Normal click
    scene_pt = view.main_plot.vb.mapViewToScene(QPointF(0.012, 10.0))
    ev_normal = DummyPGSceneClickEvent(scene_pt, button=Qt.LeftButton)
    view._on_plot_clicked(ev_normal)

    # 2. Click negative time coordinates
    scene_pt_neg = view.main_plot.vb.mapViewToScene(QPointF(-0.02, 0.0))
    ev_neg = DummyPGSceneClickEvent(scene_pt_neg, button=Qt.LeftButton)
    view._on_plot_clicked(ev_neg)

    # 3. Click past duration
    scene_pt_large = view.main_plot.vb.mapViewToScene(QPointF(5.0, 0.0))
    ev_large = DummyPGSceneClickEvent(scene_pt_large, button=Qt.LeftButton)
    view._on_plot_clicked(ev_large)


def test_multitrace_fixed_pg_view_click_and_culling(qtbot, populated_state):
    """Verify MultiTraceFixedPGView click bounds validation and view culling under rapid sweeps."""
    view = MultiTraceFixedPGView(state=populated_state)
    qtbot.addWidget(view)
    view.show()
    qtbot.waitExposed(view)

    # 1. Click outside line range (e.g. y = -5 or y = 50)
    scene_pt_below = view.main_plot.vb.mapViewToScene(QPointF(0.01, -3.0))
    ev_below = DummyPGSceneClickEvent(scene_pt_below, button=Qt.LeftButton)
    view._on_click(ev_below)

    scene_pt_above = view.main_plot.vb.mapViewToScene(QPointF(0.01, 50.0))
    ev_above = DummyPGSceneClickEvent(scene_pt_above, button=Qt.LeftButton)
    view._on_click(ev_above)

    # 2. View culling test: when hidden, new sweeps flag dirty without rendering
    view.hide()
    assert view.is_view_culled() is True
    view._on_new_sweep_core(21)
    assert view._dirty is True

    # Showing re-renders dirty state
    view.show()
    assert view._dirty is False


def test_unit_pg_view_click_and_culling(qtbot, populated_state):
    """Verify UnitPGView distance-based click handling and culled background execution."""
    view = UnitPGView(state=populated_state)
    qtbot.addWidget(view)
    view.show()
    qtbot.waitExposed(view)

    # Click on snippet area
    scene_pt = view.main_plot.vb.mapViewToScene(QPointF(0.0, 0.0))
    ev = DummyPGSceneClickEvent(scene_pt, button=Qt.LeftButton)
    view._on_click(ev)

    # View culling test
    view.hide()
    assert view.is_view_culled() is True
    view._on_new_sweep_core()
    assert view._dirty is True

    view.show()
    assert view._dirty is False


def test_channel_group_container_live_streaming_stability(qtbot, populated_state):
    """Verify ChannelGroupContainer hosting nested sub-views under high sweep arrival rates."""
    container = ChannelGroupContainerView(state=populated_state)
    qtbot.addWidget(container)
    container.show()
    qtbot.waitExposed(container)

    # Add both Matplotlib and PyQtGraph views into container
    v_mt = container.add_view_by_key("MultiTracePG")
    v_st = container.add_view_by_key("SingleTracePG")
    v_mtf = container.add_view_by_key("MultiTraceFixedPG")
    v_u = container.add_view_by_key("UnitPG")
    assert len(container.sub_views) == 4

    # Enable follow live inputs
    container.set_follow_live(True)
    assert container.is_following_live() is True

    # Simulate 30 rapid live sweeps at ~50 Hz
    for i in range(20, 50):
        t_sec = i * 0.02
        w0 = (np.random.randn(500) * 15.0).astype(np.float32)
        w1 = (np.random.randn(500) * 10.0).astype(np.float32)
        populated_state.append_sweep(t_sec, w0, aux_sweeps={"CH1": w1})
        container._on_new_sweep(i)

    # Process events and verify container and children remain stable
    QApplication.processEvents()
    assert container.isVisible()
