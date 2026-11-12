import time
import numpy as np
import pytest
import quantities as pq
from PySide6.QtCore import Qt, QPoint, QEvent
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication, QWidget, QPushButton

from spikespy.ViewerState import ViewerState
from spikespy.MultiTraceView import MultiTraceView
from spikespy.MultiTracePGView import MultiTracePGView
from spikespy.SingleTraceView import SingleTraceView
from spikespy.ViewSettingsPanel import ViewSettingsPanel
from spikespy.SpikeSpy import MdiView


@pytest.fixture()
def streaming_state():
    state = ViewerState()
    state.start_live_session(sampling_rate=10000.0, window_size_ms=100.0, units=pq.uV)
    yield state
    state.stop_live_session()


def test_cached_axes_image_zoom_during_live_streaming(qtbot, streaming_state):
    """
    Test zooming into a Matplotlib MultiTraceView with CachedAxesImage
    while live streaming sweeps arrive, ensuring no AttributeError or crash occurs.
    """
    view = MultiTraceView(state=streaming_state)
    qtbot.addWidget(view)
    view.show()
    qtbot.waitExposed(view)

    # Stream first few sweeps
    for i in range(5):
        w = np.random.randn(1000).astype(np.float32) * 20.0
        streaming_state.append_sweep(i * 0.1, w)

    # Force a zoom by directly updating ax limits (simulating NavigationToolbar zoom)
    view.ax.set_xlim(200, 600)
    view.ax.set_ylim(4, 1)

    # Trigger a render that invokes CachedAxesImage.make_image
    view.view.draw()

    # Now append more sweeps while zoomed in
    for i in range(5, 10):
        w = np.random.randn(1000).astype(np.float32) * 20.0
        streaming_state.append_sweep(i * 0.1, w)
        view.on_new_sweep(i)

    # Ensure make_image does not crash with NoneType or transformed error
    view.view.draw()

    # View limits in X should remain zoomed, not snapped back to (0, 1)
    xlim = view.ax.get_xlim()
    assert xlim[0] >= 100
    assert xlim[1] <= 700


def test_multitrace_pgview_zoom_preserved_during_live_streaming(qtbot, streaming_state):
    """
    Test that zooming/panning in MultiTracePGView is preserved when sweeps arrive
    and doesn't snap back to full range on every sweep.
    """
    view = MultiTracePGView(state=streaming_state)
    qtbot.addWidget(view)
    view.show()
    qtbot.waitExposed(view)

    # Append initial sweeps
    for i in range(5):
        w = np.random.randn(1000).astype(np.float32) * 20.0
        streaming_state.append_sweep(i * 0.1, w)
        view.on_new_sweep(i)

    # User zooms into a specific range
    view.main_plot.setXRange(250, 450, padding=0)
    view.main_plot.setYRange(1, 3, padding=0)

    # Live sweeps continue arriving
    for i in range(5, 10):
        w = np.random.randn(1000).astype(np.float32) * 20.0
        streaming_state.append_sweep(i * 0.1, w)
        view.on_new_sweep(i)

    # Check that user's X zoom is preserved
    cur_x = view.main_plot.viewRange()[0]
    assert 200 <= cur_x[0] <= 300
    assert 400 <= cur_x[1] <= 500


def test_settings_panel_focus_tracking_ancestor_traversal(qtbot, synthetic_state):
    """
    Test that clicking or focusing canvas/sub-widgets inside MultiTraceView,
    MultiTracePGView, or SingleTraceView updates the ViewSettingsPanel active view.
    """
    panel = ViewSettingsPanel(state=synthetic_state)
    qtbot.addWidget(panel)
    panel.show()

    mt_view = MultiTraceView(state=synthetic_state)
    qtbot.addWidget(mt_view)
    mt_view.show()

    pg_view = MultiTracePGView(state=synthetic_state)
    qtbot.addWidget(pg_view)
    pg_view.show()

    st_view = SingleTraceView(state=synthetic_state)
    qtbot.addWidget(st_view)
    st_view.show()

    # Pass the internal canvas of mt_view
    panel.setActiveView(mt_view.view)
    assert panel.active_view == mt_view
    assert panel.grp_multitrace.isVisible()

    # Pass the internal graphics widget of pg_view
    panel.setActiveView(pg_view.glw)
    assert panel.active_view == pg_view
    assert panel.grp_multitrace.isVisible()

    # Pass the internal canvas of st_view
    panel.setActiveView(st_view.view)
    assert panel.active_view == st_view
    assert panel.grp_singletrace.isVisible()

    # Interacting with a widget inside panel itself should not reset or clear active view
    panel.setActiveView(panel.combo_mt_mode)
    assert panel.active_view == st_view


def test_mouse_press_event_updates_settings_panel_via_main_window(qtbot, synthetic_state):
    """
    Test that clicking anywhere inside a view (even its internal canvas) triggers
    the QApplication event filter in MdiView and sets the active view in settings_panel.
    """
    win = MdiView(state=synthetic_state)
    qtbot.addWidget(win)
    win.show()
    qtbot.waitExposed(win)

    # Create MultiTrace view window
    mt = win.newWindow("MultiTrace")
    # Simulate mouse press event on the canvas widget of MultiTraceView
    event = QMouseEvent(QEvent.MouseButtonPress, QPoint(10, 10), Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
    QApplication.sendEvent(mt.view, event)

    assert win.settings_panel.active_view == mt
    assert win.settings_panel.grp_multitrace.isVisible()


def test_live_streaming_2hz_responsiveness(qtbot, streaming_state):
    """
    Ensure that 2 Hz (and higher) stream sweep arrivals process smoothly without UI freeze.
    """
    mt_view = MultiTraceView(state=streaming_state)
    qtbot.addWidget(mt_view)
    mt_view.show()

    pg_view = MultiTracePGView(state=streaming_state)
    qtbot.addWidget(pg_view)
    pg_view.show()

    start_time = time.perf_counter()
    n_sweeps = 10  # 10 sweeps
    for i in range(n_sweeps):
        w = np.random.randn(1000).astype(np.float32) * 20.0
        streaming_state.append_sweep(i * 0.5, w)  # 2 Hz simulation
        QApplication.processEvents()

    elapsed = time.perf_counter() - start_time
    # Processing 10 sweeps with both views active must comfortably fit within 2 Hz budget (500 ms per sweep = 5.0s)
    assert elapsed < 5.0, f"Live streaming processing took too long: {elapsed:.2f}s for {n_sweeps} sweeps"

