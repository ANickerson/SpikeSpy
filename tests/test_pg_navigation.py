import numpy as np
import pytest
import pyqtgraph as pg
from PySide6.QtCore import Qt, QPoint, QEvent
from PySide6.QtGui import QMouseEvent

from spikespy.ViewerState import ViewerState
from spikespy.MultiTracePGView import MultiTracePGView
from spikespy.SingleTracePGView import SingleTracePGView
from spikespy.MultiTraceFixedPGView import MultiTraceFixedPGView
from spikespy.UnitPGView import UnitPGView
from spikespy.UnitFootprintPGView import UnitFootprintPGView
from spikespy.ViewSettingsPanel import ViewSettingsPanel


def test_pg_views_navigation_modes(qtbot, synthetic_state):
    """
    Test that all PyQtGraph plots support Matplotlib-like navigation modes
    (Normal, Pan, Zoom to Rect), updating ViewBox mouse modes and cursors.
    """
    views = [
        MultiTracePGView(state=synthetic_state),
        SingleTracePGView(state=synthetic_state),
        MultiTraceFixedPGView(state=synthetic_state),
        UnitPGView(state=synthetic_state),
        UnitFootprintPGView(state=synthetic_state),
    ]

    for view in views:
        qtbot.addWidget(view)
        view.show()

        # Initial mode is normal/empty
        assert view.get_nav_mode() == ""
        assert not view.is_nav_active()

        # Enable Pan mode
        view.set_nav_mode("pan")
        assert view.get_nav_mode() == "pan"
        assert view.is_nav_active()
        plot = view.get_primary_nav_plot()
        assert plot is not None
        assert plot.vb.state["mouseMode"] == pg.ViewBox.PanMode
        assert plot.vb.state["mouseEnabled"] == [True, True]
        assert plot.vb.cursor().shape() == Qt.OpenHandCursor

        # Enable Zoom mode (box / rect zoom)
        view.set_nav_mode("zoom")
        assert view.get_nav_mode() == "zoom"
        assert view.is_nav_active()
        assert plot.vb.state["mouseMode"] == pg.ViewBox.RectMode
        assert plot.vb.state["mouseEnabled"] == [True, True]
        assert plot.vb.cursor().shape() == Qt.CrossCursor

        # Return to Normal mode
        view.set_nav_mode("")
        assert view.get_nav_mode() == ""
        assert not view.is_nav_active()
        assert plot.vb.state["mouseEnabled"] == [False, False]
        assert plot.vb.cursor().shape() == Qt.ArrowCursor


def test_clicks_suppressed_when_nav_active(qtbot, synthetic_state):
    """
    Verify that plot click actions (e.g. changing active sweep in MultiTracePGView
    or setting spike latency in SingleTracePGView) are suppressed when Pan or Zoom
    is active, matching Matplotlib toolbar behavior.
    """
    mt_view = MultiTracePGView(state=synthetic_state)
    qtbot.addWidget(mt_view)
    mt_view.show()

    initial_stim = synthetic_state.stimno

    # In Pan mode, clicking on plot should NOT change stimno
    mt_view.set_nav_mode("pan")
    # Simulate a left click on the ViewBox
    vb = mt_view.main_plot.vb
    scene_pos = vb.mapViewToScene(pg.Point(100, 5))

    class MockClickEvent:
        def button(self):
            return Qt.LeftButton

        def scenePos(self):
            return scene_pos

        def accept(self):
            pass

    event = MockClickEvent()
    mt_view._on_plot_clicked(event)
    assert synthetic_state.stimno == initial_stim

    # In Normal mode, clicking DOES select the sweep
    mt_view.set_nav_mode("")
    mt_view._on_plot_clicked(event)
    assert synthetic_state.stimno == 5


def test_view_settings_panel_controls_navigation_mode(qtbot, synthetic_state):
    """
    Test that the Properties window (ViewSettingsPanel) contains controls to
    enable Pan, Zoom, and Normal modes on the active PyQtGraph view.
    """
    panel = ViewSettingsPanel(state=synthetic_state)
    qtbot.addWidget(panel)
    panel.show()

    pg_view = SingleTracePGView(state=synthetic_state)
    qtbot.addWidget(pg_view)
    pg_view.show()

    panel.setActiveView(pg_view)
    assert panel.grp_navigation.isVisible()
    assert panel.btn_nav_normal.isChecked()

    # Click Pan button in panel
    panel.btn_nav_pan.click()
    assert pg_view.get_nav_mode() == "pan"
    assert pg_view.is_nav_active()

    # Click Zoom button in panel
    panel.btn_nav_zoom.click()
    assert pg_view.get_nav_mode() == "zoom"
    assert pg_view.is_nav_active()

    # Click Normal button in panel
    panel.btn_nav_normal.click()
    assert pg_view.get_nav_mode() == ""
    assert not pg_view.is_nav_active()


def test_view_settings_panel_manual_xy_positions(qtbot, synthetic_state):
    """
    Test that the user can manually specify x/y positions in ViewSettingsPanel,
    that applying them updates the plot, and that plot range changes update the spinboxes.
    """
    panel = ViewSettingsPanel(state=synthetic_state)
    qtbot.addWidget(panel)
    panel.show()

    mt_view = MultiTracePGView(state=synthetic_state)
    qtbot.addWidget(mt_view)
    mt_view.show()

    panel.setActiveView(mt_view)

    # Manually specify X and Y limits in the panel
    panel.spin_xlim_min.setValue(50.0)
    panel.spin_xlim_max.setValue(250.0)
    panel.spin_ylim_min.setValue(2.0)
    panel.spin_ylim_max.setValue(8.0)

    panel.btn_apply_limits.click()

    # Verify the plot's range updated
    (xmin, xmax), (ymin, ymax) = mt_view.get_view_range()
    assert np.isclose(xmin, 50.0, atol=1.0)
    assert np.isclose(xmax, 250.0, atol=1.0)
    assert np.isclose(ymin, 2.0, atol=0.5)
    assert np.isclose(ymax, 8.0, atol=0.5)

    # Now zoom the plot programmatically (or via mouse drag)
    mt_view.main_plot.setXRange(100.0, 180.0, padding=0)
    mt_view.main_plot.setYRange(3.0, 6.0, padding=0)

    # The spinboxes in the panel should reflect the new limits
    assert np.isclose(panel.spin_xlim_min.value(), 100.0, atol=1.0)
    assert np.isclose(panel.spin_xlim_max.value(), 180.0, atol=1.0)
    assert np.isclose(panel.spin_ylim_min.value(), 3.0, atol=0.5)
    assert np.isclose(panel.spin_ylim_max.value(), 6.0, atol=0.5)


def test_view_settings_panel_reset_button(qtbot, synthetic_state):
    """
    Test that the Reset / Auto Range button in ViewSettingsPanel resets view range.
    """
    panel = ViewSettingsPanel(state=synthetic_state)
    qtbot.addWidget(panel)
    panel.show()

    fixed_view = MultiTraceFixedPGView(state=synthetic_state)
    qtbot.addWidget(fixed_view)
    fixed_view.show()

    panel.setActiveView(fixed_view)

    # Zoom in tightly
    panel.spin_xlim_min.setValue(0.01)
    panel.spin_xlim_max.setValue(0.02)
    panel.btn_apply_limits.click()

    cur_x = fixed_view.get_view_range()[0]
    assert np.isclose(cur_x[0], 0.01, atol=1e-3)
    assert np.isclose(cur_x[1], 0.02, atol=1e-3)

    # Click Reset in the panel
    panel.btn_nav_reset.click()

    # View range should be restored to full sweep range (> 0.05s)
    new_x = fixed_view.get_view_range()[0]
    assert new_x[1] > 0.05
