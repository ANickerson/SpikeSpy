import os
import pytest
import numpy as np
import quantities as pq
import neo
from PySide6.QtWidgets import QApplication

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from spikespy.ViewerState import ViewerState, tracked_neuron_unit
from spikespy.ViewSettingsPanel import ViewSettingsPanel
from spikespy.MultiTraceView import MultiTraceView
from spikespy.SingleTraceView import SingleTraceView
from spikespy.MultiTraceFixedView import MultiTraceFixedView
from spikespy.UnitView import UnitView


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

    state.segment = seg
    state.set_data(analog_signal=seg.analogsignals[0], event_signal=event_sig, spike_groups=[u0])
    return state


def test_view_settings_panel_multitrace(mock_state):
    app = QApplication.instance() or QApplication([])

    panel = ViewSettingsPanel(state=mock_state)
    panel.show()
    app.processEvents()

    assert panel.active_view is None
    assert panel.placeholder.isVisible()

    mt_view = MultiTraceView(state=mock_state)
    mt_view.show()
    app.processEvents()

    panel.setActiveView(mt_view)
    app.processEvents()

    assert panel.active_view is mt_view
    assert panel.grp_multitrace.isVisible()
    assert not panel.grp_singletrace.isVisible()

    # Test changing mode via panel
    panel.combo_mt_mode.setCurrentIndex(1)  # Lines / Traces
    app.processEvents()
    assert mt_view.mode == "lines"

    panel.combo_mt_mode.setCurrentIndex(0)  # Heatmap
    app.processEvents()
    assert mt_view.mode == "heatmap"

    # Test changing percentiles / scale
    panel.spin_mt_lower.setValue(1.5)
    panel.spin_mt_upper.setValue(3.5)
    app.processEvents()
    assert mt_view.lowerSpinBox.value() == 1.5
    assert mt_view.upperSpinBox.value() == 3.5

    # Test lock to stim checkbox
    panel.chk_mt_lock_stim.setChecked(True)
    app.processEvents()
    assert mt_view.lock_to_stimCheckBox.isChecked() is True

    panel.close()
    mt_view.close()
    app.processEvents()


def test_view_settings_panel_singletrace_and_fixed(mock_state):
    app = QApplication.instance() or QApplication([])

    panel = ViewSettingsPanel(state=mock_state)
    panel.show()
    app.processEvents()

    st_view = SingleTraceView(state=mock_state)
    st_view.show()
    app.processEvents()

    panel.setActiveView(st_view)
    app.processEvents()
    assert panel.grp_singletrace.isVisible()

    # Toggle multichannel from panel
    panel.chk_st_multi.setChecked(True)
    app.processEvents()
    assert st_view.channel_bar.is_multichannel() is True

    # Adjust spacing
    panel.spin_st_spacing.setValue(25)
    app.processEvents()
    assert st_view.spin_chan_spacing.value() == 25

    # Switch active view to MultiTraceFixedView
    fix_view = MultiTraceFixedView(state=mock_state)
    fix_view.show()
    app.processEvents()

    panel.setActiveView(fix_view)
    app.processEvents()

    assert not panel.grp_singletrace.isVisible()
    assert panel.grp_fixed.isVisible()

    # Adjust fixed view parameters
    panel.spin_fix_scale.setValue(3.0)
    panel.spin_fix_pre.setValue(8)
    panel.spin_fix_post.setValue(12)
    app.processEvents()

    assert fix_view.scale_spinner.value() == 3.0
    assert fix_view.pre_spinner.value() == 8
    assert fix_view.post_spinner.value() == 12

    panel.close()
    st_view.close()
    fix_view.close()
    app.processEvents()


def test_view_settings_panel_unit_and_switching(mock_state):
    app = QApplication.instance() or QApplication([])

    panel = ViewSettingsPanel(state=mock_state)
    panel.show()
    app.processEvents()

    u_view = UnitView(state=mock_state)
    u_view.show()
    app.processEvents()

    panel.setActiveView(u_view)
    app.processEvents()
    assert panel.grp_unit.isVisible()

    panel.chk_unit_neighbors.setChecked(True)
    app.processEvents()
    assert u_view.chk_show_neighbors.isChecked() is True

    # Switch to None
    panel.setActiveView(None)
    app.processEvents()
    assert panel.active_view is None
    assert panel.placeholder.isVisible()
    assert not panel.grp_unit.isVisible()

    panel.close()
    u_view.close()
    app.processEvents()


def test_unit_group_selector_standalone_and_channel_group(mock_state):
    from spikespy.ChannelGroupContainer import ChannelGroupContainerView
    app = QApplication.instance() or QApplication([])

    # Add a second unit to mock_state for testing
    times = np.arange(5) * 0.1 * pq.s
    evt1 = neo.Event(times + 0.02 * pq.s, name="unit_1", units=pq.s)
    evt1.annotations["channel"] = "CH1"
    u1 = tracked_neuron_unit(event=evt1, channel="CH1", neighbor_channels=[])
    mock_state.spike_groups.append(u1)
    mock_state.onUnitGroupChange.emit()

    panel = ViewSettingsPanel(state=mock_state)
    panel.show()
    app.processEvents()

    # 1. Standalone view
    st_view = SingleTraceView(state=mock_state)
    st_view.show()
    app.processEvents()

    panel.setActiveView(st_view)
    app.processEvents()

    assert panel.grp_unit_selector.isVisible()
    # By default, follows active unit (item 0)
    assert panel.combo_unit.currentIndex() == 0

    # Select Unit 1 in panel
    u1_idx = panel.combo_unit.findData(1)
    assert u1_idx >= 0
    panel.combo_unit.setCurrentIndex(u1_idx)
    app.processEvents()

    assert st_view.channel_bar.get_bound_unit() == 1

    # 2. View within ChannelGroupContainerView
    container = ChannelGroupContainerView(state=mock_state)
    c_child1 = SingleTraceView(parent=container, state=mock_state)
    c_child2 = MultiTraceView(parent=container, state=mock_state)
    container.add_view(c_child1, title="Child1")
    container.add_view(c_child2, title="Child2")
    container.show()
    app.processEvents()

    # Select child1 in settings panel
    panel.setActiveView(c_child1)
    app.processEvents()

    # Change unit via panel to Unit 0
    u0_idx = panel.combo_unit.findData(0)
    assert u0_idx >= 0
    panel.combo_unit.setCurrentIndex(u0_idx)
    app.processEvents()

    # Verify that changing it within a channel group affected all groups / children
    assert container.channel_bar.get_bound_unit() == 0
    assert c_child1.channel_bar.get_bound_unit() == 0
    assert c_child2.channel_bar.get_bound_unit() == 0

    panel.close()
    st_view.close()
    container.close()
    app.processEvents()


def test_hide_view_top_bars_global_option(mock_state):
    from spikespy.ChannelGroupContainer import ChannelGroupContainerView
    from PySide6.QtWidgets import QToolBar
    app = QApplication.instance() or QApplication([])

    panel = ViewSettingsPanel(state=mock_state)
    panel.show()
    app.processEvents()

    mt_view = MultiTraceView(state=mock_state)
    mt_view.show()
    container = ChannelGroupContainerView(state=mock_state)
    st_child = SingleTraceView(parent=container, state=mock_state)
    container.add_view(st_child, title="ST Child")
    container.show()
    app.processEvents()

    # Initially toolbars are visible
    mt_toolbars = mt_view.findChildren(QToolBar)
    assert any(tb.isVisible() for tb in mt_toolbars)
    assert container.channel_bar.isVisible()
    assert container.control_bar.isVisible()

    # Test with MdiView to ensure main application toolbar is preserved
    from spikespy.SpikeSpy import MdiView
    win = MdiView(state=mock_state)
    win.show()
    app.processEvents()

    assert win.toolbar.isVisible() is True

    # Enable remove top bars via panel checkbox
    panel.chk_hide_top_bars.setChecked(True)
    app.processEvents()

    assert mock_state.hide_view_top_bars is True
    # Application toolbar must remain visible!
    assert win.toolbar.isVisible() is True
    # All toolbars on standalone and container should be hidden
    assert all(not tb.isVisible() for tb in mt_view.findChildren(QToolBar))
    assert not container.channel_bar.isVisible()
    assert not container.control_bar.isVisible()
    assert all(not tb.isVisible() for tb in st_child.findChildren(QToolBar))

    # Disable remove top bars
    panel.chk_hide_top_bars.setChecked(False)
    app.processEvents()

    assert mock_state.hide_view_top_bars is False
    assert win.toolbar.isVisible() is True
    assert any(tb.isVisible() for tb in mt_view.findChildren(QToolBar))
    assert container.channel_bar.isVisible()
    assert container.control_bar.isVisible()
    # Child's channel bar remains hidden due to channel group container deferral rule
    assert not st_child.channel_bar.isVisible()

    panel.close()
    mt_view.close()
    container.close()
    win.close()
    app.processEvents()


def test_settings_follows_selected_window_via_titlebar_and_tabs(mock_state):
    from PySide6.QtCore import QPoint, Qt, QEvent
    from PySide6.QtGui import QMouseEvent
    from spikespy.SpikeSpy import MdiView
    app = QApplication.instance() or QApplication([])

    win = MdiView(state=mock_state)
    win.show()
    app.processEvents()

    mt = win.newWindow("MultiTrace")
    st = win.newWindow("SingleTraceView")
    u = win.newWindow("UnitView")
    app.processEvents()

    # Find dock widgets
    mt_dw = [dw for dw in win.dock_manager.dockWidgets() if dw.widget() == mt][0]
    st_dw = [dw for dw in win.dock_manager.dockWidgets() if dw.widget() == st][0]
    u_dw = [dw for dw in win.dock_manager.dockWidgets() if dw.widget() == u][0]

    press = QMouseEvent(QEvent.MouseButtonPress, QPoint(10, 10), Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)

    # 1. Clicking dock title bars switches active view in settings panel
    tb_mt = mt_dw.dockAreaWidget().titleBar()
    win.eventFilter(tb_mt, press)
    app.processEvents()
    assert win.settings_panel.active_view == mt
    assert win.settings_panel.grp_multitrace.isVisible()

    tb_st = st_dw.dockAreaWidget().titleBar()
    win.eventFilter(tb_st, press)
    app.processEvents()
    assert win.settings_panel.active_view == st
    assert win.settings_panel.grp_singletrace.isVisible()

    tb_u = u_dw.dockAreaWidget().titleBar()
    win.eventFilter(tb_u, press)
    app.processEvents()
    assert win.settings_panel.active_view == u
    assert win.settings_panel.grp_unit.isVisible()

    # 2. Tab switching updates active view
    area = mt_dw.dockAreaWidget()
    win.dock_manager.addDockWidgetTabToArea(st_dw, area)
    app.processEvents()

    mt_dw.setAsCurrentTab()
    app.processEvents()
    assert win.settings_panel.active_view == mt

    st_dw.setAsCurrentTab()
    app.processEvents()
    assert win.settings_panel.active_view == st

    win.close()
    app.processEvents()


def test_scale_mode_mv_and_std_scaling(mock_state):
    from spikespy.MultiTraceFixedPGView import MultiTraceFixedPGView
    from spikespy.MultiTracePGView import MultiTracePGView
    app = QApplication.instance() or QApplication([])

    panel = ViewSettingsPanel(state=mock_state)
    panel.show()
    app.processEvents()

    # 1. MultiTraceFixedView (MPL)
    fix_view = MultiTraceFixedView(state=mock_state)
    fix_view.show()
    panel.setActiveView(fix_view)
    app.processEvents()

    assert fix_view.get_settings().get("scale_mode") == "mV"
    assert panel.combo_fix_scale_mode.currentText() == "mV"

    # Switch to std scaling
    panel.combo_fix_scale_mode.setCurrentText("std")
    app.processEvents()
    assert fix_view.get_settings().get("scale_mode") == "std"
    assert fix_view.scale_mode_combo.currentText() == "std"

    # Switch back to mV
    panel.combo_fix_scale_mode.setCurrentText("mV")
    app.processEvents()
    assert fix_view.get_settings().get("scale_mode") == "mV"

    # 2. MultiTraceFixedPGView (PyQtGraph)
    fix_pg = MultiTraceFixedPGView(state=mock_state)
    fix_pg.show()
    panel.setActiveView(fix_pg)
    app.processEvents()

    assert fix_pg.get_settings().get("scale_mode") == "mV"
    panel.combo_fix_scale_mode.setCurrentText("std")
    app.processEvents()
    assert fix_pg.get_settings().get("scale_mode") == "std"
    assert fix_pg.scale_mode_combo.currentText() == "std"

    # 3. MultiTraceView (MPL)
    mt_view = MultiTraceView(state=mock_state)
    mt_view.show()
    panel.setActiveView(mt_view)
    app.processEvents()

    assert mt_view.get_settings().get("scale_mode") == "std"
    assert panel.combo_mt_scale_mode.currentText() == "std"

    panel.combo_mt_scale_mode.setCurrentText("mV")
    app.processEvents()
    assert mt_view.get_settings().get("scale_mode") == "mV"
    assert mt_view.scale_mode_combo.currentText() == "mV"

    # 4. MultiTracePGView (PyQtGraph)
    mt_pg = MultiTracePGView(state=mock_state)
    mt_pg.show()
    panel.setActiveView(mt_pg)
    app.processEvents()

    assert mt_pg.get_settings().get("scale_mode") == "std"
    panel.combo_mt_scale_mode.setCurrentText("mV")
    app.processEvents()
    assert mt_pg.get_settings().get("scale_mode") == "mV"
    assert mt_pg.scale_mode_combo.currentText() == "mV"

    panel.close()
    fix_view.close()
    fix_pg.close()
    mt_view.close()
    mt_pg.close()
    app.processEvents()


def test_view_settings_appears_on_clicking_component(mock_state):
    """
    Test that clicking any component view:
    1. Updates active view and shows its corresponding configuration in ViewSettingsPanel.
    2. Opens and unhides View Settings if it was closed or hidden.
    3. Raises and selects the View Settings tab if it was tabbed behind another panel.
    4. Handles sub-views inside ChannelGroupContainerView.
    """
    from PySide6.QtCore import Qt, QPoint
    from PySide6.QtTest import QTest
    from spikespy.SpikeSpy import MdiView
    import PySide6QtAds as QtAds

    app = QApplication.instance() or QApplication([])

    win = MdiView(state=mock_state)
    win.show()
    app.processEvents()

    # 1. Create components
    mt = win.newWindow("MultiTrace")
    st = win.newWindow("SingleTraceView")
    pg_mt = win.newWindow("MultiTrace (PyQtGraph)")
    pg_st = win.newWindow("SingleTrace (PyQtGraph)")
    app.processEvents()

    # Click mt component (canvas)
    QTest.mouseClick(mt.view, Qt.LeftButton, pos=QPoint(10, 10))
    app.processEvents()
    assert win.settings_panel.active_view == mt
    assert win.settings_panel.grp_multitrace.isVisible()

    # Click st component (canvas)
    QTest.mouseClick(st.view, Qt.LeftButton, pos=QPoint(10, 10))
    app.processEvents()
    assert win.settings_panel.active_view == st
    assert win.settings_panel.grp_singletrace.isVisible()

    # Click pg_mt component (viewport)
    QTest.mouseClick(pg_mt.glw.viewport(), Qt.LeftButton, pos=QPoint(10, 10))
    app.processEvents()
    assert win.settings_panel.active_view == pg_mt
    assert win.settings_panel.grp_multitrace.isVisible()

    # Click pg_st component (viewport)
    QTest.mouseClick(pg_st.glw.viewport(), Qt.LeftButton, pos=QPoint(10, 10))
    app.processEvents()
    assert win.settings_panel.active_view == pg_st
    assert win.settings_panel.grp_singletrace.isVisible()

    # 2. If View Settings was closed, clicking a component makes it appear
    win.dock_settings.toggleView(False)
    app.processEvents()
    assert win.dock_settings.isClosed() is True

    QTest.mouseClick(mt.view, Qt.LeftButton, pos=QPoint(10, 10))
    app.processEvents()
    assert win.dock_settings.isClosed() is False
    assert win.dock_settings.isVisible() is True
    assert win.settings_panel.active_view == mt

    # 3. If View Settings was tabbed behind another panel, clicking a component brings it to front
    tbl = win.newWindow("SpikeGroupTable")
    dw_tbl = [dw for dw in win.dock_manager.dockWidgets() if dw.widget() == tbl][0]
    area = win.dock_settings.dockAreaWidget()
    win.dock_manager.addDockWidgetTabToArea(dw_tbl, area)
    dw_tbl.setAsCurrentTab()
    app.processEvents()
    assert area.currentDockWidget() == dw_tbl

    # Clicking st component switches tab back to View Settings
    QTest.mouseClick(st.view, Qt.LeftButton, pos=QPoint(10, 10))
    app.processEvents()
    assert area.currentDockWidget() == win.dock_settings
    assert win.settings_panel.active_view == st

    # 4. Clicking inside ChannelGroupContainer sub-view components updates settings panel
    cgc = win.newWindow("ChannelGroupContainer")
    cgc.add_view_by_key("MultiTrace")
    cgc.add_view_by_key("SingleTrace (PyQtGraph)")
    app.processEvents()

    sub_mt = cgc.sub_views[0]
    sub_pg_st = cgc.sub_views[1]

    # Click sub_mt canvas
    QTest.mouseClick(sub_mt.view, Qt.LeftButton, pos=QPoint(10, 10))
    app.processEvents()
    assert win.settings_panel.active_view == sub_mt
    assert win.settings_panel.grp_multitrace.isVisible()

    # Click sub_pg_st viewport
    QTest.mouseClick(sub_pg_st.glw.viewport(), Qt.LeftButton, pos=QPoint(10, 10))
    app.processEvents()
    assert win.settings_panel.active_view == sub_pg_st
    assert win.settings_panel.grp_singletrace.isVisible()

    win.close()
    app.processEvents()


def test_view_settings_panel_saved_layout_loading(mock_state):
    """
    Test that loading a saved layout/perspective:
    1. Keeps or restores the View Settings panel in an open, visible state without needing manual reopening.
    2. Automatically binds the settings panel and follows the active or first configurable view.
    3. Clicking components in the restored layout switches the active view in View Settings.
    """
    from PySide6.QtCore import Qt, QPoint
    from PySide6.QtTest import QTest
    from spikespy.SpikeSpy import MdiView

    app = QApplication.instance() or QApplication([])

    win = MdiView(state=mock_state)
    win.show()
    app.processEvents()

    mt = win.newWindow("MultiTrace")
    st = win.newWindow("SingleTraceView")
    app.processEvents()

    # Save custom layout
    layout_name = "TestSavedLayout"
    win.save_perspective_as(layout_name)
    app.processEvents()

    # Load custom layout
    loaded = win.load_perspective(layout_name)
    assert loaded is True
    app.processEvents()

    # View Settings must be present, open/visible, and active_view set
    assert win.dock_settings is not None
    assert win.dock_settings.isClosed() is False
    assert win.dock_settings.isVisible() is True
    assert win.settings_panel is not None
    assert win.settings_panel.active_view is not None

    # Clicking components in the restored layout updates the settings panel
    cur_st = [dw.widget() for dw in win.dock_manager.dockWidgets() if dw.windowTitle() == "SingleTraceView"][0]
    QTest.mouseClick(cur_st.view, Qt.LeftButton, pos=QPoint(10, 10))
    app.processEvents()
    assert win.settings_panel.active_view == cur_st
    assert win.settings_panel.grp_singletrace.isVisible()

    cur_mt = [dw.widget() for dw in win.dock_manager.dockWidgets() if dw.windowTitle() == "MultiTrace"][0]
    QTest.mouseClick(cur_mt.view, Qt.LeftButton, pos=QPoint(10, 10))
    app.processEvents()
    assert win.settings_panel.active_view == cur_mt
    assert win.settings_panel.grp_multitrace.isVisible()

    win.close()
    app.processEvents()


