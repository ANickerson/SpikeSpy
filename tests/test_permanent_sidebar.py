import numpy as np
import PySide6QtAds as QtAds
import quantities as pq
import neo
import pytest

from spikespy.ActiveUnitInfoView import ActiveUnitInfoView
from spikespy.NeoSettingsView import NeoSettingsView
from spikespy.SharedChannelPanel import SharedChannelPanel
from spikespy.SpikeSpy import MdiView
from spikespy.ViewerState import ViewerState
from spikespy.ViewSettingsPanel import ViewSettingsPanel
from spikespy.helpers import notify_view_selected


@pytest.fixture
def sidebar_state():
    sr = 1000.0 * pq.Hz
    sig = neo.AnalogSignal(
        np.random.randn(2000, 2) * pq.uV,
        sampling_rate=sr,
        t_start=0.0 * pq.s,
        name="ch0",
    )
    ev = neo.Event(times=np.array([0.2, 0.7, 1.2]) * pq.s, name="stim")
    seg = neo.Segment(name="test_seg")
    seg.analogsignals.append(sig)
    seg.events.append(ev)
    state = ViewerState()
    state.set_data(analog_signal=sig, event_signal=ev, segment=seg)
    return state


def test_permanent_sidebar_initial_state(qtbot, sidebar_state):
    mdi = MdiView(state=sidebar_state, check_updates=False)
    qtbot.addWidget(mdi)
    mdi.show()
    qtbot.wait(20)

    # Expanded by default on the left with all 4 permanent panels
    assert mdi.is_sidebar_collapsed() is False
    assert len(mdi._permanent_sidebar_docks) == 4

    widget_types = [type(dw.widget()) for dw in mdi._permanent_sidebar_docks]
    assert widget_types == [
        ViewSettingsPanel,
        ActiveUnitInfoView,
        SharedChannelPanel,
        NeoSettingsView,
    ]

    # All 4 share the same left dock area and are non-closable + pinnable
    areas = {dw.dockAreaWidget() for dw in mdi._permanent_sidebar_docks}
    assert len(areas) == 1
    area = next(iter(areas))
    assert area is not None
    assert area.currentDockWidget() is mdi.dock_settings

    for dw in mdi._permanent_sidebar_docks:
        assert not dw.features().testFlag(QtAds.CDockWidget.DockWidgetClosable)
        assert dw.features().testFlag(QtAds.CDockWidget.DockWidgetPinnable)


def test_permanent_sidebar_survives_closing_all_windows(qtbot, sidebar_state):
    mdi = MdiView(state=sidebar_state, check_updates=False)
    qtbot.addWidget(mdi)
    mdi.show()
    qtbot.wait(20)

    mt = mdi.newWindow("MultiTrace")
    st = mdi.newWindow("SingleTraceView")
    qtbot.wait(20)

    # Close workspace windows individually
    for dw in list(mdi.dock_manager.dockWidgets()):
        if not mdi._is_permanent_sidebar_dock(dw):
            dw.closeDockWidget()
            mdi.dock_manager.removeDockWidget(dw)
    qtbot.wait(20)

    # All 4 permanent sidebar docks remain docked and open
    remaining = mdi.dock_manager.dockWidgets()
    for dw in mdi._permanent_sidebar_docks:
        assert dw in remaining
        assert not dw.isClosed()

    # Also test clear_dock_area()
    mdi.newWindow("MultiTrace")
    mdi.clear_dock_area()
    qtbot.wait(20)
    remaining_after_clear = mdi.dock_manager.dockWidgets()
    workspace_after_clear = [
        dw for dw in remaining_after_clear if not mdi._is_permanent_sidebar_dock(dw)
    ]
    assert len(workspace_after_clear) == 0
    for dw in mdi._permanent_sidebar_docks:
        assert dw in remaining_after_clear


def test_permanent_sidebar_collapse_expand_and_focus_respect(qtbot, sidebar_state):
    mdi = MdiView(state=sidebar_state, check_updates=False)
    qtbot.addWidget(mdi)
    mdi.show()
    qtbot.wait(20)

    mt = mdi.newWindow("MultiTrace")
    st = mdi.newWindow("SingleTraceView")
    qtbot.wait(20)

    # Collapse into left auto-hide sidebar
    assert mdi.toggle_sidebar_collapsed() is True
    qtbot.wait(20)
    assert mdi.is_sidebar_collapsed() is True
    left_rail = mdi.dock_manager.autoHideSideBar(QtAds.SideBarLeft)
    assert left_rail is not None
    assert left_rail.count() == 4

    # Clicking/selecting a workspace view updates settings_panel without un-collapsing the sidebar
    notify_view_selected(st, ensure_panel_visible=True)
    qtbot.wait(20)
    assert mdi.settings_panel.active_view is st
    assert mdi.is_sidebar_collapsed() is True

    # Closing all workspace windows while collapsed still keeps all 4 tabs in the left rail
    mdi.clear_dock_area()
    qtbot.wait(20)
    assert mdi.is_sidebar_collapsed() is True
    assert mdi.dock_manager.autoHideSideBar(QtAds.SideBarLeft).count() == 4

    # Expand back into a single shared left dock area
    assert mdi.toggle_sidebar_collapsed() is False
    qtbot.wait(20)
    assert mdi.is_sidebar_collapsed() is False
    assert mdi.dock_manager.autoHideSideBar(QtAds.SideBarLeft).count() == 0
    areas = {dw.dockAreaWidget() for dw in mdi._permanent_sidebar_docks}
    assert len(areas) == 1
    assert next(iter(areas)) is not None


def test_permanent_sidebar_open_helpers_and_layouts_no_duplicates(qtbot, sidebar_state):
    mdi = MdiView(state=sidebar_state, check_updates=False)
    qtbot.addWidget(mdi)
    mdi.show()
    qtbot.wait(20)

    assert mdi.open_or_show_view_settings_panel() is mdi.settings_panel
    assert mdi.open_or_show_active_unit_info() is mdi.active_unit_info_panel
    assert mdi.open_or_show_shared_channel_panel() is mdi.shared_channel_panel
    assert mdi.open_or_show_neo_settings() is mdi.neo_settings_panel

    # Layouts reuse permanent sidebar panels instead of duplicating them
    mdi.apply_default_layout("Shared Channel Group Suite")
    qtbot.wait(20)
    shared_docks = [
        dw for dw in mdi.dock_manager.dockWidgetsMap().values()
        if isinstance(dw.widget(), SharedChannelPanel)
    ]
    assert len(shared_docks) == 1

    mdi.apply_default_layout("Live Tracking Layout (Protocol Execution)")
    qtbot.wait(20)
    unit_docks = [
        dw for dw in mdi.dock_manager.dockWidgetsMap().values()
        if isinstance(dw.widget(), ActiveUnitInfoView)
    ]
    assert len(unit_docks) == 1

    # Named perspective save & load preserves permanent sidebar without duplicates
    mdi.save_perspective_as("SidebarTestPerspective")
    mdi.clear_dock_area()
    assert mdi.load_perspective("SidebarTestPerspective") is True
    qtbot.wait(20)
    for cls in (ViewSettingsPanel, ActiveUnitInfoView, SharedChannelPanel, NeoSettingsView):
        matches = [
            dw for dw in mdi.dock_manager.dockWidgetsMap().values()
            if isinstance(dw.widget(), cls)
        ]
        assert len(matches) == 1
