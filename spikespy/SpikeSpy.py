import cProfile
import copy
import sys
from dataclasses import dataclass, field
from datetime import datetime
from os import environ
from pathlib import Path
from typing import Any, List, Optional, Union
from csv import writer

import matplotlib
import matplotlib.style as mplstyle
import neo
import numpy as np
import PySide6
import quantities as pq
import logging
from matplotlib.widgets import PolygonSelector
from neo.io import NixIO, NixIOFr
from PySide6.QtCore import (
    QAbstractTableModel,
    QEvent,
    QModelIndex,
    QObject,
    Qt,
    Signal,
    Slot,
    QSettings,
)
from PySide6.QtGui import QAction, QColor, QShortcut, QKeySequence, QIcon, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QMainWindow,
    QMdiArea,
    QMdiSubWindow,
    QMenu,
    QMenuBar,
    QPushButton,
    QSpinBox,
    QTableView,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QMessageBox,
    QScrollArea,
    QTextEdit,
    QLabel,
)

from spikespy.check_update import ReleaseNotesDialog, VersionCheckThread
from spikespy.processing import MultiAnalogSignal

# import PySide6QtAds as QtAds
from .mng_file_selector import QNeoSelector
from .legacy_mpl import (
    MultiTraceView,
    SingleTraceView,
    UnitView,
    UnitFootprintView,
    EventView,
    MultiTraceFixedView,
)
from .NeoSettingsView import NeoSettingsView
from .SpikeGroupTable import SpikeGroupTableView
from .TrackingView import TrackingView
from .ViewerState import ViewerState, prompt_for_neo_file, tracked_neuron_unit
from .io_manager import (
    export_units_csv,
    export_units_npz,
    import_units_csv,
    prompt_for_neo_folder,
    save_state_to_nix,
)
from .UnitSuggestor import UnitSuggestor
from .open_ephys_addin import OpenEphysLiveView
from .ChannelGroupBar import ChannelGroupBar, ChannelGroupDialog
from .SharedChannelPanel import SharedChannelPanel
from .ChannelGroupContainer import ChannelGroupContainerView
from .ViewSettingsPanel import ViewSettingsPanel
from .ActiveUnitInfoView import ActiveUnitInfoView
from .SpikeProposalConfigView import SpikeProposalConfigView
from .helpers import apply_top_bars_visibility, notify_view_selected
from .dsp.detectors import (
    DetectorParamSpec,
    SpikeDetectorPlugin,
    SpikeDetectorRegistry,
    register_spike_detector,
)
import PySide6QtAds as QtAds
from typing import Dict

try:
    from .MultiTracePGView import MultiTracePGView
    from .MultiTraceFixedPGView import MultiTraceFixedPGView
    from .SingleTracePGView import SingleTracePGView
    from .UnitPGView import UnitPGView
    from .UnitFootprintPGView import UnitFootprintPGView
    _has_pg = True
except Exception as _e:
    logging.warning(f"PyQtGraph views not loaded: {_e}")
    _has_pg = False

mplstyle.use("fast")

window_options = {
    # 'TraceAnnotation': TraceView,
    "MultiTrace": MultiTraceView,
    "MultiTraceFixedView": MultiTraceFixedView,
    "UnitView": UnitView,
    "UnitFootprintView": UnitFootprintView,
    "SpikeGroupTable": SpikeGroupTableView,
    "SingleTraceView": SingleTraceView,
    "Settings": NeoSettingsView,
    "TrackingView": TrackingView,
    "Data": QNeoSelector,
    "Events": EventView,
    "OpenEphysLive": OpenEphysLiveView,
    "SharedChannelPanel": SharedChannelPanel,
    "ChannelGroupContainer": ChannelGroupContainerView,
    "ViewSettings": ViewSettingsPanel,
    "ActiveUnitInfo": ActiveUnitInfoView,
    "SpikeProposalConfig": SpikeProposalConfigView,
}
if _has_pg:
    window_options.update(
        {
            "MultiTrace (PyQtGraph)": MultiTracePGView,
            "MultiTraceFixed (PyQtGraph)": MultiTraceFixedPGView,
            "SingleTrace (PyQtGraph)": SingleTracePGView,
            "UnitView (PyQtGraph)": UnitPGView,
            "UnitFootprint (PyQtGraph)": UnitFootprintPGView,
        }
    )

suggestors: Dict[str, UnitSuggestor] = {}


class MdiView(QMainWindow):
    # signals
    loadFile = Signal(str, str)

    def clear_dock_area(self):
        """Closes all open dock widgets and clears the dock area."""
        for dw in list(self.dock_manager.dockWidgets()):
            w = dw.widget()
            # Cancel pending idle draws on canvases before close/deletion
            for target in ([w] + (w.get_child_views() if hasattr(w, "get_child_views") else [])):
                for c_attr in ("view", "canvas", "figure_canvas"):
                    cv = getattr(target, c_attr, None)
                    if cv is not None:
                        try:
                            cv._draw_pending = False
                            PySide6.QtCore.QCoreApplication.removePostedEvents(cv)
                        except Exception:
                            pass
            dw.closeDockWidget()
            self.dock_manager.removeDockWidget(dw)
        self.cur_windows.clear()
        self.dock_settings = None
        self.settings_panel = None
        self._open_ephys_live_dock = None
        if hasattr(self, "extra_settings_panels"):
            self.extra_settings_panels.clear()

    def save_perspective_as(self, name: str):
        if not name:
            return
        logging.info(f"Saving perspective '{name}' to settings file: {self.settings_file.fileName()}")
        self.dock_manager.addPerspective(name)
        group_key = f"perspective_{name}" if name != "main" else "view"
        self.settings_file.beginGroup(group_key)
        self.dock_manager.savePerspectives(self.settings_file)
        class_to_str = {v: k for k, v in self.window_options.items()}
        view_widgets = [
            x
            for x in self.dock_manager.dockWidgets()
            if not isinstance(x.widget(), ViewSettingsPanel)
            and x is not getattr(self, "dock_settings", None)
            and x.windowTitle() != "View Settings"
        ]
        self.settings_file.beginWriteArray("window", len(view_widgets))
        for i, x in enumerate(view_widgets):
            self.settings_file.setArrayIndex(i)
            self.settings_file.setValue("name", x.windowTitle())
            w_widget = x.widget()
            self.settings_file.setValue("class", class_to_str.get(type(w_widget), ""))
            try:
                settings = w_widget.get_settings()
            except Exception:
                settings = {}
            self.settings_file.setValue("settings", settings)

        self.settings_file.endArray()
        self.settings_file.endGroup()
        self.settings_file.sync()

    def load_perspective(self, name: str) -> bool:
        if not name:
            return False
        logging.info(f"Loading perspective '{name}' from settings file: {self.settings_file.fileName()}")
        group_key = f"perspective_{name}"
        if group_key not in self.settings_file.childGroups() and name == "main":
            group_key = "view"

        self.clear_dock_area()

        self.settings_file.beginGroup(group_key)
        self.dock_manager.loadPerspectives(self.settings_file)
        n_windows = self.settings_file.beginReadArray("window")
        for x in range(n_windows):
            self.settings_file.setArrayIndex(x)
            cls_name = self.settings_file.value("class")
            win_name = self.settings_file.value("name")
            settings = self.settings_file.value("settings")
            if cls_name in self.window_options:
                w = self.newWindow(cls_name, name=win_name)
                if hasattr(w, "set_settings") and settings:
                    try:
                        w.set_settings(settings)
                    except Exception:
                        pass
        self.settings_file.endArray()
        self.settings_file.endGroup()

        opened = False
        if name in self.dock_manager.perspectiveNames():
            self.dock_manager.openPerspective(name)
            opened = True

        # Reconnect or open the view settings panel
        self.open_or_show_view_settings_panel()

        # Follow the active or first configurable view in the newly loaded layout
        first_view = None
        for dw in self.dock_manager.dockWidgets():
            w = dw.widget()
            if w is not None and w is not getattr(self, "settings_panel", None):
                if hasattr(self, "settings_panel") and self.settings_panel is not None:
                    resolved = self.settings_panel._resolve_view(w)
                    if self.settings_panel._is_configurable_view(resolved):
                        first_view = resolved
                        break
        if first_view is not None:
            notify_view_selected(first_view, ensure_panel_visible=True)

        return opened

    def savePerspectives(self):
        self.save_perspective_as("main")

    def loadPerspectives(self):
        self.load_perspective("main")

    def prompt_save_perspective(self):
        default_name = "Custom Layout"
        if QApplication.instance() and QApplication.instance().platformName() == "offscreen":
            name = default_name
            ok = True
        else:
            name, ok = QInputDialog.getText(self, "Save Current Layout", "Layout Name:", text=default_name)
        if ok and name and name.strip():
            self.save_perspective_as(name.strip())

    def prompt_load_perspective(self):
        names = list(self.dock_manager.perspectiveNames())
        all_groups = self.settings_file.childGroups()
        for g in all_groups:
            if g.startswith("perspective_"):
                p_name = g[len("perspective_") :]
                if p_name not in names:
                    names.append(p_name)
        if not names:
            if QApplication.instance() and QApplication.instance().platformName() != "offscreen":
                QMessageBox.information(self, "No Layouts", "No saved layouts found.")
            return
        if QApplication.instance() and QApplication.instance().platformName() == "offscreen":
            selected = names[0]
            ok = True
        else:
            selected, ok = QInputDialog.getItem(self, "Load Layout", "Select Layout:", names, 0, False)
        if ok and selected:
            self.load_perspective(selected)

    def open_or_show_shared_channel_panel(self):
        """Opens or activates the SharedChannelPanel controller."""
        for dw in self.dock_manager.dockWidgets():
            if isinstance(dw.widget(), SharedChannelPanel):
                dw.show()
                dw.raise_()
                return dw.widget()
        return self.newWindow("SharedChannelPanel", pos=QtAds.BottomDockWidgetArea)

    def open_or_show_active_unit_info(self):
        """Opens or activates the ActiveUnitInfoView dock panel."""
        for dw in self.dock_manager.dockWidgets():
            if isinstance(dw.widget(), ActiveUnitInfoView):
                if dw.isClosed():
                    dw.toggleView(True)
                    if dw.dockAreaWidget() is None:
                        self.dock_manager.addDockWidget(QtAds.RightDockWidgetArea, dw)
                dw.show()
                dw.raise_()
                if dw.dockAreaWidget() is not None:
                    try:
                        dw.dockAreaWidget().setCurrentDockWidget(dw)
                    except Exception:
                        pass
                if hasattr(dw, "setAsCurrentTab") and callable(getattr(dw, "setAsCurrentTab")):
                    try:
                        dw.setAsCurrentTab()
                    except Exception:
                        pass
                return dw.widget()
        return self.newWindow("ActiveUnitInfo", pos=QtAds.RightDockWidgetArea)

    def open_or_show_spike_proposal_config(self):
        """Opens or activates the SpikeProposalConfigView dock panel."""
        for dw in self.dock_manager.dockWidgets():
            if isinstance(dw.widget(), SpikeProposalConfigView):
                if dw.isClosed():
                    dw.toggleView(True)
                    if dw.dockAreaWidget() is None:
                        self.dock_manager.addDockWidget(QtAds.RightDockWidgetArea, dw)
                dw.show()
                dw.raise_()
                if dw.dockAreaWidget() is not None:
                    try:
                        dw.dockAreaWidget().setCurrentDockWidget(dw)
                    except Exception:
                        pass
                if hasattr(dw, "setAsCurrentTab") and callable(getattr(dw, "setAsCurrentTab")):
                    try:
                        dw.setAsCurrentTab()
                    except Exception:
                        pass
                return dw.widget()
        return self.newWindow("SpikeProposalConfig", pos=QtAds.RightDockWidgetArea)

    def open_or_show_view_settings_panel(self):
        """Opens or brings to front the View Settings side panel on the left side."""
        try:
            if hasattr(self, "settings_panel") and self.settings_panel is not None:
                self.settings_panel.isVisible()
            else:
                self.settings_panel = ViewSettingsPanel(parent=self, state=self.state)
        except RuntimeError:
            self.settings_panel = ViewSettingsPanel(parent=self, state=self.state)

        dock_valid = False
        if hasattr(self, "dock_settings") and self.dock_settings is not None:
            try:
                if self.dock_settings in self.dock_manager.dockWidgets():
                    dock_valid = True
            except RuntimeError:
                self.dock_settings = None

        if not dock_valid:
            for dw in self.dock_manager.dockWidgets():
                if dw.windowTitle() == "View Settings" or isinstance(dw.widget(), ViewSettingsPanel):
                    self.dock_settings = dw
                    if isinstance(dw.widget(), ViewSettingsPanel):
                        self.settings_panel = dw.widget()
                    else:
                        self.dock_settings.setWidget(self.settings_panel)
                    dock_valid = True
                    break

        if not dock_valid:
            self.dock_settings = QtAds.CDockWidget("View Settings")
            self.dock_settings.setWidget(self.settings_panel)
            self.dock_settings.setFeature(QtAds.CDockWidget.DockWidgetDeleteOnClose, False)
            self.dock_manager.addDockWidget(QtAds.LeftDockWidgetArea, self.dock_settings)
        elif self.dock_settings.isClosed():
            self.dock_settings.toggleView(True)
            if self.dock_settings.dockAreaWidget() is None:
                self.dock_manager.addDockWidget(QtAds.LeftDockWidgetArea, self.dock_settings)
        self.dock_settings.show()
        self.dock_settings.raise_()
        if self.dock_settings.dockAreaWidget() is not None:
            try:
                self.dock_settings.dockAreaWidget().setCurrentDockWidget(self.dock_settings)
            except Exception:
                pass
        if hasattr(self.dock_settings, "setAsCurrentTab") and callable(getattr(self.dock_settings, "setAsCurrentTab")):
            try:
                self.dock_settings.setAsCurrentTab()
            except Exception:
                pass
        return self.dock_settings.widget()

    def _refresh_all_settings_panel_targets(self):
        panels: List[ViewSettingsPanel] = []
        if getattr(self, "settings_panel", None) is not None:
            panels.append(self.settings_panel)
        for sp in list(getattr(self, "extra_settings_panels", [])):
            if sp is not None and sp not in panels:
                panels.append(sp)
        for sp in panels:
            try:
                if hasattr(sp, "refresh_target_views"):
                    sp.refresh_target_views()
            except RuntimeError:
                pass

    def new_settings_panel(
        self,
        target_view: Optional[QWidget] = None,
        pinned: bool = False,
        page: str = "view",
        title: Optional[str] = None,
        dock_area=QtAds.RightDockWidgetArea,
    ) -> ViewSettingsPanel:
        panel = ViewSettingsPanel(parent=self, state=self.state, initial_page=page)
        target = (
            target_view
            if target_view is not None
            else (
                self.settings_panel.active_view
                if getattr(self, "settings_panel", None) is not None
                else None
            )
        )
        if target is not None:
            if pinned:
                panel.pin_to_view(target)
            else:
                panel.setActiveView(target)

        if title is None:
            if pinned and panel.active_view is not None:
                view_title = panel._get_view_display_name(panel.active_view)
                dock_title = f"Settings ({page.capitalize()}): {view_title}"
            else:
                dock_title = f"{page.capitalize()} Settings"
        else:
            dock_title = title

        dock = QtAds.CDockWidget(dock_title)
        dock.setWidget(panel)
        dock.setFeature(QtAds.CDockWidget.DockWidgetDeleteOnClose, True)
        dock.setFeature(QtAds.CDockWidget.DockWidgetForceCloseWithArea, True)
        self.dock_manager.addDockWidget(dock_area, dock)
        dock.show()
        dock.raise_()

        if not hasattr(self, "extra_settings_panels"):
            self.extra_settings_panels = []
        self.extra_settings_panels.append(panel)
        self._refresh_all_settings_panel_targets()
        return panel

    def open_channel_groups_editor_pane(self):
        """Opens the View Settings panel and expands the Channel Groups editor pane."""
        panel = self.open_or_show_view_settings_panel()
        if panel is not None and hasattr(panel, "show_channel_groups_editor"):
            panel.show_channel_groups_editor(open_pane=True)
        return panel

    def _on_hide_top_bars_changed(self, hide: bool):
        if hasattr(self, "act_hide_top_bars") and self.act_hide_top_bars.isChecked() != hide:
            self.act_hide_top_bars.setChecked(hide)
        if hasattr(self, "settings_file"):
            self.settings_file.setValue("hide_view_top_bars", hide)
        self.apply_top_bars_visibility_all(hide)

    def apply_top_bars_visibility_all(self, hide: bool):
        if not hasattr(self, "dock_manager") or self.dock_manager is None:
            return
        for dw in self.dock_manager.dockWidgets():
            w = dw.widget()
            if w is not None and w is not getattr(self, "settings_panel", None):
                apply_top_bars_visibility(w, hide)

    def _on_dock_area_created(self, area):
        if area is not None and not getattr(area, "_spikespy_current_changed_connected", False):
            area.currentChanged.connect(self._on_dock_area_current_changed)
            area._spikespy_current_changed_connected = True

    def _on_dock_area_current_changed(self, index: int):
        area = self.sender()
        if area is not None and hasattr(area, "currentDockWidget") and callable(getattr(area, "currentDockWidget")):
            dw = area.currentDockWidget()
            if dw and hasattr(dw, "widget") and dw.widget():
                notify_view_selected(dw.widget())

    def _on_focused_dock_widget_changed(self, old, new):
        if new and new.widget():
            notify_view_selected(new.widget())

    def _on_app_focus_changed(self, old, new):
        if new is not None:
            notify_view_selected(new, ensure_panel_visible=False)

    def closeEvent(self, event):
        app = QApplication.instance()
        if app is not None:
            try:
                app.focusChanged.disconnect(self._on_app_focus_changed)
            except (TypeError, RuntimeError):
                pass
            try:
                app.removeEventFilter(self)
            except (TypeError, RuntimeError):
                pass
        super().closeEvent(event)

    def eventFilter(self, obj, event):
        if event.type() in (QEvent.MouseButtonPress, QEvent.MouseButtonRelease):
            if isinstance(obj, QWidget):
                notify_view_selected(obj, ensure_panel_visible=(event.type() == QEvent.MouseButtonPress))
        return super().eventFilter(obj, event)

    def apply_default_layout(self, layout_name: str):
        """
        Applies a predefined multi-panel dock layout.
        Supported layouts:
        - 'Shared Channel Group Suite' / 'Shared Channel Group Suite (Synced)':
            MultiTrace on Left (bound to Shared Group A)
            SingleTraceView on Right Top (bound to Shared Group A)
            UnitView on Right Bottom (bound to Shared Group A)
            SharedChannelPanel docked at Bottom
        - 'Default Analysis Layout':
            MultiTrace on Left
            SingleTraceView on Right Top
            UnitView on Right Bottom
            SpikeGroupTable at Bottom
        - 'Dual Unit Compare':
            Unit 0 MultiTrace on Left
            Unit 1 MultiTrace on Right
        - 'Single Trace Inspector':
            SingleTraceView on Left
            UnitFootprintView on Right
            SpikeGroupTable at Bottom
        """
        self.clear_dock_area()
        layout_clean = layout_name.strip()

        if "Shared Channel Group Suite" in layout_clean:
            self.newWindow("MultiTrace", pos=QtAds.LeftDockWidgetArea, shared_set="Shared Group A")
            self.newWindow("SingleTraceView", pos=QtAds.RightDockWidgetArea, shared_set="Shared Group A")
            self.newWindow("UnitView", pos=QtAds.RightDockWidgetArea, shared_set="Shared Group A")
            self.newWindow("SharedChannelPanel", pos=QtAds.BottomDockWidgetArea)

        elif layout_clean in ("Default Analysis Layout", "Reset to Default Layout"):
            self.newWindow("MultiTrace", pos=QtAds.LeftDockWidgetArea)
            self.newWindow("SingleTraceView", pos=QtAds.RightDockWidgetArea)
            self.newWindow("UnitView", pos=QtAds.RightDockWidgetArea)
            self.newWindow("SpikeGroupTable", pos=QtAds.BottomDockWidgetArea)

        elif "Dual Unit Compare" in layout_clean:
            u0 = 0 if self.state and len(self.state.spike_groups) > 0 else None
            u1 = 1 if self.state and len(self.state.spike_groups) > 1 else u0
            self.newWindow("MultiTrace", pos=QtAds.LeftDockWidgetArea, target_unit=u0)
            self.newWindow("MultiTrace", pos=QtAds.RightDockWidgetArea, target_unit=u1)

        elif "Single Trace Inspector" in layout_clean:
            self.newWindow("SingleTraceView", pos=QtAds.LeftDockWidgetArea)
            self.newWindow("UnitFootprintView", pos=QtAds.RightDockWidgetArea)
            self.newWindow("SpikeGroupTable", pos=QtAds.BottomDockWidgetArea)

        elif "Live Search Layout" in layout_clean:
            mt_key = "MultiTrace (PyQtGraph)" if _has_pg and "MultiTrace (PyQtGraph)" in self.window_options else "MultiTrace"
            st_key = "SingleTrace (PyQtGraph)" if _has_pg and "SingleTrace (PyQtGraph)" in self.window_options else "SingleTraceView"
            self.newWindow("OpenEphysLive", pos=QtAds.LeftDockWidgetArea)
            self.newWindow(mt_key, pos=QtAds.LeftDockWidgetArea)
            self.newWindow(st_key, pos=QtAds.RightDockWidgetArea)
            self.newWindow("SpikeGroupTable", pos=QtAds.BottomDockWidgetArea)
            if self.state is not None and hasattr(self.state, "set_auto_scan_live"):
                self.state.set_auto_scan_live(True)

        elif "Live Tracking Layout" in layout_clean:
            mt_key = "MultiTrace (PyQtGraph)" if _has_pg and "MultiTrace (PyQtGraph)" in self.window_options else "MultiTrace"
            st_key = "SingleTrace (PyQtGraph)" if _has_pg and "SingleTrace (PyQtGraph)" in self.window_options else "SingleTraceView"
            uv_key = "UnitView (PyQtGraph)" if _has_pg and "UnitView (PyQtGraph)" in self.window_options else "UnitView"
            self.newWindow(mt_key, pos=QtAds.LeftDockWidgetArea, shared_set="Shared Group A")
            self.newWindow(st_key, pos=QtAds.RightDockWidgetArea, shared_set="Shared Group A")
            self.newWindow(uv_key, pos=QtAds.RightDockWidgetArea, shared_set="Shared Group A")
            self.newWindow("ActiveUnitInfo", pos=QtAds.RightDockWidgetArea)
            self.newWindow("SpikeGroupTable", pos=QtAds.BottomDockWidgetArea)

    def handle_version_check_result(self, result):
        if result is not None:
            version, release_notes = result
            dialog = ReleaseNotesDialog(version, release_notes)
            dialog.exec()

    def __init__(
        self,
        parent: PySide6.QtWidgets.QWidget = None,
        state: ViewerState = None,
        check_updates=False,
        **kwargs,
    ) -> None:
        super().__init__(parent)
        self.settings_file = QSettings(
            QSettings.Format.IniFormat, QSettings.Scope.UserScope, "spikespy"
        )

        self.window_options = window_options
        self.cur_windows = []
        self.extra_settings_panels: List[ViewSettingsPanel] = []

        self.version_check_thread = VersionCheckThread()
        self.version_check_thread.worker.result.connect(
            self.handle_version_check_result
        )
        if check_updates:
            self.version_check_thread.start()

        from importlib.metadata import version as get_version

        version = get_version("spikespy")
        self.setWindowTitle(f"SpikeSpy - {version}")
        self.state = state or ViewerState(**kwargs)
        self.loadFile.connect(self._dispatch_load_file)
        self.state.onFileLoaded.connect(self.updateRecents)
        self.state.onLoadNewFile.connect(
            lambda self=self: self.setWindowTitle(
                f"SpikeSpy ({version}) - {Path(self.state.title).name}"
            )
        )
        self._open_ephys_live_dock = None
        self.dock_manager = QtAds.CDockManager(self)
        self.dock_manager.addPerspective("main")
        self.dock_manager.dockAreaCreated.connect(self._on_dock_area_created)
        if hasattr(self.dock_manager, "openedDockAreas"):
            for da in self.dock_manager.openedDockAreas():
                self._on_dock_area_created(da)

        # Active View Settings Side Panel (follows focused window)
        self.settings_panel = ViewSettingsPanel(parent=self, state=self.state)
        self.dock_settings = QtAds.CDockWidget("View Settings")
        self.dock_settings.setWidget(self.settings_panel)
        self.dock_settings.setFeature(QtAds.CDockWidget.DockWidgetDeleteOnClose, False)
        self.dock_manager.addDockWidget(QtAds.LeftDockWidgetArea, self.dock_settings)
        self.dock_manager.focusedDockWidgetChanged.connect(self._on_focused_dock_widget_changed)

        if hasattr(self.state, "onHideViewTopBarsChanged"):
            self.state.onHideViewTopBarsChanged.connect(self._on_hide_top_bars_changed)

        saved_hide_top_bars = self.settings_file.value("hide_view_top_bars", False, type=bool)
        if self.state is not None and saved_hide_top_bars:
            self.state.set_hide_view_top_bars(True)

        app = QApplication.instance()
        if app is not None:
            app.focusChanged.connect(self._on_app_focus_changed)
            app.installEventFilter(self)

        self.mdi = QMdiArea()
        # self.setCentralWidget(self.mdi)

        # for k in [
        #     "MultiTrace",
        #     "UnitView",
        #     "SpikeGroupTable",
        #     "SingleTraceView",
        #     "TrackingView",
        # ]:
        #     self.newWindow(k)

        self.toolbar = self.addToolBar("Main")

        # 1. [+ Add Panel ▾] QToolButton with categorized popup menu
        self.btn_add_panel = QToolButton(self)
        self.btn_add_panel.setText("+ Add Panel ▾")
        self.btn_add_panel.setPopupMode(QToolButton.InstantPopup)
        add_panel_menu = QMenu(self.btn_add_panel)

        menu_traces = add_panel_menu.addMenu("Trace & Heatmap Views")
        menu_traces.addAction("MultiTrace (Matplotlib)", lambda: self.newWindow("MultiTrace"))
        if "MultiTrace (PyQtGraph)" in self.window_options:
            menu_traces.addAction("MultiTrace (PyQtGraph)", lambda: self.newWindow("MultiTrace (PyQtGraph)"))
        menu_traces.addAction("MultiTraceFixedView (Matplotlib)", lambda: self.newWindow("MultiTraceFixedView"))
        if "MultiTraceFixed (PyQtGraph)" in self.window_options:
            menu_traces.addAction("MultiTraceFixed (PyQtGraph)", lambda: self.newWindow("MultiTraceFixed (PyQtGraph)"))
        menu_traces.addAction("SingleTraceView (Matplotlib)", lambda: self.newWindow("SingleTraceView"))
        if "SingleTrace (PyQtGraph)" in self.window_options:
            menu_traces.addAction("SingleTrace (PyQtGraph)", lambda: self.newWindow("SingleTrace (PyQtGraph)"))
        menu_traces.addAction("TrackingView", lambda: self.newWindow("TrackingView"))
        menu_traces.addAction("OpenEphysLive", lambda: self.newWindow("OpenEphysLive"))

        menu_unit = add_panel_menu.addMenu("Unit Analysis")
        menu_unit.addAction("Active Unit Info & Tracking", lambda: self.newWindow("ActiveUnitInfo"))
        menu_unit.addAction("Spike Proposal Config", lambda: self.open_or_show_spike_proposal_config())
        menu_unit.addAction("UnitView (Matplotlib)", lambda: self.newWindow("UnitView"))
        if "UnitView (PyQtGraph)" in self.window_options:
            menu_unit.addAction("UnitView (PyQtGraph)", lambda: self.newWindow("UnitView (PyQtGraph)"))
        menu_unit.addAction("UnitFootprintView (Matplotlib)", lambda: self.newWindow("UnitFootprintView"))
        if "UnitFootprint (PyQtGraph)" in self.window_options:
            menu_unit.addAction("UnitFootprint (PyQtGraph)", lambda: self.newWindow("UnitFootprint (PyQtGraph)"))

        menu_channels = add_panel_menu.addMenu("Channel Groups")
        menu_channels.addAction("Channel Group Sub-Window", lambda: self.newWindow("ChannelGroupContainer"))
        menu_channels.addAction("Shared Channel Controller", lambda: self.open_or_show_shared_channel_panel())

        menu_tables = add_panel_menu.addMenu("Tables & Controllers")
        menu_tables.addAction("Active Unit Info & Tracking", lambda: self.newWindow("ActiveUnitInfo"))
        menu_tables.addAction("Spike Proposal Config", lambda: self.open_or_show_spike_proposal_config())
        menu_tables.addAction("View Settings Panel", lambda: self.open_or_show_view_settings_panel())
        menu_tables.addAction("Pinned Settings for Active View", lambda: self.new_settings_panel(pinned=True))
        menu_tables.addAction("Channel & Unit Settings Panel", lambda: self.new_settings_panel(page="channels", title="Channel & Unit Settings"))
        menu_tables.addAction("Live Streaming Settings Panel", lambda: self.new_settings_panel(page="stream", title="Streaming Settings"))
        menu_tables.addAction("SharedChannelPanel", lambda: self.open_or_show_shared_channel_panel())
        menu_tables.addAction("SpikeGroupTable", lambda: self.newWindow("SpikeGroupTable"))
        menu_tables.addAction("Events", lambda: self.newWindow("Events"))
        menu_tables.addAction("Data", lambda: self.newWindow("Data"))
        menu_tables.addAction("Settings", lambda: self.newWindow("Settings"))

        self.btn_add_panel.setMenu(add_panel_menu)
        self.toolbar.addWidget(self.btn_add_panel)

        # 2. [ 🎛 Shared Channel Controller ] QAction
        self.act_shared_ctrl = QAction("🎛 Shared Channel Controller", self)
        self.act_shared_ctrl.setToolTip("Open or bring to focus the Shared Channel Controller panel")
        self.act_shared_ctrl.triggered.connect(self.open_or_show_shared_channel_panel)
        self.toolbar.addAction(self.act_shared_ctrl)

        self.act_unit_info = QAction("🎯 Active Unit Info", self)
        self.act_unit_info.setToolTip("Open or bring to focus the Active Unit Info & Tracking panel")
        self.act_unit_info.triggered.connect(self.open_or_show_active_unit_info)
        self.toolbar.addAction(self.act_unit_info)

        self.act_proposal_config = QAction("🔬 Proposal Config", self)
        self.act_proposal_config.setToolTip("Open or bring to focus the Spike Proposal & Candidate Search Configuration panel")
        self.act_proposal_config.triggered.connect(self.open_or_show_spike_proposal_config)
        self.toolbar.addAction(self.act_proposal_config)

        # 3. [ 📐 Layouts ▾ ] QToolButton with popup menu
        self.btn_layouts = QToolButton(self)
        self.btn_layouts.setText("📐 Layouts ▾")
        self.btn_layouts.setPopupMode(QToolButton.InstantPopup)
        layout_menu = QMenu(self.btn_layouts)

        layout_menu.addAction(
            "🔍 Live Search Layout (Electrode Hunting)",
            lambda: self.apply_default_layout("Live Search Layout (Electrode Hunting)"),
        )
        layout_menu.addAction(
            "🎯 Live Tracking Layout (Protocol Execution)",
            lambda: self.apply_default_layout("Live Tracking Layout (Protocol Execution)"),
        )
        layout_menu.addSeparator()
        layout_menu.addAction(
            "Shared Channel Group Suite (Synced)",
            lambda: self.apply_default_layout("Shared Channel Group Suite"),
        )
        layout_menu.addAction(
            "Default Analysis Layout",
            lambda: self.apply_default_layout("Default Analysis Layout"),
        )
        layout_menu.addAction(
            "Dual Unit Compare",
            lambda: self.apply_default_layout("Dual Unit Compare"),
        )
        layout_menu.addAction(
            "Single Trace Inspector",
            lambda: self.apply_default_layout("Single Trace Inspector"),
        )
        layout_menu.addSeparator()
        layout_menu.addAction("Save Current Layout As...", self.prompt_save_perspective)
        layout_menu.addAction("Load Saved Layout...", self.prompt_load_perspective)
        layout_menu.addAction(
            "Reset to Default Layout",
            lambda: self.apply_default_layout("Default Analysis Layout"),
        )

        self.btn_layouts.setMenu(layout_menu)
        self.toolbar.addWidget(self.btn_layouts)

        # 4. Live Cockpit Telemetry & Transport Bar
        self.toolbar.addSeparator()
        self.lbl_live_status_pill = QLabel("○ Offline", self)
        self.lbl_live_status_pill.setToolTip("Live acquisition status and current sweep index")
        self.toolbar.addWidget(self.lbl_live_status_pill)

        self.btn_follow_live_pill = QPushButton("📡 Follow Live", self)
        self.btn_follow_live_pill.setCheckable(True)
        self.btn_follow_live_pill.setChecked(bool(getattr(self.state, "follow_live", True)))
        self.btn_follow_live_pill.setToolTip("Toggle automatic following of incoming live sweeps (Space / Ctrl+L)")
        self.btn_follow_live_pill.clicked.connect(lambda _checked=False: self._toggle_follow_live())
        self.toolbar.addWidget(self.btn_follow_live_pill)

        self.btn_add_marker = QToolButton(self)
        self.btn_add_marker.setText("🏷 + Marker (M) ▾")
        self.btn_add_marker.setPopupMode(QToolButton.MenuButtonPopup)
        self.btn_add_marker.setToolTip("Add a protocol marker at the current stimulus sweep (M)")
        self.btn_add_marker.clicked.connect(self.prompt_add_protocol_marker)
        marker_menu = QMenu(self.btn_add_marker)
        for preset_label in (
            "Baseline Start",
            "2 Hz Tetanus Start",
            "2 Hz Tetanus End",
            "Recovery",
            "Mechanical Stim",
            "Patient Movement / Artifact",
        ):
            marker_menu.addAction(
                preset_label,
                lambda _checked=False, lbl=preset_label: self.add_quick_protocol_marker(lbl),
            )
        marker_menu.addSeparator()
        marker_menu.addAction("Custom Marker (M)...", self.prompt_add_protocol_marker)
        self.btn_add_marker.setMenu(marker_menu)
        self.toolbar.addWidget(self.btn_add_marker)

        for sig_name in (
            "onStreamStateChanged",
            "onNewSweep",
            "onStimNoChange",
            "onFollowLiveChanged",
            "onLoadNewFile",
        ):
            sig = getattr(self.state, sig_name, None)
            if sig is not None and hasattr(sig, "connect"):
                sig.connect(self._update_live_cockpit_bar)
        self._update_live_cockpit_bar()

        # Menu bar
        self.menubar = self.menuBar()

        file_menu = self.menubar.addMenu("&File")
        file_menu.addAction(
            QAction("Open...", self, shortcut="Ctrl+O", triggered=lambda: self.open())
        )
        file_menu.addAction(
            QAction(
                "Open Folder...",
                self,
                shortcut="Ctrl+Shift+O",
                triggered=self.open_folder,
            )
        )
        self.recent_menu = file_menu.addMenu("Open Recent")
        file_menu.addAction(
            QAction(
                "Save as nixio",
                self,
                shortcut="Ctrl+S",
                triggered=lambda: self.save_as(),
            )
        )
        file_menu.addAction(
            QAction(
                "Export spikes as csv",
                self,
                shortcut="Ctrl+E",
                triggered=lambda: self.export_csv(),
            )
        )
        file_menu.addAction(
            QAction(
                "import spikes csv",
                self,
                shortcut="Ctrl+I",
                triggered=lambda: self.import_csv(),
            )
        )

        self.updateRecents(None, None)

        self.settings_file.endArray()

        edit_menu = self.menubar.addMenu("&Edit")
        edit_menu.addAction(
            QAction(
                "undo", self, shortcut="Ctrl+Z", triggered=lambda: self.state.undo()
            )
        )
        edit_menu.addAction(
            QAction("save perspective", self, triggered=self.savePerspectives)
        )
        edit_menu.addAction(
            QAction("load perspective", self, triggered=self.loadPerspectives)
        )

        channels_menu = self.menubar.addMenu("&Channels")
        channels_menu.addAction(
            QAction(
                "New Channel Group Sub-Window...",
                self,
                triggered=lambda: self.newWindow("ChannelGroupContainer"),
            )
        )
        channels_menu.addAction(
            QAction(
                "Shared Channel Controller...",
                self,
                triggered=self.open_or_show_shared_channel_panel,
            )
        )
        channels_menu.addAction(
            QAction(
                "Manage Channel Groups...",
                self,
                triggered=lambda: ChannelGroupDialog(parent=self, state=self.state).exec(),
            )
        )
        channels_menu.addAction(
            QAction(
                "Open Dual Unit View (Unit 0 & Unit 1)...",
                self,
                triggered=self.open_dual_unit_view,
            )
        )

        view_menu = self.menubar.addMenu("&View")
        self.act_hide_top_bars = QAction("Remove View Top Bars", self, checkable=True)
        self.act_hide_top_bars.setChecked(getattr(self.state, "hide_view_top_bars", False))
        self.act_hide_top_bars.setToolTip(
            "Globally remove top bars from views so settings are controlled exclusively from the properties pane."
        )
        self.act_hide_top_bars.triggered.connect(lambda checked: self.state.set_hide_view_top_bars(checked))
        view_menu.addAction(self.act_hide_top_bars)
        view_menu.addAction(
            QAction("View Settings Panel...", self, triggered=self.open_or_show_view_settings_panel)
        )
        view_menu.addAction(
            QAction(
                "Pinned Settings for Active View",
                self,
                triggered=lambda: self.new_settings_panel(pinned=True),
            )
        )
        view_menu.addAction(
            QAction(
                "Channel & Unit Settings Panel",
                self,
                triggered=lambda: self.new_settings_panel(
                    page="channels", title="Channel & Unit Settings"
                ),
            )
        )
        view_menu.addAction(
            QAction(
                "Live Streaming Settings Panel",
                self,
                triggered=lambda: self.new_settings_panel(
                    page="stream", title="Streaming Settings"
                ),
            )
        )

        help_menu = self.menubar.addMenu("Help")

        def showInfo():
            info = InfoDialog()
            info.exec()

        help_menu.addAction(QAction("About", self, triggered=showInfo))
        help_menu.addAction(
            QAction(
                "Check for updates",
                self,
                triggered=lambda: self.version_check_thread.start(),
            )
        )

        # key shortcuts
        self.profiler = cProfile.Profile()
        self.is_profiling = False
        self.shortcut_profile = QShortcut(QKeySequence(Qt.Key_F1), self)

        def profile():
            if self.is_profiling:
                self.profiler.disable()
                self.is_profiling = False
                nom = QFileDialog.getSaveFileName(self, "Export profile")
                if nom is None:

                    return
                self.profiler.dump_stats(nom[0])
                self.profiler = cProfile.Profile()
                import subprocess

                subprocess.run(["snakeviz", nom[0]])

            else:
                self.is_profiling = True
                self.profiler.enable()

        self.shortcut_profile.activated.connect(lambda: profile())

        self.shortcut_next = QShortcut(
            QKeySequence(Qt.Key_Down), self, context=Qt.ApplicationShortcut
        )
        self.shortcut_next.activated.connect(lambda: self._navigate_stim(1))

        self.shortcut_skip_next = QShortcut(
            QKeySequence(Qt.Key_PageDown), self, context=Qt.ApplicationShortcut
        )
        self.shortcut_skip_next.activated.connect(lambda: self._navigate_stim(5))

        self.shortcut_prev = QShortcut(
            QKeySequence(Qt.Key_Up), self, context=Qt.ApplicationShortcut
        )
        self.shortcut_prev.activated.connect(lambda: self._navigate_stim(-1))

        self.shortcut_skip_prev = QShortcut(
            QKeySequence(Qt.Key_PageUp), self, context=Qt.ApplicationShortcut
        )
        self.shortcut_skip_prev.activated.connect(lambda: self._navigate_stim(-5))

        self.shortcut_del = QShortcut(
            QKeySequence(Qt.Key_Backspace), self, context=Qt.ApplicationShortcut
        )
        self.shortcut_del.activated.connect(self._on_backspace_pressed)

        self.shortcut_copylast = QShortcut(
            QKeySequence(Qt.Key_Space), self, context=Qt.ApplicationShortcut
        )
        self.shortcut_copylast.activated.connect(self._on_space_pressed)
        self.shortcut_follow_live = self.shortcut_copylast

        self.shortcut_copynext = QShortcut(
            QKeySequence(Qt.Modifier.SHIFT | Qt.Key.Key_Space),
            self,
            context=Qt.ApplicationShortcut,
        )
        self.shortcut_copynext.activated.connect(lambda: self._active_viewer_state().setUnit(-2))

        self.move_mode = "snap"

        def move(dist=1, mode=None):
            print(f"move {dist}")
            s = self._active_viewer_state()
            sg = s.spike_groups[s.cur_spike_group]
            sr = 10000.0
            if hasattr(s, "sampling_rate") and s.sampling_rate is not None:
                sr = float(s.sampling_rate.rescale(pq.Hz).magnitude) if hasattr(s.sampling_rate, "magnitude") else float(s.sampling_rate)
            elif s.analog_signal is not None and hasattr(s.analog_signal, "sampling_rate"):
                sr = float(s.analog_signal.sampling_rate.rescale(pq.Hz).magnitude) if hasattr(s.analog_signal.sampling_rate, "magnitude") else float(s.analog_signal.sampling_rate)

            cur_point = None
            if s.event_signal is not None and s.stimno < len(s.event_signal) and hasattr(sg, "get_latencies"):
                cur_ev = s.event_signal[s.stimno]
                lat = sg.get_latencies(np.array([cur_ev.magnitude]) * cur_ev.units)[0]
                if not np.isnan(lat):
                    cur_point = int(round(float((lat.rescale(pq.s) * sr).magnitude)))
            elif getattr(sg, "idx_arr", None) is not None and s.stimno < len(sg.idx_arr):
                item = sg.idx_arr[s.stimno]
                if item is not None:
                    cur_point = item[0]

            if cur_point is None:
                return

            if mode is None:
                mode = self.move_mode
            if mode == "snap":
                pts = s.get_peaks()[s.stimno]
                i = pts.searchsorted(cur_point)

                if dist > 0 and pts[i] != cur_point:
                    dist -= 1

                dist = pts[i + dist] - cur_point

            s.setUnit(cur_point + dist)  # TODO: make method

        self.shortcut_left = QShortcut(
            QKeySequence(Qt.Key_Left), self, context=Qt.ApplicationShortcut
        )
        self.shortcut_left.activated.connect(lambda: move(-1))

        self.shortcut_right = QShortcut(
            QKeySequence(Qt.Key_Right), self, context=Qt.ApplicationShortcut
        )
        self.shortcut_right.activated.connect(lambda: move(1))

        self.shortcut_numbers = [
            QShortcut(QKeySequence(Qt.Key_1), self, context=Qt.ApplicationShortcut),
            QShortcut(QKeySequence(Qt.Key_2), self, context=Qt.ApplicationShortcut),
            QShortcut(QKeySequence(Qt.Key_3), self, context=Qt.ApplicationShortcut),
            QShortcut(QKeySequence(Qt.Key_4), self, context=Qt.ApplicationShortcut),
            QShortcut(QKeySequence(Qt.Key_5), self, context=Qt.ApplicationShortcut),
            QShortcut(QKeySequence(Qt.Key_6), self, context=Qt.ApplicationShortcut),
            QShortcut(QKeySequence(Qt.Key_7), self, context=Qt.ApplicationShortcut),
            QShortcut(QKeySequence(Qt.Key_8), self, context=Qt.ApplicationShortcut),
            QShortcut(QKeySequence(Qt.Key_9), self, context=Qt.ApplicationShortcut),
        ]
        for i, x in enumerate(self.shortcut_numbers):
            x.activated.connect(lambda i=i: self.state.setUnitGroup(i))

        def add_select_new_unit():
            self.state.addUnitGroup()
            x = len(self.state.spike_groups)
            self.state.setUnitGroup(x - 1)

        self.shortcut_new_unit = QShortcut(
            QKeySequence(Qt.CTRL | Qt.Key_N), self, context=Qt.ApplicationShortcut
        )
        self.shortcut_new_unit.activated.connect(add_select_new_unit)

        def toggle_snap():
            self.move_mode = None if self.move_mode == "snap" else "snap"

        self.shortcut_snap = QShortcut(
            QKeySequence(Qt.Key_S), self, context=Qt.ApplicationShortcut
        )
        self.shortcut_snap.activated.connect(toggle_snap)

        def copy_previous():
            s = self._active_viewer_state()
            sg = s.spike_groups[s.cur_spike_group]
            sr = 10000.0
            if hasattr(s, "sampling_rate") and s.sampling_rate is not None:
                sr = float(s.sampling_rate.rescale(pq.Hz).magnitude) if hasattr(s.sampling_rate, "magnitude") else float(s.sampling_rate)
            elif s.analog_signal is not None and hasattr(s.analog_signal, "sampling_rate"):
                sr = float(s.analog_signal.sampling_rate.rescale(pq.Hz).magnitude) if hasattr(s.analog_signal.sampling_rate, "magnitude") else float(s.analog_signal.sampling_rate)

            new_x = None
            if s.event_signal is not None and s.stimno > 0 and hasattr(sg, "get_latencies"):
                search_evts = s.event_signal[0 : s.stimno]
                lats = sg.get_latencies(search_evts)
                for k in range(len(lats) - 1, -1, -1):
                    if not np.isnan(lats[k]):
                        new_x = int(round(float((lats[k].rescale(pq.s) * sr).magnitude)))
                        break
            elif getattr(sg, "idx_arr", None) is not None:
                try:
                    new_x = next(
                        sg.idx_arr[x][0]
                        for x in range(s.stimno - 1, -1, -1)
                        if sg.idx_arr[x] is not None
                    )
                except StopIteration:
                    pass

            if new_x is not None:
                # localise to nearest peak
                cur_erp = s.get_erp()[s.stimno]
                start_w = max(0, new_x - 300)
                end_w = min(len(cur_erp), new_x + 300)
                if end_w > start_w:
                    offset = np.argmax(cur_erp[start_w:end_w]) - (new_x - start_w)
                    s.setUnit(new_x + offset)

        self.shortcut_copy_previous = QShortcut(
            QKeySequence(Qt.Key_C), self, context=Qt.ApplicationShortcut
        )
        self.shortcut_copy_previous.activated.connect(copy_previous)

        self.shortcut_toggle_live_events = QShortcut(
            QKeySequence("Ctrl+L"), self, context=Qt.ApplicationShortcut
        )
        self.shortcut_toggle_live_events.activated.connect(self._toggle_follow_live)

        self.shortcut_marker = QShortcut(
            QKeySequence(Qt.Key_M), self, context=Qt.ApplicationShortcut
        )
        self.shortcut_marker.activated.connect(self.prompt_add_protocol_marker)

    def _active_viewer_state(self):
        """State backing the focused view (a container's local overlay if any)."""
        panel = getattr(self, "settings_panel", None)
        view = getattr(panel, "active_view", None) if panel is not None else None
        if view is not None:
            state = getattr(view, "view_state", None) or getattr(view, "state", None)
            if state is not None:
                return state
        return self.state

    def _navigate_stim(self, delta: int):
        state = self._active_viewer_state()
        if state is not None:
            state.setStimNo(state.stimno + delta)

    def _toggle_follow_live(self):
        state = self._active_viewer_state()
        if state is not None:
            state.set_follow_live(not bool(getattr(state, "follow_live", True)))
            self._update_live_cockpit_bar()
            return True
        return False

    def _on_space_pressed(self):
        if not self._toggle_follow_live():
            state = self._active_viewer_state()
            if state is not None:
                state.setUnit(-1)

    def _on_backspace_pressed(self):
        s = self._active_viewer_state()
        if s is None:
            return
        if getattr(s, "is_streaming", False) and getattr(s, "follow_live", True):
            return
        s.setUnit(None)

    def add_quick_protocol_marker(self, label: str, category: str = "protocol"):
        if self.state is not None and hasattr(self.state, "add_protocol_marker"):
            return self.state.add_protocol_marker(
                label, stimno=self.state.stimno, category=category
            )
        return None

    def prompt_add_protocol_marker(self):
        presets = [
            "Baseline Start",
            "2 Hz Tetanus Start",
            "2 Hz Tetanus End",
            "Recovery",
            "Mechanical Stim",
            "Patient Movement / Artifact",
        ]
        app = QApplication.instance()
        if app is not None and app.platformName() == "offscreen":
            return self.add_quick_protocol_marker("Marker")
        text, ok = QInputDialog.getItem(
            self,
            "Add Protocol Marker",
            f"Marker at Sweep #{int(getattr(self.state, 'stimno', 0) or 0)}:",
            presets,
            0,
            True,
        )
        if ok and text and text.strip():
            return self.add_quick_protocol_marker(text.strip())
        return None

    @Slot()
    @Slot(bool)
    @Slot(int)
    def _update_live_cockpit_bar(self, *args):
        if not hasattr(self, "lbl_live_status_pill") or not hasattr(self, "btn_follow_live_pill"):
            return
        state = self.state
        is_streaming = bool(getattr(state, "is_streaming", False)) if state is not None else False
        follow_live = bool(getattr(state, "follow_live", True)) if state is not None else True
        ev_sig = getattr(state, "event_signal", None) if state is not None else None
        n_sweeps = len(ev_sig) if ev_sig is not None else 0
        stimno = int(getattr(state, "stimno", 0) or 0) if state is not None else 0

        if not is_streaming and n_sweeps == 0:
            self.lbl_live_status_pill.setText("○ Offline | Sweep #0")
            self.lbl_live_status_pill.setStyleSheet(
                "background: #f1f5f9; color: #475569; border: 1px solid #cbd5e1; "
                "border-radius: 10px; padding: 2px 8px; font-weight: bold;"
            )
        elif not follow_live:
            total_idx = max(0, n_sweeps - 1)
            self.lbl_live_status_pill.setText(f"⏸ PAUSED (#{stimno} / #{total_idx})")
            self.lbl_live_status_pill.setStyleSheet(
                "background: #fef3c7; color: #92400e; border: 1px solid #f59e0b; "
                "border-radius: 10px; padding: 2px 8px; font-weight: bold;"
            )
        elif is_streaming:
            self.lbl_live_status_pill.setText(f"● LIVE | Sweep #{stimno}")
            self.lbl_live_status_pill.setStyleSheet(
                "background: #dcfce7; color: #166534; border: 1px solid #86efac; "
                "border-radius: 10px; padding: 2px 8px; font-weight: bold;"
            )
        else:
            total_idx = max(0, n_sweeps - 1)
            self.lbl_live_status_pill.setText(f"○ Offline | Sweep #{stimno} / #{total_idx}")
            self.lbl_live_status_pill.setStyleSheet(
                "background: #f1f5f9; color: #475569; border: 1px solid #cbd5e1; "
                "border-radius: 10px; padding: 2px 8px; font-weight: bold;"
            )

        self.btn_follow_live_pill.blockSignals(True)
        self.btn_follow_live_pill.setChecked(follow_live)
        if not follow_live and n_sweeps > 0:
            self.btn_follow_live_pill.setText("⏸ History (Click to Follow Live)")
            self.btn_follow_live_pill.setStyleSheet(
                "background: #fef3c7; color: #92400e; border: 1px solid #f59e0b; "
                "border-radius: 6px; padding: 2px 8px; font-weight: bold;"
            )
        elif follow_live:
            self.btn_follow_live_pill.setText("📡 Following Live")
            self.btn_follow_live_pill.setStyleSheet("")
        else:
            self.btn_follow_live_pill.setText("📡 Follow Live")
            self.btn_follow_live_pill.setStyleSheet("")
        self.btn_follow_live_pill.blockSignals(False)

    def export_csv(self):
        save_filename = QFileDialog.getSaveFileName(self, "Export")[0]
        if not save_filename:
            return
        export_units_csv(save_filename, unit_evts=self.state.spike_groups, stim_evt=self.state.event_signal)

    def updateRecents(self, filename, type):

        if filename is not None:
            to_save = [(filename, type)]
            for i in range(self.settings_file.beginReadArray("Recent")):
                self.settings_file.setArrayIndex(i)
                prevVal = self.settings_file.value("filename")
                if prevVal == filename:
                    continue
                to_save.append((prevVal, self.settings_file.value("type", "h5")))
            self.settings_file.endArray()

            self.settings_file.beginWriteArray("Recent")
            for i, (fname, ftype) in enumerate(to_save):
                self.settings_file.setArrayIndex(i)
                self.settings_file.setValue("filename", fname)
                self.settings_file.setValue("type", ftype)
            self.settings_file.endArray()

        self.recent_menu.clear()
        for i in range(self.settings_file.beginReadArray("Recent")):
            self.settings_file.setArrayIndex(i)
            fname = self.settings_file.value("filename")
            ftype = self.settings_file.value("type", "h5")
            self.recent_menu.addAction(
                QAction(
                    Path(fname).name,
                    self,
                    triggered=lambda n, fname=fname, ftype=ftype: self.loadFile.emit(
                        fname, ftype
                    ),
                )
            )

    def export_npz(self):
        save_filename = QFileDialog.getSaveFileName(self, "Export NPZ", filter="*.npz")[0]
        if not save_filename:
            return
        export_units_npz(save_filename, self.state)

    def import_csv(self):
        open_filename = QFileDialog.getOpenFileName(self, "csv import")[0]
        if not open_filename:
            return
        imported_units = import_units_csv(open_filename)
        if imported_units:
            self.state.set_data(spike_groups=(self.state.spike_groups or []) + imported_units)

    def newWindow(
        self,
        k,
        pos=QtAds.TopDockWidgetArea,
        name=None,
        target_unit=None,
        channels=None,
        shared_set=None,
        wrap_in_channel_group=None,
    ):
        # 1. Singleton OpenEphysLive view: reuse existing instance without re-instantiating
        if k == "OpenEphysLive" and getattr(self, "_open_ephys_live_dock", None) is not None:
            dock = self._open_ephys_live_dock
            if dock.isClosed() or not dock.isVisible():
                dock.toggleView(True)
                if pos is not None and pos != QtAds.NoDockWidgetArea:
                    self.dock_manager.addDockWidget(pos, dock)
            dock.show()
            dock.raise_()
            return dock.widget()

        # 2. Channel group container wrapping:
        # If requested or if component is channel/unit specific, wrap into a ChannelGroupContainer
        channel_specific_views = (
            "MultiTrace",
            "MultiTraceFixedView",
            "UnitView",
            "UnitFootprintView",
            "SingleTraceView",
            "MultiTrace (PyQtGraph)",
            "MultiTraceFixed (PyQtGraph)",
            "SingleTrace (PyQtGraph)",
            "UnitView (PyQtGraph)",
            "UnitFootprint (PyQtGraph)",
        )
        should_wrap = wrap_in_channel_group or (
            wrap_in_channel_group is None and (target_unit is not None or channels is not None)
        )

        if should_wrap and k in channel_specific_views and k in self.window_options:
            container = ChannelGroupContainerView(parent=self, state=self.state)
            inner_w = self.window_options[k](parent=container, state=self.state)
            container.add_view(inner_w, title=k)
            if shared_set is not None:
                container.channel_bar.bind_to_shared_set(shared_set)
            if target_unit is not None:
                container.channel_bar.bind_to_unit(target_unit)
                if channels is not None:
                    container.channel_bar.bind_to_channels(channels)
                elif shared_set is None and hasattr(container.channel_bar, "set_channel_mode_unit"):
                    container.channel_bar.set_channel_mode_unit()
            elif channels is not None:
                container.channel_bar.bind_to_channels(channels)
            w = container
        else:
            w = self.window_options[k](parent=self, state=self.state)

            # Apply target unit, channels, or shared set binding if component supports it
            if hasattr(w, "channel_bar") and w.channel_bar is not None:
                # By default, hide the top channel bar on individual views so the simplified selector
                # lives in the View Settings side panel and ChannelGroupContainer headers.
                if not isinstance(w, ChannelGroupContainerView):
                    w.channel_bar.setVisible(False)
                if shared_set is not None:
                    w.channel_bar.bind_to_shared_set(shared_set)
                if target_unit is not None:
                    w.channel_bar.bind_to_unit(target_unit)
                    if channels is not None:
                        w.channel_bar.bind_to_channels(channels)
                    elif shared_set is None and hasattr(w.channel_bar, "set_channel_mode_unit"):
                        w.channel_bar.set_channel_mode_unit()
                elif channels is not None:
                    w.channel_bar.bind_to_channels(channels)

        window_no = len(
            [
                x
                for x in self.dock_manager.dockWidgets()
                if isinstance(x.widget(), self.window_options[k])
            ]
        )

        if name is None:
            if isinstance(w, ChannelGroupContainerView):
                prefix = f"Channel Group [{k}]"
                if target_unit is not None:
                    prefix += f" (Unit {target_unit})"
                name = prefix
            elif shared_set is not None:
                name = f"{k} [{shared_set}]"
            elif target_unit is not None and self.state and 0 <= target_unit < len(self.state.spike_groups):
                u_name = getattr(self.state.spike_groups[target_unit].event, "name", f"unit_{target_unit}")
                name = f"{k} [Unit {target_unit}: {u_name}]"
            else:
                name = k + (f" {window_no}" if window_no > 0 else "")

        w2 = QtAds.CDockWidget(name)
        w2.setWidget(w)
        if k == "OpenEphysLive":
            w2.setFeature(QtAds.CDockWidget.DockWidgetDeleteOnClose, False)
            self._open_ephys_live_dock = w2
        else:
            w2.setFeature(QtAds.CDockWidget.DockWidgetDeleteOnClose, True)
        w2.setFeature(QtAds.CDockWidget.DockWidgetForceCloseWithArea, True)
        if pos is not None and pos != QtAds.NoDockWidgetArea:
            self.dock_manager.addDockWidget(pos, w2)
        else:
            self.dock_manager.addDockWidget(QtAds.NoDockWidgetArea, w2)
        self.cur_windows.append(w2)
        w2.show()
        area = w2.dockAreaWidget()
        if area is not None and not getattr(area, "_spikespy_current_changed_connected", False):
            area.currentChanged.connect(self._on_dock_area_current_changed)
            area._spikespy_current_changed_connected = True
        if getattr(self.state, "hide_view_top_bars", False):
            apply_top_bars_visibility(w, True)
        if hasattr(self, "settings_panel") and self.settings_panel is not None:
            notify_view_selected(w)
        self._refresh_all_settings_panel_targets()
        return w

    def open_dual_unit_view(self):
        """Convenience action to open comparison windows for Unit 0 and Unit 1 side-by-side."""
        if not self.state or len(self.state.spike_groups) < 1:
            QMessageBox.information(self, "No Units", "No tracked units available in state.")
            return
        w1 = self.newWindow("MultiTrace", target_unit=0)
        if len(self.state.spike_groups) >= 2:
            w2 = self.newWindow("MultiTrace", target_unit=1)

    @Slot()
    def open(self, type=None):
        """
        triggered on file->open
        """
        fname, type = prompt_for_neo_file(type)
        if not fname:
            return
        self.loadFile.emit(fname, type)

    @Slot()
    def open_folder(self):
        """File -> Open Folder... for folder-based formats (Open Ephys, APTrack, MATLAB)."""
        fname, type = prompt_for_neo_folder(self)
        if not fname:
            return
        self.loadFile.emit(fname, type)

    @Slot(str, str)
    def _dispatch_load_file(self, fname, type):
        """Route file loads through the state's non-blocking loader."""
        self.state.loadFile(fname, type, blocking=False)

    @Slot()
    def save_as(
        self,
    ):  # TODO: move out
        """
        triggered on file->save
        """
        fname = QFileDialog.getSaveFileName(self, "Save as", filter="*.h5 *.nwb")[0]
        # s = neo.Segment()
        if fname == "":
            return

        if self.state.segment is None:
            s = neo.Segment()

            s.events.append(self.state.event_signal)
            s.analogsignals.append(self.state.analog_signal)
        else:
            s = self.state.segment
        save_file(
            fname,
            self.state.spike_groups,
            s,
            candidates=self.state.export_candidate_store(),
        )


save_file = save_state_to_nix


def align_spikegroup(spikegroup, erp_arr, event_signal=None, sampling_rate=10000):
    """given a spike group attempt to align using convolution - note this should go outside of the UI component"""
    window = 200
    arr = []
    if event_signal is not None and hasattr(spikegroup, "get_latencies"):
        lats = spikegroup.get_latencies(event_signal)
        sr = float(sampling_rate.rescale(pq.Hz).magnitude) if hasattr(sampling_rate, "magnitude") else float(sampling_rate)
        for i, lat in enumerate(lats):
            if not np.isnan(lat) and i < erp_arr.shape[0]:
                c_s = int(round(float((lat.rescale(pq.s) * sr).magnitude)))
                if c_s - window >= 0 and c_s + window <= erp_arr.shape[1]:
                    arr.append(erp_arr[i, c_s - window : c_s + window])
    elif getattr(spikegroup, "idx_arr", None) is not None:
        for i, x in enumerate(spikegroup.idx_arr):
            if x is not None and i < erp_arr.shape[0]:
                c_s = x[0]
                if c_s - window >= 0 and c_s + window <= erp_arr.shape[1]:
                    arr.append(erp_arr[i, c_s - window : c_s + window])

    if len(arr) > 0:
        np.convolve(arr, arr, axis=-1)
    pass


def run(filename=None):
    app = QApplication.instance() or QApplication(sys.argv)
    icon_path = Path(sys.modules[__name__].__file__).parent.joinpath("ui/icon.svg")
    app.setWindowIcon(QIcon(QPixmap(str(icon_path))))
    # data, signal_chan, event_signal, spike_groups = load_file(
    # )
    # w = TraceView(
    #     analog_signal=signal_chan, event_signal=event_signal, spike_groups=spike_groups
    # )
    w = MdiView()
    if filename is not None:
        w.state.loadFile(filename, blocking=False)
    elif len(sys.argv) > 1:
        w.state.loadFile(sys.argv[1], blocking=False)

    # w = SpikeGroupView()
    w.showMaximized()
    print(app)
    sys.exit(app.exec())


from importlib.metadata import version, metadata, packages_distributions


def run_spikespy(viewerState):
    app = QApplication.instance() or QApplication(sys.argv)
    icon_path = Path(sys.modules[__name__].__file__).parent.joinpath("ui/icon.svg")
    app.setWindowIcon(QIcon(QPixmap(str(icon_path))))
    # data, signal_chan, event_signal, spike_groups = load_file(
    # )
    # w = TraceView(
    #     analog_signal=signal_chan, event_signal=event_signal, spike_groups=spike_groups
    # )
    w = MdiView(state=viewerState)
    w.showMaximized()
    app.exec()

    # w = SpikeGroupView()


class InfoDialog(QDialog):
    # show license info and about

    def __init__(self):
        super().__init__()
        # self.textwidget = QScrollArea()
        self.textarea = QTextEdit()

        self.license_text = self.buid_licence_text()
        self.textarea.setText(self.license_text)
        # self.textwidget.setWidget(self.textarea)
        lo = QVBoxLayout()
        self.setLayout(lo)

        lo.addWidget(QLabel("SpikeSpy"))
        lo.addWidget(QLabel("Open source packages & licenses"))
        lo.addWidget(self.textarea)

    def buid_licence_text(self):
        out = ""
        packages = sorted(list(set(sum(packages_distributions().values(), []))))
        for x in packages:
            md = metadata(x)
            out += f"""## {md.get("name")}
            """
            for k in ["version", "author", "license", "Project-URL"]:
                if k not in md:
                    continue
                out += f"\n{k}: {md.get(k)}"

            out += "\n\n"
        return out


export_csv = export_units_csv


def register_plugin(plugin_name, plugin_class):
    is_suggestor = isinstance(plugin_class, UnitSuggestor) or (
        isinstance(plugin_class, type) and issubclass(plugin_class, UnitSuggestor)
    )
    if is_suggestor:
        suggestors[plugin_name] = plugin_class
    else:
        window_options[plugin_name] = plugin_class


if __name__ == "__main__":
    run()
