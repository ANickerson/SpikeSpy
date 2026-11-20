import logging
from typing import Any, Dict, List, Optional, Union

import PySide6
from PySide6.QtCore import QSignalBlocker, Qt, Signal, Slot
from PySide6.QtGui import QAction, QIcon, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
import PySide6QtAds as QtAds

from .ChannelGroupBar import ChannelGroupBar
from .ViewerState import ViewerState
from .helpers import qsignal_throttle, notify_view_selected
from .live_orchestrator import LiveFrameConsumerMixin


class SubViewCard(QFrame):
    """
    Sub-window frame within a ChannelGroupContainer.
    Maintained for backward compatibility.
    """

    closeRequested = Signal(QWidget)

    def __init__(self, title: str, child_widget: QWidget, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.child_widget = child_widget
        self.setFrameShape(QFrame.StyledPanel)
        self.setFrameShadow(QFrame.Raised)
        self.setStyleSheet(
            "SubViewCard { border: 1px solid #c0c0c0; border-radius: 4px; margin: 2px; background: #ffffff; }"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(1)

        # Header bar
        header = QWidget(self)
        header.setStyleSheet("background: #f0f0f4; border-bottom: 1px solid #d0d0d8; padding: 2px;")
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(4, 2, 4, 2)

        self.lbl_title = QLabel(f"<b>{title}</b>", header)
        h_layout.addWidget(self.lbl_title)
        h_layout.addStretch()

        self.btn_close = QPushButton("✕", header)
        self.btn_close.setFixedSize(18, 18)
        self.btn_close.setToolTip("Close this sub-window")
        self.btn_close.setStyleSheet(
            "QPushButton { border: none; font-weight: bold; color: #666; } QPushButton:hover { color: #d00; }"
        )
        self.btn_close.clicked.connect(lambda: self.closeRequested.emit(self))
        h_layout.addWidget(self.btn_close)

        layout.addWidget(header)
        layout.addWidget(child_widget, 1)


class ChannelGroupContainerView(LiveFrameConsumerMixin, QMainWindow):
    """
    Sub-window container representing a Channel Group within SpikeSpy.
    Can host multiple nested sub-windows (e.g. MultiTrace, SingleTrace, UnitView)
    using the advanced QtAds docking system.

    Features:
    - Nested advanced window management: Drag-and-drop docking, horizontal/vertical splits,
      grid arrangements, and tabbed grouping within each channel group window.
    - Follow Live Inputs option: dynamically tracks live streaming channels.
    - Deferral of child channel / unit selection to this container's ChannelGroupBar.
    - Ability to nest channel group containers within other channel group containers.
    """

    viewAdded = Signal(QWidget)
    viewRemoved = Signal(QWidget)

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        state: Optional[ViewerState] = None,
        name: str = "Channel Group",
        **kwargs,
    ):
        super().__init__(parent)
        self.state = state
        self.view_state = None
        self.container_name = name
        self.sub_views: List[QWidget] = []
        self._cards: List[Any] = []
        self._dock_map: Dict[QWidget, QtAds.CDockWidget] = {}
        self._current_orientation: str = "vertical"
        self._current_layout: str = "vertical"
        self._connections: List = []
        self._on_new_sweep_throttled = qsignal_throttle(self._on_new_sweep_core, 50)

        # 1. Advanced Docking Manager for nested sub-windows
        self.dock_manager = QtAds.CDockManager(self)
        self.dock_manager.dockAreaCreated.connect(self._on_dock_area_created)
        self.dock_manager.focusedDockWidgetChanged.connect(self._on_focused_dock_widget_changed)

        # 2. Top Channel Group Bar
        self.channel_bar = ChannelGroupBar(parent=self, state=state)
        self.channel_bar.bindingChanged.connect(self._on_channel_bar_changed)
        self.addToolBar(Qt.TopToolBarArea, self.channel_bar)

        # 3. Controls Toolbar for adding child views, arrangements, and live input tracking
        self.control_bar = QToolBar("Sub-Views", self)

        # Add Sub-View button
        self.btn_add_subview = QToolButton(self)
        self.btn_add_subview.setText("+ Add Sub-View ▾")
        self.btn_add_subview.setPopupMode(QToolButton.InstantPopup)
        self._build_add_menu()
        self.control_bar.addWidget(self.btn_add_subview)

        # Layout arrangements dropdown
        self.btn_layouts = QToolButton(self)
        self.btn_layouts.setText("📐 Layout ▾")
        self.btn_layouts.setPopupMode(QToolButton.InstantPopup)
        self._build_layout_menu()
        self.control_bar.addWidget(self.btn_layouts)

        # Split orientation toggle button
        self.btn_orient = QPushButton("⬍ Horizontal / Vertical", self)
        self.btn_orient.setToolTip("Toggle horizontal or vertical arrangement of sub-windows")
        self.btn_orient.clicked.connect(self._toggle_orientation)
        self.control_bar.addWidget(self.btn_orient)

        # Follow Live Input checkbox
        self.chk_follow_live = QCheckBox("📡 Follow Live", self)
        self.chk_follow_live.setToolTip(
            "When checked, this channel group automatically follows live streaming inputs."
        )
        self.chk_follow_live.toggled.connect(self._on_follow_live_toggled)
        self.control_bar.addWidget(self.chk_follow_live)

        # Follow Live Events checkbox
        self.chk_follow_events = QCheckBox("Follow Events", self)
        self.chk_follow_events.setToolTip(
            "Follow live new events (keep event index in sync with latest). Shortcut: Ctrl+L"
        )
        self.chk_follow_events.toggled.connect(self._on_follow_events_toggled)
        self.control_bar.addWidget(self.chk_follow_events)

        # Sync Visuals checkbox
        self.chk_sync_visuals = QCheckBox("🔗 Sync Visuals", self)
        self.chk_sync_visuals.setChecked(True)
        self.chk_sync_visuals.setToolTip(
            "Synchronize visual settings (scale mode, color scale, trace scale, X range) across views in this container."
        )
        self.control_bar.addWidget(self.chk_sync_visuals)

        self.addToolBar(Qt.TopToolBarArea, self.control_bar)

        if state is not None:
            self.set_state(state)

    @property
    def splitter(self):
        """Compatibility proxy for code or tests accessing .splitter.orientation()."""
        container = self

        class _SplitterCompat:
            def orientation(self):
                return Qt.Horizontal if container._current_orientation == "horizontal" else Qt.Vertical

            def setOrientation(self, orient):
                mode = "horizontal" if orient == Qt.Horizontal else "vertical"
                container.arrange_layout(mode)

        return _SplitterCompat()

    def _build_add_menu(self):
        menu = QMenu(self.btn_add_subview)

        # Trace & Heatmaps
        m_traces = menu.addMenu("Traces & Heatmaps")
        m_traces.addAction("MultiTrace (Matplotlib)", lambda: self.add_view_by_key("MultiTrace"))
        m_traces.addAction("MultiTrace (PyQtGraph)", lambda: self.add_view_by_key("MultiTracePG"))
        m_traces.addAction("SingleTrace (Matplotlib)", lambda: self.add_view_by_key("SingleTraceView"))
        m_traces.addAction("SingleTrace (PyQtGraph)", lambda: self.add_view_by_key("SingleTracePG"))
        m_traces.addAction("MultiTraceFixed (Matplotlib)", lambda: self.add_view_by_key("MultiTraceFixedView"))
        m_traces.addAction("MultiTraceFixed (PyQtGraph)", lambda: self.add_view_by_key("MultiTraceFixedPG"))

        # Unit Analysis
        m_units = menu.addMenu("Unit Analysis")
        m_units.addAction("UnitView (Matplotlib)", lambda: self.add_view_by_key("UnitView"))
        m_units.addAction("UnitView (PyQtGraph)", lambda: self.add_view_by_key("UnitPG"))
        m_units.addAction("UnitFootprint (Matplotlib)", lambda: self.add_view_by_key("UnitFootprintView"))
        m_units.addAction("UnitFootprint (PyQtGraph)", lambda: self.add_view_by_key("UnitFootprintPG"))

        # Nested Channel Group
        menu.addSeparator()
        menu.addAction("Nested Channel Group Container", lambda: self.add_view_by_key("ChannelGroupContainer"))

        self.btn_add_subview.setMenu(menu)

    def _build_layout_menu(self):
        menu = QMenu(self.btn_layouts)
        menu.addAction("Vertical Split (Stacked)", lambda: self.arrange_layout("vertical"))
        menu.addAction("Horizontal Split (Side by Side)", lambda: self.arrange_layout("horizontal"))
        menu.addAction("Grid (2x2)", lambda: self.arrange_layout("grid"))
        menu.addAction("Tabbed (All Tabs)", lambda: self.arrange_layout("tabbed"))
        self.btn_layouts.setMenu(menu)

    def _toggle_orientation(self):
        new_mode = "horizontal" if self._current_orientation == "vertical" else "vertical"
        self.arrange_layout(new_mode)

    def arrange_layout(self, mode: str = "vertical"):
        """
        Arranges the child dock widgets into a specific layout arrangement:
        - 'vertical': Stacked vertically
        - 'horizontal': Side by side horizontally
        - 'grid': 2-column grid
        - 'tabbed': All tabbed into one central area
        """
        self._current_layout = mode
        mode_lower = mode.lower()
        if mode_lower in ("horizontal", "vertical"):
            self._current_orientation = mode_lower

        docks = [self._dock_map[v] for v in self.sub_views if v in self._dock_map]
        if not docks:
            return

        if mode_lower == "horizontal":
            self._current_orientation = "horizontal"
            for i, d in enumerate(docks):
                if i == 0:
                    self.dock_manager.addDockWidget(QtAds.LeftDockWidgetArea, d)
                else:
                    self.dock_manager.addDockWidget(QtAds.RightDockWidgetArea, d)

        elif mode_lower == "vertical":
            self._current_orientation = "vertical"
            for i, d in enumerate(docks):
                if i == 0:
                    self.dock_manager.addDockWidget(QtAds.TopDockWidgetArea, d)
                else:
                    self.dock_manager.addDockWidget(QtAds.BottomDockWidgetArea, d)

        elif mode_lower == "tabbed":
            for i, d in enumerate(docks):
                if i == 0:
                    self.dock_manager.addDockWidget(QtAds.CenterDockWidgetArea, d)
                else:
                    self.dock_manager.addDockWidgetTab(QtAds.CenterDockWidgetArea, d)

        elif mode_lower == "grid":
            for i, d in enumerate(docks):
                if i == 0:
                    self.dock_manager.addDockWidget(QtAds.LeftDockWidgetArea, d)
                elif i == 1:
                    self.dock_manager.addDockWidget(QtAds.RightDockWidgetArea, d)
                elif i % 2 == 0:
                    self.dock_manager.addDockWidget(QtAds.BottomDockWidgetArea, d)
                else:
                    self.dock_manager.addDockWidget(QtAds.BottomDockWidgetArea, d)

    def is_following_live(self) -> bool:
        """Returns True if this channel group container is set to follow live streaming inputs."""
        return self.chk_follow_live.isChecked()

    def set_follow_live(self, enable: bool = True):
        """Enables or disables following live inputs."""
        self.chk_follow_live.setChecked(enable)

    def is_following_live_events(self) -> bool:
        """Determines if this channel group is following live new events (i.e. idx kept in sync with latest)."""
        if hasattr(self, "channel_bar") and self.channel_bar is not None:
            return self.channel_bar.is_following_live_events()
        local_state = getattr(self, "view_state", None) or self.state
        if local_state is not None and hasattr(local_state, "is_following_live_events"):
            return local_state.is_following_live_events()
        return self.chk_follow_events.isChecked()

    def set_follow_live_events(self, enable: bool = True):
        """Enables or disables following live new events."""
        if hasattr(self, "channel_bar") and self.channel_bar is not None:
            self.channel_bar.set_follow_live_events(enable)
        else:
            local_state = getattr(self, "view_state", None) or self.state
            if local_state is not None and hasattr(local_state, "set_follow_live"):
                local_state.set_follow_live(enable)
        self._sync_follow_events_checkbox()

    def toggle_follow_live_events(self):
        """Toggles following live new events."""
        self.set_follow_live_events(not self.is_following_live_events())

    def _on_follow_events_toggled(self, checked: bool):
        self.set_follow_live_events(checked)

    def _sync_follow_events_checkbox(self, *args):
        following = self.is_following_live_events()
        if self.chk_follow_events.isChecked() != following:
            with QSignalBlocker(self.chk_follow_events):
                self.chk_follow_events.setChecked(following)

    def _on_follow_live_toggled(self, checked: bool):
        if checked:
            self.channel_bar.set_channel_mode_live()
        else:
            self.channel_bar.set_channel_mode_global()
        self._sync_all_children()

    def _on_channel_bar_changed(self):
        mode, _ = self.channel_bar.get_channel_mode()
        is_live = mode == ChannelGroupBar.MODE_FOLLOW_LIVE
        if self.chk_follow_live.isChecked() != is_live:
            self.chk_follow_live.blockSignals(True)
            self.chk_follow_live.setChecked(is_live)
            self.chk_follow_live.blockSignals(False)
        self._sync_all_children()

    def _on_new_sweep_core(self, stimno: Optional[int] = None):
        if self.is_following_live():
            avail = tuple(self.state.get_available_channels()) if (self.state and hasattr(self.state, "get_available_channels")) else ()
            n_units = len(self.state.spike_groups) if (self.state and hasattr(self.state, "spike_groups")) else 0
            sig = (avail, n_units, self.channel_bar.get_bound_unit())
            if getattr(self, "_last_live_sync_sig", None) != sig:
                self._last_live_sync_sig = sig
                self._sync_all_children()

    def _on_new_sweep(self, stimno: Optional[int] = None):
        if hasattr(self, "_on_new_sweep_throttled"):
            self._on_new_sweep_throttled(stimno)
        else:
            self._on_new_sweep_core(stimno)

    def set_state(self, state: ViewerState):
        from .container_view_state import ContainerViewerState

        self.state = state
        if state is None:
            self.view_state = None
        elif getattr(self, "view_state", None) is None:
            self.view_state = ContainerViewerState(state, parent=self)
        else:
            self.view_state.rebase(state)
        child_state = self.view_state if self.view_state is not None else state

        self.channel_bar.setState(child_state)
        for v in self.sub_views:
            if hasattr(v, "set_state"):
                v.set_state(child_state)
            elif hasattr(v, "setState"):
                v.setState(child_state)

        for sig, slot in self._connections:
            try:
                sig.disconnect(slot)
            except (TypeError, RuntimeError):
                pass
        self._connections.clear()

        # Live sweeps arrive through the state's LiveOrchestrator (LIFO);
        # the GUI-side throttle keeps child syncing rate-limited.
        self._unregister_live_frames()
        self._register_live_frames(state, lambda stimno: self._on_new_sweep(stimno))

        if child_state is not None and hasattr(child_state, "onFollowLiveChanged"):
            child_state.onFollowLiveChanged.connect(self._sync_follow_events_checkbox)
            self._connections.append((child_state.onFollowLiveChanged, self._sync_follow_events_checkbox))
        if child_state is not None and hasattr(child_state, "onStimNoChange"):
            child_state.onStimNoChange.connect(self._sync_follow_events_checkbox)
            self._connections.append((child_state.onStimNoChange, self._sync_follow_events_checkbox))
        self._sync_follow_events_checkbox()

    def add_view_by_key(self, key: str) -> Optional[QWidget]:
        """Creates a child view by string key and adds it into this container."""
        from .SpikeSpy import window_options

        # Support alias mapping for both standard and PG views
        alias_map = {
            "MultiTracePG": "MultiTrace (PyQtGraph)",
            "SingleTracePG": "SingleTrace (PyQtGraph)",
            "MultiTraceFixedPG": "MultiTraceFixed (PyQtGraph)",
            "UnitPG": "UnitView (PyQtGraph)",
            "UnitFootprintPG": "UnitFootprint (PyQtGraph)",
            "ChannelGroupContainer": "ChannelGroupContainer",
        }
        lookup = alias_map.get(key, key)
        cls = window_options.get(lookup, window_options.get(key))
        if cls is None:
            # Fallback direct import attempts if not in window_options yet
            if "PG" in key or "PyQtGraph" in lookup:
                try:
                    if "MultiTraceFixed" in key:
                        from .MultiTraceFixedPGView import MultiTraceFixedPGView as cls
                    elif "MultiTrace" in key:
                        from .MultiTracePGView import MultiTracePGView as cls
                    elif "SingleTrace" in key:
                        from .SingleTracePGView import SingleTracePGView as cls
                    elif "UnitFootprint" in key or "Footprint" in key:
                        from .UnitFootprintPGView import UnitFootprintPGView as cls
                    elif "Unit" in key:
                        from .UnitPGView import UnitPGView as cls
                except Exception as e:
                    logging.warning(f"Could not import PG view {key}: {e}")
            elif key == "ChannelGroupContainer":
                cls = ChannelGroupContainerView

        if cls is None:
            logging.error(f"Unknown sub-view type: {key}")
            return None

        child = cls(parent=self, state=self.view_state or self.state)
        self.add_view(child, title=key)
        return child

    def add_view(self, child: QWidget, title: Optional[str] = None, pos: int = QtAds.TopDockWidgetArea):
        """
        Embeds a child view into this container using the nested docking manager.
        Hides the child's own channel selector toolbar and binds it
        to this container's ChannelGroupBar.
        """
        if title is None:
            title = child.__class__.__name__

        # Views embedded here browse stimuli through this container's local
        # overlay state so each channel group can follow (or not) independently.
        if self.view_state is not None and getattr(child, "state", None) is not self.view_state:
            if hasattr(child, "set_state"):
                child.set_state(self.view_state)
            elif hasattr(child, "setState"):
                child.setState(self.view_state)

        # Defer channel selector: hide child's own channel bar if present
        if hasattr(child, "channel_bar") and child.channel_bar is not None:
            child.channel_bar.setVisible(False)

        # Wrap in CDockWidget for advanced nested window management
        dock = QtAds.CDockWidget(title)
        dock.setWidget(child)
        dock.setFeature(QtAds.CDockWidget.DockWidgetDeleteOnClose, True)
        dock.closeRequested.connect(lambda: self.remove_view(child))
        dock.closed.connect(lambda: self._on_dock_closed(child))

        self.sub_views.append(child)
        self._cards.append(child)
        self._dock_map[child] = dock

        # Determine dock placement
        if len(self.sub_views) > 1:
            if self._current_orientation == "horizontal":
                area = QtAds.RightDockWidgetArea
            else:
                area = QtAds.BottomDockWidgetArea
        else:
            area = pos

        self.dock_manager.addDockWidget(area, dock)
        d_area = dock.dockAreaWidget()
        if d_area is not None and not getattr(d_area, "_spikespy_current_changed_connected", False):
            d_area.currentChanged.connect(lambda idx, a=d_area: self._on_subview_area_changed(a))
            d_area._spikespy_current_changed_connected = True

        # Synchronize child to this container's channel/unit settings immediately
        self._sync_child(child)

        # Synchronize child to container visual settings when enabled
        if (
            self.chk_sync_visuals.isChecked()
            and self.view_state is not None
            and hasattr(self.view_state, "get_visual_settings")
            and hasattr(child, "apply_shared_visual_settings")
        ):
            vis = self.view_state.get_visual_settings() or {}
            non_none_vis = {k: v for k, v in vis.items() if v is not None}
            if non_none_vis:
                child.apply_shared_visual_settings(non_none_vis)

        if self.state and getattr(self.state, "hide_view_top_bars", False):
            from .helpers import apply_top_bars_visibility
            apply_top_bars_visibility(child, True)

        notify_view_selected(child)
        self.viewAdded.emit(child)

    def _on_dock_area_created(self, area):
        if area is not None and not getattr(area, "_spikespy_current_changed_connected", False):
            area.currentChanged.connect(lambda idx, a=area: self._on_subview_area_changed(a))
            area._spikespy_current_changed_connected = True

    def _on_focused_dock_widget_changed(self, old, new):
        if new and hasattr(new, "widget") and new.widget():
            notify_view_selected(new.widget())

    def _on_subview_area_changed(self, area):
        if area and hasattr(area, "currentDockWidget") and area.currentDockWidget():
            child = area.currentDockWidget().widget()
            notify_view_selected(child)

    def _on_dock_closed(self, child: QWidget):
        if child in self.sub_views:
            self.remove_view(child)

    def remove_view(self, child: QWidget):
        """Removes a child view from this container."""
        if child not in self.sub_views:
            return

        for c_attr in ("view", "canvas", "figure_canvas"):
            cv = getattr(child, c_attr, None)
            if cv is not None:
                try:
                    cv._draw_pending = False
                    PySide6.QtCore.QCoreApplication.removePostedEvents(cv)
                except Exception:
                    pass

        self.sub_views.remove(child)
        if child in self._cards:
            self._cards.remove(child)

        dock = self._dock_map.pop(child, None)
        if dock is not None:
            try:
                self.dock_manager.removeDockWidget(dock)
                dock.closeDockWidget()
                dock.deleteLater()
            except Exception:
                pass

        self.viewRemoved.emit(child)

    def closeEvent(self, event):
        for child in list(self.sub_views):
            for c_attr in ("view", "canvas", "figure_canvas"):
                cv = getattr(child, c_attr, None)
                if cv is not None:
                    try:
                        cv._draw_pending = False
                        PySide6.QtCore.QCoreApplication.removePostedEvents(cv)
                    except Exception:
                        pass
        super().closeEvent(event)

    def _sync_all_children(self):
        """Propagates channel / unit binding changes to all embedded children."""
        if getattr(self, "_is_syncing_children", False):
            return
        self._is_syncing_children = True
        try:
            for child in self.sub_views:
                self._sync_child(child)
        finally:
            self._is_syncing_children = False

    def _sync_child(self, child: QWidget):
        """Synchronizes an individual child view's channel / unit binding to this container."""
        if hasattr(child, "channel_bar") and child.channel_bar is not None:
            cbar: ChannelGroupBar = child.channel_bar

            # Sync unit binding
            bound_unit = self.channel_bar.get_bound_unit()
            cbar.bind_to_unit(bound_unit)

            # Sync live mode, shared set, or channel selection
            if self.is_following_live():
                cbar.set_channel_mode_live()
            else:
                shared_set = self.channel_bar.get_bound_shared_set()
                if shared_set:
                    cbar.bind_to_shared_set(shared_set)
                else:
                    sel_mode = self.channel_bar.combo_channel.currentData()
                    if isinstance(sel_mode, (tuple, list)) and len(sel_mode) == 2:
                        mode, val = sel_mode
                        if mode == ChannelGroupBar.MODE_FOLLOW_UNIT:
                            cbar.set_channel_mode_unit()
                        elif mode == ChannelGroupBar.MODE_FOLLOW_GLOBAL:
                            idx = cbar.find_channel_data_index(ChannelGroupBar.MODE_FOLLOW_GLOBAL)
                            if idx >= 0:
                                cbar.combo_channel.setCurrentIndex(idx)
                        elif mode == ChannelGroupBar.MODE_FOLLOW_LIVE:
                            cbar.set_channel_mode_live()
                        elif mode in (ChannelGroupBar.MODE_GROUP, ChannelGroupBar.MODE_SINGLE):
                            cbar.bind_to_channels(self.channel_bar.get_selected_channels())

            # Sync multichannel checkbox
            cbar.chk_multichannel.setChecked(self.channel_bar.is_multichannel())

        elif isinstance(child, ChannelGroupContainerView):
            # Propagate to nested container
            child.channel_bar.bind_to_unit(self.channel_bar.get_bound_unit())
            if self.is_following_live():
                child.set_follow_live(True)
            else:
                child.set_follow_live(False)
                child.channel_bar.bind_to_channels(self.channel_bar.get_selected_channels())
            child.channel_bar.chk_multichannel.setChecked(self.channel_bar.is_multichannel())

    def broadcast_visual_settings(self, **kwargs):
        """Update container visual settings and apply them to all child views."""
        clean_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        if not clean_kwargs:
            return
        if self.view_state is not None and hasattr(self.view_state, "update_visual_settings"):
            self.view_state.update_visual_settings(**clean_kwargs)
        for child in self.sub_views:
            if hasattr(child, "apply_shared_visual_settings"):
                child.apply_shared_visual_settings(clean_kwargs)

    def get_child_views(self) -> List[QWidget]:
        return list(self.sub_views)

    def get_settings(self) -> Dict[str, Any]:
        """Returns serializable settings for perspective save/load."""
        from .SpikeSpy import window_options

        class_to_str = {v: k for k, v in window_options.items()}
        children_settings = []
        for child in self.sub_views:
            c_cls = class_to_str.get(type(child), child.__class__.__name__)
            c_sett = child.get_settings() if hasattr(child, "get_settings") else {}
            children_settings.append({"class": c_cls, "settings": c_sett})

        return {
            "channel_bar": self.channel_bar.get_settings(),
            "orientation": self._current_orientation,
            "layout": self._current_layout,
            "follow_live": self.is_following_live(),
            "follow_live_events": self.is_following_live_events(),
            "sync_visuals": self.chk_sync_visuals.isChecked(),
            "children": children_settings,
        }

    def set_settings(self, values: Dict[str, Any]):
        """Restores settings from serialized configuration."""
        if "channel_bar" in values:
            self.channel_bar.set_settings(values["channel_bar"])
        if "orientation" in values:
            self.arrange_layout(values["orientation"])
        if "layout" in values:
            self.arrange_layout(values["layout"])
        if "follow_live" in values:
            self.set_follow_live(bool(values["follow_live"]))
        if "follow_live_events" in values:
            self.set_follow_live_events(bool(values["follow_live_events"]))
        if "sync_visuals" in values:
            self.chk_sync_visuals.setChecked(bool(values["sync_visuals"]))
        if "children" in values:
            for c_info in values["children"]:
                child = self.add_view_by_key(c_info.get("class", ""))
                if child and "settings" in c_info and hasattr(child, "set_settings"):
                    try:
                        child.set_settings(c_info["settings"])
                    except Exception as e:
                        logging.warning(f"Error restoring sub-view settings: {e}")
