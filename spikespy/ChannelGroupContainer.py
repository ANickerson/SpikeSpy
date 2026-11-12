import logging
from typing import Any, Dict, List, Optional, Union

import PySide6
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import (
    QApplication,
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

from .ChannelGroupBar import ChannelGroupBar
from .ViewerState import ViewerState


class SubViewCard(QFrame):
    """
    Sub-window frame within a ChannelGroupContainer.
    Wraps a child view with a compact header bar (title and close button).
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


class ChannelGroupContainerView(QMainWindow):
    """
    Sub-window container representing a Channel Group within SpikeSpy.
    Can host multiple nested sub-windows (e.g. MultiTrace, SingleTrace, UnitView).
    
    Frames placed inside this container defer their channel / unit selection
    to this container's ChannelGroupBar, hiding their own individual channel selectors.
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
        self.container_name = name
        self.sub_views: List[QWidget] = []
        self._cards: List[SubViewCard] = []

        # 1. Top Channel Group Bar
        self.channel_bar = ChannelGroupBar(parent=self, state=state)
        self.channel_bar.bindingChanged.connect(self._sync_all_children)
        self.addToolBar(Qt.TopToolBarArea, self.channel_bar)

        # 2. Controls Toolbar for adding child views & arranging layout
        self.control_bar = QToolBar("Sub-Views", self)
        self.btn_add_subview = QToolButton(self)
        self.btn_add_subview.setText("+ Add Sub-View ▾")
        self.btn_add_subview.setPopupMode(QToolButton.InstantPopup)
        self._build_add_menu()
        self.control_bar.addWidget(self.btn_add_subview)

        # Split orientation toggle button
        self.btn_orient = QPushButton("⬍ Horizontal / Vertical", self)
        self.btn_orient.setToolTip("Toggle splitter layout orientation")
        self.btn_orient.clicked.connect(self._toggle_orientation)
        self.control_bar.addWidget(self.btn_orient)

        self.addToolBar(Qt.TopToolBarArea, self.control_bar)

        # 3. Central splitter for hosting nested views
        self.splitter = QSplitter(Qt.Vertical, self)
        self.splitter.setChildrenCollapsible(False)
        self.setCentralWidget(self.splitter)

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
        m_units.addAction("UnitFootprintView", lambda: self.add_view_by_key("UnitFootprintView"))

        # Nested Channel Group
        menu.addSeparator()
        menu.addAction("Nested Channel Group Container", lambda: self.add_view_by_key("ChannelGroupContainer"))

        self.btn_add_subview.setMenu(menu)

    def _toggle_orientation(self):
        if self.splitter.orientation() == Qt.Vertical:
            self.splitter.setOrientation(Qt.Horizontal)
        else:
            self.splitter.setOrientation(Qt.Vertical)

    def set_state(self, state: ViewerState):
        self.state = state
        self.channel_bar.setState(state)
        for v in self.sub_views:
            if hasattr(v, "set_state"):
                v.set_state(state)
            elif hasattr(v, "setState"):
                v.setState(state)

    def add_view_by_key(self, key: str) -> Optional[QWidget]:
        """Creates a child view by string key and adds it into this container."""
        from .SpikeSpy import window_options

        # Support alias mapping for both standard and PG views
        alias_map = {
            "MultiTracePG": "MultiTrace (PyQtGraph)",
            "SingleTracePG": "SingleTrace (PyQtGraph)",
            "MultiTraceFixedPG": "MultiTraceFixed (PyQtGraph)",
            "UnitPG": "UnitView (PyQtGraph)",
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
                    elif "Unit" in key:
                        from .UnitPGView import UnitPGView as cls
                except Exception as e:
                    logging.warning(f"Could not import PG view {key}: {e}")
            elif key == "ChannelGroupContainer":
                cls = ChannelGroupContainerView

        if cls is None:
            logging.error(f"Unknown sub-view type: {key}")
            return None

        child = cls(parent=self, state=self.state)
        self.add_view(child, title=key)
        return child

    def add_view(self, child: QWidget, title: Optional[str] = None):
        """
        Embeds a child view into this container.
        Hides the child's own channel selector toolbar and binds it
        to this container's ChannelGroupBar.
        """
        if title is None:
            title = child.__class__.__name__

        # Defer channel selector: hide child's own channel bar if present
        if hasattr(child, "channel_bar") and child.channel_bar is not None:
            child.channel_bar.setVisible(False)

        # Wrap in a card with close button
        card = SubViewCard(title=title, child_widget=child, parent=self.splitter)
        card.closeRequested.connect(self._on_card_close_requested)

        self.sub_views.append(child)
        self._cards.append(card)
        self.splitter.addWidget(card)

        # Synchronize child to this container's channel/unit settings immediately
        self._sync_child(child)

        self.viewAdded.emit(child)

    def remove_view(self, child: QWidget):
        """Removes a child view from this container."""
        if child not in self.sub_views:
            return
        idx = self.sub_views.index(child)
        card = self._cards[idx]

        for c_attr in ("view", "canvas", "figure_canvas"):
            cv = getattr(child, c_attr, None)
            if cv is not None:
                try:
                    cv._draw_pending = False
                    PySide6.QtCore.QCoreApplication.removePostedEvents(cv)
                except Exception:
                    pass

        self.sub_views.remove(child)
        self._cards.remove(card)

        card.setParent(None)
        card.deleteLater()
        self.viewRemoved.emit(child)

    def closeEvent(self, event):
        for child in self.sub_views:
            for c_attr in ("view", "canvas", "figure_canvas"):
                cv = getattr(child, c_attr, None)
                if cv is not None:
                    try:
                        cv._draw_pending = False
                        PySide6.QtCore.QCoreApplication.removePostedEvents(cv)
                    except Exception:
                        pass
        super().closeEvent(event)

    def _on_card_close_requested(self, card: SubViewCard):
        child = card.child_widget
        self.remove_view(child)

    def _sync_all_children(self):
        """Propagates channel / unit binding changes to all embedded children."""
        for child in self.sub_views:
            self._sync_child(child)

    def _sync_child(self, child: QWidget):
        """Synchronizes an individual child view's channel / unit binding to this container."""
        if hasattr(child, "channel_bar") and child.channel_bar is not None:
            cbar: ChannelGroupBar = child.channel_bar

            # Sync unit binding
            bound_unit = self.channel_bar.get_bound_unit()
            cbar.bind_to_unit(bound_unit)

            # Sync shared set or channel selection
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
                    elif mode in (ChannelGroupBar.MODE_GROUP, ChannelGroupBar.MODE_SINGLE):
                        cbar.bind_to_channels(self.channel_bar.get_selected_channels())

            # Sync multichannel checkbox
            cbar.chk_multichannel.setChecked(self.channel_bar.is_multichannel())

        elif isinstance(child, ChannelGroupContainerView):
            # Propagate to nested container
            child.channel_bar.bind_to_unit(self.channel_bar.get_bound_unit())
            child.channel_bar.bind_to_channels(self.channel_bar.get_selected_channels())
            child.channel_bar.chk_multichannel.setChecked(self.channel_bar.is_multichannel())

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
            "orientation": "horizontal" if self.splitter.orientation() == Qt.Horizontal else "vertical",
            "children": children_settings,
        }

    def set_settings(self, values: Dict[str, Any]):
        """Restores settings from serialized configuration."""
        if "channel_bar" in values:
            self.channel_bar.set_settings(values["channel_bar"])
        if "orientation" in values:
            orient = Qt.Horizontal if values["orientation"] == "horizontal" else Qt.Vertical
            self.splitter.setOrientation(orient)
        if "children" in values:
            for c_info in values["children"]:
                child = self.add_view_by_key(c_info.get("class", ""))
                if child and "settings" in c_info and hasattr(child, "set_settings"):
                    try:
                        child.set_settings(c_info["settings"])
                    except Exception as e:
                        logging.warning(f"Error restoring sub-view settings: {e}")
