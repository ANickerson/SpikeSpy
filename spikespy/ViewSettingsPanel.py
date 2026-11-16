import logging
from typing import Any, Dict, Optional, Union

import PySide6
from PySide6.QtCore import QSignalBlocker, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class ViewSettingsPanel(QWidget):
    """
    Settings side panel for individual view settings that follows the focused / selected window
    rather than having a cluttered toolbar on top of each view.
    """

    settingsChanged = Signal()

    def __init__(self, parent: Optional[QWidget] = None, state=None):
        super().__init__(parent)
        self.state = state
        self.active_view: Optional[QWidget] = None
        self._block_updates = False

        self.setWindowTitle("View Settings")
        self.setMinimumWidth(240)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(6)

        # 1. Header with Selected Window Name
        header_frame = QFrame(self)
        header_frame.setFrameShape(QFrame.StyledPanel)
        header_frame.setStyleSheet(
            "background: #eef2f7; border-radius: 4px; padding: 4px;"
        )
        h_layout = QVBoxLayout(header_frame)
        h_layout.setContentsMargins(4, 4, 4, 4)

        self.lbl_view_title = QLabel("<b>No View Selected</b>", header_frame)
        self.lbl_view_title.setWordWrap(True)
        h_layout.addWidget(self.lbl_view_title)

        self.lbl_view_sub = QLabel(
            "Click any view window to edit its settings.", header_frame
        )
        self.lbl_view_sub.setStyleSheet("color: #666; font-size: 11px;")
        h_layout.addWidget(self.lbl_view_sub)

        main_layout.addWidget(header_frame)

        # 2. Scroll area containing dynamic settings editors
        self.scroll = QScrollArea(self)
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)

        self.container_widget = QWidget()
        self.container_layout = QVBoxLayout(self.container_widget)
        self.container_layout.setContentsMargins(0, 0, 0, 0)
        self.container_layout.setSpacing(8)

        # Empty placeholder widget
        self.placeholder = QLabel("Select a view to display its configuration options.")
        self.placeholder.setAlignment(Qt.AlignCenter)
        self.placeholder.setStyleSheet("color: #888; padding: 20px;")
        self.container_layout.addWidget(self.placeholder)

        # MultiTrace settings group
        self.grp_multitrace = self._create_multitrace_group()
        self.container_layout.addWidget(self.grp_multitrace)
        self.grp_multitrace.setVisible(False)

        # SingleTrace settings group
        self.grp_singletrace = self._create_singletrace_group()
        self.container_layout.addWidget(self.grp_singletrace)
        self.grp_singletrace.setVisible(False)

        # MultiTraceFixed settings group
        self.grp_fixed = self._create_fixed_group()
        self.container_layout.addWidget(self.grp_fixed)
        self.grp_fixed.setVisible(False)

        # Unit analysis settings group
        self.grp_unit = self._create_unit_group()
        self.container_layout.addWidget(self.grp_unit)
        self.grp_unit.setVisible(False)

        # Plot Navigation & Limits group
        self.grp_navigation = self._create_navigation_group()
        self.container_layout.addWidget(self.grp_navigation)
        self.grp_navigation.setVisible(False)

        self.container_layout.addStretch()
        self.scroll.setWidget(self.container_widget)
        main_layout.addWidget(self.scroll, 1)

    # ---------------------------------------------------------
    # Editor Creation
    # ---------------------------------------------------------
    def _create_multitrace_group(self) -> QGroupBox:
        grp = QGroupBox("MultiTrace Configuration")
        layout = QVBoxLayout(grp)

        # Display Mode
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Mode:"))
        self.combo_mt_mode = QComboBox()
        self.combo_mt_mode.addItems(["Heatmap", "Lines / Traces", "Unit Only"])
        self.combo_mt_mode.currentIndexChanged.connect(self._on_mt_mode_changed)
        mode_layout.addWidget(self.combo_mt_mode)
        layout.addLayout(mode_layout)

        # Min / Max Color scale multipliers/percentiles
        scale_form = QFormLayout()
        self.spin_mt_lower = QDoubleSpinBox()
        self.spin_mt_lower.setRange(0.0, 100.0)
        self.spin_mt_lower.setValue(0.5)
        self.spin_mt_lower.setSingleStep(0.1)
        self.spin_mt_lower.valueChanged.connect(self._on_mt_percentiles_changed)
        scale_form.addRow("Min Scale:", self.spin_mt_lower)

        self.spin_mt_upper = QDoubleSpinBox()
        self.spin_mt_upper.setRange(0.0, 100.0)
        self.spin_mt_upper.setValue(2.0)
        self.spin_mt_upper.setSingleStep(0.1)
        self.spin_mt_upper.valueChanged.connect(self._on_mt_percentiles_changed)
        scale_form.addRow("Max Scale:", self.spin_mt_upper)
        layout.addLayout(scale_form)

        # Stimulus and Unit checkboxes
        self.chk_mt_lock_stim = QCheckBox("Lock to Stimulus Sweep")
        self.chk_mt_lock_stim.toggled.connect(self._on_mt_lock_stim_toggled)
        layout.addWidget(self.chk_mt_lock_stim)

        self.chk_mt_inc_unit = QCheckBox("Show Active Unit Overlay")
        self.chk_mt_inc_unit.toggled.connect(self._on_mt_inc_unit_toggled)
        layout.addWidget(self.chk_mt_inc_unit)

        self.chk_mt_inc_all = QCheckBox("Show All Units Overlay")
        self.chk_mt_inc_all.toggled.connect(self._on_mt_inc_all_toggled)
        layout.addWidget(self.chk_mt_inc_all)

        # Tools
        btn_layout = QHBoxLayout()
        self.btn_mt_poly = QPushButton("PolySelect")
        self.btn_mt_poly.clicked.connect(self._on_mt_poly_clicked)
        btn_layout.addWidget(self.btn_mt_poly)

        self.btn_mt_del = QPushButton("Delete Region")
        self.btn_mt_del.clicked.connect(self._on_mt_del_clicked)
        btn_layout.addWidget(self.btn_mt_del)
        layout.addLayout(btn_layout)

        return grp

    def _create_singletrace_group(self) -> QGroupBox:
        grp = QGroupBox("SingleTrace Configuration")
        layout = QVBoxLayout(grp)

        self.chk_st_multi = QCheckBox("Multi-Channel Overlay")
        self.chk_st_multi.toggled.connect(self._on_st_multi_toggled)
        layout.addWidget(self.chk_st_multi)

        self.chk_st_conv = QCheckBox("Show Conv Peaks")
        self.chk_st_conv.toggled.connect(self._on_st_conv_toggled)
        layout.addWidget(self.chk_st_conv)

        form = QFormLayout()
        self.spin_st_spacing = QDoubleSpinBox()
        self.spin_st_spacing.setRange(0.0, 1000.0)
        self.spin_st_spacing.setValue(1.0)
        self.spin_st_spacing.setSingleStep(0.2)
        self.spin_st_spacing.valueChanged.connect(self._on_st_spacing_changed)
        form.addRow("Channel Spacing:", self.spin_st_spacing)

        self.spin_st_maxima = QSpinBox()
        self.spin_st_maxima.setRange(1, 500)
        self.spin_st_maxima.setValue(1)
        self.spin_st_maxima.valueChanged.connect(self._on_st_maxima_changed)
        form.addRow("Maxima Width:", self.spin_st_maxima)
        layout.addLayout(form)

        return grp

    def _create_fixed_group(self) -> QGroupBox:
        grp = QGroupBox("MultiTraceFixed Configuration")
        layout = QFormLayout(grp)

        self.spin_fix_scale = QDoubleSpinBox()
        self.spin_fix_scale.setRange(0.001, 100.0)
        self.spin_fix_scale.setValue(1.0)
        self.spin_fix_scale.setSingleStep(0.1)
        self.spin_fix_scale.valueChanged.connect(self._on_fix_scale_changed)
        layout.addRow("Trace Amplitude Scale:", self.spin_fix_scale)

        self.spin_fix_pre = QSpinBox()
        self.spin_fix_pre.setRange(0, 100)
        self.spin_fix_pre.setValue(5)
        self.spin_fix_pre.valueChanged.connect(self._on_fix_sweeps_changed)
        layout.addRow("Pre-Sweeps:", self.spin_fix_pre)

        self.spin_fix_post = QSpinBox()
        self.spin_fix_post.setRange(0, 100)
        self.spin_fix_post.setValue(5)
        self.spin_fix_post.valueChanged.connect(self._on_fix_sweeps_changed)
        layout.addRow("Post-Sweeps:", self.spin_fix_post)

        return grp

    def _create_unit_group(self) -> QGroupBox:
        grp = QGroupBox("Unit Analysis Configuration")
        layout = QVBoxLayout(grp)

        self.chk_unit_neighbors = QCheckBox("Show Neighbor Channels")
        self.chk_unit_neighbors.toggled.connect(self._on_unit_neighbors_toggled)
        layout.addWidget(self.chk_unit_neighbors)

        form = QFormLayout()
        self.spin_unit_window = QDoubleSpinBox()
        self.spin_unit_window.setRange(0.5, 500.0)
        self.spin_unit_window.setValue(4.0)
        self.spin_unit_window.setSuffix(" ms")
        self.spin_unit_window.valueChanged.connect(self._on_unit_window_changed)
        form.addRow("Window Width:", self.spin_unit_window)

        self.spin_unit_last_n = QSpinBox()
        self.spin_unit_last_n.setRange(1, 500)
        self.spin_unit_last_n.setValue(20)
        self.spin_unit_last_n.valueChanged.connect(self._on_unit_last_n_changed)
        form.addRow("Last N Events:", self.spin_unit_last_n)
        layout.addLayout(form)

        return grp

    def _create_navigation_group(self) -> QGroupBox:
        grp = QGroupBox("Plot Navigation & Limits")
        layout = QVBoxLayout(grp)

        # Mode row: Normal, Pan, Zoom, Reset
        mode_box = QHBoxLayout()
        mode_box.addWidget(QLabel("Mode:"))

        self.btn_nav_normal = QPushButton("Normal")
        self.btn_nav_normal.setCheckable(True)
        self.btn_nav_normal.setChecked(True)
        self.btn_nav_normal.setToolTip("Normal selection / annotation mode")

        self.btn_nav_pan = QPushButton("Pan")
        self.btn_nav_pan.setCheckable(True)
        self.btn_nav_pan.setToolTip(
            "Pan with left-drag, zoom with right-drag / scroll wheel"
        )

        self.btn_nav_zoom = QPushButton("Zoom")
        self.btn_nav_zoom.setCheckable(True)
        self.btn_nav_zoom.setToolTip("Zoom to rectangle (drag with left mouse)")

        self.btn_nav_reset = QPushButton("Reset")
        self.btn_nav_reset.setToolTip("Reset view range to default limits")

        self.nav_button_group = QButtonGroup(self)
        self.nav_button_group.addButton(self.btn_nav_normal)
        self.nav_button_group.addButton(self.btn_nav_pan)
        self.nav_button_group.addButton(self.btn_nav_zoom)
        self.nav_button_group.setExclusive(True)

        mode_box.addWidget(self.btn_nav_normal)
        mode_box.addWidget(self.btn_nav_pan)
        mode_box.addWidget(self.btn_nav_zoom)
        mode_box.addWidget(self.btn_nav_reset)
        layout.addLayout(mode_box)

        self.btn_nav_normal.clicked.connect(lambda: self._on_nav_mode_btn_clicked(""))
        self.btn_nav_pan.clicked.connect(lambda: self._on_nav_mode_btn_clicked("pan"))
        self.btn_nav_zoom.clicked.connect(lambda: self._on_nav_mode_btn_clicked("zoom"))
        self.btn_nav_reset.clicked.connect(self._on_nav_reset_clicked)

        # Manual Limits Form
        form = QFormLayout()

        # X Range
        x_layout = QHBoxLayout()
        self.spin_xlim_min = QDoubleSpinBox()
        self.spin_xlim_min.setRange(-1e9, 1e9)
        self.spin_xlim_min.setDecimals(4)
        self.spin_xlim_min.setSingleStep(1.0)
        self.spin_xlim_min.setKeyboardTracking(False)

        self.spin_xlim_max = QDoubleSpinBox()
        self.spin_xlim_max.setRange(-1e9, 1e9)
        self.spin_xlim_max.setDecimals(4)
        self.spin_xlim_max.setSingleStep(1.0)
        self.spin_xlim_max.setKeyboardTracking(False)

        x_layout.addWidget(self.spin_xlim_min)
        x_layout.addWidget(QLabel("to"))
        x_layout.addWidget(self.spin_xlim_max)
        form.addRow("X Limits:", x_layout)

        # Y Range
        y_layout = QHBoxLayout()
        self.spin_ylim_min = QDoubleSpinBox()
        self.spin_ylim_min.setRange(-1e9, 1e9)
        self.spin_ylim_min.setDecimals(4)
        self.spin_ylim_min.setSingleStep(1.0)
        self.spin_ylim_min.setKeyboardTracking(False)

        self.spin_ylim_max = QDoubleSpinBox()
        self.spin_ylim_max.setRange(-1e9, 1e9)
        self.spin_ylim_max.setDecimals(4)
        self.spin_ylim_max.setSingleStep(1.0)
        self.spin_ylim_max.setKeyboardTracking(False)

        y_layout.addWidget(self.spin_ylim_min)
        y_layout.addWidget(QLabel("to"))
        y_layout.addWidget(self.spin_ylim_max)
        form.addRow("Y Limits:", y_layout)

        layout.addLayout(form)

        # Action Buttons
        btn_action_layout = QHBoxLayout()
        self.btn_apply_limits = QPushButton("Apply Limits")
        self.btn_apply_limits.clicked.connect(self._on_apply_limits_clicked)
        btn_action_layout.addWidget(self.btn_apply_limits)

        self.btn_auto_limits = QPushButton("Auto Range")
        self.btn_auto_limits.clicked.connect(self._on_auto_limits_clicked)
        btn_action_layout.addWidget(self.btn_auto_limits)
        layout.addLayout(btn_action_layout)

        self.spin_xlim_min.editingFinished.connect(self._on_apply_limits_clicked)
        self.spin_xlim_max.editingFinished.connect(self._on_apply_limits_clicked)
        self.spin_ylim_min.editingFinished.connect(self._on_apply_limits_clicked)
        self.spin_ylim_max.editingFinished.connect(self._on_apply_limits_clicked)

        return grp

    # ---------------------------------------------------------
    # Active View Tracking & Population
    # ---------------------------------------------------------
    def _is_view_alive(self, w: Optional[QWidget]) -> bool:
        if w is None:
            return False
        try:
            _ = w.objectName()
            return True
        except RuntimeError:
            return False

    def _is_configurable_view(self, w: Optional[QWidget]) -> bool:
        if not self._is_view_alive(w):
            return False
        c_name = w.__class__.__name__
        if any(
            k in c_name
            for k in (
                "MultiTraceView",
                "MultiTracePGView",
                "SingleTraceView",
                "SingleTracePGView",
                "MultiTraceFixed",
                "UnitView",
                "UnitFootprint",
                "UnitPG",
            )
        ):
            return True
        return hasattr(w, "populate_settings_panel") and callable(
            getattr(w, "populate_settings_panel")
        )

    def _disconnect_current_view(self):
        """Cleanly disconnect any signals from the previously active view."""
        self._disconnect_navigation_signals()
        self.active_view = None

    def _on_focus_changed(self, old: Optional[QWidget], new: Optional[QWidget]):
        if new is not None:
            self.setActiveView(new)

    def closeEvent(self, event):
        if getattr(self, "_focus_connected", False):
            app = QApplication.instance()
            if app is not None:
                try:
                    app.focusChanged.disconnect(self._on_focus_changed)
                except (TypeError, RuntimeError):
                    pass
            self._focus_connected = False
        self._disconnect_current_view()
        super().closeEvent(event)

    def _resolve_view(self, w: Optional[QWidget]) -> Optional[QWidget]:
        if w is None:
            return None
        # Check if w is part of ViewSettingsPanel itself
        curr = w
        while curr is not None:
            if curr is self:
                return (
                    self.active_view
                )  # User is interacting with ViewSettingsPanel; do not switch view!
            curr = curr.parentWidget() if hasattr(curr, "parentWidget") else None

        # Now search up for a view widget
        curr = w
        found_view = None
        while curr is not None:
            c_name = curr.__class__.__name__
            if any(
                k in c_name
                for k in (
                    "MultiTrace",
                    "SingleTrace",
                    "UnitView",
                    "UnitFootprint",
                    "UnitPG",
                    "ChannelGroupContainer",
                )
            ) or hasattr(curr, "populate_settings_panel"):
                found_view = curr
                break
            # Check CDockWidget or CDockWidgetTab unwrapping
            if (
                c_name == "CDockWidget"
                and hasattr(curr, "widget")
                and callable(getattr(curr, "widget"))
            ):
                try:
                    inner = curr.widget()
                    if inner is not None and (
                        self._is_configurable_view(inner)
                        or inner.__class__.__name__ == "ChannelGroupContainerView"
                    ):
                        found_view = inner
                        break
                except TypeError:
                    pass
            elif (
                c_name == "CDockWidgetTab"
                and hasattr(curr, "dockWidget")
                and callable(getattr(curr, "dockWidget"))
            ):
                try:
                    dw = curr.dockWidget()
                    inner = dw.widget() if dw is not None else None
                    if inner is not None and (
                        self._is_configurable_view(inner)
                        or inner.__class__.__name__ == "ChannelGroupContainerView"
                    ):
                        found_view = inner
                        break
                except TypeError:
                    pass
            curr = curr.parentWidget() if hasattr(curr, "parentWidget") else None

        if found_view is not None:
            # If it's a ChannelGroupContainerView, see if w was inside one of its sub_views
            if hasattr(found_view, "sub_views") and len(found_view.sub_views) > 0:
                for sv in found_view.sub_views:
                    temp = w
                    while temp is not None and temp is not found_view:
                        if temp is sv:
                            return sv
                        temp = (
                            temp.parentWidget()
                            if hasattr(temp, "parentWidget")
                            else None
                        )
                return found_view.sub_views[0]
            if self._is_configurable_view(found_view):
                return found_view
            return None
        return None

    def setActiveView(self, view: Optional[QWidget]):
        """Sets the active view to display/edit in this settings panel.
        Retains the latest valid configurable visualizer when non-configurable panels are clicked.
        """
        if view is None:
            self._disconnect_current_view()
            self.lbl_view_title.setText("<b>No View Selected</b>")
            self.lbl_view_sub.setText("Click any view window to edit its settings.")
            self.grp_multitrace.setVisible(False)
            self.grp_singletrace.setVisible(False)
            self.grp_fixed.setVisible(False)
            self.grp_unit.setVisible(False)
            self.grp_navigation.setVisible(False)
            if (
                hasattr(self, "_custom_plugin_grp")
                and self._custom_plugin_grp is not None
            ):
                self._custom_plugin_grp.setVisible(False)
                self._custom_plugin_grp.deleteLater()
                self._custom_plugin_grp = None
            self.placeholder.setVisible(True)
            self._block_updates = False
            return

        resolved = self._resolve_view(view)
        if not self._is_configurable_view(resolved):
            if self._is_configurable_view(self.active_view):
                return
            # Look through parent's dock_manager if available
            parent_widget = self.parent()
            found_fallback = None
            if (
                parent_widget is not None
                and hasattr(parent_widget, "dock_manager")
                and parent_widget.dock_manager is not None
            ):
                for dw in parent_widget.dock_manager.dockWidgets():
                    if hasattr(dw, "widget") and callable(getattr(dw, "widget")):
                        candidate = self._resolve_view(dw.widget())
                        if self._is_configurable_view(candidate):
                            found_fallback = candidate
                            break
            if found_fallback is not None:
                resolved = found_fallback
            else:
                return

        if self.active_view == resolved and self._is_view_alive(self.active_view):
            return

        self.active_view = resolved
        view = resolved
        self._block_updates = True

        # Hide all groups first
        self.placeholder.setVisible(False)
        self.grp_multitrace.setVisible(False)
        self.grp_singletrace.setVisible(False)
        self.grp_fixed.setVisible(False)
        self.grp_unit.setVisible(False)
        self.grp_navigation.setVisible(False)
        self._disconnect_navigation_signals()
        if hasattr(self, "_custom_plugin_grp") and self._custom_plugin_grp is not None:
            self._custom_plugin_grp.setVisible(False)
            self._custom_plugin_grp.deleteLater()
            self._custom_plugin_grp = None

        if view is None:
            self.lbl_view_title.setText("<b>No View Selected</b>")
            self.lbl_view_sub.setText("Click any view window to edit its settings.")
            self.placeholder.setVisible(True)
            self._block_updates = False
            return

        c_name = view.__class__.__name__
        self.lbl_view_title.setText(f"<b>{c_name}</b>")
        self.lbl_view_sub.setText("Settings follow this active view.")

        # Detect MultiTrace (Matplotlib or PG)
        if "MultiTraceView" in c_name or "MultiTracePGView" in c_name:
            self.grp_multitrace.setVisible(True)
            self._populate_multitrace(view)

        # Detect SingleTrace (Matplotlib or PG)
        elif "SingleTraceView" in c_name or "SingleTracePGView" in c_name:
            self.grp_singletrace.setVisible(True)
            self._populate_singletrace(view)

        # Detect MultiTraceFixed (Matplotlib or PG)
        elif "MultiTraceFixed" in c_name:
            self.grp_fixed.setVisible(True)
            self._populate_fixed(view)

        # Detect UnitView / UnitFootprint (Matplotlib or PG)
        elif "UnitView" in c_name or "UnitFootprint" in c_name or "UnitPG" in c_name:
            self.grp_unit.setVisible(True)
            self._populate_unit(view)

        elif hasattr(view, "populate_settings_panel") and callable(
            getattr(view, "populate_settings_panel")
        ):
            self._custom_plugin_grp = QGroupBox(f"{c_name} Configuration")
            custom_form = QFormLayout(self._custom_plugin_grp)
            view.populate_settings_panel(self, custom_form)
            self.container_layout.insertWidget(
                self.container_layout.count() - 1, self._custom_plugin_grp
            )
            self._custom_plugin_grp.setVisible(True)

        else:
            self.placeholder.setText(
                f"{c_name} has no view-specific visual parameters."
            )
            self.placeholder.setVisible(True)

        # Show Navigation & Limits group if supported
        if (
            hasattr(view, "get_view_range")
            and callable(getattr(view, "get_view_range"))
        ) or hasattr(view, "ax"):
            self.grp_navigation.setVisible(True)
            self._populate_navigation(view)

        self._block_updates = False

    def _populate_multitrace(self, view):
        # Mode
        mode = getattr(view, "mode", "heatmap")
        if mode == "lines":
            self.combo_mt_mode.setCurrentIndex(1)
        elif mode == "unitonly":
            self.combo_mt_mode.setCurrentIndex(2)
        else:
            self.combo_mt_mode.setCurrentIndex(0)

        # Percentiles / Scale
        if hasattr(view, "lowerSpinBox") and hasattr(view, "upperSpinBox"):
            self.spin_mt_lower.setRange(
                view.lowerSpinBox.minimum(), view.lowerSpinBox.maximum()
            )
            self.spin_mt_upper.setRange(
                view.upperSpinBox.minimum(), view.upperSpinBox.maximum()
            )
            self.spin_mt_lower.setValue(float(view.lowerSpinBox.value()))
            self.spin_mt_upper.setValue(float(view.upperSpinBox.value()))

        # Checkboxes
        if hasattr(view, "lock_to_stimCheckBox"):
            self.chk_mt_lock_stim.setChecked(view.lock_to_stimCheckBox.isChecked())
        elif hasattr(view, "lock_to_stim"):
            self.chk_mt_lock_stim.setChecked(bool(view.lock_to_stim))

        if hasattr(view, "includeUnitCheckbox"):
            self.chk_mt_inc_unit.setChecked(view.includeUnitCheckbox.isChecked())

        if hasattr(view, "includeAllUnitsCheckBox"):
            self.chk_mt_inc_all.setChecked(view.includeAllUnitsCheckBox.isChecked())

    def _populate_singletrace(self, view):
        if hasattr(view, "channel_bar") and view.channel_bar is not None:
            self.chk_st_multi.setChecked(view.channel_bar.is_multichannel())
        elif hasattr(view, "chk_multichannel"):
            self.chk_st_multi.setChecked(view.chk_multichannel.isChecked())

        if hasattr(view, "chk_multichannel_conv"):
            self.chk_st_conv.setChecked(view.chk_multichannel_conv.isChecked())
        elif hasattr(view, "chk_conv_peaks"):
            self.chk_st_conv.setChecked(view.chk_conv_peaks.isChecked())

        if hasattr(view, "spin_chan_spacing"):
            self.spin_st_spacing.setValue(float(view.spin_chan_spacing.value()))
        elif hasattr(view, "spacing"):
            self.spin_st_spacing.setValue(float(view.spacing))

        if hasattr(view, "select_local_maxima_width"):
            self.spin_st_maxima.setValue(int(view.select_local_maxima_width))

    def _populate_fixed(self, view):
        if hasattr(view, "scale_spinner"):
            self.spin_fix_scale.setValue(float(view.scale_spinner.value()))
        elif hasattr(view, "scale"):
            self.spin_fix_scale.setValue(float(view.scale))

        if hasattr(view, "pre_spinner"):
            self.spin_fix_pre.setValue(int(view.pre_spinner.value()))
        elif hasattr(view, "pre_sweeps"):
            self.spin_fix_pre.setValue(int(view.pre_sweeps))

        if hasattr(view, "post_spinner"):
            self.spin_fix_post.setValue(int(view.post_spinner.value()))
        elif hasattr(view, "post_sweeps"):
            self.spin_fix_post.setValue(int(view.post_sweeps))

    def _populate_unit(self, view):
        if hasattr(view, "chk_show_neighbors"):
            self.chk_unit_neighbors.setChecked(view.chk_show_neighbors.isChecked())

        if hasattr(view, "spin_window_ms"):
            self.spin_unit_window.setValue(float(view.spin_window_ms.value()))
        elif hasattr(view, "window_size"):
            self.spin_unit_window.setValue(float(view.window_size))

        if hasattr(view, "spin_last_n"):
            self.spin_unit_last_n.setValue(int(view.spin_last_n.value()))

    # ---------------------------------------------------------
    # Event Handlers to update active view
    # ---------------------------------------------------------
    def _on_mt_mode_changed(self, idx: int):
        if self._block_updates or not self.active_view:
            return
        modes = ["heatmap", "lines", "unitonly"]
        m = modes[idx]
        if hasattr(self.active_view, "set_mode"):
            self.active_view.set_mode(m)
        elif hasattr(self.active_view, "mode"):
            self.active_view.mode = m
            if hasattr(self.active_view, "update_axis"):
                self.active_view.update_axis()
            elif hasattr(self.active_view, "update_plot"):
                self.active_view.update_plot()
            elif hasattr(self.active_view, "setup_figure"):
                self.active_view.setup_figure()
        self.settingsChanged.emit()

    def _on_mt_percentiles_changed(self):
        if self._block_updates or not self.active_view:
            return
        low = self.spin_mt_lower.value()
        high = self.spin_mt_upper.value()
        if hasattr(self.active_view, "lowerSpinBox"):
            self.active_view.lowerSpinBox.setValue(low)
        if hasattr(self.active_view, "upperSpinBox"):
            self.active_view.upperSpinBox.setValue(high)
        if hasattr(self.active_view, "set_percentiles"):
            self.active_view.set_percentiles(low, high)
        elif hasattr(self.active_view, "update_axis"):
            self.active_view.update_axis()
        self.settingsChanged.emit()

    def _on_mt_lock_stim_toggled(self, checked: bool):
        if self._block_updates or not self.active_view:
            return
        if hasattr(self.active_view, "lock_to_stimCheckBox"):
            self.active_view.lock_to_stimCheckBox.setChecked(checked)
        elif hasattr(self.active_view, "lock_to_stim"):
            self.active_view.lock_to_stim = checked
            if hasattr(self.active_view, "update_axis"):
                self.active_view.update_axis()
        self.settingsChanged.emit()

    def _on_mt_inc_unit_toggled(self, checked: bool):
        if self._block_updates or not self.active_view:
            return
        if hasattr(self.active_view, "includeUnitCheckbox"):
            self.active_view.includeUnitCheckbox.setChecked(checked)
        elif hasattr(self.active_view, "update_axis"):
            self.active_view.update_axis()
        self.settingsChanged.emit()

    def _on_mt_inc_all_toggled(self, checked: bool):
        if self._block_updates or not self.active_view:
            return
        if hasattr(self.active_view, "includeAllUnitsCheckBox"):
            self.active_view.includeAllUnitsCheckBox.setChecked(checked)
        elif hasattr(self.active_view, "update_axis"):
            self.active_view.update_axis()
        self.settingsChanged.emit()

    def _on_mt_poly_clicked(self):
        if not self.active_view:
            return
        if hasattr(self.active_view, "polySelectButton"):
            self.active_view.polySelectButton.click()
        elif hasattr(self.active_view, "toggle_polySelector"):
            self.active_view.toggle_polySelector()

    def _on_mt_del_clicked(self):
        if not self.active_view:
            return
        if hasattr(self.active_view, "deleterButton"):
            self.active_view.deleterButton.click()
        elif hasattr(self.active_view, "toggle_deleter"):
            self.active_view.toggle_deleter()

    def _on_st_multi_toggled(self, checked: bool):
        if self._block_updates or not self.active_view:
            return
        if (
            hasattr(self.active_view, "channel_bar")
            and self.active_view.channel_bar is not None
        ):
            self.active_view.channel_bar.chk_multichannel.setChecked(checked)
        elif hasattr(self.active_view, "chk_multichannel"):
            self.active_view.chk_multichannel.setChecked(checked)
        self.settingsChanged.emit()

    def _on_st_conv_toggled(self, checked: bool):
        if self._block_updates or not self.active_view:
            return
        if hasattr(self.active_view, "chk_multichannel_conv"):
            self.active_view.chk_multichannel_conv.setChecked(checked)
        elif hasattr(self.active_view, "chk_conv_peaks"):
            self.active_view.chk_conv_peaks.setChecked(checked)
        self.settingsChanged.emit()

    def _on_st_spacing_changed(self, val: float):
        if self._block_updates or not self.active_view:
            return
        if hasattr(self.active_view, "spin_chan_spacing"):
            self.active_view.spin_chan_spacing.setValue(int(round(val)))
        elif hasattr(self.active_view, "spin_spacing"):
            self.active_view.spin_spacing.setValue(val)
        elif hasattr(self.active_view, "spacing"):
            self.active_view.spacing = val
            if hasattr(self.active_view, "update_plot"):
                self.active_view.update_plot()
        self.settingsChanged.emit()

    def _on_st_maxima_changed(self, val: int):
        if self._block_updates or not self.active_view:
            return
        if hasattr(self.active_view, "select_local_maxima_width"):
            self.active_view.select_local_maxima_width = val
        self.settingsChanged.emit()

    def _on_fix_scale_changed(self, val: float):
        if self._block_updates or not self.active_view:
            return
        if hasattr(self.active_view, "scale_spinner"):
            self.active_view.scale_spinner.setValue(val)
        elif hasattr(self.active_view, "scale"):
            self.active_view.scale = val
            if hasattr(self.active_view, "update_plot"):
                self.active_view.update_plot()
            elif hasattr(self.active_view, "plot"):
                self.active_view.plot()
        self.settingsChanged.emit()

    def _on_fix_sweeps_changed(self):
        if self._block_updates or not self.active_view:
            return
        pre = self.spin_fix_pre.value()
        post = self.spin_fix_post.value()
        if hasattr(self.active_view, "pre_spinner"):
            self.active_view.pre_spinner.setValue(pre)
        elif hasattr(self.active_view, "pre_sweeps"):
            self.active_view.pre_sweeps = pre

        if hasattr(self.active_view, "post_spinner"):
            self.active_view.post_spinner.setValue(post)
        elif hasattr(self.active_view, "post_sweeps"):
            self.active_view.post_sweeps = post

        if hasattr(self.active_view, "update_plot"):
            self.active_view.update_plot()
        elif hasattr(self.active_view, "plot"):
            self.active_view.plot()
        self.settingsChanged.emit()

    def _on_unit_neighbors_toggled(self, checked: bool):
        if self._block_updates or not self.active_view:
            return
        if hasattr(self.active_view, "chk_show_neighbors"):
            self.active_view.chk_show_neighbors.setChecked(checked)
        self.settingsChanged.emit()

    def _on_unit_window_changed(self, val: float):
        if self._block_updates or not self.active_view:
            return
        if hasattr(self.active_view, "spin_window_ms"):
            self.active_view.spin_window_ms.setValue(val)
        elif hasattr(self.active_view, "window_size"):
            self.active_view.window_size = val
            if hasattr(self.active_view, "update_plot"):
                self.active_view.update_plot()
        self.settingsChanged.emit()

    def _on_unit_last_n_changed(self, val: int):
        if self._block_updates or not self.active_view:
            return
        if hasattr(self.active_view, "spin_last_n"):
            self.active_view.spin_last_n.setValue(val)
        self.settingsChanged.emit()

    # ---------------------------------------------------------
    # Navigation & Limits Handlers
    # ---------------------------------------------------------
    def _populate_navigation(self, view):
        self._block_updates = True
        try:
            # Mode
            mode = ""
            if hasattr(view, "get_nav_mode") and callable(
                getattr(view, "get_nav_mode")
            ):
                mode = view.get_nav_mode()
            elif (
                hasattr(view, "matplotlib_toolbar")
                and view.matplotlib_toolbar is not None
            ):
                tb_mode = getattr(view.matplotlib_toolbar, "mode", "")
                if "pan" in tb_mode.lower():
                    mode = "pan"
                elif "zoom" in tb_mode.lower():
                    mode = "zoom"

            if mode == "pan":
                self.btn_nav_pan.setChecked(True)
            elif mode == "zoom":
                self.btn_nav_zoom.setChecked(True)
            else:
                self.btn_nav_normal.setChecked(True)

            # Range
            if hasattr(view, "get_view_range") and callable(
                getattr(view, "get_view_range")
            ):
                (xmin, xmax), (ymin, ymax) = view.get_view_range()
                self.spin_xlim_min.setValue(xmin)
                self.spin_xlim_max.setValue(xmax)
                self.spin_ylim_min.setValue(ymin)
                self.spin_ylim_max.setValue(ymax)

                plot = (
                    view.get_primary_nav_plot()
                    if hasattr(view, "get_primary_nav_plot")
                    else None
                )
                if plot is not None and hasattr(plot, "vb") and plot.vb is not None:
                    self._connected_vb = plot.vb
                    plot.vb.sigRangeChanged.connect(self._on_plot_range_changed)

                if hasattr(view, "register_nav_mode_callback"):
                    self._connected_nav_view = view
                    view.register_nav_mode_callback(self._on_view_nav_mode_changed)

            elif hasattr(view, "ax"):
                xlim = view.ax.get_xlim()
                ylim = view.ax.get_ylim()
                self.spin_xlim_min.setValue(float(min(xlim)))
                self.spin_xlim_max.setValue(float(max(xlim)))
                self.spin_ylim_min.setValue(float(min(ylim)))
                self.spin_ylim_max.setValue(float(max(ylim)))
        finally:
            self._block_updates = False

    def _disconnect_navigation_signals(self):
        if hasattr(self, "_connected_vb") and self._connected_vb is not None:
            try:
                self._connected_vb.sigRangeChanged.disconnect(
                    self._on_plot_range_changed
                )
            except Exception:
                pass
            self._connected_vb = None

        if (
            hasattr(self, "_connected_nav_view")
            and self._connected_nav_view is not None
        ):
            try:
                self._connected_nav_view.unregister_nav_mode_callback(
                    self._on_view_nav_mode_changed
                )
            except Exception:
                pass
            self._connected_nav_view = None

    def _on_plot_range_changed(self, *args):
        if self._block_updates or not self._is_view_alive(self.active_view):
            return
        if (
            self.spin_xlim_min.hasFocus()
            or self.spin_xlim_max.hasFocus()
            or self.spin_ylim_min.hasFocus()
            or self.spin_ylim_max.hasFocus()
        ):
            return
        if hasattr(self.active_view, "get_view_range") and callable(
            getattr(self.active_view, "get_view_range")
        ):
            (xmin, xmax), (ymin, ymax) = self.active_view.get_view_range()
            self._block_updates = True
            try:
                self.spin_xlim_min.setValue(xmin)
                self.spin_xlim_max.setValue(xmax)
                self.spin_ylim_min.setValue(ymin)
                self.spin_ylim_max.setValue(ymax)
            finally:
                self._block_updates = False

    def _on_view_nav_mode_changed(self, mode: str):
        if self._block_updates:
            return
        self._block_updates = True
        try:
            if mode == "pan":
                self.btn_nav_pan.setChecked(True)
            elif mode == "zoom":
                self.btn_nav_zoom.setChecked(True)
            else:
                self.btn_nav_normal.setChecked(True)
        finally:
            self._block_updates = False

    def _on_nav_mode_btn_clicked(self, mode: str):
        if self._block_updates or not self._is_view_alive(self.active_view):
            return
        if hasattr(self.active_view, "set_nav_mode") and callable(
            getattr(self.active_view, "set_nav_mode")
        ):
            self.active_view.set_nav_mode(mode)
        elif (
            hasattr(self.active_view, "matplotlib_toolbar")
            and self.active_view.matplotlib_toolbar is not None
        ):
            tb = self.active_view.matplotlib_toolbar
            current_tb_mode = getattr(tb, "mode", "").lower()
            if mode == "pan" and "pan" not in current_tb_mode:
                tb.pan()
            elif mode == "zoom" and "zoom" not in current_tb_mode:
                tb.zoom()
            elif mode == "" and current_tb_mode != "":
                if "pan" in current_tb_mode:
                    tb.pan()
                elif "zoom" in current_tb_mode:
                    tb.zoom()
        self.settingsChanged.emit()

    def _on_apply_limits_clicked(self):
        if self._block_updates or not self._is_view_alive(self.active_view):
            return
        xmin = float(self.spin_xlim_min.value())
        xmax = float(self.spin_xlim_max.value())
        ymin = float(self.spin_ylim_min.value())
        ymax = float(self.spin_ylim_max.value())
        if xmin >= xmax:
            xmax = xmin + 1.0
            self.spin_xlim_max.setValue(xmax)
        if ymin >= ymax:
            ymax = ymin + 1.0
            self.spin_ylim_max.setValue(ymax)

        self._block_updates = True
        try:
            if hasattr(self.active_view, "set_view_range") and callable(
                getattr(self.active_view, "set_view_range")
            ):
                self.active_view.set_view_range((xmin, xmax), (ymin, ymax))
            elif hasattr(self.active_view, "ax"):
                self.active_view.ax.set_xlim(xmin, xmax)
                self.active_view.ax.set_ylim(ymin, ymax)
                if hasattr(self.active_view, "view") and hasattr(
                    self.active_view.view, "draw_idle"
                ):
                    self.active_view.view.draw_idle()
                elif hasattr(self.active_view, "canvas") and hasattr(
                    self.active_view.canvas, "draw_idle"
                ):
                    self.active_view.canvas.draw_idle()
        finally:
            self._block_updates = False
        self.settingsChanged.emit()

    def _on_auto_limits_clicked(self):
        self._on_nav_reset_clicked()

    def _on_nav_reset_clicked(self):
        if self._block_updates or not self._is_view_alive(self.active_view):
            return
        if hasattr(self.active_view, "reset_view") and callable(
            getattr(self.active_view, "reset_view")
        ):
            self.active_view.reset_view()
        elif hasattr(self.active_view, "matplotlib_toolbar") and hasattr(
            self.active_view.matplotlib_toolbar, "home"
        ):
            self.active_view.matplotlib_toolbar.home()

        if hasattr(self.active_view, "get_view_range") and callable(
            getattr(self.active_view, "get_view_range")
        ):
            (xmin, xmax), (ymin, ymax) = self.active_view.get_view_range()
            self._block_updates = True
            try:
                self.spin_xlim_min.setValue(xmin)
                self.spin_xlim_max.setValue(xmax)
                self.spin_ylim_min.setValue(ymin)
                self.spin_ylim_max.setValue(ymax)
            finally:
                self._block_updates = False
        elif hasattr(self.active_view, "ax"):
            xlim = self.active_view.ax.get_xlim()
            ylim = self.active_view.ax.get_ylim()
            self._block_updates = True
            try:
                self.spin_xlim_min.setValue(float(min(xlim)))
                self.spin_xlim_max.setValue(float(max(xlim)))
                self.spin_ylim_min.setValue(float(min(ylim)))
                self.spin_ylim_max.setValue(float(max(ylim)))
            finally:
                self._block_updates = False
        self.settingsChanged.emit()
