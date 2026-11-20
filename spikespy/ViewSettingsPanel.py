import logging
from typing import Any, Dict, Optional, Union

import PySide6
from PySide6.QtCore import QSignalBlocker, Qt, Signal, Slot
from PySide6.QtGui import QAction, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .ActiveUnitInfoView import ActiveUnitInfoView
from .ChannelGroupBar import ChannelGroupBar, ChannelGroupEditorPane
from .MultiTracePlotManager import MultiTracePlotManager


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
        self._active_state_connections = []
        self._current_view_connections = []
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

        self.chk_hide_top_bars = QCheckBox("Remove Top Bars from Views", header_frame)
        self.chk_hide_top_bars.setToolTip(
            "Globally remove top bars from views so settings are controlled exclusively from this properties pane."
        )
        if self.state is not None:
            self.chk_hide_top_bars.setChecked(getattr(self.state, "hide_view_top_bars", False))
        self.chk_hide_top_bars.toggled.connect(self._on_hide_top_bars_toggled)
        h_layout.addWidget(self.chk_hide_top_bars)

        main_layout.addWidget(header_frame)

        # 1b. Channel & Unit Selector for Selected Window
        self.grp_unit_selector = QGroupBox("Channel & Unit Selection", self)
        self.grp_channel_selector = self.grp_unit_selector
        grp_layout = QVBoxLayout(self.grp_unit_selector)
        grp_layout.setContentsMargins(6, 4, 6, 4)
        grp_layout.setSpacing(4)

        # Unit selection row
        row_unit = QHBoxLayout()
        row_unit.addWidget(QLabel("Unit:"))
        self.combo_unit = QComboBox(self.grp_unit_selector)
        self.combo_unit.setToolTip(
            "Select target unit group for the active window.\n"
            "When inside a channel group, changing this affects all views in the group."
        )
        self.combo_unit.currentIndexChanged.connect(self._on_unit_selection_changed)
        row_unit.addWidget(self.combo_unit, 1)
        grp_layout.addLayout(row_unit)

        # Channel / Group selection row + Settings Button
        row_chan = QHBoxLayout()
        row_chan.addWidget(QLabel("Channel:"))
        self.combo_channel = QComboBox(self.grp_unit_selector)
        self.combo_channel.setToolTip(
            "Select channel or channel group:\n"
            "- Auto (Active Channel): follow global active recording channel\n"
            "- Unit Assigned Channels: primary & neighbor channels configured for the unit\n"
            "- Unit Detected Channels: channels dynamically detected from the unit's footprint\n"
            "- Auto (Live Stream Input): track incoming live streaming channels\n"
            "- [Group] ...: use a saved channel group\n"
            "- Individual Channel: pick a specific channel"
        )
        self.combo_channel.currentIndexChanged.connect(self._on_channel_selection_changed)
        row_chan.addWidget(self.combo_channel, 1)

        self.btn_channel_settings = QToolButton(self.grp_unit_selector)
        self.btn_channel_settings.setText("⚙")
        self.btn_channel_settings.setCheckable(True)
        self.btn_channel_settings.setToolTip("Open / toggle Channel Groups editor pane")
        self.btn_channel_settings.toggled.connect(self._on_toggle_channel_groups_editor)
        row_chan.addWidget(self.btn_channel_settings)
        grp_layout.addLayout(row_chan)

        # Follow Live Events and Sync Visuals checkboxes row
        row_live = QHBoxLayout()
        self.chk_follow_events = QCheckBox("📡 Follow Live Events", self.grp_unit_selector)
        self.chk_follow_events.setToolTip(
            "Follow live new events (keep event index in sync with latest). Shortcut: Ctrl+L"
        )
        self.chk_follow_events.toggled.connect(self._on_follow_events_toggled)
        row_live.addWidget(self.chk_follow_events)

        self.chk_sync_group_visuals = QCheckBox("🔗 Sync Visuals Across Group", self.grp_unit_selector)
        self.chk_sync_group_visuals.setChecked(True)
        self.chk_sync_group_visuals.setToolTip(
            "Broadcast visual setting changes (scale mode, color scale, trace scale, X range) "
            "to sibling views in the same channel group container or shared channel set."
        )
        row_live.addWidget(self.chk_sync_group_visuals)
        grp_layout.addLayout(row_live)

        # Compatibility controls kept hidden in layout (so existing tests or callers can still use them)
        row_opts = QHBoxLayout()
        self.chk_multichannel = QCheckBox("Multi-Ch", self.grp_unit_selector)
        self.chk_multichannel.setToolTip("Display or integrate all channels in the selected group/unit")
        self.chk_multichannel.toggled.connect(self._on_multichannel_toggled)
        self.chk_multichannel.setVisible(False)
        row_opts.addWidget(self.chk_multichannel)

        self.btn_shift_prev = QPushButton("<", self.grp_unit_selector)
        self.btn_shift_prev.setFixedWidth(24)
        self.btn_shift_prev.clicked.connect(lambda: self._shift_channel_group(-1))
        self.btn_shift_prev.setVisible(False)
        row_opts.addWidget(self.btn_shift_prev)

        self.btn_shift_next = QPushButton(">", self.grp_unit_selector)
        self.btn_shift_next.setFixedWidth(24)
        self.btn_shift_next.clicked.connect(lambda: self._shift_channel_group(1))
        self.btn_shift_next.setVisible(False)
        row_opts.addWidget(self.btn_shift_next)

        self.btn_group_menu = QToolButton(self.grp_unit_selector)
        self.btn_group_menu.setText("Group ▾")
        self.btn_group_menu.setPopupMode(QToolButton.InstantPopup)
        self._build_panel_group_menu()
        self.btn_group_menu.setVisible(False)
        row_opts.addWidget(self.btn_group_menu)
        grp_layout.addLayout(row_opts)

        # Collapsible Channel Groups Editor Pane
        self.channel_groups_editor = ChannelGroupEditorPane(parent=self, state=self.state)
        self.channel_groups_editor.setVisible(False)
        self.channel_groups_editor.groupsChanged.connect(self._on_editor_groups_changed)
        grp_layout.addWidget(self.channel_groups_editor)

        main_layout.addWidget(self.grp_unit_selector)

        # Keyboard shortcut for toggling following live events
        self.shortcut_toggle_live_events = QShortcut(
            QKeySequence("Ctrl+L"), self, context=Qt.WidgetWithChildrenShortcut
        )
        self.shortcut_toggle_live_events.activated.connect(self.toggle_follow_live_events)

        if self.state is not None:
            self._connect_state_signals()
            self._refresh_unit_combo()
            self._refresh_channel_combo()
            self._sync_follow_events_checkbox()

        # 2. Scroll area containing dynamic settings editors
        self.scroll = QScrollArea(self)
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)

        self.container_widget = QWidget()
        self.container_layout = QVBoxLayout(self.container_widget)
        self.container_layout.setContentsMargins(0, 0, 0, 0)
        self.container_layout.setSpacing(8)

        # Compact Active Unit Info & Tracking section
        self.grp_active_unit_info = QGroupBox("Active Unit Info & Tracking", self)
        unit_info_layout = QVBoxLayout(self.grp_active_unit_info)
        unit_info_layout.setContentsMargins(4, 4, 4, 4)
        self.unit_info_widget = ActiveUnitInfoView(
            parent=self.grp_active_unit_info, state=self.state, compact=True
        )
        unit_info_layout.addWidget(self.unit_info_widget)
        self.container_layout.addWidget(self.grp_active_unit_info)

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
        self.combo_mt_scale_mode = QComboBox()
        self.combo_mt_scale_mode.addItems(["std", "mV"])
        self.combo_mt_scale_mode.currentTextChanged.connect(self._on_mt_scale_mode_changed)
        scale_form.addRow("Scale Mode:", self.combo_mt_scale_mode)

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

        # Per-plot management (channel plots and summary line graphs)
        self.lbl_mt_plots = QLabel("<b>Plots</b>")
        self.lbl_mt_plots.setToolTip(
            "Add, remove, reorder and configure each channel or summary plot"
        )
        layout.addWidget(self.lbl_mt_plots)
        self.lbl_mt_plots.setVisible(False)
        self.mt_plot_manager = MultiTracePlotManager(grp)
        self.mt_plot_manager.setVisible(False)
        layout.addWidget(self.mt_plot_manager)

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

        self.combo_fix_scale_mode = QComboBox()
        self.combo_fix_scale_mode.addItems(["mV", "std"])
        self.combo_fix_scale_mode.currentTextChanged.connect(self._on_fix_scale_mode_changed)
        layout.addRow("Scale Mode:", self.combo_fix_scale_mode)

        self.combo_fix_axis_mode = QComboBox()
        self.combo_fix_axis_mode.addItems(["Sweep", "Time"])
        self.combo_fix_axis_mode.currentIndexChanged.connect(
            self._on_fix_axis_mode_changed
        )
        layout.addRow("Y-Axis Mode:", self.combo_fix_axis_mode)

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

        self.spin_fix_time_window = QDoubleSpinBox()
        self.spin_fix_time_window.setRange(0.1, 3600.0)
        self.spin_fix_time_window.setValue(10.0)
        self.spin_fix_time_window.setSingleStep(1.0)
        self.spin_fix_time_window.setSuffix(" s")
        self.spin_fix_time_window.valueChanged.connect(self._on_fix_time_window_changed)
        layout.addRow("Time Window:", self.spin_fix_time_window)

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

        # Range Mode: Absolute vs Diff (Center + Width/Height)
        range_mode_layout = QHBoxLayout()
        range_mode_layout.addWidget(QLabel("Range Mode:"))
        self.combo_range_mode = QComboBox()
        self.combo_range_mode.addItems(["Absolute (min/max)", "Diff (center + width/height)"])
        self.combo_range_mode.currentIndexChanged.connect(self._on_range_mode_changed)
        range_mode_layout.addWidget(self.combo_range_mode)
        layout.addLayout(range_mode_layout)

        # Manual Limits Form
        form = QFormLayout()

        # X Range - Absolute mode
        self.x_abs_layout = QHBoxLayout()
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

        self.x_abs_layout.addWidget(self.spin_xlim_min)
        self.x_abs_layout.addWidget(QLabel("to"))
        self.x_abs_layout.addWidget(self.spin_xlim_max)

        # X Range - Diff mode
        self.x_diff_layout = QHBoxLayout()
        self.spin_x_center = QDoubleSpinBox()
        self.spin_x_center.setRange(-1e9, 1e9)
        self.spin_x_center.setDecimals(4)
        self.spin_x_center.setSingleStep(1.0)
        self.spin_x_center.setKeyboardTracking(False)
        self.spin_x_center.setToolTip("Center of X range")

        self.spin_x_width = QDoubleSpinBox()
        self.spin_x_width.setRange(0.0, 2e9)
        self.spin_x_width.setDecimals(4)
        self.spin_x_width.setSingleStep(1.0)
        self.spin_x_width.setKeyboardTracking(False)
        self.spin_x_width.setToolTip("Width of X range")

        self.x_diff_layout.addWidget(QLabel("Center:"))
        self.x_diff_layout.addWidget(self.spin_x_center)
        self.x_diff_layout.addWidget(QLabel("Width:"))
        self.x_diff_layout.addWidget(self.spin_x_width)

        form.addRow("X Range:", self.x_abs_layout)

        # Y Range - Absolute mode
        self.y_abs_layout = QHBoxLayout()
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

        self.y_abs_layout.addWidget(self.spin_ylim_min)
        self.y_abs_layout.addWidget(QLabel("to"))
        self.y_abs_layout.addWidget(self.spin_ylim_max)

        # Y Range - Diff mode
        self.y_diff_layout = QHBoxLayout()
        self.spin_y_center = QDoubleSpinBox()
        self.spin_y_center.setRange(-1e9, 1e9)
        self.spin_y_center.setDecimals(4)
        self.spin_y_center.setSingleStep(1.0)
        self.spin_y_center.setKeyboardTracking(False)
        self.spin_y_center.setToolTip("Center of Y range")

        self.spin_y_height = QDoubleSpinBox()
        self.spin_y_height.setRange(0.0, 2e9)
        self.spin_y_height.setDecimals(4)
        self.spin_y_height.setSingleStep(1.0)
        self.spin_y_height.setKeyboardTracking(False)
        self.spin_y_height.setToolTip("Height of Y range")

        self.y_diff_layout.addWidget(QLabel("Center:"))
        self.y_diff_layout.addWidget(self.spin_y_center)
        self.y_diff_layout.addWidget(QLabel("Height:"))
        self.y_diff_layout.addWidget(self.spin_y_height)

        form.addRow("Y Range:", self.y_abs_layout)

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
        self.spin_x_center.editingFinished.connect(self._on_apply_limits_clicked)
        self.spin_x_width.editingFinished.connect(self._on_apply_limits_clicked)
        self.spin_y_center.editingFinished.connect(self._on_apply_limits_clicked)
        self.spin_y_height.editingFinished.connect(self._on_apply_limits_clicked)

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
                "ChannelGroupContainer",
            )
        ):
            return True
        return hasattr(w, "populate_settings_panel") and callable(
            getattr(w, "populate_settings_panel")
        )

    def _disconnect_current_view(self):
        """Cleanly disconnect any signals from the previously active view."""
        self._disconnect_navigation_signals()
        self._disconnect_active_state_signals()
        for sig, slot in getattr(self, "_current_view_connections", []):
            try:
                sig.disconnect(slot)
            except (TypeError, RuntimeError):
                pass
        if hasattr(self, "_current_view_connections"):
            self._current_view_connections.clear()
        if hasattr(self, "mt_plot_manager"):
            self.mt_plot_manager.set_view(None)
        self.active_view = None

    def _disconnect_active_state_signals(self):
        for sig, slot in self._active_state_connections:
            try:
                sig.disconnect(slot)
            except (TypeError, RuntimeError):
                pass
        self._active_state_connections.clear()

    def _connect_active_state_signals(self, view: Optional[QWidget]):
        """Listen to the active view's own (per-container) follow/index signals."""
        self._disconnect_active_state_signals()
        if view is None:
            return
        state = getattr(view, "view_state", None) or getattr(view, "state", None)
        if state is None:
            return
        for name in ("onFollowLiveChanged", "onStimNoChange", "onNewSweep"):
            sig = getattr(state, name, None)
            if sig is None:
                continue
            try:
                sig.connect(self._sync_follow_events_checkbox)
                self._active_state_connections.append(
                    (sig, self._sync_follow_events_checkbox)
                )
            except (TypeError, RuntimeError):
                pass

    def _on_focus_changed(self, old: Optional[QWidget], new: Optional[QWidget]):
        if new is not None:
            self.setActiveView(new)

    def set_state(self, state):
        self.state = state
        if hasattr(self, "unit_info_widget") and self.unit_info_widget is not None:
            self.unit_info_widget.set_state(state)
        if hasattr(self, "channel_groups_editor") and self.channel_groups_editor is not None:
            self.channel_groups_editor.setState(state)
        if self.state is not None:
            self._connect_state_signals()
            self._refresh_unit_combo()
            self._refresh_channel_combo()
            self._sync_follow_events_checkbox()
            if hasattr(self, "chk_hide_top_bars"):
                self.chk_hide_top_bars.setChecked(getattr(self.state, "hide_view_top_bars", False))

    def setState(self, state):
        self.set_state(state)

    def show_channel_groups_editor(self, open_pane: bool = True, target_bar: Optional[ChannelGroupBar] = None):
        """Shows or hides the Channel Groups editor pane inside the View Settings panel."""
        self.grp_unit_selector.setVisible(True)
        with QSignalBlocker(self.btn_channel_settings):
            self.btn_channel_settings.setChecked(bool(open_pane))
        self.channel_groups_editor.setVisible(bool(open_pane))
        if open_pane:
            bar = target_bar if target_bar is not None else self._get_active_channel_bar()
            self.channel_groups_editor.set_active_bar(bar)
            self.channel_groups_editor.refresh()

    def _on_toggle_channel_groups_editor(self, checked: bool):
        self.channel_groups_editor.setVisible(checked)
        if checked:
            self.channel_groups_editor.set_active_bar(self._get_active_channel_bar())
            self.channel_groups_editor.refresh()

    def _on_editor_groups_changed(self):
        self._refresh_channel_combo()
        self._sync_channel_selection_from_bar()
        container = self._find_parent_container(self.active_view)
        if container is not None and hasattr(container, "_sync_all_children"):
            container._sync_all_children()
        self.settingsChanged.emit()

    def _connect_state_signals(self):
        if self.state is None:
            return
        if not hasattr(self, "_state_connections"):
            self._state_connections = []
        for sig, slot in self._state_connections:
            try:
                sig.disconnect(slot)
            except (TypeError, RuntimeError):
                pass
        self._state_connections.clear()

        if hasattr(self.state, "onUnitGroupChange"):
            self._state_connections.append((self.state.onUnitGroupChange, self._on_state_unit_changed))
            self.state.onUnitGroupChange.connect(self._on_state_unit_changed)
        if hasattr(self.state, "onActiveUnitChanged"):
            self._state_connections.append((self.state.onActiveUnitChanged, self._on_state_unit_changed))
            self.state.onActiveUnitChanged.connect(self._on_state_unit_changed)
        if hasattr(self.state, "onUnitListChanged"):
            self._state_connections.append((self.state.onUnitListChanged, self._refresh_unit_combo))
            self.state.onUnitListChanged.connect(self._refresh_unit_combo)
        if hasattr(self.state, "onLoadNewFile"):
            self._state_connections.append((self.state.onLoadNewFile, self._refresh_unit_combo))
            self.state.onLoadNewFile.connect(self._refresh_unit_combo)
        if hasattr(self.state, "onHideViewTopBarsChanged"):
            self._state_connections.append((self.state.onHideViewTopBarsChanged, self._on_hide_top_bars_state_changed))
            self.state.onHideViewTopBarsChanged.connect(self._on_hide_top_bars_state_changed)
        if hasattr(self.state, "onChannelGroupsChanged"):
            self._state_connections.append((self.state.onChannelGroupsChanged, self._refresh_channel_combo))
            self.state.onChannelGroupsChanged.connect(self._refresh_channel_combo)
        if hasattr(self.state, "onChannelChange"):
            self._state_connections.append((self.state.onChannelChange, self._refresh_channel_combo))
            self.state.onChannelChange.connect(self._refresh_channel_combo)
        if hasattr(self.state, "onSharedChannelSetChanged"):
            self._state_connections.append((self.state.onSharedChannelSetChanged, self._refresh_channel_combo))
            self.state.onSharedChannelSetChanged.connect(self._refresh_channel_combo)
        if hasattr(self.state, "onFollowLiveChanged"):
            self._state_connections.append((self.state.onFollowLiveChanged, self._sync_follow_events_checkbox))
            self.state.onFollowLiveChanged.connect(self._sync_follow_events_checkbox)
        if hasattr(self.state, "onStimNoChange"):
            self._state_connections.append((self.state.onStimNoChange, self._sync_follow_events_checkbox))
            self.state.onStimNoChange.connect(self._sync_follow_events_checkbox)
        if hasattr(self.state, "onNewSweep"):
            self._state_connections.append((self.state.onNewSweep, self._sync_follow_events_checkbox))
            self.state.onNewSweep.connect(self._sync_follow_events_checkbox)

    def _find_parent_container(self, view: Optional[QWidget]) -> Optional[Any]:
        if view is None:
            return None
        if not self._is_view_alive(view):
            return None
        if view.__class__.__name__ == "ChannelGroupContainerView":
            return view
        curr = view.parentWidget() if hasattr(view, "parentWidget") else None
        while curr is not None:
            if curr.__class__.__name__ == "ChannelGroupContainerView":
                return curr
            curr = curr.parentWidget() if hasattr(curr, "parentWidget") else None
        win = self.window()
        if win is not None:
            for cgc in win.findChildren(QWidget):
                if cgc.__class__.__name__ == "ChannelGroupContainerView":
                    if hasattr(cgc, "sub_views") and view in cgc.sub_views:
                        return cgc
        return None

    def _refresh_unit_combo(self):
        curr_data = self.combo_unit.currentData()
        with QSignalBlocker(self.combo_unit):
            self.combo_unit.clear()
            self.combo_unit.addItem("(Follow Active Unit)", userData=None)
            if self.state and hasattr(self.state, "spike_groups") and self.state.spike_groups:
                for i, sg in enumerate(self.state.spike_groups):
                    u_name = getattr(sg.event, "name", f"unit_{i}") or f"unit_{i}"
                    ch = getattr(sg, "channel", "")
                    ch_info = f" [{ch}]" if ch else ""
                    self.combo_unit.addItem(f"Unit {i}: {u_name}{ch_info}", userData=i)
            restored = False
            if curr_data is not None:
                for idx in range(self.combo_unit.count()):
                    if self.combo_unit.itemData(idx) == curr_data:
                        self.combo_unit.setCurrentIndex(idx)
                        restored = True
                        break
            if not restored:
                self.combo_unit.setCurrentIndex(0)

    def _resolve_active_view_bound_unit(self, view: Optional[QWidget]) -> Optional[int]:
        if view is None or not self._is_view_alive(view):
            return None
        container = self._find_parent_container(view)
        if container is not None and hasattr(container, "channel_bar") and container.channel_bar is not None:
            return container.channel_bar.get_bound_unit()
        if hasattr(view, "channel_bar") and view.channel_bar is not None:
            return view.channel_bar.get_bound_unit()
        if hasattr(view, "get_target_unit_idx") and callable(getattr(view, "get_target_unit_idx")):
            if hasattr(view, "channel_bar") and hasattr(view.channel_bar, "is_unit_locked") and view.channel_bar.is_unit_locked():
                return view.get_target_unit_idx()
        return None

    def _on_state_unit_changed(self, *args):
        self._refresh_unit_combo()
        if self.active_view is not None:
            bound = self._resolve_active_view_bound_unit(self.active_view)
            with QSignalBlocker(self.combo_unit):
                target_idx = 0
                for idx in range(self.combo_unit.count()):
                    if self.combo_unit.itemData(idx) == bound:
                        target_idx = idx
                        break
                self.combo_unit.setCurrentIndex(target_idx)
            if hasattr(self, "unit_info_widget") and self.unit_info_widget is not None:
                self.unit_info_widget.set_target_unit(bound)
        else:
            if hasattr(self, "unit_info_widget") and self.unit_info_widget is not None:
                self.unit_info_widget.set_target_unit(None)

    def _on_unit_selection_changed(self, idx: int):
        if self._block_updates or self.active_view is None:
            return
        selected_unit = self.combo_unit.itemData(idx)
        container = self._find_parent_container(self.active_view)
        if container is not None:
            # When within a channel group, changing unit affects all groups
            if hasattr(container, "channel_bar") and container.channel_bar is not None:
                container.channel_bar.bind_to_unit(selected_unit)
            if hasattr(container, "_sync_all_children"):
                container._sync_all_children()

            win = self.window()
            if win is not None:
                for cgc in win.findChildren(QWidget):
                    if cgc is not container and cgc.__class__.__name__ == "ChannelGroupContainerView":
                        if hasattr(cgc, "channel_bar") and cgc.channel_bar is not None:
                            cgc.channel_bar.bind_to_unit(selected_unit)
                        if hasattr(cgc, "_sync_all_children"):
                            cgc._sync_all_children()

            if hasattr(self.active_view, "channel_bar") and self.active_view.channel_bar is not None:
                self.active_view.channel_bar.bind_to_unit(selected_unit)
        else:
            if hasattr(self.active_view, "channel_bar") and self.active_view.channel_bar is not None:
                self.active_view.channel_bar.bind_to_unit(selected_unit)
            elif hasattr(self.active_view, "bind_to_unit"):
                self.active_view.bind_to_unit(selected_unit)
        if hasattr(self, "unit_info_widget") and self.unit_info_widget is not None:
            self.unit_info_widget.set_target_unit(selected_unit)
        self.settingsChanged.emit()

    def _get_active_channel_bar(self) -> Optional[ChannelGroupBar]:
        if self.active_view is None:
            return None
        if not self._is_view_alive(self.active_view):
            return None
        container = self._find_parent_container(self.active_view)
        if container is not None and hasattr(container, "channel_bar") and container.channel_bar is not None:
            return container.channel_bar
        if hasattr(self.active_view, "channel_bar") and self.active_view.channel_bar is not None:
            return self.active_view.channel_bar
        return None

    def _refresh_channel_combo(self):
        curr_data = self.combo_channel.currentData()
        target_bar = self._get_active_channel_bar()
        if target_bar is not None and curr_data is None:
            curr_data = target_bar.get_channel_mode()

        with QSignalBlocker(self.combo_channel):
            self.combo_channel.clear()
            self.combo_channel.addItem("Auto (Active Channel)", userData=(ChannelGroupBar.MODE_FOLLOW_GLOBAL, None))
            self.combo_channel.addItem("Unit Assigned Channels", userData=(ChannelGroupBar.MODE_FOLLOW_UNIT, None))
            self.combo_channel.addItem("Unit Detected Channels", userData=(ChannelGroupBar.MODE_UNIT_DETECTED, None))
            self.combo_channel.addItem("Auto (Live Stream Input)", userData=(ChannelGroupBar.MODE_FOLLOW_LIVE, None))

            if self.state:
                all_groups = self.state.get_all_channel_groups() if hasattr(self.state, "get_all_channel_groups") else {}
                if all_groups:
                    self.combo_channel.insertSeparator(self.combo_channel.count())
                    for g_name, g_chans in all_groups.items():
                        preview = f" ({len(g_chans)} ch: {', '.join(g_chans[:3])}{'...' if len(g_chans)>3 else ''})"
                        self.combo_channel.addItem(f"[Group] {g_name}{preview}", userData=(ChannelGroupBar.MODE_GROUP, g_name))

                self.combo_channel.insertSeparator(self.combo_channel.count())
                self.combo_channel.addItem("Shared Group A", userData=(ChannelGroupBar.MODE_SHARED_SET, "Shared Group A"))
                self.combo_channel.addItem("Shared Group B", userData=(ChannelGroupBar.MODE_SHARED_SET, "Shared Group B"))

                avail = self.state.get_available_channels() if hasattr(self.state, "get_available_channels") else []
                if avail:
                    self.combo_channel.insertSeparator(self.combo_channel.count())
                    for ch in avail:
                        self.combo_channel.addItem(ch, userData=(ChannelGroupBar.MODE_SINGLE, ch))

            restored = False
            if curr_data is not None and isinstance(curr_data, (tuple, list)) and len(curr_data) == 2:
                for idx in range(self.combo_channel.count()):
                    d = self.combo_channel.itemData(idx)
                    if isinstance(d, (tuple, list)) and len(d) == 2 and d[0] == curr_data[0] and (curr_data[1] is None or d[1] == curr_data[1]):
                        self.combo_channel.setCurrentIndex(idx)
                        restored = True
                        break
            if not restored:
                self.combo_channel.setCurrentIndex(0)

    def _build_panel_group_menu(self):
        menu = QMenu(self.btn_group_menu)
        menu.addAction("Edit Channels in Group...", self._open_channel_group_dialog)
        menu.addAction("Quick 4-Ch Tetrode Group", self._quick_select_tetrode)
        menu.addSeparator()
        menu.addAction("Shift Group -4 Channels", lambda: self._shift_channel_group(-4))
        menu.addAction("Shift Group +4 Channels", lambda: self._shift_channel_group(4))
        menu.addAction("Shift by Group Span (-N)", lambda: self._shift_group_by_span(-1))
        menu.addAction("Shift by Group Span (+N)", lambda: self._shift_group_by_span(1))
        menu.addSeparator()
        menu.addAction("Assign Group to Bound Unit", self._assign_group_to_unit)
        menu.addAction("Save Current Group as Named Preset...", self._save_group_preset)
        menu.addAction("Auto-Detect Unit Channels (Footprint)", self._auto_detect_unit_channels)
        self.btn_group_menu.setMenu(menu)

    def _on_channel_selection_changed(self, idx: int):
        if self._block_updates:
            return
        curr_data = self.combo_channel.itemData(idx)
        if curr_data is None or not isinstance(curr_data, (tuple, list)) or len(curr_data) != 2:
            return
        mode, val = curr_data
        target_bar = self._get_active_channel_bar()
        if target_bar is not None:
            c_idx = target_bar.find_channel_data_index(mode, val)
            if c_idx >= 0 and target_bar.combo_channel.currentIndex() != c_idx:
                target_bar.combo_channel.setCurrentIndex(c_idx)
            elif c_idx < 0:
                if mode == ChannelGroupBar.MODE_SINGLE:
                    target_bar.bind_to_channel(str(val))
                elif mode == ChannelGroupBar.MODE_GROUP:
                    target_bar.bind_to_group(str(val))
                elif mode == ChannelGroupBar.MODE_FOLLOW_GLOBAL:
                    target_bar.set_channel_mode_global()
                elif mode == ChannelGroupBar.MODE_FOLLOW_LIVE:
                    target_bar.set_channel_mode_live()
                elif mode == ChannelGroupBar.MODE_FOLLOW_UNIT:
                    target_bar.set_channel_mode_unit()
                elif mode == ChannelGroupBar.MODE_UNIT_DETECTED:
                    target_bar.set_channel_mode_unit_detected()
                elif mode == ChannelGroupBar.MODE_SHARED_SET:
                    target_bar.bind_to_shared_set(str(val))

            container = self._find_parent_container(self.active_view)
            if container is not None and hasattr(container, "_sync_all_children"):
                container._sync_all_children()
            with QSignalBlocker(self.chk_multichannel):
                self.chk_multichannel.setChecked(target_bar.is_multichannel())
        elif self.state is not None:
            if mode == ChannelGroupBar.MODE_SINGLE and hasattr(self.state, "set_channel"):
                self.state.set_channel(str(val))
        self.settingsChanged.emit()

    def _on_multichannel_toggled(self, checked: bool):
        if self._block_updates:
            return
        target_bar = self._get_active_channel_bar()
        if target_bar is not None:
            if target_bar.chk_multichannel.isChecked() != checked:
                target_bar.chk_multichannel.setChecked(checked)
            container = self._find_parent_container(self.active_view)
            if container is not None and hasattr(container, "_sync_all_children"):
                container._sync_all_children()
        self.settingsChanged.emit()

    def _shift_channel_group(self, step: int):
        target_bar = self._get_active_channel_bar()
        if target_bar is not None:
            target_bar.shift_group(step)
            container = self._find_parent_container(self.active_view)
            if container is not None and hasattr(container, "_sync_all_children"):
                container._sync_all_children()
            self._sync_channel_selection_from_bar()
        elif self.state is not None:
            avail = self.state.get_available_channels() if hasattr(self.state, "get_available_channels") else []
            if avail:
                cur = getattr(self.state.analog_signal, "name", avail[0])
                shifted = self.state.shift_channels([cur], step=step)
                if shifted and hasattr(self.state, "set_channel"):
                    self.state.set_channel(shifted[0])
            self._refresh_channel_combo()
        self.settingsChanged.emit()

    def _shift_group_by_span(self, direction: int = 1):
        target_bar = self._get_active_channel_bar()
        if target_bar is not None:
            target_bar.shift_group_by_span(direction)
            container = self._find_parent_container(self.active_view)
            if container is not None and hasattr(container, "_sync_all_children"):
                container._sync_all_children()
            self._sync_channel_selection_from_bar()

    def _open_channel_group_dialog(self):
        target_bar = self._get_active_channel_bar()
        if target_bar is not None:
            target_bar.open_channel_group_dialog()
            container = self._find_parent_container(self.active_view)
            if container is not None and hasattr(container, "_sync_all_children"):
                container._sync_all_children()
            self._sync_channel_selection_from_bar()
        else:
            from .ChannelGroupBar import ChannelGroupDialog
            dlg = ChannelGroupDialog(parent=self, state=self.state)
            if dlg.exec() == QDialog.Accepted:
                self._refresh_channel_combo()

    def _quick_select_tetrode(self):
        target_bar = self._get_active_channel_bar()
        if target_bar is not None:
            target_bar.quick_select_tetrode()
            container = self._find_parent_container(self.active_view)
            if container is not None and hasattr(container, "_sync_all_children"):
                container._sync_all_children()
            self._sync_channel_selection_from_bar()

    def _assign_group_to_unit(self):
        target_bar = self._get_active_channel_bar()
        if target_bar is not None:
            target_bar.assign_group_to_unit()
            container = self._find_parent_container(self.active_view)
            if container is not None and hasattr(container, "_sync_all_children"):
                container._sync_all_children()
            self._sync_channel_selection_from_bar()

    def _save_group_preset(self):
        target_bar = self._get_active_channel_bar()
        if target_bar is not None:
            target_bar.save_group_preset()
            self._refresh_channel_combo()

    def _auto_detect_unit_channels(self):
        target_bar = self._get_active_channel_bar()
        if target_bar is not None:
            target_bar.auto_detect_unit_channels()
            container = self._find_parent_container(self.active_view)
            if container is not None and hasattr(container, "_sync_all_children"):
                container._sync_all_children()
            self._sync_channel_selection_from_bar()

    def _sync_channel_selection_from_bar(self):
        target_bar = self._get_active_channel_bar()
        if target_bar is not None:
            mode, val = target_bar.get_channel_mode()
            with QSignalBlocker(self.combo_channel):
                for idx in range(self.combo_channel.count()):
                    d = self.combo_channel.itemData(idx)
                    if isinstance(d, (tuple, list)) and len(d) == 2 and d[0] == mode and (val is None or d[1] == val):
                        self.combo_channel.setCurrentIndex(idx)
                        break
            with QSignalBlocker(self.chk_multichannel):
                self.chk_multichannel.setChecked(target_bar.is_multichannel())
            if hasattr(self, "channel_groups_editor") and self.channel_groups_editor is not None:
                self.channel_groups_editor.set_active_bar(target_bar)
        self._sync_follow_events_checkbox()

    def is_following_live_events(self) -> bool:
        """Determines if the active view / container / state is following live new events."""
        if self.active_view is not None and not self._is_view_alive(self.active_view):
            self.active_view = None
        target_bar = self._get_active_channel_bar()
        if target_bar is not None and hasattr(target_bar, "is_following_live_events"):
            return target_bar.is_following_live_events()
        container = self._find_parent_container(self.active_view)
        if container is not None and hasattr(container, "is_following_live_events"):
            return container.is_following_live_events()
        if self.state is not None and hasattr(self.state, "is_following_live_events"):
            return self.state.is_following_live_events()
        return self.chk_follow_events.isChecked()

    def set_follow_live_events(self, enable: bool = True):
        """Enables or disables following live new events."""
        target_bar = self._get_active_channel_bar()
        container = self._find_parent_container(self.active_view)
        if target_bar is not None and hasattr(target_bar, "set_follow_live_events"):
            target_bar.set_follow_live_events(enable)
        elif container is not None and hasattr(container, "set_follow_live_events"):
            container.set_follow_live_events(enable)
        elif self.state is not None and hasattr(self.state, "set_follow_live"):
            self.state.set_follow_live(enable)
        self._sync_follow_events_checkbox()

    def toggle_follow_live_events(self):
        """Toggles following live new events."""
        self.set_follow_live_events(not self.is_following_live_events())

    def _on_follow_events_toggled(self, checked: bool):
        if self._block_updates:
            return
        self.set_follow_live_events(checked)

    def _sync_follow_events_checkbox(self, *args):
        following = self.is_following_live_events()
        if self.chk_follow_events.isChecked() != following:
            with QSignalBlocker(self.chk_follow_events):
                self.chk_follow_events.setChecked(following)

    def _on_hide_top_bars_toggled(self, checked: bool):
        if self.state is not None:
            self.state.set_hide_view_top_bars(checked)

    def _on_hide_top_bars_state_changed(self, hide: bool):
        if self.chk_hide_top_bars.isChecked() != hide:
            with QSignalBlocker(self.chk_hide_top_bars):
                self.chk_hide_top_bars.setChecked(hide)
        from .helpers import apply_top_bars_visibility
        app = QApplication.instance()
        if app is not None:
            for top_w in app.topLevelWidgets():
                if top_w is not self:
                    apply_top_bars_visibility(top_w, hide)

    def closeEvent(self, event):
        if getattr(self, "_focus_connected", False):
            app = QApplication.instance()
            if app is not None:
                try:
                    app.focusChanged.disconnect(self._on_focus_changed)
                except (TypeError, RuntimeError):
                    pass
            self._focus_connected = False
        if hasattr(self, "_state_connections"):
            for sig, slot in self._state_connections:
                try:
                    sig.disconnect(slot)
                except (TypeError, RuntimeError):
                    pass
            self._state_connections.clear()
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

        # Now search up for a view widget or dock area
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

            # Check CDockWidget unwrapping
            if (
                c_name == "CDockWidget"
                and hasattr(curr, "widget")
                and callable(getattr(curr, "widget"))
            ):
                try:
                    inner = curr.widget()
                    if inner is self:
                        return self.active_view
                    if inner is not None:
                        found_view = inner
                        break
                except TypeError:
                    pass

            # Check CDockWidgetTab unwrapping
            elif (
                c_name == "CDockWidgetTab"
                and hasattr(curr, "dockWidget")
                and callable(getattr(curr, "dockWidget"))
            ):
                try:
                    dw = curr.dockWidget()
                    inner = dw.widget() if dw is not None else None
                    if inner is self:
                        return self.active_view
                    if inner is not None:
                        found_view = inner
                        break
                except TypeError:
                    pass

            # Check CDockAreaWidget or title bar / tab bar having dockAreaWidget()
            if hasattr(curr, "dockAreaWidget") and callable(getattr(curr, "dockAreaWidget")):
                try:
                    area = curr.dockAreaWidget()
                    if area and hasattr(area, "currentDockWidget") and callable(getattr(area, "currentDockWidget")):
                        dw = area.currentDockWidget()
                        inner = dw.widget() if dw and hasattr(dw, "widget") else None
                        if inner is self:
                            return self.active_view
                        if inner is not None:
                            found_view = inner
                            break
                except TypeError:
                    pass

            # Check if curr itself is a CDockAreaWidget
            if hasattr(curr, "currentDockWidget") and callable(getattr(curr, "currentDockWidget")):
                try:
                    dw = curr.currentDockWidget()
                    inner = dw.widget() if dw and hasattr(dw, "widget") else None
                    if inner is self:
                        return self.active_view
                    if inner is not None:
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
                # Check if w was in an area of container's dock_manager
                if hasattr(found_view, "dock_manager") and found_view.dock_manager is not None:
                    curr_d = w
                    while curr_d is not None and curr_d is not found_view:
                        if hasattr(curr_d, "dockAreaWidget") and callable(getattr(curr_d, "dockAreaWidget")):
                            area = curr_d.dockAreaWidget()
                            if area and hasattr(area, "currentDockWidget") and area.currentDockWidget():
                                candidate = area.currentDockWidget().widget()
                                if candidate in found_view.sub_views:
                                    return candidate
                        if hasattr(curr_d, "currentDockWidget") and callable(getattr(curr_d, "currentDockWidget")):
                            candidate = curr_d.currentDockWidget().widget()
                            if candidate in found_view.sub_views:
                                return candidate
                        curr_d = curr_d.parentWidget() if hasattr(curr_d, "parentWidget") else None
                    focused = found_view.dock_manager.focusedDockWidget()
                    if focused and hasattr(focused, "widget") and focused.widget() in found_view.sub_views:
                        return focused.widget()
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
            self.grp_unit_selector.setVisible(False)
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
            if hasattr(self, "unit_info_widget") and self.unit_info_widget is not None:
                self.unit_info_widget.set_target_unit(None)
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
        self._connect_active_state_signals(resolved)
        view = resolved
        self._block_updates = True
        try:
            # Hide all groups first
            self.placeholder.setVisible(False)
            self.grp_multitrace.setVisible(False)
            if hasattr(self, "mt_plot_manager"):
                self.mt_plot_manager.set_view(None)
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
                self.grp_unit_selector.setVisible(False)
                self.placeholder.setVisible(True)
                return

            self.grp_unit_selector.setVisible(True)
            bound_unit = self._resolve_active_view_bound_unit(view)

            with QSignalBlocker(self.combo_unit):
                target_idx = 0
                for idx in range(self.combo_unit.count()):
                    if self.combo_unit.itemData(idx) == bound_unit:
                        target_idx = idx
                        break
                self.combo_unit.setCurrentIndex(target_idx)

            if hasattr(self, "unit_info_widget") and self.unit_info_widget is not None:
                self.unit_info_widget.set_target_unit(bound_unit)

            for sig, slot in getattr(self, "_current_view_connections", []):
                try:
                    sig.disconnect(slot)
                except (TypeError, RuntimeError):
                    pass
            if hasattr(self, "_current_view_connections"):
                self._current_view_connections.clear()

            target_bar = self._get_active_channel_bar()
            if target_bar is not None:
                self._sync_channel_selection_from_bar()
                try:
                    target_bar.bindingChanged.connect(self._on_active_bar_binding_changed)
                    self._current_view_connections.append((target_bar.bindingChanged, self._on_active_bar_binding_changed))
                except Exception:
                    pass
            else:
                self._refresh_channel_combo()
                self._sync_follow_events_checkbox()

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
        finally:
            self._block_updates = False

    def _populate_multitrace(self, view):
        supports_plots = hasattr(view, "get_plot_specs") and callable(
            getattr(view, "get_plot_specs")
        )
        if hasattr(self, "mt_plot_manager"):
            self.mt_plot_manager.setVisible(supports_plots)
            if hasattr(self, "lbl_mt_plots"):
                self.lbl_mt_plots.setVisible(supports_plots)
            self.mt_plot_manager.set_view(view if supports_plots else None)

        # Scale mode
        cur_scale_mode = getattr(view, "scale_mode", "std")
        if hasattr(view, "get_settings") and callable(getattr(view, "get_settings")):
            try:
                cur_scale_mode = view.get_settings().get("scale_mode", cur_scale_mode)
            except Exception:
                pass
        idx = self.combo_mt_scale_mode.findText(cur_scale_mode)
        if idx >= 0:
            self.combo_mt_scale_mode.setCurrentIndex(idx)

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
        # Scale mode
        cur_scale_mode = "mV"
        if hasattr(view, "get_settings") and callable(getattr(view, "get_settings")):
            try:
                cur_scale_mode = view.get_settings().get("scale_mode", cur_scale_mode)
            except Exception:
                pass
        elif hasattr(view, "settings") and isinstance(view.settings, dict):
            cur_scale_mode = view.settings.get("scale_mode", cur_scale_mode)
        if hasattr(self, "combo_fix_scale_mode"):
            idx = self.combo_fix_scale_mode.findText(cur_scale_mode)
            if idx >= 0:
                self.combo_fix_scale_mode.setCurrentIndex(idx)

        if hasattr(self, "combo_fix_axis_mode"):
            if hasattr(view, "y_axis_combo"):
                self.combo_fix_axis_mode.setCurrentIndex(
                    view.y_axis_combo.currentIndex()
                )
            elif hasattr(view, "settings") and "y_axis_mode" in view.settings:
                idx = 1 if view.settings.get("y_axis_mode") == "time" else 0
                self.combo_fix_axis_mode.setCurrentIndex(idx)

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

        if hasattr(self, "spin_fix_time_window"):
            if hasattr(view, "time_window_spinner"):
                self.spin_fix_time_window.setValue(
                    float(view.time_window_spinner.value())
                )
            elif hasattr(view, "settings") and "time_window_s" in view.settings:
                self.spin_fix_time_window.setValue(
                    float(view.settings.get("time_window_s", 10.0))
                )

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
    # Shared Visual Settings Broadcasting
    # ---------------------------------------------------------
    def _on_active_bar_binding_changed(self):
        self._sync_channel_selection_from_bar()
        if self.active_view is not None and self._is_view_alive(self.active_view):
            bound = self._resolve_active_view_bound_unit(self.active_view)
            with QSignalBlocker(self.combo_unit):
                target_idx = 0
                for idx in range(self.combo_unit.count()):
                    if self.combo_unit.itemData(idx) == bound:
                        target_idx = idx
                        break
                self.combo_unit.setCurrentIndex(target_idx)
            if hasattr(self, "unit_info_widget") and self.unit_info_widget is not None:
                self.unit_info_widget.set_target_unit(bound)

    def _broadcast_shared_visual_settings(self, **kwargs):
        if (
            not hasattr(self, "chk_sync_group_visuals")
            or not self.chk_sync_group_visuals.isChecked()
            or not kwargs
        ):
            return

        clean_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        if not clean_kwargs:
            return

        # 1. Broadcast to ChannelGroupContainerView if active view lives inside one
        container = self._find_parent_container(self.active_view)
        if container is not None:
            if (
                hasattr(container, "chk_sync_visuals")
                and not container.chk_sync_visuals.isChecked()
            ):
                pass
            else:
                if (
                    getattr(container, "view_state", None) is not None
                    and hasattr(container.view_state, "update_visual_settings")
                ):
                    container.view_state.update_visual_settings(**clean_kwargs)
                if hasattr(container, "sub_views"):
                    for sv in container.sub_views:
                        if sv is not self.active_view and hasattr(sv, "apply_shared_visual_settings"):
                            sv.apply_shared_visual_settings(clean_kwargs)

        # 2. Broadcast to shared channel set if bound to one
        target_bar = self._get_active_channel_bar()
        if target_bar is not None and hasattr(target_bar, "get_channel_mode"):
            mode, val = target_bar.get_channel_mode()
            if (
                mode == ChannelGroupBar.MODE_SHARED_SET
                and val
                and self.state is not None
                and hasattr(self.state, "update_shared_visual_settings")
            ):
                self.state.update_shared_visual_settings(val, **clean_kwargs)

    # ---------------------------------------------------------
    # Event Handlers to update active view
    # ---------------------------------------------------------
    def _on_mt_scale_mode_changed(self, text: str):
        if self._block_updates or not self.active_view:
            return
        if hasattr(self.active_view, "scale_mode_combo"):
            self.active_view.scale_mode_combo.setCurrentText(text)
        elif hasattr(self.active_view, "_on_scale_mode_changed"):
            self.active_view.scale_mode = text
            self.active_view._on_scale_mode_changed(text)
        elif hasattr(self.active_view, "set_settings"):
            self.active_view.set_settings({"scale_mode": text})
        self._broadcast_shared_visual_settings(scale_mode=text)
        self.settingsChanged.emit()

    def _on_fix_scale_mode_changed(self, text: str):
        if self._block_updates or not self.active_view:
            return
        if hasattr(self.active_view, "scale_mode_combo"):
            self.active_view.scale_mode_combo.setCurrentText(text)
        elif hasattr(self.active_view, "set_scale_setting"):
            self.active_view.set_scale_setting(text)
        elif hasattr(self.active_view, "set_settings"):
            self.active_view.set_settings({"scale_mode": text})
        self._broadcast_shared_visual_settings(scale_mode=text)
        self.settingsChanged.emit()

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
        self._broadcast_shared_visual_settings(color_scale=(float(low), float(high)))
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
        self._broadcast_shared_visual_settings(trace_scale=float(val))
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

    def _on_fix_axis_mode_changed(self, idx: int):
        if self._block_updates or not self.active_view:
            return
        mode = "time" if idx == 1 else "sweep"
        if hasattr(self.active_view, "set_settings"):
            self.active_view.set_settings({"y_axis_mode": mode})
        elif hasattr(self.active_view, "y_axis_combo"):
            self.active_view.y_axis_combo.setCurrentIndex(idx)
        self.settingsChanged.emit()

    def _on_fix_time_window_changed(self, val: float):
        if self._block_updates or not self.active_view:
            return
        if hasattr(self.active_view, "set_settings"):
            self.active_view.set_settings({"time_window_s": val})
        elif hasattr(self.active_view, "time_window_spinner"):
            self.active_view.time_window_spinner.setValue(val)
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
                
                # Populate based on current range mode
                if self.combo_range_mode.currentIndex() == 0:  # Absolute mode
                    self.spin_xlim_min.setValue(xmin)
                    self.spin_xlim_max.setValue(xmax)
                    self.spin_ylim_min.setValue(ymin)
                    self.spin_ylim_max.setValue(ymax)
                else:  # Diff mode
                    self.spin_x_center.setValue((xmin + xmax) / 2)
                    self.spin_x_width.setValue(xmax - xmin)
                    self.spin_y_center.setValue((ymin + ymax) / 2)
                    self.spin_y_height.setValue(ymax - ymin)

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
                xmin, xmax = float(min(xlim)), float(max(xlim))
                ymin, ymax = float(min(ylim)), float(max(ylim))
                
                if self.combo_range_mode.currentIndex() == 0:  # Absolute mode
                    self.spin_xlim_min.setValue(xmin)
                    self.spin_xlim_max.setValue(xmax)
                    self.spin_ylim_min.setValue(ymin)
                    self.spin_ylim_max.setValue(ymax)
                else:  # Diff mode
                    self.spin_x_center.setValue((xmin + xmax) / 2)
                    self.spin_x_width.setValue(xmax - xmin)
                    self.spin_y_center.setValue((ymin + ymax) / 2)
                    self.spin_y_height.setValue(ymax - ymin)
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
            or self.spin_x_center.hasFocus()
            or self.spin_x_width.hasFocus()
            or self.spin_y_center.hasFocus()
            or self.spin_y_height.hasFocus()
        ):
            return
        if hasattr(self.active_view, "get_view_range") and callable(
            getattr(self.active_view, "get_view_range")
        ):
            (xmin, xmax), (ymin, ymax) = self.active_view.get_view_range()
            self._block_updates = True
            try:
                if self.combo_range_mode.currentIndex() == 0:  # Absolute mode
                    self.spin_xlim_min.setValue(xmin)
                    self.spin_xlim_max.setValue(xmax)
                    self.spin_ylim_min.setValue(ymin)
                    self.spin_ylim_max.setValue(ymax)
                else:  # Diff mode
                    self.spin_x_center.setValue((xmin + xmax) / 2)
                    self.spin_x_width.setValue(xmax - xmin)
                    self.spin_y_center.setValue((ymin + ymax) / 2)
                    self.spin_y_height.setValue(ymax - ymin)
            finally:
                self._block_updates = False

    def _on_range_mode_changed(self, idx: int):
        if self._block_updates or not self._is_view_alive(self.active_view):
            return
        # Switch between absolute and diff layouts
        is_absolute = (idx == 0)
        
        # Update form layout - replace the widget in the row
        if is_absolute:
            # Show absolute, hide diff
            self._replace_form_row("X Range:", self.x_abs_layout)
            self._replace_form_row("Y Range:", self.y_abs_layout)
        else:
            # Show diff, hide absolute
            self._replace_form_row("X Range:", self.x_diff_layout)
            self._replace_form_row("Y Range:", self.y_diff_layout)
        
        # Re-populate from current view range
        if hasattr(self.active_view, "get_view_range") and callable(
            getattr(self.active_view, "get_view_range")
        ):
            (xmin, xmax), (ymin, ymax) = self.active_view.get_view_range()
            if is_absolute:
                self.spin_xlim_min.setValue(xmin)
                self.spin_xlim_max.setValue(xmax)
                self.spin_ylim_min.setValue(ymin)
                self.spin_ylim_max.setValue(ymax)
            else:
                self.spin_x_center.setValue((xmin + xmax) / 2)
                self.spin_x_width.setValue(xmax - xmin)
                self.spin_y_center.setValue((ymin + ymax) / 2)
                self.spin_y_height.setValue(ymax - ymin)
        elif hasattr(self.active_view, "ax"):
            xlim = self.active_view.ax.get_xlim()
            ylim = self.active_view.ax.get_ylim()
            xmin, xmax = float(min(xlim)), float(max(xlim))
            ymin, ymax = float(min(ylim)), float(max(ylim))
            if is_absolute:
                self.spin_xlim_min.setValue(xmin)
                self.spin_xlim_max.setValue(xmax)
                self.spin_ylim_min.setValue(ymin)
                self.spin_ylim_max.setValue(ymax)
            else:
                self.spin_x_center.setValue((xmin + xmax) / 2)
                self.spin_x_width.setValue(xmax - xmin)
                self.spin_y_center.setValue((ymin + ymax) / 2)
                self.spin_y_height.setValue(ymax - ymin)

    def _replace_form_row(self, label: str, new_layout: QHBoxLayout):
        """Replace a form row's layout while keeping the label."""
        # Find the form layout in grp_navigation
        form_layout = None
        for i in range(self.grp_navigation.layout().count()):
            item = self.grp_navigation.layout().itemAt(i)
            if item and item.layout() and isinstance(item.layout(), QFormLayout):
                form_layout = item.layout()
                break
        if form_layout:
            # Find the row index
            for row in range(form_layout.rowCount()):
                if form_layout.itemAt(row, QFormLayout.LabelRole):
                    lbl = form_layout.itemAt(row, QFormLayout.LabelRole).widget()
                    if lbl and lbl.text().startswith(label.replace(":", "")):
                        # Remove old field layout
                        old_field = form_layout.itemAt(row, QFormLayout.FieldRole)
                        if old_field:
                            # Remove all widgets from old layout
                            while old_field.layout().count():
                                child = old_field.layout().takeAt(0)
                                if child.widget():
                                    child.widget().setParent(None)
                            form_layout.removeRow(row)
                        # Insert new layout
                        form_layout.insertRow(row, label, new_layout)
                        break

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
        
        is_absolute = (self.combo_range_mode.currentIndex() == 0)
        
        if is_absolute:
            xmin = float(self.spin_xlim_min.value())
            xmax = float(self.spin_xlim_max.value())
            ymin = float(self.spin_ylim_min.value())
            ymax = float(self.spin_ylim_max.value())
        else:
            x_center = float(self.spin_x_center.value())
            x_width = float(self.spin_x_width.value())
            y_center = float(self.spin_y_center.value())
            y_height = float(self.spin_y_height.value())
            xmin = x_center - x_width / 2
            xmax = x_center + x_width / 2
            ymin = y_center - y_height / 2
            ymax = y_center + y_height / 2
        
        if xmin >= xmax:
            xmax = xmin + 1.0
            if is_absolute:
                self.spin_xlim_max.setValue(xmax)
            else:
                self.spin_x_width.setValue(1.0)
        if ymin >= ymax:
            ymax = ymin + 1.0
            if is_absolute:
                self.spin_ylim_max.setValue(ymax)
            else:
                self.spin_y_height.setValue(1.0)

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
        self._broadcast_shared_visual_settings(x_range=(xmin, xmax))
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
                is_absolute = (self.combo_range_mode.currentIndex() == 0)
                if is_absolute:
                    self.spin_xlim_min.setValue(xmin)
                    self.spin_xlim_max.setValue(xmax)
                    self.spin_ylim_min.setValue(ymin)
                    self.spin_ylim_max.setValue(ymax)
                else:
                    self.spin_x_center.setValue((xmin + xmax) / 2)
                    self.spin_x_width.setValue(xmax - xmin)
                    self.spin_y_center.setValue((ymin + ymax) / 2)
                    self.spin_y_height.setValue(ymax - ymin)
            finally:
                self._block_updates = False
        elif hasattr(self.active_view, "ax"):
            xlim = self.active_view.ax.get_xlim()
            ylim = self.active_view.ax.get_ylim()
            xmin, xmax = float(min(xlim)), float(max(xlim))
            ymin, ymax = float(min(ylim)), float(max(ylim))
            self._block_updates = True
            try:
                is_absolute = (self.combo_range_mode.currentIndex() == 0)
                if is_absolute:
                    self.spin_xlim_min.setValue(xmin)
                    self.spin_xlim_max.setValue(xmax)
                    self.spin_ylim_min.setValue(ymin)
                    self.spin_ylim_max.setValue(ymax)
                else:
                    self.spin_x_center.setValue((xmin + xmax) / 2)
                    self.spin_x_width.setValue(xmax - xmin)
                    self.spin_y_center.setValue((ymin + ymax) / 2)
                    self.spin_y_height.setValue(ymax - ymin)
            finally:
                self._block_updates = False
        self.settingsChanged.emit()
