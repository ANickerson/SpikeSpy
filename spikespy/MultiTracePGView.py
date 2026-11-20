import itertools
import sys
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import quantities as pq
from neo import Event

import PySide6
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, Signal, Slot, QTimer, QRectF, QPointF, QSignalBlocker
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QToolBar,
    QLabel,
    QDoubleSpinBox,
    QSpinBox,
    QCheckBox,
    QButtonGroup,
    QRadioButton,
    QPushButton,
    QToolButton,
    QMenu,
    QInputDialog,
    QComboBox,
    QDialog,
    QApplication,
    QFrame,
)
import pyqtgraph as pg
import matplotlib.pyplot as plt

from .ViewerState import ViewerState, tracked_neuron_unit
from .ChannelGroupBar import ChannelGroupBar
from .MultiTraceView import DialogSignalSelect, DialogPolySelect, DialogDeleteRegion
from .MultiTracePlotManager import MultiTracePlotManager
from .helpers import qsignal_throttle, quiet_fail, notify_view_selected
from .live_orchestrator import LiveFrameConsumerMixin
from .viewer_worker import get_task_manager, TaskType, TaskPriority
from .PGNavigation import PGNavigationMixin

logger = logging.getLogger(__name__)


def _get_mpl_lut(name: str = "gray_r", n: int = 256) -> np.ndarray:
    """Sample a Matplotlib colormap into an RGBA lookup table for PyQtGraph."""
    cmap = plt.get_cmap(name)
    colors = cmap(np.linspace(0, 1, n))
    return (colors * 255).astype(np.uint8)


# 8 qualitative colors from Matplotlib's 'Set2'
SET2_COLORS = [
    (102, 194, 165),
    (252, 141, 98),
    (141, 160, 203),
    (231, 138, 195),
    (166, 216, 84),
    (255, 217, 47),
    (229, 196, 148),
    (179, 179, 179),
]


class EphysTimeAxisItem(pg.AxisItem):
    """Custom AxisItem converting sample index to milliseconds (stimulus-locked time)."""

    def __init__(self, get_sampling_rate, orientation="bottom", *args, **kwargs):
        super().__init__(orientation, *args, **kwargs)
        self.get_sampling_rate = get_sampling_rate

    def tickStrings(self, values, scale, spacing):
        sr = self.get_sampling_rate()
        if not sr or sr <= 0:
            return [f"{v:g}" for v in values]
        return [f"{1000 * v / sr:g}" for v in values]


class ChannelPlotPanel:
    """Encapsulates a single channel's visual plot inside MultiTracePGView."""

    def __init__(
        self,
        slot_idx: int,
        channel_name: Optional[str],
        plot: pg.PlotItem,
        img_item: pg.ImageItem,
        lines_curve: pg.PlotDataItem,
        active_sweep_curve: pg.PlotDataItem,
        stim_line: pg.InfiniteLine,
        tracked_unit_scatter: pg.ScatterPlotItem,
        empty_text_item: pg.TextItem,
        time_axis: Optional[EphysTimeAxisItem] = None,
        highlight_segments: Optional[pg.PlotDataItem] = None,
        points_scatter: Optional[pg.ScatterPlotItem] = None,
    ):
        self.slot_idx: int = slot_idx
        self.channel_name: Optional[str] = channel_name
        self.plot: pg.PlotItem = plot
        self.img_item: pg.ImageItem = img_item
        self.lines_curve: pg.PlotDataItem = lines_curve
        self.active_sweep_curve: pg.PlotDataItem = active_sweep_curve
        self.highlight_segments: Optional[pg.PlotDataItem] = highlight_segments
        self.points_scatter: Optional[pg.ScatterPlotItem] = points_scatter
        self.stim_line: pg.InfiniteLine = stim_line
        self.tracked_unit_scatter: pg.ScatterPlotItem = tracked_unit_scatter
        self.other_unit_scatters: List[pg.ScatterPlotItem] = []
        self.legend_items: List[pg.TextItem] = []
        self.empty_text_item: pg.TextItem = empty_text_item
        self.time_axis: Optional[EphysTimeAxisItem] = time_axis
        self.std: float = 1.0
        self.has_data: bool = False

        # Independent X-axis linking, unit pinning, and window groups
        self.x_linked: bool = True
        self.pinned_unit_idx: Optional[int] = None
        self.window_width_ms: float = 20.0
        self.window_group: Optional[str] = None


class SummaryLinePanel:
    """Encapsulates a summary line graph panel (channel row stat or global metric)."""

    def __init__(
        self,
        key: str,
        label: str,
        plot: pg.PlotItem,
        curve: pg.PlotDataItem,
        stim_line: pg.InfiniteLine,
        empty_text_item: pg.TextItem,
        is_channel_stat: bool = False,
        slot_idx: Optional[int] = None,
        channel_name: Optional[str] = None,
        has_data: bool = False,
    ):
        self.key: str = key
        self.label: str = label
        self.plot: pg.PlotItem = plot
        self.curve: pg.PlotDataItem = curve
        self.stim_line: pg.InfiniteLine = stim_line
        self.empty_text_item: pg.TextItem = empty_text_item
        self.is_channel_stat: bool = is_channel_stat
        self.slot_idx: Optional[int] = slot_idx
        self.channel_name: Optional[str] = channel_name
        self.has_data: bool = has_data


class MultiTracePGView(LiveFrameConsumerMixin, PGNavigationMixin, QMainWindow):
    """
    High-performance PyQtGraph port of MultiTraceView.
    Delivers 60 FPS GPU-accelerated waterfall heatmaps, multi-trace visualization,
    and interactive click-to-draw polygon corridor spike tracking.
    """

    #: Emitted whenever the set of plots or their per-plot options change.
    plotsChanged = Signal()

    right_ax_data: Dict = {}

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        state: Optional[ViewerState] = None,
    ):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setFocusPolicy(Qt.ClickFocus)

        self.state: Optional[ViewerState] = None
        self.mode: str = "heatmap"
        self.lock_to_stim: bool = False
        self.fixed_height: bool = False
        self.y_axis_mode: str = "sweep"
        self.n_lines_pre: int = 5
        self.n_lines_post: int = 5
        self.time_window_s: float = 10.0
        self.post_seconds: float = 0.0
        self.scale: float = 1.0
        self.scale_mode: str = "std"
        self.link_x_axis: bool = True
        self.std: float = 1.0
        self.rightPlots: Dict[str, bool] = {}
        self.right_plots: Dict[str, pg.PlotItem] = {}
        self.summary_panels: Dict[str, SummaryLinePanel] = {}
        self._last_summary_keys: Optional[Tuple[str, ...]] = None
        self.connections: List[Tuple] = []
        self.references = self.connections
        self.legend_items: List[pg.TextItem] = []
        self.message_items: List[pg.TextItem] = []
        self.other_unit_scatters: List[pg.ScatterPlotItem] = []

        # Channel group display linkage
        self.group_size_n: int = 1
        self.displayed_channels: List[Optional[str]] = []
        self.channel_panels: List[ChannelPlotPanel] = []
        self.selector_channel_name: Optional[str] = None
        self.selector_panel: Optional[ChannelPlotPanel] = None

        # Color LUT for heatmap
        self._cmap_lut = _get_mpl_lut("gray_r")

        # Debounce timer for spinner changes
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.updateAll)
        self.update_timer.setInterval(50)

        # Selector state
        self.selector_active: bool = False
        self.selector_points: List[List[float]] = []
        self.selector_width: float = 100.0  # samples

        self._dirty: bool = False
        self._user_hidden: bool = False
        self._was_shown: bool = False
        self._has_initialized_ranges: bool = False
        self._last_setup_mode: Optional[str] = None
        self._applying_shared_visuals: bool = False
        self.protocol_marker_items: List[Any] = []

        # Background task manager
        self._task_manager = get_task_manager()
        self._viewer_id = f"multitrace_pg_{id(self)}"

        # Build UI layout
        self._init_ui()

        # Connect dialogs
        self.dialogPolySelect = DialogPolySelect(self)
        self.dialogPolySelect.onSubmit.connect(self.polySelect)
        self.dialogPolySelect.finished.connect(lambda *args: self.toggle_polySelector(False))
        self.dialogPolySelect.changeSelection.connect(self._on_selector_window_change)

        self.dialogDeleteRegion = DialogDeleteRegion(self)
        self.dialogDeleteRegion.onSubmit.connect(self.deleteRegion)
        self.dialogDeleteRegion.finished.connect(lambda *args: self.toggle_deleter(False))

        self.settingsDialog: Optional[DialogSignalSelect] = None

        if state is not None:
            self.set_state(state)

        self.setup_figure()

    def _init_ui(self):
        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        self.live_paused_banner = QFrame(central)
        self.live_paused_banner.setObjectName("live_paused_banner")
        self.live_paused_banner.setStyleSheet(
            "QFrame#live_paused_banner { background: #fff3cd; border: 1px solid #ffe69c; border-radius: 4px; padding: 2px 6px; }"
        )
        banner_layout = QHBoxLayout(self.live_paused_banner)
        banner_layout.setContentsMargins(6, 2, 6, 2)
        self.lbl_live_paused = QLabel("⏸ Viewing History —", self.live_paused_banner)
        self.btn_return_live = QPushButton("📡 Return to Live (Ctrl+L)", self.live_paused_banner)
        self.btn_return_live.clicked.connect(self._on_return_to_live_clicked)
        banner_layout.addWidget(self.lbl_live_paused)
        banner_layout.addStretch()
        banner_layout.addWidget(self.btn_return_live)
        self.live_paused_banner.setVisible(False)
        layout.addWidget(self.live_paused_banner)

        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground("w")  # Clean white background matching Matplotlib

        # Interactive Selector Graphics (Corridor polygon + Polyline with handles)
        self.corridor_item = QtWidgets.QGraphicsPolygonItem()
        self.corridor_item.setPen(pg.mkPen("purple", width=1, style=Qt.DashLine))
        self.corridor_item.setBrush(pg.mkBrush(160, 32, 240, 50))
        self.corridor_item.setZValue(20)
        self.corridor_item.setVisible(False)

        self.selector_line_item = pg.PlotDataItem(
            pen=pg.mkPen("purple", width=2),
            symbol="o",
            symbolSize=7,
            symbolBrush=pg.mkBrush("purple"),
            symbolPen=pg.mkPen("w", width=1),
        )
        self.selector_line_item.setZValue(21)
        self.selector_line_item.setVisible(False)

        # Handle mouse clicks and movement across the graphics layout
        self.glw.scene().sigMouseClicked.connect(self._on_plot_clicked)
        self.glw.scene().sigMouseMoved.connect(self._on_plot_mouse_moved)

        # Handle view range changes to dynamically render visible area
        self._range_timer = QTimer(self)
        self._range_timer.setSingleShot(True)
        self._range_timer.setInterval(60)
        self._range_timer.timeout.connect(self._on_range_timer_timeout)

        # Build initial channel panels
        self._rebuild_channel_panels()

        layout.addWidget(self.glw)
        self.setCentralWidget(central)

        # Build Toolbars
        self._build_toolbars()

    def _build_toolbars(self):
        toolbar = QToolBar("Controls", self)

        # Scale & Contrast Spinners
        self.scale_spinner = QDoubleSpinBox(self)
        self.scale_spinner.setDecimals(4)
        self.scale_spinner.setRange(0.0001, 1000.0)
        self.scale_spinner.setSingleStep(0.05)
        self.scale_spinner.setValue(self.scale)
        self.scale_spinner.setMinimumWidth(80)
        self.scale_spinner.valueChanged.connect(self._on_scale_changed)

        self.lowerSpinBox = QDoubleSpinBox(self)
        self.lowerSpinBox.setDecimals(2)
        self.lowerSpinBox.setSingleStep(0.1)
        self.lowerSpinBox.setRange(0, 10)
        self.lowerSpinBox.setValue(0.5)
        self.lowerSpinBox.setMinimumWidth(80)
        self.lowerSpinBox.valueChanged.connect(lambda x: self.update_timer.start())

        self.upperSpinBox = QDoubleSpinBox(self)
        self.upperSpinBox.setDecimals(2)
        self.upperSpinBox.setSingleStep(0.1)
        self.upperSpinBox.setRange(0, 10)
        self.upperSpinBox.setValue(2.0)
        self.upperSpinBox.setMinimumWidth(80)
        self.upperSpinBox.valueChanged.connect(lambda x: self.update_timer.start())

        self.lowerSpinBox.valueChanged.connect(self.upperSpinBox.setMinimum)
        self.upperSpinBox.valueChanged.connect(self.lowerSpinBox.setMaximum)

        toolbar.addWidget(QLabel("Scale: "))
        self.scale_mode_combo = QComboBox(self)
        self.scale_mode_combo.addItems(["std", "mV"])
        self.scale_mode_combo.setCurrentText(getattr(self, "scale_mode", "std"))
        self.scale_mode_combo.currentTextChanged.connect(self._on_scale_mode_changed)
        toolbar.addWidget(self.scale_mode_combo)
        toolbar.addWidget(self.scale_spinner)
        self.contrast_min_label = QLabel("Min")
        self.contrast_max_label = QLabel("Max")
        toolbar.addWidget(self.contrast_min_label)
        toolbar.addWidget(self.lowerSpinBox)
        toolbar.addWidget(self.contrast_max_label)
        toolbar.addWidget(self.upperSpinBox)
        toolbar.addSeparator()

        # Unit Checkboxes
        self.includeUnitCheckbox = QCheckBox("Tracked")
        self.includeUnitCheckbox.setChecked(True)
        self.includeUnitCheckbox.stateChanged.connect(lambda x: self.render())

        self.includeAllUnitsCheckBox = QCheckBox("Others")
        self.includeAllUnitsCheckBox.stateChanged.connect(lambda x: self.render())

        toolbar.addWidget(QLabel("Units:"))
        toolbar.addWidget(self.includeUnitCheckbox)
        toolbar.addWidget(self.includeAllUnitsCheckBox)
        toolbar.addSeparator()

        # Mode Radio Buttons & Fixed Height & Y-Axis Mode
        self.lock_to_stimCheckBox = QCheckBox("lock")
        self.lock_to_stimCheckBox.stateChanged.connect(self._on_lock_changed)

        self.fixed_height_checkbox = QCheckBox("Fixed Height")
        self.fixed_height_checkbox.setChecked(self.fixed_height)
        self.fixed_height_checkbox.stateChanged.connect(self._on_fixed_height_changed)

        self.y_axis_combo = QComboBox(self)
        self.y_axis_combo.addItems(["Sweep", "Time"])
        self.y_axis_combo.setCurrentText("Time" if self.y_axis_mode == "time" else "Sweep")
        self.y_axis_combo.currentTextChanged.connect(self._on_y_axis_mode_changed)

        self.pre_label = QLabel("Pre:")
        self.pre_spinner = QSpinBox(self)
        self.pre_spinner.setRange(0, 500)
        self.pre_spinner.setValue(self.n_lines_pre)
        self.pre_spinner.valueChanged.connect(self._on_pre_changed)

        self.post_label = QLabel("Post:")
        self.post_spinner = QSpinBox(self)
        self.post_spinner.setRange(0, 500)
        self.post_spinner.setValue(self.n_lines_post)
        self.post_spinner.valueChanged.connect(self._on_post_changed)

        self.time_window_label = QLabel("Window:")
        self.time_window_spinner = QDoubleSpinBox(self)
        self.time_window_spinner.setRange(0.1, 3600.0)
        self.time_window_spinner.setValue(self.time_window_s)
        self.time_window_spinner.setSuffix(" s")
        self.time_window_spinner.valueChanged.connect(self._on_time_window_changed)

        linesRadio = QRadioButton("lines")
        heatmapRadio = QRadioButton("heatmap")
        unitOnlyRadio = QRadioButton("unit only")
        heatmapRadio.setChecked(True)

        self.butgrp = QButtonGroup(self)
        self.butgrp.addButton(linesRadio)
        self.butgrp.addButton(heatmapRadio)
        self.butgrp.addButton(unitOnlyRadio)

        def on_mode_toggle(btn, checked):
            if not checked:
                return
            if heatmapRadio.isChecked():
                self.mode = "heatmap"
            elif linesRadio.isChecked():
                self.mode = "lines"
            elif unitOnlyRadio.isChecked():
                self.mode = "unitonly"
            self._update_toolbar_visibility()
            self.setup_figure()

        self.butgrp.buttonToggled.connect(on_mode_toggle)

        toolbar.addWidget(QLabel("Mode:"))
        toolbar.addWidget(self.lock_to_stimCheckBox)
        toolbar.addWidget(self.fixed_height_checkbox)
        toolbar.addWidget(self.y_axis_combo)
        toolbar.addWidget(self.pre_label)
        toolbar.addWidget(self.pre_spinner)
        toolbar.addWidget(self.post_label)
        toolbar.addWidget(self.post_spinner)
        toolbar.addWidget(self.time_window_label)
        toolbar.addWidget(self.time_window_spinner)
        toolbar.addWidget(linesRadio)
        toolbar.addWidget(heatmapRadio)
        toolbar.addWidget(unitOnlyRadio)
        toolbar.addSeparator()

        # Channel Display Controls (Add, Remove, Group Size N, Presets, Link X)
        toolbar.addWidget(QLabel("Channels: "))

        self.btn_add_channel = QToolButton(self)
        self.btn_add_channel.setText("+ Channel ▾")
        self.btn_add_channel.setToolTip("Add channel or empty screen slot to group display")
        self.btn_add_channel.setPopupMode(QToolButton.InstantPopup)
        toolbar.addWidget(self.btn_add_channel)

        self.btn_remove_channel = QToolButton(self)
        self.btn_remove_channel.setText("- Channel ▾")
        self.btn_remove_channel.setToolTip("Remove channel or slot from group display")
        self.btn_remove_channel.setPopupMode(QToolButton.InstantPopup)
        toolbar.addWidget(self.btn_remove_channel)

        toolbar.addWidget(QLabel("Group Size (N):"))
        self.groupSizeSpinBox = QSpinBox(self)
        self.groupSizeSpinBox.setRange(1, 32)
        self.groupSizeSpinBox.setValue(self.group_size_n)
        self.groupSizeSpinBox.setToolTip(
            "Target number of channel screens in group (N).\n"
            "Retains this count when switching groups or channels, keeping empty screens for missing channels."
        )
        self.groupSizeSpinBox.valueChanged.connect(self.set_group_size_n)
        toolbar.addWidget(self.groupSizeSpinBox)

        self.btn_presets = QToolButton(self)
        self.btn_presets.setText("Presets ▾")
        self.btn_presets.setToolTip("Quick group size presets (1-Ch, 2-Ch, 4-Ch Tetrode, 8-Ch)")
        self.btn_presets.setPopupMode(QToolButton.InstantPopup)
        toolbar.addWidget(self.btn_presets)

        self.link_x_checkbox = QCheckBox("Link X")
        self.link_x_checkbox.setChecked(self.link_x_axis)
        self.link_x_checkbox.setToolTip("Link horizontal time/sample zooming across channels")
        self.link_x_checkbox.stateChanged.connect(self._on_link_x_changed)
        toolbar.addWidget(self.link_x_checkbox)

        self._build_channel_menus()
        toolbar.addSeparator()

        # Action Buttons
        self.polySelectButton = QPushButton("Selector")
        self.polySelectButton.setCheckable(True)
        self.polySelectButton.clicked.connect(lambda *args: self.toggle_polySelector())
        toolbar.addWidget(self.polySelectButton)

        self.deleterButton = QPushButton("Deleter")
        self.deleterButton.clicked.connect(lambda *args: self.toggle_deleter())
        toolbar.addWidget(self.deleterButton)

        self.settingsButton = QPushButton("Settings")
        self.settingsButton.clicked.connect(self.open_settings_dialog)
        toolbar.addWidget(self.settingsButton)

        self.plotsButton = QPushButton("Plots...")
        self.plotsButton.setToolTip(
            "Add, remove, reorder and configure each channel or summary plot"
        )
        self.plotsButton.clicked.connect(lambda *_: self.open_plot_manager())
        toolbar.addWidget(self.plotsButton)

        self.toggleMaxMeanButton = QPushButton("Show max", self)
        self.toggleMaxMeanButton.setCheckable(True)
        self.toggleMaxMeanButton.clicked.connect(self.toggleMaxMean)
        toolbar.addWidget(self.toggleMaxMeanButton)

        self.btn_summary_menu = QToolButton(self)
        self.btn_summary_menu.setText("Summary ▾")
        self.btn_summary_menu.setToolTip("Quick toggles for summary line graphs (Channel Mean/Max, Stim. Freq., Latency Diff.)")
        self.btn_summary_menu.setPopupMode(QToolButton.InstantPopup)
        toolbar.addWidget(self.btn_summary_menu)
        self._build_summary_menu()

        self.showMessages = QPushButton("Show messages", self)
        self.showMessages.setCheckable(True)
        self.showMessages.clicked.connect(self.toggleShowMessages)
        toolbar.addWidget(self.showMessages)

        self.showLegend = QPushButton("Show legend", self)
        self.showLegend.setCheckable(True)
        self.showLegend.clicked.connect(lambda x: self.render())
        toolbar.addWidget(self.showLegend)

        # Quick View-Reset button
        resetViewBtn = QPushButton("Reset View")
        resetViewBtn.clicked.connect(self.update_axis)
        toolbar.addWidget(resetViewBtn)

        # Matplotlib-like navigation actions
        self.add_navigation_actions(toolbar)
        self.init_navigation()

        # Independent channel group and unit binding toolbar (hidden by default; controlled via ViewSettingsPanel or container)
        self.channel_bar = ChannelGroupBar(parent=self, state=self.state)
        self.channel_bar.bindingChanged.connect(self._on_channel_bar_changed)
        self.addToolBar(Qt.TopToolBarArea, self.channel_bar)
        self.channel_bar.setVisible(False)
        self.addToolBarBreak()
        self.addToolBar(Qt.TopToolBarArea, toolbar)
        self._update_toolbar_visibility()

    def _on_scale_changed(self, val: float):
        self.scale = float(val)
        if self.mode == "lines":
            self.setup_figure()

    def _on_fixed_height_changed(self, state):
        self.fixed_height = bool(state)
        self._update_toolbar_visibility()
        if self.state and self.state.stimno is not None:
            self.update_ylim(self.state.stimno)
        self.setup_figure()

    def _on_y_axis_mode_changed(self, mode_text: str):
        self.y_axis_mode = mode_text.lower()
        self._update_toolbar_visibility()
        if self.state and self.state.stimno is not None:
            self.update_ylim(self.state.stimno)
        self.setup_figure()

    def _on_pre_changed(self, val: int):
        self.n_lines_pre = int(val)
        if self.fixed_height and self.state and self.state.stimno is not None:
            self.update_ylim(self.state.stimno)
            self.setup_figure()

    def _on_post_changed(self, val: int):
        self.n_lines_post = int(val)
        if self.fixed_height and self.state and self.state.stimno is not None:
            self.update_ylim(self.state.stimno)
            self.setup_figure()

    def _on_time_window_changed(self, val: float):
        self.time_window_s = float(val)
        if self.fixed_height and self.state and self.state.stimno is not None:
            self.update_ylim(self.state.stimno)
            self.setup_figure()

    def _on_link_x_changed(self, state):
        self.link_x_axis = bool(state)
        self._update_axis_linking()

    def _update_toolbar_visibility(self):
        is_lines = self.mode == "lines"
        is_fixed = getattr(self, "fixed_height", False)
        is_time = getattr(self, "y_axis_mode", "sweep") == "time"

        if hasattr(self, "scale_spinner"):
            self.scale_spinner.setVisible(is_lines)
        if hasattr(self, "contrast_min_label"):
            self.contrast_min_label.setVisible(not is_lines)
            self.contrast_max_label.setVisible(not is_lines)
            self.lowerSpinBox.setVisible(not is_lines)
            self.upperSpinBox.setVisible(not is_lines)

        if hasattr(self, "pre_label"):
            self.pre_label.setVisible(is_fixed and not is_time)
            self.pre_spinner.setVisible(is_fixed and not is_time)
            self.post_label.setVisible(is_fixed and not is_time)
            self.post_spinner.setVisible(is_fixed and not is_time)

        if hasattr(self, "time_window_label"):
            self.time_window_label.setVisible(is_fixed and is_time)
            self.time_window_spinner.setVisible(is_fixed and is_time)

    def _build_channel_menus(self):
        # + Channel menu
        menu_add = QMenu(self)
        menu_add.aboutToShow.connect(self._populate_add_channel_menu)
        self.btn_add_channel.setMenu(menu_add)

        # - Channel menu
        menu_remove = QMenu(self)
        menu_remove.aboutToShow.connect(self._populate_remove_channel_menu)
        self.btn_remove_channel.setMenu(menu_remove)

        # Presets menu
        menu_presets = QMenu(self)
        menu_presets.addAction("1-Ch (Single)", lambda: self.set_group_size_n(1))
        menu_presets.addAction("2-Ch (Stereotrode)", lambda: self.set_group_size_n(2))
        menu_presets.addAction("4-Ch (Tetrode)", lambda: self.set_group_size_n(4))
        menu_presets.addAction("8-Ch (Octrode)", lambda: self.set_group_size_n(8))
        self.btn_presets.setMenu(menu_presets)

    def _populate_add_channel_menu(self):
        menu = self.btn_add_channel.menu()
        if menu is None:
            return
        menu.clear()

        avail = self.state.get_available_channels() if self.state else []
        displayed_set = set(c for c in self.displayed_channels if c)

        unadded = [c for c in avail if c not in displayed_set]
        if unadded:
            sec_header = menu.addAction("Add Recording Channel:")
            sec_header.setEnabled(False)
            for ch in unadded:
                act = menu.addAction(f"+ {ch}")
                act.triggered.connect(lambda checked=False, c=ch: self.add_channel_to_display(c))
        else:
            act_none = menu.addAction("All available channels added")
            act_none.setEnabled(False)

        menu.addSeparator()
        act_empty = menu.addAction("+ Add Empty Screen (Slot)")
        act_empty.triggered.connect(self.add_empty_slot)

        act_custom = menu.addAction("Custom Channel Name...")
        act_custom.triggered.connect(self._prompt_custom_channel)

    def _populate_remove_channel_menu(self):
        menu = self.btn_remove_channel.menu()
        if menu is None:
            return
        menu.clear()

        displayed_valid = [c for c in self.displayed_channels if c]
        if displayed_valid:
            sec_header = menu.addAction("Remove Channel from Display:")
            sec_header.setEnabled(False)
            for ch in displayed_valid:
                act = menu.addAction(f"✕ {ch}")
                act.triggered.connect(
                    lambda checked=False, c=ch: self.remove_channel_from_display(c, keep_empty_slot=False)
                )

            menu.addSeparator()
            sub_clear = menu.addMenu("Clear to Empty Screen (Keep Slot)")
            for ch in displayed_valid:
                act_c = sub_clear.addAction(f"Clear {ch}")
                act_c.triggered.connect(
                    lambda checked=False, c=ch: self.remove_channel_from_display(c, keep_empty_slot=True)
                )
        else:
            act_none = menu.addAction("No active channels to remove")
            act_none.setEnabled(False)

        menu.addSeparator()
        if any(c is None for c in self.displayed_channels):
            act_trim = menu.addAction("Remove Empty Screens")
            act_trim.triggered.connect(self._remove_empty_screens)

    def _show_channel_context_menu(self, panel: ChannelPlotPanel, screen_pos: QtCore.QPoint):
        menu = QMenu(self)
        ch_name = panel.channel_name
        slot_label = f"Slot {panel.slot_idx + 1} of {len(self.channel_panels)}"
        header = menu.addAction(f"Channel: {ch_name or '(Empty)'} ({slot_label})")
        header.setEnabled(False)
        menu.addSeparator()

        act_edit = menu.addAction("Configure This Plot...")
        act_edit.triggered.connect(
            lambda: self.open_plot_manager(f"channel:{panel.slot_idx}")
        )
        menu.addSeparator()

        if ch_name:
            primary_ch = self.channel_bar.get_primary_channel() if hasattr(self, "channel_bar") and self.channel_bar else ""
            if ch_name != primary_ch:
                act_prim = menu.addAction("★ Set as Primary Channel")
                act_prim.triggered.connect(lambda: self._set_channel_as_primary(panel.slot_idx))

            act_clear = menu.addAction("Clear to Empty Screen")
            act_clear.triggered.connect(lambda: self.clear_slot_to_empty(panel.slot_idx))

            act_rem = menu.addAction("✕ Remove Channel & Screen")
            act_rem.triggered.connect(lambda: self.remove_channel_from_display(ch_name, keep_empty_slot=False))

        avail = self.state.get_available_channels() if self.state else []
        replace_menu = menu.addMenu("Set Channel..." if not ch_name else "Replace Channel with...")
        for ch in avail:
            if ch != ch_name:
                act_ch = replace_menu.addAction(ch)
                act_ch.triggered.connect(
                    lambda checked=False, c=ch, idx=panel.slot_idx: self._replace_slot_channel(idx, c)
                )

        menu.addSeparator()
        act_add = menu.addAction("+ Add Channel Screen...")
        act_add.triggered.connect(self.add_empty_slot)

        # X-Axis Linking & Pinning Submenu
        menu.addSeparator()
        x_menu = menu.addMenu("X-Axis Linking & Pinning")

        act_link = x_menu.addAction("Link X with Ch 1")
        act_link.setCheckable(True)
        act_link.setChecked(getattr(panel, "x_linked", True) and getattr(panel, "pinned_unit_idx", None) is None)
        def toggle_x_link(checked):
            panel.x_linked = checked
            if checked:
                panel.pinned_unit_idx = None
            self._update_axis_linking()
            self._update_channel_titles()
            self.setup_figure()
        act_link.triggered.connect(toggle_x_link)

        pin_menu = x_menu.addMenu("Pin X to Unit Latency...")
        act_unpin = pin_menu.addAction("Unpinned (Free)")
        act_unpin.setCheckable(True)
        act_unpin.setChecked(getattr(panel, "pinned_unit_idx", None) is None)
        act_unpin.triggered.connect(lambda: self._pin_channel_to_unit(panel.slot_idx, None))

        if self.state and hasattr(self.state, "spike_groups"):
            sr = float(self.state.sampling_rate) if self.state.sampling_rate else 10000.0
            for u_i, sg in enumerate(self.state.spike_groups):
                u_name = getattr(sg.event, "name", str(u_i)) or str(u_i)
                approx_lat = ""
                if getattr(sg, "idx_arr", None) is not None:
                    valid_samples = [float(a[0]) for a in sg.idx_arr if a is not None]
                    if valid_samples:
                        approx_lat = f" (~{np.median(valid_samples) / sr * 1000.0:.1f} ms)"
                act_u = pin_menu.addAction(f"Unit {u_i}: {u_name}{approx_lat}")
                act_u.setCheckable(True)
                act_u.setChecked(getattr(panel, "pinned_unit_idx", None) == u_i)
                act_u.triggered.connect(lambda chk=False, s_idx=panel.slot_idx, u_idx=u_i: self._pin_channel_to_unit(s_idx, u_idx))

        act_width = pin_menu.addAction(f"Window Width: {panel.window_width_ms:.1f} ms...")
        def set_window_w():
            val, ok = QInputDialog.getDouble(self, "Window Width", "Window width around latency (ms):", panel.window_width_ms, 0.5, 1000.0, 1)
            if ok:
                panel.window_width_ms = float(val)
                self._apply_pinned_unit_windows()
        act_width.triggered.connect(set_window_w)

        grp_menu = x_menu.addMenu("Window Group...")
        for g_name in [None, "Group A", "Group B", "Group C"]:
            label = "None (Independent)" if g_name is None else g_name
            act_g = grp_menu.addAction(label)
            act_g.setCheckable(True)
            act_g.setChecked(getattr(panel, "window_group", None) == g_name)
            act_g.triggered.connect(lambda chk=False, s_idx=panel.slot_idx, g=g_name: self._set_channel_window_group(s_idx, g))

        if hasattr(self, "channel_bar") and self.channel_bar is not None:
            menu.addSeparator()
            act_prev = menu.addAction("Shift Group -1 (<)")
            act_prev.triggered.connect(lambda: self.channel_bar.shift_group(-1))
            act_next = menu.addAction("Shift Group +1 (>)")
            act_next.triggered.connect(lambda: self.channel_bar.shift_group(1))

        if self.state is not None:
            menu.addSeparator()
            act_seed = menu.addAction("🎯 Seed & Auto-Track New Unit at Current Sweep")
            act_seed.triggered.connect(
                lambda checked=False, p=panel: self._seed_unit_at_current_sweep(p)
            )
            act_reanchor = menu.addAction("📍 Re-Anchor Active Unit at Current Sweep")
            act_reanchor.triggered.connect(
                lambda checked=False, p=panel: self._reanchor_active_unit_at_current_sweep(p)
            )

        menu.exec(screen_pos)

    def _build_summary_menu(self):
        menu = QMenu(self)
        menu.aboutToShow.connect(self._populate_summary_menu)
        self.btn_summary_menu.setMenu(menu)

    def _populate_summary_menu(self):
        menu = self.btn_summary_menu.menu()
        if menu is None:
            return
        menu.clear()

        act_ch_all = menu.addAction("Channel Summary Graphs (All in Group)")
        act_ch_all.setCheckable(True)
        act_ch_all.setChecked(self.rightPlots.get("__channel_summaries__", True))
        act_ch_all.triggered.connect(lambda checked: self.toggle_channel_summaries(checked))

        act_sf = menu.addAction("Stim. Freq. (Hz)")
        act_sf.setCheckable(True)
        act_sf.setChecked(self.rightPlots.get("Stim. Freq. (Hz)", True))
        act_sf.triggered.connect(lambda checked: self.toggle_summary_graph("Stim. Freq. (Hz)", checked))

        act_ld = menu.addAction("Latency Diff.")
        act_ld.setCheckable(True)
        act_ld.setChecked(self.rightPlots.get("Latency Diff.", True))
        act_ld.triggered.connect(lambda checked: self.toggle_summary_graph("Latency Diff.", checked))

        menu.addSeparator()

        stat_lbl = "Switch to Mean" if self.toggleMaxMeanButton.isChecked() else "Switch to Max"
        act_stat = menu.addAction(stat_lbl)
        act_stat.triggered.connect(self.toggleMaxMean)

        menu.addSeparator()
        act_cfg = menu.addAction("Configure All Summary Signals...")
        act_cfg.triggered.connect(self.open_settings_dialog)

        act_manage = menu.addAction("Manage Plots...")
        act_manage.triggered.connect(lambda *_: self.open_plot_manager())

    def toggle_channel_summaries(self, visible: bool):
        """Toggle all channel summary graphs on or off."""
        self.rightPlots["__channel_summaries__"] = visible
        self.plot_right_axis()
        self.plotsChanged.emit()

    def toggle_summary_graph(self, key: str, visible: bool):
        """Toggle visibility of a specific summary line graph."""
        self.rightPlots[key] = visible
        self.plot_right_axis()
        self.plotsChanged.emit()

    def open_settings_dialog(self):
        """Open settings dialog to configure which summary line graphs are displayed."""
        all_options = dict(self.rightPlots)
        all_options.setdefault("__channel_summaries__", True)
        all_options.setdefault("Stim. Freq. (Hz)", True)
        all_options.setdefault("Latency Diff.", True)

        for ch in self.displayed_channels:
            if ch and ch not in all_options:
                all_options[ch] = True

        if self.right_ax_data:
            for k in self.right_ax_data.keys():
                if k not in all_options:
                    all_options[k] = True

        if self.settingsDialog is not None:
            try:
                self.settingsDialog.close()
                self.settingsDialog.deleteLater()
            except Exception:
                pass

        self.settingsDialog = DialogSignalSelect(self, options=all_options)
        self.settingsDialog.setWindowTitle("Configure Summary Line Graphs")

        def updateView(k, v):
            self.rightPlots[k] = v
            self.plot_right_axis()
            self.render()

        self.settingsDialog.changeSelection.connect(updateView)
        self.settingsDialog.show()

    def open_plot_manager(self, plot_key: Optional[str] = None):
        """Open the per-plot add/remove/configure dialog for this view."""
        dialog = getattr(self, "plot_manager_dialog", None)
        try:
            if dialog is not None and dialog.isVisible():
                if plot_key is not None:
                    manager = getattr(self, "plot_manager", None)
                    if manager is not None:
                        manager.select_plot(plot_key)
                dialog.raise_()
                dialog.activateWindow()
                return
        except RuntimeError:
            dialog = None

        dialog = QDialog(self)
        dialog.setWindowTitle("MultiTrace Plots")
        dialog.resize(420, 520)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(6, 6, 6, 6)
        manager = MultiTracePlotManager(dialog, view=self)
        layout.addWidget(manager, 1)
        close_btn = QPushButton("Close", dialog)
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)

        self.plot_manager_dialog = dialog
        self.plot_manager = manager
        dialog.finished.connect(self._on_plot_manager_closed)
        dialog.show()
        if plot_key is not None:
            manager.select_plot(plot_key)

    def _on_plot_manager_closed(self, *args):
        manager = getattr(self, "plot_manager", None)
        if manager is not None:
            manager.set_view(None)
        self.plot_manager = None
        dialog = getattr(self, "plot_manager_dialog", None)
        self.plot_manager_dialog = None
        if dialog is not None:
            dialog.deleteLater()

    def _show_summary_context_menu(self, screen_pos, key: str, panel: SummaryLinePanel):
        menu = QMenu(self)

        act_edit = menu.addAction("Configure This Plot...")
        act_edit.triggered.connect(lambda: self.open_plot_manager(key))
        menu.addSeparator()

        if panel.is_channel_stat:
            stat_act = menu.addAction(
                "Switch to Mean" if self.toggleMaxMeanButton.isChecked() else "Switch to Max"
            )
            stat_act.triggered.connect(self.toggleMaxMean)
            menu.addSeparator()

            if panel.slot_idx is not None:
                if panel.channel_name:
                    primary_ch = (
                        self.channel_bar.get_primary_channel()
                        if hasattr(self, "channel_bar") and self.channel_bar
                        else ""
                    )
                    if panel.channel_name != primary_ch:
                        act_pri = menu.addAction(f"★ Set '{panel.channel_name}' as Primary")
                        act_pri.triggered.connect(lambda: self._set_channel_as_primary(panel.slot_idx))

                replace_menu = menu.addMenu("Set Channel..." if not panel.channel_name else "Replace Channel with...")
                avail = self.state.get_available_channels() if self.state else []
                for ch in avail:
                    if ch != panel.channel_name:
                        act_ch = replace_menu.addAction(ch)
                        act_ch.triggered.connect(
                            lambda checked=False, c=ch, idx=panel.slot_idx: self._replace_slot_channel(idx, c)
                        )

                if panel.channel_name is not None:
                    act_clr = menu.addAction("Clear Slot to (Empty Screen)")
                    act_clr.triggered.connect(lambda: self.clear_slot_to_empty(panel.slot_idx))

            menu.addSeparator()

        act_hide = menu.addAction(f"Hide '{panel.label}'")
        act_hide.triggered.connect(lambda: self.toggle_summary_graph(key, False))

        menu.exec(screen_pos)

    def _format_summary_title(
        self, slot_idx: int, total_n: int, ch_name: Optional[str], primary_ch: str = "", stat_name: str = "Mean"
    ) -> str:
        """Format informative label linking summary line graph to its channel group position."""
        star = "★ " if ch_name and ch_name == primary_ch else ""
        slot_label = f"{slot_idx + 1}/{total_n} in group"
        if not ch_name:
            return f"[{slot_label}] (Empty)"

        has_data = False
        if self.state is not None:
            erp = self.get_channel_erp_signal(ch_name)
            has_data = erp is not None and len(erp) > 0

        status = f"({stat_name})" if has_data else "(No data)"
        return f"{star}[{slot_label}] {ch_name} {status}"

    def _pin_channel_to_unit(self, slot_idx: int, unit_idx: Optional[int]):
        if 0 <= slot_idx < len(self.channel_panels):
            p = self.channel_panels[slot_idx]
            p.pinned_unit_idx = unit_idx
            if unit_idx is not None:
                p.x_linked = False
            else:
                p.x_linked = True
            self._update_axis_linking()
            self._apply_pinned_unit_windows()
            self._update_channel_titles()
            self.plotsChanged.emit()

    def _set_channel_window_group(self, slot_idx: int, group_name: Optional[str]):
        if 0 <= slot_idx < len(self.channel_panels):
            p = self.channel_panels[slot_idx]
            p.window_group = group_name
            self._update_axis_linking()
            self._update_channel_titles()
            self.plotsChanged.emit()

    def _update_channel_titles(self):
        primary_ch = self.channel_bar.get_primary_channel() if hasattr(self, "channel_bar") and self.channel_bar else ""
        for i, p in enumerate(self.channel_panels):
            title = self._format_channel_title(i, len(self.channel_panels), p.channel_name, primary_ch, panel=p)
            p.plot.setTitle(title, color="#333", size="9pt")

    def _update_axis_linking(self):
        if not self.channel_panels:
            return
        leader_0 = self.channel_panels[0].plot

        group_leaders: Dict[str, pg.PlotItem] = {}
        for p in self.channel_panels:
            if p.window_group:
                if p.window_group not in group_leaders:
                    group_leaders[p.window_group] = p.plot

        for i, panel in enumerate(self.channel_panels):
            if i > 0:
                panel.plot.setYLink(leader_0)

            if not getattr(self, "link_x_axis", True) or not getattr(panel, "x_linked", True) or getattr(panel, "pinned_unit_idx", None) is not None:
                panel.plot.setXLink(None)
            elif panel.window_group:
                leader = group_leaders.get(panel.window_group)
                if leader is not None and leader is not panel.plot:
                    panel.plot.setXLink(leader)
                else:
                    panel.plot.setXLink(None)
            else:
                if i > 0:
                    panel.plot.setXLink(leader_0)
                else:
                    panel.plot.setXLink(None)

    def _apply_pinned_unit_windows(self):
        if self.state is None or not hasattr(self.state, "spike_groups"):
            return
        sr = float(self.state.sampling_rate) if self.state.sampling_rate else 10000.0
        cur_stim = self.state.stimno if self.state.stimno is not None else 0

        for panel in self.channel_panels:
            u_idx = getattr(panel, "pinned_unit_idx", None)
            if u_idx is None or not (0 <= u_idx < len(self.state.spike_groups)):
                continue
            sg = self.state.spike_groups[u_idx]

            center_sample = None
            if getattr(sg, "idx_arr", None) is not None and 0 <= cur_stim < len(sg.idx_arr):
                entry = sg.idx_arr[cur_stim]
                if entry is not None:
                    center_sample = float(entry[0])

            if center_sample is None and getattr(sg, "idx_arr", None) is not None:
                valid_samples = [float(a[0]) for a in sg.idx_arr if a is not None]
                if valid_samples:
                    center_sample = float(np.median(valid_samples))

            if center_sample is None and hasattr(sg, "get_latencies") and self.state.event_signal is not None:
                lats = sg.get_latencies(self.state.event_signal)
                valid_lats = [float(l.rescale("s").magnitude) * sr for l in lats if not np.isnan(l)]
                if valid_lats:
                    center_sample = float(np.median(valid_lats))

            if center_sample is not None:
                half_w_samples = (panel.window_width_ms / 1000.0) * sr / 2.0
                panel.plot.setXRange(center_sample - half_w_samples, center_sample + half_w_samples, padding=0)

    def _get_stim_times_s(self) -> Optional[np.ndarray]:
        if self.state is None or self.state.event_signal is None or len(self.state.event_signal) == 0:
            return None
        try:
            ev = self.state.event_signal
            if hasattr(ev, "times"):
                times = ev.times
                if hasattr(times, "rescale"):
                    return np.asarray(times.rescale(pq.s).magnitude, dtype=float)
                return np.asarray(times, dtype=float)
            elif hasattr(ev, "rescale"):
                return np.asarray(ev.rescale(pq.s).magnitude, dtype=float)
            return np.asarray(ev, dtype=float)
        except Exception:
            return None

    def _format_channel_title(
        self, slot_idx: int, total_n: int, ch_name: Optional[str], primary_ch: str = "", panel: Optional[ChannelPlotPanel] = None
    ) -> str:
        """Format informative label linking channel to its channel group position (N in group with label)."""
        star = "★ " if ch_name and ch_name == primary_ch else ""
        slot_label = f"{slot_idx + 1}/{total_n} in group"
        if not ch_name:
            return f"[{slot_label}] (Empty)"

        has_data = False
        if self.state is not None:
            erp = self.get_channel_erp_signal(ch_name)
            has_data = erp is not None and len(erp) > 0

        status = "" if has_data else " (No data)"

        extra = ""
        if panel is not None:
            if getattr(panel, "pinned_unit_idx", None) is not None and self.state and 0 <= panel.pinned_unit_idx < len(self.state.spike_groups):
                u_name = getattr(self.state.spike_groups[panel.pinned_unit_idx].event, "name", str(panel.pinned_unit_idx))
                extra = f" [Pin: Unit {u_name}]"
            elif getattr(panel, "window_group", None):
                extra = f" [{panel.window_group}]"
            elif not getattr(panel, "x_linked", True):
                extra = " [X-Detached]"

        return f"{star}[{slot_label}] {ch_name}{status}{extra}"

    def _rebuild_channel_panels(self):
        """Construct or adjust per-channel PlotItems side-by-side in self.glw."""
        primary_ch = (
            self.channel_bar.get_primary_channel()
            if hasattr(self, "channel_bar") and self.channel_bar is not None
            else ""
        )
        if not primary_ch and self.state is not None:
            avail = self.state.get_available_channels()
            if avail:
                primary_ch = avail[0]
            elif self.state.analog_signal is not None:
                primary_ch = getattr(self.state.analog_signal, "name", "")

        target_n = max(getattr(self, "group_size_n", 1), len(self.displayed_channels), 1)
        self.group_size_n = target_n

        # If displayed_channels is empty or only [None], populate slot 0 with primary_ch if available
        if (not self.displayed_channels or self.displayed_channels == [None]) and primary_ch:
            self.displayed_channels = [primary_ch]

        # Pad displayed_channels up to target_n
        while len(self.displayed_channels) < target_n:
            self.displayed_channels.append(None)

        if len(self.channel_panels) == target_n:
            for i, p in enumerate(self.channel_panels):
                p.slot_idx = i
                p.channel_name = self.displayed_channels[i]
                title = self._format_channel_title(i, target_n, p.channel_name, primary_ch, panel=p)
                p.plot.setTitle(title, color="#333", size="9pt")
            self._update_axis_linking()
            self.plot_right_axis()
            self.plotsChanged.emit()
            return

        # Count changed - remove right plots and channel panels from glw
        for p in self.right_plots.values():
            try:
                self.glw.removeItem(p)
            except Exception:
                pass
        self.right_plots.clear()
        self.summary_panels.clear()

        for p in self.channel_panels:
            try:
                self.glw.removeItem(p.plot)
            except Exception:
                pass
        self.channel_panels.clear()
        self._last_curstim_line = None

        cur_stim = self.state.stimno if self.state and self.state.stimno is not None else 0

        for i in range(target_n):
            ch_name = self.displayed_channels[i]
            time_axis = EphysTimeAxisItem(
                lambda: self.state.sampling_rate if self.state else None,
                orientation="bottom",
            )
            plot = self.glw.addPlot(row=0, col=i, axisItems={"bottom": time_axis})
            plot.invertY(True)
            plot.setLabel("bottom", "Time", units="ms")
            plot.showGrid(x=True, y=True, alpha=0.3)
            plot.getViewBox().disableAutoRange()
            self.glw.ci.layout.setColumnStretchFactor(i, 3)

            if i == 0:
                if getattr(self, "y_axis_mode", "sweep") == "time":
                    plot.setLabel("left", "Time", units="s")
                else:
                    plot.setLabel("left", "Stimulus #")
            else:
                plot.getAxis("left").setStyle(showValues=False)

            title = self._format_channel_title(i, target_n, ch_name, primary_ch)
            plot.setTitle(title, color="#333", size="9pt")

            # 1. Heatmap ImageItem
            img_item = pg.ImageItem(axisOrder="row-major")
            img_item.setLookupTable(self._cmap_lut)
            plot.addItem(img_item)

            # 2. Lines mode curves
            lines_curve = pg.PlotDataItem(pen=pg.mkPen(color=(150, 150, 150), width=1), connect="finite")
            active_sweep_curve = pg.PlotDataItem(pen=pg.mkPen(color="purple", width=2))
            highlight_segments = pg.PlotDataItem(pen=pg.mkPen(color="red", width=2), connect="finite")
            points_scatter = pg.ScatterPlotItem(size=8, symbol="o", brush=pg.mkBrush("black"), pen=None)

            plot.addItem(lines_curve)
            plot.addItem(active_sweep_curve)
            plot.addItem(highlight_segments)
            plot.addItem(points_scatter)
            lines_curve.setVisible(False)
            active_sweep_curve.setVisible(False)
            highlight_segments.setVisible(False)
            points_scatter.setVisible(False)

            # 3. Current stimulus cursor line
            stim_line = pg.InfiniteLine(angle=0, pen=pg.mkPen("purple", width=1.5), movable=False)
            stim_line.setValue(cur_stim)
            plot.addItem(stim_line)

            # 4. Spike group scatter points
            tracked_scatter = pg.ScatterPlotItem(
                size=8, symbol="s", pen=None, brush=pg.mkBrush(255, 0, 0, 200)
            )
            plot.addItem(tracked_scatter)

            # 5. Empty Screen watermark text
            empty_text = pg.TextItem(text="", color=(120, 120, 120), anchor=(0.5, 0.5))
            plot.addItem(empty_text)
            empty_text.setVisible(False)

            # 6. Candidate preview overlay (amber open circles + dashed band)
            candidate_band = pg.LinearRegionItem(
                orientation="vertical",
                brush=pg.mkBrush(255, 170, 0, 30),
                pen=pg.mkPen((255, 170, 0), width=1, style=Qt.DashLine),
                movable=False,
            )
            candidate_band.setVisible(False)
            candidate_band.setZValue(15)
            plot.addItem(candidate_band)
            candidate_scatter = pg.ScatterPlotItem(
                size=10,
                symbol="o",
                pen=pg.mkPen((255, 170, 0), width=1.5),
                brush=pg.mkBrush(255, 170, 0, 40),
            )
            candidate_scatter.setZValue(16)
            plot.addItem(candidate_scatter)

            panel = ChannelPlotPanel(
                slot_idx=i,
                channel_name=ch_name,
                plot=plot,
                img_item=img_item,
                lines_curve=lines_curve,
                active_sweep_curve=active_sweep_curve,
                stim_line=stim_line,
                tracked_unit_scatter=tracked_scatter,
                empty_text_item=empty_text,
                time_axis=time_axis,
                highlight_segments=highlight_segments,
                points_scatter=points_scatter,
            )
            panel.candidate_band = candidate_band
            panel.candidate_scatter = candidate_scatter
            self.channel_panels.append(panel)

        # Backward compatibility aliases
        self.main_plot = self.channel_panels[0].plot
        self.img_item = self.channel_panels[0].img_item
        self.lines_curve = self.channel_panels[0].lines_curve
        self.active_sweep_curve = self.channel_panels[0].active_sweep_curve
        self.highlight_segments = self.channel_panels[0].highlight_segments
        self.points_scatter = self.channel_panels[0].points_scatter
        self.stim_line = self.channel_panels[0].stim_line
        self.tracked_unit_scatter = self.channel_panels[0].tracked_unit_scatter
        self.other_unit_scatters = self.channel_panels[0].other_unit_scatters
        self.candidate_scatter = self.channel_panels[0].candidate_scatter
        self.candidate_band = self.channel_panels[0].candidate_band
        self.time_axis = self.channel_panels[0].time_axis

        self._update_axis_linking()
        self._apply_pinned_unit_windows()

        # Attach selector items to panel 0
        self.corridor_item = QtWidgets.QGraphicsPolygonItem()
        self.corridor_item.setPen(pg.mkPen("purple", width=1, style=Qt.DashLine))
        self.corridor_item.setBrush(pg.mkBrush(160, 32, 240, 50))
        self.corridor_item.setZValue(20)
        self.corridor_item.setVisible(False)

        self.selector_line_item = pg.PlotDataItem(
            pen=pg.mkPen("purple", width=2),
            symbol="o",
            symbolSize=7,
            symbolBrush=pg.mkBrush("purple"),
            symbolPen=pg.mkPen("w", width=1),
        )
        self.selector_line_item.setZValue(21)
        self.selector_line_item.setVisible(False)

        self.main_plot.addItem(self.corridor_item)
        self.main_plot.addItem(self.selector_line_item)

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            try:
                self.main_plot.vb.sigRangeChanged.disconnect(self._on_view_range_changed)
            except Exception:
                pass
        self.main_plot.vb.sigRangeChanged.connect(self._on_view_range_changed)
        self.plotsChanged.emit()

    def add_channel_to_display(self, channel_name: str):
        """Add a channel to display, filling an empty slot if available or expanding N."""
        if not channel_name:
            return
        filled = False
        for i in range(len(self.displayed_channels)):
            if self.displayed_channels[i] is None:
                self.displayed_channels[i] = channel_name
                filled = True
                break
        if not filled:
            self.displayed_channels.append(channel_name)
            self.group_size_n = len(self.displayed_channels)
            if hasattr(self, "groupSizeSpinBox"):
                with QSignalBlocker(self.groupSizeSpinBox):
                    self.groupSizeSpinBox.setValue(self.group_size_n)

        self._sync_linked_channel_group()
        self.setup_figure()
        self.reset_right_axes_data()
        self.plotsChanged.emit()

    def remove_channel_from_display(self, channel_name: str, keep_empty_slot: bool = False):
        """Remove a channel from display and update the linked channel group."""
        if not channel_name:
            return
        for i, ch in enumerate(self.displayed_channels):
            if ch == channel_name:
                if keep_empty_slot:
                    self.displayed_channels[i] = None
                else:
                    self.displayed_channels.pop(i)
                    self.group_size_n = max(len(self.displayed_channels), 1)
                    if hasattr(self, "groupSizeSpinBox"):
                        with QSignalBlocker(self.groupSizeSpinBox):
                            self.groupSizeSpinBox.setValue(self.group_size_n)
                break

        self._sync_linked_channel_group()
        self.setup_figure()
        self.reset_right_axes_data()
        self.plotsChanged.emit()

    def clear_slot_to_empty(self, slot_idx: int):
        """Clear a specific slot to an empty screen without changing N."""
        if 0 <= slot_idx < len(self.displayed_channels):
            self.displayed_channels[slot_idx] = None
            self._sync_linked_channel_group()
            self.setup_figure()
            self.reset_right_axes_data()
            self.plotsChanged.emit()

    def add_empty_slot(self):
        """Add an empty channel screen slot to the group."""
        self.displayed_channels.append(None)
        self.group_size_n = len(self.displayed_channels)
        if hasattr(self, "groupSizeSpinBox"):
            with QSignalBlocker(self.groupSizeSpinBox):
                self.groupSizeSpinBox.setValue(self.group_size_n)
        self.setup_figure()
        self.plotsChanged.emit()

    def set_group_size_n(self, n: int):
        """Set the target group size N, preserving existing channels and padding or trimming."""
        n = max(int(n), 1)
        self.group_size_n = n
        if len(self.displayed_channels) < n:
            while len(self.displayed_channels) < n:
                self.displayed_channels.append(None)
        elif len(self.displayed_channels) > n:
            while len(self.displayed_channels) > n:
                self.displayed_channels.pop()
        self._sync_linked_channel_group()
        self.setup_figure()
        self.reset_right_axes_data()
        self.plotsChanged.emit()

    def _set_channel_as_primary(self, slot_idx: int):
        """Move the channel at slot_idx to position 0 (primary)."""
        if 0 < slot_idx < len(self.displayed_channels):
            ch = self.displayed_channels.pop(slot_idx)
            self.displayed_channels.insert(0, ch)
            self._sync_linked_channel_group()
            self.setup_figure()
            self.reset_right_axes_data()
            self.plotsChanged.emit()

    def _replace_slot_channel(self, slot_idx: int, new_ch: str):
        """Replace channel at slot_idx with new_ch."""
        self.set_plot_slot_channel(slot_idx, new_ch)

    def _remove_empty_screens(self):
        """Remove empty None slots down to minimum 1."""
        cleaned = [c for c in self.displayed_channels if c]
        if not cleaned:
            cleaned = [None]
        self.displayed_channels = cleaned
        self.group_size_n = len(self.displayed_channels)
        if hasattr(self, "groupSizeSpinBox"):
            with QSignalBlocker(self.groupSizeSpinBox):
                self.groupSizeSpinBox.setValue(self.group_size_n)
        self.setup_figure()
        self.plotsChanged.emit()

    # ------------------------------------------------------------------
    # Per-plot management API (used by MultiTracePlotManager)
    # ------------------------------------------------------------------
    def _summary_stat_name(self) -> str:
        return "Max" if self.toggleMaxMeanButton.isChecked() else "Mean"

    def get_summary_plot_availability(self) -> Dict[str, bool]:
        """Return globally available summary plots mapped to their enabled state."""
        availability: Dict[str, bool] = {
            "__channel_summaries__": bool(
                self.rightPlots.get("__channel_summaries__", True)
            ),
        }
        if (
            self.state is not None
            and self.state.event_signal is not None
            and len(self.state.event_signal) > 1
        ):
            availability["Stim. Freq. (Hz)"] = bool(
                self.rightPlots.get("Stim. Freq. (Hz)", True)
            )
        if (
            self.state is not None
            and self.state.spike_groups
            and self.state.event_signal is not None
        ):
            availability["Latency Diff."] = bool(
                self.rightPlots.get("Latency Diff.", True)
            )
        return availability

    def get_plot_specs(self) -> List[Dict[str, Any]]:
        """Describe every plot in the view for a per-plot settings UI.

        Returns channel plots first (in display order), followed by the summary
        line graphs that can be added below/next to them.
        """
        specs: List[Dict[str, Any]] = []
        primary_ch = (
            self.channel_bar.get_primary_channel()
            if hasattr(self, "channel_bar") and self.channel_bar is not None
            else ""
        )
        n_slots = len(self.channel_panels)
        for i, panel in enumerate(self.channel_panels):
            specs.append(
                {
                    "kind": "channel",
                    "key": f"channel:{i}",
                    "slot_idx": i,
                    "channel_name": panel.channel_name,
                    "label": self._format_channel_title(
                        i, n_slots, panel.channel_name, primary_ch, panel=panel
                    ),
                    "has_data": panel.has_data,
                    "visible": True,
                    "pinned_unit_idx": getattr(panel, "pinned_unit_idx", None),
                    "window_group": getattr(panel, "window_group", None),
                    "window_width_ms": float(
                        getattr(panel, "window_width_ms", 20.0)
                    ),
                    "x_linked": bool(getattr(panel, "x_linked", True)),
                }
            )

        show_channel_summaries = bool(
            self.rightPlots.get("__channel_summaries__", True)
        )
        if show_channel_summaries:
            stat_name = self._summary_stat_name()
            for i in range(self.group_size_n):
                ch_name = (
                    self.displayed_channels[i]
                    if i < len(self.displayed_channels)
                    else None
                )
                key = ch_name if ch_name is not None else f"__empty_slot_{i}__"
                specs.append(
                    {
                        "kind": "summary",
                        "key": key,
                        "slot_idx": i,
                        "channel_name": ch_name,
                        "label": self._format_summary_title(
                            i, self.group_size_n, ch_name, primary_ch, stat_name
                        ),
                        "has_data": bool(
                            ch_name
                            and ch_name in self.right_ax_data
                            and len(self.right_ax_data[ch_name][0]) > 0
                        ),
                        "visible": bool(self.rightPlots.get(key, True)),
                        "is_channel_stat": True,
                    }
                )

        global_labels = {
            "Stim. Freq. (Hz)": "Stimulus Frequency",
            "Latency Diff.": "Latency Difference",
        }
        for key, is_enabled in self.get_summary_plot_availability().items():
            if key == "__channel_summaries__":
                continue
            specs.append(
                {
                    "kind": "summary",
                    "key": key,
                    "slot_idx": None,
                    "channel_name": None,
                    "label": global_labels.get(key, key),
                    "has_data": key in self.right_ax_data,
                    "visible": bool(is_enabled),
                    "is_channel_stat": False,
                }
            )
        return specs

    def get_available_plot_channels(self) -> List[str]:
        """Available recording channels that are not displayed yet."""
        if self.state is None:
            return []
        displayed = {c for c in self.displayed_channels if c}
        return [c for c in self.state.get_available_channels() if c not in displayed]

    def remove_plot_slot(self, slot_idx: int):
        """Remove the channel plot at ``slot_idx`` and shrink the group."""
        if not (0 <= slot_idx < len(self.displayed_channels)):
            return
        self.displayed_channels.pop(slot_idx)
        if not self.displayed_channels:
            self.displayed_channels = [None]
        self.group_size_n = max(len(self.displayed_channels), 1)
        if hasattr(self, "groupSizeSpinBox"):
            with QSignalBlocker(self.groupSizeSpinBox):
                self.groupSizeSpinBox.setValue(self.group_size_n)
        self._sync_linked_channel_group()
        self.setup_figure()
        self.reset_right_axes_data()
        self.plotsChanged.emit()

    def move_plot_slot(self, slot_idx: int, delta: int):
        """Move a channel plot left/right by ``delta`` positions."""
        target = int(slot_idx) + int(delta)
        if not (0 <= slot_idx < len(self.displayed_channels)):
            return
        if not (0 <= target < len(self.displayed_channels)) or target == slot_idx:
            return
        self.displayed_channels.insert(
            target, self.displayed_channels.pop(slot_idx)
        )
        self._sync_linked_channel_group()
        self.setup_figure()
        self.reset_right_axes_data()
        self.plotsChanged.emit()

    def set_plot_slot_channel(self, slot_idx: int, channel_name: Optional[str]):
        """Set or clear the channel shown in a plot slot."""
        if not (0 <= slot_idx < len(self.displayed_channels)):
            return
        if self.displayed_channels[slot_idx] == channel_name:
            return
        self.displayed_channels[slot_idx] = channel_name or None
        self._sync_linked_channel_group()
        self.setup_figure()
        self.reset_right_axes_data()
        self.plotsChanged.emit()

    def set_plot_slot_options(self, slot_idx: int, **options):
        """Update per-plot options for a channel plot.

        Supported keys: ``channel_name``, ``pinned_unit_idx``, ``window_group``,
        ``window_width_ms`` and ``x_linked``.
        """
        if not (0 <= slot_idx < len(self.channel_panels)):
            return
        panel = self.channel_panels[slot_idx]

        if "channel_name" in options:
            self.set_plot_slot_channel(slot_idx, options["channel_name"])

        if "pinned_unit_idx" in options:
            panel.pinned_unit_idx = options["pinned_unit_idx"]
            panel.x_linked = panel.pinned_unit_idx is None
        if "x_linked" in options:
            panel.x_linked = bool(options["x_linked"])
        if "window_group" in options:
            panel.window_group = options["window_group"]
        if "window_width_ms" in options:
            panel.window_width_ms = float(options["window_width_ms"])

        self._update_axis_linking()
        self._apply_pinned_unit_windows()
        self._update_channel_titles()
        self.plotsChanged.emit()

    def set_plot_visible(self, key: str, visible: bool):
        """Show or hide a summary plot."""
        if key == "__channel_summaries__":
            self.toggle_channel_summaries(visible)
        elif (
            key in ("Stim. Freq. (Hz)", "Latency Diff.")
            or key in self.rightPlots
            or key.startswith("__empty_slot_")
        ):
            self.toggle_summary_graph(key, visible)
        self.plotsChanged.emit()

    def _prompt_custom_channel(self):
        if QApplication.instance() and QApplication.instance().platformName() != "offscreen":
            name, ok = QInputDialog.getText(self, "Add Custom Channel", "Channel Name:")
            if ok and name.strip():
                self.add_channel_to_display(name.strip())

    def _sync_linked_channel_group(self):
        """Synchronize the valid displayed channels back to the linked channel group in ViewerState."""
        if self.state is None:
            return
        valid_chans = [c for c in self.displayed_channels if c]
        if not valid_chans:
            return

        mode, val = (
            self.channel_bar.get_channel_mode()
            if hasattr(self, "channel_bar") and self.channel_bar is not None
            else ("__global__", None)
        )
        u_idx = self.get_target_unit_idx()

        if mode == ChannelGroupBar.MODE_GROUP and val:
            self.state.add_channel_group(str(val), valid_chans)
        elif mode == ChannelGroupBar.MODE_SHARED_SET and val:
            self.state.set_shared_channels(str(val), valid_chans)
        elif mode == ChannelGroupBar.MODE_FOLLOW_UNIT or (
            hasattr(self, "channel_bar") and self.channel_bar is not None and self.channel_bar.is_unit_locked()
        ):
            self.state.set_unit_channels(u_idx, valid_chans)
        else:
            if hasattr(self.state, "set_unit_channels") and self.state.spike_groups:
                self.state.set_unit_channels(u_idx, valid_chans)

    def get_channel_erp_signal(self, channel_name: Optional[str]):
        """Retrieve ERP for a specific channel name, checking live streaming buffers first."""
        if self.state is None or not channel_name:
            return None

        if getattr(self.state, "is_streaming", False):
            if hasattr(self.state, "_live_aux_erp_buffers") and channel_name in self.state._live_aux_erp_buffers:
                buf = self.state._live_aux_erp_buffers[channel_name]
                if buf is not None and len(buf) > 0:
                    return buf.data
            primary = (
                self.channel_bar.get_primary_channel()
                if hasattr(self, "channel_bar") and self.channel_bar is not None
                else None
            )
            sig_name = getattr(self.state.analog_signal, "name", "")
            if channel_name in (primary, sig_name, "rd.0"):
                buf = getattr(self.state, "_live_erp_buffer", None)
                if buf is not None and len(buf) > 0:
                    return buf.data

        if hasattr(self.state, "get_channel_erp"):
            return self.state.get_channel_erp(channel_name)
        return self.get_erp_signal()

    def _on_scale_mode_changed(self, mode: str):
        self.scale_mode = mode
        self.render()

    def _on_channel_bar_changed(self):
        self._on_channel_change()

    def _on_channel_change(self):
        self.update_window_title()

        group_chans = (
            self.channel_bar.get_selected_channels()
            if hasattr(self, "channel_bar") and self.channel_bar is not None
            else []
        )
        if not group_chans and self.state:
            avail = self.state.get_available_channels()
            if avail:
                group_chans = [avail[0]]

        target_n = max(getattr(self, "group_size_n", 1), len(group_chans), 1)
        self.group_size_n = target_n

        new_displayed: List[Optional[str]] = []
        for i in range(target_n):
            if i < len(group_chans):
                new_displayed.append(group_chans[i])
            else:
                new_displayed.append(None)

        self.displayed_channels = new_displayed

        if hasattr(self, "groupSizeSpinBox"):
            with QSignalBlocker(self.groupSizeSpinBox):
                self.groupSizeSpinBox.setValue(self.group_size_n)

        self.setup_figure()
        self.reset_right_axes_data()
        self.plotsChanged.emit()

    def update_window_title(self):
        if hasattr(self, "channel_bar"):
            summary = self.channel_bar.get_title_summary()
            self.setWindowTitle(f"MultiTrace (PyQtGraph) {summary}")
            p = self.parentWidget()
            if p is not None and hasattr(p, "setWindowTitle"):
                p.setWindowTitle(f"MultiTrace (PyQtGraph) {summary}")

    def get_target_unit(self) -> Optional[tracked_neuron_unit]:
        if hasattr(self, "channel_bar") and self.channel_bar is not None:
            return self.channel_bar.get_target_unit()
        if self.state and 0 <= self.state.cur_spike_group < len(self.state.spike_groups):
            return self.state.spike_groups[self.state.cur_spike_group]
        return None

    def get_target_unit_idx(self) -> int:
        if hasattr(self, "channel_bar") and self.channel_bar is not None:
            return self.channel_bar.get_target_unit_idx()
        return getattr(self.state, "cur_spike_group", 0) if self.state else 0

    def get_erp_signal(self):
        if self.state is None:
            return None
        if hasattr(self, "channel_bar") and self.channel_bar is not None:
            primary_ch = self.channel_bar.get_primary_channel()
            if primary_ch and hasattr(self.state, "get_channel_erp"):
                erp = self.state.get_channel_erp(primary_ch)
                if erp is not None and len(erp) > 0:
                    return erp
        return self.state.get_erp()

    def get_live_buffer(self):
        if self.state is None:
            return None
        primary = (
            self.channel_bar.get_primary_channel()
            if hasattr(self, "channel_bar") and self.channel_bar is not None
            else getattr(getattr(self.state, "analog_signal", None), "name", None)
        )
        if hasattr(self.state, "_live_aux_erp_buffers") and self.state._live_aux_erp_buffers and primary in self.state._live_aux_erp_buffers:
            return self.state._live_aux_erp_buffers[primary]
        if self.state.analog_signal is not None and primary == getattr(self.state.analog_signal, "name", None):
            return getattr(self.state, "_live_erp_buffer", None)
        return getattr(self.state, "_live_erp_buffer", None)

    def _on_lock_changed(self):
        self.lock_to_stim = self.lock_to_stimCheckBox.isChecked()
        if self.state and self.state.stimno is not None:
            self.update_ylim(self.state.stimno)

    def _on_view_range_changed(self, *args):
        if getattr(self, "_has_initialized_ranges", False) and not self.is_view_culled():
            self._range_timer.start()

    def _on_range_timer_timeout(self):
        if self.state is not None and not self.is_view_culled():
            self._schedule_render(self.state.stimno if self.state else None)
            if self.mode == "lines":
                self.setup_figure()

    def reset_view(self):
        self.update_axis()

    def mousePressEvent(self, event):
        notify_view_selected(self)
        super().mousePressEvent(event)

    def _on_plot_clicked(self, event):
        """Handle clicks on any channel plot for sweep navigation, channel context menu, or selector point placement."""
        notify_view_selected(self)
        if self.is_nav_active():
            return
        scene_pos = event.scenePos()
        target_panel = None
        for p in self.channel_panels:
            if hasattr(p.plot, "vb") and p.plot.vb.sceneBoundingRect().contains(scene_pos):
                target_panel = p
                break

        target_summary = None
        if target_panel is None:
            for sp in self.summary_panels.values():
                if hasattr(sp.plot, "vb") and sp.plot.vb.sceneBoundingRect().contains(scene_pos):
                    target_summary = sp
                    break

        if target_summary is not None:
            pos = target_summary.plot.vb.mapSceneToView(scene_pos)
            if event.button() == Qt.RightButton:
                screen_pos = event.screenPos().toPoint() if hasattr(event.screenPos(), "toPoint") else QtCore.QPoint(int(event.screenPos().x()), int(event.screenPos().y()))
                self._show_summary_context_menu(screen_pos, target_summary.key, target_summary)
                event.accept()
                return
            elif event.button() == Qt.LeftButton:
                if not np.isnan(pos.y()):
                    stimno = int(round(pos.y()))
                    if self.state and self.state.event_signal is not None:
                        if 0 <= stimno < len(self.state.event_signal):
                            self.state.setStimNo(stimno)
                event.accept()
                return

        if target_panel is None:
            target_panel = self.channel_panels[0] if self.channel_panels else None
        if target_panel is None:
            return

        pos = target_panel.plot.vb.mapSceneToView(scene_pos)

        # If selector is active, left-click adds a vertex, right-click removes the last vertex
        if self.selector_active:
            if event.button() == Qt.LeftButton:
                if self.selector_panel is not None and self.selector_panel != target_panel:
                    if self.corridor_item.scene() is not None:
                        try:
                            self.corridor_item.scene().removeItem(self.corridor_item)
                        except Exception:
                            pass
                    if self.selector_line_item.scene() is not None:
                        try:
                            self.selector_line_item.scene().removeItem(self.selector_line_item)
                        except Exception:
                            pass
                    target_panel.plot.addItem(self.corridor_item)
                    target_panel.plot.addItem(self.selector_line_item)
                self.selector_points.append([float(pos.x()), float(pos.y())])
                self.selector_channel_name = target_panel.channel_name
                self.selector_panel = target_panel
                self._redraw_selector()
                event.accept()
                return
            elif event.button() == Qt.RightButton:
                if self.selector_points:
                    self.selector_points.pop()
                    self._redraw_selector()
                    event.accept()
                    return

        # Right-click when selector not active: show channel context menu
        if event.button() == Qt.RightButton:
            screen_pos = event.screenPos().toPoint() if hasattr(event.screenPos(), "toPoint") else QtCore.QPoint(int(event.screenPos().x()), int(event.screenPos().y()))
            self._show_channel_context_menu(target_panel, screen_pos)
            event.accept()
            return

        # Normal mode: Shift+Click seeds & auto-tracks a new unit; plain left-click selects sweep
        if event.button() == Qt.LeftButton:
            modifiers = (
                event.modifiers()
                if hasattr(event, "modifiers") and callable(event.modifiers)
                else Qt.NoModifier
            )
            if modifiers & Qt.ShiftModifier:
                if not np.isnan(pos.x()) and not np.isnan(pos.y()):
                    sweep_idx = self._y_pos_to_stimno(float(pos.y()))
                    if sweep_idx is not None:
                        self.seed_new_unit_at_pos(
                            float(pos.x()),
                            stimno=sweep_idx,
                            channel_name=target_panel.channel_name,
                        )
                        event.accept()
                        return
            if not np.isnan(pos.y()):
                if getattr(self, "y_axis_mode", "sweep") == "time":
                    stim_times_s = self._get_stim_times_s()
                    if stim_times_s is not None and len(stim_times_s) > 0:
                        cur_stim = min(max(0, self.state.stimno if self.state else 0), len(stim_times_s) - 1)
                        t_selected = stim_times_s[cur_stim]
                        t_clicked = t_selected + pos.y()
                        diffs = np.abs(stim_times_s - t_clicked)
                        closest_stim = int(np.argmin(diffs))
                        if self.state and self.state.event_signal is not None and 0 <= closest_stim < len(self.state.event_signal):
                            self.state.setStimNo(closest_stim)
                else:
                    stimno = int(round(pos.y()))
                    if self.state and self.state.event_signal is not None:
                        if 0 <= stimno < len(self.state.event_signal):
                            self.state.setStimNo(stimno)

    def _on_plot_mouse_moved(self, scene_pos):
        """Draw dynamic preview from the last point to cursor when selector is active."""
        if not self.selector_active or len(self.selector_points) == 0:
            return
        target_panel = getattr(self, "selector_panel", None) or (self.channel_panels[0] if self.channel_panels else None)
        if target_panel is None:
            return
        pos = target_panel.plot.vb.mapSceneToView(scene_pos)
        pts = self.selector_points + [[float(pos.x()), float(pos.y())]]
        self._update_selector_graphics(pts)

    def _redraw_selector(self):
        if not self.selector_active:
            self.selector_line_item.setVisible(False)
            self.corridor_item.setVisible(False)
            return
        self._update_selector_graphics(self.selector_points)

    def _update_selector_graphics(self, pts_list: List[List[float]]):
        if len(pts_list) == 0:
            self.selector_line_item.setVisible(False)
            self.corridor_item.setVisible(False)
            return

        pts = np.array(pts_list)
        self.selector_line_item.setData(pts[:, 0], pts[:, 1])
        self.selector_line_item.setVisible(True)

        if len(pts) >= 2:
            w = self.selector_width
            upper = pts + np.array([w / 2.0, 0.0])
            lower = pts - np.array([w / 2.0, 0.0])
            lower = lower[::-1]
            poly_points = np.vstack([upper, lower, upper[0]])
            qpoly = QtGui.QPolygonF([QPointF(float(p[0]), float(p[1])) for p in poly_points])
            self.corridor_item.setPolygon(qpoly)
            self.corridor_item.setVisible(True)
        else:
            self.corridor_item.setVisible(False)

    def is_view_culled(self) -> bool:
        try:
            if getattr(self, "_user_hidden", False):
                return True
            win = self.window()
            if win is not None and win.isMinimized():
                return True
            if self.parentWidget() is not None and not self.isVisible():
                return True
            if getattr(self, "_was_shown", False) and not self.isVisible():
                return True
            return False
        except Exception:
            return False

    def setVisible(self, visible: bool):
        if not visible:
            self._user_hidden = True
        else:
            self._user_hidden = False
            self._was_shown = True
        super().setVisible(visible)
        if visible and getattr(self, "_dirty", False):
            self._dirty = False
            self.on_new_sweep(self.state.stimno if self.state else 0)

    def hideEvent(self, event):
        super().hideEvent(event)
        self._user_hidden = True

    def showEvent(self, event):
        super().showEvent(event)
        self._was_shown = True
        self._user_hidden = False
        if getattr(self, "_dirty", False):
            self._dirty = False
            self.on_new_sweep(self.state.stimno if self.state else 0)

    def _connect_state(self, sig, slot):
        sig.connect(slot)
        self.connections.append((sig, slot))

    def unset_connections(self):
        self._unregister_live_frames()
        for sig, slot in self.connections:
            try:
                sig.disconnect(slot)
            except (TypeError, RuntimeError):
                pass
        self.connections.clear()

    def remove_references(self):
        self.unset_connections()

    @Slot()
    def _on_data_reload(self):
        self.reset_right_axes_data()
        self.setup_figure()
        if not getattr(self, "_has_initialized_ranges", False):
            self.update_axis()

    @Slot()
    def _on_unit_updated(self, *args):
        self.plot_spikegroups()
        if getattr(self.state, "is_streaming", False):
            self.on_new_sweep_throttled()
        else:
            self.render()
            self._schedule_right_axis_update(force=True)

    @Slot()
    @Slot(int)
    def _on_stim_no_changed(self, stimno: int = -1):
        if getattr(self.state, "is_streaming", False):
            self.on_new_sweep_throttled()
        else:
            target = self.state.stimno if stimno < 0 and self.state is not None else stimno
            if target is not None:
                self.update_ylim(target)
                self.plot_curstim_line(target)
        self._update_live_paused_banner()

    @Slot()
    @Slot(int)
    def _on_new_spike_events(self, *args):
        self.plot_spikegroups()
        if getattr(self.state, "is_streaming", False):
            self.on_new_sweep_throttled()
        else:
            self._schedule_spike_group_update()

    def set_state(self, state: ViewerState):
        self.unset_connections()
        self.state = state
        if hasattr(self, "channel_bar") and self.channel_bar is not None:
            self.channel_bar.setState(state)
        if self.state is None:
            return
        self._on_channel_change()

        # Register with background task manager
        self._task_manager.register_viewer(
            self._viewer_id,
            self.state,
            {
                TaskType.MULTI_TRACE_RENDER: self._on_multi_trace_result,
                TaskType.RIGHT_AXIS_UPDATE: self._on_right_axis_result,
                TaskType.SPIKE_GROUP_UPDATE: self._on_spike_group_result,
            }
        )

        # Static file loads are processed through the same orchestrator
        # pipeline as live sweeps (single point of view processing).
        self._register_data_reload(
            state,
            self._on_data_reload,
            name="reload_setup",
        )

        self.on_new_sweep_throttled = qsignal_throttle(
            lambda: self.on_new_sweep(self.state.stimno if self.state else 0),
            40,
            parent=self,
            weight="primary",
        )
        self.reset_right_axes_throttled = qsignal_throttle(
            self.reset_right_axes_data,
            200,
            parent=self,
            weight="heavy",
        )

        def _on_unit_updated(*args):
            self.plot_spikegroups()
            if getattr(self.state, "is_streaming", False):
                self.on_new_sweep_throttled()
            else:
                self.render()
                self._schedule_right_axis_update(force=True)

        self._connect_state(self.state.onUnitGroupChange, _on_unit_updated)
        if hasattr(self.state, "onUnitListChanged"):
            self._connect_state(self.state.onUnitListChanged, _on_unit_updated)
        self._connect_state(self.state.onUnitChange, _on_unit_updated)
        if hasattr(self.state, "onCandidatesChanged"):
            self._connect_state(self.state.onCandidatesChanged, self._on_candidates_changed)
        if hasattr(self.state, "onCandidateSelectionChanged"):
            self._connect_state(
                self.state.onCandidateSelectionChanged, self._on_candidates_changed
            )
        self._connect_state(self.state.onStimNoChange, self._on_stim_no_changed)

        if hasattr(self.state, "onChannelChange"):
            self._connect_state(self.state.onChannelChange, self._on_channel_change)

        # Live sweeps arrive through the state's LiveOrchestrator (LIFO);
        # the GUI-side throttle keeps render scheduling rate-limited.
        if hasattr(self.state, "onNewSweep"):
            self._register_live_frames(state, lambda _stimno: self.on_new_sweep_throttled())
        if hasattr(self.state, "onNewSpikeEvents"):
            self._connect_state(self.state.onNewSpikeEvents, self._on_new_spike_events)
        if hasattr(self.state, "onStreamStateChanged"):
            self._connect_state(self.state.onStreamStateChanged, self._on_stream_state_changed)
        if hasattr(self.state, "onFollowLiveChanged"):
            self._connect_state(self.state.onFollowLiveChanged, self._update_live_paused_banner)
        if hasattr(self.state, "onProtocolMarkersChanged"):
            self._connect_state(
                self.state.onProtocolMarkersChanged,
                self._update_protocol_markers_overlay,
            )
        if hasattr(self.state, "onContainerVisualSettingsChanged"):
            self._connect_state(
                self.state.onContainerVisualSettingsChanged,
                self._on_container_visual_settings_changed,
            )
        if hasattr(self.state, "onSharedVisualSettingsChanged"):
            self._connect_state(
                self.state.onSharedVisualSettingsChanged,
                self._on_shared_visual_settings_changed,
            )

        self.update_axis()
        self._schedule_right_axis_update(force=True)
        self.update_window_title()
        self._update_live_paused_banner()
        self._update_protocol_markers_overlay()

    def _on_stream_state_changed(self, is_streaming: bool):
        if not is_streaming and self.state is not None:
            self._schedule_render(self.state.stimno)
            self._schedule_right_axis_update(force=True)
            self.plot_spikegroups()

    @Slot()
    @Slot(int)
    def _on_candidates_changed(self, *args):
        if not getattr(self, "channel_panels", None):
            return
        self.plot_spikegroups()
        if getattr(self.state, "is_streaming", False):
            self.on_new_sweep_throttled()
        else:
            self.render()

    @quiet_fail
    def on_new_sweep(self, stimno: Optional[int] = None):
        """Lightweight sweep handler - schedules heavy work to background thread."""
        if self.is_view_culled():
            self._dirty = True
            return
        if self.state is None:
            return
        from .helpers import run_adaptive_render
        run_adaptive_render(self, lambda: self._on_new_sweep_core(stimno), weight="primary")

    def _on_new_sweep_core(self, stimno: Optional[int] = None):
        live_buf = self.get_live_buffer()
        if live_buf is not None and len(live_buf) > 0:
            erp = live_buf.data
        else:
            erp = self.get_erp_signal()

        if erp is None or len(erp) == 0:
            return

        erp_mag = erp.magnitude if hasattr(erp, "magnitude") else erp
        if erp_mag is None or erp_mag.shape[0] == 0:
            return

        has_buf_match = live_buf is not None and len(live_buf) == erp_mag.shape[0]

        # Update running std (lightweight)
        if not hasattr(self, "std") or self.std == 0 or self.std is None or erp_mag.shape[0] <= 5:
            std_val = np.std(erp_mag)
            self.std = float(std_val.magnitude if hasattr(std_val, "magnitude") else std_val) if std_val > 0 else 1.0
        else:
            if has_buf_match:
                last_std = float(live_buf.row_std[-1]) or 1.0
            else:
                s = np.std(erp_mag[-1])
                last_std = float(s.magnitude if hasattr(s, "magnitude") else s) or 1.0
            self.std = 0.95 * float(self.std) + 0.05 * last_std

        n_sweeps, n_samples = erp_mag.shape

        # Lightweight UI updates that must stay on UI thread
        # Adapt xlim and ylim as sweeps arrive without resetting user zoom
        if not getattr(self, "_has_initialized_ranges", False):
            self.main_plot.setXRange(0, n_samples, padding=0)
            self.main_plot.setYRange(0, n_sweeps, padding=0)
            self._has_initialized_ranges = True
        elif self.lock_to_stim:
            self.update_ylim(self.state.stimno)
        elif getattr(self.state, "follow_live", False):
            current_yrange = self.main_plot.viewRange()[1]
            if abs(current_yrange[1] - n_sweeps) <= 3:
                window_height = abs(current_yrange[1] - current_yrange[0])
                if window_height < n_sweeps:
                    self.main_plot.setYRange(n_sweeps - window_height, n_sweeps, padding=0)
                else:
                    self.main_plot.setYRange(0, n_sweeps, padding=0)

        # Schedule heavy rendering to background thread
        if self.mode != getattr(self, "_last_setup_mode", None) or not any(p.has_data for p in self.channel_panels):
            self.setup_figure()
            return
        self._schedule_render(stimno)

        # Lightweight UI updates
        self.plot_curstim_line(self.state.stimno)
        self.plot_spikegroups()

    # ===== Background Task Scheduling =====

    def _extract_channel_row_stats(self, ch_names: List[str], show_max: bool) -> Dict[str, np.ndarray]:
        stats: Dict[str, np.ndarray] = {}
        if self.state is None:
            return stats
        is_streaming = getattr(self.state, "is_streaming", False)
        for ch_name in ch_names:
            if not ch_name:
                continue
            if is_streaming:
                buf = None
                if hasattr(self.state, "_live_aux_erp_buffers") and ch_name in self.state._live_aux_erp_buffers:
                    buf = self.state._live_aux_erp_buffers[ch_name]
                elif (
                    getattr(self.state, "_live_erp_buffer", None) is not None
                    and self.state.analog_signal is not None
                    and ch_name == getattr(self.state.analog_signal, "name", None)
                ):
                    buf = self.state._live_erp_buffer
                if buf is not None and len(buf) > 0:
                    rv = buf.row_max if show_max else buf.row_mean
                    if hasattr(rv, "magnitude"):
                        rv = rv.magnitude
                    stats[ch_name] = np.asarray(rv, dtype=np.float32)
            else:
                ch_erp = self.get_channel_erp_signal(ch_name)
                if ch_erp is not None and len(ch_erp) > 0:
                    ch_mag = ch_erp.magnitude if hasattr(ch_erp, "magnitude") else ch_erp
                    if hasattr(ch_mag, "ndim") and ch_mag.ndim >= 2 and ch_mag.shape[0] > 0:
                        rv = np.max(ch_mag, axis=1) if show_max else np.mean(ch_mag, axis=1)
                        if hasattr(rv, "magnitude"):
                            rv = rv.magnitude
                        stats[ch_name] = np.asarray(rv, dtype=np.float32)
        return stats

    def _schedule_render(self, stimno: Optional[int]):
        """Schedule a render task for the background worker."""
        if self.is_view_culled() or self.state is None:
            return
        
        live_buf = self.get_live_buffer()
        if live_buf is not None and len(live_buf) > 0:
            erp = live_buf.data
        else:
            erp = self.get_erp_signal()
        
        if erp is None or len(erp) == 0:
            return
        
        erp_mag = erp.magnitude if hasattr(erp, "magnitude") else erp
        full_shape = tuple(erp_mag.shape)
        n_sweeps, n_samples = full_shape

        # Visible slice determination
        y_range = self.main_plot.viewRange()[1]
        y_min = min(y_range)
        y_max = max(y_range)
        pad = max(10, int(abs(y_max - y_min) * 0.15))
        slice_start = max(0, int(np.floor(y_min - pad)))
        slice_stop = min(n_sweeps, int(np.ceil(y_max + pad)))
        if slice_stop <= slice_start:
            slice_start = 0
            slice_stop = n_sweeps

        n_slice = slice_stop - slice_start

        display = None
        if live_buf is not None and (full_shape[0] > 2048 or full_shape[1] > 2048):
            candidate = live_buf.heatmap_decimated_data
            if candidate is not None and candidate.size > 0:
                display = candidate.magnitude if hasattr(candidate, "magnitude") else candidate

        source = display if display is not None else erp_mag
        sub_display = source[slice_start:slice_stop]

        if n_slice > 1500:
            step = int(np.ceil(n_slice / 1000))
            sub_display = sub_display[::step]

        slice_rect = (0, slice_start, n_samples, n_slice)

        ch_names = self._get_channel_names()
        show_max = self.toggleMaxMeanButton.isChecked()
        payload = {
            "erp_data": sub_display,
            "erp_display": sub_display,
            "erp_full_shape": full_shape,
            "slice_rect": slice_rect,
            "running_std": float(self.std) if hasattr(self, "std") and self.std else None,
            "spike_groups": self.state.spike_groups,
            "event_signal": self.state.event_signal,
            "sampling_rate": self.state.sampling_rate,
            "mode": self.mode,
            "stimno": self.state.stimno,
            "lock_to_stim": self.lock_to_stim,
            "lower_percentile": self.lowerSpinBox.value(),
            "upper_percentile": self.upperSpinBox.value(),
            "follow_live": getattr(self.state, "follow_live", True),
            "channel_names": ch_names,
            "channel_row_stats": self._extract_channel_row_stats(ch_names, show_max),
            "current_unit_idx": self.get_target_unit_idx(),
            "show_max": show_max,
        }

        self._task_manager.submit_render(
            self._viewer_id,
            TaskType.MULTI_TRACE_RENDER,
            payload,
            priority=TaskPriority.HIGH,
        )
    
    def _get_channel_names(self) -> List[str]:
        """Get list of channel names for right axis computation."""
        names = []
        if hasattr(self, "displayed_channels"):
            for ch in self.displayed_channels:
                if ch and ch not in names:
                    names.append(ch)
        if hasattr(self, "channel_bar") and self.channel_bar is not None:
            chans = self.channel_bar.get_selected_channels()
            for ch in chans:
                if ch and ch not in names:
                    names.append(ch)
        elif self.state and self.state.analog_signal is not None:
            sig_name = getattr(self.state.analog_signal, "name", None)
            if sig_name and sig_name not in names:
                names.append(sig_name)
        return names
    
    def _schedule_right_axis_update(self, force: bool = False):
        """Schedule right axis data computation."""
        if self.state is None or self.state.event_signal is None or len(self.state.event_signal) == 0:
            return
        
        live_buf = self.get_live_buffer()
        if live_buf is not None and len(live_buf) > 0:
            erp = live_buf.data
        else:
            erp = self.get_erp_signal()
        
        if erp is None or len(erp) == 0:
            return

        now = time.monotonic()
        if not force and now - getattr(self, "_last_right_axis_update_time", 0.0) < 0.1:
            return
        self._last_right_axis_update_time = now
        
        erp_mag = erp.magnitude if hasattr(erp, "magnitude") else erp
        ch_names = self._get_channel_names()
        show_max = self.toggleMaxMeanButton.isChecked()
        
        self._task_manager.submit_render(
            self._viewer_id,
            TaskType.RIGHT_AXIS_UPDATE,
            {
                "erp_data": erp_mag,
                "event_signal": self.state.event_signal,
                "spike_groups": self.state.spike_groups,
                "sampling_rate": self.state.sampling_rate,
                "channel_names": ch_names,
                "channel_row_stats": self._extract_channel_row_stats(ch_names, show_max),
                "show_max": show_max,
            },
            priority=TaskPriority.LOW,
        )
    
    def _schedule_spike_group_update(self):
        """Schedule spike group points computation."""
        if self.state is None or self.state.event_signal is None:
            return
        
        self._task_manager.submit_render(
            self._viewer_id,
            TaskType.SPIKE_GROUP_UPDATE,
            {
                "event_signal": self.state.event_signal,
                "spike_groups": self.state.spike_groups,
                "sampling_rate": self.state.sampling_rate,
                "current_unit_idx": self.get_target_unit_idx(),
            },
            priority=TaskPriority.NORMAL,
        )
    
    # ===== Background Task Result Handlers =====

    def _on_multi_trace_result(self, result: Dict[str, Any], error: str):
        """Handle completed multi-trace render task."""
        if error:
            return
        if not result:
            return

        full_shape = result.get("full_shape")
        if not full_shape:
            return
        n_sweeps, n_samples = full_shape
        
        if self.mode == "heatmap":
            vmin = result.get("clim", (None, None))[0]
            vmax = result.get("clim", (None, None))[1]
            if vmin is not None and vmax is not None:
                self.img_item.setLevels((vmin, max(vmin + 0.001, vmax)))
            
            clipped_data = result.get("clipped_data")
            if clipped_data is not None:
                self.img_item.setImage(clipped_data, autoLevels=False, levels=(vmin, max(vmin + 0.001, vmax)))
                self.img_item.setVisible(True)
                if self.channel_panels:
                    self.channel_panels[0].has_data = True
                    self.channel_panels[0].empty_text_item.setVisible(False)
                slice_rect = result.get("slice_rect")
                if slice_rect is not None:
                    self.img_item.setRect(QRectF(*slice_rect))
                else:
                    self.img_item.setRect(QRectF(0, 0, n_samples, n_sweeps))
            # If multiple channels are displayed, update auxiliary channel panels
            if len(self.channel_panels) > 1:
                for p in self.channel_panels[1:]:
                    if not p.has_data or not p.channel_name:
                        continue
                    ch_erp = self.get_channel_erp_signal(p.channel_name)
                    if ch_erp is not None and len(ch_erp) > 0:
                        ch_mag = ch_erp.magnitude if hasattr(ch_erp, "magnitude") else ch_erp
                        ch_vmin = p.std * self.lowerSpinBox.value()
                        ch_vmax = p.std * self.upperSpinBox.value()
                        p.img_item.setLevels((ch_vmin, max(ch_vmin + 0.001, ch_vmax)))
                        p.img_item.setImage(ch_mag, autoLevels=False, levels=(ch_vmin, max(ch_vmin + 0.001, ch_vmax)))
        
        elif self.mode == "lines":
            # Lines mode rendering is handled in setup_figure
            pass
        
        # Stim line
        stimno = result.get("stimno")
        if stimno is not None:
            self.plot_curstim_line(stimno)

        rdata = result.get("right_ax_data")
        if rdata:
            self.right_ax_data.update(rdata)
            for k in rdata.keys():
                if k not in self.rightPlots:
                    self.rightPlots[k] = True
            self.plot_right_axis()
        
        # Spike groups handled by separate result
    
    def _on_right_axis_result(self, result: Dict[str, Any], error: str):
        """Handle completed right axis computation."""
        if error or not result:
            return
        
        self.right_ax_data = result
        
        # Update rightPlots dict
        for k in self.right_ax_data.keys():
            if k not in self.rightPlots:
                self.rightPlots[k] = True
        
        self.plot_right_axis()
    
    def _on_spike_group_result(self, result: Dict[int, Dict], error: str):
        """Handle completed spike group computation."""
        if error or not result:
            return

        stim_times_s = self._get_stim_times_s()
        is_time = (
            getattr(self, "y_axis_mode", "sweep") == "time"
            and stim_times_s is not None
            and len(stim_times_s) > 0
        )
        cur_stim = (
            min(
                max(0, self.state.stimno if self.state and self.state.stimno is not None else 0),
                len(stim_times_s) - 1,
            )
            if is_time
            else 0
        )
        t_selected = stim_times_s[cur_stim] if is_time else 0.0
        candidate = None
        candidate_pts = []
        if self.state is not None and hasattr(self.state, "get_active_candidate"):
            candidate = self.state.get_active_candidate()
        if candidate is not None and getattr(candidate.unit, "idx_arr", None) is not None:
            for s_idx, a in enumerate(candidate.unit.idx_arr):
                if a is not None:
                    y_val = (
                        stim_times_s[s_idx] - t_selected
                        if is_time and s_idx < len(stim_times_s)
                        else float(s_idx)
                    )
                    candidate_pts.append((a[0], y_val))
        candidate_arr = (
            np.array(candidate_pts) if candidate_pts else np.empty((0, 2))
        )

        target_unit_idx = self.get_target_unit_idx()

        for panel_idx, panel in enumerate(self.channel_panels):
            # Clear existing scatter items for this panel
            for scat in panel.other_unit_scatters:
                try:
                    panel.plot.removeItem(scat)
                except Exception:
                    pass
            panel.other_unit_scatters.clear()

            for item in panel.legend_items:
                try:
                    panel.plot.removeItem(item)
                except Exception:
                    pass
            panel.legend_items.clear()
            if panel_idx == 0:
                self.legend_items = panel.legend_items
                self.other_unit_scatters = panel.other_unit_scatters

            if not panel.has_data:
                panel.tracked_unit_scatter.clear()
                continue

            # 1. Other Units (only if "Others" checkbox is checked)
            if self.includeAllUnitsCheckBox.isChecked():
                for i, unit in enumerate(self.state.spike_groups):
                    if i == target_unit_idx or getattr(unit, "idx_arr", None) is None:
                        continue
                    if i in result:
                        data = result[i]
                        pts_arr = np.column_stack([data["samples"], data["sweeps"]])
                        if len(pts_arr) > 0:
                            c = SET2_COLORS[i % len(SET2_COLORS)]
                            scat = pg.ScatterPlotItem(
                                x=pts_arr[:, 0],
                                y=pts_arr[:, 1],
                                size=8,
                                symbol="s",
                                pen=None,
                                brush=pg.mkBrush(*c, 200),
                            )
                            panel.plot.addItem(scat)
                            panel.other_unit_scatters.append(scat)

                            if self.showLegend.isChecked():
                                unit_name = unit.event.name or str(i)
                                lbl = pg.TextItem(text=unit_name, color=c, fill=pg.mkBrush(255, 255, 255, 220))
                                lbl.setPos(pts_arr[0, 0], pts_arr[0, 1])
                                panel.plot.addItem(lbl)
                                panel.legend_items.append(lbl)

            # 2. Tracked Unit (Red)
            if self.includeUnitCheckbox.isChecked() and 0 <= target_unit_idx < len(self.state.spike_groups):
                if target_unit_idx in result:
                    data = result[target_unit_idx]
                    pts_arr = np.column_stack([data["samples"], data["sweeps"]])
                    if len(pts_arr) > 0:
                        panel.tracked_unit_scatter.setData(
                            x=pts_arr[:, 0],
                            y=pts_arr[:, 1],
                            size=10,
                            symbol="s",
                            pen=None,
                            brush=pg.mkBrush(255, 0, 0, 220),
                        )
                        panel.tracked_unit_scatter.setVisible(True)

                        if self.showLegend.isChecked():
                            sg = self.state.spike_groups[target_unit_idx]
                            unit_name = sg.event.name or str(target_unit_idx)
                            lbl = pg.TextItem(text=unit_name, color=(255, 0, 0), fill=pg.mkBrush(255, 255, 255, 220))
                            lbl.setPos(pts_arr[0, 0], pts_arr[0, 1])
                            panel.plot.addItem(lbl)
                            panel.legend_items.append(lbl)
            else:
                panel.tracked_unit_scatter.clear()

            # 3. Candidate preview (amber open circles + dashed latency band)
            show_candidate = (
                candidate is not None
                and candidate_arr.size > 0
                and panel.channel_name == candidate.channel
            )
            if show_candidate:
                panel.candidate_scatter.setData(
                    x=candidate_arr[:, 0],
                    y=candidate_arr[:, 1],
                    size=10,
                    symbol="o",
                    pen=pg.mkPen((255, 170, 0), width=1.5),
                    brush=pg.mkBrush(255, 170, 0, 40),
                )
                panel.candidate_scatter.setVisible(True)
                center = float(candidate.latency_center_samples)
                half = max(1.0, float(candidate.latency_tolerance_samples))
                panel.candidate_band.setRegion((center - half, center + half))
                panel.candidate_band.setVisible(True)
            else:
                panel.candidate_scatter.clear()
                panel.candidate_scatter.setVisible(False)
                panel.candidate_band.setVisible(False)


    @quiet_fail
    def update_axis(self):
        if self.state is None or self.state.analog_signal is None or self.state.event_signal is None:
            return
        erp = self.get_erp_signal()
        if erp is None or len(erp) == 0:
            return

        n_sweeps = erp.shape[0] if (erp is not None and getattr(erp, "ndim", 0) >= 2) else (len(self.state.event_signal) if self.state and self.state.event_signal is not None else 1)
        n_samples = erp.shape[1]

        self.main_plot.setXRange(0, n_samples, padding=0)
        self.main_plot.setYRange(0, max(n_sweeps, 1), padding=0)
        self._has_initialized_ranges = True

    @quiet_fail
    def update_ylim(self, curStim: int):
        if self.state is None or self.state.event_signal is None:
            return
        ch_n_sweeps = len(self.state.event_signal)

        if getattr(self, "fixed_height", False):
            if getattr(self, "y_axis_mode", "sweep") == "time":
                stim_times_s = self._get_stim_times_s()
                if stim_times_s is not None and len(stim_times_s) > 0:
                    time_win = float(getattr(self, "time_window_s", 10.0))
                    post_s = float(getattr(self, "post_seconds", 0.0))
                    pad = max(time_win * 0.05, 0.1)
                    y_top = -time_win - pad
                    y_bottom = max(0.0, post_s) + pad
                    self.main_plot.setYRange(y_top, y_bottom, padding=0)
            else:
                pre = int(getattr(self, "n_lines_pre", 5))
                post = int(getattr(self, "n_lines_post", 5))
                start = max(curStim - pre, 0)
                stop = min(start + pre + post + 1, ch_n_sweeps)
                self.main_plot.setYRange(start - 0.5, stop - 0.5, padding=0)
        elif self.lock_to_stim:
            y_range = self.main_plot.viewRange()[1]
            w = max(abs(y_range[1] - y_range[0]) / 2, 2)
            self.main_plot.setYRange(curStim - w, curStim + w, padding=0)

    @Slot()
    def updateAll(self):
        """Update contrast thresholds rapidly via PyQtGraph LUT levels across all channels."""
        if self.mode != "heatmap" or self.state is None:
            return
        for p in self.channel_panels:
            if p.has_data and hasattr(p, "img_item"):
                if getattr(self, "scale_mode", "std") == "mV":
                    vmin = self.lowerSpinBox.value()
                    vmax = self.upperSpinBox.value()
                else:
                    vmin = p.std * self.lowerSpinBox.value()
                    vmax = p.std * self.upperSpinBox.value()
                p.img_item.setLevels((vmin, max(vmin + 0.001, vmax)))

    @quiet_fail
    def setup_figure(self):
        if self.state is None or self.state.analog_signal is None or self.state.event_signal is None:
            return

        self._rebuild_channel_panels()
        N = len(self.channel_panels)
        primary_ch = (
            self.channel_bar.get_primary_channel()
            if hasattr(self, "channel_bar") and self.channel_bar is not None
            else ""
        )

        ref_erp = self.get_erp_signal()
        n_sweeps = 1
        n_samples = 100
        if ref_erp is not None and len(ref_erp) > 0:
            ref_mag = ref_erp.magnitude if hasattr(ref_erp, "magnitude") else ref_erp
            n_sweeps, n_samples = ref_mag.shape
        elif self.state.event_signal is not None:
            n_sweeps = max(len(self.state.event_signal), 1)

        y_range = self.main_plot.viewRange()[1]
        y_min = min(y_range)
        y_max = max(y_range)
        pad = max(10, int(abs(y_max - y_min) * 0.15))

        stim_times_s = self._get_stim_times_s()
        is_time = getattr(self, "y_axis_mode", "sweep") == "time" and stim_times_s is not None and len(stim_times_s) > 0
        cur_stim = min(max(0, self.state.stimno if self.state.stimno is not None else 0), n_sweeps - 1)
        t_selected = stim_times_s[cur_stim] if is_time and cur_stim < len(stim_times_s) else 0.0

        for i, panel in enumerate(self.channel_panels):
            ch_name = panel.channel_name
            title = self._format_channel_title(i, N, ch_name, primary_ch, panel=panel)
            panel.plot.setTitle(title, color="#333", size="9pt")

            erp = self.get_channel_erp_signal(ch_name) if ch_name else None
            if erp is None or len(erp) == 0:
                # Keep empty screen!
                panel.has_data = False
                panel.img_item.setVisible(False)
                panel.lines_curve.setVisible(False)
                panel.active_sweep_curve.setVisible(False)
                panel.highlight_segments.setVisible(False)
                panel.points_scatter.setVisible(False)
                panel.stim_line.setVisible(False)
                panel.tracked_unit_scatter.clear()

                msg = "(Empty Screen)" if not ch_name else f"No data: {ch_name}"
                panel.empty_text_item.setText(msg)
                panel.empty_text_item.setPos(n_samples / 2, n_sweeps / 2)
                panel.empty_text_item.setVisible(True)
                continue

            panel.has_data = True
            panel.empty_text_item.setVisible(False)

            erp_mag = erp.magnitude if hasattr(erp, "magnitude") else erp
            std_val = np.std(erp_mag)
            panel.std = float(std_val.magnitude if hasattr(std_val, "magnitude") else std_val) if std_val > 0 else 1.0
            if i == 0:
                self.std = panel.std

            if getattr(self, "scale_mode", "std") == "mV":
                vmin = self.lowerSpinBox.value()
                vmax = self.upperSpinBox.value()
            else:
                vmin = panel.std * self.lowerSpinBox.value()
                vmax = panel.std * self.upperSpinBox.value()

            ch_n_sweeps, n_pts = erp_mag.shape

            if self.mode == "heatmap":
                panel.lines_curve.setVisible(False)
                panel.active_sweep_curve.setVisible(False)
                panel.highlight_segments.setVisible(False)
                panel.points_scatter.setVisible(False)
                panel.img_item.setVisible(True)
                panel.stim_line.setVisible(True)

                if is_time:
                    if getattr(self, "fixed_height", False):
                        time_window_s = float(getattr(self, "time_window_s", 10.0))
                        post_s = float(getattr(self, "post_seconds", 0.0))
                        t_min = t_selected - time_window_s
                        t_max = t_selected + post_s
                        start = int(np.searchsorted(stim_times_s[:ch_n_sweeps], t_min, side="left"))
                        stop = int(np.searchsorted(stim_times_s[:ch_n_sweeps], t_max, side="right"))
                        start = max(0, min(start, cur_stim))
                        stop = max(cur_stim + 1, min(stop, ch_n_sweeps))

                        span = max(time_window_s + post_s, 0.1)
                        n_bins = int(np.clip(span / 0.02, 100, 1000))
                        dt = span / n_bins
                        time_raster = np.zeros((n_bins, n_pts), dtype=np.float32)
                        for k in range(start, stop):
                            t_k = stim_times_s[k]
                            bin_idx = int(np.clip(np.floor((t_k - t_min) / dt), 0, n_bins - 1))
                            time_raster[bin_idx] = erp_mag[k]
                            if bin_idx + 1 < n_bins and np.all(time_raster[bin_idx + 1] == 0):
                                time_raster[bin_idx + 1] = erp_mag[k]

                        panel.img_item.setImage(time_raster, autoLevels=False, levels=(vmin, max(vmin + 0.001, vmax)))
                        panel.img_item.setRect(QRectF(0, -time_window_s, n_pts, span))
                        panel.stim_line.setValue(0.0)
                    else:
                        t_start = stim_times_s[0] - t_selected
                        t_end = stim_times_s[-1] - t_selected
                        span = max(t_end - t_start, 0.1)
                        panel.img_item.setImage(erp_mag, autoLevels=False, levels=(vmin, max(vmin + 0.001, vmax)))
                        panel.img_item.setRect(QRectF(0, t_start, n_pts, span))
                        panel.stim_line.setValue(0.0)
                else:
                    panel.img_item.setImage(erp_mag, autoLevels=False, levels=(vmin, max(vmin + 0.001, vmax)))
                    panel.img_item.setRect(QRectF(0, 0, erp_mag.shape[1], erp_mag.shape[0]))
                    panel.stim_line.setValue(cur_stim)

            elif self.mode == "lines":
                panel.img_item.setVisible(False)
                panel.stim_line.setVisible(False)
                panel.lines_curve.setVisible(True)
                panel.active_sweep_curve.setVisible(True)
                panel.highlight_segments.setVisible(True)
                panel.points_scatter.setVisible(True)

                scale = float(getattr(self, "scale", 1.0))
                scale_mode = getattr(self, "scale_mode", "std")
                if scale_mode == "std":
                    std_val = float(panel.std) if panel.std > 0 else 1.0
                    eff_scale = scale / (std_val * 8.0)
                else:
                    eff_scale = scale

                if is_time:
                    if getattr(self, "fixed_height", False):
                        time_window_s = float(getattr(self, "time_window_s", 10.0))
                        post_s = float(getattr(self, "post_seconds", 0.0))
                        t_min = t_selected - time_window_s
                        t_max = t_selected + post_s
                        start = int(np.searchsorted(stim_times_s[:ch_n_sweeps], t_min, side="left"))
                        stop = int(np.searchsorted(stim_times_s[:ch_n_sweeps], t_max, side="right"))
                        start = max(0, min(start, cur_stim))
                        stop = max(cur_stim + 1, min(stop, ch_n_sweeps))
                    else:
                        start = 0
                        stop = ch_n_sweeps
                    y_baselines = stim_times_s[start:stop] - t_selected
                    active_idx = cur_stim - start
                else:
                    if getattr(self, "fixed_height", False):
                        pre = int(getattr(self, "n_lines_pre", 5))
                        post = int(getattr(self, "n_lines_post", 5))
                        start = max(cur_stim - pre, 0)
                        stop = min(start + pre + post + 1, ch_n_sweeps)
                    else:
                        slice_start = max(0, int(np.floor(y_min - pad)))
                        slice_stop = min(ch_n_sweeps, int(np.ceil(y_max + pad)))
                        if slice_stop <= slice_start:
                            slice_start = 0
                            slice_stop = ch_n_sweeps
                        start = slice_start
                        stop = slice_stop
                    y_baselines = np.arange(start, stop, dtype=float)
                    active_idx = cur_stim - start if start <= cur_stim < stop else -1

                dpts = erp_mag[start:stop]
                n_display = len(dpts)
                bg_x_list = []
                bg_y_list = []
                active_x = None
                active_y = None
                x = np.arange(n_pts)

                for idx in range(n_display):
                    y = (dpts[idx] * eff_scale * -1.0) + y_baselines[idx]
                    if idx == active_idx:
                        active_x = x
                        active_y = y
                    else:
                        bg_x_list.append(np.append(x, np.nan))
                        bg_y_list.append(np.append(y, np.nan))

                if bg_x_list:
                    panel.lines_curve.setData(
                        np.concatenate(bg_x_list), np.concatenate(bg_y_list),
                        pen=pg.mkPen(color=(150, 150, 150), width=1), connect="finite"
                    )
                else:
                    panel.lines_curve.clear()

                if active_x is not None:
                    panel.active_sweep_curve.setData(
                        active_x, active_y,
                        pen=pg.mkPen(color="purple", width=2)
                    )
                    panel.active_sweep_curve.setVisible(True)
                else:
                    panel.active_sweep_curve.setVisible(False)

                # Spike latency waveform highlights and peak dot
                hl_x = []
                hl_y = []
                scat_pts = []
                target_unit_idx = self.get_target_unit_idx()
                if 0 <= target_unit_idx < len(self.state.spike_groups):
                    ug = self.state.spike_groups[target_unit_idx]
                    idx_arr = getattr(ug, "idx_arr", None)
                    r = 20
                    if idx_arr is not None and len(idx_arr) >= stop:
                        for s_i, s_idx in enumerate(range(start, stop)):
                            entry = idx_arr[s_idx]
                            if entry is None:
                                continue
                            sample_no = int(entry[0])
                            if sample_no < 0 or sample_no >= n_pts:
                                continue
                            r_start = max(0, sample_no - r)
                            r_end = min(n_pts, sample_no + r)
                            seg_x = np.arange(r_start, r_end)
                            seg_y = (dpts[s_i][r_start:r_end] * eff_scale * -1.0) + y_baselines[s_i]
                            hl_x.append(np.append(seg_x, np.nan))
                            hl_y.append(np.append(seg_y, np.nan))
                            scat_pts.append((sample_no, (dpts[s_i][sample_no] * eff_scale * -1.0) + y_baselines[s_i]))

                if hl_x:
                    panel.highlight_segments.setData(
                        np.concatenate(hl_x), np.concatenate(hl_y),
                        pen=pg.mkPen(color="red", width=2), connect="finite"
                    )
                else:
                    panel.highlight_segments.clear()

                if scat_pts:
                    pts_arr = np.array(scat_pts)
                    panel.points_scatter.setData(
                        x=pts_arr[:, 0], y=pts_arr[:, 1],
                        symbol="o", size=8, brush=pg.mkBrush("black")
                    )
                else:
                    panel.points_scatter.clear()

            elif self.mode == "unitonly":
                panel.img_item.setVisible(False)
                panel.lines_curve.setVisible(False)
                panel.active_sweep_curve.setVisible(False)
                panel.highlight_segments.setVisible(False)
                panel.points_scatter.setVisible(False)
                panel.stim_line.setVisible(False)

        if getattr(self, "y_axis_mode", "sweep") == "time":
            self.main_plot.setLabel("left", "Time", units="s")
        else:
            self.main_plot.setLabel("left", "Stimulus #")

        if getattr(self, "fixed_height", False) and self.state and self.state.stimno is not None:
            self.update_ylim(self.state.stimno)

        self._apply_pinned_unit_windows()
        self.plot_curstim_line(self.state.stimno, force=True)
        self.plot_spikegroups()
        self.plot_right_axis()
        self._update_protocol_markers_overlay()
        self._update_live_paused_banner()
        self._last_setup_mode = self.mode

    @quiet_fail
    def plot_curstim_line(self, stimNo: Optional[int] = None, force: bool = False):
        if stimNo is None:
            return
        if (
            not force
            and getattr(self, "_last_curstim_line", None) == stimNo
            and getattr(self, "_last_curstim_mode", None) == self.mode
        ):
            return
        self._last_curstim_line = stimNo
        self._last_curstim_mode = self.mode

        if getattr(self, "fixed_height", False) or getattr(self, "y_axis_mode", "sweep") == "time":
            self.setup_figure()
            return

        for p in self.channel_panels:
            if hasattr(p, "stim_line"):
                p.stim_line.setValue(stimNo)
            if self.mode == "lines" and p.has_data and p.channel_name and self.state is not None:
                ch_erp = self.get_channel_erp_signal(p.channel_name)
                if ch_erp is not None and len(ch_erp) > stimNo >= 0:
                    erp_mag = ch_erp.magnitude if hasattr(ch_erp, "magnitude") else ch_erp
                    scale = float(getattr(self, "scale", 1.0))
                    scale_mode = getattr(self, "scale_mode", "std")
                    if scale_mode == "std":
                        std_val = float(p.std) if p.std > 0 else 1.0
                        eff_scale = scale / (std_val * 8.0)
                    else:
                        eff_scale = scale
                    y = (erp_mag[stimNo] * eff_scale * -1.0) + stimNo
                    x = np.arange(len(y))
                    p.active_sweep_curve.setData(x, y, pen=pg.mkPen(color="purple", width=2))

        for sp in self.summary_panels.values():
            if hasattr(sp, "stim_line") and sp.stim_line is not None:
                sp.stim_line.setValue(stimNo)
        self._apply_pinned_unit_windows()

    @quiet_fail
    def plot_spikegroups(self):
        if self.state is None or self.state.event_signal is None:
            return

        target_unit_idx = self.get_target_unit_idx()
        show_legend = hasattr(self, "showLegend") and self.showLegend.isChecked()
        has_tracked = self.includeUnitCheckbox.isChecked() and 0 <= target_unit_idx < len(self.state.spike_groups)
        sg = self.state.spike_groups[target_unit_idx] if has_tracked else None

        stim_times_s = self._get_stim_times_s()
        is_time = getattr(self, "y_axis_mode", "sweep") == "time" and stim_times_s is not None and len(stim_times_s) > 0
        cur_stim = min(max(0, self.state.stimno if self.state.stimno is not None else 0), len(stim_times_s) - 1) if is_time else 0
        t_selected = stim_times_s[cur_stim] if is_time else 0.0

        tracked_pts = []
        if sg and getattr(sg, "idx_arr", None) is not None:
            for s_idx, a in enumerate(sg.idx_arr):
                if a is not None:
                    y_val = stim_times_s[s_idx] - t_selected if is_time and s_idx < len(stim_times_s) else float(s_idx)
                    tracked_pts.append((a[0], y_val))
        tracked_arr = np.array(tracked_pts) if tracked_pts else np.empty((0, 2))

        other_units_data = []
        if self.includeAllUnitsCheckBox.isChecked():
            for i, unit in enumerate(self.state.spike_groups):
                if i == target_unit_idx or getattr(unit, "idx_arr", None) is None:
                    continue
                points = []
                for s_idx, a in enumerate(unit.idx_arr):
                    if a is not None:
                        y_val = stim_times_s[s_idx] - t_selected if is_time and s_idx < len(stim_times_s) else float(s_idx)
                        points.append((a[0], y_val))
                if not points:
                    continue
                pts_arr = np.array(points)
                c = SET2_COLORS[i % len(SET2_COLORS)]
                u_name = getattr(unit.event, "name", str(i)) or str(i)
                other_units_data.append((i, u_name, c, pts_arr))

        candidate = None
        candidate_pts = []
        if hasattr(self.state, "get_active_candidate"):
            candidate = self.state.get_active_candidate()
        if candidate is not None and getattr(candidate.unit, "idx_arr", None) is not None:
            for s_idx, a in enumerate(candidate.unit.idx_arr):
                if a is not None:
                    y_val = stim_times_s[s_idx] - t_selected if is_time and s_idx < len(stim_times_s) else float(s_idx)
                    candidate_pts.append((a[0], y_val))
        candidate_arr = np.array(candidate_pts) if candidate_pts else np.empty((0, 2))

        for panel_idx, panel in enumerate(self.channel_panels):
            for item in panel.legend_items:
                try:
                    panel.plot.removeItem(item)
                except Exception:
                    pass
            panel.legend_items.clear()
            if panel_idx == 0:
                self.legend_items = panel.legend_items
                self.other_unit_scatters = panel.other_unit_scatters

            if not panel.has_data and panel.channel_name:
                erp_check = self.get_channel_erp_signal(panel.channel_name)
                if erp_check is not None and len(erp_check) > 0:
                    panel.has_data = True
                    panel.empty_text_item.setVisible(False)

            if not panel.has_data:
                panel.tracked_unit_scatter.clear()
                for scat in panel.other_unit_scatters:
                    scat.clear()
                    scat.setVisible(False)
                continue

            # 1. Other Units
            other_idx = 0
            if self.includeAllUnitsCheckBox.isChecked():
                for u_id, u_name, c, pts_arr in other_units_data:
                    if other_idx < len(panel.other_unit_scatters):
                        scat = panel.other_unit_scatters[other_idx]
                        scat.setData(
                            x=pts_arr[:, 0],
                            y=pts_arr[:, 1],
                            size=8,
                            symbol="s",
                            pen=None,
                            brush=pg.mkBrush(*c, 200),
                        )
                        scat.setVisible(True)
                    else:
                        scat = pg.ScatterPlotItem(
                            x=pts_arr[:, 0],
                            y=pts_arr[:, 1],
                            size=8,
                            symbol="s",
                            pen=None,
                            brush=pg.mkBrush(*c, 200),
                        )
                        panel.plot.addItem(scat)
                        panel.other_unit_scatters.append(scat)
                    other_idx += 1

                    if show_legend:
                        lbl = pg.TextItem(text=u_name, color=c, fill=pg.mkBrush(255, 255, 255, 220))
                        lbl.setPos(pts_arr[0, 0], pts_arr[0, 1])
                        panel.plot.addItem(lbl)
                        panel.legend_items.append(lbl)

            for idx_unused in range(other_idx, len(panel.other_unit_scatters)):
                panel.other_unit_scatters[idx_unused].clear()
                panel.other_unit_scatters[idx_unused].setVisible(False)

            # 2. Tracked Unit (Red)
            if has_tracked and len(tracked_arr) > 0:
                panel.tracked_unit_scatter.setData(
                    x=tracked_arr[:, 0],
                    y=tracked_arr[:, 1],
                    size=10,
                    symbol="s",
                    pen=None,
                    brush=pg.mkBrush(255, 0, 0, 220),
                )
                panel.tracked_unit_scatter.setVisible(True)

                if show_legend:
                    unit_name = sg.event.name or str(target_unit_idx)
                    lbl = pg.TextItem(text=unit_name, color=(255, 0, 0), fill=pg.mkBrush(255, 255, 255, 220))
                    lbl.setPos(tracked_arr[0, 0], tracked_arr[0, 1])
                    panel.plot.addItem(lbl)
                    panel.legend_items.append(lbl)
            else:
                panel.tracked_unit_scatter.clear()

            # 3. Candidate preview (amber open circles + dashed latency band)
            show_candidate = (
                candidate is not None
                and candidate_arr.size > 0
                and panel.channel_name == candidate.channel
            )
            if show_candidate:
                panel.candidate_scatter.setData(
                    x=candidate_arr[:, 0],
                    y=candidate_arr[:, 1],
                    size=10,
                    symbol="o",
                    pen=pg.mkPen((255, 170, 0), width=1.5),
                    brush=pg.mkBrush(255, 170, 0, 40),
                )
                panel.candidate_scatter.setVisible(True)
                center = float(candidate.latency_center_samples)
                half = max(1.0, float(candidate.latency_tolerance_samples))
                panel.candidate_band.setRegion((center - half, center + half))
                panel.candidate_band.setVisible(True)
            else:
                panel.candidate_scatter.clear()
                panel.candidate_scatter.setVisible(False)
                panel.candidate_band.setVisible(False)

    def render(self):
        self.plot_spikegroups()

    def _sync_right_axis_data(self, force: bool = False):
        """Synchronously compute right axis summary line data for both static and streaming sessions."""
        if self.state is None:
            return

        data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        # 1. Stimulus frequency
        if self.state.event_signal is not None and len(self.state.event_signal) > 1:
            try:
                times_s = self.state.event_signal.times.rescale(pq.second).magnitude
                diffs = np.diff(times_s)
                stim_freq = np.divide(1.0, diffs, out=np.zeros_like(diffs), where=diffs != 0)
                sweeps = np.arange(1, len(times_s))
                data["Stim. Freq. (Hz)"] = (np.asarray(stim_freq, dtype=np.float32), sweeps)
            except Exception:
                pass

        # 2. Latency differences for target unit
        ug = self.get_target_unit()
        if ug is not None and self.state.event_signal is not None:
            try:
                latencies = ug.get_latencies(self.state.event_signal)
                if latencies is not None and len(latencies) > 0:
                    lat_mag = latencies.magnitude if hasattr(latencies, "magnitude") else latencies
                    idx_na = np.isnan(lat_mag)
                    if np.any(~idx_na):
                        valid_lat = lat_mag[~idx_na]
                        diffs = np.diff(valid_lat)
                        out = np.ones(lat_mag.shape, dtype=np.float32) * np.nan
                        valid_idx = np.where(~idx_na)[0]
                        if len(valid_idx) > 1:
                            out[valid_idx[1:]] = diffs
                        data["Latency Diff."] = (out, np.arange(len(lat_mag)))
            except Exception:
                pass

        # 3. Channel row statistics (Mean or Max)
        show_max = self.toggleMaxMeanButton.isChecked()
        all_chans = self._get_channel_names()
        for ch in self.displayed_channels:
            if ch and ch not in all_chans:
                all_chans.append(ch)

        channel_stats = self._extract_channel_row_stats(all_chans, show_max)
        for ch_name, stat_arr in channel_stats.items():
            if stat_arr is not None and len(stat_arr) > 0:
                data[ch_name] = (stat_arr, np.arange(len(stat_arr)))

        self.right_ax_data = data
        for k in data.keys():
            if k not in self.rightPlots:
                self.rightPlots[k] = True

    def reset_right_axes_data(self):
        """Recompute right axis summary data and schedule background updates."""
        self._sync_right_axis_data(force=True)
        self.plot_right_axis()
        self._schedule_right_axis_update(force=True)

    def plot_right_axis(self):
        if not self.right_ax_data:
            self._sync_right_axis_data()

        active_summary_specs: List[Dict[str, Any]] = []
        show_channel_summaries = self.rightPlots.get("__channel_summaries__", True)
        stat_name = "Max" if self.toggleMaxMeanButton.isChecked() else "Mean"
        primary_ch = (
            self.channel_bar.get_primary_channel()
            if hasattr(self, "channel_bar") and self.channel_bar
            else ""
        )
        if not primary_ch and self.state is not None:
            avail = self.state.get_available_channels()
            if avail:
                primary_ch = avail[0]
            elif self.state.analog_signal is not None:
                primary_ch = getattr(self.state.analog_signal, "name", "")

        # A) Channel summary graphs for each slot in self.displayed_channels (size group_size_n)
        if show_channel_summaries:
            for i in range(self.group_size_n):
                ch_name = self.displayed_channels[i] if i < len(self.displayed_channels) else None
                key = ch_name if ch_name is not None else f"__empty_slot_{i}__"
                if not self.rightPlots.get(key, True):
                    continue

                label = self._format_summary_title(i, self.group_size_n, ch_name, primary_ch, stat_name)
                if ch_name is None:
                    axis_label = f"Slot {i+1} (Empty)"
                    has_data = False
                    xy_data = None
                elif ch_name in self.right_ax_data and len(self.right_ax_data[ch_name][0]) > 0:
                    axis_label = f"{ch_name} ({stat_name})"
                    has_data = True
                    xy_data = self.right_ax_data[ch_name]
                else:
                    axis_label = f"{ch_name} (No data)"
                    has_data = False
                    xy_data = None

                active_summary_specs.append({
                    "key": key,
                    "label": label,
                    "axis_label": axis_label,
                    "is_channel": True,
                    "slot_idx": i,
                    "channel_name": ch_name,
                    "has_data": has_data,
                    "xy_data": xy_data,
                })

        # B) Global metric summary graphs
        for g_key, axis_lbl in [("Stim. Freq. (Hz)", "Freq (Hz)"), ("Latency Diff.", "Diff (s)")]:
            if g_key in self.right_ax_data and self.rightPlots.get(g_key, True):
                xy_data = self.right_ax_data[g_key]
                if xy_data is not None and len(xy_data[0]) > 0:
                    active_summary_specs.append({
                        "key": g_key,
                        "label": g_key,
                        "axis_label": axis_lbl,
                        "is_channel": False,
                        "slot_idx": None,
                        "channel_name": None,
                        "has_data": True,
                        "xy_data": xy_data,
                    })

        start_col = len(self.channel_panels)
        if not active_summary_specs:
            for p in self.right_plots.values():
                try:
                    self.glw.removeItem(p)
                except Exception:
                    pass
            self.right_plots.clear()
            self.summary_panels.clear()
            for c in range(start_col):
                self.glw.ci.layout.setColumnStretchFactor(c, 3)
            self._last_summary_keys = ()
            self.plotsChanged.emit()
            return

        active_keys = [spec["key"] for spec in active_summary_specs]
        cur_stim = self.state.stimno if self.state else 0

        # Fast path: check if existing panels match keys and are attached
        can_fast_path = (
            list(self.summary_panels.keys()) == active_keys
            and all(sp.plot.scene() is not None for sp in self.summary_panels.values())
        )
        if can_fast_path:
            for spec in active_summary_specs:
                sp = self.summary_panels[spec["key"]]
                sp.label = spec["label"]
                sp.plot.setTitle(spec["label"])
                sp.plot.setLabel("bottom", spec["axis_label"])
                sp.has_data = spec["has_data"]
                sp.stim_line.setValue(cur_stim)
                if spec["has_data"]:
                    x_data, y_data = spec["xy_data"]
                    x_val = x_data.magnitude if hasattr(x_data, "magnitude") else x_data
                    y_val = y_data.magnitude if hasattr(y_data, "magnitude") else y_data
                    sp.curve.setData(x=x_val, y=y_val)
                    sp.empty_text_item.setVisible(False)
                else:
                    sp.curve.clear()
                    txt = "(Empty Screen)" if spec["channel_name"] is None else "(No Data)"
                    sp.empty_text_item.setText(txt)
                    sp.empty_text_item.setVisible(True)
            return

        # Slow path: rebuild summary panels in grid layout
        for p in self.right_plots.values():
            try:
                self.glw.removeItem(p)
            except Exception:
                pass
        self.right_plots.clear()
        self.summary_panels.clear()

        for c in range(start_col):
            self.glw.ci.layout.setColumnStretchFactor(c, 3)

        colorwheel = itertools.cycle(["#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02", "#a6761d"])

        for offset, spec in enumerate(active_summary_specs):
            col_idx = start_col + offset
            c = next(colorwheel)
            p = self.glw.addPlot(row=0, col=col_idx)
            p.setYLink(self.main_plot)
            p.invertY(True)
            p.setTitle(spec["label"])
            p.setLabel("bottom", spec["axis_label"], color=c)
            p.getAxis("left").setStyle(showValues=False)
            p.getAxis("left").setPen(c)
            p.showGrid(x=True, y=True, alpha=0.2)
            self.glw.ci.layout.setColumnStretchFactor(col_idx, 1)

            stim_line = pg.InfiniteLine(angle=0, pen=pg.mkPen("purple", width=1.5), movable=False)
            stim_line.setValue(cur_stim)
            p.addItem(stim_line)

            curve = pg.PlotDataItem(pen=pg.mkPen(c, width=1.5), connect="finite")
            p.addItem(curve)

            txt_str = "(Empty Screen)" if spec["channel_name"] is None else ("(No Data)" if not spec["has_data"] else "")
            empty_text = pg.TextItem(txt_str, color=(140, 140, 140), anchor=(0.5, 0.5))
            p.addItem(empty_text)
            empty_text.setPos(0, max(cur_stim, 1))

            if spec["has_data"]:
                x_data, y_data = spec["xy_data"]
                x_val = x_data.magnitude if hasattr(x_data, "magnitude") else x_data
                y_val = y_data.magnitude if hasattr(y_data, "magnitude") else y_data
                curve.setData(x=x_val, y=y_val)
                empty_text.setVisible(False)
            else:
                curve.clear()
                empty_text.setVisible(True)

            panel = SummaryLinePanel(
                key=spec["key"],
                label=spec["label"],
                plot=p,
                curve=curve,
                stim_line=stim_line,
                empty_text_item=empty_text,
                is_channel_stat=spec["is_channel"],
                slot_idx=spec["slot_idx"],
                channel_name=spec["channel_name"],
                has_data=spec["has_data"],
            )
            self.summary_panels[spec["key"]] = panel
            self.right_plots[spec["key"]] = p

        self._last_summary_keys = tuple(self.summary_panels.keys())
        self.plotsChanged.emit()

    # --- PolyLine Selection and Peak Tracking ---

    def toggle_polySelector(self, mode: Optional[bool] = None):
        new_active = mode if mode is not None else (not self.selector_active)
        self.selector_active = new_active
        self.polySelectButton.setChecked(self.selector_active)

        if self.selector_active:
            self.selector_points = []
            self.main_plot.setCursor(Qt.CrossCursor)
            self.main_plot.vb.setMouseEnabled(x=False, y=False)  # disable pan while clicking vertices
            self.dialogPolySelect.show()
            self.dialogPolySelect.change()
        else:
            self.selector_points = []
            self.main_plot.setCursor(Qt.ArrowCursor)
            self.main_plot.vb.setMouseEnabled(x=True, y=True)
            self._redraw_selector()
            self.dialogPolySelect.hide()

    def _on_selector_window_change(self, window_size_ms: float, min_thresh: float):
        if self.state and self.state.analog_signal is not None:
            sr = float(self.state.analog_signal.sampling_rate.magnitude)
            self.selector_width = (window_size_ms / 1000.0) * sr
        else:
            self.selector_width = 100.0
        self._redraw_selector()

    def polySelect(self):
        """Interpolate across drawn selector vertices and detect unit action potentials."""
        if len(self.selector_points) < 2 or self.state is None:
            self.toggle_polySelector(False)
            return

        window_size_ms, min_peak = self.dialogPolySelect.getValues()
        xys = np.array(self.selector_points, dtype=float)

        sr = float(self.state.analog_signal.sampling_rate.magnitude)
        window_size = int(sr * window_size_ms / 1000.0)
        erp = (
            self.get_channel_erp_signal(self.selector_channel_name)
            if self.selector_channel_name
            else None
        )
        if erp is None:
            erp = self.get_erp_signal()
        if erp is None:
            self.toggle_polySelector(False)
            return

        target_unit_idx = self.get_target_unit_idx()

        out = []
        for i in range(len(xys) - 1):
            seg = np.array([xys[i], xys[i + 1]], dtype=float)
            seg = seg[np.argsort(seg[:, 1])]  # sort by sweep index

            start_stim = int(round(seg[0, 1]))
            end_stim = int(round(seg[1, 1]))
            stimpos = np.arange(start_stim, end_stim + 1)
            if len(stimpos) == 0:
                continue

            arr = np.interp(stimpos, seg[:, 1], seg[:, 0])
            for x, stimno in zip(arr, stimpos):
                if stimno >= erp.shape[0] or stimno < 0:
                    continue
                x2 = int(x - (window_size // 2))
                x_start = max(0, x2)
                x_end = min(erp.shape[1], x2 + window_size)
                if x_end <= x_start:
                    continue
                vals = erp[stimno, x_start:x_end]
                x2_offset = int(np.argmax((1 if min_peak >= 0 else -1) * vals))
                peak_val = vals[x2_offset]
                if (1 if min_peak >= 0 else -1) * peak_val > abs(min_peak):
                    out.append((x_start + x2_offset, stimno, peak_val))

        if len(out) > 0:
            out_arr = np.array(out)
            evt = self.state.stimno_offset_to_event(out_arr[:, 1].astype(int), out_arr[:, 0].astype(int))
            self.state.updateUnit(evt, merge=True, unitid=target_unit_idx)

        self.toggle_polySelector(False)

    # --- Deleter Tool ---

    def toggle_deleter(self, mode: Optional[bool] = None):
        if mode is not None:
            active = mode
        else:
            active = not self.dialogDeleteRegion.isVisible()
        if active:
            self.dialogDeleteRegion.show()
            self.dialogDeleteRegion.change()
        else:
            self.dialogDeleteRegion.hide()

    def deleteRegion(self):
        start_event, end_event = self.dialogDeleteRegion.getValues()
        if self.state is None:
            return
        erp = self.get_erp_signal()
        if erp is None:
            return
        target_unit_idx = self.get_target_unit_idx()
        for stimpos in range(int(start_event), int(end_event) + 1):
            if stimpos >= erp.shape[0]:
                continue
            self.state.setUnit(None, stimpos, unit_idx=target_unit_idx)

    # --- Message and Display Toggles ---

    def toggleMaxMean(self, checked: Optional[bool] = None):
        if checked is not None:
            self.toggleMaxMeanButton.setChecked(checked)
        else:
            self.toggleMaxMeanButton.setChecked(not self.toggleMaxMeanButton.isChecked())
        if self.toggleMaxMeanButton.isChecked():
            self.toggleMaxMeanButton.setText("Show Mean")
        else:
            self.toggleMaxMeanButton.setText("Show Max")
        self.reset_right_axes_data()
        self.plotsChanged.emit()

    def toggleShowMessages(self):
        if self.showMessages.isChecked():
            self.showMessages.setText("Hide messages")
        else:
            self.showMessages.setText("Show messages")
        self._update_protocol_markers_overlay()

    def _on_return_to_live_clicked(self):
        if self.state is not None and hasattr(self.state, "set_follow_live"):
            self.state.set_follow_live(True)
        elif hasattr(self, "channel_bar") and self.channel_bar is not None:
            self.channel_bar.set_follow_live_events(True)
        self._update_live_paused_banner()

    def _update_live_paused_banner(self, *args):
        if not hasattr(self, "live_paused_banner"):
            return
        if self.state is None:
            self.live_paused_banner.setVisible(False)
            return
        ev_sig = getattr(self.state, "event_signal", None)
        n_sweeps = len(ev_sig) if ev_sig is not None else 0
        follow_live = getattr(self.state, "follow_live", True)
        if not follow_live and n_sweeps > 0:
            stimno = int(getattr(self.state, "stimno", 0) or 0)
            latest = max(0, n_sweeps - 1)
            self.lbl_live_paused.setText(
                f"⏸ Viewing History (Sweep #{stimno} / #{latest}) —"
            )
            self.live_paused_banner.setVisible(True)
        elif not follow_live:
            self.lbl_live_paused.setText("⏸ Viewing History —")
            self.live_paused_banner.setVisible(True)
        else:
            self.live_paused_banner.setVisible(False)

    def _sampling_rate_hz(self) -> float:
        sr = getattr(self.state, "sampling_rate", None) if self.state is not None else None
        if sr is None and self.state is not None and getattr(self.state, "analog_signal", None) is not None:
            sr = getattr(self.state.analog_signal, "sampling_rate", None)
        if sr is None:
            return 10000.0
        return float(sr.rescale("Hz").magnitude) if hasattr(sr, "rescale") else float(sr)

    def _y_pos_to_stimno(self, y_pos: float) -> Optional[int]:
        if self.state is None or self.state.event_signal is None or len(self.state.event_signal) == 0:
            return None
        n_sweeps = len(self.state.event_signal)
        if getattr(self, "y_axis_mode", "sweep") == "time":
            stim_times_s = self._get_stim_times_s()
            if stim_times_s is not None and len(stim_times_s) > 0:
                cur_stim = min(max(0, int(getattr(self.state, "stimno", 0) or 0)), len(stim_times_s) - 1)
                t_selected = stim_times_s[cur_stim]
                t_clicked = t_selected + y_pos
                closest_stim = int(np.argmin(np.abs(stim_times_s - t_clicked)))
                return max(0, min(n_sweeps - 1, closest_stim))
        stimno = int(round(y_pos))
        return max(0, min(n_sweeps - 1, stimno))

    def _snap_peak_on_sweep(self, sample_x: float, stimno: int, channel_name: Optional[str] = None) -> int:
        erp = self.get_channel_erp_signal(channel_name) if channel_name else self.get_erp_signal()
        if erp is None or len(erp) == 0:
            erp = self.get_erp_signal()
        if erp is None or stimno < 0 or stimno >= len(erp):
            return max(0, int(round(sample_x)))
        row = erp[stimno]
        if hasattr(row, "magnitude"):
            row = row.magnitude
        row = np.asarray(row).ravel()
        n_pts = len(row)
        if n_pts == 0:
            return 0
        x_idx = max(0, min(n_pts - 1, int(round(sample_x))))
        sr = self._sampling_rate_hz()
        half_w = max(1, int(round(0.0015 * sr)))
        lo = max(0, x_idx - half_w)
        hi = min(n_pts, x_idx + half_w + 1)
        if hi > lo:
            x_idx = lo + int(np.argmax(np.abs(row[lo:hi])))
        return int(x_idx)

    def seed_new_unit_at_pos(
        self,
        sample_x: float,
        stimno: Optional[int] = None,
        channel_name: Optional[str] = None,
    ):
        if self.state is None or np.isnan(sample_x):
            return None
        if stimno is None:
            stimno = int(getattr(self.state, "stimno", 0) or 0)
        if not channel_name:
            channel_name = (
                self.channel_bar.get_primary_channel()
                if hasattr(self, "channel_bar") and self.channel_bar is not None
                else getattr(getattr(self.state, "analog_signal", None), "name", "")
            )
        peak_sample = self._snap_peak_on_sweep(sample_x, int(stimno), channel_name)
        if hasattr(self.state, "seed_unit_at"):
            return self.state.seed_unit_at(
                peak_sample,
                stimno=int(stimno),
                channel=channel_name,
                auto_track=True,
            )

        # Fallback when ViewerState does not define seed_unit_at
        self.state.addUnitGroup()
        new_idx = len(self.state.spike_groups) - 1
        if new_idx < 0:
            return None
        sg = self.state.spike_groups[new_idx]
        if channel_name:
            sg.channel = channel_name
            sg.tracking_channel = channel_name
            if hasattr(sg, "event") and hasattr(sg.event, "annotations"):
                sg.event.annotations["channel"] = channel_name
                sg.event.annotations["tracking_channel"] = channel_name
        sg.auto_track = True
        if hasattr(sg, "event") and hasattr(sg.event, "annotations"):
            sg.event.annotations["auto_track"] = True
        if hasattr(self.state, "setUnitGroup"):
            self.state.setUnitGroup(new_idx)
        self.state.setUnit(peak_sample, stimno=int(stimno), unitid=new_idx)
        return new_idx

    def _current_unit_sample_or_center(self, panel: ChannelPlotPanel, stimno: int) -> int:
        ug = self.get_target_unit()
        if ug is not None and getattr(ug, "idx_arr", None):
            if 0 <= stimno < len(ug.idx_arr) and ug.idx_arr[stimno] is not None:
                item = ug.idx_arr[stimno]
                return int(item[0] if isinstance(item, (tuple, list)) else item)
            for item in ug.idx_arr:
                if item is not None:
                    return int(item[0] if isinstance(item, (tuple, list)) else item)
        xr = panel.plot.viewRange()[0] if panel is not None else self.main_plot.viewRange()[0]
        return max(0, int(round((float(xr[0]) + float(xr[1])) / 2.0)))

    def _seed_unit_at_current_sweep(self, panel: ChannelPlotPanel):
        if self.state is None:
            return
        stimno = int(getattr(self.state, "stimno", 0) or 0)
        sample_x = float(self._current_unit_sample_or_center(panel, stimno))
        self.seed_new_unit_at_pos(sample_x, stimno=stimno, channel_name=panel.channel_name)

    def _reanchor_active_unit_at_current_sweep(self, panel: ChannelPlotPanel):
        if self.state is None:
            return
        stimno = int(getattr(self.state, "stimno", 0) or 0)
        unit_idx = self.get_target_unit_idx()
        sample_x = self._current_unit_sample_or_center(panel, stimno)
        peak_sample = self._snap_peak_on_sweep(sample_x, stimno, panel.channel_name)
        if hasattr(self.state, "reanchor_unit_at"):
            self.state.reanchor_unit_at(
                unit_idx,
                peak_sample,
                stimno=stimno,
            )
        else:
            self.state.setUnit(peak_sample, stimno=stimno, unitid=unit_idx)

    def _clear_protocol_markers_overlay(self):
        for item in getattr(self, "message_items", []):
            try:
                if item.scene() is not None:
                    self.main_plot.removeItem(item)
            except Exception:
                pass
        self.message_items.clear()
        for item in getattr(self, "protocol_marker_items", []):
            try:
                if item.scene() is not None:
                    self.main_plot.removeItem(item)
            except Exception:
                pass
        self.protocol_marker_items.clear()

    def _update_protocol_markers_overlay(self, *args):
        if self.state is None or not hasattr(self, "main_plot") or self.main_plot is None:
            return
        self._clear_protocol_markers_overlay()

        markers = []
        if hasattr(self.state, "get_protocol_markers"):
            try:
                markers = list(self.state.get_protocol_markers() or [])
            except Exception:
                markers = []

        show_messages = hasattr(self, "showMessages") and self.showMessages.isChecked()
        if not show_messages and not markers:
            return

        is_time = getattr(self, "y_axis_mode", "sweep") == "time"
        stim_times_s = self._get_stim_times_s() if is_time else None
        cur_stim = int(getattr(self.state, "stimno", 0) or 0)
        t_selected = (
            stim_times_s[min(max(0, cur_stim), len(stim_times_s) - 1)]
            if (is_time and stim_times_s is not None and len(stim_times_s) > 0)
            else 0.0
        )

        # Collect log.other messages when showMessages is checked
        if show_messages and getattr(self.state, "segment", None) is not None and getattr(self.state, "event_signal", None) is not None:
            try:
                ev_other = next(e for e in self.state.segment.events if e.name == "log.other")
                for i, t in enumerate(ev_other):
                    yy = self.state.event_signal.times.searchsorted(t, "right") - 1
                    y_pos = (
                        float(stim_times_s[yy] - t_selected)
                        if (is_time and stim_times_s is not None and 0 <= yy < len(stim_times_s))
                        else float(yy)
                    )
                    msg = ev_other.array_annotations.get("message", [""])[i]
                    txt = pg.TextItem(text=str(msg), color=(0, 0, 0), fill=pg.mkBrush(255, 255, 255, 200))
                    txt.setPos(0, y_pos)
                    self.main_plot.addItem(txt)
                    self.message_items.append(txt)
            except StopIteration:
                pass

        # Render protocol markers (lines + badges)
        for m in markers:
            if isinstance(m, dict):
                m_stim = m.get("stimno")
                m_time = m.get("time_s", m.get("timestamp_s"))
                m_label = str(m.get("label", m.get("message", m.get("name", "Marker"))))
            else:
                m_stim = getattr(m, "stimno", None)
                m_time = getattr(m, "time_s", getattr(m, "timestamp_s", None))
                m_label = str(getattr(m, "label", getattr(m, "message", getattr(m, "name", "Marker"))))

            if is_time:
                if m_time is not None:
                    y_pos = float(m_time) - t_selected
                elif m_stim is not None and stim_times_s is not None and 0 <= int(m_stim) < len(stim_times_s):
                    y_pos = float(stim_times_s[int(m_stim)] - t_selected)
                else:
                    continue
            else:
                if m_stim is not None:
                    y_pos = float(m_stim)
                elif m_time is not None and stim_times_s is not None and len(stim_times_s) > 0:
                    y_pos = float(np.searchsorted(stim_times_s, float(m_time), side="right") - 1)
                else:
                    continue

            line = pg.InfiniteLine(
                angle=0,
                pos=y_pos,
                pen=pg.mkPen((40, 110, 180, 180), width=1.2, style=Qt.DashLine),
                movable=False,
            )
            line.setZValue(14)
            self.main_plot.addItem(line)
            self.protocol_marker_items.append(line)

            badge = pg.TextItem(
                text=m_label,
                color=(20, 60, 120),
                fill=pg.mkBrush(225, 240, 255, 220),
                border=pg.mkPen((40, 110, 180, 180), width=1),
                anchor=(0, 1),
            )
            badge.setPos(0, y_pos)
            badge.setZValue(15)
            self.main_plot.addItem(badge)
            self.protocol_marker_items.append(badge)

    def _on_container_visual_settings_changed(self, settings: Dict[str, Any]):
        if isinstance(settings, dict):
            self.apply_shared_visual_settings(settings)

    def _on_shared_visual_settings_changed(self, *args):
        set_name = None
        settings = None
        if len(args) == 1:
            if isinstance(args[0], dict):
                settings = args[0]
            elif isinstance(args[0], str):
                set_name = args[0]
        elif len(args) >= 2:
            set_name = args[0]
            if isinstance(args[1], dict):
                settings = args[1]

        bound_set = (
            self.channel_bar.get_bound_shared_set()
            if hasattr(self, "channel_bar") and self.channel_bar is not None
            and hasattr(self.channel_bar, "get_bound_shared_set")
            else None
        )
        if set_name is not None and set_name != bound_set:
            return
        if settings is None and self.state is not None and set_name is not None:
            if hasattr(self.state, "get_shared_visual_settings"):
                settings = self.state.get_shared_visual_settings(set_name)
            elif hasattr(self.state, "get_shared_channel_set"):
                s = self.state.get_shared_channel_set(set_name)
                if isinstance(s, dict):
                    settings = s.get("visual_settings")
        if isinstance(settings, dict):
            self.apply_shared_visual_settings(settings)

    def apply_shared_visual_settings(self, settings: Dict[str, Any]):
        if not isinstance(settings, dict) or self._applying_shared_visuals:
            return
        self._applying_shared_visuals = True
        try:
            needs_setup = False
            needs_levels = False

            if "scale_mode" in settings and hasattr(self, "scale_mode_combo"):
                mode = str(settings["scale_mode"])
                if mode != self.scale_mode:
                    self.scale_mode = mode
                    with QSignalBlocker(self.scale_mode_combo):
                        idx = self.scale_mode_combo.findText(mode)
                        if idx >= 0:
                            self.scale_mode_combo.setCurrentIndex(idx)
                    if self.mode == "lines":
                        needs_setup = True
                    else:
                        needs_levels = True

            if "color_scale" in settings and settings["color_scale"] is not None:
                cs = settings["color_scale"]
                if len(cs) == 2 and hasattr(self, "lowerSpinBox") and hasattr(self, "upperSpinBox"):
                    lo, hi = float(cs[0]), float(cs[1])
                    with QSignalBlocker(self.lowerSpinBox), QSignalBlocker(self.upperSpinBox):
                        self.upperSpinBox.setMinimum(0.0)
                        self.lowerSpinBox.setMaximum(max(10.0, hi))
                        self.lowerSpinBox.setValue(lo)
                        self.upperSpinBox.setValue(hi)
                        self.upperSpinBox.setMinimum(lo)
                        self.lowerSpinBox.setMaximum(hi)
                    needs_levels = True

            if "trace_scale" in settings and settings["trace_scale"] is not None and hasattr(self, "scale_spinner"):
                sc = float(settings["trace_scale"])
                if abs(sc - self.scale) > 1e-9:
                    self.scale = sc
                    with QSignalBlocker(self.scale_spinner):
                        self.scale_spinner.setValue(sc)
                    if self.mode == "lines":
                        needs_setup = True

            if "x_range" in settings and settings["x_range"] is not None:
                xr = settings["x_range"]
                if len(xr) == 2 and xr[0] is not None and xr[1] is not None:
                    x0, x1 = float(xr[0]), float(xr[1])
                    if x0 < x1:
                        self.main_plot.setXRange(x0, x1, padding=0)

            if needs_setup:
                self.setup_figure()
            elif needs_levels:
                self.updateAll()
        finally:
            self._applying_shared_visuals = False

    # --- Window State & Settings ---

    def get_settings(self) -> Dict:
        x_range, y_range = self.main_plot.viewRange()
        res = {
            "xlim": x_range,
            "yrange": abs(y_range[1] - y_range[0]),
            "mode": self.mode,
            "include_all_units": self.includeAllUnitsCheckBox.isChecked(),
            "lock_to_stim": self.lock_to_stim,
            "fixed_height": getattr(self, "fixed_height", False),
            "y_axis_mode": getattr(self, "y_axis_mode", "sweep"),
            "n_lines_pre": getattr(self, "n_lines_pre", 5),
            "n_lines_post": getattr(self, "n_lines_post", 5),
            "time_window_s": getattr(self, "time_window_s", 10.0),
            "post_seconds": getattr(self, "post_seconds", 0.0),
            "scale": getattr(self, "scale", 1.0),
            "scale_mode": getattr(self, "scale_mode", "std"),
            "link_x_axis": getattr(self, "link_x_axis", True),
            "percentiles": (self.lowerSpinBox.value(), self.upperSpinBox.value()),
            "group_size_n": self.group_size_n,
            "displayed_channels": list(self.displayed_channels),
            "right_plots": dict(self.rightPlots),
            "show_max": self.toggleMaxMeanButton.isChecked(),
            "pinned_units": {p.slot_idx: p.pinned_unit_idx for p in self.channel_panels if getattr(p, "pinned_unit_idx", None) is not None},
            "window_groups": {p.slot_idx: p.window_group for p in self.channel_panels if getattr(p, "window_group", None) is not None},
            "window_widths": {p.slot_idx: p.window_width_ms for p in self.channel_panels},
        }
        if hasattr(self, "channel_bar"):
            res["channel_bar"] = self.channel_bar.get_settings()
        return res

    def set_settings(self, values: Dict):
        if "channel_bar" in values and hasattr(self, "channel_bar"):
            self.channel_bar.set_settings(values["channel_bar"])
        if "group_size_n" in values:
            self.group_size_n = int(values["group_size_n"])
            if hasattr(self, "spin_group_size"):
                self.spin_group_size.blockSignals(True)
                self.spin_group_size.setValue(self.group_size_n)
                self.spin_group_size.blockSignals(False)
            if hasattr(self, "groupSizeSpinBox"):
                self.groupSizeSpinBox.blockSignals(True)
                self.groupSizeSpinBox.setValue(self.group_size_n)
                self.groupSizeSpinBox.blockSignals(False)
        if "displayed_channels" in values:
            self.displayed_channels = list(values["displayed_channels"])
        if "right_plots" in values:
            self.rightPlots = dict(values["right_plots"])
        if "show_max" in values:
            self.toggleMaxMeanButton.setChecked(values["show_max"])
            self.toggleMaxMeanButton.setText("Show Mean" if values["show_max"] else "Show Max")
        if "xlim" in values:
            self.main_plot.setXRange(values["xlim"][0], values["xlim"][1], padding=0)
        if "mode" in values:
            self.mode = values["mode"]
        if "scale" in values:
            self.scale = float(values["scale"])
            if hasattr(self, "scale_spinner"):
                self.scale_spinner.blockSignals(True)
                self.scale_spinner.setValue(self.scale)
                self.scale_spinner.blockSignals(False)
        if "scale_mode" in values:
            self.scale_mode = values["scale_mode"]
            if hasattr(self, "scale_mode_combo"):
                idx = self.scale_mode_combo.findText(self.scale_mode)
                if idx >= 0:
                    self.scale_mode_combo.blockSignals(True)
                    self.scale_mode_combo.setCurrentIndex(idx)
                    self.scale_mode_combo.blockSignals(False)
        if "fixed_height" in values:
            self.fixed_height = bool(values["fixed_height"])
            if hasattr(self, "fixed_height_checkbox"):
                self.fixed_height_checkbox.blockSignals(True)
                self.fixed_height_checkbox.setChecked(self.fixed_height)
                self.fixed_height_checkbox.blockSignals(False)
        if "y_axis_mode" in values:
            self.y_axis_mode = values["y_axis_mode"]
            if hasattr(self, "y_axis_combo"):
                idx = self.y_axis_combo.findText("Time" if self.y_axis_mode == "time" else "Sweep")
                if idx >= 0:
                    self.y_axis_combo.blockSignals(True)
                    self.y_axis_combo.setCurrentIndex(idx)
                    self.y_axis_combo.blockSignals(False)
        if "n_lines_pre" in values:
            self.n_lines_pre = int(values["n_lines_pre"])
            if hasattr(self, "pre_spinner"):
                self.pre_spinner.setValue(self.n_lines_pre)
        if "n_lines_post" in values:
            self.n_lines_post = int(values["n_lines_post"])
            if hasattr(self, "post_spinner"):
                self.post_spinner.setValue(self.n_lines_post)
        if "time_window_s" in values:
            self.time_window_s = float(values["time_window_s"])
            if hasattr(self, "time_window_spinner"):
                self.time_window_spinner.setValue(self.time_window_s)
        if "post_seconds" in values:
            self.post_seconds = float(values["post_seconds"])
        if "link_x_axis" in values:
            self.link_x_axis = bool(values["link_x_axis"])
            if hasattr(self, "link_x_checkbox"):
                self.link_x_checkbox.setChecked(self.link_x_axis)
        if "include_all_units" in values:
            self.includeAllUnitsCheckBox.setChecked(values["include_all_units"])
        if "lock_to_stim" in values:
            self.lock_to_stim = values["lock_to_stim"]
            self.lock_to_stimCheckBox.setChecked(values["lock_to_stim"])
        if "percentiles" in values:
            self.lowerSpinBox.setValue(values["percentiles"][0])
            self.upperSpinBox.setValue(values["percentiles"][1])
        if "yrange" in values and self.state is not None:
            cur = max(self.state.stimno - abs(values["yrange"]) / 2, 0)
            self.main_plot.setYRange(cur, cur + abs(values["yrange"]), padding=0)

        # Restore per-panel pinning / grouping
        if "pinned_units" in values:
            for s_idx, u_idx in values["pinned_units"].items():
                s_int = int(s_idx)
                if s_int < len(self.channel_panels):
                    self.channel_panels[s_int].pinned_unit_idx = u_idx
                    if u_idx is not None:
                        self.channel_panels[s_int].x_linked = False
        if "window_groups" in values:
            for s_idx, g_name in values["window_groups"].items():
                s_int = int(s_idx)
                if s_int < len(self.channel_panels):
                    self.channel_panels[s_int].window_group = g_name
        if "window_widths" in values:
            for s_idx, w_ms in values["window_widths"].items():
                s_int = int(s_idx)
                if s_int < len(self.channel_panels):
                    self.channel_panels[s_int].window_width_ms = float(w_ms)

        self._update_axis_linking()
        self._update_toolbar_visibility()
        self.setup_figure()
        self.plotsChanged.emit()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_P:
            self.toggle_polySelector()
        elif e.key() == Qt.Key_Escape:
            if self.selector_active:
                self.toggle_polySelector(False)
        elif e.key() in (Qt.Key_Return, Qt.Key_Enter):
            if self.selector_active:
                self.polySelect()
        super().keyPressEvent(e)

    def closeEvent(self, event):
        self.remove_references()
        # Unregister from task manager
        if hasattr(self, "_task_manager"):
            self._task_manager.unregister_viewer(self._viewer_id)
        if hasattr(self, "dialogPolySelect") and self.dialogPolySelect is not None:
            self.dialogPolySelect.close()
        if hasattr(self, "settingsDialog") and self.settingsDialog is not None:
            self.settingsDialog.close()
        if getattr(self, "plot_manager_dialog", None) is not None:
            try:
                self.plot_manager_dialog.close()
            except RuntimeError:
                pass
        if hasattr(self, "dialogDeleteRegion") and self.dialogDeleteRegion is not None:
            self.dialogDeleteRegion.close()
        super().closeEvent(event)
