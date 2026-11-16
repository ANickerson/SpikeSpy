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
    QApplication,
)
import pyqtgraph as pg
import matplotlib.pyplot as plt

from .ViewerState import ViewerState, tracked_neuron_unit
from .ChannelGroupBar import ChannelGroupBar
from .MultiTraceView import DialogSignalSelect, DialogPolySelect, DialogDeleteRegion
from .helpers import qsignal_throttle, quiet_fail
from .live_orchestrator import LiveFrameConsumerMixin
from .viewer_worker import get_task_manager, TaskType, TaskPriority

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
    ):
        self.slot_idx: int = slot_idx
        self.channel_name: Optional[str] = channel_name
        self.plot: pg.PlotItem = plot
        self.img_item: pg.ImageItem = img_item
        self.lines_curve: pg.PlotDataItem = lines_curve
        self.active_sweep_curve: pg.PlotDataItem = active_sweep_curve
        self.stim_line: pg.InfiniteLine = stim_line
        self.tracked_unit_scatter: pg.ScatterPlotItem = tracked_unit_scatter
        self.other_unit_scatters: List[pg.ScatterPlotItem] = []
        self.legend_items: List[pg.TextItem] = []
        self.empty_text_item: pg.TextItem = empty_text_item
        self.time_axis: Optional[EphysTimeAxisItem] = time_axis
        self.std: float = 1.0
        self.has_data: bool = False


class MultiTracePGView(LiveFrameConsumerMixin, QMainWindow):
    """
    High-performance PyQtGraph port of MultiTraceView.
    Delivers 60 FPS GPU-accelerated waterfall heatmaps, multi-trace visualization,
    and interactive click-to-draw polygon corridor spike tracking.
    """

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
        self.scale_mode: str = "std"
        self.std: float = 1.0
        self.rightPlots: Dict[str, bool] = {}
        self.right_plots: Dict[str, pg.PlotItem] = {}
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

        self.settingsDialog = DialogSignalSelect()

        if state is not None:
            self.set_state(state)

        self.setup_figure()

    def _init_ui(self):
        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

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

        # Contrast Min/Max Spinners
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

        toolbar.addWidget(QLabel("Colour scale: "))
        self.scale_mode_combo = QComboBox(self)
        self.scale_mode_combo.addItems(["std", "mV"])
        self.scale_mode_combo.setCurrentText(getattr(self, "scale_mode", "std"))
        self.scale_mode_combo.currentTextChanged.connect(self._on_scale_mode_changed)
        toolbar.addWidget(self.scale_mode_combo)
        toolbar.addWidget(QLabel("Min"))
        toolbar.addWidget(self.lowerSpinBox)
        toolbar.addWidget(QLabel("Max"))
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

        # Mode Radio Buttons
        self.lock_to_stimCheckBox = QCheckBox("lock")
        self.lock_to_stimCheckBox.stateChanged.connect(self._on_lock_changed)

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
            self.setup_figure()

        self.butgrp.buttonToggled.connect(on_mode_toggle)

        toolbar.addWidget(QLabel("Mode:"))
        toolbar.addWidget(self.lock_to_stimCheckBox)
        toolbar.addWidget(linesRadio)
        toolbar.addWidget(heatmapRadio)
        toolbar.addWidget(unitOnlyRadio)
        toolbar.addSeparator()

        # Channel Display Controls (Add, Remove, Group Size N, Presets)
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
        self.settingsButton.clicked.connect(lambda: self.settingsDialog.show())
        toolbar.addWidget(self.settingsButton)

        self.toggleMaxMeanButton = QPushButton("Show max", self)
        self.toggleMaxMeanButton.setCheckable(True)
        self.toggleMaxMeanButton.clicked.connect(self.toggleMaxMean)
        toolbar.addWidget(self.toggleMaxMeanButton)

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

        # Independent channel group and unit binding toolbar
        self.channel_bar = ChannelGroupBar(parent=self, state=self.state)
        self.channel_bar.bindingChanged.connect(self._on_channel_bar_changed)
        self.addToolBar(Qt.TopToolBarArea, self.channel_bar)
        self.addToolBarBreak()
        self.addToolBar(Qt.TopToolBarArea, toolbar)

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

        if hasattr(self, "channel_bar") and self.channel_bar is not None:
            menu.addSeparator()
            act_prev = menu.addAction("Shift Group -1 (<)")
            act_prev.triggered.connect(lambda: self.channel_bar.shift_group(-1))
            act_next = menu.addAction("Shift Group +1 (>)")
            act_next.triggered.connect(lambda: self.channel_bar.shift_group(1))

        menu.exec(screen_pos)

    def _format_channel_title(
        self, slot_idx: int, total_n: int, ch_name: Optional[str], primary_ch: str = ""
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
        return f"{star}[{slot_label}] {ch_name}{status}"

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
                title = self._format_channel_title(i, target_n, p.channel_name, primary_ch)
                p.plot.setTitle(title, color="#333", size="9pt")
            return

        # Count changed - remove right plots and channel panels from glw
        for p in self.right_plots.values():
            try:
                self.glw.removeItem(p)
            except Exception:
                pass
        self.right_plots.clear()

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
                plot.setLabel("left", "Stimulus #")
            else:
                plot.setYLink(self.channel_panels[0].plot)
                plot.setXLink(self.channel_panels[0].plot)
                plot.getAxis("left").setStyle(showValues=False)

            title = self._format_channel_title(i, target_n, ch_name, primary_ch)
            plot.setTitle(title, color="#333", size="9pt")

            # 1. Heatmap ImageItem
            img_item = pg.ImageItem(axisOrder="row-major")
            img_item.setLookupTable(self._cmap_lut)
            plot.addItem(img_item)

            # 2. Lines mode curves
            lines_curve = pg.PlotDataItem(pen=pg.mkPen(color=(128, 128, 128), width=1), connect="finite")
            active_sweep_curve = pg.PlotDataItem(pen=pg.mkPen(color="purple", width=2))
            plot.addItem(lines_curve)
            plot.addItem(active_sweep_curve)
            lines_curve.setVisible(False)
            active_sweep_curve.setVisible(False)

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
            )
            self.channel_panels.append(panel)

        # Backward compatibility aliases
        self.main_plot = self.channel_panels[0].plot
        self.img_item = self.channel_panels[0].img_item
        self.lines_curve = self.channel_panels[0].lines_curve
        self.active_sweep_curve = self.channel_panels[0].active_sweep_curve
        self.stim_line = self.channel_panels[0].stim_line
        self.tracked_unit_scatter = self.channel_panels[0].tracked_unit_scatter
        self.other_unit_scatters = self.channel_panels[0].other_unit_scatters
        self.time_axis = self.channel_panels[0].time_axis

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

    def clear_slot_to_empty(self, slot_idx: int):
        """Clear a specific slot to an empty screen without changing N."""
        if 0 <= slot_idx < len(self.displayed_channels):
            self.displayed_channels[slot_idx] = None
            self._sync_linked_channel_group()
            self.setup_figure()
            self.reset_right_axes_data()

    def add_empty_slot(self):
        """Add an empty channel screen slot to the group."""
        self.displayed_channels.append(None)
        self.group_size_n = len(self.displayed_channels)
        if hasattr(self, "groupSizeSpinBox"):
            with QSignalBlocker(self.groupSizeSpinBox):
                self.groupSizeSpinBox.setValue(self.group_size_n)
        self.setup_figure()

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

    def _set_channel_as_primary(self, slot_idx: int):
        """Move the channel at slot_idx to position 0 (primary)."""
        if 0 < slot_idx < len(self.displayed_channels):
            ch = self.displayed_channels.pop(slot_idx)
            self.displayed_channels.insert(0, ch)
            self._sync_linked_channel_group()
            self.setup_figure()
            self.reset_right_axes_data()

    def _replace_slot_channel(self, slot_idx: int, new_ch: str):
        """Replace channel at slot_idx with new_ch."""
        if 0 <= slot_idx < len(self.displayed_channels):
            self.displayed_channels[slot_idx] = new_ch
            self._sync_linked_channel_group()
            self.setup_figure()
            self.reset_right_axes_data()

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
        if self.state is None or not getattr(self.state, "is_streaming", False):
            return None
        primary = (
            self.channel_bar.get_primary_channel()
            if hasattr(self, "channel_bar") and self.channel_bar is not None
            else getattr(getattr(self.state, "analog_signal", None), "name", None)
        )
        if hasattr(self.state, "_live_aux_erp_buffers") and primary in self.state._live_aux_erp_buffers:
            return self.state._live_aux_erp_buffers[primary]
        if self.state.analog_signal is not None and primary == self.state.analog_signal.name:
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

    def _on_plot_clicked(self, event):
        """Handle clicks on any channel plot for sweep navigation, channel context menu, or selector point placement."""
        scene_pos = event.scenePos()
        target_panel = None
        for p in self.channel_panels:
            if p.plot.vb.sceneBoundingRect().contains(scene_pos):
                target_panel = p
                break
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

        # Normal mode: single left-click selects stimulus sweep
        if event.button() == Qt.LeftButton:
            if not np.isnan(pos.y()):
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
        self._connect_state(self.state.onStimNoChange, self._on_stim_no_changed)

        if hasattr(self.state, "onChannelChange"):
            self._connect_state(self.state.onChannelChange, self._on_channel_change)

        # Live sweeps arrive through the state's LiveOrchestrator (LIFO);
        # the GUI-side throttle keeps render scheduling rate-limited.
        if hasattr(self.state, "onNewSweep"):
            self._register_live_frames(state, lambda _stimno: self.on_new_sweep_throttled())
        if hasattr(self.state, "onNewSpikeEvents"):
            self._connect_state(self.state.onNewSpikeEvents, self._on_new_spike_events)

        self.update_axis()
        self._schedule_right_axis_update(force=True)
        self.update_window_title()

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
        if self.state is None or not getattr(self.state, "is_streaming", False):
            return stats
        for ch_name in ch_names:
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
        if hasattr(self, "channel_bar") and self.channel_bar is not None:
            chans = self.channel_bar.get_selected_channels()
            names.extend(chans)
        elif self.state.analog_signal is not None:
            names.append(self.state.analog_signal.name)
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
        
        # Spike groups handled by separate result
    
    def _on_right_axis_result(self, result: Dict[str, Any], error: str):
        """Handle completed right axis computation."""
        if error:
            return
        if not result:
            return
        
        self.right_ax_data = result
        
        # Update rightPlots dict
        for k in self.right_ax_data.keys():
            if k not in self.rightPlots:
                self.rightPlots[k] = True
        
        # Recreate settings dialog if needed
        if self.settingsDialog is None:
            self.settingsDialog = DialogSignalSelect(options=self.rightPlots)

            def updateView(k, v):
                self.rightPlots[k] = v
                self.plot_right_axis()
                self.render()
            
            self.settingsDialog.changeSelection.connect(updateView)
        
        self.plot_right_axis()
    
    def _on_spike_group_result(self, result: Dict[int, Dict], error: str):
        """Handle completed spike group computation."""
        if error or not result:
            return

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
                else:
                    panel.tracked_unit_scatter.clear()
            else:
                panel.tracked_unit_scatter.clear()


    @quiet_fail
    def update_axis(self):
        if self.state is None or self.state.analog_signal is None or self.state.event_signal is None:
            return
        erp = self.get_erp_signal()
        if erp is None or len(erp) == 0:
            return

        n_sweeps = len(self.state.event_signal) if not getattr(self.state, "is_streaming", False) else erp.shape[0]
        n_samples = erp.shape[1]

        self.main_plot.setXRange(0, n_samples, padding=0)
        self.main_plot.setYRange(0, max(n_sweeps, 1), padding=0)
        self._has_initialized_ranges = True

    @quiet_fail
    def update_ylim(self, curStim: int):
        if self.lock_to_stim and self.state and self.state.event_signal is not None:
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

        for i, panel in enumerate(self.channel_panels):
            ch_name = panel.channel_name
            title = self._format_channel_title(i, N, ch_name, primary_ch)
            panel.plot.setTitle(title, color="#333", size="9pt")

            erp = self.get_channel_erp_signal(ch_name) if ch_name else None
            if erp is None or len(erp) == 0:
                # Keep empty screen!
                panel.has_data = False
                panel.img_item.setVisible(False)
                panel.lines_curve.setVisible(False)
                panel.active_sweep_curve.setVisible(False)
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

            if self.mode == "heatmap":
                panel.lines_curve.setVisible(False)
                panel.active_sweep_curve.setVisible(False)
                panel.img_item.setVisible(True)
                panel.stim_line.setVisible(True)

                panel.img_item.setImage(erp_mag, autoLevels=False, levels=(vmin, max(vmin + 0.001, vmax)))
                panel.img_item.setRect(QRectF(0, 0, erp_mag.shape[1], erp_mag.shape[0]))

            elif self.mode == "lines":
                panel.img_item.setVisible(False)
                panel.stim_line.setVisible(False)
                panel.lines_curve.setVisible(True)
                panel.active_sweep_curve.setVisible(True)

                ch_n_sweeps, n_pts = erp_mag.shape
                slice_start = max(0, int(np.floor(y_min - pad)))
                slice_stop = min(ch_n_sweeps, int(np.ceil(y_max + pad)))
                if slice_stop <= slice_start:
                    slice_start = 0
                    slice_stop = ch_n_sweeps

                p90 = float(panel.std.magnitude if hasattr(panel.std, "magnitude") else panel.std) * 10
                norm_erp = np.clip(erp_mag[slice_start:slice_stop], -p90, p90) / (p90 * 2)
                n_sub_sweeps = norm_erp.shape[0]
                ys = (norm_erp * -1) + (np.arange(n_sub_sweeps) + slice_start)[:, np.newaxis]
                x = np.arange(n_pts)

                all_x = np.tile(np.append(x, np.nan), n_sub_sweeps)
                all_y = np.column_stack([ys, np.full(n_sub_sweeps, np.nan)]).ravel()
                panel.lines_curve.setData(
                    all_x, all_y, pen=pg.mkPen(color=(150, 150, 150), width=1), connect="finite"
                )

                # Active sweep line
                stimno = min(max(self.state.stimno, 0), ch_n_sweeps - 1)
                panel.active_sweep_curve.setData(
                    x,
                    (np.clip(erp_mag[stimno], -p90, p90) / (p90 * 2) * -1) + stimno,
                    pen=pg.mkPen(color="purple", width=2),
                )

            elif self.mode == "unitonly":
                panel.img_item.setVisible(False)
                panel.lines_curve.setVisible(False)
                panel.active_sweep_curve.setVisible(False)
                panel.stim_line.setVisible(False)

        self.plot_curstim_line(self.state.stimno, force=True)
        self.plot_spikegroups()
        self.plot_right_axis()
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

        for p in self.channel_panels:
            if hasattr(p, "stim_line"):
                p.stim_line.setValue(stimNo)
            if self.mode == "lines" and p.has_data and p.channel_name and self.state is not None:
                ch_erp = self.get_channel_erp_signal(p.channel_name)
                if ch_erp is not None and len(ch_erp) > stimNo >= 0:
                    erp_mag = ch_erp.magnitude if hasattr(ch_erp, "magnitude") else ch_erp
                    p90 = float(p.std.magnitude if hasattr(p.std, "magnitude") else p.std) * 10
                    norm_erp = np.clip(erp_mag[stimNo], -p90, p90) / (p90 * 2)
                    y = (norm_erp * -1) + stimNo
                    x = np.arange(len(y))
                    p.active_sweep_curve.setData(x, y, pen=pg.mkPen(color="purple", width=2))

    @quiet_fail
    def plot_spikegroups(self):
        if self.state is None or self.state.event_signal is None:
            return

        target_unit_idx = self.get_target_unit_idx()
        show_legend = hasattr(self, "showLegend") and self.showLegend.isChecked()
        has_tracked = self.includeUnitCheckbox.isChecked() and 0 <= target_unit_idx < len(self.state.spike_groups)
        sg = self.state.spike_groups[target_unit_idx] if has_tracked else None
        tracked_pts = (
            [(a[0], s_idx) for s_idx, a in enumerate(sg.idx_arr) if a is not None]
            if sg and getattr(sg, "idx_arr", None) is not None
            else []
        )
        tracked_arr = np.array(tracked_pts) if tracked_pts else np.empty((0, 2))

        other_units_data = []
        if self.includeAllUnitsCheckBox.isChecked():
            for i, unit in enumerate(self.state.spike_groups):
                if i == target_unit_idx or getattr(unit, "idx_arr", None) is None:
                    continue
                points = [(a[0], s_idx) for s_idx, a in enumerate(unit.idx_arr) if a is not None]
                if not points:
                    continue
                pts_arr = np.array(points)
                c = SET2_COLORS[i % len(SET2_COLORS)]
                u_name = getattr(unit.event, "name", str(i)) or str(i)
                other_units_data.append((i, u_name, c, pts_arr))

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

    def render(self):
        self.plot_spikegroups()

    def reset_right_axes_data(self):
        """Schedule right axis data computation to background thread."""
        self._schedule_right_axis_update(force=True)

    def plot_right_axis(self):
        active_axes = [(k, v) for k, v in self.right_ax_data.items() if self.rightPlots.get(k, False)]
        start_col = len(self.channel_panels)
        if not active_axes:
            for p in self.right_plots.values():
                try:
                    self.glw.removeItem(p)
                except Exception:
                    pass
            self.right_plots.clear()
            for c in range(start_col):
                self.glw.ci.layout.setColumnStretchFactor(c, 3)
            return

        active_labels = [label for label, data in active_axes]
        if list(self.right_plots.keys()) == active_labels:
            for label, data in active_axes:
                p = self.right_plots[label]
                x_data = data[0].magnitude if hasattr(data[0], "magnitude") else data[0]
                y_data = data[1].magnitude if hasattr(data[1], "magnitude") else data[1]
                items = p.listDataItems()
                if items:
                    items[0].setData(x=x_data, y=y_data)
                else:
                    p.plot(x=x_data, y=y_data)
            return

        for p in self.right_plots.values():
            try:
                self.glw.removeItem(p)
            except Exception:
                pass
        self.right_plots.clear()

        for c in range(start_col):
            self.glw.ci.layout.setColumnStretchFactor(c, 3)
        colorwheel = itertools.cycle(["#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02", "#a6761d"])

        for offset, (label, data) in enumerate(active_axes):
            col_idx = start_col + offset
            c = next(colorwheel)
            p = self.glw.addPlot(row=0, col=col_idx)
            p.setYLink(self.main_plot)
            p.invertY(True)
            x_data = data[0].magnitude if hasattr(data[0], "magnitude") else data[0]
            y_data = data[1].magnitude if hasattr(data[1], "magnitude") else data[1]
            p.plot(x=x_data, y=y_data, pen=pg.mkPen(c, width=1.5))
            p.setLabel("bottom", label, color=c)
            p.getAxis("left").setStyle(showValues=False)
            p.getAxis("left").setPen(c)
            self.glw.ci.layout.setColumnStretchFactor(col_idx, 1)
            self.right_plots[label] = p

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

    def toggleMaxMean(self):
        if self.toggleMaxMeanButton.isChecked():
            self.toggleMaxMeanButton.setText("Show Mean")
        else:
            self.toggleMaxMeanButton.setText("Show Max")
        self.reset_right_axes_data()

    def toggleShowMessages(self):
        if self.state is None or self.state.segment is None:
            return
        try:
            ev_other = next(e for e in self.state.segment.events if e.name == "log.other")
        except StopIteration:
            return

        if self.showMessages.isChecked():
            self.showMessages.setText("Hide messages")
            for item in self.message_items:
                self.main_plot.removeItem(item)
            self.message_items.clear()

            for i, t in enumerate(ev_other):
                yy = self.state.event_signal.times.searchsorted(t, "right") - 1
                msg = ev_other.array_annotations.get("message", [""])[i]
                txt = pg.TextItem(text=str(msg), color=(0, 0, 0), fill=pg.mkBrush(255, 255, 255, 200))
                txt.setPos(0, yy)
                self.main_plot.addItem(txt)
                self.message_items.append(txt)
        else:
            self.showMessages.setText("Show messages")
            for item in self.message_items:
                self.main_plot.removeItem(item)
            self.message_items.clear()

    # --- Window State & Settings ---

    def get_settings(self) -> Dict:
        x_range, y_range = self.main_plot.viewRange()
        res = {
            "xlim": x_range,
            "yrange": abs(y_range[1] - y_range[0]),
            "mode": self.mode,
            "include_all_units": self.includeAllUnitsCheckBox.isChecked(),
            "lock_to_stim": self.lock_to_stim,
            "percentiles": (self.lowerSpinBox.value(), self.upperSpinBox.value()),
            "group_size_n": self.group_size_n,
            "displayed_channels": list(self.displayed_channels),
            "scale_mode": getattr(self, "scale_mode", "std"),
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
        if "displayed_channels" in values:
            self.displayed_channels = list(values["displayed_channels"])
        if "xlim" in values:
            self.main_plot.setXRange(values["xlim"][0], values["xlim"][1], padding=0)
        if "mode" in values:
            self.mode = values["mode"]
        if "scale_mode" in values:
            self.scale_mode = values["scale_mode"]
            if hasattr(self, "scale_mode_combo"):
                idx = self.scale_mode_combo.findText(self.scale_mode)
                if idx >= 0:
                    self.scale_mode_combo.blockSignals(True)
                    self.scale_mode_combo.setCurrentIndex(idx)
                    self.scale_mode_combo.blockSignals(False)
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
        self.setup_figure()

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
        if hasattr(self, "dialogPolySelect"):
            self.dialogPolySelect.close()
        if hasattr(self, "settingsDialog"):
            self.settingsDialog.close()
        if hasattr(self, "dialogDeleteRegion"):
            self.dialogDeleteRegion.close()
        super().closeEvent(event)
