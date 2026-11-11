import itertools
import sys
from typing import Dict, List, Optional
import numpy as np
import quantities as pq
from neo import Event

import PySide6
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, Signal, Slot, QTimer, QRectF, QPointF
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QToolBar,
    QLabel,
    QDoubleSpinBox,
    QCheckBox,
    QButtonGroup,
    QRadioButton,
    QPushButton,
    QApplication,
)
import pyqtgraph as pg
import matplotlib.pyplot as plt

from .ViewerState import ViewerState, tracked_neuron_unit
from .MultiTraceView import DialogSignalSelect, DialogPolySelect, DialogDeleteRegion


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


class MultiTracePGView(QMainWindow):
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
        self.std: float = 1.0
        self.rightPlots: Dict[str, bool] = {}
        self.right_plots: Dict[str, pg.PlotItem] = {}
        self.references: List = []
        self.legend_items: List[pg.TextItem] = []
        self.message_items: List[pg.TextItem] = []
        self.other_unit_scatters: List[pg.ScatterPlotItem] = []

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

        # Custom Time Axis (ms)
        self.time_axis = EphysTimeAxisItem(
            lambda: self.state.sampling_rate if self.state else None,
            orientation="bottom",
        )
        self.main_plot: pg.PlotItem = self.glw.addPlot(row=0, col=0, axisItems={"bottom": self.time_axis})
        self.main_plot.invertY(True)  # Sweep 0 at top, sweep N at bottom
        self.main_plot.setLabel("bottom", "Time", units="ms")
        self.main_plot.setLabel("left", "Stimulus #")
        self.main_plot.showGrid(x=True, y=True, alpha=0.3)
        self.glw.ci.layout.setColumnStretchFactor(0, 3)

        # 1. Heatmap ImageItem (axisOrder='row-major': [row, col] -> [y, x])
        self.img_item = pg.ImageItem(axisOrder="row-major")
        self.img_item.setLookupTable(self._cmap_lut)
        self.main_plot.addItem(self.img_item)

        # 2. Lines mode curves
        self.lines_curve = pg.PlotDataItem(pen=pg.mkPen(color=(128, 128, 128), width=1), connect="finite")
        self.active_sweep_curve = pg.PlotDataItem(pen=pg.mkPen(color="purple", width=2))
        self.main_plot.addItem(self.lines_curve)
        self.main_plot.addItem(self.active_sweep_curve)
        self.lines_curve.setVisible(False)
        self.active_sweep_curve.setVisible(False)

        # 3. Current stimulus cursor line
        self.stim_line = pg.InfiniteLine(angle=0, pen=pg.mkPen("purple", width=1.5), movable=False)
        self.main_plot.addItem(self.stim_line)

        # 4. Spike group scatter points
        self.tracked_unit_scatter = pg.ScatterPlotItem(size=8, symbol="s", pen=None, brush=pg.mkBrush(255, 0, 0, 200))
        self.main_plot.addItem(self.tracked_unit_scatter)

        # 5. Interactive Selector Graphics (Corridor polygon + Polyline with handles)
        self.corridor_item = QtWidgets.QGraphicsPolygonItem()
        self.corridor_item.setPen(pg.mkPen("purple", width=1, style=Qt.DashLine))
        self.corridor_item.setBrush(pg.mkBrush(160, 32, 240, 50))
        self.corridor_item.setZValue(20)
        self.main_plot.addItem(self.corridor_item)
        self.corridor_item.setVisible(False)

        self.selector_line_item = pg.PlotDataItem(
            pen=pg.mkPen("purple", width=2),
            symbol="o",
            symbolSize=7,
            symbolBrush=pg.mkBrush("purple"),
            symbolPen=pg.mkPen("w", width=1),
        )
        self.selector_line_item.setZValue(21)
        self.main_plot.addItem(self.selector_line_item)
        self.selector_line_item.setVisible(False)

        # Handle mouse clicks and movement
        self.main_plot.scene().sigMouseClicked.connect(self._on_plot_clicked)
        self.main_plot.scene().sigMouseMoved.connect(self._on_plot_mouse_moved)

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

        toolbar.addWidget(QLabel("Colour scale (std): "))
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

        self.addToolBar(Qt.TopToolBarArea, toolbar)

    def _on_lock_changed(self):
        self.lock_to_stim = self.lock_to_stimCheckBox.isChecked()
        if self.state and self.state.stimno is not None:
            self.update_ylim(self.state.stimno)

    def _on_plot_clicked(self, event):
        """Handle clicks on the plot for both normal sweep navigation and selector point placement."""
        pos = self.main_plot.vb.mapSceneToView(event.scenePos())

        # If selector is active, left-click adds a vertex, right-click removes the last vertex
        if self.selector_active:
            if event.button() == Qt.LeftButton:
                self.selector_points.append([float(pos.x()), float(pos.y())])
                self._redraw_selector()
                event.accept()
                return
            elif event.button() == Qt.RightButton:
                if self.selector_points:
                    self.selector_points.pop()
                    self._redraw_selector()
                    event.accept()
                    return

        # Normal mode: single left-click selects stimulus sweep
        if event.button() == Qt.LeftButton:
            stimno = int(round(pos.y()))
            if self.state and self.state.event_signal is not None:
                if 0 <= stimno < len(self.state.event_signal):
                    self.state.setStimNo(stimno)

    def _on_plot_mouse_moved(self, scene_pos):
        """Draw dynamic preview from the last point to cursor when selector is active."""
        if not self.selector_active or len(self.selector_points) == 0:
            return
        pos = self.main_plot.vb.mapSceneToView(scene_pos)
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

    def set_state(self, state: ViewerState):
        self.remove_references()
        self.state = state
        if self.state is None:
            return

        self.references.append(self.state.onLoadNewFile.connect(self.reset_right_axes_data))
        self.references.append(self.state.onLoadNewFile.connect(self.setup_figure))
        self.references.append(self.state.onLoadNewFile.connect(self.update_axis))

        self.references.append(self.state.onUnitGroupChange.connect(lambda *args: self.render()))
        self.references.append(self.state.onUnitChange.connect(lambda *args: self.render()))
        self.references.append(self.state.onUnitGroupChange.connect(lambda *args: self.reset_right_axes_data()))

        self.references.append(self.state.onStimNoChange.connect(self.update_ylim))
        self.references.append(self.state.onStimNoChange.connect(lambda *args: self.plot_curstim_line(self.state.stimno)))
        self.references.append(self.state.onUnitChange.connect(lambda x: self.reset_right_axes_data()))

        self.update_axis()
        self.reset_right_axes_data()

    def remove_references(self):
        if self.state is not None:
            for ref in self.references:
                try:
                    self.state.disconnect(ref)
                except Exception:
                    pass
            self.references.clear()

    def update_axis(self):
        if self.state is None or self.state.analog_signal is None or self.state.event_signal is None:
            return
        erp = self.state.analog_signal_erp
        if erp is None or len(erp) == 0:
            return

        n_sweeps = len(self.state.event_signal)
        n_samples = erp.shape[1]

        self.main_plot.setXRange(0, n_samples, padding=0)
        self.main_plot.setYRange(0, n_sweeps, padding=0)

    def update_ylim(self, curStim: int):
        if self.lock_to_stim and self.state and self.state.event_signal is not None:
            y_range = self.main_plot.viewRange()[1]
            w = max(abs(y_range[1] - y_range[0]) / 2, 2)
            self.main_plot.setYRange(curStim - w, curStim + w, padding=0)

    @Slot()
    def updateAll(self):
        """Update contrast thresholds rapidly via PyQtGraph LUT levels."""
        if self.mode != "heatmap" or self.state is None:
            return
        vmin = self.std * self.lowerSpinBox.value()
        vmax = self.std * self.upperSpinBox.value()
        self.img_item.setLevels((vmin, max(vmin + 0.001, vmax)))

    def setup_figure(self):
        if self.state is None or self.state.analog_signal is None or self.state.event_signal is None:
            return

        erp = self.state.get_erp()
        if erp is None or len(erp) == 0:
            return

        self.std = float(np.std(erp)) if np.std(erp) > 0 else 1.0
        vmin = self.std * self.lowerSpinBox.value()
        vmax = self.std * self.upperSpinBox.value()

        if self.mode == "heatmap":
            self.lines_curve.setVisible(False)
            self.active_sweep_curve.setVisible(False)
            self.img_item.setVisible(True)
            self.stim_line.setVisible(True)

            self.img_item.setImage(erp, levels=(vmin, max(vmin + 0.001, vmax)))
            self.img_item.setRect(QRectF(0, 0, erp.shape[1], erp.shape[0]))

        elif self.mode == "lines":
            self.img_item.setVisible(False)
            self.stim_line.setVisible(False)
            self.lines_curve.setVisible(True)
            self.active_sweep_curve.setVisible(True)

            # Stacked waveform traces
            p90 = self.std * 10
            norm_erp = np.clip(erp, -p90, p90) / (p90 * 2)
            n_sweeps, n_pts = norm_erp.shape
            ys = (norm_erp * -1) + np.arange(n_sweeps)[:, np.newaxis]
            x = np.arange(n_pts)

            # Vectorized line concatenation with NaN separators for single-draw performance
            all_x = np.tile(np.append(x, np.nan), n_sweeps)
            all_y = np.column_stack([ys, np.full(n_sweeps, np.nan)]).ravel()
            self.lines_curve.setData(all_x, all_y, pen=pg.mkPen(color=(150, 150, 150), width=1), connect="finite")

            # Active sweep line
            stimno = min(max(self.state.stimno, 0), n_sweeps - 1)
            self.active_sweep_curve.setData(x, ys[stimno], pen=pg.mkPen(color="purple", width=2))

        elif self.mode == "unitonly":
            self.img_item.setVisible(False)
            self.lines_curve.setVisible(False)
            self.active_sweep_curve.setVisible(False)
            self.stim_line.setVisible(False)

        self.plot_curstim_line(self.state.stimno)
        self.plot_spikegroups()
        self.plot_right_axis()

    def plot_curstim_line(self, stimNo: Optional[int] = None):
        if stimNo is None:
            return
        self.stim_line.setValue(stimNo)
        if self.mode == "lines" and self.state is not None:
            erp = self.state.get_erp()
            if erp is not None and len(erp) > stimNo >= 0:
                p90 = self.std * 10
                norm_erp = np.clip(erp[stimNo], -p90, p90) / (p90 * 2)
                y = (norm_erp * -1) + stimNo
                x = np.arange(len(y))
                self.active_sweep_curve.setData(x, y, pen=pg.mkPen(color="purple", width=2))

    def plot_spikegroups(self):
        if self.state is None or self.state.event_signal is None:
            return

        for item in self.legend_items:
            self.main_plot.removeItem(item)
        self.legend_items.clear()

        for scat in self.other_unit_scatters:
            self.main_plot.removeItem(scat)
        self.other_unit_scatters.clear()

        # 1. Other Units
        if self.includeAllUnitsCheckBox.isChecked():
            for i, unit in enumerate(self.state.spike_groups):
                points = [(a[0], s_idx) for s_idx, a in enumerate(unit.idx_arr) if a is not None]
                if not points:
                    continue
                pts_arr = np.array(points)
                c = SET2_COLORS[i % len(SET2_COLORS)]

                if i == self.state.cur_spike_group:
                    continue

                scat = pg.ScatterPlotItem(
                    x=pts_arr[:, 0],
                    y=pts_arr[:, 1],
                    size=8,
                    symbol="s",
                    pen=None,
                    brush=pg.mkBrush(*c, 200),
                )
                self.main_plot.addItem(scat)
                self.other_unit_scatters.append(scat)

                if self.showLegend.isChecked():
                    unit_name = unit.event.name or str(i)
                    lbl = pg.TextItem(text=unit_name, color=c, fill=pg.mkBrush(255, 255, 255, 220))
                    lbl.setPos(pts_arr[0, 0], pts_arr[0, 1])
                    self.main_plot.addItem(lbl)
                    self.legend_items.append(lbl)

        # 2. Tracked Unit (Red)
        if self.includeUnitCheckbox.isChecked() and self.state.cur_spike_group < len(self.state.spike_groups):
            sg = self.state.getUnitGroup(self.state.cur_spike_group)
            points = [(a[0], s_idx) for s_idx, a in enumerate(sg.idx_arr) if a is not None]
            if points:
                pts_arr = np.array(points)
                self.tracked_unit_scatter.setData(
                    x=pts_arr[:, 0],
                    y=pts_arr[:, 1],
                    size=10,
                    symbol="s",
                    pen=None,
                    brush=pg.mkBrush(255, 0, 0, 220),
                )
                self.tracked_unit_scatter.setVisible(True)

                if self.showLegend.isChecked():
                    unit_name = sg.event.name or str(self.state.cur_spike_group)
                    lbl = pg.TextItem(text=unit_name, color=(255, 0, 0), fill=pg.mkBrush(255, 255, 255, 220))
                    lbl.setPos(pts_arr[0, 0], pts_arr[0, 1])
                    self.main_plot.addItem(lbl)
                    self.legend_items.append(lbl)
            else:
                self.tracked_unit_scatter.clear()
        else:
            self.tracked_unit_scatter.clear()

    def render(self):
        self.plot_spikegroups()

    def reset_right_axes_data(self):
        if self.state is None or self.state.event_signal is None or len(self.state.event_signal) < 2:
            return

        stimFreq_data = (
            1 / np.diff(self.state.event_signal.times).rescale(pq.second),
            np.arange(1, len(self.state.event_signal.times)),
        )
        ug = self.state.getUnitGroup()
        latencies = ug.get_latencies(self.state.event_signal)
        idx_na = np.isnan(latencies)

        current_spike_diffs = np.diff(latencies[~idx_na])
        out = np.ones(latencies.shape) * np.nan * pq.ms
        if len(np.where(~idx_na)[0]) > 1:
            out[np.where(~idx_na)[0][1:]] = current_spike_diffs

        self.right_ax_data = {
            "Stim. Freq. (Hz)": stimFreq_data,
            "Latency Diff.": (out, np.arange(len(latencies))),
        }

        if self.state.segment and hasattr(self.state.segment, "analogsignals"):
            for x in self.state.segment.analogsignals:
                try:
                    erp_sig = self.state.get_erp(x, self.state.event_signal)
                    if self.toggleMaxMeanButton.isChecked():
                        val = np.max(erp_sig, axis=1)
                    else:
                        val = np.mean(erp_sig, axis=1)
                    self.right_ax_data[x.name] = (val, np.arange(len(self.state.event_signal.times)))
                except Exception:
                    pass

        if not self.rightPlots:
            self.rightPlots = {k: True for k in self.right_ax_data.keys()}

        if self.settingsDialog is not None:
            self.settingsDialog.destroy()

        self.settingsDialog = DialogSignalSelect(options=self.rightPlots)

        def updateView(k, v):
            self.rightPlots[k] = v
            self.plot_right_axis()

        self.settingsDialog.changeSelection.connect(updateView)
        self.plot_right_axis()

    def plot_right_axis(self):
        for p in self.right_plots.values():
            self.glw.removeItem(p)
        self.right_plots.clear()

        active_axes = [(k, v) for k, v in self.right_ax_data.items() if self.rightPlots.get(k, False)]
        if not active_axes:
            self.glw.ci.layout.setColumnStretchFactor(0, 1)
            return

        self.glw.ci.layout.setColumnStretchFactor(0, 3)
        colorwheel = itertools.cycle(["#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02", "#a6761d"])

        for col_idx, (label, data) in enumerate(active_axes, start=1):
            c = next(colorwheel)
            p = self.glw.addPlot(row=0, col=col_idx)
            p.setYLink(self.main_plot)
            p.invertY(True)
            p.plot(x=data[0], y=data[1], pen=pg.mkPen(c, width=1.5))
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
        erp = self.state.get_erp()
        if erp is None:
            self.toggle_polySelector(False)
            return

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
            self.state.updateUnit(evt, merge=True)

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
        erp = self.state.get_erp()
        if erp is None:
            return
        for stimpos in range(int(start_event), int(end_event) + 1):
            if stimpos >= erp.shape[0]:
                continue
            self.state.setUnit(None, stimpos)

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
        return {
            "xlim": x_range,
            "yrange": abs(y_range[1] - y_range[0]),
            "mode": self.mode,
            "include_all_units": self.includeAllUnitsCheckBox.isChecked(),
            "lock_to_stim": self.lock_to_stim,
            "percentiles": (self.lowerSpinBox.value(), self.upperSpinBox.value()),
        }

    def set_settings(self, values: Dict):
        if "xlim" in values:
            self.main_plot.setXRange(values["xlim"][0], values["xlim"][1], padding=0)
        if "mode" in values:
            self.mode = values["mode"]
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
        if hasattr(self, "dialogPolySelect"):
            self.dialogPolySelect.close()
        if hasattr(self, "settingsDialog"):
            self.settingsDialog.close()
        if hasattr(self, "dialogDeleteRegion"):
            self.dialogDeleteRegion.close()
        super().closeEvent(event)
