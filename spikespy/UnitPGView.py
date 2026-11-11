import sys
from typing import Dict, List, Optional
import numpy as np
import quantities as pq

import PySide6
from PySide6 import QtCore
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QApplication,
)
import pyqtgraph as pg

from .ViewerState import ViewerState


class UnitPGView(QMainWindow):
    """
    High-performance PyQtGraph port of UnitView.
    Displays aligned spike waveform snippets, mean waveform, and active sweep highlight.
    """

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        state: Optional[ViewerState] = None,
    ):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setFocusPolicy(Qt.ClickFocus)

        self.state: Optional[ViewerState] = None
        self.connections: List = []
        self.lines: Dict[int, np.ndarray] = {}
        self.w: int = 0

        # Plot items
        self.all_snippets_curve = pg.PlotDataItem(pen=pg.mkPen(color=(160, 160, 160, 80), width=1), connect="finite")
        self.mean_curve = pg.PlotDataItem(pen=pg.mkPen("black", width=2, style=Qt.DashLine))
        self.selected_curve = pg.PlotDataItem(pen=pg.mkPen("purple", width=2.5))
        self.zero_line = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen("black", width=1, style=Qt.DashLine))

        self._init_ui()

        if state is not None:
            self.set_state(state)

    def _init_ui(self):
        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground("w")

        self.main_plot: pg.PlotItem = self.glw.addPlot(row=0, col=0)
        self.main_plot.setLabel("bottom", "Relative Time", units="samples")
        self.main_plot.setLabel("left", "Amplitude")
        self.main_plot.showGrid(x=True, y=True, alpha=0.3)

        self.main_plot.addItem(self.all_snippets_curve)
        self.main_plot.addItem(self.zero_line)
        self.main_plot.addItem(self.mean_curve)
        self.main_plot.addItem(self.selected_curve)

        self.main_plot.scene().sigMouseClicked.connect(self._on_click)

        layout.addWidget(self.glw)
        self.setCentralWidget(central)

    def set_state(self, state: ViewerState):
        self.unset_connections()
        self.state = state
        if self.state is None:
            return

        self.connections.append(self.state.onStimNoChange.connect(self.update_curstim_line))
        self.connections.append(self.state.onUnitChange.connect(self.update_all))
        self.connections.append(self.state.onUnitGroupChange.connect(self.update_all))
        self.connections.append(self.state.onLoadNewFile.connect(self.update_all))

        self.update_all()

    def unset_connections(self):
        if self.state is not None:
            for con in self.connections:
                try:
                    self.state.disconnect(con)
                except Exception:
                    pass
        self.connections.clear()

    @Slot()
    def update_all(self):
        if self.state is None or self.state.analog_signal is None:
            return

        sr = float(self.state.sampling_rate)
        self.w = int(sr * 0.002)
        xarr = np.arange(-self.w, self.w)

        sg = self.state.getUnitGroup()
        sig_erp = self.state.analog_signal_erp
        if sig_erp is None or len(sig_erp) == 0:
            return

        idx_arr = sg.idx_arr
        self.lines = {}
        all_x = []
        all_y = []
        mean_arr = np.zeros(len(xarr), dtype=float)
        count_snippets = 0

        min_y = -0.1
        max_y = 0.1

        for i, idx, d in zip(range(len(sig_erp)), idx_arr, sig_erp):
            if idx is None:
                continue

            w_span = [-self.w, self.w]
            xmin = idx[0] - self.w
            if xmin < 0:
                w_span[0] = abs(xmin) + 1
                xmin = 0

            xmax = idx[0] + self.w
            if xmax > len(d):
                w_span[1] = abs(xmax - len(d))
                xmax = len(d)

            ydata = d[xmin:xmax]
            seg_x = np.arange(w_span[0], w_span[1])
            if len(seg_x) != len(ydata) or len(ydata) == 0:
                continue

            try:
                mean_arr[w_span[0] + self.w : w_span[1] + self.w] += ydata
                count_snippets += 1
            except Exception:
                pass

            self.lines[i] = np.vstack((seg_x, ydata)).T
            all_x.append(np.append(seg_x, np.nan))
            all_y.append(np.append(ydata, np.nan))

            min_y = min(min_y, float(np.min(ydata)))
            max_y = max(max_y, float(np.max(ydata)))

        # Plot all background snippets in one fast draw call
        if all_x:
            self.all_snippets_curve.setData(np.concatenate(all_x), np.concatenate(all_y))
        else:
            self.all_snippets_curve.clear()

        # Plot mean waveform
        if count_snippets > 0:
            self.mean_curve.setData(xarr, mean_arr / count_snippets)
            self.mean_curve.setVisible(True)
        else:
            self.mean_curve.setVisible(False)

        self.main_plot.setXRange(-self.w, self.w, padding=0.05)
        self.main_plot.setYRange(min_y, max_y, padding=0.05)

        self.update_curstim_line(self.state.stimno)

    @Slot(int)
    def update_curstim_line(self, stimno: int):
        if stimno in self.lines:
            line_data = self.lines[stimno]
            self.selected_curve.setData(line_data[:, 0], line_data[:, 1])
            self.selected_curve.setVisible(True)
        else:
            self.selected_curve.clear()
            self.selected_curve.setVisible(False)

    def _on_click(self, event):
        if event.button() != Qt.LeftButton or not self.lines or self.state is None:
            return

        pos = self.main_plot.vb.mapSceneToView(event.scenePos())
        xdata = pos.x()
        ydata = pos.y()

        closest_line = None
        closest_dist = float("inf")

        # Find closest snippet trace to mouse click
        for i, line in self.lines.items():
            for j in range(len(line) - 1):
                p1, p2 = line[j], line[j + 1]
                line_vec = p2 - p1
                point_vec = np.array([xdata, ydata]) - p1
                line_len = np.dot(line_vec, line_vec)
                if line_len == 0:
                    dist = float(np.linalg.norm(point_vec))
                else:
                    t = max(0.0, min(1.0, float(np.dot(point_vec, line_vec) / line_len)))
                    projection = p1 + t * line_vec
                    dist = float(np.linalg.norm(np.array([xdata, ydata]) - projection))

                if dist < closest_dist:
                    closest_dist = dist
                    closest_line = i

        if closest_line is not None:
            self.state.setStimNo(closest_line)

    def closeEvent(self, event):
        self.unset_connections()
        super().closeEvent(event)
