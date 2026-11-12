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
from .ChannelGroupBar import ChannelGroupBar
from .helpers import qsignal_throttle


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
        self._dirty: bool = False
        self._was_shown: bool = False
        self._user_hidden: bool = False
        self.on_new_sweep_throttled = qsignal_throttle(self._on_new_sweep_core, 50)

        # Plot items
        self.all_snippets_curve = pg.PlotDataItem(pen=pg.mkPen(color=(160, 160, 160, 80), width=1), connect="finite")
        self.mean_curve = pg.PlotDataItem(pen=pg.mkPen("black", width=2, style=Qt.DashLine))
        self.selected_curve = pg.PlotDataItem(pen=pg.mkPen("purple", width=2.5))
        self.zero_line = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen("black", width=1, style=Qt.DashLine))

        # Independent channel group and unit binding toolbar
        self.channel_bar = ChannelGroupBar(parent=self, state=state)
        self.channel_bar.bindingChanged.connect(self._on_channel_bar_changed)
        self.addToolBar(Qt.TopToolBarArea, self.channel_bar)

        self._init_ui()

        if state is not None:
            self.set_state(state)

    def _on_channel_bar_changed(self):
        if hasattr(self, "channel_bar"):
            summary = self.channel_bar.get_title_summary()
            self.setWindowTitle(f"UnitView (PyQtGraph) {summary}")
            p = self.parentWidget()
            if p is not None and hasattr(p, "setWindowTitle"):
                p.setWindowTitle(f"UnitView (PyQtGraph) {summary}")
        self.update_all()

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
        if hasattr(self, "channel_bar") and self.channel_bar is not None:
            self.channel_bar.setState(state)
        if self.state is None:
            return

        self.connections.append(self.state.onStimNoChange.connect(self.update_curstim_line))
        self.connections.append(self.state.onUnitChange.connect(self.update_all))
        self.connections.append(self.state.onUnitGroupChange.connect(self.update_all))
        self.connections.append(self.state.onLoadNewFile.connect(self.update_all))
        if hasattr(self.state, "onChannelChange"):
            self.connections.append(self.state.onChannelChange.connect(self.update_all))
        if hasattr(self.state, "onNewSweep"):
            self.connections.append(self.state.onNewSweep.connect(self.on_new_sweep_throttled))
        if hasattr(self.state, "onNewSpikeEvents"):
            self.connections.append(self.state.onNewSpikeEvents.connect(self.on_new_sweep_throttled))

        self.update_all()

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
            self.update_all()

    def hideEvent(self, event):
        super().hideEvent(event)
        self._user_hidden = True

    def showEvent(self, event):
        super().showEvent(event)
        self._was_shown = True
        self._user_hidden = False
        if getattr(self, "_dirty", False):
            self._dirty = False
            self.update_all()

    def _on_new_sweep_core(self, *args):
        self.update_all()

    def on_new_sweep(self, *args):
        if hasattr(self, "on_new_sweep_throttled"):
            self.on_new_sweep_throttled(*args)
        else:
            self._on_new_sweep_core(*args)

    def unset_connections(self):
        if self.state is not None:
            for con in self.connections:
                try:
                    self.state.disconnect(con)
                except Exception:
                    pass
        self.connections.clear()

    @Slot()
    def update_all(self, force: bool = False):
        if self.is_view_culled():
            self._dirty = True
            return
        self._dirty = False

        if self.state is None or self.state.analog_signal is None:
            return

        sr = float(self.state.sampling_rate.rescale("Hz").magnitude) if hasattr(self.state.sampling_rate, "rescale") else float(self.state.sampling_rate)
        self.w = int(sr * 0.002)
        xarr = np.arange(-self.w, self.w)

        sg = self.channel_bar.get_target_unit() if hasattr(self, "channel_bar") else self.state.getUnitGroup()
        if sg is None:
            return
        primary_name = self.channel_bar.get_primary_channel() if hasattr(self, "channel_bar") else getattr(self.state.analog_signal, "name", "")
        sig_erp = self.state.get_channel_erp(primary_name) if hasattr(self.state, "get_channel_erp") else self.state.get_erp()
        if sig_erp is None or len(sig_erp) == 0:
            return

        idx_arr = sg.idx_arr
        ug_idx = self.channel_bar.get_target_unit_idx() if hasattr(self, "channel_bar") else getattr(self.state, "cur_spike_group", 0)
        ug_spikes = len(idx_arr) if idx_arr is not None else 0
        render_key = (primary_name, ug_idx, ug_spikes, len(sig_erp))
        if not force and getattr(self, "_last_render_key", None) == render_key:
            self.update_curstim_line(self.state.stimno)
            return
        self._last_render_key = render_key

        self.lines = {}
        all_x = []
        all_y = []
        mean_arr = np.zeros(len(xarr), dtype=float)
        count_snippets = 0

        min_y = -0.1
        max_y = 0.1

        if idx_arr is None:
            self.all_snippets_curve.clear()
            self.mean_curve.clear()
            self.selected_curve.clear()
            return

        idx_arr_list = list(idx_arr)
        for i, idx, d in zip(range(len(sig_erp)), idx_arr_list, sig_erp):
            if idx is None:
                continue

            xmin = idx[0] - self.w
            xmax = idx[0] + self.w
            w_left = max(0, -xmin)
            w_right = max(0, xmax - len(d))
            w_span = [-self.w + w_left, self.w - w_right]
            xmin = max(0, xmin)
            xmax = min(len(d), xmax)

            ydata = d[xmin:xmax]
            if hasattr(ydata, "magnitude"):
                ydata = ydata.magnitude
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
            self.mean_curve.clear()
            self.mean_curve.setVisible(False)

        self.main_plot.setXRange(-self.w, self.w, padding=0.05)
        self.main_plot.setYRange(min_y, max_y, padding=0.05)

        self.update_curstim_line(self.state.stimno, force=True)

    @Slot(int)
    def update_curstim_line(self, stimno: int, force: bool = False):
        if not force and getattr(self, "_last_highlighted_stim", None) == stimno:
            return
        self._last_highlighted_stim = stimno
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
        if not hasattr(self.main_plot, "vb") or not self.main_plot.sceneBoundingRect().contains(event.scenePos()):
            return

        pos = self.main_plot.vb.mapSceneToView(event.scenePos())
        xdata = pos.x()
        ydata = pos.y()
        if np.isnan(xdata) or np.isnan(ydata):
            return

        closest_line = None
        closest_dist = float("inf")

        # Fast O(N) lookup by vertical distance near clicked sample
        target_sample = int(round(xdata))
        for i, line in self.lines.items():
            if len(line) == 0:
                continue
            x_col = line[:, 0]
            idx = np.argmin(np.abs(x_col - target_sample))
            if abs(x_col[idx] - target_sample) <= 3:
                dist = abs(line[idx, 1] - ydata)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_line = i

        if closest_line is not None and self.state.event_signal is not None:
            if 0 <= closest_line < len(self.state.event_signal):
                self.state.setStimNo(closest_line)

    def closeEvent(self, event):
        self.unset_connections()
        super().closeEvent(event)
