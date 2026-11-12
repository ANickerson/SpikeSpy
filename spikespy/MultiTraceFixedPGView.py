import sys
from typing import Dict, List, Optional
import numpy as np
import quantities as pq

import PySide6
from PySide6 import QtCore
from PySide6.QtCore import Qt, Signal, Slot, QTimer
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QToolBar,
    QLabel,
    QDoubleSpinBox,
    QSpinBox,
    QApplication,
)
import pyqtgraph as pg

from .ViewerState import ViewerState, tracked_neuron_unit
from .ChannelGroupBar import ChannelGroupBar


class FixedTimeAxisItem(pg.AxisItem):
    """Custom AxisItem converting seconds to milliseconds."""

    def tickStrings(self, values, scale, spacing):
        return [f"{1000 * v:g}" for v in values]


class MultiTraceFixedPGView(QMainWindow):
    """
    High-performance PyQtGraph port of MultiTraceFixedView.
    Displays N-pre and N-post sweeps around the active stimulus sweep.
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
        self.settings: Dict = {
            "n_lines_pre": 5,
            "n_lines_post": 5,
            "scale": 1.0,
        }

        # Graphics items
        self.lines_curve = pg.PlotDataItem(pen=pg.mkPen(color=(150, 150, 150), width=1), connect="finite")
        self.active_curve = pg.PlotDataItem(pen=pg.mkPen(color="purple", width=2))
        self.highlight_segments = pg.PlotDataItem(pen=pg.mkPen(color="red", width=2), connect="finite")
        self.points_scatter = pg.ScatterPlotItem(size=8, symbol="o", brush=pg.mkBrush("black"), pen=None)

        self._has_initialized_xrange: bool = False

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
            self.setWindowTitle(f"MultiTraceFixed (PyQtGraph) {summary}")
            p = self.parentWidget()
            if p is not None and hasattr(p, "setWindowTitle"):
                p.setWindowTitle(f"MultiTraceFixed (PyQtGraph) {summary}")
        self.update_figure()

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

    def _init_ui(self):
        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground("w")

        self.time_axis = FixedTimeAxisItem(orientation="bottom")
        self.main_plot: pg.PlotItem = self.glw.addPlot(row=0, col=0, axisItems={"bottom": self.time_axis})
        self.main_plot.getViewBox().disableAutoRange(pg.ViewBox.XAxis)
        self.main_plot.invertY(True)
        self.main_plot.setLabel("bottom", "Time", units="ms")
        self.main_plot.setLabel("left", "Stimulus Sweep")
        self.main_plot.showGrid(x=True, y=True, alpha=0.3)

        self.main_plot.addItem(self.lines_curve)
        self.main_plot.addItem(self.active_curve)
        self.main_plot.addItem(self.highlight_segments)
        self.main_plot.addItem(self.points_scatter)

        self.main_plot.scene().sigMouseClicked.connect(self._on_click)

        layout.addWidget(self.glw)
        self.setCentralWidget(central)

        self._build_toolbar()

    def _build_toolbar(self):
        toolbar = QToolBar("Settings", self)

        scale_widget = QWidget()
        scale_layout = QHBoxLayout(scale_widget)
        scale_layout.setContentsMargins(0, 0, 0, 0)

        scale_label = QLabel("Scale:")
        self.scale_spinner = QDoubleSpinBox()
        self.scale_spinner.setDecimals(5)
        self.scale_spinner.setRange(0.00001, 100.0)
        self.scale_spinner.setSingleStep(0.005)
        self.scale_spinner.setValue(self.settings.get("scale", 1.0))
        self.scale_spinner.valueChanged.connect(
            lambda value: self.set_settings({"scale": value}, update_spinners=False)
        )
        self.scale_spinner.setMinimumSize(100, 1)

        pre_label = QLabel("Pre:")
        self.pre_spinner = QSpinBox()
        self.pre_spinner.setRange(0, 100)
        self.pre_spinner.setValue(self.settings.get("n_lines_pre", 5))
        self.pre_spinner.valueChanged.connect(
            lambda value: self.set_settings({"n_lines_pre": value}, update_spinners=False)
        )
        self.pre_spinner.setMinimumSize(80, 1)

        post_label = QLabel("Post:")
        self.post_spinner = QSpinBox()
        self.post_spinner.setRange(0, 100)
        self.post_spinner.setValue(self.settings.get("n_lines_post", 5))
        self.post_spinner.valueChanged.connect(
            lambda value: self.set_settings({"n_lines_post": value}, update_spinners=False)
        )
        self.post_spinner.setMinimumSize(80, 1)

        scale_layout.addWidget(scale_label)
        scale_layout.addWidget(self.scale_spinner)
        scale_layout.addWidget(pre_label)
        scale_layout.addWidget(self.pre_spinner)
        scale_layout.addWidget(post_label)
        scale_layout.addWidget(self.post_spinner)

        toolbar.addWidget(scale_widget)
        self.addToolBar(toolbar)

    def set_state(self, state: ViewerState):
        self.unset_connections()
        self.state = state
        if hasattr(self, "channel_bar") and self.channel_bar is not None:
            self.channel_bar.setState(state)
        if self.state is None:
            return

        self.connections.append(self.state.onStimNoChange.connect(lambda x: self.update_figure()))
        self.connections.append(self.state.onUnitChange.connect(lambda x: self.update_figure()))
        self.connections.append(self.state.onUnitGroupChange.connect(lambda: self.update_figure()))
        self.connections.append(self.state.onLoadNewFile.connect(self._on_load_new_file))
        if hasattr(self.state, "onNewSweep"):
            self.connections.append(self.state.onNewSweep.connect(self.on_new_sweep))
        if hasattr(self.state, "onNewSpikeEvents"):
            self.connections.append(self.state.onNewSpikeEvents.connect(lambda *args: self.update_figure()))

        self._has_initialized_xrange = False
        self._auto_scale_needed = True
        self._auto_scale()
        erp = self.get_erp_signal()
        if erp is not None and len(erp) > 0:
            self._auto_scale_needed = False
        self.update_figure()

    def _on_load_new_file(self):
        self._has_initialized_xrange = False
        self._auto_scale_needed = True
        self.update_figure()

    def on_new_sweep(self, *args):
        if getattr(self, "_auto_scale_needed", True):
            erp = self.get_erp_signal() if self.state else None
            if erp is not None and len(erp) > 0:
                self._auto_scale()
                self._auto_scale_needed = False
        self.update_figure()

    def unset_connections(self):
        if self.state is not None:
            for con in self.connections:
                try:
                    self.state.disconnect(con)
                except Exception:
                    pass
        self.connections.clear()

    def _auto_scale(self):
        if self.state is None or self.state.analog_signal is None:
            return
        erp = self.state.get_erp()
        if erp is not None and len(erp) > 0:
            erp_mag = erp.magnitude if hasattr(erp, "magnitude") else erp
            ylim = float(np.std(np.abs(erp_mag))) * 8
            if ylim > 0:
                self.settings["scale"] = float(np.round(1.0 / ylim, decimals=4))
                self.scale_spinner.setValue(self.settings["scale"])

    def _on_click(self, event):
        if event.button() != Qt.LeftButton or self.state is None:
            return
        pos = self.main_plot.vb.mapSceneToView(event.scenePos())
        y_val = pos.y()
        stimno = self.state.stimno
        start = max(stimno - self.settings.get("n_lines_pre", 5), 0)
        target_stim = int(round(y_val) + start)
        if 0 <= target_stim < len(self.state.event_signal):
            self.state.setStimNo(target_stim)

    def update_figure(self):
        if self.state is None or self.state.analog_signal is None or self.state.event_signal is None:
            return

        erp = self.get_erp_signal()
        if erp is None or len(erp) == 0:
            return

        n_lines_pre = self.settings.get("n_lines_pre", 5)
        n_lines_post = self.settings.get("n_lines_post", 5)
        scale = float(self.settings.get("scale", 1.0))
        stimno = self.state.stimno

        start = max(stimno - n_lines_pre, 0)
        stop = min(start + n_lines_pre + n_lines_post + 1, len(erp))
        dpts = erp[start:stop]
        n_display = len(dpts)
        if n_display == 0:
            return

        dpts_mag = dpts.magnitude if hasattr(dpts, "magnitude") else dpts
        sr = float(self.state.sampling_rate.rescale("Hz").magnitude) if hasattr(self.state.sampling_rate, "rescale") else float(self.state.sampling_rate)
        x = np.arange(erp.shape[-1]) / sr
        active_idx = stimno - start

        # Background sweeps and active sweep
        bg_x_list = []
        bg_y_list = []
        active_x = None
        active_y = None

        for i in range(n_display):
            y = (dpts_mag[i] * scale * -1.0) + i
            if i == active_idx:
                active_x = x
                active_y = y
            else:
                bg_x_list.append(np.append(x, np.nan))
                bg_y_list.append(np.append(y, np.nan))

        if bg_x_list:
            self.lines_curve.setData(np.concatenate(bg_x_list), np.concatenate(bg_y_list))
        else:
            self.lines_curve.clear()

        if active_x is not None:
            self.active_curve.setData(active_x, active_y)
            self.active_curve.setVisible(True)
        else:
            self.active_curve.setVisible(False)

        # Spike latency markers
        hl_x = []
        hl_y = []
        scat_pts = []

        ug = self.get_target_unit()
        if ug is not None:
            latencies = ug.get_latencies(self.state.event_signal[start:stop])
            r = 20

            for i, l in enumerate(latencies):
                if np.isnan(l):
                    continue
                l_sec = float(l.rescale("s").magnitude)
                sample_no = int(l_sec * sr)
                if sample_no < 0 or sample_no >= len(dpts_mag[i]):
                    continue

                r_start = max(0, sample_no - r)
                r_end = min(len(dpts_mag[i]), sample_no + r)
                seg_x = np.arange(r_start, r_end) / sr
                seg_y = (dpts_mag[i][r_start:r_end] * scale * -1.0) + i

                hl_x.append(np.append(seg_x, np.nan))
                hl_y.append(np.append(seg_y, np.nan))

                scat_pts.append((l_sec, (dpts_mag[i][sample_no] * scale * -1.0) + i))

        if hl_x:
            self.highlight_segments.setData(np.concatenate(hl_x), np.concatenate(hl_y))
        else:
            self.highlight_segments.clear()

        if scat_pts:
            pts_arr = np.array(scat_pts)
            self.points_scatter.setData(x=pts_arr[:, 0], y=pts_arr[:, 1])
        else:
            self.points_scatter.clear()

        # Update Y-axis ticks with sweep indices
        ticks = [(i, str(start + i)) for i in range(n_display)]
        self.main_plot.getAxis("left").setTicks([ticks])
        self.main_plot.setYRange(-0.5, n_display - 0.5, padding=0)

        # Only initialize X-range on first sweep or reset; preserve zoom and pan across live sweeps
        if not self._has_initialized_xrange:
            self.main_plot.setXRange(0, (erp.shape[-1] - 1) / sr, padding=0)
            self._has_initialized_xrange = True

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_BracketLeft:  # Zoom out on X
            self.main_plot.vb.scaleBy((1.2, 1.0))
        elif e.key() == Qt.Key_BracketRight:  # Zoom in on X
            self.main_plot.vb.scaleBy((0.8, 1.0))
        elif e.key() in (Qt.Key_Home, Qt.Key_R):
            erp = self.get_erp_signal()
            if erp is not None and len(erp) > 0 and self.state is not None:
                sr = float(self.state.sampling_rate.rescale("Hz").magnitude) if hasattr(self.state.sampling_rate, "rescale") else float(self.state.sampling_rate)
                self.main_plot.setXRange(0, (erp.shape[-1] - 1) / sr, padding=0)
        else:
            super().keyPressEvent(e)

    def get_settings(self) -> Dict:
        res = dict(self.settings)
        res["xlim"] = self.main_plot.viewRange()[0]
        if hasattr(self, "channel_bar") and self.channel_bar is not None:
            res["channel_bar"] = self.channel_bar.get_settings()
        return res

    def set_settings(self, values: Dict, update_spinners: bool = True):
        self.settings.update(values)
        if "xlim" in values:
            self.main_plot.setXRange(values["xlim"][0], values["xlim"][1], padding=0)
            self._has_initialized_xrange = True
        if "channel_bar" in values and hasattr(self, "channel_bar") and self.channel_bar is not None:
            self.channel_bar.set_settings(values["channel_bar"])
        if update_spinners:
            if "n_lines_pre" in values:
                self.pre_spinner.setValue(values["n_lines_pre"])
            if "n_lines_post" in values:
                self.post_spinner.setValue(values["n_lines_post"])
            if "scale" in values:
                self.scale_spinner.setValue(values["scale"])
        self.update_figure()

    def closeEvent(self, event):
        self.unset_connections()
        super().closeEvent(event)
