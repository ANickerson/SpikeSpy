import sys
from typing import Dict, List, Optional
import numpy as np
import quantities as pq
from scipy.signal import find_peaks

import PySide6
from PySide6 import QtCore
from PySide6.QtCore import Qt, Signal, Slot, QObject, QRunnable, QThreadPool
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QApplication,
)
import pyqtgraph as pg

from .ViewerState import ViewerState, tracked_neuron_unit
from .processing import create_erp_signals
from .ChannelGroupBar import ChannelGroupBar
from .helpers import qsignal_throttle


class SingleTraceTimeAxisItem(pg.AxisItem):
    """Custom AxisItem converting seconds to milliseconds."""

    def tickStrings(self, values, scale, spacing):
        return [f"{1000 * v:g}" for v in values]


class WorkerSignals(QObject):
    result = Signal(np.ndarray)
    cancel = Signal()


class PGUpdateConv(QRunnable):
    """Background worker for template cross-correlation."""

    def __init__(self, analog_signal, sg, erp):
        super().__init__()
        self.signals = WorkerSignals()
        self.erp = erp
        self.sg = sg
        self.analog_signal = analog_signal
        self.cancelled = False
        self.signals.cancel.connect(self.cancel)

    def cancel(self):
        self.cancelled = True

    def create_erp_signals(self, signal_chan, idxs, offset=-1000 * pq.ms, length=30000 * pq.ms):
        arr = np.zeros(
            (
                len(idxs),
                int(np.ceil(length.rescale(pq.second).magnitude * signal_chan.sampling_rate.magnitude)),
            )
        )
        for i, x in enumerate(idxs):
            try:
                arr[i].flat[:] = signal_chan.time_slice(x + offset, x + offset + length).as_array()
                if self.cancelled:
                    return None
            except Exception:
                pass
        return arr * signal_chan.units

    def run(self):
        erp_sig = self.create_erp_signals(self.analog_signal, self.sg.event, -0.01 * pq.s, +0.02 * pq.s)
        if self.cancelled or erp_sig is None:
            return
        mean_erp = np.mean(erp_sig, axis=0)
        std_mean = np.std(mean_erp)
        if std_mean == 0 or self.cancelled:
            return

        conv = np.convolve(
            mean_erp / std_mean,
            (self.erp / std_mean).flat[:],
            mode="same",
        ).reshape(self.erp.shape)
        conv[:, 0 : len(mean_erp)] = 0
        conv[:, -len(mean_erp) :] = 0

        if not self.cancelled:
            self.signals.result.emit(conv)


class SingleTracePGView(QMainWindow):
    """
    High-performance PyQtGraph port of SingleTraceView.
    Displays single-sweep waveform with peak snapping, template correlation,
    latency histogram, and keyboard navigation.
    """

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        state: Optional[ViewerState] = None,
    ):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setFocusPolicy(Qt.ClickFocus)

        self.state: Optional[ViewerState] = state
        self.connections: List = []
        self.select_local_maxima_width: int = 1
        self.closest_pos: int = 0
        self.conv: Optional[np.ndarray] = None
        self.task: Optional[PGUpdateConv] = None
        self._dirty: bool = False
        self._was_shown: bool = False
        self._user_hidden: bool = False
        self._last_render_key = None
        self.on_new_sweep_throttled = qsignal_throttle(self._on_new_sweep_core, 40)
        self.update_histogram_throttled = qsignal_throttle(self._update_histogram_core, 500)

        # Graphics items
        self.top_lines: List[pg.InfiniteLine] = []
        self.aux_curves: List[pg.PlotDataItem] = []

        # Independent channel group and unit binding toolbar
        self.channel_bar = ChannelGroupBar(parent=self, state=self.state)
        self.channel_bar.bindingChanged.connect(self._on_channel_bar_changed)
        self.addToolBar(Qt.TopToolBarArea, self.channel_bar)

        self._init_ui()

        if state is not None:
            self.set_state(state)

    def _init_ui(self):
        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground("w")

        # Top plot: histogram and markers (Row 0)
        self.top_plot: pg.PlotItem = self.glw.addPlot(row=0, col=0)
        self.top_plot.getAxis("bottom").setStyle(showValues=False)
        self.top_plot.getAxis("left").setStyle(showValues=False)
        self.top_plot.setYRange(0, 1.0, padding=0)
        self.histogram_curve = pg.PlotDataItem(
            pen=pg.mkPen("gray", width=1),
            brush=pg.mkBrush(200, 200, 200, 120),
            stepMode="center",
            fillLevel=0,
        )
        self.top_plot.addItem(self.histogram_curve)

        # Bottom plot: main trace waveform (Row 1)
        self.time_axis = SingleTraceTimeAxisItem(orientation="bottom")
        self.main_plot: pg.PlotItem = self.glw.addPlot(row=1, col=0, axisItems={"bottom": self.time_axis})
        self.main_plot.setLabel("bottom", "Time", units="ms")
        self.main_plot.setLabel("left", "Voltage")
        self.main_plot.showGrid(x=True, y=True, alpha=0.3)
        self.top_plot.setXLink(self.main_plot)

        # Height ratios: 1 for top, 5 for main
        self.glw.ci.layout.setRowStretchFactor(0, 1)
        self.glw.ci.layout.setRowStretchFactor(1, 5)

        # Waveform line
        self.trace_curve = pg.PlotDataItem(pen=pg.mkPen("purple", width=1.5))
        self.main_plot.addItem(self.trace_curve)

        # Identified spike vertical line
        self.identified_spike_line = pg.InfiniteLine(
            angle=90, pen=pg.mkPen("blue", width=1.5, style=Qt.DashLine), movable=False
        )
        self.main_plot.addItem(self.identified_spike_line)
        self.identified_spike_line.setVisible(False)

        # Sweep label text
        self.stimno_label = pg.TextItem(text="", color=(0, 0, 0), anchor=(0, 1))
        self.main_plot.addItem(self.stimno_label)

        # Mouse click handling
        self.main_plot.scene().sigMouseClicked.connect(self._on_plot_clicked)

        layout.addWidget(self.glw)
        self.setCentralWidget(central)

    def _on_channel_bar_changed(self):
        if hasattr(self, "channel_bar"):
            summary = self.channel_bar.get_title_summary()
            self.setWindowTitle(f"SingleTrace (PyQtGraph) {summary}")
            p = self.parentWidget()
            if p is not None and hasattr(p, "setWindowTitle"):
                p.setWindowTitle(f"SingleTrace (PyQtGraph) {summary}")
        self._setup_done = False
        self.setup_figure()
        self.update_histogram()
        self.update_figure(force=True)

    def _on_channel_change_event(self, *args):
        self._setup_done = False
        self.setup_figure()
        self.update_histogram()
        self.update_figure(force=True)

    def get_target_unit(self) -> Optional[tracked_neuron_unit]:
        if hasattr(self, "channel_bar") and self.channel_bar is not None:
            return self.channel_bar.get_target_unit()
        return self.state.getUnitGroup() if self.state else None

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

    def set_state(self, state: ViewerState):
        self.unset_connections()
        self.state = state
        if hasattr(self, "channel_bar") and self.channel_bar is not None:
            self.channel_bar.setState(state)
        if self.state is None:
            return

        self.connections.append(self.state.onLoadNewFile.connect(self.setup_figure))
        self.connections.append(self.state.onStimNoChange.connect(self.update_figure))
        self.connections.append(self.state.onUnitChange.connect(self.update_figure))
        self.connections.append(self.state.onUnitGroupChange.connect(self.update_figure))
        self.connections.append(self.state.onUnitChange.connect(self.update_histogram))
        self.connections.append(self.state.onUnitGroupChange.connect(self.update_histogram))
        if hasattr(self.state, "onChannelChange"):
            self.connections.append(self.state.onChannelChange.connect(self._on_channel_change_event))
        if hasattr(self.state, "onNewSweep"):
            self.connections.append(self.state.onNewSweep.connect(self.on_new_sweep))
        if hasattr(self.state, "onNewSpikeEvents"):
            self.connections.append(self.state.onNewSpikeEvents.connect(lambda *args: (self.update_histogram(), self.update_figure())))

        self._setup_done = False
        self.setup_figure()

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
            self.update_figure()

    def hideEvent(self, event):
        super().hideEvent(event)
        self._user_hidden = True

    def showEvent(self, event):
        super().showEvent(event)
        self._was_shown = True
        self._user_hidden = False
        if getattr(self, "_dirty", False):
            self._dirty = False
            self.update_figure()

    def _on_new_sweep_core(self, stimno: Optional[int] = None):
        if self.state is None:
            return
        erp = self.get_erp_signal()
        if erp is None or len(erp) == 0:
            return
        if not getattr(self, "_setup_done", False):
            self.setup_figure()
            self._setup_done = True
        self.update_figure()

    def on_new_sweep(self, stimno: Optional[int] = None):
        if not getattr(self, "_setup_done", False):
            self._on_new_sweep_core(stimno)
        elif hasattr(self, "on_new_sweep_throttled"):
            self.on_new_sweep_throttled(stimno)
        else:
            self._on_new_sweep_core(stimno)

    def unset_connections(self):
        if self.state is not None:
            for con in self.connections:
                try:
                    self.state.disconnect(con)
                except Exception:
                    pass
        self.connections.clear()

    def setup_figure(self):
        if self.state is None or self.state.analog_signal is None:
            return
        erp = self.get_erp_signal()
        if erp is None or len(erp) == 0:
            return

        sr = float(self.state.sampling_rate.rescale("Hz").magnitude) if hasattr(self.state.sampling_rate, "rescale") else float(self.state.sampling_rate)
        duration_sec = erp.shape[-1] / sr
        self.main_plot.setXRange(0, duration_sec, padding=0)

        erp_mag = erp.magnitude if hasattr(erp, "magnitude") else erp
        ylim = float(np.std(np.abs(erp_mag)) * 8)
        if ylim > 0:
            self.main_plot.setYRange(-ylim, ylim, padding=0)

        self._setup_done = True
        self._update_histogram_core()
        self.update_figure()

    def update_histogram(self):
        if hasattr(self, "update_histogram_throttled"):
            self.update_histogram_throttled()
        else:
            self._update_histogram_core()

    def _update_histogram_core(self):
        if self.state is None or self.state.event_signal is None:
            return

        sg = self.get_target_unit()
        if sg is None or len(sg.event) == 0:
            self.histogram_curve.clear()
            return

        if self.task is not None:
            self.task.signals.cancel.emit()

        erp = self.get_erp_signal()
        if erp is not None and len(erp) > 0 and self.state.analog_signal is not None:
            self.task = PGUpdateConv(self.state.analog_signal, sg, erp)
            self.task.signals.result.connect(self._on_conv_done)
            QThreadPool.globalInstance().start(self.task)

        # Draw latency histogram
        latencies = sg.get_latencies(self.state.event_signal)
        valid = latencies[~np.isnan(latencies)]
        if len(valid) == 0:
            self.histogram_curve.clear()
            return

        sec_vals = [float(v.rescale("s").magnitude) for v in valid]
        step = 0.002  # 2 ms bins
        min_v = min(sec_vals)
        max_v = max(sec_vals)
        bins = np.arange(min_v - step, max_v + (step * 2), step)
        counts, bin_edges = np.histogram(sec_vals, bins=bins)

        if len(counts) > 0 and np.max(counts) > 0:
            norm_counts = counts / float(np.max(counts))
            self.histogram_curve.setData(x=bin_edges, y=norm_counts)
        else:
            self.histogram_curve.clear()

    @Slot(np.ndarray)
    def _on_conv_done(self, conv_result: np.ndarray):
        self.conv = conv_result
        self.update_figure()

    def update_figure(self, force: bool = False):
        if self.is_view_culled():
            self._dirty = True
            return
        self._dirty = False

        if self.state is None or self.state.event_signal is None:
            return

        stimno = self.state.stimno
        erp = self.get_erp_signal()
        if erp is None or stimno >= len(erp) or stimno < 0:
            return

        if not getattr(self, "_setup_done", False):
            self.setup_figure()
            self._setup_done = True

        ug = self.get_target_unit()
        ug_spikes = len(ug.event) if ug and hasattr(ug, "event") and ug.event is not None else 0
        ug_idx = self.get_target_unit_idx()
        primary_ch = self.channel_bar.get_primary_channel() if hasattr(self, "channel_bar") else ""
        aux_chans = tuple(sorted(self.channel_bar.get_selected_channels())) if hasattr(self, "channel_bar") else ()

        render_key = (stimno, len(erp), primary_ch, aux_chans, ug_idx, ug_spikes)
        if not force and getattr(self, "_last_render_key", None) == render_key:
            return
        self._last_render_key = render_key

        dpts = erp[stimno]
        dpts_mag = dpts.magnitude if hasattr(dpts, "magnitude") else dpts
        sr = float(self.state.sampling_rate.rescale("Hz").magnitude) if hasattr(self.state.sampling_rate, "rescale") else float(self.state.sampling_rate)
        x_sec = np.arange(len(dpts_mag)) / sr

        # Update waveform curve
        self.trace_curve.setData(x_sec, dpts_mag)

        # Multichannel overlay
        for c in self.aux_curves:
            self.main_plot.removeItem(c)
        self.aux_curves.clear()

        if hasattr(self, "channel_bar") and self.channel_bar.is_multichannel():
            primary_name = self.channel_bar.get_primary_channel()
            selected = self.channel_bar.get_selected_channels()
            aux_chans = [c for c in selected if c != primary_name]
            colors = [(100, 149, 237, 180), (220, 20, 60, 180), (46, 139, 87, 180), (255, 140, 0, 180)]
            for i, ch_name in enumerate(aux_chans):
                if hasattr(self.state, "get_channel_erp"):
                    ch_erp = self.state.get_channel_erp(ch_name)
                    if ch_erp is not None and stimno < len(ch_erp):
                        ch_dpts = ch_erp[stimno]
                        ch_mag = ch_dpts.magnitude if hasattr(ch_dpts, "magnitude") else ch_dpts
                        pen_col = colors[i % len(colors)]
                        aux_c = pg.PlotDataItem(x_sec[:len(ch_mag)], ch_mag, pen=pg.mkPen(color=pen_col, width=1.2, style=Qt.DashLine))
                        self.main_plot.addItem(aux_c)
                        self.aux_curves.append(aux_c)

        # Update sweep label text
        if len(self.state.event_signal.times) > stimno:
            cur_event_time = float(self.state.event_signal.times[stimno].magnitude)
            hours, remainder = divmod(cur_event_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            self.stimno_label.setText(f"{stimno:04} [{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}]")
        else:
            self.stimno_label.setText(f"{stimno:04}")
        self.stimno_label.setPos(0, float(self.main_plot.viewRange()[1][1]))

        # Clear existing marker lines on top plot
        for line in self.top_lines:
            self.top_plot.removeItem(line)
        self.top_lines.clear()

        sg = self.get_target_unit()
        if sg is not None and getattr(sg, "idx_arr", None) is not None:
            idxs = [i for i, x in enumerate(sg.idx_arr) if x is not None]
            values = [x[0] / sr for x in sg.idx_arr if x is not None]

            if len(self.state.event_signal.times) > stimno:
                cur_ev = self.state.event_signal.times[stimno]
                lat = sg.get_latencies(np.array([cur_ev]) * cur_ev.units)[0].rescale("s")

                if not np.isnan(lat) and lat < self.state.window_size:
                    lat_sec = float(lat.magnitude)
                    self.identified_spike_line.setValue(lat_sec)
                    self.identified_spike_line.setVisible(True)

                    line_cur = pg.InfiniteLine(pos=lat_sec, angle=90, pen=pg.mkPen("blue", width=1.5), movable=False)
                    self.top_plot.addItem(line_cur)
                    self.top_lines.append(line_cur)
                else:
                    self.identified_spike_line.setVisible(False)
            else:
                self.identified_spike_line.setVisible(False)

        # Plot previous and next identified spikes in this group
        cur_idx = int(np.searchsorted(idxs, stimno))
        if cur_idx > 0 and (stimno - idxs[cur_idx - 1]) < 10:
            l_prev = values[cur_idx - 1]
            line_prev = pg.InfiniteLine(
                pos=l_prev, angle=90, pen=pg.mkPen(0, 180, 0, 150, width=1.5), movable=False
            )
            self.top_plot.addItem(line_prev)
            self.top_lines.append(line_prev)

        if (cur_idx + 1) < len(idxs) and (idxs[cur_idx + 1] - stimno) < 10 and cur_idx >= 0:
            l_next = values[cur_idx + 1]
            line_next = pg.InfiniteLine(
                pos=l_next, angle=90, pen=pg.mkPen(220, 0, 0, 150, width=1.5), movable=False
            )
            self.top_plot.addItem(line_next)
            self.top_lines.append(line_next)

        # Correlation peaks
        if self.conv is not None and stimno < len(self.conv):
            conv_row = self.conv[stimno]
            if len(conv_row) > 2000:
                p99 = float(np.percentile(conv_row[1000:-1000], 99))
                peaks, _ = find_peaks(conv_row, height=p99)
                peaks = peaks - 1
                if len(peaks) > 0:
                    sorted_peaks = peaks[np.argsort(conv_row[peaks])[::-1]][:50]
                    for px in sorted_peaks:
                        c_line = pg.InfiniteLine(
                            pos=px / sr,
                            angle=90,
                            pen=pg.mkPen(218, 165, 32, 100, width=1, style=Qt.DashLine),
                            movable=False,
                        )
                        self.top_plot.addItem(c_line)
                        self.top_lines.append(c_line)

    def _on_plot_clicked(self, event):
        if event.button() != Qt.LeftButton or self.state is None:
            return
        if not hasattr(self.main_plot, "vb") or not self.main_plot.sceneBoundingRect().contains(event.scenePos()):
            return
        pos = self.main_plot.vb.mapSceneToView(event.scenePos())
        if np.isnan(pos.x()):
            return
        sr = float(self.state.sampling_rate) if self.state.sampling_rate is not None else 10000.0
        sample_x = pos.x() * sr
        self.set_cur_pos(sample_x)

    def set_cur_pos(self, x: float):
        if np.isnan(x):
            return
        erp = self.get_erp_signal()
        if erp is None or self.state is None or self.state.stimno < 0 or self.state.stimno >= len(erp):
            return
        dpts = erp[self.state.stimno]
        if hasattr(dpts, "magnitude"):
            dpts = dpts.magnitude
        n_pts = len(dpts)
        if n_pts == 0:
            return

        # Clamp x_idx to valid positive range within waveform bounds
        x_idx = max(0, min(n_pts - 1, int(round(x))))
        w = int(self.select_local_maxima_width)

        if w > 1:
            start_w = max(0, x_idx - w)
            end_w = min(n_pts, x_idx + w + 1)
            if end_w > start_w:
                sub_slice = np.abs(dpts[start_w:end_w])
                if len(sub_slice) > 0:
                    x_idx = start_w + int(np.argmax(sub_slice))

        target_unit_idx = self.get_target_unit_idx()
        self.state.setUnit(x_idx, unitid=target_unit_idx)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Z and self.state is not None:
            # Zoom to current spike
            ug = self.get_target_unit()
            if ug is None:
                return
            cur_ev = self.state.event_signal[self.state.stimno]
            lat = ug.get_latencies([cur_ev.magnitude] * self.state.event_signal.units)[0].rescale("s")

            if not np.isnan(lat) and lat < self.state.window_size:
                lat_sec = float(lat.magnitude)
                self.main_plot.setXRange(lat_sec - 0.005, lat_sec + 0.005, padding=0)

                # Set Y limits to max of the spike + 5%
                try:
                    vals = self.state.analog_signal.time_slice(
                        cur_ev + lat - (2 * pq.ms), cur_ev + lat + (2 * pq.ms)
                    ).magnitude
                    min_v = float(np.min(vals))
                    max_v = float(np.max(vals))
                    pad_v = 0.05 * max(abs(min_v), abs(max_v))
                    self.main_plot.setYRange(min_v - pad_v, max_v + pad_v, padding=0)
                except Exception:
                    pass

        elif (
            (e.key() in (Qt.Key_Left, Qt.Key_Right))
            and (e.modifiers() & Qt.ShiftModifier)
            and self.conv is not None
        ):
            ug = self.get_target_unit()
            if ug is None:
                return
            cur_stimpos = ug.idx_arr[self.state.stimno][0]
            conv_row = self.conv[self.state.stimno]
            if len(conv_row) > 2000:
                p99 = float(np.percentile(conv_row[1000:-1000], 99))
                peaks, _ = find_peaks(conv_row, height=p99)
                peaks = peaks - 1
                if len(peaks) > 0:
                    closest_idx = int(np.searchsorted(peaks, cur_stimpos))
                    if peaks[min(closest_idx, len(peaks) - 1)] == cur_stimpos and e.key() == Qt.Key_Right:
                        closest_idx += 1
                    elif e.key() == Qt.Key_Left:
                        closest_idx -= 1

                    if 0 <= closest_idx < len(peaks):
                        target_unit_idx = self.get_target_unit_idx()
                        self.state.setUnit(int(peaks[closest_idx]), unitid=target_unit_idx)

        elif e.key() == Qt.Key_BracketLeft:  # Zoom out
            self.main_plot.vb.scaleBy((1.2, 1.2))
        elif e.key() == Qt.Key_BracketRight:  # Zoom in
            self.main_plot.vb.scaleBy((0.8, 0.8))
        else:
            super().keyPressEvent(e)

    def get_settings(self) -> Dict:
        res = {"xlim": self.main_plot.viewRange()[0]}
        if hasattr(self, "channel_bar") and self.channel_bar is not None:
            res["channel_bar"] = self.channel_bar.get_settings()
        return res

    def set_settings(self, values: Dict):
        if "xlim" in values:
            self.main_plot.setXRange(values["xlim"][0], values["xlim"][1], padding=0)
        if "channel_bar" in values and hasattr(self, "channel_bar") and self.channel_bar is not None:
            self.channel_bar.set_settings(values["channel_bar"])

    def closeEvent(self, event):
        self.unset_connections()
        if self.task is not None:
            self.task.signals.cancel.emit()
        super().closeEvent(event)
