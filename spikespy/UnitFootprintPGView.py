import itertools
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import quantities as pq
from neo import Event

import PySide6
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, QTimer, Slot, QRectF, QPointF
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QLabel,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QToolBar,
    QVBoxLayout,
    QWidget,
)
import pyqtgraph as pg

from .ViewerState import ViewerState, tracked_neuron_unit
from .ChannelGroupBar import ChannelGroupBar


def _get_mpl_lut(name: str = "RdBu_r", n: int = 256) -> np.ndarray:
    """Sample a Matplotlib colormap into an RGBA lookup table for PyQtGraph."""
    cmap = plt.get_cmap(name)
    colors = cmap(np.linspace(0, 1, n))
    return (colors * 255).astype(np.uint8)


class UnitFootprintPGView(QMainWindow):
    """
    High-performance PyQtGraph port of UnitFootprintView.
    Delivers GPU-accelerated spatiotemporal footprint waterfall heatmaps across
    recording channels, inter-channel conduction lag (Δt), amplitude bar charts,
    and interactive multi-unit grid mode with 60 FPS live streaming.
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
        self.current_channels: List[str] = []
        self.render_count: int = 0
        self._dirty: bool = False
        self._user_hidden: bool = False
        self._was_shown: bool = False
        self.grid_plots: List[Tuple[pg.PlotItem, int, str]] = []

        # Color LUT for diverging footprint heatmap
        self._cmap_lut = _get_mpl_lut("RdBu_r")

        # Central Graphics Layout Widget
        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground("w")
        self.glw.scene().sigMouseClicked.connect(self._on_scene_clicked)

        central_widget = QWidget(self)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.glw)
        self.setCentralWidget(central_widget)

        # UI Controls
        self.combo_mode = QComboBox(self)
        self.combo_mode.addItems(["Single Unit Composite", "Multi-Unit Grid"])
        self.combo_mode.setToolTip("Switch between Single Unit Composite view and Multi-Unit Grid view")

        self.spin_last_n = QSpinBox(self)
        self.spin_last_n.setRange(1, 500)
        self.spin_last_n.setValue(20)
        self.spin_last_n.setPrefix("Last N=")
        self.spin_last_n.setSuffix(" events")
        self.spin_last_n.setToolTip("Number of most recent detected spike events to align and average")

        self.spin_window_ms = QDoubleSpinBox(self)
        self.spin_window_ms.setRange(1.0, 20.0)
        self.spin_window_ms.setValue(4.0)
        self.spin_window_ms.setSingleStep(0.5)
        self.spin_window_ms.setSuffix(" ms")
        self.spin_window_ms.setToolTip("Time window (± half-window) around the reference peak")

        self.chk_align_peaks = QCheckBox("Align to Peak", self)
        self.chk_align_peaks.setChecked(True)
        self.chk_align_peaks.setToolTip(
            "Jitter-correct spikes to local extremum on reference channel while preserving inter-channel lag"
        )

        self.chk_overlay_traces = QCheckBox("Overlay Waveforms & Lag Markers", self)
        self.chk_overlay_traces.setChecked(True)
        self.chk_overlay_traces.setToolTip("Overlay mean waveforms, individual traces, and peak lag markers")

        self.btn_auto_assign_channels = QPushButton("Auto-Detect Unit Channels", self)
        self.btn_auto_assign_channels.setToolTip(
            "Auto-identify primary and neighbor channels based on footprint amplitude"
        )

        # Build Top Control Toolbar
        self.control_toolbar = QToolBar("Footprint Controls", self)
        self.control_toolbar.addWidget(QLabel(" Mode: ", self))
        self.control_toolbar.addWidget(self.combo_mode)
        self.control_toolbar.addSeparator()
        self.control_toolbar.addWidget(self.spin_last_n)
        self.control_toolbar.addWidget(self.spin_window_ms)
        self.control_toolbar.addSeparator()
        self.control_toolbar.addWidget(self.chk_align_peaks)
        self.control_toolbar.addWidget(self.chk_overlay_traces)
        self.control_toolbar.addSeparator()
        self.control_toolbar.addWidget(self.btn_auto_assign_channels)
        self.addToolBar(self.control_toolbar)

        # Independent channel group and unit binding toolbar
        self.channel_bar = ChannelGroupBar(parent=self, state=state)
        self.channel_bar.bindingChanged.connect(self._on_channel_bar_changed)
        self.addToolBarBreak()
        self.addToolBar(Qt.TopToolBarArea, self.channel_bar)

        # Throttled render timer (30 FPS or configured)
        self.render_timer = QTimer(self)
        interval = (
            self.state.get_ui_refresh_rate_ms()
            if self.state and hasattr(self.state, "get_ui_refresh_rate_ms")
            else 33
        )
        self.render_timer.setInterval(interval)
        self.render_timer.timeout.connect(self._on_render_timer_tick)
        self.render_timer.start()

        if self.state and hasattr(self.state, "onUIRefreshRateChange"):
            self.state.onUIRefreshRateChange.connect(self.render_timer.setInterval)

        # Signal connections for UI controls
        self.combo_mode.currentIndexChanged.connect(self.update_plot)
        self.spin_last_n.valueChanged.connect(self.update_plot)
        self.spin_window_ms.valueChanged.connect(self.update_plot)
        self.chk_align_peaks.toggled.connect(self.update_plot)
        self.chk_overlay_traces.toggled.connect(self.update_plot)
        self.btn_auto_assign_channels.clicked.connect(lambda: self.auto_detect_unit_channels())

        if state is not None:
            self.set_state(state)
        self.update_window_title()

    def _on_channel_bar_changed(self):
        self.update_window_title()
        self.update_plot()

    def update_window_title(self):
        if hasattr(self, "channel_bar"):
            summary = self.channel_bar.get_title_summary()
            self.setWindowTitle(f"UnitFootprint (PyQtGraph) {summary}")
            p = self.parentWidget()
            if p is not None and hasattr(p, "setWindowTitle"):
                p.setWindowTitle(f"UnitFootprint (PyQtGraph) {summary}")

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

    def _on_render_timer_tick(self):
        try:
            if self.is_view_culled() or not getattr(self, "_dirty", False):
                return
            self._dirty = False
            self._do_update_plot()
        except RuntimeError:
            pass

    def setVisible(self, visible: bool):
        if not visible:
            self._user_hidden = True
        else:
            self._user_hidden = False
            self._was_shown = True
        super().setVisible(visible)
        if visible and getattr(self, "_dirty", False):
            self._dirty = False
            self._do_update_plot()

    def hideEvent(self, event):
        super().hideEvent(event)
        self._user_hidden = True

    def showEvent(self, event):
        super().showEvent(event)
        self._was_shown = True
        self._user_hidden = False
        if getattr(self, "_dirty", False):
            self._dirty = False
            self._do_update_plot()

    def set_state(self, state: ViewerState):
        self.state = state
        if hasattr(self, "channel_bar"):
            self.channel_bar.setState(state)
        if self.state is not None:
            if hasattr(self.state, "onUnitChange"):
                self.state.onUnitChange.connect(lambda *args: self.schedule_update())
            if hasattr(self.state, "onUnitGroupChange"):
                self.state.onUnitGroupChange.connect(lambda *args: self.schedule_update())
            if hasattr(self.state, "onLoadNewFile"):
                self.state.onLoadNewFile.connect(lambda *args: self.schedule_update())
            if hasattr(self.state, "onNewSpikeEvents"):
                self.state.onNewSpikeEvents.connect(lambda *args: self.schedule_update())
            if hasattr(self.state, "onChannelChange"):
                self.state.onChannelChange.connect(lambda *args: self.schedule_update())
            if hasattr(self.state, "onNewSweep"):
                self.state.onNewSweep.connect(lambda *args: self.schedule_update())
            if hasattr(self.state, "onUIRefreshRateChange") and hasattr(self, "render_timer"):
                self.render_timer.setInterval(self.state.get_ui_refresh_rate_ms())
                self.state.onUIRefreshRateChange.connect(self.render_timer.setInterval)
        self.update_window_title()
        self.update_plot()

    def schedule_update(self):
        self._dirty = True

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

    def compute_unit_footprint(
        self,
        sg: Optional[tracked_neuron_unit],
        last_n: Optional[int] = None,
        window_ms: Optional[float] = None,
        align_peaks: Optional[bool] = None,
    ) -> Optional[dict]:
        """
        Extracts and aligns spatiotemporal snippets across all available neural channels
        for the specified tracked unit using the last N detected events.
        """
        if sg is None or self.state is None:
            return None

        if last_n is None:
            last_n = self.spin_last_n.value()
        if window_ms is None:
            window_ms = self.spin_window_ms.value()
        if align_peaks is None:
            align_peaks = self.chk_align_peaks.isChecked()

        sr = float(
            self.state.sampling_rate
            if self.state.sampling_rate is not None
            else (
                self.state.analog_signal.sampling_rate.rescale(pq.Hz).magnitude
                if self.state.analog_signal is not None
                else 30000.0
            )
        )
        if sr <= 0:
            return None

        # Resolve available recording channels
        channels = self.state.get_available_channels()
        if not channels:
            return None

        # Resolve primary reference channel for this unit
        ref_chan = (
            getattr(sg, "channel", "")
            or getattr(getattr(self.state, "analog_signal", None), "name", "")
            or channels[0]
        )
        if ref_chan not in channels:
            ref_chan = channels[0]

        # Extract last N valid sweep indices and sample latencies for this unit
        valid_sweeps = []
        valid_samples = []

        if getattr(sg, "idx_arr", None) is not None and len(sg.idx_arr) > 0:
            for s_idx, a in enumerate(list(sg.idx_arr)):
                if a is not None and len(a) > 0 and a[0] is not None:
                    valid_sweeps.append(s_idx)
                    valid_samples.append(int(a[0]))
        elif getattr(sg, "event", None) is not None and len(sg.event) > 0 and self.state.event_signal is not None:
            sw_arr, sa_arr = sg.get_valid_sweep_indices_and_samples(
                self.state.event_signal, sr, self.state.window_size
            )
            valid_sweeps = list(sw_arr)
            valid_samples = list(sa_arr)

        if not valid_sweeps:
            return None

        # Take only the last N events
        if len(valid_sweeps) > last_n:
            valid_sweeps = valid_sweeps[-last_n:]
            valid_samples = valid_samples[-last_n:]

        window_samples = max(8, int(round((window_ms / 1000.0) * sr)))
        if window_samples % 2 != 0:
            window_samples += 1
        half_win = window_samples // 2
        t_ms = (np.arange(window_samples) - half_win) * (1000.0 / sr)

        n_ch = len(channels)
        n_events = len(valid_sweeps)

        # Fetch ERP for each channel
        chan_erps = {}
        for ch in channels:
            erp = self.state.get_channel_erp(ch)
            if erp is not None:
                erp_mag = erp.magnitude if hasattr(erp, "magnitude") else erp
                chan_erps[ch] = erp_mag
            else:
                chan_erps[ch] = None

        if ref_chan not in chan_erps or chan_erps[ref_chan] is None:
            return None

        ref_erp = chan_erps[ref_chan]
        snippets = np.zeros((n_ch, n_events, window_samples), dtype=np.float32)

        for ev_i, (s_idx, s_samp) in enumerate(zip(valid_sweeps, valid_samples)):
            if s_idx >= ref_erp.shape[0]:
                continue

            ref_trace = ref_erp[s_idx]
            n_tot_pts = len(ref_trace)

            center_samp = s_samp
            if align_peaks:
                jitter_search_w = max(4, int(0.001 * sr))
                search_start = max(0, s_samp - jitter_search_w)
                search_end = min(n_tot_pts, s_samp + jitter_search_w + 1)
                if search_end > search_start:
                    sub_win = ref_trace[search_start:search_end]
                    peak_idx = int(np.argmax(np.abs(sub_win)))
                    center_samp = search_start + peak_idx

            w_start = center_samp - half_win
            w_end = center_samp + half_win

            for ch_i, ch in enumerate(channels):
                c_erp = chan_erps[ch]
                if c_erp is not None and s_idx < c_erp.shape[0]:
                    tr = c_erp[s_idx]
                    pts_avail = len(tr)
                    src_s = max(0, w_start)
                    src_e = min(pts_avail, w_end)
                    dst_s = max(0, -w_start)
                    dst_e = dst_s + (src_e - src_s)
                    if src_e > src_s and dst_e <= window_samples:
                        snippets[ch_i, ev_i, dst_s:dst_e] = tr[src_s:src_e]

        # Compute mean footprint matrix across events: (n_channels, window_samples)
        mean_matrix = np.mean(snippets, axis=1)

        # Baseline correct each channel
        base_samples = max(3, int(0.12 * window_samples))
        for ch_i in range(n_ch):
            baseline = np.mean(mean_matrix[ch_i, :base_samples])
            mean_matrix[ch_i, :] -= baseline
            for ev_i in range(n_events):
                snippets[ch_i, ev_i, :] -= np.mean(snippets[ch_i, ev_i, :base_samples])

        ptp_amp = np.zeros(n_ch, dtype=np.float32)
        lag_ms = np.zeros(n_ch, dtype=np.float32)
        snr = np.zeros(n_ch, dtype=np.float32)

        ref_ch_idx = channels.index(ref_chan)
        ref_peak_idx = int(np.argmax(np.abs(mean_matrix[ref_ch_idx])))

        for ch_i in range(n_ch):
            ch_mean = mean_matrix[ch_i]
            ptp_amp[ch_i] = float(np.max(ch_mean) - np.min(ch_mean))

            ch_peak_idx = int(np.argmax(np.abs(ch_mean)))
            lag_ms[ch_i] = float((ch_peak_idx - ref_peak_idx) * (1000.0 / sr))

            mad = float(np.median(np.abs(ch_mean)))
            bg_std = (mad / 0.6745) if mad > 1e-9 else float(np.std(ch_mean) or 1.0)
            if bg_std < 1e-9:
                bg_std = 1.0
            snr[ch_i] = float(np.max(np.abs(ch_mean)) / bg_std)

        max_ptp = float(np.max(ptp_amp)) if len(ptp_amp) > 0 else 0.0
        detected_channels = [
            channels[i]
            for i in range(n_ch)
            if ptp_amp[i] >= 0.25 * max_ptp and snr[i] >= 2.0
        ]

        return {
            "channels": channels,
            "ref_channel": ref_chan,
            "t_ms": t_ms,
            "snippets": snippets,
            "mean_matrix": mean_matrix,
            "ptp_amp": ptp_amp,
            "peak_amps": ptp_amp,
            "lag_ms": lag_ms,
            "snr": snr,
            "detected_channels": detected_channels,
            "n_events": len(valid_sweeps),
        }

    def auto_detect_unit_channels(self, sg: Optional[tracked_neuron_unit] = None):
        """
        Automatically identifies the primary channel and neighbor channels based on
        spatiotemporal footprint amplitude and SNR, updating unit annotations.
        """
        if sg is None:
            sg = self.get_target_unit()
        if sg is None:
            return

        fp = self.compute_unit_footprint(sg)
        if fp is None or len(fp["channels"]) == 0:
            return

        best_idx = int(np.argmax(fp["ptp_amp"]))
        primary_chan = fp["channels"][best_idx]
        max_amp = fp["ptp_amp"][best_idx]

        neighbors = []
        for i, ch in enumerate(fp["channels"]):
            if ch == primary_chan:
                continue
            if max_amp > 0 and fp["ptp_amp"][i] >= 0.30 * max_amp and fp["snr"][i] >= 2.5:
                neighbors.append(ch)

        sg.channel = primary_chan
        sg.neighbor_channels = neighbors
        if hasattr(sg, "event") and sg.event is not None:
            if not hasattr(sg.event, "annotations"):
                sg.event.annotations = {}
            sg.event.annotations["channel"] = primary_chan
            sg.event.annotations["neighbor_channels"] = neighbors

        if self.state is not None and hasattr(self.state, "onUnitGroupChange"):
            self.state.onUnitGroupChange.emit()

    @Slot()
    def update_plot(self):
        """Refreshes the plot according to current mode and settings."""
        if self.is_view_culled():
            self._dirty = True
            return
        self._dirty = False
        self._do_update_plot()

    def render_views(self):
        """Forces an immediate render pass (useful in tests and manual refreshes)."""
        self._do_update_plot()

    def _do_update_plot(self):
        if self.is_view_culled():
            self._dirty = True
            return

        self.render_count = getattr(self, "render_count", 0) + 1
        self.glw.clear()
        self.grid_plots = []
        self.current_channels = []
        self.heat_plot = None
        self.img_item = None
        self.bar_plot = None
        self.lag_scatter = None
        self.trend_line = None
        self.bars = []

        if self.state is None:
            return

        mode_idx = self.combo_mode.currentIndex()
        if mode_idx == 0:
            self._plot_single_unit_composite()
        else:
            self._plot_multi_unit_grid()

    def _plot_single_unit_composite(self):
        sg = self.get_target_unit()
        if sg is None:
            p = self.glw.addPlot(row=0, col=0)
            txt = pg.TextItem(text="No unit selected", color=(100, 100, 100), anchor=(0.5, 0.5))
            p.addItem(txt)
            p.showGrid(False, False)
            return

        fp = self.compute_unit_footprint(sg)
        if fp is None:
            p = self.glw.addPlot(row=0, col=0)
            txt = pg.TextItem(text="Insufficient events or channel data for unit footprint", color=(100, 100, 100), anchor=(0.5, 0.5))
            p.addItem(txt)
            p.showGrid(False, False)
            return

        channels = fp["channels"]
        self.current_channels = channels
        t_ms = fp["t_ms"]
        mean_matrix = fp["mean_matrix"]
        ptp_amp = fp["ptp_amp"]
        lag_ms = fp["lag_ms"]
        detected_channels = fp["detected_channels"]
        ref_chan = fp["ref_channel"]
        snippets = fp["snippets"]
        n_ch = len(channels)

        # 1. Left Plot: Spatiotemporal Footprint Heatmap & Waveforms
        heat_plot = self.glw.addPlot(row=0, col=0)
        heat_plot.invertY(True)
        heat_plot.setLabel("bottom", "Time", units="ms")
        heat_plot.setLabel("left", "Recording Channel")
        heat_plot.showGrid(x=True, y=True, alpha=0.2)

        # Set Custom Channel Ticks on Y-axis
        ticks = [(i, ch) for i, ch in enumerate(channels)]
        heat_plot.getAxis("left").setTicks([ticks])

        # Title
        unit_name = (
            getattr(sg.event, "name", f"Unit {self.get_target_unit_idx()}")
            if hasattr(sg, "event") and sg.event is not None
            else f"Unit {self.get_target_unit_idx()}"
        )
        heat_plot.setTitle(f"<b>{unit_name} Footprint ({fp['n_events']} events, Ref: {ref_chan})</b>")

        # Heatmap ImageItem
        vmax = float(np.max(np.abs(mean_matrix))) if np.max(np.abs(mean_matrix)) > 0 else 1.0
        img_item = pg.ImageItem(axisOrder="row-major")
        img_item.setLookupTable(self._cmap_lut)
        img_item.setImage(mean_matrix, autoLevels=False, levels=(-vmax, vmax))
        img_item.setRect(QRectF(t_ms[0], -0.5, t_ms[-1] - t_ms[0], n_ch))
        heat_plot.addItem(img_item)

        # Reference Zero Time Vertical Line
        zero_line = pg.InfiniteLine(pos=0.0, angle=90, pen=pg.mkPen("black", width=1.5, style=Qt.DashLine))
        heat_plot.addItem(zero_line)

        self.heat_plot = heat_plot
        self.img_item = img_item
        self.zero_line = zero_line

        # Waveforms Overlay & Lag Markers
        if self.chk_overlay_traces.isChecked():
            scale = 0.38 / (vmax if vmax > 0 else 1.0)
            n_events = snippets.shape[1]

            # Individual traces in gray
            all_snip_x = []
            all_snip_y = []
            for ch_i in range(n_ch):
                for ev_i in range(n_events):
                    tr = snippets[ch_i, ev_i, :]
                    y_tr = ch_i - (tr * scale)
                    all_snip_x.extend([t_ms, [np.nan]])
                    all_snip_y.extend([y_tr, [np.nan]])

            if all_snip_x:
                snips_curve = pg.PlotDataItem(
                    np.concatenate(all_snip_x),
                    np.concatenate(all_snip_y),
                    pen=pg.mkPen(color=(120, 120, 120, 50), width=1),
                    connect="finite",
                )
                heat_plot.addItem(snips_curve)

            # Mean waveforms in black
            all_mean_x = []
            all_mean_y = []
            for ch_i in range(n_ch):
                m_tr = mean_matrix[ch_i, :]
                y_mean = ch_i - (m_tr * scale)
                all_mean_x.extend([t_ms, [np.nan]])
                all_mean_y.extend([y_mean, [np.nan]])

            if all_mean_x:
                mean_curve = pg.PlotDataItem(
                    np.concatenate(all_mean_x),
                    np.concatenate(all_mean_y),
                    pen=pg.mkPen("black", width=2),
                    connect="finite",
                )
                heat_plot.addItem(mean_curve)

            # Lag Scatter Points (Diamonds)
            lag_scatter = pg.ScatterPlotItem(
                x=lag_ms,
                y=np.arange(n_ch),
                symbol="d",
                size=12,
                brush=pg.mkBrush("#00e5ff"),
                pen=pg.mkPen("black", width=1.2),
            )
            heat_plot.addItem(lag_scatter)
            self.lag_scatter = lag_scatter

            # Lag Trend Line for detected channels
            det_indices = [i for i, ch in enumerate(channels) if ch in detected_channels]
            if len(det_indices) > 1:
                det_sorted = sorted(det_indices)
                det_x = [lag_ms[i] for i in det_sorted]
                det_y = det_sorted
                trend_line = pg.PlotDataItem(
                    det_x,
                    det_y,
                    pen=pg.mkPen(color="#ff1744", width=2, style=Qt.DashLine),
                )
                heat_plot.addItem(trend_line)
                self.trend_line = trend_line

            # Lag text callouts
            dt_range = t_ms[-1] - t_ms[0]
            for i in det_indices:
                lag = lag_ms[i]
                txt = f"{lag:+.2f} ms"
                lbl = pg.TextItem(text=txt, color="white", fill=pg.mkBrush(0, 0, 0, 180))
                lbl.setPos(lag + (dt_range * 0.02), i)
                heat_plot.addItem(lbl)

        heat_plot.setXRange(t_ms[0], t_ms[-1], padding=0)
        heat_plot.setYRange(-0.5, n_ch - 0.5, padding=0)

        # 2. Right Plot: PTP Amplitude Bar Chart
        bar_plot = self.glw.addPlot(row=0, col=1)
        bar_plot.invertY(True)
        bar_plot.setYLink(heat_plot)
        bar_plot.getAxis("left").setStyle(showValues=False)
        bar_plot.setLabel("bottom", "PTP Amplitude (uV)")
        bar_plot.showGrid(x=True, y=True, alpha=0.2)
        self.bar_plot = bar_plot

        # Draw bars categorized by channel status
        y_pos = np.arange(n_ch)
        for i, ch in enumerate(channels):
            amp = ptp_amp[i]
            if ch == ref_chan:
                c = "#1976d2"  # Primary reference: Blue
            elif ch in detected_channels:
                c = "#388e3c"  # Significant footprint neighbor: Green
            else:
                c = "#cfd8dc"  # Background contact: Gray

            bar = pg.BarGraphItem(x0=0, y0=i - 0.3, width=amp, height=0.6, brush=pg.mkBrush(c), pen=pg.mkPen("black", width=0.8))
            bar_plot.addItem(bar)
            self.bars.append(bar)

        max_amp = float(np.max(ptp_amp)) if len(ptp_amp) > 0 else 1.0
        bar_plot.setXRange(0, max_amp * 1.35 if max_amp > 0 else 1.0, padding=0)
        bar_plot.setYRange(-0.5, n_ch - 0.5, padding=0)

        self.glw.ci.layout.setColumnStretchFactor(0, 4)
        self.glw.ci.layout.setColumnStretchFactor(1, 1)

    def _plot_multi_unit_grid(self):
        units = [
            sg
            for sg in self.state.spike_groups
            if sg is not None and sg.get_number_of_events() > 0
        ]
        if not units:
            units = [sg for sg in self.state.spike_groups if sg is not None]
        if not units:
            p = self.glw.addPlot(row=0, col=0)
            txt = pg.TextItem(text="No tracked units available", color=(100, 100, 100), anchor=(0.5, 0.5))
            p.addItem(txt)
            p.showGrid(False, False)
            return

        n_units = len(units)
        n_cols = min(n_units, 3)

        for u_idx, sg in enumerate(units):
            row = u_idx // n_cols
            col = u_idx % n_cols
            plot = self.glw.addPlot(row=row, col=col)
            plot.invertY(True)

            fp = self.compute_unit_footprint(sg)
            if fp is None:
                unit_title = (
                    getattr(sg.event, "name", f"Unit {u_idx}")
                    if hasattr(sg, "event") and sg.event is not None
                    else f"Unit {u_idx}"
                )
                plot.setTitle(f"<small>{unit_title} (No Data)</small>")
                plot.showAxis("left", False)
                plot.showAxis("bottom", False)
                continue

            channels = fp["channels"]
            t_ms = fp["t_ms"]
            mean_matrix = fp["mean_matrix"]
            ptp_amp = fp["ptp_amp"]
            lag_ms = fp["lag_ms"]
            detected_channels = fp["detected_channels"]
            best_ch_idx = int(np.argmax(ptp_amp)) if len(ptp_amp) > 0 else 0
            best_ch = channels[best_ch_idx] if channels else "None"
            n_ch = len(channels)

            sg_original_idx = (
                self.state.spike_groups.index(sg)
                if sg in self.state.spike_groups
                else u_idx
            )
            self.grid_plots.append((plot, sg_original_idx, best_ch))

            vmax = float(np.max(np.abs(mean_matrix))) if np.max(np.abs(mean_matrix)) > 0 else 1.0
            img_item = pg.ImageItem(axisOrder="row-major")
            img_item.setLookupTable(self._cmap_lut)
            img_item.setImage(mean_matrix, autoLevels=False, levels=(-vmax, vmax))
            img_item.setRect(QRectF(t_ms[0], -0.5, t_ms[-1] - t_ms[0], n_ch))
            plot.addItem(img_item)

            zero_line = pg.InfiniteLine(pos=0.0, angle=90, pen=pg.mkPen("black", width=1, style=Qt.DashLine))
            plot.addItem(zero_line)

            if self.chk_overlay_traces.isChecked():
                scale = 0.38 / (vmax if vmax > 0 else 1.0)
                all_mean_x = []
                all_mean_y = []
                for ch_i in range(n_ch):
                    m_tr = mean_matrix[ch_i, :]
                    y_mean = ch_i - (m_tr * scale)
                    all_mean_x.extend([t_ms, [np.nan]])
                    all_mean_y.extend([y_mean, [np.nan]])

                if all_mean_x:
                    mean_curve = pg.PlotDataItem(
                        np.concatenate(all_mean_x),
                        np.concatenate(all_mean_y),
                        pen=pg.mkPen("black", width=1.2),
                        connect="finite",
                    )
                    plot.addItem(mean_curve)

                lag_scatter = pg.ScatterPlotItem(
                    x=lag_ms,
                    y=np.arange(n_ch),
                    symbol="d",
                    size=8,
                    brush=pg.mkBrush("#00e5ff"),
                    pen=pg.mkPen("black", width=0.8),
                )
                plot.addItem(lag_scatter)

            if col == 0:
                ticks = [(i, ch) for i, ch in enumerate(channels)]
                plot.getAxis("left").setTicks([ticks])
            else:
                plot.getAxis("left").setStyle(showValues=False)

            plot.setXRange(t_ms[0], t_ms[-1], padding=0)
            plot.setYRange(-0.5, n_ch - 0.5, padding=0)

            unit_name = (
                getattr(sg.event, "name", f"Unit {u_idx}")
                if hasattr(sg, "event") and sg.event is not None
                else f"Unit {u_idx}"
            )
            plot.setTitle(f"<b>{unit_name}</b> (N={fp['n_events']}, Best: {best_ch})")

    def _on_scene_clicked(self, event):
        """Unified click handler dispatching to composite or grid mode handler."""
        if getattr(event, "button", None) != Qt.LeftButton:
            return
        if self.combo_mode.currentIndex() == 0:
            self._on_plot_clicked(event)
        elif self.combo_mode.currentIndex() == 1:
            self._on_grid_plot_clicked(event)

    def _on_plot_clicked(self, event):
        """Click handler for selecting primary channel in Single Unit Composite mode."""
        if self.combo_mode.currentIndex() != 0 or not self.current_channels:
            return
        plot_item = self.glw.getItem(0, 0)
        if plot_item is None or not hasattr(plot_item, "vb"):
            return
        if not plot_item.sceneBoundingRect().contains(event.scenePos()):
            return
        try:
            pos = plot_item.vb.mapSceneToView(event.scenePos())
            if np.isnan(pos.y()):
                return
            ch_idx = int(round(pos.y()))
            if 0 <= ch_idx < len(self.current_channels):
                clicked_ch = self.current_channels[ch_idx]
                if self.state is not None and hasattr(self.state, "set_active_channel"):
                    self.state.set_active_channel(clicked_ch)
        except Exception:
            pass

    def _on_grid_plot_clicked(self, event):
        """Click handler for jumping to unit and channel in Multi-Unit Grid mode."""
        if self.combo_mode.currentIndex() != 1 or not self.grid_plots:
            return
        for plot, sg_idx, best_ch in self.grid_plots:
            if plot.sceneBoundingRect().contains(event.scenePos()):
                if self.state is not None:
                    if hasattr(self.state, "setUnitGroup"):
                        self.state.setUnitGroup(sg_idx)
                    if best_ch and hasattr(self.state, "set_active_channel"):
                        self.state.set_active_channel(best_ch)
                break

    def get_settings(self) -> dict:
        """Returns perspective settings dictionary."""
        res = {
            "mode": self.combo_mode.currentIndex(),
            "last_n": self.spin_last_n.value(),
            "window_ms": self.spin_window_ms.value(),
            "align_peaks": self.chk_align_peaks.isChecked(),
            "overlay_traces": self.chk_overlay_traces.isChecked(),
        }
        if hasattr(self, "channel_bar"):
            res["channel_bar"] = self.channel_bar.get_settings()
        return res

    def set_settings(self, values: dict):
        """Restores perspective settings."""
        if not isinstance(values, dict):
            return
        if "channel_bar" in values and hasattr(self, "channel_bar"):
            self.channel_bar.set_settings(values["channel_bar"])
        if "mode" in values:
            self.combo_mode.setCurrentIndex(int(values["mode"]))
        if "last_n" in values:
            self.spin_last_n.setValue(int(values["last_n"]))
        if "window_ms" in values:
            self.spin_window_ms.setValue(float(values["window_ms"]))
        if "align_peaks" in values:
            self.chk_align_peaks.setChecked(bool(values["align_peaks"]))
        if "overlay_traces" in values:
            self.chk_overlay_traces.setChecked(bool(values["overlay_traces"]))
        self.update_plot()
