import itertools
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import quantities as pq
from neo import Event

import PySide6
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, QTimer, Slot, QRectF, QPointF, QSignalBlocker
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
from .live_orchestrator import LiveFrameConsumerMixin
from .ChannelGroupBar import ChannelGroupBar
from .helpers import quiet_fail, run_adaptive_render, notify_view_selected
from .PGNavigation import PGNavigationMixin

logger = logging.getLogger(__name__)


def _get_mpl_lut(name: str = "RdBu_r", n: int = 256) -> np.ndarray:
    """Sample a Matplotlib colormap into an RGBA lookup table for PyQtGraph."""
    cmap = plt.get_cmap(name)
    colors = cmap(np.linspace(0, 1, n))
    return (colors * 255).astype(np.uint8)


class UnitFootprintPGView(LiveFrameConsumerMixin, PGNavigationMixin, QMainWindow):
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
        self.heat_plot = None
        self.img_item = None
        self.zero_line = None
        self.snips_curve = None
        self.mean_curve = None
        self.live_trace_curve = None
        self.bar_plot = None
        self.lag_scatter = None
        self.trend_line = None
        self.bars = []
        self.lag_labels = []
        self._built_mode = None
        self._grid_items = []
        self._grid_layout_size = None

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

        self.chk_show_live_trace = QCheckBox("Live Sweep", self)
        self.chk_show_live_trace.setChecked(False)
        self.chk_show_live_trace.setToolTip(
            "Overlay the latest stimulus sweep waveform aligned to the unit's mean latency"
        )

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
        self.control_toolbar.addWidget(self.chk_show_live_trace)
        self.control_toolbar.addSeparator()
        self.control_toolbar.addWidget(self.btn_auto_assign_channels)
        self.add_navigation_actions(self.control_toolbar)
        self.init_navigation()
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

        self.connections: List[Tuple] = []

        # Signal connections for UI controls
        self.combo_mode.currentIndexChanged.connect(self.update_plot)
        self.spin_last_n.valueChanged.connect(self.update_plot)
        self.spin_window_ms.valueChanged.connect(self.update_plot)
        self.chk_align_peaks.toggled.connect(self.update_plot)
        self.chk_overlay_traces.toggled.connect(self.update_plot)
        self.chk_show_live_trace.toggled.connect(self.update_plot)
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
            run_adaptive_render(self, self._do_update_plot, weight="normal")
        except RuntimeError:
            pass
        except Exception:
            logger.exception("UnitFootprint render failed; skipping frame")

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

    def closeEvent(self, event):
        self.unset_connections()
        super().closeEvent(event)

    @Slot()
    def _on_state_reset(self, *args):
        self.schedule_update()

    @Slot()
    @Slot(int)
    def _on_unit_params_changed(self, unit_idx: int = -1):
        if unit_idx < 0 or unit_idx == self.get_target_unit_idx():
            self.schedule_update()

    @Slot()
    @Slot(int)
    def _on_new_spike_events(self, unit_idx: int = -1):
        if unit_idx < 0 or unit_idx == self.get_target_unit_idx():
            self.schedule_update()

    @Slot()
    @Slot(int)
    def _on_unit_change(self, *args):
        if self.get_target_unit_idx() == getattr(self.state, "cur_spike_group", 0):
            self.schedule_update()

    @Slot()
    @Slot(str)
    def _on_channel_change(self, *args):
        self.schedule_update()

    def set_state(self, state: ViewerState):
        self.unset_connections()
        self.state = state
        if hasattr(self, "channel_bar"):
            self.channel_bar.setState(state)
        if self.state is not None:
            if hasattr(self.state, "onUnitChange"):
                self._connect_state(self.state.onUnitChange, self._on_unit_change)
            if hasattr(self.state, "onUnitGroupChange"):
                self._connect_state(self.state.onUnitGroupChange, self._on_state_reset)
            if hasattr(self.state, "onLoadNewFile"):
                self._register_data_reload(
                    self.state,
                    self._on_state_reset,
                )
            if hasattr(self.state, "onNewSpikeEvents"):
                self._connect_state(self.state.onNewSpikeEvents, self._on_new_spike_events)
            if hasattr(self.state, "onUnitParamsChanged"):
                self._connect_state(self.state.onUnitParamsChanged, self._on_unit_params_changed)
            if hasattr(self.state, "onChannelGroupsChanged"):
                self._connect_state(self.state.onChannelGroupsChanged, self._on_state_reset)
            if hasattr(self.state, "onChannelChange"):
                self._connect_state(self.state.onChannelChange, self._on_channel_change)
            if hasattr(self.state, "onNewSweep"):
                self._register_live_frames(
                    self.state,
                    lambda _stimno: self.schedule_update(),
                )
            if hasattr(self.state, "onUIRefreshRateChange") and hasattr(self, "render_timer"):
                self.render_timer.setInterval(self.state.get_ui_refresh_rate_ms())
                self._connect_state(self.state.onUIRefreshRateChange, self.render_timer.setInterval)
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
        include_latest_sweep: Optional[bool] = None,
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
        if include_latest_sweep is None:
            include_latest_sweep = (
                self.chk_show_live_trace.isChecked()
                if hasattr(self, "chk_show_live_trace")
                else False
            )

        from .dsp.footprint import compute_unit_footprint_from_state

        return compute_unit_footprint_from_state(
            self.state,
            sg,
            last_n=last_n,
            window_ms=window_ms,
            align_peaks=align_peaks,
            include_latest_sweep=include_latest_sweep,
        )

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

        from .dsp.footprint import auto_detect_channels_from_footprint

        primary_chan, neighbors = auto_detect_channels_from_footprint(fp)
        if not primary_chan:
            return

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

    def _reset_graphics_pools(self):
        self.grid_plots = []
        self.current_channels = []
        self.heat_plot = None
        self.img_item = None
        self.zero_line = None
        self.snips_curve = None
        self.mean_curve = None
        self.live_trace_curve = None
        self.bar_plot = None
        self.lag_scatter = None
        self.trend_line = None
        self.bars = []
        self.lag_labels = []
        self._built_mode = None
        self._grid_items = []
        self._grid_layout_size = None

    @quiet_fail
    def _do_update_plot(self):
        if self.is_view_culled():
            self._dirty = True
            return

        self.render_count = getattr(self, "render_count", 0) + 1

        if self.state is None:
            if self._built_mode is not None:
                self.glw.clear()
                self._reset_graphics_pools()
            return

        mode_idx = self.combo_mode.currentIndex()
        if mode_idx != self._built_mode:
            self.glw.clear()
            self._reset_graphics_pools()
            self._built_mode = mode_idx

        if mode_idx == 0:
            self._plot_single_unit_composite()
        else:
            self._plot_multi_unit_grid()

    def _show_composite_placeholder(self, text: str):
        if self.heat_plot is not None and getattr(self.heat_plot, "placeholder_text", None) is not None:
            self.heat_plot.placeholder_text.setText(text)
            return
        self.glw.clear()
        self._reset_graphics_pools()
        self._built_mode = 0
        p = self.glw.addPlot(row=0, col=0)
        p.showGrid(False, False)
        txt = pg.TextItem(text=text, color=(100, 100, 100), anchor=(0.5, 0.5))
        p.addItem(txt)
        p.placeholder_text = txt
        self.heat_plot = p
        self.current_channels = []

    def _ensure_composite_layout(self):
        if self.heat_plot is not None and getattr(self.heat_plot, "placeholder_text", None) is None:
            return
        self.glw.clear()
        self._reset_graphics_pools()
        self._built_mode = 0

        heat_plot = self.glw.addPlot(row=0, col=0)
        heat_plot.invertY(True)
        heat_plot.setLabel("bottom", "Time", units="ms")
        heat_plot.setLabel("left", "Recording Channel")
        heat_plot.showGrid(x=True, y=True, alpha=0.2)
        heat_plot.placeholder_text = None

        img_item = pg.ImageItem(axisOrder="row-major")
        img_item.setLookupTable(self._cmap_lut)
        heat_plot.addItem(img_item)

        zero_line = pg.InfiniteLine(pos=0.0, angle=90, pen=pg.mkPen("black", width=1.5, style=Qt.DashLine))
        heat_plot.addItem(zero_line)

        snips_curve = pg.PlotDataItem(
            connect="finite", pen=pg.mkPen(color=(120, 120, 120, 50), width=1)
        )
        heat_plot.addItem(snips_curve)

        mean_curve = pg.PlotDataItem(connect="finite", pen=pg.mkPen("black", width=2))
        heat_plot.addItem(mean_curve)

        live_trace_curve = pg.PlotDataItem(
            connect="finite", pen=pg.mkPen(color="#ff9100", width=1.8, style=Qt.DashLine)
        )
        heat_plot.addItem(live_trace_curve)

        lag_scatter = pg.ScatterPlotItem(
            symbol="d", size=12, brush=pg.mkBrush("#00e5ff"), pen=pg.mkPen("black", width=1.2)
        )
        heat_plot.addItem(lag_scatter)

        trend_line = pg.PlotDataItem(pen=pg.mkPen(color="#ff1744", width=2, style=Qt.DashLine))
        heat_plot.addItem(trend_line)

        bar_plot = self.glw.addPlot(row=0, col=1)
        bar_plot.invertY(True)
        bar_plot.setYLink(heat_plot)
        bar_plot.getAxis("left").setStyle(showValues=False)
        bar_plot.setLabel("bottom", "PTP Amplitude (uV)")
        bar_plot.showGrid(x=True, y=True, alpha=0.2)

        self.glw.ci.layout.setColumnStretchFactor(0, 4)
        self.glw.ci.layout.setColumnStretchFactor(1, 1)

        self.heat_plot = heat_plot
        self.img_item = img_item
        self.zero_line = zero_line
        self.snips_curve = snips_curve
        self.mean_curve = mean_curve
        self.live_trace_curve = live_trace_curve
        self.lag_scatter = lag_scatter
        self.trend_line = trend_line
        self.bar_plot = bar_plot

    def _ensure_bars(self, n_ch: int):
        while len(self.bars) > n_ch:
            bar = self.bars.pop()
            self.bar_plot.removeItem(bar)
        while len(self.bars) < n_ch:
            bar = pg.BarGraphItem(x0=0, y0=0, width=0, height=0.6, brush=pg.mkBrush("#cfd8dc"), pen=pg.mkPen("black", width=0.8))
            self.bar_plot.addItem(bar)
            self.bars.append(bar)

    def _ensure_lag_labels(self, n: int):
        while len(self.lag_labels) > n:
            lbl = self.lag_labels.pop()
            self.heat_plot.removeItem(lbl)
        while len(self.lag_labels) < n:
            lbl = pg.TextItem(text="", color="white", fill=pg.mkBrush(0, 0, 0, 180))
            self.heat_plot.addItem(lbl)
            self.lag_labels.append(lbl)

    def _plot_single_unit_composite(self):
        sg = self.get_target_unit()
        if sg is None:
            self._show_composite_placeholder("No unit selected")
            return

        fp = self.compute_unit_footprint(sg)
        if fp is None:
            self._show_composite_placeholder("Insufficient events or channel data for unit footprint")
            return

        self._ensure_composite_layout()
        self.heat_plot.placeholder_text = None

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

        vmax = float(np.max(np.abs(mean_matrix))) if np.max(np.abs(mean_matrix)) > 0 else 1.0
        scale = 0.38 / (vmax if vmax > 0 else 1.0)

        ticks = [(i, ch) for i, ch in enumerate(channels)]
        self.heat_plot.getAxis("left").setTicks([ticks])

        unit_name = (
            getattr(sg.event, "name", f"Unit {self.get_target_unit_idx()}")
            if hasattr(sg, "event") and sg.event is not None
            else f"Unit {self.get_target_unit_idx()}"
        )
        self.heat_plot.setTitle(f"<b>{unit_name} Footprint ({fp['n_events']} events, Ref: {ref_chan})</b>")

        self.img_item.setImage(mean_matrix, autoLevels=False, levels=(-vmax, vmax))
        self.img_item.setRect(QRectF(t_ms[0], -0.5, t_ms[-1] - t_ms[0], n_ch))

        if self.chk_overlay_traces.isChecked():
            n_events = snippets.shape[1]

            all_snip_x = []
            all_snip_y = []
            for ch_i in range(n_ch):
                for ev_i in range(n_events):
                    tr = snippets[ch_i, ev_i, :]
                    y_tr = ch_i - (tr * scale)
                    all_snip_x.extend([t_ms, [np.nan]])
                    all_snip_y.extend([y_tr, [np.nan]])

            if all_snip_x:
                self.snips_curve.setData(np.concatenate(all_snip_x), np.concatenate(all_snip_y))
            else:
                self.snips_curve.setData([], [])

            all_mean_x = []
            all_mean_y = []
            for ch_i in range(n_ch):
                m_tr = mean_matrix[ch_i, :]
                y_mean = ch_i - (m_tr * scale)
                all_mean_x.extend([t_ms, [np.nan]])
                all_mean_y.extend([y_mean, [np.nan]])

            if all_mean_x:
                self.mean_curve.setData(np.concatenate(all_mean_x), np.concatenate(all_mean_y))
            else:
                self.mean_curve.setData([], [])

            self.lag_scatter.setData(x=lag_ms, y=np.arange(n_ch))

            det_indices = [i for i, ch in enumerate(channels) if ch in detected_channels]
            if len(det_indices) > 1:
                det_sorted = sorted(det_indices)
                det_x = [lag_ms[i] for i in det_sorted]
                det_y = det_sorted
                self.trend_line.setData(det_x, det_y)
                self.trend_line.show()
            else:
                self.trend_line.hide()

            dt_range = t_ms[-1] - t_ms[0]
            self._ensure_lag_labels(len(det_indices))
            for lbl_i, i in enumerate(det_indices):
                lag = lag_ms[i]
                self.lag_labels[lbl_i].setText(f"{lag:+.2f} ms")
                self.lag_labels[lbl_i].setPos(lag + (dt_range * 0.02), i)
                self.lag_labels[lbl_i].show()
            self.snips_curve.show()
            self.mean_curve.show()
            self.lag_scatter.show()
        else:
            self.snips_curve.hide()
            self.mean_curve.hide()
            self.lag_scatter.hide()
            self.trend_line.hide()
            self._ensure_lag_labels(0)

        live_wfs = fp.get("latest_sweep_waveforms")
        if self.chk_show_live_trace.isChecked() and live_wfs is not None and self.live_trace_curve is not None:
            all_live_x = []
            all_live_y = []
            for ch_i in range(min(n_ch, live_wfs.shape[0])):
                l_tr = live_wfs[ch_i, :]
                y_live = ch_i - (l_tr * scale)
                all_live_x.extend([t_ms, [np.nan]])
                all_live_y.extend([y_live, [np.nan]])
            if all_live_x:
                self.live_trace_curve.setData(np.concatenate(all_live_x), np.concatenate(all_live_y))
                self.live_trace_curve.show()
            else:
                self.live_trace_curve.setData([], [])
                self.live_trace_curve.hide()
        elif self.live_trace_curve is not None:
            self.live_trace_curve.setData([], [])
            self.live_trace_curve.hide()

        self.heat_plot.setXRange(t_ms[0], t_ms[-1], padding=0)
        self.heat_plot.setYRange(-0.5, n_ch - 0.5, padding=0)

        self._ensure_bars(n_ch)
        for i, ch in enumerate(channels):
            amp = ptp_amp[i]
            if ch == ref_chan:
                c = "#1976d2"
            elif ch in detected_channels:
                c = "#388e3c"
            else:
                c = "#cfd8dc"
            self.bars[i].setOpts(x0=0, y0=i - 0.3, width=amp, height=0.6, brush=pg.mkBrush(c), pen=pg.mkPen("black", width=0.8))

        max_amp = float(np.max(ptp_amp)) if len(ptp_amp) > 0 else 1.0
        self.bar_plot.setXRange(0, max_amp * 1.35 if max_amp > 0 else 1.0, padding=0)
        self.bar_plot.setYRange(-0.5, n_ch - 0.5, padding=0)

    def _ensure_grid_layout(self, n_units: int, n_cols: int):
        expected = (n_units, n_cols)
        if self._grid_layout_size == expected and len(self._grid_items) == n_units:
            return
        self.glw.clear()
        self._reset_graphics_pools()
        self._built_mode = 1
        self._grid_layout_size = expected

        for u_idx in range(n_units):
            row = u_idx // n_cols
            col = u_idx % n_cols
            plot = self.glw.addPlot(row=row, col=col)
            plot.invertY(True)

            img_item = pg.ImageItem(axisOrder="row-major")
            img_item.setLookupTable(self._cmap_lut)
            plot.addItem(img_item)

            zero_line = pg.InfiniteLine(pos=0.0, angle=90, pen=pg.mkPen("black", width=1, style=Qt.DashLine))
            plot.addItem(zero_line)

            mean_curve = pg.PlotDataItem(connect="finite", pen=pg.mkPen("black", width=1.2))
            plot.addItem(mean_curve)

            lag_scatter = pg.ScatterPlotItem(
                symbol="d", size=8, brush=pg.mkBrush("#00e5ff"), pen=pg.mkPen("black", width=0.8)
            )
            plot.addItem(lag_scatter)

            self._grid_items.append((plot, img_item, zero_line, mean_curve, lag_scatter))
            if col != 0:
                plot.getAxis("left").setStyle(showValues=False)

    def _plot_multi_unit_grid(self):
        units = [
            sg
            for sg in self.state.spike_groups
            if sg is not None and sg.get_number_of_events() > 0
        ]
        if not units:
            units = [sg for sg in self.state.spike_groups if sg is not None]
        if not units:
            self._show_composite_placeholder("No tracked units available")
            return

        n_units = len(units)
        n_cols = min(n_units, 3)
        self._ensure_grid_layout(n_units, n_cols)
        del self.grid_plots[:]

        for u_idx, sg in enumerate(units):
            plot, img_item, zero_line, mean_curve, lag_scatter = self._grid_items[u_idx]
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
                img_item.clear()
                mean_curve.setData([], [])
                lag_scatter.setData(x=[], y=[])
                self.grid_plots.append((plot, -1, None))
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

            vmax = float(np.max(np.abs(mean_matrix))) if np.max(np.abs(mean_matrix)) > 0 else 1.0
            img_item.setImage(mean_matrix, autoLevels=False, levels=(-vmax, vmax))
            img_item.setRect(QRectF(t_ms[0], -0.5, t_ms[-1] - t_ms[0], n_ch))

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
                    mean_curve.setData(np.concatenate(all_mean_x), np.concatenate(all_mean_y))
                else:
                    mean_curve.setData([], [])
                lag_scatter.setData(x=lag_ms, y=np.arange(n_ch))
                mean_curve.show()
                lag_scatter.show()
            else:
                mean_curve.hide()
                lag_scatter.hide()

            if u_idx % n_cols == 0:
                ticks = [(i, ch) for i, ch in enumerate(channels)]
                plot.getAxis("left").setTicks([ticks])
                plot.showAxis("left", True)
            else:
                plot.getAxis("left").setStyle(showValues=False)

            plot.showAxis("bottom", True)
            plot.setXRange(t_ms[0], t_ms[-1], padding=0)
            plot.setYRange(-0.5, n_ch - 0.5, padding=0)

            unit_name = (
                getattr(sg.event, "name", f"Unit {u_idx}")
                if hasattr(sg, "event") and sg.event is not None
                else f"Unit {u_idx}"
            )
            plot.setTitle(f"<b>{unit_name}</b> (N={fp['n_events']}, Best: {best_ch})")

            self.grid_plots.append((plot, sg_original_idx, best_ch))

    def reset_view(self):
        self._schedule_render()

    def mousePressEvent(self, event):
        notify_view_selected(self)
        super().mousePressEvent(event)

    def _on_scene_clicked(self, event):
        """Unified click handler dispatching to composite or grid mode handler."""
        notify_view_selected(self)
        if self.is_nav_active() or getattr(event, "button", None) != Qt.LeftButton:
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
            "show_live_trace": self.chk_show_live_trace.isChecked(),
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
        if "show_live_trace" in values:
            self.chk_show_live_trace.setChecked(bool(values["show_live_trace"]))
        self.update_plot()
