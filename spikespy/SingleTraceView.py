import sys
from dataclasses import dataclass, field
from typing import Any, List, Optional, Union

from scipy.signal import find_peaks
import matplotlib
import matplotlib.style as mplstyle
import numpy as np
import PySide6
import quantities as pq
from matplotlib.backend_bases import MouseEvent
from matplotlib.backends.backend_qtagg import FigureCanvas, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    QObject,
    Qt,
    Signal,
    Slot,
    QThread,
    QThreadPool,
    QRunnable,
    QTimer,
)
from .helpers import qsignal_throttle
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QMainWindow,
    QMdiArea,
    QMdiSubWindow,
    QMenu,
    QMenuBar,
    QPushButton,
    QSpinBox,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from scipy.signal import fftconvolve, find_peaks
from .processing import create_erp_signals
from .ViewerState import ViewerState
from .ChannelGroupBar import ChannelGroupBar
import quantities as pq
from matplotlib.ticker import Formatter

mplstyle.use("fast")


class SweepConvProxy:
    """
    Lightweight lazy wrapper for template convolution.
    Instead of pre-allocating an (N, W) array across the entire recording,
    self.conv[stimno] computes and returns the 1D float32 multi-channel convolution
    for `stimno` on demand with an LRU cache.
    """

    def __init__(self, view: "SingleTraceView", max_cache: int = 32):
        self.view = view
        self._cache = {}
        self._max_cache = max_cache

    def clear_cache(self):
        self._cache.clear()

    def __getitem__(self, stimno: int) -> np.ndarray:
        if stimno in self._cache:
            return self._cache[stimno]
        conv_res = self.view._compute_sweep_conv(stimno)
        if len(self._cache) >= self._max_cache:
            self._cache.pop(next(iter(self._cache)))
        self._cache[stimno] = conv_res
        return conv_res

    def __len__(self):
        erp = self.view.state.get_erp() if self.view and self.view.state else None
        return erp.shape[0] if erp is not None else 0


# Custom formatter class


class SingleTraceView(QMainWindow):
    def __init__(
        self,
        parent: Optional[PySide6.QtWidgets.QWidget] = None,
        state: ViewerState = None,
    ) -> None:
        super().__init__(parent)
        self.state = state

        xsize = 1024
        ysize = 480
        dpi = 100
        self.closest_pos = 0  # the closest inflection to the current spike
        self.fig = Figure(figsize=(xsize / dpi, ysize / dpi), dpi=dpi)
        #  create widgets
        self.view = FigureCanvas(self.fig)

        self.toolbar = NavigationToolbar2QT(self.view, self)
        self.gs = self.fig.add_gridspec(2, 1, height_ratios=[1, 5], hspace=0)

        self.ax = self.fig.add_subplot(self.gs[1, 0])
        self.topax = self.fig.add_subplot(self.gs[0, 0], sharex=self.ax)
        self.topax.yaxis.set_visible(False)
        self.topax.xaxis.set_visible(False)
        self.topax.spines["top"].set_visible(False)
        self.topax.spines["left"].set_visible(False)
        self.topax.spines["bottom"].set_visible(False)
        self.topax.spines["right"].set_visible(False)
        self.topax.set_ylim(0, 1)

        self.fig.tight_layout()
        self.addToolBar(self.toolbar)

        # Multi-Channel Overlay toolbar controls
        self.chk_multichannel = QCheckBox("Multi-Channel Overlay", self)
        self.chk_multichannel.setChecked(False)
        self.chk_multichannel_conv = QCheckBox("Multi-Ch Conv", self)
        self.chk_multichannel_conv.setChecked(True)
        self.chk_multichannel_conv.setToolTip("Combine template convolution across primary and overlay/unit channels")
        self.chk_unit_neighbors_only = QCheckBox("Unit Channels Only", self)
        self.chk_unit_neighbors_only.setChecked(False)
        self.chk_unit_neighbors_only.setToolTip("Only overlay channels associated with the active unit (primary + neighbor channels)")
        self.spin_chan_spacing = QSpinBox(self)
        self.spin_chan_spacing.setRange(1, 500)
        self.spin_chan_spacing.setValue(50)
        self.spin_chan_spacing.setSuffix(" uV offset")
        self.spin_chan_spacing.setToolTip("Vertical spacing between stacked multi-channel traces")

        self.toolbar.addSeparator()
        self.toolbar.addWidget(self.chk_multichannel)
        self.toolbar.addWidget(self.chk_multichannel_conv)
        self.toolbar.addWidget(self.chk_unit_neighbors_only)
        self.toolbar.addWidget(self.spin_chan_spacing)

        # Independent channel group and unit binding toolbar
        self.channel_bar = ChannelGroupBar(parent=self, state=self.state)
        self.channel_bar.bindingChanged.connect(self._on_channel_bar_changed)
        self.addToolBarBreak()
        self.addToolBar(Qt.TopToolBarArea, self.channel_bar)

        self.aux_trace_lines = []
        self.aux_trace_labels = []

        self.identified_spike_line = self.ax.axvline(
            6000, zorder=0, visible=False, animated=True, alpha=0.3, color="blue"
        )
        self.trace_line_cache = None
        self.scatter_peaks = None

        self.setCentralWidget(self.view)

        self.render_count: int = 0
        self._dirty: bool = False
        self._user_hidden: bool = False
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

        # connect to signals
        self.updateFigure = self._mark_dirty_and_schedule

        self.updateHistogram = qsignal_throttle(self.updateHistogram_core, 1000)

        self.chk_multichannel.toggled.connect(self.setupFigure)
        self.chk_multichannel.toggled.connect(self.updateFigure)
        self.chk_multichannel_conv.toggled.connect(self.updateHistogram)
        self.chk_unit_neighbors_only.toggled.connect(self.setupFigure)
        self.chk_unit_neighbors_only.toggled.connect(self.updateFigure)
        self.spin_chan_spacing.valueChanged.connect(self.setupFigure)
        self.spin_chan_spacing.valueChanged.connect(self.updateFigure)

        if self.state is not None:
            self.state.onLoadNewFile.connect(self.updateFigure)

            self.state.onUnitChange.connect(lambda *args: setattr(self, "_histogram_dirty", True))
            self.state.onUnitChange.connect(self.updateFigure)
            self.state.onUnitChange.connect(self.updateHistogram)

            self.state.onUnitGroupChange.connect(lambda *args: setattr(self, "_histogram_dirty", True))
            self.state.onUnitGroupChange.connect(self.updateFigure)
            self.state.onUnitGroupChange.connect(self.updateHistogram)
            self.state.onStimNoChange.connect(self.updateFigure)

            if hasattr(self.state, "onNewSweep"):
                self.state.onNewSweep.connect(self.on_new_sweep)
            if hasattr(self.state, "onNewSpikeEvents"):
                self.state.onNewSpikeEvents.connect(lambda *args: setattr(self, "_histogram_dirty", True))
                self.state.onNewSpikeEvents.connect(lambda *args: self.updateFigure())
            if hasattr(self.state, "onChannelChange"):
                self.state.onChannelChange.connect(self.setupFigure)

        self.fig.canvas.mpl_connect("button_press_event", self.view_clicked)
        # self.fig.canvas.mpl_connect('key_press_event', self.keyPressEvent)
        self.select_local_maxima_width = 1
        self.closest_pos = 0

        self.setFocusPolicy(Qt.ClickFocus)
        self.setFocus()

        self.blit_data = None

        def draw_evt(evt):
            self.blit()
            self.updateFigure()

        self.fig.canvas.mpl_connect("draw_event", draw_evt)
        self.topax_lines = []

        # Persistent animated Line2D artists on self.topax
        self.topax_cur_line = self.topax.plot([], [], color="blue", animated=True)[0]
        self.topax_prev_line = self.topax.plot([], [], color="green", alpha=0.5, animated=True)[0]
        self.topax_next_line = self.topax.plot([], [], color="red", alpha=0.5, animated=True)[0]
        self.topax_conv_peaks_line = self.topax.plot([], [], color="gold", ls="--", alpha=0.3, animated=True)[0]

        self._histogram_dirty = True

        self.task = None
        self.conv = SweepConvProxy(self)
        self._templates = {}
        self.step = None
        self.setupFigure()
        self.update_window_title()

    def _on_channel_bar_changed(self):
        if hasattr(self, "channel_bar"):
            self.chk_multichannel.blockSignals(True)
            self.chk_multichannel.setChecked(self.channel_bar.is_multichannel())
            self.chk_multichannel.blockSignals(False)
            self.update_window_title()
            self._histogram_dirty = True
            self.setupFigure()
            self.updateFigure_core()

    def update_window_title(self):
        if hasattr(self, "channel_bar"):
            summary = self.channel_bar.get_title_summary()
            self.setWindowTitle(f"SingleTrace {summary}")
            p = self.parentWidget()
            if p is not None and hasattr(p, "setWindowTitle"):
                p.setWindowTitle(f"SingleTrace {summary}")

    def get_overlay_channels(self) -> List[str]:
        if not self.chk_multichannel.isChecked() or self.state is None:
            return []

        if hasattr(self, "channel_bar"):
            all_chans = self.channel_bar.get_selected_channels()
            primary_name = self.channel_bar.get_primary_channel()
            # If explicit group or unit channels selected in bar, return them directly
            mode, _ = self.channel_bar.get_channel_mode()
            if mode in (self.channel_bar.MODE_FOLLOW_UNIT, self.channel_bar.MODE_GROUP) or (len(all_chans) > 1 and mode != self.channel_bar.MODE_FOLLOW_GLOBAL):
                return [c for c in all_chans if c != primary_name]

        if hasattr(self.state, "get_available_channels"):
            chans = list(self.state.get_available_channels())
        elif getattr(self.state, "segment", None) is not None:
            chans = [s.name for s in self.state.segment.analogsignals]
        else:
            chans = []

        sg = (
            self.channel_bar.get_target_unit()
            if hasattr(self, "channel_bar")
            else (self.state.getUnitGroup() if hasattr(self.state, "getUnitGroup") else None)
        )
        unit_chans = []
        if sg is not None:
            if getattr(sg, "channel", ""):
                unit_chans.append(sg.channel)
            if getattr(sg, "neighbor_channels", None):
                unit_chans.extend(sg.neighbor_channels)

        # Filter out non-neural auxiliary channels if they start with "env." or "log." or "rec." unless explicitly in unit's channels
        filtered_chans = []
        for c in chans:
            is_aux = any(c.startswith(prefix) for prefix in ("env.", "log.", "rec."))
            if not is_aux or (c in unit_chans):
                filtered_chans.append(c)
        chans = filtered_chans

        if self.chk_unit_neighbors_only.isChecked():
            if unit_chans:
                chans = [c for c in chans if c in unit_chans]

        # Exclude the primary active channel
        primary_name = (
            self.channel_bar.get_primary_channel()
            if hasattr(self, "channel_bar")
            else getattr(getattr(self.state, "analog_signal", None), "name", "")
        )
        chans = [c for c in chans if c != primary_name]
        return chans

    def on_new_sweep(self, stimno: int):
        if self.state is None:
            return
        erp = self.state.get_erp()
        if erp is None or erp.shape[0] == 0:
            return
        if not getattr(self, "_setup_done", False) or self.trace_line_cache is None:
            self.setupFigure()
            self._setup_done = True
        self.updateFigure()


    @Slot()
    def view_clicked(self, e: MouseEvent):
        if self.toolbar.mode != "" or e.button != 1 or e.inaxes != self.ax or e.xdata is None:
            return

        try:
            if np.isnan(e.xdata):
                return
            sr = float(self.state.sampling_rate) if self.state and self.state.sampling_rate is not None else 10000.0
            self.set_cur_pos(e.xdata * sr)
        except Exception:
            pass

    @Slot()
    def setupFigure(self):
        if not hasattr(self, "stimno_label") or self.stimno_label is None:
            self.stimno_label = self.ax.text(
                0, 1.01, "{stimno}", transform=self.ax.transAxes, animated=True
            )
        if self.state is None:
            return
        if self.state.analog_signal is None or self.state.event_signal is None or len(self.state.event_signal) == 0:
            return

        erp = self.state.get_erp()
        if erp is not None and hasattr(erp, "magnitude"):
            erp = erp.magnitude
        if erp is None or erp.shape[0] == 0:
            return

        self._setup_done = True
        self.ax.grid(True, which="both")
        if self.trace_line_cache is not None:
            self.trace_line_cache.remove()
            self.trace_line_cache = None

        for line in getattr(self, "aux_trace_lines", []):
            try:
                line.remove()
            except Exception:
                pass
        for lbl in getattr(self, "aux_trace_labels", []):
            try:
                lbl.remove()
            except Exception:
                pass
        self.aux_trace_lines = []
        self.aux_trace_labels = []

        self.ax.set_xlim(0, erp.shape[-1] / self.state.sampling_rate)
        ylim = (float(np.std(np.abs(erp))) or 1.0) * 8
        if getattr(self, "chk_multichannel", None) and self.chk_multichannel.isChecked():
            overlay_chans = self.get_overlay_channels()
            n_extra = len(overlay_chans)
            spacing = self.spin_chan_spacing.value()
            ylim_top = ylim + n_extra * spacing
            self.ax.set_ylim(-ylim, ylim_top)
        else:
            self.ax.set_ylim(-ylim, ylim)

        class CustomFormatter(Formatter):
            def __init__(self, ax: Any):
                super().__init__()
                self.set_axis(ax)

            def __call__(self, x, pos=None):
                # Find the axis range
                vmin, vmax = self.axis.get_view_interval()
                major_locs = [
                    x
                    for x in self.axis.get_major_locator().tick_values(vmin, vmax)
                    if vmin <= x and x <= vmax
                ]
                if len(major_locs) >= 2:
                    return ""
                return "{0:g}".format(1000 * x)
                # tl = [
                #     x
                #     for x in self.axis.get_minor_locator().tick_values(vmin, vmax)
                #     if vmin <= x and x <= vmax
                # ]

                # if x == tl[0] or x == tl[-1]:
                #     return "{0:g}".format(1000 * x)

                # return ""

        func_formatter = matplotlib.ticker.FuncFormatter(
            lambda x, pos: "{0:g}".format(1000 * x)
        )
        self.ax.xaxis.set_major_formatter(func_formatter)
        self.ax.xaxis.set_minor_formatter(CustomFormatter(self.ax))
        # loc = matplotlib.ticker.MaxNLocator(
        #     steps=[1, 5, 10],
        # )  # this locator puts ticks at regular intervals
        loc = matplotlib.ticker.MultipleLocator(0.05)
        self.ax.xaxis.set_major_locator(loc)
        loc = matplotlib.ticker.MultipleLocator(0.01)
        self.ax.xaxis.set_minor_locator(loc)

        self.view.draw_idle()
        self.view.update()
        try:
            self.toolbar.set_history_buttons()
        except Exception:
            pass
        self.step = None
        self._histogram_dirty = True
        self.updateHistogram()

    def get_settings(self):
        return {"xlim": self.ax.get_xlim()}

    def set_settings(self, values):
        if "xlim" in values:
            self.ax.set_xlim(values["xlim"])

    def _get_conv_channels(self) -> List[str]:
        if self.state is None:
            return []
        primary_name = getattr(getattr(self.state, "analog_signal", None), "name", "")
        overlay_chans = self.get_overlay_channels()
        sg = self.state.getUnitGroup() if hasattr(self.state, "getUnitGroup") else None

        if self.chk_multichannel_conv.isChecked() and len(overlay_chans) > 0:
            chans = ([primary_name] if primary_name else []) + [
                c for c in overlay_chans if c != primary_name
            ]
            return chans
        elif (
            sg is not None
            and self.chk_multichannel_conv.isChecked()
            and (getattr(sg, "channel", "") or getattr(sg, "neighbor_channels", None))
        ):
            avail = (
                self.state.get_available_channels()
                if hasattr(self.state, "get_available_channels")
                else []
            )
            unit_chans = []
            if getattr(sg, "channel", "") and sg.channel in avail:
                unit_chans.append(sg.channel)
            elif primary_name:
                unit_chans.append(primary_name)
            if getattr(sg, "neighbor_channels", None):
                for nc in sg.neighbor_channels:
                    if nc in avail and nc not in unit_chans:
                        unit_chans.append(nc)
            return unit_chans if unit_chans else ([primary_name] if primary_name else [])
        else:
            return [primary_name] if primary_name else []

    def _update_templates(self):
        self._templates = {}
        if self.conv is not None and hasattr(self.conv, "clear_cache"):
            self.conv.clear_cache()

        if (
            self.state is None
            or self.state.event_signal is None
            or len(self.state.event_signal) == 0
        ):
            return

        sg = self.state.getUnitGroup() if hasattr(self.state, "getUnitGroup") else None
        if sg is None or sg.event is None or len(sg.event) == 0:
            return

        sr = 10000.0
        if hasattr(self.state, "sampling_rate") and self.state.sampling_rate is not None:
            sr = (
                float(self.state.sampling_rate.rescale(pq.Hz).magnitude)
                if hasattr(self.state.sampling_rate, "magnitude")
                else float(self.state.sampling_rate)
            )
        elif self.state.analog_signal is not None and hasattr(
            self.state.analog_signal, "sampling_rate"
        ):
            sr = (
                float(self.state.analog_signal.sampling_rate.rescale(pq.Hz).magnitude)
                if hasattr(self.state.analog_signal.sampling_rate, "magnitude")
                else float(self.state.analog_signal.sampling_rate)
            )

        lats = sg.get_latencies(self.state.event_signal)
        valid_sweeps = np.where(~np.isnan(lats))[0]
        if len(valid_sweeps) == 0:
            return

        valid_sweeps = valid_sweeps[-50:]
        pre_samples = int(0.010 * sr)
        post_samples = int(0.020 * sr)

        chans = self._get_conv_channels()
        for ch in chans:
            erp_ch = (
                self.state.get_channel_erp(ch)
                if hasattr(self.state, "get_channel_erp")
                else self.state.get_erp()
            )
            if erp_ch is None or len(erp_ch) == 0:
                continue
            if hasattr(erp_ch, "magnitude"):
                erp_ch = erp_ch.magnitude
            erp_ch = erp_ch.astype(np.float32, copy=False)

            snippets = []
            for s_idx in valid_sweeps:
                if s_idx >= len(erp_ch):
                    continue
                lat = lats[s_idx]
                c_idx = int(round(float((lat.rescale(pq.s) * sr).magnitude)))
                start = c_idx - pre_samples
                end = c_idx + post_samples
                sweep_row = erp_ch[s_idx]
                if start >= 0 and end <= len(sweep_row):
                    snippets.append(sweep_row[start:end])
                else:
                    pad_left = max(0, -start)
                    pad_right = max(0, end - len(sweep_row))
                    v_start = max(0, start)
                    v_end = min(len(sweep_row), end)
                    if v_end > v_start:
                        sub = sweep_row[v_start:v_end]
                        snippets.append(np.pad(sub, (pad_left, pad_right), mode="edge"))

            if snippets:
                mean_tpl = np.mean(snippets, axis=0).astype(np.float32)
                std_tpl = float(np.std(mean_tpl))
                norm_tpl = (mean_tpl / (std_tpl if std_tpl > 1e-8 else 1.0)).astype(
                    np.float32
                )
                self._templates[ch] = norm_tpl

    def _compute_sweep_conv(self, stimno: int) -> np.ndarray:
        if not getattr(self, "_templates", None) or self.state is None:
            erp = self.state.get_erp() if self.state else None
            w_len = erp.shape[1] if (erp is not None and erp.ndim == 2) else 1000
            return np.zeros(w_len, dtype=np.float32)

        composite_conv = None
        for ch, tpl in self._templates.items():
            erp_ch = (
                self.state.get_channel_erp(ch)
                if hasattr(self.state, "get_channel_erp")
                else self.state.get_erp()
            )
            if erp_ch is None or stimno >= len(erp_ch):
                continue
            sweep_data = erp_ch[stimno]
            if hasattr(sweep_data, "magnitude"):
                sweep_data = sweep_data.magnitude
            sweep_data = sweep_data.astype(np.float32, copy=False)

            std_sweep = float(np.std(sweep_data))
            norm_sweep = (
                sweep_data / (std_sweep if std_sweep > 1e-8 else 1.0)
            ).astype(np.float32)
            c = fftconvolve(norm_sweep, tpl, mode="same").astype(np.float32)
            c[0 : len(tpl)] = 0
            c[-len(tpl) :] = 0
            if composite_conv is None:
                composite_conv = c
            else:
                composite_conv += c

        if composite_conv is None:
            erp = self.state.get_erp() if self.state else None
            w_len = erp.shape[1] if (erp is not None and erp.ndim == 2) else 1000
            return np.zeros(w_len, dtype=np.float32)

        return composite_conv

    @Slot(np.ndarray)
    def updateThreadDone(self, x):
        self.updateFigure()

    def updateConv(self):
        self._update_templates()
        self.updateFigure()

    def _decimate_trace_for_display(
        self, x_arr: np.ndarray, y_arr: np.ndarray, max_points: int = 2500
    ):
        if len(x_arr) == 0:
            return x_arr, y_arr
        xlim = self.ax.get_xlim()
        x_min, x_max = min(xlim), max(xlim)
        padding = (x_max - x_min) * 0.05
        x_min_pad = x_min - padding
        x_max_pad = x_max + padding

        i_start = int(np.searchsorted(x_arr, x_min_pad, side="left"))
        i_end = int(np.searchsorted(x_arr, x_max_pad, side="right"))
        i_start = max(0, min(len(x_arr) - 1, i_start))
        i_end = max(i_start + 1, min(len(x_arr), i_end))

        x_vis = x_arr[i_start:i_end]
        y_vis = y_arr[i_start:i_end]

        n_samples = len(x_vis)
        if n_samples <= max_points:
            return x_vis, y_vis

        n_bins = max_points // 2
        pts_per_bin = n_samples // n_bins
        total_pts = pts_per_bin * n_bins
        x_trim = x_vis[:total_pts].reshape(n_bins, pts_per_bin)
        y_trim = y_vis[:total_pts].reshape(n_bins, pts_per_bin)

        min_idx = np.argmin(y_trim, axis=1)
        max_idx = np.argmax(y_trim, axis=1)

        row_indices = np.arange(n_bins)
        first_is_min = min_idx <= max_idx
        first_idx = np.where(first_is_min, min_idx, max_idx)
        second_idx = np.where(first_is_min, max_idx, min_idx)

        x1 = x_trim[row_indices, first_idx]
        y1 = y_trim[row_indices, first_idx]
        x2 = x_trim[row_indices, second_idx]
        y2 = y_trim[row_indices, second_idx]

        x_out = np.column_stack([x1, x2]).ravel()
        y_out = np.column_stack([y1, y2]).ravel()
        return x_out, y_out

    def updateHistogram_core(self):
        self._histogram_dirty = True
        self._update_templates()
        self.updateFigure()

    def draw_histogram(self):
        if not getattr(self, "_histogram_dirty", True) and self.step is not None:
            return

        sg = (
            self.channel_bar.get_target_unit()
            if hasattr(self, "channel_bar")
            else (self.state.getUnitGroup() if hasattr(self.state, "getUnitGroup") else None)
        )
        if self.step is not None:
            try:
                self.step[0].remove()
                self.step = None
            except:
                pass
        if sg is None or len(sg.event) == 0:
            self._histogram_dirty = False
            return

        values = sg.get_latencies(self.state.event_signal)
        values[values > self.state.window_size] = np.nan * pq.ms
        values = values[~np.isnan(values)]
        step = 2 * pq.ms  # self.state.sampling_rate * 0.0005 # 0.5ms bins
        if len(values) == 0:
            self._histogram_dirty = False
            return
        else:
            bins = np.arange(
                np.floor(((np.min(values) // step)) - 1) * step,
                np.ceil(np.max(values) + (step * 2)),
                step,
            )
        values_binned = np.histogram(values, bins=bins)
        self.step = self.topax.step(
            values_binned[1][1:] / 1000,
            values_binned[0] / max(values_binned[0]),
            color="gray",
            animated=True,
        )
        self._histogram_dirty = False

    def _mark_dirty_and_schedule(self, *args, **kwargs):
        self._dirty = True

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
            if self.is_view_culled():
                return
            if not getattr(self, "_dirty", False):
                return
            self._dirty = False
            self.updateFigure_core()
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
            self.updateFigure_core()

    def hideEvent(self, event):
        super().hideEvent(event)
        self._user_hidden = True
        if hasattr(self, "view") and self.view is not None:
            try:
                self.view._draw_pending = False
                from PySide6.QtCore import QCoreApplication
                QCoreApplication.removePostedEvents(self.view)
            except Exception:
                pass

    def showEvent(self, event):
        super().showEvent(event)
        self._was_shown = True
        self._user_hidden = False
        if getattr(self, "_dirty", False):
            self._dirty = False
            self.updateFigure_core()

    def updateFigure_core(self):
        if self.is_view_culled():
            self._dirty = True
            return
        self.render_count = getattr(self, "render_count", 0) + 1
        if self.state is None:
            return
        if self.state.analog_signal is None or self.state.event_signal is None or len(self.state.event_signal) == 0:
            return

        if not getattr(self, "_setup_done", False) or self.trace_line_cache is None:
            self.setupFigure()
            self._setup_done = True

        if hasattr(self, "channel_bar"):
            primary_ch = self.channel_bar.get_primary_channel()
            erp = self.state.get_channel_erp(primary_ch)
            sg = self.channel_bar.get_target_unit()
        else:
            erp = self.state.get_erp()
            sg = self.state.getUnitGroup()

        if erp is None or erp.shape[0] == 0:
            return
        if self.state.stimno >= erp.shape[0] or self.state.stimno >= len(self.state.event_signal):
            return
        dpts = erp[self.state.stimno]  # this should only happen once.
        if hasattr(dpts, "magnitude"):
            dpts = dpts.magnitude
        cur_event_time = self.state.event_signal.times[self.state.stimno]
        hours, remainder = divmod(cur_event_time.magnitude, 3600)
        minutes, seconds = divmod(remainder, 60)
        if not hasattr(self, "stimno_label") or self.stimno_label is None:
            self.stimno_label = self.ax.text(
                0, 1.01, "{stimno}", transform=self.ax.transAxes, animated=True
            )
        self.stimno_label.set_text(
            f"{self.state.stimno:04} [{hours:02.0f}:{minutes:02.0f}:{seconds:02.0f}]"
        )

        pts, _ = find_peaks(dpts)
        pts_down, _ = find_peaks(-1 * dpts)
        pts = np.sort(np.hstack([pts, pts_down]).flatten())

        raw_x = np.arange(len(dpts)) / self.state.sampling_rate
        dec_x, dec_y = self._decimate_trace_for_display(raw_x, dpts)

        if self.trace_line_cache is None:
            self.trace_line_cache = self.ax.plot(
                dec_x, dec_y, color="purple"
            )[0]
            self.trace_line_cache.set_animated(True)
        else:
            self.trace_line_cache.set_data(dec_x, dec_y)

        overlay_chans = self.get_overlay_channels()
        # Ensure aux_trace_lines and aux_trace_labels match len(overlay_chans)
        while len(self.aux_trace_lines) < len(overlay_chans):
            line = self.ax.plot([], [], alpha=0.75, animated=True)[0]
            lbl = self.ax.text(0, 0, "", fontsize="x-small", alpha=0.8, animated=True)
            self.aux_trace_lines.append(line)
            self.aux_trace_labels.append(lbl)

        for idx_c, ch_name in enumerate(overlay_chans):
            if hasattr(self.state, "get_channel_erp"):
                ch_erp = self.state.get_channel_erp(ch_name)
            else:
                ch_erp = None

            if ch_erp is not None and self.state.stimno < ch_erp.shape[0]:
                ch_dpts = ch_erp[self.state.stimno]
                if hasattr(ch_dpts, "magnitude"):
                    ch_dpts = ch_dpts.magnitude
                offset = (idx_c + 1) * self.spin_chan_spacing.value()
                ch_raw_x = np.arange(len(ch_dpts)) / self.state.sampling_rate
                ch_dec_x, ch_dec_y = self._decimate_trace_for_display(ch_raw_x, ch_dpts + offset)
                self.aux_trace_lines[idx_c].set_data(ch_dec_x, ch_dec_y)
                self.aux_trace_lines[idx_c].set_visible(True)
                self.aux_trace_labels[idx_c].set_position((0, offset))
                self.aux_trace_labels[idx_c].set_text(ch_name)
                self.aux_trace_labels[idx_c].set_visible(True)
            else:
                self.aux_trace_lines[idx_c].set_visible(False)
                self.aux_trace_labels[idx_c].set_visible(False)

        # Hide extra lines/labels if any
        for idx_c in range(len(overlay_chans), len(self.aux_trace_lines)):
            self.aux_trace_lines[idx_c].set_visible(False)
            self.aux_trace_labels[idx_c].set_visible(False)

        lats_all = sg.get_latencies(self.state.event_signal) if sg is not None else np.array([])
        valid_mask = ~np.isnan(lats_all) if len(lats_all) > 0 else np.array([], dtype=bool)
        idxs = np.where(valid_mask)[0] if len(valid_mask) > 0 else np.array([], dtype=int)
        values = [(float(lats_all[i].rescale(pq.s).magnitude) if hasattr(lats_all[i], "rescale") else float(lats_all[i])) for i in idxs]

        lat = (
            sg.get_latencies(np.array([cur_event_time]) * cur_event_time.units)[0].rescale("s")
            if sg is not None else np.nan * pq.s
        )

        if not np.isnan(lat) and lat < self.state.window_size:
            cur_point = int(lat * self.state.sampling_rate)
            self.identified_spike_line.set_data(
                (
                    [
                        lat.magnitude,
                        lat.magnitude,
                    ],
                    [0, 1],
                )
            )
            self.identified_spike_line.set_visible(True)
            i = pts.searchsorted(cur_point)
            if i > 0 and i < len(pts):
                i2 = pts[i - 1 : i + 1]
                self.closest_pos = i2[np.argmin(np.abs(cur_point - i2))]

            self.topax_cur_line.set_data([lat.magnitude, lat.magnitude], [0, 1])
            self.topax_cur_line.set_visible(True)
        else:
            self.identified_spike_line.set_visible(False)
            self.topax_cur_line.set_visible(False)

        cur_idx = np.searchsorted(
            idxs, self.state.stimno
        ) if len(idxs) > 0 else 0  # plot the previous and next identified spike in this group
        if cur_idx > 0 and (self.state.stimno - idxs[cur_idx - 1]) < 10:
            self.topax_prev_line.set_data(
                [values[cur_idx - 1], values[cur_idx - 1]], [0, 1]
            )
            self.topax_prev_line.set_visible(True)
        else:
            self.topax_prev_line.set_visible(False)

        if np.isnan(lat):
            cur_idx -= 1  # edge case where there is no unit
        if (
            (cur_idx + 1) < len(idxs)
            and (idxs[cur_idx + 1] - self.state.stimno) < 10
            and cur_idx >= 0
        ):
            self.topax_next_line.set_data(
                [values[cur_idx + 1], values[cur_idx + 1]], [0, 1]
            )
            self.topax_next_line.set_visible(True)
        else:
            self.topax_next_line.set_visible(False)

        if self.conv is not None:
            conv = self.conv[self.state.stimno]
            margin = min(1000, max(1, len(conv) // 10))
            inner = conv[margin:-margin] if len(conv) > 2 * margin else conv
            conv_high = np.percentile(inner, 99)
            highlight, _ = find_peaks(conv, conv_high)
            highlight = highlight - 1
            hl = highlight[np.argsort(conv[highlight])[::-1]]
            hl_top = hl[0:100]
            if len(hl_top) > 0:
                x_pos = hl_top / self.state.sampling_rate
                x_segs = np.column_stack([x_pos, x_pos, np.full_like(x_pos, np.nan, dtype=np.float64)]).ravel()
                y_segs = np.tile([0.0, 1.0, np.nan], len(x_pos))
                self.topax_conv_peaks_line.set_data(x_segs, y_segs)
                self.topax_conv_peaks_line.set_visible(True)
            else:
                self.topax_conv_peaks_line.set_visible(False)
        else:
            self.topax_conv_peaks_line.set_visible(False)

        self.topax_lines = [
            a for a in (self.topax_cur_line, self.topax_prev_line, self.topax_next_line, self.topax_conv_peaks_line)
            if a.get_visible()
        ]

        self.draw_histogram()

        if getattr(self, "blit_data", None) is not None:
            self.fig.canvas.restore_region(self.blit_data)
        if getattr(self, "blit_data_topax", None) is not None:
            self.fig.canvas.restore_region(self.blit_data_topax)

        self.ax.draw_artist(self.identified_spike_line)
        self.ax.draw_artist(self.trace_line_cache)
        for line in getattr(self, "aux_trace_lines", []):
            if line.get_visible():
                self.ax.draw_artist(line)
        for lbl in getattr(self, "aux_trace_labels", []):
            if lbl.get_visible():
                self.ax.draw_artist(lbl)
        self.ax.draw_artist(self.stimno_label)
        if self.step is not None:
            for x in self.step:
                self.topax.draw_artist(x)
        for x in self.topax_lines:
            self.topax.draw_artist(x)
        self.view.update()
        if getattr(self, "blit_data", None) is None:
            try:
                self.blit()
            except Exception:
                pass

    def blit(self):
        self.blit_data = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        self.blit_data_topax = self.fig.canvas.copy_from_bbox(self.topax.bbox)

    def set_cur_pos(self, x):
        if x is None or np.isnan(x) or self.state is None:
            return
        erp = self.state.get_erp()
        if erp is None or self.state.stimno < 0 or self.state.stimno >= len(erp):
            return
        dpts = erp[self.state.stimno]
        if hasattr(dpts, "magnitude"):
            dpts = dpts.magnitude
        n_pts = len(dpts)
        if n_pts == 0:
            return

        x_idx = max(0, min(n_pts - 1, int(round(x))))
        w = int(self.select_local_maxima_width)
        if w > 1:
            start_w = max(0, x_idx - w)
            end_w = min(n_pts, x_idx + w + 1)
            if end_w > start_w:
                sub_slice = np.abs(dpts[start_w:end_w])
                if len(sub_slice) > 0:
                    x_idx = start_w + int(np.argmax(sub_slice))

        target_unit_idx = self.channel_bar.get_target_unit_idx() if hasattr(self, "channel_bar") else self.state.cur_spike_group
        self.state.setUnit(x_idx, unitid=target_unit_idx)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Z:  # Although the zoom works, it can't be undone...
            # get current spike location
            spike_ts = (
                self.state.getUnitGroup()
                .get_latencies(
                    [self.state.event_signal[self.state.stimno].magnitude]
                    * self.state.event_signal.units
                )[0]
                .rescale("s")
            )
            if spike_ts == np.nan or spike_ts > self.state.window_size:
                return
            # set the xlim to 5 ms before and after spike
            self.ax.set_xlim(spike_ts.magnitude - 0.005, spike_ts.magnitude + 0.005)

            e = self.state.event_signal[self.state.stimno]
            # set the ylim to be the max of the spike + 5%
            vals = (
                self.state.analog_signal.time_slice(
                    e + spike_ts - (2 * pq.ms), e + spike_ts + (2 * pq.ms)
                )
                # .rescale(pq.mV)
                .magnitude
            )
            min_val = np.min(vals)
            max_val = np.max(vals)
            self.ax.set_ylim(
                min_val - (0.05 * np.abs(min_val)), max_val + (0.05 * np.abs(max_val))
            )
            self.fig.canvas.draw_idle()
        elif (
            (e.key() == Qt.Key_Left or e.key() == Qt.Key_Right)
            and (e.modifiers() & Qt.ShiftModifier)
            and self.conv is not None
        ):
            sg = self.state.getUnitGroup()
            cur_lat = sg.get_latencies(
                [self.state.event_signal[self.state.stimno].magnitude]
                * self.state.event_signal.units
            )[0]
            if np.isnan(cur_lat):
                cur_stimpos = 0
            else:
                cur_stimpos = int(
                    round(float((cur_lat.rescale(pq.s) * self.state.sampling_rate).magnitude))
                )
            conv_sweep = self.conv[self.state.stimno]
            margin = min(1000, max(1, len(conv_sweep) // 10))
            inner = conv_sweep[margin:-margin] if len(conv_sweep) > 2 * margin else conv_sweep
            conv_high = np.percentile(inner, 99)
            idx, _ = find_peaks(conv_sweep, conv_high)
            idx = idx - 1  # offset by one
            # idx = np.where(highlight)[0]
            closest_idx = np.searchsorted(idx, cur_stimpos)
            if (
                closest_idx < len(idx)
                and idx[closest_idx] == cur_stimpos
                and e.key() == Qt.Key_Right
                and closest_idx < len(idx)
            ):
                closest_idx += 1
            if e.key() == Qt.Key_Left:
                closest_idx -= 1
            target_unit_idx = self.channel_bar.get_target_unit_idx() if hasattr(self, "channel_bar") else self.state.cur_spike_group
            self.state.setUnit(idx[closest_idx], unitid=target_unit_idx)
        elif e.key() == Qt.Key_BracketLeft:  # Zoom out
            # get current xlim & ylim
            x_limits = [self.ax.get_xlim()[0], self.ax.get_xlim()[1]]
            y_limits = [self.ax.get_ylim()[0], self.ax.get_ylim()[1]]

            # calculate new limits
            midline = np.mean(x_limits)
            new_x_limits = [
                midline - (midline - x_limits[0]) * 1.2,
                midline + (x_limits[1] - midline) * 1.2,
            ]

            midline = np.mean(y_limits)
            new_y_limits = [
                midline - (midline - y_limits[0]) * 1.2,
                midline + (y_limits[1] - midline) * 1.2,
            ]

            # increase the lims to 120%
            self.ax.set_xlim(new_x_limits[0], new_x_limits[1])
            self.ax.set_ylim(new_y_limits[0], new_y_limits[1])

            self.fig.canvas.draw_idle()
        elif e.key() == Qt.Key_BracketRight:  # Zoom in
            # get current xlim & ylim
            x_limits = [self.ax.get_xlim()[0], self.ax.get_xlim()[1]]
            y_limits = [self.ax.get_ylim()[0], self.ax.get_ylim()[1]]

            # calculate new limits
            midline = np.mean(x_limits)
            new_x_limits = [
                midline - (midline - x_limits[0]) * 0.8,
                midline + (x_limits[1] - midline) * 0.8,
            ]

            midline = np.mean(y_limits)
            new_y_limits = [
                midline - (midline - y_limits[0]) * 0.8,
                midline + (y_limits[1] - midline) * 0.8,
            ]

            # deccrease the lims to 80%
            self.ax.set_xlim(new_x_limits[0], new_x_limits[1])
            self.ax.set_ylim(new_y_limits[0], new_y_limits[1])

            self.fig.canvas.draw_idle()


if __name__ == "__main__":

    app = QApplication([])
    state = ViewerState()
    state.loadFile(r"data/test2.h5")

    view = SingleTraceView(state=state)
    view.show()
    app.exec()
    sys.exit()
