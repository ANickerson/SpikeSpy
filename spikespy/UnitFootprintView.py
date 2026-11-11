import sys
from typing import Any, Dict, List, Optional, Union

import matplotlib.style as mplstyle
import numpy as np
import PySide6
import quantities as pq
from matplotlib.backends.backend_qtagg import FigureCanvas, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtWidgets import (
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

from .ViewerState import ViewerState, tracked_neuron_unit

mplstyle.use("fast")


class UnitFootprintView(QMainWindow):
    """
    Dockable view visualizing the spatiotemporal footprint of detected units across
    recording channels, displaying inter-channel conduction/propagation lag (Δt)
    and channel amplitudes.
    """

    def __init__(
        self,
        parent: Optional[PySide6.QtWidgets.QWidget] = None,
        state: Optional[ViewerState] = None,
    ):
        super().__init__(parent)
        self.state: Optional[ViewerState] = state
        self.current_channels: List[str] = []
        self.ax_heat = None
        self.ax_bar = None
        self.grid_axes = []

        # Central widget and figure
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        central_widget = QWidget(self)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        self.setCentralWidget(central_widget)

        # Controls
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

        # Matplotlib navigation toolbar
        self.matplotlib_toolbar = NavigationToolbar2QT(self.canvas, self)
        self.addToolBar(self.matplotlib_toolbar)

        # Throttling timer for smooth streaming updates
        self.update_timer = QTimer(self)
        self.update_timer.setSingleShot(True)
        self.update_timer.setInterval(200)
        self.update_timer.timeout.connect(self.update_plot)

        # Signal connections for UI controls
        self.combo_mode.currentIndexChanged.connect(self.update_plot)
        self.spin_last_n.valueChanged.connect(self.update_plot)
        self.spin_window_ms.valueChanged.connect(self.update_plot)
        self.chk_align_peaks.toggled.connect(self.update_plot)
        self.chk_overlay_traces.toggled.connect(self.update_plot)
        self.btn_auto_assign_channels.clicked.connect(lambda: self.auto_detect_unit_channels())

        # Canvas click handler
        self.canvas.mpl_connect("button_press_event", self.on_click)

        if state is not None:
            self.setState(state)

    def setState(self, state: ViewerState):
        self.state = state
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
        self.update_plot()

    def schedule_update(self):
        if not self.update_timer.isActive():
            self.update_timer.start()

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

        avail_chans = (
            self.state.get_available_channels()
            if hasattr(self.state, "get_available_channels")
            else []
        )
        if not avail_chans and getattr(self.state, "analog_signal", None) is not None:
            if getattr(self.state.analog_signal, "name", None):
                avail_chans = [self.state.analog_signal.name]

        sg_channels = set()
        if getattr(sg, "channel", None):
            sg_channels.add(sg.channel)
        if getattr(sg, "neighbor_channels", None):
            sg_channels.update(sg.neighbor_channels)

        channels = []
        for ch in avail_chans:
            if not ch:
                continue
            if ch in sg_channels:
                channels.append(ch)
            elif not any(ch.startswith(prefix) for prefix in ("env.", "log.", "rec.")):
                channels.append(ch)

        if not channels:
            return None

        # Reference channel for peak alignment
        ref_chan = (
            sg.channel
            if (getattr(sg, "channel", "") in channels)
            else (
                self.state.analog_signal.name
                if (
                    self.state.analog_signal
                    and getattr(self.state.analog_signal, "name", "") in channels
                )
                else channels[0]
            )
        )

        # Sampling rate in Hz
        sr = 10000.0
        if hasattr(self.state, "sampling_rate") and self.state.sampling_rate is not None:
            sr = (
                float(self.state.sampling_rate.rescale(pq.Hz).magnitude)
                if hasattr(self.state.sampling_rate, "magnitude")
                else float(self.state.sampling_rate)
            )
        elif (
            hasattr(self.state, "analog_signal")
            and self.state.analog_signal is not None
            and hasattr(self.state.analog_signal, "sampling_rate")
        ):
            asig_sr = self.state.analog_signal.sampling_rate
            sr = (
                float(asig_sr.rescale(pq.Hz).magnitude)
                if hasattr(asig_sr, "magnitude")
                else float(asig_sr)
            )

        sweep_sample_map = {}
        if self.state.event_signal is not None and hasattr(sg, "get_latencies"):
            lats = sg.get_latencies(self.state.event_signal)
            valid = ~np.isnan(lats)
            valid_sweeps = np.where(valid)[0].tolist()
            for s_i in valid_sweeps:
                sec = float(lats[s_i].rescale(pq.s).magnitude) if hasattr(lats[s_i], "rescale") else float(lats[s_i])
                sweep_sample_map[s_i] = int(round(sec * sr))
        elif getattr(sg, "idx_arr", None) is not None:
            valid_sweeps = []
            for s_i, item in enumerate(sg.idx_arr):
                if item is not None:
                    c_s = int(item[0]) if isinstance(item, (tuple, list, np.ndarray)) else int(item)
                    valid_sweeps.append(s_i)
                    sweep_sample_map[s_i] = c_s
        else:
            valid_sweeps = []

        valid_sweeps = valid_sweeps[-last_n:]
        if len(valid_sweeps) == 0:
            return None

        half_w = max(int((window_ms / 1000.0) * sr / 2.0), 5)
        t_ms = (np.arange(-half_w, half_w) / sr) * 1000.0

        channel_erps: Dict[str, Optional[np.ndarray]] = {}
        for ch in channels:
            erp = (
                self.state.get_channel_erp(ch)
                if hasattr(self.state, "get_channel_erp")
                else None
            )
            if erp is not None:
                if hasattr(erp, "magnitude"):
                    erp = erp.magnitude
                channel_erps[ch] = erp
            else:
                channel_erps[ch] = None

        ref_erp = channel_erps.get(ref_chan)
        if ref_erp is None:
            return None

        search_r = max(int(0.00075 * sr), 3)
        n_ch = len(channels)
        n_ev = len(valid_sweeps)
        snippets = np.zeros((n_ch, n_ev, 2 * half_w), dtype=float)

        for ev_i, s_i in enumerate(valid_sweeps):
            c_idx = sweep_sample_map.get(s_i, 0)

            if align_peaks:
                if s_i < len(ref_erp):
                    ref_sweep = ref_erp[s_i]
                    n_samples = len(ref_sweep)
                    search_start = max(0, c_idx - search_r)
                    search_end = min(n_samples, c_idx + search_r + 1)
                    if search_end > search_start:
                        local_peak = search_start + int(
                            np.argmax(np.abs(ref_sweep[search_start:search_end]))
                        )
                        c_aligned = local_peak
                    else:
                        c_aligned = c_idx
                else:
                    c_aligned = c_idx
            else:
                c_aligned = c_idx

            for ch_i, ch in enumerate(channels):
                ch_erp = channel_erps.get(ch)
                if ch_erp is not None and s_i < len(ch_erp):
                    sweep_ch = ch_erp[s_i]
                    n_ch_samples = len(sweep_ch)
                    start = c_aligned - half_w
                    end = c_aligned + half_w
                    pad_left = max(0, -start)
                    pad_right = max(0, end - n_ch_samples)
                    v_start = max(0, start)
                    v_end = min(n_ch_samples, end)
                    if v_end > v_start:
                        sub = sweep_ch[v_start:v_end]
                        if pad_left > 0 or pad_right > 0:
                            sub = np.pad(
                                sub,
                                (pad_left, pad_right),
                                mode="constant",
                                constant_values=0.0,
                            )
                        if len(sub) == 2 * half_w:
                            snippets[ch_i, ev_i, :] = sub
                        elif len(sub) > 2 * half_w:
                            snippets[ch_i, ev_i, :] = sub[: 2 * half_w]
                        else:
                            snippets[ch_i, ev_i, : len(sub)] = sub

        mean_matrix = np.mean(snippets, axis=1)
        ptp_amp = np.zeros(n_ch, dtype=float)
        lag_ms = np.zeros(n_ch, dtype=float)
        snr = np.zeros(n_ch, dtype=float)

        for ch_i in range(n_ch):
            ch_mean = mean_matrix[ch_i]
            ptp_amp[ch_i] = float(np.ptp(ch_mean))
            peak_sample_offset = int(np.argmax(np.abs(ch_mean))) - half_w
            lag_ms[ch_i] = float((peak_sample_offset / sr) * 1000.0)

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
            if self.state is None:
                return
            sg = self.state.getUnitGroup()
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
        """Refreshes the canvas plot according to current mode and settings."""
        if hasattr(self, "update_timer") and self.update_timer.isActive():
            self.update_timer.stop()

        self.figure.clear()
        self.ax_heat = None
        self.ax_bar = None
        self.current_channels = []
        self.grid_axes = []

        if self.state is None:
            self.canvas.draw_idle()
            return

        mode_idx = self.combo_mode.currentIndex()
        if mode_idx == 0:
            self._plot_single_unit_composite()
        else:
            self._plot_multi_unit_grid()

        self.canvas.draw_idle()

    def _plot_single_unit_composite(self):
        sg = self.state.getUnitGroup() if self.state is not None else None
        if sg is None:
            ax = self.figure.add_subplot(111)
            ax.text(
                0.5,
                0.5,
                "No unit selected",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return

        fp = self.compute_unit_footprint(sg)
        if fp is None:
            ax = self.figure.add_subplot(111)
            ax.text(
                0.5,
                0.5,
                "Insufficient events or channel data for unit footprint",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
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

        gs = GridSpec(1, 2, figure=self.figure, width_ratios=[3.5, 1.2], wspace=0.15)
        ax_heat = self.figure.add_subplot(gs[0, 0])
        ax_bar = self.figure.add_subplot(gs[0, 1], sharey=ax_heat)
        self.ax_heat = ax_heat
        self.ax_bar = ax_bar

        vmax = (
            float(np.max(np.abs(mean_matrix)))
            if np.max(np.abs(mean_matrix)) > 0
            else 1.0
        )
        ax_heat.imshow(
            mean_matrix,
            aspect="auto",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            extent=[t_ms[0], t_ms[-1], n_ch - 0.5, -0.5],
            interpolation="bilinear",
        )

        ax_heat.axvline(0.0, ls="--", color="black", alpha=0.6, zorder=5)

        if self.chk_overlay_traces.isChecked():
            global_max = vmax if vmax > 0 else 1.0
            scale = 0.38 / global_max
            n_events = snippets.shape[1]

            for ch_i in range(n_ch):
                for ev_i in range(n_events):
                    tr = snippets[ch_i, ev_i, :]
                    y_tr = ch_i - (tr * scale)
                    ax_heat.plot(
                        t_ms, y_tr, color="gray", alpha=0.18, lw=0.7, zorder=6
                    )

                m_tr = mean_matrix[ch_i, :]
                y_mean = ch_i - (m_tr * scale)
                ax_heat.plot(t_ms, y_mean, color="black", lw=1.6, zorder=7)

            ax_heat.scatter(
                lag_ms,
                range(n_ch),
                color="#00e5ff",
                edgecolor="black",
                s=45,
                marker="D",
                zorder=12,
                label="Peak Lag",
            )

            det_indices = [i for i, ch in enumerate(channels) if ch in detected_channels]
            if len(det_indices) > 1:
                det_sorted = sorted(det_indices)
                ax_heat.plot(
                    [lag_ms[i] for i in det_sorted],
                    det_sorted,
                    ls="--",
                    color="#ff1744",
                    lw=1.8,
                    alpha=0.9,
                    zorder=11,
                )

            dt_range = t_ms[-1] - t_ms[0]
            for i in det_indices:
                ch = channels[i]
                lag = lag_ms[i]
                txt = f"{lag:+.2f} ms"
                ax_heat.text(
                    lag + (dt_range * 0.02),
                    i,
                    txt,
                    va="center",
                    ha="left",
                    fontsize=8,
                    fontweight="bold",
                    color="white",
                    bbox=dict(
                        boxstyle="round,pad=0.15", fc="black", ec="none", alpha=0.65
                    ),
                    zorder=13,
                )

        ax_heat.set_yticks(range(n_ch))
        ax_heat.set_yticklabels(channels)
        for tick_lbl, ch in zip(ax_heat.get_yticklabels(), channels):
            if ch in detected_channels:
                tick_lbl.set_fontweight("bold")
                tick_lbl.set_color("#0d47a1" if ch == ref_chan else "#1b5e20")

        ax_heat.set_xlim(t_ms[0], t_ms[-1])
        ax_heat.set_ylim(n_ch - 0.5, -0.5)
        ax_heat.set_xlabel("Time (ms)")
        ax_heat.set_ylabel("Channel")

        unit_name = (
            getattr(sg.event, "name", f"Unit {self.state.cur_spike_group}")
            if hasattr(sg, "event") and sg.event is not None
            else f"Unit {self.state.cur_spike_group}"
        )
        ax_heat.set_title(
            f"{unit_name} Footprint ({fp['n_events']} events, Ref: {ref_chan})",
            fontsize=10,
            fontweight="bold",
        )

        bar_colors = [
            (
                "#1976d2"
                if ch == ref_chan
                else ("#388e3c" if ch in detected_channels else "#cfd8dc")
            )
            for ch in channels
        ]
        ax_bar.barh(
            range(n_ch),
            ptp_amp,
            color=bar_colors,
            edgecolor="black",
            height=0.6,
            alpha=0.85,
        )
        max_amp = float(np.max(ptp_amp)) if len(ptp_amp) > 0 else 1.0
        ax_bar.set_xlim(0, max_amp * 1.35 if max_amp > 0 else 1.0)
        ax_bar.set_ylim(n_ch - 0.5, -0.5)
        ax_bar.set_xlabel("PTP Amplitude")
        ax_bar.yaxis.set_visible(False)
        ax_bar.grid(axis="x", linestyle=":", alpha=0.6)

        for i, (amp, lag, ch) in enumerate(zip(ptp_amp, lag_ms, channels)):
            label_text = f"{amp:.1f} (Δt={lag:+.2f}ms)"
            is_det = ch in detected_channels
            ax_bar.text(
                amp + (max_amp * 0.02),
                i,
                label_text,
                va="center",
                ha="left",
                fontsize=7.5,
                fontweight="bold" if is_det else "normal",
                color=(
                    "#0d47a1"
                    if ch == ref_chan
                    else ("#1b5e20" if is_det else "#455a64")
                ),
            )

    def _plot_multi_unit_grid(self):
        units = [
            sg
            for sg in self.state.spike_groups
            if sg is not None and sg.get_number_of_events() > 0
        ]
        if not units:
            units = [sg for sg in self.state.spike_groups if sg is not None]
        if not units:
            ax = self.figure.add_subplot(111)
            ax.text(
                0.5,
                0.5,
                "No tracked units available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return

        n_units = len(units)
        n_cols = min(n_units, 3)
        n_rows = int(np.ceil(n_units / n_cols))

        for u_idx, sg in enumerate(units):
            ax = self.figure.add_subplot(n_rows, n_cols, u_idx + 1)
            fp = self.compute_unit_footprint(sg)
            if fp is None:
                unit_title = (
                    getattr(sg.event, "name", f"Unit {u_idx}")
                    if hasattr(sg, "event") and sg.event is not None
                    else f"Unit {u_idx}"
                )
                ax.set_title(f"{unit_title} (No Data)", fontsize=8)
                ax.axis("off")
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
            self.grid_axes.append((ax, sg_original_idx, best_ch))

            vmax = (
                float(np.max(np.abs(mean_matrix)))
                if np.max(np.abs(mean_matrix)) > 0
                else 1.0
            )
            ax.imshow(
                mean_matrix,
                aspect="auto",
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
                extent=[t_ms[0], t_ms[-1], n_ch - 0.5, -0.5],
                interpolation="bilinear",
            )
            ax.axvline(0.0, ls="--", color="black", alpha=0.5, lw=0.8)

            if self.chk_overlay_traces.isChecked():
                scale = 0.38 / (vmax if vmax > 0 else 1.0)
                for ch_i in range(n_ch):
                    m_tr = mean_matrix[ch_i, :]
                    y_mean = ch_i - (m_tr * scale)
                    ax.plot(t_ms, y_mean, color="black", lw=1.0, zorder=7)

                ax.scatter(
                    lag_ms,
                    range(n_ch),
                    color="#00e5ff",
                    edgecolor="black",
                    s=25,
                    marker="D",
                    zorder=10,
                )
                det_indices = [
                    i for i, ch in enumerate(channels) if ch in detected_channels
                ]
                if len(det_indices) > 1:
                    det_sorted = sorted(det_indices)
                    ax.plot(
                        [lag_ms[i] for i in det_sorted],
                        det_sorted,
                        ls="--",
                        color="#ff1744",
                        lw=1.2,
                        alpha=0.85,
                        zorder=9,
                    )

            ax.set_yticks(range(n_ch))
            if u_idx % n_cols == 0:
                ax.set_yticklabels(channels, fontsize=7)
            else:
                ax.set_yticklabels([])

            ax.set_xlim(t_ms[0], t_ms[-1])
            ax.set_ylim(n_ch - 0.5, -0.5)
            ax.tick_params(labelsize=7)

            unit_name = (
                getattr(sg.event, "name", f"Unit {u_idx}")
                if hasattr(sg, "event") and sg.event is not None
                else f"Unit {u_idx}"
            )
            ax.set_title(
                f"{unit_name} (N={fp['n_events']}, Best: {best_ch})",
                fontsize=8.5,
                fontweight="bold",
            )

        self.figure.tight_layout()

    def on_click(self, event):
        """Interactive click handler for channel selection and unit jumping."""
        if hasattr(self, "matplotlib_toolbar") and self.matplotlib_toolbar is not None:
            if self.matplotlib_toolbar.mode != "":
                return

        if self.combo_mode.currentIndex() == 0:
            if event.inaxes in (self.ax_heat, self.ax_bar):
                if event.ydata is not None and self.current_channels:
                    ch_idx = int(round(event.ydata))
                    if 0 <= ch_idx < len(self.current_channels):
                        clicked_ch = self.current_channels[ch_idx]
                        if self.state is not None and hasattr(
                            self.state, "set_active_channel"
                        ):
                            self.state.set_active_channel(clicked_ch)
        elif self.combo_mode.currentIndex() == 1:
            if hasattr(self, "grid_axes") and self.grid_axes:
                for ax, sg_idx, best_ch in self.grid_axes:
                    if event.inaxes == ax:
                        if self.state is not None:
                            if hasattr(self.state, "setUnitGroup"):
                                self.state.setUnitGroup(sg_idx)
                            if best_ch and hasattr(self.state, "set_active_channel"):
                                self.state.set_active_channel(best_ch)
                        break

    def get_settings(self) -> dict:
        """Returns perspective settings dictionary."""
        return {
            "mode": self.combo_mode.currentIndex(),
            "last_n": self.spin_last_n.value(),
            "window_ms": self.spin_window_ms.value(),
            "align_peaks": self.chk_align_peaks.isChecked(),
            "overlay_traces": self.chk_overlay_traces.isChecked(),
        }

    def set_settings(self, values: dict):
        """Restores perspective settings."""
        if not isinstance(values, dict):
            return
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
