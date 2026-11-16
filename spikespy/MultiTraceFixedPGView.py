import sys
from typing import Dict, List, Optional, Tuple
import numpy as np
import quantities as pq

import PySide6
from PySide6 import QtCore
from PySide6.QtCore import Qt, Signal, Slot, QTimer, QSignalBlocker
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QToolBar,
    QLabel,
    QDoubleSpinBox,
    QSpinBox,
    QComboBox,
    QApplication,
)
import pyqtgraph as pg

from .ViewerState import ViewerState, tracked_neuron_unit
from .ChannelGroupBar import ChannelGroupBar
from .helpers import qsignal_throttle, quiet_fail, run_adaptive_render
from .live_orchestrator import LiveFrameConsumerMixin
from .PGNavigation import PGNavigationMixin


class FixedTimeAxisItem(pg.AxisItem):
    """Custom AxisItem converting seconds to milliseconds."""

    def tickStrings(self, values, scale, spacing):
        return [f"{1000 * v:g}" for v in values]


class MultiTraceFixedPGView(LiveFrameConsumerMixin, PGNavigationMixin, QMainWindow):
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
        self.connections: List[Tuple] = []
        self.settings: Dict = {
            "n_lines_pre": 5,
            "n_lines_post": 5,
            "scale": 1.0,
            "scale_mode": "mV",
            "y_axis_mode": "sweep",
            "time_window_s": 10.0,
        }
        self._displayed_stim_indices: List[int] = []
        self._displayed_baselines: List[float] = []

        # Graphics items
        self.lines_curve = pg.PlotDataItem(
            pen=pg.mkPen(color=(150, 150, 150), width=1), connect="finite"
        )
        self.active_curve = pg.PlotDataItem(pen=pg.mkPen(color="purple", width=2))
        self.highlight_segments = pg.PlotDataItem(
            pen=pg.mkPen(color="red", width=2), connect="finite"
        )
        self.points_scatter = pg.ScatterPlotItem(
            size=8, symbol="o", brush=pg.mkBrush("black"), pen=None
        )

        self._has_initialized_xrange: bool = False
        self._dirty: bool = False
        self._was_shown: bool = False
        self._user_hidden: bool = False
        self._last_render_key = None
        self.on_new_sweep_throttled = qsignal_throttle(self._on_new_sweep_core, 40)

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
        self.update_figure(force=True)

    def get_target_unit(self) -> Optional[tracked_neuron_unit]:
        if hasattr(self, "channel_bar") and self.channel_bar is not None:
            return self.channel_bar.get_target_unit()
        if self.state and 0 <= self.state.cur_spike_group < len(
            self.state.spike_groups
        ):
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
        self.main_plot: pg.PlotItem = self.glw.addPlot(
            row=0, col=0, axisItems={"bottom": self.time_axis}
        )
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

        y_axis_label = QLabel("Y-Axis:")
        self.y_axis_combo = QComboBox()
        self.y_axis_combo.addItem("Sweep", "sweep")
        self.y_axis_combo.addItem("Time", "time")
        current_mode = self.settings.get("y_axis_mode", "sweep")
        self.y_axis_combo.setCurrentIndex(1 if current_mode == "time" else 0)
        self.y_axis_combo.currentIndexChanged.connect(self._on_y_axis_mode_changed)

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

        self.pre_label = QLabel("Pre:")
        self.pre_spinner = QSpinBox()
        self.pre_spinner.setRange(0, 100)
        self.pre_spinner.setValue(self.settings.get("n_lines_pre", 5))
        self.pre_spinner.valueChanged.connect(
            lambda value: self.set_settings(
                {"n_lines_pre": value}, update_spinners=False
            )
        )
        self.pre_spinner.setMinimumSize(80, 1)

        self.post_label = QLabel("Post:")
        self.post_spinner = QSpinBox()
        self.post_spinner.setRange(0, 100)
        self.post_spinner.setValue(self.settings.get("n_lines_post", 5))
        self.post_spinner.valueChanged.connect(
            lambda value: self.set_settings(
                {"n_lines_post": value}, update_spinners=False
            )
        )
        self.post_spinner.setMinimumSize(80, 1)

        self.scale_mode_combo = QComboBox()
        self.scale_mode_combo.addItems(["mV", "std"])
        self.scale_mode_combo.setCurrentText(self.settings.get("scale_mode", "mV"))
        self.scale_mode_combo.currentTextChanged.connect(
            lambda mode: self.set_settings({"scale_mode": mode})
        )

        self.time_window_label = QLabel("Window (s):")
        self.time_window_spinner = QDoubleSpinBox()
        self.time_window_spinner.setDecimals(1)
        self.time_window_spinner.setRange(0.1, 3600.0)
        self.time_window_spinner.setSingleStep(1.0)
        self.time_window_spinner.setValue(self.settings.get("time_window_s", 10.0))
        self.time_window_spinner.setSuffix(" s")
        self.time_window_spinner.setMinimumSize(80, 1)
        self.time_window_spinner.valueChanged.connect(
            lambda value: self.set_settings(
                {"time_window_s": value}, update_spinners=False
            )
        )

        scale_layout.addWidget(y_axis_label)
        scale_layout.addWidget(self.y_axis_combo)
        scale_layout.addWidget(scale_label)
        scale_layout.addWidget(self.scale_spinner)
        scale_layout.addWidget(self.scale_mode_combo)
        scale_layout.addWidget(self.pre_label)
        scale_layout.addWidget(self.pre_spinner)
        scale_layout.addWidget(self.post_label)
        scale_layout.addWidget(self.post_spinner)
        scale_layout.addWidget(self.time_window_label)
        scale_layout.addWidget(self.time_window_spinner)

        self._update_toolbar_visibility()

        toolbar.addWidget(scale_widget)
        self.add_navigation_actions(toolbar)
        self.init_navigation()
        self.addToolBar(toolbar)

    def _update_toolbar_visibility(self):
        is_time = self.settings.get("y_axis_mode", "sweep") == "time"
        if hasattr(self, "pre_label"):
            self.pre_label.setVisible(not is_time)
            self.pre_spinner.setVisible(not is_time)
            self.post_label.setVisible(not is_time)
            self.post_spinner.setVisible(not is_time)
            self.time_window_label.setVisible(is_time)
            self.time_window_spinner.setVisible(is_time)

    def _on_y_axis_mode_changed(self, index: int):
        mode = "time" if index == 1 else "sweep"
        self.set_settings({"y_axis_mode": mode}, update_spinners=False)

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

    @Slot()
    @Slot(int)
    def _on_unit_change(self, *args):
        if getattr(self.state, "is_streaming", False):
            self.on_new_sweep_throttled()
        else:
            self.update_figure()

    @Slot()
    def _on_unit_group_change(self, *args):
        self.update_figure()

    @Slot()
    @Slot(str)
    def _on_channel_change(self, *args):
        self.update_figure(force=True)

    @Slot()
    @Slot(int)
    def _on_new_spike_events(self, *args):
        if getattr(self.state, "is_streaming", False):
            self.on_new_sweep_throttled()
        else:
            self.update_figure()

    def set_state(self, state: ViewerState):
        self.unset_connections()
        self.state = state
        if hasattr(self, "channel_bar") and self.channel_bar is not None:
            self.channel_bar.setState(state)
        if self.state is None:
            return

        self._connect_state(self.state.onStimNoChange, self._on_stim_no_change)
        self._connect_state(self.state.onUnitChange, self._on_unit_change)
        self._connect_state(self.state.onUnitGroupChange, self._on_unit_group_change)
        self._register_data_reload(state, self._on_load_new_file)
        if hasattr(self.state, "onChannelChange"):
            self._connect_state(self.state.onChannelChange, self._on_channel_change)
        if hasattr(self.state, "onNewSweep"):
            self._register_live_frames(
                self.state,
                lambda stimno: self.on_new_sweep(stimno),
            )
        if hasattr(self.state, "onNewSpikeEvents"):
            self._connect_state(self.state.onNewSpikeEvents, self._on_new_spike_events)

        self._has_initialized_xrange = False
        self._auto_scale_needed = True
        self._auto_scale()
        erp = self.get_erp_signal()
        if erp is not None and len(erp) > 0:
            self._auto_scale_needed = False
        self.update_figure()

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

    def _on_load_new_file(self):
        self._has_initialized_xrange = False
        self._auto_scale_needed = True
        self.update_figure()

    def _on_stim_no_change(self, *args):
        """Route sweep-advance notifications through the throttled live path."""
        if getattr(self.state, "is_streaming", False):
            self.on_new_sweep(self.state.stimno)
        else:
            self.update_figure()

    @quiet_fail
    def _on_new_sweep_core(self, *args):
        if getattr(self, "_auto_scale_needed", True):
            erp = self.get_erp_signal() if self.state else None
            if erp is not None and len(erp) > 0:
                self._auto_scale()
                self._auto_scale_needed = False
        run_adaptive_render(
            self, self.update_figure, weight="normal", base_interval_ms=40.0
        )

    def on_new_sweep(self, *args):
        if getattr(self, "_auto_scale_needed", True):
            self._on_new_sweep_core(*args)
        elif hasattr(self, "on_new_sweep_throttled"):
            self.on_new_sweep_throttled(*args)
        else:
            self._on_new_sweep_core(*args)

    @quiet_fail
    def _auto_scale(self):
        if self.state is None or self.state.analog_signal is None:
            return
        erp = self.state.get_erp()
        if erp is not None and len(erp) > 0:
            erp_mag = erp.magnitude if hasattr(erp, "magnitude") else erp
            scale_mode = self.settings.get("scale_mode", "mV")
            y_axis_mode = self.settings.get("y_axis_mode", "sweep")
            if scale_mode == "std":
                if y_axis_mode == "time":
                    isi = 1.0
                    if (
                        self.state.event_signal is not None
                        and len(self.state.event_signal) > 1
                    ):
                        try:
                            if hasattr(self.state.event_signal, "times"):
                                t_arr = np.asarray(
                                    self.state.event_signal.times.rescale(
                                        pq.s
                                    ).magnitude
                                )
                            else:
                                t_arr = np.asarray(self.state.event_signal)
                            diffs = np.diff(t_arr)
                            diffs = diffs[diffs > 0]
                            if len(diffs) > 0:
                                isi = float(np.median(diffs))
                        except Exception:
                            isi = 1.0
                    self.settings["scale"] = float(np.round(isi, decimals=4))
                else:
                    self.settings["scale"] = 1.0
                self.scale_spinner.setValue(self.settings["scale"])
            else:
                ylim = float(np.std(np.abs(erp_mag))) * 8
                if ylim > 0:
                    if y_axis_mode == "time":
                        isi = 1.0
                        if (
                            self.state.event_signal is not None
                            and len(self.state.event_signal) > 1
                        ):
                            try:
                                if hasattr(self.state.event_signal, "times"):
                                    t_arr = np.asarray(
                                        self.state.event_signal.times.rescale(
                                            pq.s
                                        ).magnitude
                                    )
                                else:
                                    t_arr = np.asarray(self.state.event_signal)
                                diffs = np.diff(t_arr)
                                diffs = diffs[diffs > 0]
                                if len(diffs) > 0:
                                    isi = float(np.median(diffs))
                            except Exception:
                                isi = 1.0
                        self.settings["scale"] = float(np.round(isi / ylim, decimals=4))
                    else:
                        self.settings["scale"] = float(np.round(1.0 / ylim, decimals=4))
                    self.scale_spinner.setValue(self.settings["scale"])

    def reset_view(self):
        erp = self.get_erp_signal()
        if erp is not None and len(erp) > 0:
            sr = float(self.state.sampling_rate) if self.state and self.state.sampling_rate else 10000.0
            self.main_plot.setXRange(0, (erp.shape[-1] - 1) / sr, padding=0)
        self._auto_scale()
        self.update_figure()

    def _on_click(self, event):
        if self.is_nav_active() or event.button() != Qt.LeftButton or self.state is None:
            return
        if not hasattr(
            self.main_plot, "vb"
        ) or not self.main_plot.sceneBoundingRect().contains(event.scenePos()):
            return
        try:
            pos = self.main_plot.vb.mapSceneToView(event.scenePos())
            y_val = pos.y()
            if np.isnan(y_val):
                return
            y_axis_mode = self.settings.get("y_axis_mode", "sweep")
            if y_axis_mode == "time":
                time_window_s = float(self.settings.get("time_window_s", 10.0))
                post_s = float(self.settings.get("post_seconds", 0.0))
                if y_val < -time_window_s - 1.0 or y_val > post_s + 1.0:
                    return
                if (
                    not hasattr(self, "_displayed_baselines")
                    or len(self._displayed_baselines) == 0
                ):
                    return
                diffs = np.abs(np.asarray(self._displayed_baselines) - y_val)
                target_idx = int(np.argmin(diffs))
                if target_idx < len(self._displayed_stim_indices):
                    target_stim = self._displayed_stim_indices[target_idx]
                    if self.state.event_signal is not None and 0 <= target_stim < len(
                        self.state.event_signal
                    ):
                        self.state.setStimNo(target_stim)
            else:
                stimno = self.state.stimno
                pre = self.settings.get("n_lines_pre", 5)
                post = self.settings.get("n_lines_post", 1)
                n_lines = pre + post + 1
                if y_val < -1.0 or y_val > n_lines:
                    return
                start = max(stimno - pre, 0)
                target_idx = int(np.clip(round(y_val), 0, n_lines - 1))
                target_stim = target_idx + start
                if self.state.event_signal is not None and 0 <= target_stim < len(
                    self.state.event_signal
                ):
                    self.state.setStimNo(target_stim)
        except Exception:
            pass

    @quiet_fail
    def update_figure(self, force: bool = False):
        if self.is_view_culled():
            self._dirty = True
            return
        self._dirty = False

        if (
            self.state is None
            or self.state.analog_signal is None
            or self.state.event_signal is None
        ):
            return

        erp = self.get_erp_signal()
        if erp is None or len(erp) == 0:
            return

        y_axis_mode = self.settings.get("y_axis_mode", "sweep")
        time_window_s = float(self.settings.get("time_window_s", 10.0))
        post_seconds = float(self.settings.get("post_seconds", 0.0))
        n_lines_pre = self.settings.get("n_lines_pre", 5)
        n_lines_post = self.settings.get("n_lines_post", 5)
        scale = float(self.settings.get("scale", 1.0))
        scale_mode = self.settings.get("scale_mode", "mV")
        stimno = self.state.stimno
        primary_ch = (
            self.channel_bar.get_primary_channel()
            if hasattr(self, "channel_bar")
            else ""
        )
        ug = self.get_target_unit()
        ug_spikes = (
            len(ug.event) if ug and hasattr(ug, "event") and ug.event is not None else 0
        )
        ug_idx = self.get_target_unit_idx()

        render_key = (
            stimno,
            len(erp),
            scale,
            n_lines_pre,
            n_lines_post,
            primary_ch,
            ug_idx,
            ug_spikes,
            scale_mode,
            y_axis_mode,
            time_window_s,
            post_seconds,
        )
        if not force and getattr(self, "_last_render_key", None) == render_key:
            return
        self._last_render_key = render_key

        stim_times_s = None
        if (
            y_axis_mode == "time"
            and self.state.event_signal is not None
            and len(self.state.event_signal) > 0
        ):
            try:
                if hasattr(self.state.event_signal, "times"):
                    times = self.state.event_signal.times
                    if hasattr(times, "rescale"):
                        stim_times_s = np.asarray(
                            times.rescale(pq.s).magnitude, dtype=float
                        )
                    else:
                        stim_times_s = np.asarray(times, dtype=float)
                elif hasattr(self.state.event_signal, "rescale"):
                    stim_times_s = np.asarray(
                        self.state.event_signal.rescale(pq.s).magnitude, dtype=float
                    )
                else:
                    stim_times_s = np.asarray(self.state.event_signal, dtype=float)
            except Exception:
                stim_times_s = None

        if y_axis_mode == "time" and stim_times_s is not None and len(stim_times_s) > 0:
            n_total = min(len(stim_times_s), len(erp))
            cur_stim = min(max(0, stimno), n_total - 1)
            t_selected = stim_times_s[cur_stim]
            t_min = t_selected - time_window_s
            t_max = t_selected + post_seconds

            start = int(np.searchsorted(stim_times_s[:n_total], t_min, side="left"))
            stop = int(np.searchsorted(stim_times_s[:n_total], t_max, side="right"))
            start = max(0, min(start, cur_stim))
            stop = max(cur_stim + 1, min(stop, n_total))

            y_baselines = stim_times_s[start:stop] - t_selected
            active_idx = cur_stim - start
        else:
            start = max(stimno - n_lines_pre, 0)
            stop = min(start + n_lines_pre + n_lines_post + 1, len(erp))
            y_baselines = np.arange(stop - start, dtype=float)
            active_idx = stimno - start

        self._displayed_stim_indices = list(range(start, stop))
        self._displayed_baselines = list(y_baselines)

        dpts = erp[start:stop]
        n_display = len(dpts)
        if n_display == 0:
            return

        dpts_mag = dpts.magnitude if hasattr(dpts, "magnitude") else dpts
        sr = (
            float(self.state.sampling_rate.rescale("Hz").magnitude)
            if hasattr(self.state.sampling_rate, "rescale")
            else float(self.state.sampling_rate)
        )
        x = np.arange(erp.shape[-1]) / sr

        if scale_mode == "std":
            dpts_arr = np.asarray(dpts_mag)
            std_val = float(np.std(dpts_arr)) if dpts_arr.size > 0 else 1.0
            if std_val <= 0 or not np.isfinite(std_val):
                std_val = 1.0
            eff_scale = scale / (std_val * 8.0)
        else:
            eff_scale = scale

        # Background sweeps and active sweep
        bg_x_list = []
        bg_y_list = []
        active_x = None
        active_y = None

        for i in range(n_display):
            y = (dpts_mag[i] * eff_scale * -1.0) + y_baselines[i]
            if i == active_idx:
                active_x = x
                active_y = y
            else:
                bg_x_list.append(np.append(x, np.nan))
                bg_y_list.append(np.append(y, np.nan))

        if bg_x_list:
            self.lines_curve.setData(
                np.concatenate(bg_x_list), np.concatenate(bg_y_list)
            )
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
            idx_arr = getattr(ug, "idx_arr", None)
            r = 20
            if idx_arr is not None and len(idx_arr) >= stop:
                for i, s_idx in enumerate(range(start, stop)):
                    entry = idx_arr[s_idx]
                    if entry is None:
                        continue
                    sample_no = int(entry[0])
                    if sample_no < 0 or sample_no >= len(dpts_mag[i]):
                        continue
                    l_sec = sample_no / sr
                    r_start = max(0, sample_no - r)
                    r_end = min(len(dpts_mag[i]), sample_no + r)
                    seg_x = np.arange(r_start, r_end) / sr
                    seg_y = (dpts_mag[i][r_start:r_end] * eff_scale * -1.0) + y_baselines[i]

                    hl_x.append(np.append(seg_x, np.nan))
                    hl_y.append(np.append(seg_y, np.nan))
                    scat_pts.append(
                        (
                            l_sec,
                            (dpts_mag[i][sample_no] * eff_scale * -1.0) + y_baselines[i],
                        )
                    )
            else:
                latencies = ug.get_latencies(self.state.event_signal[start:stop])
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
                    seg_y = (dpts_mag[i][r_start:r_end] * eff_scale * -1.0) + y_baselines[i]

                    hl_x.append(np.append(seg_x, np.nan))
                    hl_y.append(np.append(seg_y, np.nan))

                    scat_pts.append(
                        (
                            l_sec,
                            (dpts_mag[i][sample_no] * eff_scale * -1.0) + y_baselines[i],
                        )
                    )

        if hl_x:
            self.highlight_segments.setData(np.concatenate(hl_x), np.concatenate(hl_y))
        else:
            self.highlight_segments.clear()

        if scat_pts:
            pts_arr = np.array(scat_pts)
            self.points_scatter.setData(x=pts_arr[:, 0], y=pts_arr[:, 1])
        else:
            self.points_scatter.clear()

        # Update Y-axis ticks and range
        if y_axis_mode == "time" and stim_times_s is not None:
            self.main_plot.setLabel("left", "Time", units="s")
            self.main_plot.getAxis("left").setTicks(None)
            pad = max(time_window_s * 0.05, 0.1)
            y_top = -time_window_s - pad
            y_bottom = max(0.0, post_seconds) + pad
            self.main_plot.setYRange(y_top, y_bottom, padding=0)
        else:
            self.main_plot.setLabel("left", "Stimulus Sweep")
            ticks = [(i, str(start + i)) for i in range(n_display)]
            self.main_plot.getAxis("left").setTicks([ticks])
            self.main_plot.setYRange(-1.0, max(n_display, 1), padding=0)

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
                sr = (
                    float(self.state.sampling_rate.rescale("Hz").magnitude)
                    if hasattr(self.state.sampling_rate, "rescale")
                    else float(self.state.sampling_rate)
                )
                self.main_plot.setXRange(0, (erp.shape[-1] - 1) / sr, padding=0)
        else:
            super().keyPressEvent(e)

    def get_settings(self) -> Dict:
        res = dict(self.settings)
        res["time_scale"] = self.settings.get("y_axis_mode") == "time"
        res["n_seconds"] = self.settings.get("time_window_s", 10.0)
        res["xlim"] = self.main_plot.viewRange()[0]
        if hasattr(self, "channel_bar") and self.channel_bar is not None:
            res["channel_bar"] = self.channel_bar.get_settings()
        return res

    @quiet_fail
    def set_settings(self, values: Dict, update_spinners: bool = True):
        self.settings.update(values)
        if "time_scale" in values:
            self.settings["y_axis_mode"] = "time" if values["time_scale"] else "sweep"
        if "n_seconds" in values:
            self.settings["time_window_s"] = float(values["n_seconds"])
        if "xlim" in values:
            self.main_plot.setXRange(values["xlim"][0], values["xlim"][1], padding=0)
            self._has_initialized_xrange = True
        if (
            "channel_bar" in values
            and hasattr(self, "channel_bar")
            and self.channel_bar is not None
        ):
            self.channel_bar.set_settings(values["channel_bar"])
        if update_spinners:
            if "n_lines_pre" in values and hasattr(self, "pre_spinner"):
                self.pre_spinner.setValue(values["n_lines_pre"])
            if "n_lines_post" in values and hasattr(self, "post_spinner"):
                self.post_spinner.setValue(values["n_lines_post"])
            if "scale" in values and hasattr(self, "scale_spinner"):
                self.scale_spinner.setValue(values["scale"])
            if "scale_mode" in values and hasattr(self, "scale_mode_combo"):
                idx = self.scale_mode_combo.findText(values["scale_mode"])
                if idx >= 0:
                    self.scale_mode_combo.blockSignals(True)
                    self.scale_mode_combo.setCurrentIndex(idx)
                    self.scale_mode_combo.blockSignals(False)
            if ("time_window_s" in values or "n_seconds" in values) and hasattr(
                self, "time_window_spinner"
            ):
                self.time_window_spinner.setValue(
                    self.settings.get("time_window_s", 10.0)
                )
            if ("y_axis_mode" in values or "time_scale" in values) and hasattr(
                self, "y_axis_combo"
            ):
                mode = self.settings.get("y_axis_mode", "sweep")
                idx = 1 if mode == "time" else 0
                if self.y_axis_combo.currentIndex() != idx:
                    self.y_axis_combo.setCurrentIndex(idx)
        self._update_toolbar_visibility()
        self.update_figure(force=True)

    def closeEvent(self, event):
        self.unset_connections()
        super().closeEvent(event)
