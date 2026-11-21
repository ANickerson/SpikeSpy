import time
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import quantities as pq
from scipy.signal import fftconvolve, find_peaks

from PySide6.QtCore import Qt, QTimer, Slot, QSignalBlocker
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFrame,
    QToolBar,
    QCheckBox,
    QDoubleSpinBox,
    QLabel,
    QPushButton,
)
import pyqtgraph as pg

from .ViewerState import ViewerState, tracked_neuron_unit
from .ChannelGroupBar import ChannelGroupBar
from .helpers import quiet_fail, run_adaptive_render, qsignal_throttle, notify_view_selected
from .live_orchestrator import LiveFrameConsumerMixin
from .viewer_worker import get_task_manager, TaskType, TaskPriority
from .PGNavigation import PGNavigationMixin


class SingleTraceTimeAxisItem(pg.AxisItem):
    """Custom AxisItem converting seconds to milliseconds."""

    def tickStrings(self, values, scale, spacing):
        return [f"{1000 * v:g}" for v in values]


class _SweepConvProxy:
    """Lazily compute template convolution for a single sweep.

    Storing a convolution row per sweep for the whole recording balloons memory
    (and freezes the UI) once a live session holds thousands of sweeps. This
    proxy computes only the requested sweep and keeps a small LRU cache.
    """

    def __init__(self, view: "SingleTracePGView", max_cache: int = 32):
        self.view = view
        self._cache: Dict[int, np.ndarray] = {}
        self._max_cache = max_cache

    def clear_cache(self):
        self._cache.clear()

    def __getitem__(self, stimno: int) -> Optional[np.ndarray]:
        key = int(stimno)
        if key in self._cache:
            return self._cache[key]
        conv_res = self.view._compute_sweep_conv(key)
        if conv_res is None:
            return None
        if len(self._cache) >= self._max_cache:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = conv_res
        return conv_res

    def __len__(self):
        erp = self.view.get_erp_signal() if self.view and self.view.state else None
        return erp.shape[0] if erp is not None else 0


class SingleTracePGView(LiveFrameConsumerMixin, PGNavigationMixin, QMainWindow):
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
        self.connections: List[Tuple] = []
        self.select_local_maxima_width: int = 1
        self.closest_pos: int = 0
        self.conv: Optional[_SweepConvProxy] = None
        self._templates: Dict[str, np.ndarray] = {}
        self._templates_request_time: float = 0.0
        self._conv_width: int = 0
        self._dirty: bool = False
        self._was_shown: bool = False
        self._user_hidden: bool = False
        self._last_render_key = None
        self._last_result_stimno: int = -1
        self._last_label_text: Optional[str] = None
        self._aux_curve_by_channel: Dict[str, pg.PlotDataItem] = {}
        self._aux_plot_by_channel: Dict[str, pg.PlotItem] = {}
        self._aux_trace_by_channel: Dict[str, pg.PlotDataItem] = {}
        self._separate_axes: bool = True
        self._syncing_tracking_controls: bool = False
        self._applying_shared_visuals: bool = False
        self.template_curve: Optional[pg.PlotDataItem] = None

        # Time-based LIFO convolution buffer: accumulates latest sweep, flushes periodically
        self._pending_conv_stimno: Optional[int] = None
        self._pending_conv_data: Optional[Dict] = None
        self._conv_flush_timer = QTimer(self)
        self._conv_flush_timer.setSingleShot(False)
        self._conv_flush_timer.setInterval(200)  # 5 fps for convolution during live streaming
        self._conv_flush_timer.timeout.connect(self._flush_convolution_buffer)

        # Histogram updates remain on slower timer
        self._histogram_timer = QTimer(self)
        self._histogram_timer.setSingleShot(True)
        self._histogram_timer.setInterval(500)
        self._histogram_timer.timeout.connect(self._update_histogram_core)

        self.on_new_sweep_throttled = qsignal_throttle(
            self._on_new_sweep_core, 40, parent=self, weight="primary"
        )

        # Background task manager
        self._task_manager = get_task_manager()
        self._viewer_id = f"singletrace_pg_{id(self)}"
        self.conv = _SweepConvProxy(self)

        # Graphics items (all persistent; per-sweep updates only call setData/setValue)
        self.conv_curve: Optional[pg.PlotDataItem] = None

        # Independent channel group and unit binding toolbar (hidden by default; controlled via ViewSettingsPanel or container)
        self.channel_bar = ChannelGroupBar(parent=self, state=self.state)
        self.channel_bar.bindingChanged.connect(self._on_channel_bar_changed)
        self.addToolBar(Qt.TopToolBarArea, self.channel_bar)
        self.channel_bar.setVisible(False)

        # Axis layout toolbar: one stacked axis per channel, or combined
        self.axis_bar = QToolBar("Axes", self)
        self.axis_bar.setMovable(False)
        self.separate_axes_check = QCheckBox("Separate channel axes", self)
        self.separate_axes_check.setChecked(self._separate_axes)
        self.separate_axes_check.toggled.connect(self._on_axis_mode_changed)
        self.axis_bar.addWidget(self.separate_axes_check)

        self.show_track_window_check = QCheckBox("Show search window", self)
        self.show_track_window_check.setChecked(True)
        self.show_track_window_check.setToolTip(
            "Shade the latency window used to search for the next spike"
        )
        self.show_track_window_check.toggled.connect(
            self._on_track_overlay_visibility_changed
        )
        self.axis_bar.addWidget(self.show_track_window_check)

        self.track_window_spin = QDoubleSpinBox(self)
        self.track_window_spin.setRange(0.1, 500.0)
        self.track_window_spin.setDecimals(1)
        self.track_window_spin.setSingleStep(0.5)
        self.track_window_spin.setSuffix(" ms")
        self.track_window_spin.setToolTip(
            "Tracking window for the bound unit (editable here or by dragging "
            "the shaded region)"
        )
        self.track_window_spin.valueChanged.connect(
            self._on_track_window_spin_changed
        )
        self.axis_bar.addWidget(QLabel("track window:", self))
        self.axis_bar.addWidget(self.track_window_spin)

        self.track_threshold_spin = QDoubleSpinBox(self)
        self.track_threshold_spin.setRange(-1e6, 1e6)
        self.track_threshold_spin.setDecimals(4)
        self.track_threshold_spin.setSingleStep(0.01)
        self.track_threshold_spin.setSuffix(" mV")
        self.track_threshold_spin.setToolTip(
            "Tracking threshold for the bound unit (editable here or by dragging "
            "the horizontal threshold line)"
        )
        self.track_threshold_spin.valueChanged.connect(
            self._on_track_threshold_spin_changed
        )
        self.axis_bar.addWidget(QLabel("threshold:", self))
        self.axis_bar.addWidget(self.track_threshold_spin)

        self.show_template_check = QCheckBox("Show template", self)
        self.show_template_check.setChecked(True)
        self.show_template_check.setToolTip(
            "Overlay the multi-channel template used for convolution"
        )
        self.show_template_check.toggled.connect(
            self._on_track_overlay_visibility_changed
        )
        self.axis_bar.addWidget(self.show_template_check)

        self.add_navigation_actions(self.axis_bar)
        self.addToolBarBreak(Qt.TopToolBarArea)
        self.addToolBar(Qt.TopToolBarArea, self.axis_bar)

        self._init_ui()
        self.init_navigation()

        if state is not None:
            self.set_state(state)

    def _init_ui(self):
        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        self.live_paused_banner = QFrame(central)
        self.live_paused_banner.setObjectName("live_paused_banner")
        self.live_paused_banner.setStyleSheet(
            "QFrame#live_paused_banner { background: #fff3cd; border: 1px solid #ffe69c; border-radius: 4px; padding: 2px 6px; }"
        )
        banner_layout = QHBoxLayout(self.live_paused_banner)
        banner_layout.setContentsMargins(6, 2, 6, 2)
        self.lbl_live_paused = QLabel("⏸ Viewing History", self.live_paused_banner)
        self.btn_return_live = QPushButton("📡 Return to Live (Ctrl+L)", self.live_paused_banner)
        self.btn_return_live.clicked.connect(self._on_return_to_live_clicked)
        banner_layout.addWidget(self.lbl_live_paused)
        banner_layout.addStretch()
        banner_layout.addWidget(self.btn_return_live)
        self.live_paused_banner.setVisible(False)
        layout.addWidget(self.live_paused_banner)

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

        # Waveform line: use native PyQtGraph view-aware peak downsampling so
        # zooming in preserves 100% of samples while zoomed-out views stay fast.
        self.trace_curve = pg.PlotDataItem(pen=pg.mkPen("purple", width=1.5))
        self.trace_curve.setDownsampling(auto=True, method="peak")
        self.trace_curve.setClipToView(True)
        self.main_plot.addItem(self.trace_curve)

        # Shaded tracking search window. Its width is the unit's
        # tracking_window_ms and dragging either edge updates it.
        self.track_window_region = pg.LinearRegionItem(
            values=(0.0, 0.0),
            movable=True,
            brush=pg.mkBrush(90, 140, 255, 35),
            pen=pg.mkPen((60, 110, 220), width=1, style=Qt.DashLine),
            hoverBrush=pg.mkBrush(90, 140, 255, 70),
        )
        self.track_window_region.setZValue(-10)
        self.track_window_region.setVisible(False)
        self.main_plot.addItem(self.track_window_region)
        self.track_window_region.sigRegionChangeFinished.connect(
            self._on_track_region_changed
        )

        # Draggable horizontal threshold line
        self.threshold_line = pg.InfiniteLine(
            angle=0,
            movable=True,
            pen=pg.mkPen((220, 80, 40), width=1.5, style=Qt.DashDotLine),
            hoverPen=pg.mkPen((255, 60, 20), width=2.5),
            label="Thresh: {value:.3f}",
            labelOpts={"position": 0.92, "color": (200, 50, 20), "movable": True},
        )
        self.threshold_line.setZValue(11)
        self.threshold_line.setVisible(False)
        self.main_plot.addItem(self.threshold_line)
        self.threshold_line.sigPositionChangeFinished.connect(
            self._on_threshold_line_dragged
        )

        # Identified spike vertical line
        self.identified_spike_line = pg.InfiniteLine(
            angle=90, pen=pg.mkPen("blue", width=1.5, style=Qt.DashLine), movable=False
        )
        self.main_plot.addItem(self.identified_spike_line)
        self.identified_spike_line.setVisible(False)

        # Previous / next identified spikes: persistent items, updated in place.
        self.prev_spike_line = pg.InfiniteLine(
            angle=90, pen=pg.mkPen(0, 180, 0, 150, width=1.5), movable=False
        )
        self.main_plot.addItem(self.prev_spike_line)
        self.prev_spike_line.setVisible(False)
        self.next_spike_line = pg.InfiniteLine(
            angle=90, pen=pg.mkPen(220, 0, 0, 150, width=1.5), movable=False
        )
        self.main_plot.addItem(self.next_spike_line)
        self.next_spike_line.setVisible(False)

        # Candidate preview: amber dashed latency line + open circle marker.
        # Deliberately distinct from the blue movable tracking-window band.
        if not hasattr(self, "candidate_line"):
            self.candidate_line = pg.InfiniteLine(
                angle=90,
                pen=pg.mkPen(255, 170, 0, 220, width=1.5, style=Qt.DashLine),
                movable=False,
            )
            self.candidate_line.setZValue(12)
            self.main_plot.addItem(self.candidate_line)
            self.candidate_line.setVisible(False)
            self.candidate_marker = pg.ScatterPlotItem(
                size=11,
                symbol="o",
                pen=pg.mkPen((255, 170, 0), width=1.8),
                brush=pg.mkBrush(255, 170, 0, 40),
            )
            self.candidate_marker.setZValue(13)
            self.main_plot.addItem(self.candidate_marker)
            self.candidate_marker.setVisible(False)

        # All convolution peaks drawn as one item (vertical segments y=0..1).
        self.conv_curve = pg.PlotDataItem(
            pen=pg.mkPen(218, 165, 32, 100, width=1), connect="pairs"
        )
        self.top_plot.addItem(self.conv_curve)
        self.conv_curve.setVisible(False)

        # Convolution template preview (normalized and aligned to the spike).
        self.template_curve = pg.PlotDataItem(
            pen=pg.mkPen((0, 140, 200), width=1.6, style=Qt.DashLine)
        )
        self.top_plot.addItem(self.template_curve)
        self.template_curve.setVisible(False)

        # Sweep label text
        self.stimno_label = pg.TextItem(text="", color=(0, 0, 0), anchor=(0, 1))
        self.main_plot.addItem(self.stimno_label)

        # Mouse click handling
        self.main_plot.scene().sigMouseClicked.connect(self._on_plot_clicked)

        layout.addWidget(self.glw)
        self.setCentralWidget(central)

    def _aux_channel_names(self) -> List[str]:
        if self.state is None or not hasattr(self, "channel_bar"):
            return []
        if not self.channel_bar.is_multichannel():
            return []
        primary_name = self.channel_bar.get_primary_channel()
        return [c for c in self.channel_bar.get_selected_channels() if c != primary_name]

    def _rebuild_aux_plots(self):
        """Rebuild the per-channel stacked subplots for the current selection."""
        for plot in self._aux_plot_by_channel.values():
            self.glw.removeItem(plot)
        self._aux_plot_by_channel.clear()
        self._aux_trace_by_channel.clear()
        # Combined-mode dashed curves live in the main plot; drop stale ones so
        # removed channels disappear in both modes.
        for curve in self._aux_curve_by_channel.values():
            self.main_plot.removeItem(curve)
        self._aux_curve_by_channel.clear()

        if not self._separate_axes:
            return

        aux_chans = self._aux_channel_names()
        # With stacked subplots below, only the bottom-most axis shows ticks.
        self.main_plot.getAxis("bottom").setStyle(showValues=not aux_chans)
        for row, ch_name in enumerate(aux_chans, start=2):
            show_ticks = row == 1 + len(aux_chans)
            plot = self.glw.addPlot(
                row=row,
                col=0,
                axisItems=(
                    {"bottom": SingleTraceTimeAxisItem(orientation="bottom")}
                    if show_ticks
                    else None
                ),
            )
            if not show_ticks:
                plot.getAxis("bottom").setStyle(showValues=False)
            plot.setLabel("left", ch_name)
            plot.showGrid(x=True, y=True, alpha=0.3)
            plot.setXLink(self.main_plot)
            curve = pg.PlotDataItem(pen=pg.mkPen(100, 149, 237, width=1.2))
            curve.setDownsampling(auto=True, method="peak")
            curve.setClipToView(True)
            plot.addItem(curve)
            self.glw.ci.layout.setRowStretchFactor(row, 5)
            self._aux_plot_by_channel[ch_name] = plot
            self._aux_trace_by_channel[ch_name] = curve

    def _on_axis_mode_changed(self):
        self._separate_axes = self.separate_axes_check.isChecked()
        self._setup_done = False
        self.setup_figure()

    @quiet_fail
    def _on_channel_bar_changed(self):
        self._templates = {}
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

    @quiet_fail
    def _on_channel_change_event(self, *args):
        self._templates = {}
        self._setup_done = False
        self.setup_figure()
        self.update_histogram()
        self.update_figure(force=True)

    @quiet_fail
    def _on_unit_group_changed(self):
        self._templates = {}
        self.update_figure()
        self.update_histogram()

    @quiet_fail
    def _on_stim_no_change(self, stimno: Optional[int] = None):
        """Handle active-sweep changes without running heavy work on the UI thread.

        During live streaming the trace itself is rendered by the worker; the UI
        thread only updates the cheap sweep label here. Offline/user scrubbing
        keeps the synchronous path because it is user-driven and infrequent.
        """
        if self.is_view_culled() or self.state is None:
            return
        if getattr(self.state, "is_streaming", False):
            self._update_sweep_label()
            target_stim = self.state.stimno if stimno is None else int(stimno)
            if getattr(self, "_last_live_rendered_stim", -1) != target_stim:
                self._last_live_rendered_stim = target_stim
                self.on_new_sweep(target_stim)
        else:
            self.update_figure()
        self._update_live_paused_banner()

    def _on_return_to_live_clicked(self):
        if self.state is not None and hasattr(self.state, "set_follow_live"):
            self.state.set_follow_live(True)
        elif hasattr(self, "channel_bar") and self.channel_bar is not None:
            self.channel_bar.set_follow_live_events(True)
        self._update_live_paused_banner()

    def _update_live_paused_banner(self, *args):
        if not hasattr(self, "live_paused_banner"):
            return
        if self.state is None:
            self.live_paused_banner.setVisible(False)
            return
        ev_sig = getattr(self.state, "event_signal", None)
        n_sweeps = len(ev_sig) if ev_sig is not None else 0
        follow_live = getattr(self.state, "follow_live", True)
        is_streaming = getattr(self.state, "is_streaming", False)
        stimno = int(getattr(self.state, "stimno", 0) or 0)
        latest = max(0, n_sweeps - 1)
        viewing_history = (not follow_live) or (is_streaming and n_sweeps > 0 and stimno < latest)
        if viewing_history and n_sweeps > 0:
            self.lbl_live_paused.setText(
                f"⏸ Viewing History (Sweep #{stimno} / #{latest}) —"
            )
            self.live_paused_banner.setVisible(True)
        elif not follow_live:
            self.lbl_live_paused.setText("⏸ Viewing History —")
            self.live_paused_banner.setVisible(True)
        else:
            self.live_paused_banner.setVisible(False)

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

    def _connect_state(self, sig, slot):
        sig.connect(slot)
        self.connections.append((sig, slot))

    @Slot()
    def _on_data_reloaded(self):
        self.setup_figure()

    @Slot()
    @Slot(int)
    def _on_unit_change(self, *args):
        self.update_figure()
        self.update_histogram()

    @Slot()
    @Slot(int)
    def _on_new_spike_events(self, *args):
        self.update_histogram()

    @Slot()
    @Slot(int)
    def _on_candidates_changed(self, *args):
        self.update_figure(force=True)

    def set_state(self, state: ViewerState):
        self.unset_connections()
        self.state = state
        if hasattr(self, "channel_bar") and self.channel_bar is not None:
            self.channel_bar.setState(state)
        if self.state is None:
            return

        # Register with background task manager
        self._task_manager.register_viewer(
            self._viewer_id,
            self.state,
            {
                TaskType.SINGLE_TRACE_RENDER: self._on_single_trace_result,
                TaskType.HISTOGRAM_UPDATE: self._on_histogram_result,
                TaskType.TEMPLATE_UPDATE: self._on_template_result,
            }
        )

        self._register_data_reload(state, self._on_data_reloaded)
        self._connect_state(self.state.onStimNoChange, self._on_stim_no_change)
        self._connect_state(self.state.onUnitChange, self._on_unit_change)
        self._connect_state(self.state.onUnitGroupChange, self._on_unit_group_changed)
        if hasattr(self.state, "onCandidatesChanged"):
            self._connect_state(
                self.state.onCandidatesChanged, self._on_candidates_changed
            )
        if hasattr(self.state, "onCandidateSelectionChanged"):
            self._connect_state(
                self.state.onCandidateSelectionChanged, self._on_candidates_changed
            )
        if hasattr(self.state, "onChannelChange"):
            self._connect_state(self.state.onChannelChange, self._on_channel_change_event)
        def _on_live_frame_cb(stimno):
            # Only advance when following live events. When the user pauses
            # following (manual navigation or unchecked "Follow Events"), the
            # displayed sweep must stay on state.stimno instead of jumping to
            # each newest incoming sweep.
            if not getattr(self.state, "follow_live", True):
                return
            target = self.state.stimno if stimno is None else int(stimno)
            self._last_live_rendered_stim = target
            self.on_new_sweep(target)

        self._register_live_frames(
            self.state,
            _on_live_frame_cb,
        )
        if hasattr(self.state, "onNewSpikeEvents"):
            self._connect_state(self.state.onNewSpikeEvents, self._on_new_spike_events)
        if hasattr(self.state, "onFollowLiveChanged"):
            self._connect_state(self.state.onFollowLiveChanged, self._update_live_paused_banner)
        if hasattr(self.state, "onUnitParamsChanged"):
            self._connect_state(self.state.onUnitParamsChanged, lambda *_: self._update_tracking_controls())
        if hasattr(self.state, "onContainerVisualSettingsChanged"):
            self._connect_state(
                self.state.onContainerVisualSettingsChanged,
                self._on_container_visual_settings_changed,
            )
        if hasattr(self.state, "onSharedVisualSettingsChanged"):
            self._connect_state(
                self.state.onSharedVisualSettingsChanged,
                self._on_shared_visual_settings_changed,
            )

        self._setup_done = False
        self.setup_figure()
        self._update_live_paused_banner()

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
        if hasattr(self, "_conv_flush_timer"):
            self._conv_flush_timer.stop()
        self._pending_conv_data = None
        self._pending_conv_stimno = None

    def showEvent(self, event):
        super().showEvent(event)
        self._was_shown = True
        self._user_hidden = False
        if getattr(self, "_dirty", False):
            self._dirty = False
            self.update_figure()

    @quiet_fail
    def _on_new_sweep_core(self, stimno: Optional[int] = None):
        """Lightweight sweep handler - schedules heavy work to background thread."""
        if self.is_view_culled() or self.state is None:
            return
        erp = self.get_erp_signal()
        if erp is None or len(erp) == 0:
            return
        if not getattr(self, "_setup_done", False):
            self.setup_figure()
            self._setup_done = True
        
        # Schedule background render task (lightweight, no convolution during live)
        run_adaptive_render(self, lambda: self._schedule_render(stimno), weight="primary")
        
        # During live streaming, queue convolution in LIFO buffer
        if getattr(self.state, "is_streaming", False):
            self._queue_convolution(stimno)
            if not self._conv_flush_timer.isActive():
                self._conv_flush_timer.start()

    def on_new_sweep(self, stimno: Optional[int] = None):
        if not getattr(self, "_setup_done", False):
            self._on_new_sweep_core(stimno)
        elif getattr(self.state, "is_streaming", False) and hasattr(self, "on_new_sweep_throttled"):
            self.on_new_sweep_throttled(stimno)
        else:
            self._on_new_sweep_core(stimno)

    def find_candidates_with_detector(
        self,
        algorithm: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        min_idx: int = 0,
        max_idx: Optional[int] = None,
        stim_idx: Optional[int] = None,
    ):
        """Find candidate spike peaks on the current sweep using SpikeDetectorRegistry."""
        from .dsp.detectors import CandidatePeaks, SpikeDetectorRegistry

        erp = self.get_erp_signal()
        if erp is None or len(erp) == 0 or self.state is None:
            return CandidatePeaks(np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64))
        raw_stim = self.state.stimno if stim_idx is None else int(stim_idx)
        stimno = max(0, min(int(raw_stim), len(erp) - 1))
        sweep = erp[stimno] if getattr(erp, "ndim", 1) == 2 else erp
        ug = self.get_target_unit()
        algo = algorithm or (getattr(ug, "tracking_algorithm", "basic") if ug is not None else "basic")
        merged = dict(getattr(ug, "tracking_params", {}) or {}) if ug is not None else {}
        if ug is not None:
            merged.setdefault("window_ms", float(getattr(ug, "tracking_window_ms", 5.0)))
            merged.setdefault("threshold", float(getattr(ug, "tracking_threshold_mv", 0.01)))
        if params:
            merged.update(params)
        return SpikeDetectorRegistry.find_candidates(
            algo,
            sweep,
            fs_hz=self._sampling_rate_hz(),
            params=merged,
            min_idx=min_idx,
            max_idx=max_idx,
        )

    # ===== Background Task Scheduling =====

    def _schedule_render(self, stimno: Optional[int]):
        """Schedule a render task for the background worker."""
        if self.is_view_culled() or self.state is None:
            return
        erp = self.get_erp_signal()
        if erp is None or len(erp) == 0:
            return

        ug = self.get_target_unit()
        if ug is None:
            return

        # Only the active sweep is displayed: never ship the whole history to
        # the worker, otherwise every update copies thousands of sweeps.
        stimno = self.state.stimno if stimno is None else int(stimno)
        n_sweeps = len(erp) if getattr(erp, "ndim", 1) == 2 else 1
        if n_sweeps == 0:
            return
        stimno = max(0, min(stimno, n_sweeps - 1))
        if getattr(erp, "ndim", 1) == 2:
            primary_sweep = erp[stimno]
        else:
            primary_sweep = erp

        # During live streaming, skip expensive convolution in main render
        is_streaming = getattr(self.state, "is_streaming", False)
        if is_streaming:
            conv_sweeps = {}
            templates = {}
        else:
            conv_channels = self._get_conv_channels()
            conv_sweeps: Dict[str, Any] = {}
            for ch in conv_channels:
                erp_ch = (
                    self.state.get_channel_erp(ch)
                    if hasattr(self.state, "get_channel_erp")
                    else None
                )
                if erp_ch is not None and 0 <= stimno < len(erp_ch):
                    conv_sweeps[ch] = erp_ch[stimno]
            templates = self._templates

        self._maybe_submit_template_update()

        self._task_manager.submit_render(
            self._viewer_id,
            TaskType.SINGLE_TRACE_RENDER,
            {
                "primary_erp": primary_sweep,
                "overlay_erps": self._get_overlay_erps(stimno),
                "conv_sweeps": conv_sweeps,
                "spike_group": ug,
                "event_signal": self.state.event_signal,
                "sampling_rate": self._sampling_rate_hz(),
                "stimno": stimno,
                "window_size": self.state.window_size,
                "channel_spacing": 50.0,
                "conv_channels": list(conv_sweeps.keys()),
                "templates": templates,
                "skip_convolution": is_streaming,  # Flag to skip fftconvolve in worker
                "decimate": False,
            },
            priority=TaskPriority.HIGH,
        )

        # Also schedule histogram updates at lower priority (offline only; streaming uses _histogram_timer debounce)
        if not is_streaming:
            self._task_manager.submit_render(
                self._viewer_id,
                TaskType.HISTOGRAM_UPDATE,
                {
                    "spike_group": ug,
                    "event_signal": self.state.event_signal,
                    "sampling_rate": self._sampling_rate_hz(),
                },
                priority=TaskPriority.LOW,
            )

    def _queue_convolution(self, stimno: int):
        """Queue latest sweep for convolution in LIFO buffer (live streaming only)."""
        if self.is_view_culled() or self.state is None:
            return
        erp = self.get_erp_signal()
        if erp is None or len(erp) == 0:
            return

        ug = self.get_target_unit()
        if ug is None:
            return

        n_sweeps = len(erp) if getattr(erp, "ndim", 1) == 2 else 1
        if n_sweeps == 0:
            return
        stimno = max(0, min(stimno, n_sweeps - 1))
        if getattr(erp, "ndim", 1) == 2:
            primary_sweep = erp[stimno]
        else:
            primary_sweep = erp

        conv_channels = self._get_conv_channels()
        conv_sweeps: Dict[str, Any] = {}
        for ch in conv_channels:
            erp_ch = (
                self.state.get_channel_erp(ch)
                if hasattr(self.state, "get_channel_erp")
                else None
            )
            if erp_ch is not None and 0 <= stimno < len(erp_ch):
                conv_sweeps[ch] = erp_ch[stimno]

        if not conv_sweeps or not self._templates:
            return

        # LIFO: only keep the latest sweep
        self._pending_conv_stimno = stimno
        self._pending_conv_data = {
            "primary_erp": primary_sweep,
            "conv_sweeps": conv_sweeps,
            "spike_group": ug,
            "event_signal": self.state.event_signal,
            "sampling_rate": self._sampling_rate_hz(),
            "stimno": stimno,
            "window_size": self.state.window_size,
            "channel_spacing": 50.0,
            "conv_channels": conv_channels,
            "templates": self._templates,
        }

    def _flush_convolution_buffer(self):
        """Flush the latest pending convolution to worker (called by timer)."""
        if self._pending_conv_data is None or self._pending_conv_stimno is None:
            return
        if self.is_view_culled() or self.state is None:
            return
        
        # Only flush if this is still the current sweep (avoid stale results)
        if self._pending_conv_stimno != self.state.stimno:
            return

        # Submit as a separate lower-priority convolution-only task
        self._task_manager.submit_render(
            self._viewer_id,
            TaskType.SINGLE_TRACE_RENDER,
            {**self._pending_conv_data, "skip_convolution": False, "convolution_only": True},
            priority=TaskPriority.LOW,
        )

    def _get_overlay_erps(self, stimno: int) -> Dict[str, np.ndarray]:
        """Get the active sweep for each overlay channel."""
        overlay = {}
        if hasattr(self, "channel_bar") and self.channel_bar.is_multichannel():
            primary_name = self.channel_bar.get_primary_channel()
            selected = self.channel_bar.get_selected_channels()
            aux_chans = [c for c in selected if c != primary_name]
            for ch_name in aux_chans:
                if hasattr(self.state, "get_channel_erp"):
                    ch_erp = self.state.get_channel_erp(ch_name)
                    if ch_erp is not None and len(ch_erp) > 0 and stimno < len(ch_erp):
                        overlay[ch_name] = ch_erp[stimno]
        return overlay

    def _get_conv_channels(self) -> List[str]:
        """Get channels for template convolution."""
        channels = []
        if hasattr(self, "channel_bar") and self.channel_bar.is_multichannel():
            primary_name = self.channel_bar.get_primary_channel()
            selected = self.channel_bar.get_selected_channels()
            aux_chans = [c for c in selected if c != primary_name]
            channels = [primary_name] + aux_chans if primary_name else aux_chans
        elif self.state.analog_signal is not None:
            channels = [self.state.analog_signal.name]
        return channels

    # ===== Background Task Result Handlers =====

    def _on_single_trace_result(self, result: Dict[str, Any], error: str):
        """Apply a completed render. Only setData/setValue calls happen here."""
        if error or not result:
            return

        stimno = result.get("stimno")
        if stimno is not None:
            # Only apply results the user should see. While following, older
            # frames are stale (drop them, but allow the display to keep
            # advancing). While paused, accept only the selected sweep so
            # backward navigation works and stale renders cannot overwrite it.
            if self.state is not None and self.state.stimno is not None:
                if getattr(self.state, "follow_live", True):
                    if int(stimno) < self._last_result_stimno:
                        return
                elif int(stimno) != int(self.state.stimno):
                    return
            self._last_result_stimno = stimno

        # Convolution-only flushes carry no trace data; applying their empty
        # payloads would blank the waveform, overlays, and markers.
        convolution_only = result.get("convolution_only", False)

        # Apply primary trace (already decimated by the worker)
        primary_trace = result.get("primary_trace")
        if primary_trace and not convolution_only:
            self.trace_curve.setData(primary_trace["x"], primary_trace["y"])

        # Apply overlay traces using persistent curves (no item churn)
        if not convolution_only:
            self._apply_overlay_curves(result.get("overlay_traces", {}))

        # Convolution peaks computed in the worker
        conv_result = result.get("convolution")
        if conv_result:
            self._apply_conv_peaks(conv_result.get("peaks"))

        # Peak positions for peak finding
        peaks = result.get("peaks")
        peaks_down = result.get("peaks_down")
        if peaks is not None:
            self._current_peaks = peaks
            self._current_peaks_down = peaks_down

        # Cheaper label/markers supplied by the worker
        self._update_sweep_label(
            stimno=stimno,
            stim_time=result.get("stim_time"),
        )
        if not convolution_only:
            self._set_marker_lines(
                result.get("cur_latency"),
                result.get("prev_latency"),
                result.get("next_latency"),
            )

    def _apply_overlay_curves(self, overlay_traces: Dict[str, Dict[str, Any]]):
        """Route per-channel traces to stacked axes or combined overlays."""
        if self._separate_axes and self._aux_plot_by_channel:
            self._apply_stacked_traces(overlay_traces)
            return

        colors = [
            (100, 149, 237, 180),
            (220, 20, 60, 180),
            (46, 139, 87, 180),
            (255, 140, 0, 180),
        ]
        seen = set()
        for ch_name, trace_data in (overlay_traces or {}).items():
            curve = self._aux_curve_by_channel.get(ch_name)
            if curve is None:
                color = colors[len(self._aux_curve_by_channel) % len(colors)]
                curve = pg.PlotDataItem(
                    pen=pg.mkPen(color=color, width=1.2, style=Qt.DashLine)
                )
                self.main_plot.addItem(curve)
                self._aux_curve_by_channel[ch_name] = curve
            curve.setData(trace_data["x"], trace_data["y"])
            curve.setVisible(True)
            seen.add(ch_name)

        for ch_name, curve in self._aux_curve_by_channel.items():
            if ch_name not in seen:
                curve.setVisible(False)

        for curve in self._aux_trace_by_channel.values():
            curve.clear()

    def _apply_stacked_traces(self, overlay_traces: Dict[str, Dict[str, Any]]):
        """Set data on the persistent per-channel stacked curves."""
        seen = set()
        for ch_name, trace_data in (overlay_traces or {}).items():
            curve = self._aux_trace_by_channel.get(ch_name)
            if curve is None:
                continue
            curve.setData(trace_data["x"], trace_data["y"])
            curve.setVisible(True)
            seen.add(ch_name)

        for ch_name, curve in self._aux_trace_by_channel.items():
            if ch_name not in seen:
                curve.clear()

    def _update_sweep_label(
        self, stimno: Optional[int] = None, stim_time: Any = None
    ):
        if self.state is None:
            return
        if stimno is None:
            stimno = self.state.stimno
        if stimno is None:
            return

        if stim_time is None and self.state.event_signal is not None:
            if 0 <= stimno < len(self.state.event_signal.times):
                stim_time = self.state.event_signal.times[stimno]

        if stim_time is not None:
            cur_event_time = float(
                getattr(stim_time, "magnitude", stim_time)
            )
            hours, remainder = divmod(cur_event_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            text = (
                f"{stimno:04} [{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}]"
            )
        else:
            text = f"{stimno:04}"

        if text != self._last_label_text:
            self.stimno_label.setText(text)
            self._last_label_text = text
        self.stimno_label.setPos(0, float(self.main_plot.viewRange()[1][1]))

    def _set_marker_lines(
        self,
        cur_latency: Optional[float],
        prev_latency: Optional[float],
        next_latency: Optional[float],
    ):
        """Update the three persistent marker items in place."""
        if cur_latency is not None:
            self.identified_spike_line.setValue(float(cur_latency))
            self.identified_spike_line.setVisible(True)
        else:
            self.identified_spike_line.setVisible(False)

        if prev_latency is not None:
            self.prev_spike_line.setValue(float(prev_latency))
            self.prev_spike_line.setVisible(True)
        else:
            self.prev_spike_line.setVisible(False)

        if next_latency is not None:
            self.next_spike_line.setValue(float(next_latency))
            self.next_spike_line.setVisible(True)
        else:
            self.next_spike_line.setVisible(False)

    def _sampling_rate_hz(self) -> float:
        sr = getattr(self.state, "sampling_rate", None)
        if sr is None:
            return 10000.0
        return float(sr.rescale("Hz").magnitude) if hasattr(sr, "rescale") else float(sr)

    def _target_tracking_center_s(
        self, sg: Optional[tracked_neuron_unit], stimno: int, sr: float
    ) -> Optional[float]:
        """Latency (s) to center the search-window overlay on."""
        if sg is None or sr <= 0:
            return None
        idx_arr = getattr(sg, "idx_arr", None)
        if idx_arr:
            if 0 <= stimno < len(idx_arr) and idx_arr[stimno] is not None:
                item = idx_arr[stimno]
                sample = item[0] if isinstance(item, (tuple, list)) else item
                return float(sample) / sr
            for step in (-1, 1):
                for offset in range(1, 50):
                    i = stimno + step * offset
                    if 0 <= i < len(idx_arr) and idx_arr[i] is not None:
                        item = idx_arr[i]
                        sample = item[0] if isinstance(item, (tuple, list)) else item
                        return float(sample) / sr
        if self.state is not None and self.state.event_signal is not None:
            if 0 <= stimno < len(self.state.event_signal):
                try:
                    lat = sg.get_latencies(
                        np.array([self.state.event_signal[stimno]])
                        * self.state.event_signal.units
                    )[0]
                    if not np.isnan(lat):
                        return float(lat.rescale(pq.s).magnitude)
                except Exception:
                    pass
        return None

    def _signal_display_units(self):
        """Return the physical units of the displayed waveform, or None."""
        erp = self.get_erp_signal() if self.state is not None else None
        units = getattr(erp, "units", None) if erp is not None else None
        if units is None and self.state is not None and getattr(self.state, "analog_signal", None) is not None:
            units = getattr(self.state.analog_signal, "units", None)
        return units

    def _threshold_mv_to_display(self, mv: float) -> float:
        units = self._signal_display_units()
        if units is not None:
            try:
                return float((float(mv) * pq.mV).rescale(units).magnitude)
            except Exception:
                u_str = str(units).lower()
                if "uv" in u_str or "µv" in u_str:
                    return float(mv) * 1000.0
                if "v" in u_str and "mv" not in u_str:
                    return float(mv) * 0.001
        return float(mv)

    def _display_to_threshold_mv(self, y_val: float) -> float:
        units = self._signal_display_units()
        if units is not None:
            try:
                return float((float(y_val) * units).rescale(pq.mV).magnitude)
            except Exception:
                u_str = str(units).lower()
                if "uv" in u_str or "µv" in u_str:
                    return float(y_val) / 1000.0
                if "v" in u_str and "mv" not in u_str:
                    return float(y_val) * 1000.0
        return float(y_val)

    def _update_tracking_controls(
        self, sg: Optional[tracked_neuron_unit] = None
    ) -> None:
        """Sync the tracking spinbox and search-window region from the unit."""
        if self.state is None:
            return
        if sg is None:
            sg = self.get_target_unit()
        window_ms = float(getattr(sg, "tracking_window_ms", 5.0)) if sg else 5.0
        thresh_mv = float(getattr(sg, "tracking_threshold_mv", 0.01)) if sg else 0.01

        self._syncing_tracking_controls = True
        try:
            with QSignalBlocker(self.track_window_spin):
                if abs(self.track_window_spin.value() - window_ms) > 1e-9:
                    self.track_window_spin.setValue(window_ms)

            if hasattr(self, "track_threshold_spin"):
                with QSignalBlocker(self.track_threshold_spin):
                    if abs(self.track_threshold_spin.value() - thresh_mv) > 1e-9:
                        self.track_threshold_spin.setValue(thresh_mv)

            if hasattr(self, "threshold_line"):
                if sg is not None and self.show_track_window_check.isChecked():
                    y_disp = self._threshold_mv_to_display(thresh_mv)
                    with QSignalBlocker(self.threshold_line):
                        self.threshold_line.setValue(y_disp)
                    self.threshold_line.setVisible(True)
                else:
                    self.threshold_line.setVisible(False)

            if sg is None or not self.show_track_window_check.isChecked():
                self.track_window_region.setVisible(False)
                return

            center = self._target_tracking_center_s(
                sg, int(self.state.stimno), self._sampling_rate_hz()
            )
            if center is None:
                self.track_window_region.setVisible(False)
                return

            half = max(0.00005, window_ms / 2000.0)
            self.track_window_region.setRegion((center - half, center + half))
            self.track_window_region.setVisible(True)
        finally:
            self._syncing_tracking_controls = False

    def _on_track_threshold_spin_changed(self, value: float) -> None:
        if self._syncing_tracking_controls or self.state is None:
            return
        sg = self.get_target_unit()
        if sg is None:
            return
        sg.tracking_threshold_mv = float(value)
        if hasattr(sg, "event") and hasattr(sg.event, "annotations"):
            sg.event.annotations["tracking_threshold_mv"] = float(value)
        sg._template_matcher = None
        self._update_tracking_controls(sg)
        if hasattr(self.state, "onUnitParamsChanged"):
            self.state.onUnitParamsChanged.emit(self.get_target_unit_idx())

    def _on_threshold_line_dragged(self) -> None:
        if self._syncing_tracking_controls or self.state is None:
            return
        sg = self.get_target_unit()
        if sg is None:
            return
        y_val = float(self.threshold_line.value())
        thresh_mv = self._display_to_threshold_mv(y_val)
        sg.tracking_threshold_mv = float(thresh_mv)
        if hasattr(sg, "event") and hasattr(sg.event, "annotations"):
            sg.event.annotations["tracking_threshold_mv"] = float(thresh_mv)
        sg._template_matcher = None
        self._syncing_tracking_controls = True
        try:
            if hasattr(self, "track_threshold_spin"):
                with QSignalBlocker(self.track_threshold_spin):
                    self.track_threshold_spin.setValue(float(thresh_mv))
        finally:
            self._syncing_tracking_controls = False
        if hasattr(self.state, "onUnitParamsChanged"):
            self.state.onUnitParamsChanged.emit(self.get_target_unit_idx())

    def _on_track_window_spin_changed(self, value: float) -> None:
        if self._syncing_tracking_controls or self.state is None:
            return
        sg = self.get_target_unit()
        if sg is None:
            return
        sg.tracking_window_ms = float(value)
        if hasattr(sg, "event") and hasattr(sg.event, "annotations"):
            sg.event.annotations["tracking_window_ms"] = float(value)
        sg._template_matcher = None
        self._update_tracking_controls(sg)
        if hasattr(self.state, "onUnitParamsChanged"):
            self.state.onUnitParamsChanged.emit(self.get_target_unit_idx())

    def _on_track_region_changed(self) -> None:
        if self._syncing_tracking_controls or self.state is None:
            return
        sg = self.get_target_unit()
        if sg is None:
            return
        lo, hi = self.track_window_region.getRegion()
        width_ms = abs(float(hi) - float(lo)) * 1000.0
        if width_ms < 0.1:
            return
        sg.tracking_window_ms = float(width_ms)
        if hasattr(sg, "event") and hasattr(sg.event, "annotations"):
            sg.event.annotations["tracking_window_ms"] = float(width_ms)
        sg._template_matcher = None
        self._syncing_tracking_controls = True
        try:
            with QSignalBlocker(self.track_window_spin):
                self.track_window_spin.setValue(float(width_ms))
        finally:
            self._syncing_tracking_controls = False
        if hasattr(self.state, "onUnitParamsChanged"):
            self.state.onUnitParamsChanged.emit(self.get_target_unit_idx())

    def _on_track_overlay_visibility_changed(self, *args) -> None:
        self._update_tracking_controls()
        self._update_template_curve()

    def _update_template_curve(
        self, sg: Optional[tracked_neuron_unit] = None
    ) -> None:
        """Draw the convolution template currently used for this unit."""
        if self.template_curve is None:
            return
        if not self.show_template_check.isChecked() or not self._templates:
            self.template_curve.setVisible(False)
            return
        if sg is None:
            sg = self.get_target_unit()

        primary = (
            self.channel_bar.get_primary_channel()
            if hasattr(self, "channel_bar")
            else ""
        )
        tpl = self._templates.get(primary) if primary else None
        if tpl is None:
            tpl = next(iter(self._templates.values()))
        tpl = np.asarray(tpl, dtype=np.float64).ravel()
        if tpl.size < 2 or not np.all(np.isfinite(tpl)):
            self.template_curve.setVisible(False)
            return

        lo = float(np.min(tpl))
        hi = float(np.max(tpl))
        if hi - lo < 1e-12:
            self.template_curve.setVisible(False)
            return

        y = 0.05 + 0.90 * (tpl - lo) / (hi - lo)
        sr = self._sampling_rate_hz()
        if sr <= 0:
            self.template_curve.setVisible(False)
            return

        center = self._target_tracking_center_s(
            sg, int(self.state.stimno), sr
        ) if self.state is not None else None
        start_s = (center - 0.010) if center is not None else 0.0
        x = start_s + np.arange(tpl.size) / sr
        self.template_curve.setData(x=x, y=y)
        self.template_curve.setVisible(True)

    def _apply_conv_peaks(self, peaks: Optional[np.ndarray]):
        """Draw convolution peaks using a single pair-connected curve."""
        if self.conv_curve is None:
            return
        if peaks is None or len(peaks) == 0:
            self.conv_curve.setVisible(False)
            return

        sr = self._sampling_rate_hz()
        if sr <= 0:
            return

        times = np.asarray(peaks, dtype=np.float64) / sr
        xs = np.repeat(times, 2)
        ys = np.tile(np.array([0.0, 1.0]), len(times))
        self.conv_curve.setData(x=xs, y=ys, connect="pairs")
        self.conv_curve.setVisible(True)

    def _update_spike_markers(self):
        """Update markers for non-live/interactive paths using in-place items."""
        sg = self.get_target_unit()
        if (
            sg is None
            or not getattr(sg, "idx_arr", None)
            or self.state is None
            or self.state.event_signal is None
        ):
            self._set_marker_lines(None, None, None)
            return

        sr = self._sampling_rate_hz()
        stimno = self.state.stimno
        if not (0 <= stimno < len(self.state.event_signal.times)):
            self._set_marker_lines(None, None, None)
            return

        cur_ev = self.state.event_signal.times[stimno]
        try:
            lat = sg.get_latencies(np.array([cur_ev]) * cur_ev.units)[0].rescale("s")
            cur_latency = (
                float(lat.magnitude)
                if not np.isnan(lat) and lat < self.state.window_size
                else None
            )
        except Exception:
            cur_latency = None

        def _nearest(step: int) -> Optional[float]:
            idx_arr = sg.idx_arr
            for offset in range(1, 10):
                i = stimno + step * offset
                if 0 <= i < len(idx_arr) and idx_arr[i] is not None:
                    return idx_arr[i][0] / sr
            return None

        self._set_marker_lines(cur_latency, _nearest(-1), _nearest(1))

    def _update_candidate_overlay(self, stimno: int, dpts_mag, sr: float):
        if not hasattr(self, "candidate_line"):
            return
        candidate = (
            self.state.get_active_candidate()
            if self.state is not None and hasattr(self.state, "get_active_candidate")
            else None
        )
        if candidate is None:
            self.candidate_line.setVisible(False)
            self.candidate_marker.clear()
            self.candidate_marker.setVisible(False)
            return
        primary = (
            self.channel_bar.get_primary_channel()
            if hasattr(self, "channel_bar") and self.channel_bar is not None
            else getattr(getattr(self.state, "analog_signal", None), "name", "")
        )
        if candidate.channel and primary and candidate.channel != primary:
            self.candidate_line.setVisible(False)
            self.candidate_marker.clear()
            self.candidate_marker.setVisible(False)
            return
        sr = float(sr) or 1.0
        center_s = float(candidate.latency_center_samples) / sr
        self.candidate_line.setValue(center_s)
        self.candidate_line.setVisible(True)
        item = None
        idx_arr = getattr(candidate.unit, "idx_arr", None) or []
        if 0 <= int(stimno) < len(idx_arr):
            item = idx_arr[int(stimno)]
        if item is not None:
            sample = int(item[0] if isinstance(item, (tuple, list)) else item)
            if 0 <= sample < len(dpts_mag):
                self.candidate_marker.setData(
                    x=[sample / sr],
                    y=[float(dpts_mag[sample])],
                    size=11,
                    symbol="o",
                    pen=pg.mkPen((255, 170, 0), width=1.8),
                    brush=pg.mkBrush(255, 170, 0, 40),
                )
                self.candidate_marker.setVisible(True)
                return
        self.candidate_marker.clear()
        self.candidate_marker.setVisible(False)

    def _on_histogram_result(self, result: Dict[str, Any], error: str):
        """Handle completed histogram computation."""
        if error or not result:
            self.histogram_curve.clear()
            return
        bins = result.get("bins")
        counts = result.get("counts")
        if bins is None or counts is None:
            self.histogram_curve.clear()
            return
        bins = np.asarray(bins)
        counts = np.asarray(counts)
        if bins.size == counts.size:
            # Older payloads returned right edges; add the final edge so the
            # step-mode curve has one more x value than y values.
            step = bins[1] - bins[0] if bins.size > 1 else 1.0
            bins = np.append(bins, bins[-1] + step)
        if bins.size == counts.size + 1:
            self.histogram_curve.setData(x=bins, y=counts)
        else:
            self.histogram_curve.clear()

    def _on_template_result(self, result: Dict[str, np.ndarray], error: str):
        """Handle completed template computation (worker thread)."""
        if error:
            return
        self._templates = result or {}
        if self.conv is not None:
            self.conv.clear_cache()
        self._update_template_curve()
        if not getattr(self.state, "is_streaming", False):
            self.update_figure(force=True)

    def unset_connections(self):
        self._unregister_live_frames()
        for sig, slot in self.connections:
            try:
                sig.disconnect(slot)
            except (TypeError, RuntimeError):
                pass
        self.connections.clear()

    @quiet_fail
    def setup_figure(self):
        if self.state is None or self.state.analog_signal is None:
            return
        erp = self.get_erp_signal()
        if erp is None or len(erp) == 0:
            return

        self._rebuild_aux_plots()

        sr = self._sampling_rate_hz()
        duration_sec = erp.shape[-1] / sr
        self.main_plot.setXRange(0, duration_sec, padding=0)

        # Scale to the sweep actually displayed instead of scanning the full
        # sweep history, which can be thousands of sweeps during a live session.
        erp_mag = erp.magnitude if hasattr(erp, "magnitude") else erp
        if getattr(erp_mag, "ndim", 1) == 2 and erp_mag.shape[0] > 0:
            stimno = max(0, min(int(self.state.stimno), erp_mag.shape[0] - 1))
            display_mag = np.asarray(erp_mag[stimno])
        else:
            display_mag = np.asarray(erp_mag)
        ylim = float(np.std(np.abs(display_mag)) * 8)
        if ylim <= 0 or not np.isfinite(ylim):
            ylim = 1.0
        self.main_plot.setYRange(-ylim, ylim, padding=0)

        # Per-channel stacked subplots follow the same x range; scale each
        # y axis to its own displayed sweep.
        if self._separate_axes and self._aux_plot_by_channel:
            for ch_name, plot in self._aux_plot_by_channel.items():
                ch_erp = (
                    self.state.get_channel_erp(ch_name)
                    if hasattr(self.state, "get_channel_erp")
                    else None
                )
                if ch_erp is None or len(ch_erp) == 0:
                    continue
                ch_mag = ch_erp.magnitude if hasattr(ch_erp, "magnitude") else ch_erp
                if getattr(ch_mag, "ndim", 1) == 2 and ch_mag.shape[0] > 0:
                    i = max(0, min(int(self.state.stimno), ch_mag.shape[0] - 1))
                    ch_disp = np.asarray(ch_mag[i])
                else:
                    ch_disp = np.asarray(ch_mag)
                ch_ylim = float(np.std(np.abs(ch_disp)) * 8)
                if ch_ylim > 0:
                    plot.setYRange(-ch_ylim, ch_ylim, padding=0)

        self._setup_done = True
        self._update_histogram_core()
        self.update_figure()

    @quiet_fail
    def update_histogram(self, *args):
        if not self._histogram_timer.isActive():
            self._histogram_timer.start()

    def _update_histogram_core(self):
        """Refresh histogram/templates; both are computed by the worker."""
        if self.state is None or self.state.event_signal is None:
            return

        sg = self.get_target_unit()
        if sg is None or len(sg.event) == 0:
            self.histogram_curve.clear()
            self._templates = {}
            return

        self._submit_template_update()
        self._task_manager.submit_render(
            self._viewer_id,
            TaskType.HISTOGRAM_UPDATE,
            {
                "spike_group": sg,
                "event_signal": self.state.event_signal,
                "sampling_rate": self._sampling_rate_hz(),
            },
            priority=TaskPriority.LOW,
        )

    def _submit_template_update(self):
        """Queue template computation on the background worker."""
        if self.state is None or self.state.event_signal is None:
            return
        sg = self.get_target_unit()
        if sg is None or sg.event is None or len(sg.event) == 0:
            self._templates = {}
            return

        channel_erps: Dict[str, Any] = {}
        for ch in self._get_conv_channels():
            erp_ch = (
                self.state.get_channel_erp(ch)
                if hasattr(self.state, "get_channel_erp")
                else None
            )
            if erp_ch is not None and len(erp_ch) > 0:
                channel_erps[ch] = erp_ch

        if not channel_erps:
            return

        self._templates_request_time = time.monotonic()
        self._task_manager.submit_template_update(
            self._viewer_id,
            {
                "spike_group": sg,
                "event_signal": self.state.event_signal,
                "sampling_rate": self._sampling_rate_hz(),
                "channels": list(channel_erps.keys()),
                "channel_erps": channel_erps,
            },
        )

    def _maybe_submit_template_update(self):
        now = time.monotonic()
        if (now - self._templates_request_time) < 1.0:
            return
        self._submit_template_update()

    def _update_templates(self, force: bool = False):
        """Ensure templates exist; they are built by the worker, never here."""
        if force or not self._templates:
            self._maybe_submit_template_update()

    def _compute_sweep_conv(self, stimno: int) -> Optional[np.ndarray]:
        """Template correlation for one sweep only."""
        if self.state is None:
            return None
        self._update_templates()
        erp = self.get_erp_signal()
        if erp is None or len(erp) == 0:
            return None
        erp_mag = erp.magnitude if hasattr(erp, "magnitude") else erp
        if getattr(erp_mag, "ndim", 1) != 2:
            return None
        width = int(erp_mag.shape[1])
        self._conv_width = width
        if not self._templates or stimno < 0 or stimno >= erp_mag.shape[0]:
            return np.zeros(width, dtype=np.float32)

        composite_conv = None
        for ch, tpl in self._templates.items():
            erp_ch = (
                self.state.get_channel_erp(ch)
                if hasattr(self.state, "get_channel_erp")
                else erp
            )
            if erp_ch is None or stimno >= len(erp_ch):
                continue
            sweep_data = erp_ch[stimno]
            if hasattr(sweep_data, "magnitude"):
                sweep_data = sweep_data.magnitude
            sweep_data = np.asarray(sweep_data, dtype=np.float32)
            if sweep_data.size == 0:
                continue

            tpl = np.asarray(tpl, dtype=np.float32)[: sweep_data.size]
            if tpl.size == 0:
                continue
            std_sweep = float(np.std(sweep_data))
            norm_sweep = sweep_data / (std_sweep if std_sweep > 1e-8 else 1.0)
            c = fftconvolve(norm_sweep, tpl, mode="same").astype(np.float32)
            c = c[: sweep_data.size]
            c[0 : len(tpl)] = 0
            c[-len(tpl) :] = 0
            if composite_conv is None:
                composite_conv = c
            else:
                composite_conv += c

        if composite_conv is None:
            return np.zeros(width, dtype=np.float32)
        return composite_conv

    def _get_conv_peaks(self, conv_row: Optional[np.ndarray]) -> np.ndarray:
        """Extract peak sample positions from a single-sweep convolution row."""
        if conv_row is None or len(conv_row) == 0:
            return np.array([], dtype=int)
        n = len(conv_row)
        margin = min(1000, max(1, n // 10))
        inner = conv_row[margin:-margin] if n > 2 * margin else conv_row
        if len(inner) == 0:
            return np.array([], dtype=int)
        p99 = float(np.percentile(inner, 99))
        peaks, _ = find_peaks(conv_row, height=p99)
        return peaks - 1

    @quiet_fail
    def update_figure(self, force: bool = False):
        """Lightweight UI update - heavy computation done in background."""
        if self.is_view_culled():
            self._dirty = True
            return
        self._dirty = False
        run_adaptive_render(self, lambda: self._do_update_figure_core(force=force), weight="primary")

    def _do_update_figure_core(self, force: bool = False):
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
        candidate = (
            self.state.get_active_candidate()
            if hasattr(self.state, "get_active_candidate")
            else None
        )
        candidate_key = (
            (
                int(candidate.candidate_id),
                int(candidate.hit_count),
                int(candidate.latency_center_samples),
            )
            if candidate is not None
            else None
        )

        render_key = (
            stimno,
            len(erp),
            primary_ch,
            aux_chans,
            ug_idx,
            ug_spikes,
            candidate_key,
        )
        if not force and getattr(self, "_last_render_key", None) == render_key:
            return
        self._last_render_key = render_key

        dpts = erp[stimno]
        dpts_mag = dpts.magnitude if hasattr(dpts, "magnitude") else dpts
        sr = self._sampling_rate_hz()
        x_sec = np.arange(len(dpts_mag)) / sr

        # Update waveform curve (lightweight)
        self.trace_curve.setData(x_sec, dpts_mag)

        # Multichannel overlay: reuse persistent curves
        overlay_data: Dict[str, Dict[str, Any]] = {}
        if hasattr(self, "channel_bar") and self.channel_bar.is_multichannel():
            primary_name = self.channel_bar.get_primary_channel()
            selected = self.channel_bar.get_selected_channels()
            aux_chans = [c for c in selected if c != primary_name]
            for ch_name in aux_chans:
                if hasattr(self.state, "get_channel_erp"):
                    ch_erp = self.state.get_channel_erp(ch_name)
                    if ch_erp is not None and stimno < len(ch_erp):
                        ch_dpts = ch_erp[stimno]
                        ch_mag = (
                            ch_dpts.magnitude
                            if hasattr(ch_dpts, "magnitude")
                            else ch_dpts
                        )
                        overlay_data[ch_name] = {
                            "x": x_sec[: len(ch_mag)],
                            "y": ch_mag,
                        }
        self._apply_overlay_curves(overlay_data)

        # Update sweep label and markers in place (no item churn)
        self._update_sweep_label(stimno)
        self._update_spike_markers()
        self._update_candidate_overlay(stimno, dpts_mag, sr)
        self._update_tracking_controls(ug)
        self._update_template_curve(ug)

        # During live streaming the worker supplies convolution peaks; only
        # compute the fallback for offline/interactive paths.
        if not getattr(self.state, "is_streaming", False):
            conv_row = self.conv[stimno] if self.conv is not None else None
            self._apply_conv_peaks(self._get_conv_peaks(conv_row))

    def reset_view(self):
        erp = self.get_erp_signal()
        if erp is None or len(erp) == 0:
            return
        sr = self._sampling_rate_hz()
        duration_sec = erp.shape[-1] / sr
        self.main_plot.setXRange(0, duration_sec, padding=0)
        erp_mag = erp.magnitude if hasattr(erp, "magnitude") else erp
        if getattr(erp_mag, "ndim", 1) == 2 and erp_mag.shape[0] > 0:
            stimno = max(0, min(int(self.state.stimno), erp_mag.shape[0] - 1)) if self.state else 0
            display_mag = np.asarray(erp_mag[stimno])
        else:
            display_mag = np.asarray(erp_mag)
        ylim = float(np.std(np.abs(display_mag)) * 8)
        if ylim > 0:
            self.main_plot.setYRange(-ylim, ylim, padding=0)

    def mousePressEvent(self, event):
        notify_view_selected(self)
        super().mousePressEvent(event)

    def _plot_channel_name(self, plot: pg.PlotItem) -> str:
        if plot is self.main_plot:
            if hasattr(self, "channel_bar") and self.channel_bar is not None:
                ch = self.channel_bar.get_primary_channel()
                if ch:
                    return ch
            if self.state is not None and getattr(self.state, "analog_signal", None) is not None:
                return getattr(self.state.analog_signal, "name", "") or ""
            return ""
        for ch_name, aux_plot in self._aux_plot_by_channel.items():
            if aux_plot is plot:
                return ch_name
        return ""

    def seed_new_unit_at_pos(self, x: float, channel_name: Optional[str] = None):
        if np.isnan(x) or self.state is None:
            return None
        erp = None
        if channel_name and hasattr(self.state, "get_channel_erp"):
            erp = self.state.get_channel_erp(channel_name)
        if erp is None or len(erp) == 0:
            erp = self.get_erp_signal()
        if erp is None or len(erp) == 0:
            return None

        stimno = int(getattr(self.state, "stimno", 0) or 0)
        if stimno < 0 or stimno >= len(erp):
            return None
        dpts = erp[stimno]
        if hasattr(dpts, "magnitude"):
            dpts = dpts.magnitude
        dpts = np.asarray(dpts).ravel()
        n_pts = len(dpts)
        if n_pts == 0:
            return None

        x_idx = max(0, min(n_pts - 1, int(round(x))))
        w = max(5, int(self.select_local_maxima_width))
        start_w = max(0, x_idx - w)
        end_w = min(n_pts, x_idx + w + 1)
        if end_w > start_w:
            sub_slice = np.abs(dpts[start_w:end_w])
            if len(sub_slice) > 0:
                x_idx = start_w + int(np.argmax(sub_slice))

        if not channel_name:
            channel_name = self._plot_channel_name(self.main_plot)

        if hasattr(self.state, "seed_unit_at"):
            return self.state.seed_unit_at(
                x_idx,
                stimno=stimno,
                channel=channel_name,
                auto_track=True,
            )

        # Fallback for legacy ViewerState without seed_unit_at
        self.state.addUnitGroup()
        new_idx = len(self.state.spike_groups) - 1
        if new_idx < 0:
            return None
        sg = self.state.spike_groups[new_idx]
        if channel_name:
            sg.channel = channel_name
            sg.tracking_channel = channel_name
            if hasattr(sg, "event") and hasattr(sg.event, "annotations"):
                sg.event.annotations["channel"] = channel_name
                sg.event.annotations["tracking_channel"] = channel_name
        peak_disp = abs(float(dpts[x_idx]))
        thresh_mv = max(0.001, self._display_to_threshold_mv(peak_disp * 0.7))
        sg.tracking_threshold_mv = thresh_mv
        sg.auto_track = True
        if hasattr(sg, "event") and hasattr(sg.event, "annotations"):
            sg.event.annotations["tracking_threshold_mv"] = thresh_mv
            sg.event.annotations["auto_track"] = True
        if hasattr(self.state, "setUnitGroup"):
            self.state.setUnitGroup(new_idx)
        self.state.setUnit(x_idx, stimno=stimno, unitid=new_idx)
        return new_idx

    def _on_plot_clicked(self, event):
        notify_view_selected(self)
        if self.is_nav_active() or event.button() != Qt.LeftButton or self.state is None:
            return
        plots = [self.main_plot] + list(self._aux_plot_by_channel.values())
        plot = next(
            (
                p
                for p in plots
                if hasattr(p, "vb")
                and p.sceneBoundingRect().contains(event.scenePos())
            ),
            None,
        )
        if plot is None:
            return
        pos = plot.vb.mapSceneToView(event.scenePos())
        if np.isnan(pos.x()):
            return
        sr = float(self.state.sampling_rate) if self.state.sampling_rate is not None else 10000.0
        sample_x = pos.x() * sr
        modifiers = (
            event.modifiers()
            if hasattr(event, "modifiers") and callable(event.modifiers)
            else Qt.NoModifier
        )
        if modifiers & Qt.ShiftModifier:
            ch_name = self._plot_channel_name(plot)
            self.seed_new_unit_at_pos(sample_x, ch_name)
            return
        self.set_cur_pos(sample_x)

    def _on_container_visual_settings_changed(self, settings: Dict[str, Any]):
        if isinstance(settings, dict):
            self.apply_shared_visual_settings(settings)

    def _on_shared_visual_settings_changed(self, *args):
        set_name = None
        settings = None
        if len(args) == 1:
            if isinstance(args[0], dict):
                settings = args[0]
            elif isinstance(args[0], str):
                set_name = args[0]
        elif len(args) >= 2:
            set_name = args[0]
            if isinstance(args[1], dict):
                settings = args[1]

        bound_set = (
            self.channel_bar.get_bound_shared_set()
            if hasattr(self, "channel_bar") and self.channel_bar is not None
            and hasattr(self.channel_bar, "get_bound_shared_set")
            else None
        )
        if set_name is not None and set_name != bound_set:
            return
        if settings is None and self.state is not None and set_name is not None:
            if hasattr(self.state, "get_shared_visual_settings"):
                settings = self.state.get_shared_visual_settings(set_name)
            elif hasattr(self.state, "get_shared_channel_set"):
                s = self.state.get_shared_channel_set(set_name)
                if isinstance(s, dict):
                    settings = s.get("visual_settings")
        if isinstance(settings, dict):
            self.apply_shared_visual_settings(settings)

    def apply_shared_visual_settings(self, settings: Dict[str, Any]):
        if not isinstance(settings, dict) or self._applying_shared_visuals:
            return
        self._applying_shared_visuals = True
        try:
            if "show_track_window" in settings and hasattr(self, "show_track_window_check"):
                val = bool(settings["show_track_window"])
                with QSignalBlocker(self.show_track_window_check):
                    self.show_track_window_check.setChecked(val)
                self._update_tracking_controls()
            if "show_template" in settings and hasattr(self, "show_template_check"):
                val = bool(settings["show_template"])
                with QSignalBlocker(self.show_template_check):
                    self.show_template_check.setChecked(val)
                self._update_template_curve()
            if "x_range" in settings and settings["x_range"] is not None:
                xr = settings["x_range"]
                if len(xr) == 2 and xr[0] is not None and xr[1] is not None:
                    x0, x1 = float(xr[0]), float(xr[1])
                    if x0 < x1:
                        self.main_plot.setXRange(x0, x1, padding=0)
        finally:
            self._applying_shared_visuals = False

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
            if ug is None or not getattr(ug, "idx_arr", None):
                return
            item = (
                ug.idx_arr[self.state.stimno]
                if 0 <= self.state.stimno < len(ug.idx_arr)
                else None
            )
            if item is None:
                return
            cur_stimpos = item[0]
            conv_row = self.conv[self.state.stimno]
            peaks = self._get_conv_peaks(conv_row)
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
        res = {
            "xlim": self.main_plot.viewRange()[0],
            "separate_axes": self._separate_axes,
            "show_track_window": self.show_track_window_check.isChecked(),
            "show_template": self.show_template_check.isChecked(),
        }
        if hasattr(self, "channel_bar") and self.channel_bar is not None:
            res["channel_bar"] = self.channel_bar.get_settings()
        return res

    def set_settings(self, values: Dict):
        if "separate_axes" in values:
            self._separate_axes = bool(values["separate_axes"])
            if hasattr(self, "separate_axes_check"):
                with QSignalBlocker(self.separate_axes_check):
                    self.separate_axes_check.setChecked(self._separate_axes)
        if "show_track_window" in values:
            with QSignalBlocker(self.show_track_window_check):
                self.show_track_window_check.setChecked(
                    bool(values["show_track_window"])
                )
        if "show_template" in values:
            with QSignalBlocker(self.show_template_check):
                self.show_template_check.setChecked(bool(values["show_template"]))
        if "xlim" in values:
            self.main_plot.setXRange(values["xlim"][0], values["xlim"][1], padding=0)
        if "channel_bar" in values and hasattr(self, "channel_bar") and self.channel_bar is not None:
            self.channel_bar.set_settings(values["channel_bar"])
        self._update_tracking_controls()
        self._update_template_curve()

    def closeEvent(self, event):
        self.unset_connections()
        if hasattr(self, "_conv_flush_timer"):
            self._conv_flush_timer.stop()
        if hasattr(self, "_histogram_timer"):
            self._histogram_timer.stop()
        # Unregister from task manager
        if hasattr(self, "_task_manager"):
            self._task_manager.unregister_viewer(self._viewer_id)
        super().closeEvent(event)
