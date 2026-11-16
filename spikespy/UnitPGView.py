import sys
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import quantities as pq

import PySide6
from PySide6 import QtCore
from PySide6.QtCore import Qt, Signal, Slot, QSignalBlocker
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QApplication,
    QToolBar,
)
import pyqtgraph as pg

from .ViewerState import ViewerState
from .ChannelGroupBar import ChannelGroupBar
from .helpers import qsignal_throttle, quiet_fail, run_adaptive_render, notify_view_selected
from .live_orchestrator import LiveFrameConsumerMixin
from .viewer_worker import get_task_manager, TaskType, TaskPriority
from .PGNavigation import PGNavigationMixin


class UnitPGView(LiveFrameConsumerMixin, PGNavigationMixin, QMainWindow):
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
        self.connections: List[Tuple] = []
        self.lines: Dict[int, np.ndarray] = {}
        self.w: int = 0
        self._dirty: bool = False
        self._last_render_key = None
        self._was_shown: bool = False
        self._user_hidden: bool = False
        self.on_new_sweep_throttled = qsignal_throttle(self._on_new_sweep_core, 50)

        # Background task manager
        self._task_manager = get_task_manager()
        self._viewer_id = f"unit_pg_{id(self)}"

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

    def get_target_unit_idx(self) -> int:
        if hasattr(self, "channel_bar") and self.channel_bar is not None:
            return self.channel_bar.get_target_unit_idx()
        return getattr(self.state, "cur_spike_group", 0) if self.state is not None else 0

    @Slot()
    @Slot(int)
    def _on_stim_no_change(self, stimno: int = -1):
        if stimno >= 0:
            self.update_curstim_line(stimno)
        elif self.state is not None:
            self.update_curstim_line(self.state.stimno)

    @Slot()
    @Slot(int)
    def _on_unit_change(self, *args):
        if getattr(self, "compare_mode", False):
            self._dirty = True
            self._handled_in_unit_change = True
            self.update_all()
            return
        if self.get_target_unit_idx() != getattr(self.state, "cur_spike_group", 0):
            return
        self._dirty = True
        self._handled_in_unit_change = True
        self.update_all()

    @Slot()
    def _on_unit_group_change(self, *args):
        self.update_all()

    @Slot()
    @Slot(str)
    def _on_channel_change(self, *args):
        self.update_all()

    @Slot()
    def _on_load_new_file(self, *args):
        self.update_all()

    @Slot()
    @Slot(int)
    def _on_new_spike_events(self, unit_idx: int = -1):
        if getattr(self, "_handled_in_unit_change", False):
            self._handled_in_unit_change = False
            return
        if not getattr(self, "compare_mode", False) and unit_idx >= 0 and unit_idx != self.get_target_unit_idx():
            return
        self._dirty = True
        if not getattr(self.state, "is_streaming", False):
            self.update_all()
        else:
            self.on_new_sweep_throttled()

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

        self.nav_toolbar = QToolBar("Navigation", self)
        self.add_navigation_actions(self.nav_toolbar)
        self.addToolBar(self.nav_toolbar)
        self.init_navigation()

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
                TaskType.UNIT_VIEW_RENDER: self._on_unit_view_result,
            }
        )

        self._connect_state(self.state.onStimNoChange, self._on_stim_no_change)
        self._connect_state(self.state.onUnitChange, self._on_unit_change)
        self._connect_state(self.state.onUnitGroupChange, self._on_unit_group_change)
        self._register_data_reload(state, self._on_load_new_file)
        if hasattr(self.state, "onChannelChange"):
            self._connect_state(self.state.onChannelChange, self._on_channel_change)
        # Live sweeps arrive through the state's LiveOrchestrator (LIFO);
        # the GUI-side throttle keeps update scheduling rate-limited.
        self._register_live_frames(state, lambda _stimno: self.on_new_sweep_throttled())
        if hasattr(self.state, "onNewSpikeEvents"):
            self._connect_state(self.state.onNewSpikeEvents, self._on_new_spike_events)

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

    @quiet_fail
    def _on_new_sweep_core(self, *args):
        """Lightweight sweep handler - schedules heavy work to background thread."""
        if self.is_view_culled() or self.state is None:
            if self.state is not None:
                self._dirty = True
            return
        run_adaptive_render(self, self._schedule_update_all, weight="normal", base_interval_ms=50.0)

    def on_new_sweep(self, *args):
        if hasattr(self, "on_new_sweep_throttled"):
            self.on_new_sweep_throttled(*args)
        else:
            self._on_new_sweep_core(*args)


    def _schedule_update_all(self):
        """Schedule update_all to background thread."""
        if self.is_view_culled() or self.state is None:
            if self.state is not None:
                self._dirty = True
            return
        
        self._task_manager.submit_render(
            self._viewer_id,
            TaskType.UNIT_VIEW_RENDER,
            {
                "primary_erp": self._get_primary_erp(),
                "neighbor_erps": self._get_neighbor_erps(),
                "spike_group": self._get_target_unit(),
                "event_signal": self.state.event_signal,
                "sampling_rate": self.state.sampling_rate,
                "stimno": self.state.stimno,
                "window_ms": 2.0,
                "show_neighbors": hasattr(self, "channel_bar") and self.channel_bar.is_multichannel(),
                "align_waveforms": True,
            },
            priority=TaskPriority.HIGH,
        )

    def _get_primary_erp(self):
        """Get primary channel ERP."""
        if self.state is None:
            return None
        primary_name = self.channel_bar.get_primary_channel() if hasattr(self, "channel_bar") else getattr(self.state.analog_signal, "name", "")
        return self.state.get_channel_erp(primary_name) if hasattr(self.state, "get_channel_erp") else self.state.get_erp()

    def _get_neighbor_erps(self) -> Dict[str, np.ndarray]:
        """Get neighbor channel ERPs."""
        neighbor_erps = {}
        if hasattr(self, "channel_bar") and self.channel_bar.is_multichannel():
            primary_name = self.channel_bar.get_primary_channel()
            selected = self.channel_bar.get_selected_channels()
            neighbors = [c for c in selected if c != primary_name]
            for ch_name in neighbors:
                if hasattr(self.state, "get_channel_erp"):
                    ch_erp = self.state.get_channel_erp(ch_name)
                    if ch_erp is not None and len(ch_erp) > 0:
                        neighbor_erps[ch_name] = ch_erp
        return neighbor_erps

    def _get_target_unit(self):
        """Get target unit."""
        if hasattr(self, "channel_bar") and self.channel_bar is not None:
            return self.channel_bar.get_target_unit()
        return self.state.getUnitGroup() if self.state else None

    @quiet_fail
    def _sync_lines_from_state(self):
        """Populate snippet line data immediately for this lightweight view state.

        Heavy mean/neighbor rendering still runs in the background; this keeps
        click selection and the active-sweep highlight responsive right away.
        """
        if self.state is None:
            return
        erp = self._get_primary_erp()
        sg = self._get_target_unit()
        if erp is None or sg is None:
            return
        idx_arr = getattr(sg, "idx_arr", None)
        if not idx_arr:
            self.lines = {}
            return
        erp_mag = erp.magnitude if hasattr(erp, "magnitude") else np.asarray(erp)
        if erp_mag.ndim != 2 or erp_mag.shape[0] == 0:
            return
        sr = self.state.sampling_rate
        sr = float(sr.rescale("Hz").magnitude) if hasattr(sr, "rescale") else float(sr or 10000.0)
        w = max(int(sr * 0.002), 1)
        self.w = w
        lines = {}
        for i, item in enumerate(idx_arr):
            if item is None or i >= erp_mag.shape[0]:
                continue
            center = int(item[0] if isinstance(item, (tuple, list, np.ndarray)) else item)
            row = erp_mag[i]
            xmin = max(0, center - w)
            xmax = min(len(row), center + w)
            if xmax <= xmin:
                continue
            y = np.asarray(row[xmin:xmax], dtype=float)
            x = np.arange(xmin - center, xmax - center)
            lines[i] = np.column_stack([x, y])
        self.lines = lines

    @Slot()
    @quiet_fail
    def update_all(self, force: bool = False):
        """Lightweight UI update - heavy computation done in background."""
        if self.is_view_culled():
            self._dirty = True
            return
        self._dirty = False

        primary_ch = getattr(getattr(self.state, "analog_signal", None), "name", "")
        if not primary_ch and hasattr(self, "channel_bar") and self.channel_bar is not None:
            primary_ch = self.channel_bar.get_primary_channel()
        sg = self._get_target_unit()
        ug_spikes = len(sg.event) if sg is not None and getattr(sg, "event", None) is not None else 0
        ug_idx = self.channel_bar.get_target_unit_idx() if hasattr(self, "channel_bar") else 0
        n_sweeps = len(self.state.event_signal) if getattr(self.state, "event_signal", None) is not None else 0
        render_key = (primary_ch, n_sweeps, ug_idx, ug_spikes)
        if not force and self._last_render_key == render_key:
            return
        self._last_render_key = render_key

        self._sync_lines_from_state()
        if self.lines and self.state is not None:
            self._last_highlighted_stim = None
            self.update_curstim_line(self.state.stimno, force=True)
        self._schedule_update_all()

    def _on_unit_view_result(self, result: Dict[str, Any], error: str):
        """Handle completed unit view render task."""
        if error:
            return
        if not result:
            return
        
        xarr = result.get("xarr")
        mean_waveform = result.get("mean_waveform")
        lines_data = result.get("lines_data", {})
        neighbor_data = result.get("neighbor_data", {})
        ylims = result.get("ylims", [-0.1, 0.1])
        cur_stim_line = result.get("cur_stim_line")

        if xarr is not None:
            xarr = np.asarray(getattr(xarr, "magnitude", xarr), dtype=float)
        if mean_waveform is not None:
            mean_waveform = np.asarray(getattr(mean_waveform, "magnitude", mean_waveform), dtype=float)
        ylims = [float(getattr(v, "magnitude", v)) for v in (ylims or [-0.1, 0.1])]

        if lines_data:
            self.lines = {}
            for k, v in lines_data.items():
                arr = np.asarray(getattr(v, "magnitude", v), dtype=float)
                if arr.ndim == 2 and arr.shape[1] >= 2:
                    self.lines[int(k)] = arr[:, :2]

        if xarr is not None and len(xarr) > 0:
            self.w = max(abs(int(xarr[0])), abs(int(xarr[-1])))
        
        if xarr is not None and mean_waveform is not None:
            # Plot all background snippets
            if lines_data:
                all_x = []
                all_y = []
                for line_data in lines_data.values():
                    all_x.append(np.append(line_data[:, 0], np.nan))
                    all_y.append(np.append(line_data[:, 1], np.nan))
                if all_x:
                    self.all_snippets_curve.setData(np.concatenate(all_x), np.concatenate(all_y))
                else:
                    self.all_snippets_curve.clear()
            else:
                self.all_snippets_curve.clear()
            
            # Plot mean waveform
            self.mean_curve.setData(xarr, mean_waveform)
            self.mean_curve.setVisible(True)
            
            # Plot neighbor waveforms
            for ch_name, data in neighbor_data.items():
                # Add neighbor curves if needed (could extend to support multiple)
                pass
            
            self.main_plot.setXRange(-self.w, self.w, padding=0.05)
            self.main_plot.setYRange(ylims[0], ylims[1], padding=0.05)
        
        # Update current stim line
        if cur_stim_line and cur_stim_line.get("visible"):
            sample = cur_stim_line.get("sample", 0)
            # Store for update_curstim_line
            self._current_stim_sample = sample
            self._last_highlighted_stim = None
            self.update_curstim_line(self.state.stimno, force=True)

    @Slot(int)
    @quiet_fail
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

    def reset_view(self):
        self.update_all()

    def mousePressEvent(self, event):
        notify_view_selected(self)
        super().mousePressEvent(event)

    def _on_click(self, event):
        notify_view_selected(self)
        if self.is_nav_active() or event.button() != Qt.LeftButton or not self.lines or self.state is None:
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
        # Unregister from task manager
        if hasattr(self, "_task_manager"):
            self._task_manager.unregister_viewer(self._viewer_id)
        super().closeEvent(event)
