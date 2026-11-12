import sys
from typing import Any, List, Optional, Union

import matplotlib.style as mplstyle
import numpy as np
import PySide6
import quantities as pq
from matplotlib.backend_tools import ToolToggleBase
from matplotlib.backends.backend_qtagg import FigureCanvas, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.path import Path
from matplotlib.widgets import PolygonSelector
from PySide6.QtCore import QAbstractTableModel, QModelIndex, QObject, Qt, Signal, Slot, QTimer
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


from .ViewerState import ViewerState
from .QMatplotlib import QMatplotlib
from .ChannelGroupBar import ChannelGroupBar

mplstyle.use("fast")


class UnitView(QMatplotlib):
    def __init__(
        self,
        parent: PySide6.QtWidgets.QWidget = None,
        state: ViewerState = None,
    ):

        self.axes = None
        self.w = None
        self.selected_line = None
        self.lines = None
        self.blit_data = None
        self.mean_line = None
        super().__init__(state=state, parent=parent, include_matplotlib_toolbar=True)

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

        self.chk_show_neighbors = QCheckBox("Show Neighbor Channels", self)
        self.chk_show_neighbors.setChecked(False)
        self.chk_show_neighbors.setToolTip("Plot the unit's mean action potential waveform across nearby channels")
        self.chk_show_neighbors.toggled.connect(self.updateAll)

        if hasattr(self, "matplotlib_toolbar") and self.matplotlib_toolbar is not None:
            self.matplotlib_toolbar.addSeparator()
            self.matplotlib_toolbar.addWidget(self.chk_show_neighbors)

        self.channel_bar = ChannelGroupBar(parent=self, state=state)
        self.channel_bar.bindingChanged.connect(self._on_channel_bar_changed)
        self.addToolBar(Qt.TopToolBarArea, self.channel_bar)
        self.update_window_title()

        def draw_evt(evt):
            self.blit()
            self.update_curstim_line(self.state.stimno)
            self.canvas.update()

        self.canvas.mpl_connect("draw_event", draw_evt)

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
            self._do_updateAll_core()
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
            self._do_updateAll_core()

    def hideEvent(self, event):
        super().hideEvent(event)
        self._user_hidden = True

    def showEvent(self, event):
        super().showEvent(event)
        self._was_shown = True
        self._user_hidden = False
        if getattr(self, "_dirty", False):
            self._dirty = False
            self._do_updateAll_core()

    def _on_channel_bar_changed(self):
        self.update_window_title()
        self.updateAll()

    def update_window_title(self):
        if hasattr(self, "channel_bar"):
            summary = self.channel_bar.get_title_summary()
            self.setWindowTitle(f"UnitView {summary}")
            p = self.parentWidget()
            if p is not None and hasattr(p, "setWindowTitle"):
                p.setWindowTitle(f"UnitView {summary}")

    def blit(self):
        if self.axes is None or "main" not in self.axes:
            return
        self.blit_data = self.canvas.copy_from_bbox(self.axes["main"].bbox)

    def setState(self, state):
        super().setState(state)
        if hasattr(self, "channel_bar"):
            self.channel_bar.setState(state)
        self.state.onStimNoChange.connect(self.update_curstim_line)
        self.state.onUnitChange.connect(self.updateAll)
        self.state.onUnitGroupChange.connect(self.updateAll)
        self.state.onLoadNewFile.connect(self.updateAll)
        if hasattr(self.state, "onNewSpikeEvents"):
            self.state.onNewSpikeEvents.connect(lambda *args: self.updateAll())
        if hasattr(self.state, "onChannelChange"):
            self.state.onChannelChange.connect(self.updateAll)
        if hasattr(self.state, "onNewSweep"):
            self.state.onNewSweep.connect(lambda *args: self.schedule_update())
        if hasattr(self.state, "onUIRefreshRateChange") and hasattr(self, "render_timer"):
            self.render_timer.setInterval(self.state.get_ui_refresh_rate_ms())
            self.state.onUIRefreshRateChange.connect(self.render_timer.setInterval)

    def schedule_update(self):
        self._dirty = True

    @Slot()
    def updateAll(self):
        if self.is_view_culled():
            self._dirty = True
            return
        self._dirty = False
        self._do_updateAll_core()

    def _do_updateAll_core(self):
        if self.is_view_culled():
            self._dirty = True
            return
        self.render_count = getattr(self, "render_count", 0) + 1
        for ax in list(self.figure.axes):
            ax.remove()
        self.setup_figure()

    def setup_figure(self):
        self.axes = {"main": self.figure.add_subplot(111)}
        if hasattr(self, "channel_bar"):
            primary_name = self.channel_bar.get_primary_channel()
            sig_erp = self.state.get_channel_erp(primary_name)
            sg = self.channel_bar.get_target_unit()
        else:
            primary_name = getattr(self.state.analog_signal, "name", "")
            sig_erp = self.state.analog_signal_erp
            sg = self.state.getUnitGroup()

        if sig_erp is None or len(sig_erp) == 0:
            return
        if getattr(self.state, "sampling_rate", None) is None:
            return
        self.w = int(self.state.sampling_rate * 0.002)
        ax = self.axes["main"]
        xarr = np.arange(-self.w, +self.w)
        self.selected_line = ax.plot(
            xarr, np.zeros(len(xarr)), color="purple", zorder=10, animated=True
        )[0]

        ax.axvline(0, ls="--", color="black")

        if sg is None:
            return

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
            sr = (
                float(self.state.analog_signal.sampling_rate.rescale(pq.Hz).magnitude)
                if hasattr(self.state.analog_signal.sampling_rate, "magnitude")
                else float(self.state.analog_signal.sampling_rate)
            )

        if self.state.event_signal is not None and hasattr(sg, "get_latencies"):
            lats = sg.get_latencies(self.state.event_signal)
            sample_arr = []
            for lat in lats:
                if np.isnan(lat):
                    sample_arr.append(None)
                else:
                    sec = float(lat.rescale(pq.s).magnitude) if hasattr(lat, "rescale") else float(lat)
                    sample_arr.append((int(round(sec * sr)), 1))
        elif getattr(sg, "idx_arr", None) is not None:
            sample_arr = sg.idx_arr
        else:
            sample_arr = []

        idx_arr = sample_arr
        self.lines = {}
        ylims = [-0.1, 0.1]
        mean_arr = np.zeros((len(xarr)))
        aligned_centers = {}
        should_align = getattr(self, "chk_show_neighbors", None) and self.chk_show_neighbors.isChecked()
        search_r = max(int(0.00075 * sr), 3)

        for i, idx, d in zip(range(len(sig_erp)), idx_arr, sig_erp):
            if idx is None:
                continue
            c_idx = idx[0]
            if should_align:
                search_start = max(0, c_idx - search_r)
                search_end = min(len(d), c_idx + search_r + 1)
                if search_end > search_start:
                    c_aligned = search_start + int(
                        np.argmax(np.abs(d[search_start:search_end]))
                    )
                else:
                    c_aligned = c_idx
            else:
                c_aligned = c_idx
            aligned_centers[i] = c_aligned

            w = [-self.w, self.w]
            xmin = c_aligned - self.w
            if xmin < 0:
                w[0] = abs(xmin) + 1
                xmin = 0

            xmax = c_aligned + self.w
            if xmax > len(d):
                w[1] = abs(xmax - len(d))
                xmax = len(d)

            ydata = d[xmin:xmax]
            try:
                mean_arr[w[0] + self.w : w[1] + self.w] += ydata
                self.lines[i] = np.vstack((np.arange(w[0], w[1]), ydata)).T
            except:
                pass
            ylims[0] = min(ylims[0], np.min(ydata))
            ylims[1] = max(ylims[1], np.max(ydata))

        from matplotlib.collections import LineCollection

        lc = LineCollection(self.lines.values(), alpha=0.3, color="gray")
        denom = len(self.lines) if len(self.lines) > 0 else 1
        self.mean_line = ax.plot(
            np.arange(-self.w, self.w),
            mean_arr / denom,
            ls="--",
            color="black",
            zorder=10,
            label=primary_name or "Primary",
        )[0]
        ax.add_artist(lc)

        # Plot neighbor channels footprint if enabled
        show_neighbors = (getattr(self, "chk_show_neighbors", None) and self.chk_show_neighbors.isChecked()) or (hasattr(self, "channel_bar") and self.channel_bar.is_multichannel())
        if show_neighbors and sg is not None:
            if hasattr(self, "channel_bar"):
                selected_chans = self.channel_bar.get_selected_channels()
                neighbors = [c for c in selected_chans if c != primary_name]
            else:
                neighbors = getattr(sg, "neighbor_channels", []) or []

            if not neighbors and hasattr(self.state, "get_available_channels"):
                avail = [c for c in self.state.get_available_channels() if c != primary_name]
                neighbors = avail[:3]

            colors = ["#e41a1c", "#377eb8", "#4daf4a", "#ff7f00", "#984ea3"]
            has_neighbor_plotted = False

            for n_idx, ch_name in enumerate(neighbors):
                if ch_name == primary_name:
                    continue
                ch_erp = (
                    self.state.get_channel_erp(ch_name)
                    if hasattr(self.state, "get_channel_erp")
                    else None
                )
                if ch_erp is not None and len(ch_erp) > 0:
                    n_mean_arr = np.zeros(len(xarr))
                    n_count = 0
                    for s_i, idx in enumerate(idx_arr):
                        if idx is None or s_i >= len(ch_erp):
                            continue
                        d_ch = ch_erp[s_i]
                        if hasattr(d_ch, "magnitude"):
                            d_ch = d_ch.magnitude
                        c_aligned = aligned_centers.get(s_i, idx[0])
                        w = [-self.w, self.w]
                        xmin = c_aligned - self.w
                        if xmin < 0:
                            w[0] = abs(xmin) + 1
                            xmin = 0
                        xmax = c_aligned + self.w
                        if xmax > len(d_ch):
                            w[1] = abs(xmax - len(d_ch))
                            xmax = len(d_ch)
                        ydata_ch = d_ch[xmin:xmax]
                        try:
                            n_mean_arr[w[0] + self.w : w[1] + self.w] += ydata_ch
                            n_count += 1
                            ylims[0] = min(ylims[0], np.min(ydata_ch))
                            ylims[1] = max(ylims[1], np.max(ydata_ch))
                        except Exception:
                            pass

                    if n_count > 0:
                        c_color = colors[n_idx % len(colors)]
                        n_mean = n_mean_arr / n_count
                        peak_offset = int(np.argmax(np.abs(n_mean))) - self.w
                        lag_ms = float((peak_offset / sr) * 1000.0)
                        lbl = f"{ch_name} (Δt={lag_ms:+.2f}ms)"
                        ax.plot(
                            xarr,
                            n_mean,
                            ls="--",
                            color=c_color,
                            label=lbl,
                            zorder=9,
                        )
                        has_neighbor_plotted = True

            if has_neighbor_plotted:
                ax.legend(loc="upper right", fontsize="small")

        ax.set_ylim(ylims)

        self.canvas.draw_idle()

    @Slot()
    def update_curstim_line(self, stimno):
        if self.state is None or self.axes is None or "main" not in self.axes:
            return
        ug = self.state.getUnitGroup()
        if ug is None:
            return

        has_spike = False
        if self.state.event_signal is not None and stimno < len(self.state.event_signal) and hasattr(ug, "get_latencies"):
            cur_ev = self.state.event_signal[stimno]
            lat = ug.get_latencies(np.array([cur_ev.magnitude]) * cur_ev.units)[0]
            has_spike = not np.isnan(lat)
        elif getattr(ug, "idx_arr", None) is not None and stimno < len(ug.idx_arr):
            has_spike = (ug.idx_arr[stimno] is not None)

        if self.blit_data is not None:
            self.canvas.restore_region(self.blit_data)
        if self.lines and stimno in self.lines:
            grayline = self.lines[stimno]
            if stimno == self.state.stimno:
                self.selected_line.set_visible(True)
                self.selected_line.set_ydata(grayline[:, 1])
        else:
            if self.selected_line is not None:
                self.selected_line.set_visible(False)
        if has_spike and self.selected_line is not None:
            self.selected_line.set_visible(True)
            self.axes["main"].draw_artist(self.selected_line)
        self.canvas.update()

    def draw_figure(self):
        pass  # return super().draw_figure()

    def on_click(self, event):
        if event.inaxes != self.axes["main"]:
            return False
        if self.matplotlib_toolbar.mode != "":
            return False
        # find the closest line
        xdata = event.xdata
        ydata = event.ydata
        if xdata is None or ydata is None:
            return False
        xdata = int(xdata)
        ydata = int(ydata)
        # find the closest line
        closest_line = None
        closest_dist = 1000000
        for i, line in self.lines.items():
            # Calculate the distance from the point to the line segments
            for j in range(len(line) - 1):
                p1, p2 = line[j], line[j + 1]
                # Vector math to calculate the distance from a point to a line segment
                line_vec = p2 - p1
                point_vec = np.array([xdata, ydata]) - p1
                line_len = np.dot(line_vec, line_vec)
                if line_len == 0:  # Avoid division by zero for degenerate line segments
                    dist = np.linalg.norm(point_vec)
                else:
                    t = max(0, min(1, np.dot(point_vec, line_vec) / line_len))
                    projection = p1 + t * line_vec
                    dist = np.linalg.norm(np.array([xdata, ydata]) - projection)

                if dist < closest_dist:
                    closest_dist = dist
                    closest_line = i
        if closest_line is not None:
            self.state.setStimNo(closest_line)

        return super().on_click(event)

    def keyPressEvent(self, event: PySide6.QtGui.QKeyEvent) -> None:
        return self.parentWidget().keyPressEvent(event)
