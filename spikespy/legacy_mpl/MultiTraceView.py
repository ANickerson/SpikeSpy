import itertools
import sys
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any
import matplotlib
import matplotlib.axes
from matplotlib.collections import PatchCollection
import matplotlib.style as mplstyle
import numpy as np
import PySide6
import quantities as pq
from matplotlib.backend_bases import MouseEvent
from matplotlib.backend_tools import ToolToggleBase
from matplotlib.backends.backend_qtagg import FigureCanvas, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.path import Path
from matplotlib.widgets import PolygonSelector
from neo import Event
from PySide6.QtCore import Qt, Signal, Slot

logger = logging.getLogger(__name__)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QHBoxLayout,
    QMainWindow,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QGroupBox,
    QRadioButton,
    QButtonGroup,
    QDialog,
    QPushButton,
    QFormLayout,
    QDoubleSpinBox,
    QToolBar,
    QLabel,
    QComboBox,
    QSpacerItem,
    QSizePolicy,
)
from PySide6 import QtCore
from PySide6.QtCore import QTimer
from ..NeoSettingsView import NeoSettingsView
from ..ViewerState import ViewerState, tracked_neuron_unit
from ..ChannelGroupBar import ChannelGroupBar

from .QMatplotlib import QMatplotlib
from ..live_orchestrator import LiveFrameConsumerMixin
from matplotlib.lines import Line2D
import time

# mplstyle.use("fast")

from matplotlib.axes import Axes
from ..helpers import qsignal_throttle_wrapper, qsignal_throttle, notify_view_selected
from matplotlib.image import AxesImage
from .CachedAxesImage import CachedAxesImage
from ..viewer_worker import get_task_manager, TaskType, TaskPriority


class falling_leaf_plotter:
    def __init__(self):
        self.ax_track_cmap = None
        self.ax_track_leaf = None
        self.percentiles = []
        self.cb1 = None
        self.cb2 = None
        self.mode = None
        self.partial = False
        self.hline = None

    def setup(self, ax: Axes, erp, sampling_rate=1000, mode="heatmap"):
        if erp is not None and hasattr(erp, "magnitude"):
            erp = erp.magnitude
        if erp is None or erp.size == 0 or erp.shape[0] == 0:
            self.percentiles = np.zeros(100)
        else:
            self.percentiles = (
                np.arange(0, 100) * 2 * float(np.sqrt(np.mean(erp**2)))
            )  # np.percentile(erp, np.arange(100))
        self.mode = mode

        # self.cb2 = ax.callbacks.connect("ylim_changed", self.setup)

        func_formatter = matplotlib.ticker.FuncFormatter(
            lambda x, pos: "{0:g}".format(1000 * x / sampling_rate)
        )
        ax.xaxis.set_major_formatter(func_formatter)
        loc = matplotlib.ticker.MultipleLocator(
            base=sampling_rate / 100
        )  # this locator puts ticks at regular intervals
        ax.xaxis.set_major_locator(loc)

    def plot_main(self, mode, ax: Axes, erp, partial=True, clim=None):
        self.mode = mode

        if erp is not None and hasattr(erp, "magnitude"):
            erp = erp.magnitude
        if erp is None or erp.size == 0 or erp.shape[0] == 0:
            return

        ylim = ax.get_ylim()
        ylim = [max(int(np.floor(ylim[0])), 0), int(np.ceil(ylim[1]))]
        xlim = ax.get_xlim()
        xlim = [max(int(np.floor(xlim[0])), 0), int(np.ceil(xlim[1]))]
        if partial:
            im_data = erp[ylim[0] : ylim[1], xlim[0] : xlim[1]]
        else:
            im_data = erp
            xlim = [0, erp.shape[1]]
            ylim = [0, erp.shape[0]]
        if hasattr(im_data, "magnitude"):
            im_data = im_data.magnitude
        if mode == "heatmap":
            if self.ax_track_leaf is not None:
                try:
                    self.ax_track_leaf.remove()
                except:
                    pass
                self.ax_track_leaf = None

            if self.ax_track_cmap is not None:
                try:
                    self.ax_track_cmap.remove()
                    self.ax_track_cmap = None
                except:
                    pass
            #    self.ax_track_cmap.remove()
            ax.set_autoscale_on(False)

            self.ax_track_cmap = CachedAxesImage(
                ax,
                cmap="gray_r",
                interpolation="antialiased",
                rolling_max=True,
            )

            self.ax_track_cmap.set_data(np.clip(im_data, 0, float(np.max(im_data)) if im_data.size else 1.0))
            self.ax_track_cmap.set_extent((xlim[0], xlim[1], ylim[1], ylim[0]))
            if clim is None:
                clim = (self.percentiles[40], self.percentiles[95])
            self.ax_track_cmap.set_clim(*clim)
            self.ax_track_cmap._scale_norm(None, None, None)
            ax.add_artist(self.ax_track_cmap)
            # self.ax_track_cmap.set_animated(True)
            # ax.draw_artist(self.ax_track_cmap)
            # self.ax_track_cmap.set_visible(False)
        elif mode == "lines":
            if self.ax_track_cmap is not None:
                self.ax_track_cmap.set_visible(False)

            p90 = (float(np.std(im_data)) if im_data.size else 1.0) * 10  # self.percentiles[95] * 4
            analog_signal_erp_norm = np.clip(
                im_data,
                -p90,
                p90,
            ) / (p90 * 2)
            if self.ax_track_leaf is not None:
                # return
                self.ax_track_leaf.remove()
                self.ax_track_leaf = None

            x = np.arange(analog_signal_erp_norm.shape[1]) + (xlim[0] if partial else 0)
            ys = (analog_signal_erp_norm * -1) + (
                np.arange(analog_signal_erp_norm.shape[0]) + (ylim[0] if partial else 0)
            )[:, np.newaxis]
            segs = [np.column_stack([x, y]) for y in ys]
            from matplotlib.collections import LineCollection

            self.ax_track_leaf = LineCollection(
                segs, array=x, linestyles="solid", color="gray"
            )

            ax.add_artist(self.ax_track_leaf)

            # self.ax_track_leaf = ax.plot(  # could increase performance to just plot lines in view. but then no blitting...
            #     (
            #         (analog_signal_erp_norm * -1)
            #         + (np.arange(analog_signal_erp_norm.shape[0]) + ylim[0])[
            #             :, np.newaxis
            #         ]
            #     ).T,
            #     color="gray",
            #     zorder=10,
            # )

    def highlight_stim(self, ax, stimNo, partial=True):
        if stimNo is None:
            return
        if self.mode == "lines":
            from matplotlib import lines

            ax_leaf_paths = self.ax_track_leaf.get_paths()

            to_color_idx = stimNo - (np.floor(ax.get_ylim()[0]) if partial else 0)
            new_colors = [
                "purple" if x == to_color_idx else "gray"
                for x in range(len(ax_leaf_paths))
            ]
            self.ax_track_leaf.set_colors(new_colors)
            # ax.draw_artist(self.ax_track_leaf)
            return self.ax_track_leaf

            # op: lines.Line2D = self.mainPlotter.ax_track_leaf[stimNo]

            # self.hline = lines.Line2D(
            #     *self.mainPlotter.ax_track_leaf[to_color_idx].get_data()
            # )
            # self.hline.update_from(op)
            # self.hline.set_color("purple")
        else:
            # self.hline.set_visible(True)
            if self.hline is not None:
                try:
                    self.hline.remove()
                except:
                    pass
            self.hline = ax.axhline(stimNo)
            # self.hline.set_animated(True)

            # ax.draw_artist(self.hline)
            return self.hline

    def plot_spikegroup(self, ax, sg, event_signal=None, sampling_rate=10000, **kwargs):
        kwargs = {"alpha": 0.5, **kwargs}

        if event_signal is not None and hasattr(sg, "get_latencies"):
            lats = sg.get_latencies(event_signal)
            valid = ~np.isnan(lats)
            sweeps = np.where(valid)[0]
            lat_vals = [(float(lats[i].rescale(pq.s).magnitude) if hasattr(lats[i], "rescale") else float(lats[i])) for i in sweeps]
            sr = (
                float(sampling_rate.rescale(pq.Hz).magnitude)
                if hasattr(sampling_rate, "magnitude")
                else float(sampling_rate)
            )
            samples = [int(round(lv * sr)) for lv in lat_vals]
            points = np.column_stack([samples, sweeps]) if len(samples) > 0 else np.empty((0, 2))
        elif getattr(sg, "idx_arr", None) is not None:
            points = np.array(
                [(x[0], i) for i, x in enumerate(sg.idx_arr) if x is not None]
            )
        else:
            points = np.empty((0, 2))

        if len(points) == 0:
            return
        scat = ax.scatter(points[:, 0], points[:, 1], s=4, **kwargs)
        # scat.set_animated(True)
        # ax.draw_artist(scat)
        return [scat]


class MultiTraceView(LiveFrameConsumerMixin, QMainWindow):
    right_ax_data = {}

    def __init__(
        self,
        parent: PySide6.QtWidgets.QWidget = None,
        state: ViewerState = None,
    ):

        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setFocusPolicy(Qt.ClickFocus)
        self.state: ViewerState = None
        self.ax_track_cmap = None
        self.ax_track_leaf = None
        self.points_spikegroup = None
        self.hline = None
        self.figcache = None
        self.references = []
        self.Patches = {}
        self._last_settings_segment = None
        self.render_count: int = 0
        self._dirty: bool = False
        self._user_hidden: bool = False
        self.messages = None
        self.legend = None
        self.scale_mode = "std"

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

        # Background task manager for heavy computations
        self._task_manager = get_task_manager()
        self._viewer_id = f"multitrace_{id(self)}"
        self._pending_render_task = None
        
        # throttle the update for upateAll to every 500ms when using the comboboxes.
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.updateAll)
        self.update_timer.setInterval(500)
        self.close_conn = self.destroyed.connect(lambda: self.closeEvent())

        xsize = 1024
        ysize = 480
        dpi = 80
        self.plotter = falling_leaf_plotter()

        self.fig = Figure(figsize=(xsize / dpi, ysize / dpi), dpi=dpi)
        self.fig.canvas.mpl_connect("button_press_event", self.view_clicked)

        self.mode = "heatmap"

        self.gs = self.fig.add_gridspec(1, 2, width_ratios=[3, 1])

        self.ax = self.fig.add_subplot(self.gs[0, 0])
        self.ax_right = []
        # self.ax_right_fig = self.fig.add_subfigure(self.gs[0,1])

        # self.fig.canvas.mpl_connect("button_press_event", self.view_clicked)
        # create widgets
        self.view = FigureCanvas(self.fig)
        # self.view.update()
        # self.ps = PolygonSelectorTool(self.fig)
        # self.ps.enable()
        self.toolbar = NavigationToolbar2QT(self.view, self)
        # act = self.toolbar.addAction("selector")

        self.lock_to_stim = False

        self.addToolBar(self.toolbar)
        layout = QVBoxLayout()
        self.lowerSpinBox = QDoubleSpinBox(self)
        self.lowerSpinBox.setDecimals(2)
        self.lowerSpinBox.setSingleStep(0.1)
        self.lowerSpinBox.setRange(0, 10)
        self.lowerSpinBox.valueChanged.connect(lambda x: self.update_timer.start())
        self.lowerSpinBox.setValue(0.5)
        self.lowerSpinBox.setMinimumWidth(100)

        self.upperSpinBox = QDoubleSpinBox(self)
        self.upperSpinBox.setRange(0, 10)
        self.upperSpinBox.setDecimals(2)
        self.upperSpinBox.setSingleStep(0.1)
        self.upperSpinBox.setValue(2)
        self.upperSpinBox.setMinimumWidth(100)

        self.lowerSpinBox.valueChanged.connect(self.upperSpinBox.setMinimum)
        self.upperSpinBox.valueChanged.connect(self.lowerSpinBox.setMaximum)
        self.upperSpinBox.valueChanged.connect(lambda x: self.update_timer.start())

        self.references = []
        self.lock_to_stimCheckBox = QCheckBox("lock")

        def set_lock(*args):
            self.lock_to_stim = self.lock_to_stimCheckBox.isChecked()
            self.update_ylim(self.state.stimno)

        self.lock_to_stimCheckBox.stateChanged.connect(set_lock)

        self.includeUnitCheckbox = QCheckBox("Tracked")
        self.includeUnitCheckbox.setChecked(True)
        self.includeUnitCheckbox.stateChanged.connect(lambda x: self.render())
        self.includeAllUnitsCheckBox = QCheckBox("Others")
        self.includeAllUnitsCheckBox.stateChanged.connect(lambda x: self.render())

        butgrp = QButtonGroup()

        linesRadio = QRadioButton("lines")
        heatmapRadio = QRadioButton("heatmap")
        unitOnlyRadio = QRadioButton("unit only")

        heatmapRadio.setChecked(True)
        butgrp.addButton(linesRadio)
        butgrp.addButton(heatmapRadio)
        butgrp.addButton(unitOnlyRadio)
        self.butgrp = butgrp

        def buttonToggled(id, checked):
            if heatmapRadio.isChecked():
                mode = "heatmap"
            elif linesRadio.isChecked():
                mode = "lines"
            elif unitOnlyRadio.isChecked():
                mode = "unitonly"
            if checked:
                self.mode = mode
                self.setup_figure()
            self.view.draw()

        butgrp.idToggled.connect(buttonToggled)

        self.polySelectButton = QPushButton("Selector")
        self.polySelectButton.clicked.connect(lambda *args: self.toggle_polySelector())
        self.deleterButton = QPushButton("Deleter")
        self.deleterButton.clicked.connect(lambda *args: self.toggle_deleter())

        self.settingsButton = QPushButton("Settings")
        self.settingsDialog = DialogSignalSelect()
        self.settingsButton.clicked.connect(lambda: self.settingsDialog.show())
        self.rightPlots = {}

        self.toggleMaxMeanButton = QPushButton("Show max", self)
        self.toggleMaxMeanButton.setCheckable(True)
        self.toggleMaxMeanButton.clicked.connect(lambda: self.toggleMaxMean())

        self.showMessages = QPushButton("Show messages", self)
        self.showMessages.setCheckable(True)
        self.showMessages.clicked.connect(lambda: self.toggleShowMessages())

        self.showLegend = QPushButton("Show legend", self)
        self.showLegend.setCheckable(True)
        self.showLegend.clicked.connect(lambda x: self.render())

        toolbar2 = QToolBar("Controls", self)
        toolbar2.addWidget(QLabel("Colour scale: "))
        self.scale_mode_combo = QComboBox(self)
        self.scale_mode_combo.addItems(["std", "mV"])
        self.scale_mode_combo.setCurrentText(getattr(self, "scale_mode", "std"))
        self.scale_mode_combo.currentTextChanged.connect(self._on_scale_mode_changed)
        toolbar2.addWidget(self.scale_mode_combo)
        toolbar2.addWidget(QLabel("Min"))
        toolbar2.addWidget(self.lowerSpinBox)
        toolbar2.addWidget(QLabel("Max"))
        toolbar2.addWidget(self.upperSpinBox)
        toolbar2.addSeparator()
        toolbar2.addWidget(QLabel("Units:"))
        toolbar2.addWidget(self.includeUnitCheckbox)
        toolbar2.addWidget(self.includeAllUnitsCheckBox)
        toolbar2.addSeparator()
        toolbar2.addWidget(QLabel("Mode:"))
        toolbar2.addWidget(self.lock_to_stimCheckBox)
        toolbar2.addWidget(linesRadio)
        toolbar2.addWidget(heatmapRadio)
        toolbar2.addWidget(unitOnlyRadio)
        toolbar2.addSeparator()

        toolbar2.addWidget(self.polySelectButton)
        toolbar2.addWidget(self.deleterButton)
        toolbar2.addWidget(self.settingsButton)
        toolbar2.addWidget(self.toggleMaxMeanButton)
        toolbar2.addWidget(self.showMessages)
        toolbar2.addWidget(self.showLegend)

        self.addToolBarBreak()
        self.addToolBar(Qt.TopToolBarArea, toolbar2)

        # Independent channel group and unit binding toolbar (hidden by default; controlled via ViewSettingsPanel or container)
        self.channel_bar = ChannelGroupBar(parent=self, state=state)
        self.channel_bar.bindingChanged.connect(self._on_channel_bar_changed)
        self.addToolBarBreak()
        self.addToolBar(Qt.TopToolBarArea, self.channel_bar)
        self.channel_bar.setVisible(False)

        # layout.addLayout(layout2)
        layout.addWidget(self.view)
        w = QWidget(self)
        w.setLayout(layout)
        self.setCentralWidget(w)

        self.percentiles = []

        if state is not None:
            self.set_state(state)
        self.setup_figure()
        self.pg_selector = LineSelector(
            self.ax,
            lambda *args: None,
            props=dict(color="purple", linestyle="none", linewidth=2, alpha=0.5),
            # useblit=True,
        )  # useblit does not work (for unknown reasons)
        self.pg_selector.set_active(False)
        self.update_axis()
        self.deleter = False
        # @qsignal_throttle_wrapper(interval=33)
        # def draw_evt(evt):
        #     with self.fig.canvas.callbacks.blocked():
        #         # self.blit()
        #         self.render()

        # self.fig.canvas.mpl_connect("draw_event", draw_evt)
        # self.ax.callbacks.connect("xlim_changed", draw_evt)
        # self.ax.callbacks.connect("ylim_changed", draw_evt)

        self.dialogPolySelect = DialogPolySelect(self)
        self.dialogPolySelect.onSubmit.connect(self.polySelect)
        self.dialogPolySelect.finished.connect(
            lambda *args: self.toggle_polySelector(False)
        )
        self.dialogPolySelect.changeSelection.connect(
            lambda x, y: self.pg_selector.set_w(
                int(self.state.analog_signal.sampling_rate * x / 1000)
            )
        )

        self.dialogDeleteRegion = DialogDeleteRegion(self)
        self.dialogDeleteRegion.onSubmit.connect(self.deleteRegion)
        self.dialogDeleteRegion.finished.connect(
            lambda *args: self.toggle_deleter(False)
        )

    def get_settings(self):
        s = {
            "xlim": self.ax.get_xlim(),
            "yrange": np.diff(self.ax.get_ylim()),
            "mode": self.mode,
            "include_all_units": self.includeAllUnitsCheckBox.isChecked(),
            "lock_to_stim": self.lock_to_stim,
            "percentiles": (self.lowerSpinBox.value(), self.upperSpinBox.value()),
            "scale_mode": getattr(self, "scale_mode", "std"),
        }
        if hasattr(self, "channel_bar"):
            s["channel_bar"] = self.channel_bar.get_settings()
        return s

    def _on_scale_mode_changed(self, mode: str):
        self.scale_mode = mode
        self.updateAll()

    def set_settings(self, values):
        if "channel_bar" in values and hasattr(self, "channel_bar"):
            self.channel_bar.set_settings(values["channel_bar"])
        if "xlim" in values:
            self.ax.set_xlim(values["xlim"])
        if "mode" in values:
            self.mode = values["mode"]
        if "scale_mode" in values:
            self.scale_mode = values["scale_mode"]
            if hasattr(self, "scale_mode_combo"):
                idx = self.scale_mode_combo.findText(self.scale_mode)
                if idx >= 0:
                    self.scale_mode_combo.blockSignals(True)
                    self.scale_mode_combo.setCurrentIndex(idx)
                    self.scale_mode_combo.blockSignals(False)
        if "include_all_units" in values:
            self.includeAllUnitsCheckBox.setChecked(values["include_all_units"])
        if "lock_to_stim" in values:
            self.lock_to_stim = values["lock_to_stim"]
            self.lock_to_stimCheckBox.setChecked(values["lock_to_stim"])
        if "percentiles" in values:
            self.lowerSpinBox.setValue(values["percentiles"][0])
            self.upperSpinBox.setValue(values["percentiles"][1])
        if "yrange" in values:
            # cur= self.ax.get_ylim()[0]
            cur = max(self.state.stimno - np.abs(values["yrange"]) // 2, 0)
            self.ax.set_ylim(cur + np.abs(values["yrange"]), cur)

        self.setup_figure()

    def polySelect(self):
        """
        create new event based on the lineSelector
        """
        # find all the interpolated spikes

        window_size, min_peak = self.dialogPolySelect.getValues()
        xys = np.array(self.pg_selector._xys[:-1])
        window_size = int(self.state.analog_signal.sampling_rate * window_size / 1000)

        target_unit_idx = self.channel_bar.get_target_unit_idx() if hasattr(self, "channel_bar") else self.state.cur_spike_group
        if hasattr(self, "channel_bar"):
            chans = self.channel_bar.get_selected_channels()
            if self.channel_bar.is_multichannel() and len(chans) > 1:
                erp = self.state.get_multichannel_erp(chans, combine_mode="mean")
            else:
                erp = self.state.get_channel_erp(self.channel_bar.get_primary_channel())
        else:
            erp = self.state.get_erp()
        if erp is not None and hasattr(erp, "magnitude"):
            erp = erp.magnitude
        out = []

        for i in range(xys.shape[0] - 1):
            seg = np.array([xys[i], xys[i + 1]], dtype=int)
            seg = seg[np.argsort(seg[:, 1])]

            stimpos = np.arange(seg[0, 1], seg[1, 1])
            arr = np.interp(stimpos, seg[:, 1], seg[:, 0])
            for x, stimno in zip(arr, stimpos):
                if stimno >= erp.shape[0]:  # if you go off the image
                    continue
                x2 = int(x - (window_size // 2))
                vals = erp[stimno, x2 : x2 + window_size]
                x2_offset = np.argmax((1 if min_peak >= 0 else -1) * (vals))
                if (1 if min_peak >= 0 else -1) * vals[x2_offset] > abs(
                    min_peak
                ):  # if min_peak is negative, must be negative
                    out.append((x2 + x2_offset, stimno, vals[x2_offset]))
        # in case our threshold is very high and it returns nothing
        if len(out) > 0:
            out = np.array(out)
            evt = self.state.stimno_offset_to_event(
                out[:, 1].astype(int), out[:, 0].astype(int)
            )
            self.state.updateUnit(evt, merge=True, unitid=target_unit_idx)
        self.pg_selector.clear()

    def toggle_polySelector(self, mode=None):
        self.pg_selector.set_active(mode or (not self.pg_selector.active))
        if self.pg_selector.active == 1:
            self.pg_selector.ax = self.ax
            if self.pg_selector._selection_artist not in self.ax.lines:
                self.ax.add_line(self.pg_selector._selection_artist)
            self.pg_selector.connect_default_events()
            self.pg_selector._selection_completed = False
            self.pg_selector.set_visible(True)
            self.dialogPolySelect.show()
            self.dialogPolySelect.change()
            # self.dialogPolySelect.activate()
        else:
            self.pg_selector._selection_completed = True
            self.pg_selector.clear()
            self.pg_selector._xys = [(0, 0)]  # TODO: move to clear
            self.dialogPolySelect.hide()
            self.pg_selector.set_visible(False)

    def deleteRegion(self):
        """
        Delete events
        """
        start_event, end_event = self.dialogDeleteRegion.getValues()

        target_unit_idx = self.channel_bar.get_target_unit_idx() if hasattr(self, "channel_bar") else self.state.cur_spike_group
        if hasattr(self, "channel_bar"):
            chans = self.channel_bar.get_selected_channels()
            if self.channel_bar.is_multichannel() and len(chans) > 1:
                erp = self.state.get_multichannel_erp(chans, combine_mode="mean")
            else:
                erp = self.state.get_channel_erp(self.channel_bar.get_primary_channel())
        else:
            erp = self.state.get_erp()

        for stimpos in range(start_event, end_event + 1):
            if erp is not None and stimpos >= len(erp):  # if you go off the image
                continue
            self.state.setUnit(None, stimpos, unitid=target_unit_idx)

    def toggle_deleter(self, mode=None):
        self.deleter = mode or (not self.deleter)
        if self.deleter == 1:
            self.dialogDeleteRegion.show()
            self.dialogDeleteRegion.change()

        else:
            self.dialogDeleteRegion.hide()

    def keyPressEvent(self, e):
        if e.key() == (Qt.Key_P):
            self.toggle_polySelector()
        if e.key() == (Qt.Key_Escape):
            self.pg_selector.clear()
        if (
            e.key() in (Qt.Key_Return, Qt.Key_Enter) and self.pg_selector.active
        ):  # action when pg_selector is done. #TODO: move elsewhere

            self.polySelect()

        self.update()

    def closeEvent(self, *args):
        if hasattr(self, "view") and self.view is not None:
            try:
                self.view._draw_pending = False
                QtCore.QCoreApplication.removePostedEvents(self.view)
            except Exception:
                pass
        self.remove_references()
        # Unregister from task manager
        if hasattr(self, "_task_manager"):
            self._task_manager.unregister_viewer(self._viewer_id)
        self.dialogPolySelect.close()
        self.settingsDialog.close()
        self.pg_selector.set_active(False)

    def remove_references(self, *args, **kwargs):
        self._unregister_live_frames()
        if self.state is not None:
            for ref in self.references:
                try:
                    self.state.disconnect(ref)
                except Exception:
                    pass
            self.references = []

    def set_state(self, state):
        self.state = state

        self.remove_references()

        if self.state is None:
            return

        # Register with background task manager
        self._task_manager.register_viewer(
            self._viewer_id,
            self.state,
            {
                TaskType.MULTI_TRACE_RENDER: self._on_multi_trace_result,
                TaskType.RIGHT_AXIS_UPDATE: self._on_right_axis_result,
                TaskType.SPIKE_GROUP_UPDATE: self._on_spike_group_result,
            }
        )

        # Static file loads are processed through the same orchestrator
        # pipeline as live sweeps (single point of view processing).
        self._register_data_reload(
            state,
            lambda: (
                self.reset_right_axes_data(),
                self.setup_figure(),
                self.update_axis(),
            ),
            name="reload_setup",
        )

        self.on_new_sweep_throttled = qsignal_throttle(
            lambda: self.on_new_sweep(self.state.stimno),
            40,
            parent=self,
            weight="heavy",
        )
        self.reset_right_axes_throttled = qsignal_throttle(
            self.reset_right_axes_data,
            250,
            parent=self,
            weight="heavy",
        )

        def _on_unit_updated(*args):
            if getattr(self.state, "is_streaming", False):
                self._dirty = True
            else:
                self.render()
                self._schedule_right_axis_update()

        self.references.append(self.state.onUnitGroupChange.connect(_on_unit_updated))
        self.references.append(self.state.onUnitChange.connect(_on_unit_updated))

        def _on_stimno_changed(*args):
            if getattr(self.state, "is_streaming", False):
                self._dirty = True
            else:
                self.update_ylim(self.state.stimno)
                self.plot_curstim_line(self.state.stimno)

        self.references.append(self.state.onStimNoChange.connect(_on_stimno_changed))

        if hasattr(self.state, "onChannelChange"):
            self.references.append(self.state.onChannelChange.connect(self._on_channel_bar_changed))

        if hasattr(self.state, "onNewSweep"):
            self._register_live_frames(state, lambda _stimno: self.on_new_sweep_throttled())
        if hasattr(self.state, "onNewSpikeEvents"):
            def _on_spike_events(*args):
                if getattr(self.state, "is_streaming", False):
                    self._dirty = True
                else:
                    self._schedule_spike_group_update()
            self.references.append(self.state.onNewSpikeEvents.connect(_on_spike_events))

        if hasattr(self.state, "onUIRefreshRateChange") and hasattr(self, "render_timer"):
            self.render_timer.setInterval(self.state.get_ui_refresh_rate_ms())
            self.references.append(self.state.onUIRefreshRateChange.connect(self.render_timer.setInterval))
        if hasattr(self, "channel_bar"):
            self.channel_bar.setState(state)

        self.update_axis()
        self._schedule_right_axis_update()
        self.update_window_title()

    def _on_channel_bar_changed(self):
        self.update_window_title()
        if getattr(self, "pg_selector", None) is not None:
            self.setup_figure()
            self.render()

    # ===== Background Task Scheduling =====
    
    def _schedule_render(self, stimno: int):
        """Schedule a render task for the background worker."""
        if self.is_view_culled() or self.state is None:
            if self.state is not None:
                self._dirty = True
            return
        
        if hasattr(self, "channel_bar"):
            chans = self.channel_bar.get_selected_channels()
            if self.channel_bar.is_multichannel() and len(chans) > 1:
                erp = self.state.get_multichannel_erp(chans, combine_mode="mean")
                primary = chans[0] if chans else None
            else:
                primary = self.channel_bar.get_primary_channel()
                erp = self.state.get_channel_erp(primary)
        else:
            primary = getattr(getattr(self.state, "analog_signal", None), "name", None)
            erp = self.state.get_erp()
        
        if erp is not None and hasattr(erp, "magnitude"):
            erp = erp.magnitude
        if erp is None or erp.shape[0] == 0:
            return
        
        # Submit to background thread (coalesces by viewer+type)
        self._task_manager.submit_render(
            self._viewer_id,
            TaskType.MULTI_TRACE_RENDER,
            {
                "erp_data": erp,
                "spike_groups": self.state.spike_groups,
                "event_signal": self.state.event_signal,
                "sampling_rate": self.state.sampling_rate,
                "mode": self.mode,
                "stimno": stimno,
                "lock_to_stim": self.lock_to_stim,
                "lower_percentile": self.lowerSpinBox.value(),
                "upper_percentile": self.upperSpinBox.value(),
                "follow_live": getattr(self.state, "follow_live", True),
                "channel_names": self._get_channel_names(),
                "viewer_ref": self,
                "current_unit_idx": self.channel_bar.get_target_unit_idx() if hasattr(self, "channel_bar") else self.state.cur_spike_group,
                "show_max": self.toggleMaxMeanButton.isChecked(),
            },
            priority=TaskPriority.HIGH,
        )
    
    def _get_channel_names(self) -> List[str]:
        """Get list of channel names for right axis computation."""
        names = []
        if hasattr(self, "channel_bar"):
            chans = self.channel_bar.get_selected_channels()
            for ch in chans:
                if ch and ch not in names:
                    names.append(ch)
        elif self.state and self.state.analog_signal is not None:
            sig_name = getattr(self.state.analog_signal, "name", None)
            if sig_name and sig_name not in names:
                names.append(sig_name)

        if self.state is not None:
            avail = self.state.get_available_channels() if hasattr(self.state, "get_available_channels") else []
            role_aux_names = set()
            if hasattr(self.state, "get_channel_roles"):
                try:
                    for role in self.state.get_channel_roles():
                        if not isinstance(role, dict):
                            continue
                        if not role.get("enabled", True):
                            continue
                        probe_type = str(role.get("probe_type", "AS")).upper()
                        purpose = str(role.get("purpose", "") or "")
                        r_name = str(role.get("name") or role.get("target_name") or "")
                        if r_name and probe_type == "AS" and purpose != "probe":
                            role_aux_names.add(r_name)
                except Exception:
                    pass

            for ch in avail:
                if not ch or ch in names:
                    continue
                if ch.startswith("env.") or ch.startswith("rec.") or ch in role_aux_names:
                    names.append(ch)

            if hasattr(self, "rightPlots") and isinstance(self.rightPlots, dict):
                avail_set = set(avail)
                for k, enabled in self.rightPlots.items():
                    if enabled and k in avail_set and k not in names:
                        names.append(k)
        return names
    
    def _schedule_right_axis_update(self):
        """Schedule right axis data computation."""
        if self.state is None or self.state.event_signal is None or len(self.state.event_signal) == 0:
            return
        if self.is_view_culled():
            self._dirty = True
            return
        
        erp = None
        if hasattr(self, "channel_bar"):
            chans = self.channel_bar.get_selected_channels()
            if self.channel_bar.is_multichannel() and len(chans) > 1:
                erp = self.state.get_multichannel_erp(chans, combine_mode="mean")
            else:
                primary = self.channel_bar.get_primary_channel()
                erp = self.state.get_channel_erp(primary)
        else:
            erp = self.state.get_erp()
        
        if erp is not None and hasattr(erp, "magnitude"):
            erp = erp.magnitude
        if erp is None or erp.size == 0:
            return
        
        self._task_manager.submit_render(
            self._viewer_id,
            TaskType.RIGHT_AXIS_UPDATE,
            {
                "erp_data": erp,
                "event_signal": self.state.event_signal,
                "spike_groups": self.state.spike_groups,
                "sampling_rate": self.state.sampling_rate,
                "viewer_ref": self,
                "channel_names": self._get_channel_names(),
                "show_max": self.toggleMaxMeanButton.isChecked(),
            },
            priority=TaskPriority.LOW,
        )
    
    def _schedule_spike_group_update(self):
        """Schedule spike group points computation."""
        if self.state is None or self.state.event_signal is None:
            return
        if self.is_view_culled():
            self._dirty = True
            return
        
        self._task_manager.submit_render(
            self._viewer_id,
            TaskType.SPIKE_GROUP_UPDATE,
            {
                "event_signal": self.state.event_signal,
                "spike_groups": self.state.spike_groups,
                "sampling_rate": self.state.sampling_rate,
                "current_unit_idx": self.channel_bar.get_target_unit_idx() if hasattr(self, "channel_bar") else self.state.cur_spike_group,
            },
            priority=TaskPriority.NORMAL,
        )
    
    # ===== Background Task Result Handlers =====
    
    def _on_multi_trace_result(self, result: Dict[str, Any], error: str):
        """Handle completed multi-trace render task."""
        if error:
            logger.error(f"MultiTrace render failed: {error}")
            return
        if not result:
            return
        
        # Apply heatmap update
        if self.mode == "heatmap" and self.plotter.ax_track_cmap is not None:
            clipped_data = result.get("clipped_data")
            extent = result.get("extent")
            clim = result.get("clim")
            if clipped_data is not None and extent is not None:
                if hasattr(self.plotter.ax_track_cmap, "update_data_fast"):
                    self.plotter.ax_track_cmap.update_data_fast(clipped_data, extent)
                else:
                    self.plotter.ax_track_cmap.set_data(clipped_data)
                    self.plotter.ax_track_cmap.set_extent(extent)
                if clim is not None:
                    self.plotter.ax_track_cmap.set_clim(*clim)
        
        # Update ylim
        ylim = result.get("ylim")
        if ylim is not None:
            self.ax.set_ylim(ylim)
        
        # Update xlim if needed
        xlim = result.get("xlim")
        if xlim is not None and abs(self.ax.get_xlim()[1] - self.ax.get_xlim()[0]) <= 1.0:
            self.ax.set_xlim(xlim)
            self.update_axis()
        
        # Highlight current stim
        stimno = result.get("stimno")
        if stimno is not None:
            self.plotter.highlight_stim(self.ax, stimno, partial=False)
        
        # Spike groups will be handled by separate result
        self.view.draw_idle()
    
    def _on_right_axis_result(self, result: Dict[str, Any], error: str):
        """Handle completed right axis computation."""
        if error:
            logger.error(f"Right axis update failed: {error}")
            return
        if not result:
            return
        
        self.right_ax_data = result
        
        # Update rightPlots dict
        for k in self.right_ax_data.keys():
            if k not in self.rightPlots:
                self.rightPlots[k] = True
        
        # Recreate settings dialog if needed
        cur_seg = getattr(self.state, "segment", None)
        if self.settingsDialog is None or self._last_settings_segment != cur_seg:
            if self.settingsDialog is not None:
                try:
                    self.settingsDialog.destroy()
                except Exception:
                    pass
            self.settingsDialog = DialogSignalSelect(options=self.rightPlots)
            self._last_settings_segment = cur_seg
            
            def updateView(k, v):
                self.rightPlots[k] = v
                self.plot_right_axis()
                self.render()
            
            self.settingsDialog.changeSelection.connect(updateView)
        
        self.plot_right_axis()
        self.view.draw_idle()
    
    def _on_spike_group_result(self, result: Dict[int, Dict], error: str):
        """Handle completed spike group computation."""
        if error:
            logger.error(f"Spike group update failed: {error}")
            return
        if not result:
            return
        
        # Update Patches with new data
        import matplotlib
        colors = matplotlib.colormaps["Set2"].colors
        
        target_unit_idx = self.channel_bar.get_target_unit_idx() if hasattr(self, "channel_bar") else self.state.cur_spike_group
        active_unit_indices = set()
        if self.includeAllUnitsCheckBox.isChecked():
            active_unit_indices.update(range(len(self.state.spike_groups)))
        if self.includeUnitCheckbox.isChecked() and target_unit_idx is not None:
            active_unit_indices.add(target_unit_idx)
        
        # Hide unused
        for u_idx, line in list(self.Patches.items()):
            if u_idx not in active_unit_indices or u_idx >= len(self.state.spike_groups):
                line.set_visible(False)
        
        legend = []
        for i, data in result.items():
            if i >= len(self.state.spike_groups):
                continue
            x_unit = self.state.spike_groups[i]
            samples = data["samples"]
            sweeps = data["sweeps"]
            
            is_cur = (i == target_unit_idx)
            c = "red" if is_cur else colors[i % len(colors)]
            alpha = (0.6 if self.mode == "heatmap" else 1.0) if is_cur else 0.8
            zorder = 12 if is_cur else 11
            
            if i not in self.Patches or self.Patches[i] not in self.ax.lines:
                line = Line2D(
                    samples,
                    sweeps,
                    marker="s",
                    linestyle="None",
                    markersize=4,
                    markerfacecolor="none",
                    markeredgecolor=c,
                    markeredgewidth=1.2,
                    alpha=alpha,
                    zorder=zorder,
                )
                self.ax.add_line(line)
                self.Patches[i] = line
            else:
                line = self.Patches[i]
                line.set_data(samples, sweeps)
                line.set_markeredgecolor(c)
                line.set_alpha(alpha)
                line.set_zorder(zorder)
                line.set_visible(True)
            
            if len(samples) > 0 and self.showLegend.isChecked():
                ss = x_unit.event.name or f"unit_{i}"
                tt = self.ax.text(samples[0], sweeps[0], ss)
                tt.set_color(c)
                tt.set_backgroundcolor("white")
                tt.set_fontweight("bold")
                legend.append(tt)
        
        self.legend = legend
        self.points_spikegroups = list(self.Patches.values())
        self.view.draw_idle()

    def update_window_title(self):
        if hasattr(self, "channel_bar"):
            summary = self.channel_bar.get_title_summary()
            self.setWindowTitle(f"MultiTrace {summary}")
            p = self.parentWidget()
            if p is not None and hasattr(p, "setWindowTitle"):
                p.setWindowTitle(f"MultiTrace {summary}")

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
            self.on_new_sweep(self.state.stimno if self.state else 0)
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
            self.on_new_sweep(self.state.stimno if self.state else 0)

    def hideEvent(self, event):
        super().hideEvent(event)
        self._user_hidden = True
        if hasattr(self, "view") and self.view is not None:
            try:
                self.view._draw_pending = False
                QtCore.QCoreApplication.removePostedEvents(self.view)
            except Exception:
                pass

    def showEvent(self, event):
        super().showEvent(event)
        self._was_shown = True
        self._user_hidden = False
        if getattr(self, "_dirty", False):
            self._dirty = False
            self.on_new_sweep(self.state.stimno if self.state else 0)

    def _sync_right_axis_from_live_buffers(self):
        """Populate right-axis data once from live row statistics.

        LiveERPBuffer already maintains row_mean/row_max, so this is O(sweeps)
        per channel and only runs on the first frame before async updates take over.
        """
        if self.state is None or self.state.event_signal is None:
            return
        if not getattr(self.state, "is_streaming", False):
            return

        data = {}
        try:
            stim_times = self.state.event_signal.times.rescale(pq.second).magnitude
            if len(stim_times) > 1:
                data["Stim. Freq. (Hz)"] = (
                    1.0 / np.diff(stim_times),
                    np.arange(1, len(stim_times)),
                )
        except Exception:
            pass

        chans = self._get_channel_names()
        if not chans and self.state.analog_signal is not None:
            chans = [self.state.analog_signal.name]

        buffers = getattr(self.state, "_live_aux_erp_buffers", {}) or {}
        for ch_name in chans:
            buf = buffers.get(ch_name)
            if buf is None:
                primary_buf = getattr(self.state, "_live_erp_buffer", None)
                if primary_buf is not None and ch_name == getattr(
                    self.state.analog_signal, "name", None
                ):
                    buf = primary_buf
            if buf is None or len(buf) == 0:
                continue
            values = buf.row_max if self.toggleMaxMeanButton.isChecked() else buf.row_mean
            values = np.asarray(getattr(values, "magnitude", values))
            data[ch_name] = (values, np.arange(len(values)))

        if not data:
            return
        self.right_ax_data = data
        for key in data:
            self.rightPlots.setdefault(key, True)
        self.plot_right_axis()

    def on_new_sweep(self, stimno: int):
        """Lightweight sweep handler - schedules heavy work to background thread."""
        if self.is_view_culled():
            self._dirty = True
            return
        from ..helpers import run_adaptive_render
        run_adaptive_render(self, lambda: self._on_new_sweep_core(stimno), weight="heavy")

    def _on_new_sweep_core(self, stimno: int):
        self.render_count = getattr(self, "render_count", 0) + 1
        if self.state is None:
            return

        # Get ERP data for std computation (lightweight)
        if hasattr(self, "channel_bar"):
            chans = self.channel_bar.get_selected_channels()
            if self.channel_bar.is_multichannel() and len(chans) > 1:
                erp = self.state.get_multichannel_erp(chans, combine_mode="mean")
                primary = chans[0] if chans else None
            else:
                primary = self.channel_bar.get_primary_channel()
                erp = self.state.get_channel_erp(primary)
        else:
            primary = getattr(getattr(self.state, "analog_signal", None), "name", None)
            erp = self.state.get_erp()

        if erp is not None and hasattr(erp, "magnitude"):
            erp = erp.magnitude
        if erp is None or erp.shape[0] == 0:
            return

        if not self.right_ax_data:
            self._sync_right_axis_from_live_buffers()

        # Update running std (lightweight)
        live_buf = getattr(self.state, "_live_aux_erp_buffers", {}).get(
            primary, getattr(self.state, "_live_erp_buffer", None)
        )
        has_buf_match = live_buf is not None and len(live_buf) == erp.shape[0]

        if not hasattr(self, "std") or self.std == 0 or self.std is None or erp.shape[0] <= 5:
            self.std = float(np.std(erp)) or 1.0
        else:
            if has_buf_match:
                last_std = float(live_buf.row_std[-1]) or 1.0
            else:
                last_std = float(np.std(erp[-1])) or 1.0
            self.std = 0.95 * float(self.std) + 0.05 * last_std

        # Check if user is interacting (lightweight)
        toolbar = getattr(self, "toolbar", None)
        if toolbar is None and hasattr(self, "view") and hasattr(self.view, "canvas"):
            toolbar = getattr(self.view.canvas, "toolbar", None)
        is_user_interacting = toolbar is not None and getattr(toolbar, "mode", "") != ""

        # Schedule heavy rendering to background thread
        self._schedule_render(stimno)

        # Lightweight UI updates that must stay on UI thread
        if not is_user_interacting:
            if abs(self.ax.get_xlim()[1] - self.ax.get_xlim()[0]) <= 1.0:
                self.ax.set_xlim(0, erp.shape[1])
                self.update_axis()

            if self.lock_to_stim:
                self.update_ylim(self.state.stimno)
            else:
                current_ylim = self.ax.get_ylim()
                if (
                    current_ylim == (0.0, 1.0)
                    or current_ylim[1] == 1.0
                    or (getattr(self.state, "follow_live", True) and abs(current_ylim[0] - erp.shape[0]) <= 5)
                ):
                    top_lim = current_ylim[1] if (current_ylim[1] != 1.0 and current_ylim != (0.0, 1.0)) else 0
                    self.ax.set_ylim(erp.shape[0], top_lim)

        self.plotter.highlight_stim(self.ax, self.state.stimno, partial=False)
        # plot_spikegroups and reset_right_axes_data now handled by background tasks
        self.view.draw_idle()

    # @qsignal_throttle_wrapper(1000)
    def reset_right_axes_data(self):
        """Schedule right axis data computation to background thread."""
        self._schedule_right_axis_update()
        self.view.draw_idle()

    def setup_figure(self):
        mode = self.mode
        if self.state is None:
            return
        if self.state.analog_signal is None or self.state.event_signal is None or len(self.state.event_signal) == 0:
            return

        # self.ax_right_fig.clear()
        if hasattr(self, "channel_bar"):
            chans = self.channel_bar.get_selected_channels()
            if self.channel_bar.is_multichannel() and len(chans) > 1:
                erp = self.state.get_multichannel_erp(chans, combine_mode="mean")
            else:
                primary = self.channel_bar.get_primary_channel()
                erp = self.state.get_channel_erp(primary)
        else:
            erp = self.state.get_erp()

        if erp is not None and hasattr(erp, "magnitude"):
            erp = erp.magnitude
        if erp is None or erp.shape[0] == 0:
            return
        # erp = np.clip(erp, 0, np.max(erp))
        self.std = float(np.std(erp)) or 1.0
        # self.percentiles = self.percentiles = np.linspace(0,10,101)*np.std((erp - np.mean(erp)))#np.percentile(erp, np.arange(100))#np.percentile(erp[erp>0], np.arange(101))
        ylim = self.ax.get_ylim()
        xlim = self.ax.get_xlim()
        if self.plotter is not None:
            try:
                self.plotter.ax_track_cmap.remove()
            except:
                pass
            try:
                self.plotter.ax_track_leaf.remove()
            except:
                pass
            try:
                self.plotter.hline.remove()
            except:
                pass

        self.plotter.setup(
            self.ax, erp, sampling_rate=self.state.sampling_rate, mode=mode
        )
        self.plotter.plot_main(
            mode=self.mode,
            ax=self.ax,
            erp=erp,
            partial=False,
            clim=(
                (self.lowerSpinBox.value() if getattr(self, "scale_mode", "std") == "mV" else self.std * self.lowerSpinBox.value()),
                (self.upperSpinBox.value() if getattr(self, "scale_mode", "std") == "mV" else self.std * self.upperSpinBox.value()),
            ),
        )
        xlim_init = abs(xlim[1] - xlim[0]) <= 1.0 or xlim == (0.0, 1.0)
        if xlim_init:
            self.ax.set_xlim(0, erp.shape[1])
        else:
            self.ax.set_xlim(xlim)

        ylim_init = abs(ylim[1] - ylim[0]) <= 1.0 or ylim == (0.0, 1.0) or ylim == (1.0, 0.0)
        if self.lock_to_stim:
            self.update_ylim(self.state.stimno)
        elif ylim_init:
            self.ax.set_ylim(erp.shape[0], 0)
        else:
            self.ax.set_ylim(ylim)

        self.ax.set_autoscale_on(False)
        self.update_axis()

        self.view.update()
        # self.blit()

        self.plotter.highlight_stim(self.ax, self.state.stimno)

        self.plot_spikegroups()
        # self.plot_curstim_line(self.state.stimno)

        self.plot_right_axis()

        self.fig.tight_layout()
        return [x for x in [self.ax_track_cmap, self.ax_track_leaf] if x is not None]
        # self.view.draw_idle()

    # @qsignal_throttle_wrapper(1000)
    def plot_right_axis(self):
        active_items = [(k, v) for k, v in self.right_ax_data.items() if self.rightPlots.get(k, True)]
        active_labels = [k for k, _ in active_items]

        # Fast path: reuse existing subplots and Line2D artists when active labels match
        if (
            getattr(self, "_right_axis_labels", None) == active_labels
            and len(self.ax_right) == len(active_items)
            and len(active_items) > 0
        ):
            for ax, (label, data) in zip(self.ax_right, active_items):
                x_d = data[0].magnitude if hasattr(data[0], "magnitude") else np.asarray(data[0])
                y_d = data[1].magnitude if hasattr(data[1], "magnitude") else np.asarray(data[1])
                if ax.lines:
                    ax.lines[0].set_data(x_d, y_d)
                    try:
                        finite_x = x_d[np.isfinite(x_d)]
                        if len(finite_x) > 0:
                            xmin, xmax = float(np.min(finite_x)), float(np.max(finite_x))
                            if xmin == xmax:
                                xmin -= 1.0
                                xmax += 1.0
                            pad = 0.05 * (xmax - xmin)
                            ax.set_xlim(xmin - pad, xmax + pad)
                    except Exception:
                        pass
            return

        for ax in self.ax_right:
            ax.remove()
        self.ax_right = []
        self._right_axis_labels = active_labels
        count_axes = len(active_items)

        gs00 = matplotlib.gridspec.GridSpecFromSubplotSpec(
            1, max(count_axes, 1), subplot_spec=self.gs[0, 1]
        )
        colorwheel = itertools.cycle(iter(["r", "g", "b", "orange", "purple", "green"]))
        for i, (label, data) in enumerate(active_items):
            self.ax_right.append(self.fig.add_subplot(gs00[0, i], sharey=self.ax))
            if i > 0:
                self.ax_right[i].tick_params(axis="y", labelleft=False)
            c = next(colorwheel)
            self.ax_right[i].plot(*data, label=label, color=c)
            self.ax_right[i].set_xlabel(label)
            self.ax_right[i].xaxis.label.set_color(c)
            self.ax_right[i].tick_params(axis="y", colors=c)

    def update_ylim(self, curStim):
        if self.lock_to_stim:
            cur_lims = self.ax.get_ylim()
            w = max(abs(cur_lims[1] - cur_lims[0]) // 2, 2)
            self.ax.set_ylim(curStim + w, curStim - w)
            self.view.draw_idle()

    def update_axis(self):  # TODO: plot these data using x as the time in millis.
        if self.state is None:
            return
        if self.state.analog_signal is None or self.state.event_signal is None or len(self.state.event_signal) == 0:
            return
        erp = self.state.analog_signal_erp
        if erp is None or len(erp) == 0 or erp.shape[0] == 0:
            return
        cur_xlim = self.ax.get_xlim()
        if abs(cur_xlim[1] - cur_xlim[0]) <= 1.0 or cur_xlim == (0.0, 1.0):
            self.ax.set_xlim(0, len(self.state.analog_signal_erp[0]))
        cur_ylim = self.ax.get_ylim()
        if abs(cur_ylim[1] - cur_ylim[0]) <= 1.0 or cur_ylim == (0.0, 1.0) or cur_ylim == (1.0, 0.0):
            self.ax.set_ylim(len(self.state.event_signal), 0)

        class CustomFormatter(matplotlib.ticker.Formatter):
            def __init__(self, ax, func):
                super().__init__()
                self.set_axis(ax)
                self.func = func

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
        sr = self.state.sampling_rate
        if sr is None and self.state.analog_signal is not None:
            sr = getattr(self.state.analog_signal, "sampling_rate", None)
        if sr is None:
            sr = 1000
        sr = float(sr.magnitude) if hasattr(sr, "magnitude") else float(sr)

        func = lambda x, pos: "{0:g}".format(1000 * x / sr)
        func_formatter = matplotlib.ticker.FuncFormatter(func)

        self.ax.xaxis.set_major_formatter(func_formatter)
        self.ax.xaxis.set_minor_formatter(CustomFormatter(self.ax, func))
        # loc = matplotlib.ticker.MaxNLocator(
        #     steps=[1, 5, 10],
        # )  # this locator puts ticks at regular intervals
        loc = matplotlib.ticker.MultipleLocator(sr / 10)
        self.ax.xaxis.set_major_locator(loc)
        loc = matplotlib.ticker.MultipleLocator(sr / 100)
        self.ax.xaxis.set_minor_locator(loc)

        self.ax.xaxis.set_major_formatter(func_formatter)

        # loc = matplotlib.ticker.MultipleLocator(
        #     base=self.state.sampling_rate / 100
        # )  # this locator puts ticks at regular intervals
        # self.ax.xaxis.set_major_locator(loc)
        # self.ax.set_xticks(
        #     np.arange(
        #         0, self.state.analog_signal_erp.shape[1], self.state.sampling_rate / 100
        #     )
        # )
        # self.ax.set_xticks(
        #     np.arange(
        #         0, self.state.analog_signal_erp.shape[1], self.state.sampling_rate / 1000
        #     ),
        #     minor=True,
        # )
        # self.ax.grid(True, which="both")

    def _sync_spike_group_points(self):
        """Build spike-group points directly for the first paint.

        Later updates still come from the background SPIKE_GROUP_UPDATE task;
        this only avoids an empty plot until the first async result arrives.
        """
        result = {}
        if self.state is None or self.state.event_signal is None:
            return result
        target_unit_idx = (
            self.channel_bar.get_target_unit_idx()
            if hasattr(self, "channel_bar")
            else self.state.cur_spike_group
        )
        for i, sg in enumerate(self.state.spike_groups):
            idx_arr = getattr(sg, "idx_arr", None)
            if not idx_arr:
                continue
            points = []
            for sweep, item in enumerate(idx_arr):
                if item is None:
                    continue
                sample = item[0] if isinstance(item, (tuple, list, np.ndarray)) else item
                points.append((sample, sweep))
            if not points:
                continue
            pts = np.asarray(points, dtype=float)
            result[i] = {
                "samples": pts[:, 0],
                "sweeps": pts[:, 1],
                "is_current": i == target_unit_idx,
            }
        return result

    def _recolor_spike_group_patches(self):
        """Keep patch colors/z-order in sync with the current target unit."""
        if not self.Patches:
            return
        import matplotlib
        colors = matplotlib.colormaps["Set2"].colors
        target_unit_idx = (
            self.channel_bar.get_target_unit_idx()
            if hasattr(self, "channel_bar")
            else self.state.cur_spike_group
        )
        for i, line in self.Patches.items():
            is_cur = i == target_unit_idx
            color = "red" if is_cur else colors[i % len(colors)]
            alpha = (0.6 if self.mode == "heatmap" else 1.0) if is_cur else 0.8
            line.set_markeredgecolor(color)
            line.set_alpha(alpha)
            line.set_zorder(12 if is_cur else 11)

    points_spikegroups = None

    def plot_spikegroups(self, sgidx=None):
        """Lightweight UI update for spike groups - heavy computation done in background.
        This method is kept for compatibility but the actual data comes from _on_spike_group_result."""
        # This is now a no-op - the background task handles computation
        # and _on_spike_group_result applies the results to the UI
        if self.state is None or self.state.event_signal is None or len(self.state.event_signal) == 0:
            for line in getattr(self, "Patches", {}).values():
                line.set_visible(False)
            return list(getattr(self, "Patches", {}).values())
        
        # Just ensure Patches dict exists
        if not hasattr(self, "Patches") or self.Patches is None:
            self.Patches = {}

        if not self.Patches:
            result = self._sync_spike_group_points()
            if result:
                self._on_spike_group_result(result, "")
        
        # Hide any unused unit artists (based on checkboxes)
        target_unit_idx = self.channel_bar.get_target_unit_idx() if hasattr(self, "channel_bar") else self.state.cur_spike_group
        active_unit_indices = set()
        if self.includeAllUnitsCheckBox.isChecked():
            active_unit_indices.update(range(len(self.state.spike_groups)))
        if self.includeUnitCheckbox.isChecked() and target_unit_idx is not None:
            active_unit_indices.add(target_unit_idx)
        
        for u_idx, line in list(self.Patches.items()):
            if u_idx not in active_unit_indices or u_idx >= len(self.state.spike_groups):
                line.set_visible(False)

        self._recolor_spike_group_patches()
        
        return list(self.Patches.values())

    # @qsignal_throttle_wrapper(interval=33)
    def render(self):
        if self.is_view_culled():
            self._dirty = True
            return
        self.render_count = getattr(self, "render_count", 0) + 1

        self.view.draw_idle()

        # self.fig.canvas.restore_region(self.blit_data)
        # o = self.plot_curstim_line(self.state.stimno)

        o2 = self.plot_spikegroups()

        if getattr(self, "pg_selector", None) and self.pg_selector.active:
            self.pg_selector.set_visible(True)

            # self.pg_selector._draw_polygon()

            # self.pg_selector.draw()

        self.view.update()
        # return  o2
        # try:
        #     self.ax.redraw_in_frame()
        # except:
        #     self.fig.canvas.draw()
        # self.view.update()

    def plot_curstim_line(self, stimNo=None, force=False):
        if stimNo is None:
            return
        if not force and getattr(self, "_last_curstim_line", None) == stimNo:
            return
        self._last_curstim_line = stimNo
        self.plotter.highlight_stim(self.ax, stimNo, partial=False)
        self.view.draw_idle()

    @Slot()
    def updateAll(self):
        if self.mode != "heatmap" or self.plotter.ax_track_cmap is None:
            pass
        else:
            if getattr(self, "scale_mode", "std") == "mV":
                vmin = self.lowerSpinBox.value()
                vmax = self.upperSpinBox.value()
            else:
                vmin = self.std * self.lowerSpinBox.value()
                vmax = self.std * self.upperSpinBox.value()
            self.plotter.ax_track_cmap.set_clim(
                vmin,
                max(vmin + 0.001, vmax),
            )
            # self.ax_track_cmap.axes.draw_artist(self.ax_track_cmap)
            self.view.draw_idle()

    def mousePressEvent(self, event):
        notify_view_selected(self)
        super().mousePressEvent(event)

    def view_clicked(self, e: MouseEvent):
        notify_view_selected(self)
        if self.toolbar.mode != "" or e.button != 1 or e.inaxes != self.ax or e.ydata is None:
            return

        try:
            if np.isnan(e.ydata):
                return
            stimno = int(round(e.ydata))
            if self.state and self.state.event_signal is not None:
                if 0 <= stimno < len(self.state.event_signal):
                    self.state.setStimNo(stimno)
        except Exception:
            pass

    def toggleMaxMean(self):
        if self.toggleMaxMeanButton.isChecked():
            self.toggleMaxMeanButton.setText("Show Mean")
        else:
            self.toggleMaxMeanButton.setText("Show Max")
        self.reset_right_axes_data()

    def toggleShowMessages(self):
        i = 0
        i = next(
            i for i, e in enumerate(self.state.segment.events) if "log.other" == e.name
        )

        if self.showMessages.isChecked():
            self.showMessages.setText("Hide messages")

        else:
            self.showMessages.setText("Show messages")
            # self.plotter.highlight_stim(self.ax, stimNo, partial=False)

        self.toggle_plot_messages(self.state.segment.events[i])
        self.view.update()
        self.view.draw_idle()
        self.view.draw()

    def toggle_plot_messages(self, events):
        if self.messages is not None:
            for txt in self.messages:
                txt.remove()
            self.messages = None
        else:
            messages = []
            for i, t in enumerate(events):
                yy = self.state.event_signal.times.searchsorted(t, "right") - 1
                xx = 0
                ss = events.array_annotations["message"][i]
                messages.append(self.ax.text(xx, yy, ss))

            self.messages = messages

    def __del__(self):
        if hasattr(self, "view") and self.view is not None:
            try:
                self.view._draw_pending = False
                QtCore.QCoreApplication.removePostedEvents(self.view)
            except Exception:
                pass
        self.remove_references()
        try:
            self.destroyed.disconnect(self.close_conn)
        except Exception:
            pass
        try:
            super().__del__()
        except Exception:
            pass


import matplotlib.patches


class LineSelector(PolygonSelector):
    # self.outerlines = None
    # def calculate_hits
    def __init__(self, *args, w=1000, **kwargs):
        self._selection_window = matplotlib.patches.Polygon(
            [(0, 0)], closed=False, edgecolor="purple", facecolor="none"
        )
        self.w = w

        super().__init__(*args, **kwargs)
        self.ax.add_patch(self._selection_window)

    def set_w(self, w):
        self.w = w
        self._draw_polygon()

    @property
    def artists(self):
        others = super().artists
        return others + (self._selection_window,)

    def set_visible(self, visible):
        return super().set_visible(visible)

    def clear(self):
        super().clear()
        self._selection_window.set_visible(False)
        self._selection_window.remove()
        self._selection_window = matplotlib.patches.Polygon(
            [(0, 0)], closed=False, edgecolor="purple", facecolor="none"
        )
        self.ax.add_patch(self._selection_window)

    def _draw_polygon_without_update(self):
        super()._draw_polygon_without_update()
        if len(self._xys) < 2:
            return

        # Create a polygon with width "w" around the line described by self._xys
        w = self.w  # You can adjust this width as needed
        xys = np.array(self._xys)
        if len(xys) < 2:
            return

        # #Calculate the perpendicular offsets for the polygon
        # dx = np.diff(xys[:, 0])
        # dy = np.diff(xys[:, 1])
        # length = np.sqrt(dx**2 + dy**2)
        # nx = -dy / length  # Normalized perpendicular vector (x-component)
        # ny = dx / length   # Normalized perpendicular vector (y-component)

        # Create the upper and lower edges of the polygon
        upper = xys + np.array([w / 2, 0])
        lower = xys - np.array([w / 2, 0])
        lower = lower[::-1]  # Reverse the lower edge to close the polygon

        # Combine the edges to form the polygon
        poly_points = np.vstack([upper, lower, upper[0]])
        if self._selection_window is None:
            self._selection_window = matplotlib.patches.Polygon(
                poly_points, closed=True, edgecolor="purple", alpha=0.5
            )

        self._selection_window.set_xy(poly_points)

    def _release(self, event):
        """Button release event handler."""
        # Release active tool handle.
        if self._active_handle_idx >= 0:
            if event.button == 3:
                self._remove_vertex(self._active_handle_idx)
                self._draw_polygon()
            self._active_handle_idx = -1

        # Place new vertex.
        elif (
            not self._selection_completed
            and "move_all" not in self._state
            and "move_vertex" not in self._state
        ):
            self._xys.insert(-1, (event.xdata, event.ydata))

    def _on_key_release(self, event):
        """Key release event handler."""
        # Add back the pending vertex if leaving the 'move_vertex' or
        # 'move_all' mode (by checking the released key)
        if not self._selection_completed and (
            event.key == self._state_modifier_keys.get("move_vertex")
            or event.key == self._state_modifier_keys.get("move_all")
        ):
            self._xys.append((event.xdata, event.ydata))
            self._draw_polygon()
        # Reset the polygon if the released key is the 'clear' key.
        elif event.key == self._state_modifier_keys.get("clear"):
            event = self._clean_event(event)
            self._xys = [(event.xdata, event.ydata)]
            self._selection_completed = False
            self._remove_box()
            self.set_visible(True)


class DialogSignalSelect(QDialog):

    changeSelection = Signal(str, bool)

    def __init__(self, parent=None, options={}):
        super().__init__(parent, QtCore.Qt.Tool)
        self.initUI(options)

    def initUI(self, options):
        self.vbox = QVBoxLayout()
        self.cboxes = []
        for k, v in options.items():
            op = QCheckBox(text=k)
            op.setChecked(v)
            op.stateChanged.connect(lambda x, k=k: self.changeSelection.emit(k, x > 0))

            self.vbox.addWidget(op)
            self.cboxes.append(op)

        self.setLayout(self.vbox)


class DialogPolySelect(QDialog):
    changeSelection = Signal(float, float)
    onSubmit = Signal(float, float)

    def __init__(self, parent=None, options={}):
        super().__init__(parent, QtCore.Qt.Tool)
        self.initUI(options)

    def initUI(self, options):
        self.vbox = QFormLayout()
        self.cboxes = []

        self.window_size_input = QSpinBox(self)
        self.window_size_input.setMinimum(0)
        self.window_size_input.setMaximum(200)
        self.window_size_input.setValue(10)
        self.window_size_input.setSuffix("ms")
        self.window_size_input.setBaseSize(100, 10)
        self.window_size_input.valueChanged.connect(lambda *args: self.change())
        self.vbox.addRow(self.tr("window size"), self.window_size_input)

        self.minimumThreshold = QDoubleSpinBox(self)
        self.minimumThreshold.valueChanged.connect(lambda *args: self.change())
        self.minimumThreshold.setBaseSize(100, 10)
        self.minimumThreshold.setDecimals(2)
        self.minimumThreshold.setMinimum(-999999)
        self.minimumThreshold.setMaximum(999999)
        self.vbox.addRow(self.tr("minimumThreshold"), self.minimumThreshold)

        self.goButton = QPushButton("track")
        self.goButton.clicked.connect(lambda: self.onSubmit.emit(*self.getValues()))
        self.vbox.addRow(self.goButton)

        self.setLayout(self.vbox)

    def change(self):
        self.changeSelection.emit(
            self.window_size_input.value(), self.minimumThreshold.value()
        )

    def getValues(self):
        return (self.window_size_input.value(), self.minimumThreshold.value())


class DialogDeleteRegion(QDialog):
    changeSelection = Signal(float, float)
    onSubmit = Signal(float, float)

    def __init__(self, parent=None, options={}):
        super().__init__(parent, QtCore.Qt.Tool)
        self.initUI(options)

    def initUI(self, options):
        self.vbox = QFormLayout()
        self.cboxes = []

        self.start_event_input = QSpinBox(self)
        self.start_event_input.setMinimum(0)
        self.start_event_input.setMaximum(999999)
        self.start_event_input.setValue(0)
        self.start_event_input.setBaseSize(100, 10)
        self.start_event_input.valueChanged.connect(lambda *args: self.change())
        self.vbox.addRow(self.tr("start event"), self.start_event_input)

        self.end_event_input = QSpinBox(self)
        self.end_event_input.setMinimum(0)
        self.end_event_input.setMaximum(999999)
        self.end_event_input.setValue(0)
        self.end_event_input.setBaseSize(100, 10)
        self.end_event_input.valueChanged.connect(lambda *args: self.change())
        self.vbox.addRow(self.tr("end event"), self.end_event_input)

        self.goButton = QPushButton("delete")
        self.goButton.clicked.connect(lambda: self.onSubmit.emit(*self.getValues()))
        self.vbox.addRow(self.goButton)

        self.setLayout(self.vbox)

    def change(self):
        self.changeSelection.emit(
            self.start_event_input.value(), self.end_event_input.value()
        )

    def getValues(self):
        return (self.start_event_input.value(), self.end_event_input.value())


if __name__ == "__main__":

    app = QApplication([])
    state = ViewerState()
    # view = DialogSignalSelect()
    state.loadFile(r"data/test2.h5")

    view = MultiTraceView(state=state)
    view.show()
    app.exec()
    sys.exit()
