import sys
import numpy as np
from PySide6.QtWidgets import QApplication, QToolBar, QSpinBox
from PySide6.QtGui import QAction
from spikespy.QMatplotlib import QMatplotlib
import matplotlib
from matplotlib.ticker import Formatter
from matplotlib.collections import LineCollection
from spikespy.ViewerState import ViewerState
import matplotlib.style as mplstyle

mplstyle.use("fast")


class MultiTraceFixedView(QMatplotlib):
    def __init__(self, parent=None, state: ViewerState = None):
        self.connections = []
        super().__init__(parent=parent, state=state, include_matplotlib_toolbar=True)
        self.settings = {
            "n_lines_pre": 5,
            "n_lines_post": 5,
        }

    def setState(self, state: ViewerState):
        super().setState(state)
        self.unsetCons()
        con = self.state.onStimNoChange.connect(lambda x: self.update_figure())
        self.connections.append(con)
        if hasattr(self.state, "onNewSweep"):
            con2 = self.state.onNewSweep.connect(lambda *args: self.update_figure())
            self.connections.append(con2)
        if hasattr(self.state, "onNewSpikeEvents"):
            con3 = self.state.onNewSpikeEvents.connect(lambda *args: self.update_figure())
            self.connections.append(con3)
        self.set_scale_setting()

        self.plot_update_signal.emit(True)

    def set_scale_setting(self):
        if self.state is None:
            return
        if self.state.event_signal is None or len(self.state.event_signal) == 0:
            return
        if self.get_settings().get("scale") is None:
            erp = self.state.get_erp()
            if erp is None or len(erp) == 0 or erp.shape[0] == 0:
                return
            ylim = (np.std(np.abs(erp)) or 1.0) * 8
            self.settings["scale"] = np.round(1 / ylim, decimals=4)

    def onLoadNewFile(self):
        super().onLoadNewFile()

    def create_toolbar(self):
        toolbar = QToolBar("Settings")
        # self.addToolBar(toolbar)

        # Add a spinner with label "Scale"

        scale_widget = QWidget()
        scale_layout = QHBoxLayout(scale_widget)
        scale_layout.setContentsMargins(0, 0, 0, 0)

        scale_label = QLabel("Scale:")
        scale_spinner = QDoubleSpinBox()
        scale_spinner.setDecimals(5)
        scale_spinner.setRange(0.00001, 100.0)
        scale_spinner.setSingleStep(0.005)
        scale_spinner.setValue(self.get_settings().get("scale", 1))
        scale_spinner.valueChanged.connect(
            lambda value: self.set_settings({"scale": value}, update_spinners=False)
        )
        scale_spinner.setMinimumSize(100, 1)
        self.scale_spinner = scale_spinner

        scale_layout.addWidget(scale_label)
        scale_layout.addWidget(scale_spinner)
        pre_label = QLabel("Pre:")
        pre_spinner = QSpinBox()
        pre_spinner.setRange(0, 100)
        pre_spinner.setValue(self.get_settings().get("n_lines_pre", 5))
        pre_spinner.valueChanged.connect(
            lambda value: self.set_settings(
                {"n_lines_pre": value}, update_spinners=False
            )
        )
        pre_spinner.setMinimumSize(100, 1)
        self.pre_spinner = pre_spinner

        post_label = QLabel("Post:")
        post_spinner = QSpinBox()
        post_spinner.setRange(0, 100)
        post_spinner.setValue(self.get_settings().get("n_lines_post", 5))
        post_spinner.valueChanged.connect(
            lambda value: self.set_settings(
                {"n_lines_post": value}, update_spinners=False
            )
        )
        post_spinner.setMinimumSize(100, 1)
        self.post_spinner = post_spinner

        scale_layout.addWidget(pre_label)
        scale_layout.addWidget(pre_spinner)
        scale_layout.addWidget(post_label)
        scale_layout.addWidget(post_spinner)
        toolbar.addWidget(scale_widget)
        return toolbar

    def unsetCons(self):
        for con in self.connections:
            try:
                self.state.disconnect(con)
            except:
                pass
        self.connections = []

    def __del__(self):
        self.unsetCons()

    def set_settings(self, values, clear=False, update_spinners=True):
        super().set_settings(values, clear)
        if update_spinners:
            self.post_spinner.setValue(self.get_settings().get("n_lines_post", 5))
            self.pre_spinner.setValue(self.get_settings().get("n_lines_pre", 5))
            self.scale_spinner.setValue(self.get_settings().get("scale", 1))

    def get_settings(self):
        return self.settings

    def on_click(self, event):
        if self.matplotlib_toolbar.mode != "":
            return False
        if self.state is not None:
            stimno = self.state.stimno
            start = max(stimno - self.settings.get("n_lines_pre", 0), 0)
            pos = np.round(event.ydata)
            self.state.setStimNo(int(pos + start))
        return super().on_click(event)

    def setup_figure(self):
        n_lines = self.get_settings().get("n_lines_pre", 10)
        n_lines_post = self.get_settings().get("n_lines_post", 10)

        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        # Create a LineCollection
        self.lines = LineCollection([], colors="grey", zorder=1)
        self.ax.add_collection(self.lines)
        self.points = self.ax.scatter([], [], color="red")

        func_formatter = matplotlib.ticker.FuncFormatter(
            lambda x, pos: "{0:g}".format(1000 * x)
        )
        self.ax.xaxis.set_major_formatter(func_formatter)
        self.ax.xaxis.set_minor_formatter(CustomFormatter(self.ax))
        self.ax.set_ylim(n_lines + n_lines_post + 1, -1)
        self.ax.set_xlim(0, 0.5)
        self.ax.callbacks.connect("ylim_changed", self.on_limits_change)

    def draw_figure(self):
        if self.state is None:
            return False
        if self.state.event_signal is None or len(self.state.event_signal) == 0:
            return False
        erp = self.state.get_erp()
        if erp is None or len(erp) == 0 or erp.shape[0] == 0:
            return False
        n_lines_pre = self.get_settings().get("n_lines_pre", 10)
        n_lines_post = self.get_settings().get("n_lines_post", 10)
        scale = self.get_settings().get("scale", 1)
        stimno = self.state.stimno
        start = max(stimno - n_lines_pre, 0)
        stop = min(start + n_lines_pre + n_lines_post + 1, len(erp))
        dpts = erp[start:stop]

        segments = []
        colors = []
        for i in range(len(dpts)):
            x = np.arange(erp.shape[-1]) / self.state.sampling_rate
            y = (dpts[i] * scale * -1) + i
            segments.append(np.column_stack([x, y]))
            if i == stimno - start:
                colors.append("purple")
            else:
                colors.append("grey")
        x_ys = []
        for i, l in enumerate(
            self.state.spike_groups[self.state.cur_spike_group].get_latencies(
                self.state.event_signal[start:stop]
            )
        ):
            if np.isnan(l):
                continue
            l = l.rescale("s")
            sample_no = int(l * self.state.sampling_rate)
            if sample_no >= len(dpts[i]) or sample_no < 0:
                continue
            r = 20
            x = np.arange(sample_no - r, sample_no + r)
            y = ((dpts[i][x] * scale) * -1) + i
            segments.append(np.column_stack([x / self.state.sampling_rate, y]))
            colors.append("red")
            x_ys.append((float(l.magnitude), (dpts[i][sample_no] * scale * -1) + i))
        if self.points is not None:
            self.points.remove()
            self.points = None

        # print(x_ys)

        self.lines.set_segments(segments)
        self.lines.set_color(colors)
        if len(x_ys) > 0:
            self.points = self.ax.scatter(*zip(*x_ys), s=10, color="black", zorder=10)

        self.ax.set_yticks(np.arange(0, len(dpts)), np.arange(start, stop))
        self.ax.set_ylim(len(dpts), -1)

    def on_limits_change(self, event_ax):
        ylim = event_ax.get_ylim()
        n_lines_pre = self.get_settings().get("n_lines_pre", 10)
        n_lines_post = self.get_settings().get("n_lines_post", 10)

        if (ylim[0] != n_lines_post + n_lines_pre + 1) or (ylim[1] != -1):
            event_ax.set_ylim(n_lines_post + n_lines_pre + 1, -1)


class CustomFormatter(Formatter):
    def __init__(self, ax):
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


import sys
import time
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QLabel, QDoubleSpinBox, QHBoxLayout, QWidget

# Assuming MultiLinePlot and ViewerState are defined in the same module


class TestMultiLinePlot(QMainWindow):
    def __init__(self, state: ViewerState, rate=5000):
        super().__init__()
        self.viewer_state = state
        self.plot = []  # MultiLinePlot(self.viewer_state)
        self.setCentralWidget(self.plot)
        self.stimno = 200

        # Timer to update stimno every 5 seconds
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_stimno)
        self.timer.start(rate)  # 5000 ms = 5 seconds
        self.direction = 1

    def update_stimno(self):
        print("next stimno")
        self.stimno += self.direction
        if self.stimno == 0:
            self.direction = 1
        if self.stimno == len(self.viewer_state.event_signal):
            self.direction = -1

        self.viewer_state.setStimNo(self.stimno)


if __name__ == "__main__":
    state = ViewerState()
    # view = DialogSignalSelect()
    state.loadFile(r"data/test2.h5")
    app = QApplication(sys.argv)
    test_window = TestMultiLinePlot(state, rate=10)
    test_window.show()
    sys.exit(app.exec())
