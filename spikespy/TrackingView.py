import sys
from dataclasses import dataclass, field
from re import I
from typing import Any, List, Optional, Union

import matplotlib.style as mplstyle
import numpy as np
import PySide6
import quantities as pq
from matplotlib.path import Path
from matplotlib.widgets import PolygonSelector
from PySide6.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QStyle,
    QWidget,
)

from .basic_tracking import track_basic
from .ViewerState import ViewerState
from neo import Event

mplstyle.use("fast")


class TrackingView(QMainWindow):
    def __init__(
        self,
        parent: Optional[PySide6.QtWidgets.QWidget] = None,
        state: ViewerState = None,
    ) -> None:
        super().__init__(parent)
        layout = QFormLayout()
        self.state = state
        w = QWidget()
        w.setLayout(layout)
        self.setCentralWidget(w)

        qsb_window_size = QDoubleSpinBox(self)
        qsb_window_size.setMinimumWidth(100)
        qsb_window_size.setMaximum(100)
        qsb_window_size.setDecimals(1)
        qsb_window_size.setMinimum(0)
        qsb_window_size.setValue(5)
        qsb_window_size.setSingleStep(0.5)
        self.qsb_window_size = qsb_window_size

        layout.addRow(self.tr("window size (ms)"), qsb_window_size)

        qsb_threshold = QDoubleSpinBox(self)
        qsb_threshold.setMinimumWidth(100)
        qsb_threshold.setMaximum(1000000)
        qsb_threshold.setMinimum(-1000000)
        qsb_threshold.setDecimals(4)
        qsb_threshold.setValue(0.01)
        qsb_threshold.setSingleStep(0.01)

        self.qsb_threshold = qsb_threshold
        try:
            self.updateThresholdFromUnit()
        except:
            pass
        ql = QHBoxLayout()
        ql.addWidget(qsb_threshold)
        qbut_threshold_update = QPushButton(
            icon=self.style().standardIcon(QStyle.SP_FileDialogContentsView)
        )
        qbut_threshold_update.clicked.connect(self.updateThresholdFromUnit)
        ql.addWidget(qbut_threshold_update)

        layout.addRow(self.tr("Threshold (mV)"), ql)

        qsb_max_skip = QSpinBox(self)
        qsb_max_skip.setMaximum(10)
        qsb_max_skip.setMinimum(0)
        qsb_max_skip.setValue(1)
        self.qsb_max_skip = qsb_max_skip
        layout.addRow(self.tr("Maximum skips"), qsb_max_skip)

        qsb_track_limit = QSpinBox(self)
        qsb_track_limit.setMaximum(1000)
        qsb_track_limit.setMinimum(0)
        qsb_track_limit.setValue(0)
        self.qsb_track_limit = qsb_track_limit
        layout.addRow(self.tr("Limit tracking"), qsb_track_limit)

        self.qcb_auto_track = PySide6.QtWidgets.QCheckBox(self.tr("Live Auto-Track"), self)
        layout.addRow(self.qcb_auto_track)

        # qbut_threshold_update.setFixedHeight(20)
        # layout.addRow(qbut_threshold_update)
        qbut_go = QPushButton("Track")
        layout.addRow(qbut_go)

        qbut_go.clicked.connect(self.trackUnit)

        if self.state is not None and hasattr(self.state, "onNewSweep"):
            self.state.onNewSweep.connect(self.on_new_sweep)

    def on_new_sweep(self, stimno: int):
        if not self.qcb_auto_track.isChecked() or self.state is None:
            return
        sg = self.state.getUnitGroup()
        if sg is None or len(sg.event) == 0:
            return

        if self.state.event_signal is None or stimno >= len(self.state.event_signal):
            return

        # Find the most recent tracked spike in sg within self.qsb_max_skip.value() + 1 sweeps prior to stimno
        max_skips = self.qsb_max_skip.value() + 1
        prev_latency = None
        min_prev = max(stimno - 1 - max_skips, 0)
        if self.state.event_signal is not None and min_prev < stimno and hasattr(sg, "get_latencies"):
            search_events = self.state.event_signal[min_prev:stimno]
            lats = sg.get_latencies(search_events)
            for k in range(len(lats) - 1, -1, -1):
                if not np.isnan(lats[k]):
                    prev_latency = lats[k].rescale(pq.s)
                    break
        elif getattr(sg, "idx_arr", None) is not None:
            for prev_stim in range(stimno - 1, max(stimno - 1 - max_skips, -1), -1):
                if prev_stim < len(sg.idx_arr) and sg.idx_arr[prev_stim] is not None:
                    prev_sample = sg.idx_arr[prev_stim][0]
                    prev_latency = (prev_sample / self.state.sampling_rate) * pq.s
                    break

        if prev_latency is None:
            return

        window = self.qsb_window_size.value() * pq.ms
        threshold = self.qsb_threshold.value() * pq.mV
        stim_time = self.state.event_signal[stimno]
        raw_data = self.state.analog_signal

        max_ts = None
        # Check if basic_tracking has track_single_sweep
        from . import basic_tracking
        if hasattr(basic_tracking, "track_single_sweep"):
            max_ts = basic_tracking.track_single_sweep(raw_data, stim_time, prev_latency, threshold, window)
        else:
            flip = False
            thresh_val = threshold
            if threshold < 0:
                flip = True
                thresh_val = -threshold

            timeslice = raw_data.time_slice(
                stim_time + prev_latency - (window / 2),
                stim_time + prev_latency + (window / 2),
            )
            if len(timeslice) > 0:
                ts_data = timeslice * -1 if flip else timeslice
                max_idx = np.argmax(ts_data)
                max_val = ts_data[max_idx]
                if max_val >= thresh_val and thresh_val >= 0:
                    max_ts = timeslice.times[max_idx]

        if max_ts is not None:
            latency_samples = int(
                np.round(
                    ((max_ts - stim_time).rescale("s") * self.state.analog_signal.sampling_rate.rescale("Hz")).magnitude
                )
            )
            self.state.setUnit(latency_samples, stimno=stimno)

    def updateThresholdFromUnit(self):
        if self.state is None:
            return
        unit_events = self.state.getUnitGroup().event.rescale(pq.s)
        last_event = unit_events.searchsorted(
            (self.state.event_signal[self.state.stimno] + (0.5 * pq.s)).rescale(pq.s)
        )  # find the most recent event

        starting_time = unit_events[max(last_event - 1, 0)]
        dat = self.state.analog_signal.time_slice(
            starting_time, starting_time + 0.1 * pq.ms
        )
        threshold = (dat[0] * 0.8).rescale("mV")
        self.qsb_threshold.setValue(threshold)

    def trackUnit(self):
        unit_events = self.state.getUnitGroup().event.rescale(pq.s)
        last_event = unit_events.searchsorted(
            (self.state.event_signal[self.state.stimno] + (0.5 * pq.s)).rescale(pq.s)
        )  # find the most recent event

        starting_time = unit_events[max(last_event - 1, 0)]
        window = self.qsb_window_size.value() * pq.ms
        threshold = self.qsb_threshold.value() * pq.mV

        evt2 = track_basic(
            self.state.analog_signal,
            self.state.event_signal,
            starting_time=starting_time,
            window=window,
            threshold=threshold,
            max_skip=self.qsb_max_skip.value(),
            track_limit=self.qsb_track_limit.value()
        )

        self.state.updateUnit(
            event=evt2, merge=True
        )  # Perhaps create new units to merge later?


if __name__ == "__main__":
    app = QApplication([])
    view = TrackingView()
    view.show()
    app.exec()
    sys.exit()
