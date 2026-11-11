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
    QLabel,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QStyle,
    QVBoxLayout,
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
        self.state = state
        self._syncing = False

        main_vbox = QVBoxLayout()
        header_layout = QHBoxLayout()
        self.lbl_active_unit = QLabel("Unit: unit_0")
        self.lbl_unit_channel = QLabel("Channel: CH0")
        header_layout.addWidget(self.lbl_active_unit)
        header_layout.addWidget(self.lbl_unit_channel)
        main_vbox.addLayout(header_layout)

        layout = QFormLayout()

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

        qbut_go = QPushButton("Track")
        layout.addRow(qbut_go)
        qbut_go.clicked.connect(self.trackUnit)

        # Batch & unit helper buttons
        self.qbut_apply_all = QPushButton("Apply Params to All Units")
        self.qbut_apply_all.clicked.connect(self.apply_params_to_all_units)
        layout.addRow(self.qbut_apply_all)

        btn_row = QHBoxLayout()
        self.qbut_enable_all = QPushButton("Enable All Units")
        self.qbut_enable_all.clicked.connect(self.enable_all_units)
        self.qbut_disable_all = QPushButton("Disable All Units")
        self.qbut_disable_all.clicked.connect(self.disable_all_units)
        btn_row.addWidget(self.qbut_enable_all)
        btn_row.addWidget(self.qbut_disable_all)
        layout.addRow(btn_row)

        self.qbut_track_all = QPushButton("Track All Enabled Units")
        self.qbut_track_all.clicked.connect(self.track_all_enabled_units)
        layout.addRow(self.qbut_track_all)

        main_vbox.addLayout(layout)
        w = QWidget()
        w.setLayout(main_vbox)
        self.setCentralWidget(w)

        # Connect control changes to update active unit
        self.qsb_window_size.valueChanged.connect(self._on_params_changed)
        self.qsb_threshold.valueChanged.connect(self._on_params_changed)
        self.qsb_max_skip.valueChanged.connect(self._on_params_changed)
        self.qcb_auto_track.toggled.connect(self._on_params_changed)

        if self.state is not None:
            if hasattr(self.state, "onUnitGroupChange"):
                self.state.onUnitGroupChange.connect(self.sync_from_active_unit)
            if hasattr(self.state, "onUnitChange"):
                self.state.onUnitChange.connect(self.sync_from_active_unit)
            if hasattr(self.state, "onNewSweep"):
                self.state.onNewSweep.connect(self.on_new_sweep)
            self.sync_from_active_unit()

    def sync_from_active_unit(self):
        if self.state is None:
            return
        sg = None
        try:
            sg = self.state.getUnitGroup()
        except Exception:
            pass

        if sg is None and hasattr(self.state, "spike_groups") and len(self.state.spike_groups) > 0:
            if 0 <= self.state.cur_spike_group < len(self.state.spike_groups):
                sg = self.state.spike_groups[self.state.cur_spike_group]

        if sg is None:
            return

        self._syncing = True
        self.qsb_window_size.blockSignals(True)
        self.qsb_threshold.blockSignals(True)
        self.qsb_max_skip.blockSignals(True)
        self.qcb_auto_track.blockSignals(True)

        name = getattr(sg.event, "name", "") if (getattr(sg, "event", None) is not None and getattr(sg.event, "name", None)) else f"unit_{self.state.cur_spike_group}"
        ch = getattr(sg, "tracking_channel", "") or getattr(sg, "channel", "") or "default"
        self.lbl_active_unit.setText(f"Unit: {name}")
        self.lbl_unit_channel.setText(f"Channel: {ch}")

        self.qsb_window_size.setValue(float(getattr(sg, "tracking_window_ms", 5.0)))
        self.qsb_threshold.setValue(float(getattr(sg, "tracking_threshold_mv", 0.01)))
        self.qsb_max_skip.setValue(int(getattr(sg, "tracking_max_skips", 1)))
        self.qcb_auto_track.setChecked(bool(getattr(sg, "auto_track", False)))

        self.qsb_window_size.blockSignals(False)
        self.qsb_threshold.blockSignals(False)
        self.qsb_max_skip.blockSignals(False)
        self.qcb_auto_track.blockSignals(False)
        self._syncing = False

    def _on_params_changed(self, *args):
        if self._syncing or self.state is None:
            return
        sg = None
        try:
            sg = self.state.getUnitGroup()
        except Exception:
            pass
        if sg is None and hasattr(self.state, "spike_groups") and 0 <= self.state.cur_spike_group < len(self.state.spike_groups):
            sg = self.state.spike_groups[self.state.cur_spike_group]
        if sg is None:
            return

        win_val = float(self.qsb_window_size.value())
        thresh_val = float(self.qsb_threshold.value())
        skip_val = int(self.qsb_max_skip.value())
        track_val = bool(self.qcb_auto_track.isChecked())

        sg.tracking_window_ms = win_val
        sg.tracking_threshold_mv = thresh_val
        sg.tracking_max_skips = skip_val
        sg.auto_track = track_val

        if hasattr(sg, "event") and hasattr(sg.event, "annotations"):
            sg.event.annotations["tracking_window_ms"] = win_val
            sg.event.annotations["tracking_threshold_mv"] = thresh_val
            sg.event.annotations["tracking_max_skips"] = skip_val
            sg.event.annotations["auto_track"] = track_val

        if hasattr(self.state, "onUnitGroupChange"):
            self.state.onUnitGroupChange.emit()

    def apply_params_to_all_units(self):
        if self.state is None or not hasattr(self.state, "spike_groups"):
            return
        win_val = float(self.qsb_window_size.value())
        thresh_val = float(self.qsb_threshold.value())
        skip_val = int(self.qsb_max_skip.value())
        for sg in self.state.spike_groups:
            sg.tracking_window_ms = win_val
            sg.tracking_threshold_mv = thresh_val
            sg.tracking_max_skips = skip_val
            if hasattr(sg, "event") and hasattr(sg.event, "annotations"):
                sg.event.annotations["tracking_window_ms"] = win_val
                sg.event.annotations["tracking_threshold_mv"] = thresh_val
                sg.event.annotations["tracking_max_skips"] = skip_val
        if hasattr(self.state, "onUnitGroupChange"):
            self.state.onUnitGroupChange.emit()

    def enable_all_units(self):
        if self.state is None or not hasattr(self.state, "spike_groups"):
            return
        for sg in self.state.spike_groups:
            sg.auto_track = True
            if hasattr(sg, "event") and hasattr(sg.event, "annotations"):
                sg.event.annotations["auto_track"] = True
        self.sync_from_active_unit()
        if hasattr(self.state, "onUnitGroupChange"):
            self.state.onUnitGroupChange.emit()

    def disable_all_units(self):
        if self.state is None or not hasattr(self.state, "spike_groups"):
            return
        for sg in self.state.spike_groups:
            sg.auto_track = False
            if hasattr(sg, "event") and hasattr(sg.event, "annotations"):
                sg.event.annotations["auto_track"] = False
        self.sync_from_active_unit()
        if hasattr(self.state, "onUnitGroupChange"):
            self.state.onUnitGroupChange.emit()

    def track_all_enabled_units(self):
        if self.state is None or self.state.event_signal is None:
            return
        n_sweeps = len(self.state.event_signal)
        for s in range(n_sweeps):
            self.state.track_stimulus_all_units(s)

    def on_new_sweep(self, stimno: int):
        if self.state is None:
            return
        self.state.track_stimulus_all_units(stimno)


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
