import sys
from dataclasses import dataclass, field
from re import I
from typing import Any, Dict, List, Optional, Union

import numpy as np
import PySide6
import quantities as pq
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
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

from .basic_tracking import track_basic, track_sweep_waveform
from .dsp.detectors import DetectorParamSpec, SpikeDetectorRegistry
from .ViewerState import ViewerState
from .live_orchestrator import BatchingMode
from neo import Event

_CORE_PARAM_NAMES = {"window_ms", "window_size", "threshold", "max_skip", "track_limit"}


class TrackingView(QMainWindow):
    def __init__(
        self,
        parent: Optional[PySide6.QtWidgets.QWidget] = None,
        state: ViewerState = None,
    ) -> None:
        super().__init__(parent)
        self.state = state
        self._syncing = False
        self.extra_param_controls: Dict[str, QWidget] = {}

        main_vbox = QVBoxLayout()
        header_layout = QHBoxLayout()
        self.lbl_active_unit = QLabel("Unit: unit_0")
        self.lbl_unit_channel = QLabel("Channel: CH0")
        header_layout.addWidget(self.lbl_active_unit)
        header_layout.addWidget(self.lbl_unit_channel)
        main_vbox.addLayout(header_layout)

        layout = QFormLayout()
        self.form_layout = layout

        self.qcb_algorithm = QComboBox(self)
        self.refresh_algorithms()
        layout.addRow(self.tr("Algorithm"), self.qcb_algorithm)

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
        except Exception:
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

        # Dynamic container for algorithm-specific DetectorParamSpec parameters
        self.extra_params_widget = QWidget(self)
        self.extra_params_layout = QFormLayout(self.extra_params_widget)
        self.extra_params_layout.setContentsMargins(0, 0, 0, 0)
        layout.addRow(self.extra_params_widget)

        self.qcb_auto_track = QCheckBox(self.tr("Live Auto-Track"), self)
        layout.addRow(self.qcb_auto_track)

        qbut_go = QPushButton("Track")
        self.qbut_go = qbut_go
        self.qpb_track = qbut_go
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

        # Build initial dynamic parameter controls
        self._rebuild_extra_param_controls(self.get_selected_algorithm())

        # Connect control changes to update active unit
        self.qcb_algorithm.currentIndexChanged.connect(self.on_algorithm_changed)
        self.qsb_window_size.valueChanged.connect(self._on_params_changed)
        self.qsb_threshold.valueChanged.connect(self._on_params_changed)
        self.qsb_max_skip.valueChanged.connect(self._on_params_changed)
        self.qsb_track_limit.valueChanged.connect(self._on_params_changed)
        self.qcb_auto_track.toggled.connect(self._on_params_changed)

        if self.state is not None:
            if hasattr(self.state, "onUnitGroupChange"):
                self.state.onUnitGroupChange.connect(self.sync_from_active_unit)
            if hasattr(self.state, "onUnitChange"):
                self.state.onUnitChange.connect(self.sync_from_active_unit)
            orch = getattr(self.state, "live_orchestrator", None)
            if orch is not None:
                orch.register(
                    f"trackingview_{id(self)}",
                    "auto_track",
                    self._live_tracking_fn,
                    self._on_live_tracking_result,
                    batching=BatchingMode.BATCH,
                )
            self.sync_from_active_unit()

    def refresh_algorithms(self) -> None:
        """Populate algorithm combo box from SpikeDetectorRegistry."""
        current_algo = self.get_selected_algorithm()
        self.qcb_algorithm.blockSignals(True)
        self.qcb_algorithm.clear()
        for plugin in SpikeDetectorRegistry.list_detectors():
            self.qcb_algorithm.addItem(plugin.display_name, plugin.name)
        idx = self.qcb_algorithm.findData(current_algo)
        if idx >= 0:
            self.qcb_algorithm.setCurrentIndex(idx)
        self.qcb_algorithm.blockSignals(False)

    def get_selected_algorithm(self) -> str:
        if hasattr(self, "qcb_algorithm") and self.qcb_algorithm.count() > 0:
            data = self.qcb_algorithm.currentData()
            if data:
                return str(data)
        return "basic"

    def _rebuild_extra_param_controls(self, algorithm: str) -> None:
        while self.extra_params_layout.rowCount() > 0:
            self.extra_params_layout.removeRow(0)
        self.extra_param_controls.clear()

        detector = SpikeDetectorRegistry.get(algorithm)
        for spec in detector.param_schema():
            if spec.name in _CORE_PARAM_NAMES:
                continue
            ctrl = self._create_param_widget(spec)
            self.extra_param_controls[spec.name] = ctrl
            self.extra_params_layout.addRow(spec.label, ctrl)

        self.extra_params_widget.setVisible(len(self.extra_param_controls) > 0)

    def _create_param_widget(self, spec: DetectorParamSpec) -> QWidget:
        if spec.param_type == "bool":
            w = QCheckBox(self.extra_params_widget)
            w.setChecked(bool(spec.default))
            w.toggled.connect(self._on_params_changed)
            return w
        if spec.param_type == "choice":
            w = QComboBox(self.extra_params_widget)
            for ch in spec.choices or []:
                w.addItem(str(ch), ch)
            idx = w.findData(spec.default)
            if idx >= 0:
                w.setCurrentIndex(idx)
            w.currentIndexChanged.connect(self._on_params_changed)
            return w
        if spec.param_type == "int":
            w = QSpinBox(self.extra_params_widget)
            w.setRange(int(spec.min_val), int(spec.max_val))
            w.setSingleStep(max(1, int(spec.step)))
            w.setValue(int(spec.default))
            w.valueChanged.connect(self._on_params_changed)
            return w
        w = QDoubleSpinBox(self.extra_params_widget)
        w.setDecimals(int(spec.decimals))
        w.setRange(float(spec.min_val), float(spec.max_val))
        w.setSingleStep(float(spec.step))
        w.setValue(float(spec.default))
        w.valueChanged.connect(self._on_params_changed)
        return w

    def _read_extra_params(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for name, w in self.extra_param_controls.items():
            if isinstance(w, QCheckBox):
                out[name] = bool(w.isChecked())
            elif isinstance(w, QComboBox):
                out[name] = w.currentData() if w.currentData() is not None else w.currentText()
            elif isinstance(w, QSpinBox):
                out[name] = int(w.value())
            elif isinstance(w, QDoubleSpinBox):
                out[name] = float(w.value())
        return out

    def _write_extra_params(self, params: Dict[str, Any]) -> None:
        for name, w in self.extra_param_controls.items():
            if name not in params:
                continue
            val = params[name]
            w.blockSignals(True)
            try:
                if isinstance(w, QCheckBox):
                    w.setChecked(bool(val))
                elif isinstance(w, QComboBox):
                    idx = w.findData(val)
                    if idx < 0:
                        idx = w.findText(str(val))
                    if idx >= 0:
                        w.setCurrentIndex(idx)
                elif isinstance(w, QSpinBox):
                    w.setValue(int(val))
                elif isinstance(w, QDoubleSpinBox):
                    w.setValue(float(val))
            finally:
                w.blockSignals(False)

    def on_algorithm_changed(self, *args) -> None:
        if self._syncing:
            return
        algo = self.get_selected_algorithm()
        self._rebuild_extra_param_controls(algo)
        self._on_params_changed()

    def updateView(self):
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
        self.qcb_algorithm.blockSignals(True)
        self.qsb_window_size.blockSignals(True)
        self.qsb_threshold.blockSignals(True)
        self.qsb_max_skip.blockSignals(True)
        self.qsb_track_limit.blockSignals(True)
        self.qcb_auto_track.blockSignals(True)

        name = getattr(sg.event, "name", "") if (getattr(sg, "event", None) is not None and getattr(sg.event, "name", None)) else f"unit_{self.state.cur_spike_group}"
        ch = getattr(sg, "tracking_channel", "") or getattr(sg, "channel", "") or "default"
        self.lbl_active_unit.setText(f"Unit: {name}")
        self.lbl_unit_channel.setText(f"Channel: {ch}")

        algo = getattr(sg, "tracking_algorithm", "basic") or "basic"
        idx = self.qcb_algorithm.findData(algo)
        if idx < 0:
            self.refresh_algorithms()
            idx = self.qcb_algorithm.findData(algo)
        if idx >= 0 and self.qcb_algorithm.currentIndex() != idx:
            self.qcb_algorithm.setCurrentIndex(idx)
        self._rebuild_extra_param_controls(algo)
        self._write_extra_params(getattr(sg, "tracking_params", {}) or {})

        self.qsb_window_size.setValue(float(getattr(sg, "tracking_window_ms", 5.0)))
        self.qsb_threshold.setValue(float(getattr(sg, "tracking_threshold_mv", 0.01)))
        self.qsb_max_skip.setValue(int(getattr(sg, "tracking_max_skips", 1)))
        self.qsb_track_limit.setValue(int(getattr(sg, "track_limit", 0)))
        self.qcb_auto_track.setChecked(bool(getattr(sg, "auto_track", False)))

        self.qcb_algorithm.blockSignals(False)
        self.qsb_window_size.blockSignals(False)
        self.qsb_threshold.blockSignals(False)
        self.qsb_max_skip.blockSignals(False)
        self.qsb_track_limit.blockSignals(False)
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

        algo_val = self.get_selected_algorithm()
        win_val = float(self.qsb_window_size.value())
        thresh_val = float(self.qsb_threshold.value())
        skip_val = int(self.qsb_max_skip.value())
        limit_val = int(self.qsb_track_limit.value())
        track_val = bool(self.qcb_auto_track.isChecked())
        extra_params = self._read_extra_params()

        sg.tracking_algorithm = algo_val
        sg.tracking_window_ms = win_val
        sg.tracking_threshold_mv = thresh_val
        sg.tracking_max_skips = skip_val
        sg.track_limit = limit_val
        sg.auto_track = track_val
        if not hasattr(sg, "tracking_params") or sg.tracking_params is None:
            sg.tracking_params = {}
        sg.tracking_params.update(extra_params)

        if hasattr(sg, "event") and hasattr(sg.event, "annotations"):
            sg.event.annotations["tracking_algorithm"] = algo_val
            sg.event.annotations["tracking_window_ms"] = win_val
            sg.event.annotations["tracking_threshold_mv"] = thresh_val
            sg.event.annotations["tracking_max_skips"] = skip_val
            sg.event.annotations["track_limit"] = limit_val
            sg.event.annotations["auto_track"] = track_val
            sg.event.annotations["tracking_params"] = dict(sg.tracking_params)

        if hasattr(self.state, "onUnitGroupChange"):
            self.state.onUnitGroupChange.emit()

    def apply_params_to_all_units(self):
        if self.state is None or not hasattr(self.state, "spike_groups"):
            return
        algo_val = self.get_selected_algorithm()
        win_val = float(self.qsb_window_size.value())
        thresh_val = float(self.qsb_threshold.value())
        skip_val = int(self.qsb_max_skip.value())
        extra_params = self._read_extra_params()
        for sg in self.state.spike_groups:
            sg.tracking_algorithm = algo_val
            sg.tracking_window_ms = win_val
            sg.tracking_threshold_mv = thresh_val
            sg.tracking_max_skips = skip_val
            if not hasattr(sg, "tracking_params") or sg.tracking_params is None:
                sg.tracking_params = {}
            sg.tracking_params.update(extra_params)
            if hasattr(sg, "event") and hasattr(sg.event, "annotations"):
                sg.event.annotations["tracking_algorithm"] = algo_val
                sg.event.annotations["tracking_window_ms"] = win_val
                sg.event.annotations["tracking_threshold_mv"] = thresh_val
                sg.event.annotations["tracking_max_skips"] = skip_val
                sg.event.annotations["tracking_params"] = dict(sg.tracking_params)
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

    def _live_tracking_fn(self, batch):
        """Runs on the orchestrator worker thread for a batch of new sweeps."""
        state = self.state
        detections = []
        if state is None or state.event_signal is None or not batch.items:
            return detections

        sr = float(
            state.sampling_rate
            if state.sampling_rate is not None
            else 30000.0
        )
        window_size = state.window_size

        units = [
            (i, sg)
            for i, sg in enumerate(state.spike_groups)
            if getattr(sg, "auto_track", False)
        ]
        if not units:
            return detections

        for item in batch.items:
            stimno = item.stim_index
            if stimno < 0 or stimno >= len(state.event_signal):
                continue
            for unit_idx, sg in units:
                if sg.get_sweep_sample(stimno, state.event_signal, sr, window_size) is not None:
                    continue
                ch = (
                    getattr(sg, "tracking_channel", "")
                    or getattr(sg, "channel", "")
                )
                row = item.erp_row(ch) if ch else None
                if row is None or row.size == 0:
                    continue
                sig_units = getattr(getattr(state, "analog_signal", None), "units", None)
                if sig_units is not None:
                    row = row * sig_units

                prev_latency = None
                max_skips = getattr(sg, "tracking_max_skips", 1)
                min_stim = max(0, stimno - 1 - max_skips)
                for prev_s in range(stimno - 1, min_stim - 1, -1):
                    prev_sample = sg.get_sweep_sample(
                        prev_s, state.event_signal, sr, window_size
                    )
                    if prev_sample is not None:
                        prev_latency = (prev_sample / sr) * pq.s
                        break
                    idx_arr = getattr(sg, "idx_arr", None)
                    if idx_arr is not None and prev_s < len(idx_arr) and idx_arr[prev_s] is not None:
                        prev_latency = (idx_arr[prev_s][0] / sr) * pq.s
                        break
                if prev_latency is None:
                    continue

                algo = getattr(sg, "tracking_algorithm", "basic") or "basic"
                unit_params = getattr(sg, "tracking_params", {}) or {}
                detected = track_sweep_waveform(
                    row,
                    sr,
                    prev_latency,
                    threshold=getattr(sg, "tracking_threshold_mv", 0.01),
                    window_ms=getattr(sg, "tracking_window_ms", 5.0),
                    algorithm=algo,
                    params=unit_params,
                    template_matcher=getattr(sg, "_template_matcher", None),
                )
                if detected is not None:
                    detections.append((unit_idx, int(detected), int(stimno)))

        return detections

    def _on_live_tracking_result(self, detections, error):
        """Applies orchestrator detections on the GUI thread."""
        if error or not detections or self.state is None:
            return
        for unit_idx, detected_sample, stimno in detections:
            self.state.setUnit(detected_sample, stimno=stimno, unit_idx=unit_idx, save_undo=False)

    def closeEvent(self, event):
        state = getattr(self, "state", None)
        if state is not None:
            orch = getattr(state, "live_orchestrator", None)
            if orch is not None:
                orch.unregister(f"trackingview_{id(self)}")
        super().closeEvent(event)

    def updateThresholdFromUnit(self):
        if self.state is None:
            return
        unit_events = self.state.getUnitGroup().event.rescale(pq.s)
        last_event = unit_events.searchsorted(
            (self.state.event_signal[self.state.stimno] + (0.5 * pq.s)).rescale(pq.s)
        )

        starting_time = unit_events[max(last_event - 1, 0)]
        dat = self.state.analog_signal.time_slice(
            starting_time, starting_time + 0.1 * pq.ms
        )
        threshold = float(np.asarray((dat[0] * 0.8).rescale("mV").magnitude).flat[0])
        self.qsb_threshold.setValue(threshold)

    def trackUnit(self):
        sg = self.state.getUnitGroup()
        unit_events = sg.event.rescale(pq.s)
        last_event = unit_events.searchsorted(
            (self.state.event_signal[self.state.stimno] + (0.5 * pq.s)).rescale(pq.s)
        )

        starting_time = unit_events[max(last_event - 1, 0)]
        window = self.qsb_window_size.value() * pq.ms
        threshold = self.qsb_threshold.value() * pq.mV
        algo = self.get_selected_algorithm()
        extra_params = self._read_extra_params()

        evt2 = track_basic(
            self.state.analog_signal,
            self.state.event_signal,
            starting_time=starting_time,
            window=window,
            threshold=threshold,
            max_skip=self.qsb_max_skip.value(),
            track_limit=self.qsb_track_limit.value(),
            algorithm=algo,
            params=extra_params,
        )

        self.state.updateUnit(
            event=evt2, merge=True
        )


if __name__ == "__main__":
    app = QApplication([])
    view = TrackingView()
    view.show()
    app.exec()
    sys.exit()
