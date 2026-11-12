"""Open Ephys Live View Dockable Widget for SpikeSpy.

Provides GUI controls to connect to Open Ephys v1.0+, manage streaming,
filter parameters, trigger sources, closed-loop bidirectional control,
and live stream diagnostics.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import quantities as pq
from neo import AnalogSignal, Event, Segment
from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from spikespy.ViewerState import ViewerState, tracked_neuron_unit

from .oe_controller import OpenEphysController
from .oe_stream_worker import OpenEphysStreamWorker

logger = logging.getLogger(__name__)


class OpenEphysLiveView(QWidget):
    """Dockable Open Ephys Live View Widget.

    Embeds into SpikeSpy's docking manager (MdiView.window_options).
    """

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        state: Optional[ViewerState] = None,
    ):
        super().__init__(parent)
        self.state: ViewerState = state or ViewerState()
        self.worker: Optional[OpenEphysStreamWorker] = None
        self.controller: Optional[OpenEphysController] = None
        self._demo_server_process = None
        self._is_streaming = False

        self._init_ui()
        self._init_controller()
        self._connect_state_signals()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(10)

        # -------------------------------------------------------------
        # 1. Connection Group
        # -------------------------------------------------------------
        conn_group = QGroupBox("Open Ephys Connection (v1.0+)")
        conn_layout = QFormLayout(conn_group)

        self.edit_host = QLineEdit("127.0.0.1")
        self.spin_data_port = QSpinBox()
        self.spin_data_port.setRange(1024, 65535)
        self.spin_data_port.setValue(5556)

        self.spin_event_port = QSpinBox()
        self.spin_event_port.setRange(1024, 65535)
        self.spin_event_port.setValue(5557)

        self.spin_http_port = QSpinBox()
        self.spin_http_port.setRange(1024, 65535)
        self.spin_http_port.setValue(37497)

        self.combo_stream_mode = QComboBox()
        self.combo_stream_mode.addItems(
            [
                "Standard OE ZMQ + Event Broadcaster",
                "Custom SpikeSpy C++ Plugin (Pre-Sliced)",
            ]
        )

        conn_layout.addRow("Host:", self.edit_host)
        conn_layout.addRow("ZMQ Data Port:", self.spin_data_port)
        conn_layout.addRow("ZMQ Event / Heartbeat Port:", self.spin_event_port)
        conn_layout.addRow("OE HTTP API Port:", self.spin_http_port)
        conn_layout.addRow("Stream Mode:", self.combo_stream_mode)

        self.btn_toggle_stream = QPushButton("Start Live Stream")
        self.btn_toggle_stream.setStyleSheet("font-weight: bold; padding: 6px;")
        self.btn_toggle_stream.clicked.connect(self.toggle_stream)
        conn_layout.addRow(self.btn_toggle_stream)

        main_layout.addWidget(conn_group)

        # -------------------------------------------------------------
        # 2. Channel & Filter Group
        # -------------------------------------------------------------
        chan_group = QGroupBox("Channel & Online Slicing")
        chan_layout = QFormLayout(chan_group)

        self.spin_rec_chan = QSpinBox()
        self.spin_rec_chan.setRange(0, 512)
        self.spin_rec_chan.setValue(0)
        self.spin_rec_chan.valueChanged.connect(self._on_rec_channel_changed)

        self.edit_active_channels = QLineEdit("0-3")
        self.edit_active_channels.setToolTip(
            "Comma-separated channels or ranges to buffer simultaneously, e.g., 0-3, 5, 7"
        )
        self.edit_active_channels.editingFinished.connect(self._on_active_channels_edited)

        self.combo_trigger_source = QComboBox()
        self.combo_trigger_source.addItems(["Digital TTL Event", "Analog ADC Channel"])

        self.spin_adc_trigger_chan = QSpinBox()
        self.spin_adc_trigger_chan.setRange(0, 64)
        self.spin_adc_trigger_chan.setValue(1)

        self.spin_window_ms = QDoubleSpinBox()
        self.spin_window_ms.setRange(10.0, 5000.0)
        self.spin_window_ms.setValue(500.0)
        self.spin_window_ms.setSuffix(" ms")

        self.spin_pre_ms = QDoubleSpinBox()
        self.spin_pre_ms.setRange(0.0, 500.0)
        self.spin_pre_ms.setValue(0.0)
        self.spin_pre_ms.setSuffix(" ms")

        self.chk_online_filter = QCheckBox("Online Bandpass (500 – 7000 Hz)")
        self.chk_online_filter.setChecked(True)
        self.chk_online_filter.toggled.connect(self._on_filter_toggled)

        self.chk_follow_live = QCheckBox("Follow Live Sweep (Auto-advance view)")
        self.chk_follow_live.setChecked(True)

        chan_layout.addRow("Recording Channel:", self.spin_rec_chan)
        chan_layout.addRow("Multi-Channel Active Set:", self.edit_active_channels)
        chan_layout.addRow("Trigger Source:", self.combo_trigger_source)
        chan_layout.addRow("Analog ADC Trigger Chan:", self.spin_adc_trigger_chan)
        chan_layout.addRow("Sweep Window:", self.spin_window_ms)
        chan_layout.addRow("Pre-Stimulus Time:", self.spin_pre_ms)
        chan_layout.addRow("", self.chk_online_filter)
        chan_layout.addRow("", self.chk_follow_live)

        main_layout.addWidget(chan_group)

        # -------------------------------------------------------------
        # 3. Bidirectional Closed-Loop Control Group
        # -------------------------------------------------------------
        bidi_group = QGroupBox("Bidirectional Open Ephys Control")
        bidi_layout = QFormLayout(bidi_group)

        self.chk_sync_annotations = QCheckBox("Sync Annotations to MessageCenter")
        self.chk_sync_annotations.setChecked(True)

        self.spin_target_proc = QSpinBox()
        self.spin_target_proc.setRange(0, 9999)
        self.spin_target_proc.setValue(0)
        self.spin_target_proc.setSpecialValueText("None (Broadcast Only)")

        btn_hlayout = QHBoxLayout()
        self.btn_query_chain = QPushButton("Query Signal Chain")
        self.btn_query_chain.clicked.connect(self._query_signal_chain)
        self.btn_toggle_oe_acq = QPushButton("Start OE Acquisition")
        self.btn_toggle_oe_acq.clicked.connect(self._toggle_oe_acquisition)
        btn_hlayout.addWidget(self.btn_query_chain)
        btn_hlayout.addWidget(self.btn_toggle_oe_acq)

        volt_hlayout = QHBoxLayout()
        self.spin_stim_volt = QDoubleSpinBox()
        self.spin_stim_volt.setRange(0.0, 100.0)
        self.spin_stim_volt.setValue(5.0)
        self.spin_stim_volt.setSuffix(" V")
        self.btn_send_volt = QPushButton("Send Voltage")
        self.btn_send_volt.clicked.connect(self._send_stim_voltage)
        volt_hlayout.addWidget(self.spin_stim_volt)
        volt_hlayout.addWidget(self.btn_send_volt)

        bidi_layout.addRow("", self.chk_sync_annotations)
        bidi_layout.addRow("Target Processor ID:", self.spin_target_proc)
        bidi_layout.addRow("OE Control:", btn_hlayout)
        bidi_layout.addRow("Stimulus Voltage:", volt_hlayout)

        main_layout.addWidget(bidi_group)

        # -------------------------------------------------------------
        # 4. Live Diagnostics & Demo Stream Group
        # -------------------------------------------------------------
        diag_group = QGroupBox("Diagnostics & Live Metrics")
        diag_layout = QFormLayout(diag_group)

        self.lbl_status = QLabel("Disconnected")
        self.lbl_status.setStyleSheet("color: gray; font-weight: bold;")

        self.lbl_sweeps_count = QLabel("0")
        self.lbl_spikes_count = QLabel("0")
        self.lbl_stim_freq = QLabel("0.0 Hz")

        diag_layout.addRow("Status:", self.lbl_status)
        diag_layout.addRow("Sweeps Received:", self.lbl_sweeps_count)
        diag_layout.addRow("Spikes Detected:", self.lbl_spikes_count)
        diag_layout.addRow("Stimulus Frequency:", self.lbl_stim_freq)

        # Demo stream parameter controls
        self.spin_demo_stim_freq = QDoubleSpinBox()
        self.spin_demo_stim_freq.setRange(0.5, 50.0)
        self.spin_demo_stim_freq.setValue(5.0)
        self.spin_demo_stim_freq.setSingleStep(0.5)
        self.spin_demo_stim_freq.setSuffix(" Hz")
        self.spin_demo_stim_freq.valueChanged.connect(self._on_demo_stim_freq_changed)

        self.combo_demo_units = QComboBox()
        self.combo_demo_units.addItems(
            [
                "1 Unit (Fast A-beta)",
                "2 Units (A-beta + A-delta)",
                "3 Units (A-beta, A-delta, C-fiber)",
            ]
        )
        self.combo_demo_units.setCurrentIndex(2)

        diag_layout.addRow("Stimulation Frequency (Hz):", self.spin_demo_stim_freq)
        diag_layout.addRow("Demo Units:", self.combo_demo_units)

        self.btn_start_demo = QPushButton("Start Built-in Demo Stream")
        self.btn_start_demo.setStyleSheet("background-color: #2b5c8f; color: white; font-weight: bold; padding: 6px;")
        self.btn_start_demo.clicked.connect(self.start_demo_stream)
        diag_layout.addRow(self.btn_start_demo)

        main_layout.addWidget(diag_group)
        main_layout.addStretch()

    def _init_controller(self):
        host = self.edit_host.text().strip()
        port = self.spin_http_port.value()
        self.controller = OpenEphysController(host=host, port=port, parent=self)
        self.controller.statusChanged.connect(self._on_oe_status_changed)

    def _connect_state_signals(self):
        # When user annotates a unit in SpikeSpy, sync if enabled
        if hasattr(self.state, "onUnitAnnotated"):
            self.state.onUnitAnnotated.connect(self._on_unit_annotated)
        if hasattr(self.state, "onUnitChange"):
            self.state.onUnitChange.connect(self._on_unit_changed_in_state)

    def get_settings(self) -> Dict[str, Any]:
        """Return dict of current settings for perspective persistence."""
        return {
            "host": self.edit_host.text(),
            "data_port": self.spin_data_port.value(),
            "event_port": self.spin_event_port.value(),
            "http_port": self.spin_http_port.value(),
            "stream_mode": self.combo_stream_mode.currentIndex(),
            "rec_chan": self.spin_rec_chan.value(),
            "active_channels": self.edit_active_channels.text(),
            "trigger_source": self.combo_trigger_source.currentIndex(),
            "window_ms": self.spin_window_ms.value(),
            "online_filter": self.chk_online_filter.isChecked(),
            "sync_annotations": self.chk_sync_annotations.isChecked(),
            "demo_stim_freq": self.spin_demo_stim_freq.value(),
            "demo_units": self.combo_demo_units.currentIndex(),
        }

    def set_settings(self, settings: Dict[str, Any]):
        """Restore settings from dict."""
        if not settings:
            return
        if "host" in settings:
            self.edit_host.setText(settings["host"])
        if "data_port" in settings:
            self.spin_data_port.setValue(int(settings["data_port"]))
        if "event_port" in settings:
            self.spin_event_port.setValue(int(settings["event_port"]))
        if "http_port" in settings:
            self.spin_http_port.setValue(int(settings["http_port"]))
        if "stream_mode" in settings:
            self.combo_stream_mode.setCurrentIndex(int(settings["stream_mode"]))
        if "rec_chan" in settings:
            self.spin_rec_chan.setValue(int(settings["rec_chan"]))
        if "active_channels" in settings:
            self.edit_active_channels.setText(str(settings["active_channels"]))
        if "trigger_source" in settings:
            self.combo_trigger_source.setCurrentIndex(int(settings["trigger_source"]))
        if "window_ms" in settings:
            self.spin_window_ms.setValue(float(settings["window_ms"]))
        if "online_filter" in settings:
            self.chk_online_filter.setChecked(bool(settings["online_filter"]))
        if "sync_annotations" in settings:
            self.chk_sync_annotations.setChecked(bool(settings["sync_annotations"]))
        if "demo_stim_freq" in settings:
            self.spin_demo_stim_freq.setValue(float(settings["demo_stim_freq"]))
        if "demo_units" in settings:
            self.combo_demo_units.setCurrentIndex(int(settings["demo_units"]))

    @Slot(float)
    def _on_demo_stim_freq_changed(self, val: float):
        """Update running demo server's stimulation frequency dynamically."""
        if self._demo_server_process is not None and self._demo_server_process.is_running():
            self._demo_server_process.set_stim_freq(val)

    @Slot()
    def toggle_stream(self):
        if self._is_streaming:
            self.stop_stream()
        else:
            self.start_stream()

    def _parse_active_channels(self) -> List[int]:
        """Parse comma-separated channels and ranges, e.g. '0-3, 5' -> [0, 1, 2, 3, 5]."""
        text = self.edit_active_channels.text().strip()
        chans = set()
        if text:
            for part in text.split(","):
                part = part.strip()
                if not part:
                    continue
                if "-" in part:
                    sub = part.split("-")
                    if len(sub) == 2 and sub[0].strip().isdigit() and sub[1].strip().isdigit():
                        start, end = int(sub[0].strip()), int(sub[1].strip())
                        if start <= end:
                            chans.update(range(start, end + 1))
                        else:
                            chans.update(range(end, start + 1))
                elif part.isdigit():
                    chans.add(int(part))
        # Ensure current recording channel is always included
        chans.add(self.spin_rec_chan.value())
        return sorted(list(chans))

    @Slot()
    def _on_active_channels_edited(self):
        active_chans = self._parse_active_channels()
        if self.worker is not None:
            self.worker.set_active_channels(active_chans, primary_channel=self.spin_rec_chan.value())

    def start_stream(self):
        host = self.edit_host.text().strip()
        data_port = self.spin_data_port.value()
        event_port = self.spin_event_port.value()
        http_port = self.spin_http_port.value()
        is_sliced = self.combo_stream_mode.currentIndex() == 1
        mode = "sliced_erp" if is_sliced else "zmq_raw"
        trig_source = "ttl" if self.combo_trigger_source.currentIndex() == 0 else "analog"
        active_chans = self._parse_active_channels()

        # Update controller
        self.controller.host = host
        self.controller.port = http_port
        target_pid = self.spin_target_proc.value()
        self.controller.target_processor_id = target_pid if target_pid > 0 else None

        self.worker = OpenEphysStreamWorker(
            host=host,
            data_port=data_port,
            event_port=event_port,
            heartbeat_port=event_port,
            mode=mode,
            rec_channel=self.spin_rec_chan.value(),
            active_channels=active_chans,
            trigger_source=trig_source,
            analog_trigger_channel=self.spin_adc_trigger_chan.value(),
            window_ms=self.spin_window_ms.value(),
            pre_stim_ms=self.spin_pre_ms.value(),
            apply_filter=self.chk_online_filter.isChecked(),
            parent=self,
        )

        self.worker.sweepReady.connect(self._on_sweep_ready)
        self.worker.spikeEventsReady.connect(self._on_spike_events_ready)
        self.worker.protocolEventReady.connect(self._on_protocol_event_ready)
        self.worker.statsUpdated.connect(self._on_stats_updated)
        self.worker.connectionStatusChanged.connect(self._on_connection_status_changed)

        if hasattr(self.state, "start_live_session"):
            self.state.start_live_session(
                sampling_rate=30000.0,
                window_size_ms=self.spin_window_ms.value(),
                channel_name=f"CH{self.spin_rec_chan.value()}",
                units=pq.uV,
                aux_channels=[f"CH{c}" for c in active_chans],
            )

        self.worker.start()
        self._is_streaming = True
        self.btn_toggle_stream.setText("Stop Live Stream")
        self.btn_toggle_stream.setStyleSheet("background-color: #d9534f; color: white; font-weight: bold; padding: 6px;")

    def stop_stream(self):
        if self.worker is not None:
            self.worker.stop()
            self.worker = None

        if hasattr(self.state, "stop_live_session"):
            self.state.stop_live_session()

        self._is_streaming = False
        self.btn_toggle_stream.setText("Start Live Stream")
        self.btn_toggle_stream.setStyleSheet("font-weight: bold; padding: 6px;")
        self.lbl_status.setText("Stopped")
        self.lbl_status.setStyleSheet("color: gray; font-weight: bold;")

    @Slot()
    def start_demo_stream(self):
        """Spin up the in-process synthetic microneurography Open Ephys mock server and connect."""
        if self._demo_server_process is not None and self._demo_server_process.is_running():
            self._demo_server_process.stop()
            self._demo_server_process = None
            self.btn_start_demo.setText("Start Built-in Demo Stream")
            self.btn_start_demo.setStyleSheet("background-color: #2b5c8f; color: white; font-weight: bold; padding: 6px;")
            self.stop_stream()
            return

        try:
            from tests.mock_open_ephys_server import MockOpenEphysServer
        except ImportError:
            try:
                import sys
                from pathlib import Path

                root = str(Path(__file__).resolve().parent.parent.parent)
                if root not in sys.path:
                    sys.path.insert(0, root)
                from tests.mock_open_ephys_server import MockOpenEphysServer
            except Exception as e:
                QMessageBox.critical(self, "Demo Error", f"Could not load mock server: {e}")
                return

        demo_data_port = 5566
        demo_event_port = 5567
        demo_http_port = 37499

        demo_freq = self.spin_demo_stim_freq.value()
        num_units = self.combo_demo_units.currentIndex() + 1
        n_channels = max(6, num_units * 2)

        self._demo_server_process = MockOpenEphysServer(
            data_port=demo_data_port,
            event_port=demo_event_port,
            http_port=demo_http_port,
            stim_freq_hz=demo_freq,
            num_units=num_units,
            n_channels=n_channels,
            sample_rate=30000.0,
        )
        self._demo_server_process.start()

        # Update UI port fields
        self.spin_data_port.setValue(demo_data_port)
        self.spin_event_port.setValue(demo_event_port)
        self.spin_http_port.setValue(demo_http_port)

        # Ensure active channels and window accommodate simulated units
        active_set_str = f"0-{n_channels - 1}"
        self.edit_active_channels.setText(active_set_str)
        if self.spin_window_ms.value() < 120.0:
            self.spin_window_ms.setValue(150.0)

        # Automatically initialize state with corresponding spike_groups for active demo units
        unit_meta = [
            ("unit_0", "CH0", ["CH1"], "Fast A-fiber (A-beta)"),
            ("unit_1", "CH2", ["CH3"], "Medium A-delta fiber"),
            ("unit_2", "CH4", ["CH5"], "Slow C-fiber"),
        ][:num_units]

        sgs = []
        for i, (uname, ch, neighbors, ftype) in enumerate(unit_meta):
            evt = Event(np.array([], dtype=float) * pq.s, name=uname)
            evt.annotations["type"] = "unit"
            evt.annotations["unit"] = i
            evt.annotations["channel"] = ch
            evt.annotations["neighbor_channels"] = neighbors
            evt.annotations["fibre_type"] = ftype
            sg = tracked_neuron_unit(
                event=evt,
                channel=ch,
                neighbor_channels=neighbors,
                fibre_type=ftype,
                idx_arr=[],
            )
            sgs.append(sg)

        self.state.spike_groups = sgs
        self.state.cur_spike_group = 0
        self.state.update_idx_arrs()
        self.state.onUnitGroupChange.emit()

        self.btn_start_demo.setText("Stop Built-in Demo Stream")
        self.btn_start_demo.setStyleSheet("background-color: #d9534f; color: white; font-weight: bold; padding: 6px;")

        # Start stream worker automatically
        self.start_stream()

    @Slot(dict)
    def _on_sweep_ready(self, payload: Dict[str, Any]):
        """Handler for newly sliced sweep."""
        waveform = payload["waveform"]
        stim_ts = payload["stim_timestamp_s"]
        pre_offset_s = payload.get("pre_offset_s", 0.0)
        spikes = payload.get("spike_latencies_s", {})
        aux_sweeps = payload.get("aux_sweeps", {})
        sample_rate = payload.get("sample_rate", 30000.0)

        if hasattr(self.state, "append_sweep"):
            if not getattr(self.state, "is_streaming", False) or self.state.event_signal is None:
                active_chans = self._parse_active_channels()
                self.state.start_live_session(
                    sampling_rate=sample_rate,
                    window_size_ms=self.spin_window_ms.value(),
                    channel_name=f"CH{self.spin_rec_chan.value()}",
                    units=pq.uV,
                    aux_channels=[f"CH{c}" for c in active_chans],
                )
            self.state.follow_live = self.chk_follow_live.isChecked()
            self.state.append_sweep(
                stim_timestamp=stim_ts,
                sweep_waveform=waveform,
                pre_offset_s=pre_offset_s,
                spike_latencies_s=spikes,
                aux_sweeps=aux_sweeps,
            )
        else:
            if self.state.analog_signal is None or len(self.state.event_signal or []) == 0:
                asig = AnalogSignal(waveform, pq.uV, sampling_rate=sample_rate * pq.Hz, name="CH1")
                evt = Event(np.array([stim_ts]) * pq.s, name="Stimulus")
                seg = Segment(name="Live_OE_Stream")
                seg.analogsignals.append(asig)
                seg.events.append(evt)
                self.state.segment = seg
                self.state.set_data(asig, evt)
            else:
                self._fallback_append_sweep(waveform, stim_ts, sample_rate)

            for unit_name, lat_s in spikes.items():
                self._append_spike_to_state(unit_name, stim_ts + lat_s)

            if self.chk_follow_live.isChecked() and self.state.event_signal is not None:
                total_sweeps = len(self.state.event_signal)
                if total_sweeps > 0 and self.state.stimno != total_sweeps - 1:
                    self.state.setStimNo(total_sweeps - 1)

    def _fallback_append_sweep(self, waveform: np.ndarray, stim_ts: float, sample_rate: float):
        """Append a sweep to ViewerState's event_signal and analog_signal if append_sweep is not defined."""
        # Append timestamp to event_signal
        cur_times = list(self.state.event_signal.rescale(pq.s).magnitude)
        cur_times.append(stim_ts)
        new_event = Event(np.array(cur_times) * pq.s, name=self.state.event_signal.name)

        # Append waveform samples
        cur_data = self.state.analog_signal.magnitude
        if cur_data.ndim == 1:
            cur_data = cur_data[:, np.newaxis]
        wf = waveform if waveform.ndim == 2 else waveform[:, np.newaxis]
        new_data = np.vstack([cur_data, wf])

        new_asig = AnalogSignal(
            new_data,
            pq.uV,
            sampling_rate=sample_rate * pq.Hz,
            name=self.state.analog_signal.name,
        )

        if self.state.segment is not None:
            self.state.segment.events = [new_event]
            self.state.segment.analogsignals = [new_asig]

        self.state.set_data(new_asig, new_event, spike_groups=self.state.spike_groups)

    def _append_spike_to_state(self, unit_name: str, spike_time_s: float):
        """Append an individual spike event to the corresponding spike group in state."""
        target_sg = None
        for sg in self.state.spike_groups:
            if sg.event.name == unit_name:
                target_sg = sg
                break

        if target_sg is None:
            active_chans = self._parse_active_channels()
            primary_ch = f"CH{self.spin_rec_chan.value()}"
            neighbors = [f"CH{c}" for c in active_chans if c != self.spin_rec_chan.value()]
            evt = Event(np.array([spike_time_s]) * pq.s, name=unit_name)
            evt.annotations["type"] = "unit"
            evt.annotations["channel"] = primary_ch
            evt.annotations["neighbor_channels"] = neighbors
            evt.annotations["fibre_type"] = "C-fibre"
            target_sg = tracked_neuron_unit(
                event=evt,
                channel=primary_ch,
                neighbor_channels=neighbors,
                fibre_type="C-fibre",
            )
            self.state.spike_groups.append(target_sg)
        else:
            old_times = list(target_sg.event.rescale(pq.s).magnitude)
            old_times.append(spike_time_s)
            target_sg.event = Event(
                np.sort(np.array(old_times)) * pq.s,
                name=unit_name,
                **target_sg.event.annotations,
            )

        self.state.update_idx_arrs()
        self.state.onUnitGroupChange.emit()

    @Slot(str, object)
    def _on_spike_events_ready(self, unit_name: str, spike_ts_arr: np.ndarray):
        """Handler for upstream spike events arriving from Open Ephys."""
        if hasattr(self.state, "append_unit_spikes"):
            self.state.append_unit_spikes(unit_name, spike_ts_arr)
        else:
            for ts in spike_ts_arr:
                self._append_spike_to_state(unit_name, float(ts))

    @Slot(str, float, dict)
    def _on_protocol_event_ready(self, event_type: str, ts: float, details: dict):
        logger.info("Protocol Event [%s] at %.3fs: %s", event_type, ts, details)

    @Slot(dict)
    def _on_stats_updated(self, stats: Dict[str, Any]):
        self.lbl_sweeps_count.setText(str(stats.get("sweeps_received", 0)))
        self.lbl_spikes_count.setText(str(stats.get("spikes_received", 0)))
        freq = stats.get("last_stim_freq_hz", 0.0)
        self.lbl_stim_freq.setText(f"{freq:.2f} Hz")

    @Slot(bool, str)
    def _on_connection_status_changed(self, connected: bool, message: str):
        if connected:
            self.lbl_status.setText("Streaming (Connected)")
            self.lbl_status.setStyleSheet("color: green; font-weight: bold;")
        else:
            self.lbl_status.setText(f"Disconnected ({message})")
            self.lbl_status.setStyleSheet("color: red; font-weight: bold;")

    @Slot(int)
    def _on_rec_channel_changed(self, chan: int):
        if self.worker is not None:
            self.worker.set_rec_channel(chan)

        # If f"CH{chan}" is buffered in self.state, switch the active displayed channel
        ch_name = f"CH{chan}"
        if hasattr(self.state, "set_active_channel"):
            self.state.set_active_channel(ch_name)

    @Slot(bool)
    def _on_filter_toggled(self, enabled: bool):
        if self.worker is not None:
            self.worker.set_filtering(enabled)

    # -------------------------------------------------------------
    # Bidirectional controls
    # -------------------------------------------------------------
    @Slot()
    def _query_signal_chain(self):
        procs = self.controller.get_processors()
        if not procs:
            QMessageBox.information(
                self,
                "Signal Chain",
                "No processors found or Open Ephys HTTP API unreachable.",
            )
            return

        lines = [f"ID {p.get('id')}: {p.get('name', 'Processor')} ({p.get('type', '')})" for p in procs]
        QMessageBox.information(
            self,
            "Open Ephys Signal Chain",
            f"Active Processors ({len(procs)}):\n\n" + "\n".join(lines),
        )

    @Slot()
    def _toggle_oe_acquisition(self):
        status = self.controller.get_status()
        if status in ("ACQUIRE", "RECORD"):
            ok = self.controller.set_status("IDLE")
            if ok:
                self.btn_toggle_oe_acq.setText("Start OE Acquisition")
        else:
            ok = self.controller.set_status("ACQUIRE")
            if ok:
                self.btn_toggle_oe_acq.setText("Stop OE Acquisition")

    @Slot(str)
    def _on_oe_status_changed(self, status: str):
        if status in ("ACQUIRE", "RECORD"):
            self.btn_toggle_oe_acq.setText("Stop OE Acquisition")
        else:
            self.btn_toggle_oe_acq.setText("Start OE Acquisition")

    @Slot()
    def _send_stim_voltage(self):
        v = self.spin_stim_volt.value()
        msg = f"setStimVoltage: {v}"
        ok = self.controller.send_message(msg)
        if ok:
            QMessageBox.information(self, "Stimulus Voltage", f"Sent: {msg}")
        else:
            QMessageBox.warning(self, "Stimulus Voltage", "Failed to send message to Open Ephys.")

    @Slot(dict)
    def _on_unit_annotated(self, annotation_info: Dict[str, Any]):
        if self.chk_sync_annotations.isChecked():
            self.controller.sync_unit_annotation(annotation_info)

    @Slot(int)
    def _on_unit_changed_in_state(self, stimno: int):
        if not self.chk_sync_annotations.isChecked():
            return
        sg = self.state.getUnitGroup()
        if sg is None:
            return

        sr = 10000.0
        if hasattr(self.state, "sampling_rate") and self.state.sampling_rate is not None:
            sr = (
                float(self.state.sampling_rate.rescale(pq.Hz).magnitude)
                if hasattr(self.state.sampling_rate, "magnitude")
                else float(self.state.sampling_rate)
            )

        latency_samples = None
        latency_ms = None
        if self.state.event_signal is not None and stimno < len(self.state.event_signal) and hasattr(sg, "get_latencies"):
            cur_ev = self.state.event_signal[stimno]
            lat = sg.get_latencies(np.array([cur_ev.magnitude]) * cur_ev.units)[0]
            if not np.isnan(lat):
                sec = float(lat.rescale(pq.s).magnitude) if hasattr(lat, "rescale") else float(lat)
                latency_samples = int(round(sec * sr))
                latency_ms = float(lat.rescale(pq.ms).magnitude) if hasattr(lat, "rescale") else (sec * 1000.0)
        elif getattr(sg, "idx_arr", None) is not None and stimno < len(sg.idx_arr):
            item = sg.idx_arr[stimno]
            if item is not None:
                latency_samples = item[0]
                latency_ms = (latency_samples / sr) * 1000.0

        if latency_samples is None:
            return

        channel = getattr(sg, "channel", "") or getattr(
            self.state.analog_signal, "name", f"CH{self.spin_rec_chan.value()}"
        )
        neighbor_channels = getattr(sg, "neighbor_channels", [])

        info = {
            "unit": self.state.cur_spike_group,
            "stimno": stimno,
            "latency_samples": latency_samples,
            "latency_ms": latency_ms,
            "fibre_type": getattr(sg, "fibre_type", ""),
            "channel": channel,
            "neighbor_channels": neighbor_channels,
        }
        self.controller.sync_unit_annotation(info)

    def closeEvent(self, event):
        self.stop_stream()
        if self._demo_server_process is not None:
            self._demo_server_process.stop()
            self._demo_server_process = None
        super().closeEvent(event)
