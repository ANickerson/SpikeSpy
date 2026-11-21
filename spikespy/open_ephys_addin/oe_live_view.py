"""Open Ephys Live View Dockable Widget for SpikeSpy.

Provides GUI controls to connect to Open Ephys v1.0+, manage streaming,
filter parameters, trigger sources, closed-loop bidirectional control,
and live stream diagnostics.
"""

from __future__ import annotations

from collections import deque
import json
import logging
from typing import Any, Deque, Dict, List, Optional

import numpy as np
import quantities as pq
from neo import AnalogSignal, Event, Segment
from PySide6.QtCore import Qt, QSignalBlocker, QTimer, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from spikespy.ChannelPurposeView import ChannelPurposeWidget
from spikespy.ViewerState import ViewerState, tracked_neuron_unit

from .oe_controller import OpenEphysController
from .oe_stream_worker import OpenEphysStreamWorker
from .oe_debug_window import OpenEphysLiveDebugDialog

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

        self._sweep_queue: Deque[Dict[str, Any]] = deque(maxlen=256)
        self._sweep_drain_timer = QTimer(self)
        self._sweep_drain_timer.setSingleShot(True)
        self._sweep_drain_timer.setInterval(0)
        self._sweep_drain_timer.timeout.connect(self._drain_sweep_queue)

        self._perf_timer = QTimer(self)
        self._perf_timer.setInterval(500)
        self._perf_timer.timeout.connect(self._update_perf_diagnostics)
        self._perf_timer.start()
        self._diag_tick = 0
        self._last_synced_annotation_key = None
        self._syncing_follow_live = False
        self._syncing_live_config = False

        # Replay of an already-loaded recording through the live pipeline.
        self._is_replaying = False
        self._replay_timer = QTimer(self)
        self._replay_timer.setSingleShot(True)
        self._replay_timer.timeout.connect(self._replay_next_sweep)
        self._replay_events_s = np.empty(0, dtype=np.float64)
        self._replay_channels: List[tuple] = []
        self._replay_primary_name = ""
        self._replay_units = pq.uV
        self._replay_sr = 0.0
        self._replay_pre_s = 0.0
        self._replay_window_ms = 500.0
        self._replay_window_samples = 0
        self._replay_spike_latencies: List[Dict[str, float]] = []
        self._replay_index = 0
        self._replay_n = 0
        self._replay_t0 = 0.0
        self._syncing_timeline = False
        self._timeline_seek_timer = QTimer(self)
        self._timeline_seek_timer.setSingleShot(True)
        self._timeline_seek_timer.setInterval(150)
        self._timeline_seek_timer.timeout.connect(self._on_timeline_seek_timeout)
        self._debug_dialog: Optional[OpenEphysLiveDebugDialog] = None

        self._init_ui()
        self._init_controller()
        self._connect_state_signals()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(6)

        # -------------------------------------------------------------
        # 1. Connection & Live Stream Control (Always Visible)
        # -------------------------------------------------------------
        self.conn_group = QGroupBox("Connection & Live Stream Control")
        conn_vbox = QVBoxLayout(self.conn_group)
        conn_vbox.setContentsMargins(8, 8, 8, 8)
        conn_vbox.setSpacing(6)

        # Compact live status summary header
        self.status_summary_bar = QFrame()
        self.status_summary_bar.setFrameShape(QFrame.StyledPanel)
        self.status_summary_bar.setStyleSheet(
            "QFrame { border: 1px solid rgba(128, 128, 128, 0.3); "
            "border-radius: 4px; background-color: rgba(128, 128, 128, 0.08); }"
        )
        summary_layout = QVBoxLayout(self.status_summary_bar)
        summary_layout.setContentsMargins(8, 5, 8, 5)
        summary_layout.setSpacing(2)

        self.lbl_summary_status = QLabel("○ Disconnected")
        self.lbl_summary_status.setStyleSheet(
            "color: gray; font-weight: bold; border: none; background: transparent;"
        )
        summary_layout.addWidget(self.lbl_summary_status)

        summary_meta_row = QHBoxLayout()
        summary_meta_row.setContentsMargins(0, 0, 0, 0)
        summary_meta_row.setSpacing(8)
        self.lbl_summary_oe = QLabel("OE: ○ Disconnected")
        self.lbl_summary_oe.setStyleSheet(
            "color: gray; font-size: 11px; border: none; background: transparent;"
        )
        self.lbl_summary_sweeps = QLabel("Sweeps: 0")
        self.lbl_summary_sweeps.setStyleSheet(
            "font-size: 11px; font-weight: bold; border: none; background: transparent;"
        )
        summary_meta_row.addWidget(self.lbl_summary_oe)
        summary_meta_row.addStretch()
        summary_meta_row.addWidget(self.lbl_summary_sweeps)
        summary_layout.addLayout(summary_meta_row)

        conn_vbox.addWidget(self.status_summary_bar)

        conn_grid = QGridLayout()
        conn_grid.setContentsMargins(0, 0, 0, 0)
        conn_grid.setHorizontalSpacing(8)
        conn_grid.setVerticalSpacing(4)

        self.edit_host = QLineEdit("127.0.0.1")
        self.edit_host.editingFinished.connect(self._push_live_config_to_state)
        self.spin_data_port = QSpinBox()
        self.spin_data_port.setRange(1024, 65535)
        self.spin_data_port.setValue(5556)
        self._prev_data_port = 5556
        self.spin_data_port.valueChanged.connect(self._on_data_port_changed)

        self.spin_event_port = QSpinBox()
        self.spin_event_port.setRange(1024, 65535)
        self.spin_event_port.setValue(5557)
        self.spin_event_port.valueChanged.connect(self._push_live_config_to_state)

        self.spin_http_port = QSpinBox()
        self.spin_http_port.setRange(1024, 65535)
        self.spin_http_port.setValue(37497)
        self.spin_http_port.valueChanged.connect(self._push_live_config_to_state)

        self.combo_stream_mode = QComboBox()
        self.combo_stream_mode.addItems(
            [
                "Standard OE ZMQ + Event Broadcaster",
                "Custom SpikeSpy C++ Plugin (Pre-Sliced)",
            ]
        )
        self.combo_stream_mode.currentIndexChanged.connect(
            self._push_live_config_to_state
        )

        conn_grid.addWidget(QLabel("Host:"), 0, 0)
        conn_grid.addWidget(self.edit_host, 0, 1)
        conn_grid.addWidget(QLabel("HTTP Port:"), 0, 2)
        conn_grid.addWidget(self.spin_http_port, 0, 3)

        conn_grid.addWidget(QLabel("Data Port:"), 1, 0)
        conn_grid.addWidget(self.spin_data_port, 1, 1)
        conn_grid.addWidget(QLabel("Event Port:"), 1, 2)
        conn_grid.addWidget(self.spin_event_port, 1, 3)

        conn_grid.addWidget(QLabel("Stream Mode:"), 2, 0)
        conn_grid.addWidget(self.combo_stream_mode, 2, 1, 1, 3)
        conn_grid.setColumnStretch(1, 1)
        conn_grid.setColumnStretch(3, 1)
        conn_vbox.addLayout(conn_grid)

        self.btn_toggle_stream = QPushButton("Start Live Stream")
        self.btn_toggle_stream.setStyleSheet("font-weight: bold; padding: 6px;")
        self.btn_toggle_stream.clicked.connect(self.toggle_stream)

        self.btn_reset_session = QPushButton("New / Clear Session")
        self.btn_reset_session.setStyleSheet("padding: 6px;")
        self.btn_reset_session.setToolTip(
            "Discard recorded sweeps and start a fresh session"
        )
        self.btn_reset_session.clicked.connect(self.reset_session)

        stream_btn_layout = QHBoxLayout()
        stream_btn_layout.setSpacing(6)
        stream_btn_layout.addWidget(self.btn_toggle_stream)
        stream_btn_layout.addWidget(self.btn_reset_session)
        conn_vbox.addLayout(stream_btn_layout)

        main_layout.addWidget(self.conn_group)

        # -------------------------------------------------------------
        # Tabbed Controls Below Connection Group
        # -------------------------------------------------------------
        self.tabs_live = QTabWidget(self)
        main_layout.addWidget(self.tabs_live, 1)

        def _wrap_tab_group(group_box: QGroupBox) -> QWidget:
            page = QWidget(self.tabs_live)
            page_layout = QVBoxLayout(page)
            page_layout.setContentsMargins(4, 6, 4, 4)
            page_layout.setSpacing(6)
            page_layout.addWidget(group_box)
            page_layout.addStretch(1)
            return page

        # -------------------------------------------------------------
        # Tab 1: Acquisition & Channels
        # -------------------------------------------------------------
        self.chan_group = QGroupBox("Channel & Online Slicing")
        chan_layout = QFormLayout(self.chan_group)
        chan_layout.setVerticalSpacing(6)

        self.spin_rec_chan = QSpinBox()
        self.spin_rec_chan.setRange(0, 512)
        self.spin_rec_chan.setValue(0)
        self.spin_rec_chan.valueChanged.connect(self._on_rec_channel_changed)

        self.edit_active_channels = QLineEdit("0-3")
        self.edit_active_channels.setToolTip(
            "Comma-separated channels or ranges to buffer simultaneously, e.g., 0-3, 5, 7"
        )
        self.edit_active_channels.editingFinished.connect(
            self._on_active_channels_edited
        )

        self.combo_trigger_source = QComboBox()
        self.combo_trigger_source.addItems(["Digital TTL Event", "Analog ADC Channel"])
        self.combo_trigger_source.currentIndexChanged.connect(
            self._push_live_config_to_state
        )

        self.spin_adc_trigger_chan = QSpinBox()
        self.spin_adc_trigger_chan.setRange(0, 64)
        self.spin_adc_trigger_chan.setValue(1)
        self.spin_adc_trigger_chan.valueChanged.connect(
            self._push_live_config_to_state
        )

        self.spin_window_ms = QDoubleSpinBox()
        self.spin_window_ms.setRange(10.0, 5000.0)
        self.spin_window_ms.setValue(500.0)
        self.spin_window_ms.setSuffix(" ms")
        self.spin_window_ms.valueChanged.connect(self._push_live_config_to_state)

        self.spin_pre_ms = QDoubleSpinBox()
        self.spin_pre_ms.setRange(0.0, 500.0)
        self.spin_pre_ms.setValue(0.0)
        self.spin_pre_ms.setSuffix(" ms")
        self.spin_pre_ms.valueChanged.connect(self._push_live_config_to_state)

        self.chk_online_filter = QCheckBox("Online Bandpass (500 – 7000 Hz)")
        self.chk_online_filter.setChecked(True)
        self.chk_online_filter.toggled.connect(self._on_filter_toggled)

        self.chk_follow_live = QCheckBox("Follow Live Sweep (Auto-advance view)")
        self.chk_follow_live.setChecked(True)
        self.chk_follow_live.toggled.connect(self._on_follow_live_toggled)

        self.btn_configure_channel_purposes = QPushButton(
            "🎛 Configure Channel Purposes & Preprocessing ▾"
        )
        self.btn_configure_channel_purposes.setCheckable(True)
        self.btn_configure_channel_purposes.setChecked(False)
        self.btn_configure_channel_purposes.setToolTip(
            "Expand or collapse inline Channel Purposes & Preprocessing controls"
        )
        self.btn_configure_channel_purposes.toggled.connect(
            self._on_channel_purposes_toggled
        )

        chan_layout.addRow("Recording Channel:", self.spin_rec_chan)
        chan_layout.addRow("Multi-Channel Active Set:", self.edit_active_channels)
        chan_layout.addRow("Trigger Source:", self.combo_trigger_source)
        chan_layout.addRow("Analog ADC Trigger Chan:", self.spin_adc_trigger_chan)
        chan_layout.addRow("Sweep Window:", self.spin_window_ms)
        chan_layout.addRow("Pre-Stimulus Time:", self.spin_pre_ms)
        chan_layout.addRow("", self.chk_online_filter)
        chan_layout.addRow("", self.chk_follow_live)
        chan_layout.addRow("", self.btn_configure_channel_purposes)

        self.channel_purpose_group = QGroupBox("Channel Purposes & Preprocessing")
        purpose_vbox = QVBoxLayout(self.channel_purpose_group)
        purpose_vbox.setContentsMargins(4, 6, 4, 4)
        purpose_vbox.setSpacing(4)
        self.channel_purpose_widget = ChannelPurposeWidget(
            parent=self.channel_purpose_group,
            state=self.state,
            compact=True,
        )
        purpose_vbox.addWidget(self.channel_purpose_widget)
        self.channel_purpose_group.setVisible(False)

        acq_tab_page = QWidget(self.tabs_live)
        acq_tab_layout = QVBoxLayout(acq_tab_page)
        acq_tab_layout.setContentsMargins(4, 6, 4, 4)
        acq_tab_layout.setSpacing(6)
        acq_tab_layout.addWidget(self.chan_group)
        acq_tab_layout.addWidget(self.channel_purpose_group)
        acq_tab_layout.addStretch(1)
        self.tabs_live.addTab(acq_tab_page, "📡 Acquisition & Channels")

        # -------------------------------------------------------------
        # Tab 2: Closed-Loop & OE
        # -------------------------------------------------------------
        self.bidi_group = QGroupBox("Bidirectional Open Ephys Control")
        bidi_layout = QFormLayout(self.bidi_group)
        bidi_layout.setVerticalSpacing(6)

        self.chk_sync_annotations = QCheckBox("Sync Annotations to MessageCenter")
        self.chk_sync_annotations.setChecked(True)
        self.chk_sync_annotations.toggled.connect(self._push_live_config_to_state)

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

        self.tabs_live.addTab(_wrap_tab_group(self.bidi_group), "⚡ Closed-Loop & OE")

        # -------------------------------------------------------------
        # Tab 3: Diagnostics & Demo
        # -------------------------------------------------------------
        self.diag_group = QGroupBox("Diagnostics & Live Metrics")
        diag_layout = QFormLayout(self.diag_group)
        diag_layout.setVerticalSpacing(5)

        self.lbl_status = QLabel("○ Disconnected")
        self.lbl_status.setStyleSheet("color: gray; font-weight: bold;")

        self.lbl_oe_gui_status = QLabel("○ Disconnected")
        self.lbl_oe_gui_status.setStyleSheet("color: gray; font-weight: bold;")

        self.lbl_plugin_detected = QLabel("Not detected")
        self.lbl_plugin_detected.setStyleSheet("color: gray;")

        self.lbl_sweeps_count = QLabel("0")
        self.lbl_spikes_count = QLabel("0")
        self.lbl_stim_freq = QLabel("0.0 Hz")
        self.lbl_ui_perf = QLabel("Normal (0.0 ms lag)")
        self.lbl_ui_perf.setStyleSheet("color: #2b7a2b;")

        metrics_row = QHBoxLayout()
        metrics_row.setSpacing(6)
        metrics_row.addWidget(QLabel("Sweeps:"))
        metrics_row.addWidget(self.lbl_sweeps_count)
        metrics_row.addWidget(QLabel("| Spikes:"))
        metrics_row.addWidget(self.lbl_spikes_count)
        metrics_row.addWidget(QLabel("| Freq:"))
        metrics_row.addWidget(self.lbl_stim_freq)
        metrics_row.addStretch()

        diag_layout.addRow("Stream Status:", self.lbl_status)
        diag_layout.addRow("Open Ephys GUI:", self.lbl_oe_gui_status)
        diag_layout.addRow("Streamer Plugin:", self.lbl_plugin_detected)
        diag_layout.addRow("Counters:", metrics_row)
        diag_layout.addRow("UI Strain / Loop Lag:", self.lbl_ui_perf)

        diag_divider = QFrame()
        diag_divider.setFrameShape(QFrame.HLine)
        diag_divider.setFrameShadow(QFrame.Sunken)
        diag_layout.addRow(diag_divider)

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

        diag_layout.addRow("Demo Stim Freq (Hz):", self.spin_demo_stim_freq)
        diag_layout.addRow("Demo Units:", self.combo_demo_units)

        self.btn_start_demo = QPushButton("Start Built-in Demo Stream")
        self.btn_start_demo.setStyleSheet(
            "background-color: #2b5c8f; color: white; font-weight: bold; padding: 6px;"
        )
        self.btn_start_demo.clicked.connect(self.start_demo_stream)
        diag_layout.addRow(self.btn_start_demo)

        self.btn_open_debug = QPushButton("🔍 Open Live Stream & Trigger Debugger")
        self.btn_open_debug.setStyleSheet(
            "background-color: #2b7a2b; color: white; font-weight: bold; padding: 6px;"
        )
        self.btn_open_debug.setToolTip(
            "Open the live continuous oscilloscope to view raw signal, monitor incoming TTL/crossing events, "
            "and diagnose trigger clock synchronization in real time."
        )
        self.btn_open_debug.clicked.connect(self._open_debug_dialog)
        diag_layout.addRow(self.btn_open_debug)

        self.btn_dump_diag = QPushButton("Show UI Performance & Freeze Diagnostics")
        self.btn_dump_diag.clicked.connect(self._show_ui_diagnostics_dialog)
        diag_layout.addRow(self.btn_dump_diag)

        self.tabs_live.addTab(
            _wrap_tab_group(self.diag_group), "📊 Diagnostics & Demo"
        )

        # -------------------------------------------------------------
        # Tab 4: Replay Signal
        # -------------------------------------------------------------
        self.replay_group = QGroupBox("Replay Loaded Signal")
        replay_layout = QFormLayout(self.replay_group)
        replay_layout.setVerticalSpacing(6)

        self.spin_replay_speed = QDoubleSpinBox()
        self.spin_replay_speed.setRange(0.1, 1000.0)
        self.spin_replay_speed.setValue(1.0)
        self.spin_replay_speed.setSingleStep(0.5)
        self.spin_replay_speed.setDecimals(2)
        self.spin_replay_speed.setSuffix(" x real time")
        self.spin_replay_speed.setToolTip(
            "Playback speed relative to the original inter-stimulus intervals."
        )

        self.chk_replay_spikes = QCheckBox("Replay recorded unit spikes")
        self.chk_replay_spikes.setChecked(True)
        self.chk_replay_spikes.setToolTip(
            "Re-create unit spike annotations from the loaded recording for each sweep."
        )

        self.btn_replay_loaded = QPushButton("Replay Loaded Signal as Live")
        self.btn_replay_loaded.setStyleSheet("font-weight: bold; padding: 6px;")
        self.btn_replay_loaded.setToolTip(
            "Slice the signal currently loaded in SpikeSpy at its stimulus events and "
            "feed those sweeps through the live pipeline, using the sweep window and "
            "pre-stimulus time configured in Acquisition & Channels."
        )
        self.btn_replay_loaded.clicked.connect(self.toggle_replay)

        self.lbl_replay_info = QLabel(
            "Load a recording with stimulus events to replay it."
        )
        self.lbl_replay_info.setWordWrap(True)
        self.lbl_replay_info.setStyleSheet("color: gray;")

        self.btn_replay_prev = QPushButton("Prev Event")
        self.btn_replay_prev.setToolTip(
            "Jump backward to the previous stimulus event, skipping the interval "
            "without events in between."
        )
        self.btn_replay_prev.clicked.connect(lambda: self._skip_replay_event(-1))
        self.btn_replay_prev.setEnabled(False)

        self.btn_replay_next = QPushButton("Next Event")
        self.btn_replay_next.setToolTip(
            "Jump forward to the next stimulus event, skipping the interval "
            "without events in between."
        )
        self.btn_replay_next.clicked.connect(lambda: self._skip_replay_event(1))
        self.btn_replay_next.setEnabled(False)

        self.slider_replay_timeline = QSlider(Qt.Horizontal)
        self.slider_replay_timeline.setRange(0, 1)
        self.slider_replay_timeline.setValue(0)
        self.slider_replay_timeline.setEnabled(False)
        self.slider_replay_timeline.setToolTip(
            "Replay timeline. Drag to jump through the recording; gaps between "
            "stimulus events are crossed by jumping to the nearest event."
        )
        self.slider_replay_timeline.valueChanged.connect(
            self._on_timeline_value_changed
        )

        timeline_layout = QHBoxLayout()
        timeline_layout.addWidget(self.btn_replay_prev)
        timeline_layout.addWidget(self.slider_replay_timeline, 1)
        timeline_layout.addWidget(self.btn_replay_next)

        self.lbl_replay_time = QLabel("No replay loaded")
        self.lbl_replay_time.setStyleSheet("color: gray;")

        replay_layout.addRow("Playback Speed:", self.spin_replay_speed)
        replay_layout.addRow("", self.chk_replay_spikes)
        replay_layout.addRow("Timeline:", timeline_layout)
        replay_layout.addRow("Position:", self.lbl_replay_time)
        replay_layout.addRow(self.btn_replay_loaded)
        replay_layout.addRow(self.lbl_replay_info)

        self.tabs_live.addTab(_wrap_tab_group(self.replay_group), "⏪ Replay Signal")
        self._sync_header_summary()

    def _sync_header_summary(self):
        """Mirror key stream, OE GUI, and sweep metrics into the top header bar."""
        if hasattr(self, "lbl_summary_status") and hasattr(self, "lbl_status"):
            self.lbl_summary_status.setText(self.lbl_status.text())
            base_style = self.lbl_status.styleSheet()
            self.lbl_summary_status.setStyleSheet(
                f"{base_style} border: none; background: transparent;"
            )
        if hasattr(self, "lbl_summary_oe") and hasattr(self, "lbl_oe_gui_status"):
            self.lbl_summary_oe.setText(f"OE: {self.lbl_oe_gui_status.text()}")
            oe_style = self.lbl_oe_gui_status.styleSheet()
            self.lbl_summary_oe.setStyleSheet(
                f"{oe_style} font-size: 11px; border: none; background: transparent;"
            )
        if hasattr(self, "lbl_summary_sweeps") and hasattr(self, "lbl_sweeps_count"):
            self.lbl_summary_sweeps.setText(f"Sweeps: {self.lbl_sweeps_count.text()}")

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
        # The follow mode lives in ViewerState so every pane (this one, View
        # Settings, channel groups) edits the same value instead of fighting.
        if hasattr(self.state, "onFollowLiveChanged"):
            self.state.onFollowLiveChanged.connect(self._sync_follow_live_checkbox)
        self._sync_follow_live_checkbox()
        if self.state is not None and hasattr(self.state, "onLiveConfigChanged"):
            self.state.onLiveConfigChanged.connect(self._on_state_live_config_changed)
            self._on_state_live_config_changed()

    @Slot()
    def _push_live_config_to_state(self, *args):
        if self._syncing_live_config:
            return
        if self.state is None or not hasattr(self.state, "update_live_config"):
            return
        stream_mode = (
            "sliced_erp" if self.combo_stream_mode.currentIndex() == 1 else "zmq_raw"
        )
        trigger_source = (
            "ttl" if self.combo_trigger_source.currentIndex() == 0 else "analog"
        )
        self.state.update_live_config(
            host=self.edit_host.text().strip(),
            http_port=self.spin_http_port.value(),
            data_port=self.spin_data_port.value(),
            event_port=self.spin_event_port.value(),
            stream_mode=stream_mode,
            rec_chan=self.spin_rec_chan.value(),
            active_channels=self.edit_active_channels.text().strip(),
            trigger_source=trigger_source,
            adc_trigger_chan=self.spin_adc_trigger_chan.value(),
            window_ms=self.spin_window_ms.value(),
            pre_ms=self.spin_pre_ms.value(),
            online_filter=self.chk_online_filter.isChecked(),
            sync_annotations=self.chk_sync_annotations.isChecked(),
        )

    @Slot()
    def _on_state_live_config_changed(self):
        if self._syncing_live_config:
            return
        if self.state is None or not hasattr(self.state, "get_live_config"):
            return
        cfg = self.state.get_live_config()
        self._syncing_live_config = True
        try:
            with QSignalBlocker(self.edit_host):
                self.edit_host.setText(str(cfg.get("host", "127.0.0.1")))
            with QSignalBlocker(self.spin_http_port):
                self.spin_http_port.setValue(int(cfg.get("http_port", 37497)))
            with QSignalBlocker(self.spin_data_port):
                data_port = int(cfg.get("data_port", 5556))
                self.spin_data_port.setValue(data_port)
                self._prev_data_port = data_port
            with QSignalBlocker(self.spin_event_port):
                self.spin_event_port.setValue(int(cfg.get("event_port", 5557)))
            with QSignalBlocker(self.combo_stream_mode):
                mode_val = cfg.get("stream_mode", "zmq_raw")
                mode_idx = (
                    1
                    if (mode_val == "sliced_erp" or mode_val == 1)
                    else 0
                )
                self.combo_stream_mode.setCurrentIndex(mode_idx)
            with QSignalBlocker(self.spin_rec_chan):
                self.spin_rec_chan.setValue(int(cfg.get("rec_chan", 0)))
            with QSignalBlocker(self.edit_active_channels):
                self.edit_active_channels.setText(
                    str(cfg.get("active_channels", "0-3"))
                )
            with QSignalBlocker(self.combo_trigger_source):
                trig_val = cfg.get("trigger_source", "ttl")
                trig_idx = 1 if (trig_val == "analog" or trig_val == 1) else 0
                self.combo_trigger_source.setCurrentIndex(trig_idx)
            with QSignalBlocker(self.spin_adc_trigger_chan):
                self.spin_adc_trigger_chan.setValue(int(cfg.get("adc_trigger_chan", 1)))
            with QSignalBlocker(self.spin_window_ms):
                self.spin_window_ms.setValue(float(cfg.get("window_ms", 500.0)))
            with QSignalBlocker(self.spin_pre_ms):
                self.spin_pre_ms.setValue(float(cfg.get("pre_ms", 0.0)))
            with QSignalBlocker(self.chk_online_filter):
                self.chk_online_filter.setChecked(bool(cfg.get("online_filter", True)))
                low_hz = float(cfg.get("filter_low_hz", 500.0))
                high_hz = float(cfg.get("filter_high_hz", 7000.0))
                self.chk_online_filter.setText(
                    f"Online Bandpass ({low_hz:g} \u2013 {high_hz:g} Hz)"
                )
            with QSignalBlocker(self.chk_sync_annotations):
                self.chk_sync_annotations.setChecked(
                    bool(cfg.get("sync_annotations", True))
                )

            if self.worker is not None:
                if hasattr(self.worker, "set_channel_roles") and self._should_apply_channel_roles(cfg):
                    roles_val = (
                        self.state.get_channel_roles()
                        if hasattr(self.state, "get_channel_roles")
                        else cfg.get("channel_roles")
                    )
                    if roles_val is not None:
                        self.worker.set_channel_roles(roles_val)
                self.worker.set_rec_channel(int(cfg["rec_chan"]))
                self.worker.set_active_channels(
                    self._parse_active_channels(str(cfg["active_channels"]))
                )
                self.worker.set_filtering(bool(cfg["online_filter"]))
                if hasattr(self.worker, "set_filter_band"):
                    self.worker.set_filter_band(
                        float(cfg["filter_low_hz"]),
                        float(cfg["filter_high_hz"]),
                    )
                if hasattr(self.worker, "set_ttl_params"):
                    self.worker.set_ttl_params(
                        threshold=float(cfg["ttl_threshold_v"]),
                        refractory_ms=float(cfg["ttl_refractory_ms"]),
                    )
                if hasattr(self.worker, "set_window_params"):
                    self.worker.set_window_params(
                        window_ms=float(cfg["window_ms"]),
                        pre_ms=float(cfg["pre_ms"]),
                    )
                self.worker.auto_align_clock = bool(
                    cfg.get("auto_align_clock", True)
                )
                self.worker.use_analog_trigger = cfg.get("trigger_source") == "analog"
                self.worker.adc_trigger_chan = int(cfg.get("adc_trigger_chan", 1))
        finally:
            self._syncing_live_config = False

    @Slot(bool)
    def _on_channel_purposes_toggled(self, checked: bool):
        if hasattr(self, "channel_purpose_group"):
            self.channel_purpose_group.setVisible(bool(checked))
            self.btn_configure_channel_purposes.setText(
                "🎛 Configure Channel Purposes & Preprocessing ▴"
                if checked
                else "🎛 Configure Channel Purposes & Preprocessing ▾"
            )

    @Slot()
    def _open_channel_purposes_tab(self):
        if hasattr(self, "tabs_live"):
            self.tabs_live.setCurrentIndex(0)
        if hasattr(self, "btn_configure_channel_purposes"):
            self.btn_configure_channel_purposes.setChecked(True)

    @Slot(bool)
    def _on_follow_live_toggled(self, checked: bool):
        if self._syncing_follow_live:
            return
        if hasattr(self.state, "set_follow_live"):
            self.state.set_follow_live(checked)
        else:
            self.state.follow_live = bool(checked)

    def _sync_follow_live_checkbox(self, *args):
        following = bool(getattr(self.state, "follow_live", True))
        if self.chk_follow_live.isChecked() == following:
            return
        self._syncing_follow_live = True
        try:
            with QSignalBlocker(self.chk_follow_live):
                self.chk_follow_live.setChecked(following)
        finally:
            self._syncing_follow_live = False

    def get_settings(self) -> Dict[str, Any]:
        """Return dict of current settings for perspective persistence."""
        cfg = (
            self.state.get_live_config()
            if self.state is not None and hasattr(self.state, "get_live_config")
            else {}
        )
        return {
            "host": self.edit_host.text(),
            "data_port": self.spin_data_port.value(),
            "event_port": self.spin_event_port.value(),
            "http_port": self.spin_http_port.value(),
            "stream_mode": self.combo_stream_mode.currentIndex(),
            "rec_chan": self.spin_rec_chan.value(),
            "active_channels": self.edit_active_channels.text(),
            "trigger_source": self.combo_trigger_source.currentIndex(),
            "adc_trigger_chan": self.spin_adc_trigger_chan.value(),
            "window_ms": self.spin_window_ms.value(),
            "pre_ms": self.spin_pre_ms.value(),
            "online_filter": self.chk_online_filter.isChecked(),
            "filter_low_hz": float(cfg.get("filter_low_hz", 500.0)),
            "filter_high_hz": float(cfg.get("filter_high_hz", 7000.0)),
            "ttl_threshold_v": float(cfg.get("ttl_threshold_v", 2.5)),
            "ttl_refractory_ms": float(cfg.get("ttl_refractory_ms", 100.0)),
            "auto_align_clock": bool(cfg.get("auto_align_clock", True)),
            "sync_annotations": self.chk_sync_annotations.isChecked(),
            "demo_stim_freq": self.spin_demo_stim_freq.value(),
            "demo_units": self.combo_demo_units.currentIndex(),
            "replay_speed": self.spin_replay_speed.value(),
            "replay_spikes": self.chk_replay_spikes.isChecked(),
            "active_tab": self.tabs_live.currentIndex(),
        }

    def set_settings(self, settings: Dict[str, Any]):
        """Restore settings from dict."""
        if not settings:
            return
        self._syncing_live_config = True
        try:
            if "host" in settings:
                self.edit_host.setText(settings["host"])
            if "data_port" in settings:
                self.spin_data_port.setValue(int(settings["data_port"]))
            if "event_port" in settings:
                self.spin_event_port.setValue(int(settings["event_port"]))
            if "http_port" in settings:
                self.spin_http_port.setValue(int(settings["http_port"]))
            if "stream_mode" in settings:
                sm = settings["stream_mode"]
                idx = 1 if (sm == "sliced_erp" or sm == 1) else 0
                self.combo_stream_mode.setCurrentIndex(idx)
            if "rec_chan" in settings:
                self.spin_rec_chan.setValue(int(settings["rec_chan"]))
            if "active_channels" in settings:
                self.edit_active_channels.setText(str(settings["active_channels"]))
            if "trigger_source" in settings:
                ts = settings["trigger_source"]
                idx = 1 if (ts == "analog" or ts == 1) else 0
                self.combo_trigger_source.setCurrentIndex(idx)
            if "adc_trigger_chan" in settings:
                self.spin_adc_trigger_chan.setValue(int(settings["adc_trigger_chan"]))
            if "window_ms" in settings:
                self.spin_window_ms.setValue(float(settings["window_ms"]))
            if "pre_ms" in settings:
                self.spin_pre_ms.setValue(float(settings["pre_ms"]))
            if "online_filter" in settings:
                self.chk_online_filter.setChecked(bool(settings["online_filter"]))
            if "sync_annotations" in settings:
                self.chk_sync_annotations.setChecked(bool(settings["sync_annotations"]))
            if "demo_stim_freq" in settings:
                self.spin_demo_stim_freq.setValue(float(settings["demo_stim_freq"]))
            if "demo_units" in settings:
                self.combo_demo_units.setCurrentIndex(int(settings["demo_units"]))
            if "replay_speed" in settings:
                self.spin_replay_speed.setValue(float(settings["replay_speed"]))
            if "replay_spikes" in settings:
                self.chk_replay_spikes.setChecked(bool(settings["replay_spikes"]))
            if "active_tab" in settings:
                tab_idx = int(settings["active_tab"])
                if 0 <= tab_idx < self.tabs_live.count():
                    self.tabs_live.setCurrentIndex(tab_idx)
        finally:
            self._syncing_live_config = False

        if self.state is not None and hasattr(self.state, "update_live_config"):
            updates: Dict[str, Any] = {
                "host": self.edit_host.text().strip(),
                "http_port": self.spin_http_port.value(),
                "data_port": self.spin_data_port.value(),
                "event_port": self.spin_event_port.value(),
                "stream_mode": (
                    "sliced_erp"
                    if self.combo_stream_mode.currentIndex() == 1
                    else "zmq_raw"
                ),
                "rec_chan": self.spin_rec_chan.value(),
                "active_channels": self.edit_active_channels.text().strip(),
                "trigger_source": (
                    "ttl"
                    if self.combo_trigger_source.currentIndex() == 0
                    else "analog"
                ),
                "adc_trigger_chan": self.spin_adc_trigger_chan.value(),
                "window_ms": self.spin_window_ms.value(),
                "pre_ms": self.spin_pre_ms.value(),
                "online_filter": self.chk_online_filter.isChecked(),
                "sync_annotations": self.chk_sync_annotations.isChecked(),
            }
            for opt_key in (
                "ttl_threshold_v",
                "ttl_refractory_ms",
                "filter_low_hz",
                "filter_high_hz",
                "auto_align_clock",
            ):
                if opt_key in settings:
                    updates[opt_key] = settings[opt_key]
            self.state.update_live_config(**updates)

    @Slot(float)
    def _on_demo_stim_freq_changed(self, val: float):
        """Update running demo server's stimulation frequency dynamically."""
        if (
            self._demo_server_process is not None
            and self._demo_server_process.is_running()
        ):
            self._demo_server_process.set_stim_freq(val)

    @Slot()
    def toggle_stream(self):
        if self._is_replaying:
            self.stop_replay()
            return
        if self._is_streaming:
            self.stop_stream()
        else:
            self.start_stream()

    def _parse_active_channels(self, text: Optional[str] = None) -> List[int]:
        """Parse comma-separated channels and ranges, e.g. '0-3, 5' -> [0, 1, 2, 3, 5]."""
        if text is None:
            text = self.edit_active_channels.text()
        text = text.strip()
        chans = set()
        if text:
            for part in text.split(","):
                part = part.strip()
                if not part:
                    continue
                if "-" in part:
                    sub = part.split("-")
                    if (
                        len(sub) == 2
                        and sub[0].strip().isdigit()
                        and sub[1].strip().isdigit()
                    ):
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
            self.worker.set_active_channels(
                active_chans, primary_channel=self.spin_rec_chan.value()
            )
        self._push_live_config_to_state()

    def _should_apply_channel_roles(self, cfg: Dict[str, Any]) -> bool:
        """Only apply channel_roles to OpenEphysStreamWorker when explicitly configured.

        ViewerState initializes default APTrack roles (only CH1/rd.0 bandpass-filtered),
        so uncustomized multi-channel Open Ephys sessions should keep filtering all active
        channels unless the user or config explicitly applied channel roles.
        """
        if self.state is not None and getattr(self.state, "_channel_roles_customized", False):
            return True
        if cfg.get("channel_roles_customized"):
            return True
        return False

    def start_stream(self):
        if self._is_replaying:
            self._cancel_replay()
        host = self.edit_host.text().strip()
        data_port = self.spin_data_port.value()
        event_port = self.spin_event_port.value()
        http_port = self.spin_http_port.value()
        is_sliced = self.combo_stream_mode.currentIndex() == 1
        mode = "sliced_erp" if is_sliced else "zmq_raw"
        trig_source = (
            "ttl" if self.combo_trigger_source.currentIndex() == 0 else "analog"
        )
        active_chans = self._parse_active_channels()

        # Update controller
        self.controller.host = host
        self.controller.port = http_port
        target_pid = self.spin_target_proc.value()
        self.controller.target_processor_id = target_pid if target_pid > 0 else None

        cfg = (
            self.state.get_live_config()
            if self.state is not None and hasattr(self.state, "get_live_config")
            else {}
        )
        filter_low_hz = float(cfg.get("filter_low_hz", 500.0))
        filter_high_hz = float(cfg.get("filter_high_hz", 7000.0))
        ttl_threshold_v = float(cfg.get("ttl_threshold_v", 2.5))
        ttl_refractory_ms = float(cfg.get("ttl_refractory_ms", 100.0))
        auto_align_clock = bool(cfg.get("auto_align_clock", True))

        hb_port = event_port if is_sliced else None
        channel_roles = (
            cfg.get("channel_roles") if self._should_apply_channel_roles(cfg) else None
        )
        self.worker = OpenEphysStreamWorker(
            host=host,
            data_port=data_port,
            event_port=event_port,
            heartbeat_port=hb_port,
            mode=mode,
            rec_channel=self.spin_rec_chan.value(),
            active_channels=active_chans,
            trigger_source=trig_source,
            analog_trigger_channel=self.spin_adc_trigger_chan.value(),
            window_ms=self.spin_window_ms.value(),
            pre_stim_ms=self.spin_pre_ms.value(),
            apply_filter=self.chk_online_filter.isChecked(),
            filter_low=filter_low_hz,
            filter_high=filter_high_hz,
            ttl_threshold_v=ttl_threshold_v,
            ttl_refractory_ms=ttl_refractory_ms,
            channel_roles=channel_roles,
            parent=self,
        )
        self.worker.auto_align_clock = auto_align_clock
        if hasattr(self.worker, "set_filter_band"):
            self.worker.set_filter_band(filter_low_hz, filter_high_hz)
        if hasattr(self.worker, "set_ttl_params"):
            self.worker.set_ttl_params(
                threshold=ttl_threshold_v,
                refractory_ms=ttl_refractory_ms,
            )

        self.worker.sweepsReady.connect(self._on_sweeps_ready)
        self.worker.spikeEventsReady.connect(self._on_spike_events_ready)
        self.worker.protocolEventReady.connect(self._on_protocol_event_ready)
        self.worker.statsUpdated.connect(self._on_stats_updated)
        self.worker.connectionStatusChanged.connect(self._on_connection_status_changed)
        if self._debug_dialog is not None:
            self._debug_dialog._connect_worker()

        can_resume = (
            hasattr(self.state, "has_live_session")
            and self.state.has_live_session()
            and getattr(self.state, "event_signal", None) is not None
            and len(self.state.event_signal) > 0
        )
        if can_resume:
            self.state.resume_live_session()
        elif hasattr(self.state, "start_live_session"):
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
        self.btn_toggle_stream.setStyleSheet(
            "background-color: #d9534f; color: white; font-weight: bold; padding: 6px;"
        )
        self.lbl_status.setText(f"◌ Connecting ({host}:{data_port})...")
        self.lbl_status.setStyleSheet("color: #d9822b; font-weight: bold;")
        self._sync_header_summary()

    def stop_stream(self):
        self._cancel_replay()
        if self.worker is not None:
            self.worker.stop()
            self.worker = None

        if hasattr(self.state, "stop_live_session"):
            self.state.stop_live_session()

        self._is_streaming = False
        sweeps_count = (
            len(self.state.event_signal)
            if self.state and self.state.event_signal is not None
            else 0
        )
        if sweeps_count > 0:
            self.btn_toggle_stream.setText("Resume Live Stream")
            self.lbl_status.setText(f"○ Stopped ({sweeps_count} sweeps recorded)")
        else:
            self.btn_toggle_stream.setText("Start Live Stream")
            self.lbl_status.setText("○ Stopped")
        self.btn_toggle_stream.setStyleSheet("font-weight: bold; padding: 6px;")
        self.lbl_status.setStyleSheet("color: gray; font-weight: bold;")
        self._sync_header_summary()

    @Slot()
    def reset_session(self):
        sweeps_count = (
            len(self.state.event_signal)
            if self.state and self.state.event_signal is not None
            else 0
        )
        if sweeps_count > 0:
            from PySide6.QtWidgets import QMessageBox

            if (
                QMessageBox.question(
                    self,
                    "New Recording Session",
                    "Are you sure you want to discard the recorded sweeps and start a new session?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                != QMessageBox.Yes
            ):
                return

        if self._is_streaming:
            self.stop_stream()

        active_chans = self._parse_active_channels()
        if hasattr(self.state, "start_live_session"):
            self.state.start_live_session(
                sampling_rate=30000.0,
                window_size_ms=self.spin_window_ms.value(),
                channel_name=f"CH{self.spin_rec_chan.value()}",
                units=pq.uV,
                aux_channels=[f"CH{c}" for c in active_chans],
                resume=False,
            )

        self.lbl_sweeps_count.setText("0")
        self.lbl_spikes_count.setText("0")
        self.lbl_stim_freq.setText("0.0 Hz")
        self.btn_toggle_stream.setText("Start Live Stream")
        self.lbl_status.setText("○ Ready (New Session)")
        self.lbl_status.setStyleSheet("color: gray; font-weight: bold;")
        self._sync_header_summary()

    # -------------------------------------------------------------
    # Replay of the currently loaded recording
    # -------------------------------------------------------------
    @property
    def is_replaying(self) -> bool:
        return self._is_replaying

    @Slot()
    def toggle_replay(self):
        if self._is_replaying:
            self.stop_replay()
        else:
            self.start_replay()

    @Slot()
    def start_replay(self):
        """Replay the currently loaded recording through the live pipeline.

        The continuous signal is sliced at each stimulus event using the sweep
        window / pre-stimulus controls, then fed to ``ViewerState.append_sweep``
        exactly like incoming Open Ephys sweeps.
        """
        if self._is_replaying:
            return
        if self._is_streaming:
            self.stop_stream()

        if not self._prepare_replay():
            return

        if not hasattr(self.state, "start_live_session"):
            QMessageBox.warning(
                self,
                "Replay Loaded Signal",
                "The current viewer state does not support live sessions.",
            )
            return

        self.state.start_live_session(
            sampling_rate=self._replay_sr,
            window_size_ms=self._replay_window_ms,
            channel_name=self._replay_primary_name,
            units=self._replay_units,
            aux_channels=[name for name, _ in self._replay_channels[1:]],
            resume=False,
        )

        self._replay_index = 0
        self._is_replaying = True
        self._timeline_seek_timer.stop()
        self._set_replay_controls(True)
        self.btn_replay_loaded.setText("Stop Replay")
        self.btn_replay_loaded.setStyleSheet(
            "background-color: #d9534f; color: white; font-weight: bold; padding: 6px;"
        )
        self.lbl_sweeps_count.setText("0")
        self.lbl_spikes_count.setText("0")
        self.lbl_stim_freq.setText("0.0 Hz")
        self.lbl_status.setText(f"Replaying loaded signal (0/{self._replay_n})")
        self.lbl_status.setStyleSheet("color: green; font-weight: bold;")
        self._sync_header_summary()

        self._replay_next_sweep()

    @Slot()
    def stop_replay(self):
        if not self._is_replaying:
            return
        played = self._replay_index
        total = self._replay_n
        self._cancel_replay()
        if getattr(self.state, "is_streaming", False) and hasattr(
            self.state, "stop_live_session"
        ):
            self.state.stop_live_session()
        self.lbl_status.setText(f"Replay stopped ({played}/{total} sweeps)")
        self.lbl_status.setStyleSheet("color: gray; font-weight: bold;")
        self._sync_header_summary()

    def _cancel_replay(self):
        """Stop the replay timer and restore controls without touching the session."""
        self._replay_timer.stop()
        self._timeline_seek_timer.stop()
        self._is_replaying = False
        self._set_replay_controls(False)
        self.btn_replay_loaded.setText("Replay Loaded Signal as Live")
        self.btn_replay_loaded.setStyleSheet("font-weight: bold; padding: 6px;")

    def _finish_replay(self):
        self._replay_timer.stop()
        self._timeline_seek_timer.stop()
        self._is_replaying = False
        if getattr(self.state, "is_streaming", False) and hasattr(
            self.state, "stop_live_session"
        ):
            self.state.stop_live_session()
        self._set_replay_controls(False)
        self.btn_replay_loaded.setText("Replay Loaded Signal as Live")
        self.btn_replay_loaded.setStyleSheet("font-weight: bold; padding: 6px;")
        self.lbl_sweeps_count.setText(str(self._replay_n))
        self.lbl_status.setText(f"Replay complete ({self._replay_n} sweeps)")
        self.lbl_status.setStyleSheet("color: green; font-weight: bold;")
        self._sync_header_summary()

    def _set_replay_controls(self, active: bool):
        self.btn_toggle_stream.setEnabled(not active)
        self.btn_start_demo.setEnabled(not active)
        self.btn_reset_session.setEnabled(not active)
        self.slider_replay_timeline.setEnabled(active)
        self.btn_replay_prev.setEnabled(active)
        self.btn_replay_next.setEnabled(active)

    @Slot(int)
    def _on_timeline_value_changed(self, value: int):
        if self._syncing_timeline or self._replay_n <= 0:
            return
        self._update_replay_time_label(value)
        if self._is_replaying:
            self._timeline_seek_timer.start()

    @Slot()
    def _on_timeline_seek_timeout(self):
        if not self._is_replaying or self._replay_n <= 0:
            return
        target = self._event_index_for_timeline_value(
            self.slider_replay_timeline.value()
        )
        self._seek_replay_to_index(target)

    def _skip_replay_event(self, direction: int):
        """Jump to the previous/next stimulus event, skipping silent gaps."""
        if not self._is_replaying or self._replay_n <= 0:
            return
        current = self._replay_index - 1 if self._replay_index > 0 else 0
        target = max(0, min(current + int(direction), self._replay_n - 1))
        if target != current:
            self._seek_replay_to_index(target)

    def _seek_replay_to_index(self, target_index: int):
        """Restart the replay session at a stimulus event and continue from there."""
        if not self._is_replaying or self._replay_n <= 0:
            return
        target_index = max(0, min(int(target_index), self._replay_n - 1))
        current = self._replay_index - 1 if self._replay_index > 0 else -1
        if target_index == current:
            self._update_timeline_position(current)
            return

        self._timeline_seek_timer.stop()
        self._replay_timer.stop()
        self._restart_replay_session()
        self._replay_index = target_index
        self._update_timeline_position(target_index)
        self._replay_next_sweep()

    def _restart_replay_session(self):
        if not hasattr(self.state, "start_live_session"):
            return
        self.state.start_live_session(
            sampling_rate=self._replay_sr,
            window_size_ms=self._replay_window_ms,
            channel_name=self._replay_primary_name,
            units=self._replay_units,
            aux_channels=[name for name, _ in self._replay_channels[1:]],
            resume=False,
        )
        self.lbl_spikes_count.setText("0")
        self.lbl_stim_freq.setText("0.0 Hz")

    def _configure_replay_timeline(self):
        self._replay_t0 = float(self._replay_events_s[0]) if self._replay_n > 0 else 0.0
        duration_ms = 0
        if self._replay_n > 1:
            duration_ms = int(
                round((float(self._replay_events_s[-1]) - self._replay_t0) * 1000.0)
            )
        self._syncing_timeline = True
        try:
            with QSignalBlocker(self.slider_replay_timeline):
                self.slider_replay_timeline.setRange(0, max(1, duration_ms))
                self.slider_replay_timeline.setValue(0)
        finally:
            self._syncing_timeline = False
        self._update_replay_time_label(0, event_index=0 if self._replay_n > 0 else None)

    def _update_timeline_position(self, event_index: int):
        if self._replay_n <= 0 or event_index < 0:
            return
        event_index = min(int(event_index), self._replay_n - 1)
        value = int(
            round(
                (float(self._replay_events_s[event_index]) - self._replay_t0) * 1000.0
            )
        )
        self._syncing_timeline = True
        try:
            with QSignalBlocker(self.slider_replay_timeline):
                self.slider_replay_timeline.setValue(value)
        finally:
            self._syncing_timeline = False
        self._update_replay_time_label(value, event_index=event_index)

    def _event_index_for_timeline_value(self, value: int) -> int:
        if self._replay_n <= 0:
            return 0
        target_s = self._replay_t0 + float(value) / 1000.0
        index = int(np.searchsorted(self._replay_events_s, target_s, side="left"))
        index = max(0, min(index, self._replay_n - 1))
        if index > 0:
            before = float(self._replay_events_s[index - 1])
            after = float(self._replay_events_s[index])
            if abs(before - target_s) <= abs(after - target_s):
                index -= 1
        return index

    def _update_replay_time_label(self, value: int, event_index: Optional[int] = None):
        if self._replay_n <= 0:
            self.lbl_replay_time.setText("No replay loaded")
            return
        if event_index is None:
            event_index = self._event_index_for_timeline_value(value)
        event_index = max(0, min(event_index, self._replay_n - 1))
        relative_s = float(self._replay_events_s[event_index]) - self._replay_t0
        total_s = max(0.0, float(self._replay_events_s[-1]) - self._replay_t0)
        self.lbl_replay_time.setText(
            f"{relative_s:.3f} s / {total_s:.3f} s  |  "
            f"event {event_index + 1}/{self._replay_n}"
        )

    @Slot()
    def _replay_next_sweep(self):
        if not self._is_replaying:
            return
        if self._replay_index >= self._replay_n:
            self._finish_replay()
            return

        index = self._replay_index
        stim_ts = float(self._replay_events_s[index])
        try:
            waveform = self._build_replay_sweep(stim_ts)
            latencies = (
                self._replay_spike_latencies[index]
                if index < len(self._replay_spike_latencies)
                else {}
            )
            self.state.append_sweep(
                stim_timestamp=stim_ts,
                sweep_waveform=waveform,
                pre_offset_s=self._replay_pre_s,
                spike_latencies_s=latencies,
                emit_signals=True,
            )
        except Exception:
            logger.exception("Failed to replay sweep %d", index)

        self._replay_index = index + 1
        self.lbl_sweeps_count.setText(str(self._replay_index))
        self._update_timeline_position(index)
        if index > 0:
            prev_ts = float(self._replay_events_s[index - 1])
            if stim_ts > prev_ts:
                self.lbl_stim_freq.setText(f"{1.0 / (stim_ts - prev_ts):.2f} Hz")
        self.lbl_status.setText(
            f"Replaying loaded signal ({self._replay_index}/{self._replay_n})"
        )
        self._sync_header_summary()

        if self._replay_index >= self._replay_n:
            self._finish_replay()
            return

        next_ts = float(self._replay_events_s[self._replay_index])
        speed = max(0.01, float(self.spin_replay_speed.value()))
        delay_ms = (next_ts - stim_ts) / speed * 1000.0
        self._replay_timer.start(max(1, int(round(delay_ms))))

    def _prepare_replay(self) -> bool:
        """Capture the loaded signal/events into replay buffers."""
        state = self.state
        segment = getattr(state, "segment", None)
        analog_signal = getattr(state, "analog_signal", None)

        candidates = []
        for sig in list(getattr(segment, "analogsignals", []) or []):
            annotations = getattr(sig, "annotations", {}) or {}
            if annotations.get("class_") == "multianalogsignal":
                continue
            if str(getattr(sig, "name", "")).startswith("ERP"):
                continue
            if self._signal_sample_count(sig) <= 0:
                continue
            candidates.append(sig)
        if not candidates and analog_signal is not None:
            candidates = [analog_signal]

        if not candidates:
            QMessageBox.warning(
                self,
                "Replay Loaded Signal",
                "No loaded signal is available to replay. Open a recording first.",
            )
            return False

        sr = self._signal_sampling_rate(candidates[0])
        if sr is None or sr <= 0:
            QMessageBox.warning(
                self,
                "Replay Loaded Signal",
                "The loaded signal has no usable sampling rate.",
            )
            return False
        candidates = [
            sig
            for sig in candidates
            if self._signal_sampling_rate(sig) is not None
            and abs(self._signal_sampling_rate(sig) - sr) < 1e-6
        ]

        channels: List[tuple] = []
        seen_names = set()
        for i, sig in enumerate(candidates):
            name = str(getattr(sig, "name", "") or f"CH{i}")
            if name in seen_names:
                continue
            seen_names.add(name)
            channels.append((name, sig))
        if not channels:
            QMessageBox.warning(
                self,
                "Replay Loaded Signal",
                "The loaded signal has no usable channels.",
            )
            return False

        event_signal = getattr(state, "event_signal", None)
        if event_signal is None or len(event_signal) == 0:
            QMessageBox.warning(
                self,
                "Replay Loaded Signal",
                "The loaded recording has no stimulus events to lock sweeps to.",
            )
            return False
        events_s = (
            np.asarray(event_signal.rescale(pq.s).magnitude, dtype=np.float64)
            if hasattr(event_signal, "rescale")
            else np.asarray(event_signal, dtype=np.float64)
        ).ravel()
        events_s = events_s[np.isfinite(events_s)]
        if events_s.size == 0:
            QMessageBox.warning(
                self,
                "Replay Loaded Signal",
                "The loaded stimulus events are empty or invalid.",
            )
            return False
        events_s = np.sort(events_s)

        window_ms = float(self.spin_window_ms.value())
        pre_s = max(0.0, float(self.spin_pre_ms.value()) / 1000.0)
        if pre_s >= window_ms / 1000.0:
            QMessageBox.warning(
                self,
                "Replay Loaded Signal",
                "Pre-stimulus time must be smaller than the sweep window.",
            )
            return False

        parsed_channels = self._parse_active_channels()
        wanted_names = {f"CH{idx}" for idx in parsed_channels}
        name_matches = [(name, sig) for name, sig in channels if name in wanted_names]
        if name_matches:
            selected = name_matches
        else:
            selected = [
                channels[idx] for idx in parsed_channels if 0 <= idx < len(channels)
            ]
        if not selected:
            selected = list(channels)

        rec_name = f"CH{self.spin_rec_chan.value()}"
        primary_index = next(
            (i for i, (name, _) in enumerate(selected) if name == rec_name), None
        )
        if primary_index is None:
            current_name = str(getattr(analog_signal, "name", "") or "")
            primary_index = next(
                (i for i, (name, _) in enumerate(selected) if name == current_name),
                None,
            )
        if primary_index is None:
            primary_index = 0
        selected = (
            [selected[primary_index]]
            + selected[:primary_index]
            + selected[primary_index + 1 :]
        )
        primary_name, primary_signal = selected[0]

        window_samples = max(1, int(sr * window_ms / 1000.0))

        sorted_event = Event(
            events_s * pq.s,
            name=str(getattr(event_signal, "name", "stim") or "stim"),
            units=pq.s,
        )
        latencies_per_sweep: List[Dict[str, float]] = []
        if self.chk_replay_spikes.isChecked():
            unit_names = [
                str(getattr(sg.event, "name", f"unit_{i}"))
                for i, sg in enumerate(getattr(state, "spike_groups", []) or [])
            ]
            unit_latencies = []
            window_size = (window_ms / 1000.0) * pq.s
            for sg in getattr(state, "spike_groups", []) or []:
                try:
                    lat_samples = sg.get_sweep_sample_latencies(
                        sorted_event, sr, window_size
                    )
                except Exception:
                    lat_samples = np.full(events_s.size, np.nan, dtype=np.float64)
                if len(lat_samples) != events_s.size:
                    lat_samples = np.full(events_s.size, np.nan, dtype=np.float64)
                unit_latencies.append(np.asarray(lat_samples, dtype=np.float64))

            for sweep_idx in range(events_s.size):
                latencies = {}
                for unit_name, lat_arr in zip(unit_names, unit_latencies):
                    lat_samples = float(lat_arr[sweep_idx])
                    if np.isfinite(lat_samples) and lat_samples >= 0:
                        latencies[unit_name] = lat_samples / sr
                latencies_per_sweep.append(latencies)
        else:
            latencies_per_sweep = [{} for _ in range(events_s.size)]

        self._replay_events_s = events_s
        self._replay_channels = selected
        self._replay_primary_name = primary_name
        self._replay_units = getattr(primary_signal, "units", pq.uV)
        self._replay_sr = float(sr)
        self._replay_pre_s = pre_s
        self._replay_window_ms = window_ms
        self._replay_window_samples = window_samples
        self._replay_spike_latencies = latencies_per_sweep
        self._replay_n = int(events_s.size)
        self._configure_replay_timeline()
        self.lbl_replay_info.setText(
            f"{self._replay_n} sweeps across {len(selected)} channel(s) at {sr:g} Hz."
        )
        return True

    @staticmethod
    def _signal_sampling_rate(signal) -> Optional[float]:
        sampling_rate = getattr(signal, "sampling_rate", None)
        if sampling_rate is None:
            return None
        try:
            return float(
                sampling_rate.rescale(pq.Hz).magnitude
                if hasattr(sampling_rate, "rescale")
                else sampling_rate
            )
        except Exception:
            return None

    @staticmethod
    def _signal_sample_count(signal) -> int:
        magnitude = getattr(signal, "magnitude", None)
        if magnitude is not None:
            array = np.asarray(magnitude)
            return int(array.shape[0]) if array.ndim > 0 else 0
        shape = getattr(signal, "shape", None)
        if shape:
            try:
                return int(shape[0])
            except Exception:
                return 0
        return 0

    def _build_replay_sweep(self, stim_timestamp_s: float) -> np.ndarray:
        start_s = stim_timestamp_s - self._replay_pre_s
        rows = [
            self._slice_loaded_signal(signal, start_s, self._replay_window_samples)
            for _, signal in self._replay_channels
        ]
        if len(rows) == 1:
            return rows[0]
        return np.vstack(rows)

    def _slice_loaded_signal(
        self, signal, start_s: float, n_samples: int
    ) -> np.ndarray:
        """Extract a fixed-length window, zero-padding outside the recording."""
        out = np.zeros(max(0, n_samples), dtype=np.float32)
        if n_samples <= 0:
            return out
        sr = self._replay_sr

        if hasattr(signal, "magnitude"):
            raw = np.asarray(signal.magnitude)
            t_start = getattr(signal, "t_start", 0.0)
            try:
                t_start_s = float(
                    t_start.rescale(pq.s).magnitude
                    if hasattr(t_start, "rescale")
                    else t_start
                )
            except Exception:
                t_start_s = 0.0
            index_offset = int(round((start_s - t_start_s) * sr))
        else:
            try:
                sliced = signal.time_slice(
                    start_s * pq.s, (start_s + n_samples / sr) * pq.s
                )
            except Exception:
                return out
            if hasattr(sliced, "magnitude"):
                raw = np.asarray(sliced.magnitude)
            elif hasattr(sliced, "signals"):
                parts = [
                    np.asarray(getattr(s, "magnitude", s)).reshape(-1)
                    for s in sliced.signals
                ]
                raw = np.concatenate(parts) if parts else np.empty(0)
            else:
                raw = np.asarray(sliced)
            sliced_start = getattr(sliced, "t_start", start_s * pq.s)
            try:
                sliced_start_s = float(
                    sliced_start.rescale(pq.s).magnitude
                    if hasattr(sliced_start, "rescale")
                    else sliced_start
                )
            except Exception:
                sliced_start_s = start_s
            index_offset = int(round((sliced_start_s - start_s) * sr))

        if raw.ndim > 1:
            raw = raw[:, 0]
        raw = raw.reshape(-1)

        signal_units = getattr(signal, "units", None)
        if raw.size and signal_units is not None and signal_units != self._replay_units:
            try:
                scale = float(
                    (1.0 * signal_units).rescale(self._replay_units).magnitude
                )
                if scale != 1.0:
                    raw = raw * scale
            except Exception:
                pass

        src_start = max(0, index_offset)
        src_end = min(raw.size, index_offset + n_samples)
        if src_end > src_start:
            dst_start = src_start - index_offset
            out[dst_start : dst_start + (src_end - src_start)] = raw[
                src_start:src_end
            ].astype(np.float32, copy=False)
        return out

    @Slot()
    def start_demo_stream(self):
        """Spin up the in-process synthetic microneurography Open Ephys mock server and connect."""
        if (
            self._demo_server_process is not None
            and self._demo_server_process.is_running()
        ):
            self._demo_server_process.stop()
            self._demo_server_process = None
            self.btn_start_demo.setText("Start Built-in Demo Stream")
            self.btn_start_demo.setStyleSheet(
                "background-color: #2b5c8f; color: white; font-weight: bold; padding: 6px;"
            )
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
                QMessageBox.critical(
                    self, "Demo Error", f"Could not load mock server: {e}"
                )
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
        self.btn_start_demo.setStyleSheet(
            "background-color: #d9534f; color: white; font-weight: bold; padding: 6px;"
        )

        # Start stream worker automatically
        self.start_stream()

    @Slot(list)
    def _on_sweeps_ready(self, payloads: list):
        """Handler for a batch of newly sliced sweeps.

        Each payload is processed independently: a malformed or out-of-sync
        sweep is logged and dropped instead of aborting the batch (and with
        it the whole GUI event loop) at high input rates.
        """
        for payload in payloads:
            try:
                self._on_sweep_ready(payload)
            except Exception:
                logger.exception("Dropped malformed live sweep payload")

    @Slot(dict)
    def _on_sweep_ready(self, payload: Dict[str, Any]):
        """Handler for newly sliced sweep.

        If multiple sweeps queue up while the GUI thread is busy rendering multiple windows,
        they are drained in a batch where intermediate sweeps skip redundant UI signal emissions.
        """
        self._sweep_queue.append(payload)
        if len(self._sweep_queue) == 1 and not self._sweep_drain_timer.isActive():
            # Process first sweep immediately if not already in a drain batch
            self._drain_sweep_queue()
        elif not self._sweep_drain_timer.isActive():
            self._sweep_drain_timer.start(0)

    def _drain_sweep_queue(self):
        if not self._sweep_queue:
            return
        # Drain up to 16 queued sweeps per event-loop slice so we never starve OS messages
        batch: List[Dict[str, Any]] = []
        while self._sweep_queue and len(batch) < 16:
            batch.append(self._sweep_queue.popleft())

        n_batch = len(batch)
        for idx, payload in enumerate(batch):
            is_last = (idx == n_batch - 1) and (len(self._sweep_queue) == 0)
            self._process_single_sweep_payload(payload, emit_signals=is_last)

        if self._sweep_queue and not self._sweep_drain_timer.isActive():
            self._sweep_drain_timer.start(5)

    def _process_single_sweep_payload(
        self, payload: Dict[str, Any], emit_signals: bool = True
    ):
        waveform = payload["waveform"]
        stim_ts = payload["stim_timestamp_s"]
        pre_offset_s = payload.get("pre_offset_s", 0.0)
        spikes = payload.get("spike_latencies_s", {})
        aux_sweeps = payload.get("aux_sweeps", {})
        sample_rate = payload.get("sample_rate", 30000.0)

        if hasattr(self.state, "append_sweep"):
            if (
                not getattr(self.state, "is_streaming", False)
                or self.state.event_signal is None
            ):
                active_chans = self._parse_active_channels()
                self.state.start_live_session(
                    sampling_rate=sample_rate,
                    window_size_ms=self.spin_window_ms.value(),
                    channel_name=f"CH{self.spin_rec_chan.value()}",
                    units=pq.uV,
                    aux_channels=[f"CH{c}" for c in active_chans],
                )
            try:
                self.state.append_sweep(
                    stim_timestamp=stim_ts,
                    sweep_waveform=waveform,
                    pre_offset_s=pre_offset_s,
                    spike_latencies_s=spikes,
                    aux_sweeps=aux_sweeps,
                    emit_signals=emit_signals,
                )
            except TypeError:
                self.state.append_sweep(
                    stim_timestamp=stim_ts,
                    sweep_waveform=waveform,
                    pre_offset_s=pre_offset_s,
                    spike_latencies_s=spikes,
                    aux_sweeps=aux_sweeps,
                )
        else:
            if (
                self.state.analog_signal is None
                or len(self.state.event_signal or []) == 0
            ):
                asig = AnalogSignal(
                    waveform, pq.uV, sampling_rate=sample_rate * pq.Hz, name="CH1"
                )
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

            if (
                emit_signals
                and getattr(self.state, "follow_live", True)
                and self.state.event_signal is not None
            ):
                total_sweeps = len(self.state.event_signal)
                if total_sweeps > 0 and self.state.stimno != total_sweeps - 1:
                    self.state.setStimNo(total_sweeps - 1, user_set=False)

    def _fallback_append_sweep(
        self, waveform: np.ndarray, stim_ts: float, sample_rate: float
    ):
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
            neighbors = [
                f"CH{c}" for c in active_chans if c != self.spin_rec_chan.value()
            ]
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
        try:
            if hasattr(self.state, "append_unit_spikes"):
                self.state.append_unit_spikes(unit_name, spike_ts_arr)
            else:
                for ts in spike_ts_arr:
                    self._append_spike_to_state(unit_name, float(ts))
        except Exception:
            logger.exception("Failed to record spike events for %s", unit_name)

    @Slot(str, float, dict)
    def _on_protocol_event_ready(self, event_type: str, ts: float, details: dict):
        logger.info("Protocol Event [%s] at %.3fs: %s", event_type, ts, details)
        if event_type == "ttl_role" and self.state is not None:
            details = details or {}
            purpose = str(details.get("purpose", "") or "").lower()
            evt_name = str(details.get("name", "rec.button") or "rec.button")
            seg = getattr(self.state, "segment", None)
            if seg is not None:
                existing_idx = None
                for i, ev in enumerate(getattr(seg, "events", []) or []):
                    if getattr(ev, "name", None) == evt_name:
                        existing_idx = i
                        break
                if existing_idx is not None:
                    old_ev = seg.events[existing_idx]
                    old_times = (
                        np.asarray(old_ev.rescale(pq.s).magnitude, dtype=np.float64).ravel()
                        if hasattr(old_ev, "rescale")
                        else np.asarray(old_ev, dtype=np.float64).ravel()
                    )
                    new_times = np.sort(np.append(old_times, float(ts))) * pq.s
                    updated_ev = Event(
                        new_times,
                        name=evt_name,
                        units=pq.s,
                        **dict(getattr(old_ev, "annotations", {}) or {}),
                    )
                    seg.events[existing_idx] = updated_ev
                else:
                    new_ev = Event(
                        np.array([float(ts)], dtype=np.float64) * pq.s,
                        name=evt_name,
                        units=pq.s,
                    )
                    if purpose:
                        new_ev.annotations["channel_purpose"] = purpose
                    seg.events.append(new_ev)
            if (purpose == "button" or evt_name == "rec.button") and hasattr(
                self.state, "add_protocol_marker"
            ):
                self.state.add_protocol_marker(
                    label=f"Button ({evt_name})",
                    timestamp_s=ts,
                    category="button",
                )
            elif hasattr(self.state, "onEventsUpdated"):
                self.state.onEventsUpdated.emit()

    @Slot(dict)
    def _on_stats_updated(self, stats: Dict[str, Any]):
        self.lbl_sweeps_count.setText(str(stats.get("sweeps_received", 0)))
        self.lbl_spikes_count.setText(str(stats.get("spikes_received", 0)))
        freq = stats.get("last_stim_freq_hz", 0.0)
        self.lbl_stim_freq.setText(f"{freq:.2f} Hz")

        plugin_conn = stats.get("plugin_connected", False)
        plugin_info = stats.get("plugin_info")
        if plugin_conn:
            if plugin_info and isinstance(plugin_info, dict):
                p_name = plugin_info.get("plugin", "SpikeSpy Streamer")
                sweeps = plugin_info.get("sweeps_sent", "")
                sweeps_str = f" ({sweeps} sweeps sent)" if sweeps != "" else ""
                self.lbl_plugin_detected.setText(f"● {p_name}{sweeps_str}")
                self.lbl_plugin_detected.setStyleSheet("color: #2b7a2b; font-weight: bold;")
            elif "Detected" not in self.lbl_plugin_detected.text() and "SpikeSpy" not in self.lbl_plugin_detected.text():
                self.lbl_plugin_detected.setText("● Connected")
                self.lbl_plugin_detected.setStyleSheet("color: #2b7a2b; font-weight: bold;")
        elif not self._is_streaming and self.controller and self.controller.get_status() == "DISCONNECTED":
            self.lbl_plugin_detected.setText("○ Disconnected")
            self.lbl_plugin_detected.setStyleSheet("color: gray;")
        self._sync_header_summary()

    @Slot()
    def _update_perf_diagnostics(self):
        watchdog = getattr(self.state, "ui_watchdog", None)
        scheduler = getattr(self.state, "frame_scheduler", None)
        lag_ms = watchdog.ema_loop_lag_ms if watchdog is not None else 0.0
        if scheduler is not None and watchdog is not None:
            scheduler.update_loop_lag(lag_ms)

        strain = scheduler.strain_level if scheduler is not None else 0.0
        render_ms = scheduler.ema_total_render_ms if scheduler is not None else 0.0
        eff_ms = (
            scheduler.get_effective_interval_ms(weight="heavy")
            if scheduler is not None
            else 33
        )

        if strain >= 1.5 or lag_ms >= 80.0:
            self.lbl_ui_perf.setText(
                f"High Strain (render {render_ms:.1f}ms, lag {lag_ms:.0f}ms, throttle {eff_ms}ms)"
            )
            self.lbl_ui_perf.setStyleSheet("color: #d9534f; font-weight: bold;")
        elif strain >= 0.4 or lag_ms >= 30.0:
            self.lbl_ui_perf.setText(
                f"Moderate Load (render {render_ms:.1f}ms, lag {lag_ms:.0f}ms, throttle {eff_ms}ms)"
            )
            self.lbl_ui_perf.setStyleSheet("color: #d9822b; font-weight: bold;")
        else:
            self.lbl_ui_perf.setText(
                f"Normal (render {render_ms:.1f}ms, lag {lag_ms:.1f}ms)"
            )
            self.lbl_ui_perf.setStyleSheet("color: #2b7a2b;")

        self._diag_tick = getattr(self, "_diag_tick", 0) + 1
        # Periodically check Open Ephys GUI status and auto-discover plugin (every 2.0s / 4 ticks)
        if self._diag_tick % 4 == 0 and self.controller is not None:
            oe_status = self.controller.get_status()
            if oe_status != "DISCONNECTED":
                if self.spin_target_proc.value() == 0:
                    found_id = self.controller.find_spikespy_streamer_processor()
                    if found_id is not None:
                        self.spin_target_proc.setValue(found_id)
                        self.lbl_plugin_detected.setText(f"● SpikeSpy Streamer (ID {found_id})")
                        self.lbl_plugin_detected.setStyleSheet("color: #2b7a2b; font-weight: bold;")

    @Slot()
    def _show_ui_diagnostics_dialog(self):
        scheduler = getattr(self.state, "frame_scheduler", None)
        watchdog = getattr(self.state, "ui_watchdog", None)
        summary = scheduler.get_diagnostics_summary() if scheduler is not None else {}
        reports = watchdog.reports if watchdog is not None else []

        lines = [
            f"Base Refresh Interval: {summary.get('base_interval_ms', 33)} ms",
            f"Current Strain Level: {summary.get('strain_level', 0.0)}",
            f"EMA Frame Render Time: {summary.get('ema_total_render_ms', 0.0)} ms",
            f"Event Loop Lag (EMA / Max): "
            f"{watchdog.ema_loop_lag_ms:.1f} ms / {watchdog.max_loop_lag_ms:.1f} ms"
            if watchdog is not None
            else "Event Loop Watchdog: inactive",
            f"Queued Sweeps Backlog: {len(self._sweep_queue)}",
            "",
            "--- Per-View Render Timings (Slowest First) ---",
        ]
        views = summary.get("views", {})
        if views:
            for vname, vinfo in views.items():
                lines.append(
                    f"  {vname} [{vinfo['weight']}]: EMA {vinfo['ema_ms']} ms "
                    f"(last {vinfo['last_ms']} ms, n={vinfo['count']}) -> interval {vinfo['effective_interval_ms']} ms"
                )
        else:
            lines.append("  (No view renders recorded yet)")

        lines.append("")
        lines.append(f"--- Detected Freeze / Stall Reports ({len(reports)}) ---")
        if reports:
            for idx, r in enumerate(reports[-3:], 1):
                lines.append(
                    f"\n[Stall #{idx}: {r.duration_ms:.0f} ms]\n{r.stack_trace}"
                )
        else:
            lines.append("  No main-thread freezes (>750 ms) detected.")

        QMessageBox.information(
            self,
            "SpikeSpy UI Performance & Freeze Diagnostics",
            "\n".join(lines),
        )

    @Slot()
    def _open_debug_dialog(self):
        """Open or focus the live continuous oscilloscope and trigger inspector dialog."""
        if self._debug_dialog is None:
            self._debug_dialog = OpenEphysLiveDebugDialog(self, parent=self.window())
        self._debug_dialog.show()
        self._debug_dialog.raise_()
        self._debug_dialog.activateWindow()

    @Slot(bool, str)
    def _on_connection_status_changed(self, connected: bool, message: str):
        if connected:
            self.lbl_status.setText(f"● Connected ({message})")
            self.lbl_status.setStyleSheet("color: #2b7a2b; font-weight: bold;")
        else:
            if self._is_streaming:
                self.lbl_status.setText(f"◌ Waiting ({message})")
                self.lbl_status.setStyleSheet("color: #d9822b; font-weight: bold;")
            else:
                self.lbl_status.setText(f"○ Disconnected ({message})")
                self.lbl_status.setStyleSheet("color: gray; font-weight: bold;")
        self._sync_header_summary()

    @Slot(int)
    def _on_data_port_changed(self, new_data_port: int):
        if not hasattr(self, "_prev_data_port"):
            self._prev_data_port = 5556
        if self.spin_event_port.value() == self._prev_data_port + 1:
            with QSignalBlocker(self.spin_event_port):
                self.spin_event_port.setValue(new_data_port + 1)
        self._prev_data_port = new_data_port
        self._push_live_config_to_state()

    @Slot(int)
    def _on_rec_channel_changed(self, chan: int):
        if self.worker is not None:
            self.worker.set_rec_channel(chan)

        # If f"CH{chan}" is buffered in self.state, switch the active displayed channel
        ch_name = f"CH{chan}"
        if hasattr(self.state, "set_active_channel"):
            self.state.set_active_channel(ch_name)
        self._push_live_config_to_state()

    @Slot(bool)
    def _on_filter_toggled(self, enabled: bool):
        if self.worker is not None:
            self.worker.set_filtering(enabled)
        self._push_live_config_to_state()

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

        found_streamer = None
        lines = []
        for p in procs:
            p_id = p.get("id")
            p_name = p.get("name", "Processor")
            p_type = p.get("type", "")
            lines.append(f"ID {p_id}: {p_name} ({p_type})")
            if "spikespy" in str(p_name).lower() or "spikespy" in str(p_type).lower():
                found_streamer = p_id

        if found_streamer is not None and self.spin_target_proc.value() == 0:
            self.spin_target_proc.setValue(found_streamer)
            self.lbl_plugin_detected.setText(f"● SpikeSpy Streamer (ID {found_streamer})")
            self.lbl_plugin_detected.setStyleSheet("color: #2b7a2b; font-weight: bold;")

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
            color = "#2b7a2b" if status == "ACQUIRE" else "#d9534f"
            self.lbl_oe_gui_status.setText(f"● {status}")
            self.lbl_oe_gui_status.setStyleSheet(f"color: {color}; font-weight: bold;")
        elif status == "IDLE":
            self.btn_toggle_oe_acq.setText("Start OE Acquisition")
            self.lbl_oe_gui_status.setText("◌ IDLE")
            self.lbl_oe_gui_status.setStyleSheet("color: #d9822b; font-weight: bold;")
        else:
            self.btn_toggle_oe_acq.setText("Start OE Acquisition")
            self.lbl_oe_gui_status.setText("○ Disconnected")
            self.lbl_oe_gui_status.setStyleSheet("color: gray; font-weight: bold;")
        self._sync_header_summary()

    @Slot()
    def _send_stim_voltage(self):
        v = self.spin_stim_volt.value()
        msg = f"setStimVoltage: {v}"
        ok = self.controller.send_message(msg)
        if ok:
            QMessageBox.information(self, "Stimulus Voltage", f"Sent: {msg}")
        else:
            QMessageBox.warning(
                self, "Stimulus Voltage", "Failed to send message to Open Ephys."
            )

    @Slot(dict)
    def _on_unit_annotated(self, annotation_info: Dict[str, Any]):
        if not self.chk_sync_annotations.isChecked():
            return
        dedup_key = (
            annotation_info.get("unit_idx", annotation_info.get("unit")),
            annotation_info.get("stimno"),
            annotation_info.get("latency_samples"),
        )
        self._last_synced_annotation_key = dedup_key
        is_automated = bool(annotation_info.get("automated", False))
        if (
            self._is_streaming
            and is_automated
            and hasattr(self.controller, "sync_unit_annotation_async")
        ):
            self.controller.sync_unit_annotation_async(annotation_info)
        else:
            self.controller.sync_unit_annotation(annotation_info)

    @Slot(int)
    def _on_unit_changed_in_state(self, stimno: int):
        if not self.chk_sync_annotations.isChecked():
            return
        sg = self.state.getUnitGroup()
        if sg is None:
            return

        sr = 10000.0
        if (
            hasattr(self.state, "sampling_rate")
            and self.state.sampling_rate is not None
        ):
            sr = (
                float(self.state.sampling_rate.rescale(pq.Hz).magnitude)
                if hasattr(self.state.sampling_rate, "magnitude")
                else float(self.state.sampling_rate)
            )

        latency_samples = None
        latency_ms = None
        if getattr(sg, "idx_arr", None) is not None and 0 <= stimno < len(sg.idx_arr):
            item = sg.idx_arr[stimno]
            if item is not None:
                latency_samples = (
                    item[0] if isinstance(item, (tuple, list)) else int(item)
                )
                latency_ms = (latency_samples / sr) * 1000.0
        elif (
            self.state.event_signal is not None
            and stimno < len(self.state.event_signal)
            and hasattr(sg, "get_latencies")
        ):
            cur_ev = self.state.event_signal[stimno]
            lat = sg.get_latencies(np.array([cur_ev.magnitude]) * cur_ev.units)[0]
            if not np.isnan(lat):
                sec = (
                    float(lat.rescale(pq.s).magnitude)
                    if hasattr(lat, "rescale")
                    else float(lat)
                )
                latency_samples = int(round(sec * sr))
                latency_ms = (
                    float(lat.rescale(pq.ms).magnitude)
                    if hasattr(lat, "rescale")
                    else (sec * 1000.0)
                )

        if latency_samples is None:
            return

        dedup_key = (self.state.cur_spike_group, int(stimno), int(latency_samples))
        if dedup_key == self._last_synced_annotation_key:
            # Already handled by _on_unit_annotated from the same setUnit() call
            return
        self._last_synced_annotation_key = dedup_key

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
        if self._is_streaming and hasattr(
            self.controller, "sync_unit_annotation_async"
        ):
            self.controller.sync_unit_annotation_async(info)
        else:
            self.controller.sync_unit_annotation(info)

    def closeEvent(self, event):
        self._perf_timer.stop()
        self._sweep_drain_timer.stop()
        self._sweep_queue.clear()
        self.stop_stream()
        if self._demo_server_process is not None:
            self._demo_server_process.stop()
            self._demo_server_process = None
        super().closeEvent(event)
