"""Live Stream & Trigger Debug Window for Open Ephys in SpikeSpy.

Provides real-time continuous oscilloscope visualization of raw incoming
multichannel streaming data, live monitoring of TTL/crossing-detector events,
sample clock alignment diagnostics, and manual trigger controls.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

try:
    import pyqtgraph as pg
    _HAS_PG = True
except Exception:
    _HAS_PG = False

logger = logging.getLogger(__name__)


class OpenEphysLiveDebugDialog(QDialog):
    """Non-modal real-time continuous oscilloscope and trigger inspector."""

    def __init__(self, live_view, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.live_view = live_view
        self.setWindowTitle("SpikeSpy – Open Ephys Live Stream & Trigger Debugger")
        self.resize(980, 720)

        self._refresh_timer = QTimer(self)
        self._refresh_timer.setInterval(40)  # ~25 FPS
        self._refresh_timer.timeout.connect(self._on_refresh_tick)

        self._recent_events: List[Dict[str, Any]] = []
        self._max_table_rows = 100

        self._init_ui()
        self._connect_worker()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # -------------------------------------------------------------
        # 1. Top Control & Status Toolbar
        # -------------------------------------------------------------
        top_bar = QHBoxLayout()

        self.lbl_conn_info = QLabel("Status: Connecting...")
        self.lbl_conn_info.setStyleSheet("font-weight: bold; color: #2b5c8f;")
        top_bar.addWidget(self.lbl_conn_info)

        top_bar.addStretch()

        self.lbl_chan = QLabel("View Channel:")
        self.combo_channel = QComboBox()
        self.combo_channel.addItems([f"CH {i}" for i in range(16)])
        self.combo_channel.currentIndexChanged.connect(self._on_channel_changed)
        top_bar.addWidget(self.lbl_chan)
        top_bar.addWidget(self.combo_channel)

        self.lbl_dur = QLabel("Time Window:")
        self.combo_duration = QComboBox()
        self.combo_duration.addItems(["0.5 s", "1.0 s", "2.0 s", "5.0 s"])
        self.combo_duration.setCurrentIndex(1)  # 1.0s default
        top_bar.addWidget(self.lbl_dur)
        top_bar.addWidget(self.combo_duration)

        self.chk_auto_align = QCheckBox("Auto-Align Trigger Sample Clock")
        self.chk_auto_align.setChecked(True)
        self.chk_auto_align.setToolTip(
            "Automatically map relative or desynchronized Crossing Detector sample numbers "
            "to the continuous stream head so sweeps are never dropped due to clock skew."
        )
        self.chk_auto_align.toggled.connect(self._on_auto_align_toggled)
        top_bar.addWidget(self.chk_auto_align)

        self.btn_force_trigger = QPushButton("⚡ Force Trigger / Cut Sweep Now")
        self.btn_force_trigger.setStyleSheet(
            "background-color: #f0ad4e; color: #222; font-weight: bold; padding: 5px 12px;"
        )
        self.btn_force_trigger.setToolTip(
            "Immediately slice a sweep from the current buffer position and send it to SpikeSpy's viewer. "
            "Use this to verify sweep cutting and rendering independently of incoming TTL pulses."
        )
        self.btn_force_trigger.clicked.connect(self._on_force_trigger_clicked)
        top_bar.addWidget(self.btn_force_trigger)

        layout.addLayout(top_bar)

        # Main splitter (Top: Oscilloscope, Bottom: Trigger & Event Inspector)
        splitter = QSplitter(Qt.Vertical)

        # -------------------------------------------------------------
        # 2. Continuous Oscilloscope Group
        # -------------------------------------------------------------
        osc_group = QGroupBox("Live Continuous Signal Oscilloscope (Raw Data Ingest)")
        osc_layout = QVBoxLayout(osc_group)

        if _HAS_PG:
            pg.setConfigOptions(antialias=True)
            self.plot_widget = pg.PlotWidget()
            self.plot_widget.setBackground("#181818")
            self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
            self.plot_widget.setLabel("left", "Amplitude", units="uV")
            self.plot_widget.setLabel("bottom", "Time relative to current head", units="s")
            self.curve = self.plot_widget.plot(pen=pg.mkPen(color="#00e5ff", width=1.5))
            osc_layout.addWidget(self.plot_widget)
        else:
            self.lbl_no_pg = QLabel("PyQtGraph is not available. Raw waveform plotting disabled.")
            osc_layout.addWidget(self.lbl_no_pg)

        # Metrics bar below plot
        metrics_bar = QHBoxLayout()
        self.lbl_stats_pk = QLabel("Vp-p: -- uV")
        self.lbl_stats_rms = QLabel("RMS: -- uV")
        self.lbl_stats_rate = QLabel("Sample Rate: -- Hz")
        self.lbl_stats_packets = QLabel("Packets: 0 (0.0 pkt/s)")
        self.lbl_stats_ring = QLabel("Buffer Head: --")

        font = QFont("Monospace", 9)
        for lbl in (self.lbl_stats_pk, self.lbl_stats_rms, self.lbl_stats_rate, self.lbl_stats_packets, self.lbl_stats_ring):
            lbl.setFont(font)
            metrics_bar.addWidget(lbl)
        metrics_bar.addStretch()
        osc_layout.addLayout(metrics_bar)

        splitter.addWidget(osc_group)

        # -------------------------------------------------------------
        # 3. Live Trigger & Event Diagnostics Group
        # -------------------------------------------------------------
        diag_group = QGroupBox("Trigger & Event Inspector (Crossing Detector / TTL Diagnostics)")
        diag_layout = QVBoxLayout(diag_group)

        # Table showing recent packets / events
        self.tbl_events = QTableWidget()
        self.tbl_events.setColumnCount(9)
        self.tbl_events.setHorizontalHeaderLabels([
            "Time", "Topic", "Type", "Line", "State", "Event Sample #", "RingBuffer Sample #", "Clock Skew", "Status / Action"
        ])
        header = self.tbl_events.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(7, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(8, QHeaderView.Stretch)
        self.tbl_events.setSelectionBehavior(QTableWidget.SelectRows)
        self.tbl_events.setSelectionMode(QTableWidget.SingleSelection)
        self.tbl_events.itemSelectionChanged.connect(self._on_table_row_selected)
        diag_layout.addWidget(self.tbl_events, 2)

        # Raw payload inspection text box
        self.txt_payload = QTextEdit()
        self.txt_payload.setReadOnly(True)
        self.txt_payload.setMaximumHeight(80)
        self.txt_payload.setFont(QFont("Monospace", 8))
        self.txt_payload.setPlaceholderText("Select any event in the table above to view its full JSON payload...")
        diag_layout.addWidget(self.txt_payload, 1)

        btn_bar = QHBoxLayout()
        self.btn_clear_events = QPushButton("Clear Event Log")
        self.btn_clear_events.clicked.connect(self._clear_events)
        btn_bar.addWidget(self.btn_clear_events)
        btn_bar.addStretch()

        diag_layout.addLayout(btn_bar)

        splitter.addWidget(diag_group)
        layout.addWidget(splitter)

        # Splitter sizing: 55% oscilloscope, 45% event inspector
        splitter.setSizes([380, 280])

    def showEvent(self, event):
        super().showEvent(event)
        self._connect_worker()
        self._refresh_timer.start()

    def hideEvent(self, event):
        self._refresh_timer.stop()
        super().hideEvent(event)

    def _connect_worker(self):
        worker = getattr(self.live_view, "worker", None)
        if worker is not None and getattr(self, "_connected_worker", None) is not worker:
            if hasattr(worker, "debugEventEmitted"):
                worker.debugEventEmitted.connect(self._on_debug_event)
            self._connected_worker = worker

    @Slot(dict)
    def _on_debug_event(self, evt: Dict[str, Any]):
        """Receive a live debug event from OpenEphysStreamWorker."""
        self._recent_events.append(evt)
        if len(self._recent_events) > self._max_table_rows:
            self._recent_events.pop(0)

        # Insert row in table
        row = self.tbl_events.rowCount()
        if row >= self._max_table_rows:
            self.tbl_events.removeRow(0)
            row = self.tbl_events.rowCount()

        self.tbl_events.insertRow(row)

        t_str = time.strftime("%H:%M:%S", time.localtime(evt.get("timestamp", time.time())))
        topic = str(evt.get("topic", ""))
        evt_type = str(evt.get("type", ""))
        line = str(evt.get("line", "--"))
        state = str(evt.get("state", "--"))
        evt_samp = str(evt.get("sample_number", "--"))
        buf_samp = str(evt.get("buffer_latest_sample", "--"))
        skew = str(evt.get("clock_skew", "--"))
        status = str(evt.get("status", ""))

        items = [
            QTableWidgetItem(t_str),
            QTableWidgetItem(topic),
            QTableWidgetItem(evt_type),
            QTableWidgetItem(line),
            QTableWidgetItem(state),
            QTableWidgetItem(evt_samp),
            QTableWidgetItem(buf_samp),
            QTableWidgetItem(skew),
            QTableWidgetItem(status),
        ]

        # Color coding
        color = QColor("#ffffff")
        if "QUEUED" in status or "SLICED" in status or "FORCE" in status:
            color = QColor("#c8e6c9")  # soft green
        elif "DROPPED" in status or "SKEW" in status or "ERROR" in status:
            color = QColor("#ffcdd2")  # soft red
        elif "IGNORED" in status:
            color = QColor("#eeeeee")  # gray

        for item in items:
            item.setBackground(color)

        for col, item in enumerate(items):
            self.tbl_events.setItem(row, col, item)

        self.tbl_events.scrollToBottom()

    def _on_table_row_selected(self):
        sel = self.tbl_events.selectedIndexes()
        if not sel:
            return
        row = sel[0].row()
        if 0 <= row < len(self._recent_events):
            evt_dict = self._recent_events[row]
            self.txt_payload.setText(json.dumps(evt_dict, indent=2))

    def _clear_events(self):
        self._recent_events.clear()
        self.tbl_events.setRowCount(0)
        self.txt_payload.clear()

    def _on_channel_changed(self):
        pass

    def _on_auto_align_toggled(self, checked: bool):
        worker = getattr(self.live_view, "worker", None)
        if worker is not None:
            worker.auto_align_clock = checked

    def _on_force_trigger_clicked(self):
        worker = getattr(self.live_view, "worker", None)
        if worker is not None and hasattr(worker, "force_trigger"):
            success = worker.force_trigger()
            if success:
                logger.info("Manual force trigger sent to OpenEphysStreamWorker.")
            else:
                logger.warning("Force trigger failed: continuous buffer has no samples yet.")
        else:
            logger.warning("Worker is not running or does not support force_trigger.")

    def _on_refresh_tick(self):
        """Update live oscilloscope waveform and stream metrics."""
        self._connect_worker()
        worker = getattr(self.live_view, "worker", None)
        if worker is None or not getattr(worker, "_running", False):
            self.lbl_conn_info.setText("Status: Worker Stopped (Click 'Start Live Stream' in main panel)")
            self.lbl_conn_info.setStyleSheet("color: #d9534f; font-weight: bold;")
            return

        self.lbl_conn_info.setText(f"Status: Streaming from {worker.host}:{worker.data_port} (Mode: {worker.mode})")
        self.lbl_conn_info.setStyleSheet("color: #2b7a2b; font-weight: bold;")

        # Update available channels in combo
        avail_chans = worker.ring_buffer.n_channels
        if self.combo_channel.count() != avail_chans:
            cur = self.combo_channel.currentIndex()
            self.combo_channel.blockSignals(True)
            self.combo_channel.clear()
            self.combo_channel.addItems([f"CH {i}" for i in range(avail_chans)])
            self.combo_channel.setCurrentIndex(min(cur, avail_chans - 1))
            self.combo_channel.blockSignals(False)

        # Get duration
        dur_text = self.combo_duration.currentText().split()[0]
        try:
            dur_s = float(dur_text)
        except ValueError:
            dur_s = 1.0

        chan = self.combo_channel.currentIndex()
        if hasattr(worker, "get_recent_continuous"):
            data_pair = worker.get_recent_continuous(channel=chan, duration_s=dur_s)
        else:
            data_pair = None

        if data_pair is not None and _HAS_PG:
            t, sig = data_pair
            if len(sig) > 0:
                self.curve.setData(t, sig)
                pk_val = float(np.ptp(sig))
                rms_val = float(np.sqrt(np.mean(sig ** 2)))
                self.lbl_stats_pk.setText(f"Vp-p: {pk_val:.1f} uV")
                self.lbl_stats_rms.setText(f"RMS: {rms_val:.1f} uV")

        # Ingestion metrics
        pkts = getattr(worker, "_packets_count", 0)
        sweeps = getattr(worker, "_sweeps_count", 0)
        sr = getattr(worker, "sample_rate", 30000.0)
        self.lbl_stats_rate.setText(f"Sample Rate: {sr:.0f} Hz")
        self.lbl_stats_packets.setText(f"Packets: {pkts} | Sweeps Sliced: {sweeps}")

        latest = worker.ring_buffer.latest_sample
        earliest = worker.ring_buffer.earliest_sample
        if latest is not None and earliest is not None:
            self.lbl_stats_ring.setText(f"Buffer: [{earliest} .. {latest}] ({latest - earliest} samples)")
        else:
            self.lbl_stats_ring.setText("Buffer: Empty")
