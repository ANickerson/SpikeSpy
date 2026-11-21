from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import quantities as pq
from neo import Event
from PySide6.QtCore import QSignalBlocker, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .ChannelGroupBar import ChannelGroupDialog
from .dsp.detectors import DetectorParamSpec, SpikeDetectorRegistry
from .helpers import qsignal_throttle
from .ViewerState import ViewerState, tracked_neuron_unit

# Qualitative palette matching MultiTracePGView.SET2_COLORS
SET2_COLORS = [
    (102, 194, 165),
    (252, 141, 98),
    (141, 160, 203),
    (231, 138, 195),
    (166, 216, 84),
    (255, 217, 47),
    (229, 196, 148),
    (179, 179, 179),
]

_CORE_PARAM_NAMES = {"window_ms", "window_size", "threshold", "max_skip", "track_limit"}

FIBRE_TYPE_PRESETS = [
    "",
    "CM",
    "CMH",
    "CMi",
    "CMiHi",
    "CH",
    "A-beta",
    "A-delta",
    "Sympathetic",
    "C-fibre",
]

_HEALTH_STYLES = {
    "locked": (
        "background: #dcfce7; color: #166534; border: 1px solid #86efac; "
        "border-radius: 10px; padding: 2px 8px; font-weight: bold;"
    ),
    "skipping": (
        "background: #fef9c3; color: #854d0e; border: 1px solid #fde047; "
        "border-radius: 10px; padding: 2px 8px; font-weight: bold;"
    ),
    "lost": (
        "background: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; "
        "border-radius: 10px; padding: 2px 8px; font-weight: bold;"
    ),
    "neutral": (
        "background: #f1f5f9; color: #475569; border: 1px solid #cbd5e1; "
        "border-radius: 10px; padding: 2px 8px; font-weight: bold;"
    ),
}


def _default_unit_hex_color(unit_idx: int) -> str:
    rgb = SET2_COLORS[max(0, int(unit_idx)) % len(SET2_COLORS)]
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


class ActiveUnitInfoView(QWidget):
    """Unified inspector & controller for the active (or pinned) tracked neuron unit.

    Can be used both as a standalone dockable panel (``window_options["ActiveUnitInfo"]``)
    and embedded inside ``ViewSettingsPanel``.
    """

    unitSelected = Signal(int)

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        state: Optional[ViewerState] = None,
        compact: bool = False,
    ) -> None:
        super().__init__(parent)
        self.state: Optional[ViewerState] = None
        self._compact: bool = bool(compact)
        self._pinned_unit_idx: Optional[int] = None
        self._target_unit_override: Optional[int] = None
        self._syncing: bool = False
        self._updating_ui: bool = False
        self._state_connections: List[tuple] = []
        self.extra_param_controls: Dict[str, QWidget] = {}

        self.setWindowTitle("Active Unit Info")

        main_layout = QVBoxLayout(self)
        margin = 4 if self._compact else 6
        spacing = 4 if self._compact else 6
        main_layout.setContentsMargins(margin, margin, margin, margin)
        main_layout.setSpacing(spacing)

        # ------------------------------------------------------------------
        # 0. Unit Selector Bar
        # ------------------------------------------------------------------
        unit_bar = QHBoxLayout()
        unit_bar.setSpacing(4)
        self.combo_unit = QComboBox(self)
        self.combo_unit.setToolTip(
            "Follow the active view / global unit or pin this inspector to a specific unit."
        )
        self.combo_unit.currentIndexChanged.connect(self._on_unit_combo_changed)
        unit_bar.addWidget(self.combo_unit, 1)

        self.btn_new_unit = QPushButton("+ New Unit", self)
        self.btn_new_unit.setToolTip("Add a new tracked unit group and select it")
        self.btn_new_unit.clicked.connect(self._on_new_unit_clicked)
        unit_bar.addWidget(self.btn_new_unit)
        main_layout.addLayout(unit_bar)

        # ------------------------------------------------------------------
        # 1. Unit Color & Health Header Card
        # ------------------------------------------------------------------
        self.header_card = QFrame(self)
        self.header_card.setFrameShape(QFrame.StyledPanel)
        self.header_card.setStyleSheet(
            "QFrame { background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 6px; }"
        )
        card_layout = QVBoxLayout(self.header_card)
        card_layout.setContentsMargins(6, 6, 6, 6)
        card_layout.setSpacing(4)

        top_row = QHBoxLayout()
        top_row.setSpacing(6)

        self.lbl_color_swatch = QLabel(self.header_card)
        self.lbl_color_swatch.setFixedSize(16, 16)
        self.lbl_color_swatch.setStyleSheet(
            "background: #b3b3b3; border: 1px solid rgba(0,0,0,0.25); border-radius: 4px;"
        )
        top_row.addWidget(self.lbl_color_swatch)

        self.lbl_unit_title = QLabel("<b>No Unit</b>", self.header_card)
        self.lbl_unit_title.setStyleSheet("border: none; background: transparent;")
        top_row.addWidget(self.lbl_unit_title, 1)

        self.lbl_health_badge = QLabel("EMPTY", self.header_card)
        self.lbl_health_badge.setAlignment(Qt.AlignCenter)
        self.lbl_health_badge.setStyleSheet(_HEALTH_STYLES["neutral"])
        top_row.addWidget(self.lbl_health_badge)

        card_layout.addLayout(top_row)

        self.lbl_metrics = QLabel("Spikes: 0  |  Hit: --  |  Lat: --  |  Δ --", self.header_card)
        self.lbl_metrics.setWordWrap(True)
        self.lbl_metrics.setStyleSheet(
            "border: none; background: transparent; color: #334155; font-size: 11px;"
        )
        card_layout.addWidget(self.lbl_metrics)
        main_layout.addWidget(self.header_card)

        # Quick Candidate Triage Bar (visible only when previewing an active candidate)
        self.candidate_actions_widget = QWidget(self)
        cand_act_layout = QHBoxLayout(self.candidate_actions_widget)
        cand_act_layout.setContentsMargins(0, 0, 0, 0)
        cand_act_layout.setSpacing(6)
        self.btn_accept_candidate = QPushButton("✓ Accept Candidate", self.candidate_actions_widget)
        self.btn_accept_candidate.setToolTip("Promote this candidate to a tracked unit")
        self.btn_accept_candidate.clicked.connect(self._on_accept_candidate_clicked)
        self.btn_discard_candidate = QPushButton("✕ Discard Candidate", self.candidate_actions_widget)
        self.btn_discard_candidate.setToolTip("Discard this candidate proposal")
        self.btn_discard_candidate.clicked.connect(self._on_discard_candidate_clicked)
        cand_act_layout.addWidget(self.btn_accept_candidate, 1)
        cand_act_layout.addWidget(self.btn_discard_candidate, 1)
        self.candidate_actions_widget.setVisible(False)
        main_layout.addWidget(self.candidate_actions_widget)

        # ------------------------------------------------------------------
        # 2. Identity & Metadata Section
        # ------------------------------------------------------------------
        self.grp_identity = QGroupBox("Identity & Metadata", self)
        id_form = QFormLayout(self.grp_identity)
        id_form.setContentsMargins(6, 6, 6, 6)
        id_form.setSpacing(4)

        self.edit_name = QLineEdit(self.grp_identity)
        self.edit_name.setPlaceholderText("Unit name (e.g. unit_0)")
        self.edit_name.editingFinished.connect(self._on_identity_changed)
        id_form.addRow("Name:", self.edit_name)

        self.combo_fibre_type = QComboBox(self.grp_identity)
        self.combo_fibre_type.setEditable(True)
        self.combo_fibre_type.addItems(FIBRE_TYPE_PRESETS)
        self.combo_fibre_type.currentTextChanged.connect(self._on_identity_changed)
        id_form.addRow("Fibre Type:", self.combo_fibre_type)

        self.edit_notes = QLineEdit(self.grp_identity)
        self.edit_notes.setPlaceholderText("Notes / annotations")
        self.edit_notes.editingFinished.connect(self._on_identity_changed)
        id_form.addRow("Notes:", self.edit_notes)

        main_layout.addWidget(self.grp_identity)

        # ------------------------------------------------------------------
        # 3. Channel Footprint Section
        # ------------------------------------------------------------------
        self.grp_channels = QGroupBox("Channel Footprint", self)
        ch_vbox = QVBoxLayout(self.grp_channels)
        ch_vbox.setContentsMargins(6, 6, 6, 6)
        ch_vbox.setSpacing(4)

        ch_primary_row = QHBoxLayout()
        ch_primary_row.addWidget(QLabel("Primary:", self.grp_channels))
        self.combo_primary_channel = QComboBox(self.grp_channels)
        self.combo_primary_channel.currentIndexChanged.connect(
            self._on_primary_channel_changed
        )
        ch_primary_row.addWidget(self.combo_primary_channel, 1)
        ch_vbox.addLayout(ch_primary_row)

        ch_foot_row = QHBoxLayout()
        ch_foot_row.setSpacing(4)
        self.lbl_neighbors = QLabel("Neighbors: (none)", self.grp_channels)
        self.lbl_neighbors.setWordWrap(True)
        self.lbl_neighbors.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        ch_foot_row.addWidget(self.lbl_neighbors, 1)

        self.btn_shift_prev = QPushButton("<", self.grp_channels)
        self.btn_shift_prev.setFixedWidth(24)
        self.btn_shift_prev.setToolTip("Shift unit channel group by -1 channel")
        self.btn_shift_prev.clicked.connect(lambda: self._shift_unit_channels(-1))
        ch_foot_row.addWidget(self.btn_shift_prev)

        self.btn_shift_next = QPushButton(">", self.grp_channels)
        self.btn_shift_next.setFixedWidth(24)
        self.btn_shift_next.setToolTip("Shift unit channel group by +1 channel")
        self.btn_shift_next.clicked.connect(lambda: self._shift_unit_channels(1))
        ch_foot_row.addWidget(self.btn_shift_next)

        self.btn_edit_channels = QPushButton("Channels...", self.grp_channels)
        self.btn_edit_channels.setToolTip("Edit primary and neighbor channels for this unit")
        self.btn_edit_channels.clicked.connect(self._open_channel_dialog)
        ch_foot_row.addWidget(self.btn_edit_channels)

        ch_vbox.addLayout(ch_foot_row)
        main_layout.addWidget(self.grp_channels)

        # ------------------------------------------------------------------
        # 4. Tracking & Detection Parameters Section
        # ------------------------------------------------------------------
        self.grp_tracking = QGroupBox("Tracking & Detection", self)
        tr_form = QFormLayout(self.grp_tracking)
        tr_form.setContentsMargins(6, 6, 6, 6)
        tr_form.setSpacing(4)

        self.chk_auto_track = QCheckBox("Live Auto-Track", self.grp_tracking)
        self.chk_auto_track.toggled.connect(self._on_tracking_params_changed)
        tr_form.addRow(self.chk_auto_track)

        self.combo_algorithm = QComboBox(self.grp_tracking)
        self.refresh_algorithms()
        self.combo_algorithm.currentIndexChanged.connect(self._on_algorithm_changed)
        tr_form.addRow("Algorithm:", self.combo_algorithm)

        self.spin_window_ms = QDoubleSpinBox(self.grp_tracking)
        self.spin_window_ms.setRange(0.1, 500.0)
        self.spin_window_ms.setDecimals(1)
        self.spin_window_ms.setSingleStep(0.5)
        self.spin_window_ms.setSuffix(" ms")
        self.spin_window_ms.setValue(5.0)
        self.spin_window_ms.valueChanged.connect(self._on_tracking_params_changed)
        tr_form.addRow("Window:", self.spin_window_ms)

        thresh_row = QHBoxLayout()
        thresh_row.setSpacing(4)
        self.spin_threshold_mv = QDoubleSpinBox(self.grp_tracking)
        self.spin_threshold_mv.setRange(-1e6, 1e6)
        self.spin_threshold_mv.setDecimals(4)
        self.spin_threshold_mv.setSingleStep(0.01)
        self.spin_threshold_mv.setSuffix(" mV")
        self.spin_threshold_mv.setValue(0.01)
        self.spin_threshold_mv.valueChanged.connect(self._on_tracking_params_changed)
        thresh_row.addWidget(self.spin_threshold_mv, 1)

        self.btn_auto_threshold = QPushButton("Auto (80%)", self.grp_tracking)
        self.btn_auto_threshold.setToolTip(
            "Estimate threshold as 80% of the latest annotated spike amplitude"
        )
        self.btn_auto_threshold.clicked.connect(self._on_auto_threshold_clicked)
        thresh_row.addWidget(self.btn_auto_threshold)
        tr_form.addRow("Threshold:", thresh_row)

        self.spin_max_skips = QSpinBox(self.grp_tracking)
        self.spin_max_skips.setRange(0, 50)
        self.spin_max_skips.setValue(1)
        self.spin_max_skips.valueChanged.connect(self._on_tracking_params_changed)
        tr_form.addRow("Max Skips:", self.spin_max_skips)

        self.spin_track_limit = QSpinBox(self.grp_tracking)
        self.spin_track_limit.setRange(0, 10000)
        self.spin_track_limit.setValue(0)
        self.spin_track_limit.valueChanged.connect(self._on_tracking_params_changed)
        tr_form.addRow("Track Limit:", self.spin_track_limit)

        self.extra_params_widget = QWidget(self.grp_tracking)
        self.extra_params_layout = QFormLayout(self.extra_params_widget)
        self.extra_params_layout.setContentsMargins(0, 0, 0, 0)
        self.extra_params_layout.setSpacing(4)
        tr_form.addRow(self.extra_params_widget)

        main_layout.addWidget(self.grp_tracking)

        # ------------------------------------------------------------------
        # 5. Quick Actions Row
        # ------------------------------------------------------------------
        actions_row = QHBoxLayout()
        actions_row.setSpacing(4)

        self.btn_track_now = QPushButton("Track Forward", self)
        self.btn_track_now.setToolTip("Track forward from the current stimulus sweep")
        self.btn_track_now.clicked.connect(self._on_track_forward_clicked)
        actions_row.addWidget(self.btn_track_now)

        self.btn_fill_missing = QPushButton("Fill Missing", self)
        self.btn_fill_missing.setToolTip("Fill gaps between existing spike annotations")
        self.btn_fill_missing.clicked.connect(self._on_fill_missing_clicked)
        actions_row.addWidget(self.btn_fill_missing)

        self.btn_rebuild_template = QPushButton("Rebuild Template", self)
        self.btn_rebuild_template.setToolTip("Rebuild multi-channel template matcher for this unit")
        self.btn_rebuild_template.clicked.connect(self._on_rebuild_template_clicked)
        actions_row.addWidget(self.btn_rebuild_template)

        main_layout.addLayout(actions_row)
        if not self._compact:
            main_layout.addStretch(1)

        self._rebuild_extra_param_controls(self.get_selected_algorithm())

        self._sync_throttled = qsignal_throttle(
            self.sync_from_unit, 150, parent=self, weight="light"
        )

        if state is not None:
            self.set_state(state)
        else:
            self.sync_from_unit()

    # ------------------------------------------------------------------
    # State Connection & Lifecycle
    # ------------------------------------------------------------------
    def unset_connections(self) -> None:
        for sig, slot in list(getattr(self, "_state_connections", [])):
            try:
                sig.disconnect(slot)
            except (TypeError, RuntimeError):
                pass
        self._state_connections.clear()

    def closeEvent(self, event) -> None:
        self.unset_connections()
        super().closeEvent(event)

    def setState(self, state: Optional[ViewerState]) -> None:
        self.set_state(state)

    def set_state(self, state: Optional[ViewerState]) -> None:
        self.unset_connections()
        self.state = state
        if self.state is not None:
            for sig_name in (
                "onUnitGroupChange",
                "onActiveUnitChanged",
                "onUnitListChanged",
                "onUnitParamsChanged",
                "onUnitChange",
                "onNewSpikeEvents",
                "onNewSweep",
                "onLoadNewFile",
                "onChannelGroupsChanged",
                "onCandidateSelectionChanged",
                "onCandidatesChanged",
            ):
                sig = getattr(self.state, sig_name, None)
                if sig is not None and hasattr(sig, "connect"):
                    try:
                        sig.connect(self._on_state_signal)
                        self._state_connections.append((sig, self._on_state_signal))
                    except (TypeError, RuntimeError):
                        pass
        self.sync_from_unit()

    @Slot()
    @Slot(int)
    @Slot(int, int)
    @Slot(str)
    def _on_state_signal(self, *args) -> None:
        if self._updating_ui:
            return
        if self.state is not None and getattr(self.state, "is_streaming", False):
            self._sync_throttled()
        else:
            self.sync_from_unit()

    # ------------------------------------------------------------------
    # Target Unit Resolution & Settings Persistence
    # ------------------------------------------------------------------
    def _get_active_candidate(self):
        """Return the active CandidateUnit when following global active selection."""
        if (
            self.state is None
            or self._pinned_unit_idx is not None
            or self._target_unit_override is not None
        ):
            return None
        if hasattr(self.state, "get_active_candidate"):
            return self.state.get_active_candidate()
        return None

    def set_target_unit(self, unit_idx: Optional[int]) -> None:
        """Point this panel at a view-bound unit index (or None to follow global)."""
        self._target_unit_override = (
            int(unit_idx) if unit_idx is not None else None
        )
        self.sync_from_unit()

    def get_target_unit_idx(self) -> Optional[int]:
        if self.state is None:
            return None
        groups = getattr(self.state, "spike_groups", None) or []
        if not groups:
            return None

        if self._pinned_unit_idx is not None:
            if 0 <= self._pinned_unit_idx < len(groups):
                return self._pinned_unit_idx
            self._pinned_unit_idx = None

        if self._target_unit_override is not None:
            if 0 <= self._target_unit_override < len(groups):
                return self._target_unit_override

        cur = int(getattr(self.state, "cur_spike_group", 0) or 0)
        if 0 <= cur < len(groups):
            return cur
        return 0

    def get_target_unit(self) -> Optional[tracked_neuron_unit]:
        idx = self.get_target_unit_idx()
        if idx is None or self.state is None:
            return None
        groups = getattr(self.state, "spike_groups", None) or []
        if 0 <= idx < len(groups):
            return groups[idx]
        return None

    def get_settings(self) -> Dict[str, Any]:
        return {
            "pinned_unit_idx": self._pinned_unit_idx,
            "compact": self._compact,
        }

    def set_settings(self, settings: Optional[Dict[str, Any]]) -> None:
        if not isinstance(settings, dict):
            return
        pinned = settings.get("pinned_unit_idx", None)
        if pinned is not None:
            try:
                self._pinned_unit_idx = int(pinned)
            except (TypeError, ValueError):
                self._pinned_unit_idx = None
        else:
            self._pinned_unit_idx = None
        self.sync_from_unit()

    # ------------------------------------------------------------------
    # Algorithm & Extra Parameter Schema Helpers
    # ------------------------------------------------------------------
    def refresh_algorithms(self) -> None:
        current_algo = self.get_selected_algorithm()
        with QSignalBlocker(self.combo_algorithm):
            self.combo_algorithm.clear()
            for plugin in SpikeDetectorRegistry.list_detectors():
                self.combo_algorithm.addItem(plugin.display_name, plugin.name)
            idx = self.combo_algorithm.findData(current_algo)
            if idx >= 0:
                self.combo_algorithm.setCurrentIndex(idx)

    def get_selected_algorithm(self) -> str:
        if hasattr(self, "combo_algorithm") and self.combo_algorithm.count() > 0:
            data = self.combo_algorithm.currentData()
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
            self.extra_params_layout.addRow(f"{spec.label}:", ctrl)

        self.extra_params_widget.setVisible(len(self.extra_param_controls) > 0)

    def _create_param_widget(self, spec: DetectorParamSpec) -> QWidget:
        if spec.param_type == "bool":
            w = QCheckBox(self.extra_params_widget)
            w.setChecked(bool(spec.default))
            w.toggled.connect(self._on_tracking_params_changed)
            return w
        if spec.param_type == "choice":
            w = QComboBox(self.extra_params_widget)
            for ch in spec.choices or []:
                w.addItem(str(ch), ch)
            idx = w.findData(spec.default)
            if idx >= 0:
                w.setCurrentIndex(idx)
            w.currentIndexChanged.connect(self._on_tracking_params_changed)
            return w
        if spec.param_type == "int":
            w = QSpinBox(self.extra_params_widget)
            w.setRange(int(spec.min_val), int(spec.max_val))
            w.setSingleStep(max(1, int(spec.step)))
            w.setValue(int(spec.default))
            w.valueChanged.connect(self._on_tracking_params_changed)
            return w
        w = QDoubleSpinBox(self.extra_params_widget)
        w.setDecimals(int(spec.decimals))
        w.setRange(float(spec.min_val), float(spec.max_val))
        w.setSingleStep(float(spec.step))
        w.setValue(float(spec.default))
        w.valueChanged.connect(self._on_tracking_params_changed)
        return w

    def _read_extra_params(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for name, w in self.extra_param_controls.items():
            if isinstance(w, QCheckBox):
                out[name] = bool(w.isChecked())
            elif isinstance(w, QComboBox):
                out[name] = (
                    w.currentData() if w.currentData() is not None else w.currentText()
                )
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
            with QSignalBlocker(w):
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

    # ------------------------------------------------------------------
    # Health & Metrics Computation (with ViewerState fallback)
    # ------------------------------------------------------------------
    def _unit_hex_color(self, unit_idx: Optional[int]) -> str:
        if unit_idx is None:
            return "#b3b3b3"
        if self.state is not None and hasattr(self.state, "get_unit_hex_color"):
            try:
                hex_c = self.state.get_unit_hex_color(unit_idx)
                if hex_c:
                    return str(hex_c)
            except Exception:
                pass
        return _default_unit_hex_color(unit_idx)

    def _compute_unit_health(
        self, unit_idx: int, sg: tracked_neuron_unit
    ) -> Dict[str, Any]:
        """Use state.get_unit_health(unit_idx) if present, else compute fallback."""
        if self.state is not None and hasattr(self.state, "get_unit_health"):
            try:
                raw_misses = int(getattr(sg, "_consecutive_misses", 0) or 0)
                res = self.state.get_unit_health(unit_idx)
                if isinstance(res, dict):
                    out = dict(res)
                    # Normalize key names between ViewerState.get_unit_health and fallback
                    if (
                        out.get("latest_latency_ms") is None
                        and out.get("last_latency_ms") is not None
                    ):
                        out["latest_latency_ms"] = out["last_latency_ms"]
                    if "hit_count" not in out and "recent_hits" in out:
                        out["hit_count"] = out["recent_hits"]
                    if "hit_window" not in out and "recent_window" in out:
                        out["hit_window"] = out["recent_window"]

                    auto_track = bool(
                        out.get("auto_track", getattr(sg, "auto_track", False))
                    )
                    max_skips = int(
                        out.get("max_skips", getattr(sg, "tracking_max_skips", 1)) or 0
                    )
                    computed_misses = int(out.get("consecutive_misses", 0) or 0)
                    # Respect sg._consecutive_misses if set directly (e.g. live miss tracking or tests)
                    if raw_misses > computed_misses:
                        sg._consecutive_misses = raw_misses
                        out["consecutive_misses"] = raw_misses
                        if auto_track:
                            if raw_misses > max_skips:
                                out["status"] = "lost"
                            elif raw_misses >= 1:
                                out["status"] = "skipping"
                            else:
                                out["status"] = "locked"
                    return out
            except Exception:
                pass

        auto_track = bool(getattr(sg, "auto_track", False))
        consecutive_misses = int(getattr(sg, "_consecutive_misses", 0) or 0)
        max_skips = int(getattr(sg, "tracking_max_skips", 1) or 0)

        event_signal = getattr(self.state, "event_signal", None) if self.state else None
        sr_raw = getattr(self.state, "sampling_rate", None) if self.state else None
        if sr_raw is None and self.state is not None and getattr(self.state, "analog_signal", None) is not None:
            sr_raw = getattr(self.state.analog_signal, "sampling_rate", None)
        sr_hz = (
            float(sr_raw.rescale(pq.Hz).magnitude)
            if sr_raw is not None and hasattr(sr_raw, "rescale")
            else (float(sr_raw) if sr_raw is not None else 30000.0)
        )
        win_size = getattr(self.state, "window_size", None) if self.state else None

        total_spikes = 0
        latest_latency_ms: Optional[float] = None
        latency_shift_ms: Optional[float] = None
        hit_count = 0
        hit_window = 0

        sweeps = np.empty(0, dtype=np.int64)
        samples = np.empty(0, dtype=np.int64)

        if (
            event_signal is not None
            and len(event_signal) > 0
            and hasattr(sg, "get_valid_sweep_indices_and_samples")
        ):
            try:
                sweeps, samples = sg.get_valid_sweep_indices_and_samples(
                    event_signal, sr_hz, win_size
                )
            except Exception:
                sweeps = np.empty(0, dtype=np.int64)
                samples = np.empty(0, dtype=np.int64)

        if len(samples) > 0:
            total_spikes = int(len(samples))
            latencies_ms = (samples.astype(np.float64) / sr_hz) * 1000.0
            latest_latency_ms = float(latencies_ms[-1])
            if len(latencies_ms) >= 2:
                latency_shift_ms = float(latencies_ms[-1] - latencies_ms[-2])
            else:
                latency_shift_ms = 0.0

            n_sweeps = len(event_signal)
            hit_window = min(20, n_sweeps)
            start_sw = max(0, n_sweeps - hit_window)
            hit_count = int(np.sum(sweeps >= start_sw))
        else:
            idx_arr = getattr(sg, "idx_arr", None)
            if idx_arr is not None:
                valid_entries = [
                    (i, x[0] if isinstance(x, (tuple, list)) else x)
                    for i, x in enumerate(idx_arr)
                    if x is not None
                ]
                total_spikes = len(valid_entries)
                if total_spikes > 0:
                    latest_latency_ms = float(valid_entries[-1][1]) / sr_hz * 1000.0
                    if total_spikes >= 2:
                        prev_lat = float(valid_entries[-2][1]) / sr_hz * 1000.0
                        latency_shift_ms = latest_latency_ms - prev_lat
                    else:
                        latency_shift_ms = 0.0
                n_sweeps = len(idx_arr)
                hit_window = min(20, n_sweeps)
                if hit_window > 0:
                    start_sw = n_sweeps - hit_window
                    hit_count = sum(1 for i, _ in valid_entries if i >= start_sw)
            elif getattr(sg, "event", None) is not None:
                try:
                    total_spikes = int(len(sg.event))
                except Exception:
                    total_spikes = 0

        if auto_track:
            if consecutive_misses > max_skips:
                status = "lost"
            elif consecutive_misses >= 1:
                status = "skipping"
            else:
                status = "locked"
        else:
            status = "manual" if total_spikes > 0 else "empty"

        return {
            "status": status,
            "auto_track": auto_track,
            "consecutive_misses": consecutive_misses,
            "max_skips": max_skips,
            "total_spikes": total_spikes,
            "hit_count": hit_count,
            "hit_window": hit_window,
            "hit_rate": (hit_count / hit_window) if hit_window > 0 else None,
            "latest_latency_ms": latest_latency_ms,
            "latency_shift_ms": latency_shift_ms,
        }

    def _apply_health_header(
        self, unit_idx: Optional[int], sg: Optional[tracked_neuron_unit]
    ) -> None:
        if unit_idx is None or sg is None:
            self.lbl_color_swatch.setStyleSheet(
                "background: #b3b3b3; border: 1px solid rgba(0,0,0,0.25); border-radius: 4px;"
            )
            self.lbl_unit_title.setText("<b>No Unit</b>")
            self.lbl_health_badge.setText("EMPTY")
            self.lbl_health_badge.setStyleSheet(_HEALTH_STYLES["neutral"])
            self.lbl_metrics.setText("Spikes: 0  |  Hit: --  |  Lat: --  |  Δ --")
            return

        hex_color = self._unit_hex_color(unit_idx)
        self.lbl_color_swatch.setStyleSheet(
            f"background: {hex_color}; border: 1px solid rgba(0,0,0,0.3); border-radius: 4px;"
        )

        ev_name = (
            getattr(getattr(sg, "event", None), "name", None) or f"unit_{unit_idx}"
        )
        pin_suffix = " (pinned)" if self._pinned_unit_idx is not None else ""
        self.lbl_unit_title.setText(f"<b>Unit {unit_idx}: {ev_name}</b>{pin_suffix}")

        health = self._compute_unit_health(unit_idx, sg)
        status = str(health.get("status", "")).lower()
        auto_track = bool(health.get("auto_track", getattr(sg, "auto_track", False)))
        misses = int(health.get("consecutive_misses", getattr(sg, "_consecutive_misses", 0)) or 0)
        max_skips = int(health.get("max_skips", getattr(sg, "tracking_max_skips", 1)) or 0)
        total_spikes = int(health.get("total_spikes", 0) or 0)

        if not status:
            if auto_track:
                if misses > max_skips:
                    status = "lost"
                elif misses >= 1:
                    status = "skipping"
                else:
                    status = "locked"
            else:
                status = "manual" if total_spikes > 0 else "empty"

        if status == "locked":
            self.lbl_health_badge.setText("LOCKED")
            self.lbl_health_badge.setStyleSheet(_HEALTH_STYLES["locked"])
        elif status == "skipping":
            self.lbl_health_badge.setText(f"SKIPPING ({misses})")
            self.lbl_health_badge.setStyleSheet(_HEALTH_STYLES["skipping"])
        elif status == "lost":
            self.lbl_health_badge.setText(f"LOST ({misses} misses)")
            self.lbl_health_badge.setStyleSheet(_HEALTH_STYLES["lost"])
        elif status == "manual":
            self.lbl_health_badge.setText("MANUAL")
            self.lbl_health_badge.setStyleSheet(_HEALTH_STYLES["neutral"])
        else:
            self.lbl_health_badge.setText("EMPTY")
            self.lbl_health_badge.setStyleSheet(_HEALTH_STYLES["neutral"])

        recent_hits = health.get("recent_hits", health.get("hit_count", 0))
        recent_window = health.get("recent_window", health.get("hit_window", 0))
        hit_rate = health.get("hit_rate")
        if recent_window and recent_window > 0:
            rate_val = (
                float(hit_rate)
                if hit_rate is not None
                else float(recent_hits) / float(recent_window)
            )
            pct = int(round(100.0 * rate_val))
            hit_str = f"{int(recent_hits)}/{int(recent_window)} ({pct}%)"
        elif hit_rate is not None:
            pct = int(round(100.0 * float(hit_rate)))
            hit_str = f"{pct}%"
        else:
            hit_str = "--"

        lat_ms = health.get("last_latency_ms")
        if lat_ms is None:
            lat_ms = health.get("latest_latency_ms")
        lat_str = f"{float(lat_ms):.1f} ms" if lat_ms is not None and not np.isnan(lat_ms) else "--"

        shift_ms = health.get("latency_shift_ms")
        if shift_ms is not None and not np.isnan(shift_ms):
            shift_str = f"Δ {float(shift_ms):+.1f} ms"
        else:
            shift_str = "Δ --"

        self.lbl_metrics.setText(
            f"N={total_spikes}  |  Hit: {hit_str}  |  Lat: {lat_str}  |  {shift_str}"
        )

    # ------------------------------------------------------------------
    # Refresh / Sync From State -> UI
    # ------------------------------------------------------------------
    def _refresh_unit_combo(self) -> None:
        with QSignalBlocker(self.combo_unit):
            self.combo_unit.clear()
            self.combo_unit.addItem("(Follow Active Unit)", userData=None)
            groups = getattr(self.state, "spike_groups", None) or [] if self.state else []
            for i, sg in enumerate(groups):
                u_name = getattr(getattr(sg, "event", None), "name", None) or f"unit_{i}"
                ch = getattr(sg, "channel", "") or ""
                ch_info = f" [{ch}]" if ch else ""
                self.combo_unit.addItem(f"Unit {i}: {u_name}{ch_info}", userData=i)

            if self._pinned_unit_idx is not None:
                idx = self.combo_unit.findData(self._pinned_unit_idx)
                self.combo_unit.setCurrentIndex(idx if idx >= 0 else 0)
            else:
                self.combo_unit.setCurrentIndex(0)

    def _refresh_primary_channel_combo(self, current_channel: str) -> None:
        avail: List[str] = []
        if self.state is not None and hasattr(self.state, "get_available_channels"):
            try:
                avail = list(self.state.get_available_channels() or [])
            except Exception:
                avail = []
        if current_channel and current_channel not in avail:
            avail = [current_channel] + avail

        with QSignalBlocker(self.combo_primary_channel):
            self.combo_primary_channel.clear()
            if not avail:
                self.combo_primary_channel.addItem("(none)", userData="")
            else:
                for ch in avail:
                    self.combo_primary_channel.addItem(str(ch), userData=str(ch))
                if current_channel:
                    idx = self.combo_primary_channel.findData(current_channel)
                    if idx >= 0:
                        self.combo_primary_channel.setCurrentIndex(idx)

    def _set_controls_enabled(self, enabled: bool) -> None:
        for w in (
            self.grp_identity,
            self.grp_channels,
            self.grp_tracking,
            self.btn_track_now,
            self.btn_fill_missing,
            self.btn_rebuild_template,
        ):
            w.setEnabled(enabled)

    def _sync_from_candidate(self, cand) -> None:
        self.candidate_actions_widget.setVisible(True)
        self.lbl_color_swatch.setStyleSheet(
            "background: #ffaa00; border: 1px solid rgba(0,0,0,0.3); border-radius: 4px;"
        )
        self.lbl_unit_title.setText(f"<b>Candidate: {cand.name}</b>")
        self.lbl_health_badge.setText("CANDIDATE")
        self.lbl_health_badge.setStyleSheet(
            "background: #fef3c7; color: #92400e; border: 1px solid #f59e0b; "
            "border-radius: 10px; padding: 2px 8px; font-weight: bold;"
        )
        sr = 30000.0
        if self.state is not None and hasattr(self.state, "_candidate_sampling_rate"):
            try:
                sr = float(self.state._candidate_sampling_rate()) or 30000.0
            except Exception:
                sr = 30000.0
        lat_ms = (float(getattr(cand, "latency_center_samples", 0.0) or 0.0) / sr) * 1000.0
        ev = getattr(cand, "evidence", None)
        jitter_ms = float(getattr(ev, "jitter_ms", 0.0) or 0.0)
        snr = float(getattr(ev, "median_snr", 0.0) or 0.0)
        self.lbl_metrics.setText(
            f"N={int(cand.hit_count)}  |  Score: {float(cand.score):.2f}  |  "
            f"Lat: {lat_ms:.1f} ms  |  Jitter: {jitter_ms:.2f} ms  |  SNR: {snr:.1f}"
        )

        # Allow editing identity (name/notes) while disabling unit-only tracking actions
        self.grp_identity.setEnabled(True)
        self.grp_channels.setEnabled(False)
        self.grp_tracking.setEnabled(False)
        self.btn_track_now.setEnabled(False)
        self.btn_fill_missing.setEnabled(False)
        self.btn_rebuild_template.setEnabled(False)

        primary_ch = str(getattr(cand, "channel", "") or "")
        neighbors_list = [
            str(c) for c in (getattr(cand, "neighbor_channels", None) or []) if str(c).strip()
        ]
        self._refresh_primary_channel_combo(primary_ch)
        self.lbl_neighbors.setText(
            f"Neighbors: {', '.join(neighbors_list)}" if neighbors_list else "Neighbors: (none)"
        )

        with (
            QSignalBlocker(self.edit_name),
            QSignalBlocker(self.combo_fibre_type),
            QSignalBlocker(self.edit_notes),
            QSignalBlocker(self.chk_auto_track),
            QSignalBlocker(self.combo_algorithm),
            QSignalBlocker(self.spin_window_ms),
            QSignalBlocker(self.spin_threshold_mv),
            QSignalBlocker(self.spin_max_skips),
            QSignalBlocker(self.spin_track_limit),
        ):
            self.edit_name.setText(str(cand.name or ""))
            self.combo_fibre_type.setCurrentText("")
            self.edit_notes.setText(str(cand.notes or ""))
            self.chk_auto_track.setChecked(True)
            algo = str(getattr(cand, "algorithm", "zscore") or "zscore")
            algo_idx = self.combo_algorithm.findData(algo)
            if algo_idx < 0:
                self.refresh_algorithms()
                algo_idx = self.combo_algorithm.findData(algo)
            if algo_idx >= 0 and self.combo_algorithm.currentIndex() != algo_idx:
                self.combo_algorithm.setCurrentIndex(algo_idx)
            self._rebuild_extra_param_controls(algo)
            self._write_extra_params(getattr(cand, "detector_params", {}) or {})
            self.spin_window_ms.setValue(float(getattr(cand, "tracking_window_ms", 5.0)))
            self.spin_threshold_mv.setValue(float(getattr(cand, "tracking_threshold_mv", 0.01)))
            self.spin_max_skips.setValue(int(getattr(cand, "tracking_max_skips", 3)))
            self.spin_track_limit.setValue(0)

    def sync_from_unit(self) -> None:
        if self._updating_ui:
            return
        self._syncing = True
        try:
            self._refresh_unit_combo()
            cand = self._get_active_candidate()
            if cand is not None:
                self._sync_from_candidate(cand)
                return

            self.candidate_actions_widget.setVisible(False)
            unit_idx = self.get_target_unit_idx()
            sg = self.get_target_unit()
            self._apply_health_header(unit_idx, sg)

            if sg is None or unit_idx is None:
                self._set_controls_enabled(False)
                with (
                    QSignalBlocker(self.edit_name),
                    QSignalBlocker(self.combo_fibre_type),
                    QSignalBlocker(self.edit_notes),
                    QSignalBlocker(self.combo_primary_channel),
                ):
                    self.edit_name.setText("")
                    self.combo_fibre_type.setCurrentText("")
                    self.edit_notes.setText("")
                    self.lbl_neighbors.setText("Neighbors: (none)")
                    self._refresh_primary_channel_combo("")
                return

            self._set_controls_enabled(True)
            ann = getattr(getattr(sg, "event", None), "annotations", None) or {}

            ev_name = getattr(getattr(sg, "event", None), "name", None) or f"unit_{unit_idx}"
            fibre_type = str(ann.get("fibre_type", getattr(sg, "fibre_type", "")) or "")
            notes = str(ann.get("notes", getattr(sg, "notes", "")) or "")
            primary_ch = str(getattr(sg, "channel", "") or ann.get("channel", "") or "")
            neighbors = getattr(sg, "neighbor_channels", None)
            if neighbors is None:
                neighbors = ann.get("neighbor_channels", [])
            neighbors_list = [str(c) for c in (neighbors or []) if str(c).strip()]

            self._refresh_primary_channel_combo(primary_ch)
            self.lbl_neighbors.setText(
                f"Neighbors: {', '.join(neighbors_list)}" if neighbors_list else "Neighbors: (none)"
            )

            with (
                QSignalBlocker(self.edit_name),
                QSignalBlocker(self.combo_fibre_type),
                QSignalBlocker(self.edit_notes),
                QSignalBlocker(self.chk_auto_track),
                QSignalBlocker(self.combo_algorithm),
                QSignalBlocker(self.spin_window_ms),
                QSignalBlocker(self.spin_threshold_mv),
                QSignalBlocker(self.spin_max_skips),
                QSignalBlocker(self.spin_track_limit),
            ):
                if self.edit_name.text() != ev_name:
                    self.edit_name.setText(str(ev_name))

                ft_idx = self.combo_fibre_type.findText(fibre_type)
                if ft_idx >= 0:
                    self.combo_fibre_type.setCurrentIndex(ft_idx)
                else:
                    self.combo_fibre_type.setCurrentText(fibre_type)

                if self.edit_notes.text() != notes:
                    self.edit_notes.setText(notes)

                self.chk_auto_track.setChecked(bool(getattr(sg, "auto_track", False)))

                algo = str(getattr(sg, "tracking_algorithm", "basic") or "basic")
                algo_idx = self.combo_algorithm.findData(algo)
                if algo_idx < 0:
                    self.refresh_algorithms()
                    algo_idx = self.combo_algorithm.findData(algo)
                if algo_idx >= 0 and self.combo_algorithm.currentIndex() != algo_idx:
                    self.combo_algorithm.setCurrentIndex(algo_idx)
                self._rebuild_extra_param_controls(algo)
                self._write_extra_params(getattr(sg, "tracking_params", {}) or {})

                self.spin_window_ms.setValue(
                    float(getattr(sg, "tracking_window_ms", 5.0))
                )
                self.spin_threshold_mv.setValue(
                    float(getattr(sg, "tracking_threshold_mv", 0.01))
                )
                self.spin_max_skips.setValue(
                    int(getattr(sg, "tracking_max_skips", 1))
                )
                self.spin_track_limit.setValue(
                    int(getattr(sg, "track_limit", 0))
                )
        finally:
            self._syncing = False

    def updateView(self) -> None:
        self.sync_from_unit()

    # ------------------------------------------------------------------
    # UI -> State Mutations
    # ------------------------------------------------------------------
    def _emit_unit_metadata_changed(self, unit_idx: int) -> None:
        if self.state is None:
            return
        if hasattr(self.state, "onUnitParamsChanged"):
            self.state.onUnitParamsChanged.emit(int(unit_idx))
        if hasattr(self.state, "onUnitGroupChange"):
            self.state.onUnitGroupChange.emit()

    @Slot(int)
    def _on_unit_combo_changed(self, idx: int) -> None:
        if self._syncing or self.state is None:
            return
        selected = self.combo_unit.itemData(idx)
        if selected is None:
            self._pinned_unit_idx = None
        else:
            unit_idx = int(selected)
            self._pinned_unit_idx = unit_idx
            if hasattr(self.state, "setUnitGroup"):
                self._updating_ui = True
                try:
                    self.state.setUnitGroup(unit_idx)
                finally:
                    self._updating_ui = False
            self.unitSelected.emit(unit_idx)
        self.sync_from_unit()

    @Slot()
    def _on_new_unit_clicked(self) -> None:
        if self.state is None or not hasattr(self.state, "addUnitGroup"):
            return
        self._updating_ui = True
        try:
            self.state.addUnitGroup()
            new_idx = max(0, len(getattr(self.state, "spike_groups", [])) - 1)
            if self._pinned_unit_idx is not None:
                self._pinned_unit_idx = new_idx
            if hasattr(self.state, "setUnitGroup"):
                self.state.setUnitGroup(new_idx)
        finally:
            self._updating_ui = False
        self.sync_from_unit()

    @Slot()
    def _on_accept_candidate_clicked(self) -> None:
        if self.state is None:
            return
        cand = self._get_active_candidate()
        if cand is not None and hasattr(self.state, "accept_candidate"):
            self.state.accept_candidate(int(cand.candidate_id))

    @Slot()
    def _on_discard_candidate_clicked(self) -> None:
        if self.state is None:
            return
        cand = self._get_active_candidate()
        if cand is not None and hasattr(self.state, "discard_candidate"):
            self.state.discard_candidate(int(cand.candidate_id))

    @Slot()
    def _on_identity_changed(self, *args) -> None:
        if self._syncing or self.state is None:
            return
        cand = self._get_active_candidate()
        if cand is not None:
            new_name = self.edit_name.text().strip()
            new_notes = self.edit_notes.text()
            self._updating_ui = True
            try:
                if new_name and new_name != cand.name:
                    cand.name = new_name
                    cand.name_edited = True
                    if getattr(cand.unit, "event", None) is not None:
                        cand.unit.event.name = new_name
                if new_notes != cand.notes:
                    cand.notes = new_notes
                    cand.notes_edited = True
                    cand.unit.notes = new_notes
                self.lbl_unit_title.setText(f"<b>Candidate: {cand.name}</b>")
                if hasattr(self.state, "onCandidatesChanged"):
                    self.state.onCandidatesChanged.emit()
            finally:
                self._updating_ui = False
            return

        unit_idx = self.get_target_unit_idx()
        sg = self.get_target_unit()
        if sg is None or unit_idx is None:
            return

        new_name = self.edit_name.text().strip()
        new_fibre = self.combo_fibre_type.currentText().strip()
        new_notes = self.edit_notes.text()

        self._updating_ui = True
        try:
            if getattr(sg, "event", None) is None:
                sg.event = Event()
            if not hasattr(sg.event, "annotations") or sg.event.annotations is None:
                sg.event.annotations = {}

            if new_name:
                sg.event.name = new_name
            sg.fibre_type = new_fibre
            sg.event.annotations["fibre_type"] = new_fibre
            sg.notes = new_notes
            sg.event.annotations["notes"] = new_notes

            self._apply_health_header(unit_idx, sg)
            self._refresh_unit_combo()
            self._emit_unit_metadata_changed(unit_idx)
        finally:
            self._updating_ui = False

    @Slot(int)
    def _on_primary_channel_changed(self, idx: int) -> None:
        if self._syncing or self.state is None:
            return
        unit_idx = self.get_target_unit_idx()
        sg = self.get_target_unit()
        if sg is None or unit_idx is None:
            return

        new_ch = str(self.combo_primary_channel.itemData(idx) or "").strip()
        if not new_ch:
            return

        self._updating_ui = True
        try:
            old_neighbors = [
                str(c) for c in (getattr(sg, "neighbor_channels", None) or []) if str(c) != new_ch
            ]
            if hasattr(self.state, "set_unit_channels"):
                self.state.set_unit_channels(unit_idx, [new_ch] + old_neighbors)
            else:
                sg.channel = new_ch
                sg.neighbor_channels = old_neighbors
                if getattr(sg, "event", None) is not None:
                    if not hasattr(sg.event, "annotations") or sg.event.annotations is None:
                        sg.event.annotations = {}
                    sg.event.annotations["channel"] = new_ch
                    sg.event.annotations["neighbor_channels"] = old_neighbors
                self._emit_unit_metadata_changed(unit_idx)
        finally:
            self._updating_ui = False
        self.sync_from_unit()

    def _shift_unit_channels(self, step: int) -> None:
        if self.state is None:
            return
        unit_idx = self.get_target_unit_idx()
        if unit_idx is None:
            return
        if hasattr(self.state, "shift_unit_channels"):
            self.state.shift_unit_channels(unit_idx, step=int(step))
        self.sync_from_unit()

    @Slot()
    def _open_channel_dialog(self) -> None:
        if self.state is None:
            return
        unit_idx = self.get_target_unit_idx()
        if unit_idx is None:
            return
        dlg = ChannelGroupDialog(parent=self, state=self.state, unit_idx=unit_idx)
        if dlg.exec() == QDialog.Accepted:
            selected = dlg.get_selected_channels()
            if selected and hasattr(self.state, "set_unit_channels"):
                self.state.set_unit_channels(unit_idx, selected)
            self.sync_from_unit()

    @Slot(int)
    def _on_algorithm_changed(self, *args) -> None:
        if self._syncing:
            return
        algo = self.get_selected_algorithm()
        self._rebuild_extra_param_controls(algo)
        self._on_tracking_params_changed()

    @Slot()
    def _on_tracking_params_changed(self, *args) -> None:
        if self._syncing or self.state is None:
            return
        unit_idx = self.get_target_unit_idx()
        sg = self.get_target_unit()
        if sg is None or unit_idx is None:
            return

        algo_val = self.get_selected_algorithm()
        win_val = float(self.spin_window_ms.value())
        thresh_val = float(self.spin_threshold_mv.value())
        skip_val = int(self.spin_max_skips.value())
        limit_val = int(self.spin_track_limit.value())
        track_val = bool(self.chk_auto_track.isChecked())
        extra_params = self._read_extra_params()

        self._updating_ui = True
        try:
            sg.tracking_algorithm = algo_val
            sg.tracking_window_ms = win_val
            sg.tracking_threshold_mv = thresh_val
            sg.tracking_max_skips = skip_val
            sg.track_limit = limit_val
            sg.auto_track = track_val
            if not hasattr(sg, "tracking_params") or sg.tracking_params is None:
                sg.tracking_params = {}
            sg.tracking_params.update(extra_params)
            sg._template_matcher = None

            if (
                track_val
                and getattr(self.state, "is_streaming", False)
                and hasattr(self.state, "ensure_live_tracking_processor")
            ):
                try:
                    self.state.ensure_live_tracking_processor()
                except Exception:
                    pass

            if getattr(sg, "event", None) is not None and hasattr(sg.event, "annotations"):
                sg.event.annotations["tracking_algorithm"] = algo_val
                sg.event.annotations["tracking_window_ms"] = win_val
                sg.event.annotations["tracking_threshold_mv"] = thresh_val
                sg.event.annotations["tracking_max_skips"] = skip_val
                sg.event.annotations["track_limit"] = limit_val
                sg.event.annotations["auto_track"] = track_val
                sg.event.annotations["tracking_params"] = dict(sg.tracking_params)

            self._apply_health_header(unit_idx, sg)

            if hasattr(self.state, "onUnitParamsChanged"):
                self.state.onUnitParamsChanged.emit(int(unit_idx))
            elif hasattr(self.state, "onUnitGroupChange"):
                self.state.onUnitGroupChange.emit()
        finally:
            self._updating_ui = False

    @Slot()
    def _on_auto_threshold_clicked(self) -> None:
        if self.state is None:
            return
        sg = self.get_target_unit()
        if sg is None or getattr(sg, "event", None) is None or len(sg.event) == 0:
            return
        event_signal = getattr(self.state, "event_signal", None)
        if event_signal is None or len(event_signal) == 0:
            return

        ch = (
            getattr(sg, "tracking_channel", "")
            or getattr(sg, "channel", "")
            or (getattr(self.state.analog_signal, "name", "") if self.state.analog_signal is not None else "")
        )
        sig = (
            self.state.get_channel_signal(ch)
            if ch and hasattr(self.state, "get_channel_signal")
            else None
        )
        if sig is None:
            sig = getattr(self.state, "analog_signal", None)
        if sig is None:
            return

        try:
            unit_events = sg.event.rescale(pq.s)
            stimno = max(0, min(int(getattr(self.state, "stimno", 0)), len(event_signal) - 1))
            last_event = int(
                unit_events.searchsorted((event_signal[stimno] + (0.5 * pq.s)).rescale(pq.s))
            )
            starting_time = unit_events[max(last_event - 1, 0)]
            dat = sig.time_slice(starting_time, starting_time + 0.1 * pq.ms)
            threshold = float(np.asarray((dat[0] * 0.8).rescale("mV").magnitude).flat[0])
            self.spin_threshold_mv.setValue(threshold)
        except Exception:
            pass

    @Slot()
    def _on_track_forward_clicked(self) -> None:
        if self.state is None or not hasattr(self.state, "track_unit_range"):
            return
        unit_idx = self.get_target_unit_idx()
        if unit_idx is None:
            return
        stimno = int(getattr(self.state, "stimno", 0) or 0)
        self.state.track_unit_range(unit_idx, start_stim=stimno, only_missing=False)
        self.sync_from_unit()

    @Slot()
    def _on_fill_missing_clicked(self) -> None:
        if self.state is None or not hasattr(self.state, "track_unit_range"):
            return
        unit_idx = self.get_target_unit_idx()
        if unit_idx is None:
            return
        self.state.track_unit_range(unit_idx, only_missing=True)
        self.sync_from_unit()

    @Slot()
    def _on_rebuild_template_clicked(self) -> None:
        if self.state is None or not hasattr(self.state, "rebuild_unit_template"):
            return
        unit_idx = self.get_target_unit_idx()
        if unit_idx is None:
            return
        self.state.rebuild_unit_template(unit_idx)
        self.sync_from_unit()
