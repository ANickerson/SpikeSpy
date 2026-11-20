from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional

from PySide6.QtCore import QSignalBlocker, Qt, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .candidate_detection import CandidateScanConfig
from .dsp.detectors import DetectorParamSpec, SpikeDetectorRegistry
from .ViewerState import ViewerState

PROPOSAL_PRESETS: Dict[str, Dict[str, Any]] = {
    "Balanced (Default)": {
        "algorithm": "zscore",
        "enabled_algorithms": [],
        "detector_params": {},
        "blank_ms": 1.0,
        "min_peak_distance_ms": 0.5,
        "min_peak_snr": 3.0,
        "tolerance_ms": 1.0,
        "merge_tolerance_ms": 1.0,
        "min_hits": 3,
        "max_jitter_ms": 2.0,
        "max_candidates": 8,
        "max_per_channel": 3,
        "accepted_overlap_suppress": 0.6,
        "reactivation_score_margin": 0.15,
        "live_scan_window_sweeps": 40,
        "recency_window_sweeps": 10,
        "scan_interval_ms": 2000,
        "review_quiet_ms": 2000,
        "max_scan_sweeps": 20000,
        "template_refine": True,
        "template_corr_threshold": 0.35,
        "max_template_refine_sweeps": 2000,
    },
    "Sensitive (Low-SNR C-Fibre)": {
        "algorithm": "zscore",
        "enabled_algorithms": [],
        "detector_params": {
            "polarity": "both",
            "z_thresh": 2.25,
            "use_mad": True,
            "min_distance_ms": 0.5,
        },
        "blank_ms": 1.0,
        "min_peak_distance_ms": 0.5,
        "min_peak_snr": 2.25,
        "tolerance_ms": 1.5,
        "merge_tolerance_ms": 1.5,
        "min_hits": 3,
        "max_jitter_ms": 3.0,
        "max_candidates": 12,
        "max_per_channel": 4,
        "accepted_overlap_suppress": 0.6,
        "reactivation_score_margin": 0.10,
        "template_refine": True,
        "template_corr_threshold": 0.25,
        "max_template_refine_sweeps": 2000,
    },
    "Strict (High-SNR Fast Search)": {
        "algorithm": "zscore",
        "enabled_algorithms": [],
        "detector_params": {
            "polarity": "both",
            "z_thresh": 4.5,
            "use_mad": True,
            "min_distance_ms": 1.0,
        },
        "blank_ms": 1.5,
        "min_peak_distance_ms": 1.0,
        "min_peak_snr": 4.5,
        "tolerance_ms": 0.75,
        "merge_tolerance_ms": 0.75,
        "min_hits": 5,
        "max_jitter_ms": 1.0,
        "max_candidates": 6,
        "max_per_channel": 2,
        "accepted_overlap_suppress": 0.5,
        "reactivation_score_margin": 0.20,
        "template_refine": False,
        "template_corr_threshold": 0.5,
        "max_template_refine_sweeps": 1000,
    },
    "Wavelet Denoised Search": {
        "algorithm": "wavelet",
        "enabled_algorithms": [],
        "detector_params": {
            "threshold_sigma": 3.0,
            "wavelet": "sym7",
            "level": 5,
        },
        "blank_ms": 1.0,
        "min_peak_distance_ms": 0.5,
        "min_peak_snr": 3.0,
        "tolerance_ms": 1.0,
        "merge_tolerance_ms": 1.0,
        "min_hits": 3,
        "max_jitter_ms": 2.0,
        "template_refine": True,
        "template_corr_threshold": 0.35,
    },
    "Multi-Detector Ensemble (Z-Score + Wavelet)": {
        "algorithm": "zscore",
        "enabled_algorithms": ["zscore", "wavelet"],
        "detector_params": {
            "polarity": "both",
            "z_thresh": 3.0,
            "threshold_sigma": 3.0,
            "use_mad": True,
            "min_distance_ms": 0.5,
        },
        "blank_ms": 1.0,
        "min_peak_distance_ms": 0.5,
        "min_peak_snr": 3.0,
        "tolerance_ms": 1.2,
        "merge_tolerance_ms": 1.2,
        "min_hits": 3,
        "max_jitter_ms": 2.5,
        "template_refine": True,
        "template_corr_threshold": 0.35,
    },
}


class SpikeProposalConfigView(QWidget):
    """Dedicated panel for configuring spike proposal and candidate search."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        state: Optional[ViewerState] = None,
    ) -> None:
        super().__init__(parent)
        self.state: Optional[ViewerState] = None
        self._syncing: bool = False
        self._scanning: bool = False
        self._state_connections: List[tuple] = []
        self.ensemble_checks: Dict[str, QCheckBox] = {}
        self.extra_param_controls: Dict[str, QWidget] = {}

        self.setWindowTitle("Spike Proposal Config")

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(6, 6, 6, 6)
        root_layout.setSpacing(6)

        # ------------------------------------------------------------------
        # Top Action & Preset Bar
        # ------------------------------------------------------------------
        preset_row = QHBoxLayout()
        preset_row.setSpacing(6)
        preset_row.addWidget(QLabel("Preset:", self))
        self.combo_preset = QComboBox(self)
        for name in PROPOSAL_PRESETS:
            self.combo_preset.addItem(name, name)
        self.combo_preset.currentTextChanged.connect(self._on_preset_selected)
        preset_row.addWidget(self.combo_preset, 1)
        root_layout.addLayout(preset_row)

        action_row = QHBoxLayout()
        action_row.setSpacing(6)
        self.btn_scan_now = QPushButton("🔍 Scan Now", self)
        self.btn_scan_now.setToolTip("Run a candidate spike proposal scan now")
        self.btn_scan_now.clicked.connect(lambda: self.trigger_scan())
        action_row.addWidget(self.btn_scan_now)

        self.chk_auto_scan = QCheckBox("📡 Live Auto-Scan", self)
        self.chk_auto_scan.setToolTip(
            "Periodically run candidate search on recent sweeps while streaming"
        )
        self.chk_auto_scan.toggled.connect(self._on_auto_scan_toggled)
        action_row.addWidget(self.chk_auto_scan)

        self.btn_rescan_discarded = QPushButton("↺ Rescan Discarded", self)
        self.btn_rescan_discarded.setToolTip(
            "Re-evaluate previously discarded candidates in the next scan"
        )
        self.btn_rescan_discarded.clicked.connect(self._on_rescan_discarded_clicked)
        action_row.addWidget(self.btn_rescan_discarded)

        self.btn_clear_candidates = QPushButton("✕ Clear", self)
        self.btn_clear_candidates.setToolTip("Clear all candidate proposals")
        self.btn_clear_candidates.clicked.connect(self._on_clear_candidates_clicked)
        action_row.addWidget(self.btn_clear_candidates)
        action_row.addStretch(1)
        root_layout.addLayout(action_row)

        self.lbl_scan_status = QLabel("Pending: 0 | Dormant: 0 | Discarded: 0", self)
        self.lbl_scan_status.setStyleSheet("color: #334155; font-size: 11px;")
        root_layout.addWidget(self.lbl_scan_status)

        # ------------------------------------------------------------------
        # Scrollable Configuration Body
        # ------------------------------------------------------------------
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll_content = QWidget(scroll)
        body_layout = QVBoxLayout(scroll_content)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(6)

        # 1. Spike Proposal Algorithm & Detector Parameters
        self.grp_algorithm = QGroupBox(
            "1. Spike Proposal Algorithm & Detector Parameters", scroll_content
        )
        algo_layout = QVBoxLayout(self.grp_algorithm)
        algo_layout.setContentsMargins(8, 8, 8, 8)
        algo_layout.setSpacing(6)

        algo_form = QFormLayout()
        algo_form.setContentsMargins(0, 0, 0, 0)
        algo_form.setSpacing(4)
        self.combo_algorithm = QComboBox(self.grp_algorithm)
        self._populate_algorithm_combo()
        self.combo_algorithm.currentIndexChanged.connect(self._on_algorithm_changed)
        algo_form.addRow("Primary Algorithm:", self.combo_algorithm)

        self.lbl_algo_description = QLabel("", self.grp_algorithm)
        self.lbl_algo_description.setWordWrap(True)
        self.lbl_algo_description.setStyleSheet("color: #475569; font-size: 11px;")
        algo_form.addRow("Description:", self.lbl_algo_description)
        algo_layout.addLayout(algo_form)

        ensemble_row = QHBoxLayout()
        ensemble_row.setSpacing(8)
        ensemble_row.addWidget(QLabel("Ensemble Detectors:", self.grp_algorithm))
        for plugin in SpikeDetectorRegistry.list_detectors():
            if plugin.name == "template":
                continue
            chk = QCheckBox(plugin.display_name, self.grp_algorithm)
            chk.setProperty("detector_key", plugin.name)
            chk.toggled.connect(self._on_ensemble_toggled)
            self.ensemble_checks[plugin.name] = chk
            ensemble_row.addWidget(chk)
        ensemble_row.addStretch(1)
        algo_layout.addLayout(ensemble_row)

        self.extra_params_widget = QWidget(self.grp_algorithm)
        self.extra_params_layout = QFormLayout(self.extra_params_widget)
        self.extra_params_layout.setContentsMargins(0, 0, 0, 0)
        self.extra_params_layout.setSpacing(4)
        algo_layout.addWidget(self.extra_params_widget)

        body_layout.addWidget(self.grp_algorithm)

        # 2. Multi-Channel Template Refinement
        self.grp_template_refine = QGroupBox(
            "2. Multi-Channel Template Refinement", scroll_content
        )
        tr_form = QFormLayout(self.grp_template_refine)
        tr_form.setContentsMargins(8, 8, 8, 8)
        tr_form.setSpacing(4)

        self.chk_template_refine = QCheckBox(
            "Enable Multi-Channel Template Refinement", self.grp_template_refine
        )
        self.chk_template_refine.toggled.connect(self._on_config_control_changed)
        tr_form.addRow(self.chk_template_refine)

        self.spin_template_corr = QDoubleSpinBox(self.grp_template_refine)
        self.spin_template_corr.setRange(0.05, 1.0)
        self.spin_template_corr.setDecimals(2)
        self.spin_template_corr.setSingleStep(0.05)
        self.spin_template_corr.valueChanged.connect(self._on_config_control_changed)
        tr_form.addRow("Template Correlation Threshold:", self.spin_template_corr)

        self.spin_max_template_sweeps = QSpinBox(self.grp_template_refine)
        self.spin_max_template_sweeps.setRange(10, 100000)
        self.spin_max_template_sweeps.setSingleStep(100)
        self.spin_max_template_sweeps.valueChanged.connect(
            self._on_config_control_changed
        )
        tr_form.addRow("Max Template Refine Sweeps:", self.spin_max_template_sweeps)

        body_layout.addWidget(self.grp_template_refine)

        # 3. Latency Band Clustering & Filtering
        self.grp_clustering = QGroupBox(
            "3. Latency Band Clustering & Filtering", scroll_content
        )
        cl_form = QFormLayout(self.grp_clustering)
        cl_form.setContentsMargins(8, 8, 8, 8)
        cl_form.setSpacing(4)

        self.spin_blank_ms = self._make_double_spin(
            self.grp_clustering, 0.0, 100.0, 2, 0.5, " ms"
        )
        cl_form.addRow("Stimulus Artifact Blank:", self.spin_blank_ms)

        self.spin_min_peak_snr = self._make_double_spin(
            self.grp_clustering, 0.5, 50.0, 2, 0.25, ""
        )
        self.spin_min_peak_snr.valueChanged.connect(self._on_snr_spin_changed)
        cl_form.addRow("Min Peak SNR:", self.spin_min_peak_snr)

        self.spin_min_peak_distance_ms = self._make_double_spin(
            self.grp_clustering, 0.1, 100.0, 2, 0.1, " ms"
        )
        self.spin_min_peak_distance_ms.valueChanged.connect(
            self._on_distance_spin_changed
        )
        cl_form.addRow("Min Peak Distance:", self.spin_min_peak_distance_ms)

        self.spin_tolerance_ms = self._make_double_spin(
            self.grp_clustering, 0.1, 50.0, 2, 0.1, " ms"
        )
        cl_form.addRow("Band Latency Tolerance:", self.spin_tolerance_ms)

        self.spin_merge_tolerance_ms = self._make_double_spin(
            self.grp_clustering, 0.1, 50.0, 2, 0.1, " ms"
        )
        cl_form.addRow("Merge Tolerance:", self.spin_merge_tolerance_ms)

        self.spin_min_hits = self._make_int_spin(self.grp_clustering, 1, 1000, 1, "")
        cl_form.addRow("Min Hits:", self.spin_min_hits)

        self.spin_max_jitter_ms = self._make_double_spin(
            self.grp_clustering, 0.05, 50.0, 2, 0.1, " ms"
        )
        cl_form.addRow("Max Jitter:", self.spin_max_jitter_ms)

        self.spin_max_candidates = self._make_int_spin(
            self.grp_clustering, 1, 100, 1, ""
        )
        cl_form.addRow("Max Candidates:", self.spin_max_candidates)

        self.spin_max_per_channel = self._make_int_spin(
            self.grp_clustering, 1, 50, 1, ""
        )
        cl_form.addRow("Max Per Channel:", self.spin_max_per_channel)

        self.spin_overlap_suppress = self._make_double_spin(
            self.grp_clustering, 0.05, 1.0, 2, 0.05, ""
        )
        cl_form.addRow("Accepted Overlap Suppress:", self.spin_overlap_suppress)

        self.spin_reactivation_margin = self._make_double_spin(
            self.grp_clustering, 0.0, 1.0, 2, 0.05, ""
        )
        cl_form.addRow("Reactivation Score Margin:", self.spin_reactivation_margin)

        for spin in (
            self.spin_blank_ms,
            self.spin_tolerance_ms,
            self.spin_merge_tolerance_ms,
            self.spin_min_hits,
            self.spin_max_jitter_ms,
            self.spin_max_candidates,
            self.spin_max_per_channel,
            self.spin_overlap_suppress,
            self.spin_reactivation_margin,
        ):
            spin.valueChanged.connect(self._on_config_control_changed)

        body_layout.addWidget(self.grp_clustering)

        # 4. Live Scan Window & Cadence
        self.grp_live_cadence = QGroupBox(
            "4. Live Scan Window & Cadence", scroll_content
        )
        live_form = QFormLayout(self.grp_live_cadence)
        live_form.setContentsMargins(8, 8, 8, 8)
        live_form.setSpacing(4)

        self.spin_live_window_sweeps = self._make_int_spin(
            self.grp_live_cadence, 5, 5000, 5, ""
        )
        live_form.addRow("Live Scan Window (sweeps):", self.spin_live_window_sweeps)

        self.spin_recency_sweeps = self._make_int_spin(
            self.grp_live_cadence, 1, 1000, 1, ""
        )
        live_form.addRow("Recency Window (sweeps):", self.spin_recency_sweeps)

        self.spin_scan_interval_ms = self._make_int_spin(
            self.grp_live_cadence, 100, 60000, 100, " ms"
        )
        live_form.addRow("Scan Interval:", self.spin_scan_interval_ms)

        self.spin_review_quiet_ms = self._make_int_spin(
            self.grp_live_cadence, 0, 60000, 100, " ms"
        )
        live_form.addRow("Review Quiet Period:", self.spin_review_quiet_ms)

        self.spin_max_scan_sweeps = self._make_int_spin(
            self.grp_live_cadence, 100, 500000, 500, ""
        )
        live_form.addRow("Max Scan Sweeps:", self.spin_max_scan_sweeps)

        for spin in (
            self.spin_live_window_sweeps,
            self.spin_recency_sweeps,
            self.spin_scan_interval_ms,
            self.spin_review_quiet_ms,
            self.spin_max_scan_sweeps,
        ):
            spin.valueChanged.connect(self._on_config_control_changed)

        body_layout.addWidget(self.grp_live_cadence)

        # 5. Proposed Candidates Summary & Quick Triage
        self.grp_candidates_preview = QGroupBox(
            "5. Proposed Candidates Summary & Quick Triage", scroll_content
        )
        cand_layout = QVBoxLayout(self.grp_candidates_preview)
        cand_layout.setContentsMargins(8, 8, 8, 8)
        cand_layout.setSpacing(6)

        self.list_candidates = QListWidget(self.grp_candidates_preview)
        self.list_candidates.setMinimumHeight(110)
        self.list_candidates.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Preferred
        )
        self.list_candidates.currentItemChanged.connect(
            self._on_candidate_item_selected
        )
        cand_layout.addWidget(self.list_candidates)

        triage_row = QHBoxLayout()
        triage_row.setSpacing(6)
        self.btn_accept_candidate = QPushButton(
            "✓ Accept Selected", self.grp_candidates_preview
        )
        self.btn_accept_candidate.clicked.connect(self._on_accept_selected_clicked)
        self.btn_discard_candidate = QPushButton(
            "✕ Discard Selected", self.grp_candidates_preview
        )
        self.btn_discard_candidate.clicked.connect(self._on_discard_selected_clicked)
        triage_row.addWidget(self.btn_accept_candidate)
        triage_row.addWidget(self.btn_discard_candidate)
        triage_row.addStretch(1)
        cand_layout.addLayout(triage_row)

        body_layout.addWidget(self.grp_candidates_preview)
        body_layout.addStretch(1)

        scroll.setWidget(scroll_content)
        root_layout.addWidget(scroll, 1)

        self._rebuild_extra_param_controls(self._selected_algorithm())
        if state is not None:
            self.set_state(state)
        else:
            self.sync_from_state()

    # ------------------------------------------------------------------
    # Widget construction helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _make_double_spin(
        parent: QWidget,
        min_val: float,
        max_val: float,
        decimals: int,
        step: float,
        suffix: str,
    ) -> QDoubleSpinBox:
        spin = QDoubleSpinBox(parent)
        spin.setRange(float(min_val), float(max_val))
        spin.setDecimals(int(decimals))
        spin.setSingleStep(float(step))
        if suffix:
            spin.setSuffix(suffix)
        return spin

    @staticmethod
    def _make_int_spin(
        parent: QWidget,
        min_val: int,
        max_val: int,
        step: int,
        suffix: str,
    ) -> QSpinBox:
        spin = QSpinBox(parent)
        spin.setRange(int(min_val), int(max_val))
        spin.setSingleStep(int(step))
        if suffix:
            spin.setSuffix(suffix)
        return spin

    def _populate_algorithm_combo(self) -> None:
        with QSignalBlocker(self.combo_algorithm):
            self.combo_algorithm.clear()
            for plugin in SpikeDetectorRegistry.list_detectors():
                self.combo_algorithm.addItem(plugin.display_name, plugin.name)

    def _selected_algorithm(self) -> str:
        if self.combo_algorithm.count() > 0:
            val = self.combo_algorithm.currentData()
            if val:
                return str(val)
        return "zscore"

    # ------------------------------------------------------------------
    # State lifecycle
    # ------------------------------------------------------------------
    def unset_connections(self) -> None:
        for sig, slot in list(self._state_connections):
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
            mappings = [
                ("onCandidateConfigChanged", self._on_state_config_changed),
                ("onCandidatesChanged", self._on_candidates_changed),
                ("onCandidateSelectionChanged", self._on_candidate_selection_changed),
                ("onCandidateScanStateChanged", self._on_scan_state_changed),
                ("onStreamStateChanged", self._on_stream_state_changed),
            ]
            for sig_name, slot in mappings:
                sig = getattr(self.state, sig_name, None)
                if sig is not None and hasattr(sig, "connect"):
                    try:
                        sig.connect(slot)
                        self._state_connections.append((sig, slot))
                    except (TypeError, RuntimeError):
                        pass
        self.sync_from_state()

    # ------------------------------------------------------------------
    # Dynamic detector parameter schema controls
    # ------------------------------------------------------------------
    def _rebuild_extra_param_controls(
        self,
        algorithm: str,
        initial_values: Optional[Dict[str, Any]] = None,
    ) -> None:
        while self.extra_params_layout.rowCount() > 0:
            self.extra_params_layout.removeRow(0)
        self.extra_param_controls.clear()

        detector = SpikeDetectorRegistry.get(algorithm)
        self.lbl_algo_description.setText(str(getattr(detector, "description", "") or ""))
        for spec in detector.param_schema():
            ctrl = self._create_param_widget(spec)
            self.extra_param_controls[spec.name] = ctrl
            self.extra_params_layout.addRow(f"{spec.label}:", ctrl)
        self.extra_params_widget.setVisible(len(self.extra_param_controls) > 0)

        if initial_values is not None:
            self._write_extra_params(initial_values)

    def _create_param_widget(self, spec: DetectorParamSpec) -> QWidget:
        if spec.param_type == "bool":
            w = QCheckBox(self.extra_params_widget)
            w.setChecked(bool(spec.default))
            w.toggled.connect(lambda _v, n=spec.name: self._on_extra_param_changed(n))
            return w
        if spec.param_type == "choice":
            w = QComboBox(self.extra_params_widget)
            for ch in spec.choices or []:
                w.addItem(str(ch), ch)
            idx = w.findData(spec.default)
            if idx >= 0:
                w.setCurrentIndex(idx)
            w.currentIndexChanged.connect(
                lambda _v, n=spec.name: self._on_extra_param_changed(n)
            )
            return w
        if spec.param_type == "int":
            w = QSpinBox(self.extra_params_widget)
            w.setRange(int(spec.min_val), int(spec.max_val))
            w.setSingleStep(max(1, int(spec.step)))
            w.setValue(int(spec.default))
            w.valueChanged.connect(
                lambda _v, n=spec.name: self._on_extra_param_changed(n)
            )
            return w
        w = QDoubleSpinBox(self.extra_params_widget)
        w.setDecimals(int(spec.decimals))
        w.setRange(float(spec.min_val), float(spec.max_val))
        w.setSingleStep(float(spec.step))
        w.setValue(float(spec.default))
        w.valueChanged.connect(
            lambda _v, n=spec.name: self._on_extra_param_changed(n)
        )
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

    def _effective_extra_values(self, cfg: CandidateScanConfig) -> Dict[str, Any]:
        algo = str(cfg.algorithm or "zscore")
        merged: Dict[str, Any] = {}
        if algo == "zscore":
            merged["polarity"] = "both"
            merged["z_thresh"] = float(cfg.min_peak_snr)
            merged["use_mad"] = True
            merged["min_distance_ms"] = float(cfg.min_peak_distance_ms)
        elif algo == "wavelet":
            merged["threshold_sigma"] = float(cfg.min_peak_snr)
        merged.update(dict(cfg.detector_params or {}))
        return merged

    # ------------------------------------------------------------------
    # State -> UI Sync
    # ------------------------------------------------------------------
    def sync_from_state(self) -> None:
        if self._syncing:
            return
        self._syncing = True
        try:
            cfg = (
                self.state.candidate_config
                if self.state is not None
                else CandidateScanConfig()
            )
            algo = str(cfg.algorithm or "zscore")
            with QSignalBlocker(self.combo_algorithm):
                idx = self.combo_algorithm.findData(algo)
                if idx < 0:
                    self._populate_algorithm_combo()
                    idx = self.combo_algorithm.findData(algo)
                if idx >= 0:
                    self.combo_algorithm.setCurrentIndex(idx)

            effective_extra = self._effective_extra_values(cfg)
            self._rebuild_extra_param_controls(algo, initial_values=effective_extra)

            enabled_set = set(str(a) for a in (cfg.enabled_algorithms or []))
            for key, chk in self.ensemble_checks.items():
                with QSignalBlocker(chk):
                    chk.setChecked(key in enabled_set)

            with QSignalBlocker(self.chk_auto_scan):
                self.chk_auto_scan.setChecked(
                    bool(getattr(self.state, "auto_scan_live", False))
                    if self.state is not None
                    else False
                )

            with QSignalBlocker(self.chk_template_refine):
                self.chk_template_refine.setChecked(bool(cfg.template_refine))
            with QSignalBlocker(self.spin_template_corr):
                self.spin_template_corr.setValue(float(cfg.template_corr_threshold))
            with QSignalBlocker(self.spin_max_template_sweeps):
                self.spin_max_template_sweeps.setValue(
                    int(cfg.max_template_refine_sweeps)
                )

            with QSignalBlocker(self.spin_blank_ms):
                self.spin_blank_ms.setValue(float(cfg.blank_ms))
            with QSignalBlocker(self.spin_min_peak_snr):
                self.spin_min_peak_snr.setValue(float(cfg.min_peak_snr))
            with QSignalBlocker(self.spin_min_peak_distance_ms):
                self.spin_min_peak_distance_ms.setValue(
                    float(cfg.min_peak_distance_ms)
                )
            with QSignalBlocker(self.spin_tolerance_ms):
                self.spin_tolerance_ms.setValue(float(cfg.tolerance_ms))
            with QSignalBlocker(self.spin_merge_tolerance_ms):
                self.spin_merge_tolerance_ms.setValue(float(cfg.merge_tolerance_ms))
            with QSignalBlocker(self.spin_min_hits):
                self.spin_min_hits.setValue(int(cfg.min_hits))
            with QSignalBlocker(self.spin_max_jitter_ms):
                self.spin_max_jitter_ms.setValue(float(cfg.max_jitter_ms))
            with QSignalBlocker(self.spin_max_candidates):
                self.spin_max_candidates.setValue(int(cfg.max_candidates))
            with QSignalBlocker(self.spin_max_per_channel):
                self.spin_max_per_channel.setValue(int(cfg.max_per_channel))
            with QSignalBlocker(self.spin_overlap_suppress):
                self.spin_overlap_suppress.setValue(
                    float(cfg.accepted_overlap_suppress)
                )
            with QSignalBlocker(self.spin_reactivation_margin):
                self.spin_reactivation_margin.setValue(
                    float(cfg.reactivation_score_margin)
                )

            with QSignalBlocker(self.spin_live_window_sweeps):
                self.spin_live_window_sweeps.setValue(
                    int(cfg.live_scan_window_sweeps)
                )
            with QSignalBlocker(self.spin_recency_sweeps):
                self.spin_recency_sweeps.setValue(int(cfg.recency_window_sweeps))
            with QSignalBlocker(self.spin_scan_interval_ms):
                self.spin_scan_interval_ms.setValue(int(cfg.scan_interval_ms))
            with QSignalBlocker(self.spin_review_quiet_ms):
                self.spin_review_quiet_ms.setValue(int(cfg.review_quiet_ms))
            with QSignalBlocker(self.spin_max_scan_sweeps):
                self.spin_max_scan_sweeps.setValue(int(cfg.max_scan_sweeps))

            self._refresh_candidates_list()
            self._update_status_label()
        finally:
            self._syncing = False

    def _refresh_candidates_list(self) -> None:
        if self.state is None:
            with QSignalBlocker(self.list_candidates):
                self.list_candidates.clear()
            self.btn_accept_candidate.setEnabled(False)
            self.btn_discard_candidate.setEnabled(False)
            return

        sr = 30000.0
        if hasattr(self.state, "_candidate_sampling_rate"):
            try:
                sr = float(self.state._candidate_sampling_rate()) or 30000.0
            except Exception:
                sr = 30000.0
        visible = list(self.state.get_visible_candidates())
        active_id = getattr(self.state, "active_candidate_id", None)

        with QSignalBlocker(self.list_candidates):
            self.list_candidates.clear()
            selected_row = -1
            for idx, cand in enumerate(visible):
                cid = int(cand.candidate_id)
                lat_ms = (float(cand.latency_center_samples) / sr) * 1000.0
                ev = getattr(cand, "evidence", None)
                snr = float(getattr(ev, "median_snr", 0.0) or 0.0)
                jitter = float(getattr(ev, "jitter_ms", 0.0) or 0.0)
                text = (
                    f"{cand.name} | {cand.channel} | {lat_ms:.2f} ms | "
                    f"score={float(cand.score):.2f} | hits={int(cand.hit_count)} | "
                    f"SNR={snr:.1f} | jitter={jitter:.2f} ms | {cand.algorithm}"
                )
                item = QListWidgetItem(text)
                item.setData(Qt.UserRole, cid)
                self.list_candidates.addItem(item)
                if active_id is not None and cid == int(active_id):
                    selected_row = idx
            if selected_row >= 0:
                self.list_candidates.setCurrentRow(selected_row)

        has_sel = self._selected_candidate_id() is not None
        self.btn_accept_candidate.setEnabled(has_sel)
        self.btn_discard_candidate.setEnabled(has_sel)

    def _update_status_label(self) -> None:
        if self._scanning:
            self.lbl_scan_status.setText("Scanning...")
            return
        if self.state is None:
            self.lbl_scan_status.setText("Pending: 0 | Dormant: 0 | Discarded: 0")
            return
        units = list(getattr(self.state, "candidate_units", {}).values())
        pending = sum(1 for c in units if getattr(c, "status", "") == "pending")
        dormant = sum(1 for c in units if getattr(c, "status", "") == "dormant")
        discarded = sum(1 for c in units if getattr(c, "status", "") == "discarded")
        self.lbl_scan_status.setText(
            f"Pending: {pending} | Dormant: {dormant} | Discarded: {discarded}"
        )

    # ------------------------------------------------------------------
    # UI -> State push
    # ------------------------------------------------------------------
    def _collect_config_kwargs(self) -> Dict[str, Any]:
        existing_params: Dict[str, Any] = {}
        if self.state is not None:
            existing_params = dict(
                getattr(self.state.candidate_config, "detector_params", {}) or {}
            )
        existing_params.update(self._read_extra_params())
        enabled = [
            key for key, chk in self.ensemble_checks.items() if chk.isChecked()
        ]
        return {
            "algorithm": self._selected_algorithm(),
            "enabled_algorithms": enabled,
            "detector_params": existing_params,
            "template_refine": bool(self.chk_template_refine.isChecked()),
            "template_corr_threshold": float(self.spin_template_corr.value()),
            "max_template_refine_sweeps": int(self.spin_max_template_sweeps.value()),
            "blank_ms": float(self.spin_blank_ms.value()),
            "min_peak_snr": float(self.spin_min_peak_snr.value()),
            "min_peak_distance_ms": float(self.spin_min_peak_distance_ms.value()),
            "tolerance_ms": float(self.spin_tolerance_ms.value()),
            "merge_tolerance_ms": float(self.spin_merge_tolerance_ms.value()),
            "min_hits": int(self.spin_min_hits.value()),
            "max_jitter_ms": float(self.spin_max_jitter_ms.value()),
            "max_candidates": int(self.spin_max_candidates.value()),
            "max_per_channel": int(self.spin_max_per_channel.value()),
            "accepted_overlap_suppress": float(self.spin_overlap_suppress.value()),
            "reactivation_score_margin": float(self.spin_reactivation_margin.value()),
            "live_scan_window_sweeps": int(self.spin_live_window_sweeps.value()),
            "recency_window_sweeps": int(self.spin_recency_sweeps.value()),
            "scan_interval_ms": int(self.spin_scan_interval_ms.value()),
            "review_quiet_ms": int(self.spin_review_quiet_ms.value()),
            "max_scan_sweeps": int(self.spin_max_scan_sweeps.value()),
        }

    def _push_config_to_state(self) -> None:
        if self.state is None:
            return
        kwargs = self._collect_config_kwargs()
        self._syncing = True
        try:
            self.state.update_candidate_config(**kwargs)
        finally:
            self._syncing = False

    # ------------------------------------------------------------------
    # Slots & Public Actions
    # ------------------------------------------------------------------
    def apply_preset(self, preset_name: str) -> None:
        preset = PROPOSAL_PRESETS.get(preset_name)
        if preset is None:
            return
        with QSignalBlocker(self.combo_preset):
            idx = self.combo_preset.findText(preset_name)
            if idx >= 0:
                self.combo_preset.setCurrentIndex(idx)
        if self.state is not None:
            self.state.update_candidate_config(**preset)
        else:
            self.sync_from_state()

    @Slot(str)
    def _on_preset_selected(self, preset_name: str) -> None:
        if self._syncing:
            return
        self.apply_preset(preset_name)

    def trigger_scan(self, synchronous: bool = False) -> bool:
        if self.state is None:
            return False
        scope = "live" if getattr(self.state, "is_streaming", False) else "offline"
        return bool(
            self.state.request_candidate_scan(synchronous=synchronous, scope=scope)
        )

    @Slot(bool)
    def _on_auto_scan_toggled(self, checked: bool) -> None:
        if self._syncing or self.state is None:
            return
        self.state.set_auto_scan_live(bool(checked))

    @Slot()
    def _on_rescan_discarded_clicked(self) -> None:
        if self.state is None:
            return
        self.state._candidate_rescan_discarded = True
        self.trigger_scan()

    @Slot()
    def _on_clear_candidates_clicked(self) -> None:
        if self.state is None:
            return
        self.state.clear_candidates()

    @Slot(int)
    def _on_algorithm_changed(self, *args) -> None:
        if self._syncing:
            return
        algo = self._selected_algorithm()
        existing = (
            dict(getattr(self.state.candidate_config, "detector_params", {}) or {})
            if self.state is not None
            else {}
        )
        if algo == "zscore":
            existing.setdefault("polarity", "both")
            existing["z_thresh"] = float(self.spin_min_peak_snr.value())
            existing["min_distance_ms"] = float(self.spin_min_peak_distance_ms.value())
        elif algo == "wavelet":
            existing["threshold_sigma"] = float(self.spin_min_peak_snr.value())
        self._rebuild_extra_param_controls(algo, initial_values=existing)
        self._push_config_to_state()

    @Slot(bool)
    def _on_ensemble_toggled(self, _checked: bool) -> None:
        if self._syncing:
            return
        self._push_config_to_state()

    def _on_extra_param_changed(self, param_name: str) -> None:
        if self._syncing:
            return
        ctrl = self.extra_param_controls.get(param_name)
        if ctrl is not None:
            if param_name in ("z_thresh", "threshold_sigma") and isinstance(
                ctrl, (QDoubleSpinBox, QSpinBox)
            ):
                with QSignalBlocker(self.spin_min_peak_snr):
                    self.spin_min_peak_snr.setValue(float(ctrl.value()))
            elif param_name == "min_distance_ms" and isinstance(
                ctrl, (QDoubleSpinBox, QSpinBox)
            ):
                with QSignalBlocker(self.spin_min_peak_distance_ms):
                    self.spin_min_peak_distance_ms.setValue(float(ctrl.value()))
        self._push_config_to_state()

    @Slot(float)
    def _on_snr_spin_changed(self, value: float) -> None:
        if self._syncing:
            return
        for param_name in ("z_thresh", "threshold_sigma"):
            ctrl = self.extra_param_controls.get(param_name)
            if isinstance(ctrl, QDoubleSpinBox):
                with QSignalBlocker(ctrl):
                    ctrl.setValue(float(value))
        self._push_config_to_state()

    @Slot(float)
    def _on_distance_spin_changed(self, value: float) -> None:
        if self._syncing:
            return
        ctrl = self.extra_param_controls.get("min_distance_ms")
        if isinstance(ctrl, QDoubleSpinBox):
            with QSignalBlocker(ctrl):
                ctrl.setValue(float(value))
        self._push_config_to_state()

    @Slot()
    def _on_config_control_changed(self, *args) -> None:
        if self._syncing:
            return
        self._push_config_to_state()

    def _selected_candidate_id(self) -> Optional[int]:
        item = self.list_candidates.currentItem()
        if item is not None:
            data = item.data(Qt.UserRole)
            if data is not None:
                return int(data)
        if self.state is not None and self.state.active_candidate_id is not None:
            return int(self.state.active_candidate_id)
        return None

    @Slot()
    def _on_candidate_item_selected(self, current: Optional[QListWidgetItem], _prev=None) -> None:
        if self._syncing or self.state is None:
            return
        cid = current.data(Qt.UserRole) if current is not None else None
        self.state.set_active_candidate(int(cid) if cid is not None else None)
        has_sel = cid is not None
        self.btn_accept_candidate.setEnabled(has_sel)
        self.btn_discard_candidate.setEnabled(has_sel)

    @Slot()
    def _on_accept_selected_clicked(self) -> None:
        if self.state is None:
            return
        cid = self._selected_candidate_id()
        if cid is not None:
            self.state.accept_candidate(cid)

    @Slot()
    def _on_discard_selected_clicked(self) -> None:
        if self.state is None:
            return
        cid = self._selected_candidate_id()
        if cid is not None:
            self.state.discard_candidate(cid)

    @Slot()
    def _on_state_config_changed(self) -> None:
        if not self._syncing:
            self.sync_from_state()

    @Slot()
    def _on_candidates_changed(self) -> None:
        self._refresh_candidates_list()
        self._update_status_label()

    @Slot(int)
    def _on_candidate_selection_changed(self, candidate_id: int) -> None:
        with QSignalBlocker(self.list_candidates):
            if candidate_id is None or int(candidate_id) < 0:
                self.list_candidates.clearSelection()
                self.list_candidates.setCurrentRow(-1)
            else:
                for row in range(self.list_candidates.count()):
                    item = self.list_candidates.item(row)
                    if item is not None and int(item.data(Qt.UserRole)) == int(
                        candidate_id
                    ):
                        self.list_candidates.setCurrentRow(row)
                        break
        has_sel = self._selected_candidate_id() is not None
        self.btn_accept_candidate.setEnabled(has_sel)
        self.btn_discard_candidate.setEnabled(has_sel)

    @Slot(bool)
    def _on_scan_state_changed(self, scanning: bool) -> None:
        self._scanning = bool(scanning)
        self.btn_scan_now.setEnabled(not self._scanning)
        self._update_status_label()

    @Slot(bool)
    def _on_stream_state_changed(self, _streaming: bool) -> None:
        with QSignalBlocker(self.chk_auto_scan):
            self.chk_auto_scan.setChecked(
                bool(getattr(self.state, "auto_scan_live", False))
                if self.state is not None
                else False
            )

    # ------------------------------------------------------------------
    # Perspective Settings Persistence
    # ------------------------------------------------------------------
    def get_settings(self) -> Dict[str, Any]:
        if self.state is not None:
            data = asdict(self.state.candidate_config)
            data["auto_scan_live"] = bool(getattr(self.state, "auto_scan_live", False))
        else:
            data = self._collect_config_kwargs()
            data["auto_scan_live"] = bool(self.chk_auto_scan.isChecked())
        data["preset"] = self.combo_preset.currentText()
        return data

    def set_settings(self, settings: Optional[Dict[str, Any]]) -> None:
        if not isinstance(settings, dict):
            return
        preset_name = settings.get("preset")
        if isinstance(preset_name, str):
            with QSignalBlocker(self.combo_preset):
                idx = self.combo_preset.findText(preset_name)
                if idx >= 0:
                    self.combo_preset.setCurrentIndex(idx)

        config_fields = set(asdict(CandidateScanConfig()).keys())
        updates = {k: v for k, v in settings.items() if k in config_fields}
        if self.state is not None:
            if updates:
                self.state.update_candidate_config(**updates)
            if "auto_scan_live" in settings:
                self.state.set_auto_scan_live(bool(settings["auto_scan_live"]))
        self.sync_from_state()
