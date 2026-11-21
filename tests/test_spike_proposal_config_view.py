import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import neo
import numpy as np
import pytest
import quantities as pq
from PySide6.QtCore import Qt

from spikespy.candidate_detection import (
    CandidateScanConfig,
    _detector_params,
    detect_candidates,
)
from spikespy.SpikeGroupTable import SpikeGroupTableView
from spikespy.SpikeProposalConfigView import PROPOSAL_PRESETS, SpikeProposalConfigView
from spikespy.SpikeSpy import MdiView, window_options
from spikespy.ViewerState import ViewerState


def _make_state_with_peaks(peaks_by_sweep, *, amplitude=1.0, n_samples=1000):
    state = ViewerState()
    n_sweeps = len(peaks_by_sweep)
    times = np.arange(n_sweeps) * 0.1 * pq.s
    event = neo.Event(times, name="stim", units=pq.s)
    rng = np.random.default_rng(42)
    data = rng.normal(0.0, 0.02, size=(n_sweeps, n_samples)).astype(np.float32)
    for i, peak in enumerate(peaks_by_sweep):
        if peak is not None:
            data[i, int(peak)] += float(amplitude)
    sig = neo.AnalogSignal(
        data.flatten(), units=pq.mV, sampling_rate=10000.0 * pq.Hz, name="CH0"
    )
    seg = neo.Segment("proposal_cfg_test")
    seg.analogsignals.append(sig)
    seg.events.append(event)
    state.segment = seg
    state.window_size = 60 * pq.ms
    state.set_data(analog_signal=sig, event_signal=event, spike_groups=[])
    return state


def test_config_view_initialization_and_bidirectional_sync(qtbot):
    state = _make_state_with_peaks([120] * 10)
    view = SpikeProposalConfigView(state=state)
    qtbot.addWidget(view)

    # Initial defaults match CandidateScanConfig
    cfg = state.candidate_config
    assert view.combo_algorithm.currentData() == cfg.algorithm
    assert view.chk_template_refine.isChecked() is cfg.template_refine
    assert view.spin_template_corr.value() == pytest.approx(cfg.template_corr_threshold)
    assert view.spin_max_template_sweeps.value() == cfg.max_template_refine_sweeps
    assert view.spin_blank_ms.value() == pytest.approx(cfg.blank_ms)
    assert view.spin_min_peak_snr.value() == pytest.approx(cfg.min_peak_snr)
    assert view.spin_min_peak_distance_ms.value() == pytest.approx(cfg.min_peak_distance_ms)
    assert view.spin_tolerance_ms.value() == pytest.approx(cfg.tolerance_ms)
    assert view.spin_merge_tolerance_ms.value() == pytest.approx(cfg.merge_tolerance_ms)
    assert view.spin_min_hits.value() == cfg.min_hits
    assert view.spin_max_jitter_ms.value() == pytest.approx(cfg.max_jitter_ms)
    assert view.spin_max_candidates.value() == cfg.max_candidates
    assert view.spin_max_per_channel.value() == cfg.max_per_channel
    assert view.spin_overlap_suppress.value() == pytest.approx(cfg.accepted_overlap_suppress)
    assert view.spin_reactivation_margin.value() == pytest.approx(cfg.reactivation_score_margin)
    assert view.spin_live_window_sweeps.value() == cfg.live_scan_window_sweeps
    assert view.spin_recency_sweeps.value() == cfg.recency_window_sweeps
    assert view.spin_scan_interval_ms.value() == cfg.scan_interval_ms
    assert view.spin_review_quiet_ms.value() == cfg.review_quiet_ms
    assert view.spin_max_scan_sweeps.value() == cfg.max_scan_sweeps

    # UI -> State updates across sections
    with qtbot.waitSignal(state.onCandidateConfigChanged):
        view.spin_min_hits.setValue(5)
    assert state.candidate_config.min_hits == 5

    view.spin_blank_ms.setValue(2.5)
    view.spin_tolerance_ms.setValue(1.8)
    view.spin_merge_tolerance_ms.setValue(1.4)
    view.spin_max_jitter_ms.setValue(3.2)
    view.spin_max_candidates.setValue(10)
    view.spin_max_per_channel.setValue(4)
    view.spin_overlap_suppress.setValue(0.75)
    view.spin_reactivation_margin.setValue(0.25)
    assert state.candidate_config.blank_ms == pytest.approx(2.5)
    assert state.candidate_config.tolerance_ms == pytest.approx(1.8)
    assert state.candidate_config.merge_tolerance_ms == pytest.approx(1.4)
    assert state.candidate_config.max_jitter_ms == pytest.approx(3.2)
    assert state.candidate_config.max_candidates == 10
    assert state.candidate_config.max_per_channel == 4
    assert state.candidate_config.accepted_overlap_suppress == pytest.approx(0.75)
    assert state.candidate_config.reactivation_score_margin == pytest.approx(0.25)

    view.chk_template_refine.setChecked(False)
    view.spin_template_corr.setValue(0.65)
    view.spin_max_template_sweeps.setValue(500)
    assert state.candidate_config.template_refine is False
    assert state.candidate_config.template_corr_threshold == pytest.approx(0.65)
    assert state.candidate_config.max_template_refine_sweeps == 500

    view.spin_live_window_sweeps.setValue(60)
    view.spin_recency_sweeps.setValue(15)
    view.spin_scan_interval_ms.setValue(3500)
    view.spin_review_quiet_ms.setValue(1500)
    view.spin_max_scan_sweeps.setValue(5000)
    assert state.candidate_config.live_scan_window_sweeps == 60
    assert state.candidate_config.recency_window_sweeps == 15
    assert state.candidate_config.scan_interval_ms == 3500
    assert state.candidate_config.review_quiet_ms == 1500
    assert state.candidate_config.max_scan_sweeps == 5000

    # Dynamic detector parameters & bidirectional SNR / distance sync
    assert "z_thresh" in view.extra_param_controls
    assert "polarity" in view.extra_param_controls
    assert "use_mad" in view.extra_param_controls
    assert "min_distance_ms" in view.extra_param_controls

    view.spin_min_peak_snr.setValue(4.25)
    assert view.extra_param_controls["z_thresh"].value() == pytest.approx(4.25)
    assert state.candidate_config.detector_params["z_thresh"] == pytest.approx(4.25)

    view.extra_param_controls["z_thresh"].setValue(5.5)
    assert view.spin_min_peak_snr.value() == pytest.approx(5.5)
    assert state.candidate_config.min_peak_snr == pytest.approx(5.5)

    view.extra_param_controls["min_distance_ms"].setValue(1.7)
    assert view.spin_min_peak_distance_ms.value() == pytest.approx(1.7)
    assert state.candidate_config.min_peak_distance_ms == pytest.approx(1.7)

    pol_combo = view.extra_param_controls["polarity"]
    pol_combo.setCurrentIndex(pol_combo.findData("positive"))
    assert state.candidate_config.detector_params["polarity"] == "positive"

    # Switch algorithm to wavelet -> dynamic controls rebuild
    idx_wav = view.combo_algorithm.findData("wavelet")
    assert idx_wav >= 0
    view.combo_algorithm.setCurrentIndex(idx_wav)
    assert state.candidate_config.algorithm == "wavelet"
    assert "wavelet" in view.extra_param_controls
    assert "level" in view.extra_param_controls
    assert "threshold_sigma" in view.extra_param_controls
    assert "Denoises" in view.lbl_algo_description.text()

    # Ensemble checkboxes
    view.ensemble_checks["zscore"].setChecked(True)
    view.ensemble_checks["wavelet"].setChecked(True)
    assert set(state.candidate_config.enabled_algorithms) == {"zscore", "wavelet"}

    # State -> UI sync when update_candidate_config is called externally
    state.update_candidate_config(
        algorithm="zscore",
        enabled_algorithms=[],
        min_hits=7,
        min_peak_snr=3.75,
        template_refine=True,
    )
    assert view.combo_algorithm.currentData() == "zscore"
    assert view.spin_min_hits.value() == 7
    assert view.spin_min_peak_snr.value() == pytest.approx(3.75)
    assert view.chk_template_refine.isChecked() is True
    assert not view.ensemble_checks["zscore"].isChecked()
    assert not view.ensemble_checks["wavelet"].isChecked()


def test_presets_and_settings_roundtrip(qtbot):
    state = _make_state_with_peaks([120] * 10)
    view = SpikeProposalConfigView(state=state)
    qtbot.addWidget(view)

    for preset_name in PROPOSAL_PRESETS:
        view.combo_preset.setCurrentText(preset_name)
        expected = PROPOSAL_PRESETS[preset_name]
        assert state.candidate_config.algorithm == expected["algorithm"]
        assert list(state.candidate_config.enabled_algorithms) == list(
            expected.get("enabled_algorithms", [])
        )

    view.apply_preset("Multi-Detector Ensemble (Z-Score + Wavelet)")
    assert state.candidate_config.enabled_algorithms == ["zscore", "wavelet"]
    assert view.ensemble_checks["zscore"].isChecked()
    assert view.ensemble_checks["wavelet"].isChecked()

    saved = view.get_settings()
    assert saved["preset"] == "Multi-Detector Ensemble (Z-Score + Wavelet)"
    assert saved["enabled_algorithms"] == ["zscore", "wavelet"]

    state2 = _make_state_with_peaks([120] * 10)
    view2 = SpikeProposalConfigView(state=state2)
    qtbot.addWidget(view2)
    view2.set_settings(saved)
    assert state2.candidate_config.enabled_algorithms == ["zscore", "wavelet"]
    assert view2.combo_preset.currentText() == "Multi-Detector Ensemble (Z-Score + Wavelet)"
    assert view2.ensemble_checks["zscore"].isChecked()
    assert view2.ensemble_checks["wavelet"].isChecked()


def test_scan_preview_accept_discard_and_clear_from_view(qtbot):
    state = ViewerState()
    n_sweeps = 16
    times = np.arange(n_sweeps) * 0.1 * pq.s
    event = neo.Event(times, name="stim", units=pq.s)
    data = np.zeros((n_sweeps, 1000), dtype=np.float32)
    data[:, 120] = 1.5
    sig = neo.AnalogSignal(
        data.flatten(), units=pq.mV, sampling_rate=10000.0 * pq.Hz, name="CH0"
    )
    seg = neo.Segment("proposal_cfg_test")
    seg.analogsignals.append(sig)
    seg.events.append(event)
    state.segment = seg
    state.window_size = 20 * pq.ms
    state.set_data(analog_signal=sig, event_signal=event, spike_groups=[])

    view = SpikeProposalConfigView(state=state)
    qtbot.addWidget(view)

    assert view.trigger_scan(synchronous=True)
    assert view.list_candidates.count() == 1
    assert "Pending: 1" in view.lbl_scan_status.text()

    # Selecting candidate in list previews it on state
    view.list_candidates.setCurrentRow(0)
    cid = view.list_candidates.item(0).data(Qt.UserRole)
    assert state.active_candidate_id == cid

    # Discard selected candidate
    view.btn_discard_candidate.click()
    assert state.candidate_units[cid].status == "discarded"
    assert view.list_candidates.count() == 0
    assert "Discarded: 1" in view.lbl_scan_status.text()

    # Rescan discarded revives if stronger
    state.candidate_units[cid].score_at_discard = -1.0
    view.btn_rescan_discarded.click()
    qtbot.waitUntil(lambda: view.list_candidates.count() == 1, timeout=5000)

    # Select and accept
    view.list_candidates.setCurrentRow(0)
    view.btn_accept_candidate.click()
    assert len(state.spike_groups) == 2
    assert view.list_candidates.count() == 0

    # Run another scan on another peak and clear
    view.btn_clear_candidates.click()
    assert state.candidate_units == {}
    assert "Pending: 0 | Dormant: 0 | Discarded: 0" in view.lbl_scan_status.text()

    # Auto scan toggle syncs with state
    view.chk_auto_scan.setChecked(True)
    assert state.auto_scan_live is True
    view.chk_auto_scan.setChecked(False)
    assert state.auto_scan_live is False


def test_update_candidate_config_updates_auto_scan_interval(qtbot):
    state = ViewerState()
    state.start_live_session(
        sampling_rate=10000.0, window_size_ms=20.0, units=pq.mV, channel_name="CH0"
    )
    try:
        state.set_auto_scan_live(True)
        state.update_candidate_config(scan_interval_ms=750)
        assert state.candidate_config.scan_interval_ms == 750
        assert state._candidate_scan_manager is not None
        assert state._candidate_scan_manager._timer.interval() == 750
    finally:
        state.set_auto_scan_live(False)
        state.stop_live_session()


def test_detector_params_override_and_multi_algorithm_ensemble():
    cfg_pos = CandidateScanConfig(
        algorithm="zscore",
        min_peak_snr=3.0,
        detector_params={"polarity": "positive", "use_mad": False, "z_thresh": 4.0},
        template_refine=False,
    )
    p = _detector_params(cfg_pos, sigma=0.1)
    assert p["polarity"] == "positive"
    assert p["use_mad"] is False
    assert p["z_thresh"] == pytest.approx(4.0)

    # With polarity="positive", negative-only peaks are ignored while positive peaks are found
    erp_neg = np.zeros((10, 300), dtype=np.float32)
    erp_neg[:, 120] = -1.0
    assert detect_candidates({"CH0": erp_neg}, 10000.0, 0, 9, config=cfg_pos) == []

    erp_pos = np.zeros((10, 300), dtype=np.float32)
    erp_pos[:, 120] = 1.0
    pos_props = detect_candidates({"CH0": erp_pos}, 10000.0, 0, 9, config=cfg_pos)
    assert len(pos_props) == 1

    # Multi-algorithm ensemble calls both detectors and dedupes
    called_algos = []

    def tracking_peak_finder(algo, trace, fs_hz, params):
        called_algos.append(algo)
        from spikespy.dsp.detectors import SpikeDetectorRegistry

        return SpikeDetectorRegistry.find_candidates(
            algo, trace, fs_hz=fs_hz, params=params
        )

    rng = np.random.default_rng(11)
    erp = rng.normal(0.0, 0.005, size=(12, 300)).astype(np.float32)
    erp[:, 140] += 1.5
    cfg_ens = CandidateScanConfig(
        algorithm="zscore",
        enabled_algorithms=["zscore", "wavelet"],
        min_peak_snr=4.0,
        template_refine=False,
    )
    proposals = detect_candidates(
        {"CH0": erp}, 10000.0, 0, 11, config=cfg_ens, peak_finder=tracking_peak_finder
    )
    assert "zscore" in called_algos
    assert "wavelet" in called_algos
    assert len(proposals) == 1
    assert abs(proposals[0].latency_center_samples - 140.0) <= 2.0


def test_mdi_and_spike_group_table_integration(qtbot):
    assert "SpikeProposalConfig" in window_options
    assert window_options["SpikeProposalConfig"] is SpikeProposalConfigView

    state = _make_state_with_peaks([120] * 10)
    mdi = MdiView(state=state, check_updates=False)
    qtbot.addWidget(mdi)

    assert hasattr(mdi, "act_proposal_config")
    w1 = mdi.open_or_show_spike_proposal_config()
    assert isinstance(w1, SpikeProposalConfigView)
    w2 = mdi.open_or_show_spike_proposal_config()
    assert w2 is w1

    docks = [
        dw
        for dw in mdi.dock_manager.dockWidgets()
        if isinstance(dw.widget(), SpikeProposalConfigView)
    ]
    assert len(docks) == 1

    # SpikeGroupTableView has proposalConfigButton and opens the panel
    table = mdi.newWindow("SpikeGroupTable")
    assert isinstance(table, SpikeGroupTableView)
    assert hasattr(table, "proposalConfigButton")
    table.proposalConfigButton.click()
    docks_after = [
        dw
        for dw in mdi.dock_manager.dockWidgets()
        if isinstance(dw.widget(), SpikeProposalConfigView)
    ]
    assert len(docks_after) == 1


def _make_multichannel_state(channel_peaks, *, n_sweeps=20, n_samples=1000):
    """Build a multi-channel ViewerState with per-channel peak locations by sweep."""
    state = ViewerState()
    times = np.arange(n_sweeps) * 0.1 * pq.s
    event = neo.Event(times, name="stim", units=pq.s)
    seg = neo.Segment("multich_candidate_test")
    signals = []
    for ch_name, peaks_by_sweep in channel_peaks.items():
        data = np.zeros((n_sweeps, n_samples), dtype=np.float32)
        for i, peak in enumerate(peaks_by_sweep):
            if peak is not None and i < n_sweeps:
                data[i, int(peak)] = 1.5
        sig = neo.AnalogSignal(
            data.flatten(), units=pq.mV, sampling_rate=10000.0 * pq.Hz, name=ch_name
        )
        seg.analogsignals.append(sig)
        signals.append(sig)
    seg.events.append(event)
    state.segment = seg
    state.window_size = 40 * pq.ms
    state.set_data(analog_signal=signals[0], event_signal=event, spike_groups=[])
    return state


def test_specify_scan_channels_controls_which_channels_are_scanned(qtbot):
    state = _make_multichannel_state(
        {
            "CH0": [120] * 20,
            "CH1": [200] * 20,
            "CH2": [280] * 20,
        }
    )
    view = SpikeProposalConfigView(state=state)
    qtbot.addWidget(view)

    # Defaults to Auto (active channel CH0 only)
    assert view.chk_scan_use_active_channels.isChecked()
    assert state.candidate_config.scan_channels == []
    assert view.list_scan_channels.count() == 3
    assert view.trigger_scan(synchronous=True)
    assert {c.channel for c in state.get_visible_candidates()} == {"CH0"}

    # Select only CH1 and CH2 via the channel checklist
    state.clear_candidates()
    view.btn_scan_channels_none.click()
    assert not view.chk_scan_use_active_channels.isChecked()
    for row in range(view.list_scan_channels.count()):
        item = view.list_scan_channels.item(row)
        if item.text() in ("CH1", "CH2"):
            item.setCheckState(Qt.Checked)

    assert state.candidate_config.scan_channels == ["CH1", "CH2"]
    assert "2 of 3" in view.lbl_scan_channels_summary.text()
    assert view.trigger_scan(synchronous=True)
    assert {c.channel for c in state.get_visible_candidates()} == {"CH1", "CH2"}

    # Settings roundtrip preserves scan_channels
    saved = view.get_settings()
    assert saved["scan_channels"] == ["CH1", "CH2"]
    state2 = _make_multichannel_state({"CH0": [120] * 10, "CH1": [200] * 10, "CH2": [280] * 10})
    view2 = SpikeProposalConfigView(state=state2)
    qtbot.addWidget(view2)
    view2.set_settings(saved)
    assert state2.candidate_config.scan_channels == ["CH1", "CH2"]
    assert not view2.chk_scan_use_active_channels.isChecked()

    # Clicking "All" scans all available channels
    state.clear_candidates()
    view.btn_scan_channels_all.click()
    assert state.candidate_config.scan_channels == ["CH0", "CH1", "CH2"]
    assert view.trigger_scan(synchronous=True)
    assert {c.channel for c in state.get_visible_candidates()} == {"CH0", "CH1", "CH2"}

    # Re-enabling Auto resets scan_channels to []
    view.chk_scan_use_active_channels.setChecked(True)
    assert state.candidate_config.scan_channels == []


def test_clicking_candidate_displays_as_current_unit_across_ui_and_quick_discard(qtbot):
    from spikespy.ActiveUnitInfoView import ActiveUnitInfoView
    from spikespy.MultiTraceFixedPGView import MultiTraceFixedPGView
    from spikespy.SingleTracePGView import SingleTracePGView
    from spikespy.UnitFootprintPGView import UnitFootprintPGView
    from spikespy.UnitPGView import UnitPGView

    # CH0 fires on all sweeps; CH2 fires only on sweeps 10..19
    state = _make_multichannel_state(
        {
            "CH0": [120] * 20,
            "CH1": [None] * 20,
            "CH2": [None] * 10 + [260] * 10,
        }
    )
    state.stimno = 0

    proposal_view = SpikeProposalConfigView(state=state)
    info_view = ActiveUnitInfoView(state=state)
    unit_view = UnitPGView(state=state)
    footprint_view = UnitFootprintPGView(state=state)
    single_view = SingleTracePGView(state=state)
    fixed_view = MultiTraceFixedPGView(state=state)
    for w in (proposal_view, info_view, unit_view, footprint_view, single_view, fixed_view):
        qtbot.addWidget(w)

    # Scan all channels
    proposal_view.btn_scan_channels_all.click()
    assert proposal_view.trigger_scan(synchronous=True)
    assert proposal_view.list_candidates.count() == 2

    # Find the CH2 candidate row and click it
    ch2_row = None
    ch2_cid = None
    for row in range(proposal_view.list_candidates.count()):
        item = proposal_view.list_candidates.item(row)
        cid = int(item.data(Qt.UserRole))
        if state.candidate_units[cid].channel == "CH2":
            ch2_row = row
            ch2_cid = cid
            break
    assert ch2_row is not None and ch2_cid is not None

    ch2_cand = state.candidate_units[ch2_cid]
    proposal_view.list_candidates.setCurrentRow(ch2_row)
    proposal_view._on_candidate_item_clicked(proposal_view.list_candidates.item(ch2_row))

    # State focused on CH2 and jumped to sweep 10 (nearest sweep with a detection)
    assert state.active_candidate_id == ch2_cid
    assert state.analog_signal.name == "CH2"
    assert state.stimno == 10

    # UnitPGView, UnitFootprintPGView, ActiveUnitInfoView, SingleTrace, MultiTraceFixed all reflect the candidate
    assert unit_view._get_target_unit() is ch2_cand.unit
    assert len(unit_view.lines) == 10
    assert footprint_view.get_target_unit() is ch2_cand.unit
    assert ch2_cand.name in footprint_view.windowTitle()

    assert info_view.lbl_health_badge.text() == "CANDIDATE"
    assert ch2_cand.name in info_view.lbl_unit_title.text()
    assert not info_view.candidate_actions_widget.isHidden()

    single_view.update_figure(force=True)
    assert single_view.candidate_line.isVisible()
    assert single_view.candidate_marker.isVisible()

    fixed_view.update_figure(force=True)
    assert fixed_view.candidate_scatter.isVisible()

    # Quick discard removes CH2 candidate and auto-advances to the remaining CH0 candidate
    proposal_view.btn_discard_candidate.click()
    assert state.candidate_units[ch2_cid].status == "discarded"
    assert proposal_view.list_candidates.count() == 1
    remaining_cid = int(proposal_view.list_candidates.item(0).data(Qt.UserRole))
    assert state.active_candidate_id == remaining_cid
    assert state.analog_signal.name == "CH0"
    assert unit_view._get_target_unit() is state.candidate_units[remaining_cid].unit

    # Selecting a regular unit cleanly exits candidate preview across the UI
    state.setUnitGroup(0)
    assert state.active_candidate_id is None
    assert info_view.candidate_actions_widget.isHidden()
    assert unit_view._get_target_unit() is state.spike_groups[0]
    assert footprint_view.get_target_unit() is state.spike_groups[0]

