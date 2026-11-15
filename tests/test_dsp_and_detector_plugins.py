import ast
from pathlib import Path

import neo
import numpy as np
import pytest
import quantities as pq
from PySide6.QtWidgets import QDoubleSpinBox

from spikespy.dsp import (
    BasicWindowThresholdDetector,
    CandidatePeaks,
    DetectorParamSpec,
    MultiChannelTemplateDetector,
    SpikeDetectorPlugin,
    SpikeDetectorRegistry,
    WaveletDenoiseDetector,
    ZScoreThresholdDetector,
    auto_detect_channels_from_footprint,
    compute_isi_and_firing_rate,
    compute_latency_histogram,
    compute_unit_footprint_from_state,
    register_spike_detector,
    select_spikes_in_polygon,
)
from spikespy.SingleTracePGView import SingleTracePGView
from spikespy.SpikeGroupTable import SpikeGroupTableModel
from spikespy.TrackingView import TrackingView
from spikespy.UnitFootprintPGView import UnitFootprintPGView
from spikespy.UnitSuggestor import UnitSuggestor, suggest_units_with_detector
from spikespy.ViewerState import ViewerState, tracked_neuron_unit
from spikespy.ViewSettingsPanel import ViewSettingsPanel


def _make_synthetic_state(n_sweeps: int = 6, fs_hz: float = 10000.0) -> ViewerState:
    """Create an in-memory ViewerState with synthetic multi-channel sweeps and spikes."""
    state = ViewerState()
    n_samples = int(fs_hz * 1.5)  # 1.5s total
    rng = np.random.default_rng(42)
    ch0 = rng.normal(0.0, 0.005, size=n_samples)
    ch1 = rng.normal(0.0, 0.005, size=n_samples)

    stim_times = np.array([0.1 + 0.2 * i for i in range(n_sweeps)], dtype=np.float64)
    spike_lat_s = 0.025  # 25 ms latency (sample 250 in sweep)

    # Inject a sharp spike on ch0 and slightly delayed spike on ch1 for each sweep
    kernel_t = np.linspace(-0.001, 0.001, 21)
    spike_kernel = 0.5 * np.exp(-((kernel_t / 0.00025) ** 2))
    for st in stim_times:
        center0 = int(round((st + spike_lat_s) * fs_hz))
        center1 = center0 + 3
        if 10 <= center0 < n_samples - 11:
            ch0[center0 - 10 : center0 + 11] += spike_kernel
        if 10 <= center1 < n_samples - 11:
            ch1[center1 - 10 : center1 + 11] += 0.6 * spike_kernel

    seg = neo.Segment(name="synthetic")
    sig0 = neo.AnalogSignal(
        ch0.reshape(-1, 1) * pq.mV,
        t_start=0.0 * pq.s,
        sampling_rate=fs_hz * pq.Hz,
        name="CH0",
    )
    sig1 = neo.AnalogSignal(
        ch1.reshape(-1, 1) * pq.mV,
        t_start=0.0 * pq.s,
        sampling_rate=fs_hz * pq.Hz,
        name="CH1",
    )
    ev = neo.Event(times=stim_times * pq.s, name="Trig")
    seg.analogsignals.extend([sig0, sig1])
    seg.events.append(ev)
    state.segment = seg

    u_ev = neo.Event(np.array([]) * pq.s, name="unit_0")
    sg = tracked_neuron_unit(event=u_ev, channel="CH0")
    state.set_data(analog_signal=sig0, event_signal=ev, spike_groups=[sg])
    return state


def test_dsp_package_has_zero_qt_imports():
    """Verify every module in spikespy/dsp/ has zero PySide6 or pyqtgraph imports."""
    dsp_dir = Path(__file__).resolve().parent.parent / "spikespy" / "dsp"
    py_files = sorted(dsp_dir.glob("*.py"))
    assert len(py_files) >= 6

    forbidden = {"PySide6", "pyqtgraph", "QtAds", "PySide6QtAds"}
    for path in py_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_pkg = alias.name.split(".")[0]
                    assert root_pkg not in forbidden, f"{path.name} imports {alias.name}"
            elif isinstance(node, ast.ImportFrom) and node.module:
                root_pkg = node.module.split(".")[0]
                assert root_pkg not in forbidden, f"{path.name} imports from {node.module}"


def test_builtin_detectors_and_custom_plugin_registration():
    """Verify built-in detectors and custom @register_spike_detector plugins work."""
    keys = SpikeDetectorRegistry.list_keys()
    for expected in ("basic", "template", "zscore", "wavelet"):
        assert expected in keys

    fs_hz = 10000.0
    rng = np.random.default_rng(7)
    trace = rng.normal(0.0, 0.005, size=1000)
    trace[250] = 0.8  # Clear positive spike at index 250

    for key in ("basic", "template", "zscore", "wavelet"):
        det = SpikeDetectorRegistry.get(key)
        cands = det.find_candidates(
            trace, fs_hz, {"threshold": 0.1, "z_thresh": 3.0, "threshold_sigma": 3.0}
        )
        assert isinstance(cands, CandidatePeaks)
        assert 250 in cands.indices
        tracked_idx = det.track_sweep(
            trace,
            fs_hz,
            expected_idx=245,
            params={
                "threshold": 0.1,
                "z_thresh": 3.0,
                "threshold_sigma": 3.0,
                "window_ms": 5.0,
            },
        )
        assert tracked_idx == 250

    # Register a custom detector plugin via decorator
    @register_spike_detector
    class CustomMaxDetector(SpikeDetectorPlugin):
        key = "custom_max_test"
        display_name = "Custom Max Test Detector"
        supports_multi_channel = False
        param_schema = (
            DetectorParamSpec(
                "min_amp",
                "Min Amplitude",
                "float",
                default=0.2,
                min_val=0.0,
                max_val=10.0,
                step=0.05,
            ),
        )

        def find_candidates(self, trace_1d, fs_hz, params=None, **kwargs):
            p = self.merge_params(params)
            idx = int(np.argmax(trace_1d))
            if trace_1d[idx] >= float(p["min_amp"]):
                return CandidatePeaks(
                    indices=np.array([idx], dtype=np.int64),
                    amplitudes=np.array([trace_1d[idx]], dtype=np.float64),
                )
            return CandidatePeaks(
                indices=np.array([], dtype=np.int64),
                amplitudes=np.array([], dtype=np.float64),
            )

        def track_sweep(self, trace_1d, fs_hz, expected_idx, params=None, **kwargs):
            cands = self.find_candidates(trace_1d, fs_hz, params)
            return int(cands.indices[0]) if len(cands.indices) else None

    try:
        assert "custom_max_test" in SpikeDetectorRegistry.list_keys()
        det = SpikeDetectorRegistry.get("custom_max_test")
        assert det.track_sweep(trace, fs_hz, 200, {"min_amp": 0.5}) == 250
        assert det.track_sweep(trace, fs_hz, 200, {"min_amp": 0.95}) is None
    finally:
        SpikeDetectorRegistry.unregister("custom_max_test")


def test_detector_reuse_across_viewerstate_trackingview_singletrace_and_suggestor(qtbot):
    """Verify registered detectors are reusable across ViewerState, TrackingView, SingleTracePGView, and UnitSuggestor."""
    state = _make_synthetic_state(n_sweeps=5)

    # Seed unit 0 at stim 0 (latency ~25ms -> sample 250)
    state.setStimNo(0)
    state.setUnit(250)
    sg = state.spike_groups[0]
    sg.auto_track = True
    sg.tracking_algorithm = "zscore"
    sg.tracking_params["z_thresh"] = 3.0
    sg.window = 5.0 * pq.ms

    undo_len_before = len(state.undo_queue)
    tracked_results = state.track_stimulus_all_units(1)
    assert len(tracked_results) == 1
    assert tracked_results[0]["unit_idx"] == 0
    assert tracked_results[0]["sample"] == 250
    # Automated tracking must not bloat the manual undo stack
    assert len(state.undo_queue) == undo_len_before

    # TrackingView exposes all registered detectors and builds dynamic parameter controls
    tv = TrackingView(state=state)
    qtbot.addWidget(tv)
    algo_keys = [tv.qcb_algorithm.itemData(i) for i in range(tv.qcb_algorithm.count())]
    for key in ("basic", "template", "zscore", "wavelet"):
        assert key in algo_keys

    z_idx = tv.qcb_algorithm.findData("zscore")
    tv.qcb_algorithm.setCurrentIndex(z_idx)
    assert "z_thresh" in tv.extra_param_controls
    ctrl = tv.extra_param_controls["z_thresh"]
    assert isinstance(ctrl, QDoubleSpinBox)
    ctrl.setValue(4.25)
    assert sg.tracking_params["z_thresh"] == pytest.approx(4.25)

    # SingleTracePGView reuses SpikeDetectorRegistry for candidate discovery and uses native peak downsampling
    st_view = SingleTracePGView(state=state)
    qtbot.addWidget(st_view)
    assert st_view.trace_curve.opts.get("downsampleMethod") == "peak"
    cands = st_view.find_candidates_with_detector(
        algorithm="zscore", params={"z_thresh": 3.0}, stim_idx=0
    )
    assert 250 in cands.indices

    # UnitSuggestor reuses SpikeDetectorRegistry
    suggested = suggest_units_with_detector(
        state,
        algorithm="zscore",
        params={"z_thresh": 3.5},
    )
    assert len(suggested) >= 1


def test_qt_responsiveness_table_incremental_update_and_window_cache(qtbot):
    """Verify SpikeGroupTableModel emits dataChanged (not modelReset) on spike updates and get_window() is cached."""
    state = _make_synthetic_state(n_sweeps=4)
    model = SpikeGroupTableModel(lambda: state.spike_groups)
    state.onUnitChange.connect(model.update)

    reset_count = 0
    data_changed_count = 0

    def on_reset():
        nonlocal reset_count
        reset_count += 1

    def on_data_changed(*_args):
        nonlocal data_changed_count
        data_changed_count += 1

    model.modelReset.connect(on_reset)
    model.dataChanged.connect(on_data_changed)

    # Updating spike times without changing the number of units should emit dataChanged only
    state.setStimNo(0)
    state.setUnit(250)
    assert reset_count == 0
    assert data_changed_count >= 1

    # Verify O(1) cached get_window() on tracked_neuron_unit and invalidation on setUnit
    sg = state.spike_groups[0]
    w1 = sg.get_window()
    assert sg._cached_window == w1
    state.setStimNo(1)
    state.setUnit(250)
    w2 = sg.get_window()
    assert sg._cached_window == w2


def test_view_settings_panel_retains_configurable_view(qtbot):
    """Verify ViewSettingsPanel stays bound to the last configurable visualizer when non-configurable panels are clicked."""
    state = _make_synthetic_state(n_sweeps=3)
    panel = ViewSettingsPanel()
    qtbot.addWidget(panel)

    st_view = SingleTracePGView(state=state)
    qtbot.addWidget(st_view)
    tv = TrackingView(state=state)
    qtbot.addWidget(tv)

    panel.setActiveView(st_view)
    assert panel.active_view is st_view

    # Clicking TrackingView should retain SingleTracePGView in the settings inspector
    panel.setActiveView(tv)
    assert panel.active_view is st_view
    # Explicit None clears active_view
    panel.setActiveView(None)
    assert panel.active_view is None


def test_unit_footprint_pg_view_live_sweep_overlay_and_dsp_helpers(qtbot):
    """Verify UnitFootprintPGView delegates to spikespy.dsp.footprint and renders latest_sweep_waveforms."""
    state = _make_synthetic_state(n_sweeps=4)
    state.setStimNo(0)
    state.setUnit(250)
    state.setStimNo(1)
    state.setUnit(250)

    sg = state.spike_groups[0]
    fp = compute_unit_footprint_from_state(
        state, sg, include_latest_sweep=True
    )
    assert fp is not None
    assert fp["latest_sweep_waveforms"] is not None
    assert fp["latest_sweep_waveforms"].shape[0] == 2

    primary, neighbors = auto_detect_channels_from_footprint(fp)
    assert primary == "CH0"

    view = UnitFootprintPGView(state=state)
    qtbot.addWidget(view)
    view.chk_show_live_trace.setChecked(True)
    view.render_views()
    assert view.live_trace_curve is not None
    x_data, y_data = view.live_trace_curve.getData()
    assert x_data is not None and len(x_data) > 0

    # Also verify pure DSP latency & ISI helpers
    counts, edges = compute_latency_histogram([25.0, 25.2, 24.9], bins=10)
    assert len(counts) == 10
    assert len(edges) == 11
    isi_info = compute_isi_and_firing_rate([0.1, 0.3, 0.5], [0, 1, 2], total_stims=3)
    assert "Instantaneous ISI" in isi_info
    assert "Firing Rate (Hz)" in isi_info

    erp_2d = state.get_erp()
    stims, samples = select_spikes_in_polygon(
        erp_2d,
        fs_hz=10000.0,
        polygon_xy=[(20.0, -0.5), (30.0, -0.5), (30.0, 1.5), (20.0, 1.5)],
    )
    assert list(stims) == [0, 1]
    assert list(samples) == [250, 250]


def test_scoped_unit_signals_and_pinned_view_filtering(qtbot):
    """Verify scoped ViewerState signals, TrackingView parameter signals, and pinned view event filtering."""
    state = _make_synthetic_state(n_sweeps=4)
    # Add a second unit so we have Unit 0 and Unit 1
    state.addUnitGroup()

    active_unit_events = []
    unit_list_events = []
    unit_params_events = []
    unit_spike_events = []

    state.onActiveUnitChanged.connect(lambda idx: active_unit_events.append(idx))
    state.onUnitListChanged.connect(lambda: unit_list_events.append(True))
    state.onUnitParamsChanged.connect(lambda idx: unit_params_events.append(idx))
    state.onUnitSpikeChanged.connect(lambda u_idx, stim: unit_spike_events.append((u_idx, stim)))

    state.setUnitGroup(1)
    assert active_unit_events == [1]

    state.setStimNo(2)
    state.setUnit(250, unit_idx=1)
    assert (1, 2) in unit_spike_events

    # TrackingView spinbox change should emit onUnitParamsChanged without triggering ChannelGroupBar.refresh_units
    tv = TrackingView(state=state)
    qtbot.addWidget(tv)
    fp_view = UnitFootprintPGView(state=state)
    qtbot.addWidget(fp_view)
    fp_view.channel_bar.bind_to_unit(1)

    refresh_units_calls = 0
    orig_refresh_units = fp_view.channel_bar.refresh_units

    def tracked_refresh_units(*args, **kwargs):
        nonlocal refresh_units_calls
        refresh_units_calls += 1
        return orig_refresh_units(*args, **kwargs)

    fp_view.channel_bar.refresh_units = tracked_refresh_units

    # Change tracking window on TrackingView
    tv.qsb_window_size.setValue(tv.qsb_window_size.value() + 1.0)
    assert len(unit_params_events) >= 1
    assert refresh_units_calls == 0

    # Pinned UnitFootprintPGView (pinned to Unit 1) should ignore spike events for Unit 0
    fp_view._dirty = False
    state.onNewSpikeEvents.emit(0)
    assert fp_view._dirty is False
    state.onNewSpikeEvents.emit(1)
    assert fp_view._dirty is True
    fp_view._dirty = False


def test_signal_disconnection_on_view_close(qtbot):
    """Verify closing views cleanly disconnects their slots from ViewerState."""
    from spikespy.SpikeGroupTable import SpikeGroupTableView

    state = _make_synthetic_state(n_sweeps=3)
    fp_view = UnitFootprintPGView(state=state)
    st_view = SingleTracePGView(state=state)
    tv = TrackingView(state=state)
    tbl = SpikeGroupTableView(state=state)
    for w in (fp_view, st_view, tv, tbl):
        qtbot.addWidget(w)
        w.close()

    assert len(fp_view.connections) == 0
    assert len(st_view.connections) == 0
    assert len(tv._state_connections) == 0
    assert len(tbl._state_connections) == 0

    # Emitting state signals after views are closed should not trigger timers or updates on closed views
    fp_view._dirty = False
    state.onNewSpikeEvents.emit(0)
    state.onUnitGroupChange.emit()
    assert fp_view._dirty is False


def test_legacy_mpl_package_and_left_docked_property_pane(qtbot):
    """Verify legacy Matplotlib views live in spikespy.legacy_mpl and MdiView docks ViewSettings on the left."""
    import PySide6QtAds as QtAds
    from spikespy.SpikeSpy import MdiView
    from spikespy.legacy_mpl import (
        MultiTraceView,
        SingleTraceView,
        MultiTraceFixedView,
        UnitView,
        UnitFootprintView,
        EventView,
    )
    import spikespy.MultiTraceView as legacy_alias

    assert legacy_alias.MultiTraceView is MultiTraceView
    assert MultiTraceView.__module__.startswith("spikespy.legacy_mpl")
    assert SingleTraceView.__module__.startswith("spikespy.legacy_mpl")
    assert MultiTraceFixedView.__module__.startswith("spikespy.legacy_mpl")
    assert UnitView.__module__.startswith("spikespy.legacy_mpl")
    assert UnitFootprintView.__module__.startswith("spikespy.legacy_mpl")
    assert EventView.__module__.startswith("spikespy.legacy_mpl")

    state = _make_synthetic_state(n_sweeps=3)
    mdi = MdiView(state=state, check_updates=False)
    qtbot.addWidget(mdi)

    # ViewSettings dock widget is present and persists across layout switches
    assert mdi.dock_settings in mdi.dock_manager.dockWidgets()
    w_st = mdi.newWindow("SingleTrace (PyQtGraph)")
    assert mdi.settings_panel.active_view is w_st

    # Clicking or opening a non-configurable panel (SpikeGroupTable) retains SingleTracePGView in property pane
    w_tbl = mdi.newWindow("SpikeGroupTable")
    assert mdi.settings_panel.active_view is w_st
    mdi.settings_panel.setActiveView(mdi.btn_add_panel)
    assert mdi.settings_panel.active_view is w_st

