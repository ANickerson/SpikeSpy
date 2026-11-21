"""Tests for channel purpose configuration, preprocessing, APTrack reloading, and live sync."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import neo
import numpy as np
import pytest
import quantities as pq
from PySide6.QtWidgets import QApplication

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from spikespy.APTrack_experiment_import import (
    ChannelRoleSpec,
    TypeID,
    get_default_channel_roles,
    normalize_channel_roles,
    preprocess_segment_with_roles,
)
from spikespy.ChannelPurposeView import (
    ChannelPurposeDialog,
    ChannelPurposeWidget,
    PURPOSE_PRESETS,
)
from spikespy.ViewerState import ViewerState, tracked_neuron_unit
from spikespy.io_manager import APTrackDialog
from spikespy.open_ephys_addin.oe_live_view import OpenEphysLiveView


def _write_continuous_file(
    path: Path,
    samples_i16: np.ndarray,
    sample_rate: float = 10000.0,
    bit_volts: float = 0.001,
    channel_name: str = "CH1",
    recording: int = 0,
) -> None:
    """Write a minimal valid legacy Open Ephys .continuous file."""
    header_str = (
        "header.format = 'Open Ephys Data Format'; "
        "header.version = 0.4; "
        f"header.sampleRate = {int(sample_rate)}; "
        f"header.bitVolts = {bit_volts}; "
        f"header.channel = '{channel_name}'; "
        "header.date_created = '22-Nov-2022 12:00:00';"
    )
    header_bytes = header_str.encode("utf-8")[:1024].ljust(1024, b" ")

    n_blocks = max(1, int(np.ceil(len(samples_i16) / 1024.0)))
    padded = np.zeros(n_blocks * 1024, dtype=np.int16)
    padded[: len(samples_i16)] = samples_i16

    dtype = np.dtype(
        [
            ("timestamp", "<i8"),
            ("N", np.uint16),
            ("recording", np.uint16),
            ("data", ">i2", 1024),
            ("rec", np.uint8, 10),
        ]
    )
    records = np.zeros(n_blocks, dtype=dtype)
    for b in range(n_blocks):
        records[b]["timestamp"] = b * 1024
        records[b]["N"] = 1024
        records[b]["recording"] = recording
        records[b]["data"] = padded[b * 1024 : (b + 1) * 1024]

    with open(path, "wb") as fh:
        fh.write(header_bytes)
        records.tofile(fh)


def _make_synthetic_segment(sr_hz: float = 10000.0, duration_s: float = 1.0) -> neo.Segment:
    """Create a synthetic neo.Segment with CH1, ADC4, ADC5, ADC8 and a unit event."""
    sr = sr_hz * pq.Hz
    n = int(sr_hz * duration_s)
    t = np.arange(n, dtype=np.float64) / sr_hz

    # CH1: slow 5 Hz drift + 1000 Hz spike carrier
    ch1_data = 200.0 * np.sin(2.0 * np.pi * 5.0 * t) + 40.0 * np.sin(2.0 * np.pi * 1000.0 * t)

    # ADC4: two 5V pulses at 0.1s and 0.5s (plus a closely spaced pulse at 0.52s for refractory testing)
    adc4_data = np.zeros(n, dtype=np.float64)
    adc4_data[int(0.10 * sr_hz) : int(0.105 * sr_hz)] = 5.0
    adc4_data[int(0.50 * sr_hz) : int(0.505 * sr_hz)] = 5.0
    adc4_data[int(0.52 * sr_hz) : int(0.525 * sr_hz)] = 5.0

    # ADC5: three 5V pulses at 0.2s, 0.4s, 0.7s
    adc5_data = np.zeros(n, dtype=np.float64)
    for ts in (0.20, 0.40, 0.70):
        adc5_data[int(ts * sr_hz) : int((ts + 0.005) * sr_hz)] = 5.0

    # ADC8: button press at 0.8s
    adc8_data = np.zeros(n, dtype=np.float64)
    adc8_data[int(0.80 * sr_hz) : int(0.82 * sr_hz)] = 5.0

    seg = neo.Segment(name="Synthetic")
    seg.analogsignals.append(neo.AnalogSignal(ch1_data, units=pq.uV, sampling_rate=sr, name="CH1"))
    seg.analogsignals.append(neo.AnalogSignal(adc4_data, units=pq.V, sampling_rate=sr, name="ADC4"))
    seg.analogsignals.append(neo.AnalogSignal(adc5_data, units=pq.V, sampling_rate=sr, name="ADC5"))
    seg.analogsignals.append(neo.AnalogSignal(adc8_data, units=pq.V, sampling_rate=sr, name="ADC8"))

    u0 = neo.Event(np.array([0.11, 0.51]) * pq.s, name="unit_0", units=pq.s)
    u0.annotations["type"] = "unit"
    u0.annotations["channel"] = "rd.0.bp"
    u0.annotations["notes"] = "keep-me"
    seg.events.append(u0)
    return seg


def test_channel_role_spec_defaults_and_normalization():
    defaults = get_default_channel_roles()
    assert len(defaults) >= 6
    assert any(r.name == "rd.0" and r.purpose == "probe" and r.bandpass for r in defaults)
    assert any(r.name == "env.stim" and r.probe_type == TypeID.TTL for r in defaults)

    # Legacy dict normalization
    legacy_roles = normalize_channel_roles(
        {
            "rd.0": "CH2",
            "stimVolt": "ADC3",
            "button": "ADC7",
            "aux.temp": "ADC6",
        }
    )
    by_name = {(r.name, r.probe_type): r for r in legacy_roles}
    assert by_name[("rd.0", TypeID.ANALOG)].source == "CH2"
    assert by_name[("env.stimVolt", TypeID.TTL)].source == "ADC3"
    assert by_name[("rec.button", TypeID.TTL)].source == "ADC7"
    assert by_name[("aux.temp", TypeID.ANALOG)].source == "ADC6"

    # Dict list with 'roles' wrapper & ChannelRoleSpec roundtrip
    spec_dicts = [r.to_dict() for r in defaults[:2]]
    roundtrip = normalize_channel_roles({"roles": spec_dicts})
    assert len(roundtrip) == 2
    assert isinstance(roundtrip[0], ChannelRoleSpec)
    assert roundtrip[0].source == defaults[0].source


def test_preprocess_segment_with_roles_bandpass_ttl_and_reprocess():
    seg = _make_synthetic_segment(sr_hz=10000.0, duration_s=1.0)

    roles_v1 = [
        {
            "source": "CH1",
            "name": "rd.0",
            "probe_type": "AS",
            "purpose": "probe",
            "bandpass": True,
            "bandpass_low": 500.0,
            "bandpass_high": 3500.0,
            "is_adc": False,
        },
        {
            "source": "ADC4",
            "name": "env.stim",
            "probe_type": "TTL",
            "purpose": "stim_trigger",
            "ttl_threshold": 2.5,
            "ttl_refractory_ms": 100.0,
            "ttl_min_width_ms": 0.4,
        },
        {
            "source": "ADC8",
            "name": "rec.button",
            "probe_type": "TTL",
            "purpose": "button",
            "ttl_threshold": 2.5,
            "ttl_refractory_ms": 100.0,
        },
    ]

    seg1 = preprocess_segment_with_roles(seg, roles_v1, in_place=False)
    sig_names1 = [s.name for s in seg1.analogsignals]
    assert "rd.0" in sig_names1
    assert "rd.0.bp" in sig_names1

    raw_sig = next(s for s in seg1.analogsignals if s.name == "rd.0")
    bp_sig = next(s for s in seg1.analogsignals if s.name == "rd.0.bp")
    # 5 Hz drift removed in bandpass signal
    assert np.std(bp_sig.magnitude) < np.std(raw_sig.magnitude)
    assert np.max(np.abs(bp_sig.magnitude[1000:-1000])) < 60.0

    # Refractory 100 ms merges 0.50s and 0.52s pulses -> 2 triggers (0.10s and 0.50s)
    stim_evt1 = next(e for e in seg1.events if e.name == "env.stim")
    assert len(stim_evt1) == 2
    assert np.allclose(stim_evt1.rescale(pq.s).magnitude, [0.10, 0.50], atol=1e-3)

    btn_evt = next(e for e in seg1.events if e.name == "rec.button")
    assert len(btn_evt) == 1
    assert btn_evt.rescale(pq.s).magnitude[0] == pytest.approx(0.80, abs=1e-3)

    # Re-run on seg1: switch trigger to ADC5 (3 pulses) and turn bandpass OFF
    roles_v2 = [
        {
            "source": "CH1",
            "name": "rd.0",
            "probe_type": "AS",
            "purpose": "probe",
            "bandpass": False,
            "is_adc": False,
        },
        {
            "source": "ADC5",
            "name": "env.stim",
            "probe_type": "TTL",
            "purpose": "stim_trigger",
            "ttl_threshold": 2.5,
            "ttl_refractory_ms": 50.0,
        },
        {
            "source": "ADC8",
            "name": "rec.button",
            "probe_type": "TTL",
            "purpose": "button",
            "ttl_threshold": 2.5,
        },
    ]

    seg2 = preprocess_segment_with_roles(seg1, roles_v2, in_place=True)
    sig_names2 = [s.name for s in seg2.analogsignals]
    assert "rd.0" in sig_names2
    assert "rd.0.bp" not in sig_names2
    assert sig_names2.count("rd.0") == 1

    stim_evts2 = [e for e in seg2.events if e.name == "env.stim"]
    assert len(stim_evts2) == 1
    assert np.allclose(stim_evts2[0].rescale(pq.s).magnitude, [0.20, 0.40, 0.70], atol=1e-3)

    # Unit annotations preserved across re-runs
    unit_evts = [e for e in seg2.events if e.name == "unit_0"]
    assert len(unit_evts) == 1
    assert unit_evts[0].annotations.get("notes") == "keep-me"


def test_viewer_state_roles_and_aptrack_folder_reload(tmp_path):
    sr = 10000.0
    n = 10240  # ~1.024s
    t = np.arange(n, dtype=np.float64) / sr

    ch1 = (1000.0 * np.sin(2.0 * np.pi * 5.0 * t) + 200.0 * np.sin(2.0 * np.pi * 1000.0 * t)).astype(np.int16)
    adc4 = np.zeros(n, dtype=np.int16)
    adc4[1000:1040] = 5000
    adc4[5000:5040] = 5000
    adc5 = np.zeros(n, dtype=np.int16)
    for idx in (2000, 4000, 7000):
        adc5[idx : idx + 40] = 5000
    adc8 = np.zeros(n, dtype=np.int16)
    adc8[8000:8100] = 5000

    _write_continuous_file(tmp_path / "100_CH1.continuous", ch1, sample_rate=sr, bit_volts=0.195, channel_name="CH1")
    _write_continuous_file(tmp_path / "100_ADC4.continuous", adc4, sample_rate=sr, bit_volts=0.001, channel_name="ADC4")
    _write_continuous_file(tmp_path / "100_ADC5.continuous", adc5, sample_rate=sr, bit_volts=0.001, channel_name="ADC5")
    _write_continuous_file(tmp_path / "100_ADC8.continuous", adc8, sample_rate=sr, bit_volts=0.001, channel_name="ADC8")

    state = ViewerState()
    initial_cfg = {
        "record": 1,
        "roles": [
            {
                "source": "CH1",
                "name": "rd.0",
                "probe_type": "AS",
                "purpose": "probe",
                "bandpass": True,
                "bandpass_low": 500.0,
                "bandpass_high": 3500.0,
                "is_adc": False,
            },
            {
                "source": "ADC4",
                "name": "env.stim",
                "probe_type": "TTL",
                "purpose": "stim_trigger",
                "ttl_threshold": 2.5,
                "ttl_refractory_ms": 100.0,
            },
        ],
    }

    state.loadFile(str(tmp_path), type="APTrack", blocking=True, aptrack_config=initial_cfg)
    assert state.analog_signal.name == "rd.0.bp"
    assert state.event_signal.name == "env.stim"
    assert len(state.event_signal) == 2

    # Add a tracked unit spike before reloading to verify unit preservation
    state.setUnit(150, stimno=0, unit_idx=0)
    assert len(state.spike_groups[0].event) == 1

    # Reload with updated roles switching stim_trigger to ADC5
    updated_roles = [
        {
            "source": "CH1",
            "name": "rd.0",
            "probe_type": "AS",
            "purpose": "probe",
            "bandpass": True,
            "bandpass_low": 500.0,
            "bandpass_high": 3500.0,
            "is_adc": False,
        },
        {
            "source": "ADC5",
            "name": "env.stim",
            "probe_type": "TTL",
            "purpose": "stim_trigger",
            "ttl_threshold": 2.5,
            "ttl_refractory_ms": 50.0,
        },
        {
            "source": "ADC8",
            "name": "rec.button",
            "probe_type": "TTL",
            "purpose": "button",
            "ttl_threshold": 2.5,
        },
    ]
    assert state.reload_with_channel_roles(updated_roles, record_no=1) is True
    assert state.event_signal.name == "env.stim"
    assert len(state.event_signal) == 3
    assert any(e.name == "rec.button" and len(e) == 1 for e in state.segment.events)
    assert len(state.spike_groups[0].event) == 1

    # In-memory preprocessing via set_channel_roles(apply_preprocess=True)
    seg_mem = _make_synthetic_segment(sr_hz=10000.0, duration_s=1.0)
    state_mem = ViewerState(segment=seg_mem)
    state_mem.set_data(seg_mem.analogsignals[0], seg_mem.events[0])
    state_mem.set_channel_roles(updated_roles, apply_preprocess=True)
    assert state_mem.analog_signal.name == "rd.0.bp"
    assert state_mem.event_signal.name == "env.stim"
    assert len(state_mem.event_signal) == 3


def test_aptrack_dialog_roles_and_config(qtbot, tmp_path):
    _write_continuous_file(tmp_path / "100_CH1.continuous", np.zeros(1024, dtype=np.int16), channel_name="CH1")
    _write_continuous_file(tmp_path / "100_ADC4.continuous", np.zeros(1024, dtype=np.int16), channel_name="ADC4")

    dlg = APTrackDialog(folder=tmp_path)
    qtbot.addWidget(dlg)

    roles = dlg.get_roles()
    assert len(roles) >= 1
    cfg = dlg.get_config()
    assert "roles" in cfg
    assert cfg["rd.0"] == "CH1"

    dlg._on_add_role_clicked()
    assert dlg.roleTable.rowCount() == len(roles) + 1
    dlg._on_remove_role_clicked()
    assert dlg.roleTable.rowCount() == len(roles)
    dlg._on_reset_defaults_clicked()
    assert dlg.roleTable.rowCount() == len(get_default_channel_roles())


def test_channel_purpose_widget_dialog_presets_and_live_sync(qtbot, tmp_path):
    seg = _make_synthetic_segment(sr_hz=10000.0, duration_s=1.0)
    state = ViewerState(segment=seg)
    state.set_data(seg.analogsignals[0])

    # Attach an OpenEphysLiveView with a stubbed active worker to test live sync
    live_view = OpenEphysLiveView(state=state)
    qtbot.addWidget(live_view)
    live_view.show()
    recorded_calls = []

    class _DummyWorker:
        def __init__(self):
            self.use_analog_trigger = False
            self.adc_trigger_chan = 1
            self.auto_align_clock = True

        def set_channel_roles(self, roles):
            recorded_calls.append(("roles", len(roles)))

        def set_rec_channel(self, ch):
            recorded_calls.append(("rec", ch))

        def set_active_channels(self, chans):
            recorded_calls.append(("active", list(chans)))

        def set_filtering(self, enabled):
            recorded_calls.append(("filter", enabled))

        def set_filter_band(self, lo, hi):
            recorded_calls.append(("band", lo, hi))

        def set_ttl_params(self, threshold=None, refractory_ms=None):
            recorded_calls.append(("ttl", threshold, refractory_ms))

        def set_window_params(self, window_ms=None, pre_ms=None):
            recorded_calls.append(("window", window_ms, pre_ms))

        def stop(self):
            pass

    live_view.worker = _DummyWorker()

    # Collapsible ChannelPurposeWidget inside OpenEphysLiveView Tab 1
    assert live_view.tabs_live.count() == 4
    assert live_view.channel_purpose_group.isVisible() is False
    live_view.btn_configure_channel_purposes.click()
    QApplication.processEvents()
    assert live_view.channel_purpose_group.isVisible() is True

    widget = ChannelPurposeWidget(state=state)
    qtbot.addWidget(widget)

    # Set clean 2-row roles and test UI interaction: change row 1 purpose -> thermode
    widget.set_roles(
        [
            {
                "enabled": True,
                "purpose": "probe",
                "source_channel": "CH1",
                "target_name": "rd.0",
                "mode": "analog_bp",
                "bandpass_low": 500.0,
                "bandpass_high": 3500.0,
                "is_adc": False,
            },
            {
                "enabled": True,
                "purpose": "stim_trigger",
                "source_channel": "ADC5",
                "target_name": "env.stim",
                "mode": "ttl",
                "ttl_threshold_v": 2.5,
                "ttl_refractory_ms": 50.0,
                "is_adc": True,
            },
            {
                "enabled": True,
                "purpose": "aux",
                "source_channel": "ADC8",
                "target_name": "env.aux",
                "mode": "analog",
                "is_adc": True,
            },
        ]
    )

    combo_purpose = widget.table.cellWidget(2, widget.COL_PURPOSE)
    combo_purpose.setCurrentIndex(combo_purpose.findData("thermode"))
    QApplication.processEvents()
    roles_after_purpose_change = widget.get_roles()
    assert roles_after_purpose_change[2]["target_name"] == "env.thermode"
    assert roles_after_purpose_change[2]["mode"] == "analog"

    # Change row 2 back to button TTL and click Apply & Preprocess (Reload)
    combo_purpose.setCurrentIndex(combo_purpose.findData("button"))
    widget.table.selectRow(2)
    widget.spin_ttl_thresh.setValue(2.5)
    QApplication.processEvents()

    applied_emitted = []
    widget.rolesApplied.connect(lambda r: applied_emitted.append(r))
    widget.btn_apply_preprocess.click()
    QApplication.processEvents()

    assert len(applied_emitted) == 1
    assert state.analog_signal.name == "rd.0.bp"
    assert state.event_signal.name == "env.stim"
    assert len(state.event_signal) == 3
    assert any(e.name == "rec.button" and len(e) == 1 for e in state.segment.events)

    # Verify live stream worker received updates without dropping connection
    assert any(c[0] == "roles" for c in recorded_calls)
    assert ("band", 500.0, 3500.0) in recorded_calls
    assert ("filter", True) in recorded_calls
    assert live_view.worker is not None

    # Save/load user preset & JSON export/import
    widget.save_user_preset("UnitTestPreset")
    assert widget.apply_preset("Single Probe + TTL")
    assert len(widget.get_roles()) == 2
    assert widget.apply_preset("UnitTestPreset")
    assert len(widget.get_roles()) == 3

    json_file = tmp_path / "roles.json"
    assert widget.export_roles_to_json(json_file)
    widget2 = ChannelPurposeWidget()
    qtbot.addWidget(widget2)
    assert widget2.load_roles_from_json(json_file)
    assert len(widget2.get_roles()) == 3

    # Dialog wrapper
    dlg = ChannelPurposeDialog(state=state)
    qtbot.addWidget(dlg)
    assert dlg.get_config()["record"] == 1
    assert len(dlg.get_roles()) == 3
    assert PURPOSE_PRESETS["probe"]["default_target_name"] == "rd.0"


def test_thermode_role_displayed_in_multitrace_pgview_offline_and_live(qtbot):
    """Verify mapping a channel to Thermode Temperature (env.thermode) enables viewing it
    as a right-axis summary trace and as a channel plot on MultiTracePGView in offline and live modes.
    """
    from spikespy.MultiTracePGView import MultiTracePGView
    from spikespy.MultiTracePlotManager import MultiTracePlotManager
    from spikespy.legacy_mpl.MultiTraceView import MultiTraceView

    sr_hz = 10000.0
    sr = sr_hz * pq.Hz
    n_samples = 20000  # 2.0 s so 0.5 s ERP windows at 0.10, 0.35, 0.60 s fit cleanly
    t = np.arange(n_samples, dtype=np.float64) / sr_hz

    # Probe channel (CH1) and thermode temperature ramp channel (ADC7) + stim TTL (ADC4)
    ch1_data = 50.0 * np.sin(2.0 * np.pi * 1000.0 * t)
    adc7_data = np.linspace(32.0, 46.0, n_samples, dtype=np.float64)
    adc4_data = np.zeros(n_samples, dtype=np.float64)
    for ts in (0.10, 0.35, 0.60):
        adc4_data[int(ts * sr_hz) : int((ts + 0.005) * sr_hz)] = 5.0

    seg = neo.Segment(name="ThermodeOffline")
    seg.analogsignals.append(neo.AnalogSignal(ch1_data, units=pq.uV, sampling_rate=sr, name="CH1"))
    seg.analogsignals.append(neo.AnalogSignal(adc7_data, units=pq.V, sampling_rate=sr, name="ADC7"))
    seg.analogsignals.append(neo.AnalogSignal(adc4_data, units=pq.V, sampling_rate=sr, name="ADC4"))

    state = ViewerState(segment=seg)
    state.set_data(seg.analogsignals[0], neo.Event(np.array([0.10, 0.35, 0.60]) * pq.s, name="stim"))

    pg_view = MultiTracePGView(state=state)
    qtbot.addWidget(pg_view)
    pg_view.show()
    QApplication.processEvents()

    # Map CH1 -> rd.0, ADC4 -> env.stim, ADC7 -> env.thermode (Thermode Temperature)
    roles = [
        {
            "source": "CH1",
            "name": "rd.0",
            "probe_type": "AS",
            "purpose": "probe",
            "bandpass": True,
            "bandpass_low": 500.0,
            "bandpass_high": 3500.0,
            "is_adc": False,
        },
        {
            "source": "ADC4",
            "name": "env.stim",
            "probe_type": "TTL",
            "purpose": "stim_trigger",
            "ttl_threshold": 2.5,
            "ttl_refractory_ms": 50.0,
        },
        {
            "source": "ADC7",
            "name": "env.thermode",
            "probe_type": "AS",
            "purpose": "thermode",
            "bandpass": False,
            "is_adc": True,
        },
    ]
    state.set_channel_roles(roles, apply_preprocess=True)
    QApplication.processEvents()

    assert "env.thermode" in state.get_available_channels()
    assert "env.thermode" in pg_view._get_channel_names()

    # 1. Right-axis summary trace auto-available and rendered for env.thermode
    avail_summary = pg_view.get_summary_plot_availability()
    assert "env.thermode" in avail_summary
    assert avail_summary["env.thermode"] is True
    assert "env.thermode" in pg_view.summary_panels
    x_sum, y_sum = pg_view.summary_panels["env.thermode"].curve.getData()
    assert len(x_sum) == 3
    assert x_sum[-1] > x_sum[0]  # rising thermode temperature ramp across sweeps

    # Verify Summary menu and Plot Manager expose env.thermode with friendly role label
    pg_view._populate_summary_menu()
    menu_texts = [a.text() for a in pg_view.btn_summary_menu.menu().actions()]
    assert any("env.thermode" in txt and "Thermode Temperature" in txt for txt in menu_texts)

    mgr = MultiTracePlotManager(view=pg_view)
    qtbot.addWidget(mgr)
    mgr.show()
    QApplication.processEvents()
    mgr._populate_add_menu()
    add_texts = [a.text() for a in mgr._menu_add.actions()]
    assert any("env.thermode" in txt and "Thermode Temperature" in txt for txt in add_texts)

    # Can toggle env.thermode summary off and back on via set_plot_visible / add_summary_channel
    pg_view.set_plot_visible("env.thermode", False)
    assert "env.thermode" not in pg_view.summary_panels
    pg_view.add_summary_channel("env.thermode")
    assert "env.thermode" in pg_view.summary_panels

    # 2. Can also add env.thermode as a full channel plot on MultiTracePGView
    pg_view.add_channel_to_display("env.thermode")
    QApplication.processEvents()
    assert "env.thermode" in pg_view.displayed_channels
    thermode_panel = next(p for p in pg_view.channel_panels if p.channel_name == "env.thermode")
    assert thermode_panel.has_data is True

    # Legacy Matplotlib MultiTraceView also includes env.thermode in _get_channel_names
    mpl_view = MultiTraceView(state=state)
    qtbot.addWidget(mpl_view)
    assert "env.thermode" in mpl_view._get_channel_names()

    # 3. Live streaming mode: map CH1 -> env.thermode (Thermode Temperature) and stream sweeps
    live_state = ViewerState()
    live_state.start_live_session(
        sampling_rate=10000.0,
        window_size_ms=100.0,
        units=pq.uV,
        channels=["CH0", "CH1"],
    )
    try:
        live_pg = MultiTracePGView(state=live_state)
        qtbot.addWidget(live_pg)
        live_pg.show()
        QApplication.processEvents()

        live_state.set_channel_roles(
            [
                {
                    "source": "CH0",
                    "name": "rd.0",
                    "probe_type": "AS",
                    "purpose": "probe",
                    "bandpass": False,
                    "is_adc": False,
                },
                {
                    "source": "CH1",
                    "name": "env.thermode",
                    "probe_type": "AS",
                    "purpose": "thermode",
                    "bandpass": False,
                    "is_adc": True,
                },
            ],
            apply_preprocess=True,
        )
        QApplication.processEvents()

        # Stream 3 sweeps with rising thermode temperatures (32 C, 38 C, 44 C)
        for sw_i, temp in enumerate((32.0, 38.0, 44.0)):
            w_probe = np.random.RandomState(sw_i).randn(1000).astype(np.float32) * 5.0
            w_temp = np.full(1000, temp, dtype=np.float32)
            live_state.append_sweep(
                stim_timestamp=1.0 + sw_i * 0.5,
                sweep_waveform=w_probe,
                aux_sweeps={"env.thermode": w_temp},
            )
        live_pg.reset_right_axes_data()
        QApplication.processEvents()

        assert "env.thermode" in live_state.get_available_channels()
        assert "env.thermode" in live_pg.summary_panels
        lx, _ = live_pg.summary_panels["env.thermode"].curve.getData()
        assert len(lx) == 3
        assert np.allclose(lx, [32.0, 38.0, 44.0], atol=1e-3)

        # Also viewable as a channel plot in live mode
        live_pg.add_channel_to_display("env.thermode")
        QApplication.processEvents()
        live_thermode_panel = next(
            p for p in live_pg.channel_panels if p.channel_name == "env.thermode"
        )
        assert live_thermode_panel.has_data is True
    finally:
        live_state.stop_live_session()

