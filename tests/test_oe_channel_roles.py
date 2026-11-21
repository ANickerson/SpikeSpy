"""Tests for OpenEphysStreamWorker channel purpose / role mapping, selective filtering,
secondary TTL detection, and semantic aux_sweeps aliasing.
"""

from __future__ import annotations

import json
import numpy as np
import pytest

from spikespy.ViewerState import ViewerState
from spikespy.open_ephys_addin.oe_live_view import OpenEphysLiveView
from spikespy.open_ephys_addin.oe_stream_worker import (
    OpenEphysStreamWorker,
    resolve_source_to_index,
)


def test_resolve_source_to_index():
    """Verify source string parsing for digits, CH0/CH1 conventions, and ADC channels."""
    assert resolve_source_to_index(0) == 0
    assert resolve_source_to_index("2") == 2
    assert resolve_source_to_index("CH0") == 0
    assert resolve_source_to_index("CH1", has_ch0=False) == 0
    assert resolve_source_to_index("CH2", has_ch0=False) == 1
    assert resolve_source_to_index("CH1", has_ch0=True) == 1
    # Standard Open Ephys 16 headstage + 8 ADC channels
    assert resolve_source_to_index("ADC1", n_channels=24) == 16
    assert resolve_source_to_index("ADC4", n_channels=24) == 19
    assert resolve_source_to_index("ADC8", n_channels=24) == 23
    # Compact stream fallback (<= 16 channels)
    assert resolve_source_to_index("ADC2", n_channels=8) == 2
    assert resolve_source_to_index("ADC7", n_channels=8) == 7


def test_worker_channel_roles_selective_filtering_and_aliases():
    """Verify per-channel selective bandpass filtering and semantic aliases in aux_sweeps."""
    sr = 30000.0
    n_samples = 6000  # 200 ms
    n_channels = 8

    roles = [
        {
            "source": "CH1",
            "name": "rd.0",
            "probe_type": "AS",
            "purpose": "probe",
            "bandpass": True,
            "bandpass_low": 500.0,
            "bandpass_high": 7000.0,
        },
        {
            "source": "ADC2",
            "channel_index": 2,
            "name": "env.potentiometer",
            "probe_type": "AS",
            "purpose": "potentiometer",
            "bandpass": False,
        },
        {
            "source": "ADC7",
            "channel_index": 7,
            "name": "env.thermode",
            "probe_type": "AS",
            "purpose": "thermode",
            "bandpass": False,
        },
        {
            "source": "ADC5",
            "channel_index": 5,
            "name": "env.stim",
            "probe_type": "TTL",
            "purpose": "stim_trigger",
            "ttl_threshold": 2.5,
            "ttl_refractory_ms": 100.0,
            "trigger_source": "analog",
        },
        {
            "source": "ADC4",
            "channel_index": 4,
            "name": "env.stimVolt",
            "probe_type": "TTL",
            "purpose": "stim_voltage",
            "ttl_threshold": 2.5,
            "ttl_refractory_ms": 100.0,
        },
        {
            "source": "ADC6",
            "channel_index": 6,
            "name": "rec.button",
            "probe_type": "TTL",
            "purpose": "button",
            "ttl_threshold": 2.5,
            "ttl_refractory_ms": 100.0,
        },
    ]

    worker = OpenEphysStreamWorker(
        mode="zmq_raw",
        rec_channel=0,
        active_channels=[0],
        window_ms=80.0,
        pre_stim_ms=10.0,
        apply_filter=True,
        sample_rate=sr,
        channel_roles=roles,
    )

    assert worker.filtered_channels == {0}
    assert worker.analog_trigger_channel == 5
    assert worker.trigger_source == "analog"
    assert "rec.button" in worker._aux_ttl_detectors
    assert "env.stimVolt" in worker._aux_ttl_detectors

    protocol_events = []
    worker.protocolEventReady.connect(
        lambda kind, ts, details: protocol_events.append((kind, ts, details))
    )
    sweeps = []
    worker.sweepsReady.connect(sweeps.extend)

    t = np.arange(n_samples, dtype=np.float32) / sr
    chunk = np.zeros((n_channels, n_samples), dtype=np.float32)
    # CH0 (rd.0): slow 10 Hz drift (to be removed by bandpass) + 1000 Hz spike tone
    chunk[0, :] = 200.0 * np.sin(2 * np.pi * 10.0 * t) + 30.0 * np.sin(2 * np.pi * 1000.0 * t)
    # CH2 (env.potentiometer): DC 3.3 V (must remain unfiltered!)
    chunk[2, :] = 3.3
    # CH7 (env.thermode): slow temperature ramp 32.0 -> 45.0 degC (must remain unfiltered!)
    chunk[7, :] = np.linspace(32.0, 45.0, n_samples, dtype=np.float32)
    # CH5 (env.stim): stimulus trigger pulse at sample 1000
    chunk[5, 1000:1060] = 5.0
    # CH4 (env.stimVolt): stim voltage pulse at sample 1000
    chunk[4, 1000:1060] = 4.2
    # CH6 (rec.button): manual button pulse at sample 2000
    chunk[6, 2000:2100] = 5.0

    hdr = json.dumps({
        "stream": "continuous",
        "channel_count": n_channels,
        "sample_rate": sr,
        "sample_number": 0,
        "num_samples": n_samples,
        "dtype": "float32",
    }).encode("utf-8")

    worker._parse_raw_zmq_data_packet([b"data", hdr, chunk.tobytes()])
    worker._check_pending_sweeps()
    worker._flush_sweep_payloads(force=True)

    # Secondary TTL events emitted for rec.button and env.stimVolt
    ttl_roles = [e for e in protocol_events if e[0] == "ttl_role"]
    role_names = {e[2]["name"]: e for e in ttl_roles}
    assert "rec.button" in role_names
    assert role_names["rec.button"][2]["purpose"] == "button"
    assert role_names["rec.button"][2]["sample_number"] == 2000
    assert role_names["rec.button"][1] == pytest.approx(2000 / sr)
    assert "env.stimVolt" in role_names
    assert role_names["env.stimVolt"][2]["sample_number"] == 1000

    # Sweep sliced and populated with semantic aliases
    assert len(sweeps) == 1
    sw = sweeps[0]
    aux = sw["aux_sweeps"]
    assert "CH0" in aux
    assert "rd.0" in aux
    assert "rd.0.bp" in aux
    assert "env.potentiometer" in aux
    assert "env.thermode" in aux

    # Bandpass was applied to rd.0 / CH0 (removes 200 uV 10 Hz drift)
    assert np.max(np.abs(aux["rd.0"][300:])) < 80.0
    assert np.allclose(aux["rd.0"], aux["rd.0.bp"])

    # Auxiliary analog channels were NOT bandpass-filtered
    assert np.allclose(aux["env.potentiometer"], 3.3, atol=1e-4)
    expected_ramp = np.linspace(32.0, 45.0, n_samples, dtype=np.float32)[700:3400]
    assert np.allclose(aux["env.thermode"], expected_ramp, atol=1e-4)


def test_worker_reload_channel_roles_while_streaming_and_sliced_erp():
    """Verify set_channel_roles can be reloaded dynamically and works in sliced_erp mode."""
    worker = OpenEphysStreamWorker(
        mode="sliced_erp",
        rec_channel=0,
        apply_filter=False,
        sample_rate=30000.0,
    )
    sweeps = []
    worker.sweepsReady.connect(sweeps.extend)

    # Initial packet without roles
    pkt1_hdr = json.dumps({
        "type": "erp_sweep",
        "sample_rate": 30000.0,
        "pre_offset_s": 0.01,
        "stim_timestamp_s": 1.0,
        "waveform": [1.0] * 100,
        "waveforms": [[1.0] * 100, [2.0] * 100],
    }).encode("utf-8")
    worker._parse_sliced_erp_packet([pkt1_hdr])
    worker._flush_sweep_payloads(force=True)
    assert len(sweeps) == 1
    assert "rd.0" not in sweeps[0]["aux_sweeps"]

    # Reload roles while "streaming"
    worker.set_channel_roles([
        {"source": "CH1", "name": "rd.0", "probe_type": "AS", "purpose": "probe", "bandpass": True},
        {"source": "CH2", "name": "env.thermode", "probe_type": "AS", "purpose": "thermode", "bandpass": False},
    ])
    pkt2_hdr = json.dumps({
        "type": "erp_sweep",
        "sample_rate": 30000.0,
        "pre_offset_s": 0.01,
        "stim_timestamp_s": 2.0,
        "waveform": [3.0] * 100,
        "waveforms": [[3.0] * 100, [4.0] * 100],
    }).encode("utf-8")
    worker._parse_sliced_erp_packet([pkt2_hdr])
    worker._flush_sweep_payloads(force=True)
    assert len(sweeps) == 2
    aux2 = sweeps[1]["aux_sweeps"]
    assert "rd.0" in aux2
    assert "rd.0.bp" in aux2
    assert "env.thermode" in aux2
    assert np.allclose(aux2["env.thermode"], 4.0)


def test_live_view_button_ttl_role_creates_protocol_marker(qtbot):
    """Verify ttl_role button event in OpenEphysLiveView records a protocol marker in ViewerState."""
    state = ViewerState()
    live_view = OpenEphysLiveView(state=state)
    qtbot.addWidget(live_view)

    live_view._on_protocol_event_ready(
        "ttl_role",
        1.25,
        {"name": "rec.button", "purpose": "button", "sample_number": 37500, "source": "ADC8"},
    )
    markers = state.get_protocol_markers()
    assert len(markers) == 1
    assert "rec.button" in markers[0]["label"]
    assert markers[0]["category"] == "button"
    assert markers[0]["timestamp_s"] == pytest.approx(1.25)
