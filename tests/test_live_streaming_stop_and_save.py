"""Tests verifying that stopping live streaming preserves all recorded history,
allows saving to NIX / loading back with all channels/events/spikes intact,
and allows continuing/resuming recording without wiping previous sweeps.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np
import pytest
import quantities as pq
from PySide6.QtWidgets import QApplication

from spikespy.ViewerState import ViewerState
from spikespy.MultiTracePGView import MultiTracePGView
from spikespy.SingleTracePGView import SingleTracePGView
from spikespy.io_manager import save_state_to_nix, load_file
from spikespy.open_ephys_addin.oe_live_view import OpenEphysLiveView


@pytest.fixture()
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_stop_live_session_preserves_history(qapp):
    state = ViewerState()
    state.start_live_session(
        sampling_rate=10000.0,
        window_size_ms=50.0,
        units=pq.uV,
        channel_name="CH0",
        aux_channels=["CH1"],
    )

    n_samples = 500  # 50ms at 10kHz
    for i in range(10):
        wave = np.ones((2, n_samples), dtype=np.float32) * (i + 1)
        state.append_sweep(
            stim_timestamp=float(i) * 0.1,
            sweep_waveform=wave,
            spike_latencies_s={0: 0.01 + i * 0.001},
        )

    assert state.is_streaming is True
    assert len(state.get_erp()) == 10

    # Stop live session
    state.stop_live_session()

    # Verify state is not streaming but history is preserved
    assert state.is_streaming is False
    assert state.has_live_session() is True
    assert len(state.event_signal) == 10
    assert len(state.get_erp()) == 10
    assert state.get_channel_erp("CH0") is not None
    assert len(state.get_channel_erp("CH0")) == 10
    assert state.get_channel_erp("CH1") is not None
    assert len(state.get_channel_erp("CH1")) == 10
    assert len(state.spike_groups[0].event) == 10

    # Available channels still reported accurately
    channels = state.get_available_channels()
    assert "CH0" in channels and "CH1" in channels

    # Channel switching works seamlessly while stopped
    state.set_active_channel("CH1")
    assert state.analog_signal.name == "CH1"
    assert len(state.get_erp()) == 10


def test_views_render_stopped_history(qapp):
    state = ViewerState()
    state.start_live_session(
        sampling_rate=10000.0,
        window_size_ms=50.0,
        units=pq.uV,
        channel_name="CH0",
        aux_channels=["CH1"],
    )

    n_samples = 500
    for i in range(8):
        wave = np.ones((2, n_samples), dtype=np.float32) * (i + 1)
        state.append_sweep(
            stim_timestamp=float(i) * 0.1,
            sweep_waveform=wave,
            spike_latencies_s={0: 0.01 + i * 0.001},
        )

    multi_view = MultiTracePGView()
    multi_view.set_state(state)
    single_view = SingleTracePGView()
    single_view.set_state(state)

    state.stop_live_session()
    qapp.processEvents()

    # MultiTracePGView live buffer remains accessible
    live_buf = multi_view.get_live_buffer()
    assert live_buf is not None
    assert len(live_buf) == 8

    # SingleTracePGView ERP signal is accessible
    single_erp = single_view.get_erp_signal()
    assert single_erp is not None
    assert len(single_erp) == 8

    multi_view.close()
    single_view.close()


def test_save_and_reload_stopped_live_session_nix(qapp):
    state = ViewerState()
    state.start_live_session(
        sampling_rate=10000.0,
        window_size_ms=50.0,
        units=pq.uV,
        channel_name="CH0",
        aux_channels=["CH1"],
    )

    n_samples = 500
    for i in range(6):
        wave = np.ones((2, n_samples), dtype=np.float32) * (i + 1)
        state.append_sweep(
            stim_timestamp=float(i) * 0.1,
            sweep_waveform=wave,
            spike_latencies_s={0: 0.012 + i * 0.001},
        )

    state.stop_live_session()

    with tempfile.TemporaryDirectory() as td:
        h5_path = Path(td) / "saved_live_session.h5"
        save_state_to_nix(str(h5_path), state.spike_groups, state.segment)
        assert h5_path.exists()

        # Reload the saved file
        loaded_data, loaded_sig, loaded_evt, loaded_sgs = load_file(str(h5_path), "h5")

        assert len(loaded_evt) == 6
        assert len(loaded_sgs) >= 1
        assert len(loaded_sgs[0].event) == 6

        # Check analog signals
        sig_names = [s.name for s in loaded_data.analogsignals]
        assert "CH0" in sig_names or "CH1" in sig_names


def test_continue_recording_after_stop(qapp):
    state = ViewerState()
    state.start_live_session(
        sampling_rate=10000.0,
        window_size_ms=50.0,
        units=pq.uV,
        channel_name="CH0",
        aux_channels=["CH1"],
    )

    n_samples = 500
    for i in range(5):
        wave = np.ones((2, n_samples), dtype=np.float32) * (i + 1)
        state.append_sweep(
            stim_timestamp=float(i) * 0.1,
            sweep_waveform=wave,
            spike_latencies_s={0: 0.01 + i * 0.001},
        )

    assert len(state.get_erp()) == 5
    state.stop_live_session()
    assert state.is_streaming is False
    assert len(state.get_erp()) == 5

    # Resume session to continue recording
    state.resume_live_session()
    assert state.is_streaming is True

    # Append 4 more sweeps
    for i in range(5, 9):
        wave = np.ones((2, n_samples), dtype=np.float32) * (i + 1)
        state.append_sweep(
            stim_timestamp=float(i) * 0.1,
            sweep_waveform=wave,
            spike_latencies_s={0: 0.01 + i * 0.001},
        )

    # All 9 sweeps should be present seamlessly
    assert len(state.get_erp()) == 9
    assert len(state.event_signal) == 9
    assert len(state.spike_groups[0].event) == 9
    # Sweep values should reflect continuous history
    np.testing.assert_allclose(state.get_erp().magnitude[0, 0], 1.0)
    np.testing.assert_allclose(state.get_erp().magnitude[8, 0], 9.0)

    state.stop_live_session()
    assert len(state.get_erp()) == 9


def test_oe_live_view_button_and_reset(qapp, monkeypatch):
    state = ViewerState()
    live_view = OpenEphysLiveView(state=state)

    state.start_live_session(
        sampling_rate=10000.0,
        window_size_ms=50.0,
        units=pq.uV,
        channel_name="CH0",
        aux_channels=["CH1"],
    )

    n_samples = 500
    for i in range(4):
        wave = np.ones((2, n_samples), dtype=np.float32) * (i + 1)
        state.append_sweep(
            stim_timestamp=float(i) * 0.1,
            sweep_waveform=wave,
            spike_latencies_s={0: 0.01 + i * 0.001},
        )

    # Stop stream via UI
    live_view.stop_stream()
    assert "Resume Live Stream" in live_view.btn_toggle_stream.text()
    assert "4 sweeps recorded" in live_view.lbl_status.text()
    assert len(state.get_erp()) == 4

    # Reset session (simulate user clicking Yes to confirm)
    from PySide6.QtWidgets import QMessageBox

    monkeypatch.setattr(
        QMessageBox, "question", lambda *args, **kwargs: QMessageBox.Yes
    )

    live_view.reset_session()
    assert "Start Live Stream" in live_view.btn_toggle_stream.text()
    assert len(state.get_erp()) == 0

    live_view.close()
