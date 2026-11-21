import neo
import numpy as np
import pytest
import quantities as pq

from spikespy.ViewerState import ViewerState, tracked_neuron_unit
from spikespy.open_ephys_addin.oe_live_view import OpenEphysLiveView


def _loaded_two_channel_state(stim_seconds=(0.5, 1.0, 1.5)):
    sampling_rate = 1000.0
    stim_seconds = np.asarray(stim_seconds, dtype=np.float64)
    n_samples = int((float(stim_seconds.max()) + 1.0) * sampling_rate)
    samples = np.arange(n_samples, dtype=np.float64)

    ch0 = neo.AnalogSignal(
        samples * pq.uV, sampling_rate=sampling_rate * pq.Hz, name="CH0"
    )
    ch1 = neo.AnalogSignal(
        2.0 * samples * pq.uV, sampling_rate=sampling_rate * pq.Hz, name="CH1"
    )

    stim_times = stim_seconds * pq.s
    stim_evt = neo.Event(stim_times, name="stim")

    spike_times = (stim_seconds + 0.02) * pq.s
    spike_evt = neo.Event(spike_times, name="unit_0")
    spike_evt.annotations["type"] = "unit"
    spike_evt.annotations["channel"] = "CH0"
    spike_group = tracked_neuron_unit(event=spike_evt, channel="CH0")

    segment = neo.Segment(name="loaded_recording")
    segment.analogsignals.append(ch0)
    segment.analogsignals.append(ch1)
    segment.events.append(stim_evt)
    segment.events.append(spike_evt)

    state = ViewerState()
    state.segment = segment
    state.set_data(analog_signal=ch0, event_signal=stim_evt, spike_groups=[spike_group])
    return state


def test_replay_loaded_signal_as_live(qtbot):
    state = _loaded_two_channel_state()
    view = OpenEphysLiveView(state=state)
    qtbot.addWidget(view)

    view.spin_window_ms.setValue(100.0)
    view.spin_pre_ms.setValue(10.0)
    view.edit_active_channels.setText("0-1")
    view.spin_rec_chan.setValue(0)
    view.spin_replay_speed.setValue(1000.0)

    view.start_replay()
    assert view.is_replaying
    qtbot.waitUntil(lambda: not view.is_replaying, timeout=5000)

    assert state.segment.name == "OpenEphys Live"
    assert len(state.event_signal) == 3
    np.testing.assert_allclose(
        state.event_signal.rescale(pq.s).magnitude, [0.5, 1.0, 1.5]
    )

    erp = state.get_erp()
    assert erp.shape == (3, 110)
    np.testing.assert_allclose(erp.magnitude[:, 10], [500.0, 1000.0, 1500.0])

    assert "CH1" in state._live_aux_erp_buffers
    ch1_erp = state._live_aux_erp_buffers["CH1"].data
    np.testing.assert_allclose(ch1_erp.magnitude[:, 10], [1000.0, 2000.0, 3000.0])

    np.testing.assert_array_equal(
        [state.spike_groups[0].idx_arr[i][0] for i in range(3)], [30, 30, 30]
    )
    np.testing.assert_allclose(
        state.spike_groups[0].get_latencies(state.event_signal).rescale(pq.ms).magnitude,
        [20.0, 20.0, 20.0],
    )

    assert "Replay complete" in view.lbl_status.text()
    assert view.btn_replay_loaded.text() == "Replay Loaded Signal as Live"


def test_replay_can_be_stopped_early(qtbot):
    state = _loaded_two_channel_state()
    view = OpenEphysLiveView(state=state)
    qtbot.addWidget(view)

    view.spin_window_ms.setValue(100.0)
    view.spin_pre_ms.setValue(10.0)
    view.edit_active_channels.setText("0")
    view.spin_replay_speed.setValue(1.0)

    view.start_replay()
    assert view.is_replaying
    view.stop_replay()

    assert not view.is_replaying
    assert not state.is_streaming
    assert "Replay stopped" in view.lbl_status.text()


def test_replay_timeline_skips_silent_gaps(qtbot):
    state = _loaded_two_channel_state(stim_seconds=(0.5, 2.5, 5.0))
    view = OpenEphysLiveView(state=state)
    qtbot.addWidget(view)

    view.spin_window_ms.setValue(100.0)
    view.spin_pre_ms.setValue(10.0)
    view.edit_active_channels.setText("0")
    view.spin_replay_speed.setValue(1.0)

    view.start_replay()
    try:
        assert view.slider_replay_timeline.isEnabled()
        assert view.slider_replay_timeline.maximum() == 4500
        assert len(state.event_signal) == 1
        np.testing.assert_allclose(state.event_signal[0].magnitude, 0.5)

        # Next Event jumps over the 2 s silent gap instead of waiting for it.
        view.btn_replay_next.click()
        assert len(state.event_signal) == 1
        np.testing.assert_allclose(state.event_signal[0].magnitude, 2.5)
        assert view.slider_replay_timeline.value() == 2000

        # Prev Event jumps back over the previous gap.
        view.btn_replay_prev.click()
        np.testing.assert_allclose(state.event_signal[0].magnitude, 0.5)
        assert view.slider_replay_timeline.value() == 0

        # Dragging the timeline scrubs to the nearest event.
        view.slider_replay_timeline.setValue(4500)
        qtbot.waitUntil(
            lambda: abs(state.event_signal[0].magnitude - 5.0) < 1e-9,
            timeout=2000,
        )
        assert view.slider_replay_timeline.value() == 4500
    finally:
        view.stop_replay()


def test_replay_requires_events(qtbot):
    state = ViewerState()
    sig = neo.AnalogSignal(
        np.zeros(100) * pq.uV, sampling_rate=1000 * pq.Hz, name="CH0"
    )
    state.segment = neo.Segment(name="no_events")
    state.segment.analogsignals.append(sig)
    state.set_data(analog_signal=sig, event_signal=neo.Event())

    view = OpenEphysLiveView(state=state)
    qtbot.addWidget(view)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "spikespy.open_ephys_addin.oe_live_view.QMessageBox.warning",
            lambda *args, **kwargs: None,
        )
        view.start_replay()

    assert not view.is_replaying
