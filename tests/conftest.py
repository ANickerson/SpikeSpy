import numpy as np
import pytest
import neo
import quantities as pq

from spikespy.ViewerState import ViewerState, tracked_neuron_unit


@pytest.fixture()
def synthetic_state():
    state = ViewerState()
    sampling_rate = 10000 * pq.Hz
    n_samples = 30000
    sig_data = np.random.randn(n_samples) * 0.1
    asig = neo.AnalogSignal(sig_data, units=pq.mV, sampling_rate=sampling_rate, name="sig1")

    ev_times = np.arange(0.1, 2.5, 0.2) * pq.s
    ev = neo.Event(ev_times, name="stim")

    seg = neo.Segment("test_seg")
    seg.analogsignals.append(asig)
    seg.events.append(ev)
    state.segment = seg

    u_times = ev_times + 0.015 * pq.s
    u_ev = neo.Event(u_times, name="unit_0")
    sg = tracked_neuron_unit(event=u_ev)

    state.set_data(analog_signal=asig, event_signal=ev, spike_groups=[sg])
    return state
