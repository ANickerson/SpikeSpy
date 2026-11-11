import pytest
import numpy as np
import neo
import quantities as pq
from spikespy.ViewerState import ViewerState, tracked_neuron_unit
from spikespy.MultiTraceView import MultiTraceView
from spikespy.MultiTracePGView import MultiTracePGView


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


def test_multitrace_pgview_init(qtbot, synthetic_state):
    view = MultiTracePGView(state=synthetic_state)
    qtbot.addWidget(view)
    view.show()

    assert view.state == synthetic_state
    assert view.mode == "heatmap"

    # Step through stimuli
    synthetic_state.setStimNo(3)
    assert synthetic_state.stimno == 3

    # Test settings round-trip
    settings = view.get_settings()
    assert "xlim" in settings
    assert "mode" in settings
    view.set_settings(settings)

    # Test mode toggle
    view.mode = "lines"
    view.setup_figure()
    assert view.lines_curve.isVisible()

    view.mode = "heatmap"
    view.setup_figure()
    assert view.img_item.isVisible()


def test_views_comparable(qtbot, synthetic_state):
    mpl_view = MultiTraceView(state=synthetic_state)
    pg_view = MultiTracePGView(state=synthetic_state)
    qtbot.addWidget(mpl_view)
    qtbot.addWidget(pg_view)

    mpl_view.show()
    pg_view.show()

    # Verify both respond to state stimulus update
    synthetic_state.setStimNo(2)
    assert mpl_view.state.stimno == 2
    assert pg_view.state.stimno == 2
