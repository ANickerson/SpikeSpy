import pytest
import numpy as np
import neo
import quantities as pq
from spikespy.ViewerState import ViewerState, tracked_neuron_unit
from spikespy.MultiTraceView import MultiTraceView
from spikespy.MultiTracePGView import MultiTracePGView
from spikespy.MultiTraceFixedPGView import MultiTraceFixedPGView
from spikespy.UnitPGView import UnitPGView
from spikespy.SingleTracePGView import SingleTracePGView


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


def test_multitrace_pgview_init_and_selector(qtbot, synthetic_state):
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

    # Test selector toggle & point placement
    view.toggle_polySelector(True)
    assert view.selector_active is True

    # Place vertices
    view.selector_points = [[150.0, 0.0], [155.0, 3.0], [160.0, 6.0]]
    view._redraw_selector()
    assert view.selector_line_item.isVisible()
    assert view.corridor_item.isVisible()

    # Execute polySelect
    view.dialogPolySelect.window_size_input.setValue(10)
    view.dialogPolySelect.minimumThreshold.setValue(0.0)
    view.polySelect()

    # Selector should be inactive and cleared after track
    assert view.selector_active is False
    assert len(view.selector_points) == 0


def test_multitrace_fixed_pgview(qtbot, synthetic_state):
    view = MultiTraceFixedPGView(state=synthetic_state)
    qtbot.addWidget(view)
    view.show()

    assert view.state == synthetic_state
    synthetic_state.setStimNo(4)
    assert view.active_curve.isVisible()

    # Settings test
    settings = view.get_settings()
    assert "n_lines_pre" in settings
    assert "scale" in settings
    view.set_settings({"n_lines_pre": 3, "scale": 2.0})
    assert view.settings["n_lines_pre"] == 3


def test_unit_pgview(qtbot, synthetic_state):
    view = UnitPGView(state=synthetic_state)
    qtbot.addWidget(view)
    view.show()

    assert view.state == synthetic_state
    synthetic_state.setStimNo(2)
    assert len(view.lines) > 0
    assert view.selected_curve.isVisible()


def test_single_trace_pgview(qtbot, synthetic_state):
    view = SingleTracePGView(state=synthetic_state)
    qtbot.addWidget(view)
    view.show()

    assert view.state == synthetic_state
    synthetic_state.setStimNo(1)
    assert view.trace_curve.isVisible()

    # Test zoom shortcut
    from PySide6.QtGui import QKeyEvent
    from PySide6.QtCore import QEvent, Qt

    event = QKeyEvent(QEvent.KeyPress, Qt.Key_Z, Qt.NoModifier)
    view.keyPressEvent(event)

    settings = view.get_settings()
    assert "xlim" in settings
