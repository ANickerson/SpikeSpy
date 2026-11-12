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


@pytest.fixture()
def live_streaming_state():
    state = ViewerState()
    state.start_live_session(sampling_rate=10000.0, window_size_ms=100.0, units=pq.uV)
    yield state
    state.stop_live_session()


def test_multitrace_pgview_live_streaming(qtbot, live_streaming_state):
    assert live_streaming_state.is_streaming is True
    view = MultiTracePGView(state=live_streaming_state)
    qtbot.addWidget(view)
    view.show()

    # Append first sweep
    w0 = np.random.randn(1000).astype(np.float32) * 50.0
    idx0 = live_streaming_state.append_sweep(1.0, w0)
    assert idx0 == 0
    view.on_new_sweep(idx0)

    assert view.img_item.isVisible()
    x_range = view.main_plot.viewRange()[0]
    y_range = view.main_plot.viewRange()[1]
    assert x_range[1] == 1000
    assert y_range[1] >= 1

    # Test mode toggle in live mode
    view.mode = "lines"
    view.on_new_sweep(idx0)
    assert view.lines_curve.isVisible()

    view.mode = "unitonly"
    view.on_new_sweep(idx0)
    assert not view.img_item.isVisible()
    assert not view.lines_curve.isVisible()

    view.mode = "heatmap"
    view.on_new_sweep(idx0)
    assert view.img_item.isVisible()

    # Test visibility culling
    view.setVisible(False)
    assert view.is_view_culled() is True

    w1 = np.random.randn(1000).astype(np.float32) * 50.0
    idx1 = live_streaming_state.append_sweep(2.0, w1)
    view.on_new_sweep(idx1)
    assert view._dirty is True

    view.setVisible(True)
    assert view.is_view_culled() is False
    assert view._dirty is False


def test_multitrace_fixed_pgview_live_streaming(qtbot, live_streaming_state):
    view = MultiTraceFixedPGView(state=live_streaming_state)
    qtbot.addWidget(view)
    view.show()

    # Initially before sweeps arrive
    assert view._auto_scale_needed is True

    # First sweep arrives
    w0 = np.random.randn(1000).astype(np.float32) * 50.0
    idx0 = live_streaming_state.append_sweep(1.0, w0)
    assert idx0 == 0

    assert view._auto_scale_needed is False
    assert view.active_curve.isVisible()

    x_range = view.main_plot.viewRange()[0]
    assert np.isclose(x_range[1], (1000 - 1) / 10000.0, atol=1e-3)

    # Append more sweeps
    for i in range(5):
        w = np.random.randn(1000).astype(np.float32) * 50.0
        live_streaming_state.append_sweep(2.0 + i, w)
    view.update_figure()
    assert view.lines_curve.isVisible()


def test_single_trace_pgview_live_streaming(qtbot, live_streaming_state):
    view = SingleTracePGView(state=live_streaming_state)
    qtbot.addWidget(view)
    view.show()

    assert getattr(view, "_setup_done", False) is False

    # First sweep arrives
    w0 = np.random.randn(1000).astype(np.float32) * 50.0
    idx0 = live_streaming_state.append_sweep(1.0, w0)
    assert idx0 == 0

    assert view._setup_done is True
    assert view.trace_curve.isVisible()
    x_range = view.main_plot.viewRange()[0]
    assert np.isclose(x_range[1], 1000 / 10000.0, atol=1e-3)

    # Subsequent sweeps
    w1 = np.random.randn(1000).astype(np.float32) * 50.0
    idx1 = live_streaming_state.append_sweep(2.0, w1)
    assert idx1 == 1
    assert view.trace_curve.isVisible()


def test_unit_pgview_live_streaming(qtbot, live_streaming_state):
    view = UnitPGView(state=live_streaming_state)
    qtbot.addWidget(view)
    view.show()

    # Sweep 0 with unit spike
    w0 = np.zeros(1000, dtype=np.float32)
    w0[250] = 100.0
    idx0 = live_streaming_state.append_sweep(1.0, w0)
    live_streaming_state.setUnit(250, stimno=0)

    assert len(view.lines) > 0
    assert view.mean_curve.isVisible()
    assert view.selected_curve.isVisible()

    # Sweep 1 with unit spike
    w1 = np.zeros(1000, dtype=np.float32)
    w1[260] = 95.0
    idx1 = live_streaming_state.append_sweep(2.0, w1)
    live_streaming_state.setUnit(260, stimno=1)

    assert len(view.lines) == 2
    assert view.mean_curve.isVisible()
