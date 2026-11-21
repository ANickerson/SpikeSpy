import numpy as np
import pytest
import quantities as pq
from PySide6.QtCore import Qt

from spikespy.ViewerState import ViewerState
from spikespy.SingleTracePGView import SingleTracePGView
from spikespy.MultiTracePGView import MultiTracePGView
from spikespy.MultiTraceFixedPGView import MultiTraceFixedPGView
from spikespy.legacy_mpl.SingleTraceView import SingleTraceView
from spikespy.legacy_mpl.MultiTraceView import MultiTraceView
from spikespy.legacy_mpl.MultiTraceFixedView import MultiTraceFixedView


def test_viewer_state_pre_stim_live_session():
    """Verify ViewerState initializes pre-stimulus offset, retains full sweep, and supports negative latencies."""
    state = ViewerState()
    # 20 ms pre-stimulus, 80 ms post-stimulus at 10 kHz -> 200 pre samples, 800 post samples = 1000 total samples
    state.start_live_session(sampling_rate=10000.0, window_size_ms=80.0, pre_ms=20.0, units=pq.uV)

    assert state.pre_offset_s == pytest.approx(0.02)
    assert state.pre_stim_ms == pytest.approx(20.0)
    assert state.get_erp().shape == (0, 1000)

    # Append sweep with pre-stimulus spike at sample 50 (-15 ms) and post-stimulus spike at sample 450 (+25 ms)
    sweep_data = np.zeros(1000, dtype=np.float32)
    sweep_data[50] = 50.0   # -15 ms latency
    sweep_data[450] = 100.0 # +25 ms latency
    idx = state.append_sweep(1.0, sweep_data)
    assert idx == 0
    assert state.get_erp().shape == (1, 1000)

    # Annotate pre-stimulus spike at sample 50
    state.setUnit(50, stimno=0)
    sg = state.getUnitGroup()
    assert sg.idx_arr[0][0] == 50

    # Latencies should reflect negative latency (-15 ms)
    lats = sg.get_latencies(state.event_signal)
    assert len(lats) == 1
    lat_ms = float(lats[0].rescale(pq.ms).magnitude)
    assert lat_ms == pytest.approx(-15.0)

    # Post-stimulus annotation at sample 450 (+25 ms)
    state.setUnit(450, stimno=0)
    assert sg.idx_arr[0][0] == 450
    lats = sg.get_latencies(state.event_signal)
    assert float(lats[0].rescale(pq.ms).magnitude) == pytest.approx(25.0)

    state.stop_live_session()


def test_single_trace_pg_and_mpl_views_pre_stim(qtbot):
    """Verify SingleTracePGView and SingleTraceView center on 0 with negative pre-stimulus x-range."""
    state = ViewerState()
    state.start_live_session(sampling_rate=10000.0, window_size_ms=100.0, pre_ms=25.0, units=pq.uV)

    sweep = np.zeros(1250, dtype=np.float32)  # 250 pre-samples (-25 ms) + 1000 post-samples (100 ms)
    sweep[100] = 75.0   # Pre-stim spike at -15 ms
    sweep[500] = 120.0  # Post-stim spike at +25 ms
    state.append_sweep(1.0, sweep)

    # 1. SingleTracePGView
    pg_view = SingleTracePGView(state=state)
    qtbot.addWidget(pg_view)
    pg_view.setup_figure()
    pg_view.update_figure()

    # Stim zero line must be at 0.0 s
    assert hasattr(pg_view, "stim_zero_line")
    assert pg_view.stim_zero_line.value() == pytest.approx(0.0)

    # X-axis range should span [-pre_offset_s, post_window_s]
    x_min, x_max = pg_view.main_plot.viewRange()[0]
    assert x_min == pytest.approx(-0.025, abs=0.005)
    assert x_max == pytest.approx(0.100, abs=0.005)

    # Click at -15 ms (-0.015 s) should set unit to sample 100
    pg_view.set_cur_pos((-0.015 + state.pre_offset_s) * 10000.0)
    assert state.getUnitGroup().idx_arr[0][0] == 100
    lat_ms = float(state.getUnitGroup().get_latencies(state.event_signal)[0].rescale(pq.ms).magnitude)
    assert lat_ms == pytest.approx(-15.0)

    # 2. Legacy Matplotlib SingleTraceView
    mpl_view = SingleTraceView(state=state)
    qtbot.addWidget(mpl_view)
    mpl_view.setupFigure()
    mpl_view.updateFigure_core()

    assert hasattr(mpl_view, "stim_zero_line")
    xlim = mpl_view.ax.get_xlim()
    assert xlim[0] == pytest.approx(-0.025, abs=0.005)
    assert xlim[1] == pytest.approx(0.100, abs=0.005)

    state.stop_live_session()


def test_multi_trace_pg_and_mpl_views_pre_stim(qtbot):
    """Verify MultiTracePGView and MultiTraceView axis formatting and stim zero lines."""
    state = ViewerState()
    state.start_live_session(sampling_rate=10000.0, window_size_ms=50.0, pre_ms=20.0, units=pq.uV)

    # 200 pre samples + 500 post samples = 700 samples
    for i in range(3):
        sweep = np.zeros(700, dtype=np.float32)
        state.append_sweep(float(i + 1), sweep)

    # 1. MultiTracePGView
    pg_view = MultiTracePGView(state=state)
    qtbot.addWidget(pg_view)
    pg_view.setup_figure()

    # Time axis tick formatting: sample 0 is -20 ms, sample 200 is 0 ms, sample 700 is 50 ms
    time_axis = pg_view.time_axis
    ticks = time_axis.tickStrings([0, 200, 700], 1.0, 100)
    assert float(ticks[0]) == pytest.approx(-20.0)
    assert float(ticks[1]) == pytest.approx(0.0)
    assert float(ticks[2]) == pytest.approx(50.0)

    # Zero line exists in channel plots
    assert hasattr(pg_view, "channel_panels")
    for panel in pg_view.channel_panels:
        assert hasattr(panel, "stim_zero_line")
        assert panel.stim_zero_line.value() == pytest.approx(200.0)
        assert panel.stim_zero_line.isVisible() is True

    # 2. Legacy Matplotlib MultiTraceView
    mpl_view = MultiTraceView(state=state)
    qtbot.addWidget(mpl_view)
    mpl_view.setup_figure()

    formatter = mpl_view.ax.xaxis.get_major_formatter()
    assert formatter(0, None) == "-20"
    assert formatter(200, None) == "0"
    assert formatter(700, None) == "50"
    assert hasattr(mpl_view, "stim_zero_line")
    assert mpl_view.stim_zero_line.get_xdata()[0] == pytest.approx(200.0)

    state.stop_live_session()


def test_multi_trace_fixed_pg_and_mpl_views_pre_stim(qtbot):
    """Verify MultiTraceFixedPGView and MultiTraceFixedView center on 0 with negative pre-stimulus x-range."""
    state = ViewerState()
    state.start_live_session(sampling_rate=10000.0, window_size_ms=60.0, pre_ms=15.0, units=pq.uV)

    # 150 pre samples + 600 post samples = 750 samples
    for i in range(4):
        sweep = np.zeros(750, dtype=np.float32)
        state.append_sweep(float(i + 1), sweep)

    # 1. MultiTraceFixedPGView
    pg_view = MultiTraceFixedPGView(state=state)
    qtbot.addWidget(pg_view)
    pg_view.update_figure()

    assert hasattr(pg_view, "stim_zero_line")
    assert pg_view.stim_zero_line.value() == pytest.approx(0.0)

    x_min, x_max = pg_view.main_plot.viewRange()[0]
    assert x_min == pytest.approx(-0.015, abs=0.005)
    assert x_max == pytest.approx(0.060, abs=0.005)

    # 2. Legacy Matplotlib MultiTraceFixedView
    mpl_view = MultiTraceFixedView(state=state)
    qtbot.addWidget(mpl_view)
    mpl_view.setup_figure()
    mpl_view.draw_figure()

    assert hasattr(mpl_view, "stim_zero_line")
    xlim = mpl_view.ax.get_xlim()
    assert xlim[0] == pytest.approx(-0.015, abs=0.005)
    assert xlim[1] == pytest.approx(0.060, abs=0.005)

    state.stop_live_session()
