import time
import numpy as np
import pytest
import quantities as pq
import neo
from PySide6.QtCore import Qt, QThreadPool, QTimer
from PySide6.QtWidgets import QApplication

from spikespy.ViewerState import ViewerState, tracked_neuron_unit
from spikespy.SingleTraceView import SingleTraceView
from spikespy.MultiTraceView import MultiTraceView
from spikespy.UnitView import UnitView
from spikespy.UnitFootprintView import UnitFootprintView
from spikespy.SpikeGroupTable import SpikeGroupTableView
from spikespy.processing import LiveERPBuffer


def test_hidden_views_do_not_render(qtbot):
    """Verify that views that are hidden do NOT render (render count == 0),
    and render once when made visible.
    """
    state = ViewerState()
    state.start_live_session(sampling_rate=10000.0, window_size_ms=50.0, units=pq.uV)

    st_view = SingleTraceView(state=state)
    mt_view = MultiTraceView(state=state)
    u_view = UnitView(state=state)
    uf_view = UnitFootprintView(state=state)

    tbl_view = SpikeGroupTableView(state=state)

    qtbot.addWidget(st_view)
    qtbot.addWidget(mt_view)
    qtbot.addWidget(u_view)
    qtbot.addWidget(uf_view)
    qtbot.addWidget(tbl_view)

    # Hide all views
    st_view.hide()
    mt_view.hide()
    u_view.hide()
    uf_view.hide()
    tbl_view.hide()

    # Reset render counts to 0
    st_view.render_count = 0
    mt_view.render_count = 0
    u_view.render_count = 0
    uf_view.render_count = 0
    tbl_view.render_count = 0

    # Feed 30 sweeps via state.append_sweep(...)
    for i in range(30):
        w = np.zeros(500, dtype=np.float32)
        state.append_sweep(float(i) * 0.01, w)
        QApplication.processEvents()

    # Wait for any background timers or throttles
    qtbot.wait(100)

    # Assert that view.render_count == 0 for all hidden views!
    assert st_view.render_count == 0, f"SingleTraceView rendered while hidden: {st_view.render_count}"
    assert mt_view.render_count == 0, f"MultiTraceView rendered while hidden: {mt_view.render_count}"
    assert u_view.render_count == 0, f"UnitView rendered while hidden: {u_view.render_count}"
    assert uf_view.render_count == 0, f"UnitFootprintView rendered while hidden: {uf_view.render_count}"
    assert tbl_view.render_count == 0, f"SpikeGroupTableView rendered while hidden: {tbl_view.render_count}"

    # Show SingleTraceView (view.show()), process Qt events
    st_view.show()
    qtbot.wait(100)
    QApplication.processEvents()

    # Assert view.render_count >= 1
    assert st_view.render_count >= 1, "SingleTraceView failed to render upon becoming visible"

    # Show other views as well and verify they also render once visible
    mt_view.show()
    u_view.show()
    uf_view.show()
    tbl_view.show()
    qtbot.wait(100)
    QApplication.processEvents()

    assert mt_view.render_count >= 1, "MultiTraceView failed to render upon becoming visible"
    assert u_view.render_count >= 1, "UnitView failed to render upon becoming visible"
    assert uf_view.render_count >= 1, "UnitFootprintView failed to render upon becoming visible"
    assert tbl_view.render_count >= 1, "SpikeGroupTableView failed to render upon becoming visible"

    state.stop_live_session()


def test_ui_refresh_rate_configuration(qtbot):
    """Verify that ViewerState ui refresh rate updates timers across views."""
    state = ViewerState()
    assert state.get_ui_refresh_rate_ms() == 33

    st_view = SingleTraceView(state=state)
    mt_view = MultiTraceView(state=state)
    u_view = UnitView(state=state)
    uf_view = UnitFootprintView(state=state)
    tbl_view = SpikeGroupTableView(state=state)

    for v in [st_view, mt_view, u_view, uf_view]:
        qtbot.addWidget(v)
        assert hasattr(v, "render_timer")
        assert v.render_timer.interval() == 33

    assert hasattr(tbl_view, "update_timer")
    assert tbl_view.update_timer.interval() == 33

    # Update UI refresh rate in ViewerState
    state.set_ui_refresh_rate_ms(50)
    assert state.get_ui_refresh_rate_ms() == 50

    for v in [st_view, mt_view, u_view, uf_view]:
        assert v.render_timer.interval() == 50
    assert tbl_view.update_timer.interval() == 50


def test_ui_rendering_refresh_rate_cap(qtbot):
    """Verify that UI render frame rate is bounded by the target refresh rate (e.g. 30 FPS / ~33 ms).
    Simulating a high-frequency stimulus burst (100 Hz) coalesces incoming sweeps rather
    than triggering 100 individual full redraws.
    """
    state = ViewerState()
    state.start_live_session(sampling_rate=10000.0, window_size_ms=50.0, units=pq.uV)

    st_view = SingleTraceView(state=state)
    qtbot.addWidget(st_view)
    st_view.show()
    qtbot.wait(100)
    QApplication.processEvents()

    # Reset render count before burst
    st_view.render_count = 0

    # Blast 100 sweeps in ~50 ms (simulating a 100 Hz burst)
    t_start = time.time()
    for i in range(100):
        w = np.zeros(500, dtype=np.float32)
        state.append_sweep(float(i) * 0.01, w)
        QApplication.processEvents()
        time.sleep(0.0005)  # 0.5 ms between incoming packets

    t_elapsed = time.time() - t_start

    # Count render calls: assert view.render_count < 30
    assert st_view.render_count < 30, (
        f"Render count was not throttled/capped: {st_view.render_count} renders in {t_elapsed * 1000:.1f} ms"
    )

    state.stop_live_session()


def test_visibility_culling_in_background_tabs(qtbot):
    """Hidden views in background tabs must not execute rendering or increment render_count."""
    from PySide6.QtWidgets import QTabWidget
    state = ViewerState()
    state.start_live_session(sampling_rate=10000.0, window_size_ms=50.0, units=pq.uV)

    tabs = QTabWidget()
    qtbot.addWidget(tabs)

    single_view = SingleTraceView(state=state)
    multi_view = MultiTraceView(state=state)
    table_view = SpikeGroupTableView(state=state)

    tabs.addTab(single_view, "SingleTrace")
    tabs.addTab(multi_view, "MultiTrace")
    tabs.addTab(table_view, "Table")

    tabs.show()
    qtbot.waitExposed(tabs)

    # Initial state: only active tab (tab 0: single_view) is visible; background tabs are culled
    assert not single_view.is_view_culled()
    assert multi_view.is_view_culled()
    assert table_view.is_view_culled()

    # Reset render counts to baseline
    single_view.render_count = 0
    multi_view.render_count = 0
    table_view.render_count = 0

    # Trigger renders across all views while tab 0 is active
    single_view.updateFigure_core()
    multi_view.on_new_sweep(0)
    table_view.update()

    # Active tab rendered
    assert single_view.render_count == 1
    # Background tabs culled (did NOT render)
    assert multi_view.render_count == 0
    assert table_view.render_count == 0

    # All culled views marked dirty
    assert multi_view._dirty is True
    assert table_view._dirty is True

    # Switch to tab 1 (MultiTraceView): showEvent should fire and trigger rendering
    tabs.setCurrentIndex(1)
    assert not multi_view.is_view_culled()
    assert multi_view.render_count == 1
    assert multi_view._dirty is False

    # Switch to tab 2 (SpikeGroupTableView)
    tabs.setCurrentIndex(2)
    assert not table_view.is_view_culled()
    assert table_view.render_count == 1
    assert table_view._dirty is False

    state.stop_live_session()


def test_async_spike_tracking_throughput_and_responsiveness(qtbot):
    """Verify that spike tracking runs asynchronously without blocking the UI thread,
    keeps up with high-frequency stimulus streams, and tracks multiple units across channels.
    """
    state = ViewerState()
    state.start_live_session(sampling_rate=10000.0, window_size_ms=50.0, units=pq.uV)

    # Setup 3 recording channels
    buf0 = LiveERPBuffer(window_samples=500, units=pq.uV)
    buf1 = LiveERPBuffer(window_samples=500, units=pq.uV)
    buf2 = LiveERPBuffer(window_samples=500, units=pq.uV)
    state._live_erp_buffer = buf0
    state._live_aux_erp_buffers = {"CH0": buf0, "CH1": buf1, "CH2": buf2}
    state.active_channel_names = ["CH0", "CH1", "CH2"]

    # Configure 3 units with auto_track=True on different channels
    sg0 = tracked_neuron_unit(
        event=neo.Event([], name="unit_0", units=pq.s),
        channel="CH0",
        tracking_channel="CH0",
        auto_track=True,
        tracking_threshold_mv=0.02,
        tracking_window_ms=5.0,
    )
    sg1 = tracked_neuron_unit(
        event=neo.Event([], name="unit_1", units=pq.s),
        channel="CH1",
        tracking_channel="CH1",
        auto_track=True,
        tracking_threshold_mv=0.02,
        tracking_window_ms=5.0,
    )
    sg2 = tracked_neuron_unit(
        event=neo.Event([], name="unit_2", units=pq.s),
        channel="CH2",
        tracking_channel="CH2",
        auto_track=True,
        tracking_threshold_mv=0.02,
        tracking_window_ms=5.0,
    )
    state.spike_groups = [sg0, sg1, sg2]

    # Sweep 0: initial seed spikes
    w0 = np.zeros(500, dtype=np.float32)
    w0[100] = 50.0  # 10.0 ms
    w1 = np.zeros(500, dtype=np.float32)
    w1[150] = 50.0  # 15.0 ms
    w2 = np.zeros(500, dtype=np.float32)
    w2[200] = 50.0  # 20.0 ms

    state.append_sweep(0.0, w0, aux_sweeps={"CH1": w1, "CH2": w2})
    state.setUnit(100, stimno=0, unit_idx=0)
    state.setUnit(150, stimno=0, unit_idx=1)
    state.setUnit(200, stimno=0, unit_idx=2)

    # Monitor UI thread responsiveness with a high-frequency Qt timer
    ui_heartbeat_ticks = []
    heartbeat_timer = QTimer()
    heartbeat_timer.setInterval(5)  # 5 ms tick
    heartbeat_timer.timeout.connect(lambda: ui_heartbeat_ticks.append(time.time()))
    heartbeat_timer.start()

    # Append 20 sweeps with synthetic spikes drifting slightly
    for s in range(1, 21):
        w0_s = np.zeros(500, dtype=np.float32)
        w0_s[100 + (s % 4)] = 50.0
        w1_s = np.zeros(500, dtype=np.float32)
        w1_s[150 + (s % 4)] = 50.0
        w2_s = np.zeros(500, dtype=np.float32)
        w2_s[200 + (s % 4)] = 50.0

        state.append_sweep(float(s) * 0.05, w0_s, aux_sweeps={"CH1": w1_s, "CH2": w2_s})
        QApplication.processEvents()
        time.sleep(0.002)

    heartbeat_timer.stop()

    # Wait for background tracking to complete
    QThreadPool.globalInstance().waitForDone(2000)
    qtbot.wait(100)
    QApplication.processEvents()

    # Assert all 3 units have detected spikes across the sweeps
    for u_idx, sg in enumerate(state.spike_groups):
        tracked = [x for x in sg.idx_arr[1:21] if x is not None]
        assert len(tracked) >= 18, (
            f"Unit {u_idx} failed to detect spikes asynchronously across sweeps: {len(tracked)}/20 detected"
        )

    # Measure UI thread responsiveness during the stream: heartbeat timer ticked continuously
    assert len(ui_heartbeat_ticks) > 5, "UI thread was completely blocked during live streaming"

    state.stop_live_session()
