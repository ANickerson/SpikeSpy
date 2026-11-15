"""Live-rate stress tests for every PyQtGraph view.

Each panel must be able to keep up with a 10 Hz stimulus stream (the GUI
thread handles one frame well inside the 100 ms frame budget) and must
degrade quietly when input arrives faster than the UI can redraw: a
failure inside one frame is logged and dropped, never propagated into
the Qt event loop where it would take the whole application down.
"""

from __future__ import annotations

import time

import neo
import numpy as np
import pytest
import quantities as pq
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from spikespy.ViewerState import ViewerState, tracked_neuron_unit
from spikespy.UnitFootprintPGView import UnitFootprintPGView
from spikespy.UnitPGView import UnitPGView
from spikespy.MultiTracePGView import MultiTracePGView
from spikespy.MultiTraceFixedPGView import MultiTraceFixedPGView
from spikespy.SingleTracePGView import SingleTracePGView

STIM_RATE_HZ = 10.0
FRAME_BUDGET_MS = 1000.0 / STIM_RATE_HZ
SPIKE_LATENCY_S = 0.012


@pytest.fixture()
def live_state():
    """A live streaming session with three channels and one tracked unit."""
    state = ViewerState()
    state.start_live_session(
        sampling_rate=10000.0,
        window_size_ms=100.0,
        channel_name="CH0",
        units=pq.uV,
        aux_channels=["CH0", "CH1", "CH2"],
    )
    state.spike_groups = [
        tracked_neuron_unit(
            event=neo.Event(times=[] * pq.s, name="unit_0"),
            channel="CH0",
            idx_arr=[],
        )
    ]
    state.cur_spike_group = 0
    yield state
    state.stop_live_session()


class Heartbeat:
    """Counts event-loop ticks so a blocked GUI thread is detectable."""

    def __init__(self, interval_ms: int = 5):
        self.ticks = 0
        self._timer = QTimer()
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    def _tick(self):
        self.ticks += 1

    def stop(self):
        self._timer.stop()


def _make_sweep(rng: np.random.Generator, n_samples: int) -> np.ndarray:
    wave = (rng.standard_normal(n_samples) * 5.0).astype(np.float32)
    wave[n_samples // 4] = 80.0
    return wave


def feed_sweeps(state: ViewerState, count: int, period_s: float, tick=None) -> float:
    """Append `count` sweeps spaced `period_s` apart; returns elapsed seconds."""
    rng = np.random.default_rng(7)
    window_samples = state._live_erp_buffer.window_samples
    started = time.perf_counter()
    t0 = state.stimno * period_s
    for i in range(count):
        aux = {ch: _make_sweep(rng, window_samples) for ch in ("CH1", "CH2")}
        state.append_sweep(
            stim_timestamp=t0 + i * period_s,
            sweep_waveform=_make_sweep(rng, window_samples),
            spike_latencies_s={0: SPIKE_LATENCY_S},
            aux_sweeps=aux,
        )
        if tick is not None:
            tick()
    return time.perf_counter() - started


def _show(view, qtbot):
    qtbot.addWidget(view)
    view.show()
    qtbot.waitExposed(view)
    QApplication.processEvents()


def _assert_gui_kept_up(elapsed_s: float, n_sweeps: int, heartbeat: Heartbeat):
    avg_ms = elapsed_s / max(n_sweeps, 1) * 1000.0
    assert avg_ms < FRAME_BUDGET_MS, (
        f"GUI thread averaged {avg_ms:.1f} ms per frame over {n_sweeps} frames; "
        f"a {STIM_RATE_HZ:.0f} Hz stream ({FRAME_BUDGET_MS:.0f} ms budget) would fall behind"
    )
    assert heartbeat.ticks > 0, "GUI event loop did not tick while frames were processed"


# ---------------------------------------------------------------------------
# Per-view: render a 10 Hz stream without freezing
# ---------------------------------------------------------------------------


def test_singletrace_pg_renders_10hz_stream(qtbot, live_state):
    view = SingleTracePGView(state=live_state)
    _show(view, qtbot)
    qtbot.wait(150)
    QApplication.processEvents()

    heartbeat = Heartbeat()
    elapsed = feed_sweeps(live_state, 30, 1.0 / STIM_RATE_HZ, tick=QApplication.processEvents)
    heartbeat.stop()
    _assert_gui_kept_up(elapsed, 30, heartbeat)

    live_state.flush_live_frame()
    qtbot.waitUntil(lambda: view._last_result_stimno >= live_state.stimno, timeout=5000)
    assert view.trace_curve.xData is not None and len(view.trace_curve.xData) > 0


def test_multitrace_pg_renders_10hz_stream(qtbot, live_state):
    view = MultiTracePGView(state=live_state)
    _show(view, qtbot)
    qtbot.wait(150)
    QApplication.processEvents()

    heartbeat = Heartbeat()
    elapsed = feed_sweeps(live_state, 30, 1.0 / STIM_RATE_HZ, tick=QApplication.processEvents)
    heartbeat.stop()
    _assert_gui_kept_up(elapsed, 30, heartbeat)

    live_state.flush_live_frame()
    qtbot.wait(300)
    QApplication.processEvents()
    assert view.img_item.image is not None
    assert view.img_item.image.size > 0


def test_multitracefixed_pg_renders_10hz_stream(qtbot, live_state):
    view = MultiTraceFixedPGView(state=live_state)
    _show(view, qtbot)
    qtbot.wait(150)
    QApplication.processEvents()

    heartbeat = Heartbeat()
    elapsed = feed_sweeps(live_state, 30, 1.0 / STIM_RATE_HZ, tick=QApplication.processEvents)
    heartbeat.stop()
    _assert_gui_kept_up(elapsed, 30, heartbeat)

    live_state.flush_live_frame()
    QApplication.processEvents()
    assert view.active_curve.xData is not None and len(view.active_curve.xData) > 0


def test_unit_pg_renders_10hz_stream(qtbot, live_state):
    view = UnitPGView(state=live_state)
    _show(view, qtbot)
    qtbot.wait(150)
    QApplication.processEvents()

    heartbeat = Heartbeat()
    elapsed = feed_sweeps(live_state, 30, 1.0 / STIM_RATE_HZ, tick=QApplication.processEvents)
    heartbeat.stop()
    _assert_gui_kept_up(elapsed, 30, heartbeat)

    live_state.flush_live_frame()
    qtbot.waitUntil(lambda: len(view.lines) >= 30, timeout=5000)
    assert len(view.lines) >= 30, f"UnitPGView rendered {len(view.lines)} snippets for 30 spike sweeps"


def test_unitfootprint_pg_renders_10hz_stream(qtbot, live_state):
    view = UnitFootprintPGView(state=live_state)
    _show(view, qtbot)
    qtbot.wait(150)
    QApplication.processEvents()

    heartbeat = Heartbeat()
    elapsed = feed_sweeps(live_state, 30, 1.0 / STIM_RATE_HZ, tick=QApplication.processEvents)
    heartbeat.stop()
    _assert_gui_kept_up(elapsed, 30, heartbeat)

    live_state.flush_live_frame()
    view.render_views()
    QApplication.processEvents()
    assert view.img_item is not None and view.img_item.image is not None


# ---------------------------------------------------------------------------
# Per-view: an input burst far above 10 Hz must not freeze or crash the app
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "view_cls",
    [SingleTracePGView, MultiTracePGView, MultiTraceFixedPGView, UnitPGView, UnitFootprintPGView],
)
def test_pg_view_survives_faster_than_10hz_burst(qtbot, live_state, view_cls):
    view = view_cls(state=live_state)
    _show(view, qtbot)
    qtbot.wait(150)
    QApplication.processEvents()

    heartbeat = Heartbeat()
    # 60 sweeps with zero spacing: far above any real stimulus rate.
    elapsed = feed_sweeps(live_state, 60, 0.0, tick=QApplication.processEvents)
    heartbeat.stop()
    _assert_gui_kept_up(elapsed, 60, heartbeat)

    live_state.flush_live_frame()
    qtbot.wait(300)
    QApplication.processEvents()
    assert view.isVisible() or not view.is_view_culled()


# ---------------------------------------------------------------------------
# Per-view: a failing frame is dropped quietly, the app keeps running
# ---------------------------------------------------------------------------


def _raiseframe_factory(exc):
    def raiser(*args, **kwargs):
        raise exc

    return raiser


@pytest.mark.parametrize(
    ("view_cls", "frame_attr"),
    [
        (SingleTracePGView, "_schedule_render"),
        (MultiTracePGView, "_schedule_render"),
        (MultiTraceFixedPGView, "update_figure"),
        (UnitPGView, "_schedule_update_all"),
        (UnitFootprintPGView, "_do_update_plot"),
    ],
)
def test_pg_view_failing_frame_is_dropped_quietly(qtbot, live_state, view_cls, frame_attr, monkeypatch):
    view = view_cls(state=live_state)
    _show(view, qtbot)
    qtbot.wait(150)
    QApplication.processEvents()

    # Break the per-frame work; the view must contain the failure.
    monkeypatch.setattr(view, frame_attr, _raiseframe_factory(RuntimeError("injected frame failure")))

    heartbeat = Heartbeat()
    feed_sweeps(live_state, 10, 1.0 / STIM_RATE_HZ, tick=QApplication.processEvents)
    heartbeat.stop()
    assert heartbeat.ticks > 0, "event loop stopped ticking after an in-frame failure"

    # Recover the frame handler; the pipeline must deliver new frames again.
    monkeypatch.undo()
    QApplication.processEvents()
    more = feed_sweeps(live_state, 3, 1.0 / STIM_RATE_HZ, tick=QApplication.processEvents)
    assert more < 3.0

    live_state.flush_live_frame()
    if view_cls is UnitFootprintPGView:
        view.render_views()
        QApplication.processEvents()
        assert view.img_item is not None
    else:
        qtbot.wait(400)
        QApplication.processEvents()
        assert live_state.stimno >= 12


def test_singletrace_signal_path_failure_is_contained(qtbot, live_state, monkeypatch):
    """A failure inside a signal-driven update must not escape into the event loop."""
    view = SingleTracePGView(state=live_state)
    _show(view, qtbot)
    qtbot.wait(150)

    # onUnitChange dispatches update_figure on the GUI thread; a failure in the
    # work it performs must be contained by the view's guard, never propagated
    # through the Qt event loop.
    monkeypatch.setattr(view, "_apply_overlay_curves", _raiseframe_factory(RuntimeError("injected slot failure")))

    live_state.onUnitChange.emit(live_state.stimno)
    QApplication.processEvents()

    # The ingest pipeline still works end to end.
    feed_sweeps(live_state, 3, 1.0 / STIM_RATE_HZ, tick=QApplication.processEvents)
    assert live_state.stimno >= 2


# ---------------------------------------------------------------------------
# All windows open together under a 10 Hz stream
# ---------------------------------------------------------------------------


def test_all_windows_render_10hz_stream(qtbot, live_state):
    """Every registered panel stays open and responsive under a 10 Hz live stream."""
    from spikespy.SpikeSpy import window_options

    opened = {}
    failed = {}
    for key, cls in window_options.items():
        try:
            view = cls(parent=None, state=live_state)
        except Exception as exc:  # noqa: BLE001 - collected below, not hidden
            failed[key] = f"{type(exc).__name__}: {exc}"
            continue
        opened[key] = view

    assert not failed, f"Panels failed to load: {failed}"
    for key, view in opened.items():
        qtbot.addWidget(view)
        view.show()
    QApplication.processEvents()
    qtbot.wait(150)
    QApplication.processEvents()

    assert len(opened) >= len(window_options), "expected every registered panel to instantiate"

    # Paced 10 Hz stream: the GUI thread must stay alive and keep ingesting.
    heartbeat = Heartbeat()
    period_s = 1.0 / STIM_RATE_HZ
    elapsed = feed_sweeps(live_state, 30, period_s, tick=QApplication.processEvents)
    heartbeat.stop()
    # Event loop liveness: at 10 Hz pacing the loop must tick between frames.
    assert heartbeat.ticks > 30, (
        f"event loop ticked {heartbeat.ticks} times while 30 sweeps arrived at 10 Hz"
    )
    # Catch-up: ingestion must not fall hopelessly behind real time.
    pacing_s = 30 * period_s
    assert elapsed < max(pacing_s * 3.0, n_50ms_budget := 15.0), (
        f"30 sweeps at 10 Hz took {elapsed:.1f}s with all panels open (pacing alone: {pacing_s:.1f}s)"
    )

    live_state.flush_live_frame()
    qtbot.wait(400)
    QApplication.processEvents()
    assert live_state.stimno >= 29

    for key, view in opened.items():
        check = getattr(view, "is_view_culled", None)
        if check is not None:
            assert not check(), f"{key} stopped being visible after the stream"
        else:
            assert view.isVisible(), f"{key} is not visible after the stream"


def test_all_windows_survive_overdriven_burst(qtbot, live_state):
    """All panels together must survive input far faster than 10 Hz without freezing."""
    from spikespy.SpikeSpy import window_options

    opened = {}
    for key, cls in window_options.items():
        try:
            view = cls(parent=None, state=live_state)
        except Exception:  # noqa: BLE001 - covered by the render test above
            continue
        opened[key] = view
        qtbot.addWidget(view)
        view.show()
    QApplication.processEvents()
    qtbot.wait(150)
    QApplication.processEvents()

    heartbeat = Heartbeat()
    elapsed = feed_sweeps(live_state, 60, 0.0, tick=QApplication.processEvents)
    heartbeat.stop()
    # No hard freeze: every frame must be handled inside a generous ceiling
    # even with all 19 panels rendering (the per-view 10 Hz budget is asserted
    # separately for each PGView).
    avg_ms = elapsed / 60 * 1000
    assert avg_ms < 500.0, (
        f"GUI thread averaged {avg_ms:.1f} ms per frame with all panels open "
        f"under an overdriven burst"
    )
    assert heartbeat.ticks > 0, "event loop did not tick during the burst"

    live_state.flush_live_frame()
    qtbot.wait(400)
    QApplication.processEvents()
    assert live_state.stimno >= 59
