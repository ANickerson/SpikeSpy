"""Tests for the live orchestrator: LIFO coalescing, numpy batching, and view integration."""

from __future__ import annotations

import threading

import neo
import numpy as np
import pytest
import quantities as pq

from spikespy.ViewerState import ViewerState, tracked_neuron_unit
from spikespy.live_orchestrator import (
    BatchingMode,
    LiveOrchestrator,
    LiveReloadItem,
    LiveSweepItem,
    RELOAD,
    SWEEP,
)


@pytest.fixture()
def live_state():
    state = ViewerState()
    state.start_live_session(
        sampling_rate=10000.0, window_size_ms=100.0, units=pq.uV, channel_name="CH0"
    )
    yield state
    state.stop_live_session()


def _item(stim_index, n_samples=100, channel="CH0", amplitude=1.0):
    return LiveSweepItem(
        stim_index=stim_index,
        stim_timestamp_s=stim_index * 0.1,
        channel_sweeps={channel: np.full(n_samples, amplitude, dtype=np.float32)},
        sampling_rate=10000.0,
    )


def test_synchronous_latest_dispatch(qtbot):
    orch = LiveOrchestrator()
    try:
        results = []
        orch.register(
            "owner_a",
            "frame",
            lambda batch: batch.latest_stim_index,
            on_result=lambda res, err: results.append((res, err)),
            batching=BatchingMode.LATEST,
        )
        # Synchronous submissions run the whole pipeline inline.
        orch.submit(_item(0), synchronous=True)
        orch.submit(_item(1), synchronous=True)
        assert [r for r, _ in results] == [0, 1]
        assert all(err is None for _, err in results)
    finally:
        orch.shutdown()


def test_batch_numpy_stacking(qtbot):
    from spikespy.live_orchestrator import LiveSweepBatch

    batch = LiveSweepBatch(
        [
            _item(0, amplitude=3.0),
            _item(1, amplitude=5.0),
            _item(2, amplitude=7.0),
        ]
    )
    stacked = batch.stacked_rows("CH0")
    assert stacked.shape == (3, 100)
    # Batch-mode functions can vectorize per-sweep stats across the batch.
    means = np.mean(stacked, axis=1)
    assert means.tolist() == [3.0, 5.0, 7.0]
    assert batch.stim_indices.tolist() == [0, 1, 2]
    assert batch.latest_stim_index == 2

    # ERP rows clip pre-stimulus offset and window like the live buffers.
    item = LiveSweepItem(
        stim_index=3,
        stim_timestamp_s=0.3,
        channel_sweeps={"CH0": np.arange(1000, dtype=np.float32)},
        pre_offset_s=0.01,
        sampling_rate=10000.0,
    )
    row = item.erp_row("CH0", window_samples=500)
    assert row.shape == (500,)
    assert row[0] == 100.0  # starts at the stimulus sample


def test_async_lifo_drops_stale(qtbot):
    orch = LiveOrchestrator()
    try:
        gate = threading.Event()
        entered = threading.Event()
        calls = []
        results = []

        def fn(batch):
            entered.set()
            if not calls:
                gate.wait(timeout=5.0)
            calls.append(batch.stim_indices.tolist())
            return batch.latest_stim_index

        orch.register(
            "owner_lifo",
            "frame",
            fn,
            on_result=lambda res, err: results.append(res),
            batching=BatchingMode.LATEST,
        )

        orch.submit(_item(0))
        qtbot.waitUntil(entered.is_set, timeout=5000)
        # These pile up while the worker is held inside the first call.
        for i in range(1, 10):
            orch.submit(_item(i))
        gate.set()

        qtbot.waitUntil(lambda: len(results) >= 2, timeout=5000)

        # LIFO: only the newest pending sweep is processed; stale ones dropped.
        assert results[-1] == 9
        assert len(calls) == 2
        assert calls[0] == [0]
        assert calls[1] == [9]
    finally:
        gate.set()
        orch.shutdown()


def test_async_batch_concatenates(qtbot):
    orch = LiveOrchestrator()
    try:
        gate = threading.Event()
        entered = threading.Event()
        calls = []

        def fn(batch):
            entered.set()
            if not calls:
                gate.wait(timeout=5.0)
            calls.append((batch.n, batch.stim_indices.tolist()))
            return batch.n

        orch.register(
            "owner_batch",
            "stats",
            fn,
            on_result=lambda res, err: None,
            batching=BatchingMode.BATCH,
        )

        orch.submit(_item(0))
        qtbot.waitUntil(entered.is_set, timeout=5000)
        for i in range(1, 6):
            orch.submit(_item(i))
        gate.set()

        qtbot.waitUntil(lambda: len(calls) == 2, timeout=5000)

        # Everything accumulated while busy arrives in one numpy-friendly batch.
        n, stim_idx = calls[1]
        assert n == 5
        assert stim_idx == [1, 2, 3, 4, 5]
    finally:
        gate.set()
        orch.shutdown()


def test_unregister_stops_delivery(qtbot):
    orch = LiveOrchestrator()
    try:
        results = []
        orch.register(
            "owner_gone",
            "frame",
            lambda batch: batch.latest_stim_index,
            on_result=lambda res, err: results.append(res),
        )
        assert orch.unregister("owner_gone") == 1
        orch.submit(_item(0), synchronous=True)
        orch.submit(_item(1), synchronous=True)
        assert results == []
    finally:
        orch.shutdown()


def test_ragged_rows_do_not_stack(qtbot):
    from spikespy.live_orchestrator import LiveSweepBatch

    batch = LiveSweepBatch([_item(0, n_samples=10), _item(1, n_samples=20)])
    assert batch.stacked_rows("CH0") is None
    assert len(batch.rows("CH0")) == 2


def test_workers_run_registrations_in_parallel(qtbot):
    """Independent registrations execute concurrently on the worker pool."""
    orch = LiveOrchestrator(max_workers=2)
    try:
        # Both functions must arrive at the barrier; if the orchestrator
        # serialized registrations this would time out and raise.
        barrier = threading.Barrier(2, timeout=5.0)
        done = []

        def make_fn(tag):
            def fn(batch):
                barrier.wait()
                done.append(tag)
                return tag

            return fn

        orch.register("par_a", "frame", make_fn("a"), on_result=lambda r, e: None)
        orch.register("par_b", "frame", make_fn("b"), on_result=lambda r, e: None)

        orch.submit(_item(0))
        qtbot.waitUntil(lambda: len(done) == 2, timeout=6000)
        assert sorted(done) == ["a", "b"]
    finally:
        orch.shutdown()


def test_reload_events_are_routed_separately(qtbot):
    """Sweep registrations don't see reload events and vice versa."""
    orch = LiveOrchestrator()
    try:
        received = {"sweeps": [], "reloads": []}
        orch.register(
            "sweep_owner",
            "frame",
            lambda batch: received["sweeps"].append(batch.latest_stim_index),
            on_result=lambda r, e: None,
            batching=BatchingMode.BATCH,
            events=(SWEEP,),
        )
        orch.register(
            "reload_owner",
            "file",
            lambda batch: received["reloads"].append(batch.items[-1].reason),
            on_result=lambda r, e: None,
            events=(RELOAD,),
        )

        orch.submit(_item(0), synchronous=True)
        orch.submit(LiveReloadItem(reason="load", n_sweeps=5), synchronous=True)
        orch.submit(_item(1), synchronous=True)

        assert received["sweeps"] == [0, 1]
        assert received["reloads"] == ["load"]
    finally:
        orch.shutdown()


def test_view_reload_registration_processes_static_load(qtbot):
    """Views registered through the mixin receive file/window reloads through the orchestrator."""
    from spikespy.SingleTraceView import SingleTraceView

    state = ViewerState()
    view = SingleTraceView(state=state)
    qtbot.addWidget(view)
    view._dirty = False

    # A window-size change is a static dataset reload (ERP cache cleared).
    state.set_window_size(100)
    assert view._dirty is True


def test_viewerstate_append_sweep_feeds_orchestrator(qtbot, live_state):
    received = []
    live_state.live_orchestrator.register(
        "recorder",
        "sweeps",
        lambda batch: received.append(batch) or len(batch),
        on_result=lambda res, err: None,
        batching=BatchingMode.BATCH,
    )

    wave = np.full(1000, 7.0, dtype=np.float32)
    aux = {"CH1": np.full(1000, 3.0, dtype=np.float32)}
    idx = live_state.append_sweep(
        stim_timestamp=1.0,
        sweep_waveform=wave,
        spike_latencies_s={0: 0.012},
        aux_sweeps=aux,
    )
    assert idx == 0
    # live_frame_interval_ms is 0 in tests, so submission was synchronous.
    assert len(received) == 1
    batch = received[0]
    assert batch.stim_indices.tolist() == [0]
    item = batch.items[0]
    assert set(item.channel_sweeps.keys()) == {"CH0", "CH1"}
    assert item.spike_latencies_s == {0: 0.012}
    assert item.sampling_rate == 10000.0
    # ERP row is clipped like the live ERP buffers.
    assert item.erp_row("CH0").shape == (1000,)


def test_trackingview_batched_detection(qtbot, live_state):
    from spikespy.TrackingView import TrackingView

    evt = neo.Event(np.array([], dtype=float) * pq.s, name="unit_0")
    sg = tracked_neuron_unit(
        event=evt,
        channel="CH0",
        auto_track=True,
        tracking_window_ms=5.0,
        tracking_threshold_mv=0.01,
        tracking_max_skips=1,
        idx_arr=[],
    )
    sg.event.annotations["auto_track"] = True
    live_state.spike_groups = [sg]
    live_state.cur_spike_group = 0
    live_state.update_idx_arrs()

    view = TrackingView(state=live_state)
    qtbot.addWidget(view)

    # First sweep: no previous latency, so seed the unit manually.
    w0 = np.zeros(1000, dtype=np.float32)
    w0[500] = 50.0
    live_state.append_sweep(0.0, w0)
    live_state.setUnit(500, stimno=0, unit_idx=0)
    assert sg.idx_arr[0] == (500, 1)

    # Second sweep: the orchestrator's batched processor must detect the spike
    # off the GUI thread and apply it via the GUI-side result callback.
    w1 = np.zeros(1000, dtype=np.float32)
    w1[505] = 45.0
    live_state.append_sweep(0.1, w1)

    detected = sg.get_sweep_sample(1, live_state.event_signal, 10000.0, live_state.window_size)
    assert detected is not None
    assert 490 <= detected <= 520
