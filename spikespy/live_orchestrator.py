"""Live-stream processing orchestrator.

In a live session, new stimulus sweeps are ingested by
``ViewerState.append_sweep`` on the GUI thread.  Instead of every view
connecting to ``onNewSweep`` and running its own bespoke throttling /
processing pipeline, views *register* processing functions with the
``LiveOrchestrator`` owned by the ``ViewerState``:

    state.live_orchestrator.register(
        owner_id="trackingview_123",
        name="auto_track",
        fn=my_processing_fn,        # runs on the orchestrator worker thread
        on_result=my_callback,      # runs on the GUI thread with (result, error)
        batching=BatchingMode.BATCH,
    )

Every newly ingested sweep is fed to all registered functions with two
coalescing guarantees:

* LIFO (latest-wins): while a function is still busy, incoming sweeps pile up
  on its pending stack; when it becomes free, only the newest pending sweep is
  delivered and the stale ones are dropped.
* Batching: functions declared ``BatchingMode.BATCH`` receive every sweep that
  accumulated while they were busy as a single ``LiveSweepBatch`` whose
  waveforms are stacked into 2-D numpy arrays, so per-sweep work can be
  vectorized across the batch.
* Multithreading: registered functions run on a small pool of worker threads,
  so independent registrations process in parallel.  Each registration is
  still executed serially (at most one in-flight batch), which is what makes
  the LIFO/batching guarantees meaningful.  Real parallelism materializes for
  numpy/scipy-heavy functions that release the GIL.

Both live sweeps and static file loads flow through the same pipeline: a
dataset (re)load is submitted as a ``LiveReloadItem`` (``kind="reload"``),
and registrations subscribe to the event kinds they want (``sweeps`` and/or
``reload``), so views have a single point of data processing for live and
offline data alike.

The result of each call is marshalled back to the GUI thread and delivered to
``on_result``.  Bound methods are held via ``weakref.WeakMethod`` so destroyed
views are dropped automatically; views should still call ``unregister`` in
their teardown paths.
"""

from __future__ import annotations

import enum
import inspect
import logging
import threading
import weakref
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from PySide6.QtCore import QObject, Signal

logger = logging.getLogger(__name__)


class BatchingMode(enum.Enum):
    """How pending sweeps are delivered to a registered function."""

    LATEST = "latest"
    BATCH = "batch"


#: Event kinds a registration can subscribe to.
SWEEP = "sweep"
RELOAD = "reload"


def _qt_receiver_alive(obj: Any) -> bool:
    """False for Python wrappers of already-deleted C++ Qt objects."""
    try:
        from shiboken6 import isValid
        from PySide6.QtCore import QObject as _QObject

        if isinstance(obj, _QObject):
            return isValid(obj)
        return True
    except Exception:  # noqa: BLE001 - shiboken always present with PySide6
        return True


class LiveSweepItem:
    """One newly ingested stimulus sweep.

    Holds references (not copies) to the raw waveform arrays; they are never
    mutated after ``ViewerState.append_sweep`` stores them, so they are safe
    to read from the orchestrator worker thread.
    """

    __slots__ = (
        "stim_index",
        "stim_timestamp_s",
        "channel_sweeps",
        "spike_latencies_s",
        "pre_offset_s",
        "sampling_rate",
    )

    kind = SWEEP

    def __init__(
        self,
        stim_index: int,
        stim_timestamp_s: float,
        channel_sweeps: Dict[str, Any],
        spike_latencies_s: Optional[Dict[Any, Any]] = None,
        pre_offset_s: float = 0.0,
        sampling_rate: Optional[float] = None,
    ):
        self.stim_index = int(stim_index)
        self.stim_timestamp_s = float(stim_timestamp_s)
        self.channel_sweeps = dict(channel_sweeps or {})
        self.spike_latencies_s = dict(spike_latencies_s or {})
        self.pre_offset_s = float(pre_offset_s or 0.0)
        self.sampling_rate = float(sampling_rate) if sampling_rate else None

    def pre_idx(self) -> int:
        """Sample offset of the stimulus inside the raw waveform."""
        if self.pre_offset_s > 0 and self.sampling_rate:
            return int(round(self.pre_offset_s * self.sampling_rate))
        return 0

    def erp_row(self, channel: str, window_samples: Optional[int] = None) -> Optional[np.ndarray]:
        """Post-stimulus (ERP) row for a channel, clipped like the live ERP buffers."""
        wave = self.channel_sweeps.get(channel)
        if wave is None:
            return None
        row = np.asarray(getattr(wave, "magnitude", wave), dtype=np.float32).ravel()
        start = self.pre_idx()
        end = None if window_samples is None else start + int(window_samples)
        return row[start:end]


class LiveSweepBatch:
    """A batch of sweeps delivered to a registered function.

    ``items`` keeps the per-sweep records; the ``stacked``/``stim_indices``/...
    accessors materialize numpy arrays so batch-mode functions can vectorize.
    """

    def __init__(self, items: List[LiveSweepItem]):
        self.items: List[LiveSweepItem] = list(items)

    def __len__(self) -> int:
        return len(self.items)

    @property
    def n(self) -> int:
        return len(self.items)

    @property
    def latest_stim_index(self) -> Optional[int]:
        return self.items[-1].stim_index if self.items else None

    @property
    def stim_indices(self) -> np.ndarray:
        return np.asarray([it.stim_index for it in self.items], dtype=np.int64)

    @property
    def stim_timestamps(self) -> np.ndarray:
        return np.asarray([it.stim_timestamp_s for it in self.items], dtype=np.float64)

    def channels(self) -> List[str]:
        seen: List[str] = []
        for it in self.items:
            for ch in it.channel_sweeps:
                if ch not in seen:
                    seen.append(ch)
        return seen

    def rows(self, channel: str) -> List[np.ndarray]:
        """Raw waveforms (1-D float32) for one channel across the batch."""
        rows = []
        for it in self.items:
            wave = it.channel_sweeps.get(channel)
            if wave is None:
                continue
            rows.append(np.asarray(getattr(wave, "magnitude", wave), dtype=np.float32).ravel())
        return rows

    def erp_rows(self, channel: str, window_samples: Optional[int] = None) -> List[np.ndarray]:
        """ERP-clipped rows for one channel across the batch."""
        rows = []
        for it in self.items:
            row = it.erp_row(channel, window_samples)
            if row is not None:
                rows.append(row)
        return rows

    def stacked(self, rows: List[np.ndarray]) -> Optional[np.ndarray]:
        """Stack equal-length rows into an (n, samples) array; None if ragged."""
        if not rows:
            return None
        length = rows[0].shape[0]
        if any(r.shape[0] != length for r in rows):
            return None
        return np.stack(rows, axis=0)

    def stacked_rows(self, channel: str) -> Optional[np.ndarray]:
        return self.stacked(self.rows(channel))

    def stacked_erp_rows(self, channel: str, window_samples: Optional[int] = None) -> Optional[np.ndarray]:
        return self.stacked(self.erp_rows(channel, window_samples))


def latest_stim_index(batch: LiveSweepBatch) -> Optional[int]:
    """Default processor body: yield the newest stimulus index of the batch."""
    return batch.latest_stim_index


class LiveReloadItem:
    """Static dataset (re)load event, e.g. a file was opened or the window changed.

    Carries a short ``reason`` plus dataset shape; heavy data stays on the
    ``ViewerState`` and is read by the view's processor or GUI callback.
    """

    __slots__ = ("reason", "n_sweeps", "sampling_rate", "segment_name")

    kind = RELOAD

    def __init__(
        self,
        reason: str = "load",
        n_sweeps: int = 0,
        sampling_rate: Optional[float] = None,
        segment_name: str = "",
    ):
        self.reason = str(reason)
        self.n_sweeps = int(n_sweeps)
        self.sampling_rate = float(sampling_rate) if sampling_rate else None
        self.segment_name = str(segment_name)


def latest_reload_reason(batch: LiveSweepBatch) -> Optional[str]:
    """Default processor body for reload registrations: yield the reload reason."""
    return batch.items[-1].reason if batch.items else None


class _Registration:
    __slots__ = (
        "owner_id",
        "name",
        "batching",
        "events",
        "fn",
        "fn_weak",
        "on_result",
        "or_weak",
        "pending",
        "busy",
        "order",
    )

    def __init__(
        self,
        owner_id: str,
        name: str,
        fn: Callable,
        on_result: Optional[Callable],
        batching: BatchingMode,
        events,
        order: int,
    ):
        self.owner_id = owner_id
        self.name = name
        self.batching = batching
        self.events = frozenset(events or (SWEEP,))
        self.fn = None if inspect.ismethod(fn) else fn
        self.fn_weak = weakref.WeakMethod(fn) if inspect.ismethod(fn) else None
        self.on_result = None if (on_result is not None and inspect.ismethod(on_result)) else on_result
        self.or_weak = weakref.WeakMethod(on_result) if (on_result is not None and inspect.ismethod(on_result)) else None
        self.pending: List[Any] = []
        self.busy = False
        self.order = order

    def resolve_fn(self) -> Optional[Callable]:
        if self.fn_weak is not None:
            return self.fn_weak()
        return self.fn

    def resolve_on_result(self) -> Optional[Callable]:
        if self.or_weak is not None:
            return self.or_weak()
        return self.on_result

    @property
    def dead(self) -> bool:
        return self.resolve_fn() is None and self.fn_weak is not None


class LiveOrchestrator(QObject):
    """Runs view-registered functions on incoming live sweeps and file loads.

    Functions execute on a pool of background worker threads (one in-flight
    batch per registration, several registrations in parallel); results are
    delivered to ``on_result`` callbacks on the GUI thread via a queued
    signal.  ``submit(..., synchronous=True)`` bypasses the workers and runs
    the whole pipeline inline (used when ``live_frame_interval_ms <= 0``,
    e.g. in tests).
    """

    _resultReady = Signal(str, str, object, str)  # owner_id, name, result, error

    DEFAULT_MAX_WORKERS = 4

    def __init__(self, parent=None, max_workers: int = DEFAULT_MAX_WORKERS):
        super().__init__(parent)
        self._cond = threading.Condition()
        self._registrations: "OrderedDict[Tuple[str, str], _Registration]" = OrderedDict()
        self._stopped = False
        self._threads = []
        for i in range(max(1, int(max_workers))):
            t = threading.Thread(
                target=self._run, name=f"LiveOrchestrator-{i}", daemon=True
            )
            t.start()
            self._threads.append(t)
        self._resultReady.connect(self._dispatch_result)

    # ------------------------------------------------------------------
    # Registration API
    # ------------------------------------------------------------------

    def register(
        self,
        owner_id: str,
        name: str,
        fn: Callable[[LiveSweepBatch], Any],
        on_result: Optional[Callable[[Any, Optional[str]], None]] = None,
        batching: BatchingMode = BatchingMode.LATEST,
        events=None,
    ) -> Tuple[str, str]:
        """Register a processing function for new live data.

        ``fn(batch)`` runs on a worker thread.  With ``BatchingMode.LATEST``
        it receives a one-item batch with the newest pending sweep (stale
        sweeps are dropped); with ``BatchingMode.BATCH`` it receives all sweeps
        accumulated since its last call, stacked into numpy arrays.
        ``on_result(result, error)`` runs on the GUI thread.
        ``events`` selects which submitted items the function receives:
        ``(SWEEP,)`` (default), ``(RELOAD,)`` for static file loads, or both.
        Re-registering the same ``(owner_id, name)`` replaces the previous
        registration.
        """
        if not owner_id or not name:
            raise ValueError("owner_id and name are required")
        if not callable(fn):
            raise TypeError("fn must be callable")
        with self._cond:
            key = (str(owner_id), str(name))
            reg = _Registration(
                str(owner_id),
                str(name),
                fn,
                on_result,
                batching,
                events,
                len(self._registrations),
            )
            self._registrations[key] = reg
            self._cond.notify_all()
        return key

    def unregister(self, owner_id: str, name: Optional[str] = None) -> int:
        """Remove registrations for an owner (optionally a single name)."""
        removed = 0
        with self._cond:
            keys = [
                k
                for k in self._registrations
                if k[0] == owner_id and (name is None or k[1] == name)
            ]
            for k in keys:
                self._registrations.pop(k, None)
                removed += 1
        return removed

    def has_registration(self, owner_id: str, name: Optional[str] = None) -> bool:
        with self._cond:
            return any(
                k[0] == owner_id and (name is None or k[1] == name)
                for k in self._registrations
            )

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def submit(self, item: Any, synchronous: bool = False) -> None:
        """Feed one new sweep or reload event to the subscribed functions."""
        if synchronous:
            self._process_inline(item)
            return
        with self._cond:
            if self._stopped:
                return
            kind = getattr(item, "kind", SWEEP)
            for reg in self._registrations.values():
                if kind in reg.events:
                    reg.pending.append(item)
            self._cond.notify()

    def shutdown(self) -> None:
        """Stop the worker threads and drop all pending items."""
        with self._cond:
            self._stopped = True
            for reg in self._registrations.values():
                reg.pending.clear()
            self._cond.notify_all()
        for t in self._threads:
            t.join(timeout=2.0)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _drain_locked(self, reg: _Registration) -> List[LiveSweepItem]:
        """Pop the sweeps this registration should process next.

        LIFO: the newest pending sweep wins.  BATCH: everything accumulated.
        """
        if not reg.pending:
            return []
        if reg.batching is BatchingMode.BATCH:
            items = reg.pending
            reg.pending = []
            return items
        items = [reg.pending[-1]]
        reg.pending = []
        return items

    def _execute(self, reg: _Registration, items: List[LiveSweepItem], inline: bool):
        fn = reg.resolve_fn()
        if fn is None:
            self.unregister(reg.owner_id, reg.name)
            return
        batch = LiveSweepBatch(items)
        try:
            result = fn(batch)
            error = None
        except Exception as exc:  # noqa: BLE001 - reported to the view
            logger.exception("Live processor %s/%s failed", reg.owner_id, reg.name)
            result = None
            error = f"{type(exc).__name__}: {exc}"
        if inline:
            self._deliver(reg, result, error)
        else:
            self._resultReady.emit(reg.owner_id, reg.name, result, error)

    def _process_inline(self, item: Any) -> None:
        with self._cond:
            kind = getattr(item, "kind", SWEEP)
            jobs = []
            for reg in self._registrations.values():
                if kind in reg.events:
                    reg.pending.append(item)
            for reg in self._registrations.values():
                if kind in reg.events:
                    items = self._drain_locked(reg)
                    if items:
                        jobs.append((reg, items))
        for reg, items in jobs:
            self._execute(reg, items, inline=True)

    def _run(self) -> None:
        while True:
            with self._cond:
                reg = None
                while reg is None:
                    if self._stopped:
                        return
                    reg = next(
                        (
                            r
                            for r in self._registrations.values()
                            if r.pending and not r.busy
                        ),
                        None,
                    )
                    if reg is None:
                        self._cond.wait(timeout=0.25)
                items = self._drain_locked(reg)
                reg.busy = True
            try:
                self._execute(reg, items, inline=False)
            finally:
                with self._cond:
                    reg.busy = False
                    self._cond.notify_all()

    def _dispatch_result(self, owner_id: str, name: str, result: Any, error: str) -> None:
        with self._cond:
            reg = self._registrations.get((owner_id, name))
        if reg is None:
            return
        self._deliver(reg, result, error)

    def _deliver(self, reg: _Registration, result: Any, error: Optional[str]) -> None:
        on_result = reg.resolve_on_result()
        if on_result is None:
            if reg.or_weak is not None:
                # View was destroyed; drop the registration.
                self.unregister(reg.owner_id, reg.name)
            return
        receiver = getattr(on_result, "__self__", None)
        if receiver is not None and not _qt_receiver_alive(receiver):
            # Qt object (e.g. a channel bar) already deleted in C++.
            self.unregister(reg.owner_id, reg.name)
            return
        try:
            on_result(result, error)
        except Exception:  # noqa: BLE001 - never break the orchestrator
            logger.exception("on_result for %s/%s failed", reg.owner_id, reg.name)


class LiveFrameConsumerMixin:
    """Mixin for views that consume live sweeps and file loads via the orchestrator.

    Subclasses call ``_register_live_frames(state, on_frame)`` and/or
    ``_register_data_reload(state, on_reload)`` once their state is set.
    ``on_frame(stimno)`` is invoked on the GUI thread for the newest sweep
    (LIFO: frames that arrive while a previous one is still being processed
    are dropped in favour of the newest).  ``on_reload()`` is invoked on the
    GUI thread when a static dataset is (re)loaded, replacing direct
    ``onLoadNewFile`` connections so live and offline data share a single
    processing point.  Destroyed views are dropped automatically via weak
    references; call ``_unregister_live_frames`` in teardown for prompt
    cleanup.
    """

    def _register_live_frames(self, state, on_frame, batching: BatchingMode = BatchingMode.LATEST):
        orch = getattr(state, "live_orchestrator", None)
        if orch is None:
            return None
        self._live_frame_state = state
        self._live_frame_cb = on_frame
        self._live_frame_owner = f"{type(self).__name__}_{id(self)}"
        return orch.register(
            self._live_frame_owner,
            "live_frame",
            latest_stim_index,
            on_result=self._live_frame_result,
            batching=batching,
        )

    def _register_data_reload(self, state, on_reload, name: str = "dataset_loaded"):
        """Register a static dataset (re)load processor (replaces onLoadNewFile hooks)."""
        orch = getattr(state, "live_orchestrator", None)
        if orch is None:
            return None
        self._live_frame_state = state
        self._live_reload_cb = on_reload
        self._live_frame_owner = f"{type(self).__name__}_{id(self)}"
        return orch.register(
            self._live_frame_owner,
            name,
            latest_reload_reason,
            on_result=self._live_reload_result,
            events=(RELOAD,),
        )

    def _unregister_live_frames(self) -> None:
        state = getattr(self, "_live_frame_state", None)
        if state is not None:
            orch = getattr(state, "live_orchestrator", None)
            if orch is not None:
                orch.unregister(getattr(self, "_live_frame_owner", ""))
        self._live_frame_state = None
        self._live_frame_cb = None
        self._live_reload_cb = None

    def _live_frame_result(self, result, error) -> None:
        cb = getattr(self, "_live_frame_cb", None)
        if cb is None or error or result is None:
            return
        try:
            cb(int(result))
        except Exception:  # noqa: BLE001 - view callbacks must not kill delivery
            logger.exception("live frame callback failed for %s", type(self).__name__)

    def _live_reload_result(self, result, error) -> None:
        if error:
            return
        cb = getattr(self, "_live_reload_cb", None)
        if cb is None:
            return
        try:
            cb()
        except Exception:  # noqa: BLE001 - view callbacks must not kill delivery
            logger.exception("live reload callback failed for %s", type(self).__name__)
