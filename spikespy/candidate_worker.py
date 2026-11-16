"""Background scan manager for candidate-unit search.

The manager owns a small ``QThreadPool`` task per scan, keeps a monotonically
increasing generation so stale results are ignored, and exposes a synchronous
path so tests can run deterministically.  It is deliberately free of
``ViewerState`` imports: the state supplies a snapshot (plain numpy arrays and
metadata) and receives the proposals through ``on_result`` on the GUI thread.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, Signal

from .candidate_detection import detect_candidates

logger = logging.getLogger(__name__)


class _CandidateScanTask(QRunnable):
    def __init__(
        self, manager: "CandidateScanManager", generation: int, snapshot: Dict[str, Any]
    ):
        super().__init__()
        self.manager = manager
        self.generation = int(generation)
        self.snapshot = snapshot

    def run(self) -> None:
        snapshot = self.snapshot
        try:
            proposals = detect_candidates(
                snapshot["erp_by_channel"],
                snapshot["sampling_rate_hz"],
                snapshot["scan_start_stim"],
                snapshot["scan_end_stim"],
                existing_units=snapshot.get("existing_units"),
                candidate_memory=snapshot.get("candidate_memory"),
                config=snapshot.get("config"),
                peak_finder=snapshot.get("peak_finder"),
                waveform_units=snapshot.get("waveform_units"),
            )
            error = ""
        except Exception as exc:  # noqa: BLE001 - reported to the state
            logger.exception("Candidate scan failed")
            proposals = []
            error = f"{type(exc).__name__}: {exc}"
        self.manager._task_finished(self.generation, proposals, error)


class CandidateScanManager(QObject):
    """Runs candidate scans off the GUI thread with latest-generation wins."""

    resultReady = Signal(int, object, str)  # generation, proposals, error
    scanStateChanged = Signal(bool)

    def __init__(
        self,
        on_result: Callable[[int, Any, int, int, str], None],
        tick_cb: Optional[Callable[[], None]] = None,
        parent: Optional[QObject] = None,
        max_threads: int = 1,
    ):
        super().__init__(parent)
        self._on_result = on_result
        self._tick_cb = tick_cb
        self._generation = 0
        self._busy = False
        self._meta: Dict[int, Dict[str, Any]] = {}
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(max(1, int(max_threads)))
        self._timer = QTimer(self)
        self._timer.setSingleShot(False)
        self._timer.timeout.connect(self._on_tick)
        self.resultReady.connect(self._on_result_ready)

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def busy(self) -> bool:
        return self._busy

    def request(
        self,
        snapshot: Dict[str, Any],
        scope: str = "auto",
        synchronous: bool = False,
    ) -> Optional[int]:
        """Start one scan.  Returns the generation, or None when one is already running."""
        if self._busy and not synchronous:
            return None
        self._generation += 1
        generation = self._generation
        self._meta[generation] = {
            "scope": scope,
            "scan_start_stim": int(snapshot["scan_start_stim"]),
            "scan_end_stim": int(snapshot["scan_end_stim"]),
        }
        self._busy = True
        self.scanStateChanged.emit(True)
        if synchronous:
            try:
                proposals = detect_candidates(
                    snapshot["erp_by_channel"],
                    snapshot["sampling_rate_hz"],
                    snapshot["scan_start_stim"],
                    snapshot["scan_end_stim"],
                    existing_units=snapshot.get("existing_units"),
                    candidate_memory=snapshot.get("candidate_memory"),
                    config=snapshot.get("config"),
                    peak_finder=snapshot.get("peak_finder"),
                    waveform_units=snapshot.get("waveform_units"),
                )
                error = ""
            except Exception as exc:  # noqa: BLE001 - reported to the state
                logger.exception("Candidate scan failed")
                proposals = []
                error = f"{type(exc).__name__}: {exc}"
            self._finish(generation, proposals, error)
            return generation
        self._pool.start(_CandidateScanTask(self, generation, snapshot))
        return generation

    def start_auto(self, interval_ms: int) -> None:
        self._timer.setInterval(max(200, int(interval_ms)))
        if self._tick_cb is not None and not self._timer.isActive():
            self._timer.start()

    def stop_auto(self) -> None:
        self._timer.stop()

    def shutdown(self) -> None:
        self.stop_auto()
        self._generation += 1
        self._busy = False

    def _on_tick(self) -> None:
        if self._tick_cb is not None:
            try:
                self._tick_cb()
            except Exception:  # noqa: BLE001 - timer must not die
                logger.exception("Candidate auto-scan tick failed")

    def _task_finished(self, generation: int, proposals: Any, error: str) -> None:
        self.resultReady.emit(int(generation), proposals, str(error))

    def _on_result_ready(self, generation: int, proposals: Any, error: str) -> None:
        self._finish(generation, proposals, error)

    def _finish(self, generation: int, proposals: Any, error: str) -> None:
        self._busy = False
        self.scanStateChanged.emit(False)
        meta = self._meta.pop(int(generation), {})
        if int(generation) != self._generation:
            return
        try:
            self._on_result(
                int(generation),
                proposals,
                int(meta.get("scan_start_stim", 0)),
                int(meta.get("scan_end_stim", 0)),
                str(error),
            )
        except Exception:  # noqa: BLE001 - never break result delivery
            logger.exception("Candidate scan result callback failed")
