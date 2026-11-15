import logging
import sys
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional

from qtpy import QtCore

logger = logging.getLogger(__name__)


@dataclass
class StallReport:
    """Record of a main-thread event loop stall detected by UIWatchdog."""
    timestamp: float
    duration_ms: float
    stack_trace: str
    active_views: List[str] = field(default_factory=list)


class UIWatchdog(QtCore.QObject):
    """Detects main-thread Qt event loop freezes and captures Python call stacks.

    Uses a lightweight QTimer on the main GUI thread to update a monotonic
    heartbeat timestamp, and a daemon background thread that checks whether the
    heartbeat has stalled beyond `stall_threshold_ms`. When a stall is detected,
    it dumps the main thread's Python stack via `sys._current_frames()` so
    "Not Responding" freezes on Windows can be diagnosed immediately without
    attaching an external debugger.
    """

    stallDetected = QtCore.Signal(object)  # emits StallReport

    def __init__(
        self,
        parent: Optional[QtCore.QObject] = None,
        heartbeat_interval_ms: int = 100,
        stall_threshold_ms: float = 750.0,
        max_reports: int = 25,
        log_to_stderr: bool = True,
    ):
        super().__init__(parent)
        self.heartbeat_interval_ms = max(20, int(heartbeat_interval_ms))
        self.stall_threshold_ms = max(100.0, float(stall_threshold_ms))
        self.max_reports = max_reports
        self.log_to_stderr = log_to_stderr

        self._main_thread_id: int = threading.main_thread().ident or threading.get_ident()
        self._last_heartbeat: float = time.monotonic()
        self._last_loop_lag_ms: float = 0.0
        self._ema_loop_lag_ms: float = 0.0
        self._max_loop_lag_ms: float = 0.0
        self._running: bool = False
        self._in_stall: bool = False
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._reports: Deque[StallReport] = deque(maxlen=max_reports)
        self._lock = threading.Lock()

        self._heartbeat_timer = QtCore.QTimer(self)
        self._heartbeat_timer.setInterval(self.heartbeat_interval_ms)
        self._heartbeat_timer.timeout.connect(self._on_heartbeat)

    @property
    def last_loop_lag_ms(self) -> float:
        return self._last_loop_lag_ms

    @property
    def ema_loop_lag_ms(self) -> float:
        return self._ema_loop_lag_ms

    @property
    def max_loop_lag_ms(self) -> float:
        return self._max_loop_lag_ms

    @property
    def reports(self) -> List[StallReport]:
        with self._lock:
            return list(self._reports)

    def clear_reports(self) -> None:
        with self._lock:
            self._reports.clear()
            self._max_loop_lag_ms = 0.0

    def start(self) -> None:
        if self._running:
            return
        self._main_thread_id = threading.get_ident()
        self._last_heartbeat = time.monotonic()
        self._in_stall = False
        self._stop_event.clear()
        self._running = True
        self._heartbeat_timer.start(self.heartbeat_interval_ms)

        self._thread = threading.Thread(
            target=self._watchdog_loop,
            name="SpikeSpy-UIWatchdog",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self._stop_event.set()
        try:
            if self._heartbeat_timer.isActive():
                self._heartbeat_timer.stop()
        except RuntimeError:
            pass
        if self._thread is not None and self._thread.is_alive():
            if threading.get_ident() != self._thread.ident:
                self._thread.join(timeout=0.5)
        self._thread = None

    def _on_heartbeat(self) -> None:
        now = time.monotonic()
        elapsed_ms = (now - self._last_heartbeat) * 1000.0
        expected_ms = float(self.heartbeat_interval_ms)
        lag_ms = max(0.0, elapsed_ms - expected_ms)

        self._last_heartbeat = now
        self._last_loop_lag_ms = lag_ms
        self._ema_loop_lag_ms = 0.25 * lag_ms + 0.75 * self._ema_loop_lag_ms
        if lag_ms > self._max_loop_lag_ms:
            self._max_loop_lag_ms = lag_ms
        self._in_stall = False

    def capture_main_thread_stack(self) -> str:
        """Capture the current Python stack trace of the main GUI thread."""
        frames = sys._current_frames()
        frame = frames.get(self._main_thread_id)
        if frame is None:
            return "<Main thread frame unavailable>"
        return "".join(traceback.format_stack(frame))

    def _watchdog_loop(self) -> None:
        check_interval_s = max(0.03, min(0.1, self.stall_threshold_ms / 3000.0))
        while not self._stop_event.wait(check_interval_s):
            if not self._running:
                break
            now = time.monotonic()
            elapsed_ms = (now - self._last_heartbeat) * 1000.0
            if elapsed_ms >= self.stall_threshold_ms and not self._in_stall:
                self._in_stall = True
                stack_str = self.capture_main_thread_stack()
                report = StallReport(
                    timestamp=time.time(),
                    duration_ms=elapsed_ms,
                    stack_trace=stack_str,
                )
                with self._lock:
                    self._reports.append(report)

                msg = (
                    f"\n[UI WATCHDOG] Main GUI thread stall detected "
                    f"({elapsed_ms:.0f} ms >= {self.stall_threshold_ms:.0f} ms)!\n"
                    f"Main thread call stack:\n{stack_str}"
                )
                logger.warning(msg)
                if self.log_to_stderr:
                    try:
                        sys.stderr.write(msg + "\n")
                        sys.stderr.flush()
                    except Exception:
                        pass
                try:
                    self.stallDetected.emit(report)
                except RuntimeError:
                    break


class AdaptiveFrameScheduler(QtCore.QObject):
    """Monitors per-view render costs and event-loop strain to dynamically throttle heavy views.

    When multiple windows are open or high-frequency streaming (e.g., 50 Hz) pushes
    the main thread toward saturation, this scheduler increases the effective refresh
    interval for normal and heavy views so the OS window message pump always has idle
    breathing room (preventing Windows "Not Responding" freezes).
    """

    strainChanged = QtCore.Signal(float, int)  # (strain_level, effective_interval_ms)

    # View weight classes:
    # - "primary": fast primary trace views (1.0x multiplier under moderate load)
    # - "normal": secondary views (1.5x multiplier under load)
    # - "heavy": expensive views like Matplotlib canvases, footprints, tables (2.5x multiplier under load)
    WEIGHT_MULTIPLIERS = {
        "primary": 1.0,
        "normal": 1.5,
        "heavy": 2.5,
    }

    def __init__(
        self,
        parent: Optional[QtCore.QObject] = None,
        base_interval_ms: int = 33,
        min_idle_gap_ms: int = 12,
        max_interval_ms: int = 400,
    ):
        super().__init__(parent)
        self.base_interval_ms = max(16, int(base_interval_ms))
        self.min_idle_gap_ms = max(5, int(min_idle_gap_ms))
        self.max_interval_ms = max(100, int(max_interval_ms))

        self._view_ema_ms: Dict[str, float] = {}
        self._view_last_ms: Dict[str, float] = {}
        self._view_counts: Dict[str, int] = {}
        self._view_weights: Dict[str, str] = {}

        self._frame_render_sum_ms: float = 0.0
        self._last_frame_Rollup_ts: float = time.monotonic()
        self._ema_total_render_ms: float = 0.0
        self._external_loop_lag_ms: float = 0.0
        self._strain_level: float = 0.0

    @property
    def strain_level(self) -> float:
        """Current strain level: 0.0 = normal, 1.0 = moderate, 2.0+ = heavy strain."""
        return self._strain_level

    @property
    def ema_total_render_ms(self) -> float:
        return self._ema_total_render_ms

    def set_base_interval_ms(self, interval_ms: int) -> None:
        self.base_interval_ms = max(16, int(interval_ms))
        self._recompute_strain()

    def update_loop_lag(self, lag_ms: float) -> None:
        self._external_loop_lag_ms = max(0.0, float(lag_ms))
        self._recompute_strain()

    def record_render_time(
        self,
        view_name: str,
        elapsed_ms: float,
        weight: str = "normal",
    ) -> int:
        """Record a view's render duration in milliseconds and return its next recommended interval (ms)."""
        elapsed_ms = max(0.0, float(elapsed_ms))
        prev_ema = self._view_ema_ms.get(view_name, elapsed_ms)
        self._view_ema_ms[view_name] = 0.3 * elapsed_ms + 0.7 * prev_ema
        self._view_last_ms[view_name] = elapsed_ms
        self._view_counts[view_name] = self._view_counts.get(view_name, 0) + 1
        self._view_weights[view_name] = weight

        now = time.monotonic()
        window_s = max(0.001, float(self.base_interval_ms) / 1000.0)
        if (now - self._last_frame_Rollup_ts) >= window_s:
            self._ema_total_render_ms = (
                0.35 * self._frame_render_sum_ms + 0.65 * self._ema_total_render_ms
            )
            self._frame_render_sum_ms = elapsed_ms
            self._last_frame_Rollup_ts = now
        else:
            self._frame_render_sum_ms += elapsed_ms
            if self._frame_render_sum_ms > self._ema_total_render_ms:
                self._ema_total_render_ms = (
                    0.5 * self._frame_render_sum_ms + 0.5 * self._ema_total_render_ms
                )

        self._recompute_strain()
        return self.get_effective_interval_ms(view_name=view_name, weight=weight)

    def _recompute_strain(self) -> None:
        budget_ms = max(16.0, float(self.base_interval_ms) * 0.75)
        render_ratio = self._ema_total_render_ms / budget_ms
        lag_ratio = self._external_loop_lag_ms / max(20.0, float(self.base_interval_ms))

        raw_strain = max(0.0, max(render_ratio, lag_ratio) - 0.7)
        prev = self._strain_level
        # Fast ramp-up on strain, gradual decay when recovering
        if raw_strain > prev:
            self._strain_level = min(5.0, 0.5 * raw_strain + 0.5 * prev)
        else:
            self._strain_level = max(0.0, 0.2 * raw_strain + 0.8 * prev)
            if self._strain_level < 0.05:
                self._strain_level = 0.0

        if abs(self._strain_level - prev) >= 0.25:
            try:
                self.strainChanged.emit(
                    self._strain_level,
                    self.get_effective_interval_ms(weight="normal"),
                )
            except RuntimeError:
                pass

    def get_effective_interval_ms(
        self,
        view_name: Optional[str] = None,
        weight: str = "normal",
    ) -> int:
        """Compute the adaptive timer interval (ms) for a given view and weight class."""
        if view_name and view_name in self._view_weights:
            weight = self._view_weights[view_name]

        mult = self.WEIGHT_MULTIPLIERS.get(weight, 1.5)
        scale = 1.0 + (self._strain_level * mult)
        target_ms = float(self.base_interval_ms) * scale

        if view_name and view_name in self._view_ema_ms:
            view_cost = self._view_ema_ms[view_name]
            # Guarantee at least `min_idle_gap_ms` of idle breathing room
            # and keep duty cycle of any single view <= 50% (or <= 30% for heavy views)
            duty_factor = 3.0 if weight == "heavy" else 2.0
            min_safe_ms = max(
                view_cost + self.min_idle_gap_ms,
                view_cost * duty_factor,
            )
            target_ms = max(target_ms, min_safe_ms)

        return int(min(self.max_interval_ms, max(self.base_interval_ms, round(target_ms))))

    def should_skip_nonessential_work(self) -> bool:
        """Return True when the UI is under significant strain and optional redraws should be deferred."""
        return self._strain_level >= 0.8

    def get_diagnostics_summary(self) -> Dict[str, object]:
        """Return a snapshot of per-view render timings and current strain metrics."""
        views = {}
        for name, ema in sorted(
            self._view_ema_ms.items(), key=lambda kv: kv[1], reverse=True
        ):
            views[name] = {
                "ema_ms": round(ema, 2),
                "last_ms": round(self._view_last_ms.get(name, 0.0), 2),
                "count": self._view_counts.get(name, 0),
                "weight": self._view_weights.get(name, "normal"),
                "effective_interval_ms": self.get_effective_interval_ms(name),
            }
        return {
            "base_interval_ms": self.base_interval_ms,
            "strain_level": round(self._strain_level, 2),
            "ema_total_render_ms": round(self._ema_total_render_ms, 2),
            "external_loop_lag_ms": round(self._external_loop_lag_ms, 2),
            "views": views,
        }

    def reset_metrics(self) -> None:
        self._view_ema_ms.clear()
        self._view_last_ms.clear()
        self._view_counts.clear()
        self._frame_render_sum_ms = 0.0
        self._ema_total_render_ms = 0.0
        self._external_loop_lag_ms = 0.0
        self._strain_level = 0.0
