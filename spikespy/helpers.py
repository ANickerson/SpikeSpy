import time
from PySide6.QtCore import QObject, QTimer
from functools import wraps


def qsignal_throttle(func, interval=33, parent=None, weight="normal"):
    """Throttle Qt signal callbacks with latest-argument coalescing and adaptive strain backoff.

    Ensures at least 12ms of event-loop breathing room after each execution so
    heavy draws cannot starve the OS window message pump (preventing Windows "Not Responding").
    """
    throttleSignalTimer = QTimer(parent)
    throttleSignalTimer.setSingleShot(True)
    pending_call = {"args": (), "kwargs": {}, "has_pending": False, "last_elapsed_ms": 0.0}

    func_name = getattr(func, "__qualname__", getattr(func, "__name__", "throttled_fn"))
    bound_self = getattr(func, "__self__", None)

    def _resolve_interval_ms():
        base_ms = max(16, int(interval))
        state = getattr(bound_self, "state", None) if bound_self is not None else None
        scheduler = getattr(state, "frame_scheduler", None) if state is not None else None
        if scheduler is not None:
            base_ms = max(base_ms, scheduler.get_effective_interval_ms(func_name, weight=weight))
        last_ms = pending_call["last_elapsed_ms"]
        if last_ms > 0.0:
            adaptive_ms = int(max(last_ms + 12.0, last_ms * 2.0))
            base_ms = max(base_ms, min(400, adaptive_ms))
        return base_ms

    def _invoke():
        pending_call["has_pending"] = False
        if bound_self is not None and isinstance(bound_self, QObject):
            try:
                bound_self.objectName()
            except RuntimeError:
                return
        t0 = time.perf_counter()
        try:
            func(*pending_call["args"], **pending_call["kwargs"])
        except TypeError:
            func()
        except RuntimeError:
            return
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        pending_call["last_elapsed_ms"] = elapsed_ms

        state = getattr(bound_self, "state", None) if bound_self is not None else None
        scheduler = getattr(state, "frame_scheduler", None) if state is not None else None
        if scheduler is not None:
            scheduler.record_render_time(func_name, elapsed_ms, weight=weight)

        if pending_call["has_pending"]:
            try:
                throttleSignalTimer.start(_resolve_interval_ms())
            except RuntimeError:
                pass

    throttleSignalTimer.timeout.connect(_invoke)

    def start(*args, **kwargs):
        pending_call["args"] = args
        pending_call["kwargs"] = kwargs
        try:
            if not throttleSignalTimer.isActive():
                throttleSignalTimer.start(_resolve_interval_ms())
            else:
                pending_call["has_pending"] = True
        except RuntimeError:
            pass

    start.timer = throttleSignalTimer
    return start


def run_adaptive_render(view, render_fn, weight="normal", base_interval_ms=None):
    """Execute `render_fn`, record its render cost on `view.state.frame_scheduler`, and adjust `view.render_timer`."""
    view_name = type(view).__name__
    t0 = time.perf_counter()
    result = render_fn()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    state = getattr(view, "state", None)
    scheduler = getattr(state, "frame_scheduler", None) if state is not None else None
    if scheduler is not None:
        next_ms = scheduler.record_render_time(view_name, elapsed_ms, weight=weight)
    else:
        fallback_base = base_interval_ms or getattr(state, "ui_refresh_rate_ms", 33) or 33
        next_ms = int(min(400, max(fallback_base, elapsed_ms * 2.0, elapsed_ms + 12.0)))

    if base_interval_ms is not None:
        next_ms = max(int(base_interval_ms), next_ms)

    for timer_attr in ("render_timer", "update_timer"):
        timer = getattr(view, timer_attr, None)
        if timer is not None:
            try:
                if timer.isActive() and abs(timer.interval() - next_ms) >= 5:
                    timer.setInterval(next_ms)
            except RuntimeError:
                pass
    return result


def qsignal_throttle2(func, interval=33):
    """Throttle a function using a QTimer."""

    def wrapper(*args, **kwargs):
        # Ensure each instance gets its own QTimer
        if not hasattr(wrapper, "_throttle_timer"):
            wrapper._throttle_timer = QTimer()
            wrapper._throttle_timer.setInterval(interval)
            wrapper._throttle_timer.setSingleShot(True)
            wrapper._throttle_timer.timeout.connect(lambda: func(*args, **kwargs))
            func(*args, **kwargs)  # Call the function immediately

        elif not wrapper._throttle_timer.isActive():
            # func(*args, **kwargs)  # Call the function immediately
            wrapper._throttle_timer.start()

    return wrapper


def qsignal_throttle_wrapper(interval=33):
    """Decorator to throttle a function using qsignal_throttle."""

    def decorator(func):
        @wraps(func)
        def wrapped_func(*args, **kwargs):
            # Create a new throttled version of the function for each instance
            if not hasattr(wrapped_func, "_throttle_func"):
                print("new timer")
                wrapped_func._throttle_func = qsignal_throttle2(func, interval)
            return wrapped_func._throttle_func(*args, **kwargs)

        return wrapped_func

    return decorator
