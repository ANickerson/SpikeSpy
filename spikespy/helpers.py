import logging
import time
from functools import wraps
from PySide6.QtCore import QObject, QTimer

logger = logging.getLogger(__name__)


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


def quiet_fail(func):
    """Run a data-driven UI callback, logging instead of propagating failures.

    Live streaming drives view update callbacks many times per second; an
    error raised inside one frame must be logged and dropped, never
    propagated into the Qt event loop where it can abort the whole
    application.
    """
    log = logging.getLogger(func.__module__)

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            log.exception("Suppressed error in %s", func.__qualname__)
            return None

    return wrapper


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


def apply_top_bars_visibility(widget, hide: bool):
    """
    Shows or hides top toolbars on a view or channel group container.
    When hide is True, all QToolBar instances on the view are hidden so
    controls are managed exclusively from the properties pane.
    When hide is False, toolbars are restored (respecting channel bar deferral).
    Does NOT hide the main application toolbar.
    """
    if widget is None:
        return
    from PySide6.QtWidgets import QToolBar

    # If widget is the main application window (MdiView), delegate only to its docked views
    if widget.__class__.__name__ == "MdiView" or (
        hasattr(widget, "dock_manager") and hasattr(widget, "settings_panel")
    ):
        if hasattr(widget, "dock_manager") and widget.dock_manager is not None:
            for dw in widget.dock_manager.dockWidgets():
                w = dw.widget()
                if w is not None and w is not getattr(widget, "settings_panel", None):
                    apply_top_bars_visibility(w, hide)
        return

    # If widget is a ChannelGroupContainerView
    if widget.__class__.__name__ == "ChannelGroupContainerView":
        if hasattr(widget, "channel_bar") and widget.channel_bar is not None:
            widget.channel_bar.setVisible(not hide)
        if hasattr(widget, "control_bar") and widget.control_bar is not None:
            widget.control_bar.setVisible(not hide)
        if hasattr(widget, "sub_views"):
            for sv in widget.sub_views:
                apply_top_bars_visibility(sv, hide)
        return

    # Check if widget is nested within a ChannelGroupContainerView
    is_inside_container = False
    curr = widget.parentWidget() if hasattr(widget, "parentWidget") else None
    while curr is not None:
        if curr.__class__.__name__ == "ChannelGroupContainerView":
            is_inside_container = True
            break
        curr = curr.parentWidget() if hasattr(curr, "parentWidget") else None

    # Handle all QToolBar children of the view
    try:
        toolbars = widget.findChildren(QToolBar)
    except Exception:
        toolbars = []

    for tb in toolbars:
        # Never hide application-level toolbars
        win = tb.window()
        if win is not None and win.__class__.__name__ == "MdiView" and tb.parent() is win:
            continue
        # If inside a channel group container, child view's channel_bar was deferred and should stay hidden
        if is_inside_container and hasattr(widget, "channel_bar") and tb is widget.channel_bar:
            tb.setVisible(False)
        else:
            tb.setVisible(not hide)


def notify_view_selected(view_or_child, ensure_panel_visible: bool = True):
    """Notifies the main MdiView's ViewSettingsPanel that a view (or one of its child widgets)
    has been clicked or focused, and ensures the View Settings dock panel is opened/visible.
    """
    if view_or_child is None:
        return
    from PySide6.QtWidgets import QApplication, QWidget

    if not isinstance(view_or_child, QWidget):
        return

    # Find MdiView
    mdi = None
    curr = view_or_child
    while curr is not None:
        if curr.__class__.__name__ == "MdiView" or (
            hasattr(curr, "settings_panel") and hasattr(curr, "dock_manager")
        ):
            mdi = curr
            break
        curr = curr.parentWidget() if hasattr(curr, "parentWidget") else None
    if mdi is None:
        win = view_or_child.window()
        if win is not None and (
            win.__class__.__name__ == "MdiView" or hasattr(win, "settings_panel")
        ):
            mdi = win
    if mdi is None:
        app = QApplication.instance()
        if app is not None:
            for top_w in app.topLevelWidgets():
                if top_w.__class__.__name__ == "MdiView" or hasattr(top_w, "settings_panel"):
                    mdi = top_w
                    break

    if mdi is not None and hasattr(mdi, "settings_panel") and mdi.settings_panel is not None:
        try:
            resolved = mdi.settings_panel._resolve_view(view_or_child)
            if mdi.settings_panel._is_configurable_view(resolved):
                mdi.settings_panel.setActiveView(resolved)
                if ensure_panel_visible and hasattr(mdi, "open_or_show_view_settings_panel"):
                    mdi.open_or_show_view_settings_panel()
        except RuntimeError:
            pass
