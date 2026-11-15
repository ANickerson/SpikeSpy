import logging
from PySide6.QtCore import QTimer


def qsignal_throttle(func, interval=33):
    # connect to signals
    throttleSignalTimer = QTimer()
    throttleSignalTimer.setInterval(interval)  # 30 fps
    throttleSignalTimer.timeout.connect(func)
    throttleSignalTimer.setSingleShot(True)

    def start(*args, **kwargs):
        if not throttleSignalTimer.isActive():
            throttleSignalTimer.start()

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


from PySide6.QtCore import QTimer
from functools import wraps


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
