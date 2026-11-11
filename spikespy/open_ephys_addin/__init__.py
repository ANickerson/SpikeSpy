"""Open Ephys Real-Time Add-In for SpikeSpy.

Provides real-time streaming, ERP slicing, and closed-loop bidirectional control
with Open Ephys GUI v1.0+.
"""

from .oe_controller import OpenEphysController
from .oe_live_view import OpenEphysLiveView
from .oe_stream_worker import OpenEphysStreamWorker


def register_open_ephys_addin(window_options: dict) -> None:
    """Register OpenEphysLiveView into SpikeSpy window_options dictionary."""
    window_options["OpenEphysLive"] = OpenEphysLiveView


__all__ = [
    "OpenEphysController",
    "OpenEphysStreamWorker",
    "OpenEphysLiveView",
    "register_open_ephys_addin",
]
