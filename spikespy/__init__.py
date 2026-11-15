import sys
import importlib

# Aliases for legacy Matplotlib modules moved to spikespy.legacy_mpl
_legacy_module_names = [
    "MultiTraceView",
    "SingleTraceView",
    "MultiTraceFixedView",
    "UnitView",
    "UnitFootprintView",
    "EventView",
    "QMatplotlib",
    "CachedAxesImage",
]

for _name in _legacy_module_names:
    _mod = importlib.import_module(f".legacy_mpl.{_name}", package=__name__)
    sys.modules[f"{__name__}.{_name}"] = _mod
    globals()[_name] = _mod

from .SpikeSpy import run as spikespy
from .SpikeSpy import export_csv, register_plugin
from .dsp.detectors import (
    DetectorParamSpec,
    SpikeDetectorPlugin,
    SpikeDetectorRegistry,
    register_spike_detector,
)

__all__ = [
    "spikespy",
    "export_csv",
    "register_plugin",
    "DetectorParamSpec",
    "SpikeDetectorPlugin",
    "SpikeDetectorRegistry",
    "register_spike_detector",
    "MultiTraceView",
    "SingleTraceView",
    "MultiTraceFixedView",
    "UnitView",
    "UnitFootprintView",
    "EventView",
    "QMatplotlib",
    "CachedAxesImage",
]
