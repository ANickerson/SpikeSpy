from __future__ import annotations

from typing import Any, Dict, Optional
import numpy as np
import quantities as pq
from neo import Event

from .ViewerState import ViewerState
from .dsp.detectors import SpikeDetectorRegistry


class UnitSuggestor:
    update_on_unit_change = True
    update_on_unit_group_change = True

    def __init__(self, state=None, settings=None):
        self._suggestions = None
        self.settings = settings or {}
        self.state: ViewerState = None

        if state is not None:
            self.setState(state)
        if settings is not None:
            self.set_settings(settings)

    def setState(self, state):
        self.state: ViewerState = state
        if self.state is None:
            return
        if self.update_on_unit_change:
            self.state.onUnitChange.connect(self.updateSuggestions)
        self.state.onUnitGroupChange.connect(self.updateSuggestions)
        self.state.onLoadNewFile.connect(self.updateSuggestions)

    def updateSuggestions(self):
        self._suggestions = self.suggest()

    def get_predictions(self):
        return self._suggestions

    def settings_dialog(self):
        pass

    def set_settings(self, settings):
        self.settings = settings or {}

    def get_settings(self):
        return {**(self.settings or {})}

    def suggest(self):
        pass


class threshold_suggestor(UnitSuggestor):
    update_on_unit_change = False
    update_on_unit_group_change = False

    def get_settings(self):
        settings = super().get_settings()
        settings.setdefault("threshold", None)
        settings.setdefault("algorithm", "zscore")
        return settings

    def suggest(self):
        settings = self.get_settings()
        asig = self.state.analog_signal
        if asig is None:
            return Event(np.array([]) * pq.s)

        algorithm = settings.get("algorithm")
        if algorithm and algorithm in SpikeDetectorRegistry.names():
            sr = float(asig.sampling_rate.rescale(pq.Hz).magnitude)
            arr = np.asarray(asig.magnitude, dtype=np.float64).ravel()
            z_thresh = float(settings.get("zscore_threshold", settings.get("zscore_threhsold", 5.0)))
            params = {"z_thresh": z_thresh, **settings}
            cand = SpikeDetectorRegistry.find_candidates(algorithm, arr, fs_hz=sr, params=params)
            times = asig.t_start + (cand.sample_indices / asig.sampling_rate)
            return Event(times.rescale(pq.s))

        threshold = settings.get("threshold")
        if threshold is None:
            threshold = np.std(asig) * settings.get("zscore_threhsold", 5)

        threshold_crossings = asig > threshold
        threshold_crossings = np.where(threshold_crossings)[0]
        threshold_corssings = asig.t_start + (threshold_crossings / asig.sampling_rate)
        suggestions = Event(threshold_corssings)
        return suggestions


def suggest_units_with_detector(
    state: ViewerState,
    algorithm: str = "zscore",
    params: Optional[Dict[str, Any]] = None,
) -> Event:
    """Use any registered SpikeDetectorPlugin in SpikeDetectorRegistry to suggest candidate spike events."""
    if state is None or state.analog_signal is None:
        return Event(np.array([]) * pq.s)
    asig = state.analog_signal
    sr = float(asig.sampling_rate.rescale(pq.Hz).magnitude)
    arr = np.asarray(asig.magnitude, dtype=np.float64).ravel()
    cand = SpikeDetectorRegistry.find_candidates(algorithm, arr, fs_hz=sr, params=params)
    times = asig.t_start + (cand.sample_indices / asig.sampling_rate)
    return Event(times.rescale(pq.s))
