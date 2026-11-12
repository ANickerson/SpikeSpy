import logging
from typing import Optional, Dict, Any, List
import numpy as np
import quantities as pq
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot

from . import basic_tracking

logger = logging.getLogger("AsyncTracking")


class WorkerSignals(QObject):
    """Signals for background spike tracking workers."""
    spike_detected = Signal(int, int, int)  # (unit_idx, detected_sample, stimno)
    finished = Signal()


class AsyncTrackingWorker(QRunnable):
    """Worker task executing spike tracking on a background thread.
    Zero Qt GUI calls are made inside run().
    """

    def __init__(
        self,
        unit_idx: int,
        unit_name: str,
        stimno: int,
        sweep_waveform: np.ndarray,
        sampling_rate: float,
        prev_latency: float,
        threshold: float,
        window_ms: float,
        callback=None,
    ):
        super().__init__()
        self.unit_idx = int(unit_idx)
        self.unit_name = str(unit_name)
        self.stimno = int(stimno)
        # Ensure detached pure NumPy array for thread safety
        if hasattr(sweep_waveform, "magnitude"):
            sweep_waveform = sweep_waveform.magnitude
        self.sweep_waveform = np.asarray(sweep_waveform, dtype=np.float32).copy()
        self.sampling_rate = float(sampling_rate)
        self.prev_latency = float(prev_latency)
        self.threshold = float(threshold)
        self.window_ms = float(window_ms)
        self.callback = callback
        self.signals = WorkerSignals()

    @Slot()
    def run(self):
        try:
            detected_sample = basic_tracking.track_sweep_waveform(
                self.sweep_waveform,
                self.sampling_rate,
                self.prev_latency,
                threshold=self.threshold,
                window_ms=self.window_ms,
            )
            if detected_sample is not None:
                detected_sample = int(detected_sample)
                self.signals.spike_detected.emit(
                    self.unit_idx, detected_sample, self.stimno
                )
                if self.callback is not None:
                    self.callback(self.unit_idx, detected_sample, self.stimno)
        except Exception as e:
            logger.error(f"Error in AsyncTrackingWorker: {e}", exc_info=True)
        finally:
            self.signals.finished.emit()


class LiveTrackingManager(QObject):
    """Manages background tracking tasks for all auto_track enabled units.
    Dispatches tasks to QThreadPool.globalInstance() and marshals results
    safely back to the main UI thread via spike_tracked Signal.
    """

    spike_tracked = Signal(int, int, int)  # (unit_idx, detected_sample, stimno)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._active_workers: List[AsyncTrackingWorker] = []
        self._active_task_count: int = 0

    def _on_spike_detected(self, unit_idx: int, detected_sample: int, stimno: int):
        self.spike_tracked.emit(unit_idx, detected_sample, stimno)

    def _on_worker_finished(self):
        self._active_task_count = max(0, getattr(self, "_active_task_count", 1) - 1)

    def track_stimulus_async(
        self,
        state,
        stimno: int,
        waveforms_by_channel: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Dispatches background tracking tasks for all enabled units for the given stimulus sweep.
        Returns the number of tracking tasks dispatched.
        """
        if getattr(self, "_active_task_count", 0) >= 12:
            return 0

        if (
            state is None
            or state.event_signal is None
            or stimno < 0
            or stimno >= len(state.event_signal)
        ):
            return 0

        sr = float(
            state.sampling_rate
            if state.sampling_rate is not None
            else (
                state.analog_signal.sampling_rate.rescale(pq.Hz).magnitude
                if state.analog_signal is not None
                else 30000.0
            )
        )

        dispatched = 0
        for unit_idx, sg in enumerate(state.spike_groups):
            if not getattr(sg, "auto_track", False):
                continue

            # Check if unit already has an annotated spike on this sweep
            existing_sample = sg.get_sweep_sample(
                stimno, state.event_signal, sr, state.window_size
            )
            if existing_sample is not None:
                continue

            # Find most recent spike latency in prior sweeps within tracking_max_skips
            max_skips = getattr(sg, "tracking_max_skips", 1)
            prev_latency = None
            min_stim = max(0, stimno - 1 - max_skips)
            for prev_s in range(stimno - 1, min_stim - 1, -1):
                prev_sample = sg.get_sweep_sample(
                    prev_s, state.event_signal, sr, state.window_size
                )
                if prev_sample is not None:
                    prev_latency = (prev_sample / sr) * pq.s
                    break
                elif (
                    getattr(sg, "idx_arr", None) is not None
                    and prev_s < len(sg.idx_arr)
                    and sg.idx_arr[prev_s] is not None
                ):
                    prev_samp = sg.idx_arr[prev_s][0]
                    prev_latency = (prev_samp / sr) * pq.s
                    break

            if prev_latency is None:
                continue

            ch = (
                getattr(sg, "tracking_channel", "")
                or getattr(sg, "channel", "")
                or (state.analog_signal.name if state.analog_signal else "")
            )

            sweep_wf = None
            if waveforms_by_channel and ch in waveforms_by_channel:
                sweep_wf = waveforms_by_channel[ch]
            elif waveforms_by_channel and None in waveforms_by_channel:
                sweep_wf = waveforms_by_channel[None]
            else:
                ch_erp = state.get_channel_erp(ch) if ch else state.get_erp()
                if ch_erp is None:
                    ch_erp = state.get_erp()
                if ch_erp is not None and stimno < len(ch_erp):
                    sweep_wf = ch_erp[stimno]

            if sweep_wf is None:
                continue

            if hasattr(sweep_wf, "magnitude"):
                sweep_wf = sweep_wf.magnitude
            sweep_wf = np.asarray(sweep_wf, dtype=np.float32).copy()

            thresh_mv = float(getattr(sg, "tracking_threshold_mv", 0.01))
            win_ms = float(getattr(sg, "tracking_window_ms", 5.0))
            prev_lat_s = float(
                prev_latency.rescale(pq.s).magnitude
                if hasattr(prev_latency, "rescale")
                else prev_latency
            )
            unit_name = (
                getattr(sg.event, "name", f"unit_{unit_idx}")
                if getattr(sg, "event", None) is not None
                else f"unit_{unit_idx}"
            )

            worker = AsyncTrackingWorker(
                unit_idx=unit_idx,
                unit_name=unit_name,
                stimno=stimno,
                sweep_waveform=sweep_wf,
                sampling_rate=sr,
                prev_latency=prev_lat_s,
                threshold=thresh_mv,
                window_ms=win_ms,
            )
            # Connect worker signal to manager slot
            worker.signals.spike_detected.connect(self._on_spike_detected)
            worker.signals.finished.connect(self._on_worker_finished)
            self._active_task_count = getattr(self, "_active_task_count", 0) + 1
            QThreadPool.globalInstance().start(worker)
            dispatched += 1

        return dispatched
