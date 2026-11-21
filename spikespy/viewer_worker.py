"""Background worker infrastructure for offloading viewer computations from UI thread.
Supports task coalescing, prioritization, and batching."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Set
from collections import OrderedDict, deque
import threading

import numpy as np
from PySide6.QtCore import QObject, QThread, QThreadPool, QRunnable, Signal, Slot, QMutex, QWaitCondition
from scipy.signal import find_peaks, fftconvolve

from .processing import MultiChannelTemplateMatcher
from .ViewerState import ViewerState, tracked_neuron_unit
import quantities as pq

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Types of viewer tasks for coalescing."""
    MULTI_TRACE_RENDER = "multi_trace_render"
    SINGLE_TRACE_RENDER = "single_trace_render"
    UNIT_VIEW_RENDER = "unit_view_render"
    TEMPLATE_UPDATE = "template_update"
    HISTOGRAM_UPDATE = "histogram_update"
    RIGHT_AXIS_UPDATE = "right_axis_update"
    SPIKE_GROUP_UPDATE = "spike_group_update"
    FOOTPRINT_RENDER = "footprint_render"


class TaskPriority(Enum):
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3


@dataclass
class ViewerTask:
    """A unit of work for viewer processing."""
    task_type: TaskType
    priority: TaskPriority
    viewer_id: str  # Unique identifier for the viewer instance
    payload: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    task_id: int = 0
    
    # Coalescing key - tasks with same key can be combined
    coalesce_key: str = ""
    
    def __lt__(self, other):
        # For priority queue ordering
        if self.priority != other.priority:
            return self.priority.value > other.priority.value  # Higher priority first
        return self.timestamp < other.timestamp  # Older first


@dataclass
class TaskResult:
    """Result of a completed task."""
    task: ViewerTask
    result: Any
    error: Optional[str] = None


class CoalescingStrategy(ABC):
    """Strategy for combining/skipping tasks."""
    
    @abstractmethod
    def should_coalesce(self, existing: ViewerTask, new: ViewerTask) -> bool:
        """Return True if new task should replace existing."""
        pass
    
    @abstractmethod
    def coalesce(self, existing: ViewerTask, new: ViewerTask) -> ViewerTask:
        """Combine two tasks into one."""
        pass


class LatestWinsCoalescing(CoalescingStrategy):
    """Newer task always replaces older task of same type/viewer."""
    
    def should_coalesce(self, existing: ViewerTask, new: ViewerTask) -> bool:
        return (existing.task_type == new.task_type and 
                existing.viewer_id == new.viewer_id and
                existing.coalesce_key == new.coalesce_key)
    
    def coalesce(self, existing: ViewerTask, new: ViewerTask) -> ViewerTask:
        # Keep newer task's payload, but preserve higher priority
        priority = max(existing.priority, new.priority, key=lambda p: p.value)
        return ViewerTask(
            task_type=new.task_type,
            priority=priority,
            viewer_id=new.viewer_id,
            payload=new.payload,
            timestamp=new.timestamp,
            task_id=new.task_id,
            coalesce_key=new.coalesce_key,
        )


class BatchCoalescing(CoalescingStrategy):
    """Batch multiple tasks of same type into single execution."""
    
    def __init__(self, max_batch_size: int = 10, max_batch_age_ms: float = 50.0):
        self.max_batch_size = max_batch_size
        self.max_batch_age_ms = max_batch_age_ms
        self._pending_batches: Dict[str, List[ViewerTask]] = {}
    
    def should_coalesce(self, existing: ViewerTask, new: ViewerTask) -> bool:
        return (existing.task_type == new.task_type and 
                existing.viewer_id == new.viewer_id and
                existing.coalesce_key == new.coalesce_key)
    
    def coalesce(self, existing: ViewerTask, new: ViewerTask) -> ViewerTask:
        # This is called for each new task - we batch in the queue instead
        return new


class TaskQueue:
    """Thread-safe task queue with coalescing and prioritization."""
    
    def __init__(self, coalescing_strategy: Optional[CoalescingStrategy] = None):
        self._queue: List[ViewerTask] = []
        self._task_counter = 0
        self._mutex = QMutex()
        self._condition = QWaitCondition()
        self._coalescing = coalescing_strategy or LatestWinsCoalescing()
        self._pending_by_key: Dict[str, ViewerTask] = {}
        self._shutdown = False
    
    def enqueue(self, task: ViewerTask) -> int:
        """Add task to queue, applying coalescing. Returns task ID."""
        self._mutex.lock()
        try:
            if self._shutdown:
                return -1
            
            self._task_counter += 1
            task.task_id = self._task_counter
            
            key = f"{task.task_type.value}:{task.viewer_id}:{task.coalesce_key}"
            
            # Check if we can coalesce with existing pending task
            if key in self._pending_by_key:
                existing = self._pending_by_key[key]
                if self._coalescing.should_coalesce(existing, task):
                    # Replace with coalesced task
                    coalesced = self._coalescing.coalesce(existing, task)
                    coalesced.task_id = existing.task_id  # Keep original ID
                    self._pending_by_key[key] = coalesced
                    # Re-sort queue
                    self._rebuild_queue()
                    return coalesced.task_id
            
            # Add new task
            self._pending_by_key[key] = task
            self._insert_sorted(task)
            self._condition.wakeOne()
            return task.task_id
        finally:
            self._mutex.unlock()
    
    def _insert_sorted(self, task: ViewerTask):
        """Insert task maintaining priority order."""
        # Simple insertion sort - queue is typically small
        idx = 0
        while idx < len(self._queue) and not (task < self._queue[idx]):
            idx += 1
        self._queue.insert(idx, task)
    
    def _rebuild_queue(self):
        """Rebuild queue from pending dict."""
        self._queue = list(self._pending_by_key.values())
        self._queue.sort()
    
    def dequeue(self, timeout_ms: int = -1) -> Optional[ViewerTask]:
        """Get next task, blocking until available."""
        self._mutex.lock()
        try:
            while not self._shutdown and not self._queue:
                if timeout_ms < 0:
                    self._condition.wait(self._mutex)
                else:
                    if not self._condition.wait(self._mutex, timeout_ms):
                        return None  # Timeout
            
            if self._shutdown:
                return None
            
            task = self._queue.pop(0)
            key = f"{task.task_type.value}:{task.viewer_id}:{task.coalesce_key}"
            self._pending_by_key.pop(key, None)
            return task
        finally:
            self._mutex.unlock()
    
    def peek(self) -> Optional[ViewerTask]:
        """View next task without removing."""
        self._mutex.lock()
        try:
            return self._queue[0] if self._queue else None
        finally:
            self._mutex.unlock()
    
    def cancel_viewer_tasks(self, viewer_id: str) -> int:
        """Cancel all pending tasks for a viewer."""
        self._mutex.lock()
        try:
            cancelled = 0
            keys_to_remove = [k for k in self._pending_by_key if k.split(":")[1] == viewer_id]
            for key in keys_to_remove:
                self._pending_by_key.pop(key, None)
                cancelled += 1
            self._rebuild_queue()
            return cancelled
        finally:
            self._mutex.unlock()
    
    def cancel_task_type(self, task_type: TaskType, viewer_id: str = "") -> int:
        """Cancel all pending tasks of a specific type."""
        self._mutex.lock()
        try:
            cancelled = 0
            keys_to_remove = []
            for key, task in self._pending_by_key.items():
                parts = key.split(":")
                if parts[0] == task_type.value and (not viewer_id or parts[1] == viewer_id):
                    keys_to_remove.append(key)
            for key in keys_to_remove:
                self._pending_by_key.pop(key, None)
                cancelled += 1
            self._rebuild_queue()
            return cancelled
        finally:
            self._mutex.unlock()
    
    def get_pending_count(self, viewer_id: str = "") -> int:
        """Get count of pending tasks."""
        self._mutex.lock()
        try:
            if not viewer_id:
                return len(self._pending_by_key)
            return sum(1 for k in self._pending_by_key if k.split(":")[1] == viewer_id)
        finally:
            self._mutex.unlock()
    
    def shutdown(self):
        """Signal shutdown to waiting threads."""
        self._mutex.lock()
        try:
            self._shutdown = True
            self._condition.wakeAll()
        finally:
            self._mutex.unlock()


class ViewerWorker(QThread):
    """Background worker thread that processes viewer tasks."""
    
    task_completed = Signal(int, object, str)  # task_id, result, error
    footprintRenderReady = Signal(dict)
    
    def __init__(self, task_queue: TaskQueue, parent=None):
        super().__init__(parent)
        self._queue = task_queue
        self._running = True
        self._viewer_states: Dict[str, ViewerState] = {}
    
    def register_viewer(self, viewer_id: str, state: ViewerState):
        """Register a viewer's state for access during processing."""
        self._viewer_states[viewer_id] = state
    
    def unregister_viewer(self, viewer_id: str):
        """Unregister a viewer and cancel its pending tasks."""
        self._viewer_states.pop(viewer_id, None)
        self._queue.cancel_viewer_tasks(viewer_id)
    
    def run(self):
        """Main worker loop."""
        logger.debug("ViewerWorker started")
        while self._running:
            task = self._queue.dequeue(timeout_ms=100)
            if task is None:
                continue
            
            try:
                result = self._process_task(task)
                self.task_completed.emit(task.task_id, result, "")
            except Exception as e:
                logger.exception(f"Task {task.task_id} failed")
                self.task_completed.emit(task.task_id, None, str(e))
        
        logger.debug("ViewerWorker stopped")
    
    def stop(self):
        """Stop the worker thread."""
        self._running = False
        self._queue.shutdown()
        self.wait(2000)
    
    def _process_task(self, task: ViewerTask) -> Any:
        """Route task to appropriate processor."""
        if task.task_type == TaskType.MULTI_TRACE_RENDER:
            return self._process_multi_trace(task)
        elif task.task_type == TaskType.SINGLE_TRACE_RENDER:
            return self._process_single_trace(task)
        elif task.task_type == TaskType.UNIT_VIEW_RENDER:
            return self._process_unit_view(task)
        elif task.task_type == TaskType.TEMPLATE_UPDATE:
            return self._process_template_update(task)
        elif task.task_type == TaskType.HISTOGRAM_UPDATE:
            return self._process_histogram(task)
        elif task.task_type == TaskType.RIGHT_AXIS_UPDATE:
            return self._process_right_axis(task)
        elif task.task_type == TaskType.SPIKE_GROUP_UPDATE:
            return self._process_spike_group(task)
        elif task.task_type == TaskType.FOOTPRINT_RENDER:
            return self._process_footprint_render(task)
        return None
    
    def _process_multi_trace(self, task: ViewerTask) -> Dict[str, Any]:
        """Process MultiTraceView render task."""
        p = task.payload
        erp = p["erp_data"]
        if erp is None or erp.size == 0 or erp.shape[0] == 0:
            return {}
        
        # Color scaling
        std_val = p.get("running_std")
        if std_val is None or not np.isfinite(std_val) or std_val <= 0:
            std_val = float(np.std(erp)) if erp.size else 1.0
        if not np.isfinite(std_val) or std_val <= 0:
            std_val = 1.0
        percentiles = np.arange(0, 100) * 2 * std_val
        clim = (percentiles[int(p["lower_percentile"])], percentiles[int(p["upper_percentile"])])
        
        full_shape = p.get("erp_full_shape")
        if full_shape is None:
            full_shape = erp.shape
        full_shape = tuple(full_shape)
        display = p.get("erp_display")
        if display is None or display.size == 0:
            display = erp

        max_val = float(np.max(display)) if display.size else 1.0
        if not np.isfinite(max_val):
            max_val = 1.0
        clipped_data = np.clip(np.nan_to_num(display, nan=0.0, posinf=max_val, neginf=0.0), 0, max_val)
        
        # Right axis data
        right_ax_data = self._compute_right_axis_data(p, erp)
        
        # Y-lim
        ylim = None
        if p.get("lock_to_stim", False):
            w = max(5, full_shape[0] // 50)
            ylim = (p["stimno"] + w, p["stimno"] - w)
        elif p.get("follow_live", False):
            ylim = (full_shape[0], 0)
        
        return {
            "clipped_data": clipped_data,
            "full_shape": full_shape,
            "extent": (0, full_shape[1], full_shape[0], 0),
            "slice_rect": p.get("slice_rect"),
            "clim": clim,
            "right_ax_data": right_ax_data,
            "ylim": ylim,
            "xlim": (0, full_shape[1]),
            "stimno": p["stimno"],
        }
    
    def _compute_right_axis_data(self, p: Dict, erp: np.ndarray) -> Dict[str, Any]:
        """Compute right-axis plots data."""
        data = {}
        
        # Stim frequency
        # Check pre-extracted arrays first
        stim_times = None
        if "right_axis_stim_times_s" in p and p["right_axis_stim_times_s"] is not None:
            stim_times = np.asarray(p["right_axis_stim_times_s"], dtype=np.float64)
        elif p.get("event_signal") is not None and len(p["event_signal"]) > 1:
            try:
                stim_times = p["event_signal"].times.rescale(pq.second).magnitude
            except Exception:
                stim_times = None

        if stim_times is not None and len(stim_times) > 1:
            diffs = np.diff(stim_times)
            stim_freq = np.divide(
                1.0, diffs, out=np.zeros_like(diffs), where=diffs != 0
            )
            data["Stim. Freq. (Hz)"] = (stim_freq, np.arange(1, len(stim_times)))
        
        # Latency diffs
        # Check pre-extracted arrays first
        if "right_axis_event_samples" in p and p["right_axis_event_samples"] is not None:
            try:
                samples = np.asarray(p["right_axis_event_samples"], dtype=np.float64)
                fs = float(p.get("right_axis_fs_hz", 30000.0) or 30000.0)
                latencies = samples / fs
                idx_na = np.isnan(latencies)
                if np.any(~idx_na):
                    valid_lat = latencies[~idx_na]
                    current_spike_diffs = np.diff(valid_lat)
                    out = np.ones(latencies.shape) * np.nan
                    valid_idx = np.where(~idx_na)[0]
                    if len(valid_idx) > 1:
                        out[valid_idx[1:]] = current_spike_diffs
                    data["Latency Diff."] = (out, np.arange(len(latencies)))
            except Exception:
                pass
        elif p.get("viewer_ref") and hasattr(p["viewer_ref"], "state") and p["viewer_ref"].state:
            ug = p["viewer_ref"].state.getUnitGroup()
            if ug is not None and p.get("event_signal") is not None:
                try:
                    latencies = ug.get_latencies(p["event_signal"])
                    idx_na = np.isnan(latencies)
                    if np.any(~idx_na):
                        valid_lat = latencies[~idx_na]
                        current_spike_diffs = np.diff(valid_lat)
                        out = np.ones(latencies.shape) * np.nan
                        valid_idx = np.where(~idx_na)[0]
                        if len(valid_idx) > 1:
                            out[valid_idx[1:]] = current_spike_diffs
                        data["Latency Diff."] = (out, np.arange(len(latencies)))
                except Exception:
                    pass
        
        # Row stats for channels
        channel_row_stats = p.get("channel_row_stats")
        if isinstance(channel_row_stats, dict) and channel_row_stats:
            for ch_name, row_values in channel_row_stats.items():
                if row_values is not None and len(row_values) > 0:
                    r_arr = np.asarray(row_values, dtype=np.float32)
                    data[ch_name] = (r_arr, np.arange(len(r_arr)))
        elif p.get("channel_names") and p.get("viewer_ref"):
            for ch_name in p["channel_names"]:
                try:
                    if hasattr(p["viewer_ref"], "state") and p["viewer_ref"].state:
                        state = p["viewer_ref"].state
                        if getattr(state, "is_streaming", False):
                            buf = None
                            if hasattr(state, "_live_aux_erp_buffers") and ch_name in state._live_aux_erp_buffers:
                                buf = state._live_aux_erp_buffers[ch_name]
                            elif (
                                getattr(state, "_live_erp_buffer", None) is not None
                                and state.analog_signal is not None
                                and ch_name == getattr(state.analog_signal, "name", None)
                            ):
                                buf = state._live_erp_buffer
                            
                            if buf is not None and len(buf) > 0:
                                row_values = buf.row_max if p.get("show_max", False) else buf.row_mean
                                if hasattr(row_values, "magnitude"):
                                    row_values = row_values.magnitude
                                data[ch_name] = (
                                    row_values,
                                    np.arange(len(row_values)),
                                )
                except Exception:
                    pass
        
        return data
    
    def _process_single_trace(self, task: ViewerTask) -> Dict[str, Any]:
        """Process SingleTraceView render task."""
        p = task.payload
        
        # Check if this is a convolution-only task (LIFO buffer flush)
        convolution_only = p.get("convolution_only", False)
        decimate = p.get("decimate", True)
        pre_offset_s = float(p.get("pre_offset_s", 0.0))
        
        # Primary trace
        if not convolution_only:
            raw_primary = p["primary_erp"]
            sr = float(p["sampling_rate"])
            if decimate:
                primary_x, primary_y = self._decimate_trace(
                    raw_primary, sr, pre_offset_s=pre_offset_s
                )
            else:
                if raw_primary is None or len(raw_primary) == 0:
                    primary_x, primary_y = np.array([]), np.array([])
                else:
                    arr = np.asarray(getattr(raw_primary, "magnitude", raw_primary), dtype=np.float32)
                    pre_samples = int(round(pre_offset_s * sr))
                    primary_x = (np.arange(len(arr)) - pre_samples) / sr
                    primary_y = arr
        else:
            primary_x, primary_y = np.array([]), np.array([])
        
        # Overlay traces
        overlay_data = {}
        if not convolution_only:
            sr = float(p["sampling_rate"])
            for ch_name, ch_erp in p["overlay_erps"].items():
                if ch_erp is not None and len(ch_erp) > 0:
                    if decimate:
                        ch_x, ch_y = self._decimate_trace(
                            ch_erp, sr, pre_offset_s=pre_offset_s
                        )
                    else:
                        arr = np.asarray(getattr(ch_erp, "magnitude", ch_erp), dtype=np.float32)
                        pre_samples = int(round(pre_offset_s * sr))
                        ch_x = (np.arange(len(arr)) - pre_samples) / sr
                        ch_y = arr
                    overlay_data[ch_name] = {"x": ch_x, "y": ch_y}
        
        # Template convolution (skip if flagged)
        conv_result = None
        if not p.get("skip_convolution", False):
            conv_result = self._compute_convolution(p)
        
        # Peak finding (skip if convolution-only)
        if not convolution_only:
            peaks, peaks_down = self._find_peaks(p["primary_erp"])
        else:
            peaks, peaks_down = np.array([]), np.array([])

        event_signal = p.get("event_signal")
        stim_time = None
        if event_signal is not None and 0 <= p["stimno"] < len(event_signal):
            stim_time = event_signal.times[p["stimno"]]

        markers = (None, None, None)
        if not convolution_only:
            markers = self._compute_sweep_markers(p)

        return {
            "convolution_only": convolution_only,
            "primary_trace": {"x": primary_x, "y": primary_y},
            "overlay_traces": overlay_data,
            "convolution": conv_result,
            "peaks": peaks,
            "peaks_down": peaks_down,
            "stimno": p["stimno"],
            "stim_time": stim_time,
            "cur_latency": markers[0],
            "prev_latency": markers[1],
            "next_latency": markers[2],
        }

    def _compute_sweep_markers(self, p: Dict) -> tuple:
        """Latency of the current spike and the nearest identified neighbours.

        Only scans +/-10 sweeps, so this stays constant-time as the session grows.
        """
        sg = p.get("spike_group")
        ev = p.get("event_signal")
        stimno = p["stimno"]
        sr = float(p["sampling_rate"])
        window = p.get("window_size")

        cur_latency = None
        if sg is not None and ev is not None and 0 <= stimno < len(ev):
            try:
                cur_ev = ev[stimno]
                lat = sg.get_latencies(np.array([cur_ev.magnitude]) * cur_ev.units)[0]
                lat = lat.rescale(pq.s)
                if not np.isnan(lat) and (window is None or lat < window):
                    cur_latency = float(lat.magnitude)
            except Exception:
                cur_latency = None

        idx_arr = getattr(sg, "idx_arr", None) if sg is not None else None
        prev_latency = None
        next_latency = None
        pre_offset_s = float(p.get("pre_offset_s", 0.0))
        pre_samples = int(round(pre_offset_s * sr))
        if idx_arr:
            for offset in range(1, 10):
                i = stimno - offset
                if 0 <= i < len(idx_arr) and idx_arr[i] is not None:
                    prev_latency = (idx_arr[i][0] - pre_samples) / sr
                    break
            for offset in range(1, 10):
                i = stimno + offset
                if 0 <= i < len(idx_arr) and idx_arr[i] is not None:
                    next_latency = (idx_arr[i][0] - pre_samples) / sr
                    break

        return cur_latency, prev_latency, next_latency
    
    def _decimate_trace(
        self, trace: np.ndarray, sr: float, max_points: int = 2500, pre_offset_s: float = 0.0
    ) -> tuple:
        """Decimate trace for efficient display."""
        if trace is None or len(trace) == 0:
            return np.array([]), np.array([])
        
        pre_samples = int(round(pre_offset_s * sr))
        x_arr = (np.arange(len(trace)) - pre_samples) / sr
        n_samples = len(x_arr)
        if n_samples <= max_points:
            return x_arr, trace
        
        n_bins = max_points // 2
        pts_per_bin = n_samples // n_bins
        total_pts = pts_per_bin * n_bins
        x_trim = x_arr[:total_pts].reshape(n_bins, pts_per_bin)
        y_trim = trace[:total_pts].reshape(n_bins, pts_per_bin)
        
        min_idx = np.argmin(y_trim, axis=1)
        max_idx = np.argmax(y_trim, axis=1)
        row_indices = np.arange(n_bins)
        first_is_min = min_idx <= max_idx
        first_idx = np.where(first_is_min, min_idx, max_idx)
        second_idx = np.where(first_is_min, max_idx, min_idx)
        
        x_out = np.column_stack([x_trim[row_indices, first_idx], x_trim[row_indices, second_idx]]).ravel()
        y_out = np.column_stack([y_trim[row_indices, first_idx], y_trim[row_indices, second_idx]]).ravel()
        return x_out, y_out
    
    def _compute_convolution(self, p: Dict) -> Optional[Dict]:
        """Compute multi-channel template convolution for one sweep."""
        templates = p.get("templates") or {}
        sweeps = p.get("conv_sweeps") or {}
        if not templates or not sweeps:
            return None

        composite_conv = None
        for ch, tpl in templates.items():
            sweep_data = sweeps.get(ch)
            if sweep_data is None:
                continue
            if hasattr(sweep_data, "magnitude"):
                sweep_data = sweep_data.magnitude
            sweep_data = np.asarray(sweep_data, dtype=np.float32)
            if sweep_data.size == 0:
                continue

            tpl = np.asarray(tpl, dtype=np.float32)[: sweep_data.size]
            if tpl.size == 0:
                continue
            std_sweep = float(np.std(sweep_data))
            norm_sweep = (
                sweep_data / (std_sweep if std_sweep > 1e-8 else 1.0)
            ).astype(np.float32)
            c = fftconvolve(norm_sweep, tpl, mode="same").astype(np.float32)
            c = c[: sweep_data.size]
            c[0 : len(tpl)] = 0
            c[-len(tpl) :] = 0
            if composite_conv is None:
                composite_conv = c
            else:
                composite_conv += c

        if composite_conv is None:
            return None

        margin = min(1000, max(1, len(composite_conv) // 10))
        inner = (
            composite_conv[margin:-margin]
            if len(composite_conv) > 2 * margin
            else composite_conv
        )
        conv_high = np.percentile(inner, 99)
        highlight, _ = find_peaks(composite_conv, conv_high)
        highlight = highlight - 1
        hl = highlight[np.argsort(composite_conv[highlight])[::-1]]
        hl_top = hl[0:100]

        return {
            "conv": composite_conv,
            "peaks": hl_top,
            "high_thresh": conv_high,
        }
    
    def _compute_histogram(self, p: Dict) -> Optional[Dict]:
        """Compute latency histogram."""
        sg = p.get("spike_group")
        if sg is None or len(sg.event) == 0:
            return None
        
        try:
            values = sg.get_latencies(p["event_signal"])
            values = values[~np.isnan(values)]
            if len(values) == 0:
                return None
            
            step = 2 * pq.ms
            bins = np.arange(
                np.floor(((np.min(values) // step)) - 1) * step,
                np.ceil(np.max(values) + (step * 2)),
                step,
            )
            values_binned = np.histogram(values, bins=bins)
            return {
                # Bin edges (length n+1) match PyQtGraph's stepMode="center".
                "bins": values_binned[1] / 1000,
                "counts": values_binned[0] / max(values_binned[0]),
            }
        except Exception:
            return None
    
    def _find_peaks(self, trace: np.ndarray) -> tuple:
        if trace is None or len(trace) == 0:
            return np.array([]), np.array([])
        pts, _ = find_peaks(trace)
        pts_down, _ = find_peaks(-trace)
        return pts, pts_down
    
    def _process_unit_view(self, task: ViewerTask) -> Dict[str, Any]:
        """Process UnitView render task."""
        p = task.payload
        if p["primary_erp"] is None or len(p["primary_erp"]) == 0:
            return {}
        
        sr = p["sampling_rate"]
        w = int(sr * 0.002)
        xarr = np.arange(-w, +w)
        
        idx_arr = self._get_idx_arr(p)
        aligned_centers = {}
        mean_arr = np.zeros(len(xarr))
        lines_data = {}
        ylims = [-0.1, 0.1]
        count = 0
        
        search_r = max(int(0.00075 * sr), 3)
        
        for i, (idx, d) in enumerate(zip(idx_arr, p["primary_erp"])):
            if idx is None:
                continue
            c_idx = idx[0]
            if p["align_waveforms"]:
                search_start = max(0, c_idx - search_r)
                search_end = min(len(d), c_idx + search_r + 1)
                if search_end > search_start:
                    c_aligned = search_start + int(np.argmax(np.abs(d[search_start:search_end])))
                else:
                    c_aligned = c_idx
            else:
                c_aligned = c_idx
            aligned_centers[i] = c_aligned
            
            w_range = [-w, w]
            xmin = c_aligned - w
            if xmin < 0:
                w_range[0] = abs(xmin) + 1
                xmin = 0
            xmax = c_aligned + w
            if xmax > len(d):
                w_range[1] = abs(xmax - len(d))
                xmax = len(d)
            
            ydata = d[xmin:xmax]
            try:
                mean_arr[w_range[0] + w : w_range[1] + w] += ydata
                lines_data[i] = np.vstack((np.arange(w_range[0], w_range[1]), ydata)).T
                count += 1
                ylims[0] = min(ylims[0], np.min(ydata))
                ylims[1] = max(ylims[1], np.max(ydata))
            except Exception:
                pass
        
        mean_waveform = mean_arr / max(count, 1) if count > 0 else np.zeros(len(xarr))
        
        # Neighbor channels
        neighbor_data = {}
        if p["show_neighbors"] and p["neighbor_erps"]:
            colors = ["#e41a1c", "#377eb8", "#4daf4a", "#ff7f00", "#984ea3"]
            for n_idx, (ch_name, ch_erp) in enumerate(p["neighbor_erps"].items()):
                if ch_erp is None or len(ch_erp) == 0:
                    continue
                n_mean_arr = np.zeros(len(xarr))
                n_count = 0
                for s_i, idx in enumerate(idx_arr):
                    if idx is None or s_i >= len(ch_erp):
                        continue
                    d_ch = ch_erp[s_i]
                    if hasattr(d_ch, "magnitude"):
                        d_ch = d_ch.magnitude
                    c_aligned = aligned_centers.get(s_i, idx[0])
                    w_range = [-w, w]
                    xmin = c_aligned - w
                    if xmin < 0:
                        w_range[0] = abs(xmin) + 1
                        xmin = 0
                    xmax = c_aligned + w
                    if xmax > len(d_ch):
                        w_range[1] = abs(xmax - len(d_ch))
                        xmax = len(d_ch)
                    ydata_ch = d_ch[xmin:xmax]
                    try:
                        n_mean_arr[w_range[0] + w : w_range[1] + w] += ydata_ch
                        n_count += 1
                        ylims[0] = min(ylims[0], np.min(ydata_ch))
                        ylims[1] = max(ylims[1], np.max(ydata_ch))
                    except Exception:
                        pass
                
                if n_count > 0:
                    n_mean = n_mean_arr / n_count
                    peak_offset = int(np.argmax(np.abs(n_mean))) - w
                    lag_ms = float((peak_offset / sr) * 1000.0)
                    neighbor_data[ch_name] = {
                        "waveform": n_mean,
                        "lag_ms": lag_ms,
                        "color": colors[n_idx % len(colors)],
                    }
        
        # Current stim line
        cur_stim_line = None
        if p["spike_group"] is not None and p["event_signal"] is not None and p["stimno"] < len(p["event_signal"]):
            cur_ev = p["event_signal"][p["stimno"]]
            lat = p["spike_group"].get_latencies(np.array([cur_ev.magnitude]) * cur_ev.units)[0]
            if not np.isnan(lat):
                cur_stim_line = {
                    "sample": int(lat * sr),
                    "visible": True,
                }
        
        return {
            "xarr": xarr,
            "mean_waveform": mean_waveform,
            "lines_data": lines_data,
            "neighbor_data": neighbor_data,
            "ylims": ylims,
            "cur_stim_line": cur_stim_line,
            "stimno": p["stimno"],
        }
    
    def _get_idx_arr(self, p: Dict) -> List:
        sg = p.get("spike_group")
        if sg is None:
            return [None] * len(p["primary_erp"])
        
        if p["event_signal"] is not None and hasattr(sg, "get_latencies"):
            lats = sg.get_latencies(p["event_signal"])
            sample_arr = []
            for lat in lats:
                if np.isnan(lat):
                    sample_arr.append(None)
                else:
                    sec = float(lat.rescale(pq.s).magnitude) if hasattr(lat, "rescale") else float(lat)
                    sample_arr.append((int(round(sec * p["sampling_rate"])), 1))
            return sample_arr
        elif getattr(sg, "idx_arr", None) is not None:
            return sg.idx_arr
        return [None] * len(p["primary_erp"])
    
    def _process_template_update(self, task: ViewerTask) -> Dict[str, np.ndarray]:
        """Compute templates for spike detection (separate task)."""
        p = task.payload
        templates = {}
        
        sg = p.get("spike_group")
        if sg is None or sg.event is None or len(sg.event) == 0:
            return templates
        
        sr = p["sampling_rate"]
        pre_samples = int(0.010 * sr)
        post_samples = int(0.020 * sr)
        
        lats = sg.get_latencies(p["event_signal"])
        valid_sweeps = np.where(~np.isnan(lats))[0]
        if len(valid_sweeps) == 0:
            return templates
        
        valid_sweeps = valid_sweeps[-50:]  # Last 50 spikes
        
        for ch in p["channels"]:
            erp_ch = p["channel_erps"].get(ch)
            if erp_ch is None or len(erp_ch) == 0:
                continue
            if hasattr(erp_ch, "magnitude"):
                erp_ch = erp_ch.magnitude
            erp_ch = erp_ch.astype(np.float32, copy=False)
            
            snippets = []
            for s_idx in valid_sweeps:
                if s_idx >= len(erp_ch):
                    continue
                lat = lats[s_idx]
                c_idx = int(round(float((lat.rescale(pq.s) * sr).magnitude)))
                start = c_idx - pre_samples
                end = c_idx + post_samples
                sweep_row = erp_ch[s_idx]
                if start >= 0 and end <= len(sweep_row):
                    snippets.append(sweep_row[start:end])
                else:
                    pad_left = max(0, -start)
                    pad_right = max(0, end - len(sweep_row))
                    v_start = max(0, start)
                    v_end = min(len(sweep_row), end)
                    if v_end > v_start:
                        sub = sweep_row[v_start:v_end]
                        snippets.append(np.pad(sub, (pad_left, pad_right), mode="edge"))
            
            if snippets:
                mean_tpl = np.mean(snippets, axis=0).astype(np.float32)
                std_tpl = float(np.std(mean_tpl))
                norm_tpl = (mean_tpl / (std_tpl if std_tpl > 1e-8 else 1.0)).astype(np.float32)
                templates[ch] = norm_tpl
        
        return templates
    
    def _process_histogram(self, task: ViewerTask) -> Optional[Dict]:
        """Compute histogram separately."""
        return self._compute_histogram(task.payload)
    
    def _process_right_axis(self, task: ViewerTask) -> Dict[str, Any]:
        """Compute right axis data separately."""
        return self._compute_right_axis_data(task.payload, task.payload["erp_data"])
    
    def _process_spike_group(self, task: ViewerTask) -> Dict[int, Dict]:
        """Process spike group points separately."""
        p = task.payload
        spikegroup_points = {}
        sr = p["sampling_rate"]
        pre_offset_s = float(p.get("pre_offset_s", 0.0))
        for i, sg in enumerate(p["spike_groups"]):
            if sg is None or sg.event is None or len(sg.event) == 0:
                continue
            lats = sg.get_latencies(p["event_signal"])
            valid = ~np.isnan(lats)
            sweeps = np.where(valid)[0]
            if len(sweeps) == 0:
                continue
            lat_vals = [
                float(lats[s].rescale(pq.s).magnitude) if hasattr(lats[s], "rescale") else float(lats[s])
                for s in sweeps
            ]
            samples = np.array(
                [int(round((lv + pre_offset_s) * sr)) for lv in lat_vals],
                dtype=float,
            )
            spikegroup_points[i] = {
                "samples": samples,
                "sweeps": sweeps,
                "is_current": i == p.get("current_unit_idx", -1),
            }
        return spikegroup_points

    def _process_footprint_render(self, task: ViewerTask) -> Dict[str, Any]:
        """Process UnitFootprint render task off the GUI thread."""
        p = task.payload
        result = {}
        try:
            from .dsp.footprint import compute_unit_footprint_arrays
            result = compute_unit_footprint_arrays(p)
        except Exception:
            # Fallback footprint array computation from pre-extracted arrays/callable
            calc_fn = p.get("calc_fn")
            if callable(calc_fn):
                result = calc_fn()
            else:
                channels = p.get("channels", [])
                ref_chan = p.get("ref_channel", "") or (channels[0] if channels else "")
                chan_erps = p.get("chan_erps", {})
                valid_sweeps = p.get("valid_sweeps", [])
                valid_samples = p.get("valid_samples", [])
                sr = float(p.get("sampling_rate", 30000.0))
                window_ms = float(p.get("window_ms", 4.0))
                align_peaks = bool(p.get("align_peaks", True))
                last_n = int(p.get("last_n", 20))

                if last_n > 0 and len(valid_sweeps) > last_n:
                    valid_sweeps = valid_sweeps[-last_n:]
                    valid_samples = valid_samples[-last_n:]
                window_samples = max(8, int(round((window_ms / 1000.0) * sr)))
                if window_samples % 2 != 0:
                    window_samples += 1
                half_win = window_samples // 2
                t_ms = (np.arange(window_samples) - half_win) * (1000.0 / sr)

                n_ch = len(channels)
                n_events = len(valid_sweeps)
                ref_erp = chan_erps.get(ref_chan)

                if ref_erp is not None and n_events > 0 and n_ch > 0:
                    snippets = np.zeros((n_ch, n_events, window_samples), dtype=np.float32)
                    for ev_i, (s_idx, s_samp) in enumerate(zip(valid_sweeps, valid_samples)):
                        if s_idx >= ref_erp.shape[0]:
                            continue
                        ref_trace = ref_erp[s_idx]
                        n_tot_pts = len(ref_trace)
                        center_samp = s_samp
                        if align_peaks:
                            jitter_search_w = max(4, int(0.001 * sr))
                            search_start = max(0, s_samp - jitter_search_w)
                            search_end = min(n_tot_pts, s_samp + jitter_search_w + 1)
                            if search_end > search_start:
                                sub_win = ref_trace[search_start:search_end]
                                peak_idx = int(np.argmax(np.abs(sub_win)))
                                center_samp = search_start + peak_idx

                        w_start = center_samp - half_win
                        w_end = center_samp + half_win

                        for ch_i, ch in enumerate(channels):
                            c_erp = chan_erps.get(ch)
                            if c_erp is not None and s_idx < c_erp.shape[0]:
                                tr = c_erp[s_idx]
                                pts_avail = len(tr)
                                src_s = max(0, w_start)
                                src_e = min(pts_avail, w_end)
                                dst_s = max(0, -w_start)
                                dst_e = dst_s + (src_e - src_s)
                                if src_e > src_s and dst_e <= window_samples:
                                    snippets[ch_i, ev_i, dst_s:dst_e] = tr[src_s:src_e]

                    mean_matrix = np.mean(snippets, axis=1)
                    base_samples = max(3, int(0.12 * window_samples))
                    for ch_i in range(n_ch):
                        baseline = np.mean(mean_matrix[ch_i, :base_samples])
                        mean_matrix[ch_i, :] -= baseline
                        for ev_i in range(n_events):
                            snippets[ch_i, ev_i, :] -= np.mean(snippets[ch_i, ev_i, :base_samples])

                    ptp_amp = np.zeros(n_ch, dtype=np.float32)
                    lag_ms = np.zeros(n_ch, dtype=np.float32)
                    snr = np.zeros(n_ch, dtype=np.float32)

                    ref_ch_idx = channels.index(ref_chan) if ref_chan in channels else 0
                    ref_peak_idx = int(np.argmax(np.abs(mean_matrix[ref_ch_idx])))

                    for ch_i in range(n_ch):
                        ch_mean = mean_matrix[ch_i]
                        ptp_amp[ch_i] = float(np.max(ch_mean) - np.min(ch_mean))
                        ch_peak_idx = int(np.argmax(np.abs(ch_mean)))
                        lag_ms[ch_i] = float((ch_peak_idx - ref_peak_idx) * (1000.0 / sr))
                        mad = float(np.median(np.abs(ch_mean)))
                        bg_std = (mad / 0.6745) if mad > 1e-9 else float(np.std(ch_mean) or 1.0)
                        if bg_std < 1e-9:
                            bg_std = 1.0
                        snr[ch_i] = float(np.max(np.abs(ch_mean)) / bg_std)

                    max_ptp = float(np.max(ptp_amp)) if len(ptp_amp) > 0 else 0.0
                    detected_channels = [
                        channels[i]
                        for i in range(n_ch)
                        if ptp_amp[i] >= 0.25 * max_ptp and snr[i] >= 2.0
                    ]
                    result = {
                        "channels": channels,
                        "ref_channel": ref_chan,
                        "t_ms": t_ms,
                        "mean_matrix": mean_matrix,
                        "snippets": snippets,
                        "ptp_amp": ptp_amp,
                        "lag_ms": lag_ms,
                        "snr": snr,
                        "detected_channels": detected_channels,
                    }

        if hasattr(self, "footprintRenderReady"):
            self.footprintRenderReady.emit(result)
        return result


class ViewerTaskManager(QObject):
    """High-level manager for submitting and handling viewer tasks."""

    footprintRenderReady = Signal(dict)

    def __init__(self, max_workers: int = 2, parent=None):
        super().__init__(parent)
        self._queue = TaskQueue(LatestWinsCoalescing())

        self._callbacks: Dict[int, Callable[[Any, Optional[str]], None]] = {}
        self._viewer_callbacks: Dict[str, Dict[TaskType, Callable]] = {}
        self._task_metadata: Dict[int, tuple] = {}
        self._metadata_lock = threading.Lock()
        self._task_counter = 0

        worker_count = max(1, int(max_workers or 1))
        self._workers: List[ViewerWorker] = []
        for _ in range(worker_count):
            worker = ViewerWorker(self._queue)
            worker.task_completed.connect(self._on_task_completed)
            worker.footprintRenderReady.connect(self.footprintRenderReady.emit)
            self._workers.append(worker)
        self._worker = self._workers[0]
        for worker in self._workers:
            worker.start()
    
    def register_viewer(self, viewer_id: str, state: ViewerState, callbacks: Dict[TaskType, Callable]):
        """Register a viewer with its result callbacks."""
        for worker in self._workers:
            worker.register_viewer(viewer_id, state)
        with self._metadata_lock:
            self._viewer_callbacks[viewer_id] = callbacks
    
    def unregister_viewer(self, viewer_id: str):
        """Unregister a viewer."""
        for worker in self._workers:
            worker.unregister_viewer(viewer_id)
        with self._metadata_lock:
            self._viewer_callbacks.pop(viewer_id, None)
        self._cleanup_cancelled_metadata(viewer_id=viewer_id)
    
    def submit_task(
        self,
        viewer_id: str,
        task_type: TaskType,
        payload: Dict[str, Any],
        priority: TaskPriority = TaskPriority.NORMAL,
        coalesce_key: str = "",
    ) -> int:
        """Submit a task for background processing."""
        self._task_counter += 1
        task = ViewerTask(
            task_type=task_type,
            priority=priority,
            viewer_id=viewer_id,
            payload=payload,
            task_id=self._task_counter,
            coalesce_key=coalesce_key or task_type.value,
        )
        with self._metadata_lock:
            self._task_metadata[task.task_id] = (viewer_id, task_type)
            task_id = self._queue.enqueue(task)
            if task_id != task.task_id:
                self._task_metadata.pop(task.task_id, None)
                if task_id >= 0:
                    self._task_metadata[task_id] = (viewer_id, task_type)
        return task_id
    
    def submit_render(
        self,
        viewer_id: str,
        task_type: TaskType,
        payload: Dict[str, Any],
        priority: TaskPriority = TaskPriority.NORMAL,
    ) -> int:
        """Submit a render task (auto-coalesces by viewer+type)."""
        return self.submit_task(viewer_id, task_type, payload, priority, coalesce_key=task_type.value)
    
    def submit_template_update(self, viewer_id: str, payload: Dict[str, Any]) -> int:
        """Submit template computation (lower priority, can be delayed)."""
        return self.submit_task(
            viewer_id, TaskType.TEMPLATE_UPDATE, payload, 
            TaskPriority.LOW, coalesce_key="template_update"
        )
    
    def submit_histogram_update(self, viewer_id: str, payload: Dict[str, Any]) -> int:
        """Submit histogram computation."""
        return self.submit_task(
            viewer_id, TaskType.HISTOGRAM_UPDATE, payload,
            TaskPriority.LOW, coalesce_key="histogram"
        )
    
    def submit_right_axis_update(self, viewer_id: str, payload: Dict[str, Any]) -> int:
        """Submit right axis computation."""
        return self.submit_task(
            viewer_id, TaskType.RIGHT_AXIS_UPDATE, payload,
            TaskPriority.LOW, coalesce_key="right_axis"
        )
    
    def submit_spike_group_update(self, viewer_id: str, payload: Dict[str, Any]) -> int:
        """Submit spike group points computation."""
        return self.submit_task(
            viewer_id, TaskType.SPIKE_GROUP_UPDATE, payload,
            TaskPriority.NORMAL, coalesce_key="spike_groups"
        )

    def request_footprint_render(self, payload: dict, priority: int = 0) -> int:
        """Submit footprint render calculation."""
        prio = TaskPriority.NORMAL
        if priority > 1:
            prio = TaskPriority.HIGH
        elif priority < 0:
            prio = TaskPriority.LOW
        viewer_id = str(payload.get("viewer_id", "footprint_viewer"))
        return self.submit_task(
            viewer_id, TaskType.FOOTPRINT_RENDER, payload,
            prio, coalesce_key="footprint_render"
        )
    
    def cancel_viewer_tasks(self, viewer_id: str):
        """Cancel all pending tasks for a viewer."""
        self._queue.cancel_viewer_tasks(viewer_id)
        self._cleanup_cancelled_metadata(viewer_id=viewer_id)
    
    def cancel_task_type(self, viewer_id: str, task_type: TaskType):
        """Cancel specific task type for viewer."""
        self._queue.cancel_task_type(task_type, viewer_id)
        self._cleanup_cancelled_metadata(viewer_id=viewer_id, task_type=task_type)

    def _cleanup_cancelled_metadata(self, viewer_id: str = "", task_type: Optional[TaskType] = None):
        """Best-effort removal of result routing for tasks no longer pending."""
        with self._metadata_lock:
            task_ids = [
                task_id
                for task_id, (meta_viewer, meta_type) in self._task_metadata.items()
                if (not viewer_id or meta_viewer == viewer_id)
                and (task_type is None or meta_type == task_type)
            ]
            for task_id in task_ids:
                self._task_metadata.pop(task_id, None)
                self._callbacks.pop(task_id, None)
    
    def _on_task_completed(self, task_id: int, result: Any, error: str):
        """Handle completed task."""
        with self._metadata_lock:
            metadata = self._task_metadata.pop(task_id, None)

        if metadata is not None:
            viewer_id, task_type = metadata
            with self._metadata_lock:
                callback = self._viewer_callbacks.get(viewer_id, {}).get(task_type)
            if callback is not None:
                try:
                    callback(result, error)
                except Exception:
                    logger.exception(f"Viewer callback failed for {viewer_id}/{task_type}")

        # Also call direct callback if registered
        with self._metadata_lock:
            callback = self._callbacks.pop(task_id, None)
        if callback is not None:
            try:
                callback(result, error)
            except Exception as e:
                logger.exception(f"Task callback failed for {task_id}")
    
    def submit_with_callback(
        self,
        viewer_id: str,
        task_type: TaskType,
        payload: Dict[str, Any],
        callback: Callable[[Any, Optional[str]], None],
        priority: TaskPriority = TaskPriority.NORMAL,
    ) -> int:
        """Submit task with direct callback."""
        self._task_counter += 1
        task = ViewerTask(
            task_type=task_type,
            priority=priority,
            viewer_id=viewer_id,
            payload=payload,
            task_id=self._task_counter,
            coalesce_key=task_type.value,
        )
        with self._metadata_lock:
            self._callbacks[task.task_id] = callback
            self._task_metadata[task.task_id] = (viewer_id, task_type)
            task_id = self._queue.enqueue(task)
            if task_id != task.task_id:
                self._callbacks.pop(task.task_id, None)
                self._task_metadata.pop(task.task_id, None)
                if task_id >= 0:
                    self._callbacks[task_id] = callback
                    self._task_metadata[task_id] = (viewer_id, task_type)
        return task_id
    
    def get_pending_count(self, viewer_id: str = "") -> int:
        return self._queue.get_pending_count(viewer_id)
    
    def shutdown(self):
        """Shutdown the task manager."""
        self._queue.shutdown()
        for worker in self._workers:
            worker.stop()


# Global instance
_task_manager: Optional[ViewerTaskManager] = None


def _shutdown_global_task_manager():
    global _task_manager
    if _task_manager is not None:
        try:
            _task_manager.shutdown()
        except Exception:
            pass
        _task_manager = None


import atexit
atexit.register(_shutdown_global_task_manager)


def get_task_manager() -> ViewerTaskManager:
    global _task_manager
    if _task_manager is None:
        _task_manager = ViewerTaskManager(max_workers=2)
        try:
            from PySide6.QtCore import QCoreApplication
            app = QCoreApplication.instance()
            if app is not None:
                app.aboutToQuit.connect(_shutdown_global_task_manager)
        except Exception:
            pass
    return _task_manager