import numpy as np
import pywt
from neo import AnalogSignal, SpikeTrain
from quantities import second
from scipy import signal
from tqdm import tqdm
from csv import DictReader
import logging
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union
import neo


def wavelet_denoise(in_arr, threshold=0.4, level=None, wlet="sym7", noise_arr=None):
    """
    perform wavelet denoising on an input signal
    """
    if noise_arr is None:
        noise_arr = in_arr
    threshold_vals = calculate_wavelet_denoise_thresholds(
        noise_arr, threshold=threshold, wlet=wlet, level=level
    )
    recon = wavelet_denoise_prespecified(in_arr, threshold_vals, wlet=wlet)
    return recon


def calculate_wavelet_denoise_thresholds(
    in_arr, threshold=1, level=None, wlet="sym7", method="NEW"
):
    """
    calculates the threshold values for the wavelet denoise filter
    method=NEW:
        T=σ * sqrt(2ln(N))
        where σ is std of gaus noise for each level.

    """
    if type(wlet) is str:
        wlet = pywt.Wavelet(wlet)

    coeffs = pywt.wavedec(in_arr, wlet, level=level)
    factor = threshold * np.sqrt(2 * np.log(in_arr.shape[-1]))  # np.log = ln
    coef_thresholds = []
    for i in coeffs[1:]:  # coeffs[0] is approximations (cA)
        if method == "NEW":
            # https://ieeexplore.ieee.org/document/1179130 - although not quite?
            sigma = np.median(np.abs(i)) / 0.6745  # 0.6745 = 75th percentile
            t = sigma * factor
        else:
            t = np.max(
                threshold * max(i)
            )  # This is incorrect... but i cant figure out the original logic.

        coef_thresholds.append(t)

    return coef_thresholds


def wavelet_denoise_prespecified(in_arr, threshold_arr=None, wlet="sym7"):
    """
    A generic class to perform wavelet denoising given an array of thresholds.
    """
    if type(wlet) is str:
        wlet = pywt.Wavelet(wlet)

    coeffs = pywt.wavedec(in_arr, wlet, level=len(threshold_arr))
    coeffs2 = []
    for threshold, c in zip(
        threshold_arr, coeffs[1:]
    ):  # coeffs[0] is approximations (cA)
        coeffs2.append(pywt.threshold(c, threshold, mode="hard"))
    recon = pywt.waverec([coeffs[0], *coeffs2], wlet)
    return recon


def create_erp(signal_chan, idxs, offset=-1000, length=30000):
    arr = np.zeros((len(idxs), length))
    for i, x in enumerate(idxs):
        arr[i].flat[:] = signal_chan[int(x + offset) : int(x + offset + length)]
    return arr


# def create_erp_signals(signal, event, offset, length, channel=0):
#     start = int(np.array(signal.sampling_rate.base) // ((1 / offset)))
#     end = int(np.array(signal.sampling_rate.base) // ((1 / length)))
#     erp = create_erp(
#         signal.as_array()[:, channel],
#         (event.as_array() - signal.t_start.base)
#         * np.array(signal.sampling_rate, dtype=int),
#         start,
#         end,
#     )
#     return erp
import quantities as pq


def create_erp_signals(
    signal_chan,
    idxs,
    offset=-1000 * pq.ms,
    length=30000 * pq.ms,
    channel: int = 0,
    dtype=np.float32,
):
    sr_hz = float(
        signal_chan.sampling_rate.rescale(pq.Hz).magnitude
        if hasattr(signal_chan.sampling_rate, "rescale")
        else float(signal_chan.sampling_rate)
    )
    len_sec = float(
        length.rescale(pq.second).magnitude
        if hasattr(length, "rescale")
        else float(length)
    )
    window_samples = int(np.ceil(len_sec * sr_hz))

    if len(idxs) == 0 or window_samples <= 0:
        return pq.Quantity(
            np.zeros((0, max(0, window_samples)), dtype=dtype),
            units=signal_chan.units,
            dtype=dtype,
        )

    # Fast vectorized path when isinstance(signal_chan, neo.AnalogSignal) and not MultiAnalogSignal
    if (
        isinstance(signal_chan, neo.AnalogSignal)
        and not isinstance(signal_chan, MultiAnalogSignal)
        and hasattr(signal_chan, "magnitude")
    ):
        raw = signal_chan.magnitude
        if raw.ndim == 2:
            ch_idx = min(max(int(channel), 0), raw.shape[1] - 1)
            raw_1d = raw[:, ch_idx]
        else:
            raw_1d = raw.reshape(-1)

        ev_sec = (
            idxs.rescale(pq.s).magnitude
            if hasattr(idxs, "rescale")
            else np.asarray(idxs, dtype=np.float64)
        )
        t_start_s = float(
            signal_chan.t_start.rescale(pq.s).magnitude
            if hasattr(signal_chan.t_start, "rescale")
            else float(signal_chan.t_start)
        )
        offset_s = float(
            offset.rescale(pq.s).magnitude
            if hasattr(offset, "rescale")
            else float(offset)
        )
        start_samples = np.round((ev_sec + offset_s - t_start_s) * sr_hz).astype(np.int64)

        arr = np.zeros((len(start_samples), window_samples), dtype=dtype)
        n_raw = len(raw_1d)

        valid_mask = (start_samples >= 0) & (start_samples + window_samples <= n_raw)

        if np.any(valid_mask):
            idx_2d = (
                start_samples[valid_mask, None]
                + np.arange(window_samples, dtype=np.int64)[None, :]
            )
            arr[valid_mask] = raw_1d[idx_2d].astype(dtype, copy=False)

        # Partially in-bounds edge sweeps
        if not np.all(valid_mask):
            for i in np.where(~valid_mask)[0]:
                s_start = start_samples[i]
                s_end = s_start + window_samples
                c_start = max(0, s_start)
                c_end = min(n_raw, s_end)
                if c_start < c_end:
                    d_start = c_start - s_start
                    d_end = d_start + (c_end - c_start)
                    arr[i, d_start:d_end] = raw_1d[c_start:c_end].astype(dtype, copy=False)

        return pq.Quantity(arr, units=signal_chan.units, dtype=dtype)

    # Fallback path for MultiAnalogSignal or non-contiguous signals
    arr = np.zeros((len(idxs), window_samples), dtype=dtype)
    for i, x in enumerate(idxs):
        try:
            sliced = signal_chan.time_slice(x + offset, x + offset + length)
            arr_slice = (
                sliced.as_array()
                if hasattr(sliced, "as_array")
                else np.asarray(sliced)
            )
            if arr_slice.ndim == 2 and arr_slice.shape[1] > 1:
                ch_idx = min(max(int(channel), 0), arr_slice.shape[1] - 1)
                arr_slice = arr_slice[:, ch_idx]
            n_copy = min(len(arr_slice.flat), window_samples)
            arr[i, :n_copy] = np.asarray(arr_slice.flat)[:n_copy].astype(dtype, copy=False)
        except Exception:
            pass  # handle case where time slice is outside range of signal

    return pq.Quantity(arr, units=signal_chan.units, dtype=dtype)


# a neo class that creates an api like analogsignal for multiple short signals with different start/stop times. they must not overlap.
from neo.core.basesignal import BaseSignal
from typing import List, Optional, Union
from neo.core import Group, Segment


class MultiAnalogSignal:
    proxy_for = AnalogSignal
    segment: Segment = None

    def __init__(self, signals: List[AnalogSignal] = None):
        if signals is None or len(signals) == 0:
            self.signals = []
            self.sampling_rate = 30000 * pq.Hz
            self.units = pq.mV
            self.name = "live"
            self.t_start = 0 * pq.s
            self.t_stop = 0 * pq.s
            self.shape = (0, 0)
            self.t_starts = []
            self.array_annotations = {}
            self.annotations = {}
        else:
            self.signals = list(signals)
            self.t_start = min([x.t_start for x in signals])
            self.t_stop = max([x.t_stop for x in signals])
            self.sampling_rate = signals[0].sampling_rate
            self.units = signals[0].units
            self.shape = (len(signals), signals[0].shape[0])
            self.array_annotations = {}
            self.annotations = {}
            self.name = signals[0].name

            # ensure all the signals have the same sample rate
            for x in signals:
                assert x.sampling_rate == self.sampling_rate

            # ensure that the signals dont overlap
            for i, x in enumerate(signals[:-1]):
                assert x.t_stop <= signals[i + 1].t_start

            self.t_starts = [x.t_start for x in signals]

    def append_signal(self, sig: AnalogSignal):
        if len(self.signals) == 0:
            self.sampling_rate = sig.sampling_rate
            self.units = sig.units
            self.name = sig.name or self.name
            self.t_start = sig.t_start
            self.t_stop = sig.t_stop
        else:
            self.t_stop = max(self.t_stop, sig.t_stop)
        self.signals.append(sig)
        self.t_starts.append(sig.t_start)
        self.shape = (len(self.signals), sig.shape[0])

    def rescale(self, unit):
        if self.units == unit:
            return self
        if len(self.signals) == 0:
            res = MultiAnalogSignal()
            res.sampling_rate = self.sampling_rate
            res.units = unit
            res.name = self.name
            return res
        rescaled_signals = [sig.rescale(unit) for sig in self.signals]
        res = MultiAnalogSignal(rescaled_signals)
        res.name = self.name
        return res

    def time_slice(self, start, stop):
        # find the signals that overlap with the start/stop
        # then slice them and concatenate them
        if len(self.signals) == 0:
            raise ValueError(f"No signal found for the time slice {start}-{stop}")

        sig_start = int(np.searchsorted(self.t_starts, start, side="right") - 1)
        sig_end = int(np.searchsorted(self.t_starts, stop, side="right") - 1)

        if sig_start < 0:
            sig_start = 0
        if sig_end < 0:
            sig_end = 0
        if sig_end >= len(self.signals):
            sig_end = len(self.signals) - 1

        try:
            x = self.signals[sig_start]
            start_clamped = max(x.t_start, min(start, x.t_stop))
            start_idx = x.time_index(start_clamped)

            x2 = self.signals[sig_end]
            stop_clamped = max(x2.t_start, min(stop, x2.t_stop))
            stop_idx = x2.time_index(stop_clamped)

        except (ValueError, IndexError):
            # Handle the case where start or stop is outside the range of x
            raise ValueError(f"No signal found for the time slice {start}-{stop}")

        if sig_end != sig_start:
            logging.warning(
                "region spans mutliple blocks, will return a multianalogsignal"
            )

            sigs = [x[start_idx:]]
            sigs += self.signals[sig_start + 1 : sig_end]
            sigs += [x2[:stop_idx]]

            return MultiAnalogSignal(sigs)

        return x[start_idx:stop_idx]

    @property
    def times(self):
        if len(self.signals) == 0:
            return np.array([]) * pq.s
        return np.concatenate([x.times for x in self.signals])

    @property
    def sampling_period(self):
        """
        Interval between two samples.

        (1/:attr:`sampling_rate`)
        """
        return 1.0 / self.sampling_rate

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, item):
        return self.signals[item]


class LiveERPBuffer:
    """
    A growable 2D numpy buffer for storing stimulus-locked ERP sweeps
    of shape (n_sweeps, window_samples) with units.
    """

    def __init__(
        self,
        window_samples: int,
        initial_capacity: int = 128,
        units=pq.uV,
        dtype=np.float32,
    ):
        self.window_samples = int(window_samples)
        self.units = units
        self.dtype = dtype
        self._count = 0
        self._capacity = max(int(initial_capacity), 16)
        self._buffer = np.zeros((self._capacity, self.window_samples), dtype=self.dtype)
        self._clipped_buffer = np.zeros_like(self._buffer)
        self._row_max = np.zeros(self._capacity, dtype=np.float32)
        self._row_mean = np.zeros(self._capacity, dtype=np.float32)
        self._row_std = np.zeros(self._capacity, dtype=np.float32)

    def append_sweep(self, sweep_1d: Union[np.ndarray, pq.Quantity]):
        if hasattr(sweep_1d, "magnitude"):
            arr = np.asarray(sweep_1d.rescale(self.units).magnitude, dtype=self.dtype).flatten()
        else:
            arr = np.asarray(sweep_1d, dtype=self.dtype).flatten()

        if len(arr) < self.window_samples:
            pad = np.zeros(self.window_samples - len(arr), dtype=self.dtype)
            arr = np.concatenate([arr, pad])
        else:
            arr = arr[: self.window_samples]

        if self._count >= self._capacity:
            new_capacity = self._capacity * 2
            new_buffer = np.zeros((new_capacity, self.window_samples), dtype=self.dtype)
            new_buffer[: self._count, :] = self._buffer[: self._count, :]
            self._buffer = new_buffer

            new_clipped = np.zeros((new_capacity, self.window_samples), dtype=self.dtype)
            new_clipped[: self._count, :] = self._clipped_buffer[: self._count, :]
            self._clipped_buffer = new_clipped

            new_row_max = np.zeros(new_capacity, dtype=np.float32)
            new_row_max[: self._count] = self._row_max[: self._count]
            self._row_max = new_row_max

            new_row_mean = np.zeros(new_capacity, dtype=np.float32)
            new_row_mean[: self._count] = self._row_mean[: self._count]
            self._row_mean = new_row_mean

            new_row_std = np.zeros(new_capacity, dtype=np.float32)
            new_row_std[: self._count] = self._row_std[: self._count]
            self._row_std = new_row_std

            self._capacity = new_capacity

        self._buffer[self._count, :] = arr
        self._clipped_buffer[self._count, :] = np.maximum(arr, 0.0)
        self._row_max[self._count] = float(np.max(arr)) if len(arr) > 0 else 0.0
        self._row_mean[self._count] = float(np.mean(arr)) if len(arr) > 0 else 0.0
        self._row_std[self._count] = float(np.std(arr)) if len(arr) > 0 else 0.0
        self._count += 1

    @property
    def data(self):
        return self._buffer[: self._count, : self.window_samples] * self.units

    @property
    def clipped_data(self):
        return self._clipped_buffer[: self._count, : self.window_samples]

    @property
    def row_max(self):
        return self._row_max[: self._count] * self.units

    @property
    def row_mean(self):
        return self._row_mean[: self._count] * self.units

    @property
    def row_std(self):
        return self._row_std[: self._count]

    def clear(self):
        self._count = 0

    def resize_window(self, window_samples: int):
        new_window = int(window_samples)
        if new_window == self.window_samples:
            return
        new_buffer = np.zeros((self._capacity, new_window), dtype=self.dtype)
        new_clipped = np.zeros((self._capacity, new_window), dtype=self.dtype)
        min_w = min(self.window_samples, new_window)
        if self._count > 0:
            new_buffer[: self._count, :min_w] = self._buffer[: self._count, :min_w]
            new_clipped[: self._count, :min_w] = self._clipped_buffer[: self._count, :min_w]
            # Recalculate stats for the resized window if count > 0
            for i in range(self._count):
                row_arr = new_buffer[i, :new_window]
                self._row_max[i] = float(np.max(row_arr)) if len(row_arr) > 0 else 0.0
                self._row_mean[i] = float(np.mean(row_arr)) if len(row_arr) > 0 else 0.0
                self._row_std[i] = float(np.std(row_arr)) if len(row_arr) > 0 else 0.0
        self.window_samples = new_window
        self._buffer = new_buffer
        self._clipped_buffer = new_clipped

    def __len__(self):
        return self._count

    def __getitem__(self, item):
        return self.data[item]


class BoundedERPCache:
    """Bounded in-memory LRU cache for 2D ERP numpy arrays and quantities.

    Avoids disk temp-file leaks and controls memory footprint with max_entries and max_bytes.
    Can be used directly or as a decorator with .cache_clear().
    """

    def __init__(
        self,
        max_entries: int = 8,
        max_bytes: int = 256 * 1024 * 1024,
    ):
        self.max_entries = int(max_entries)
        self.max_bytes = int(max_bytes)
        self.cache: OrderedDict[str, Any] = OrderedDict()
        self.current_bytes = 0

    def _item_nbytes(self, arr: Any) -> int:
        if hasattr(arr, "nbytes"):
            return int(arr.nbytes)
        if hasattr(arr, "magnitude") and hasattr(arr.magnitude, "nbytes"):
            return int(arr.magnitude.nbytes)
        return 1024

    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key: str, arr: Any):
        if key in self.cache:
            old = self.cache.pop(key)
            self.current_bytes -= self._item_nbytes(old)

        item_bytes = self._item_nbytes(arr)
        self.cache[key] = arr
        self.current_bytes += item_bytes

        while len(self.cache) > self.max_entries or (
            self.current_bytes > self.max_bytes and len(self.cache) > 1
        ):
            k, removed = self.cache.popitem(last=False)
            self.current_bytes -= self._item_nbytes(removed)

    def clear(self):
        self.cache.clear()
        self.current_bytes = 0

    def cache_clear(self):
        self.clear()

    def __call__(self, func=None, keyfunc=None):
        def decorator(f):
            if keyfunc is None:
                def default_k(*args, **kwargs):
                    return ";".join(
                        [str(hash(a)) for a in args]
                        + [f"{k}={hash(v)}" for k, v in kwargs.items()]
                    )
                k_func = default_k
            else:
                k_func = keyfunc

            def wrapper(*args, **kwargs):
                k = k_func(*args, **kwargs)
                hit = self.get(k)
                if hit is not None:
                    return hit
                res = f(*args, **kwargs)
                self.put(k, res)
                return res

            wrapper.cache_clear = self.clear
            wrapper.cache = self
            return wrapper

        if func is not None:
            return decorator(func)
        return decorator


class MultiChannelTemplateMatcher:
    """High-performance, ultra-low-memory Multi-Channel Spatio-Temporal Template Matcher.

    Computes normalized templates across active and neighbor channels, calculates
    SNR-weighted composite convolution scores on-the-fly without allocating
    large full-recording convolution matrices, and caches single-sweep scores in a bounded cache.
    """

    def __init__(
        self,
        pre_s: float = 0.01,
        post_s: float = 0.02,
        max_template_spikes: int = 50,
    ):
        self.pre_s = float(pre_s)
        self.post_s = float(post_s)
        self.max_template_spikes = int(max_template_spikes)
        self.templates: Dict[str, np.ndarray] = {}
        self.weights: Dict[str, float] = {}
        self.version = 0
        self._sweep_cache: OrderedDict[Tuple[int, Tuple[str, ...]], np.ndarray] = OrderedDict()
        self._sweep_cache_capacity = 32

    def build_template(self, state, sg, channels: List[str] = None) -> bool:
        if state is None or sg is None or state.analog_signal is None:
            return False

        if channels is None or len(channels) == 0:
            primary_name = getattr(state.analog_signal, "name", "CH0") or "CH0"
            channels = [primary_name]

        sr = float(
            state.sampling_rate
            if state.sampling_rate is not None
            else state.analog_signal.sampling_rate.rescale(pq.Hz).magnitude
        )
        pre_samples = int(np.round(self.pre_s * sr))
        post_samples = int(np.round(self.post_s * sr))
        template_len = pre_samples + post_samples

        if template_len <= 0:
            return False

        # Get valid sweep indices and sample latencies
        if hasattr(sg, "get_valid_sweep_indices_and_samples") and state.event_signal is not None:
            sweeps, samples = sg.get_valid_sweep_indices_and_samples(
                state.event_signal, sr, state.window_size
            )
        elif hasattr(sg, "idx_arr") and sg.idx_arr is not None:
            pairs = [
                (i, (x[0] if isinstance(x, (tuple, list)) else x))
                for i, x in enumerate(sg.idx_arr)
                if x is not None
            ]
            if pairs:
                sweeps = np.array([p[0] for p in pairs], dtype=np.int64)
                samples = np.array([p[1] for p in pairs], dtype=np.int64)
            else:
                sweeps = np.empty(0, dtype=np.int64)
                samples = np.empty(0, dtype=np.int64)
        else:
            return False

        if len(sweeps) == 0:
            return False

        # Use at most the last max_template_spikes
        if len(sweeps) > self.max_template_spikes:
            sweeps = sweeps[-self.max_template_spikes :]
            samples = samples[-self.max_template_spikes :]

        new_templates: Dict[str, np.ndarray] = {}
        new_weights: Dict[str, float] = {}
        ptps: Dict[str, float] = {}

        for ch in channels:
            ch_erp = state.get_channel_erp(ch)
            if ch_erp is None:
                continue
            erp_arr = ch_erp.magnitude if hasattr(ch_erp, "magnitude") else np.asarray(ch_erp)
            if erp_arr.ndim != 2 or erp_arr.shape[0] == 0:
                continue

            n_sweeps, n_samples = erp_arr.shape
            snippets = []
            for sw_idx, samp in zip(sweeps, samples):
                if 0 <= sw_idx < n_sweeps:
                    start_s = samp - pre_samples
                    end_s = samp + post_samples
                    if 0 <= start_s and end_s <= n_samples:
                        snip = erp_arr[sw_idx, start_s:end_s].astype(np.float32, copy=False)
                        snippets.append(snip)
                    else:
                        snip = np.zeros(template_len, dtype=np.float32)
                        c_start = max(0, start_s)
                        c_end = min(n_samples, end_s)
                        if c_start < c_end:
                            d_start = c_start - start_s
                            d_end = d_start + (c_end - c_start)
                            snip[d_start:d_end] = erp_arr[sw_idx, c_start:c_end]
                        snippets.append(snip)

            if not snippets:
                continue

            mean_wf = np.mean(snippets, axis=0).astype(np.float32)
            ptp_val = float(np.ptp(mean_wf))
            std_val = float(np.std(mean_wf))

            if std_val > 1e-9:
                norm_wf = (mean_wf - np.mean(mean_wf)) / std_val
            else:
                norm_wf = mean_wf - np.mean(mean_wf)

            new_templates[ch] = norm_wf.astype(np.float32)
            ptps[ch] = max(ptp_val, 1e-6)

        if not new_templates:
            return False

        total_ptp = sum(ptps.values())
        for ch in new_templates:
            new_weights[ch] = ptps[ch] / total_ptp if total_ptp > 0 else 1.0 / len(new_templates)

        self.templates = new_templates
        self.weights = new_weights
        self.version += 1
        self._sweep_cache.clear()
        return True

    def score_sweep(
        self,
        state,
        stimno: int,
        channels: List[str] = None,
        use_correlate: bool = False,
    ) -> Optional[np.ndarray]:
        if not self.templates:
            return None

        if channels is None or len(channels) == 0:
            channels = [next(iter(self.templates.keys()))]

        active_chans = [ch for ch in channels if ch in self.templates]
        if not active_chans:
            return None

        cache_key = (int(stimno), tuple(active_chans))
        if cache_key in self._sweep_cache:
            self._sweep_cache.move_to_end(cache_key)
            return self._sweep_cache[cache_key]

        composite_score = None
        weights_sum = 0.0

        for ch in active_chans:
            ch_erp = state.get_channel_erp(ch)
            if ch_erp is None:
                continue
            erp_arr = ch_erp.magnitude if hasattr(ch_erp, "magnitude") else np.asarray(ch_erp)
            if erp_arr.ndim != 2 or stimno < 0 or stimno >= erp_arr.shape[0]:
                continue

            sweep_data = erp_arr[stimno].astype(np.float32, copy=False)
            std_sw = float(np.std(sweep_data))
            if std_sw > 1e-9:
                sweep_norm = (sweep_data - np.mean(sweep_data)) / std_sw
            else:
                sweep_norm = sweep_data - np.mean(sweep_data)

            tmpl_norm = self.templates[ch]
            # Fast FFT convolution
            if use_correlate:
                conv = signal.fftconvolve(sweep_norm, tmpl_norm[::-1], mode="same")
            else:
                conv = signal.fftconvolve(sweep_norm, tmpl_norm, mode="same")

            L = len(tmpl_norm)
            if L < len(conv):
                conv[:L] = 0.0
                conv[-L:] = 0.0

            w = self.weights.get(ch, 1.0)
            if composite_score is None:
                composite_score = conv.astype(np.float32) * w
            else:
                composite_score += conv.astype(np.float32) * w
            weights_sum += w

        if composite_score is None:
            return None

        if weights_sum > 0:
            composite_score /= float(weights_sum)

        self._sweep_cache[cache_key] = composite_score
        while len(self._sweep_cache) > self._sweep_cache_capacity:
            self._sweep_cache.popitem(last=False)

        return composite_score

    def find_candidate_peaks(
        self,
        score_1d: np.ndarray,
        top_k: int = 100,
        percentile: float = 99.0,
    ) -> np.ndarray:
        if score_1d is None or len(score_1d) == 0:
            return np.empty(0, dtype=np.int64)

        n = len(score_1d)
        margin = min(1000, max(1, n // 10))

        if margin * 2 >= n:
            margin = 0

        eval_region = score_1d[margin : n - margin] if margin > 0 else score_1d
        if len(eval_region) == 0:
            return np.empty(0, dtype=np.int64)

        thresh = float(np.percentile(eval_region, percentile))
        peaks, props = signal.find_peaks(score_1d, height=thresh)

        if len(peaks) == 0:
            return np.empty(0, dtype=np.int64)

        if margin > 0:
            valid_mask = (peaks >= margin) & (peaks < n - margin)
            peaks = peaks[valid_mask]
            if len(peaks) == 0:
                return np.empty(0, dtype=np.int64)

        peak_heights = score_1d[peaks]
        sort_order = np.argsort(peak_heights)[::-1]
        top_peaks = peaks[sort_order][:top_k]
        return top_peaks.astype(np.int64)


class OnlineBandpassFilter:
    """Causal online bandpass filter using Second-Order Sections (SOS) to process streaming chunks
    without edge discontinuities across chunk boundaries.
    """

    def __init__(
        self,
        lowHz: Optional[float] = None,
        highHz: Optional[float] = None,
        sample_rate: Optional[float] = None,
        order: int = 2,
        lowcut: Optional[float] = None,
        highcut: Optional[float] = None,
        fs: Optional[float] = None,
        sampling_rate: Optional[float] = None,
        num_channels: Optional[int] = None,
    ):
        low_val = lowHz if lowHz is not None else (lowcut if lowcut is not None else 500.0)
        high_val = highHz if highHz is not None else (highcut if highcut is not None else 7000.0)
        sr_val = (
            sample_rate
            if sample_rate is not None
            else (
                sampling_rate
                if sampling_rate is not None
                else (fs if fs is not None else 30000.0)
            )
        )

        self.lowHz = float(low_val)
        self.highHz = float(high_val)
        self.sample_rate = float(sr_val)
        self.order = int(order)
        self.lowcut = self.lowHz
        self.highcut = self.highHz
        self.fs = self.sample_rate

        self.sos = signal.butter(
            self.order,
            [self.lowHz, self.highHz],
            btype="bandpass",
            fs=self.sample_rate,
            output="sos",
        )
        self.zi_init = signal.sosfilt_zi(self.sos)
        self.zi = self.zi_init.copy()
        if num_channels is not None and num_channels > 1:
            zi_base = signal.sosfilt_zi(self.sos)
            self.zi = np.repeat(zi_base[:, np.newaxis, :], num_channels, axis=1)

    def process(self, chunk: np.ndarray) -> np.ndarray:
        if chunk.size == 0:
            return chunk.astype(np.float32) if hasattr(chunk, "astype") else chunk
        is_qty = hasattr(chunk, "magnitude")
        arr = chunk.magnitude if is_qty else chunk

        if arr.ndim == 1:
            if self.zi is None or self.zi.ndim != 2:
                self.zi = signal.sosfilt_zi(self.sos) * (arr[0] if len(arr) > 0 else 0.0)
        elif arr.ndim == 2:
            n_channels = arr.shape[0]
            if self.zi is None or self.zi.ndim != 3 or self.zi.shape[1] != n_channels:
                zi_base = signal.sosfilt_zi(self.sos)[:, np.newaxis, :]
                first_col = arr[:, 0:1] if arr.shape[1] > 0 else np.zeros((n_channels, 1))
                self.zi = zi_base * first_col

        filtered, self.zi = signal.sosfilt(self.sos, arr, axis=-1, zi=self.zi)
        return filtered * chunk.units if is_qty else filtered

    def reset(self, num_channels: Optional[int] = None):
        if num_channels is not None and num_channels > 1:
            zi_base = signal.sosfilt_zi(self.sos)
            self.zi = np.repeat(zi_base[:, np.newaxis, :], num_channels, axis=1)
        else:
            self.zi = signal.sosfilt_zi(self.sos)


class TriggerEvent(int):
    """Integer sample index subclass that also acts as a dictionary for backwards compatibility."""

    def __new__(cls, val, data_dict):
        obj = super().__new__(cls, int(val))
        obj.data = data_dict
        return obj

    def __getitem__(self, key):
        return self.data[key]

    def __contains__(self, key):
        return key in self.data

    def get(self, key, default=None):
        return self.data.get(key, default)

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    def __iter__(self):
        return iter(self.data)


class OnlineTTLDetector:
    """Stateful rising-edge detector for 1D streaming analog chunks.
    Detects threshold-crossing pulses and enforces min width and refractory period.
    """

    def __init__(
        self,
        threshold: float = 2.5,
        min_width_samples: int = 1,
        refractory_samples: Optional[int] = None,
        refractory_s: float = 0.100,
        refractory_ms: Optional[float] = None,
        min_interval_samples: Optional[int] = None,
    ):
        self.threshold = float(threshold)
        self.min_width_samples = int(min_width_samples)
        if refractory_ms is not None:
            self.refractory_s = float(refractory_ms) / 1000.0
        else:
            self.refractory_s = float(refractory_s)

        if min_interval_samples is not None:
            self.refractory_samples = int(min_interval_samples)
        else:
            self.refractory_samples = refractory_samples
        self.reset()

    def reset(self):
        self.in_pulse = False
        self.pulse_start_sample = None
        self.pulse_max_val = -float("inf")
        self.pulse_width = 0
        self.last_trigger_sample = None

    def process_chunk(
        self,
        chunk: np.ndarray,
        start_sample: Optional[int] = None,
        sample_rate: Optional[float] = None,
        start_sample_index: Optional[int] = None,
    ) -> List[Any]:
        if start_sample is None:
            start_sample = start_sample_index if start_sample_index is not None else 0
        if sample_rate is None:
            sample_rate = 30000.0

        chunk_arr = np.asarray(
            chunk.magnitude if hasattr(chunk, "magnitude") else chunk
        ).flatten()

        if self.refractory_samples is not None:
            refractory = self.refractory_samples
        else:
            refractory = int(self.refractory_s * sample_rate)

        triggers = []
        for i, val in enumerate(chunk_arr):
            cur_sample = start_sample + i
            if val >= self.threshold:
                if not self.in_pulse:
                    self.in_pulse = True
                    self.pulse_start_sample = cur_sample
                    self.pulse_max_val = float(val)
                    self.pulse_width = 1
                else:
                    self.pulse_width += 1
                    if val > self.pulse_max_val:
                        self.pulse_max_val = float(val)
            else:
                if self.in_pulse:
                    duration_samples = cur_sample - self.pulse_start_sample
                    if duration_samples >= self.min_width_samples:
                        if (
                            self.last_trigger_sample is None
                            or (self.pulse_start_sample - self.last_trigger_sample)
                            >= refractory
                        ):
                            self.last_trigger_sample = self.pulse_start_sample
                            info = {
                                "sample_index": int(self.pulse_start_sample),
                                "sample": int(self.pulse_start_sample),
                                "timestamp_s": float(
                                    self.pulse_start_sample / sample_rate
                                ),
                                "timestamp": float(
                                    self.pulse_start_sample / sample_rate
                                ),
                                "duration_s": float(duration_samples / sample_rate),
                                "max_val": float(self.pulse_max_val),
                            }
                            triggers.append(
                                TriggerEvent(self.pulse_start_sample, info)
                            )
                    self.in_pulse = False
                    self.pulse_start_sample = None
                    self.pulse_max_val = -float("inf")
                    self.pulse_width = 0

        return triggers


from neo import Event
import quantities as pq
import warnings


# function to create multianalogsignal like erp
def create_neo_multianalogsignal(
    signal_chan: AnalogSignal, idxs: Event, offset=-1, length=2
):

    # if the offset or length is not a pyquantity assume it is seconds otherwise rescale it to seconds
    if not hasattr(offset, "rescale"):
        offset = offset * pq.second

    if not hasattr(length, "rescale"):
        length = length * pq.second

    offset = offset.rescale("s")
    length = length.rescale("s")

    # create a list of signals
    signals = []
    # create the (start, stop) for segments, where start= idx+offset and end=idx+offset+length ensuring there are no overlaps of idx-offset and idx-offset+length.
    regions = []
    start = idxs[0] + offset
    stop = idxs[0] + offset + length
    for i in range(1, len(idxs)):
        next_start = idxs[i] + offset
        if next_start < stop:
            stop = next_start + offset + length
        else:
            regions.append((start, stop))
            start = next_start
            stop = next_start + offset + length
    regions.append((start, stop))

    for start, stop in regions:
        if start < signal_chan.t_start or stop > signal_chan.t_stop:
            warnings.warn("start and/or stop are outside the range of signal_chan")
            stop = min(signal_chan.t_stop, stop)
            start = max(signal_chan.t_start, start)
        if start == stop:
            warnings.warn("start and stop are the same, skipping")
            continue

        signals.append(signal_chan.time_slice(start, stop).copy())

    return MultiAnalogSignal(signals)


def read_spike_csv(open_filename):
    with open(open_filename, "r") as f:
        r = DictReader(f)
        try:
            k = next(x for x in r.fieldnames if "timestamp" in x.lower())
        except StopIteration:
            import logging

            logging.warn("'timestamp' column not found")
            return

        unit = pq.ms
        out = {}
        for row in r:
            spikeid = str(row.get("SpikeID", "0"))
            out[spikeid] = out.get(spikeid, []) + [row[k]]

        out2 = []
        for k, v in out.items():
            if k.isdigit():
                spikeid = "unit_" + k
            else:
                spikeid = k

            timestamps = neo.Event(
                np.array(v, dtype=np.float64), units=unit, name=spikeid, type="unit"
            )
            out2.append(timestamps.rescale(pq.s))
    return out2
