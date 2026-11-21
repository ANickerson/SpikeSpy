"""Channel Purpose & Preprocessing Configuration View for SpikeSpy.

Provides a clean, reloadable UI widget (``ChannelPurposeWidget``) and modal
dialog (``ChannelPurposeDialog``) for assigning channel roles (probe, stimulus
triggers, thermode, potentiometer, button, auxiliary channels) and configuring
preprocessing (bandpass filtering, TTL pulse extraction, ADC scaling) for both
offline recordings (APTrack, Open Ephys, NIX, NWB, etc.) and live streaming.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import neo
import numpy as np
import quantities as pq
from PySide6.QtCore import QSettings, QSignalBlocker, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .APTrack_experiment_import import (
    APTrackRecording,
    TypeID,
    apply_bandpass,
    as_neo as aptrack_as_neo,
)

try:
    from .APTrack_experiment_import import (
        ChannelRoleSpec,
        preprocess_segment_with_roles as _core_preprocess_segment_with_roles,
    )
except ImportError:
    ChannelRoleSpec = None
    _core_preprocess_segment_with_roles = None

logger = logging.getLogger(__name__)


MODE_ANALOG = "analog"
MODE_ANALOG_BP = "analog_bp"
MODE_TTL = "ttl"

MODE_OPTIONS: List[Tuple[str, str]] = [
    ("Analog (Raw)", MODE_ANALOG),
    ("Analog + Bandpass (.bp)", MODE_ANALOG_BP),
    ("TTL -> Event Pulses", MODE_TTL),
]


PURPOSE_PRESETS: Dict[str, Dict[str, Any]] = {
    "probe": {
        "label": "Probe / Recording (rd.0)",
        "default_target_name": "rd.0",
        "probe_type": "AS",
        "bandpass": True,
        "bandpass_low": 500.0,
        "bandpass_high": 7000.0,
        "is_adc": False,
    },
    "stim_trigger": {
        "label": "Stimulus Trigger TTL (env.stim)",
        "default_target_name": "env.stim",
        "probe_type": "TTL",
        "bandpass": False,
        "is_adc": True,
    },
    "stim_voltage": {
        "label": "Stimulus Voltage TTL (env.stimVolt)",
        "default_target_name": "env.stimVolt",
        "probe_type": "TTL",
        "bandpass": False,
        "is_adc": True,
    },
    "potentiometer": {
        "label": "Potentiometer (env.potentiometer)",
        "default_target_name": "env.potentiometer",
        "probe_type": "AS",
        "bandpass": False,
        "is_adc": True,
    },
    "stim_analog": {
        "label": "Stimulus Analog Trace (env.stim)",
        "default_target_name": "env.stim",
        "probe_type": "AS",
        "bandpass": False,
        "is_adc": True,
    },
    "thermode": {
        "label": "Thermode Temperature (env.thermode)",
        "default_target_name": "env.thermode",
        "probe_type": "AS",
        "bandpass": False,
        "is_adc": True,
    },
    "button": {
        "label": "Manual Button / Marker (rec.button)",
        "default_target_name": "rec.button",
        "probe_type": "TTL",
        "bandpass": False,
        "is_adc": True,
    },
    "aux": {
        "label": "Custom Auxiliary Analog",
        "default_target_name": "env.aux",
        "probe_type": "AS",
        "bandpass": False,
        "is_adc": True,
    },
    "ttl_custom": {
        "label": "Custom TTL Event",
        "default_target_name": "evt.custom",
        "probe_type": "TTL",
        "bandpass": False,
        "is_adc": True,
    },
}


SETUP_PRESETS: Dict[str, List[Dict[str, Any]]] = {
    "APTrack Default Setup": [
        {
            "enabled": True,
            "purpose": "probe",
            "source_channel": "CH1",
            "target_name": "rd.0",
            "mode": MODE_ANALOG_BP,
            "probe_type": "AS",
            "bandpass": True,
            "bandpass_low": 500.0,
            "bandpass_high": 7000.0,
            "is_adc": False,
        },
        {
            "enabled": True,
            "purpose": "stim_voltage",
            "source_channel": "ADC4",
            "target_name": "env.stimVolt",
            "mode": MODE_TTL,
            "probe_type": "TTL",
            "bandpass": False,
            "is_adc": True,
        },
        {
            "enabled": True,
            "purpose": "stim_trigger",
            "source_channel": "ADC4",
            "target_name": "env.stim",
            "mode": MODE_TTL,
            "probe_type": "TTL",
            "bandpass": False,
            "is_adc": True,
        },
        {
            "enabled": True,
            "purpose": "potentiometer",
            "source_channel": "ADC2",
            "target_name": "env.potentiometer",
            "mode": MODE_ANALOG,
            "probe_type": "AS",
            "bandpass": False,
            "is_adc": True,
        },
        {
            "enabled": True,
            "purpose": "stim_analog",
            "source_channel": "ADC5",
            "target_name": "env.stim",
            "mode": MODE_ANALOG,
            "probe_type": "AS",
            "bandpass": False,
            "is_adc": True,
        },
        {
            "enabled": True,
            "purpose": "thermode",
            "source_channel": "ADC7",
            "target_name": "env.thermode",
            "mode": MODE_ANALOG,
            "probe_type": "AS",
            "bandpass": False,
            "is_adc": True,
        },
        {
            "enabled": True,
            "purpose": "button",
            "source_channel": "ADC8",
            "target_name": "rec.button",
            "mode": MODE_TTL,
            "probe_type": "TTL",
            "bandpass": False,
            "is_adc": True,
        },
    ],
    "Single Probe + TTL": [
        {
            "enabled": True,
            "purpose": "probe",
            "source_channel": "CH0",
            "target_name": "rd.0",
            "mode": MODE_ANALOG_BP,
            "probe_type": "AS",
            "bandpass": True,
            "bandpass_low": 500.0,
            "bandpass_high": 7000.0,
            "is_adc": False,
        },
        {
            "enabled": True,
            "purpose": "stim_trigger",
            "source_channel": "ADC1",
            "target_name": "env.stim",
            "mode": MODE_TTL,
            "probe_type": "TTL",
            "bandpass": False,
            "is_adc": True,
        },
    ],
    "Multi-Channel Tetrode (CH0-CH3)": [
        {
            "enabled": True,
            "purpose": "probe",
            "source_channel": "CH0",
            "target_name": "rd.0",
            "mode": MODE_ANALOG_BP,
            "probe_type": "AS",
            "bandpass": True,
            "bandpass_low": 500.0,
            "bandpass_high": 7000.0,
            "is_adc": False,
        },
        {
            "enabled": True,
            "purpose": "probe",
            "source_channel": "CH1",
            "target_name": "rd.1",
            "mode": MODE_ANALOG_BP,
            "probe_type": "AS",
            "bandpass": True,
            "bandpass_low": 500.0,
            "bandpass_high": 7000.0,
            "is_adc": False,
        },
        {
            "enabled": True,
            "purpose": "probe",
            "source_channel": "CH2",
            "target_name": "rd.2",
            "mode": MODE_ANALOG_BP,
            "probe_type": "AS",
            "bandpass": True,
            "bandpass_low": 500.0,
            "bandpass_high": 7000.0,
            "is_adc": False,
        },
        {
            "enabled": True,
            "purpose": "probe",
            "source_channel": "CH3",
            "target_name": "rd.3",
            "mode": MODE_ANALOG_BP,
            "probe_type": "AS",
            "bandpass": True,
            "bandpass_low": 500.0,
            "bandpass_high": 7000.0,
            "is_adc": False,
        },
        {
            "enabled": True,
            "purpose": "stim_trigger",
            "source_channel": "ADC1",
            "target_name": "env.stim",
            "mode": MODE_TTL,
            "probe_type": "TTL",
            "bandpass": False,
            "is_adc": True,
        },
    ],
}


STANDARD_CHANNEL_CHOICES: List[str] = (
    [f"CH{i}" for i in range(1, 17)]
    + [f"CH{i}" for i in range(0, 16)]
    + [f"ADC{i}" for i in range(1, 9)]
)


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        s = str(item).strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _infer_purpose_from_target(target_name: str, probe_type: str = "AS") -> str:
    t = str(target_name or "").strip().lower()
    pt = str(probe_type or "AS").strip().upper()
    if t.startswith("rd"):
        return "probe"
    if t == "env.stimvolt" or t == "stimvolt":
        return "stim_voltage"
    if t == "env.potentiometer" or "pot" in t:
        return "potentiometer"
    if t == "env.thermode" or "thermode" in t or "temp" in t:
        return "thermode"
    if t == "rec.button" or "button" in t:
        return "button"
    if t == "env.stim" or t == "stim":
        return "stim_trigger" if pt == "TTL" else "stim_analog"
    if pt == "TTL":
        return "ttl_custom"
    return "aux"


def _to_role_dict_for_state(role: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a UI role dict into a dict compatible with ChannelRoleSpec and UI."""
    norm = normalize_role(role)
    thresh_v = float(norm.get("ttl_threshold_v", 0.0))
    return {
        **norm,
        # ChannelRoleSpec keys
        "source": norm["source_channel"],
        "name": norm["target_name"],
        "ttl_threshold": thresh_v if thresh_v > 0.0 else None,
        "ttl_std_mult": float(norm.get("ttl_sd_multiplier", 10.0)),
        "ttl_min_width_ms": float(norm.get("ttl_min_width_ms", 0.4)),
        "ttl_refractory_ms": float(norm.get("ttl_refractory_ms", 100.0)),
        "details": norm.get("description", ""),
    }


def normalize_role(role: Any) -> Dict[str, Any]:
    """Convert a role dict or ``ChannelRoleSpec`` into a canonical UI role dictionary."""
    if isinstance(role, dict):
        getter = role.get
    else:
        getter = lambda k, default=None: getattr(role, k, default)

    purpose = str(getter("purpose", "") or getter("role", "") or "").strip()
    target_name = str(
        getter("target_name", None)
        or getter("name", None)
        or getter("target", "")
        or ""
    ).strip()
    raw_pt = getter("probe_type", None) or getter("type", None)
    if hasattr(raw_pt, "value"):
        raw_pt = raw_pt.value
    probe_type = str(raw_pt or "").strip().upper()

    if not purpose or purpose not in PURPOSE_PRESETS:
        purpose = _infer_purpose_from_target(target_name, probe_type or "AS")

    preset = PURPOSE_PRESETS[purpose]
    if not target_name:
        target_name = preset["default_target_name"]
    if not probe_type:
        probe_type = preset["probe_type"]

    mode = str(getter("mode", "") or "").strip().lower()
    bandpass_val = getter("bandpass", None)
    if mode not in (MODE_ANALOG, MODE_ANALOG_BP, MODE_TTL):
        if probe_type == "TTL":
            mode = MODE_TTL
        elif bandpass_val is True or (bandpass_val is None and preset.get("bandpass", False)):
            mode = MODE_ANALOG_BP
        else:
            mode = MODE_ANALOG

    if mode == MODE_TTL:
        probe_type = "TTL"
        bandpass = False
    elif mode == MODE_ANALOG_BP:
        probe_type = "AS"
        bandpass = True
    else:
        probe_type = "AS"
        bandpass = False

    source_channel = str(
        getter("source_channel", None)
        or getter("source", None)
        or getter("channel", None)
        or getter("filename", "")
        or "CH1"
    ).strip()

    bp_low = float(
        getter("bandpass_low", None)
        if getter("bandpass_low", None) is not None
        else preset.get("bandpass_low", 500.0)
    )
    bp_high = float(
        getter("bandpass_high", None)
        if getter("bandpass_high", None) is not None
        else preset.get("bandpass_high", 7000.0)
    )

    raw_thresh = getter("ttl_threshold_v", None)
    if raw_thresh is None:
        raw_thresh = getter("ttl_threshold", None)
    ttl_thresh = float(raw_thresh) if raw_thresh is not None else 0.0

    raw_sd = getter("ttl_sd_multiplier", None)
    if raw_sd is None:
        raw_sd = getter("ttl_std_mult", None)
    ttl_sd = float(raw_sd) if raw_sd is not None else 10.0

    raw_refrac = getter("ttl_refractory_ms", None)
    ttl_refrac = float(raw_refrac) if raw_refrac is not None else 100.0

    raw_width = getter("ttl_min_width_ms", None)
    ttl_min_width = float(raw_width) if raw_width is not None else 0.4

    raw_is_adc = getter("is_adc", None)
    is_adc = bool(preset.get("is_adc", True) if raw_is_adc is None else raw_is_adc)

    enabled = bool(True if getter("enabled", None) is None else getter("enabled"))
    description = str(
        getter("description", None)
        or getter("details", None)
        or preset["label"]
    )

    return {
        "enabled": enabled,
        "purpose": purpose,
        "source_channel": source_channel,
        "source": source_channel,
        "target_name": target_name,
        "name": target_name,
        "mode": mode,
        "probe_type": probe_type,
        "bandpass": bandpass,
        "bandpass_low": bp_low,
        "bandpass_high": bp_high,
        "ttl_threshold_v": ttl_thresh,
        "ttl_threshold": ttl_thresh if ttl_thresh > 0.0 else None,
        "ttl_sd_multiplier": ttl_sd,
        "ttl_std_mult": ttl_sd,
        "ttl_refractory_ms": ttl_refrac,
        "ttl_min_width_ms": ttl_min_width,
        "is_adc": is_adc,
        "description": description,
        "details": description,
    }


def format_role_parameters(role: Dict[str, Any]) -> str:
    """Return a concise inline parameter summary string for a role row."""
    mode = role.get("mode", MODE_ANALOG)
    if mode == MODE_ANALOG_BP:
        low = float(role.get("bandpass_low", 500.0))
        high = float(role.get("bandpass_high", 7000.0))
        adc_tag = " | ADC (V)" if role.get("is_adc", False) else ""
        return f"{low:g}-{high:g} Hz{adc_tag}"
    if mode == MODE_TTL:
        thresh = float(role.get("ttl_threshold_v", 0.0))
        sd_mult = float(role.get("ttl_sd_multiplier", 10.0))
        refrac = float(role.get("ttl_refractory_ms", 100.0))
        min_w = float(role.get("ttl_min_width_ms", 0.4))
        t_str = f"Auto ({sd_mult:g} SD)" if thresh <= 0.0 else f"{thresh:g}V"
        return f"{t_str} / {refrac:g}ms (w>={min_w:g}ms)"
    return "ADC (V)" if role.get("is_adc", True) else "Raw (uV)"


def _scan_folder_channels(folder: Optional[Union[str, Path]]) -> List[str]:
    """Inspect a recording folder for available continuous channel names."""
    if not folder:
        return []
    p = Path(folder)
    if not p.exists() or not p.is_dir():
        return []

    found: List[str] = []
    try:
        cont_files = sorted(p.glob("*.continuous"))
    except OSError:
        cont_files = []

    for f in cont_files:
        stem = f.stem
        m = re.search(r"_((?:CH|ADC|AUX)\d+)(?:_\d+)?$", stem, re.IGNORECASE)
        if m:
            found.append(m.group(1).upper())
            continue
        m_num = re.search(r"_(\d+)(?:_\d+)?$", stem)
        if m_num:
            num = int(m_num.group(1))
            if 17 <= num <= 24:
                found.append(f"ADC{num - 16}")
            else:
                found.append(f"CH{num}")
            continue
        found.append(stem)

    oebin = p / "structure.oebin"
    if oebin.exists():
        try:
            meta = json.loads(oebin.read_text(encoding="utf-8"))
            for cont in meta.get("continuous", []) or []:
                for ch in cont.get("channels", []) or []:
                    ch_name = ch.get("channel_name")
                    if ch_name:
                        found.append(str(ch_name))
        except Exception:
            pass

    return _dedupe_preserve_order(found)


def detect_channels_from_state(
    state: Any = None, folder: Optional[Union[str, Path]] = None
) -> List[str]:
    """Gather available/detected channel names from ViewerState and/or folder."""
    detected: List[str] = []

    if folder is not None:
        detected.extend(_scan_folder_channels(folder))

    if state is not None:
        loaded_path = getattr(state, "_loaded_file_path", None)
        if loaded_path and Path(loaded_path).is_dir():
            detected.extend(_scan_folder_channels(loaded_path))

        seg = getattr(state, "segment", None)
        if seg is not None:
            orig_folder = (getattr(seg, "annotations", None) or {}).get(
                "original_folder"
            )
            if orig_folder:
                detected.extend(_scan_folder_channels(orig_folder))
            for sig in getattr(seg, "analogsignals", []) or []:
                for key in ("raw_channel_name", "source_channel"):
                    val = (getattr(sig, "annotations", None) or {}).get(key)
                    if val:
                        detected.append(str(val))
                name = getattr(sig, "name", None)
                if name:
                    detected.append(str(name))

        if hasattr(state, "get_available_channels"):
            try:
                detected.extend(state.get_available_channels() or [])
            except Exception:
                pass

        asig = getattr(state, "analog_signal", None)
        if asig is not None and getattr(asig, "name", None):
            detected.append(str(asig.name))

    detected.extend(STANDARD_CHANNEL_CHOICES)
    return _dedupe_preserve_order(detected)


def _find_square_pulses_safe(
    arr: np.ndarray, min_width_samples: int, threshold: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized rising-edge square pulse detector safe against edge cases."""
    arr = np.asarray(arr, dtype=np.float64).ravel()
    if arr.size < 2:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
        )
    min_w = max(1, int(min_width_samples))
    up = arr > float(threshold)
    transitions_up = np.where(~up[:-1] & up[1:])[0] + 1
    transitions_down = np.where(up[:-1] & ~up[1:])[0] + 1

    if transitions_up.size == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
        )

    starts: List[int] = []
    ends: List[int] = []
    maxes: List[float] = []
    down_idx = 0
    n_down = transitions_down.size

    for start in transitions_up:
        while down_idx < n_down and transitions_down[down_idx] <= start:
            down_idx += 1
        end = int(transitions_down[down_idx]) if down_idx < n_down else int(arr.size)
        if end - int(start) >= min_w:
            starts.append(int(start))
            ends.append(end)
            maxes.append(float(np.max(arr[int(start) : end])))

    return (
        np.array(starts, dtype=np.int64),
        np.array(ends, dtype=np.int64),
        np.array(maxes, dtype=np.float64),
    )


def preprocess_segment_with_roles(
    segment: neo.Segment, roles: List[Any], in_place: bool = True
) -> neo.Segment:
    """Apply channel roles (bandpass & TTL extraction) to a neo.Segment."""
    if segment is None:
        return segment
    if _core_preprocess_segment_with_roles is not None:
        converted = [_to_role_dict_for_state(normalize_role(r)) for r in (roles or [])]
        return _core_preprocess_segment_with_roles(segment, converted, in_place=in_place)

    sig_lookup: Dict[str, neo.AnalogSignal] = {}
    for sig in getattr(segment, "analogsignals", []) or []:
        name = str(getattr(sig, "name", "") or "")
        if name and name not in sig_lookup:
            sig_lookup[name] = sig
        for key in ("raw_channel_name", "source_channel"):
            src_ann = str((getattr(sig, "annotations", None) or {}).get(key, "") or "")
            if src_ann and src_ann not in sig_lookup:
                sig_lookup[src_ann] = sig

    def _find_sig(src_name: str) -> Optional[neo.AnalogSignal]:
        if src_name in sig_lookup:
            return sig_lookup[src_name]
        low = src_name.lower()
        for k, s in sig_lookup.items():
            if k.lower() == low:
                return s
        return None

    def _replace_or_append_signal(new_sig: neo.AnalogSignal):
        for idx, existing in enumerate(segment.analogsignals):
            if getattr(existing, "name", None) == new_sig.name:
                segment.analogsignals[idx] = new_sig
                return
        segment.analogsignals.append(new_sig)

    def _replace_or_append_event(new_evt: neo.Event):
        for idx, existing in enumerate(segment.events):
            if getattr(existing, "name", None) == new_evt.name:
                segment.events[idx] = new_evt
                return
        segment.events.append(new_evt)

    for raw_role in roles or []:
        role = normalize_role(raw_role)
        if not role.get("enabled", True):
            continue

        src_name = role["source_channel"]
        target_name = role["target_name"]
        src_sig = _find_sig(src_name) or _find_sig(target_name)
        if src_sig is None:
            continue

        sr = src_sig.sampling_rate
        sr_hz = float(sr.rescale(pq.Hz).magnitude if hasattr(sr, "rescale") else sr)
        t_start = getattr(src_sig, "t_start", 0.0 * pq.s)
        mode = role["mode"]

        if mode in (MODE_ANALOG, MODE_ANALOG_BP):
            raw_sig = neo.AnalogSignal(
                src_sig.magnitude,
                units=src_sig.units,
                sampling_rate=sr,
                t_start=t_start,
                name=target_name,
                description=role.get("description", ""),
            )
            raw_sig.annotations.update(dict(getattr(src_sig, "annotations", {}) or {}))
            raw_sig.annotations["source_channel"] = src_name
            raw_sig.annotations["channel_purpose"] = role["purpose"]
            _replace_or_append_signal(raw_sig)
            sig_lookup.setdefault(target_name, raw_sig)

            if mode == MODE_ANALOG_BP and sr_hz > 0:
                bp_low = max(0.5, float(role.get("bandpass_low", 500.0)))
                nyquist = 0.49 * sr_hz
                bp_high = min(nyquist, max(bp_low + 1.0, float(role.get("bandpass_high", 7000.0))))
                if bp_low < bp_high < 0.5 * sr_hz:
                    bp_arr = apply_bandpass(src_sig, lowHz=bp_low, highHz=bp_high, sampleRate=sr_hz)
                    bp_name = target_name if target_name.endswith(".bp") else f"{target_name}.bp"
                    bp_sig = neo.AnalogSignal(
                        bp_arr,
                        sampling_rate=sr,
                        t_start=t_start,
                        name=bp_name,
                        description=f"{role.get('description', '')}\nbandpass {bp_low:g}-{bp_high:g}Hz",
                    )
                    bp_sig.annotations.update(dict(getattr(src_sig, "annotations", {}) or {}))
                    bp_sig.annotations["source_channel"] = src_name
                    bp_sig.annotations["channel_purpose"] = "bandpass"
                    _replace_or_append_signal(bp_sig)

        elif mode == MODE_TTL and sr_hz > 0:
            thresh_v = float(role.get("ttl_threshold_v", 0.0))
            sd_mult = float(role.get("ttl_sd_multiplier", 10.0))
            refrac_ms = float(role.get("ttl_refractory_ms", 100.0))
            min_width_ms = float(role.get("ttl_min_width_ms", 0.4))

            arr_1d = np.asarray(
                src_sig.magnitude if hasattr(src_sig, "magnitude") else src_sig,
                dtype=np.float64,
            )
            if arr_1d.ndim > 1:
                arr_1d = arr_1d[:, 0]
            arr_1d = arr_1d.ravel()

            if thresh_v > 0.0:
                try:
                    arr_v = np.asarray(src_sig.rescale(pq.V).magnitude, dtype=np.float64)
                    if arr_v.ndim > 1:
                        arr_v = arr_v[:, 0]
                    arr_detect = arr_v.ravel()
                    unit_for_max = pq.V
                except Exception:
                    arr_detect = arr_1d
                    unit_for_max = src_sig.units
                threshold = thresh_v
            else:
                arr_detect = arr_1d
                unit_for_max = src_sig.units
                m = float(np.mean(arr_detect)) if arr_detect.size else 0.0
                sd = float(np.std(arr_detect - m)) if arr_detect.size else 0.0
                threshold = m + sd_mult * sd

            min_width_samples = max(1, int(round(sr_hz * (min_width_ms / 1000.0))))
            starts, ends, maxes = _find_square_pulses_safe(arr_detect, min_width_samples, threshold)

            t_rising_ms = (starts / sr_hz) * 1000.0
            keep = np.ones(len(t_rising_ms), dtype=bool)
            last_kept_ms = -1e18
            for i, t_ms in enumerate(t_rising_ms):
                if i > 0 and (t_ms - last_kept_ms) < refrac_ms:
                    keep[i] = False
                else:
                    last_kept_ms = float(t_ms)

            t_kept_s = (t_rising_ms[keep] / 1000.0) * pq.s
            t_start_s = t_start.rescale(pq.s) if hasattr(t_start, "rescale") else (float(t_start) * pq.s)
            durations = ((ends[keep] - starts[keep]) / sr_hz) * pq.s
            max_qty = maxes[keep] * unit_for_max
            try:
                max_mv = max_qty.rescale(pq.mV)
            except Exception:
                max_mv = max_qty

            evt = neo.Event(
                t_kept_s + t_start_s,
                name=target_name,
                type_id=TypeID.TTL.value,
                array_annotations={
                    "duration": durations,
                    "maximum": max_mv,
                },
                description=role.get("description", ""),
            )
            evt.annotations["source_channel"] = src_name
            evt.annotations["channel_purpose"] = role["purpose"]
            _replace_or_append_event(evt)

    return segment


def _parse_channel_int(ch_name: str, default: int = 0) -> int:
    m = re.search(r"(\d+)", str(ch_name or ""))
    return int(m.group(1)) if m else default


class ChannelPurposeWidget(QWidget):
    """UI widget for configuring channel purposes (roles) and preprocessing.

    Works standalone as a dockable panel (``MdiView.window_options["ChannelPurpose"]``),
    embedded inside ``ViewSettingsPanel`` / ``OpenEphysLiveView`` (``compact=True``),
    and wrapped by ``ChannelPurposeDialog``.
    """

    rolesChanged = Signal(list)
    rolesApplied = Signal(list)
    presetChanged = Signal(str)

    COL_ENABLED = 0
    COL_PURPOSE = 1
    COL_SOURCE = 2
    COL_TARGET = 3
    COL_MODE = 4
    COL_PARAMS = 5

    _SETTINGS_ORG = "spikespy"
    _SETTINGS_KEY = "channel_purpose_user_presets"

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        state: Optional[Any] = None,
        compact: bool = False,
        show_record_no: Optional[bool] = None,
    ) -> None:
        super().__init__(parent)
        self.state = None
        self._compact = bool(compact)
        self._show_record_no_override = show_record_no
        self._folder_path: Optional[Path] = None
        self._state_connections: List[Tuple[Any, Any]] = []
        self._row_metadata: List[Dict[str, Any]] = []
        self._available_channels: List[str] = list(STANDARD_CHANNEL_CHOICES)
        self._user_presets: Dict[str, List[Dict[str, Any]]] = self._load_user_presets()
        self._syncing = False
        self._updating_params = False
        self._global_defaults: Dict[str, Any] = {
            "bandpass_low": 500.0,
            "bandpass_high": 7000.0,
            "ttl_threshold_v": 0.0,
            "ttl_sd_multiplier": 10.0,
            "ttl_refractory_ms": 100.0,
            "ttl_min_width_ms": 0.4,
            "is_adc": True,
        }

        self.setWindowTitle("Channel Purposes & Preprocessing")
        self._init_ui()

        if state is not None:
            self.setState(state)
        else:
            self.apply_preset("APTrack Default Setup", emit_signals=False)

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------
    def _init_ui(self) -> None:
        margin = 4 if self._compact else 8
        spacing = 4 if self._compact else 6

        root = QVBoxLayout(self)
        root.setContentsMargins(margin, margin, margin, margin)
        root.setSpacing(spacing)

        # 1. Top Bar: Preset selector, Preset/JSON actions, Record #
        top_frame = QFrame(self)
        top_frame.setFrameShape(QFrame.StyledPanel if not self._compact else QFrame.NoFrame)
        top_vbox = QVBoxLayout(top_frame)
        pad = 4 if not self._compact else 0
        top_vbox.setContentsMargins(pad, pad, pad, pad)
        top_vbox.setSpacing(4)

        preset_row = QHBoxLayout()
        preset_row.setSpacing(4)
        preset_row.addWidget(QLabel("Preset:", top_frame))

        self.combo_preset = QComboBox(top_frame)
        self.combo_preset.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._populate_preset_combo()
        self.combo_preset.currentTextChanged.connect(self._on_preset_combo_changed)
        preset_row.addWidget(self.combo_preset, 1)

        self.lbl_record_no = QLabel("Rec #:", top_frame)
        self.sb_record_no = QSpinBox(top_frame)
        self.sb_record_no.setRange(0, 999)
        self.sb_record_no.setValue(1)
        self.sb_record_no.setToolTip(
            "Open Ephys / APTrack recording index to load or reload (1 = primary recording)"
        )
        preset_row.addWidget(self.lbl_record_no)
        preset_row.addWidget(self.sb_record_no)
        top_vbox.addLayout(preset_row)

        io_row = QHBoxLayout()
        io_row.setSpacing(4)
        self.btn_save_preset = QPushButton("Save Preset...", top_frame)
        self.btn_save_preset.setToolTip("Save current channel role table as a reusable preset")
        self.btn_save_preset.clicked.connect(self._on_save_preset_clicked)
        io_row.addWidget(self.btn_save_preset)

        self.btn_load_json = QPushButton("Load JSON...", top_frame)
        self.btn_load_json.setToolTip("Load channel roles from a JSON file")
        self.btn_load_json.clicked.connect(self._on_load_json_clicked)
        io_row.addWidget(self.btn_load_json)

        self.btn_export_json = QPushButton("Export JSON...", top_frame)
        self.btn_export_json.setToolTip("Export current channel roles to a JSON file")
        self.btn_export_json.clicked.connect(self._on_export_json_clicked)
        io_row.addWidget(self.btn_export_json)

        io_row.addStretch(1)
        top_vbox.addLayout(io_row)
        root.addWidget(top_frame)

        # 2. Roles Table
        self.table = QTableWidget(0, 6, self)
        self.table.setHorizontalHeaderLabels(
            ["On", "Purpose", "Source Ch", "Target Name", "Preprocess / Mode", "Parameters"]
        )
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(26)
        self.table.setMinimumHeight(140 if self._compact else 190)

        header = self.table.horizontalHeader()
        header.setSectionResizeMode(self.COL_ENABLED, QHeaderView.Fixed)
        self.table.setColumnWidth(self.COL_ENABLED, 34)
        header.setSectionResizeMode(self.COL_PURPOSE, QHeaderView.Stretch)
        header.setSectionResizeMode(self.COL_SOURCE, QHeaderView.Interactive)
        self.table.setColumnWidth(self.COL_SOURCE, 86)
        header.setSectionResizeMode(self.COL_TARGET, QHeaderView.Interactive)
        self.table.setColumnWidth(self.COL_TARGET, 115)
        header.setSectionResizeMode(self.COL_MODE, QHeaderView.Interactive)
        self.table.setColumnWidth(self.COL_MODE, 150)
        header.setSectionResizeMode(self.COL_PARAMS, QHeaderView.Stretch)

        self.table.itemSelectionChanged.connect(self._on_table_selection_changed)
        root.addWidget(self.table, 1)

        # 3. Preprocessing Parameters Form for Selected Row (or Defaults)
        self.grp_params = QGroupBox("Preprocessing Parameters (Selected Row)", self)
        params_layout = QVBoxLayout(self.grp_params)
        params_layout.setContentsMargins(6, 6, 6, 6)
        params_layout.setSpacing(4)

        self.lbl_selected_row = QLabel("Editing parameters: (no row selected)", self.grp_params)
        self.lbl_selected_row.setStyleSheet("color: #475569; font-size: 11px;")
        params_layout.addWidget(self.lbl_selected_row)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(4)

        self.spin_bp_low = QDoubleSpinBox(self.grp_params)
        self.spin_bp_low.setRange(1.0, 10000.0)
        self.spin_bp_low.setDecimals(1)
        self.spin_bp_low.setSingleStep(50.0)
        self.spin_bp_low.setValue(500.0)
        self.spin_bp_low.setSuffix(" Hz")
        self.spin_bp_low.valueChanged.connect(self._on_param_widget_changed)

        self.spin_bp_high = QDoubleSpinBox(self.grp_params)
        self.spin_bp_high.setRange(10.0, 20000.0)
        self.spin_bp_high.setDecimals(1)
        self.spin_bp_high.setSingleStep(100.0)
        self.spin_bp_high.setValue(7000.0)
        self.spin_bp_high.setSuffix(" Hz")
        self.spin_bp_high.valueChanged.connect(self._on_param_widget_changed)

        self.spin_ttl_thresh = QDoubleSpinBox(self.grp_params)
        self.spin_ttl_thresh.setRange(0.0, 100.0)
        self.spin_ttl_thresh.setDecimals(2)
        self.spin_ttl_thresh.setSingleStep(0.5)
        self.spin_ttl_thresh.setValue(0.0)
        self.spin_ttl_thresh.setSpecialValueText("Auto (SD)")
        self.spin_ttl_thresh.setSuffix(" V")
        self.spin_ttl_thresh.setToolTip("0 = Auto threshold using SD Multiplier")
        self.spin_ttl_thresh.valueChanged.connect(self._on_param_widget_changed)

        self.spin_ttl_sd = QDoubleSpinBox(self.grp_params)
        self.spin_ttl_sd.setRange(0.5, 50.0)
        self.spin_ttl_sd.setDecimals(1)
        self.spin_ttl_sd.setSingleStep(0.5)
        self.spin_ttl_sd.setValue(10.0)
        self.spin_ttl_sd.setSuffix(" SD")
        self.spin_ttl_sd.valueChanged.connect(self._on_param_widget_changed)

        self.spin_ttl_refrac = QDoubleSpinBox(self.grp_params)
        self.spin_ttl_refrac.setRange(1.0, 5000.0)
        self.spin_ttl_refrac.setDecimals(1)
        self.spin_ttl_refrac.setSingleStep(10.0)
        self.spin_ttl_refrac.setValue(100.0)
        self.spin_ttl_refrac.setSuffix(" ms")
        self.spin_ttl_refrac.valueChanged.connect(self._on_param_widget_changed)

        self.spin_ttl_min_width = QDoubleSpinBox(self.grp_params)
        self.spin_ttl_min_width.setRange(0.05, 100.0)
        self.spin_ttl_min_width.setDecimals(2)
        self.spin_ttl_min_width.setSingleStep(0.1)
        self.spin_ttl_min_width.setValue(0.4)
        self.spin_ttl_min_width.setSuffix(" ms")
        self.spin_ttl_min_width.valueChanged.connect(self._on_param_widget_changed)

        self.chk_is_adc = QCheckBox("Is ADC (V vs uV)", self.grp_params)
        self.chk_is_adc.setChecked(True)
        self.chk_is_adc.toggled.connect(self._on_param_widget_changed)

        if self._compact:
            grid.addWidget(QLabel("BP Low:"), 0, 0)
            grid.addWidget(self.spin_bp_low, 0, 1)
            grid.addWidget(QLabel("BP High:"), 0, 2)
            grid.addWidget(self.spin_bp_high, 0, 3)

            grid.addWidget(QLabel("TTL Thresh:"), 1, 0)
            grid.addWidget(self.spin_ttl_thresh, 1, 1)
            grid.addWidget(QLabel("TTL SD:"), 1, 2)
            grid.addWidget(self.spin_ttl_sd, 1, 3)

            grid.addWidget(QLabel("Refractory:"), 2, 0)
            grid.addWidget(self.spin_ttl_refrac, 2, 1)
            grid.addWidget(QLabel("Min Width:"), 2, 2)
            grid.addWidget(self.spin_ttl_min_width, 2, 3)

            grid.addWidget(self.chk_is_adc, 3, 0, 1, 4)
        else:
            grid.addWidget(QLabel("Bandpass Low (Hz):"), 0, 0)
            grid.addWidget(self.spin_bp_low, 0, 1)
            grid.addWidget(QLabel("Bandpass High (Hz):"), 0, 2)
            grid.addWidget(self.spin_bp_high, 0, 3)

            grid.addWidget(QLabel("TTL Threshold (V, 0=Auto):"), 1, 0)
            grid.addWidget(self.spin_ttl_thresh, 1, 1)
            grid.addWidget(QLabel("TTL SD Multiplier:"), 1, 2)
            grid.addWidget(self.spin_ttl_sd, 1, 3)

            grid.addWidget(QLabel("TTL Refractory (ms):"), 2, 0)
            grid.addWidget(self.spin_ttl_refrac, 2, 1)
            grid.addWidget(QLabel("Min Width (ms):"), 2, 2)
            grid.addWidget(self.spin_ttl_min_width, 2, 3)

            grid.addWidget(self.chk_is_adc, 3, 0, 1, 4)

        params_layout.addLayout(grid)
        root.addWidget(self.grp_params)

        # 4. Action Row
        edit_row = QHBoxLayout()
        edit_row.setSpacing(6)
        self.btn_add_role = QPushButton("+ Add Channel Role", self)
        self.btn_add_role.clicked.connect(lambda: self.add_role())
        edit_row.addWidget(self.btn_add_role)

        self.btn_remove_role = QPushButton("- Remove Selected", self)
        self.btn_remove_role.clicked.connect(self.remove_selected_roles)
        edit_row.addWidget(self.btn_remove_role)

        self.btn_reset_defaults = QPushButton("Reset Defaults", self)
        self.btn_reset_defaults.clicked.connect(self.reset_defaults)
        edit_row.addWidget(self.btn_reset_defaults)
        edit_row.addStretch(1)
        root.addLayout(edit_row)

        apply_row = QHBoxLayout()
        apply_row.setSpacing(8)
        self.btn_apply_preprocess = QPushButton("🔄 Apply & Preprocess (Reload)", self)
        self.btn_apply_preprocess.setStyleSheet(
            "QPushButton { background-color: #2563eb; color: white; font-weight: bold; "
            "padding: 6px 12px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #1d4ed8; }"
        )
        self.btn_apply_preprocess.clicked.connect(self.apply_and_preprocess)
        apply_row.addWidget(self.btn_apply_preprocess)

        self.chk_auto_apply_live = QCheckBox("Auto-sync to Live Stream", self)
        self.chk_auto_apply_live.setChecked(True)
        self.chk_auto_apply_live.setToolTip(
            "Immediately push role, bandpass, and trigger updates to an active live stream without dropping the connection"
        )
        apply_row.addWidget(self.chk_auto_apply_live)
        apply_row.addStretch(1)
        root.addLayout(apply_row)

        self.lbl_status = QLabel("", self)
        self.lbl_status.setStyleSheet("color: #475569; font-size: 11px;")
        self.lbl_status.setVisible(False)
        root.addWidget(self.lbl_status)

        self._update_record_no_visibility()

    # ------------------------------------------------------------------
    # State & Signal Wiring
    # ------------------------------------------------------------------
    def unset_connections(self) -> None:
        for sig, slot in self._state_connections:
            try:
                sig.disconnect(slot)
            except (TypeError, RuntimeError):
                pass
        self._state_connections.clear()

    def closeEvent(self, event) -> None:
        self.unset_connections()
        super().closeEvent(event)

    def setState(self, state: Any) -> None:
        self.unset_connections()
        self.state = state
        if self.state is not None:
            for sig_name, slot in (
                ("onLoadNewFile", self._on_state_data_changed),
                ("onChannelGroupsChanged", self._on_state_channels_changed),
                ("onStreamStateChanged", self._on_stream_state_changed),
                ("onLiveConfigChanged", self._on_state_live_config_changed),
                ("onChannelRolesChanged", self._on_state_roles_changed),
            ):
                sig = getattr(self.state, sig_name, None)
                if sig is not None:
                    try:
                        sig.connect(slot)
                        self._state_connections.append((sig, slot))
                    except (TypeError, RuntimeError):
                        pass

        self.refresh_available_channels()
        self._sync_from_state()
        self._update_record_no_visibility()

    set_state = setState

    def set_folder(self, folder: Optional[Union[str, Path]]) -> None:
        self._folder_path = Path(folder) if folder else None
        self.refresh_available_channels(folder=self._folder_path)
        self._update_record_no_visibility()

    def set_record_no_visible(self, visible: bool) -> None:
        self._show_record_no_override = bool(visible)
        self.lbl_record_no.setVisible(bool(visible))
        self.sb_record_no.setVisible(bool(visible))

    def _has_aptrack_folder(self) -> bool:
        if self._folder_path is not None and self._folder_path.exists():
            return True
        if self.state is not None:
            if getattr(self.state, "_loaded_file_type", None) == "APTrack":
                return True
            seg = getattr(self.state, "segment", None)
            if seg is not None:
                orig = (getattr(seg, "annotations", None) or {}).get("original_folder")
                if orig:
                    return True
        return False

    def _update_record_no_visibility(self) -> None:
        if self._show_record_no_override is not None:
            vis = self._show_record_no_override
        elif self._has_aptrack_folder():
            vis = True
        else:
            vis = not self._compact
        self.lbl_record_no.setVisible(vis)
        self.sb_record_no.setVisible(vis)

    @Slot()
    def _on_state_data_changed(self, *args) -> None:
        if self._syncing:
            return
        self.refresh_available_channels()
        self._update_record_no_visibility()
        self._sync_from_state()

    @Slot()
    def _on_state_channels_changed(self, *args) -> None:
        self.refresh_available_channels()

    @Slot(bool)
    def _on_stream_state_changed(self, _is_streaming: bool) -> None:
        self.refresh_available_channels()

    @Slot()
    def _on_state_live_config_changed(self, *args) -> None:
        if self._syncing or self.state is None:
            return
        cfg = (
            self.state.get_live_config()
            if hasattr(self.state, "get_live_config")
            else getattr(self.state, "live_config", {})
        )
        roles = cfg.get("channel_roles") if isinstance(cfg, dict) else None
        if isinstance(roles, list) and roles:
            self.set_roles(roles)

    @Slot()
    def _on_state_roles_changed(self, *args) -> None:
        if self._syncing or self.state is None:
            return
        self._sync_from_state()

    def _sync_from_state(self) -> None:
        if self.state is None:
            return
        roles = None
        if hasattr(self.state, "get_channel_roles"):
            try:
                roles = self.state.get_channel_roles()
            except Exception:
                roles = None
        if not roles and hasattr(self.state, "channel_roles"):
            roles = getattr(self.state, "channel_roles", None)
        if not roles:
            cfg = (
                self.state.get_live_config()
                if hasattr(self.state, "get_live_config")
                else getattr(self.state, "live_config", {})
            )
            if isinstance(cfg, dict):
                roles = cfg.get("channel_roles")

        last_cfg = getattr(self.state, "_last_aptrack_config", None)
        if isinstance(last_cfg, dict) and "record" in last_cfg:
            with QSignalBlocker(self.sb_record_no):
                self.sb_record_no.setValue(int(last_cfg["record"]))

        if isinstance(roles, list) and roles:
            self.set_roles(roles)
            return

        seg = getattr(self.state, "segment", None)
        if seg is not None and (getattr(seg, "analogsignals", None) or getattr(seg, "events", None)):
            inferred = self._infer_roles_from_segment(seg)
            if inferred:
                self.set_roles(inferred)
                return

        if self.table.rowCount() == 0:
            self.apply_preset("APTrack Default Setup", emit_signals=False)

    def _infer_roles_from_segment(self, seg: neo.Segment) -> List[Dict[str, Any]]:
        roles: List[Dict[str, Any]] = []
        sig_names = {
            str(getattr(s, "name", "") or "")
            for s in getattr(seg, "analogsignals", []) or []
        }

        for sig in getattr(seg, "analogsignals", []) or []:
            name = str(getattr(sig, "name", "") or "")
            if not name or name.endswith(".bp") or name.startswith("ERP"):
                continue
            anns = getattr(sig, "annotations", None) or {}
            src = str(anns.get("raw_channel_name") or anns.get("source_channel") or name)
            purpose = str(
                anns.get("channel_purpose")
                or anns.get("purpose")
                or _infer_purpose_from_target(name, "AS")
            )
            has_bp = f"{name}.bp" in sig_names or purpose == "probe"
            bp_sig = next(
                (s for s in getattr(seg, "analogsignals", []) or [] if getattr(s, "name", "") == f"{name}.bp"),
                None,
            )
            bp_anns = getattr(bp_sig, "annotations", {}) or {} if bp_sig is not None else {}
            roles.append(
                normalize_role(
                    {
                        "enabled": True,
                        "purpose": purpose,
                        "source_channel": src,
                        "target_name": name,
                        "mode": MODE_ANALOG_BP if has_bp else MODE_ANALOG,
                        "bandpass": has_bp,
                        "bandpass_low": bp_anns.get("bandpass_low", 500.0),
                        "bandpass_high": bp_anns.get("bandpass_high", 7000.0),
                        "is_adc": purpose != "probe",
                    }
                )
            )

        for evt in getattr(seg, "events", []) or []:
            name = str(getattr(evt, "name", "") or "")
            if not name or name.startswith("unit") or name.startswith("log."):
                continue
            anns = getattr(evt, "annotations", None) or {}
            if anns.get("type") == "unit":
                continue
            src = str(anns.get("source_channel") or name)
            purpose = str(
                anns.get("channel_purpose")
                or anns.get("purpose")
                or _infer_purpose_from_target(name, "TTL")
            )
            roles.append(
                normalize_role(
                    {
                        "enabled": True,
                        "purpose": purpose,
                        "source_channel": src,
                        "target_name": name,
                        "mode": MODE_TTL,
                        "probe_type": "TTL",
                        "bandpass": False,
                        "is_adc": True,
                    }
                )
            )

        return roles

    # ------------------------------------------------------------------
    # Channel Choices & Presets
    # ------------------------------------------------------------------
    def refresh_available_channels(
        self, folder: Optional[Union[str, Path]] = None
    ) -> None:
        target_folder = folder or self._folder_path
        self._available_channels = detect_channels_from_state(
            self.state, folder=target_folder
        )
        for row in range(self.table.rowCount()):
            combo = self.table.cellWidget(row, self.COL_SOURCE)
            if isinstance(combo, QComboBox):
                curr = combo.currentText().strip()
                with QSignalBlocker(combo):
                    combo.clear()
                    choices = _dedupe_preserve_order(
                        ([curr] if curr else []) + self._available_channels
                    )
                    combo.addItems(choices)
                    if curr:
                        combo.setCurrentText(curr)

    def _load_user_presets(self) -> Dict[str, List[Dict[str, Any]]]:
        try:
            settings = QSettings(
                QSettings.Format.IniFormat,
                QSettings.Scope.UserScope,
                self._SETTINGS_ORG,
            )
            raw = settings.value(self._SETTINGS_KEY, "")
            if raw:
                data = json.loads(raw)
                if isinstance(data, dict):
                    return {
                        str(k): [normalize_role(r) for r in v]
                        for k, v in data.items()
                        if isinstance(v, list)
                    }
        except Exception:
            pass
        return {}

    def _save_user_presets(self) -> None:
        try:
            settings = QSettings(
                QSettings.Format.IniFormat,
                QSettings.Scope.UserScope,
                self._SETTINGS_ORG,
            )
            settings.setValue(self._SETTINGS_KEY, json.dumps(self._user_presets))
            settings.sync()
        except Exception:
            pass

    def _populate_preset_combo(self, selected: Optional[str] = None) -> None:
        curr = selected or (
            self.combo_preset.currentText() if hasattr(self, "combo_preset") else ""
        )
        with QSignalBlocker(self.combo_preset):
            self.combo_preset.clear()
            for name in SETUP_PRESETS:
                self.combo_preset.addItem(name, name)
            for name in sorted(self._user_presets.keys()):
                if name not in SETUP_PRESETS:
                    self.combo_preset.addItem(name, name)
            if curr:
                idx = self.combo_preset.findText(curr)
                if idx >= 0:
                    self.combo_preset.setCurrentIndex(idx)

    @Slot(str)
    def _on_preset_combo_changed(self, name: str) -> None:
        if self._syncing or not name:
            return
        self.apply_preset(name, emit_signals=True)

    def apply_preset(self, preset_name: str, emit_signals: bool = True) -> bool:
        roles = SETUP_PRESETS.get(preset_name) or self._user_presets.get(preset_name)
        if roles is None:
            return False
        self._syncing = True
        try:
            with QSignalBlocker(self.combo_preset):
                idx = self.combo_preset.findText(preset_name)
                if idx >= 0:
                    self.combo_preset.setCurrentIndex(idx)
            self._populate_table_from_roles([normalize_role(r) for r in roles])
        finally:
            self._syncing = False
        if emit_signals:
            self.presetChanged.emit(preset_name)
            self.rolesChanged.emit(self.get_roles())
        return True

    def save_user_preset(self, name: str) -> None:
        clean = str(name or "").strip()
        if not clean:
            return
        self._user_presets[clean] = self.get_roles()
        self._save_user_presets()
        self._populate_preset_combo(selected=clean)

    @Slot()
    def _on_save_preset_clicked(self) -> None:
        app = QApplication.instance()
        default_name = f"Custom Setup ({self.table.rowCount()} ch)"
        if app is not None and app.platformName() == "offscreen":
            name, ok = default_name, True
        else:
            name, ok = QInputDialog.getText(
                self, "Save Channel Purpose Preset", "Preset Name:", text=default_name
            )
        if ok and str(name).strip():
            self.save_user_preset(str(name).strip())

    @Slot()
    def _on_load_json_clicked(self) -> None:
        fname, _ = QFileDialog.getOpenFileName(
            self,
            "Load Channel Purposes JSON",
            "",
            "JSON files (*.json);;All files (*)",
        )
        if fname:
            self.load_roles_from_json(fname)

    @Slot()
    def _on_export_json_clicked(self) -> None:
        fname, _ = QFileDialog.getSaveFileName(
            self,
            "Export Channel Purposes JSON",
            "channel_purposes.json",
            "JSON files (*.json);;All files (*)",
        )
        if fname:
            self.export_roles_to_json(fname)

    def load_roles_from_json(self, path: Union[str, Path]) -> bool:
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                if "record_no" in raw:
                    self.sb_record_no.setValue(int(raw["record_no"]))
                elif "record" in raw:
                    self.sb_record_no.setValue(int(raw["record"]))
                roles_list = raw.get("roles", [])
            elif isinstance(raw, list):
                roles_list = raw
            else:
                return False
            self.set_roles(roles_list)
            return True
        except Exception as exc:
            logger.warning("Failed to load channel purpose JSON %s: %s", path, exc)
            app = QApplication.instance()
            if app is not None and app.platformName() != "offscreen":
                QMessageBox.warning(self, "Load JSON Failed", str(exc))
            return False

    def export_roles_to_json(self, path: Union[str, Path]) -> bool:
        try:
            payload = {
                "record_no": int(self.sb_record_no.value()),
                "roles": self.get_roles(),
            }
            Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
            return True
        except Exception as exc:
            logger.warning("Failed to export channel purpose JSON %s: %s", path, exc)
            app = QApplication.instance()
            if app is not None and app.platformName() != "offscreen":
                QMessageBox.warning(self, "Export JSON Failed", str(exc))
            return False

    # ------------------------------------------------------------------
    # Table Row Management
    # ------------------------------------------------------------------
    def _populate_table_from_roles(self, roles: List[Dict[str, Any]]) -> None:
        self.table.setRowCount(0)
        self._row_metadata.clear()
        for role in roles:
            self._append_role_row(normalize_role(role))
        if self.table.rowCount() > 0:
            self.table.selectRow(0)
            self._load_row_params_into_form(0)
        else:
            self._load_row_params_into_form(-1)

    def _append_role_row(self, role: Dict[str, Any]) -> int:
        row = self.table.rowCount()
        self.table.insertRow(row)

        preset = PURPOSE_PRESETS.get(role["purpose"], PURPOSE_PRESETS["aux"])
        custom_target = bool(
            role.get("target_name") and role["target_name"] != preset["default_target_name"]
        )
        meta = dict(role)
        meta["custom_target_name"] = custom_target
        self._row_metadata.append(meta)

        # Col 0: Enabled checkbox centered in cell
        chk_container = QWidget(self.table)
        chk_layout = QHBoxLayout(chk_container)
        chk_layout.setContentsMargins(0, 0, 0, 0)
        chk_layout.setAlignment(Qt.AlignCenter)
        chk = QCheckBox(chk_container)
        chk.setChecked(bool(role.get("enabled", True)))
        chk.toggled.connect(lambda checked, c=chk: self._on_row_enabled_toggled(c, checked))
        chk_layout.addWidget(chk)
        self.table.setCellWidget(row, self.COL_ENABLED, chk_container)

        # Col 1: Purpose combo
        combo_purpose = QComboBox(self.table)
        for key, info in PURPOSE_PRESETS.items():
            combo_purpose.addItem(info["label"], key)
        p_idx = combo_purpose.findData(role["purpose"])
        if p_idx >= 0:
            combo_purpose.setCurrentIndex(p_idx)
        combo_purpose.currentIndexChanged.connect(
            lambda _idx, w=combo_purpose: self._on_row_purpose_changed(w)
        )
        self.table.setCellWidget(row, self.COL_PURPOSE, combo_purpose)

        # Col 2: Source Ch editable combo
        combo_src = QComboBox(self.table)
        combo_src.setEditable(True)
        src_val = str(role.get("source_channel", "CH1"))
        choices = _dedupe_preserve_order(
            ([src_val] if src_val else []) + self._available_channels
        )
        combo_src.addItems(choices)
        combo_src.setCurrentText(src_val)
        combo_src.currentTextChanged.connect(
            lambda text, w=combo_src: self._on_row_source_changed(w, text)
        )
        self.table.setCellWidget(row, self.COL_SOURCE, combo_src)

        # Col 3: Target Name QLineEdit
        edit_target = QLineEdit(str(role.get("target_name", "")), self.table)
        edit_target.textEdited.connect(
            lambda text, w=edit_target: self._on_row_target_edited(w, text)
        )
        edit_target.textChanged.connect(
            lambda text, w=edit_target: self._on_row_target_changed(w, text)
        )
        self.table.setCellWidget(row, self.COL_TARGET, edit_target)

        # Col 4: Preprocess / Mode combo
        combo_mode = QComboBox(self.table)
        for label, m_key in MODE_OPTIONS:
            combo_mode.addItem(label, m_key)
        m_idx = combo_mode.findData(role.get("mode", MODE_ANALOG))
        if m_idx >= 0:
            combo_mode.setCurrentIndex(m_idx)
        combo_mode.currentIndexChanged.connect(
            lambda _idx, w=combo_mode: self._on_row_mode_changed(w)
        )
        self.table.setCellWidget(row, self.COL_MODE, combo_mode)

        # Col 5: Parameters summary item (read-only)
        params_item = QTableWidgetItem(format_role_parameters(role))
        params_item.setFlags(params_item.flags() & ~Qt.ItemIsEditable)
        self.table.setItem(row, self.COL_PARAMS, params_item)

        return row

    def _find_widget_row(self, widget: QWidget, col: int) -> int:
        for row in range(self.table.rowCount()):
            cell_w = self.table.cellWidget(row, col)
            if cell_w is widget:
                return row
            if cell_w is not None and widget.parentWidget() is cell_w:
                return row
        return -1

    def _update_row_param_summary(self, row: int) -> None:
        if not (0 <= row < len(self._row_metadata)):
            return
        summary = format_role_parameters(self._row_metadata[row])
        item = self.table.item(row, self.COL_PARAMS)
        if item is None:
            item = QTableWidgetItem(summary)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, self.COL_PARAMS, item)
        else:
            item.setText(summary)

    @Slot()
    def _on_row_enabled_toggled(self, chk: QCheckBox, checked: bool) -> None:
        row = self._find_widget_row(chk, self.COL_ENABLED)
        if 0 <= row < len(self._row_metadata):
            self._row_metadata[row]["enabled"] = bool(checked)
            if not self._syncing:
                self.rolesChanged.emit(self.get_roles())

    @Slot()
    def _on_row_purpose_changed(self, combo: QComboBox) -> None:
        row = self._find_widget_row(combo, self.COL_PURPOSE)
        if not (0 <= row < len(self._row_metadata)):
            return
        purpose_key = str(combo.currentData() or "aux")
        preset = PURPOSE_PRESETS.get(purpose_key, PURPOSE_PRESETS["aux"])
        meta = self._row_metadata[row]
        meta["purpose"] = purpose_key
        meta["description"] = preset["label"]
        meta["details"] = preset["label"]
        meta["is_adc"] = bool(preset.get("is_adc", True))

        if not meta.get("custom_target_name", False):
            edit_target = self.table.cellWidget(row, self.COL_TARGET)
            default_target = preset["default_target_name"]
            meta["target_name"] = default_target
            meta["name"] = default_target
            if isinstance(edit_target, QLineEdit):
                with QSignalBlocker(edit_target):
                    edit_target.setText(default_target)

        if preset["probe_type"] == "TTL":
            new_mode = MODE_TTL
        elif preset.get("bandpass", False):
            new_mode = MODE_ANALOG_BP
        else:
            new_mode = MODE_ANALOG

        meta["mode"] = new_mode
        meta["probe_type"] = preset["probe_type"]
        meta["bandpass"] = bool(preset.get("bandpass", False))
        if "bandpass_low" in preset:
            meta["bandpass_low"] = float(preset["bandpass_low"])
        if "bandpass_high" in preset:
            meta["bandpass_high"] = float(preset["bandpass_high"])

        combo_mode = self.table.cellWidget(row, self.COL_MODE)
        if isinstance(combo_mode, QComboBox):
            with QSignalBlocker(combo_mode):
                m_idx = combo_mode.findData(new_mode)
                if m_idx >= 0:
                    combo_mode.setCurrentIndex(m_idx)

        self._update_row_param_summary(row)
        if self.table.currentRow() == row:
            self._load_row_params_into_form(row)
        if not self._syncing:
            self.rolesChanged.emit(self.get_roles())

    @Slot()
    def _on_row_source_changed(self, combo: QComboBox, text: str) -> None:
        row = self._find_widget_row(combo, self.COL_SOURCE)
        if 0 <= row < len(self._row_metadata):
            clean = str(text).strip()
            self._row_metadata[row]["source_channel"] = clean
            self._row_metadata[row]["source"] = clean
            if not self._syncing:
                self.rolesChanged.emit(self.get_roles())

    @Slot()
    def _on_row_target_edited(self, edit: QLineEdit, text: str) -> None:
        row = self._find_widget_row(edit, self.COL_TARGET)
        if 0 <= row < len(self._row_metadata):
            meta = self._row_metadata[row]
            preset = PURPOSE_PRESETS.get(meta["purpose"], PURPOSE_PRESETS["aux"])
            clean = str(text).strip()
            meta["custom_target_name"] = bool(
                clean and clean != preset["default_target_name"]
            )

    @Slot()
    def _on_row_target_changed(self, edit: QLineEdit, text: str) -> None:
        row = self._find_widget_row(edit, self.COL_TARGET)
        if 0 <= row < len(self._row_metadata):
            clean = str(text).strip()
            self._row_metadata[row]["target_name"] = clean
            self._row_metadata[row]["name"] = clean
            if self.table.currentRow() == row:
                self._update_selected_row_label(row)
            if not self._syncing:
                self.rolesChanged.emit(self.get_roles())

    @Slot()
    def _on_row_mode_changed(self, combo: QComboBox) -> None:
        row = self._find_widget_row(combo, self.COL_MODE)
        if not (0 <= row < len(self._row_metadata)):
            return
        mode = str(combo.currentData() or MODE_ANALOG)
        meta = self._row_metadata[row]
        meta["mode"] = mode
        if mode == MODE_TTL:
            meta["probe_type"] = "TTL"
            meta["bandpass"] = False
        elif mode == MODE_ANALOG_BP:
            meta["probe_type"] = "AS"
            meta["bandpass"] = True
        else:
            meta["probe_type"] = "AS"
            meta["bandpass"] = False

        self._update_row_param_summary(row)
        if self.table.currentRow() == row:
            self._load_row_params_into_form(row)
        if not self._syncing:
            self.rolesChanged.emit(self.get_roles())

    @Slot()
    def _on_table_selection_changed(self) -> None:
        row = self.table.currentRow()
        self._load_row_params_into_form(row)

    def _update_selected_row_label(self, row: int) -> None:
        if 0 <= row < len(self._row_metadata):
            meta = self._row_metadata[row]
            target = meta.get("target_name") or "(unnamed)"
            src = meta.get("source_channel") or "?"
            self.lbl_selected_row.setText(
                f"Editing row {row + 1}: <b>{target}</b> (from {src})"
            )
        else:
            self.lbl_selected_row.setText("Editing global defaults (select a row to customize)")

    def _load_row_params_into_form(self, row: int) -> None:
        self._updating_params = True
        try:
            self._update_selected_row_label(row)
            src = (
                self._row_metadata[row]
                if 0 <= row < len(self._row_metadata)
                else self._global_defaults
            )
            with QSignalBlocker(self.spin_bp_low):
                self.spin_bp_low.setValue(float(src.get("bandpass_low", 500.0)))
            with QSignalBlocker(self.spin_bp_high):
                self.spin_bp_high.setValue(float(src.get("bandpass_high", 7000.0)))
            with QSignalBlocker(self.spin_ttl_thresh):
                self.spin_ttl_thresh.setValue(float(src.get("ttl_threshold_v", 0.0)))
            with QSignalBlocker(self.spin_ttl_sd):
                self.spin_ttl_sd.setValue(float(src.get("ttl_sd_multiplier", 10.0)))
            with QSignalBlocker(self.spin_ttl_refrac):
                self.spin_ttl_refrac.setValue(float(src.get("ttl_refractory_ms", 100.0)))
            with QSignalBlocker(self.spin_ttl_min_width):
                self.spin_ttl_min_width.setValue(float(src.get("ttl_min_width_ms", 0.4)))
            with QSignalBlocker(self.chk_is_adc):
                self.chk_is_adc.setChecked(bool(src.get("is_adc", True)))

            mode = src.get("mode", MODE_ANALOG)
            is_bp = mode == MODE_ANALOG_BP or row < 0
            is_ttl = mode == MODE_TTL or row < 0
            self.spin_bp_low.setEnabled(is_bp)
            self.spin_bp_high.setEnabled(is_bp)
            self.spin_ttl_thresh.setEnabled(is_ttl)
            self.spin_ttl_sd.setEnabled(is_ttl)
            self.spin_ttl_refrac.setEnabled(is_ttl)
            self.spin_ttl_min_width.setEnabled(is_ttl)
        finally:
            self._updating_params = False

    @Slot()
    def _on_param_widget_changed(self, *args) -> None:
        if self._updating_params:
            return
        bp_low = float(self.spin_bp_low.value())
        bp_high = float(self.spin_bp_high.value())
        if bp_high <= bp_low:
            bp_high = bp_low + 10.0
            with QSignalBlocker(self.spin_bp_high):
                self.spin_bp_high.setValue(bp_high)

        thresh_v = float(self.spin_ttl_thresh.value())
        sd_mult = float(self.spin_ttl_sd.value())
        refrac_ms = float(self.spin_ttl_refrac.value())
        min_w_ms = float(self.spin_ttl_min_width.value())

        params = {
            "bandpass_low": bp_low,
            "bandpass_high": bp_high,
            "ttl_threshold_v": thresh_v,
            "ttl_threshold": thresh_v if thresh_v > 0.0 else None,
            "ttl_sd_multiplier": sd_mult,
            "ttl_std_mult": sd_mult,
            "ttl_refractory_ms": refrac_ms,
            "ttl_min_width_ms": min_w_ms,
            "is_adc": bool(self.chk_is_adc.isChecked()),
        }

        row = self.table.currentRow()
        if 0 <= row < len(self._row_metadata):
            self._row_metadata[row].update(params)
            self._update_row_param_summary(row)
        else:
            self._global_defaults.update(params)

        if not self._syncing:
            self.rolesChanged.emit(self.get_roles())

    # ------------------------------------------------------------------
    # Public Role Accessors & Actions
    # ------------------------------------------------------------------
    def add_role(self, role: Optional[Any] = None) -> int:
        if role is None:
            idx = self.table.rowCount() + 1
            next_ch = f"ADC{min(8, idx)}"
            norm = normalize_role(
                {
                    "enabled": True,
                    "purpose": "aux",
                    "source_channel": next_ch,
                    "target_name": f"env.aux{idx}",
                    "mode": MODE_ANALOG,
                    **self._global_defaults,
                }
            )
        else:
            norm = normalize_role(role)
        row = self._append_role_row(norm)
        self.table.selectRow(row)
        self._load_row_params_into_form(row)
        if not self._syncing:
            self.rolesChanged.emit(self.get_roles())
        return row

    @Slot()
    def remove_selected_roles(self) -> None:
        row = self.table.currentRow()
        if 0 <= row < self.table.rowCount():
            self.table.removeRow(row)
            if 0 <= row < len(self._row_metadata):
                self._row_metadata.pop(row)
            new_row = min(row, self.table.rowCount() - 1)
            if new_row >= 0:
                self.table.selectRow(new_row)
            self._load_row_params_into_form(new_row)
            if not self._syncing:
                self.rolesChanged.emit(self.get_roles())

    @Slot()
    def reset_defaults(self) -> None:
        self.apply_preset("APTrack Default Setup", emit_signals=True)

    def get_roles(self) -> List[dict]:
        """Return the list of configured channel role dicts."""
        roles: List[dict] = []
        for row in range(self.table.rowCount()):
            meta = dict(self._row_metadata[row]) if row < len(self._row_metadata) else {}

            chk_container = self.table.cellWidget(row, self.COL_ENABLED)
            if chk_container is not None:
                chk = chk_container.findChild(QCheckBox)
                if chk is not None:
                    meta["enabled"] = bool(chk.isChecked())

            combo_purpose = self.table.cellWidget(row, self.COL_PURPOSE)
            if isinstance(combo_purpose, QComboBox):
                meta["purpose"] = str(combo_purpose.currentData() or "aux")

            combo_src = self.table.cellWidget(row, self.COL_SOURCE)
            if isinstance(combo_src, QComboBox):
                src_text = combo_src.currentText().strip()
                meta["source_channel"] = src_text
                meta["source"] = src_text

            edit_target = self.table.cellWidget(row, self.COL_TARGET)
            if isinstance(edit_target, QLineEdit):
                target_text = edit_target.text().strip()
                meta["target_name"] = target_text
                meta["name"] = target_text

            combo_mode = self.table.cellWidget(row, self.COL_MODE)
            if isinstance(combo_mode, QComboBox):
                meta["mode"] = str(combo_mode.currentData() or MODE_ANALOG)

            meta.pop("custom_target_name", None)
            roles.append(normalize_role(meta))
        return roles

    def set_roles(self, roles: List[Any]) -> None:
        """Populate the table from a list of role dicts or ``ChannelRoleSpec`` objects."""
        normalized = [normalize_role(r) for r in (roles or [])]
        self._syncing = True
        try:
            self._populate_table_from_roles(normalized)
        finally:
            self._syncing = False
        self.rolesChanged.emit(self.get_roles())

    def get_aptrack_config(self) -> dict:
        """Return an APTrack-compatible config dict (including ``roles`` and legacy keys)."""
        roles = self.get_roles()
        config: Dict[str, Any] = {
            "record": int(self.sb_record_no.value()),
            "roles": [_to_role_dict_for_state(r) for r in roles],
        }
        for role in roles:
            if not role.get("enabled", True):
                continue
            purpose = role["purpose"]
            src = role["source_channel"]
            target = role["target_name"]

            if purpose == "probe" and "rd.0" not in config:
                config["rd.0"] = src
            elif purpose == "stim_voltage":
                config["stimVolt"] = src
            elif purpose == "potentiometer":
                config["stimVolt_pot"] = src
            elif purpose == "stim_analog":
                config["stim"] = src
            elif purpose == "stim_trigger" and "stimVolt" not in config:
                config["stimVolt"] = src
            elif purpose == "thermode":
                config["thermode"] = src
            elif purpose == "button":
                config["button"] = src
            elif target:
                config[target] = src
        return config

    def set_from_aptrack_config(self, config: dict) -> None:
        """Populate roles from an APTrack config dict (legacy flat keys or ``roles``)."""
        if not config:
            return
        if "record" in config:
            self.sb_record_no.setValue(int(config["record"]))
        elif "record_no" in config:
            self.sb_record_no.setValue(int(config["record_no"]))

        if isinstance(config.get("roles"), list):
            self.set_roles(config["roles"])
            return

        key_map = {
            "rd.0": ("probe", "rd.0", MODE_ANALOG_BP, False),
            "stimVolt": ("stim_voltage", "env.stimVolt", MODE_TTL, True),
            "env.stimVolt": ("stim_voltage", "env.stimVolt", MODE_TTL, True),
            "stim": ("stim_analog", "env.stim", MODE_ANALOG, True),
            "env.stim": ("stim_trigger", "env.stim", MODE_TTL, True),
            "stimVolt_pot": ("potentiometer", "env.potentiometer", MODE_ANALOG, True),
            "env.potentiometer": ("potentiometer", "env.potentiometer", MODE_ANALOG, True),
            "thermode": ("thermode", "env.thermode", MODE_ANALOG, True),
            "env.thermode": ("thermode", "env.thermode", MODE_ANALOG, True),
            "button": ("button", "rec.button", MODE_TTL, True),
            "rec.button": ("button", "rec.button", MODE_TTL, True),
        }

        roles: List[Dict[str, Any]] = []
        for k, v in config.items():
            if k in ("record", "record_no"):
                continue
            src = str(v).strip()
            if not src:
                continue
            if k in key_map:
                purpose, target, mode, is_adc = key_map[k]
            else:
                purpose = _infer_purpose_from_target(k, "AS")
                target = k
                preset = PURPOSE_PRESETS[purpose]
                mode = (
                    MODE_TTL
                    if preset["probe_type"] == "TTL"
                    else (MODE_ANALOG_BP if preset.get("bandpass") else MODE_ANALOG)
                )
                is_adc = preset.get("is_adc", True)
            roles.append(
                normalize_role(
                    {
                        "enabled": True,
                        "purpose": purpose,
                        "source_channel": src,
                        "target_name": target,
                        "mode": mode,
                        "is_adc": is_adc,
                    }
                )
            )
            if k == "stimVolt" and "env.stim" not in config and "stim" not in config:
                roles.append(
                    normalize_role(
                        {
                            "enabled": True,
                            "purpose": "stim_trigger",
                            "source_channel": src,
                            "target_name": "env.stim",
                            "mode": MODE_TTL,
                            "is_adc": True,
                        }
                    )
                )

        if roles:
            self.set_roles(roles)

    # ------------------------------------------------------------------
    # Apply, Preprocess & Live Stream Sync
    # ------------------------------------------------------------------
    def _derive_live_updates(self, roles: List[Dict[str, Any]]) -> Dict[str, Any]:
        state_roles = [_to_role_dict_for_state(r) for r in roles]
        updates: Dict[str, Any] = {"channel_roles": state_roles}
        enabled = [r for r in roles if r.get("enabled", True)]
        probe_roles = [r for r in enabled if r["purpose"] == "probe"]
        if not probe_roles:
            probe_roles = [r for r in enabled if r["mode"] in (MODE_ANALOG, MODE_ANALOG_BP)]

        if probe_roles:
            primary = probe_roles[0]
            updates["rec_chan"] = _parse_channel_int(primary["source_channel"], 0)
            updates["online_filter"] = bool(primary["mode"] == MODE_ANALOG_BP)
            updates["filter_low_hz"] = float(primary.get("bandpass_low", 500.0))
            updates["filter_high_hz"] = float(primary.get("bandpass_high", 7000.0))

        analog_nums = []
        for r in enabled:
            if r["mode"] in (MODE_ANALOG, MODE_ANALOG_BP):
                m = re.search(r"(\d+)", str(r.get("source_channel", "")))
                if m:
                    num = int(m.group(1))
                    if num not in analog_nums:
                        analog_nums.append(num)
        if analog_nums:
            updates["active_channels"] = ",".join(str(n) for n in sorted(analog_nums))

        trig_roles = [
            r
            for r in enabled
            if r["purpose"] in ("stim_trigger", "stim_voltage", "ttl_custom")
            or r["mode"] == MODE_TTL
        ]
        if trig_roles:
            trig = trig_roles[0]
            src_upper = str(trig["source_channel"]).upper()
            if src_upper.startswith("ADC") or trig["mode"] == MODE_TTL:
                updates["trigger_source"] = "analog" if src_upper.startswith("ADC") else "ttl"
                updates["adc_trigger_chan"] = _parse_channel_int(src_upper, 1)
            thresh_v = float(trig.get("ttl_threshold_v", 0.0))
            if thresh_v > 0.0:
                updates["ttl_threshold_v"] = thresh_v
            updates["ttl_refractory_ms"] = float(trig.get("ttl_refractory_ms", 100.0))

        return updates

    def sync_to_live_stream(self, roles: Optional[List[Dict[str, Any]]] = None) -> None:
        """Push role, bandpass, and trigger settings to ViewerState and active workers without disconnecting."""
        roles = roles if roles is not None else self.get_roles()
        live_updates = self._derive_live_updates(roles)

        if self.state is not None:
            if hasattr(self.state, "update_live_config"):
                self.state.update_live_config(**live_updates)
            elif hasattr(self.state, "live_config") and isinstance(self.state.live_config, dict):
                self.state.live_config.update(live_updates)
                if hasattr(self.state, "onLiveConfigChanged"):
                    self.state.onLiveConfigChanged.emit()

        app = QApplication.instance()
        if app is None:
            return

        live_views: List[Any] = []
        curr = self.parentWidget()
        while curr is not None:
            if hasattr(curr, "worker") and hasattr(curr, "controller"):
                live_views.append(curr)
                break
            curr = curr.parentWidget() if hasattr(curr, "parentWidget") else None

        for top_w in app.topLevelWidgets():
            dm = getattr(top_w, "dock_manager", None)
            if dm is not None and hasattr(dm, "dockWidgets"):
                for dw in dm.dockWidgets():
                    w = dw.widget() if hasattr(dw, "widget") else None
                    if w is not None and hasattr(w, "worker") and w not in live_views:
                        live_views.append(w)

        for lv in live_views:
            worker = getattr(lv, "worker", None)
            if worker is None:
                continue
            try:
                if hasattr(worker, "set_channel_roles"):
                    worker.set_channel_roles(live_updates["channel_roles"])
                if "rec_chan" in live_updates and hasattr(worker, "set_rec_channel"):
                    worker.set_rec_channel(int(live_updates["rec_chan"]))
                if "online_filter" in live_updates and hasattr(worker, "set_filtering"):
                    worker.set_filtering(bool(live_updates["online_filter"]))
                if (
                    "filter_low_hz" in live_updates
                    and "filter_high_hz" in live_updates
                    and hasattr(worker, "set_filter_band")
                ):
                    low = float(live_updates["filter_low_hz"])
                    high = float(live_updates["filter_high_hz"])
                    if 0 < low < high:
                        worker.set_filter_band(low, high)
                if hasattr(worker, "set_ttl_params"):
                    worker.set_ttl_params(
                        threshold=live_updates.get("ttl_threshold_v"),
                        refractory_ms=live_updates.get("ttl_refractory_ms"),
                    )
                if "trigger_source" in live_updates:
                    worker.use_analog_trigger = live_updates["trigger_source"] == "analog"
                if "adc_trigger_chan" in live_updates:
                    worker.adc_trigger_chan = int(live_updates["adc_trigger_chan"])
            except Exception as exc:
                logger.warning("Failed to live-sync channel roles to worker: %s", exc)

    def _reload_aptrack_with_roles(
        self, folder: Path, roles: List[Dict[str, Any]], record_no: int
    ) -> Optional[neo.Segment]:
        all_files = list(folder.glob("*.continuous"))
        if not all_files:
            return None

        def _find_channel_file(chname: str) -> Optional[Path]:
            digits = re.sub(r"[^\d]", "", chname or "")
            if not digits:
                return None
            channo = int(digits)
            if chname.upper().startswith("ADC"):
                channo += 16
            rex = f"_(((CH)?{channo})|({re.escape(chname)}))"
            if record_no > 1:
                rex += f"(_{record_no})"
            rex += r"\.continuous"
            matches = [x for x in all_files if len(re.findall(rex, str(x.name))) == 1]
            return matches[0] if matches else None

        recordings: List[APTrackRecording] = []
        for role in roles:
            if not role.get("enabled", True):
                continue
            src_file = _find_channel_file(role["source_channel"])
            if src_file is None:
                continue
            pt = TypeID.TTL if role["mode"] == MODE_TTL else TypeID.ANALOG
            recordings.append(
                APTrackRecording(
                    filename=str(src_file),
                    probe_type=pt,
                    name=role["target_name"],
                    details=role.get("description", ""),
                    is_adc=bool(role.get("is_adc", True)),
                )
            )

        if not recordings:
            return None

        messages = folder / (
            "messages" + (f"_{record_no}" if record_no > 1 else "") + ".events"
        )
        seg = aptrack_as_neo(
            recordings,
            str(messages) if messages.exists() else None,
            record_no=record_no,
        )
        seg.annotate(original_folder=str(folder))
        preprocess_segment_with_roles(seg, roles, in_place=True)
        return seg

    @Slot()
    def apply_and_preprocess(self) -> None:
        """Save channel roles into ViewerState, preprocess/reload offline data, and live-sync."""
        roles = self.get_roles()
        state_roles = [_to_role_dict_for_state(r) for r in roles]
        self._syncing = True
        try:
            if self.state is not None:
                if hasattr(self.state, "set_channel_roles"):
                    is_aptrack = getattr(self.state, "_loaded_file_type", None) == "APTrack"
                    try:
                        self.state.set_channel_roles(
                            state_roles,
                            apply_preprocess=True,
                            reload_if_aptrack=is_aptrack,
                        )
                    except TypeError:
                        try:
                            self.state.set_channel_roles(state_roles, apply_preprocess=True)
                        except TypeError:
                            self.state.set_channel_roles(state_roles)
                else:
                    live_updates = self._derive_live_updates(roles)
                    if hasattr(self.state, "update_live_config"):
                        self.state.update_live_config(**live_updates)
                    elif hasattr(self.state, "live_config") and isinstance(
                        self.state.live_config, dict
                    ):
                        self.state.live_config.update(live_updates)

                    seg = getattr(self.state, "segment", None)
                    orig_folder = (
                        (getattr(seg, "annotations", None) or {}).get("original_folder")
                        if seg is not None
                        else None
                    )
                    target_folder = self._folder_path or (
                        Path(orig_folder) if orig_folder else None
                    )

                    new_seg = None
                    if (
                        target_folder is not None
                        and target_folder.exists()
                        and not getattr(self.state, "is_streaming", False)
                    ):
                        try:
                            new_seg = self._reload_aptrack_with_roles(
                                target_folder, roles, int(self.sb_record_no.value())
                            )
                        except Exception as exc:
                            logger.warning("APTrack role reload failed: %s", exc)

                    if new_seg is not None and hasattr(self.state, "set_segment"):
                        self.state.set_segment(new_seg)
                    elif seg is not None and not getattr(self.state, "is_streaming", False):
                        preprocess_segment_with_roles(seg, roles, in_place=True)
                        if hasattr(self.state, "set_segment"):
                            self.state.set_segment(seg)
                        else:
                            if hasattr(self.state, "onEventsUpdated"):
                                self.state.onEventsUpdated.emit()
                            if hasattr(self.state, "onChannelGroupsChanged"):
                                self.state.onChannelGroupsChanged.emit()

                if self.chk_auto_apply_live.isChecked():
                    self.sync_to_live_stream(roles)

            self.lbl_status.setText(
                f"Applied {sum(1 for r in roles if r['enabled'])} active channel role(s)."
            )
            self.lbl_status.setVisible(True)
        finally:
            self._syncing = False

        self.rolesApplied.emit(roles)

    # ------------------------------------------------------------------
    # Perspective Persistence
    # ------------------------------------------------------------------
    def get_settings(self) -> dict:
        return {
            "preset": self.combo_preset.currentText(),
            "record_no": int(self.sb_record_no.value()),
            "auto_apply_live": bool(self.chk_auto_apply_live.isChecked()),
            "roles": self.get_roles(),
        }

    def set_settings(self, settings: dict) -> None:
        if not settings:
            return
        self._syncing = True
        try:
            if "record_no" in settings:
                self.sb_record_no.setValue(int(settings["record_no"]))
            if "auto_apply_live" in settings:
                self.chk_auto_apply_live.setChecked(bool(settings["auto_apply_live"]))
            if "preset" in settings:
                idx = self.combo_preset.findText(str(settings["preset"]))
                if idx >= 0:
                    with QSignalBlocker(self.combo_preset):
                        self.combo_preset.setCurrentIndex(idx)
            if isinstance(settings.get("roles"), list):
                self._populate_table_from_roles(
                    [normalize_role(r) for r in settings["roles"]]
                )
        finally:
            self._syncing = False


class ChannelPurposeDialog(QDialog):
    """Modal dialog wrapping ``ChannelPurposeWidget`` with OK / Cancel buttons.

    Usable when importing an APTrack folder (as a rich companion/drop-in to
    ``APTrackDialog``) or opening "Channel Purposes & Preprocessing..." from
    the application menu.
    """

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        state: Optional[Any] = None,
        folder: Optional[Union[str, Path]] = None,
        initial_roles: Optional[List[Any]] = None,
        initial_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Channel Purposes & Preprocessing")
        self.resize(840, 560)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.widget = ChannelPurposeWidget(
            parent=self, state=state, compact=False, show_record_no=True
        )
        if folder is not None:
            self.widget.set_folder(folder)
        if initial_roles is not None:
            self.widget.set_roles(initial_roles)
        elif initial_config is not None:
            self.widget.set_from_aptrack_config(initial_config)

        layout.addWidget(self.widget, 1)

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

    def get_config(self) -> dict:
        return self.widget.get_aptrack_config()

    def get_aptrack_config(self) -> dict:
        return self.widget.get_aptrack_config()

    def set_from_aptrack_config(self, config: dict) -> None:
        self.widget.set_from_aptrack_config(config)

    def get_roles(self) -> List[dict]:
        return self.widget.get_roles()

    def set_roles(self, roles: List[Any]) -> None:
        self.widget.set_roles(roles)

    def set_folder(self, folder: Union[str, Path]) -> None:
        self.widget.set_folder(folder)
