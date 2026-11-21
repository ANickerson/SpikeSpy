# Code to load data generated using APTrack into neo
# ensure works with 04/04 data

import copy
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import neo
import numpy as np
import quantities as pq
from neo.core import AnalogSignal, Block, Event, Group, Segment
from scipy import signal


def apply_bandpass(data: AnalogSignal, lowHz=500, highHz=7000, sampleRate=None):
    """
    Apply a zero-phase Butterworth bandpass filter to the data.
    """
    out = np.array(data).copy()
    sr = sampleRate if sampleRate is not None else getattr(data, "sampling_rate", None)
    if sr is None:
        raise ValueError("sampleRate must be provided when data has no sampling_rate")
    sr_hz = float(sr.rescale(pq.Hz).magnitude) if hasattr(sr, "rescale") else float(sr)
    nyq = 0.5 * sr_hz
    low_norm = max(1e-6, min(float(lowHz) / nyq, 0.999999))
    high_norm = max(low_norm + 1e-6, min(float(highHz) / nyq, 0.999999))
    sos = signal.butter(2, [low_norm, high_norm], btype="bandpass", output="sos")
    arr = np.asarray(data)
    if arr.ndim == 2 and arr.shape[1] == 1:
        filt = signal.sosfiltfilt(sos, arr[:, 0])[:, None]
    elif arr.ndim == 2:
        filt = signal.sosfiltfilt(sos, arr, axis=0)
    else:
        filt = signal.sosfiltfilt(sos, arr)
    out[:] = filt
    units = getattr(data, "units", pq.dimensionless)
    return out * units


class TypeID(Enum):
    ANALOG = "AS"  # analogsignal channel
    TTL = "TTL"  # electricalstimulus - convert to timestamps
    EVENTS = "EVT"  # events files to convert into timestamps


def _coerce_probe_type(probe_type: Union[TypeID, str]) -> TypeID:
    if isinstance(probe_type, TypeID):
        return probe_type
    if isinstance(probe_type, str):
        val = probe_type.strip()
        upper = val.upper()
        for member in TypeID:
            if member.value.upper() == upper or member.name.upper() == upper:
                return member
    raise ValueError(f"Unsupported probe_type: {probe_type!r}")


@dataclass
class ChannelRoleSpec:
    """Declarative specification for a recording channel's role and preprocessing."""

    source: str
    name: str
    probe_type: Union[TypeID, str] = TypeID.ANALOG
    purpose: str = ""
    bandpass: bool = False
    bandpass_low: float = 500.0
    bandpass_high: float = 7000.0
    ttl_threshold: Optional[float] = None
    ttl_std_mult: float = 10.0
    ttl_min_width_ms: float = 0.4
    ttl_refractory_ms: float = 100.0
    is_adc: bool = True
    enabled: bool = True
    details: str = ""

    def __post_init__(self):
        self.probe_type = _coerce_probe_type(self.probe_type)
        self.source = str(self.source)
        self.name = str(self.name)
        self.purpose = str(self.purpose or "")
        self.bandpass = bool(self.bandpass)
        self.bandpass_low = float(self.bandpass_low)
        self.bandpass_high = float(self.bandpass_high)
        if self.ttl_threshold is not None:
            self.ttl_threshold = float(self.ttl_threshold)
        self.ttl_std_mult = float(self.ttl_std_mult)
        self.ttl_min_width_ms = float(self.ttl_min_width_ms)
        self.ttl_refractory_ms = float(self.ttl_refractory_ms)
        self.is_adc = bool(self.is_adc)
        self.enabled = bool(self.enabled)
        self.details = str(self.details or "")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "name": self.name,
            "probe_type": self.probe_type.value
            if isinstance(self.probe_type, TypeID)
            else str(self.probe_type),
            "purpose": self.purpose,
            "bandpass": self.bandpass,
            "bandpass_low": self.bandpass_low,
            "bandpass_high": self.bandpass_high,
            "ttl_threshold": self.ttl_threshold,
            "ttl_std_mult": self.ttl_std_mult,
            "ttl_min_width_ms": self.ttl_min_width_ms,
            "ttl_refractory_ms": self.ttl_refractory_ms,
            "is_adc": self.is_adc,
            "enabled": self.enabled,
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ChannelRoleSpec":
        if isinstance(d, ChannelRoleSpec):
            return copy.deepcopy(d)
        return cls(
            source=d.get("source", ""),
            name=d.get("name", d.get("source", "")),
            probe_type=d.get("probe_type", TypeID.ANALOG),
            purpose=d.get("purpose", ""),
            bandpass=d.get("bandpass", False),
            bandpass_low=d.get("bandpass_low", 500.0),
            bandpass_high=d.get("bandpass_high", 7000.0),
            ttl_threshold=d.get("ttl_threshold", None),
            ttl_std_mult=d.get("ttl_std_mult", 10.0),
            ttl_min_width_ms=d.get("ttl_min_width_ms", 0.4),
            ttl_refractory_ms=d.get("ttl_refractory_ms", 100.0),
            is_adc=d.get("is_adc", True),
            enabled=d.get("enabled", True),
            details=d.get("details", ""),
        )


DEFAULT_APTRACK_ROLES: List[ChannelRoleSpec] = [
    ChannelRoleSpec(
        source="CH1",
        name="rd.0",
        probe_type=TypeID.ANALOG,
        purpose="probe",
        bandpass=True,
        bandpass_low=500.0,
        bandpass_high=7000.0,
        is_adc=False,
        details="microneurography probe",
    ),
    ChannelRoleSpec(
        source="ADC4",
        name="env.stimVolt",
        probe_type=TypeID.TTL,
        purpose="stim_voltage",
        is_adc=True,
        details="A TTL of the stimulation voltage",
    ),
    ChannelRoleSpec(
        source="ADC4",
        name="env.stim",
        probe_type=TypeID.TTL,
        purpose="stim_trigger",
        is_adc=True,
        details="A TTL of the stimulation",
    ),
    ChannelRoleSpec(
        source="ADC2",
        name="env.potentiometer",
        probe_type=TypeID.ANALOG,
        purpose="potentiometer",
        is_adc=True,
        details="Potentiometer stimulation voltage",
    ),
    ChannelRoleSpec(
        source="ADC5",
        name="env.stim",
        probe_type=TypeID.ANALOG,
        purpose="stim_analog",
        is_adc=True,
        details="Analog signal of the stimulation",
    ),
    ChannelRoleSpec(
        source="ADC7",
        name="env.thermode",
        probe_type=TypeID.ANALOG,
        purpose="thermode",
        is_adc=True,
        details="Thermal stimulation temperatures",
    ),
    ChannelRoleSpec(
        source="ADC8",
        name="rec.button",
        probe_type=TypeID.TTL,
        purpose="button",
        is_adc=True,
        details="Manual button press, usually to signify a change in protocol or mechanical stimulation",
    ),
]


def get_default_channel_roles() -> List[ChannelRoleSpec]:
    return [copy.deepcopy(r) for r in DEFAULT_APTRACK_ROLES]


def _guess_purpose_from_name(name: str, probe_type: TypeID) -> str:
    low = (name or "").lower()
    if low.startswith("rd"):
        return "probe"
    if "stimvolt" in low:
        return "stim_voltage"
    if "pot" in low:
        return "potentiometer"
    if "thermode" in low or "temp" in low:
        return "thermode"
    if "button" in low:
        return "button"
    if "stim" in low:
        return "stim_trigger" if probe_type == TypeID.TTL else "stim_analog"
    return "aux"


def normalize_channel_roles(
    config_or_roles: Any = None,
    extra_channels: Optional[List[Any]] = None,
) -> List[ChannelRoleSpec]:
    """Normalize a legacy config dict, list of dicts, or list of ChannelRoleSpec into List[ChannelRoleSpec]."""
    roles: List[ChannelRoleSpec] = []

    if config_or_roles is None:
        roles = get_default_channel_roles()
    elif isinstance(config_or_roles, (list, tuple)):
        for item in config_or_roles:
            if isinstance(item, ChannelRoleSpec):
                roles.append(copy.deepcopy(item))
            elif isinstance(item, dict):
                roles.append(ChannelRoleSpec.from_dict(item))
            else:
                raise TypeError(f"Unsupported role entry: {type(item)!r}")
    elif isinstance(config_or_roles, dict):
        if "roles" in config_or_roles and config_or_roles["roles"] is not None:
            for item in config_or_roles["roles"]:
                if isinstance(item, ChannelRoleSpec):
                    roles.append(copy.deepcopy(item))
                elif isinstance(item, dict):
                    roles.append(ChannelRoleSpec.from_dict(item))
                else:
                    raise TypeError(f"Unsupported role entry in config['roles']: {type(item)!r}")
        else:
            # Convert legacy key/value dictionary to standard roles + any extra custom keys
            legacy_defaults = get_default_channel_roles()
            rd0_src = config_or_roles.get("rd.0")
            stim_volt_src = config_or_roles.get("stimVolt")
            stim_pot_src = config_or_roles.get("stimVolt_pot")
            stim_src = config_or_roles.get("stim")
            thermode_src = config_or_roles.get("thermode")
            button_src = config_or_roles.get("button")

            for r in legacy_defaults:
                if r.name == "rd.0" and r.purpose == "probe" and rd0_src:
                    r.source = str(rd0_src)
                    r.is_adc = str(rd0_src).upper().startswith("ADC")
                elif r.name == "env.stimVolt" and stim_volt_src:
                    r.source = str(stim_volt_src)
                elif r.name == "env.stim" and r.probe_type == TypeID.TTL and stim_volt_src:
                    r.source = str(stim_volt_src)
                elif r.name == "env.potentiometer" and stim_pot_src:
                    r.source = str(stim_pot_src)
                elif r.name == "env.stim" and r.probe_type == TypeID.ANALOG and stim_src:
                    r.source = str(stim_src)
                elif r.name == "env.thermode" and thermode_src:
                    r.source = str(thermode_src)
                elif r.name == "rec.button" and button_src:
                    r.source = str(button_src)
                roles.append(r)

            known_legacy_keys = {
                "record",
                "roles",
                "rd.0",
                "rd.0.bp",
                "stimVolt",
                "stimVolt_pot",
                "stim",
                "thermode",
                "button",
            }
            for k, v in config_or_roles.items():
                if k in known_legacy_keys or not isinstance(v, str) or not v.strip():
                    continue
                roles.append(
                    ChannelRoleSpec(
                        source=v.strip(),
                        name=str(k),
                        probe_type=TypeID.ANALOG,
                        purpose=_guess_purpose_from_name(str(k), TypeID.ANALOG),
                        bandpass=str(k).lower().startswith("rd"),
                        is_adc=v.strip().upper().startswith("ADC"),
                    )
                )
    else:
        raise TypeError(f"Unsupported config_or_roles type: {type(config_or_roles)!r}")

    if extra_channels:
        for ch in extra_channels:
            if isinstance(ch, ChannelRoleSpec):
                roles.append(copy.deepcopy(ch))
            elif isinstance(ch, dict):
                roles.append(ChannelRoleSpec.from_dict(ch))
            elif isinstance(ch, APTrackRecording):
                roles.append(
                    ChannelRoleSpec(
                        source=str(ch.filename),
                        name=ch.name,
                        probe_type=ch.probe_type,
                        purpose=ch.purpose or _guess_purpose_from_name(ch.name, ch.probe_type),
                        bandpass=bool(ch.bandpass) if ch.bandpass is not None else ch.name.lower().startswith("rd"),
                        bandpass_low=ch.bandpass_low,
                        bandpass_high=ch.bandpass_high,
                        ttl_threshold=ch.ttl_threshold,
                        ttl_std_mult=ch.ttl_std_mult,
                        ttl_min_width_ms=ch.ttl_min_width_ms,
                        ttl_refractory_ms=ch.ttl_refractory_ms,
                        is_adc=ch.is_adc,
                        details=ch.details,
                    )
                )

    return roles


def readHeader(f):
    header = {}
    h = f.read(1024).decode().replace("\n", "").replace("header.", "")
    for i, item in enumerate(h.split(";")):
        if "=" in item:
            header[item.split(" = ")[0].strip()] = item.split(" = ")[1]

    try:
        header["date_created"] = datetime.strptime(
            header["date_created"], "'%d-%b-%Y %H:%M:%S'"
        )
    except Exception:
        header["date_created"] = datetime.strptime(
            header["date_created"], "'%d-%b-%Y %H%M%S'"
        )  # odd in 22-Nov-22
    return header


def readContinous(filename):
    NUM_HEADER_BYTES = 1024
    openephysdtype = np.dtype(
        [
            ("timestamp", "<i8"),
            ("N", np.uint16),
            ("recording", np.uint16),
            ("data", ">i2", 1024),
            ("rec", np.uint8, 10),
        ]
    )
    with open(filename, "rb") as fh:
        header = readHeader(fh)
    mmap = np.memmap(filename, mode="r", dtype=openephysdtype, offset=NUM_HEADER_BYTES)
    return (header, mmap)


@dataclass
class APTrackRecording:
    filename: str
    probe_type: TypeID
    name: str
    details: str = ""
    summarize: str = None
    is_adc: bool = True
    bandpass: Optional[bool] = None
    bandpass_low: float = 500.0
    bandpass_high: float = 7000.0
    ttl_threshold: Optional[float] = None
    ttl_std_mult: float = 10.0
    ttl_min_width_ms: float = 0.4
    ttl_refractory_ms: float = 100.0
    purpose: str = ""


def extract_ttl_event_from_array(
    data,
    sampling_rate,
    t_start=0 * pq.s,
    units=pq.V,
    name: str = "env.stim",
    threshold: Optional[float] = None,
    std_mult: float = 10.0,
    min_width_ms: float = 0.4,
    refractory_ms: float = 100.0,
    file_origin: Optional[str] = None,
    date_created: Any = None,
    description: str = "",
) -> Event:
    """Extract rising-edge TTL events from a 1D analog array."""
    arr = np.asarray(data, dtype=np.float64).reshape(-1)
    sr_q = sampling_rate if hasattr(sampling_rate, "rescale") else (float(sampling_rate) * pq.Hz)
    sr_hz = float(sr_q.rescale(pq.Hz).magnitude)
    t_start_q = t_start if hasattr(t_start, "rescale") else (float(t_start) * pq.s)

    if arr.size < 2:
        kwargs = {
            "units": pq.s,
            "type_id": TypeID.TTL.value,
            "name": name,
            "array_annotations": {
                "duration": np.array([], dtype=np.float64) * pq.s,
                "maximum": np.array([], dtype=np.float64) * pq.mV,
            },
            "description": description,
        }
        if file_origin is not None:
            kwargs["file_origin"] = file_origin
        if date_created is not None:
            kwargs["date_created"] = date_created
        return Event(np.array([], dtype=np.float64) * pq.s, **kwargs)

    if threshold is None or threshold <= 0:
        m = float(np.mean(arr))
        eff_threshold = (float(std_mult) * float(np.std(arr - m))) + m
    else:
        eff_threshold = float(threshold)

    width_samples = max(0, int(round(sr_hz * (float(min_width_ms) / 1000.0))))
    idxs = find_square_pulse_numpy(arr, width_samples, eff_threshold)
    idxs_rising = idxs[0]
    t_rising_ms = (idxs_rising / sr_q).rescale("ms")

    to_keep = np.ones(shape=len(t_rising_ms), dtype=bool)
    refractory_val = float(refractory_ms)
    if refractory_val > 0 and len(t_rising_ms) > 1:
        last_kept_t = float(t_rising_ms[0].magnitude)
        for i in range(1, len(t_rising_ms)):
            cur_t = float(t_rising_ms[i].magnitude)
            if (cur_t - last_kept_t) < refractory_val:
                to_keep[i] = False
            else:
                last_kept_t = cur_t

    t_rising_kept = t_rising_ms[to_keep]
    durations = (np.array(idxs[1][to_keep] - idxs[0][to_keep] - 1) / sr_q).rescale("s")
    raw_max = idxs[2][to_keep] * units
    try:
        maximums = raw_max.rescale("mV")
    except Exception:
        maximums = raw_max

    evt_kwargs: Dict[str, Any] = {
        "type_id": TypeID.TTL.value,
        "name": name,
        "array_annotations": {
            "duration": durations,
            "maximum": maximums,
        },
        "description": description,
    }
    if file_origin is not None:
        evt_kwargs["file_origin"] = file_origin
    if date_created is not None:
        evt_kwargs["date_created"] = date_created

    return Event(t_rising_kept.rescale("s") + t_start_q, **evt_kwargs)


def as_neo(
    mng_files: List[APTrackRecording],
    aptrack_events: str = None,
    record_no=1,
    fix_timestamps=True,
):
    """
    Takes the openephys files and creates a single neo object containing the experiment data

    mng_files: defines the categories of mng files
    mng_events: the filename of the messages.events file to use (should identify spikes)
    """

    seg = Segment()
    sampling_rate = 30000.0 * pq.Hz
    for f in mng_files:
        probe_type = _coerce_probe_type(f.probe_type)
        header, ch_mmap = readContinous(f.filename)
        sampling_rate = float(header["sampleRate"]) * pq.Hz
        if f.is_adc:
            ch_units_probe = pq.CompoundUnit(f'{header["bitVolts"]} * V')
        else:
            ch_units_probe = pq.CompoundUnit(f'{header["bitVolts"]} * uV')

        if record_no is not None and record_no >= 1 and False:  # DISABLED
            ch_mmap2 = ch_mmap[ch_mmap["recording"] == record_no - 1]
        else:
            ch_mmap2 = ch_mmap

        if len(np.unique(ch_mmap2["recording"])) > 1:
            data = np.zeros(
                shape=(ch_mmap2["timestamp"][-1] + 1024) - ch_mmap2["timestamp"][0],
                dtype=np.int16,
            )
            for x in ch_mmap2:  # SLOOOOOW
                ts = x["timestamp"] - ch_mmap2["timestamp"][0]
                data[ts : ts + 1024] = x["data"]
        else:
            data = ch_mmap2["data"].flat

        t_start = (ch_mmap2["timestamp"][0] / float(header["sampleRate"])) * pq.s
        if probe_type == TypeID.ANALOG:
            asig = AnalogSignal(
                data,
                sampling_rate=sampling_rate,
                units=ch_units_probe,
                type_id=probe_type.value,
                name=f.name,
                description=f.details,
                t_start=t_start,
                file_origin=str(f.filename),
                date_created=header.get("date_created"),
            )
            if f.purpose:
                asig.annotations["channel_purpose"] = f.purpose
            asig.annotations.setdefault("raw_channel_name", f.name)
            seg.analogsignals.append(asig)
            do_bp = f.bandpass if f.bandpass is not None else f.name.lower().startswith("rd")
            if do_bp:
                bp_sig = AnalogSignal(
                    apply_bandpass(asig, f.bandpass_low, f.bandpass_high),
                    sampling_rate=sampling_rate,
                    type_id=probe_type.value,
                    name=f.name + ".bp",
                    description=f.details + f"\n bandpass {f.bandpass_low:g}-{f.bandpass_high:g}Hz",
                    t_start=t_start,
                    file_origin=str(f.filename),
                    date_created=header.get("date_created"),
                )
                bp_sig.annotations["derived_from_role"] = True
                bp_sig.annotations["source_channel"] = f.name
                bp_sig.annotations["channel_purpose"] = "bandpass"
                seg.analogsignals.append(bp_sig)
        elif probe_type == TypeID.TTL:
            evt = extract_ttl_event_from_array(
                data,
                sampling_rate=sampling_rate,
                t_start=t_start,
                units=ch_units_probe,
                name=f.name,
                threshold=f.ttl_threshold,
                std_mult=f.ttl_std_mult,
                min_width_ms=f.ttl_min_width_ms,
                refractory_ms=f.ttl_refractory_ms,
                file_origin=str(f.filename),
                date_created=header.get("date_created"),
                description=f.details,
            )
            if f.purpose:
                evt.annotations["channel_purpose"] = f.purpose
            evt.annotations["derived_from_role"] = True
            seg.events.append(evt)
        elif probe_type == TypeID.EVENTS:
            raise Exception("WIP")
        else:
            raise Exception(f"Unsupported probe type '{f.probe_type}'")

    if aptrack_events is not None:
        (
            spike_events,
            stimChange_events,
            protocol_events,
            extra_events,
        ) = parse_APTrackEvents(aptrack_events)
        # HACK - find nearest event and add offset - due to bug in APTrack not saving the true latencies
        if fix_timestamps and len(spike_events) > 0:
            stim_evt = next((x for x in seg.events if x.name == "env.stim"), None)
            if stim_evt is None:
                stim_evt = next(
                    (
                        x
                        for x in seg.events
                        if x.annotations.get("type_id") == TypeID.TTL.value
                        and not (x.name or "").startswith("unit")
                    ),
                    None,
                )
            if stim_evt is not None and len(stim_evt) > 0:
                new_spike_events = []
                for i in range(len(spike_events)):
                    x = spike_events[i]
                    idx = np.searchsorted(stim_evt, x) - 1
                    if idx == -1:
                        new_spike_events.append(x)
                        continue
                    spike_ts = stim_evt[idx]
                    new_spike_events.append(
                        spike_ts
                        + (
                            (
                                spike_events.array_annotations["spikeSampleLatency"][i]
                                / sampling_rate.base
                            )
                            * pq.s
                        )
                    )
                spike_events = Event(
                    np.array(new_spike_events),
                    units=pq.second,
                    name=spike_events.name,
                    array_annotations=spike_events.array_annotations,
                )

        units = []
        try:
            for x in np.unique(spike_events.array_annotations["spikeGroup"]):
                e = spike_events[spike_events.array_annotations["spikeGroup"] == x]
                e.name = f"unit_{x}"
                units.append(e)
        except Exception:
            pass
        seg.events += units
        seg.events += [stimChange_events, protocol_events, extra_events]
    return seg


def _is_preserved_event(evt: Event) -> bool:
    name = getattr(evt, "name", "") or ""
    if name.startswith("unit") or evt.annotations.get("type") == "unit":
        return True
    if name.startswith("log."):
        return True
    return False


def _match_source_signal(
    source_name: str,
    role_name: str,
    candidates: List[AnalogSignal],
) -> Optional[AnalogSignal]:
    """Find a raw/source AnalogSignal matching role.source or role.name."""
    if not candidates:
        return None
    targets = [t for t in (source_name, role_name) if t]
    # 1. Exact match on raw_channel_name or current name
    for target in targets:
        for sig in candidates:
            raw_name = sig.annotations.get("raw_channel_name")
            if raw_name == target or sig.name == target:
                return sig
    # 2. Case-insensitive match
    targets_low = [t.lower() for t in targets]
    for target_low in targets_low:
        for sig in candidates:
            raw_name = (sig.annotations.get("raw_channel_name") or "").lower()
            cur_name = (sig.name or "").lower()
            if raw_name == target_low or cur_name == target_low:
                return sig
    # 3. CH1 <-> CH0 style fallback when explicit 1-based/0-based Open Ephys naming differs
    m = re.fullmatch(r"ch(\d+)", (source_name or "").strip(), re.IGNORECASE)
    if m:
        idx_1based = int(m.group(1))
        alt_names = {f"ch{idx_1based - 1}", f"ch_{idx_1based - 1}"} if idx_1based >= 1 else set()
        for sig in candidates:
            raw_name = (sig.annotations.get("raw_channel_name") or "").lower()
            cur_name = (sig.name or "").lower()
            if raw_name in alt_names or cur_name in alt_names:
                return sig
    return None


def preprocess_segment_with_roles(
    seg: Segment,
    roles: Union[List[Union[ChannelRoleSpec, Dict[str, Any]]], Dict[str, Any], None],
    in_place: bool = False,
) -> Segment:
    """Apply or re-apply ChannelRoleSpec preprocessing rules to any neo.Segment."""
    target_seg = seg if in_place else copy.deepcopy(seg)
    norm_roles = normalize_channel_roles(roles)

    # Identify original/source analog signals (exclude previously derived .bp or aliased role copies)
    raw_signals: List[AnalogSignal] = []
    for sig in target_seg.analogsignals:
        sig_name = getattr(sig, "name", "") or ""
        if sig.annotations.get("derived_from_role") or sig_name.endswith(".bp"):
            continue
        sig.annotations.setdefault("raw_channel_name", sig_name)
        raw_signals.append(sig)

    # Count how many enabled ANALOG roles reference each raw signal so we know when renaming vs aliasing
    analog_roles_per_sig: Dict[int, int] = {}
    matched_sources: List[Optional[AnalogSignal]] = []
    for role in norm_roles:
        if not role.enabled:
            matched_sources.append(None)
            continue
        src_sig = _match_source_signal(role.source, role.name, raw_signals)
        matched_sources.append(src_sig)
        if src_sig is not None and _coerce_probe_type(role.probe_type) == TypeID.ANALOG:
            analog_roles_per_sig[id(src_sig)] = analog_roles_per_sig.get(id(src_sig), 0) + 1

    # Remove previously derived analog signals (e.g. .bp or derived_from_role)
    role_names = {r.name for r in norm_roles if r.enabled}
    kept_analog: List[AnalogSignal] = list(raw_signals)

    # Remove previously derived events (keep unit events, log.* events, and non-derived events not overwritten by roles)
    ttl_role_names = {
        r.name for r in norm_roles if r.enabled and _coerce_probe_type(r.probe_type) == TypeID.TTL
    }
    kept_events: List[Event] = []
    for evt in target_seg.events:
        if _is_preserved_event(evt):
            kept_events.append(evt)
            continue
        if evt.annotations.get("derived_from_role"):
            continue
        if (evt.name or "") in ttl_role_names:
            continue
        kept_events.append(evt)

    # Track renamed signals vs newly created aliases/bandpass signals
    renamed_sigs = set()
    appended_analog: List[AnalogSignal] = []
    appended_events: List[Event] = []

    for role, src_sig in zip(norm_roles, matched_sources):
        if not role.enabled or src_sig is None:
            continue
        probe_type = _coerce_probe_type(role.probe_type)
        raw_name = src_sig.annotations.get("raw_channel_name", src_sig.name)

        if probe_type == TypeID.ANALOG:
            if (
                role.name != src_sig.name
                and analog_roles_per_sig.get(id(src_sig), 0) == 1
                and id(src_sig) not in renamed_sigs
            ):
                src_sig.name = role.name
                src_sig.annotations["source_channel"] = raw_name
                if role.purpose:
                    src_sig.annotations["channel_purpose"] = role.purpose
                if role.details:
                    src_sig.description = role.details
                renamed_sigs.add(id(src_sig))
                active_sig = src_sig
            elif role.name == src_sig.name:
                src_sig.annotations["source_channel"] = raw_name
                if role.purpose:
                    src_sig.annotations["channel_purpose"] = role.purpose
                if role.details and not src_sig.description:
                    src_sig.description = role.details
                active_sig = src_sig
            else:
                # Alias/copy when multiple roles share the same source signal
                alias_sig = AnalogSignal(
                    np.asarray(src_sig),
                    units=src_sig.units,
                    sampling_rate=src_sig.sampling_rate,
                    t_start=src_sig.t_start,
                    name=role.name,
                    description=role.details or src_sig.description or "",
                    file_origin=src_sig.file_origin,
                )
                alias_sig.annotations.update(copy.deepcopy(src_sig.annotations))
                alias_sig.annotations["derived_from_role"] = True
                alias_sig.annotations["raw_channel_name"] = raw_name
                alias_sig.annotations["source_channel"] = raw_name
                if role.purpose:
                    alias_sig.annotations["channel_purpose"] = role.purpose
                appended_analog.append(alias_sig)
                active_sig = alias_sig

            if role.bandpass:
                bp_arr = apply_bandpass(
                    active_sig,
                    lowHz=role.bandpass_low,
                    highHz=role.bandpass_high,
                    sampleRate=active_sig.sampling_rate,
                )
                bp_sig = AnalogSignal(
                    bp_arr,
                    sampling_rate=active_sig.sampling_rate,
                    type_id=TypeID.ANALOG.value,
                    name=f"{role.name}.bp",
                    description=(role.details or active_sig.description or "")
                    + f"\n bandpass {role.bandpass_low:g}-{role.bandpass_high:g}Hz",
                    t_start=active_sig.t_start,
                    file_origin=active_sig.file_origin,
                )
                bp_sig.annotations["derived_from_role"] = True
                bp_sig.annotations["source_channel"] = raw_name
                bp_sig.annotations["channel_purpose"] = "bandpass"
                appended_analog.append(bp_sig)

        elif probe_type == TypeID.TTL:
            arr_1d = np.asarray(src_sig.magnitude).reshape(-1)
            evt = extract_ttl_event_from_array(
                arr_1d,
                sampling_rate=src_sig.sampling_rate,
                t_start=src_sig.t_start,
                units=src_sig.units,
                name=role.name,
                threshold=role.ttl_threshold,
                std_mult=role.ttl_std_mult,
                min_width_ms=role.ttl_min_width_ms,
                refractory_ms=role.ttl_refractory_ms,
                file_origin=src_sig.file_origin,
                description=role.details,
            )
            evt.annotations["derived_from_role"] = True
            evt.annotations["source_channel"] = raw_name
            if role.purpose:
                evt.annotations["channel_purpose"] = role.purpose
            # Replace any previously appended event with same name
            appended_events = [e for e in appended_events if e.name != role.name]
            appended_events.append(evt)

    # Deduplicate appended_analog by name (later role wins) while keeping raw_signals intact
    dedup_appended: List[AnalogSignal] = []
    existing_names = {s.name for s in kept_analog}
    for s in appended_analog:
        if s.name in existing_names:
            kept_analog = [k for k in kept_analog if k.name != s.name or not k.annotations.get("derived_from_role")]
        dedup_appended = [d for d in dedup_appended if d.name != s.name]
        dedup_appended.append(s)

    target_seg.analogsignals = kept_analog + dedup_appended
    target_seg.events = kept_events + appended_events
    target_seg.annotations["channel_roles"] = [r.to_dict() for r in norm_roles]
    return target_seg


def parse_APTrackEvents(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
    stimulation_volts = []
    spikeInfo = []
    protocolInfo = []
    curProtocol = ""
    curProtocolNo = 0
    curProtocolStep = 0
    sampleRate = 30000  # TODO: this should be found... somewhere? or neo may be able to handle timestamps
    sep = None
    other_messages = []
    for l in lines:
        if sep is None:
            val = re.finditer("\\d+([, ])", l)
            try:
                sep = next(val).group(1)
            except Exception:
                continue
        l2 = l.split(sep, 1)
        if len(l2) == 1:
            continue
        timestamp = int(l2[0])
        message = l2[1].strip()
        if message.find("Processor: ") >= 0:
            sampleRate = int(re.findall(r"(\d*)Hz", message)[0])
        elif message.find("setStimVoltage") >= 0:
            voltage = float(message.split(":")[1])
            stimulation_volts.append([timestamp, {"voltage": voltage}])
            continue
        elif message.find("spikeSampleNumber") >= 0:
            d = json.loads(
                message.replace("'", '"')
            )  # TODO: fix the json encoding in the plugin
            spikeInfo.append([timestamp, d])
        elif message.find("starting stimulus") >= 0:
            curProtocol = message.split("starting stimulus protocol ")[1]
            curProtocolNo += 1
            curProtocolStep = 0

        elif message.find("voltage:") >= 0:  # TODO: convert this into json encoded.
            elements = message.split(";")
            curProtocolStep += 1
            d = {e.split(":")[0].strip(): float(e.split(":")[1]) for e in elements}
            d["protocol"] = curProtocol
            d["protocolIdx"] = curProtocolNo
            d["step"] = curProtocolStep
            protocolInfo.append([timestamp, d])

        else:
            logging.warning(f"did not parse log line: {message}")
            other_messages.append([timestamp, {"message": message}])

    def make_evt(arr):  # an arr containing [(timestamp, {details}),(...)]
        array_annotations = {}
        for _, evt in arr:
            for k, v in evt.items():
                array_annotations[k] = array_annotations.get(k, []) + [v]
        for k, v in array_annotations.items():
            array_annotations[k] = np.array(v)

        return Event(  # A bit clunky - for spikeEvents this uses spikeSampleNumber, otherwise use the current timestamp
            np.array(
                [
                    (
                        (
                            x[0]
                        )
                        / sampleRate
                    )
                    for x in arr
                ]
            )
            * pq.s,
            array_annotations=array_annotations,
        )

    evt_spikeInfo = make_evt(
        spikeInfo
    )  # TODO: this needs to split by spikeGroup to work in viewer

    evt_stimulation_volts = make_evt(stimulation_volts)
    evt_stimulation_volts.name = "log.stimVolt"
    evt_protocolInfo = make_evt(protocolInfo)
    evt_protocolInfo.name = "log.protocol"

    evt_other = make_evt(other_messages)
    evt_other.name = "log.other"

    return [evt_spikeInfo, evt_stimulation_volts, evt_protocolInfo, evt_other]


def process_folder(
    foldername: str,
    record_no=1,
    config=None,
    extra_channels: List[APTrackRecording] = [],
):
    """
    Taking a given folder, convert to neo using the as_neo function.

    config may be:
      - None (uses DEFAULT_APTRACK_ROLES)
      - a legacy dict (e.g. {"rd.0": "CH1", "stimVolt": "ADC4", ...})
      - a dict containing "roles": List[ChannelRoleSpec | dict]
      - a List[ChannelRoleSpec | dict]
    """
    all_files = list(Path(foldername).glob("*.continuous"))

    def find_channel(chname, foldername=None):
        digits = re.sub("[^\\d]", "", str(chname))
        if not digits:
            matches = [x for x in all_files if str(chname) in str(x.name)]
            return matches[0] if matches else None
        channo = int(digits)
        if str(chname).upper().startswith("ADC"):
            channo += 16  # sometimes there is just the channel number for ADC
        rex = f"_(((CH)?{channo})|({re.escape(str(chname))}))"
        if record_no > 1:
            rex += f"(_{record_no})"
        rex += ".continuous"

        matches = [x for x in all_files if len(re.findall(rex, str(x.name))) == 1]
        if len(matches) == 0:
            logging.warning(f"did not find channel: {chname}")
            return None
        return matches[0]

    roles = normalize_channel_roles(config, extra_channels=extra_channels)

    signals: List[APTrackRecording] = []
    for role in roles:
        if not role.enabled:
            continue
        probe_type = _coerce_probe_type(role.probe_type)
        ch_file = find_channel(role.source)
        if ch_file is None:
            continue
        signals.append(
            APTrackRecording(
                filename=ch_file,
                probe_type=probe_type,
                name=role.name,
                details=role.details,
                is_adc=role.is_adc,
                bandpass=role.bandpass,
                bandpass_low=role.bandpass_low,
                bandpass_high=role.bandpass_high,
                ttl_threshold=role.ttl_threshold,
                ttl_std_mult=role.ttl_std_mult,
                ttl_min_width_ms=role.ttl_min_width_ms,
                ttl_refractory_ms=role.ttl_refractory_ms,
                purpose=role.purpose,
            )
        )

    messages = Path(foldername) / (
        "messages" + (f"_{record_no}" if record_no > 1 else "") + ".events"
    )

    seg = as_neo(
        signals, str(messages) if messages.exists() else None, record_no=record_no
    )

    settings_info = Path(foldername) / (
        "settings" + (f"_{record_no}" if record_no > 1 else "") + ".xml"
    )
    if settings_info.exists():
        seg.annotate(settings_info=settings_info.read_text())
    settings_info = Path(foldername) / (
        "Continuous_Data" + (f"_{record_no}" if record_no > 1 else "") + ".openephys"
    )
    if settings_info.exists():
        seg.annotate(Continuous_Data=settings_info.read_text())

    seg.annotate(
        original_folder=str(foldername),
        channel_roles=[r.to_dict() for r in roles],
    )

    return seg


from numpy.lib.stride_tricks import sliding_window_view


def find_square_pulse_numpy(arr, width, threshold):
    up = arr > threshold

    transitions_up = np.where(~up[:-1] & up[1:])[0]
    transitions_down = np.where(up[:-1] & ~up[1:])[0]

    if len(transitions_up) == 0 or len(transitions_down) == 0:
        return (
            np.array([], dtype=int),
            np.array([], dtype=int),
            np.array([], dtype=arr.dtype if hasattr(arr, "dtype") else float),
        )

    transitions_down_idx = 0
    starts = []
    ends = []
    maxes = []
    for start in transitions_up:
        while (
            transitions_down_idx < len(transitions_down)
            and transitions_down[transitions_down_idx] <= start
        ):
            transitions_down_idx += 1
        if transitions_down_idx >= len(transitions_down):
            break
        end = transitions_down[transitions_down_idx]

        if end - start > width:
            maxval = np.max(arr[start + 1 : end + 1])
            starts.append(start)
            ends.append(end)
            maxes.append(maxval)

    return np.array(starts, dtype=int), np.array(ends, dtype=int), np.array(maxes)


def find_square_pulse(arr, width, threshold):
    """
    convert analog arr into timepoints.
    arr = a 1d array of the signal
    width = number of points used for a local average to use when identifying step changes
    threshold = the difference between the signal and average to indicate a change

    returns (
            indexes,
            direction (1 for step up, 0 for step down),
            maxima (the maximum value since the previous index
        )
    """
    maxno = 20000
    out = np.zeros(maxno, dtype=np.int32)
    direction = np.zeros(maxno, dtype=np.int8)
    maxima = np.zeros(maxno, dtype=np.float32)
    buffer = np.zeros(width, dtype=arr.dtype)

    out_i = 0
    prev = 0
    max_val = 0
    i = 0
    sum_val = np.sum(arr[0:width])
    for x in range(arr.shape[0]):
        start = x - width
        if start < 0:
            start = 0
        sum_val = (
            sum_val + arr[x] - arr[start]
        )  # recalculate average. increase in coding complexity, reduces runtime dramatically
        v = np.abs(arr[x] - (sum_val / buffer.shape[0]))

        if v > threshold and prev < threshold:
            out[out_i] = x
            direction[out_i] = 0
            maxima[out_i] = max_val
            max_val = 0
            out_i += 1

        if v < threshold and prev > threshold:
            out[out_i] = x
            direction[out_i] = 1
            maxima[out_i] = max_val
            max_val = 0
            out_i += 1
        if max_val < v:
            max_val = v
        prev = v
        buffer[i] = v
        i += 1

        if i > width - 1:
            i = 0
    return (out[:out_i], direction[:out_i], maxima[:out_i])


def process_oe_binary(folder, config=None):
    """
    Load an Open Ephys binary recording folder and preprocess channels using ChannelRoleSpec roles.
    """
    from open_ephys.analysis import Session

    sess = Session(folder)
    recording = sess.recordings[0]
    dat = recording.continuous[0]

    sampling_rate = dat.metadata["sample_rate"] * pq.Hz
    t_start = (dat.sample_numbers[0] / sampling_rate) * pq.s
    channel_names = list(dat.metadata["channel_names"])
    bit_volts = list(dat.metadata["bit_volts"])

    roles = normalize_channel_roles(config)
    seg = Segment()

    # Load all raw channels referenced by enabled roles (plus preserve raw names)
    raw_cache: Dict[str, AnalogSignal] = {}
    for ch_idx, ch_name in enumerate(channel_names):
        ch_units_probe = pq.UnitQuantity(
            "kmicroVolts", pq.uV * bit_volts[ch_idx], symbol="kuV"
        )
        asig = AnalogSignal(
            dat.samples[:, ch_idx].flat,
            sampling_rate=sampling_rate,
            units=ch_units_probe,
            type_id=TypeID.ANALOG.value,
            name=ch_name,
            t_start=t_start,
            file_origin=str(folder),
        )
        asig.annotations["raw_channel_name"] = ch_name
        raw_cache[ch_name] = asig

    # Build a temporary segment containing the referenced raw signals (or all raw signals)
    referenced_sources = {r.source for r in roles if r.enabled}
    for ch_name, asig in raw_cache.items():
        if ch_name in referenced_sources or not referenced_sources:
            seg.analogsignals.append(asig)

    seg = preprocess_segment_with_roles(seg, roles, in_place=True)

    all_txt = []
    all_tstamps = []
    for p in Path(recording.directory).glob("events/**/TEXT_*"):
        all_txt.append(np.load(p / "text.npy"))
        all_tstamps.append(np.load(p / "timestamps.npy"))

    if all_txt:
        all_txt = np.concatenate(all_txt)
        all_tstamps = np.concatenate(all_tstamps)
        spike_events, stimChange_events, protocol_events = parse_APTrackvents_bin(
            all_txt, all_tstamps, sampling_rate
        )

        units = []
        try:
            for x in np.unique(spike_events.array_annotations["spikeGroup"]):
                e = spike_events[spike_events.array_annotations["spikeGroup"] == x]
                e.name = f"unit_{x}"
                units.append(e)
        except Exception:
            pass
        seg.events += [stimChange_events, protocol_events]
        seg.events += units
    return seg


def parse_APTrackvents_bin(events, tstamps, sampleRate=30000):
    stimulation_volts = []
    spikeInfo = []
    protocolInfo = []
    curProtocol = ""
    curProtocolNo = 0
    curProtocolStep = 0

    for l, timestamp in zip(events, tstamps):

        message = l.decode()
        if message.find("Processor: ") >= 0:
            sampleRate = int(re.findall(r"(\d*)Hz", message)[0])
        if message.find("setStimVoltage") >= 0:
            voltage = float(message.split(":")[1])
            stimulation_volts.append([timestamp, {"voltage": voltage}])
            continue
        elif message.find("spikeSampleNumber") >= 0:
            d = json.loads(
                message.replace("'", '"')
            )  # TODO: fix the json encoding in the plugin
            spikeInfo.append([timestamp, d])
        elif message.find("starting stimulus") >= 0:
            curProtocol = message.split("starting stimulus protocol ")[1]
            curProtocolNo += 1
            curProtocolStep = 0

        elif message.find("voltage:") >= 0:  # TODO: convert this into json encoded.
            elements = message.split(";")
            curProtocolStep += 1
            d = {e.split(":")[0].strip(): float(e.split(":")[1]) for e in elements}
            d["protocol"] = curProtocol
            d["protocolIdx"] = curProtocolNo
            d["step"] = curProtocolStep
            protocolInfo.append([timestamp, d])

        else:
            logging.warning(f"did not parse log line: {message}")

    def make_evt(arr):
        array_annotations = {}
        for _, evt in arr:
            for k, v in evt.items():

                array_annotations[k] = array_annotations.get(k, []) + [v]
        for k, v in array_annotations.items():
            array_annotations[k] = np.array(v)

        return Event(  # A bit clunky - for spikeEvents this uses spikeSampleNumber, otherwise use the current timestamp
            np.array(
                [
                    (((x[1].get("spikeSampleNumber", 0)) or x[0]) / sampleRate)
                    for x in arr
                ]
            )
            * pq.s,
            array_annotations=array_annotations,
        )

    evt_spikeInfo = make_evt(
        spikeInfo
    )  # TODO: this needs to split by spikeGroup to work in viewer

    evt_stimulation_volts = make_evt(stimulation_volts)
    evt_stimulation_volts.name = "log.stimVolt"
    evt_protocolInfo = make_evt(protocolInfo)
    evt_protocolInfo.name = "log.protocol"

    return [evt_spikeInfo, evt_stimulation_volts, evt_protocolInfo]


try:
    import numba

    find_square_pulse = numba.jit(find_square_pulse)
except:
    logging.info("optional dependency numba will speed up find_square_pulse method")

# ##!
# def test_load():
#     path = "/Users/xs19785/Library/CloudStorage/OneDrive-UniversityofBristol/Microneurography/Data/2023-03-14_14-11-52/Record Node 104"
#     data = process_folder(path)
