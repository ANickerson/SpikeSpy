"""I/O Manager for SpikeSpy: File loading, export, and persistence.

Extracts file loading (load_file, prompt_for_neo_file, open_matlab_to_neo,
APTrackDialog, expand_multichannel_segment) and persistence helpers
(export_units_csv, import_units_csv, export_units_npz, save_state_to_nix)
so ViewerState and SpikeSpy can delegate cleanly.
"""

from __future__ import annotations

import csv
import logging
from csv import DictReader, writer
from datetime import datetime
from os import environ
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import neo
import numpy as np
import quantities as pq
from neo import Event
from neo.core.objectlist import ObjectList
from neo.io import NixIO
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from .APTrack_experiment_import import (
    APTrackRecording,
    TypeID,
    process_folder as open_aptrack,
)
from .processing import MultiAnalogSignal, create_erp_signals

logger = logging.getLogger(__name__)


def expand_multichannel_segment(seg: neo.Segment) -> neo.Segment:
    """Inspects seg.analogsignals. If any AnalogSignal asig has asig.ndim == 2 and asig.shape[1] > 1,
    extracts channel names or generates CH_i, and splits each column into its own single-channel AnalogSignal.
    """
    if seg is None or not hasattr(seg, "analogsignals"):
        return seg

    new_analogsignals = []
    for asig in seg.analogsignals:
        if asig.ndim == 2 and asig.shape[1] > 1:
            ch_names = []
            if hasattr(asig, "array_annotations") and "channel_names" in asig.array_annotations:
                raw_names = asig.array_annotations["channel_names"]
                ch_names = [str(n) for n in raw_names]
            if len(ch_names) != asig.shape[1]:
                prefix = asig.name or "CH"
                ch_names = [f"{prefix}_{i}" for i in range(asig.shape[1])]

            for i in range(asig.shape[1]):
                col_data = asig[:, i]
                ch_sig = neo.AnalogSignal(
                    col_data,
                    units=asig.units,
                    sampling_rate=asig.sampling_rate,
                    t_start=asig.t_start,
                    name=ch_names[i],
                )
                if hasattr(asig, "annotations"):
                    ch_sig.annotations.update(asig.annotations)
                new_analogsignals.append(ch_sig)
        else:
            new_analogsignals.append(asig)
    seg.analogsignals = new_analogsignals
    return seg


def open_matlab_to_neo(folder: str | Path) -> neo.Segment:
    from scipy.io import loadmat

    matfiles = Path(folder).glob("*.mat")
    seg = neo.Segment()
    for m in matfiles:
        mf = loadmat(m)
        asig = neo.AnalogSignal(
            mf["data"].T,
            pq.V,
            sampling_rate=mf["samplerate"][0, 0] * pq.Hz,
            name=m.stem,
        )
        seg.analogsignals.append(asig)
    return seg


class APTrackDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Load APTrack Data")

        self.bottomLayout = QFormLayout()
        self.topLayout = QFormLayout()
        self.mainLayout = QVBoxLayout()

        self.topLayout.addRow(QLabel("custom parameters"))
        self.sbRecordNumber = QSpinBox()
        self.sbRecordNumber.setMinimum(0)

        self.inputsConfig = {}
        self.labelBoxes = []
        self.inputBoxes = []

        self.topLayout.addRow(QLabel("Rec#"), self.sbRecordNumber)

        labelBox = QLabel("Rec. channel")
        inputBox = QLineEdit("CH1")
        self.topLayout.addRow(labelBox, inputBox)
        self.labelBoxes.append(labelBox)
        self.inputBoxes.append(inputBox)

        labelBox = QLabel("Stimulus ch.")
        inputBox = QLineEdit("ADC2")
        self.topLayout.addRow(labelBox, inputBox)
        self.labelBoxes.append(labelBox)
        self.inputBoxes.append(inputBox)

        config = {
            "button": "ADC8",
        }
        for k, v in config.items():
            labelBox = QLineEdit(k)
            inputBox = QLineEdit(v)
            self.labelBoxes.append(labelBox)
            self.inputBoxes.append(inputBox)
            self.topLayout.addRow(labelBox, inputBox)

        self.addButton = QPushButton("+")
        self.addButton.clicked.connect(self.addWidget)
        self.removeButton = QPushButton("-")
        self.removeButton.clicked.connect(self.removeWidget)

        self.bottomLayout.addRow(self.addButton, self.removeButton)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.bottomLayout.addRow(self.buttons)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        self.mainLayout.addLayout(self.topLayout)
        self.mainLayout.addLayout(self.bottomLayout)
        self.setLayout(self.mainLayout)

    def get_config(self):
        config = {}
        config["record"] = self.sbRecordNumber.value()
        config["rd.0"] = self.inputBoxes[0].text()
        config["stimVolt"] = self.inputBoxes[1].text()

        for i in range(len(self.inputBoxes) - 2):
            config[self.labelBoxes[i + 2].text()] = self.inputBoxes[i + 2].text()

        return config

    def addWidget(self):
        labelBox = QLineEdit("new_input")
        inputBox = QLineEdit("ADC8")
        self.topLayout.addRow(labelBox, inputBox)
        self.labelBoxes.append(labelBox)
        self.inputBoxes.append(inputBox)

    def removeWidget(self):
        if self.topLayout.rowCount() < 5:
            return
        self.topLayout.removeRow(self.topLayout.rowCount() - 1)
        self.labelBoxes.pop()
        self.inputBoxes.pop()


def prompt_for_neo_file(type=None):
    if type is None:
        type = QInputDialog().getItem(
            None,
            "Select file type",
            "Filetype",
            ["h5", "openEphys", "APTrack", "matlab", "spike2", "nwb"],
        )[0]

    if type in ("h5", "spike2", "nwb"):
        fname = QFileDialog.getOpenFileName(None, "Open")[0]
    elif type in ("openEphys", "APTrack", "matlab"):
        fname = QFileDialog.getExistingDirectory(None, "Open OpenEphys")
    else:
        raise Exception(f"Unknown filetype: {type}")
    return fname, type


def load_file(data_path, type="h5", **kwargs):
    from .ViewerState import tracked_neuron_unit

    if type == "h5":
        data = NixIO(data_path, mode="ro").read_block(0).segments
    elif type == "openEphys":
        data = neo.OpenEphysBinaryIO(data_path).read_block(0).segments[0]
    elif type == "nwb":
        data = neo.NWBIO(data_path).read_block(block_index=0).segments[0]
    elif type == "matlab":
        data = open_matlab_to_neo(data_path)
    elif type == "APTrack":
        d = APTrackDialog()
        val = d.exec_() if hasattr(d, "exec_") else d.exec()
        config = d.get_config()
        t = config.pop("record")

        skip_list = ["rd.0", "rd.0.bp", "stimVolt", "stim"]
        additional_channels = []
        for k, v in config.items():
            if any(k in x for x in skip_list):
                continue
            extra_channel = APTrackRecording(v, TypeID.ANALOG, k, "")
            additional_channels.append(extra_channel)

        data = open_aptrack(data_path, t, config, additional_channels)
    elif type == "spike2":
        data = neo.Spike2IO(data_path)
    else:
        raise ValueError(f"Unknown file type: {type}")

    if isinstance(data, (list, ObjectList)):
        final_sig = next(
            x
            for x in data
            if x.name != "ERP"
            and x.annotations.get("class_", "") != "multianalogsignal"
        )
        for seg in data:
            if (
                seg.name == "ERP"
                or seg.annotations.get("class_", "") == "multianalogsignal"
            ):
                erp = MultiAnalogSignal(seg.analogsignals)
                final_sig.analogsignals.append(erp)
        data = final_sig

    if data is not None:
        data = expand_multichannel_segment(data)

    if len(data.events) == 0:
        event_signal = Event()
    else:
        event_signal = data.events[0]
    signal_chan = data.analogsignals[0] if len(data.analogsignals) > 0 else None

    spike_event_groups = []
    for x in data.events:
        x: Event = x
        if (not x.name.startswith("unit")) and (
            x.annotations.get("type", "") != "unit"
        ):
            continue

        spike_event_groups.append(
            tracked_neuron_unit(
                event=x,
                channel=x.annotations.get("channel", ""),
                neighbor_channels=list(x.annotations.get("neighbor_channels", [])),
                fibre_type=str(x.annotations.get("fibre_type", "")),
                notes=str(x.annotations.get("notes", "")),
            )
        )

    data.file_origin = data_path
    return data, signal_chan, event_signal, spike_event_groups


def export_units_csv(save_filename: str, unit_evts, stim_evt: neo.Event):
    """Export unit spikes to CSV including Channel, Neighbor_channels, and Tracking metadata."""
    with open(save_filename, "w", newline="") as f:
        w = writer(f)
        w.writerow([
            "SpikeID",
            "Stimulus_number",
            "Latency_ms",
            "Timestamp_ms",
            "Channel",
            "Neighbor_channels",
            "auto_track",
            "tracking_window_ms",
            "tracking_threshold_mv",
            "tracking_max_skips",
            "tracking_channel",
        ])
        for i, item in enumerate(unit_evts):
            if hasattr(item, "event"):
                evt = item.event
                channel = getattr(item, "channel", "") or evt.annotations.get("channel", "")
                neighbors = getattr(item, "neighbor_channels", []) or evt.annotations.get("neighbor_channels", [])
                auto_track = getattr(item, "auto_track", False) or evt.annotations.get("auto_track", False)
                win_ms = getattr(item, "tracking_window_ms", 5.0) if hasattr(item, "tracking_window_ms") else evt.annotations.get("tracking_window_ms", 5.0)
                thresh_mv = getattr(item, "tracking_threshold_mv", 0.01) if hasattr(item, "tracking_threshold_mv") else evt.annotations.get("tracking_threshold_mv", 0.01)
                max_skips = getattr(item, "tracking_max_skips", 1) if hasattr(item, "tracking_max_skips") else evt.annotations.get("tracking_max_skips", 1)
                trk_channel = getattr(item, "tracking_channel", "") or evt.annotations.get("tracking_channel", "")
            else:
                evt = item
                channel = evt.annotations.get("channel", "")
                neighbors = evt.annotations.get("neighbor_channels", [])
                auto_track = evt.annotations.get("auto_track", False)
                win_ms = evt.annotations.get("tracking_window_ms", 5.0)
                thresh_mv = evt.annotations.get("tracking_threshold_mv", 0.01)
                max_skips = evt.annotations.get("tracking_max_skips", 1)
                trk_channel = evt.annotations.get("tracking_channel", "")

            nom = evt.name or f"unit_{i}"
            neigh_str = ";".join(neighbors) if isinstance(neighbors, (list, tuple)) else str(neighbors)

            if len(evt) == 0:
                continue

            for timestamp in evt:
                if stim_evt is not None and len(stim_evt) > 0:
                    stim_no = int(stim_evt.searchsorted(timestamp) - 1)
                    if 0 <= stim_no < len(stim_evt):
                        latency = (timestamp - stim_evt[stim_no]).rescale(pq.ms)
                        lat_val = float(latency.magnitude)
                    else:
                        stim_no = -1
                        lat_val = float("nan")
                else:
                    stim_no = -1
                    lat_val = float("nan")

                ts_ms = float(timestamp.rescale(pq.ms).magnitude)
                w.writerow([
                    nom,
                    stim_no,
                    lat_val,
                    ts_ms,
                    channel,
                    neigh_str,
                    auto_track,
                    win_ms,
                    thresh_mv,
                    max_skips,
                    trk_channel,
                ])


# Backward compatibility alias
export_csv = export_units_csv


def import_units_csv(open_filename: str):
    """Import units from CSV, parsing SpikeID, timestamp, channel, neighbor_channels, and tracking metadata."""
    from .ViewerState import tracked_neuron_unit

    with open(open_filename, "r") as f:
        r = DictReader(f)
        fieldnames_lower = {x.lower(): x for x in (r.fieldnames or [])}
        ts_key = None
        for k in r.fieldnames or []:
            if "timestamp" in k.lower():
                ts_key = k
                break
        if ts_key is None:
            logger.warning("'timestamp' column not found in %s", open_filename)
            return []

        ch_key = fieldnames_lower.get("channel", None)
        neigh_key = fieldnames_lower.get("neighbor_channels", fieldnames_lower.get("neighbors", None))
        id_key = fieldnames_lower.get("spikeid", "SpikeID")
        auto_track_key = fieldnames_lower.get("auto_track", None)
        win_ms_key = fieldnames_lower.get("tracking_window_ms", None)
        thresh_mv_key = fieldnames_lower.get("tracking_threshold_mv", None)
        max_skips_key = fieldnames_lower.get("tracking_max_skips", None)
        trk_ch_key = fieldnames_lower.get("tracking_channel", None)

        units_dict: Dict[str, Dict[str, Any]] = {}
        for row in r:
            spikeid = str(row.get(id_key, "0"))
            if spikeid not in units_dict:
                auto_track_val = False
                if auto_track_key and row.get(auto_track_key) is not None:
                    at_raw = row[auto_track_key]
                    auto_track_val = at_raw.strip().lower() in ("true", "1", "t", "yes") if isinstance(at_raw, str) else bool(at_raw)

                win_ms_val = 5.0
                if win_ms_key and row.get(win_ms_key):
                    try:
                        win_ms_val = float(row[win_ms_key])
                    except (ValueError, TypeError):
                        pass

                thresh_mv_val = 0.01
                if thresh_mv_key and row.get(thresh_mv_key):
                    try:
                        thresh_mv_val = float(row[thresh_mv_key])
                    except (ValueError, TypeError):
                        pass

                max_skips_val = 1
                if max_skips_key and row.get(max_skips_key):
                    try:
                        max_skips_val = int(float(row[max_skips_key]))
                    except (ValueError, TypeError):
                        pass

                trk_ch_val = ""
                if trk_ch_key and row.get(trk_ch_key):
                    trk_ch_val = str(row[trk_ch_key])

                units_dict[spikeid] = {
                    "timestamps": [],
                    "channel": row.get(ch_key, "") if ch_key else "",
                    "neighbor_channels": [],
                    "auto_track": auto_track_val,
                    "tracking_window_ms": win_ms_val,
                    "tracking_threshold_mv": thresh_mv_val,
                    "tracking_max_skips": max_skips_val,
                    "tracking_channel": trk_ch_val,
                }
                if neigh_key and row.get(neigh_key):
                    raw_n = row[neigh_key]
                    sep = ";" if ";" in raw_n else ","
                    units_dict[spikeid]["neighbor_channels"] = [x.strip() for x in raw_n.split(sep) if x.strip()]

            units_dict[spikeid]["timestamps"].append(float(row[ts_key]))

        out = []
        for k, v in units_dict.items():
            name = f"unit_{k}" if k.isdigit() else k
            ts_arr = np.array(v["timestamps"], dtype=np.float64)
            evt = neo.Event(ts_arr * pq.ms, units=pq.ms, name=name)
            evt.annotations["type"] = "unit"
            evt.annotations["channel"] = v["channel"]
            evt.annotations["neighbor_channels"] = v["neighbor_channels"]
            evt.annotations["auto_track"] = v["auto_track"]
            evt.annotations["tracking_window_ms"] = v["tracking_window_ms"]
            evt.annotations["tracking_threshold_mv"] = v["tracking_threshold_mv"]
            evt.annotations["tracking_max_skips"] = v["tracking_max_skips"]
            evt.annotations["tracking_channel"] = v["tracking_channel"]

            sg = tracked_neuron_unit(
                event=evt.rescale(pq.s),
                channel=v["channel"],
                neighbor_channels=v["neighbor_channels"],
                auto_track=v["auto_track"],
                tracking_window_ms=v["tracking_window_ms"],
                tracking_threshold_mv=v["tracking_threshold_mv"],
                tracking_max_skips=v["tracking_max_skips"],
                tracking_channel=v["tracking_channel"],
            )
            out.append(sg)

        return out


def export_units_npz(save_filename: str, state, window_pre: pq.Quantity = -0.5 * pq.ms, window_post: pq.Quantity = 1.0 * pq.ms):
    """Export units and signals to a condensed .npz file package."""
    timestamps = state.event_signal
    units_erps = {}
    for i, sg in enumerate(state.spike_groups):
        nom = sg.event.name or f"unit_{i}"
        if state.analog_signal is not None and len(sg.event) > 0:
            erp = create_erp_signals(state.analog_signal, sg.event, window_pre, window_post)
            units_erps[nom] = np.asarray(erp.magnitude)

    signals_dict = {}
    if state.segment is not None:
        for asig in state.segment.analogsignals:
            if asig.name and timestamps is not None and len(timestamps) > 0:
                erp = create_erp_signals(asig, timestamps, window_pre, window_post)
                signals_dict[f"sig_{asig.name}"] = np.asarray(erp.magnitude)

    np.savez(
        save_filename,
        sampling_rate=float(state.sampling_rate or 30000.0),
        window_size_s=float(state.window_size.rescale(pq.s).magnitude if state.window_size else 0.5),
        **units_erps,
        **signals_dict,
    )


def save_state_to_nix(filename: str, spike_groups, data: neo.Segment = None, metadata=None):
    """Saves a file containing the spike groups and multi-channel signals to NIX/NWB format.
    Ensures MultiAnalogSignal is converted appropriately and saves channel/neighbor_channels in annotations.
    """
    if metadata is None:
        metadata = {
            "current_user": environ.get("USERNAME", environ.get("USER", "unknown")),
            "date": datetime.now(),
        }

    blk = neo.Block(name="main")

    if data is not None:
        from copy import deepcopy

        data2 = deepcopy(data)

        # Remove nix_name annotations that prevent saving
        for x in [*data2.analogsignals, *data2.events, data2]:
            if "nix_name" in x.annotations:
                del x.annotations["nix_name"]

        # Extract MultiAnalogSignals into separate ERP segments
        remaining_sigs = []
        for x in data2.analogsignals:
            if isinstance(x, MultiAnalogSignal):
                new_seg = neo.Segment("ERP")
                new_seg.analogsignals = x.signals
                blk.segments.append(new_seg)
            else:
                remaining_sigs.append(x)
        data2.analogsignals = remaining_sigs

        # Remove previous unit events to avoid duplicates
        data2.events = [
            x
            for x in data2.events
            if not x.name.startswith("unit") and x.annotations.get("type") != "unit"
        ]
    else:
        data2 = neo.Segment()

    for i, sg in enumerate(spike_groups):
        evt = sg.event if hasattr(sg, "event") else sg
        evt.annotate(**metadata)
        evt.name = evt.name or f"unit_{i}"
        if hasattr(sg, "channel") and sg.channel:
            evt.annotations["channel"] = sg.channel
        if hasattr(sg, "neighbor_channels") and sg.neighbor_channels:
            evt.annotations["neighbor_channels"] = list(sg.neighbor_channels)
        if hasattr(sg, "fibre_type") and sg.fibre_type:
            evt.annotations["fibre_type"] = sg.fibre_type
        if hasattr(sg, "notes") and sg.notes:
            evt.annotations["notes"] = sg.notes
        if hasattr(sg, "auto_track"):
            evt.annotations["auto_track"] = sg.auto_track
        if hasattr(sg, "tracking_window_ms"):
            evt.annotations["tracking_window_ms"] = sg.tracking_window_ms
        if hasattr(sg, "tracking_threshold_mv"):
            evt.annotations["tracking_threshold_mv"] = sg.tracking_threshold_mv
        if hasattr(sg, "tracking_max_skips"):
            evt.annotations["tracking_max_skips"] = sg.tracking_max_skips
        if hasattr(sg, "tracking_channel"):
            evt.annotations["tracking_channel"] = sg.tracking_channel
        data2.events.append(evt)

    blk.segments.append(data2)
    data2.block = blk

    if Path(filename).exists():
        Path(filename).unlink()

    blk.annotations["session_start_time"] = data2.annotations.get(
        "session_start_time", datetime.now()
    )

    ext = filename.split(".")[-1].lower()
    output_formats = {"h5": NixIO, "nwb": neo.NWBIO}
    export_class = output_formats.get(ext, NixIO)
    n = export_class(filename, mode="rw")
    n.write_block(blk)
    n.close()


# Backward compatibility alias
save_file = save_state_to_nix
