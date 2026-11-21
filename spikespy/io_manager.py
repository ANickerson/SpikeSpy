"""I/O Manager for SpikeSpy: File loading, export, and persistence.

Extracts file loading (load_file, prompt_for_neo_file, open_matlab_to_neo,
APTrackDialog, expand_multichannel_segment) and persistence helpers
(export_units_csv, import_units_csv, export_units_npz, save_state_to_nix)
so ViewerState and SpikeSpy can delegate cleanly.
"""

from __future__ import annotations

import csv
import json
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
from PySide6.QtCore import QSettings, Qt, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from .APTrack_experiment_import import (
    APTrackRecording,
    ChannelRoleSpec,
    TypeID,
    get_default_channel_roles,
    normalize_channel_roles,
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
        if (
            not isinstance(asig, MultiAnalogSignal)
            and getattr(asig, "ndim", 1) == 2
            and asig.shape[1] > 1
        ):
            ch_names = []
            if (
                hasattr(asig, "array_annotations")
                and "channel_names" in asig.array_annotations
            ):
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


_PURPOSE_LABELS = [
    ("Analog", (TypeID.ANALOG, False)),
    ("Analog + Bandpass", (TypeID.ANALOG, True)),
    ("TTL Trigger / Event", (TypeID.TTL, False)),
]

_KNOWN_PURPOSES = [
    "probe",
    "stim_trigger",
    "stim_voltage",
    "stim_analog",
    "potentiometer",
    "thermode",
    "button",
    "aux",
]


def _detect_channels_from_folder(folder: Optional[Union[str, Path]]) -> List[str]:
    if not folder:
        return []
    p = Path(folder)
    if not p.exists() or not p.is_dir():
        return []
    detected: List[str] = []
    seen = set()
    for f in sorted(p.glob("*.continuous")):
        stem = f.stem
        if "_" in stem:
            ch = stem.split("_", 1)[1]
            # Strip optional trailing recording suffix _2, _3
            parts = ch.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit() and not parts[0].isdigit():
                ch = parts[0]
        else:
            ch = stem
        if ch and ch not in seen:
            seen.add(ch)
            detected.append(ch)
    return detected


class APTrackDialog(QDialog):
    """Channel purpose & preprocessing configuration dialog (backward compatible with legacy API)."""

    _SETTINGS_KEY = "APTrackDialog/last_config"

    def __init__(
        self,
        parent=None,
        available_channels: Optional[List[str]] = None,
        folder: Optional[Union[str, Path]] = None,
        segment: Optional[neo.Segment] = None,
        initial_roles: Optional[List[Union[ChannelRoleSpec, Dict[str, Any]]]] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Load APTrack / Channel Configuration")
        self.resize(860, 480)

        self._available_channels: List[str] = list(available_channels or [])
        if folder:
            for ch in _detect_channels_from_folder(folder):
                if ch not in self._available_channels:
                    self._available_channels.append(ch)
        if segment is not None and hasattr(segment, "analogsignals"):
            for sig in segment.analogsignals:
                sig_name = getattr(sig, "name", "") or ""
                if sig.annotations.get("derived_from_role") or sig_name.endswith(".bp"):
                    continue
                raw_name = sig.annotations.get("raw_channel_name", sig_name)
                if raw_name and raw_name not in self._available_channels:
                    self._available_channels.append(raw_name)

        self.bottomLayout = QFormLayout()
        self.topLayout = QFormLayout()
        self.mainLayout = QVBoxLayout()

        self.topLayout.addRow(QLabel("Channel & Preprocessing Parameters"))
        self.sbRecordNumber = QSpinBox()
        self.sbRecordNumber.setMinimum(0)

        self.inputsConfig = {}
        self.labelBoxes = []
        self.inputBoxes = []

        self.topLayout.addRow(QLabel("Rec#"), self.sbRecordNumber)

        # Legacy quick-edit rows preserved for backward compatibility
        labelBox = QLabel("Rec. channel")
        inputBox = QLineEdit("CH1")
        self.topLayout.addRow(labelBox, inputBox)
        self.labelBoxes.append(labelBox)
        self.inputBoxes.append(inputBox)

        labelBox = QLabel("Stimulus ch.")
        inputBox = QLineEdit("ADC4")
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

        # Structured role configuration table
        self.roleTable = QTableWidget(0, 9)
        self.roleTable.setHorizontalHeaderLabels(
            [
                "Enabled",
                "Source",
                "Target Name",
                "Type / Processing",
                "Purpose",
                "BP Low (Hz)",
                "BP High (Hz)",
                "TTL Thresh",
                "TTL Refrac (ms)",
            ]
        )
        self.roleTable.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.roleTable.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)

        # Role row controls & Presets bar
        presetBar = QHBoxLayout()
        self.addRoleButton = QPushButton("Add Role")
        self.addRoleButton.clicked.connect(self._on_add_role_clicked)
        self.removeRoleButton = QPushButton("Remove Role")
        self.removeRoleButton.clicked.connect(self._on_remove_role_clicked)
        self.resetDefaultsButton = QPushButton("Reset Defaults")
        self.resetDefaultsButton.clicked.connect(self._on_reset_defaults_clicked)
        self.loadPresetButton = QPushButton("Load Preset...")
        self.loadPresetButton.clicked.connect(self._on_load_preset_clicked)
        self.savePresetButton = QPushButton("Save Preset...")
        self.savePresetButton.clicked.connect(self._on_save_preset_clicked)

        presetBar.addWidget(self.addRoleButton)
        presetBar.addWidget(self.removeRoleButton)
        presetBar.addWidget(self.resetDefaultsButton)
        presetBar.addStretch()
        presetBar.addWidget(self.loadPresetButton)
        presetBar.addWidget(self.savePresetButton)

        self.addButton = QPushButton("+")
        self.addButton.clicked.connect(self.addWidget)
        self.removeButton = QPushButton("-")
        self.removeButton.clicked.connect(self.removeWidget)

        self.bottomLayout.addRow(self.addButton, self.removeButton)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.bottomLayout.addRow(self.buttons)
        self.buttons.accepted.connect(self._on_accept)
        self.buttons.rejected.connect(self.reject)

        self.mainLayout.addLayout(self.topLayout)
        self.mainLayout.addWidget(self.roleTable)
        self.mainLayout.addLayout(presetBar)
        self.mainLayout.addLayout(self.bottomLayout)
        self.setLayout(self.mainLayout)

        self._syncing_legacy = False
        self.inputBoxes[0].textChanged.connect(self._sync_legacy_to_table)
        self.inputBoxes[1].textChanged.connect(self._sync_legacy_to_table)
        if len(self.inputBoxes) > 2:
            self.inputBoxes[2].textChanged.connect(self._sync_legacy_to_table)

        # Initialize roles from explicit argument, QSettings, or defaults
        if initial_roles is not None:
            self.set_roles(normalize_channel_roles(initial_roles))
        else:
            loaded_from_settings = self._load_from_qsettings()
            if not loaded_from_settings:
                self.set_roles(get_default_channel_roles())

    def _load_from_qsettings(self) -> bool:
        try:
            settings = QSettings("SpikeSpy", "SpikeSpy")
            raw = settings.value(self._SETTINGS_KEY, None)
            if not raw:
                return False
            data = json.loads(raw) if isinstance(raw, str) else raw
            if isinstance(data, dict):
                if "record" in data:
                    self.sbRecordNumber.setValue(int(data["record"]))
                if "roles" in data and isinstance(data["roles"], list) and data["roles"]:
                    self.set_roles(normalize_channel_roles(data["roles"]))
                    return True
        except Exception:
            logger.debug("Failed to load APTrackDialog settings", exc_info=True)
        return False

    def _save_to_qsettings(self):
        try:
            settings = QSettings("SpikeSpy", "SpikeSpy")
            cfg = self.get_config()
            settings.setValue(self._SETTINGS_KEY, json.dumps(cfg))
        except Exception:
            logger.debug("Failed to save APTrackDialog settings", exc_info=True)

    def _on_accept(self):
        self._save_to_qsettings()
        self.accept()

    def _sync_legacy_to_table(self):
        if self._syncing_legacy:
            return
        self._syncing_legacy = True
        try:
            rd0_src = self.inputBoxes[0].text().strip()
            stim_src = self.inputBoxes[1].text().strip()
            for row in range(self.roleTable.rowCount()):
                target_item = self.roleTable.item(row, 2)
                target_name = target_item.text().strip() if target_item else ""
                src_widget = self.roleTable.cellWidget(row, 1)
                if target_name == "rd.0" and rd0_src and isinstance(src_widget, QComboBox):
                    src_widget.setCurrentText(rd0_src)
                elif target_name in ("env.stimVolt", "env.stim") and stim_src:
                    type_widget = self.roleTable.cellWidget(row, 3)
                    if isinstance(type_widget, QComboBox) and "TTL" in type_widget.currentText():
                        if isinstance(src_widget, QComboBox):
                            src_widget.setCurrentText(stim_src)
                for idx in range(2, len(self.inputBoxes)):
                    lbl_w = self.labelBoxes[idx]
                    lbl = lbl_w.text().strip() if hasattr(lbl_w, "text") else ""
                    val = self.inputBoxes[idx].text().strip()
                    if lbl == "button" and target_name == "rec.button" and val:
                        if isinstance(src_widget, QComboBox):
                            src_widget.setCurrentText(val)
        finally:
            self._syncing_legacy = False

    def _sync_table_to_legacy(self):
        if self._syncing_legacy:
            return
        self._syncing_legacy = True
        try:
            roles = self.get_roles()
            for r in roles:
                if r.name == "rd.0" and r.source:
                    self.inputBoxes[0].setText(r.source)
                elif r.name in ("env.stimVolt", "env.stim") and r.probe_type == TypeID.TTL and r.source:
                    self.inputBoxes[1].setText(r.source)
                elif r.name == "rec.button" and len(self.inputBoxes) > 2 and r.source:
                    lbl_w = self.labelBoxes[2]
                    if hasattr(lbl_w, "text") and lbl_w.text().strip() == "button":
                        self.inputBoxes[2].setText(r.source)
        finally:
            self._syncing_legacy = False

    def _append_role_row(self, role: ChannelRoleSpec):
        row = self.roleTable.rowCount()
        self.roleTable.insertRow(row)

        chk = QCheckBox()
        chk.setChecked(role.enabled)
        self.roleTable.setCellWidget(row, 0, chk)

        src_combo = QComboBox()
        src_combo.setEditable(True)
        choices = list(self._available_channels)
        for default_ch in ("CH1", "CH0", "ADC2", "ADC4", "ADC5", "ADC7", "ADC8"):
            if default_ch not in choices:
                choices.append(default_ch)
        if role.source and role.source not in choices:
            choices.insert(0, role.source)
        src_combo.addItems(choices)
        src_combo.setCurrentText(role.source)
        self.roleTable.setCellWidget(row, 1, src_combo)

        name_item = QTableWidgetItem(role.name)
        self.roleTable.setItem(row, 2, name_item)

        type_combo = QComboBox()
        for label, _ in _PURPOSE_LABELS:
            type_combo.addItem(label)
        if role.probe_type == TypeID.TTL:
            type_combo.setCurrentIndex(2)
        elif role.bandpass:
            type_combo.setCurrentIndex(1)
        else:
            type_combo.setCurrentIndex(0)
        self.roleTable.setCellWidget(row, 3, type_combo)

        purpose_combo = QComboBox()
        purpose_combo.setEditable(True)
        purpose_choices = list(_KNOWN_PURPOSES)
        if role.purpose and role.purpose not in purpose_choices:
            purpose_choices.insert(0, role.purpose)
        purpose_combo.addItems(purpose_choices)
        purpose_combo.setCurrentText(role.purpose)
        self.roleTable.setCellWidget(row, 4, purpose_combo)

        bp_low = QDoubleSpinBox()
        bp_low.setRange(0.1, 25000.0)
        bp_low.setDecimals(1)
        bp_low.setValue(role.bandpass_low)
        self.roleTable.setCellWidget(row, 5, bp_low)

        bp_high = QDoubleSpinBox()
        bp_high.setRange(1.0, 50000.0)
        bp_high.setDecimals(1)
        bp_high.setValue(role.bandpass_high)
        self.roleTable.setCellWidget(row, 6, bp_high)

        ttl_thresh = QDoubleSpinBox()
        ttl_thresh.setRange(0.0, 1e6)
        ttl_thresh.setDecimals(4)
        ttl_thresh.setSpecialValueText("Auto")
        ttl_thresh.setValue(float(role.ttl_threshold) if role.ttl_threshold and role.ttl_threshold > 0 else 0.0)
        self.roleTable.setCellWidget(row, 7, ttl_thresh)

        ttl_refrac = QDoubleSpinBox()
        ttl_refrac.setRange(0.0, 60000.0)
        ttl_refrac.setDecimals(1)
        ttl_refrac.setValue(role.ttl_refractory_ms)
        self.roleTable.setCellWidget(row, 8, ttl_refrac)

    def set_roles(self, roles: List[Union[ChannelRoleSpec, Dict[str, Any]]]):
        norm = normalize_channel_roles(roles)
        self.roleTable.setRowCount(0)
        for r in norm:
            self._append_role_row(r)
        self._sync_table_to_legacy()

    def get_roles(self) -> List[ChannelRoleSpec]:
        roles: List[ChannelRoleSpec] = []
        for row in range(self.roleTable.rowCount()):
            chk = self.roleTable.cellWidget(row, 0)
            src_widget = self.roleTable.cellWidget(row, 1)
            name_item = self.roleTable.item(row, 2)
            type_widget = self.roleTable.cellWidget(row, 3)
            purpose_widget = self.roleTable.cellWidget(row, 4)
            bp_low_w = self.roleTable.cellWidget(row, 5)
            bp_high_w = self.roleTable.cellWidget(row, 6)
            ttl_thresh_w = self.roleTable.cellWidget(row, 7)
            ttl_refrac_w = self.roleTable.cellWidget(row, 8)

            source = src_widget.currentText().strip() if isinstance(src_widget, QComboBox) else ""
            name = name_item.text().strip() if name_item else source
            type_idx = type_widget.currentIndex() if isinstance(type_widget, QComboBox) else 0
            probe_type, bandpass = _PURPOSE_LABELS[max(0, min(type_idx, len(_PURPOSE_LABELS) - 1))][1]
            purpose = purpose_widget.currentText().strip() if isinstance(purpose_widget, QComboBox) else ""
            bp_low = float(bp_low_w.value()) if isinstance(bp_low_w, QDoubleSpinBox) else 500.0
            bp_high = float(bp_high_w.value()) if isinstance(bp_high_w, QDoubleSpinBox) else 7000.0
            thresh_val = float(ttl_thresh_w.value()) if isinstance(ttl_thresh_w, QDoubleSpinBox) else 0.0
            ttl_threshold = thresh_val if thresh_val > 0 else None
            ttl_refrac = float(ttl_refrac_w.value()) if isinstance(ttl_refrac_w, QDoubleSpinBox) else 100.0
            enabled = bool(chk.isChecked()) if isinstance(chk, QCheckBox) else True

            if not source and not name:
                continue

            roles.append(
                ChannelRoleSpec(
                    source=source,
                    name=name or source,
                    probe_type=probe_type,
                    purpose=purpose,
                    bandpass=bandpass,
                    bandpass_low=bp_low,
                    bandpass_high=bp_high,
                    ttl_threshold=ttl_threshold,
                    ttl_refractory_ms=ttl_refrac,
                    is_adc=source.upper().startswith("ADC"),
                    enabled=enabled,
                )
            )
        return roles

    def _on_add_role_clicked(self):
        self._append_role_row(
            ChannelRoleSpec(
                source="ADC1",
                name="aux.ch",
                probe_type=TypeID.ANALOG,
                purpose="aux",
                is_adc=True,
            )
        )

    def _on_remove_role_clicked(self):
        row = self.roleTable.currentRow()
        if row < 0:
            row = self.roleTable.rowCount() - 1
        if row >= 0:
            self.roleTable.removeRow(row)

    def _on_reset_defaults_clicked(self):
        self.set_roles(get_default_channel_roles())

    def _on_save_preset_clicked(self):
        fname, _ = QFileDialog.getSaveFileName(
            self, "Save Channel Preset", "", "JSON (*.json);;All files (*)"
        )
        if not fname:
            return
        payload = {
            "record": self.sbRecordNumber.value(),
            "roles": [r.to_dict() for r in self.get_roles()],
        }
        Path(fname).write_text(json.dumps(payload, indent=2))

    def _on_load_preset_clicked(self):
        fname, _ = QFileDialog.getOpenFileName(
            self, "Load Channel Preset", "", "JSON (*.json);;All files (*)"
        )
        if not fname:
            return
        data = json.loads(Path(fname).read_text())
        if isinstance(data, list):
            self.set_roles(normalize_channel_roles(data))
        elif isinstance(data, dict):
            if "record" in data:
                self.sbRecordNumber.setValue(int(data["record"]))
            self.set_roles(normalize_channel_roles(data))

    def get_config(self):
        config: Dict[str, Any] = {}
        config["record"] = self.sbRecordNumber.value()
        config["rd.0"] = self.inputBoxes[0].text()
        config["stimVolt"] = self.inputBoxes[1].text()

        for i in range(len(self.inputBoxes) - 2):
            config[self.labelBoxes[i + 2].text()] = self.inputBoxes[i + 2].text()

        roles = self.get_roles()
        # Reflect table edits for core roles while preserving any custom extra legacy rows
        for r in roles:
            if not r.enabled:
                continue
            if r.name == "rd.0":
                config["rd.0"] = r.source
            elif r.name == "env.stimVolt":
                config["stimVolt"] = r.source
            elif r.name == "env.potentiometer":
                config["stimVolt_pot"] = r.source
            elif r.name == "env.stim" and r.probe_type == TypeID.ANALOG:
                config["stim"] = r.source
            elif r.name == "env.thermode":
                config["thermode"] = r.source
            elif r.name == "rec.button" and "button" in config:
                config["button"] = r.source

        # If caller added custom legacy rows via addWidget(), ensure they appear in roles too
        known_names = {r.name for r in roles}
        known_legacy = {"record", "rd.0", "stimVolt", "stimVolt_pot", "stim", "thermode", "button", "roles"}
        for k, v in config.items():
            if k in known_legacy or k in known_names or not isinstance(v, str) or not v.strip():
                continue
            roles.append(
                ChannelRoleSpec(
                    source=v.strip(),
                    name=k,
                    probe_type=TypeID.ANALOG,
                    purpose="aux",
                    bandpass=k.lower().startswith("rd"),
                    is_adc=v.strip().upper().startswith("ADC"),
                )
            )

        config["roles"] = [r.to_dict() for r in roles]
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


FILE_TYPE_EXTENSIONS = {
    ".h5": "h5",
    ".hdf5": "h5",
    ".nwb": "nwb",
    ".smr": "spike2",
    ".smrx": "spike2",
}

OPEN_FILE_FILTERS = ";;".join(
    [
        "All supported recordings (*.h5 *.hdf5 *.nwb *.smr *.smrx)",
        "NIX h5 (*.h5 *.hdf5)",
        "NWB (*.nwb)",
        "Spike2 (*.smr *.smrx)",
        "All files (*)",
    ]
)

FOLDER_TYPES = ["openEphys", "APTrack", "matlab"]
ALL_TYPES = ["h5", "nwb", "spike2"] + FOLDER_TYPES


def infer_file_type(path) -> Optional[str]:
    """Guess a loader type from a file extension, e.g. '.h5' -> 'h5'."""
    return FILE_TYPE_EXTENSIONS.get(Path(path).suffix.lower())


def infer_folder_type(folder) -> Optional[str]:
    """Guess a loader type from the contents of a recording folder.

    Returns 'openEphys' for Open Ephys binary folders, 'matlab' for folders of
    .mat files, 'APTrack' for legacy *.continuous recordings, or None when the
    contents are unrecognised.
    """
    folder = Path(folder)
    try:
        entries = [x.name.lower() for x in folder.iterdir()]
    except OSError:
        return None
    if "structure.oebin" in entries or "settings.xml" in entries or "continuous" in entries:
        return "openEphys"
    if any(n.endswith(".continuous") for n in entries):
        return "APTrack"
    if any(n.endswith(".mat") for n in entries):
        return "matlab"
    return None


def _prompt_file_type(default: str = "h5") -> str:
    return str(
        QInputDialog.getItem(
            None,
            "Select file type",
            "Filetype",
            ALL_TYPES,
            max(0, ALL_TYPES.index(default)) if default in ALL_TYPES else 0,
            False,
        )[0]
    )


def prompt_for_neo_file(type=None, parent=None):
    """Ask for a recording file with a single dialog; the loader type is inferred
    from the chosen extension when not given explicitly."""
    if type in ("openEphys", "APTrack", "matlab"):
        fname = QFileDialog.getExistingDirectory(parent, "Open recording folder")[0]
        return fname, (type if fname else None)
    if type in ("h5", "spike2", "nwb"):
        fname = QFileDialog.getOpenFileName(parent, "Open")[0]
        return fname, (type if fname else None)

    fname, _selected = QFileDialog.getOpenFileName(
        parent, "Open recording", "", OPEN_FILE_FILTERS
    )
    if not fname:
        return "", None
    type = infer_file_type(fname)
    if type is None:
        type = _prompt_file_type()
    return fname, type


def prompt_for_neo_folder(parent=None):
    """Ask for a recording folder and infer the loader from its contents."""
    folder = QFileDialog.getExistingDirectory(parent, "Open recording folder")
    if not folder:
        return "", None
    type = infer_folder_type(folder)
    if type is None:
        type = _prompt_file_type()
    return folder, type


def _select_primary_event_signal(
    seg: neo.Segment, roles: Optional[List[ChannelRoleSpec]] = None
) -> Event:
    if seg is None or not seg.events:
        return Event()

    non_unit_events = [
        evt
        for evt in seg.events
        if not (getattr(evt, "name", "") or "").startswith("unit")
        and evt.annotations.get("type") != "unit"
    ]

    # 1. Prefer event named "env.stim" if non-empty
    for evt in non_unit_events:
        if getattr(evt, "name", "") == "env.stim" and len(evt) > 0:
            return evt

    # 2. Prefer event matching a role with purpose == "stim_trigger"
    if roles:
        stim_names = [
            r.name
            for r in roles
            if r.enabled and r.purpose == "stim_trigger" and r.probe_type == TypeID.TTL
        ]
        for sname in stim_names:
            for evt in non_unit_events:
                if getattr(evt, "name", "") == sname and len(evt) > 0:
                    return evt

    # 3. Prefer non-empty non-unit TTL event
    for evt in non_unit_events:
        if evt.annotations.get("type_id") == TypeID.TTL.value and len(evt) > 0:
            return evt

    # 4. Fallback to first non-empty non-unit event, then first event
    for evt in non_unit_events:
        if len(evt) > 0:
            return evt

    return seg.events[0]


def _select_primary_analog_signal(
    seg: neo.Segment, roles: Optional[List[ChannelRoleSpec]] = None
):
    if seg is None or not seg.analogsignals:
        return None

    sigs = seg.analogsignals
    # 1. Prefer "rd.0.bp" then "rd.0"
    for preferred in ("rd.0.bp", "rd.0"):
        for sig in sigs:
            if getattr(sig, "name", "") == preferred:
                return sig

    # 2. Prefer role with purpose == "probe" (its .bp variant first if bandpass is enabled)
    if roles:
        for r in roles:
            if r.enabled and r.purpose == "probe":
                candidates = [f"{r.name}.bp", r.name] if r.bandpass else [r.name]
                for cname in candidates:
                    for sig in sigs:
                        if getattr(sig, "name", "") == cname:
                            return sig

    return sigs[0]


def load_file(data_path, type="h5", aptrack_config=None, progress_cb=None):
    from .ViewerState import tracked_neuron_unit

    def _stage(msg):
        if progress_cb is not None:
            progress_cb(msg)

    normalized_roles: Optional[List[ChannelRoleSpec]] = None

    if type == "h5":
        _stage("Reading NIX file...")
        data = NixIO(data_path, mode="ro").read_block(0).segments
    elif type == "openEphys":
        _stage("Reading Open Ephys data...")
        data = neo.OpenEphysBinaryIO(data_path).read_block(0).segments[0]
    elif type == "nwb":
        _stage("Reading NWB file...")
        data = neo.NWBIO(data_path).read_block(block_index=0).segments[0]
    elif type == "matlab":
        _stage("Reading MATLAB folder...")
        data = open_matlab_to_neo(data_path)
    elif type == "APTrack":
        if aptrack_config is None:
            d = APTrackDialog(folder=data_path)
            val = d.exec_() if hasattr(d, "exec_") else d.exec()
            config = d.get_config()
        else:
            config = dict(aptrack_config)
        t = config.pop("record", 1)

        has_explicit_roles = "roles" in config and config["roles"] is not None
        skip_list = ["rd.0", "rd.0.bp", "stimVolt", "stimVolt_pot", "stim", "thermode", "button", "roles"]
        additional_channels = []
        if not has_explicit_roles:
            for k, v in config.items():
                if any(k in x for x in skip_list) or not isinstance(v, str) or not v.strip():
                    continue
                extra_channel = APTrackRecording(v, TypeID.ANALOG, k, "")
                additional_channels.append(extra_channel)

        normalized_roles = normalize_channel_roles(config, extra_channels=additional_channels)

        _stage("Reading APTrack folder...")
        data = open_aptrack(data_path, t, config, additional_channels)
        if data is not None:
            saved_cfg = dict(config)
            saved_cfg["record"] = t
            data.annotations["aptrack_config"] = saved_cfg
            data.annotations["channel_roles"] = [r.to_dict() for r in normalized_roles]
    elif type == "spike2":
        _stage("Reading Spike2 file...")
        data = neo.Spike2IO(data_path)
    else:
        raise ValueError(f"Unknown file type: {type}")

    _stage("Preparing channels...")

    if isinstance(data, (list, ObjectList)):
        final_sig = next(
            (
                x
                for x in data
                if not getattr(x, "name", "").startswith("ERP")
                and x.annotations.get("class_", "") != "multianalogsignal"
            ),
            None,
        )
        if final_sig is None and len(data) > 0:
            final_sig = data[-1]
        for seg in data:
            if seg is not final_sig and (
                getattr(seg, "name", "").startswith("ERP")
                or seg.annotations.get("class_", "") == "multianalogsignal"
            ):
                erp = MultiAnalogSignal(seg.analogsignals)
                ch_name = seg.annotations.get(
                    "channel",
                    seg.name.replace("ERP_", "")
                    if seg.name.startswith("ERP_")
                    else (seg.analogsignals[0].name if seg.analogsignals else "CH0"),
                )
                erp.name = ch_name
                final_sig.analogsignals.append(erp)
        data = final_sig

    if data is not None:
        data = expand_multichannel_segment(data)

    if normalized_roles is None and data is not None and "channel_roles" in data.annotations:
        try:
            normalized_roles = normalize_channel_roles(data.annotations["channel_roles"])
        except Exception:
            normalized_roles = None

    event_signal = _select_primary_event_signal(data, normalized_roles)
    signal_chan = _select_primary_analog_signal(data, normalized_roles)

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


class FileLoadWorker(QThread):
    """Loads a recording off the GUI thread; results are marshalled via signals."""

    loaded = Signal(str, str, object, object, object, object)
    failed = Signal(str, str, str)
    stage = Signal(str)

    def __init__(self, fname, type, aptrack_config=None, parent=None):
        super().__init__(parent)
        self.fname = fname
        self.type = type
        self.aptrack_config = aptrack_config

    def run(self):
        try:
            self.stage.emit(f"Opening {Path(self.fname).name}...")
            data, analog_signal, event_signal, spike_groups = load_file(
                self.fname, self.type, aptrack_config=self.aptrack_config, progress_cb=self.stage.emit
            )
            self.loaded.emit(self.fname, self.type, data, analog_signal, event_signal, spike_groups)
        except Exception:
            import traceback

            logger.exception("Failed to load %s using %s", self.fname, self.type)
            self.failed.emit(self.fname, self.type, traceback.format_exc())


def export_units_csv(save_filename: str, unit_evts, stim_evt: neo.Event):
    """Export unit spikes to CSV including Channel, Neighbor_channels, and Tracking metadata."""
    with open(save_filename, "w", newline="") as f:
        w = writer(f)
        w.writerow(
            [
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
            ]
        )
        for i, item in enumerate(unit_evts):
            if hasattr(item, "event"):
                evt = item.event
                channel = getattr(item, "channel", "") or evt.annotations.get(
                    "channel", ""
                )
                neighbors = getattr(
                    item, "neighbor_channels", []
                ) or evt.annotations.get("neighbor_channels", [])
                auto_track = getattr(item, "auto_track", False) or evt.annotations.get(
                    "auto_track", False
                )
                win_ms = (
                    getattr(item, "tracking_window_ms", 5.0)
                    if hasattr(item, "tracking_window_ms")
                    else evt.annotations.get("tracking_window_ms", 5.0)
                )
                thresh_mv = (
                    getattr(item, "tracking_threshold_mv", 0.01)
                    if hasattr(item, "tracking_threshold_mv")
                    else evt.annotations.get("tracking_threshold_mv", 0.01)
                )
                max_skips = (
                    getattr(item, "tracking_max_skips", 1)
                    if hasattr(item, "tracking_max_skips")
                    else evt.annotations.get("tracking_max_skips", 1)
                )
                trk_channel = getattr(
                    item, "tracking_channel", ""
                ) or evt.annotations.get("tracking_channel", "")
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
            neigh_str = (
                ";".join(neighbors)
                if isinstance(neighbors, (list, tuple))
                else str(neighbors)
            )

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
                w.writerow(
                    [
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
                    ]
                )


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
        neigh_key = fieldnames_lower.get(
            "neighbor_channels", fieldnames_lower.get("neighbors", None)
        )
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
                    auto_track_val = (
                        at_raw.strip().lower() in ("true", "1", "t", "yes")
                        if isinstance(at_raw, str)
                        else bool(at_raw)
                    )

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
                    units_dict[spikeid]["neighbor_channels"] = [
                        x.strip() for x in raw_n.split(sep) if x.strip()
                    ]

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


def export_units_npz(
    save_filename: str,
    state,
    window_pre: pq.Quantity = -0.5 * pq.ms,
    window_post: pq.Quantity = 1.0 * pq.ms,
):
    """Export units and signals to a condensed .npz file package."""
    timestamps = state.event_signal
    units_erps = {}
    for i, sg in enumerate(state.spike_groups):
        nom = sg.event.name or f"unit_{i}"
        if state.analog_signal is not None and len(sg.event) > 0:
            erp = create_erp_signals(
                state.analog_signal, sg.event, window_pre, window_post
            )
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
        window_size_s=float(
            state.window_size.rescale(pq.s).magnitude if state.window_size else 0.5
        ),
        **units_erps,
        **signals_dict,
    )


def save_state_to_nix(
    filename: str,
    spike_groups,
    data: neo.Segment = None,
    metadata=None,
    candidates=None,
):
    """Saves a file containing the spike groups and multi-channel signals to NIX/NWB format.
    Ensures MultiAnalogSignal is converted appropriately and saves channel/neighbor_channels in annotations.

    ``candidates`` may be the dict returned by ``ViewerState.export_candidate_store``
    or an already-serialized JSON string; it is stored in the segment annotation
    ``spikespy_candidates`` so the four-tuple ``load_file`` contract stays intact.
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
                ch_name = getattr(x, "name", "CH0") or "CH0"
                new_seg = neo.Segment(f"ERP_{ch_name}")
                new_seg.annotations["channel"] = ch_name
                new_seg.annotations["class_"] = "multianalogsignal"
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

    if candidates is not None:
        if isinstance(candidates, str):
            payload = candidates
        else:
            payload = json.dumps(candidates, separators=(",", ":"), default=str)
        data2.annotations["spikespy_candidates"] = payload

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
