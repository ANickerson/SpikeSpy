from typing import Any, List, Optional, Union

import neo
import PySide6
from PySide6.QtWidgets import QCheckBox, QComboBox, QFormLayout, QMainWindow, QWidget, QSpinBox
import quantities as pq
from .ViewerState import ViewerState


class NeoSettingsView(QMainWindow):
    def __init__(
        self,
        parent: Optional[PySide6.QtWidgets.QWidget] = None,
        state: ViewerState = None,
    ) -> None:
        super().__init__(parent)
        layout = QFormLayout()
        self.state = state
        if self.state is not None:
            self.state.onLoadNewFile.connect(
                lambda: self.populate_comboboxes(self.state.segment)
            )
            if hasattr(self.state, "onChannelChange"):
                self.state.onChannelChange.connect(self._on_channel_changed)
        w = QWidget()
        w.setLayout(layout)
        self.setCentralWidget(w)

        self.analogSignalCombo = QComboBox(self)
        layout.addRow("Analog signal", self.analogSignalCombo)
        self.analogSignalCombo.currentIndexChanged.connect(self._on_analog_combo_changed)

        self.eventSignalCombo = QComboBox(self)

        layout.addRow("event channel", self.eventSignalCombo)
        self.eventSignalCombo.currentIndexChanged.connect(
            lambda _: self.state.set_data(
                event_signal=self.eventSignalCombo.currentData()
            )
        )
        if self.state is not None:
            self.populate_comboboxes(seg=self.state.segment)
        self.window_size = QSpinBox()

        self.window_size.setMinimum(500)
        self.window_size.setMaximum(2000)
        if self.state is not None:
            self.window_size.setValue(int(self.state.window_size.rescale(pq.ms)))
            self.window_size.valueChanged.connect(self.state.set_window_size)
        self.window_size.setSuffix("ms")
        self.window_size.setSingleStep(100)

        layout.addRow("window_size", self.window_size)

        self.chk_auto_unit_chan = QCheckBox("Auto-switch to unit's channel", self)
        if self.state is not None:
            self.chk_auto_unit_chan.setChecked(
                getattr(self.state, "auto_switch_unit_channel", False)
            )
            self.chk_auto_unit_chan.toggled.connect(self._on_auto_unit_chan_toggled)
        layout.addRow("", self.chk_auto_unit_chan)

    def _on_analog_combo_changed(self, _index: int):
        if self.state is None:
            return
        data = self.analogSignalCombo.currentData()
        if data is None:
            return
        if hasattr(self.state, "set_active_channel"):
            self.state.set_active_channel(data)
        else:
            self.state.set_data(data)

    def _on_auto_unit_chan_toggled(self, checked: bool):
        if self.state is not None:
            self.state.auto_switch_unit_channel = bool(checked)

    def _on_channel_changed(self, channel_name: str):
        self.analogSignalCombo.blockSignals(True)
        idx = self.analogSignalCombo.findText(channel_name)
        if idx >= 0:
            self.analogSignalCombo.setCurrentIndex(idx)
        self.analogSignalCombo.blockSignals(False)

    def populate_comboboxes(self, seg: neo.Block):
        self.analogSignalCombo.blockSignals(True)
        self.eventSignalCombo.blockSignals(True)
        self.analogSignalCombo.clear()
        self.eventSignalCombo.clear()

        if seg is None:
            self.analogSignalCombo.blockSignals(False)
            self.eventSignalCombo.blockSignals(False)
            return
        for a in seg.analogsignals:
            self.analogSignalCombo.addItem(a.name, userData=a)
        for a in seg.events:
            self.eventSignalCombo.addItem(a.name, userData=a)
        self.analogSignalCombo.setCurrentIndex(
            self.analogSignalCombo.findData(self.state.analog_signal)
        )

        self.analogSignalCombo.blockSignals(False)
        self.eventSignalCombo.setCurrentIndex(
            self.eventSignalCombo.findData(self.state.event_signal)
        )
        self.eventSignalCombo.blockSignals(False)
