import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import PySide6
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QAction, QColor, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .ViewerState import ViewerState, tracked_neuron_unit


class ChannelGroupDialog(QDialog):
    """
    Dialog to quickly view, select, modify, shift, and save groups of recording channels.
    Allows selecting a block of channels (e.g. 4 channels for a tetrode / unit), shifting
    them together as a group across recording channels, and assigning them to a unit.
    """

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        state: Optional[ViewerState] = None,
        initial_channels: Optional[List[str]] = None,
        unit_idx: Optional[int] = None,
    ):
        super().__init__(parent)
        self.state = state
        self.unit_idx = unit_idx
        self.setWindowTitle("Channel Group Manager")
        self.resize(420, 500)

        layout = QVBoxLayout(self)

        # Unit context header if applicable
        if self.unit_idx is not None and self.state and 0 <= self.unit_idx < len(self.state.spike_groups):
            u_name = getattr(self.state.spike_groups[self.unit_idx].event, "name", f"unit_{self.unit_idx}")
            lbl_header = QLabel(f"<b>Configuring Channels for Unit {self.unit_idx}: {u_name}</b>", self)
            layout.addWidget(lbl_header)

        # Search / filter box
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Filter:", self))
        self.txt_filter = QLineEdit(self)
        self.txt_filter.setPlaceholderText("Filter channel names...")
        self.txt_filter.textChanged.connect(self._filter_channels)
        filter_layout.addWidget(self.txt_filter)
        layout.addLayout(filter_layout)

        # Channel checklist
        self.list_widget = QListWidget(self)
        self.list_widget.setSelectionMode(QListWidget.ExtendedSelection)
        layout.addWidget(self.list_widget)

        # Populate channels
        avail_channels = self.state.get_available_channels() if self.state else []
        selected_set = set(initial_channels or [])
        if not selected_set and self.unit_idx is not None and self.state:
            selected_set = set(self.state.get_unit_channels(self.unit_idx))

        for ch in avail_channels:
            item = QListWidgetItem(ch, self.list_widget)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if ch in selected_set else Qt.Unchecked)

        self.list_widget.itemChanged.connect(lambda item: self._update_count_label())

        # Count summary label
        self.lbl_count = QLabel(self)
        layout.addWidget(self.lbl_count)
        self._update_count_label()

        # Shift & Quick Select Buttons
        btn_layout = QHBoxLayout()
        self.btn_tetrode = QPushButton("Select 4-Ch (Tetrode)", self)
        self.btn_tetrode.setToolTip("Select 4 consecutive channels starting from selected item")
        self.btn_tetrode.clicked.connect(self._select_tetrode)
        btn_layout.addWidget(self.btn_tetrode)

        self.btn_shift_prev4 = QPushButton("<< -4", self)
        self.btn_shift_prev4.setToolTip("Shift channel selection by -4 channels")
        self.btn_shift_prev4.clicked.connect(lambda: self._shift_selection(-4))
        btn_layout.addWidget(self.btn_shift_prev4)

        self.btn_shift_prev1 = QPushButton("< -1", self)
        self.btn_shift_prev1.setToolTip("Shift channel selection by -1 channel")
        self.btn_shift_prev1.clicked.connect(lambda: self._shift_selection(-1))
        btn_layout.addWidget(self.btn_shift_prev1)

        self.btn_shift_next1 = QPushButton("+1 >", self)
        self.btn_shift_next1.setToolTip("Shift channel selection by +1 channel")
        self.btn_shift_next1.clicked.connect(lambda: self._shift_selection(1))
        btn_layout.addWidget(self.btn_shift_next1)

        self.btn_shift_next4 = QPushButton("+4 >>", self)
        self.btn_shift_next4.setToolTip("Shift channel selection by +4 channels")
        self.btn_shift_next4.clicked.connect(lambda: self._shift_selection(4))
        btn_layout.addWidget(self.btn_shift_next4)

        layout.addLayout(btn_layout)

        # Action buttons row: Select All, Clear, Save as Group Preset
        action_layout = QHBoxLayout()
        btn_all = QPushButton("All", self)
        btn_all.clicked.connect(lambda: self._set_all_checked(True))
        action_layout.addWidget(btn_all)

        btn_none = QPushButton("Clear", self)
        btn_none.clicked.connect(lambda: self._set_all_checked(False))
        action_layout.addWidget(btn_none)

        self.btn_save_preset = QPushButton("Save Group Preset...", self)
        self.btn_save_preset.clicked.connect(self._save_group_preset)
        action_layout.addWidget(self.btn_save_preset)

        if self.unit_idx is not None and self.state:
            self.btn_assign_unit = QPushButton("Apply to Unit", self)
            self.btn_assign_unit.clicked.connect(self._apply_to_unit)
            action_layout.addWidget(self.btn_assign_unit)

        layout.addLayout(action_layout)

        # Standard dialog button box
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _filter_channels(self, text: str):
        query = text.lower().strip()
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            item.setHidden(query not in item.text().lower() if query else False)

    def _update_count_label(self):
        checked = self.get_selected_channels()
        count = len(checked)
        channels_str = ", ".join(checked[:6]) + ("..." if count > 6 else "")
        self.lbl_count.setText(f"<b>{count} channels selected:</b> {channels_str if count else '(none)'}")

    def _set_all_checked(self, checked: bool):
        state = Qt.Checked if checked else Qt.Unchecked
        for i in range(self.list_widget.count()):
            self.list_widget.item(i).setCheckState(state)
        self._update_count_label()

    def get_selected_channels(self) -> List[str]:
        selected = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.checkState() == Qt.Checked:
                selected.append(item.text())
        return selected

    def set_selected_channels(self, channels: List[str]):
        ch_set = set(channels)
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            item.setCheckState(Qt.Checked if item.text() in ch_set else Qt.Unchecked)
        self._update_count_label()

    def _select_tetrode(self):
        curr_row = self.list_widget.currentRow()
        if curr_row < 0:
            curr_row = 0
        self._set_all_checked(False)
        for r in range(curr_row, min(curr_row + 4, self.list_widget.count())):
            self.list_widget.item(r).setCheckState(Qt.Checked)
        self._update_count_label()

    def _shift_selection(self, step: int):
        current = self.get_selected_channels()
        if not current or not self.state:
            return
        shifted = self.state.shift_channels(current, step=step)
        self.set_selected_channels(shifted)

    def _save_group_preset(self):
        selected = self.get_selected_channels()
        if not selected:
            if QApplication.instance() and QApplication.instance().platformName() != "offscreen":
                QMessageBox.warning(self, "No Channels", "Please select at least one channel first.")
            return
        default_name = f"Group ({len(selected)} ch: {selected[0]}..{selected[-1]})"
        if QApplication.instance() and QApplication.instance().platformName() == "offscreen":
            name, ok = default_name, True
        else:
            name, ok = QInputDialog.getText(self, "Save Channel Group", "Group Name:", text=default_name)
        if ok and name.strip():
            if self.state:
                self.state.add_channel_group(name.strip(), selected)
            if QApplication.instance() and QApplication.instance().platformName() != "offscreen":
                QMessageBox.information(self, "Saved", f"Channel group '{name.strip()}' saved successfully.")

    def _apply_to_unit(self):
        if self.unit_idx is not None and self.state:
            selected = self.get_selected_channels()
            if selected:
                self.state.set_unit_channels(self.unit_idx, selected)
                if QApplication.instance() and QApplication.instance().platformName() != "offscreen":
                    QMessageBox.information(
                        self, "Applied", f"Channels applied to Unit {self.unit_idx}: {', '.join(selected)}"
                    )


class ChannelGroupBar(QToolBar):
    """
    A reusable toolbar component for independent channel selection and unit binding.
    Every view (MultiTraceView, SingleTraceView, MultiTraceFixedView, UnitView, UnitFootprintView)
    can include this bar to allow independent channel selection, multi-channel integration,
    and binding to specific units.
    """

    bindingChanged = Signal()

    MODE_FOLLOW_GLOBAL = "__global__"
    MODE_FOLLOW_UNIT = "__unit__"
    MODE_GROUP = "__group__"
    MODE_SINGLE = "__single__"
    MODE_SHARED_SET = "__shared__"

    def __init__(self, parent: Optional[QWidget] = None, state: Optional[ViewerState] = None):
        super().__init__("Channels & Unit", parent)
        self.state: Optional[ViewerState] = state
        self._block_updates = False

        # Target Unit Selector
        self.lbl_unit = QLabel(" Unit: ", self)
        self.addWidget(self.lbl_unit)

        self.combo_unit = QComboBox(self)
        self.combo_unit.setToolTip(
            "Select target unit. Choose '(Active Unit)' to follow global selection,\n"
            "or lock this window to a specific unit (e.g. Unit 1)."
        )
        self.combo_unit.currentIndexChanged.connect(self._on_unit_selection_changed)
        self.addWidget(self.combo_unit)

        self.addSeparator()

        # Channel / Group Selector
        self.lbl_chan = QLabel(" Chans: ", self)
        self.addWidget(self.lbl_chan)

        self.combo_channel = QComboBox(self)
        self.combo_channel.setToolTip(
            "Select channel or channel group:\n"
            "- (Active Channel): follow global active recording channel\n"
            "- (Unit Channels): display all channels where the bound unit is identified\n"
            "- [Group] ...: use a saved channel group\n"
            "- Individual Channel: pick a specific channel"
        )
        self.combo_channel.currentIndexChanged.connect(self._on_channel_selection_changed)
        self.addWidget(self.combo_channel)

        # Multi-Channel Toggle
        self.chk_multichannel = QCheckBox("Multi-Ch", self)
        self.chk_multichannel.setToolTip("Display or integrate all channels in the selected group/unit")
        self.chk_multichannel.toggled.connect(self._on_multichannel_toggled)
        self.addWidget(self.chk_multichannel)

        # Quick Shift Buttons
        self.btn_shift_prev = QPushButton("<", self)
        self.btn_shift_prev.setFixedWidth(24)
        self.btn_shift_prev.setToolTip("Shift channel group by -1 channel")
        self.btn_shift_prev.clicked.connect(lambda: self.shift_group(-1))
        self.addWidget(self.btn_shift_prev)

        self.btn_shift_next = QPushButton(">", self)
        self.btn_shift_next.setFixedWidth(24)
        self.btn_shift_next.setToolTip("Shift channel group by +1 channel")
        self.btn_shift_next.clicked.connect(lambda: self.shift_group(1))
        self.addWidget(self.btn_shift_next)

        # Channel Group Menu Button
        self.btn_group_menu = QToolButton(self)
        self.btn_group_menu.setText("Group ▾")
        self.btn_group_menu.setPopupMode(QToolButton.InstantPopup)
        self._build_group_menu()
        self.addWidget(self.btn_group_menu)

        if state is not None:
            self.setState(state)

    def find_channel_data_index(self, mode: str, val: Any = None) -> int:
        """Finds index in combo_channel matching (mode, val) without relying on C++ QVariant equality."""
        for i in range(self.combo_channel.count()):
            d = self.combo_channel.itemData(i)
            if isinstance(d, (tuple, list)) and len(d) == 2:
                if d[0] == mode and (val is None or d[1] == val):
                    return i
        return -1

    def find_unit_data_index(self, unit_idx: Optional[int]) -> int:
        """Finds index in combo_unit matching unit_idx."""
        for i in range(self.combo_unit.count()):
            if self.combo_unit.itemData(i) == unit_idx:
                return i
        return -1

    def setState(self, state: ViewerState):
        self.state = state
        if self.state is not None:
            self.state.onUnitGroupChange.connect(self.refresh_units)
            self.state.onLoadNewFile.connect(self.refresh_all)
            if hasattr(self.state, "onChannelGroupsChanged"):
                self.state.onChannelGroupsChanged.connect(self.refresh_channels)
            if hasattr(self.state, "onChannelChange"):
                self.state.onChannelChange.connect(self._on_global_channel_changed)
            if hasattr(self.state, "onSharedChannelSetChanged"):
                self.state.onSharedChannelSetChanged.connect(self._on_shared_set_changed)
        self.refresh_all()

    def _build_group_menu(self):
        menu = QMenu(self)
        act_edit = menu.addAction("Edit Channels in Group...")
        act_edit.triggered.connect(self.open_channel_group_dialog)

        act_tetrode = menu.addAction("Quick 4-Ch Tetrode Group")
        act_tetrode.triggered.connect(self.quick_select_tetrode)

        menu.addSeparator()
        act_shift_m4 = menu.addAction("Shift Group -4 Channels")
        act_shift_m4.triggered.connect(lambda: self.shift_group(-4))

        act_shift_p4 = menu.addAction("Shift Group +4 Channels")
        act_shift_p4.triggered.connect(lambda: self.shift_group(4))

        act_shift_span_m = menu.addAction("Shift by Group Span (-N)")
        act_shift_span_m.triggered.connect(lambda: self.shift_group_by_span(-1))

        act_shift_span_p = menu.addAction("Shift by Group Span (+N)")
        act_shift_span_p.triggered.connect(lambda: self.shift_group_by_span(1))

        menu.addSeparator()
        act_assign = menu.addAction("Assign Group to Bound Unit")
        act_assign.triggered.connect(self.assign_group_to_unit)

        act_save = menu.addAction("Save Current Group as Named Preset...")
        act_save.triggered.connect(self.save_group_preset)

        act_detect = menu.addAction("Auto-Detect Unit Channels (Footprint)")
        act_detect.triggered.connect(self.auto_detect_unit_channels)

        self.btn_group_menu.setMenu(menu)

    def refresh_all(self):
        self._block_updates = True
        self.refresh_units()
        self.refresh_channels()
        self._block_updates = False
        self.bindingChanged.emit()

    def refresh_units(self):
        curr_data = self.combo_unit.currentData()
        self.combo_unit.blockSignals(True)
        self.combo_unit.clear()

        # Follow active unit option
        self.combo_unit.addItem("(Follow Active Unit)", userData=None)

        if self.state and self.state.spike_groups:
            for i, sg in enumerate(self.state.spike_groups):
                u_name = getattr(sg.event, "name", f"unit_{i}") or f"unit_{i}"
                ch = getattr(sg, "channel", "")
                ch_info = f" [{ch}]" if ch else ""
                self.combo_unit.addItem(f"Unit {i}: {u_name}{ch_info}", userData=i)

        # Restore previous selection if possible
        restored = False
        if curr_data is not None:
            idx = self.find_unit_data_index(curr_data)
            if idx >= 0:
                self.combo_unit.setCurrentIndex(idx)
                restored = True
        if not restored:
            self.combo_unit.setCurrentIndex(0)
        self.combo_unit.blockSignals(False)

    def refresh_channels(self):
        curr_data = self.combo_channel.currentData()
        self.combo_channel.blockSignals(True)
        self.combo_channel.clear()

        # Dynamic options
        self.combo_channel.addItem("(Follow Global Channel)", userData=(self.MODE_FOLLOW_GLOBAL, None))
        self.combo_channel.addItem("(Bound Unit Channels)", userData=(self.MODE_FOLLOW_UNIT, None))
        self.combo_channel.addItem("(🔗 Shared Group A)", userData=(self.MODE_SHARED_SET, "Shared Group A"))
        self.combo_channel.addItem("(🔗 Shared Group B)", userData=(self.MODE_SHARED_SET, "Shared Group B"))

        # Available individual channels and groups
        if self.state:
            avail = self.state.get_available_channels()
            all_groups = self.state.get_all_channel_groups() if hasattr(self.state, "get_all_channel_groups") else {}

            if all_groups:
                self.combo_channel.insertSeparator(self.combo_channel.count())
                for g_name, g_chans in all_groups.items():
                    preview = f" ({len(g_chans)} ch: {', '.join(g_chans[:3])}{'...' if len(g_chans)>3 else ''})"
                    self.combo_channel.addItem(f"[Group] {g_name}{preview}", userData=(self.MODE_GROUP, g_name))

            if avail:
                self.combo_channel.insertSeparator(self.combo_channel.count())
                for ch in avail:
                    self.combo_channel.addItem(ch, userData=(self.MODE_SINGLE, ch))

        # Restore previous selection if possible
        restored = False
        if curr_data is not None and isinstance(curr_data, (tuple, list)) and len(curr_data) == 2:
            idx = self.find_channel_data_index(curr_data[0], curr_data[1])
            if idx >= 0:
                self.combo_channel.setCurrentIndex(idx)
                restored = True
        if not restored:
            self.combo_channel.setCurrentIndex(0)
        self.combo_channel.blockSignals(False)

    def _on_unit_selection_changed(self, index: int):
        if not self._block_updates:
            self.bindingChanged.emit()

    def _on_channel_selection_changed(self, index: int):
        if not self._block_updates:
            mode, val = self.get_channel_mode()
            if mode in (self.MODE_FOLLOW_UNIT, self.MODE_GROUP, self.MODE_SHARED_SET):
                self.chk_multichannel.blockSignals(True)
                self.chk_multichannel.setChecked(True)
                self.chk_multichannel.blockSignals(False)
            self.bindingChanged.emit()

    def _on_shared_set_changed(self, set_name: str):
        mode, val = self.get_channel_mode()
        if mode == self.MODE_SHARED_SET and val == set_name:
            self.bindingChanged.emit()

    def _on_multichannel_toggled(self, checked: bool):
        if not self._block_updates:
            self.bindingChanged.emit()

    def _on_global_channel_changed(self, channel_name: str):
        mode, val = self.get_channel_mode()
        if mode == self.MODE_FOLLOW_GLOBAL:
            self.bindingChanged.emit()

    # Public Getters & Binding API

    def get_target_unit_idx(self) -> int:
        """Returns the index of the bound unit, or state.cur_spike_group if following active."""
        data = self.combo_unit.currentData()
        if data is not None and isinstance(data, int):
            return data
        return getattr(self.state, "cur_spike_group", 0) if self.state else 0

    def get_target_unit(self) -> Optional[tracked_neuron_unit]:
        """Returns the bound unit object."""
        if not self.state or not self.state.spike_groups:
            return None
        u_idx = self.get_target_unit_idx()
        if 0 <= u_idx < len(self.state.spike_groups):
            return self.state.spike_groups[u_idx]
        return None

    def is_unit_locked(self) -> bool:
        """Returns True if the view is locked to a specific unit (not following active)."""
        return self.combo_unit.currentData() is not None

    def get_bound_unit(self) -> Optional[int]:
        """Returns the unit index if locked to a unit, else None."""
        data = self.combo_unit.currentData()
        return data if isinstance(data, int) else None

    def get_bound_shared_set(self) -> Optional[str]:
        """Returns the shared channel group name if bound to one, else None."""
        mode, val = self.get_channel_mode()
        return val if mode == self.MODE_SHARED_SET else None

    def get_channel_mode(self) -> Tuple[str, Any]:
        """Returns (mode, value) tuple for the current channel dropdown selection."""
        data = self.combo_channel.currentData()
        if data is not None and isinstance(data, tuple) and len(data) == 2:
            return data
        return (self.MODE_FOLLOW_GLOBAL, None)

    def is_multichannel(self) -> bool:
        """Returns True if multi-channel display / integration is enabled."""
        return self.chk_multichannel.isChecked()

    def get_selected_channels(self) -> List[str]:
        """
        Returns the resolved list of channels selected for this component.
        Handles Follow Global, Bound Unit, Channel Groups, and Single Channels.
        """
        if not self.state:
            return []

        mode, val = self.get_channel_mode()
        avail = self.state.get_available_channels()

        if mode == self.MODE_FOLLOW_GLOBAL:
            global_name = (
                getattr(self.state.analog_signal, "name", "")
                if self.state.analog_signal is not None
                else (avail[0] if avail else "")
            )
            if self.is_multichannel():
                # Include bound unit's neighbors or nearby channels
                u_idx = self.get_target_unit_idx()
                u_chans = self.state.get_unit_channels(u_idx) if hasattr(self.state, "get_unit_channels") else []
                res = [global_name] if global_name else []
                for c in u_chans:
                    if c and c not in res:
                        res.append(c)
                return res or avail[:4]
            return [global_name] if global_name else (avail[:1] if avail else [])

        elif mode == self.MODE_FOLLOW_UNIT:
            u_idx = self.get_target_unit_idx()
            u_chans = self.state.get_unit_channels(u_idx) if hasattr(self.state, "get_unit_channels") else []
            if not u_chans and avail:
                u_chans = [avail[0]]
            return u_chans if self.is_multichannel() else u_chans[:1]

        elif mode == self.MODE_GROUP:
            g_chans = self.state.get_channel_group(val) if hasattr(self.state, "get_channel_group") else []
            if not g_chans and hasattr(self.state, "get_all_channel_groups"):
                all_g = self.state.get_all_channel_groups()
                g_chans = all_g.get(val, [])
            return g_chans if self.is_multichannel() else g_chans[:1]

        elif mode == self.MODE_SHARED_SET:
            s = self.state.get_shared_channel_set(val) if hasattr(self.state, "get_shared_channel_set") else {}
            chans = s.get("channels", [])
            if not chans and avail:
                chans = avail[:4] if len(avail) >= 4 else avail
            return chans if self.is_multichannel() else chans[:1]

        elif mode == self.MODE_SINGLE:
            ch = str(val)
            if self.is_multichannel():
                # If multi-ch is checked on a single channel, include unit neighbors
                u_idx = self.get_target_unit_idx()
                u_chans = self.state.get_unit_channels(u_idx) if hasattr(self.state, "get_unit_channels") else []
                res = [ch]
                for c in u_chans:
                    if c != ch and c not in res:
                        res.append(c)
                return res
            return [ch]

        return avail[:1] if avail else []

    def get_primary_channel(self) -> str:
        """Returns the primary active channel name for this component."""
        chans = self.get_selected_channels()
        if chans:
            return chans[0]
        if self.state and self.state.analog_signal is not None and getattr(self.state.analog_signal, "name", ""):
            return self.state.analog_signal.name
        avail = self.state.get_available_channels() if self.state else []
        return avail[0] if avail else ""

    def get_title_summary(self) -> str:
        """Returns a concise summary string suitable for window / dock titles."""
        u_idx = self.get_target_unit_idx()
        unit_str = f"Unit {u_idx}"
        if self.state and 0 <= u_idx < len(self.state.spike_groups):
            u_name = getattr(self.state.spike_groups[u_idx].event, "name", "")
            if u_name:
                unit_str = f"Unit {u_idx}: {u_name}"
        if not self.is_unit_locked():
            unit_str = "(Active Unit)"

        chans = self.get_selected_channels()
        if not chans:
            chan_str = "no chan"
        elif len(chans) == 1:
            chan_str = chans[0]
        elif len(chans) <= 4:
            chan_str = ", ".join(chans)
        else:
            chan_str = f"{len(chans)} ch: {chans[0]}..{chans[-1]}"

        return f"[{unit_str}] ({chan_str})"

    def bind_to_unit(self, unit_idx: int):
        """Programmatically locks this component to the specified unit."""
        idx = self.find_unit_data_index(unit_idx)
        if idx >= 0:
            self.combo_unit.setCurrentIndex(idx)

    def bind_to_channels(self, channels: List[str]):
        """Programmatically sets the channel list or assigns it to the bound unit."""
        if not channels or not self.state:
            return
        if len(channels) == 1:
            idx = self.find_channel_data_index(self.MODE_SINGLE, channels[0])
            if idx >= 0:
                self.combo_channel.setCurrentIndex(idx)
        else:
            # Set on bound unit or create temporary group
            u_idx = self.get_target_unit_idx()
            self.state.set_unit_channels(u_idx, channels)
            self.set_channel_mode_unit()

    def bind_to_shared_set(self, set_name: str = "Shared Group A"):
        """Locks channel selection to a shared channel set and enables multi-channel display."""
        idx = self.find_channel_data_index(self.MODE_SHARED_SET, set_name)
        if idx >= 0:
            self.combo_channel.setCurrentIndex(idx)
        self.chk_multichannel.setChecked(True)
        self.bindingChanged.emit()

    def set_channel_mode_unit(self):
        """Sets channel selection to '(Bound Unit Channels)'."""
        idx = self.find_channel_data_index(self.MODE_FOLLOW_UNIT)
        if idx >= 0:
            self.combo_channel.setCurrentIndex(idx)
            self.chk_multichannel.setChecked(True)

    def shift_group_by_span(self, direction: int = 1):
        """Shifts channel selection by the span (number of selected channels) in the given direction (+1 or -1)."""
        chans = self.get_selected_channels()
        span = len(chans) if chans else 1
        self.shift_group(span * direction)

    def shift_group(self, step: int):
        """Shifts the current channel group or bound unit's channels along recording channels."""
        if not self.state:
            return
        mode, val = self.get_channel_mode()
        if mode == self.MODE_SHARED_SET:
            if hasattr(self.state, "shift_shared_channels"):
                self.state.shift_shared_channels(val, step=step)
            return

        current = self.get_selected_channels()
        if not current:
            return
        shifted = self.state.shift_channels(current, step=step)
        if not shifted:
            return

        # If bound to unit or in unit channel mode, update unit's channels
        u_idx = self.get_target_unit_idx()
        if mode == self.MODE_FOLLOW_UNIT or self.is_unit_locked():
            self.state.set_unit_channels(u_idx, shifted)
        elif len(shifted) == 1:
            idx = self.find_channel_data_index(self.MODE_SINGLE, shifted[0])
            if idx >= 0:
                self.combo_channel.setCurrentIndex(idx)
        else:
            group_name = f"Shifted ({len(shifted)} ch: {shifted[0]}..{shifted[-1]})"
            self.state.add_channel_group(group_name, shifted)
            idx = self.find_channel_data_index(self.MODE_GROUP, group_name)
            if idx >= 0:
                self.combo_channel.setCurrentIndex(idx)

        self.bindingChanged.emit()

    def open_channel_group_dialog(self):
        """Opens the interactive Channel Group Manager dialog."""
        u_idx = self.get_target_unit_idx() if self.is_unit_locked() else None
        dlg = ChannelGroupDialog(
            parent=self,
            state=self.state,
            initial_channels=self.get_selected_channels(),
            unit_idx=u_idx,
        )
        if dlg.exec() == QDialog.Accepted:
            selected = dlg.get_selected_channels()
            if selected:
                if self.is_unit_locked():
                    self.state.set_unit_channels(self.get_target_unit_idx(), selected)
                    self.set_channel_mode_unit()
                elif len(selected) == 1:
                    idx = self.find_channel_data_index(self.MODE_SINGLE, selected[0])
                    if idx >= 0:
                        self.combo_channel.setCurrentIndex(idx)
                else:
                    g_name = f"Custom ({len(selected)} ch: {selected[0]}..{selected[-1]})"
                    self.state.add_channel_group(g_name, selected)
                    idx = self.find_channel_data_index(self.MODE_GROUP, g_name)
                    if idx >= 0:
                        self.combo_channel.setCurrentIndex(idx)
                self.bindingChanged.emit()

    def quick_select_tetrode(self):
        """Selects 4 consecutive channels starting from current primary channel."""
        avail = self.state.get_available_channels() if self.state else []
        if not avail:
            return
        primary = self.get_primary_channel()
        start_i = avail.index(primary) if primary in avail else 0
        tetrode = avail[start_i : min(start_i + 4, len(avail))]
        if len(tetrode) < 4 and len(avail) >= 4:
            tetrode = avail[max(0, len(avail) - 4) :]
        if self.is_unit_locked():
            self.state.set_unit_channels(self.get_target_unit_idx(), tetrode)
            self.set_channel_mode_unit()
        else:
            g_name = f"Tetrode ({len(tetrode)} ch: {tetrode[0]}..{tetrode[-1]})"
            self.state.add_channel_group(g_name, tetrode)
            self.refresh_channels()
            idx = self.find_channel_data_index(self.MODE_GROUP, g_name)
            if idx >= 0:
                self.combo_channel.setCurrentIndex(idx)
            self.chk_multichannel.setChecked(True)
        self.bindingChanged.emit()

    def assign_group_to_unit(self):
        """Assigns the currently selected channels to the bound unit."""
        if not self.state:
            return
        u_idx = self.get_target_unit_idx()
        chans = self.get_selected_channels()
        if chans:
            self.state.set_unit_channels(u_idx, chans)
            self.set_channel_mode_unit()
            self.bindingChanged.emit()

    def save_group_preset(self):
        """Prompts for a name and saves the current channel group."""
        chans = self.get_selected_channels()
        if not chans or not self.state:
            return
        default_name = f"Preset ({len(chans)} ch: {chans[0]}..{chans[-1]})"
        if QApplication.instance() and QApplication.instance().platformName() == "offscreen":
            name, ok = default_name, True
        else:
            name, ok = QInputDialog.getText(self, "Save Preset", "Channel Group Name:", text=default_name)
        if ok and name.strip():
            self.state.add_channel_group(name.strip(), chans)
            self.refresh_channels()
            idx = self.find_channel_data_index(self.MODE_GROUP, name.strip())
            if idx >= 0:
                self.combo_channel.setCurrentIndex(idx)

    def auto_detect_unit_channels(self):
        """Auto-detects the 4 best channels for the bound unit based on waveform amplitude."""
        u_idx = self.get_target_unit_idx()
        if not self.state or u_idx >= len(self.state.spike_groups):
            return
        sg = self.state.spike_groups[u_idx]
        from .UnitFootprintView import UnitFootprintView

        temp_view = UnitFootprintView(state=self.state)
        temp_view.auto_detect_unit_channels(sg=sg)
        self.refresh_units()
        self.set_channel_mode_unit()
        self.bindingChanged.emit()

    # Settings persistence

    def get_settings(self) -> dict:
        return {
            "unit_binding": self.combo_unit.currentData(),
            "channel_mode": self.combo_channel.currentData(),
            "multichannel": self.chk_multichannel.isChecked(),
        }

    def set_settings(self, settings: dict):
        if not settings:
            return
        self._block_updates = True
        if "unit_binding" in settings:
            idx = self.find_unit_data_index(settings["unit_binding"])
            if idx >= 0:
                self.combo_unit.setCurrentIndex(idx)
        if "channel_mode" in settings:
            m = settings["channel_mode"]
            if isinstance(m, (tuple, list)) and len(m) == 2:
                idx = self.find_channel_data_index(m[0], m[1])
                if idx >= 0:
                    self.combo_channel.setCurrentIndex(idx)
        if "multichannel" in settings:
            self.chk_multichannel.setChecked(bool(settings["multichannel"]))
        self._block_updates = False
        self.bindingChanged.emit()
