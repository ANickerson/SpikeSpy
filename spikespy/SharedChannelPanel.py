import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import PySide6
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .ViewerState import ViewerState


class SharedChannelPanel(QWidget):
    """
    SharedChannelPanel coordinates multi-panel viewing by managing shared channel sets
    (e.g. Shared Group A, Shared Group B), allowing synchronized channel navigation,
    one-click group chips, auto-splitting into tetrodes/octodes, and mass viewer linking.
    """

    def __init__(self, parent: Optional[QWidget] = None, state: Optional[ViewerState] = None):
        super().__init__(parent)
        self.state: Optional[ViewerState] = state
        self._block_updates = False
        self._chip_buttons: Dict[str, QPushButton] = {}

        self.setWindowTitle("Shared Channel Controller")
        self.setObjectName("SharedChannelPanel")

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)

        # -------------------------------------------------------------
        # 1. Top Control Bar: Set selector, Active channels badge, Unit & Multi-Ch
        # -------------------------------------------------------------
        top_box = QGroupBox("Active Shared Channel Set", self)
        top_layout = QGridLayout(top_box)
        top_layout.setContentsMargins(6, 6, 6, 6)
        top_layout.setSpacing(6)

        # Set Selector
        top_layout.addWidget(QLabel("<b>Shared Set:</b>", self), 0, 0)
        self.combo_set = QComboBox(self)
        self.combo_set.addItems(["Shared Group A", "Shared Group B"])
        self.combo_set.currentIndexChanged.connect(self._on_set_changed)
        top_layout.addWidget(self.combo_set, 0, 1)

        # Active channels badge
        top_layout.addWidget(QLabel("<b>Active Channels:</b>", self), 0, 2)
        self.lbl_badge = QLabel("(None)", self)
        self.lbl_badge.setStyleSheet(
            "QLabel { background-color: palette(alternate-base); border: 1px solid palette(mid); "
            "border-radius: 4px; padding: 2px 6px; font-weight: bold; }"
        )
        top_layout.addWidget(self.lbl_badge, 0, 3)

        # Target unit selector
        top_layout.addWidget(QLabel("<b>Target Unit:</b>", self), 1, 0)
        self.combo_unit = QComboBox(self)
        self.combo_unit.currentIndexChanged.connect(self._on_unit_selection_changed)
        top_layout.addWidget(self.combo_unit, 1, 1)

        # Multi-channel checkbox
        self.chk_multichannel = QCheckBox("Multi-Channel", self)
        self.chk_multichannel.setChecked(True)
        self.chk_multichannel.toggled.connect(self._on_multichannel_toggled)
        top_layout.addWidget(self.chk_multichannel, 1, 2, 1, 2)

        main_layout.addWidget(top_box)

        # -------------------------------------------------------------
        # 2. Quick Group Switcher & Auto-Split
        # -------------------------------------------------------------
        group_box = QGroupBox("Quick Group Switcher", self)
        group_vbox = QVBoxLayout(group_box)
        group_vbox.setContentsMargins(6, 6, 6, 6)
        group_vbox.setSpacing(6)

        # Scrollable chip container
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFixedHeight(54)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.chip_container = QWidget()
        self.chip_layout = QHBoxLayout(self.chip_container)
        self.chip_layout.setContentsMargins(2, 2, 2, 2)
        self.chip_layout.setSpacing(4)
        self.chip_layout.addStretch()
        self.scroll_area.setWidget(self.chip_container)
        group_vbox.addWidget(self.scroll_area)

        # Auto-split tool row
        split_layout = QHBoxLayout()
        split_layout.addWidget(QLabel("Auto-Split Channels into Groups of:", self))
        self.spin_split = QSpinBox(self)
        self.spin_split.setRange(1, 128)
        self.spin_split.setValue(4)
        split_layout.addWidget(self.spin_split)

        self.btn_create_groups = QPushButton("Create Groups", self)
        self.btn_create_groups.clicked.connect(self._on_auto_split_clicked)
        split_layout.addWidget(self.btn_create_groups)
        split_layout.addStretch()

        group_vbox.addLayout(split_layout)
        main_layout.addWidget(group_box)

        # -------------------------------------------------------------
        # 3. Group Navigation & Steppers
        # -------------------------------------------------------------
        nav_box = QGroupBox("Group Navigation & Stepping", self)
        nav_layout = QHBoxLayout(nav_box)
        nav_layout.setContentsMargins(6, 6, 6, 6)
        nav_layout.setSpacing(6)

        self.btn_prev_group = QPushButton("⏮ Prev Group", self)
        self.btn_prev_group.setToolTip("Switch to the previous channel group preset")
        self.btn_prev_group.clicked.connect(lambda: self._cycle_group(-1))
        nav_layout.addWidget(self.btn_prev_group)

        self.btn_prev_1 = QPushButton("◀ -1", self)
        self.btn_prev_1.setToolTip("Shift channels backward by 1 channel")
        self.btn_prev_1.clicked.connect(lambda: self._shift_channels(-1))
        nav_layout.addWidget(self.btn_prev_1)

        self.btn_next_1 = QPushButton("+1 ▶", self)
        self.btn_next_1.setToolTip("Shift channels forward by 1 channel")
        self.btn_next_1.clicked.connect(lambda: self._shift_channels(1))
        nav_layout.addWidget(self.btn_next_1)

        self.btn_next_group = QPushButton("Next Group ⏭", self)
        self.btn_next_group.setToolTip("Switch to the next channel group preset")
        self.btn_next_group.clicked.connect(lambda: self._cycle_group(1))
        nav_layout.addWidget(self.btn_next_group)

        self.btn_shift_span = QPushButton("Shift by Group Size", self)
        self.btn_shift_span.setToolTip("Shift channels forward by the count of active channels")
        self.btn_shift_span.clicked.connect(self._shift_by_group_size)
        nav_layout.addWidget(self.btn_shift_span)

        self.btn_new_group = QPushButton("+ New Group...", self)
        self.btn_new_group.setToolTip("Open Channel Group Manager to design and save a custom group")
        self.btn_new_group.clicked.connect(self._on_new_group)
        nav_layout.addWidget(self.btn_new_group)

        main_layout.addWidget(nav_box)

        # -------------------------------------------------------------
        # 4. Viewer Link Control
        # -------------------------------------------------------------
        link_box = QGroupBox("Viewer Link Control", self)
        link_layout = QHBoxLayout(link_box)
        link_layout.setContentsMargins(6, 6, 6, 6)
        link_layout.setSpacing(6)

        self.lbl_linked_viewers = QLabel("Linked Viewers: None", self)
        self.lbl_linked_viewers.setStyleSheet("QLabel { color: palette(text); font-style: italic; }")
        link_layout.addWidget(self.lbl_linked_viewers, 1)

        self.btn_link_all = QPushButton("Link All Open Viewers to This Group", self)
        self.btn_link_all.setToolTip(
            "Iterate over all open dock viewers and synchronize their channel selection to this shared set"
        )
        self.btn_link_all.clicked.connect(self.link_all_open_viewers)
        link_layout.addWidget(self.btn_link_all)

        main_layout.addWidget(link_box)

        if state is not None:
            self.setState(state)

    def setState(self, state: ViewerState):
        self.state = state
        if self.state is not None:
            if hasattr(self.state, "onSharedChannelSetChanged"):
                self.state.onSharedChannelSetChanged.connect(self._on_shared_channel_set_changed)
            if hasattr(self.state, "onChannelGroupsChanged"):
                self.state.onChannelGroupsChanged.connect(self.refresh_chips)
            if hasattr(self.state, "onUnitGroupChange"):
                self.state.onUnitGroupChange.connect(self.refresh_units)
            if hasattr(self.state, "onLoadNewFile"):
                self.state.onLoadNewFile.connect(self.refresh_all)

        self.refresh_all()

    def refresh_all(self):
        self.refresh_units()
        self.refresh_chips()
        self.update_active_set_display()
        self.update_linked_viewers_label()

    def get_active_set(self) -> str:
        return self.combo_set.currentText() or "Shared Group A"

    def get_target_unit_idx(self) -> Optional[int]:
        data = self.combo_unit.currentData()
        if isinstance(data, int):
            return data
        return None

    def get_current_shared_channels(self) -> List[str]:
        if not self.state:
            return []
        s = self.state.get_shared_channel_set(self.get_active_set())
        return list(s.get("channels", []))

    def refresh_units(self):
        curr_data = self.combo_unit.currentData()
        self.combo_unit.blockSignals(True)
        self.combo_unit.clear()

        self.combo_unit.addItem("(Follow Active Unit)", userData=None)

        if self.state and self.state.spike_groups:
            for i, sg in enumerate(self.state.spike_groups):
                u_name = getattr(sg.event, "name", f"unit_{i}") or f"unit_{i}"
                ch = getattr(sg, "channel", "")
                ch_info = f" [{ch}]" if ch else ""
                self.combo_unit.addItem(f"Unit {i}: {u_name}{ch_info}", userData=i)

        restored = False
        if curr_data is not None:
            for idx in range(self.combo_unit.count()):
                if self.combo_unit.itemData(idx) == curr_data:
                    self.combo_unit.setCurrentIndex(idx)
                    restored = True
                    break
        if not restored:
            self.combo_unit.setCurrentIndex(0)
        self.combo_unit.blockSignals(False)

    def refresh_chips(self):
        # Clear existing chips
        while self.chip_layout.count() > 1:
            item = self.chip_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._chip_buttons.clear()

        if not self.state or not hasattr(self.state, "get_all_channel_groups"):
            return

        all_groups = self.state.get_all_channel_groups()
        current_set = self.state.get_shared_channel_set(self.get_active_set())
        active_group_name = current_set.get("group_name", "")

        for g_name, g_chans in all_groups.items():
            btn = QPushButton(f"{g_name} ({len(g_chans)}ch)", self.chip_container)
            btn.setCheckable(True)
            if g_name == active_group_name:
                btn.setChecked(True)
                btn.setStyleSheet("QPushButton { font-weight: bold; background-color: palette(highlight); color: palette(highlighted-text); }")
            btn.clicked.connect(lambda checked=False, name=g_name, chans=g_chans: self._on_chip_clicked(name, chans))
            self.chip_layout.insertWidget(self.chip_layout.count() - 1, btn)
            self._chip_buttons[g_name] = btn

    def _on_chip_clicked(self, group_name: str, channels: List[str]):
        if not self.state:
            return
        active_set = self.get_active_set()
        self.state.set_shared_channels(
            set_name=active_set,
            channels=channels,
            group_name=group_name,
            unit_idx=self.get_target_unit_idx(),
            multichannel=self.chk_multichannel.isChecked(),
        )
        self._update_chip_selection(group_name)

    def _update_chip_selection(self, active_group_name: str):
        for name, btn in self._chip_buttons.items():
            is_active = (name == active_group_name)
            btn.setChecked(is_active)
            if is_active:
                btn.setStyleSheet("QPushButton { font-weight: bold; background-color: palette(highlight); color: palette(highlighted-text); }")
            else:
                btn.setStyleSheet("")

    def update_active_set_display(self):
        if not self.state:
            self.lbl_badge.setText("(No State)")
            return

        active_set = self.get_active_set()
        s = self.state.get_shared_channel_set(active_set)
        chans = s.get("channels", [])
        if chans:
            ch_str = ", ".join(chans[:6]) + ("..." if len(chans) > 6 else "")
            self.lbl_badge.setText(f"{ch_str} ({len(chans)} channels)")
        else:
            self.lbl_badge.setText("(None)")

        # Target unit
        target_unit = s.get("unit_idx")
        self.combo_unit.blockSignals(True)
        if target_unit is None:
            self.combo_unit.setCurrentIndex(0)
        else:
            for idx in range(self.combo_unit.count()):
                if self.combo_unit.itemData(idx) == target_unit:
                    self.combo_unit.setCurrentIndex(idx)
                    break
        self.combo_unit.blockSignals(False)

        # Multichannel
        self.chk_multichannel.blockSignals(True)
        self.chk_multichannel.setChecked(bool(s.get("multichannel", True)))
        self.chk_multichannel.blockSignals(False)

        self._update_chip_selection(s.get("group_name", ""))

    def _on_set_changed(self, index: int):
        self.update_active_set_display()
        self.update_linked_viewers_label()

    def _on_shared_channel_set_changed(self, set_name: str):
        if set_name == self.get_active_set():
            self.update_active_set_display()
            self.update_linked_viewers_label()

    def _on_unit_selection_changed(self, index: int):
        if not self.state:
            return
        active_set = self.get_active_set()
        s = self.state.get_shared_channel_set(active_set)
        unit_idx = self.get_target_unit_idx()
        self.state.set_shared_channels(
            set_name=active_set,
            channels=s.get("channels", []),
            group_name=s.get("group_name", ""),
            unit_idx=unit_idx,
            multichannel=self.chk_multichannel.isChecked(),
        )

    def _on_multichannel_toggled(self, checked: bool):
        if not self.state:
            return
        active_set = self.get_active_set()
        s = self.state.get_shared_channel_set(active_set)
        self.state.set_shared_channels(
            set_name=active_set,
            channels=s.get("channels", []),
            group_name=s.get("group_name", ""),
            unit_idx=self.get_target_unit_idx(),
            multichannel=checked,
        )

    def _on_auto_split_clicked(self):
        if not self.state or not hasattr(self.state, "auto_create_channel_groups"):
            return
        size = self.spin_split.value()
        created = self.state.auto_create_channel_groups(group_size=size)
        self.refresh_chips()
        if created:
            # Activate first created group
            first_name = list(created.keys())[0]
            self._on_chip_clicked(first_name, created[first_name])

    def _cycle_group(self, direction: int):
        if not self.state or not hasattr(self.state, "cycle_shared_group"):
            return
        self.state.cycle_shared_group(self.get_active_set(), direction=direction)

    def _shift_channels(self, step: int):
        if not self.state or not hasattr(self.state, "shift_shared_channels"):
            return
        self.state.shift_shared_channels(self.get_active_set(), step=step)

    def _shift_by_group_size(self):
        if not self.state:
            return
        chans = self.get_current_shared_channels()
        span = len(chans) if chans else 1
        self._shift_channels(span)

    def _on_new_group(self):
        from .ChannelGroupBar import ChannelGroupDialog

        u_idx = self.get_target_unit_idx()
        dlg = ChannelGroupDialog(
            parent=self,
            state=self.state,
            initial_channels=self.get_current_shared_channels(),
            unit_idx=u_idx,
        )
        if dlg.exec() == QDialog.Accepted:
            selected = dlg.get_selected_channels()
            if selected and self.state:
                g_name = f"Custom ({len(selected)} ch: {selected[0]}..{selected[-1]})"
                self.state.add_channel_group(g_name, selected)
                self.refresh_chips()
                self._on_chip_clicked(g_name, selected)

    # -------------------------------------------------------------
    # Viewer Link Controls
    # -------------------------------------------------------------
    def get_top_window(self) -> Optional[QWidget]:
        p = self.parent()
        while p is not None:
            if hasattr(p, "dock_manager"):
                return p
            p = p.parent()
        for w in QApplication.topLevelWidgets():
            if hasattr(w, "dock_manager"):
                return w
        return None

    def get_open_viewers(self) -> List[Tuple[Any, QWidget]]:
        top = self.get_top_window()
        viewers = []
        if top and hasattr(top, "dock_manager"):
            for dw in top.dock_manager.dockWidgets():
                inner = dw.widget()
                if inner is not None and hasattr(inner, "channel_bar"):
                    viewers.append((dw, inner))
        return viewers

    def update_linked_viewers_label(self):
        active_set = self.get_active_set()
        viewers = self.get_open_viewers()
        linked_names = []
        for dw, inner in viewers:
            cb = inner.channel_bar
            mode, val = cb.get_channel_mode()
            if mode == cb.MODE_SHARED_SET and val == active_set:
                linked_names.append(dw.windowTitle() or type(inner).__name__)

        if linked_names:
            self.lbl_linked_viewers.setText(
                f"Linked Viewers ({len(linked_names)}): {', '.join(linked_names)}"
            )
        else:
            self.lbl_linked_viewers.setText(f"Linked Viewers: None linked to {active_set}")

    def link_all_open_viewers(self):
        active_set = self.get_active_set()
        viewers = self.get_open_viewers()
        for dw, inner in viewers:
            if hasattr(inner, "channel_bar"):
                inner.channel_bar.bind_to_shared_set(active_set)
        self.update_linked_viewers_label()

    # -------------------------------------------------------------
    # Settings persistence
    # -------------------------------------------------------------
    def get_settings(self) -> dict:
        return {"active_set": self.get_active_set()}

    def set_settings(self, settings: dict):
        if not settings:
            return
        if "active_set" in settings:
            idx = self.combo_set.findText(str(settings["active_set"]))
            if idx >= 0:
                self.combo_set.setCurrentIndex(idx)
