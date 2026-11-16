"""Per-plot manager for :class:`~spikespy.MultiTracePGView.MultiTracePGView`.

The manager provides one place to add, remove, reorder and configure every
plot in a multi-trace view instead of scattering those actions across the view
toolbar and right-click context menus.
"""

from typing import Any, Dict, List, Optional

from PySide6.QtCore import Qt, QSignalBlocker, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

WINDOW_GROUPS = [None, "Group A", "Group B", "Group C"]


class MultiTracePlotManager(QWidget):
    """List every plot in a multi-trace view and edit it in place."""

    viewChanged = Signal()

    def __init__(
        self, parent: Optional[QWidget] = None, view: Optional[QWidget] = None
    ):
        super().__init__(parent)
        self._view: Optional[QWidget] = None
        self._connections: List[Any] = []
        self._specs_by_key: Dict[str, Dict[str, Any]] = {}
        self._last_signature: Optional[tuple] = None
        self._updating = False

        self._build_ui()
        self.set_view(view)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        hint = QLabel("Select a plot to edit its parameters.")
        hint.setStyleSheet("color: #666; font-size: 11px;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.list_plots = QListWidget(self)
        self.list_plots.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list_plots.setAlternatingRowColors(True)
        self.list_plots.setMinimumHeight(120)
        self.list_plots.setMaximumHeight(180)
        self.list_plots.itemSelectionChanged.connect(self._on_selection_changed)
        self.list_plots.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self.list_plots)

        button_row = QHBoxLayout()
        self.btn_add = QToolButton(self)
        self.btn_add.setText("+ Add")
        self.btn_add.setToolTip("Add a channel plot or toggle a summary plot")
        self.btn_add.setPopupMode(QToolButton.InstantPopup)
        self._menu_add = QMenu(self)
        self._menu_add.aboutToShow.connect(self._populate_add_menu)
        self.btn_add.setMenu(self._menu_add)
        button_row.addWidget(self.btn_add)

        self.btn_remove = QPushButton("Remove", self)
        self.btn_remove.setToolTip("Remove the selected plot")
        self.btn_remove.clicked.connect(self._remove_selected)
        button_row.addWidget(self.btn_remove)

        self.btn_left = QPushButton("◀", self)
        self.btn_left.setFixedWidth(28)
        self.btn_left.setToolTip("Move the selected channel plot left")
        self.btn_left.clicked.connect(lambda: self._move_selected(-1))
        button_row.addWidget(self.btn_left)

        self.btn_right = QPushButton("▶", self)
        self.btn_right.setFixedWidth(28)
        self.btn_right.setToolTip("Move the selected channel plot right")
        self.btn_right.clicked.connect(lambda: self._move_selected(1))
        button_row.addWidget(self.btn_right)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        grp = QGroupBox("Plot Parameters", self)
        form = QFormLayout(grp)
        form.setContentsMargins(6, 4, 6, 4)
        form.setSpacing(4)

        self.lbl_plot_title = QLabel("No plot selected")
        self.lbl_plot_title.setStyleSheet("font-weight: bold;")
        self.lbl_plot_title.setWordWrap(True)
        form.addRow(self.lbl_plot_title)

        self.lbl_channel = QLabel("Channel:")
        self.combo_channel = QComboBox(grp)
        self.combo_channel.setToolTip("Channel shown by this plot")
        self.combo_channel.currentIndexChanged.connect(self._on_channel_changed)
        form.addRow(self.lbl_channel, self.combo_channel)

        self.lbl_visible = QLabel("Visible:")
        self.chk_visible = QCheckBox("Show summary graph", grp)
        self.chk_visible.toggled.connect(self._on_visible_toggled)
        form.addRow(self.lbl_visible, self.chk_visible)

        self.lbl_pin = QLabel("Pin to unit:")
        self.combo_pin = QComboBox(grp)
        self.combo_pin.setToolTip(
            "Center this plot's X axis on the selected unit's latency"
        )
        self.combo_pin.currentIndexChanged.connect(self._on_pin_changed)
        form.addRow(self.lbl_pin, self.combo_pin)

        self.lbl_group = QLabel("Window group:")
        self.combo_group = QComboBox(grp)
        self.combo_group.setToolTip(
            "Link zooming/panning with other plots in the same window group"
        )
        self.combo_group.currentIndexChanged.connect(self._on_group_changed)
        form.addRow(self.lbl_group, self.combo_group)

        self.lbl_width = QLabel("Window width:")
        self.spin_width = QDoubleSpinBox(grp)
        self.spin_width.setRange(0.5, 1000.0)
        self.spin_width.setDecimals(1)
        self.spin_width.setSingleStep(1.0)
        self.spin_width.setSuffix(" ms")
        self.spin_width.setToolTip("X-axis span used when pinned to a unit latency")
        self.spin_width.valueChanged.connect(self._on_width_changed)
        form.addRow(self.lbl_width, self.spin_width)

        self.lbl_link = QLabel("X axis:")
        self.chk_link = QCheckBox("Link X with other plots", grp)
        self.chk_link.toggled.connect(self._on_link_changed)
        form.addRow(self.lbl_link, self.chk_link)

        layout.addWidget(grp)
        layout.addStretch(1)

        self._set_detail_enabled(False)

    # ------------------------------------------------------------------
    # View binding
    # ------------------------------------------------------------------
    def set_view(self, view: Optional[QWidget]):
        if view is self._view:
            self.refresh()
            return
        self._disconnect_view()
        self._view = view
        self._last_signature = None
        if view is not None:
            self._connect_view(view)
        self.refresh()
        self.viewChanged.emit()

    def _connect_view(self, view: QWidget):
        plots_changed = getattr(view, "plotsChanged", None)
        if plots_changed is not None:
            try:
                plots_changed.connect(self.refresh)
                self._connections.append((plots_changed, self.refresh))
            except (TypeError, RuntimeError):
                pass
        bar = getattr(view, "channel_bar", None)
        binding_changed = (
            getattr(bar, "bindingChanged", None) if bar is not None else None
        )
        if binding_changed is not None:
            try:
                binding_changed.connect(self.refresh)
                self._connections.append((binding_changed, self.refresh))
            except (TypeError, RuntimeError):
                pass
        state = getattr(view, "state", None)
        unit_list_changed = (
            getattr(state, "onUnitListChanged", None) if state is not None else None
        )
        if unit_list_changed is not None:
            try:
                unit_list_changed.connect(self.refresh)
                self._connections.append((unit_list_changed, self.refresh))
            except (TypeError, RuntimeError):
                pass

    def _disconnect_view(self):
        for sig, slot in self._connections:
            try:
                sig.disconnect(slot)
            except (TypeError, RuntimeError):
                pass
        self._connections.clear()

    def _safe_view(self) -> Optional[QWidget]:
        view = self._view
        if view is None:
            return None
        try:
            _ = view.objectName()
        except RuntimeError:
            self._view = None
            return None
        return view

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------
    def refresh(self):
        view = self._safe_view()
        if view is None or not hasattr(view, "get_plot_specs"):
            self._specs_by_key.clear()
            self._last_signature = None
            self._updating = True
            try:
                self.list_plots.clear()
            finally:
                self._updating = False
            self._set_detail_enabled(False)
            return

        try:
            specs: List[Dict[str, Any]] = list(view.get_plot_specs())
        except Exception:
            return

        signature = tuple(
            (
                spec.get("kind"),
                spec.get("key"),
                spec.get("label"),
                spec.get("visible"),
                spec.get("channel_name"),
                spec.get("pinned_unit_idx"),
                spec.get("window_group"),
                spec.get("window_width_ms"),
                spec.get("x_linked"),
            )
            for spec in specs
        )
        if signature == self._last_signature:
            return
        self._last_signature = signature

        current_key = self._selected_key()
        self._specs_by_key = {spec["key"]: spec for spec in specs}

        self._updating = True
        try:
            with QSignalBlocker(self.list_plots):
                self.list_plots.clear()
                for spec in specs:
                    item = QListWidgetItem(self._format_item_text(spec))
                    item.setData(Qt.UserRole, spec["key"])
                    item.setToolTip(self._format_item_tooltip(spec))
                    if spec.get("kind") == "summary":
                        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                        item.setCheckState(
                            Qt.Checked if spec.get("visible", True) else Qt.Unchecked
                        )
                    self.list_plots.addItem(item)
        finally:
            self._updating = False

        self._restore_selection(current_key)
        self._populate_detail()

    def _format_item_text(self, spec: Dict[str, Any]) -> str:
        label = spec.get("label") or "(unnamed)"
        if spec.get("kind") == "channel":
            return label
        return f"Summary: {label}"

    def _format_item_tooltip(self, spec: Dict[str, Any]) -> str:
        if spec.get("kind") == "channel":
            return "Channel plot — right-click or use the fields below to configure it."
        if spec.get("is_channel_stat"):
            return "Per-channel summary line graph (Mean/Max over time)."
        return "Global summary line graph."

    def _selected_key(self) -> Optional[str]:
        item = self.list_plots.currentItem()
        return item.data(Qt.UserRole) if item is not None else None

    def _restore_selection(self, key: Optional[str]):
        row = -1
        if key is not None:
            for i in range(self.list_plots.count()):
                if self.list_plots.item(i).data(Qt.UserRole) == key:
                    row = i
                    break
        if row < 0 and self.list_plots.count() > 0:
            row = 0
        if row >= 0:
            self.list_plots.setCurrentRow(row)

    def _selected_spec(self) -> Optional[Dict[str, Any]]:
        key = self._selected_key()
        return self._specs_by_key.get(key) if key is not None else None

    def select_plot(self, key: str):
        """Select a plot row by its spec key (e.g. ``channel:1``)."""
        self.refresh()
        for i in range(self.list_plots.count()):
            if self.list_plots.item(i).data(Qt.UserRole) == key:
                self.list_plots.setCurrentRow(i)
                return

    # ------------------------------------------------------------------
    # Detail editor
    # ------------------------------------------------------------------
    def _populate_detail(self):
        spec = self._selected_spec()
        self._updating = True
        try:
            if spec is None:
                self.lbl_plot_title.setText("No plot selected")
                self._set_detail_enabled(False)
                return

            self.lbl_plot_title.setText(spec.get("label") or "(unnamed plot)")
            self._set_detail_enabled(True)
            is_channel = spec.get("kind") == "channel"
            is_channel_stat = bool(spec.get("is_channel_stat"))

            self._populate_channel_combo(spec, allow_edit=is_channel or is_channel_stat)
            self._set_row_visible(self.lbl_visible, self.chk_visible, not is_channel)
            self.chk_visible.setChecked(bool(spec.get("visible", True)))
            self.chk_visible.setText(
                "Show channel summary" if is_channel_stat else "Show summary graph"
            )

            show_slot_options = is_channel
            self._set_row_visible(self.lbl_pin, self.combo_pin, show_slot_options)
            self._set_row_visible(self.lbl_group, self.combo_group, show_slot_options)
            self._set_row_visible(self.lbl_width, self.spin_width, show_slot_options)
            self._set_row_visible(self.lbl_link, self.chk_link, show_slot_options)

            if show_slot_options:
                self._populate_pin_combo(spec.get("pinned_unit_idx"))
                self._populate_group_combo(spec.get("window_group"))
                self.spin_width.setValue(float(spec.get("window_width_ms", 20.0)))
                self.chk_link.setChecked(bool(spec.get("x_linked", True)))
            else:
                self._populate_pin_combo(None)
                self._populate_group_combo(None)
                self.chk_link.setChecked(True)

            self.btn_remove.setEnabled(
                not (is_channel and self._channel_plot_count() <= 1)
            )
            movable = (
                spec.get("slot_idx") is not None and self.list_plots.currentRow() >= 0
            )
            self.btn_left.setEnabled(movable and self._can_move(spec, -1))
            self.btn_right.setEnabled(movable and self._can_move(spec, 1))
        finally:
            self._updating = False

    def _set_detail_enabled(self, enabled: bool):
        for w in (
            self.lbl_channel,
            self.combo_channel,
            self.lbl_visible,
            self.chk_visible,
            self.lbl_pin,
            self.combo_pin,
            self.lbl_group,
            self.combo_group,
            self.lbl_width,
            self.spin_width,
            self.lbl_link,
            self.chk_link,
        ):
            w.setEnabled(enabled)
        self.btn_remove.setEnabled(enabled)
        self.btn_left.setEnabled(False)
        self.btn_right.setEnabled(False)

    @staticmethod
    def _set_row_visible(label: QLabel, field: QWidget, visible: bool):
        label.setVisible(visible)
        field.setVisible(visible)

    def _populate_channel_combo(self, spec: Dict[str, Any], allow_edit: bool):
        self.combo_channel.clear()
        self.combo_channel.addItem("(Empty Screen)", userData=None)
        view = self._safe_view()
        channels: List[str] = []
        if view is not None and getattr(view, "state", None) is not None:
            channels = list(view.state.get_available_channels())
        current = spec.get("channel_name")
        if current and current not in channels:
            channels.append(current)
        for ch in channels:
            self.combo_channel.addItem(ch, userData=ch)
        idx = self.combo_channel.findData(current)
        if idx < 0:
            idx = 0
        self.combo_channel.setCurrentIndex(idx)
        self.combo_channel.setEnabled(
            bool(allow_edit) and self.combo_channel.count() > 1
        )

    def _populate_pin_combo(self, current: Optional[int]):
        self.combo_pin.clear()
        self.combo_pin.addItem("None (free X axis)", userData=None)
        view = self._safe_view()
        state = getattr(view, "state", None) if view is not None else None
        units = getattr(state, "spike_groups", None) or []
        for i, sg in enumerate(units):
            name = getattr(getattr(sg, "event", None), "name", None) or f"unit_{i}"
            self.combo_pin.addItem(f"Unit {i}: {name}", userData=i)
        idx = self.combo_pin.findData(current)
        if idx < 0:
            if current is not None:
                self.combo_pin.addItem(f"Unit {current}", userData=current)
                idx = self.combo_pin.count() - 1
            else:
                idx = 0
        self.combo_pin.setCurrentIndex(idx)

    def _populate_group_combo(self, current: Optional[str]):
        self.combo_group.clear()
        for group in WINDOW_GROUPS:
            self.combo_group.addItem(
                "None (independent)" if group is None else group, userData=group
            )
        idx = self.combo_group.findData(current)
        if idx < 0:
            if current is not None:
                self.combo_group.addItem(current, userData=current)
                idx = self.combo_group.count() - 1
            else:
                idx = 0
        self.combo_group.setCurrentIndex(idx)

    def _channel_plot_count(self) -> int:
        view = self._safe_view()
        return len(getattr(view, "channel_panels", [])) if view is not None else 0

    def _can_move(self, spec: Dict[str, Any], delta: int) -> bool:
        slot = spec.get("slot_idx")
        if slot is None:
            return False
        target = int(slot) + int(delta)
        return 0 <= target < self._channel_plot_count()

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _populate_add_menu(self):
        menu = self._menu_add
        menu.clear()
        view = self._safe_view()
        if view is None:
            act = menu.addAction("No view selected")
            act.setEnabled(False)
            return

        avail = []
        if hasattr(view, "get_available_plot_channels"):
            try:
                avail = list(view.get_available_plot_channels())
            except Exception:
                avail = []
        if avail:
            header = menu.addAction("Add Channel Plot:")
            header.setEnabled(False)
            for ch in avail:
                act = menu.addAction(ch)
                act.triggered.connect(
                    lambda checked=False, c=ch: view.add_channel_to_display(c)
                )
        else:
            act = menu.addAction("All channels already shown")
            act.setEnabled(False)

        act_empty = menu.addAction("Add Empty Screen")
        act_empty.triggered.connect(view.add_empty_slot)

        menu.addSeparator()
        summary_header = menu.addAction("Summary Plots:")
        summary_header.setEnabled(False)

        if hasattr(view, "get_summary_plot_availability"):
            try:
                availability = dict(view.get_summary_plot_availability())
            except Exception:
                availability = {}
            if "__channel_summaries__" in availability:
                act = menu.addAction("Channel Summary Graphs")
                act.setCheckable(True)
                act.setChecked(bool(availability["__channel_summaries__"]))
                act.triggered.connect(
                    lambda checked: view.set_plot_visible(
                        "__channel_summaries__", checked
                    )
                )
            labels = {
                "Stim. Freq. (Hz)": "Stimulus Frequency",
                "Latency Diff.": "Latency Difference",
            }
            for key, enabled in availability.items():
                if key == "__channel_summaries__":
                    continue
                act = menu.addAction(labels.get(key, key))
                act.setCheckable(True)
                act.setChecked(bool(enabled))
                act.triggered.connect(
                    lambda checked=False, k=key: view.set_plot_visible(k, checked)
                )

    def _remove_selected(self):
        view = self._safe_view()
        spec = self._selected_spec()
        if view is None or spec is None:
            return
        if spec.get("kind") == "channel":
            if self._channel_plot_count() <= 1:
                return
            view.remove_plot_slot(int(spec["slot_idx"]))
        else:
            view.set_plot_visible(spec["key"], False)

    def _move_selected(self, delta: int):
        view = self._safe_view()
        spec = self._selected_spec()
        if view is None or spec is None or spec.get("slot_idx") is None:
            return
        if not self._can_move(spec, delta):
            return
        key = spec["key"]
        view.move_plot_slot(int(spec["slot_idx"]), delta)
        # The slot-based key follows the moved plot; keep it selected.
        self._last_signature = None
        self.refresh()
        self._select_key_after_move(key, int(spec["slot_idx"]) + delta)

    def _select_key_after_move(self, key: str, new_slot: int):
        wanted = key
        if key.startswith("channel:"):
            wanted = f"channel:{new_slot}"
        for i in range(self.list_plots.count()):
            if self.list_plots.item(i).data(Qt.UserRole) == wanted:
                self.list_plots.setCurrentRow(i)
                return

    def _on_selection_changed(self):
        if self._updating:
            return
        self._populate_detail()

    def _on_item_changed(self, item: QListWidgetItem):
        if self._updating:
            return
        spec = self._specs_by_key.get(item.data(Qt.UserRole))
        view = self._safe_view()
        if spec is None or view is None or spec.get("kind") != "summary":
            return
        checked = item.checkState() == Qt.Checked
        if checked != bool(spec.get("visible", True)):
            view.set_plot_visible(spec["key"], checked)

    def _on_channel_changed(self, index: int):
        if self._updating:
            return
        spec = self._selected_spec()
        view = self._safe_view()
        if spec is None or view is None or spec.get("slot_idx") is None:
            return
        channel = self.combo_channel.itemData(index)
        view.set_plot_slot_channel(int(spec["slot_idx"]), channel)

    def _on_visible_toggled(self, checked: bool):
        if self._updating:
            return
        spec = self._selected_spec()
        view = self._safe_view()
        if spec is None or view is None or spec.get("kind") == "channel":
            return
        if checked != bool(spec.get("visible", True)):
            view.set_plot_visible(spec["key"], checked)

    def _on_pin_changed(self, index: int):
        if self._updating:
            return
        spec = self._selected_spec()
        view = self._safe_view()
        if spec is None or view is None or spec.get("kind") != "channel":
            return
        unit_idx = self.combo_pin.itemData(index)
        view.set_plot_slot_options(int(spec["slot_idx"]), pinned_unit_idx=unit_idx)

    def _on_group_changed(self, index: int):
        if self._updating:
            return
        spec = self._selected_spec()
        view = self._safe_view()
        if spec is None or view is None or spec.get("kind") != "channel":
            return
        group = self.combo_group.itemData(index)
        view.set_plot_slot_options(int(spec["slot_idx"]), window_group=group)

    def _on_width_changed(self, value: float):
        if self._updating:
            return
        spec = self._selected_spec()
        view = self._safe_view()
        if spec is None or view is None or spec.get("kind") != "channel":
            return
        view.set_plot_slot_options(int(spec["slot_idx"]), window_width_ms=float(value))

    def _on_link_changed(self, checked: bool):
        if self._updating:
            return
        spec = self._selected_spec()
        view = self._safe_view()
        if spec is None or view is None or spec.get("kind") != "channel":
            return
        view.set_plot_slot_options(int(spec["slot_idx"]), x_linked=bool(checked))
