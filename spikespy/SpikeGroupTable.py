from typing import Any, List, Optional, Tuple, Union

import PySide6


from PySide6.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    QObject,
    QSignalBlocker,
    Qt,
    Signal,
    Slot,
    QTimer,
)
from PySide6.QtGui import QAction, QColor, QFont, QKeySequence, QPalette, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMdiArea,
    QMdiSubWindow,
    QMenu,
    QMenuBar,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableView,
    QVBoxLayout,
    QWidget,
    QStyledItemDelegate,
)

from .ViewerState import ViewerState
from .live_orchestrator import LiveFrameConsumerMixin
from .ChannelGroupBar import ChannelGroupDialog


SET2_COLORS = [
    (102, 194, 165),
    (252, 141, 98),
    (141, 160, 203),
    (231, 138, 195),
    (166, 216, 84),
    (255, 217, 47),
    (229, 196, 148),
    (179, 179, 179),
]


class SpikeGroupTableView(LiveFrameConsumerMixin, QWidget):
    def __init__(
        self,
        parent: Optional[PySide6.QtWidgets.QWidget] = None,
        state: ViewerState = None,
    ) -> None:
        super().__init__(parent)
        self.render_count: int = 0
        self._dirty: bool = False
        self._user_hidden: bool = False
        self._state_connections: list[tuple] = []
        self._last_unit_change_sync_token: Optional[tuple] = None

        self.update_timer = QTimer(self)
        interval = (
            state.get_ui_refresh_rate_ms()
            if state and hasattr(state, "get_ui_refresh_rate_ms")
            else 33
        )
        self.update_timer.setInterval(interval)
        self.update_timer.timeout.connect(self._on_render_timer_tick)
        self.update_timer.start()

        if state and hasattr(state, "onUIRefreshRateChange"):
            self._state_connections.append((state.onUIRefreshRateChange, self.update_timer.setInterval))
            state.onUIRefreshRateChange.connect(self.update_timer.setInterval)

        self.spike_tablemodel = SpikeGroupTableModel(
            lambda: state.spike_groups if state is not None else [],
            candidates_func=(
                (lambda: state.get_visible_candidates())
                if state is not None and hasattr(state, "get_visible_candidates")
                else None
            ),
            state=state,
        )

        self.tbl = SpikeGroupTableView2(self)
        self.tbl.setModel(self.spike_tablemodel)
        self.state = state
        if state is not None:
            self._state_connections.append((state.onUnitChange, self._on_unit_change))
            state.onUnitChange.connect(self._on_unit_change)
            self._state_connections.append((state.onUnitGroupChange, self.update))
            state.onUnitGroupChange.connect(self.update)
            if hasattr(state, "onUnitParamsChanged"):
                self._state_connections.append((state.onUnitParamsChanged, self._on_unit_params_changed))
                state.onUnitParamsChanged.connect(self._on_unit_params_changed)
            if hasattr(state, "onLoadNewFile"):
                self._register_data_reload(state, self.update)
            if hasattr(state, "onNewSpikeEvents"):
                self._state_connections.append((state.onNewSpikeEvents, self._on_new_spike_events))
                state.onNewSpikeEvents.connect(self._on_new_spike_events)
            if hasattr(state, "onNewSweep"):
                self._register_live_frames(
                    state,
                    lambda _stimno: self.update(),
                )
            if hasattr(state, "onCandidatesChanged"):
                self._state_connections.append(
                    (state.onCandidatesChanged, self._on_candidates_changed)
                )
                state.onCandidatesChanged.connect(self._on_candidates_changed)
            if hasattr(state, "onCandidateSelectionChanged"):
                self._state_connections.append(
                    (
                        state.onCandidateSelectionChanged,
                        self._sync_candidate_selection,
                    )
                )
                state.onCandidateSelectionChanged.connect(
                    self._sync_candidate_selection
                )
            if hasattr(state, "onCandidateScanStateChanged"):
                self._state_connections.append(
                    (
                        state.onCandidateScanStateChanged,
                        self._on_candidate_scan_state,
                    )
                )
                state.onCandidateScanStateChanged.connect(
                    self._on_candidate_scan_state
                )

        self.tbl.setSelectionBehavior(QTableView.SelectRows)
        # self.tbl.setSelectionMode(QAbstractItemView.SingleSelection)
        # self.tbl.setStyleSheet("QTableView::item:selected { background: #a0c4ff; }")

        # Context menu and double-click actions on table
        self.tbl.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tbl.customContextMenuRequested.connect(self.on_table_context_menu)
        self.tbl.doubleClicked.connect(self.on_cell_double_clicked)

        # Use standard icons for add/remove if available
        style = self.style()
        add_icon = style.standardIcon(PySide6.QtWidgets.QStyle.SP_FileDialogNewFolder)
        remove_icon = style.standardIcon(PySide6.QtWidgets.QStyle.SP_TrashIcon)

        # Place add/remove buttons next to each other, aligned to the right, with standard add/delete icons
        button_layout = QHBoxLayout()
        self.scanButton = QPushButton("Scan")
        self.scanButton.setToolTip("Search the recording for candidate units")
        self.scanButton.clicked.connect(self._on_scan_clicked)
        self.autoButton = QPushButton("Auto")
        self.autoButton.setCheckable(True)
        self.autoButton.setToolTip("Periodically scan recent sweeps while streaming")
        self.autoButton.toggled.connect(self._on_auto_toggled)
        self.proposalConfigButton = QPushButton("⚙ Proposal Config...")
        self.proposalConfigButton.setToolTip(
            "Configure spike proposal algorithm, detector parameters, and search settings"
        )
        self.proposalConfigButton.clicked.connect(self._open_proposal_config_view)
        self.candidateStatusLabel = QLabel("")
        self.candidateStatusLabel.setContentsMargins(4, 0, 4, 0)
        button_layout.addWidget(self.scanButton)
        button_layout.addWidget(self.autoButton)
        button_layout.addWidget(self.proposalConfigButton)
        button_layout.addWidget(self.candidateStatusLabel)
        button_layout.addStretch(1)  # Push buttons to the right
        self.acceptCandidateButton = QPushButton("✓ Accept Candidate")
        self.acceptCandidateButton.setToolTip(
            "Accept the selected candidate into a tracked unit (Ctrl+Shift+A)"
        )
        self.acceptCandidateButton.setStyleSheet(
            "QPushButton { background-color: #2b7a2b; color: white; font-weight: bold; "
            "padding: 3px 10px; border-radius: 4px; border: 1px solid #1f5c1f; }"
            "QPushButton:hover { background-color: #236623; }"
            "QPushButton:disabled { background-color: #cbd5e1; color: #64748b; border-color: #94a3b8; }"
        )
        self.acceptCandidateButton.clicked.connect(self._accept_selected_candidate)
        self.discardCandidateButton = QPushButton("✕ Discard")
        self.discardCandidateButton.setToolTip(
            "Discard the selected candidate (Ctrl+Shift+D)"
        )
        self.discardCandidateButton.setStyleSheet(
            "QPushButton { background-color: #f1f5f9; color: #475569; "
            "padding: 3px 10px; border-radius: 4px; border: 1px solid #cbd5e1; }"
            "QPushButton:hover { background-color: #e2e8f0; color: #1e293b; }"
            "QPushButton:disabled { background-color: #f8fafc; color: #94a3b8; border-color: #e2e8f0; }"
        )
        self.discardCandidateButton.clicked.connect(self._discard_selected_candidate)
        button_layout.addWidget(self.acceptCandidateButton)
        button_layout.addWidget(self.discardCandidateButton)
        self.addButton = QPushButton()
        self.addButton.setIcon(add_icon)
        self.addButton.setToolTip("Add a new event/group")
        if self.state is not None:
            self.addButton.clicked.connect(self.state.addUnitGroup)
        self.removeButton = QPushButton()
        self.removeButton.setIcon(remove_icon)
        self.removeButton.setToolTip("Delete the selected event/group")
        self.removeButton.clicked.connect(self.remove_selected_row)
        button_layout.addWidget(self.removeButton)
        button_layout.addWidget(self.addButton)

        vlayout = QVBoxLayout()
        vlayout.addWidget(self.tbl)
        vlayout.addLayout(button_layout)
        self.setLayout(vlayout)

        self.tbl.selectionModel().selectionChanged.connect(self.set_selection)
        self.tbl.setItemDelegateForColumn(
            4, ComboBoxDelegate(self, ["cm", "ch", "cmh", "cmihi", "cmih"])
        )

        self.addButton.clicked.disconnect()
        self.addButton.clicked.connect(self._on_add_clicked)
        self.removeButton.clicked.disconnect()
        self.removeButton.clicked.connect(self._on_remove_clicked)
        self.spike_tablemodel.dataChanged.connect(self._on_model_data_changed)

        self._accept_shortcut = QShortcut(QKeySequence("Ctrl+Shift+A"), self)
        self._accept_shortcut.activated.connect(self._accept_selected_candidate)
        self._discard_shortcut = QShortcut(QKeySequence("Ctrl+Shift+D"), self)
        self._discard_shortcut.activated.connect(self._discard_selected_candidate)

        self._manual_scan_select_pending = False
        self._update_action_buttons()
        self.update()

    def _on_model_data_changed(self, *args):
        if getattr(self.spike_tablemodel, "_in_model_update", False) or self.state is None:
            return
        index = args[0] if args and hasattr(args[0], "row") else None
        row = index.row() if index is not None else -1
        column = index.column() if index is not None else -1
        item = self.spike_tablemodel.row_to_item(row) if row >= 0 else None
        if item is None:
            return
        if item[0] == "candidate":
            self._update_action_buttons()
            return
        unit_idx = item[1]
        if 0 <= unit_idx < len(getattr(self.state, "spike_groups", [])):
            sg = self.state.spike_groups[unit_idx]
            if column in (8, 9, 10):
                # Tracking parameters changed from the table: drop any cached
                # convolution template and notify the tracking view/views.
                sg._template_matcher = None
                if (
                    column == 8
                    and getattr(sg, "auto_track", False)
                    and getattr(self.state, "is_streaming", False)
                ):
                    try:
                        self.state.ensure_live_tracking_processor()
                    except Exception:
                        pass
                if hasattr(self.state, "onUnitParamsChanged"):
                    self.state.onUnitParamsChanged.emit(unit_idx)
        self.state.onUnitGroupChange.emit()

    def on_table_context_menu(self, pos):
        index = self.tbl.indexAt(pos)
        if not index.isValid():
            return
        row = index.row()
        item = self.spike_tablemodel.row_to_item(row)
        if item is None:
            return
        if item[0] == "candidate":
            self._show_candidate_context_menu(item[1], pos)
            return
        unit_idx = int(item[1])
        sg = self.tbl.model().data(self.tbl.model().index(row, 0), role="SPIKEGROUP_ROLE")
        if sg is None:
            return
        ch = getattr(sg, "channel", "")
        if not ch and hasattr(sg.event, "annotations"):
            ch = sg.event.annotations.get("channel", "")

        menu = QMenu(self)
        if ch and hasattr(self.state, "set_active_channel"):
            act_focus = menu.addAction(f"Focus Primary Channel '{ch}'")
            act_focus.triggered.connect(lambda: self.state.set_active_channel(ch))

        menu.addSeparator()
        act_edit_chans = menu.addAction(f"Configure Channels for Unit {unit_idx}...")
        act_edit_chans.triggered.connect(
            lambda: self._open_channel_dialog_for_unit(unit_idx)
        )

        menu_shift = menu.addMenu("Shift Unit Channel Group")
        menu_shift.addAction(
            "Shift -1 Channel", lambda: self.state.shift_unit_channels(unit_idx, step=-1)
        )
        menu_shift.addAction(
            "Shift +1 Channel", lambda: self.state.shift_unit_channels(unit_idx, step=1)
        )
        menu_shift.addAction(
            "Shift -4 Channels (Tetrode)",
            lambda: self.state.shift_unit_channels(unit_idx, step=-4),
        )
        menu_shift.addAction(
            "Shift +4 Channels (Tetrode)",
            lambda: self.state.shift_unit_channels(unit_idx, step=4),
        )

        merge_items = self._merge_targets(unit_idx)
        if merge_items:
            menu.addSeparator()
            menu_merge = menu.addMenu(f"Merge Unit {unit_idx} Into...")
            for other_idx, label in merge_items:
                menu_merge.addAction(
                    label,
                    lambda checked=False, keep=other_idx, rm=unit_idx: self._merge_units(
                        keep, rm
                    ),
                )

        menu.addSeparator()
        menu_track = menu.addMenu(f"Tracking for Unit {unit_idx}")
        menu_track.addAction(
            "Fill Missing Spikes",
            lambda checked=False, r=unit_idx: self._fill_missing_for_unit(r),
        )
        menu_track.addAction(
            "Rebuild Convolution Template",
            lambda checked=False, r=unit_idx: self._rebuild_unit_template(r),
        )
        menu_track.addAction(
            "Re-Anchor Active Unit at Current Sweep",
            lambda checked=False, r=unit_idx: self._reanchor_unit_at_current_sweep(r),
        )
        menu_track.addAction(
            "Open Tracking View",
            lambda checked=False, r=unit_idx: self._open_tracking_view_for_unit(r),
        )
        menu_track.addAction(
            "Open Active Unit Info",
            lambda checked=False, r=unit_idx: self._open_active_unit_info(r),
        )
        auto_label = (
            "Disable Live Auto-Track"
            if getattr(sg, "auto_track", False)
            else "Enable Live Auto-Track"
        )
        menu_track.addAction(
            auto_label,
            lambda checked=False, r=unit_idx: self._toggle_live_tracking(r),
        )

        menu.addSeparator()
        menu_open = menu.addMenu(f"Open Pinned Window for Unit {unit_idx}")
        menu_open.addAction(
            "MultiTrace View", lambda: self._open_pinned_window("MultiTrace", unit_idx)
        )
        menu_open.addAction(
            "SingleTrace View",
            lambda: self._open_pinned_window("SingleTraceView", unit_idx),
        )
        menu_open.addAction(
            "UnitView", lambda: self._open_pinned_window("UnitView", unit_idx)
        )
        menu_open.addAction(
            "MultiTraceFixed View",
            lambda: self._open_pinned_window("MultiTraceFixedView", unit_idx),
        )

        menu.exec(self.tbl.viewport().mapToGlobal(pos))

    def _show_candidate_context_menu(self, candidate_id: int, pos):
        if self.state is None:
            return
        menu = QMenu(self)
        menu.addAction(
            "Accept Candidate",
            lambda checked=False, cid=candidate_id: self.state.accept_candidate(cid),
        )
        menu.addAction(
            "Discard Candidate",
            lambda checked=False, cid=candidate_id: self.state.discard_candidate(cid),
        )
        menu.addAction(
            "Select for Preview",
            lambda checked=False, cid=candidate_id: self.state.set_active_candidate(cid),
        )
        menu.addSeparator()
        menu.addAction("Rescan Discarded Candidates", self._rescan_discarded_candidates)
        menu.exec(self.tbl.viewport().mapToGlobal(pos))

    def _fill_missing_for_unit(self, unit_idx: int) -> int:
        """Run this unit's tracker across the recording gaps between spikes."""
        if self.state is None:
            return 0
        added = self.state.track_unit_range(unit_idx, only_missing=True)
        self._last_fill_count = len(added)
        self.update()
        return len(added)

    def _rebuild_unit_template(self, unit_idx: int) -> bool:
        if self.state is None:
            return False
        ok = self.state.rebuild_unit_template(unit_idx)
        self.update()
        return ok

    def _open_tracking_view_for_unit(self, unit_idx: int):
        if self.state is None:
            return
        self.state.setUnitGroup(unit_idx)
        top_win = self.window()
        if hasattr(top_win, "newWindow"):
            top_win.newWindow("TrackingView")

    def _open_active_unit_info(self, unit_idx: int):
        if self.state is None:
            return
        self.state.setUnitGroup(unit_idx)
        top_win = self.window()
        if hasattr(top_win, "newWindow"):
            top_win.newWindow("ActiveUnitInfo")

    def _reanchor_unit_at_current_sweep(self, unit_idx: int) -> Optional[int]:
        if self.state is None or not (
            0 <= unit_idx < len(getattr(self.state, "spike_groups", []))
        ):
            return None
        import numpy as np
        from scipy.signal import find_peaks

        sg = self.state.spike_groups[unit_idx]
        stimno = int(getattr(self.state, "stimno", 0) or 0)

        ch = (
            getattr(sg, "tracking_channel", "")
            or getattr(sg, "channel", "")
            or (
                sg.event.annotations.get("channel", "")
                if getattr(sg, "event", None) is not None
                and getattr(sg.event, "annotations", None)
                else ""
            )
        )
        erp = None
        if ch and hasattr(self.state, "get_channel_erp"):
            try:
                erp = self.state.get_channel_erp(ch)
            except Exception:
                erp = None
        if erp is None and hasattr(self.state, "get_erp"):
            try:
                erp = self.state.get_erp()
            except Exception:
                erp = None

        sr_raw = getattr(self.state, "sampling_rate", None)
        if sr_raw is None and getattr(self.state, "analog_signal", None) is not None:
            sr_raw = getattr(self.state.analog_signal, "sampling_rate", None)
        try:
            sr_hz = (
                float(sr_raw.rescale("Hz").magnitude)
                if sr_raw is not None and hasattr(sr_raw, "rescale")
                else (float(sr_raw) if sr_raw is not None else 10000.0)
            )
        except Exception:
            sr_hz = 10000.0

        ref_sample: Optional[int] = None
        idx_arr = getattr(sg, "idx_arr", None)
        if idx_arr is not None and len(idx_arr) > 0:
            if 0 <= stimno < len(idx_arr) and idx_arr[stimno] is not None:
                val = idx_arr[stimno]
                ref_sample = int(val[0] if isinstance(val, (tuple, list)) else val)
            else:
                for prev_s in range(min(stimno - 1, len(idx_arr) - 1), -1, -1):
                    if idx_arr[prev_s] is not None:
                        val = idx_arr[prev_s]
                        ref_sample = int(
                            val[0] if isinstance(val, (tuple, list)) else val
                        )
                        break

        sample: Optional[int] = None
        if erp is not None and 0 <= stimno < len(erp):
            row_wf = erp[stimno]
            wf = np.asarray(
                row_wf.magnitude if hasattr(row_wf, "magnitude") else row_wf,
                dtype=np.float64,
            ).ravel()
            if wf.size > 0:
                pts_up, _ = find_peaks(wf)
                pts_dn, _ = find_peaks(-wf)
                pts = np.sort(np.hstack([pts_up, pts_dn]).astype(np.int64))
                if ref_sample is not None:
                    win_ms = float(getattr(sg, "tracking_window_ms", 5.0) or 5.0)
                    search_rad = max(int(round((win_ms / 1000.0) * sr_hz)) * 2, 10)
                    lo = max(0, ref_sample - search_rad)
                    hi = min(wf.size - 1, ref_sample + search_rad)
                    in_win = pts[(pts >= lo) & (pts <= hi)]
                    if in_win.size > 0:
                        sample = int(in_win[np.argmax(np.abs(wf[in_win]))])
                    else:
                        sample = int(lo + np.argmax(np.abs(wf[lo : hi + 1])))
                elif pts.size > 0:
                    sample = int(pts[np.argmax(np.abs(wf[pts]))])
                else:
                    sample = int(np.argmax(np.abs(wf)))

        if sample is None:
            sample = ref_sample
        if sample is None:
            return None

        if hasattr(self.state, "reanchor_unit_at"):
            self.state.reanchor_unit_at(unit_idx, int(sample))
        else:
            sg._consecutive_misses = 0
            self.state.setUnit(int(sample), stimno=stimno, unit_idx=unit_idx)
        self.update()
        return int(sample)

    def _toggle_live_tracking(self, unit_idx: int):
        if self.state is None or not (0 <= unit_idx < len(self.state.spike_groups)):
            return
        sg = self.state.spike_groups[unit_idx]
        sg.auto_track = not bool(getattr(sg, "auto_track", False))
        if hasattr(sg, "event") and hasattr(sg.event, "annotations"):
            sg.event.annotations["auto_track"] = sg.auto_track
        if sg.auto_track and getattr(self.state, "is_streaming", False):
            self.state.ensure_live_tracking_processor()
        self.state.onUnitParamsChanged.emit(unit_idx)
        self.update()

    def _open_channel_dialog_for_unit(self, unit_idx: int):
        dlg = ChannelGroupDialog(parent=self, state=self.state, unit_idx=unit_idx)
        if dlg.exec() == QDialog.Accepted:
            selected = dlg.get_selected_channels()
            if selected:
                self.state.set_unit_channels(unit_idx, selected)
                self.update()

    def _unit_label(self, unit_idx: int) -> str:
        groups = getattr(self.state, "spike_groups", None) or []
        if not (0 <= unit_idx < len(groups)):
            return f"unit {unit_idx}"
        sg = groups[unit_idx]
        name = getattr(getattr(sg, "event", None), "name", None) or f"unit_{unit_idx}"
        n_spikes = 0
        if getattr(sg, "event", None) is not None:
            try:
                n_spikes = len(sg.event)
            except Exception:
                n_spikes = 0
        return f"{name} ({n_spikes} spikes)"

    def _merge_targets(self, exclude_idx: int) -> List[Tuple[int, str]]:
        """(unit index, display label) for every unit that can absorb ``exclude_idx``."""
        if self.state is None or not hasattr(self.state, "mergeUnitGroups"):
            return []
        groups = getattr(self.state, "spike_groups", None) or []
        return [
            (i, self._unit_label(i)) for i in range(len(groups)) if i != exclude_idx
        ]

    def _merge_units(self, keep_idx: int, remove_idx: int):
        if self.state is None or not hasattr(self.state, "mergeUnitGroups"):
            return
        self.state.mergeUnitGroups(keep_idx, remove_idx)
        self.update()

    def _open_pinned_window(self, window_key: str, unit_idx: int):
        top_win = self.window()
        if hasattr(top_win, "newWindow"):
            top_win.newWindow(window_key, target_unit=unit_idx)

    def on_cell_double_clicked(self, index: QModelIndex):
        if not index.isValid():
            return
        row = index.row()
        item = self.spike_tablemodel.row_to_item(row)
        if item is None:
            return
        if item[0] == "candidate":
            self.state.set_active_candidate(int(item[1]))
            return
        # If user double-clicks the channel column (column 6)
        if index.column() == 6:
            sg = self.state.spike_groups[int(item[1])]
            ch = getattr(sg, "channel", "")
            if not ch and hasattr(sg, "event") and getattr(sg.event, "annotations", None):
                ch = sg.event.annotations.get("channel", "")
            if ch and hasattr(self.state, "set_active_channel"):
                self.state.set_active_channel(ch)

    def is_view_culled(self) -> bool:
        try:
            if getattr(self, "_user_hidden", False):
                return True
            win = self.window()
            if win is not None and win.isMinimized():
                return True
            if self.parentWidget() is not None and not self.isVisible():
                return True
            if getattr(self, "_was_shown", False) and not self.isVisible():
                return True
            return False
        except Exception:
            return False

    def _on_render_timer_tick(self):
        try:
            if self.is_view_culled() or not getattr(self, "_dirty", False):
                return
            self._dirty = False
            from .helpers import run_adaptive_render
            run_adaptive_render(self, self._do_update_core, weight="heavy")
        except RuntimeError:
            pass

    def schedule_update(self, *args):
        if self.state is not None and not getattr(self.state, "is_streaming", False):
            self.update()
        else:
            self._dirty = True

    def setVisible(self, visible: bool):
        if not visible:
            self._user_hidden = True
        else:
            self._user_hidden = False
            self._was_shown = True
        super().setVisible(visible)
        if visible and getattr(self, "_dirty", False):
            self._dirty = False
            self._do_update_core()

    def hideEvent(self, event):
        super().hideEvent(event)
        self._user_hidden = True

    def showEvent(self, event):
        super().showEvent(event)
        self._was_shown = True
        self._user_hidden = False
        if getattr(self, "_dirty", False):
            self._dirty = False
            self._do_update_core()

    def unset_connections(self):
        """Disconnect all tracked ViewerState signal connections cleanly."""
        for sig, slot in self._state_connections:
            try:
                sig.disconnect(slot)
            except (TypeError, RuntimeError):
                pass
        self._state_connections.clear()
        self._unregister_live_frames()

    def closeEvent(self, event):
        self.unset_connections()
        super().closeEvent(event)

    @Slot()
    def _on_unit_params_changed(self, *args):
        self.update()

    def _get_sync_token(self) -> Optional[tuple]:
        if not self.state or not self.state.spike_groups:
            return None
        u_idx = getattr(self.state, "cur_spike_group", 0)
        stim = getattr(self.state, "stimno", 0)
        sg_len = len(self.state.spike_groups)
        u_spikes_count = (
            len(self.state.spike_groups[u_idx].event)
            if 0 <= u_idx < sg_len and self.state.spike_groups[u_idx].event is not None
            else 0
        )
        return (sg_len, stim, u_idx, u_spikes_count)

    def _on_unit_change(self, *args):
        self.update()
        self._last_unit_change_sync_token = self._get_sync_token()

    def _on_new_spike_events(self, unit_idx: int = -1):
        if getattr(self.state, "is_streaming", False):
            self.schedule_update()
            return
        curr_token = self._get_sync_token()
        if (
            self._last_unit_change_sync_token is not None
            and curr_token == self._last_unit_change_sync_token
        ):
            # Already updated synchronously by onUnitChange right before this
            return
        self.update()

    def schedule_update(self):
        self._dirty = True

    def update(self, *args):
        if self.is_view_culled():
            self._dirty = True
            return
        self._dirty = False
        from .helpers import run_adaptive_render
        run_adaptive_render(self, self._do_update_core, weight="heavy")

    def _do_update_core(self):
        if self.is_view_culled():
            self._dirty = True
            return
        self.render_count = getattr(self, "render_count", 0) + 1
        self.spike_tablemodel.update()
        # Ensure the correct row is selected as per state
        if self.state is None:
            return
        active_candidate = (
            self.state.get_active_candidate()
            if hasattr(self.state, "get_active_candidate")
            else None
        )
        if active_candidate is not None:
            row = self.spike_tablemodel.row_for_item(
                ("candidate", int(active_candidate.candidate_id))
            )
            if row is not None and row >= 0:
                if self.tbl.currentIndex().row() != row:
                    with QSignalBlocker(self.tbl.selectionModel()):
                        self.tbl.selectRow(row)
                self._update_action_buttons()
                return
        try:
            current_group = self.state.getUnitGroup()
        except (IndexError, Exception):
            current_group = None
        if current_group is not None:
            # Find the row corresponding to the current unit group
            for row in range(self.tbl.model().rowCount()):
                item = self.spike_tablemodel.row_to_item(row)
                if item is None or item[0] != "unit":
                    continue
                sg = self.tbl.model().data(
                    self.tbl.model().index(row, 0), role="SPIKEGROUP_ROLE"
                )
                if sg is not None and current_group.event is not None and sg.event.name == current_group.event.name:
                    if self.tbl.currentIndex().row() != row:
                        with QSignalBlocker(self.tbl.selectionModel()):
                            self.tbl.selectRow(row)
                    break
        self._update_action_buttons()

    @Slot()
    def set_selection(self, x):
        y = x.toList()
        if len(y) > 0:
            row = y[0].top()
            item = self.spike_tablemodel.row_to_item(row)
            if item is not None and item[0] == "candidate":
                self.state.set_active_candidate(int(item[1]))
                self._update_action_buttons()
                return
            if item is not None and item[0] == "unit":
                if self.state is not None:
                    self.state.set_active_candidate(None)
                self.state.setUnitGroup(int(item[1]))
            else:
                self.state.setUnitGroup(0)
            self._update_action_buttons()
        else:
            self.state.set_active_candidate(None)
            self.state.setUnitGroup(0)
            self._update_action_buttons()

    def _selected_row_item(self):
        selection = self.tbl.selectionModel().selectedRows()
        if not selection:
            return None
        return self.spike_tablemodel.row_to_item(selection[0].row())

    def _accept_selected_candidate(self) -> bool:
        item = self._selected_row_item()
        if item is None or item[0] != "candidate" or self.state is None:
            return False
        return self.state.accept_candidate(int(item[1])) >= 0

    def _discard_selected_candidate(self) -> bool:
        item = self._selected_row_item()
        if item is None or item[0] != "candidate" or self.state is None:
            return False
        return bool(self.state.discard_candidate(int(item[1])))

    @Slot()
    def _on_add_clicked(self):
        item = self._selected_row_item()
        if item is not None and item[0] == "candidate":
            self._accept_selected_candidate()
        else:
            self.add_row_and_select()

    @Slot()
    def _on_remove_clicked(self):
        item = self._selected_row_item()
        if item is not None and item[0] == "candidate":
            self._discard_selected_candidate()
        else:
            self.remove_selected_row()

    def remove_selected_row(self):
        item = self._selected_row_item()
        if item is None or self.state is None:
            return
        if item[0] == "candidate":
            self.state.discard_candidate(int(item[1]))
            return
        unit_idx = int(item[1])
        if 0 <= unit_idx < len(self.state.spike_groups):
            sg = self.state.spike_groups[unit_idx]
            app = QApplication.instance()
            is_offscreen = app is not None and app.platformName() == "offscreen"
            has_events = (
                getattr(sg, "event", None) is not None and len(sg.event) > 0
            )
            if (
                getattr(self.state, "is_streaming", False)
                and has_events
                and not is_offscreen
            ):
                unit_name = getattr(sg.event, "name", None) or f"unit_{unit_idx}"
                reply = QMessageBox.question(
                    self,
                    "Delete Tracked Unit?",
                    f"Unit '{unit_name}' has {len(sg.event)} spikes and a live stream is active.\n"
                    "Are you sure you want to delete this unit?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if reply != QMessageBox.Yes:
                    return
            self.state.removeUnitGroup(sg)

    def add_row_and_select(self):
        if self.state is None:
            return
        self.state.addUnitGroup()
        new_idx = len(self.state.spike_groups) - 1
        row = self.spike_tablemodel.row_for_item(("unit", new_idx))
        if row is not None and row >= 0:
            self.tbl.selectRow(row)

    def _candidate_rows(self):
        if self.state is None or not hasattr(self.state, "get_visible_candidates"):
            return []
        return list(self.state.get_visible_candidates())

    def _on_candidates_changed(self):
        self.spike_tablemodel.update()
        self._update_scan_button()
        self._update_action_buttons()
        if self._manual_scan_select_pending and not getattr(
            self.state, "is_streaming", False
        ):
            candidates = self._candidate_rows()
            if candidates:
                self._manual_scan_select_pending = False
                row = self.spike_tablemodel.row_for_item(
                    ("candidate", int(candidates[0].candidate_id))
                )
                if row is not None and row >= 0:
                    with QSignalBlocker(self.tbl.selectionModel()):
                        self.tbl.selectRow(row)
                    self.tbl.scrollTo(
                        self.spike_tablemodel.index(row, 0),
                        QAbstractItemView.PositionAtCenter,
                    )
                    self.state.set_active_candidate(int(candidates[0].candidate_id))
        self.update()

    @Slot(int)
    def _sync_candidate_selection(self, candidate_id: int):
        if candidate_id is None or int(candidate_id) < 0:
            return
        row = self.spike_tablemodel.row_for_item(("candidate", int(candidate_id)))
        if row is None or row < 0:
            return
        if self.tbl.currentIndex().row() != row:
            with QSignalBlocker(self.tbl.selectionModel()):
                self.tbl.selectRow(row)
        self._update_action_buttons()

    def _update_action_buttons(self):
        if self.state is None:
            if hasattr(self, "acceptCandidateButton"):
                self.acceptCandidateButton.setVisible(False)
                self.acceptCandidateButton.setEnabled(False)
            if hasattr(self, "discardCandidateButton"):
                self.discardCandidateButton.setVisible(False)
                self.discardCandidateButton.setEnabled(False)
            return
        item = self._selected_row_item()
        has_candidates = bool(self._candidate_rows())
        is_candidate = item is not None and item[0] == "candidate"
        if hasattr(self, "acceptCandidateButton"):
            self.acceptCandidateButton.setVisible(is_candidate or has_candidates)
            self.acceptCandidateButton.setEnabled(is_candidate)
        if hasattr(self, "discardCandidateButton"):
            self.discardCandidateButton.setVisible(is_candidate or has_candidates)
            self.discardCandidateButton.setEnabled(is_candidate)
        style = self.style()
        if is_candidate:
            self.addButton.setIcon(
                style.standardIcon(PySide6.QtWidgets.QStyle.SP_DialogApplyButton)
            )
            self.addButton.setToolTip("Accept the selected candidate")
            self.removeButton.setIcon(
                style.standardIcon(PySide6.QtWidgets.QStyle.SP_TrashIcon)
            )
            self.removeButton.setToolTip("Discard the selected candidate")
        else:
            self.addButton.setIcon(
                style.standardIcon(PySide6.QtWidgets.QStyle.SP_FileDialogNewFolder)
            )
            self.addButton.setToolTip("Add a new event/group")
            self.removeButton.setIcon(
                style.standardIcon(PySide6.QtWidgets.QStyle.SP_TrashIcon)
            )
            self.removeButton.setToolTip("Delete the selected event/group")

    def _update_scan_button(self):
        try:
            count = len(self._candidate_rows())
        except Exception:
            count = 0
        if not hasattr(self, "scanButton"):
            return
        suffix = f" ({count})" if count else ""
        self.scanButton.setText(f"Scan{suffix}")
        if getattr(self.state, "_candidate_scan_truncated", False):
            self.scanButton.setToolTip(
                "Search the recording for candidate units (last scan covered only "
                "the configured maximum sweep count)"
            )
        else:
            self.scanButton.setToolTip("Search the recording for candidate units")
        if self.autoButton.isChecked() != bool(getattr(self.state, "auto_scan_live", False)):
            with QSignalBlocker(self.autoButton):
                self.autoButton.setChecked(bool(getattr(self.state, "auto_scan_live", False)))

    @Slot()
    def _on_scan_clicked(self):
        if self.state is None:
            return
        self._manual_scan_select_pending = False
        scope = "live" if getattr(self.state, "is_streaming", False) else "offline"
        started = self.state.request_candidate_scan(synchronous=False, scope=scope)
        if started and not getattr(self.state, "is_streaming", False):
            self._manual_scan_select_pending = True

    @Slot(bool)
    def _on_auto_toggled(self, checked: bool):
        if self.state is None:
            return
        self.state.set_auto_scan_live(bool(checked))

    @Slot(bool)
    def _on_candidate_scan_state(self, scanning: bool):
        self.scanButton.setEnabled(not scanning)
        self.candidateStatusLabel.setText("Scanning..." if scanning else "")
        self._update_scan_button()

    def _rescan_discarded_candidates(self):
        if self.state is None:
            return
        self.state._candidate_rescan_discarded = True
        scope = "live" if getattr(self.state, "is_streaming", False) else "offline"
        self.state.request_candidate_scan(synchronous=False, scope=scope)

    @Slot()
    def _open_proposal_config_view(self):
        top_win = self.window()
        if hasattr(top_win, "open_or_show_spike_proposal_config"):
            return top_win.open_or_show_spike_proposal_config()
        if hasattr(top_win, "newWindow"):
            return top_win.newWindow("SpikeProposalConfig")
        return None


class SpikeGroupTableView2(QTableView):
    pass
    # def mousePressEvent(self, event):
    #     index = self.indexAt(event.pos())
    #     if index.isValid() and index.column() == 0:
    #         self.selectRow(index.row())
    #         # Optionally emit selectionChanged if you want to trigger logic
    #     else:
    #         # Ignore selection for other columns
    #         event.ignore()
    #     super().mousePressEvent(event)


class ComboBoxDelegate(QStyledItemDelegate):
    def __init__(self, parent=None, items: List[str] = None):
        super(ComboBoxDelegate, self).__init__(parent)
        if items is None:
            items = ["Cm", "Cmh", "Cmih", "Cmihi", "Ch"]
        self.items = items

    def createEditor(self, parent, option, index):
        combo = QComboBox(parent)
        # Set the default text to match the current data
        combo.setEditable(True)
        combo.addItems(self.items)

        return combo

    def setEditorData(self, editor, index):
        value = index.model().data(index, Qt.DisplayRole)
        editor.setCurrentText(value)

    def setModelData(self, editor, model, index):
        model.setData(index, editor.currentText(), Qt.EditRole)


class SpikeGroupTableModel(QAbstractTableModel):
    def __init__(self, spikegroups_func=None, candidates_func=None, state=None):
        """
        spikegroups_func is a function which returns the spike groups;
        candidates_func optionally returns the visible candidate units.
        """
        QAbstractTableModel.__init__(
            self,
        )
        self.spikegroups_func = spikegroups_func
        self.candidates_func = candidates_func
        self._state = state
        self.spikegroups = self.spikegroups_func()
        self._last_row_count = len(self.spikegroups or [])
        self._rows: List[Tuple[str, int]] = [("unit", i) for i in range(len(self.spikegroups or []))]
        self._last_candidate_ids: Tuple[int, ...] = ()
        self._in_model_update = False
        self.headers = ["name", "start", "end", "N", "type", "notes", "channel", "neighbors", "track", "win (ms)", "thresh (mV)"]
        self.annotations_to_show = {
            "notes": "notes",
            "type": "fibre_type",
        }

    def _candidate_list(self):
        if self.candidates_func is None:
            return []
        try:
            return list(self.candidates_func() or [])
        except Exception:
            return []

    def _refresh_rows(self):
        groups = self.spikegroups_func() or []
        self.spikegroups = groups
        candidates = self._candidate_list()
        self._rows = [("unit", i) for i in range(len(groups))]
        self._rows.extend(("candidate", int(c.candidate_id)) for c in candidates)
        return groups, candidates

    def _candidate_by_id(self, candidate_id: int):
        for candidate in self._candidate_list():
            if int(candidate.candidate_id) == int(candidate_id):
                return candidate
        return None

    def row_to_item(self, row: int):
        groups = self.spikegroups_func() or []
        candidates = self._candidate_list()
        expected = len(groups) + len(candidates)
        if len(self._rows) != expected or any(
            self._rows[len(groups) + i] != ("candidate", int(c.candidate_id))
            for i, c in enumerate(candidates)
            if len(groups) + i < len(self._rows)
        ):
            self.spikegroups = groups
            self._rows = [("unit", i) for i in range(len(groups))]
            self._rows.extend(("candidate", int(c.candidate_id)) for c in candidates)
        if 0 <= int(row) < len(self._rows):
            return self._rows[int(row)]
        return None

    def row_for_item(self, item) -> Optional[int]:
        self._refresh_rows()
        try:
            return self._rows.index(tuple(item))
        except (ValueError, TypeError):
            return None

    def rowCount(self, parent=QModelIndex()):
        groups = self.spikegroups_func() or []
        candidates = self._candidate_list()
        return len(groups) + len(candidates)

    def columnCount(self, parent=QModelIndex()):
        return len(self.headers)

    def headerData(
        self, section: int, orientation: PySide6.QtCore.Qt.Orientation, role: int = ...
    ) -> Any:
        if role != Qt.DisplayRole:
            return None

        if orientation == Qt.Horizontal:
            return self.headers[section]
        else:
            return section

    @Slot()
    def update(self):
        new_groups, new_candidates = self._refresh_rows()
        new_count = len(self._rows)
        new_candidate_ids = tuple(int(c.candidate_id) for c in new_candidates)
        self._row_windows = {}
        for idx, sg in enumerate(new_groups or []):
            try:
                self._row_windows[idx] = sg.get_window()
            except Exception:
                self._row_windows[idx] = None
        self._in_model_update = True
        try:
            if (
                new_count == self._last_row_count
                and new_candidate_ids == self._last_candidate_ids
                and new_count > 0
            ):
                top_left = self.index(0, 0)
                bottom_right = self.index(new_count - 1, self.columnCount() - 1)
                self.dataChanged.emit(top_left, bottom_right)
            else:
                self.modelAboutToBeReset.emit()
                self._last_row_count = new_count
                self._last_candidate_ids = new_candidate_ids
                self.modelReset.emit()
        finally:
            self._in_model_update = False

    def _candidate_summary(self, candidate) -> str:
        evidence = candidate.evidence
        return (
            f"hits {int(getattr(evidence, 'hit_count', 0))}, "
            f"jitter {float(getattr(evidence, 'jitter_ms', 0.0)):.2f} ms, "
            f"SNR {float(getattr(evidence, 'median_snr', 0.0)):.1f}, "
            f"corr {float(getattr(evidence, 'waveform_score', 0.0)):.2f}, "
            f"support {float(getattr(evidence, 'footprint_score', 0.0)):.0%}"
        )

    def _candidate_data(self, candidate_id: int, column: int, role):
        candidate = self._candidate_by_id(candidate_id)
        if candidate is None:
            return None
        if role == "SPIKEGROUP_ROLE":
            return candidate.unit
        if role == Qt.CheckStateRole and column == 8:
            return None
        if role == Qt.BackgroundRole:
            color = QApplication.palette().color(QPalette.Midlight)
            if color.alpha() == 255:
                color.setAlpha(140)
            return color
        if role == Qt.FontRole:
            font = QFont()
            font.setItalic(True)
            return font
        if role == Qt.ToolTipRole:
            evidence = candidate.evidence
            return (
                f"{candidate.name}: candidate ({candidate.status})\n"
                f"channel: {candidate.channel}  neighbors: {', '.join(candidate.neighbor_channels or [])}\n"
                f"score: {candidate.score:.3f}  algorithm: {candidate.algorithm}"
                f"{' + template' if candidate.template_refined else ''}\n"
                f"hits: {int(getattr(evidence, 'hit_count', 0))} "
                f"(recent {int(getattr(evidence, 'recent_hit_count', 0))})\n"
                f"jitter: {float(getattr(evidence, 'jitter_ms', 0.0)):.2f} ms  "
                f"SNR: {float(getattr(evidence, 'median_snr', 0.0)):.2f}\n"
                f"waveform: {float(getattr(evidence, 'waveform_score', 0.0)):.2f}  "
                f"footprint: {float(getattr(evidence, 'footprint_score', 0.0)):.2f}\n"
                f"recency: {float(getattr(evidence, 'recency_score', 0.0)):.2f}  "
                f"distinctiveness: {float(getattr(evidence, 'distinctiveness', 0.0)):.2f}"
            )
        if role == Qt.DisplayRole:
            if column == 0:
                return candidate.name
            if column in (1, 2):
                center = float(candidate.latency_center_samples)
                half = float(candidate.latency_tolerance_samples)
                value = center - half if column == 1 else center + half
                return int(round(value))
            if column == 3:
                return candidate.hit_count
            if column == 4:
                return f"candidate ({float(candidate.score):.2f})"
            if column == 5:
                if candidate.notes_edited and candidate.notes:
                    return candidate.notes
                return self._candidate_summary(candidate)
            if column == 6:
                return candidate.channel or ""
            if column == 7:
                return ", ".join(candidate.neighbor_channels or [])
            if column == 8:
                return ""
            if column == 9:
                return float(candidate.tracking_window_ms)
            if column == 10:
                return float(candidate.tracking_threshold_mv)
            return ""
        if role == Qt.EditRole:
            if column == 0:
                return candidate.name
            if column == 5:
                return candidate.notes
            if column == 6:
                return candidate.channel or ""
            if column == 7:
                return ", ".join(candidate.neighbor_channels or [])
            if column == 9:
                return float(candidate.tracking_window_ms)
            if column == 10:
                return float(candidate.tracking_threshold_mv)
            return ""
        if role == Qt.ForegroundRole:
            return QApplication.palette().color(QPalette.Text)
        return None

    def _set_candidate_data(self, candidate_id: int, column: int, value, role) -> bool:
        if role != Qt.EditRole:
            return False
        candidate = self._candidate_by_id(candidate_id)
        if candidate is None:
            return False
        changed = False
        if column == 0:
            candidate.name = str(value) or candidate.name
            candidate.name_edited = True
            if hasattr(candidate.unit, "event") and candidate.unit.event is not None:
                candidate.unit.event.name = candidate.name
            changed = True
        elif column == 5:
            candidate.notes = str(value)
            candidate.notes_edited = True
            if hasattr(candidate.unit, "event") and candidate.unit.event is not None:
                candidate.unit.event.annotations["notes"] = candidate.notes
            changed = True
        elif column == 6:
            candidate.channel = str(value)
            candidate.unit.channel = candidate.channel
            if candidate.unit.event is not None:
                candidate.unit.event.annotations["channel"] = candidate.channel
            changed = True
        elif column == 7:
            if isinstance(value, str):
                chans = [c.strip() for c in value.split(",") if c.strip()]
            elif isinstance(value, (list, tuple)):
                chans = [str(c).strip() for c in value if str(c).strip()]
            else:
                chans = []
            candidate.neighbor_channels = chans
            candidate.unit.neighbor_channels = list(chans)
            if candidate.unit.event is not None:
                candidate.unit.event.annotations["neighbor_channels"] = list(chans)
            changed = True
        elif column == 9:
            candidate.tracking_window_ms = float(value)
            candidate.unit.tracking_window_ms = float(value)
            if candidate.unit.event is not None:
                candidate.unit.event.annotations["tracking_window_ms"] = float(value)
            changed = True
        elif column == 10:
            candidate.tracking_threshold_mv = float(value)
            candidate.unit.tracking_threshold_mv = float(value)
            if candidate.unit.event is not None:
                candidate.unit.event.annotations["tracking_threshold_mv"] = float(value)
            changed = True
        if changed:
            row = self.row_for_item(("candidate", candidate_id))
            if row is not None:
                changed_index = self.index(row, column)
                self.dataChanged.emit(changed_index, changed_index)
        return changed

    def _unit_qcolor(self, unit_idx: int) -> QColor:
        if self._state is not None and hasattr(self._state, "get_unit_color"):
            try:
                c = self._state.get_unit_color(unit_idx)
                if isinstance(c, QColor):
                    return c
                if isinstance(c, (tuple, list)) and len(c) >= 3:
                    return QColor(int(c[0]), int(c[1]), int(c[2]))
                if isinstance(c, str) and c:
                    return QColor(c)
            except Exception:
                pass
        r, g, b = SET2_COLORS[max(0, int(unit_idx)) % len(SET2_COLORS)]
        return QColor(r, g, b)

    def _tracking_health_badge(
        self, unit_idx: int, sg
    ) -> Tuple[str, Optional[QColor], bool, int, int]:
        """Return (badge_text, fg_color, auto_track, misses, max_skips) for a unit row."""
        raw_max_skips = getattr(sg, "tracking_max_skips", 1)
        max_skips = 1 if raw_max_skips is None else int(raw_max_skips)
        misses = int(getattr(sg, "_consecutive_misses", 0) or 0)
        auto_track = bool(getattr(sg, "auto_track", False))

        if not auto_track:
            return ("", None, False, misses, max_skips)

        # Prefer explicit get_tracking_health() if implemented as a 0-arg hook
        # (without overwriting _consecutive_misses via ViewerState's sweep-scan method).
        health = None
        if hasattr(sg, "get_tracking_health"):
            try:
                import inspect
                sig = inspect.signature(sg.get_tracking_health)
                required = [
                    p
                    for p in sig.parameters.values()
                    if p.default is inspect.Parameter.empty
                    and p.kind
                    in (
                        inspect.Parameter.POSITIONAL_ONLY,
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    )
                ]
                if not required and "event_signal" not in sig.parameters:
                    health = sg.get_tracking_health()
            except Exception:
                health = None

        if isinstance(health, dict):
            auto_track = bool(health.get("auto_track", auto_track))
            misses = int(health.get("consecutive_misses", misses) or 0)
            raw_ms = health.get("max_skips", max_skips)
            max_skips = max_skips if raw_ms is None else int(raw_ms)
            if not auto_track:
                return ("", None, False, misses, max_skips)
            status = str(health.get("status", "")).lower()
            if status in ("locked", "ok"):
                return ("🟢 Locked", QColor("#2b7a2b"), True, misses, max_skips)
            if status in ("skip", "skipping"):
                return ("🟡 Skip", QColor("#b86b11"), True, misses, max_skips)
            if status in ("lost", "stale"):
                return ("🔴 Lost", QColor("#c9302c"), True, misses, max_skips)
        elif isinstance(health, str) and health:
            s = health.lower()
            if "lock" in s:
                return ("🟢 Locked", QColor("#2b7a2b"), True, misses, max_skips)
            if "skip" in s:
                return ("🟡 Skip", QColor("#b86b11"), True, misses, max_skips)
            if "lost" in s:
                return ("🔴 Lost", QColor("#c9302c"), True, misses, max_skips)

        if misses == 0:
            return ("🟢 Locked", QColor("#2b7a2b"), True, misses, max_skips)
        if 1 <= misses <= max_skips:
            return ("🟡 Skip", QColor("#b86b11"), True, misses, max_skips)
        return ("🔴 Lost", QColor("#c9302c"), True, misses, max_skips)

    def _unit_spike_count(self, sg) -> int:
        if hasattr(sg, "event") and sg.event is not None and len(sg.event) > 0:
            return len(sg.event)
        if hasattr(sg, "get_number_of_events"):
            return int(sg.get_number_of_events())
        if hasattr(sg, "idx_arr") and sg.idx_arr is not None:
            return len([x for x in sg.idx_arr if x is not None])
        return 0

    def _unit_tooltip(self, unit_idx: int, sg) -> str:
        name = getattr(getattr(sg, "event", None), "name", None) or f"unit_{unit_idx}"
        ch = getattr(sg, "channel", "")
        if not ch and hasattr(sg, "event") and getattr(sg.event, "annotations", None):
            ch = sg.event.annotations.get("channel", "")
        neigh = getattr(sg, "neighbor_channels", None)
        if neigh is None and hasattr(sg, "event") and getattr(sg.event, "annotations", None):
            neigh = sg.event.annotations.get("neighbor_channels", [])
        if isinstance(neigh, (list, tuple)):
            neigh_str = ", ".join(str(x) for x in neigh if x) or "none"
        elif isinstance(neigh, str) and neigh:
            neigh_str = neigh
        else:
            neigh_str = "none"
        n_spikes = self._unit_spike_count(sg)
        badge, _fg, auto_track, misses, max_skips = self._tracking_health_badge(
            unit_idx, sg
        )
        auto_str = f"ON ({badge})" if auto_track else "OFF"
        win_ms = float(getattr(sg, "tracking_window_ms", 5.0) or 5.0)
        thresh_mv = float(getattr(sg, "tracking_threshold_mv", 0.01) or 0.01)
        return (
            f"{name} (Unit {unit_idx})\n"
            f"channel: {ch or 'unassigned'}  neighbors: {neigh_str}\n"
            f"total spikes: {n_spikes}\n"
            f"auto-track: {auto_str}\n"
            f"consecutive misses: {misses} (max skips: {max_skips})\n"
            f"window: {win_ms:.2f} ms  threshold: {thresh_mv:.4g} mV"
        )

    def data(
        self,
        index: Union[PySide6.QtCore.QModelIndex, PySide6.QtCore.QPersistentModelIndex],
        role: int = ...,
    ) -> Any:
        column = index.column()
        row = index.row()
        item = self.row_to_item(row)
        if item is None:
            return None
        if item[0] == "candidate":
            return self._candidate_data(int(item[1]), column, role)
        unit_idx = int(item[1])
        if not (0 <= unit_idx < len(self.spikegroups)):
            return None
        sg = self.spikegroups[unit_idx]

        if role == Qt.ToolTipRole:
            return self._unit_tooltip(unit_idx, sg)

        if column == 0 and role == Qt.DecorationRole:
            return self._unit_qcolor(unit_idx)

        # Column 8: track checkbox + tracking health badge
        if column == 8:
            if role == Qt.CheckStateRole:
                return Qt.Checked if getattr(sg, "auto_track", False) else Qt.Unchecked
            elif role == Qt.DisplayRole:
                badge, _fg, _auto, _misses, _max = self._tracking_health_badge(
                    unit_idx, sg
                )
                return badge
            elif role == Qt.ForegroundRole:
                _badge, fg, _auto, _misses, _max = self._tracking_health_badge(
                    unit_idx, sg
                )
                return fg
            elif role == Qt.EditRole:
                return getattr(sg, "auto_track", False)
            elif role == "SPIKEGROUP_ROLE":
                return sg
            return None

        # Column 9: win (ms)
        if column == 9:
            if role in (Qt.DisplayRole, Qt.EditRole):
                return getattr(sg, "tracking_window_ms", 5.0)
            elif role == "SPIKEGROUP_ROLE":
                return sg
            return None

        # Column 10: thresh (mV)
        if column == 10:
            if role in (Qt.DisplayRole, Qt.EditRole):
                return getattr(sg, "tracking_threshold_mv", 0.01)
            elif role == "SPIKEGROUP_ROLE":
                return sg
            return None

        if role == Qt.DisplayRole or role == Qt.EditRole:
            if column == 0:  # name / SpikeID
                return getattr(sg.event, "name", None) or item[1]
            elif column in (1, 2):  # start/end
                row_wins = getattr(self, "_row_windows", None)
                win = row_wins.get(item[1]) if row_wins is not None and item[1] in row_wins else sg.get_window()
                if win is None:
                    return ""
                return win[column - 1]
            elif column == 3:  # N (number of events)
                return self._unit_spike_count(sg)
            elif column == 4:  # type / fibre class
                return sg.event.annotations.get("fibre_type", getattr(sg, "fibre_type", ""))
            elif column == 5:  # notes
                return sg.event.annotations.get("notes", getattr(sg, "notes", ""))
            elif column == 6:  # channel
                ch = getattr(sg, "channel", "")
                if not ch and hasattr(sg, "event") and getattr(sg.event, "annotations", None):
                    ch = sg.event.annotations.get("channel", "")
                return ch or ""
            elif column == 7:  # neighbors
                neigh = getattr(sg, "neighbor_channels", None)
                if neigh is None and hasattr(sg, "event") and getattr(sg.event, "annotations", None):
                    neigh = sg.event.annotations.get("neighbor_channels", [])
                if isinstance(neigh, (list, tuple)):
                    return ", ".join(neigh)
                elif isinstance(neigh, str):
                    return neigh
                return ""
            else:
                return ""
        elif role == "SPIKEGROUP_ROLE":
            return sg
        # elif role == Qt.BackgroundRole:
        # return QColor(Qt.white)

    def setData(
        self,
        index: Union[PySide6.QtCore.QModelIndex, PySide6.QtCore.QPersistentModelIndex],
        value: Any,
        role: int = ...,
    ) -> bool:
        column = index.column()
        row = index.row()
        item = self.row_to_item(row)
        if item is None:
            return False
        if item[0] == "candidate":
            return self._set_candidate_data(int(item[1]), column, value, role)
        sg = self.spikegroups[int(item[1])]

        # Column 8: track
        if column == 8:
            if role == Qt.CheckStateRole:
                val = bool(value == Qt.Checked or value == Qt.Checked.value or value is True)
                sg.auto_track = val
                if hasattr(sg, "event") and hasattr(sg.event, "annotations"):
                    sg.event.annotations["auto_track"] = val
                self.dataChanged.emit(index, index)
                return True
            elif role == Qt.EditRole:
                val = bool(value)
                sg.auto_track = val
                if hasattr(sg, "event") and hasattr(sg.event, "annotations"):
                    sg.event.annotations["auto_track"] = val
                self.dataChanged.emit(index, index)
                return True

        if role == Qt.EditRole:
            if column == 0:
                sg.event.name = str(value)
                self.dataChanged.emit(index, index)
                return True
            elif column == 4:  # type
                sg.fibre_type = str(value)
                sg.event.annotations["fibre_type"] = str(value)
                self.dataChanged.emit(index, index)
                return True
            elif column == 5:  # notes
                sg.notes = str(value)
                sg.event.annotations["notes"] = str(value)
                self.dataChanged.emit(index, index)
                return True
            elif column == 6:  # channel
                sg.channel = str(value)
                sg.event.annotations["channel"] = str(value)
                self.dataChanged.emit(index, index)
                return True
            elif column == 7:  # neighbors
                if isinstance(value, str):
                    parsed_chans = [c.strip() for c in value.split(",") if c.strip()]
                elif isinstance(value, (list, tuple)):
                    parsed_chans = [str(c).strip() for c in value if str(c).strip()]
                else:
                    parsed_chans = []
                sg.neighbor_channels = parsed_chans
                sg.event.annotations["neighbor_channels"] = parsed_chans
                self.dataChanged.emit(index, index)
                return True
            elif column == 9:  # win (ms)
                sg.tracking_window_ms = float(value)
                if hasattr(sg, "event") and hasattr(sg.event, "annotations"):
                    sg.event.annotations["tracking_window_ms"] = float(value)
                self.dataChanged.emit(index, index)
                return True
            elif column == 10:  # thresh (mV)
                sg.tracking_threshold_mv = float(value)
                if hasattr(sg, "event") and hasattr(sg.event, "annotations"):
                    sg.event.annotations["tracking_threshold_mv"] = float(value)
                self.dataChanged.emit(index, index)
                return True

        return False

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        flags = Qt.ItemIsEnabled | Qt.ItemIsSelectable
        item = self.row_to_item(index.row()) if index.isValid() else None
        if item is not None and item[0] == "candidate":
            if index.column() in (0, 5, 6, 7, 9, 10):
                flags |= Qt.ItemIsEditable
            return flags
        # Column 8: UserCheckable + Editable
        if index.column() == 8:
            flags |= Qt.ItemIsUserCheckable | Qt.ItemIsEditable
        # Columns 0, 4, 5, 6, 7, 9, 10: Editable
        elif index.column() in (0, 4, 5, 6, 7, 9, 10):
            flags |= Qt.ItemIsEditable
        return flags



if __name__ == "__main__":

    app = QApplication([])
    state = ViewerState()
    state.loadFile(r"data/test2.h5")
    # evt = neo.Event(np.array([100]) * pq.s)
    # evt.array_annotate(info=["something"], number=[5])
    view = SpikeGroupTableView(None, state=state)
    # view = EventEdit(event=evt[0], annotations=evt.array_annotations_at_index(0))
    view.show()
    app.exec()
