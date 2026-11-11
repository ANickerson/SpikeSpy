from typing import Any, List, Optional, Union

import PySide6


from PySide6.QtCore import QAbstractTableModel, QModelIndex, QObject, Qt, Signal, Slot
from PySide6.QtGui import QAction, QColor
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
    QMainWindow,
    QMdiArea,
    QMdiSubWindow,
    QMenu,
    QMenuBar,
    QPushButton,
    QSpinBox,
    QTableView,
    QVBoxLayout,
    QWidget,
    QStyledItemDelegate,
)

from .ViewerState import ViewerState
from PySide6.QtWidgets import QComboBox


class SpikeGroupTableView(QWidget):
    def __init__(
        self,
        parent: Optional[PySide6.QtWidgets.QWidget] = None,
        state: ViewerState = None,
    ) -> None:
        super().__init__(parent)
        self.spike_tablemodel = SpikeGroupTableModel(lambda: state.spike_groups)

        self.tbl = SpikeGroupTableView2(self)
        self.tbl.setModel(self.spike_tablemodel)
        self.state = state
        if state is not None:
            state.onUnitChange.connect(self.update)
            state.onUnitGroupChange.connect(self.update)
            state.onLoadNewFile.connect(self.update)
            if hasattr(state, "onNewSpikeEvents"):
                state.onNewSpikeEvents.connect(self.update)

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
        button_layout.addStretch(1)  # Push buttons to the right
        self.addButton = QPushButton()
        self.addButton.setIcon(add_icon)
        self.addButton.setToolTip("Add a new event/group")
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
        self.addButton.clicked.connect(self.add_row_and_select)
        self.spike_tablemodel.dataChanged.connect(self.state.onUnitGroupChange)
        self.update()

    def on_table_context_menu(self, pos):
        index = self.tbl.indexAt(pos)
        if not index.isValid():
            return
        row = index.row()
        sg = self.tbl.model().data(self.tbl.model().index(row, 0), role="SPIKEGROUP_ROLE")
        if sg is None:
            return
        ch = getattr(sg, "channel", "")
        if not ch and hasattr(sg, "event") and getattr(sg.event, "annotations", None):
            ch = sg.event.annotations.get("channel", "")

        menu = QMenu(self)
        if ch and hasattr(self.state, "set_active_channel"):
            act_focus = menu.addAction(f"Focus Channel '{ch}'")
            act_focus.triggered.connect(lambda: self.state.set_active_channel(ch))
        menu.exec(self.tbl.viewport().mapToGlobal(pos))

    def on_cell_double_clicked(self, index: QModelIndex):
        if not index.isValid():
            return
        # If user double-clicks the channel column (column 6)
        if index.column() == 6:
            row = index.row()
            sg = self.tbl.model().data(self.tbl.model().index(row, 0), role="SPIKEGROUP_ROLE")
            if sg is not None:
                ch = getattr(sg, "channel", "")
                if not ch and hasattr(sg, "event") and getattr(sg.event, "annotations", None):
                    ch = sg.event.annotations.get("channel", "")
                if ch and hasattr(self.state, "set_active_channel"):
                    self.state.set_active_channel(ch)

    def update(self):
        self.spike_tablemodel.update()
        # Ensure the correct row is selected as per state
        if self.state is None:
            return
        try:
            current_group = self.state.getUnitGroup()
        except (IndexError, Exception):
            current_group = None
        if current_group is not None:
            # Find the row corresponding to the current unit group
            for row in range(self.tbl.model().rowCount()):
                sg = self.tbl.model().data(
                    self.tbl.model().index(row, 0), role="SPIKEGROUP_ROLE"
                )
                if sg is not None and current_group.event is not None and sg.event.name == current_group.event.name:
                    self.tbl.selectionModel().blockSignals(True)
                    self.tbl.selectRow(row)
                    self.tbl.selectionModel().blockSignals(False)
                    break
        # self.tbl.selectionModel().disconnect(self.set_selection)
        # self.tbl.selectionModel().setCurrentIndex(
        #     ,
        #     QItemSelectionModel.SelectCurrent,
        # )

    @Slot()
    def set_selection(self, x):
        y = x.toList()
        if len(y) > 0:
            row = y[0].top()
            self.state.setUnitGroup(row)
        else:
            self.state.setUnitGroup(0)

    def remove_selected_row(self):
        selection = self.tbl.selectionModel().selectedRows()
        if selection:
            row = selection[0].row()
            sg = self.tbl.model().data(
                self.tbl.model().index(row, 0), role="SPIKEGROUP_ROLE"
            )

            self.state.removeUnitGroup(sg)

    def add_row_and_select(self):
        self.state.addUnitGroup()
        # Select the last row after adding
        row_count = self.tbl.model().rowCount()
        if row_count > 0:
            self.tbl.selectRow(row_count - 1)


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
    def __init__(self, spikegroups_func=None):
        """
        spikegroups_func is a function which returns the spike groups
        """
        QAbstractTableModel.__init__(
            self,
        )
        self.spikegroups_func = spikegroups_func
        self.spikegroups = self.spikegroups_func()
        self.headers = ["name", "start", "end", "N", "type", "notes", "channel", "neighbors"]
        self.annotations_to_show = {
            "notes": "notes",
            "type": "fibre_type",
        }

    def rowCount(self, parent=QModelIndex()):
        return len(self.spikegroups_func() or [])

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
        self.modelAboutToBeReset.emit()
        self.spikegroups = self.spikegroups_func()
        self.modelReset.emit()

    def data(
        self,
        index: Union[PySide6.QtCore.QModelIndex, PySide6.QtCore.QPersistentModelIndex],
        role: int = ...,
    ) -> Any:
        column = index.column()
        row = index.row()
        sg = self.spikegroups_func()[row]
        if role == Qt.DisplayRole or role == Qt.EditRole:
            if column == 0:  # name / SpikeID
                return getattr(sg.event, "name", None) or row
            elif column in (1, 2):  # start/end
                if sg.get_window() is None:
                    return ""
                return sg.get_window()[column - 1]
            elif column == 3:  # N (number of events)
                if hasattr(sg, "event") and sg.event is not None and len(sg.event) > 0:
                    return len(sg.event)
                elif hasattr(sg, "get_number_of_events"):
                    return sg.get_number_of_events()
                elif hasattr(sg, "idx_arr") and sg.idx_arr is not None:
                    return len([x for x in sg.idx_arr if x is not None])
                return 0
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
        if role == Qt.EditRole:
            column = index.column()
            row = index.row()
            sg = self.spikegroups_func()[row]
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

        return False

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        flags = Qt.ItemIsEnabled | Qt.ItemIsSelectable
        # Editable columns: name (0), type (4), notes (5), channel (6), neighbors (7)
        if index.column() in (0, 4, 5, 6, 7):
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
