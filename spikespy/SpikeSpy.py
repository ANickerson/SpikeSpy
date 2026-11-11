import cProfile
import copy
import sys
from dataclasses import dataclass, field
from datetime import datetime
from os import environ
from pathlib import Path
from typing import Any, List, Optional, Union
from csv import writer

import matplotlib
import matplotlib.style as mplstyle
import neo
import numpy as np
import PySide6
import quantities as pq
import logging
from matplotlib.widgets import PolygonSelector
from neo.io import NixIO, NixIOFr
from PySide6.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    QObject,
    Qt,
    Signal,
    Slot,
    QSettings,
)
from PySide6.QtGui import QAction, QColor, QShortcut, QKeySequence, QIcon, QPixmap
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
    QMessageBox,
    QScrollArea,
    QTextEdit,
    QLabel,
)

from spikespy.check_update import ReleaseNotesDialog, VersionCheckThread
from spikespy.processing import MultiAnalogSignal

# import PySide6QtAds as QtAds
from .mng_file_selector import QNeoSelector
from .MultiTraceView import MultiTraceView
from .MultiTracePGView import MultiTracePGView
from .NeoSettingsView import NeoSettingsView
from .SingleTraceView import SingleTraceView
from .SpikeGroupTable import SpikeGroupTableView
from .TrackingView import TrackingView
from .UnitView import UnitView
from .ViewerState import ViewerState, prompt_for_neo_file, tracked_neuron_unit
from .EventView import EventView
from .MultiTraceFixedView import MultiTraceFixedView
from .UnitSuggestor import UnitSuggestor
import PySide6QtAds as QtAds
from typing import Dict

mplstyle.use("fast")

window_options = {
    # 'TraceAnnotation': TraceView,
    "MultiTrace": MultiTraceView,
    "MultiTrace (PyQtGraph)": MultiTracePGView,
    "MultiTraceFixedView": MultiTraceFixedView,
    "UnitView": UnitView,
    "SpikeGroupTable": SpikeGroupTableView,
    "SingleTraceView": SingleTraceView,
    "Settings": NeoSettingsView,
    "TrackingView": TrackingView,
    "Data": QNeoSelector,
    "Events": EventView,
}
suggestors: Dict[str, UnitSuggestor] = {}


class MdiView(QMainWindow):
    # signals
    loadFile = Signal(str, str)

    def savePerspectives(self):
        logging.info(f"using settings file: {self.settings_file.fileName()}")
        self.dock_manager.addPerspective("main")
        self.settings_file.beginGroup("view")
        self.dock_manager.savePerspectives(self.settings_file)
        class_to_str = {v: k for k, v in self.window_options.items()}
        open_widgets = [x for x in self.dock_manager.dockWidgets()]
        self.settings_file.beginWriteArray("window", len(open_widgets))
        for i, x in enumerate(open_widgets):
            self.settings_file.setArrayIndex(i)
            self.settings_file.setValue("name", x.windowTitle())
            self.settings_file.setValue("class", class_to_str[type(x.widget())])
            try:
                settings = x.widget().get_settings()
            except:
                settings = {}
            self.settings_file.setValue("settings", settings)

        self.settings_file.endArray()
        self.settings_file.endGroup()

    def loadPerspectives(self):
        # self.settings_file.

        self.settings_file.beginGroup("view")
        # TODO: Not sure, perhaps closing all open windows is the most rational.
        # class_to_str = {v: k for k, v in self.window_options.items()}
        # open_widgets = [
        #     class_to_str[type(x.widget())] for x in self.dock_manager.dockWidgets()
        # ]
        logging.info(f"using settings file: {self.settings_file.fileName()}")
        self.dock_manager.loadPerspectives(self.settings_file)
        for x in range(self.settings_file.beginReadArray("window")):
            self.settings_file.setArrayIndex(x)
            w = self.newWindow(
                self.settings_file.value("class"), name=self.settings_file.value("name")
            )
            settings = self.settings_file.value("settings")
            try:
                w.set_settings(settings)
            except:
                pass
        self.settings_file.endArray()

        self.settings_file.endGroup()

        self.dock_manager.openPerspective("main")

    def handle_version_check_result(self, result):
        if result is not None:
            version, release_notes = result
            dialog = ReleaseNotesDialog(version, release_notes)
            dialog.exec()

    def __init__(
        self,
        parent: PySide6.QtWidgets.QWidget = None,
        state: ViewerState = None,
        check_updates=False,
        **kwargs,
    ) -> None:
        super().__init__(parent)
        self.settings_file = QSettings(
            QSettings.Format.IniFormat, QSettings.Scope.UserScope, "spikespy"
        )

        self.window_options = window_options
        self.cur_windows = []

        self.version_check_thread = VersionCheckThread()
        self.version_check_thread.worker.result.connect(
            self.handle_version_check_result
        )
        if check_updates:
            self.version_check_thread.start()

        from importlib.metadata import version as get_version

        version = get_version("spikespy")
        self.setWindowTitle(f"SpikeSpy - {version}")
        self.state = state or ViewerState(**kwargs)
        self.loadFile.connect(self.state.loadFile)
        self.loadFile.connect(self.updateRecents)
        self.state.onLoadNewFile.connect(
            lambda self=self: self.setWindowTitle(
                f"SpikeSpy ({version}) - {Path(self.state.title).name}"
            )
        )
        self.dock_manager = QtAds.CDockManager(self)
        self.dock_manager.addPerspective("main")
        self.mdi = QMdiArea()
        # self.setCentralWidget(self.mdi)

        # for k in [
        #     "MultiTrace",
        #     "UnitView",
        #     "SpikeGroupTable",
        #     "SingleTraceView",
        #     "TrackingView",
        # ]:
        #     self.newWindow(k)

        self.toolbar = self.addToolBar("")
        for k in self.window_options.keys():
            act = self.toolbar.addAction(k)
            act.triggered.connect(lambda *args, k=k: self.newWindow(k))

        # Menu bar
        self.menubar = self.menuBar()

        file_menu = self.menubar.addMenu("&File")
        file_menu.addAction(
            QAction("Open...", self, shortcut="Ctrl+O", triggered=lambda: self.open())
        )
        self.recent_menu = file_menu.addMenu("Open Recent")
        file_menu.addAction(
            QAction(
                "Save as nixio",
                self,
                shortcut="Ctrl+S",
                triggered=lambda: self.save_as(),
            )
        )
        file_menu.addAction(
            QAction(
                "Export spikes as csv",
                self,
                shortcut="Ctrl+E",
                triggered=lambda: self.export_csv(),
            )
        )
        file_menu.addAction(
            QAction(
                "import spikes csv",
                self,
                shortcut="Ctrl+I",
                triggered=lambda: self.import_csv(),
            )
        )

        self.updateRecents(None, None)

        self.settings_file.endArray()

        edit_menu = self.menubar.addMenu("&Edit")
        edit_menu.addAction(
            QAction(
                "undo", self, shortcut="Ctrl+Z", triggered=lambda: self.state.undo()
            )
        )
        edit_menu.addAction(
            QAction("save perspective", self, triggered=self.savePerspectives)
        )
        edit_menu.addAction(
            QAction("load perspective", self, triggered=self.loadPerspectives)
        )

        help_menu = self.menubar.addMenu("Help")

        def showInfo():
            info = InfoDialog()
            info.exec()

        help_menu.addAction(QAction("About", self, triggered=showInfo))
        help_menu.addAction(
            QAction(
                "Check for updates",
                self,
                triggered=lambda: self.version_check_thread.start(),
            )
        )

        # key shortcuts
        self.profiler = cProfile.Profile()
        self.is_profiling = False
        self.shortcut_profile = QShortcut(QKeySequence(Qt.Key_F1), self)

        def profile():
            if self.is_profiling:
                self.profiler.disable()
                self.is_profiling = False
                nom = QFileDialog.getSaveFileName(self, "Export profile")
                if nom is None:

                    return
                self.profiler.dump_stats(nom[0])
                self.profiler = cProfile.Profile()
                import subprocess

                subprocess.run(["snakeviz", nom[0]])

            else:
                self.is_profiling = True
                self.profiler.enable()

        self.shortcut_profile.activated.connect(lambda: profile())

        self.shortcut_next = QShortcut(
            QKeySequence(Qt.Key_Down), self, context=Qt.ApplicationShortcut
        )
        self.shortcut_next.activated.connect(
            lambda: self.state.setStimNo(self.state.stimno + 1)
        )

        self.shortcut_skip_next = QShortcut(
            QKeySequence(Qt.Key_PageDown), self, context=Qt.ApplicationShortcut
        )
        self.shortcut_skip_next.activated.connect(
            lambda: self.state.setStimNo(self.state.stimno + 5)
        )

        self.shortcut_prev = QShortcut(
            QKeySequence(Qt.Key_Up), self, context=Qt.ApplicationShortcut
        )
        self.shortcut_prev.activated.connect(
            lambda: self.state.setStimNo(self.state.stimno - 1)
        )

        self.shortcut_skip_prev = QShortcut(
            QKeySequence(Qt.Key_PageUp), self, context=Qt.ApplicationShortcut
        )
        self.shortcut_skip_prev.activated.connect(
            lambda: self.state.setStimNo(self.state.stimno - 5)
        )

        self.shortcut_del = QShortcut(
            QKeySequence(Qt.Key_Backspace), self, context=Qt.ApplicationShortcut
        )
        self.shortcut_del.activated.connect(lambda: self.state.setUnit(None))

        self.shortcut_copylast = QShortcut(
            QKeySequence(Qt.Key_Space), self, context=Qt.ApplicationShortcut
        )
        self.shortcut_copylast.activated.connect(lambda: self.state.setUnit(-1))

        self.shortcut_copynext = QShortcut(
            QKeySequence(Qt.Modifier.SHIFT | Qt.Key.Key_Space),
            self,
            context=Qt.ApplicationShortcut,
        )
        self.shortcut_copynext.activated.connect(lambda: self.state.setUnit(-2))

        self.move_mode = "snap"

        def move(dist=1, mode=None):
            print(f"move {dist}")
            cur_point = (
                self.state.spike_groups[self.state.cur_spike_group].idx_arr[
                    self.state.stimno
                ]
            )[0]
            if mode is None:
                mode = self.move_mode
            if mode == "snap":
                pts = self.state.get_peaks()[self.state.stimno]
                i = pts.searchsorted(cur_point)

                if dist > 0 and pts[i] != cur_point:
                    dist -= 1

                dist = pts[i + dist] - cur_point

            self.state.setUnit(cur_point + dist)  # TODO: make method

        self.shortcut_left = QShortcut(
            QKeySequence(Qt.Key_Left), self, context=Qt.ApplicationShortcut
        )
        self.shortcut_left.activated.connect(lambda: move(-1))

        self.shortcut_right = QShortcut(
            QKeySequence(Qt.Key_Right), self, context=Qt.ApplicationShortcut
        )
        self.shortcut_right.activated.connect(lambda: move(1))

        self.shortcut_numbers = [
            QShortcut(QKeySequence(Qt.Key_1), self, context=Qt.ApplicationShortcut),
            QShortcut(QKeySequence(Qt.Key_2), self, context=Qt.ApplicationShortcut),
            QShortcut(QKeySequence(Qt.Key_3), self, context=Qt.ApplicationShortcut),
            QShortcut(QKeySequence(Qt.Key_4), self, context=Qt.ApplicationShortcut),
            QShortcut(QKeySequence(Qt.Key_5), self, context=Qt.ApplicationShortcut),
            QShortcut(QKeySequence(Qt.Key_6), self, context=Qt.ApplicationShortcut),
            QShortcut(QKeySequence(Qt.Key_7), self, context=Qt.ApplicationShortcut),
            QShortcut(QKeySequence(Qt.Key_8), self, context=Qt.ApplicationShortcut),
            QShortcut(QKeySequence(Qt.Key_9), self, context=Qt.ApplicationShortcut),
        ]
        for i, x in enumerate(self.shortcut_numbers):
            x.activated.connect(lambda i=i: self.state.setUnitGroup(i))

        def add_select_new_unit():
            self.state.addUnitGroup()
            x = len(self.state.spike_groups)
            self.state.setUnitGroup(x - 1)

        self.shortcut_new_unit = QShortcut(
            QKeySequence(Qt.CTRL | Qt.Key_N), self, context=Qt.ApplicationShortcut
        )
        self.shortcut_new_unit.activated.connect(add_select_new_unit)

        def toggle_snap():
            self.move_mode = None if self.move_mode == "snap" else "snap"

        self.shortcut_snap = QShortcut(
            QKeySequence(Qt.Key_S), self, context=Qt.ApplicationShortcut
        )
        self.shortcut_snap.activated.connect(toggle_snap)

        def copy_previous():
            try:
                sg = self.state.spike_groups[self.state.cur_spike_group].idx_arr
                new_x = next(
                    sg[x][0]
                    for x in range(self.state.stimno - 1, -1, -1)
                    if sg[x] is not None
                )
                # localise to nearest peak
                cur_erp = self.state.get_erp()[self.state.stimno]
                offset = np.argmax(cur_erp[new_x - 300 : new_x + 300]) - 300
                self.state.setUnit(new_x + offset)
            except StopIteration:
                pass

        self.shortcut_copy_previous = QShortcut(
            QKeySequence(Qt.Key_C), self, context=Qt.ApplicationShortcut
        )
        self.shortcut_copy_previous.activated.connect(copy_previous)

    def export_csv(self):
        save_filename = QFileDialog.getSaveFileName(self, "Export")[0]

        if save_filename is None:
            return
        units = [x.event for x in self.state.spike_groups]
        export_csv(save_filename, stim_evt=self.state.event_signal, unit_evts=units)

    def updateRecents(self, filename, type):

        if filename is not None:
            to_save = [filename]
            for i in range(self.settings_file.beginReadArray("Recent")):
                self.settings_file.setArrayIndex(i)
                prevVal = self.settings_file.value("filename")
                if prevVal == filename:
                    continue
                to_save.append(prevVal)
            self.settings_file.endArray()

            self.settings_file.beginWriteArray("Recent")
            for i in range(len(to_save)):
                self.settings_file.setArrayIndex(i)
                self.settings_file.setValue("filename", to_save[i])
            self.settings_file.endArray()

        self.recent_menu.clear()
        for i in range(self.settings_file.beginReadArray("Recent")):
            self.settings_file.setArrayIndex(i)
            fname = self.settings_file.value("filename")
            self.recent_menu.addAction(
                QAction(
                    Path(fname).name,
                    self,
                    triggered=lambda n, fname=fname: self.loadFile.emit(
                        fname, "h5"
                    ),  # currently assumes h5
                )
            )

    def export_npz(
        self,
    ):  # save a condensed analysis package. - should also be able to load these in spikespy
        from .processing import create_erp_signals

        # erp = self.state.get_erp()
        timestamps = self.state.event_signal
        details = {
            "sampling_rate": self.state.sampling_rate,
            "unit": "mV",
            "window_size": self.state.window_size,
            "original_data": "",  # TODO: get ref. to original file if possible
        }

        units = self.state.spike_groups
        units_erps = [
            create_erp_signals(
                self.state.analog_signal, e.event, -0.5 * pq.ms, 1 * pq.ms
            )
            for e in units
        ]
        other_events = self.state.segment.events
        signal_erps = [
            create_erp_signals(e, timestamps, -0.5 * pq.ms, 1 * pq.ms)
            for e in self.state.segment.analogsignals
        ]

    def import_csv(self):
        open_filename = QFileDialog.getOpenFileName(self, "csv import")[0]
        if open_filename is None:
            return
        from csv import DictReader

        with open(open_filename, "r") as f:
            r = DictReader(f)
            try:
                k = next(x for x in r.fieldnames if "timestamp" in x.lower())
            except StopIteration:
                import logging

                logging.warn("StopIteration not found")
                return

            unit = pq.ms
            out = {}
            for row in r:
                spikeid = str(row.get("SpikeID", "0"))
                out[spikeid] = out.get(spikeid, []) + [row[k]]

            out2 = []
            for k, v in out.items():
                timestamps = neo.Event(
                    np.array(v, dtype=np.float64), units=unit, name=f"unit_{k}"
                )
                out2.append(tracked_neuron_unit(event=timestamps.rescale(pq.s)))

            self.state.set_data(spike_groups=(self.state.spike_groups or []) + out2)

    def newWindow(self, k, pos=QtAds.TopDockWidgetArea, name=None):
        # TODO: this needs to be smarter. window names must be unique for perspectives. need to be able to recreate old windows by name

        w = self.window_options[k](parent=self, state=self.state)
        window_no = len(
            [
                x
                for x in self.dock_manager.dockWidgets()
                if isinstance(x.widget(), self.window_options[k])
            ]
        )

        if name is None:
            name = k + (f" {window_no}" if window_no > 0 else "")

        w2 = QtAds.CDockWidget(name)
        w2.setWidget(w)
        w2.setFeature(QtAds.CDockWidget.DockWidgetDeleteOnClose, True)
        w2.setFeature(QtAds.CDockWidget.DockWidgetForceCloseWithArea, True)
        self.dock_manager.addDockWidget(QtAds.NoDockWidgetArea, w2)
        self.cur_windows.append(w2)
        w2.show()
        return w

    @Slot()
    def open(self, type=None):
        """
        triggered on file->open
        """
        fname, type = prompt_for_neo_file(type)
        self.loadFile.emit(fname, type)

    @Slot()
    def save_as(
        self,
    ):  # TODO: move out
        """
        triggered on file->save
        """
        fname = QFileDialog.getSaveFileName(self, "Save as", filter="*.h5 *.nwb")[0]
        # s = neo.Segment()
        if fname == "":
            return

        if self.state.segment is None:
            s = neo.Segment()

            s.events.append(self.state.event_signal)
            s.analogsignals.append(self.state.analog_signal)
        else:
            s = self.state.segment
        save_file(
            fname,
            self.state.spike_groups,
            s,
        )


def save_file(filename, spike_groups, data: neo.Segment = None, metadata=None):
    """
    Saves a file containing the spike groups
    """
    if metadata is None:
        metadata = {
            "current_user": environ.get("USERNAME", environ.get("USER")),
            "date": datetime.now(),
        }

    # TODO: we could try just updating the events in an already existing h5
    if filename is None:

        # check if data is an opened file in rw
        # check if analogsignals are in file (by nix_name), if not either add of fail.
        # check if events are in file (by nix_name, and data)
        # remove changed events, add new ones
        pass
    blk = neo.Block(name="main")

    if data is not None:
        from copy import deepcopy

        data2 = deepcopy(data)

        # remove all 'nix_names' which prevent saving the file
        for x in [*data2.analogsignals, *data2.events, data2]:
            if "nix_name" in x.annotations:
                del x.annotations["nix_name"]

        # extract any multianalogsignals into separate segment
        for x in data2.analogsignals:
            if isinstance(x, MultiAnalogSignal):
                data2.analogsignals.remove(x)
                new_seg = neo.Segment("ERP")
                new_seg.analogsignals = x.signals
                blk.segments.append(new_seg)

        # data2.analogsignals = [x.rescale("mV") for x in data2.analogsignals]

        # remove previous unit annotations
        data2.events = [
            x
            for x in data2.events
            if not x.name.startswith("unit") and x.annotations.get("type") != "unit"
        ]
    else:
        data2 = neo.Segment()

    for i, x in enumerate(spike_groups):
        x.event.annotate(**metadata)
        x.event.name = x.event.name or f"unit_{i}"
        data2.events.append(x.event)

    blk.segments.append(data2)
    data2.block = blk
    if Path(filename).exists():
        Path(filename).unlink()  # should probably do a user check here...
    blk.annotations["session_start_time"] = data2.annotations.get(
        "session_start_time", datetime.now()
    )
    output_formats = {"h5": NixIO, "nwb": neo.NWBIO}
    export_class = output_formats[filename.split(".")[-1]]
    n = export_class(filename, mode="rw")
    n.write_block(blk)
    n.close()


def align_spikegroup(spikegroup, erp_arr):
    """given a spike group attempt to align using convolution - note this should go outside of the UI component"""
    # 1. get spike events as 2d matrix
    window = 200
    arr = [
        erp_arr[i, x[0] - window : x[0] + window]
        for i, x in enumerate(spikegroup.idx_arr)
    ]

    # 2. correlate event signals with eachother
    np.convolve(arr, arr, axis=-1)
    # 3. take n-best overlaps & create template

    # 4. align all other signals to this template.
    # TODO
    pass


def run(filename=None):
    app = QApplication.instance() or QApplication(sys.argv)
    icon_path = Path(sys.modules[__name__].__file__).parent.joinpath("ui/icon.svg")
    app.setWindowIcon(QIcon(QPixmap(str(icon_path))))
    # data, signal_chan, event_signal, spike_groups = load_file(
    # )
    # w = TraceView(
    #     analog_signal=signal_chan, event_signal=event_signal, spike_groups=spike_groups
    # )
    w = MdiView()
    if filename is not None:
        w.state.loadFile(filename)
    elif len(sys.argv) > 1:
        w.state.loadFile(sys.argv[1])

    # w = SpikeGroupView()
    w.showMaximized()
    print(app)
    sys.exit(app.exec())


from importlib.metadata import version, metadata, packages_distributions


def run_spikespy(viewerState):
    app = QApplication.instance() or QApplication(sys.argv)
    icon_path = Path(sys.modules[__name__].__file__).parent.joinpath("ui/icon.svg")
    app.setWindowIcon(QIcon(QPixmap(str(icon_path))))
    # data, signal_chan, event_signal, spike_groups = load_file(
    # )
    # w = TraceView(
    #     analog_signal=signal_chan, event_signal=event_signal, spike_groups=spike_groups
    # )
    w = MdiView(state=viewerState)
    w.showMaximized()
    app.exec()

    # w = SpikeGroupView()


class InfoDialog(QDialog):
    # show license info and about

    def __init__(self):
        super().__init__()
        # self.textwidget = QScrollArea()
        self.textarea = QTextEdit()

        self.license_text = self.buid_licence_text()
        self.textarea.setText(self.license_text)
        # self.textwidget.setWidget(self.textarea)
        lo = QVBoxLayout()
        self.setLayout(lo)

        lo.addWidget(QLabel("SpikeSpy"))
        lo.addWidget(QLabel("Open source packages & licenses"))
        lo.addWidget(self.textarea)

    def buid_licence_text(self):
        out = ""
        packages = sorted(list(set(sum(packages_distributions().values(), []))))
        for x in packages:
            md = metadata(x)
            out += f"""## {md.get("name")}
            """
            for k in ["version", "author", "license", "Project-URL"]:
                if k not in md:
                    continue
                out += f"\n{k}: {md.get(k)}"

            out += "\n\n"
        return out


def export_csv(save_filename, unit_evts, stim_evt):
    with open(save_filename, "w", newline="") as f:
        w = writer(f)
        # self.state.segment
        w.writerow(["SpikeID", "Stimulus_number", "Latency_ms", "Timestamp_ms"])
        for i, sg in enumerate(unit_evts):
            nom = sg.name
            if nom is None or nom == "":
                nom = f"unit_{i}"
            for timestamp in sg:
                stim_no = stim_evt.searchsorted(timestamp) - 1
                latency = (timestamp - stim_evt[stim_no]).rescale(pq.ms)

                w.writerow([nom, stim_no, latency.base, timestamp.rescale(pq.ms).base])


def register_plugin(plugin_name, plugin_class):
    suggestors = []
    if isinstance(plugin_class, UnitSuggestor):
        suggestors[plugin_name] = plugin_class
    else:
        window_options[plugin_name] = plugin_class


if __name__ == "__main__":
    run()
