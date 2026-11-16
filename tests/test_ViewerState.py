import pytest
from spikespy.ViewerState import ViewerState


@pytest.fixture()
def state():
    import os
    fname = r"data/test2.h5"
    if not os.path.exists(fname):
        pytest.skip(f"Test data file {fname} not found")
    state = ViewerState()
    state.loadFile(fname)
    return state

def test_undo(state: ViewerState):
    state.setStimNo(1)
    state.setUnit(300)
    unit_group = state.getUnitGroup()
    cur_latency = unit_group.idx_arr[state.stimno][0]
    assert cur_latency == 300
    state.setUnit(100)
    unit_group = state.getUnitGroup()
    new_latency = unit_group.idx_arr[state.stimno][0]
    assert cur_latency != new_latency
    assert new_latency == 100

    state.undo()
    unit_group = state.getUnitGroup()
    undo_latency = unit_group.idx_arr[state.stimno][0]
    assert cur_latency == undo_latency


@pytest.fixture()
def h5_path(tmp_path):
    import numpy as np
    import quantities as pq
    import neo
    from neo.io import NixIO

    seg = neo.Segment(name="seg0")
    seg.analogsignals.append(
        neo.AnalogSignal(
            np.random.randn(1000, 1) * pq.mV, sampling_rate=1000 * pq.Hz, name="CH0"
        )
    )
    seg.events.append(neo.Event([0.1, 0.2, 0.3] * pq.s, name="stim"))
    blk = neo.Block(name="test")
    blk.segments.append(seg)
    path = tmp_path / "async_test.h5"
    NixIO(str(path), mode="ow").write_block(blk)
    return str(path)


def test_loadFile_async_non_blocking(qtbot, h5_path):
    state = ViewerState()
    with qtbot.waitSignal(state.onFileLoaded, timeout=30000) as blocker:
        state.loadFile(h5_path, blocking=False)
    assert blocker.args == [h5_path, "h5"]
    assert state.title == h5_path
    assert state.analog_signal is not None
    assert state.event_signal is not None
    assert state._file_load_worker is None


def test_loadFile_async_cancel_discards(qtbot, h5_path):
    state = ViewerState()
    state.loadFile(h5_path, blocking=False)
    state._file_load_cancelled = True
    qtbot.waitUntil(lambda: state._file_load_worker is None, timeout=30000)
    assert state.title != h5_path


def test_loadFile_async_failure_is_reported(qtbot, tmp_path):
    state = ViewerState()
    missing = str(tmp_path / "does_not_exist.h5")
    state.loadFile(missing, blocking=False)
    qtbot.waitUntil(lambda: state._file_load_worker is None, timeout=30000)
    # a failed load must not be applied
    assert state.title != missing


def test_infer_file_type():
    from spikespy.io_manager import infer_file_type

    assert infer_file_type("rec.h5") == "h5"
    assert infer_file_type("rec.HDF5") == "h5"
    assert infer_file_type("rec.nwb") == "nwb"
    assert infer_file_type("rec.smr") == "spike2"
    assert infer_file_type("rec.smrx") == "spike2"
    assert infer_file_type("rec.unknown") is None


def test_infer_folder_type(tmp_path):
    from spikespy.io_manager import infer_folder_type

    (tmp_path / "structure.oebin").touch()
    assert infer_folder_type(tmp_path) == "openEphys"

    oe = tmp_path / "oe"
    oe.mkdir()
    (oe / "settings.xml").touch()
    assert infer_folder_type(oe) == "openEphys"

    apt = tmp_path / "apt"
    apt.mkdir()
    (apt / "101_CH1.continuous").touch()
    assert infer_folder_type(apt) == "APTrack"

    mat = tmp_path / "mat"
    mat.mkdir()
    (mat / "chan.mat").touch()
    assert infer_folder_type(mat) == "matlab"

    empty = tmp_path / "empty"
    empty.mkdir()
    assert infer_folder_type(empty) is None


def _force_real_progress_dialog(state):
    """Build a real QProgressDialog even offscreen, mirroring the production setup."""
    from pathlib import Path
    from PySide6.QtWidgets import QApplication, QProgressDialog
    from PySide6.QtCore import Qt

    def make(fname):
        dlg = QProgressDialog(
            f"Loading {Path(fname).name}...", "Cancel", 0, 0, QApplication.activeWindow()
        )
        dlg.setWindowTitle("Loading")
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setMinimumDuration(0)
        dlg.setAutoClose(False)
        dlg.setAutoReset(False)
        dlg.canceled.connect(state._on_load_cancelled)
        dlg.show()
        return dlg

    return make


def test_loadFile_async_with_progress_dialog_applies_data(qtbot, h5_path):
    # Regression: QProgressDialog.reset()/close() emit canceled(), which used to
    # mark the load as cancelled and silently discard the freshly loaded data.
    state = ViewerState()
    state._make_load_progress = _force_real_progress_dialog(state)
    with qtbot.waitSignal(state.onFileLoaded, timeout=30000):
        state.loadFile(h5_path, blocking=False)
    assert state.title == h5_path
    assert state.analog_signal is not None
    assert state.event_signal is not None
    assert state._file_load_cancelled is False
    assert state._file_load_progress is None


@pytest.fixture()
def test3_path():
    import os
    fname = r"data/test3.h5"
    if not os.path.exists(fname):
        pytest.skip(f"Test data file {fname} not found")
    return fname


def test_loadFile_async_real_file_with_progress_dialog(qtbot, test3_path):
    state = ViewerState()
    state._make_load_progress = _force_real_progress_dialog(state)
    with qtbot.waitSignal(state.onFileLoaded, timeout=120000) as blocker:
        state.loadFile(test3_path, blocking=False)
    assert blocker.args == [test3_path, "h5"]
    assert state.title == test3_path
    assert state.analog_signal is not None
    assert state.analog_signal.shape[0] > 0
    assert state._file_load_cancelled is False
    assert state._file_load_progress is None
    assert state._file_load_worker is None
