import os

import neo
import numpy as np
import pytest
import quantities as pq
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from spikespy.MultiTracePGView import MultiTracePGView
from spikespy.MultiTracePlotManager import MultiTracePlotManager
from spikespy.ViewSettingsPanel import ViewSettingsPanel
from spikespy.ViewerState import ViewerState, tracked_neuron_unit


def _make_state() -> ViewerState:
    state = ViewerState()
    seg = neo.Segment(name="Plot manager test")
    sr = 10000.0 * pq.Hz
    n_samples = 400
    n_sweeps = 5
    times = np.arange(n_sweeps) * 0.1 * pq.s
    event_sig = neo.Event(times, name="stim", units=pq.s)
    seg.events.append(event_sig)

    for i in range(4):
        w = np.zeros((n_sweeps, n_samples), dtype=np.float32)
        w[:, 50 + i * 20] = 40.0 + i * 5.0
        sig = neo.AnalogSignal(
            w.flatten(), units=pq.uV, sampling_rate=sr, name=f"CH{i}"
        )
        seg.analogsignals.append(sig)

    evt0 = neo.Event(times + (50 / 10000.0) * pq.s, name="unit_0", units=pq.s)
    u0 = tracked_neuron_unit(event=evt0, channel="CH0")
    evt1 = neo.Event(times + (130 / 10000.0) * pq.s, name="unit_1", units=pq.s)
    u1 = tracked_neuron_unit(event=evt1, channel="CH1")

    state.segment = seg
    state.set_data(
        analog_signal=seg.analogsignals[0],
        event_signal=event_sig,
        spike_groups=[u0, u1],
    )
    return state


@pytest.fixture()
def plot_state():
    return _make_state()


@pytest.fixture()
def view(plot_state):
    app = QApplication.instance() or QApplication([])
    w = MultiTracePGView(state=plot_state)
    w.show()
    app.processEvents()
    yield w
    w.close()
    app.processEvents()


def _find_item_key(manager, key):
    for i in range(manager.list_plots.count()):
        if manager.list_plots.item(i).data(Qt.UserRole) == key:
            return i
    return -1


def test_get_plot_specs_lists_channel_and_summary_plots(view):
    view.set_group_size_n(2)
    view.add_channel_to_display("CH1")
    view.setup_figure()

    specs = view.get_plot_specs()
    channels = [s for s in specs if s["kind"] == "channel"]
    assert [s["slot_idx"] for s in channels] == [0, 1]
    assert channels[0]["channel_name"] == "CH0"

    summary_keys = {s["key"] for s in specs if s["kind"] == "summary"}
    assert "CH0" in summary_keys
    assert "CH1" in summary_keys
    assert "Stim. Freq. (Hz)" in summary_keys
    assert "Latency Diff." in summary_keys

    ch_summary = next(s for s in specs if s["key"] == "CH0")
    assert ch_summary["is_channel_stat"] is True
    assert ch_summary["visible"] is True


def test_plot_slot_add_move_replace_and_remove(view):
    view.set_group_size_n(2)
    view.add_channel_to_display("CH1")
    assert view.displayed_channels == ["CH0", "CH1"]

    view.add_channel_to_display("CH2")
    assert view.displayed_channels == ["CH0", "CH1", "CH2"]

    view.move_plot_slot(0, 2)
    assert view.displayed_channels == ["CH1", "CH2", "CH0"]

    view.set_plot_slot_channel(0, "CH3")
    assert view.displayed_channels[0] == "CH3"

    view.set_plot_slot_options(
        0,
        pinned_unit_idx=0,
        window_group="Group A",
        window_width_ms=12.5,
    )
    panel = view.channel_panels[0]
    assert panel.pinned_unit_idx == 0
    assert panel.window_group == "Group A"
    assert panel.window_width_ms == 12.5
    assert panel.x_linked is False

    view.remove_plot_slot(0)
    assert view.displayed_channels == ["CH2", "CH0"]
    assert view.group_size_n == 2


def test_set_plot_visible_toggles_channel_and_global_summaries(view):
    view.set_group_size_n(2)
    view.add_channel_to_display("CH1")

    view.set_plot_visible("CH0", False)
    assert view.rightPlots["CH0"] is False
    spec = next(s for s in view.get_plot_specs() if s["key"] == "CH0")
    assert spec["visible"] is False

    view.set_plot_visible("__channel_summaries__", False)
    assert view.rightPlots["__channel_summaries__"] is False
    assert not [
        s for s in view.get_plot_specs() if s.get("is_channel_stat")
    ]

    view.set_plot_visible("Stim. Freq. (Hz)", False)
    assert view.rightPlots["Stim. Freq. (Hz)"] is False


def test_plot_manager_lists_and_edits_plot_params(view):
    app = QApplication.instance() or QApplication([])
    view.set_group_size_n(2)
    view.add_channel_to_display("CH1")

    manager = MultiTracePlotManager(view=view)
    manager.show()
    app.processEvents()

    assert manager.list_plots.count() == len(view.get_plot_specs())

    row = _find_item_key(manager, "channel:1")
    assert row >= 0
    manager.list_plots.setCurrentRow(row)
    app.processEvents()
    assert manager.combo_channel.isEnabled()
    assert manager.combo_pin.isEnabled()

    manager.spin_width.setValue(33.0)
    app.processEvents()
    assert view.channel_panels[1].window_width_ms == pytest.approx(33.0)

    pin_row = manager.combo_pin.findData(1)
    assert pin_row >= 0
    manager.combo_pin.setCurrentIndex(pin_row)
    app.processEvents()
    assert view.channel_panels[1].pinned_unit_idx == 1

    # Summary plots can be toggled from the list.
    row = _find_item_key(manager, "CH1")
    assert row >= 0
    manager.list_plots.setCurrentRow(row)
    app.processEvents()
    manager.chk_visible.setChecked(False)
    app.processEvents()
    assert view.rightPlots["CH1"] is False
    assert manager.list_plots.item(row).checkState() == Qt.Unchecked

    # View-side additions are reflected in an open manager.
    before = manager.list_plots.count()
    view.add_channel_to_display("CH2")
    app.processEvents()
    assert manager.list_plots.count() > before

    manager.close()
    app.processEvents()


def test_view_settings_panel_embeds_plot_manager(plot_state):
    app = QApplication.instance() or QApplication([])
    panel = ViewSettingsPanel(state=plot_state)
    panel.show()
    app.processEvents()

    view = MultiTracePGView(state=plot_state)
    view.show()
    app.processEvents()

    panel.setActiveView(view)
    app.processEvents()

    assert panel.grp_multitrace.isVisible()
    assert panel.mt_plot_manager.isVisible()
    assert panel.mt_plot_manager.list_plots.count() == len(view.get_plot_specs())

    panel.setActiveView(None)
    app.processEvents()
    assert not panel.mt_plot_manager.isVisible()

    panel.close()
    view.close()
    app.processEvents()
