import os

import neo
import numpy as np
import pytest
import quantities as pq
from PySide6.QtWidgets import QApplication

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from spikespy.ActiveUnitInfoView import ActiveUnitInfoView
from spikespy.ChannelGroupBar import ChannelGroupBar
from spikespy.ChannelGroupContainer import ChannelGroupContainerView
from spikespy.MultiTracePGView import MultiTracePGView
from spikespy.SharedChannelPanel import SharedChannelPanel
from spikespy.SingleTracePGView import SingleTracePGView
from spikespy.SpikeSpy import MdiView
from spikespy.ViewSettingsPanel import ViewSettingsPanel
from spikespy.ViewerState import ViewerState, tracked_neuron_unit


@pytest.fixture
def two_unit_state():
    state = ViewerState()
    seg = neo.Segment(name="TestSegment")
    sr = 10000.0 * pq.Hz
    n_samples = 500
    n_sweeps = 5
    times = np.arange(n_sweeps) * 0.2 * pq.s
    event_sig = neo.Event(times, name="stim", units=pq.s)
    seg.events.append(event_sig)

    for i in range(4):
        w = np.zeros((n_sweeps, n_samples), dtype=np.float32)
        w[:, 100] = 0.5 / (i + 1)
        sig = neo.AnalogSignal(
            w.flatten(), units=pq.mV, sampling_rate=sr, name=f"CH{i}"
        )
        seg.analogsignals.append(sig)

    evt0 = neo.Event(times + 0.01 * pq.s, name="unit_alpha", units=pq.s)
    evt0.annotations["channel"] = "CH0"
    evt0.annotations["neighbor_channels"] = ["CH1"]
    u0 = tracked_neuron_unit(event=evt0, channel="CH0", neighbor_channels=["CH1"])

    evt1 = neo.Event(times + 0.02 * pq.s, name="unit_beta", units=pq.s)
    evt1.annotations["channel"] = "CH2"
    evt1.annotations["neighbor_channels"] = ["CH3"]
    u1 = tracked_neuron_unit(event=evt1, channel="CH2", neighbor_channels=["CH3"])

    state.segment = seg
    state.set_data(
        analog_signal=seg.analogsignals[0],
        event_signal=event_sig,
        spike_groups=[u0, u1],
    )
    return state


def test_view_settings_panel_embeds_active_unit_info_and_tracks_bound_unit(
    qtbot, two_unit_state
):
    panel = ViewSettingsPanel(state=two_unit_state)
    qtbot.addWidget(panel)
    panel.show()

    assert isinstance(panel.unit_info_widget, ActiveUnitInfoView)
    assert panel.unit_info_widget._compact is True
    assert panel.grp_active_unit_info.isVisible() is True

    # Two views bound to different units
    v0 = MultiTracePGView(state=two_unit_state)
    v1 = MultiTracePGView(state=two_unit_state)
    qtbot.addWidget(v0)
    qtbot.addWidget(v1)
    v0.channel_bar.bind_to_unit(0)
    v1.channel_bar.bind_to_unit(1)

    panel.setActiveView(v0)
    assert panel.unit_info_widget.get_target_unit_idx() == 0
    assert panel.unit_info_widget.edit_name.text() == "unit_alpha"

    panel.setActiveView(v1)
    assert panel.unit_info_widget.get_target_unit_idx() == 1
    assert panel.unit_info_widget.edit_name.text() == "unit_beta"

    # Changing unit selection in ViewSettingsPanel updates unit_info_widget
    u0_idx = panel.combo_unit.findData(0)
    assert u0_idx >= 0
    panel.combo_unit.setCurrentIndex(u0_idx)
    assert panel.unit_info_widget.get_target_unit_idx() == 0
    assert panel.unit_info_widget.edit_name.text() == "unit_alpha"

    # Unbound view follows global active unit
    v_unbound = MultiTracePGView(state=two_unit_state)
    qtbot.addWidget(v_unbound)
    panel.setActiveView(v_unbound)
    two_unit_state.setUnitGroup(1)
    assert panel.unit_info_widget.get_target_unit_idx() == 1
    assert panel.unit_info_widget.edit_name.text() == "unit_beta"

    # Changing state propagates to embedded unit_info_widget
    new_state = ViewerState()
    panel.set_state(new_state)
    assert panel.unit_info_widget.state is new_state


def test_view_settings_panel_syncs_visuals_in_container_and_shared_set(
    qtbot, two_unit_state
):
    panel = ViewSettingsPanel(state=two_unit_state)
    qtbot.addWidget(panel)
    panel.show()
    assert panel.chk_sync_group_visuals.isChecked() is True

    # 1. Sibling views inside ChannelGroupContainerView
    container = ChannelGroupContainerView(state=two_unit_state)
    qtbot.addWidget(container)
    assert container.chk_sync_visuals.isChecked() is True

    mt1 = MultiTracePGView(parent=container, state=two_unit_state)
    mt2 = MultiTracePGView(parent=container, state=two_unit_state)
    st1 = SingleTracePGView(parent=container, state=two_unit_state)
    container.add_view(mt1, title="MT1")
    container.add_view(mt2, title="MT2")
    container.add_view(st1, title="ST1")

    panel.setActiveView(mt1)
    QApplication.processEvents()

    # Change scale_mode and color scale via ViewSettingsPanel
    panel.combo_mt_scale_mode.setCurrentText("mV")
    panel.spin_mt_lower.setValue(0.25)
    panel.spin_mt_upper.setValue(1.75)
    QApplication.processEvents()

    assert mt1.scale_mode == "mV"
    assert mt2.scale_mode == "mV"
    assert mt2.lowerSpinBox.value() == pytest.approx(0.25)
    assert mt2.upperSpinBox.value() == pytest.approx(1.75)
    assert container.view_state.get_visual_settings()["scale_mode"] == "mV"
    assert container.view_state.get_visual_settings()["color_scale"] == pytest.approx(
        (0.25, 1.75)
    )

    # Change X range via ViewSettingsPanel
    panel.spin_xlim_min.setValue(50.0)
    panel.spin_xlim_max.setValue(250.0)
    panel._on_apply_limits_clicked()
    QApplication.processEvents()

    assert container.view_state.get_visual_settings()["x_range"] == pytest.approx(
        (50.0, 250.0)
    )
    (st_xmin, st_xmax), _ = st1.get_view_range()
    assert st_xmin == pytest.approx(50.0, abs=1.0)
    assert st_xmax == pytest.approx(250.0, abs=1.0)

    # Newly added view receives container visual settings automatically
    mt3 = MultiTracePGView(parent=container, state=two_unit_state)
    container.add_view(mt3, title="MT3")
    assert mt3.scale_mode == "mV"
    assert mt3.lowerSpinBox.value() == pytest.approx(0.25)
    assert mt3.upperSpinBox.value() == pytest.approx(1.75)

    # Container broadcast_visual_settings updates all children & persists sync_visuals
    container.broadcast_visual_settings(scale_mode="std", color_scale=(0.4, 2.2))
    assert mt1.scale_mode == "std"
    assert mt2.scale_mode == "std"
    assert mt3.lowerSpinBox.value() == pytest.approx(0.4)
    assert mt3.upperSpinBox.value() == pytest.approx(2.2)

    container.chk_sync_visuals.setChecked(False)
    sett = container.get_settings()
    assert sett["sync_visuals"] is False
    container2 = ChannelGroupContainerView(state=two_unit_state)
    qtbot.addWidget(container2)
    container2.set_settings({"sync_visuals": False})
    assert container2.chk_sync_visuals.isChecked() is False

    # 2. Views bound to Shared Group A
    shared_a1 = MultiTracePGView(state=two_unit_state)
    shared_a2 = MultiTracePGView(state=two_unit_state)
    other_b = MultiTracePGView(state=two_unit_state)
    qtbot.addWidget(shared_a1)
    qtbot.addWidget(shared_a2)
    qtbot.addWidget(other_b)

    shared_a1.channel_bar.bind_to_shared_set("Shared Group A")
    shared_a2.channel_bar.bind_to_shared_set("Shared Group A")
    other_b.channel_bar.bind_to_shared_set("Shared Group B")

    panel.setActiveView(shared_a1)
    panel.combo_mt_scale_mode.setCurrentText("mV")
    panel.spin_mt_lower.setValue(0.3)
    panel.spin_mt_upper.setValue(1.9)
    QApplication.processEvents()

    assert shared_a2.scale_mode == "mV"
    assert shared_a2.lowerSpinBox.value() == pytest.approx(0.3)
    assert shared_a2.upperSpinBox.value() == pytest.approx(1.9)
    assert other_b.scale_mode == "std"

    # Unchecking chk_sync_group_visuals stops broadcasting
    panel.chk_sync_group_visuals.setChecked(False)
    panel.combo_mt_scale_mode.setCurrentText("std")
    QApplication.processEvents()
    assert shared_a1.scale_mode == "std"
    assert shared_a2.scale_mode == "mV"


def test_shared_channel_panel_broadcasts_visual_settings_to_linked_viewers(
    qtbot, two_unit_state
):
    win = MdiView(state=two_unit_state, check_updates=False)
    qtbot.addWidget(win)
    win.show()

    v_a = win.newWindow("MultiTrace (PyQtGraph)")
    v_b = win.newWindow("MultiTrace (PyQtGraph)")

    v_a.channel_bar.bind_to_shared_set("Shared Group A")
    v_b.channel_bar.bind_to_shared_set("Shared Group B")

    scp = win.open_or_show_shared_channel_panel()
    QApplication.processEvents()

    assert scp.get_active_set() == "Shared Group A"

    scp.combo_shared_scale_mode.setCurrentText("mV")
    scp.spin_shared_min_scale.setValue(0.15)
    scp.spin_shared_max_scale.setValue(1.45)
    scp.btn_sync_visuals_now.click()
    QApplication.processEvents()

    shared_vis = two_unit_state.get_shared_visual_settings("Shared Group A")
    assert shared_vis["scale_mode"] == "mV"
    assert shared_vis["color_scale"] == pytest.approx((0.15, 1.45))

    assert v_a.scale_mode == "mV"
    assert v_a.lowerSpinBox.value() == pytest.approx(0.15)
    assert v_a.upperSpinBox.value() == pytest.approx(1.45)

    # Shared Group B viewer should remain untouched
    assert v_b.scale_mode == "std"
    assert v_b.lowerSpinBox.value() == pytest.approx(0.5)
    assert v_b.upperSpinBox.value() == pytest.approx(2.0)

    win.close()
