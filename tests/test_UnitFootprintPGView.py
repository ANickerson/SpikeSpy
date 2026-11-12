import pytest
import numpy as np
import neo
import quantities as pq
from PySide6.QtCore import Qt, QPointF

from spikespy.ViewerState import ViewerState, tracked_neuron_unit
from spikespy.UnitFootprintPGView import UnitFootprintPGView
from spikespy.ChannelGroupContainer import ChannelGroupContainerView
from spikespy.ViewSettingsPanel import ViewSettingsPanel


@pytest.fixture()
def multichannel_state():
    state = ViewerState()
    seg = neo.Segment(name="Synthetic Multichannel")
    sr = 10000.0 * pq.Hz
    n_samples = 1000
    n_sweeps = 6

    times = np.arange(n_sweeps) * 0.1 * pq.s
    event_sig = neo.Event(times, name="stim", units=pq.s)
    seg.events.append(event_sig)

    jitters = [0, 1, -1, 0, 1, 0]

    w_ch0 = np.random.normal(0, 0.5, (n_sweeps, n_samples))
    w_ch1 = np.random.normal(0, 0.5, (n_sweeps, n_samples))
    w_ch2 = np.random.normal(0, 0.5, (n_sweeps, n_samples))
    w_ch3 = np.random.normal(0, 0.5, (n_sweeps, n_samples))

    for s_i in range(n_sweeps):
        j = jitters[s_i]
        # Primary unit: CH0 (peak 60 uV at sample 150)
        w_ch0[s_i, 150 + j] = 60.0
        w_ch0[s_i, 150 + j - 1] = 20.0
        w_ch0[s_i, 150 + j + 1] = -15.0

        # Conduction lag +0.3 ms (+3 samples at 10 kHz): CH1
        w_ch1[s_i, 153 + j] = 38.0
        w_ch1[s_i, 153 + j - 1] = 12.0
        w_ch1[s_i, 153 + j + 1] = -10.0

        # Conduction lag +0.6 ms (+6 samples at 10 kHz): CH2
        w_ch2[s_i, 156 + j] = 24.0
        w_ch2[s_i, 156 + j - 1] = 8.0
        w_ch2[s_i, 156 + j + 1] = -6.0

        # Unit 1: secondary unit on CH3 at sample 300
        w_ch3[s_i, 300 + j] = 45.0
        w_ch3[s_i, 300 + j - 1] = 15.0
        w_ch3[s_i, 300 + j + 1] = -10.0

    sig0 = neo.AnalogSignal(w_ch0.flatten(), units=pq.uV, sampling_rate=sr, name="CH0")
    sig1 = neo.AnalogSignal(w_ch1.flatten(), units=pq.uV, sampling_rate=sr, name="CH1")
    sig2 = neo.AnalogSignal(w_ch2.flatten(), units=pq.uV, sampling_rate=sr, name="CH2")
    sig3 = neo.AnalogSignal(w_ch3.flatten(), units=pq.uV, sampling_rate=sr, name="CH3")

    seg.analogsignals.extend([sig0, sig1, sig2, sig3])

    # Unit 0 on CH0
    unit0_spike_times = times + (150 / 10000.0) * pq.s
    evt_unit0 = neo.Event(unit0_spike_times, name="unit_0", units=pq.s)
    sg0 = tracked_neuron_unit(
        event=evt_unit0,
        channel="CH0",
        neighbor_channels=["CH1", "CH2"],
        idx_arr=[(150 + jitters[s_i], 1) for s_i in range(n_sweeps)],
    )

    # Unit 1 on CH3
    unit1_spike_times = times + (300 / 10000.0) * pq.s
    evt_unit1 = neo.Event(unit1_spike_times, name="unit_1", units=pq.s)
    sg1 = tracked_neuron_unit(
        event=evt_unit1,
        channel="CH3",
        neighbor_channels=[],
        idx_arr=[(300 + jitters[s_i], 1) for s_i in range(n_sweeps)],
    )

    state.segment = seg
    state.set_data(analog_signal=sig0, event_signal=event_sig, spike_groups=[sg0, sg1])
    return state


def test_unit_footprint_pgview_init_and_computation(qtbot, multichannel_state):
    view = UnitFootprintPGView(state=multichannel_state)
    qtbot.addWidget(view)
    view.show()

    assert view.state == multichannel_state
    assert view.combo_mode.currentText() == "Single Unit Composite"

    # Compute footprint
    sg0 = multichannel_state.spike_groups[0]
    fp = view.compute_unit_footprint(sg0, last_n=5, align_peaks=True)
    assert fp is not None
    assert fp["mean_matrix"].shape[0] == 4
    assert fp["n_events"] == 5

    channels = fp["channels"]
    ch0_idx = channels.index("CH0")
    ch1_idx = channels.index("CH1")
    ch2_idx = channels.index("CH2")
    ch3_idx = channels.index("CH3")

    # Lag on CH0 is ~0.0 ms
    assert abs(fp["lag_ms"][ch0_idx] - 0.0) < 0.05
    # Lag on CH1 is ~ +0.3 ms
    assert abs(fp["lag_ms"][ch1_idx] - 0.3) < 0.05
    # Lag on CH2 is ~ +0.6 ms
    assert abs(fp["lag_ms"][ch2_idx] - 0.6) < 0.05

    # Peak amplitude is highest on CH0
    assert fp["ptp_amp"][ch0_idx] > fp["ptp_amp"][ch1_idx]
    assert fp["ptp_amp"][ch1_idx] > fp["ptp_amp"][ch2_idx]


def test_unit_footprint_pgview_single_composite_render(qtbot, multichannel_state):
    view = UnitFootprintPGView(state=multichannel_state)
    qtbot.addWidget(view)
    view.show()

    # Force render
    view.render_views()
    assert view.render_count >= 1

    # Check composite elements
    assert view.img_item is not None
    assert view.img_item.image is not None
    assert view.img_item.image.shape == (4, view.img_item.image.shape[1])

    # Check lag markers and trend line
    assert view.lag_scatter is not None
    assert view.trend_line is not None

    # Check bar graph
    assert view.bar_plot is not None
    assert len(view.bars) == 4


def test_unit_footprint_pgview_multi_grid_render(qtbot, multichannel_state):
    view = UnitFootprintPGView(state=multichannel_state)
    qtbot.addWidget(view)
    view.show()

    # Switch to Multi-Unit Grid mode
    view.combo_mode.setCurrentIndex(1)
    view.render_views()

    # Verify grid plots created for all units
    assert len(view.grid_plots) == 2
    plot0, u_idx0, u_name0 = view.grid_plots[0]
    plot1, u_idx1, u_name1 = view.grid_plots[1]
    assert u_idx0 == 0
    assert u_idx1 == 1


def test_unit_footprint_pgview_settings_roundtrip(qtbot, multichannel_state):
    view = UnitFootprintPGView(state=multichannel_state)
    qtbot.addWidget(view)

    view.combo_mode.setCurrentIndex(1)
    view.spin_last_n.setValue(35)
    view.spin_window_ms.setValue(6.5)
    view.chk_align_peaks.setChecked(False)
    view.chk_overlay_traces.setChecked(False)

    settings = view.get_settings()
    assert settings["mode"] == 1
    assert settings["last_n"] == 35
    assert settings["window_ms"] == 6.5
    assert settings["align_peaks"] is False
    assert settings["overlay_traces"] is False

    # Create new view and apply settings
    view2 = UnitFootprintPGView(state=multichannel_state)
    qtbot.addWidget(view2)
    view2.set_settings(settings)

    assert view2.combo_mode.currentIndex() == 1
    assert view2.spin_last_n.value() == 35
    assert view2.spin_window_ms.value() == 6.5
    assert view2.chk_align_peaks.isChecked() is False
    assert view2.chk_overlay_traces.isChecked() is False


def test_unit_footprint_pgview_channel_group_container_integration(qtbot, multichannel_state):
    container = ChannelGroupContainerView(state=multichannel_state)
    qtbot.addWidget(container)
    container.show()

    # Add UnitFootprintPGView via key
    child = container.add_view_by_key("UnitFootprintPG")
    assert child is not None
    assert isinstance(child, UnitFootprintPGView)

    # Child's own channel bar should be hidden inside container
    assert child.channel_bar.isVisible() is False

    # Render child
    child.render_views()
    assert child.render_count >= 1


def test_unit_footprint_pgview_settings_panel_sync(qtbot, multichannel_state):
    view = UnitFootprintPGView(state=multichannel_state)
    qtbot.addWidget(view)
    view.show()

    panel = ViewSettingsPanel(state=multichannel_state)
    qtbot.addWidget(panel)
    panel.show()

    # Set active view
    panel.setActiveView(view)
    assert panel.grp_unit.isVisible() is True
    assert panel.spin_unit_last_n.value() == view.spin_last_n.value()
    assert abs(panel.spin_unit_window.value() - view.spin_window_ms.value()) < 1e-4

    # Change settings via panel
    panel.spin_unit_last_n.setValue(42)
    assert view.spin_last_n.value() == 42

    panel.spin_unit_window.setValue(8.0)
    assert abs(view.spin_window_ms.value() - 8.0) < 1e-4
