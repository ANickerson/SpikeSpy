import numpy as np
import pytest
import quantities as pq
import neo
from PySide6.QtCore import Qt, QPoint
from PySide6.QtWidgets import QApplication

from spikespy.ViewerState import ViewerState, tracked_neuron_unit
from spikespy.SingleTraceView import SingleTraceView
from spikespy.UnitView import UnitView
from spikespy.SpikeGroupTable import SpikeGroupTableView, SpikeGroupTableModel
from spikespy.TrackingView import TrackingView
from spikespy.processing import LiveERPBuffer


@pytest.fixture
def mock_multichannel_state():
    state = ViewerState()
    # Create segment with multiple analog signals
    seg = neo.Segment(name="Test Segment")
    sr = 10000.0 * pq.Hz
    n_samples = 500
    n_sweeps = 5
    
    # Event signal
    times = np.arange(n_sweeps) * 0.1 * pq.s
    event_sig = neo.Event(times, name="stim", units=pq.s)
    seg.events.append(event_sig)

    # Primary channel
    w_ch0 = np.zeros((n_sweeps, n_samples))
    w_ch0[:, 100] = 50.0  # Spike at 100
    sig0 = neo.AnalogSignal(w_ch0.flatten(), units=pq.uV, sampling_rate=sr, name="CH0")
    seg.analogsignals.append(sig0)

    # Neighbor channel 1
    w_ch1 = np.zeros((n_sweeps, n_samples))
    w_ch1[:, 100] = 30.0
    sig1 = neo.AnalogSignal(w_ch1.flatten(), units=pq.uV, sampling_rate=sr, name="CH1")
    seg.analogsignals.append(sig1)

    # Neighbor channel 2
    w_ch2 = np.zeros((n_sweeps, n_samples))
    w_ch2[:, 100] = 20.0
    sig2 = neo.AnalogSignal(w_ch2.flatten(), units=pq.uV, sampling_rate=sr, name="CH2")
    seg.analogsignals.append(sig2)

    # Non-neural aux channel
    sig_env = neo.AnalogSignal(np.ones(n_sweeps * n_samples), units=pq.V, sampling_rate=sr, name="env.stim")
    seg.analogsignals.append(sig_env)

    # Unit group with channel and neighbor_channels
    evt_unit = neo.Event(times + (100 / 10000.0) * pq.s, name="unit_0", units=pq.s)
    evt_unit.annotations["channel"] = "CH0"
    evt_unit.annotations["neighbor_channels"] = ["CH1", "CH2"]
    evt_unit.annotations["fibre_type"] = "Cmh"
    evt_unit.annotations["notes"] = "Test note"
    sg = tracked_neuron_unit(
        event=evt_unit,
        channel="CH0",
        neighbor_channels=["CH1", "CH2"],
        fibre_type="Cmh",
        notes="Test note",
    )
    
    state.segment = seg
    state.set_data(analog_signal=sig0, event_signal=event_sig, spike_groups=[sg])
    return state


def test_single_trace_view_multichannel_overlay(qtbot, mock_multichannel_state):
    state = mock_multichannel_state
    view = SingleTraceView(state=state)
    qtbot.addWidget(view)

    # Default is not multichannel
    assert not view.chk_multichannel.isChecked()
    assert len(view.get_overlay_channels()) == 0

    # Turn on multichannel
    view.chk_multichannel.setChecked(True)
    overlay_chans = view.get_overlay_channels()
    # Should exclude primary "CH0", and exclude "env.stim" (aux), so ["CH1", "CH2"]
    assert "CH1" in overlay_chans
    assert "CH2" in overlay_chans
    assert "CH0" not in overlay_chans
    assert "env.stim" not in overlay_chans

    # Unit channels only checkbox
    view.chk_unit_neighbors_only.setChecked(True)
    overlay_chans_unit = view.get_overlay_channels()
    assert "CH1" in overlay_chans_unit
    assert "CH2" in overlay_chans_unit

    # Change vertical offset
    view.spin_chan_spacing.setValue(100)
    assert view.spin_chan_spacing.value() == 100

    # Ensure updateFigure ran and aux artists exist
    view.updateFigure_core()
    assert len(view.aux_trace_lines) >= 2

    # Verify Multi-Channel Conv checkbox and SweepConvProxy
    assert hasattr(view, "chk_multichannel_conv")
    assert view.chk_multichannel_conv.isChecked()
    assert hasattr(view, "conv")
    # Proxy computes on demand
    conv_sweep = view.conv[state.stimno]
    assert isinstance(conv_sweep, np.ndarray)
    assert conv_sweep.ndim == 1
    assert len(conv_sweep) == mock_multichannel_state.analog_signal_erp.shape[1]
    assert view.conv[state.stimno] is conv_sweep  # Cached in LRU



def test_unit_view_neighbor_channels(qtbot, mock_multichannel_state):
    state = mock_multichannel_state
    view = UnitView(state=state)
    qtbot.addWidget(view)

    assert not view.chk_show_neighbors.isChecked()
    view.chk_show_neighbors.setChecked(True)
    assert view.chk_show_neighbors.isChecked()
    # updateAll triggered on check
    view.updateAll()
    assert view.axes["main"] is not None


def test_spike_group_table_channel_and_neighbors(qtbot, mock_multichannel_state):
    state = mock_multichannel_state
    tbl_view = SpikeGroupTableView(state=state)
    qtbot.addWidget(tbl_view)

    model = tbl_view.spike_tablemodel
    # Check headers
    assert model.headers[6] == "channel"
    assert model.headers[7] == "neighbors"

    # Check data display
    idx_chan = model.index(0, 6)
    idx_neigh = model.index(0, 7)
    assert model.data(idx_chan, Qt.DisplayRole) == "CH0"
    assert "CH1" in model.data(idx_neigh, Qt.DisplayRole)
    assert "CH2" in model.data(idx_neigh, Qt.DisplayRole)

    # Test editing channel column
    model.setData(idx_chan, "CH1", Qt.EditRole)
    assert model.data(idx_chan, Qt.DisplayRole) == "CH1"
    assert state.getUnitGroup().channel == "CH1"
    assert state.getUnitGroup().event.annotations["channel"] == "CH1"

    # Test editing neighbors column
    model.setData(idx_neigh, "CH0, CH2", Qt.EditRole)
    assert state.getUnitGroup().neighbor_channels == ["CH0", "CH2"]
    assert state.getUnitGroup().event.annotations["neighbor_channels"] == ["CH0", "CH2"]

    # Test double click channel activation
    assert hasattr(state, "set_active_channel")
    tbl_view.on_cell_double_clicked(idx_chan)
    # Primary channel should now be switched to CH1
    assert state.analog_signal.name == "CH1"


def test_offline_2d_multichannel_expansion_and_neo_settings_view(qtbot):
    """Verify 2D multi-channel AnalogSignals are automatically expanded into selectable per-channel signals."""
    from spikespy.NeoSettingsView import NeoSettingsView

    seg = neo.Segment(name="MultiChan2D")
    sr = 10000.0 * pq.Hz
    data_2d = np.column_stack([
        np.ones(1000) * 1.0,
        np.ones(1000) * 2.0,
        np.ones(1000) * 3.0,
    ])
    sig_2d = neo.AnalogSignal(data_2d, units=pq.uV, sampling_rate=sr, name="Probe")
    sig_2d.array_annotations["channel_names"] = np.array(["Tetrode_0", "Tetrode_1", "Tetrode_2"])
    seg.analogsignals.append(sig_2d)
    seg.events.append(neo.Event(np.array([0.01, 0.05]) * pq.s, name="stim"))

    state = ViewerState()
    state.set_segment(seg)

    avail = state.get_available_channels()
    assert avail == ["Tetrode_0", "Tetrode_1", "Tetrode_2"]
    assert state.analog_signal.name == "Tetrode_0"

    settings_view = NeoSettingsView(state=state)
    qtbot.addWidget(settings_view)
    assert settings_view.analogSignalCombo.count() == 3

    # Test auto-switch to unit's channel when selecting a unit group
    settings_view.chk_auto_unit_chan.setChecked(True)
    assert state.auto_switch_unit_channel is True

    state.addUnitGroup()
    state.spike_groups[1].channel = "Tetrode_2"
    state.setUnitGroup(1)
    assert state.analog_signal.name == "Tetrode_2"
    assert settings_view.analogSignalCombo.currentText() == "Tetrode_2"


def test_unit_footprint_view_and_lag_alignment(qtbot):
    from spikespy.UnitFootprintView import UnitFootprintView

    state = ViewerState()
    seg = neo.Segment(name="Synthetic Multichannel")
    sr = 10000.0 * pq.Hz
    n_samples = 1000
    n_sweeps = 6

    # Sweeps event signal
    times = np.arange(n_sweeps) * 0.1 * pq.s
    event_sig = neo.Event(times, name="stim", units=pq.s)
    seg.events.append(event_sig)

    # Base nominal latency is sample 150
    # Add small sweep trigger jitter: ±1 sample
    jitters = [0, 1, -1, 0, 1, 0]

    # Create 4 channels:
    # CH0: Primary unit, peak 60.0 uV at 150 + jitter
    # CH1: Conduction lag +0.3 ms (+3 samples at 10 kHz), peak 38.0 uV at 153 + jitter
    # CH2: Conduction lag +0.6 ms (+6 samples at 10 kHz), peak 24.0 uV at 156 + jitter
    # CH3: Noise only (peak ~2 uV)
    w_ch0 = np.random.normal(0, 0.5, (n_sweeps, n_samples))
    w_ch1 = np.random.normal(0, 0.5, (n_sweeps, n_samples))
    w_ch2 = np.random.normal(0, 0.5, (n_sweeps, n_samples))
    w_ch3 = np.random.normal(0, 0.5, (n_sweeps, n_samples))

    for s_i in range(n_sweeps):
        j = jitters[s_i]
        # Biphasic/negative action potential waveform shape
        w_ch0[s_i, 150 + j] = 60.0
        w_ch0[s_i, 150 + j - 1] = 20.0
        w_ch0[s_i, 150 + j + 1] = -15.0

        w_ch1[s_i, 153 + j] = 38.0
        w_ch1[s_i, 153 + j - 1] = 12.0
        w_ch1[s_i, 153 + j + 1] = -10.0

        w_ch2[s_i, 156 + j] = 24.0
        w_ch2[s_i, 156 + j - 1] = 8.0
        w_ch2[s_i, 156 + j + 1] = -6.0

    sig0 = neo.AnalogSignal(w_ch0.flatten(), units=pq.uV, sampling_rate=sr, name="CH0")
    sig1 = neo.AnalogSignal(w_ch1.flatten(), units=pq.uV, sampling_rate=sr, name="CH1")
    sig2 = neo.AnalogSignal(w_ch2.flatten(), units=pq.uV, sampling_rate=sr, name="CH2")
    sig3 = neo.AnalogSignal(w_ch3.flatten(), units=pq.uV, sampling_rate=sr, name="CH3")

    seg.analogsignals.extend([sig0, sig1, sig2, sig3])

    # Marked latency has nominal center at 150 (with the sweep jitter)
    unit_spike_times = times + (150 / 10000.0) * pq.s
    evt_unit = neo.Event(unit_spike_times, name="unit_0", units=pq.s)
    sg = tracked_neuron_unit(
        event=evt_unit,
        channel="CH0",
        neighbor_channels=[],
        idx_arr=[(150 + jitters[s_i], 1) for s_i in range(n_sweeps)],
    )

    state.segment = seg
    state.set_data(analog_signal=sig0, event_signal=event_sig, spike_groups=[sg])

    view = UnitFootprintView(state=state)
    qtbot.addWidget(view)

    # Compute unit footprint with peak alignment
    fp = view.compute_unit_footprint(state.getUnitGroup(), last_n=5, align_peaks=True)
    assert fp is not None
    assert fp["mean_matrix"].shape[0] == 4
    assert fp["n_events"] == 5

    # Check channels and conduction lag
    channels = fp["channels"]
    ch0_idx = channels.index("CH0")
    ch1_idx = channels.index("CH1")
    ch2_idx = channels.index("CH2")
    ch3_idx = channels.index("CH3")

    # Lag on CH0 is 0.0 ms (reference channel aligned to peak)
    assert abs(fp["lag_ms"][ch0_idx] - 0.0) < 0.05
    # Lag on CH1 is ~ +0.3 ms (+3 samples / 10 kHz = +0.3 ms)
    assert abs(fp["lag_ms"][ch1_idx] - 0.3) < 0.05
    # Lag on CH2 is ~ +0.6 ms (+6 samples / 10 kHz = +0.6 ms)
    assert abs(fp["lag_ms"][ch2_idx] - 0.6) < 0.05

    # Detected channels: CH0, CH1, CH2; CH3 (noise) has much lower ptp_amp
    assert "CH0" in fp["detected_channels"]
    assert "CH1" in fp["detected_channels"]
    assert "CH2" in fp["detected_channels"]
    assert "CH3" not in fp["detected_channels"]
    assert fp["ptp_amp"][ch3_idx] < 0.25 * fp["ptp_amp"][ch0_idx]

    # Test auto_detect_unit_channels
    view.auto_detect_unit_channels()
    assert sg.channel == "CH0"
    assert "CH1" in sg.neighbor_channels
    assert "CH2" in sg.neighbor_channels
    assert "CH3" not in sg.neighbor_channels
    assert sg.event.annotations["channel"] == "CH0"
    assert "CH1" in sg.event.annotations["neighbor_channels"]
    assert "CH2" in sg.event.annotations["neighbor_channels"]

    # Test settings persistence
    settings = view.get_settings()
    assert settings["align_peaks"] is True
    assert settings["mode"] == 0
    settings["mode"] = 1
    view.set_settings(settings)
    assert view.combo_mode.currentIndex() == 1

    # Switch view mode to "Multi-Unit Grid" and verify rendering succeeds without error
    view.combo_mode.setCurrentIndex(1)
    view.update_plot()
    assert len(view.figure.axes) >= 1


def test_multi_trace_view_vectorized_spike_markers(qtbot, mock_multichannel_state):
    from spikespy.MultiTraceView import MultiTraceView
    from matplotlib.lines import Line2D

    state = mock_multichannel_state
    view = MultiTraceView(state=state)
    qtbot.addWidget(view)

    # Initial plot of spike groups
    artists = view.plot_spikegroups()
    assert hasattr(view, "Patches")
    assert len(view.Patches) >= 1
    # Check that each patch artist is a vectorized Line2D rather than Rectangle/PatchCollection
    for unit_idx, artist in view.Patches.items():
        assert isinstance(artist, Line2D)
        assert artist.get_linestyle() == "None"
        assert artist.get_marker() == "s"

    # Switching sweeps or re-rendering should update existing Line2D artists in place
    line_before = view.Patches[state.cur_spike_group]
    view.render()
    line_after = view.Patches[state.cur_spike_group]
    assert line_before is line_after

    # Test settings dialog persistence
    assert view.settingsDialog is not None
    dlg_before = view.settingsDialog
    view.reset_right_axes_data()
    # Segment did not change, so settingsDialog should be preserved
    assert view.settingsDialog is dlg_before


def test_streaming_rendering_optimizations(qtbot, mock_multichannel_state):
    from spikespy.processing import LiveERPBuffer
    from spikespy.SingleTraceView import SingleTraceView
    from spikespy.CachedAxesImage import CachedAxesImage
    from spikespy.MultiTraceView import MultiTraceView
    import matplotlib.pyplot as plt

    # 1. Test LiveERPBuffer properties and incremental stats
    buf = LiveERPBuffer(window_samples=100, initial_capacity=16, units=pq.uV)
    sweep1 = np.ones(100, dtype=np.float32) * 5.0
    sweep1[10] = -3.0
    sweep1[20] = 12.0
    buf.append_sweep(sweep1)

    sweep2 = np.ones(100, dtype=np.float32) * 2.0
    sweep2[30] = 8.0
    buf.append_sweep(sweep2)

    assert len(buf) == 2
    assert buf.clipped_data.shape == (2, 100)
    assert np.all(buf.clipped_data >= 0.0)
    assert buf.clipped_data[0, 10] == 0.0
    assert buf.clipped_data[0, 20] == 12.0

    assert np.isclose(buf.row_max[0].magnitude, 12.0)
    assert np.isclose(buf.row_max[1].magnitude, 8.0)
    assert len(buf.row_mean) == 2
    assert len(buf.row_std) == 2
    assert np.isclose(buf.row_std[0], np.std(sweep1), atol=1e-4)

    # Test resizing and clear
    buf.resize_window(120)
    assert buf.window_samples == 120
    assert buf.clipped_data.shape == (2, 120)
    buf.clear()
    assert len(buf) == 0
    assert buf.clipped_data.shape == (0, 120)

    # 2. Test SingleTraceView persistent artists, histogram caching, and trace decimation
    state = mock_multichannel_state
    single_view = SingleTraceView(state=state)
    qtbot.addWidget(single_view)

    # Verify persistent Line2D topax artists exist
    assert hasattr(single_view, "topax_cur_line")
    assert hasattr(single_view, "topax_prev_line")
    assert hasattr(single_view, "topax_next_line")
    assert hasattr(single_view, "topax_conv_peaks_line")

    # Vectorized conv peaks line test
    hl_sample = np.array([50, 120, 200])
    x_pos = hl_sample / state.sampling_rate
    x_segs = np.column_stack([x_pos, x_pos, np.full_like(x_pos, np.nan, dtype=np.float64)]).ravel()
    y_segs = np.tile([0.0, 1.0, np.nan], len(x_pos))
    single_view.topax_conv_peaks_line.set_data(x_segs, y_segs)
    single_view.topax_conv_peaks_line.set_visible(True)
    assert len(single_view.topax_conv_peaks_line.get_xdata()) == len(hl_sample) * 3

    # Check histogram caching across setStimNo
    single_view.updateFigure_core()
    assert single_view.step is not None
    step_artist_before = single_view.step

    state.setStimNo(1)
    single_view.updateFigure_core()
    # When switching stimno without unit change, step artist identity should be preserved
    assert single_view.step is step_artist_before

    # Decimation test
    x_high = np.linspace(0, 1, 10000)
    y_high = np.sin(x_high * 100)
    x_dec, y_dec = single_view._decimate_trace_for_display(x_high, y_high, max_points=2500)
    assert len(x_dec) <= 2500
    assert np.isclose(y_dec.min(), y_high.min(), atol=1e-3)
    assert np.isclose(y_dec.max(), y_high.max(), atol=1e-3)

    # Decimation when points <= max_points returns original array slice without loss
    single_view.ax.set_xlim(0.0, 1.0)
    x_low = np.linspace(0, 1, 500)
    y_low = np.cos(x_low)
    x_dec_low, y_dec_low = single_view._decimate_trace_for_display(x_low, y_low, max_points=2500)
    assert len(x_dec_low) == len(x_low)

    # 3. Test CachedAxesImage with rolling_max=True
    fig, ax = plt.subplots()
    img_data = np.random.rand(300, 500).astype(np.float32)
    cached_img = CachedAxesImage(ax, rolling_max=True)
    cached_img.set_data(img_data)
    cached_img.set_extent([0, 500, 300, 0])
    ax.add_artist(cached_img)
    ax.set_xlim(50, 150)
    ax.set_ylim(150, 50)
    renderer = fig.canvas.get_renderer()
    res = cached_img.make_image(renderer)
    assert res is not None
    assert res[0] is not None
    plt.close(fig)

    # 4. Test MultiTraceView on_new_sweep with LiveERPBuffer and streaming right axes
    multi_view = MultiTraceView(state=state)
    qtbot.addWidget(multi_view)

    # Attach live erp buffer to state
    live_buf = LiveERPBuffer(window_samples=state.analog_signal_erp.shape[1], units=pq.uV)
    for row in state.analog_signal_erp:
        live_buf.append_sweep(row)
    state._live_erp_buffer = live_buf
    state.is_streaming = True
    state._live_aux_erp_buffers = {state.analog_signal.name: live_buf}

    multi_view.on_new_sweep(0)
    assert multi_view.std is not None
    assert state.analog_signal.name in multi_view.right_ax_data
    # Check that right_ax_data uses the buffer row stats directly
    vals, sweeps = multi_view.right_ax_data[state.analog_signal.name]
    assert len(vals) == len(live_buf)


def test_multi_unit_live_tracking_and_table_toggle(qtbot, mock_multichannel_state):
    """Verify live multi-unit tracking across different channels, table column toggles, and parameter synchronization."""
    state = mock_multichannel_state
    sr = 10000.0 * pq.Hz

    # State already has Unit 0 on CH0. Add Unit 1 on CH1
    evt1 = neo.Event(state.event_signal.times + (100 / 10000.0) * pq.s, name="unit_1", units=pq.s)
    evt1.annotations["channel"] = "CH1"
    evt1.annotations["tracking_channel"] = "CH1"
    sg1 = tracked_neuron_unit(
        event=evt1,
        channel="CH1",
        tracking_channel="CH1",
        auto_track=False,
        tracking_window_ms=6.0,
        tracking_threshold_mv=0.02,
    )
    state.spike_groups.append(sg1)
    state.segment.events.append(evt1)
    state.update_idx_arrs()

    # Unit 0 is on CH0
    sg0 = state.spike_groups[0]
    sg0.channel = "CH0"
    sg0.tracking_channel = "CH0"
    sg0.auto_track = False

    # Instantiate views
    tbl_view = SpikeGroupTableView(state=state)
    tr_view = TrackingView(state=state)
    qtbot.addWidget(tbl_view)
    qtbot.addWidget(tr_view)

    model = tbl_view.spike_tablemodel
    assert model.columnCount() == 11
    assert model.headers[8] == "track"
    assert model.headers[9] == "win (ms)"
    assert model.headers[10] == "thresh (mV)"

    # Test column 8 ("track") checkbox toggle via model.setData
    idx_track_0 = model.index(0, 8)
    idx_track_1 = model.index(1, 8)

    assert model.data(idx_track_0, Qt.CheckStateRole) == Qt.Unchecked
    assert sg0.auto_track is False

    # Toggle Unit 0 auto_track to True
    model.setData(idx_track_0, Qt.Checked, Qt.CheckStateRole)
    assert sg0.auto_track is True
    assert sg0.event.annotations["auto_track"] is True
    assert model.data(idx_track_0, Qt.CheckStateRole) == Qt.Checked

    # Toggle Unit 1 auto_track to True
    model.setData(idx_track_1, Qt.Checked, Qt.CheckStateRole)
    assert sg1.auto_track is True
    assert sg1.event.annotations["auto_track"] is True
    assert model.data(idx_track_1, Qt.CheckStateRole) == Qt.Checked

    # Test editing column 9 ("win (ms)") and column 10 ("thresh (mV)") in table model
    idx_win_0 = model.index(0, 9)
    idx_thresh_0 = model.index(0, 10)
    model.setData(idx_win_0, 8.0, Qt.EditRole)
    assert sg0.tracking_window_ms == 8.0
    assert sg0.event.annotations["tracking_window_ms"] == 8.0
    assert model.data(idx_win_0, Qt.DisplayRole) == 8.0

    model.setData(idx_thresh_0, 0.03, Qt.EditRole)
    assert sg0.tracking_threshold_mv == 0.03
    assert sg0.event.annotations["tracking_threshold_mv"] == 0.03
    assert model.data(idx_thresh_0, Qt.DisplayRole) == 0.03

    # Test TrackingView parameter synchronization when switching units
    state.setUnitGroup(0)
    assert tr_view.lbl_active_unit.text() == "Unit: unit_0"
    assert tr_view.lbl_unit_channel.text() == "Channel: CH0"
    assert tr_view.qsb_window_size.value() == 8.0
    assert abs(tr_view.qsb_threshold.value() - 0.03) < 1e-4
    assert tr_view.qcb_auto_track.isChecked() is True

    state.setUnitGroup(1)
    assert tr_view.lbl_active_unit.text() == "Unit: unit_1"
    assert tr_view.lbl_unit_channel.text() == "Channel: CH1"
    assert tr_view.qsb_window_size.value() == 6.0
    assert abs(tr_view.qsb_threshold.value() - 0.02) < 1e-4
    assert tr_view.qcb_auto_track.isChecked() is True

    # Prepare streaming live buffers for live sweep tracking test
    # Primary channel CH0 and aux channel CH1
    live_buf0 = LiveERPBuffer(window_samples=500, units=pq.uV)
    live_buf1 = LiveERPBuffer(window_samples=500, units=pq.uV)

    # Populate initial sweeps (0 to 4)
    erp0 = state.get_channel_erp("CH0")
    erp1 = state.get_channel_erp("CH1")
    for s_i in range(len(state.event_signal)):
        live_buf0.append_sweep(erp0[s_i])
        live_buf1.append_sweep(erp1[s_i])

    state.is_streaming = True
    state._live_erp_buffer = live_buf0
    state._live_aux_erp_buffers = {"CH0": live_buf0, "CH1": live_buf1}
    state.active_channel_names = ["CH0", "CH1"]

    # Sweep 5: spike on CH0 at sample 104, spike on CH1 at sample 106
    # Note: 50 uV = 0.05 mV >= threshold (0.03 mV for unit 0, 0.02 mV for unit 1)
    w0_new = np.zeros(500, dtype=np.float32)
    w0_new[104] = 50.0  # 50 uV peak
    w1_new = np.zeros(500, dtype=np.float32)
    w1_new[106] = 50.0

    # Append new sweep with 2D array or aux_sweeps dict
    t_new = state.event_signal.times[-1] + 0.1 * pq.s
    idx_new = state.append_sweep(
        stim_timestamp=t_new,
        sweep_waveform=w0_new,
        aux_sweeps={"CH1": w1_new},
    )
    assert idx_new == 5

    # Verify track_stimulus_all_units automatically tracked both units on their respective channels!
    assert sg0.idx_arr[5] is not None
    assert sg0.idx_arr[5][0] == 104

    assert sg1.idx_arr[5] is not None
    assert sg1.idx_arr[5][0] == 106
