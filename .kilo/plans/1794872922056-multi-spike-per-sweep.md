# Allow a unit to appear more than once in a sweep

## Goal

A tracked unit (`tracked_neuron_unit`) can hold several spikes on the same stimulus sweep (e.g. doublets), annotated manually, visible everywhere spikes are drawn, and preserved through save/load. Automatic tracking is explicitly out of scope: trackers keep exactly one primary detection per sweep.

## Resolved decisions

- **Scope**: manual annotation only. Detectors, `track_range`, live tracking, and candidate search stay one-spike-per-sweep.
- **Click semantics unchanged**: plain clicks still replace the primary spike (and Delete still clears the sweep). Adding/removing an extra spike is a separate explicit action: **context menu items and Ctrl+Click** in the trace views.
- **Representation**: `sg.event` (neo.Event) stays the canonical store — it already holds multiple timestamps per sweep and NIX/CSV/NPZ already round-trip them. `idx_arr` keeps its current shape (one `(sample, 1)` primary entry per sweep) so the ~100 existing read sites keep working unchanged. "Primary" = earliest spike in the sweep (matches today's first-spike-wins collapse).
- **Known semantics**: replacing the primary with a later latency can hand "primary" to a former extra (first-spike-wins is the global convention); latency-diff axes and template building stay primary-only.

## Why it currently fails

- `setUnit` (ViewerState.py:1078-1086) deletes **all** spikes inside the target sweep before inserting one.
- `apply_tracked_samples` (ViewerState.py:1288-1295) remove-mask deletes all spikes in keyed sweeps.
- `updateUnit(merge=True)` (ViewerState.py:1589-1592) keeps at most one spike per sweep (`break` after first).
- `get_sweep_sample_latencies` / `get_valid_sweep_indices_and_samples` / `get_sweep_sample` / `get_latencies` collapse to the first spike per sweep (ViewerState.py:287-409, 421-437) — fine for primary, but there is no spike-level accessor.
- Rasters/snippets draw one dot or line per sweep row (`viewer_worker.py:905-928, 720-753`, UnitPGView, MultiTracePG/Fixed, SingleTracePG, legacy mpl twins).
- `dsp/footprint.py` and `dsp/latencies.py` assume one spike per sweep (ISI/cumulative count undercounts).
- IO is already multi-spike-safe (`export_units_csv` writes one row per spike; NIX persists the whole event; `export_units_npz` makes one snippet per spike) — extras just disappear from view after `update_idx_arrs` rebuilds the collapsed `idx_arr`.

## Tasks (ordered)

### 1. Spike-level accessors — `spikespy/ViewerState.py`

On `tracked_neuron_unit` (near the existing methods at lines 287-419), add:

- `get_sweep_samples(stimno, event_signal, sampling_rate, window_size=None) -> List[int]` — all spike samples inside `[stim_time, next_stim_time)` and within `window_size`, searchsorted-based (mirror `get_sweep_sample` but collect all matches).
- `get_valid_sweep_indices_and_all_samples(event_signal, sampling_rate, window_size=None) -> (sweeps, samples)` — spike-level arrays; a sweep index may repeat.
- Fix `get_number_of_events` windowed branch (lines 241-245): count spikes in window (`sum(in_win)`), not unique sweeps.
- Leave every existing method and `update_idx_arrs` behavior unchanged.

### 2. Stop destructive writes — `spikespy/ViewerState.py`

- `setUnit` (1035-1185): when `latency` is not None, the `to_keep` filter removes **only the primary (earliest) spike** of the target sweep; extras survive. `latency=None` still removes all spikes in the sweep. Keep the final-index guard, undo, sentinel (-1/-2) handling, and all signal emissions.
- `apply_tracked_samples` (1216-1329): for each keyed sweep, remove only that sweep's primary spike (compute per-keyed-sweep first-spike index) instead of `remove[keys] = True` over every spike; then insert the new detection. Extras in keyed sweeps survive races.
- `updateUnit(merge=True)` (1559-1602): append **all** old+new spikes within `[t, t+gap)` per sweep (drop the `break` at 1592).

### 3. Explicit add/remove API — `spikespy/ViewerState.py`

- `addUnitSpike(unit_idx, stimno, latency_samples, save_undo=True) -> bool`:
  - Dedupe no-op if an existing spike on that sweep is within a deadband of `max(1 sample, 0.5 ms)`.
  - Merge the timestamp into `sg.event`, sort, replace the name-matched event in `self.segment.events` (pattern at 1312-1316), pad `idx_arr`, and recompute that sweep's primary entry (earliest spike, `None` if sweep emptied).
  - Set default channel like `setUnit` does (1138-1144). Save undo (event copy). Emit `onUnitChange`, `onNewSpikeEvents`, `onUnitSpikeChanged`, and `onUnitAnnotated` with an added `"extra": True` key (additive; consumers ignore unknown keys).
- `removeUnitSpike(unit_idx, stimno, near_sample=None, save_undo=True) -> bool`:
  - Removes the in-sweep spike nearest `near_sample`, or the primary when `near_sample is None`. Recompute the sweep's `idx_arr` entry afterwards (next remaining spike or `None`). Same undo/segment/signals handling.

### 4. Spike-level rendering — `spikespy/viewer_worker.py` + PyQtGraph views

- `_process_spike_group` (905-928): return spike-level `(samples, sweeps)` point arrays (sweep index repeats) instead of one point per sweep.
- `_compute_sweep_markers` (546-583): current-sweep marker set includes all in-sweep spikes; prev/next logic stays primary-based.
- `_compute_histogram` (667-692): use spike-level latencies.
- `_process_unit_view` (701-820) and `_process_template_update` (841-895): snippet lines keyed by `(sweep, k)` instead of sweep index; the current-stim line shows all in-sweep snippets.
- `MultiTracePGView`: raster (`plot_spikegroups` 2913-3070), candidate overlay (2400-2449 — candidate units stay single-per-sweep, just consume the new arrays), current-sweep highlight + dots (2809-2849) draw one marker per in-sweep spike.
- `MultiTraceFixedPGView` markers (706-771) and `SingleTracePGView` `_update_spike_markers` (1126-1163) + `_update_candidate_overlay` stays as-is.
- Legacy mpl twins per AGENTS.md: `legacy_mpl/MultiTraceView.py` raster (227-254, 1468-1494), `MultiTraceFixedView.py` markers (390-416), `SingleTraceView.py` markers (860-928).

### 5. Snippets, footprint, ISI — `UnitPGView.py`, `dsp/footprint.py`, `dsp/latencies.py`

- `UnitPGView._sync_lines_from_state` (311-348) + `update_curstim_line` (397-454): multiple snippet lines per sweep, keys `(sweep, k)`.
- `dsp/footprint.compute_unit_footprint_from_state`: collect spike-level (sweep, sample) pairs via the new accessor; snippets become `(n_ch, n_spikes, window)`; `last_n` counts spikes.
- `dsp/latencies.compute_isi_and_firing_rate`: cumulative spike count per sweep; ISI/rate from event times rather than the one-int-per-sweep array (also fixes the latent `(sample, 1)`-tuple misparse in `np.asarray(idx_arr, int64)`).

### 6. UI actions — `MultiTracePGView.py`, `SingleTracePGView.py`

- Right-click context menu on traces: "Add spike here for `<unit>`" and "Remove nearest spike for `<unit>`" → `state.addUnitSpike` / `state.removeUnitSpike` using each view's existing unit resolution (same unit the plain click annotates, e.g. `set_cur_pos` in SingleTracePGView:1583, MultiTracePGView click path).
- Ctrl+Click adds an extra spike at the clicked sample (plain click untouched).
- No changes to `SpikeGroupTable` (N column already reports `len(sg.event)`).

### 7. Tests

New `tests/test_multi_spike_per_sweep.py` (pytest + pytest-qt; offscreen):

- `addUnitSpike` appends a second timestamp; event shows 2 spikes in the sweep; `idx_arr[stimno]` stays the earliest; dedupe no-op inside deadband.
- `removeUnitSpike` removes nearest; removing the primary re-derives `idx_arr` to the next spike; removing the last sets `None`.
- `setUnit` replace keeps the extra spike; `setUnit(None)` still clears the whole sweep.
- `apply_tracked_samples` does not delete extras; `updateUnit(merge=True)` keeps multiple spikes per sweep.
- NIX save/load round-trip (tmp_path) and CSV export/import keep both spikes.
- `_process_spike_group` returns spike-level points (two dots for a doublet sweep).

Run: `$env:QT_QPA_PLATFORM="offscreen"; uv run pytest tests -q` (live-stream suites serially). Fix any tests that asserted all-spike deletion on re-annotation (none expected: replace of a single spike is observably identical).

## Out of scope

- Multi-spike auto-tracking / detectors (`detect_sweep`, `track_range`, live `spike_tracked(unit, sample, stimno)`) — one primary detection per sweep.
- Candidate search (`CandidateProposal.samples_by_sweep`) and candidate store format.
- Open Ephys C++ plugin, ZMQ payload, and `oe_controller.sync_unit_annotation` — stay single-latency per (unit, sweep); extras are SpikeSpy-side only.
- `get_latencies` first-spike semantics (latency-diff axes, template building, `move`/`copy_previous`, `align_spikegroup`).

## Risks

- Views that key data by sweep index (`lines[i]`, `lines_data[i]`) must switch to `(sweep, k)` keys — contained to UnitPGView + viewer_worker.
- Legacy mpl views are lower-value; if they regress, PG views remain authoritative.
- Live streaming: manual extras added during streaming go through the new API and survive `append_sweep`/`update_idx_arrs` (event-level, multi-safe).

## Validation

1. New test file green; full offscreen suite green.
2. Manual: load `data/test2.h5`, annotate a unit, Ctrl+Click a second spike on the same sweep → two raster dots, two UnitPG snippets, N column +1; save NIX, reload → both spikes visible; CSV round-trip keeps both rows.
