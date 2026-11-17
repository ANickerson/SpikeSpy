# Candidate Unit Search & Review

## Goal

During the microneurography search phase, automatically propose likely neurons from repeated stimulus sweeps, track each proposal provisionally, and let the user accept or discard it with minimal UI disruption. Latency consistency across sweeps is the primary identity signal; a candidate may appear, disappear, and reappear as the electrode moves, and new units may appear later at constant latencies after sensitisation.

The UX is: a small, muted set of ghost rows at the bottom of the existing unit table, with the existing Add/Remove buttons acting as Accept/Discard when a candidate is selected. No modal dialogs, no new dock panel, no automatic selection stealing during live scanning.

## Confirmed Decisions

- Offline and live candidate scanning ship in one implementation pass.
- Live: explicit `Auto` search toggle; scans the recent sweep window periodically.
- Offline: manual `Scan` button on the loaded recording.
- Candidates live in a separate store from `ViewerState.spike_groups`; they are tracked provisionally but never exported/saved as units until accepted.
- Accept promotes a candidate to a normal `tracked_neuron_unit`, keeps only the already-verified per-sweep detections, and enables auto-tracking for future sweeps with a search-friendly `tracking_max_skips`.
- Discard makes a candidate dormant, not permanently gone. Re-propose only if evidence improves materially, the active channel changes, or the user explicitly rescans discarded candidates.
- Pending, dormant, and discarded candidates persist in the NIX/NWB segment annotation and are restored on load.
- Candidate identity is anchored on channel + latency band, not continuous tracking; gaps and reappearances update the same candidate.
- Candidate trace overlays are implemented in the PyQtGraph views. Legacy Matplotlib views remain table-only for candidates in this pass.

## Data Model

### New pure module: `spikespy/candidate_detection.py`

No Qt imports. Contains:

- `CandidateScanConfig`: dataclass of tunable values and defaults.
- `CandidateProposal`: channel, latency center (samples and seconds), tolerance, score, features, `samples_by_sweep`, `hits`, `last_seen_stim`, polarity fraction, waveform correlation, SNR, neighbor support.
- `CandidateEvidence`: hit count, recent hit count, jitter ms, hit rate, median SNR, waveform score, footprint score, recency score.
- `detect_candidates(...)`: pure function (no Qt) using NumPy/SciPy plus `dsp.detectors` and the snapshot-safe parts of `MultiChannelTemplateMatcher`. It takes:
  - `erp_by_channel: dict[str, np.ndarray]` (shape `n_sweeps x n_samples`)
  - `sampling_rate_hz`
  - `scan_start_stim`, `scan_end_stim`
  - `existing_units` sample maps for suppression
  - `candidate_memory` fingerprints for identity matching
  - `config`
  - optional `peak_finder` injection for tests; defaults to `SpikeDetectorRegistry.find_candidates`

Default `CandidateScanConfig`:

| Field | Default | Notes |
|---|---:|---|
| `blank_ms` | `1.0` | Ignore stimulus artifact at sweep start |
| `min_peak_distance_ms` | `0.5` | Avoid multi-peak duplicates |
| `min_peak_snr` | `3.0` | Peak must exceed robust noise by this factor |
| `tolerance_ms` | `1.0` | Latency band width for cross-sweep support |
| `merge_tolerance_ms` | `1.0` | Duplicate/non-max suppression width |
| `min_hits` | `3` | Absolute sweeps, not hit rate |
| `max_jitter_ms` | `2.0` | Maximum latency scatter |
| `max_candidates` | `8` | Proposals per scan, globally |
| `max_per_channel` | `3` | Proposals per channel per scan |
| `live_scan_window_sweeps` | `40` | Rolling live scan range |
| `recency_window_sweeps` | `10` | Recent-support window |
| `scan_interval_ms` | `2000` | Live auto-scan period |
| `review_quiet_ms` | `2000` | Pause auto-scan after candidate interaction |
| `reactivation_score_margin` | `0.15` | Dormant/discarded reactivation threshold |
| `accepted_overlap_suppress` | `0.6` | Suppress if this fraction of hits overlap an accepted unit |
| `max_scan_sweeps` | `20000` | Offline safety cap; scan status reports truncation |
| `algorithm` | `"zscore"` | Initial peak detector; `basic` and `wavelet` allowed, `template` falls back to `zscore` initially |
| `template_refine` | `True` | Build a provisional multi-channel template and re-detect each sweep |
| `template_corr_threshold` | `0.35` | Minimum composite match score for a refined detection |
| `max_template_refine_sweeps` | `2000` | Cap template refinement cost on long recordings |

## Reusing the Existing Tracking Stack

The feature should sit on top of the existing DSP rather than reimplement peak finding and template scoring. `candidate_detection.py` stays Qt-free, but imports `dsp.detectors` and may instantiate `processing.MultiChannelTemplateMatcher` (also Qt-free).

| Existing component | Role in candidate search |
|---|---|
| `SpikeDetectorRegistry.find_candidates(...)` | Per-sweep peak extraction for the initial latency-support map. Default `zscore`, with `polarity="both"`, `z_thresh=min_peak_snr`, `use_mad=True`, `min_distance_ms=min_peak_distance_ms`. `basic` and `wavelet` have no polarity mode in `find_candidates`, so run one pass on the trace and one on the inverted trace. |
| `ZScoreThresholdDetector` | Robust MAD noise estimate and automatic polarity; already handles negative/positive spikes and refractory spacing. |
| `BasicWindowThresholdDetector` | Fallback peak detector when `algorithm="basic"`; threshold is derived from robust noise in signal units. |
| `WaveletDenoiseDetector` | Optional denoised peak extraction when selected; falls back safely if pywt is unavailable. |
| `SpikeDetectorRegistry.detect_sweep(...)` / `unit_tracking.detect_sample(...)` | Per-sweep follow of an existing candidate near its latency center. A miss is recorded as a gap and does not terminate the candidate. |
| `MultiChannelTemplateMatcher` | Second-pass refinement: build normalized templates from the candidate's best hits, set `.templates`, `.weights`, `.threshold`, then call `compute_composite_score` or `match_single_stimulus` on a snapshot getter to strengthen weak/intermittent detections and measure multi-channel footprint. |
| `unit_tracking.tracking_params(...)` | Reuse mV/uV threshold conversion and parameter merging when promoting a candidate or calling a detector. |
| `MultiChannelTemplateMatcher.find_candidate_peaks` | Not needed for the initial proposal; available for optional composite-peak refinement. |
| `track_basic`, `unit_tracking.track_range`, `detect_batch` | **Not** used for candidate identity. They are causal, stop after `max_skip` misses, and only move forward, which would split a unit that disappears and reappears. They remain available as post-accept “Fill Missing Spikes” tools. |
| `select_spikes_in_polygon` / polygon corridor | Left as the existing manual drawing tool; candidate detection is automatic and independent. |

Detection parameter source of truth:

- Initial peak extraction uses `algorithm` and its detector parameters. `CandidateScanConfig.min_peak_snr` maps to `z_thresh` for `zscore`; for `basic`, threshold is `min_peak_snr * robust_sigma` in the signal's units.
- Per-sweep follow and template refinement use the candidate's own polarity and latency window, so a positive/negative waveform inversion does not silently flip the detector each sweep.
- Accepted units inherit the candidate's `algorithm` and detector parameters. If the candidate was refined with `template`, promote with `tracking_algorithm="template"`, append the unit, then call `rebuild_unit_template(new_unit_idx)`.

### New `CandidateUnit` dataclass in `ViewerState.py`

Keep it separate from `tracked_neuron_unit`, wrapping a provisional unit payload:

- `candidate_id: int`
- `unit: tracked_neuron_unit` (holds provisional event/idx_arr/channel/neighbors and tracking params)
- `algorithm: str`
- `detector_params: dict`
- `template_refined: bool`
- `name: str`
- `name_edited: bool`
- `channel: str`
- `neighbor_channels: list[str]`
- `score: float`
- `evidence: CandidateEvidence`
- `latency_center_samples: float`
- `latency_tolerance_samples: float`
- `status: str` in `{"pending", "dormant", "discarded"}`
- `score_at_discard: Optional[float]`
- `channel_at_discard: Optional[str]`
- `first_seen_stim: int`
- `last_seen_stim: int`
- `created_at_s: float`

`ViewerState` additions:

- `self.candidate_units: dict[int, CandidateUnit]`
- `self.active_candidate_id: Optional[int]`
- `self.candidate_config: CandidateScanConfig`
- `self.auto_scan_live: bool = False`
- `self._candidate_scan_manager`
- signals: `onCandidatesChanged`, `onCandidateSelectionChanged(int)`, `onCandidateScanStateChanged(bool)`
- `get_active_candidate()`, `get_visible_candidates()` (pending only, sorted by score, capped), `set_active_candidate(id|None)`
- `accept_candidate(id) -> int`, `discard_candidate(id)`, `reactivate_candidate(id)`
- `export_candidate_store() -> dict`, `restore_candidate_store(payload|str)`
- `request_candidate_scan(synchronous=False, scope="auto")`
- `set_auto_scan_live(bool)`

## Detection Algorithm

The detector must find latency bands that recur across sweeps even when the signal vanishes between appearances.

For each channel in the scan scope:

1. Copy the relevant ERP rows as float32 and blank the first `blank_ms` samples. Do not reimplement peak finding: call `SpikeDetectorRegistry.find_candidates(algorithm, sweep, fs_hz, params=...)` for each sweep. Use `polarity="both"` for `zscore`; for `basic`/`wavelet`, run one pass on the trace and one on the inverted trace. Capture sample indices, detector scores (SNR for `zscore`), and the signed peak amplitude from the original waveform. Skip flat sweeps.
2. Build a sparse peak mask (one boolean row per sweep). Convolve along the sample axis with a boxcar of width `2 * tolerance + 1` to obtain, for every sample bin, how many distinct sweeps have a peak within tolerance. Also build a weighted version using detector SNR, and a recent-support version limited to the last `recency_window_sweeps`.
3. Find local maxima of cross-sweep support. Candidate bands are latency centers, not connected tracks. At each peak, gather contributing sweeps and, per sweep, the peak nearest the center within tolerance. This produces `samples_by_sweep` with gaps preserved.
4. If `template_refine` is enabled and the proposal has at least 3 hits, refine at most `max_candidates` proposals over the most recent `max_template_refine_sweeps` sweeps:
   - Build a normalized multi-channel template from snippets around the proposal's hits, using `MultiChannelTemplateMatcher` conventions.
   - Instantiate `MultiChannelTemplateMatcher`, set `.templates`, `.weights`, and `.threshold` (`template_corr_threshold`), and re-detect each sweep with `match_single_stimulus(getter, expected_idx=center, search_radius_samples=tolerance_samples)`.
   - Keep the refined detection when the composite score passes threshold; otherwise keep the initial peak detection for that sweep. Recompute hits, jitter, waveform score, and footprint from the refined result.
   - Use a snapshot-based `getter(channel) -> 1D waveform`; do not call `score_sweep` because it reads `ViewerState` and is not worker-thread safe.
5. Compute features:
   - `hits`: number of contributing sweeps.
   - `hit_rate`: `hits / eligible_sweeps`; display it but do not use it as the primary gate.
   - `recent_hits`: hits in the recency window.
   - `jitter_ms`: `1.4826 * MAD(latencies)` in milliseconds.
   - `median_snr`: median peak SNR.
   - `polarity_fraction`: fraction of hits with the dominant sign.
   - `waveform_score`: median normalized correlation of per-hit snippets against the median template, using a window of about `-1.5 ms .. +2.5 ms`; use the template refinement composite score when available; neutral `0.5` when fewer than 3 hits.
   - `footprint_score`: fraction of hits with a matching peak on a neighbor channel within tolerance; use the template refinement multi-channel score when available; neutral `0.5` if no neighbors are configured.
   - `recency_score`: `1.0` if a hit exists in the last `recency_window_sweeps`, decaying to `0.0` for stale bands.
   - `distinctiveness`: support at center relative to the best support just outside the NMS window.
6. Score with a recall-oriented blend:

   ```
   score = 0.30 * saturating(hits / 8)
         + 0.20 * jitter_score
         + 0.15 * waveform_score
         + 0.10 * snr_score
         + 0.10 * recency_score
         + 0.10 * footprint_score
         + 0.05 * distinctiveness_score
   ```

   Where `jitter_score = 1 / (1 + jitter_ms / tolerance_ms)`, `snr_score = min(1, median_snr / 8)`, and `saturating(x) = min(1, x)`.

7. Gate:
   - `hits >= min_hits`
   - `jitter_ms <= max_jitter_ms`
   - `median_snr >= min_peak_snr`
   - center outside the blanking period
   - not suppressed by an accepted unit (see below)
   - pending candidates need `recent_hits >= 1`; stale strong bands become dormant instead of new pending rows.
8. Non-max suppression: sort by score, keep the best per channel/latency neighborhood. Suppress two proposals within `merge_tolerance_ms` when their waveform correlation is high; otherwise keep both. Enforce `max_per_channel` and `max_candidates`.
9. Return proposals plus evidence. The caller merges them into the store; the detector itself is stateless.

Provisional tracking is scan-driven: each scan re-detects every sweep in the window for each candidate. This is what preserves gaps and reappearances. `detect_sweep` is the single-sweep follow primitive if a future per-sweep live path is added; it is not used to decide candidate identity.

Accepted-unit suppression:

- For each accepted `spike_group` on the same channel, use its `idx_arr`/`get_sweep_sample`.
- If more than `accepted_overlap_suppress` of a candidate's hit sweeps already contain that unit's spike within tolerance, drop the proposal.
- Also drop near-identical bands when the accepted unit's median latency is within tolerance and overlap is substantial. Do not suppress a candidate merely because latencies are nearby if the accepted unit is silent on those sweeps.

Artifact handling:

- Blanking the first `blank_ms` is the primary defense.
- If a strong band remains at nearly zero latency and has extremely high hit rate, show a lower score rather than hiding it; the user can discard it and it becomes dormant.

## Candidate Lifecycle

### Scan merge

On a scan result, match proposals to existing candidates by same channel and latency within `max(existing tolerance, proposal tolerance, merge_tolerance)`. If several match, choose the closest center.

- Pending match: update center as the robust median of combined detections, merge detections (replace inside the scanned range, keep older detections outside it), refresh evidence/score/last-seen, and keep pending.
- Dormant match: reactivate to pending if score improves or recent hits are present above the reactivation threshold.
- Discarded match: reactivate only if `new_score > score_at_discard + reactivation_score_margin`, the active channel changed, or the user chose Rescan discarded. Otherwise keep discarded and hidden.
- No match: create a new pending candidate if capacity allows. If at capacity, replace only a lower-scoring, non-selected pending candidate; otherwise store the proposal as dormant.
- Pending candidates not seen for longer than the recency window become dormant and disappear from the table but remain in the store for re-identification.
- Candidates overlapping an accepted unit are suppressed, not stored as pending.

Candidate identity intentionally ignores short gaps and waveform changes. Latency plus channel is the anchor. A large latency jump beyond the tolerance creates a new candidate; slow drift within tolerance updates the same candidate.

### Accept

`accept_candidate(candidate_id)`:

1. Build a normal `tracked_neuron_unit` from `candidate.unit` using the verified detections exactly as stored. Do not run historical gap filling.
2. Use a unique `unit_N` name unless the user edited the candidate name.
3. Set channel, neighbors, and tracking parameters:
   - `tracking_window_ms = clamp(2.0, 4 * jitter_ms + 0.5, 10.0)`
   - `tracking_threshold_mv` from the candidate's detection threshold, converted from the signal units to mV using `unit_tracking.tracking_params` / detector conversion helpers
   - `tracking_max_skips = 3`
   - `tracking_algorithm = candidate.algorithm` (default `"zscore"`), preserving `polarity`, `z_thresh`/threshold, `min_distance_ms`, and `use_mad` in `tracking_params`
   - `auto_track = True`
4. If the candidate was refined with `template`, promote with `tracking_algorithm="template"`, append the unit, then call `rebuild_unit_template(new_unit_idx)` so the existing template matcher is initialized for future sweeps.
5. Add provenance annotations: `candidate_source="search"`, `candidate_score`, `candidate_first_seen_stim`.
6. Append to `spike_groups`, update idx arrays, emit unit-list/group/spike signals, select the new unit, remove the candidate, and clear the active candidate.
7. Push one undo closure that removes the new unit and restores the pending candidate.

Do not modify `export_units_csv` or `export_units_npz`; candidates are never in `spike_groups`, so they are excluded automatically.

### Discard

`discard_candidate(candidate_id)`:

1. Set status to `discarded`, store `score_at_discard` and `channel_at_discard`.
2. Hide the row, clear active selection, emit candidate-changed signals.
3. Push one undo closure that restores pending status and selection.
4. `Rescan discarded` clears the discarded fingerprints or sets a one-shot override so previously discarded bands can be proposed again.

### Live auto-scan

- `Auto` is explicit and default off. It is enabled only while streaming.
- A `QTimer` in the scan manager fires every `scan_interval_ms`.
- Skip a tick when: a scan is already running, no new sweeps arrived since the last scan, the user interacted with a candidate within `review_quiet_ms`, or the pending cap is reached.
- Manual scans are allowed while streaming and while auto-scan is off.
- On stream stop, stop the timer but keep `auto_scan_live` and pending/dormant candidates. Resume on the next stream.
- Ignore results from stale generations; only the latest applied scan may update the store.

## UI

### `SpikeGroupTable.py`

Ghost rows are appended after real units, so real unit row indices stay stable and existing row-index assumptions remain valid for rows below `n_units`.

`SpikeGroupTableModel`:

- Accept `candidates_func` in addition to the existing units provider.
- Keep a `_rows` list and `row_to_item(row)` helper returning `("unit", idx)` or `("candidate", id)`.
- Candidate rows sorted by descending score, then id.
- Existing columns keep their meaning:
  - `name`: candidate name, editable.
  - `start` / `end`: latency sample range from the provisional unit's window.
  - `N`: verified hit count.
  - `type`: `candidate (<score>)`.
  - `notes`: generated evidence summary unless the user typed notes, editable.
  - `channel` / `neighbors`: from the provisional unit, editable channel if useful.
  - `track`: unchecked, not user-checkable for candidates.
  - `win (ms)` / `thresh (mV)`: editable proposed tracking parameters.
- Add `CANDIDATE_ROLE`; keep `SPIKEGROUP_ROLE` returning `candidate.unit` for compatibility.
- `ToolTipRole` for candidates shows the full breakdown: score, latency ± jitter, hits/recent hits, SNR, waveform match, footprint, last seen sweep.
- `BackgroundRole`, `ForegroundRole`, and `FontRole` use palette-aware muted styling and italic text; no hard-coded light theme colors.
- `setData` on candidate rows updates the provisional unit or candidate name/notes only; it never sets `auto_track`.
- `flags` for candidates: selectable, editable name/notes/win/thresh, not checkable.
- Force a model reset when the candidate-id list changes; otherwise emit `dataChanged`.

`SpikeGroupTableView`:

- Connect `onCandidatesChanged` to `update` and `onCandidateSelectionChanged` to selection sync.
- Selection:
  - Row < `n_units`: existing `state.setUnitGroup(row)` and clear active candidate.
  - Candidate row: `state.set_active_candidate(candidate_id)` and leave `cur_spike_group` unchanged.
- `_do_update_core` must keep a selected candidate selected instead of always re-selecting the current unit. Clear candidate selection only on an explicit unit-row selection.
- Add/Remove buttons:
  - Candidate selected: Add becomes Accept (check/apply icon), Remove becomes Discard (trash icon); tooltips update.
  - Unit selected: existing Add/Discard behavior.
- `add_row_and_select` must select the new unit row, not `rowCount() - 1`, because candidate rows are now last.
- Context menu on a candidate row: Accept, Discard, Select for preview, Rescan discarded. Do not show unit tracking actions for candidates.
- `remove_selected_row` and `on_cell_double_clicked` must use `row_to_item`.
- Add a `Scan` button and a compact checkable `Auto` button in the existing button row, plus a transient status label.
  - Offline: `Scan` runs a manual whole-recording scan.
  - Live: `Scan` runs a manual recent-window scan; `Auto` toggles live scanning.
  - During a scan: `Scanning...`, disabled; no modal.
  - After a manual scan with results: select and scroll to the first candidate. Auto-scan must never steal selection or scroll.
  - Show count in the button text, e.g. `Scan (3)`.
- Optional `Ctrl+Shift+A` / `Ctrl+Shift+D` shortcuts for accept/discard when the table has focus; do not use plain letters because existing application shortcuts consume them.

### Trace overlays (PyQtGraph only)

`MultiTracePGView.py`:

- Add `candidate_scatter` (open circles, amber pen, transparent brush) and `candidate_band` (vertical `LinearRegionItem`, translucent amber) to `ChannelPanel`.
- In `plot_spikegroups`, after drawing accepted units, draw the active candidate on panels whose channel matches the candidate channel:
  - open circles at `(sample, sweep)` for all detections.
  - a vertical band around `latency_center_samples ± latency_tolerance_samples`.
- Hide both when no candidate is active or the panel channel does not match.
- Connect `onCandidatesChanged` and `onCandidateSelectionChanged` to refresh.

`SingleTracePGView.py`:

- Add a dashed vertical candidate line and one open-circle marker on the current sweep.
- Update them in `_do_update_figure_core` after `_update_spike_markers`.
- Include `active_candidate_id` and the candidate hit count in `render_key` so selection changes force a redraw.
- Hide when no candidate, or when the view's primary channel differs from the candidate channel.

`MultiTraceFixedPGView.py`:

- Mirror the active-candidate scatter/band on its single main plot with the same helper.

Legacy Matplotlib views:

- Candidate rows still work because they are table-level. No candidate overlay in this pass.

## Persistence

Candidates are stored in a segment annotation, so `load_file`'s existing 4-tuple signature stays intact.

On save:

- `SpikeSpy.saveFile` passes `self.state.export_candidate_store()` to `save_state_to_nix(..., candidates=...)`.
- `save_state_to_nix` adds `data2.annotations["spikespy_candidates"] = json.dumps(payload, separators=(",", ":"))`.
- Existing NIX/NWB writers accept a string annotation.

Payload schema:

```json
{
  "version": 1,
  "candidates": [
    {
      "id": 3,
      "status": "pending",
      "name": "candidate_3",
      "name_edited": false,
      "channel": "CH0",
      "neighbors": ["CH1"],
      "score": 0.82,
      "score_at_discard": null,
      "channel_at_discard": null,
      "latency_center_s": 0.0124,
      "latency_tolerance_s": 0.001,
      "algorithm": "zscore",
      "detector_params": {"polarity": "negative", "z_thresh": 3.0, "min_distance_ms": 0.5, "use_mad": true},
      "template_refined": true,
      "features": {"hits": 7, "recent_hits": 3, "jitter_ms": 0.31, "snr": 5.2},
      "tracking": {"window_ms": 2.0, "threshold_mv": 0.05, "max_skips": 3},
      "detections": [[2, 124, 10.224], [5, 126, 10.526]],
      "first_seen_stim": 2,
      "last_seen_stim": 41
    }
  ]
}
```

On load:

- Parse the annotation in `ViewerState.loadFile` / `set_segment` and pass the restored candidates into `set_data` before `onLoadNewFile` fires.
- Rebuild each provisional `tracked_neuron_unit` event from stored timestamps and `idx_arr` from stored samples.
- Drop detections with out-of-range stimulus indices; keep the candidate dormant if none remain.
- Unknown versions are ignored with a logged warning; no migration needed for files without the annotation.
- Accepted candidates are not in the store; they are normal units and continue through the existing save path.

## Implementation Order

1. Add `spikespy/candidate_detection.py` with config/proposal/evidence dataclasses and the pure detector. Implement per-sweep peak extraction through `SpikeDetectorRegistry.find_candidates` (default `zscore`, polarity `both`) and the optional `MultiChannelTemplateMatcher` refinement pass. Add focused unit tests for constant-latency, intermittent/reappearing, two-unit, jitter, artifact, duplicate suppression, detector delegation, and template refinement.
2. Add `CandidateUnit`, candidate store state, signals, matching/merge logic, accept/discard/undo, and `export_candidate_store`/`restore_candidate_store` in `ViewerState.py`. Preserve the candidate's algorithm/params on accept and reuse `unit_tracking.tracking_params` for unit conversion. Add state-level tests.
3. Add `spikespy/candidate_worker.py` with a QRunnable and a manager using `QThreadPool.globalInstance()`, latest-wins generation handling, and synchronous mode for tests. Wire live auto-scan timer and manual scans.
4. Update `SpikeGroupTableModel` and `SpikeGroupTableView` with candidate rows at the bottom, selection mapping, Add/Remove repurposing, Scan/Auto controls, tooltips, and palette-aware styling. Add Qt tests.
5. Add candidate overlays to `MultiTracePGView`, `SingleTracePGView`, and `MultiTraceFixedPGView`. Add view tests.
6. Add NIX persistence in `io_manager.save_state_to_nix`, the save call in `SpikeSpy.py`, and restore in `ViewerState.loadFile`/`set_segment`. Add a save/reload round-trip test.
7. Run focused tests, then the headless suite; run `uv run black` only on changed files.

## Validation

Headless suite:

```powershell
$env:QT_QPA_PLATFORM="offscreen"; uv run pytest tests -q
```

Focused tests to add:

- `tests/test_candidate_detection.py`
  - one synthetic unit at a fixed latency is proposed once with the correct center
  - a sparse unit seen at sweeps 0, 2, 5, 9 and again at 20 remains one candidate
  - two units at different latencies produce two candidates
  - near-duplicate peaks are suppressed
  - artifact at sweep start is blanked
  - latency jumps beyond tolerance create a new candidate
  - accepted unit overlap suppresses a duplicate proposal
  - `algorithm="zscore"` uses the registry and detects both positive and negative peaks
  - a weak/intermittent candidate is strengthened by template refinement without adding gaps
  - a fake detector plugin is delegated to and its params are respected
- `tests/test_candidate_workflow.py`
  - pending candidate is not in `spike_groups`
  - accept creates a unit with exactly the verified detections, `auto_track=True`, and the candidate's algorithm/params
  - accept does not add historical gaps
  - template-refined accept initializes the unit template matcher
  - discard hides the row and does not immediately re-propose it
  - stronger evidence or channel change reactivates a discarded candidate
  - undo restores candidate/unit state
  - live auto-scan skips while reviewing and ignores stale generations
- `tests/test_SpikeGroupTable_candidates.py` (or extend `test_tracking_integration.py`)
  - candidate rows appear after real units
  - selecting a candidate does not change `cur_spike_group`
  - Add/Remove accept/discard when a candidate row is selected
  - candidate rows have muted roles and no track checkbox
- `tests/test_candidate_views.py`
  - MultiTrace and SingleTrace overlays appear for the active candidate and disappear on accept/discard
- `tests/test_candidate_persistence.py`
  - save with pending/dormant/discarded candidates, reload, and verify restoration
  - accept a candidate, save, reload, and verify it is a normal unit and absent from the candidate store

Manual validation:

- Use `OpenEphysLiveView` built-in demo stream with 1-3 units. Enable `Auto`, confirm proposals match demo unit latencies, accept one and discard another, then save/reload and confirm both decisions persist.
- Move/change the active channel and confirm dormant candidates on the old channel do not reappear as new rows unless evidence returns.

## Risks and Edge Cases

- **Row mapping:** candidate rows are last, but all row consumers in `SpikeGroupTable` must use `row_to_item`; otherwise context menus and tracking actions can address the wrong unit.
- **Selection stealing:** `_do_update_core` must not replace a selected candidate during live refresh.
- **Sparse units:** do not gate on hit rate. A unit seen in only 3 of 40 sweeps is valid if latency is tight and it is recent.
- **Artifact false positives:** default `blank_ms=1.0`; expose the setting if early-latency units are missed.
- **Unit conversion:** live signals may be uV while thresholds are stored in mV; reuse `unit_tracking.tracking_params`/detector conversion helpers and test both uV and mV.
- **Stale persistence:** validate stored stimulus indices on load; keep candidates dormant rather than crashing on changed data.
- **Performance:** live snapshots are limited to `live_scan_window_sweeps`; offline scans are capped at `max_scan_sweeps` and run off the GUI thread. Never let a scan block rendering.
- **Template refinement:** build templates only from the snapshot, never call `MultiChannelTemplateMatcher.score_sweep` or `build_template` from the worker because they read `ViewerState`. Skip refinement when templates are too short or fewer than 3 hits exist.
- **Detector availability:** `zscore` is the default; `basic` needs a threshold derived from robust noise; `template` has no template before a candidate exists, so initial extraction falls back to `zscore`; `wavelet` already falls back safely if pywt is unavailable.
- **Duplicate proposals:** latency NMS plus accepted-unit overlap suppression; if two genuine units share a latency band, keep them only when waveform correlation is low.
- **Theme:** use palette roles, not fixed light-theme grays, so ghost rows remain readable in dark themes.
- **Undo:** accept/discard each push exactly one undo action and restore both the candidate store and the unit list.

## Out of Scope

- Automatic acceptance of high-score candidates.
- Machine-learning classification or template clustering beyond normalized waveform correlation.
- Candidate overlay parity in legacy Matplotlib views.
- Changes to CSV/NPZ export formats.
- Editing candidate latency directly; editing is limited to name, notes, window, and threshold.
