# CLAUDE.md — openscope-pp project conventions

This file is the durable knowledge an LLM agent (Claude / Cursor / etc.) should
have on hand when working in this repo. Append to it whenever you discover
something that would have saved time the first time around.

## Environment
- **Python**: `python3.13` for scripts requiring `dandi` + `numpy` (both available)
- **Anaconda python** (`/opt/anaconda3/bin/python`) has numpy but NOT dandi — don't use for pipeline scripts
- **Colab outputs**: always write to local Colab disk (`/content/`), never Google Drive unless explicitly asked
- **Colab notebook refresh**: pulling/refreshing the page does NOT reload the .ipynb. Use File → Revert, or close + reopen via File → Open notebook → GitHub. Otherwise you will run a stale cached version of the notebook.

## Task Completion Rules
- Never claim a task is done if there were errors during execution or if the output file/figure hasn't been verified
- Before reporting completion: run the script, show the last ~20 lines of output, confirm figures saved
- If a background process is running, check actual output file — don't infer status from prior messages

## Data & NumPy Conventions
- **Always check for inhomogeneous arrays** before stacking cross-session data — sessions have different frame rates → different PSTH lengths
- Fix: interpolate to `COMMON_T = np.linspace(window[0], window[1], 300)` before calling `np.array([...])`
- **dF/F extraction**: use `np.interp` on the timestamp axis — robust to clock drift and gaps
- **Soma mask**: always apply `is_soma` filter before extracting fluorescence traces (mesoscope)
- When reading HDF5 lazy datasets, do a single bulk read over the time span of interest, then interpolate — don't index row-by-row
- **NaN-aware reductions everywhere downstream of `np.interp`**: edge NaNs on individual trials silently propagate via plain `.mean(axis=…)` and wipe the entire population PSTH. Use `np.nanmean` / `np.nanstd` and SEM = `nanstd / sqrt(finite_count)`. This applies to `pop_psth`, cross-session aggregation, and the `(trial, unit)` reductions in `src/openscope_pp/analysis/{oddball,orientation,rf_mapping}.py` (already patched).
- **Imaging dF/F over full sessions**: stride-read with `data[::stride, :]` (h5py-supported) to bound memory. Typical strides: 30 for mesoscope, 10 for SLAP2.

## NWB structure quick reference
The schema differs slightly between the three dandisets and between releases. Verified for the 2026-04 release of 001637:

### Ecephys (001637) — `/units` columns of interest
`amplitude`, `decoder_label`, `default_qc`, **`depth`** (µm — direct, no electrode lookup needed), `device_name` (string like `"ProbeA"…"ProbeF"`, *camel-case, no separator*), `electrodes` (DynamicTableRegion), `estimated_x` / `estimated_y` / `estimated_z` (CCF coordinates), `firing_rate`, `id`, `spike_times`, `spike_times_index`. **No `peak_channel_id` column** — earlier multi-strategy fallbacks to `general/extracellular_ephys/electrodes/rel_y` are not necessary for this release; just use `units/depth` directly.

Some sessions carry **6 probes (A–F)**, not 4. **Probe letter is a per-session device ID, not an anatomical label** — never pool by letter across subjects. Use the *insertion* (`subject × asset_id × device_name`) as the unit of observation, or use CCF `estimated_x/y/z` for cross-session anatomical comparisons.

### Mesoscope (001768)
- Per-plane dF/F at `processing/<plane>/dff_timeseries/dff_timeseries` with `data` (n_t, n_roi) + `timestamps`.
- `is_soma` mask at `processing/<plane>/image_segmentation/roi_table/is_soma`.
- Plane names look like `VISp_0..3`, `VISl_4..7`, etc.

### SLAP2 (001424)
- dF/F per DMD at `processing/ophys/Fluorescence_<DMD>/<DMD>_dFF`.
- DMD timing offsets relative to trial `start_time`: DMD1 = +0.115 s, DMD2 = −0.165 s.
- ISI in oddball / ori-tuning blocks = 700 ms — keep response windows ≤ 0.35 s.

## Git / Version Control
- Commit figures with `git add -f` (they're in `.gitignore` by default)
- Cache directories (`results/*_cache*/`, `results/rf_cache*/`, `results/stability_cache_*/`) should NOT be committed
- Commit message style: short imperative title + bullet body explaining what and why
- Colab "Created using Colab" auto-commits will arrive on `origin/main` while you work locally — pull/rebase before pushing

## Project Structure
- Scripts: `scripts/` — one file per analysis type, pickle-cached per session
- Results: `results/` — CSVs committed, pickle caches not committed
- Source: `src/openscope_pp/` — loaders in `loaders/streaming.py` and `loaders/trials.py`
- Notebooks: `notebooks/`
  - `openscope_pp_demo.ipynb` — original demo, all three modalities, single-session examples
  - `openscope_pp_comparison.ipynb` — direct cross-technique RF / orientation / oddball comparison
  - `rf_sub830794_probes_ABCD.ipynb` — publication-quality RF figures for sub-830794 (4-panel)
  - `rf_sub830794_probes_ABCD_per_probe.ipynb` — same data, one publication-quality figure per probe (RdBu_r, square tiles)
  - `rf_population_001637.ipynb` — cross-session RF population analysis, probes A–D
  - `rf_population_allprobes_001637.ipynb` — same population analysis but no probe-letter assumption (insertion-level)
  - `stability_snr_multimodal.ipynb` — recording stability / SNR-over-time across all three modalities

## Cache convention
Long pipelines drop one pickle per session under a dedicated directory; the file name is the first 8 chars of the asset id. Re-runs detect existing cache and skip those sessions, so adding new sessions is incremental.
- `results/oddball_cache/` (ecephys oddball)
- `results/oddball_cache_meso/`
- `results/oddball_cache_slap2/`
- `results/rf_cache/` (RF population, A–D)
- `results/rf_cache_allprobes/` (RF population, all probes)
- `results/stability_cache_ecephys/`
- `results/stability_cache_meso/`
- `results/stability_cache_slap2/`

## Plotting / publication-quality conventions
- Vector outputs: save both `.pdf` (vector, editable in Illustrator/Inkscape) and `.png` (raster preview). Set `pdf.fonttype = 42`, `ps.fonttype = 42`, `svg.fonttype = "none"` so text stays selectable.
- Figures with many imshow tiles: mark them `rasterized=True` so the PDF embeds a small image instead of a polygon stream. Axes / labels / colour bars stay vector.
- Memory: PDF rasterisation at high dpi can OOM free Colab. Use `savefig.dpi=200` global, `dpi=120–150` for PDFs of imshow-heavy figures, and `plt.close(fig); gc.collect()` between figures.
- Composite-mosaic trick: when you would otherwise create thousands of tiny Axes (e.g. RF gallery by depth), build a single numpy mosaic and draw it with one `imshow`. Cuts ~3000 Axes to 1 and avoids OOM.
- Probe colour palette: Wong colour-blind-friendly. Default mapping: A=#0072B2, B=#D55E00, C=#009E73, D=#CC79A7, E=#F0E442, F=#56B4E9. Probe colour is fine for *within-session* visualisation; never imply that the same letter = same anatomical location across animals.

## Paradigm / Control Pairings
| Paradigm | Control block kind |
|----------|--------------------|
| SEQUENCE | control_sequential |
| STANDARD | control_standard |
| SENSORYMOTOR | control_replay |
| DURATION | control_standard |

## Dandisets
| Technique | Dandiset | Sessions |
|-----------|----------|----------|
| Ecephys (Neuropixels) | 001637 | 46 across 14 subjects (was 6/2 before 2026-04 expansion) |
| Mesoscope (GCaMP) | 001768 | 42 |
| SLAP2 | 001424 | 9 (skip sub-794237) |

## Known Gotchas
- SLAP2 `sub-794237` uses an older NWB format — always skip it
- SLAP2 DMD timing offsets: DMD1 = +0.115 s, DMD2 = −0.165 s relative to trial `start_time`
- Mesoscope SEQUENCE paradigm: omission MI is **negative** in calcium (suppression, not mismatch) — this is real, not a bug
- Mesoscope STANDARD paradigm omission MI is positive but contaminated: real prediction-error rise lives at ~0.3–0.6 s, but the 0.1–0.8 s window also picks up an adaptation-release-amplified next-stim response at ~0.7–0.8 s (701 ms SOA + GCaMP kinetics). Genuine effect, not directly comparable to ecephys/SLAP2 MIs
- Ecephys 001637 expanded 2026-04 from 6 → 46 sessions across 14 subjects (~10–12 sessions per paradigm). Cross-session t-tests on MI vs. 0 are now appropriate; the old "show-only" caveat is obsolete
- SLAP2 oddball / ori-tuning ISI = 700 ms; response windows must stay ≤ 0.35 s. iGluSnFR is fast enough that 0.05–0.35 s captures the transient cleanly.
- SLAP2 sub-803496 / DMD2 has the strongest visual RFs and orientation tuning, but its dendrites adapt heavily to the 2887 identical standards in the oddball block, producing net-negative average standard responses. For oddball analyses use sub-801381 / DMD1 (asset `d5120d10-…`) — 129 ROIs with clean positive transients.
- Probe letters in ecephys 001637 are per-session device IDs, NOT anatomical labels. Do not pool by letter across subjects. Use insertion-level analyses (subject × session × device_name) or CCF coordinates (`units.estimated_x/y/z`) for cross-session anatomy.
- `units` table in 001637 (2026-04 release) has `depth` directly — no `peak_channel_id`. Don't waste cycles on `electrodes` table lookups for depth.
- `device_name` is camel-case (`"ProbeA"`), not lowercase or with a separator.
