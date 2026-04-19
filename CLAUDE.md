# CLAUDE.md — openscope-pp project conventions

## Environment
- **Python**: `python3.13` for scripts requiring `dandi` + `numpy` (both available)
- **Anaconda python** (`/opt/anaconda3/bin/python`) has numpy but NOT dandi — don't use for pipeline scripts
- **Colab outputs**: always write to local Colab disk (`/content/`), never Google Drive unless explicitly asked

## Task Completion Rules
- Never claim a task is done if there were errors during execution or if the output file/figure hasn't been verified
- Before reporting completion: run the script, show the last ~20 lines of output, confirm figures saved
- If a background process is running, check actual output file — don't infer status from prior messages

## Data & NumPy Conventions
- **Always check for inhomogeneous arrays** before stacking cross-session data — sessions have different frame rates → different PSTH lengths
- Fix: interpolate to `COMMON_T = np.linspace(window[0], window[1], 300)` before calling `np.array([...])`
- **dF/F extraction**: use `np.interp` on the timestamp axis — robust to clock drift and gaps
- **Soma mask**: always apply `is_soma` filter before extracting fluorescence traces
- When reading HDF5 lazy datasets, do a single bulk read over the time span of interest, then interpolate — don't index row-by-row

## Git / Version Control
- Commit figures with `git add -f` (they're in `.gitignore` by default)
- Cache directories (`results/oddball_cache*/`) should NOT be committed
- Commit message style: short imperative title + bullet body explaining what and why

## Project Structure
- Scripts: `scripts/` — one file per analysis type, pickle-cached per session
- Results: `results/` — CSVs committed, pickle caches not committed
- Notebook: `notebooks/openscope_pp_demo.ipynb` — Colab-ready, 7 sections
- Source: `src/openscope_pp/` — loaders in `loaders/streaming.py` and `loaders/trials.py`

## Dandisets
| Technique | Dandiset | Sessions |
|-----------|----------|----------|
| Ecephys (Neuropixels) | 001637 | 6 |
| Mesoscope (GCaMP) | 001768 | 42 |
| SLAP2 | 001424 | 9 (skip sub-794237) |

## Paradigm / Control Pairings
| Paradigm | Control block kind |
|----------|--------------------|
| SEQUENCE | control_sequential |
| STANDARD | control_standard |
| SENSORYMOTOR | control_replay |
| DURATION | control_standard |

## Known Gotchas
- SLAP2 sub-794237 uses an older NWB format — always skip it
- SLAP2 DMD timing offsets: DMD1 = +0.115 s, DMD2 = −0.165 s relative to trial `start_time`
- Mesoscope SEQUENCE paradigm: omission MI is **negative** in calcium (suppression, not mismatch) — this is real, not a bug
- Ecephys has only 1–2 sessions per paradigm — insufficient for t-tests, show data only
