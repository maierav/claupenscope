# openscope-pp

Analysis toolkit for the [OpenScope Community Predictive Processing](https://allenneuraldynamics.github.io/openscope-community-predictive-processing/) project.

Streams NWB files from [DANDI](https://dandiarchive.org) and provides a common trial-aligned representation across three recording techniques:

| Technique | Dandiset | Signal |
|---|---|---|
| Neuropixels ecephys | [001637](https://dandiarchive.org/dandiset/001637) | Spike-sorted units (Kilosort4) + LFP |
| Mesoscope 2-photon | [001768](https://dandiarchive.org/dandiset/001768) | ΔF/F, events (jGCaMP8s), 8 simultaneous planes |
| SLAP2 glutamate imaging | [001424](https://dandiarchive.org/dandiset/001424) | ΔF/F per DMD (dendrite-resolution) |

## Demo notebook

**[→ Open in Colab](https://colab.research.google.com/github/maierav/claupenscope/blob/main/notebooks/openscope_pp_demo.ipynb)**

Runs end-to-end in ~4 min. Covers mesoscope RF mapping (VISp plane 0, 350 soma ROIs) and ecephys oddball PSTH split by deviant type. All data stream from DANDI — nothing downloaded locally.

## Install

```bash
pip install git+https://github.com/maierav/claupenscope.git
```

## Quick start

```python
from openscope_pp.loaders import open_nwb, load_trials, load_responses

# Stream any session by DANDI asset ID (auto-detects technique)
handle = open_nwb("cd175e65-8faa-4216-86af-c1fd30e571a1")  # ecephys

# Unified trials DataFrame — same schema for all three techniques
trials = load_trials(handle)
print(trials["block_kind"].value_counts())
# paradigm_oddball    4320
# rf_mapping           540
# spontaneous          ...

oddball = trials[trials["block_kind"] == "paradigm_oddball"]
handle.close()
```

## Technique notes

### Neuropixels ecephys (dandiset 001637)

- Units classified as `sua` / `mua` / `noise` via `decoder_label`
- Visual units identified by RF peak response (≥ 5 spk/s above baseline)
- Oddball paradigms: sequence mismatch, sensorimotor mismatch, duration mismatch
- Deviant types include: `halt`, `omission`, `orientation_45`, `orientation_90`, `sequence_omission`
- LFP available for CSD analysis

```python
# Filter to visually-responsive single units
h5 = handle.h5
dl = h5["units"]["decoder_label"][:]
sua_mask = [v.decode() == "sua" for v in dl]
spike_times = h5["units"]["spike_times"][:]
```

### Mesoscope 2-photon (dandiset 001768)

- 8 simultaneous imaging planes: VISp_0–3 (primary visual) and VISl_4–7 (lateral visual)
- Soma-only ROIs via `is_soma` flag (~70–95% of ROIs per plane)
- RF mapping: 9×9 spatial grid (±40°), 15 trials/position, ~10.7 Hz imaging rate
- Response window (0.1–0.8 s) accounts for slow GCaMP dynamics
- dFF snippets extracted via `np.interp` over finite timestamps — robust to gaps and clock drift

```python
# Load dFF + soma mask for one plane
plane = "VISp_0"
ts   = h5[f"processing/{plane}/dff_timeseries/dff_timeseries/timestamps"][:]
data = h5[f"processing/{plane}/dff_timeseries/dff_timeseries/data"]        # lazy
is_soma = h5[f"processing/{plane}/image_segmentation/roi_table/is_soma"][:].astype(bool)

# Or use load_responses() for automatic handling across planes
responses = load_responses(
    handle, trials,
    signal_type="dff",      # also: "events", "neuropil_corrected", "neuropil", "raw"
    soma_only=True,
    baseline_window=(-0.5, 0.0),
    baseline_mode="zscore", # or "subtract" / "divide"
)
```

### SLAP2 glutamate imaging (dandiset 001424)

- Two DMDs (digital micromirror devices) image separate dendritic fields simultaneously
- **Timing offsets must be applied per DMD** (calibration burst at recording start creates a ~2.9 MHz phase that corrupts `dt` estimates from early timestamps):
  - DMD1: +115 ms
  - DMD2: −165 ms
- `dt` should be estimated from mid-recording: `np.median(np.diff(ts[len(ts)//2 : len(ts)//2 + 2000]))`
- RF block falls at the end of the session; DMD1 and DMD2 may stop at different times, so valid trial counts differ

```python
# Corrected trial onsets for DMD2
offset_sec = -0.165
onsets = rf_trials["start_time"].values + offset_sec

# dt from mid-recording (avoids 2.9 MHz calibration burst)
mid = len(ts) // 2
dt  = float(np.median(np.diff(ts[mid : mid + 2000])))
```

## Analysis scripts

Ready-to-run scripts in `scripts/` — each connects to DANDI, runs the analysis, and saves figures:

| Script | Technique | Description |
|---|---|---|
| `rf_mapping_ecephys.py` | Ecephys | SUA RF maps from spike histograms |
| `rf_mapping_mesoscope.py` | Mesoscope | Per-ROI RF maps across all 8 planes |
| `rf_mapping_slap2.py` | SLAP2 | RF maps for DMD1 + DMD2 with timing correction |
| `oddball_by_type_ecephys.py` | Ecephys | Oddball PSTH split by deviant type |
| `plot_oddball_responses.py` | Ecephys | Standard vs deviant population response |
| `diagnose_rf_ecephys.py` | Ecephys | RF alignment diagnostic plots |

## Package structure

```
src/openscope_pp/
├── loaders/
│   ├── streaming.py   # open_nwb() — stream from DANDI, auto-detect technique
│   ├── trials.py      # load_trials() — per-technique block/trial detection
│   ├── responses.py   # load_responses() — trial-aligned neural data
│   └── behavior.py    # load_behavior() — running speed, pupil
└── analysis/
    ├── oddball.py     # mismatch index, standard vs deviant statistics
    ├── orientation.py # OSI, DSI, preferred orientation
    ├── rf_mapping.py  # RF maps, Gaussian fits, centre-of-mass
    └── csd.py         # laminar LFP + current source density (ecephys)
```
