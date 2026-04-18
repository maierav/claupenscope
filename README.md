# openscope-pp

Analysis toolkit for the [OpenScope Community Predictive Processing](https://allenneuraldynamics.github.io/openscope-community-predictive-processing/) project.

Streams NWB files from [DANDI](https://dandiarchive.org) (dandisets 001637, 001768, 001424) and provides a common trial-aligned representation across three recording techniques:

| Technique | Dandiset | Signal |
|---|---|---|
| Neuropixels ecephys | 001637 | Spike-sorted units (Kilosort4) + LFP |
| Mesoscope 2-photon | 001768 | ΔF/F, events (jGCaMP8s) |
| SLAP2 glutamate imaging | 001424 | ΔF/F per DMD |

## Install (Colab / pip)

```python
!pip install git+https://github.com/maierav/claupenscope.git
```

## Quick start

```python
from openscope_pp.loaders import open_nwb, load_trials, load_responses

# Stream any session by DANDI asset ID
handle = open_nwb("cd175e65-8faa-4216-86af-c1fd30e571a1")  # ecephys example

# Common trials DataFrame — same schema for all three techniques
trials = load_trials(handle)
oddball = trials[trials["block_kind"] == "paradigm_oddball"]

# Trial-aligned responses → xr.DataArray (trial, unit, time)
responses = load_responses(
    handle, oddball,
    window=(-0.5, 1.0),
    unit_filter={"decoder_label": "sua"},   # ecephys: SUA only
)

handle.close()
```

## Structure

```
src/openscope_pp/
├── loaders/
│   ├── streaming.py   # open_nwb() — stream from DANDI
│   ├── trials.py      # load_trials() — per-technique block detection
│   ├── responses.py   # load_responses() — trial-aligned neural data
│   └── behavior.py    # load_behavior() — running, pupil
└── analysis/
    ├── oddball.py     # mismatch index, standard vs deviant stats
    ├── orientation.py # OSI, DSI, preferred orientation
    ├── rf_mapping.py  # RF maps, centre-of-mass
    └── csd.py         # laminar LFP + CSD (ecephys)
```

## Mesoscope signal types

```python
# Choose signal: "dff", "events", "neuropil_corrected", "neuropil", "raw"
responses = load_responses(
    handle, trials,
    signal_type="dff",
    soma_only=True,                      # filter to is_soma == 1
    baseline_window=(-0.5, 0.0),        # pre-stimulus baseline
    baseline_mode="zscore",             # or "subtract" / "divide"
)
```
