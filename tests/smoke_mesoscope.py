"""End-to-end smoke test: mesoscope pipeline.

Streams the real NWB file from DANDI, runs load_trials → load_responses
with all signal types + baseline correction, and checks shapes/values.
"""
import sys, time
import numpy as np

print("=" * 60)
print("MESOSCOPE SMOKE TEST")
print("=" * 60)

# --- Step 1: open_nwb ---
print("\n[1/6] Opening NWB file (streaming from DANDI)...")
t0 = time.time()

from openscope_pp.loaders.streaming import open_nwb

# sub-832700, OPTICAL_SESSION1_SENSORYMOTOR
ASSET_ID = "55babc82-9551-4df7-b64f-572d6ec21415"
handle = open_nwb(ASSET_ID)

print(f"  technique: {handle.technique}")
print(f"  session_id: {handle.nwb.session_id}")
print(f"  subject: {handle.nwb.subject.subject_id}")
print(f"  genotype: {handle.nwb.subject.genotype}")
print(f"  opened in {time.time() - t0:.1f}s")
assert handle.technique == "mesoscope"

# --- Step 2: load_trials ---
print("\n[2/6] Loading trials...")
t0 = time.time()

from openscope_pp.loaders.trials import load_trials

trials = load_trials(handle)
print(f"  {len(trials)} total rows, {trials['block'].nunique()} blocks")
print(f"  block_kind values: {sorted(trials['block_kind'].unique())}")
print(f"  paradigm values: {sorted(trials['paradigm'].dropna().unique())}")
print(f"  loaded in {time.time() - t0:.1f}s")

assert len(trials) > 0
assert "start_time" in trials.columns
assert trials["start_time"].is_monotonic_increasing

oddball = trials[trials["block_kind"] == "paradigm_oddball"]
print(f"  oddball trials: {len(oddball)}, trial_types: {sorted(oddball['trial_type'].unique())}")
assert len(oddball) > 0

# --- Step 3: dFF responses (soma-only, no baseline) ---
print("\n[3/6] Loading dFF responses (soma-only, single plane, 20 trials)...")
t0 = time.time()

from openscope_pp.loaders.responses import load_responses

test_trials = oddball.head(20).copy()

resp_dff = load_responses(
    handle, test_trials,
    window=(-0.5, 1.0),
    signal_type="dff",
    soma_only=True,
    plane_filter=["VISp_0"],   # just one plane to keep it fast
)
print(f"  shape: {resp_dff.shape}  (trial x unit x time)")
print(f"  soma ROIs in VISp_0: {resp_dff.sizes['unit']}")
print(f"  time range: [{float(resp_dff.coords['time_sec'].min()):.3f}, {float(resp_dff.coords['time_sec'].max()):.3f}]")
print(f"  sample rate: {resp_dff.attrs['sample_rate_hz']:.2f} Hz")
print(f"  mean dF/F: {float(np.nanmean(resp_dff.values)):.4f}")
print(f"  loaded in {time.time() - t0:.1f}s")

assert resp_dff.dims == ("trial", "unit", "time")
assert resp_dff.sizes["trial"] == 20
assert resp_dff.sizes["unit"] > 0, "No soma ROIs found — is_soma filter broken?"
assert resp_dff.sizes["time"] > 0
assert resp_dff.attrs["signal_type"] == "dff"
assert resp_dff.attrs["soma_only"] is True

# --- Step 4: z-scored dFF with pre-stimulus baseline ---
print("\n[4/6] Loading z-scored dFF (baseline -0.5 to 0.0 s)...")
t0 = time.time()

resp_z = load_responses(
    handle, test_trials,
    window=(-0.5, 1.0),
    signal_type="dff",
    soma_only=True,
    plane_filter=["VISp_0"],
    baseline_window=(-0.5, 0.0),
    baseline_mode="zscore",
)
print(f"  shape: {resp_z.shape}")
print(f"  baseline mean (should be ~0): {float(np.nanmean(resp_z.values[:, :, :5])):.4f}")
print(f"  post-stim mean: {float(np.nanmean(resp_z.values[:, :, 5:])):.4f}")
print(f"  loaded in {time.time() - t0:.1f}s")

# After z-scoring, baseline period should average near 0
bl_mean = float(np.nanmean(resp_z.values[:, :, :5]))
assert abs(bl_mean) < 0.5, f"Baseline mean after z-score too large: {bl_mean}"

# --- Step 5: events signal ---
print("\n[5/6] Loading deconvolved events (soma-only, VISp_0, 20 trials)...")
t0 = time.time()

resp_ev = load_responses(
    handle, test_trials,
    window=(-0.5, 1.0),
    signal_type="events",
    soma_only=True,
    plane_filter=["VISp_0"],
)
print(f"  shape: {resp_ev.shape}")
print(f"  events >= 0: {bool(np.all(np.nan_to_num(resp_ev.values) >= 0))}")
print(f"  mean event rate: {float(np.nanmean(resp_ev.values)):.5f}")
print(f"  loaded in {time.time() - t0:.1f}s")

assert resp_ev.sizes["unit"] == resp_dff.sizes["unit"], "ROI count mismatch between signals"
assert resp_ev.attrs["signal_type"] == "events"

# --- Step 6: behavior ---
print("\n[6/6] Loading behavior...")
t0 = time.time()

from openscope_pp.loaders.behavior import load_behavior

behavior = load_behavior(handle)
print(f"  columns: {list(behavior.columns)}")
print(f"  rows: {len(behavior)}")
print(f"  time range: [{behavior.index.min():.1f}, {behavior.index.max():.1f}]")
print(f"  loaded in {time.time() - t0:.1f}s")

assert len(behavior) > 0
assert "running_speed" in behavior.columns

# --- Summary ---
print("\n" + "=" * 60)
print("ALL CHECKS PASSED")
print("=" * 60)
print(f"\nKey facts about this session:")
print(f"  Paradigm: {sorted(trials['paradigm'].dropna().unique())}")
print(f"  Total ROIs (VISp_0 somas): {resp_dff.sizes['unit']}")
print(f"  Imaging rate: {resp_dff.attrs['sample_rate_hz']:.2f} Hz")
print(f"  Oddball trial types: {sorted(oddball['trial_type'].unique())}")

handle.close()
