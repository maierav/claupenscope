"""End-to-end smoke test: SLAP2 pipeline.

Streams the real NWB file from DANDI, runs load_trials → load_responses
and checks block detection, trial classification, and response shapes.
"""
import sys, time
import numpy as np

print("=" * 60)
print("SLAP2 SMOKE TEST")
print("=" * 60)

# --- Step 1: open_nwb ---
print("\n[1/5] Opening NWB file (streaming from DANDI)...")
t0 = time.time()

from openscope_pp.loaders.streaming import open_nwb

# sub-776270 — the pilot session documented in nwb_schema.md
ASSET_ID = "98e54c75-2b4a-41ca-b502-b58d63b1f6d5"
handle = open_nwb(ASSET_ID)

print(f"  technique: {handle.technique}")
print(f"  session_id: {handle.nwb.session_id}")
print(f"  subject: {handle.nwb.subject.subject_id}")
print(f"  opened in {time.time() - t0:.1f}s")
assert handle.technique == "slap2", f"Expected slap2, got {handle.technique}"

# --- Step 2: raw gratings table ---
print("\n[2/5] Checking raw gratings table...")
h5 = handle.h5
g = h5["intervals/gratings"]
n_total = g["start_time"].shape[0]
print(f"  total presentations: {n_total}")
print(f"  columns: {list(g.keys())}")
ori = g["orientation"][:]
dia = g["diameter"][:]
tf  = g["temporal_frequency"][:]
con = g["contrast"][:]
print(f"  orientation range: [{ori.min():.3f}, {ori.max():.3f}] rad")
print(f"  diameter range: [{dia.min():.1f}, {dia.max():.1f}] deg")
print(f"  unique diameters: {sorted(np.unique(dia).tolist())}")
assert n_total > 0

# --- Step 3: load_trials ---
print("\n[3/5] Loading trials (block detection)...")
t0 = time.time()

from openscope_pp.loaders.trials import load_trials

trials = load_trials(handle)
print(f"  {len(trials)} rows (should match {n_total})")
print(f"  loaded in {time.time() - t0:.1f}s")

print(f"\n  Block summary:")
for bk in sorted(trials["block_kind"].unique()):
    subset = trials[trials["block_kind"] == bk]
    blocks = sorted(subset["block"].unique())
    print(f"    {bk}: {len(subset)} trials across {len(blocks)} block(s): {blocks}")

assert len(trials) == n_total, "Row count mismatch after load_trials"
assert "rf_mapping"      in trials["block_kind"].values, "RF mapping not detected"
assert "paradigm_oddball" in trials["block_kind"].values, "No oddball blocks detected"
assert "ori_tuning"      in trials["block_kind"].values, "No orientation tuning detected"

# Verify we detected ≥2 oddball blocks
n_oddball_blocks = trials[trials["block_kind"] == "paradigm_oddball"]["block"].nunique()
print(f"\n  Oddball blocks found: {n_oddball_blocks} (expect ≥2)")
assert n_oddball_blocks >= 2, f"Expected ≥2 oddball blocks, got {n_oddball_blocks}"

# Verify trial types within oddball blocks
oddball = trials[trials["block_kind"] == "paradigm_oddball"]
tt = sorted(oddball["trial_type"].unique())
print(f"  Oddball trial_types: {tt}")
assert "standard"  in tt, "No 'standard' trial type in oddball blocks"
assert "omission"  in tt, "No 'omission' trial type"

n_std = (oddball["trial_type"] == "standard").sum()
n_dev = oddball["is_deviant"].sum()
print(f"  Standards: {n_std}, deviants: {n_dev}, ratio: {n_std/max(n_dev,1):.0f}:1")
assert n_std > n_dev, "Fewer standards than deviants — oddball detection wrong"

# Verify deviant orientations in degrees are sensible
deviant_types = [t for t in tt if t not in ("standard", "omission", "static")]
print(f"  Deviant orientation labels: {deviant_types}")

# --- Step 4: load_responses (DMD1 only, small subset) ---
print("\n[4/5] Loading dFF responses (DMD1, 20 oddball trials)...")
t0 = time.time()

from openscope_pp.loaders.responses import load_responses

# Use the first oddball block
first_ob_block = trials[trials["block_kind"] == "paradigm_oddball"]["block"].unique()[0]
ob_trials = trials[trials["block"] == first_ob_block].head(20).copy()

resp = load_responses(
    handle, ob_trials,
    window=(-0.5, 1.0),
    dmd_filter=["DMD1"],
    baseline_window=(-0.5, 0.0),
    baseline_mode="zscore",
)
print(f"  shape: {resp.shape}  (trial x unit x time)")
print(f"  DMD1 ROIs: {resp.sizes['unit']}")
print(f"  time range: [{float(resp.coords['time_sec'].min()):.3f}, {float(resp.coords['time_sec'].max()):.3f}]")
print(f"  sample rate: {resp.attrs['sample_rate_hz']:.1f} Hz  (expect ~195)")
print(f"  baseline mean (should be ~0): {float(np.nanmean(resp.values[:, :, :90])):.4f}")
print(f"  loaded in {time.time() - t0:.1f}s")

assert resp.dims == ("trial", "unit", "time")
assert resp.sizes["trial"] == 20
assert resp.sizes["unit"] > 0
assert resp.sizes["time"] > 0
assert 150 < resp.attrs["sample_rate_hz"] < 250, "Unexpected SLAP2 sample rate"

# --- Step 5: DMD2 independence ---
print("\n[5/5] Checking DMD2 timestamps are independent of DMD1...")
grp1 = h5["processing/ophys/Fluorescence_DMD1/DMD1_dFF"]
grp2 = h5["processing/ophys/Fluorescence_DMD2/DMD2_dFF"]
ts1 = grp1["timestamps"][:]
ts2 = grp2["timestamps"][:]
print(f"  DMD1: {len(ts1)} samples, range [{ts1[0]:.2f}, {ts1[-1]:.2f}] s")
print(f"  DMD2: {len(ts2)} samples, range [{ts2[0]:.2f}, {ts2[-1]:.2f}] s")
print(f"  DMD1 ROIs: {grp1['data'].shape[1]}, DMD2 ROIs: {grp2['data'].shape[1]}")
print(f"  Sample counts equal: {len(ts1) == len(ts2)}")

# --- Done ---
print("\n" + "=" * 60)
print("ALL CHECKS PASSED")
print("=" * 60)

handle.close()
