"""End-to-end smoke test: ecephys pipeline.

Streams the real NWB file from DANDI, runs load_trials → load_responses
on a small subset, and checks shapes/dtypes.
"""
import sys, time

print("=" * 60)
print("ECEPHYS SMOKE TEST")
print("=" * 60)

# --- Step 1: open_nwb ---
print("\n[1/5] Opening NWB file (streaming from DANDI)...")
t0 = time.time()

from openscope_pp.loaders.streaming import open_nwb

# sub-820459, session 2025-11-12 (the one we documented in nwb_schema.md)
ASSET_ID = "cd175e65-8faa-4216-86af-c1fd30e571a1"
handle = open_nwb(ASSET_ID)

print(f"  technique: {handle.technique}")
print(f"  session_id: {handle.nwb.session_id}")
print(f"  subject: {handle.nwb.subject.subject_id}")
print(f"  opened in {time.time() - t0:.1f}s")
assert handle.technique == "ecephys", f"Expected ecephys, got {handle.technique}"

# --- Step 2: load_trials ---
print("\n[2/5] Loading trials...")
t0 = time.time()

from openscope_pp.loaders.trials import load_trials

trials = load_trials(handle)
print(f"  {len(trials)} total rows, {trials['block'].nunique()} blocks")
print(f"  columns: {list(trials.columns)}")
print(f"  block_kind values: {sorted(trials['block_kind'].unique())}")
print(f"  paradigm values: {sorted(trials['paradigm'].dropna().unique())}")
print(f"  loaded in {time.time() - t0:.1f}s")

# Basic sanity checks
assert len(trials) > 0, "No trials loaded"
assert "start_time" in trials.columns
assert "is_deviant" in trials.columns
assert trials["start_time"].is_monotonic_increasing, "Trials not sorted by time"

# Check that paradigm oddball blocks have trial_type values
oddball_trials = trials[trials["block_kind"] == "paradigm_oddball"]
if len(oddball_trials) > 0:
    tt_values = sorted(oddball_trials["trial_type"].unique())
    print(f"  oddball trial_types: {tt_values}")
    assert "standard" in tt_values, "No 'standard' trial type in oddball blocks"
    n_std = (oddball_trials["trial_type"] == "standard").sum()
    n_dev = (oddball_trials["is_deviant"]).sum()
    print(f"  standards: {n_std}, deviants: {n_dev}, ratio: {n_std/max(n_dev,1):.0f}:1")

# Check optotagging blocks detected
opto = trials[trials["block_kind"] == "optotagging"]
print(f"  optotagging rows: {len(opto)}")

# --- Step 3: load_responses (small subset) ---
print("\n[3/5] Loading spike responses (first 20 oddball trials, SUA only)...")
t0 = time.time()

from openscope_pp.loaders.responses import load_responses

# Take a small slice to keep it fast
test_trials = oddball_trials.head(20).copy() if len(oddball_trials) >= 20 else trials.head(20).copy()

responses = load_responses(
    handle, test_trials,
    window=(-0.2, 0.5),
    bin_size=0.01,
    unit_filter={"decoder_label": "sua"},
)
print(f"  shape: {responses.shape}  (trial x unit x time)")
print(f"  units: {responses.sizes['unit']}")
print(f"  time bins: {responses.sizes['time']}")
print(f"  time range: [{float(responses.coords['time_sec'].min()):.3f}, {float(responses.coords['time_sec'].max()):.3f}]")
print(f"  dtype: {responses.dtype}")
if responses.sizes["unit"] > 0:
    print(f"  mean spike count per bin: {float(responses.mean()):.4f}")
    print(f"  max spike count per bin: {float(responses.max()):.0f}")
else:
    print("  WARNING: 0 units matched filter!")
print(f"  loaded in {time.time() - t0:.1f}s")

assert responses.dims == ("trial", "unit", "time")
assert responses.sizes["trial"] == len(test_trials)
assert responses.sizes["time"] > 0
assert responses.sizes["unit"] > 0, "Unit filter returned 0 units — byte-string decoding bug?"
assert float(responses.min()) >= 0, "Negative spike counts"

# --- Step 4: load_behavior ---
print("\n[4/5] Loading behavior...")
t0 = time.time()

from openscope_pp.loaders.behavior import load_behavior

behavior = load_behavior(handle)
print(f"  columns: {list(behavior.columns)}")
print(f"  rows: {len(behavior)}")
print(f"  time range: [{behavior.index.min():.1f}, {behavior.index.max():.1f}]")
print(f"  loaded in {time.time() - t0:.1f}s")

assert len(behavior) > 0
assert "running_speed" in behavior.columns

# --- Step 5: CSD metadata ---
print("\n[5/5] Checking LFP/CSD metadata...")
t0 = time.time()

from openscope_pp.analysis.csd import list_probes, load_lfp_metadata

probes = list_probes(handle)
print(f"  available probes: {probes}")

if probes:
    meta = load_lfp_metadata(handle, probe=probes[0])
    print(f"  {meta.probe}: {meta.n_channels} channels, depth range {meta.depths_sorted.min():.0f}-{meta.depths_sorted.max():.0f} um")
    print(f"  loaded in {time.time() - t0:.1f}s")

# --- Done ---
print("\n" + "=" * 60)
print("ALL CHECKS PASSED")
print("=" * 60)

handle.close()
