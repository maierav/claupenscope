"""Ecephys oddball PSTH split by deviant type — visual SUA only.

Shows one line per trial type (standard + each deviant) so the different
mismatch signatures don't blur together.

Saves: oddball_by_type_ecephys.png
"""
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from openscope_pp.loaders.streaming import open_nwb
from openscope_pp.loaders.trials import load_trials

# ── Config ────────────────────────────────────────────────────────────
ASSET_ID     = "cd175e65-8faa-4216-86af-c1fd30e571a1"
RF_WINDOW    = (-0.1, 0.5)
RF_BIN       = 0.01
RESP_WIN     = (0.03, 0.25)
RESP_THRESH  = 5.0           # spk/s — visual unit threshold
RF_SMOOTH    = 0.8

ODD_WINDOW   = (-0.5, 1.0)
ODD_BIN      = 0.025
N_STD_MAX    = 300

COLORS = {
    "standard":         "#4878CF",
    "halt":             "#D65F5F",
    "omission":         "#E07B39",
    "orientation_45":   "#6BAF6B",
    "orientation_90":   "#9B59B6",
    "sequence_omission":"#C0392B",
}

# ── Helpers ───────────────────────────────────────────────────────────
def decode(arr):
    if arr.dtype.kind in ("S", "O"):
        return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in arr])
    return arr

def bin_spikes(sua_idx, all_spikes, spike_index, t0_arr, window, bin_size):
    pre, post = window
    edges   = np.arange(pre, post + bin_size, bin_size)
    centers = 0.5 * (edges[:-1] + edges[1:])
    n_bins  = len(centers)
    out = np.zeros((len(t0_arr), len(sua_idx), n_bins), dtype=np.float32)
    for j, uid in enumerate(sua_idx):
        i0  = int(spike_index[uid - 1]) if uid > 0 else 0
        i1  = int(spike_index[uid])
        spk = all_spikes[i0:i1]
        for i, t0 in enumerate(t0_arr):
            rel    = spk - t0
            in_win = rel[(rel >= pre) & (rel < post)]
            if len(in_win):
                cnt, _ = np.histogram(in_win, bins=edges)
                out[i, j, :] = cnt / bin_size
    return out, centers

def zscore_unit(arr, centers, bl=(-0.5, 0.0)):
    """Z-score each unit using its own baseline mean/std across all provided trials."""
    bl_mask = (centers >= bl[0]) & (centers < bl[1])
    mu  = arr[:, :, bl_mask].mean(axis=(0, 2), keepdims=True)
    sig = arr[:, :, bl_mask].std(axis=(0, 2), keepdims=True)
    sig = np.where(sig > 0.1, sig, 1.0)
    return (arr - mu) / sig

def mean_sem(arr):
    """(trial, unit, time) → mean and SEM of per-trial unit-averages."""
    per_trial = arr.mean(axis=1)
    return per_trial.mean(axis=0), per_trial.std(axis=0) / np.sqrt(per_trial.shape[0])

# ── Load ──────────────────────────────────────────────────────────────
print("Opening NWB…")
t0 = time.time()
handle = open_nwb(ASSET_ID)
trials = load_trials(handle)
print(f"  opened in {time.time()-t0:.1f}s")

h5          = handle.h5
units_grp   = h5["units"]
n_total     = int(units_grp["id"].shape[0])
dl          = decode(units_grp["decoder_label"][:])
sua_idx     = np.where(dl == "sua")[0]
n_sua       = len(sua_idx)
all_spikes  = units_grp["spike_times"][:]
spike_index = units_grp["spike_times_index"][:]

# ── Step 1: identify visual units via RF mapping ──────────────────────
print("Finding visual units via RF mapping…")
rf_t = trials[trials["block_kind"] == "rf_mapping"].copy().reset_index(drop=True)

t1 = time.time()
rf_arr, rf_centers = bin_spikes(
    sua_idx, all_spikes, spike_index,
    rf_t["start_time"].values, RF_WINDOW, RF_BIN,
)
print(f"  RF binned in {time.time()-t1:.1f}s")

xs = np.array(sorted(rf_t["x"].unique()))
ys = np.array(sorted(rf_t["y"].unique()))
bl_mask   = rf_centers < 0
resp_mask = (rf_centers >= RESP_WIN[0]) & (rf_centers < RESP_WIN[1])
bl_rate   = rf_arr[:, :, bl_mask].mean(axis=2)
resp_rate = rf_arr[:, :, resp_mask].mean(axis=2)
delta     = resp_rate - bl_rate

rf_maps  = np.zeros((n_sua, len(ys), len(xs)), dtype=np.float32)
for xi, x in enumerate(xs):
    for yi, y in enumerate(ys):
        m = (rf_t["x"].values == x) & (rf_t["y"].values == y)
        rf_maps[:, yi, xi] = delta[m, :].mean(axis=0)

peak_dfr    = np.array([gaussian_filter(rf_maps[u], RF_SMOOTH).max() for u in range(n_sua)])
vis_mask    = peak_dfr >= RESP_THRESH
vis_idx     = sua_idx[vis_mask]
n_vis       = len(vis_idx)

print(f"\n  Unit filter summary:")
print(f"    All units : {n_total:4d}  (100%)")
print(f"    SUA       : {n_sua:4d}  ({100*n_sua/n_total:.1f}% of all)")
print(f"    Visual SUA: {n_vis:4d}  ({100*n_vis/n_sua:.1f}% of SUA, {100*n_vis/n_total:.1f}% of all)")

# ── Step 2: oddball trials, split by type ────────────────────────────
odd_t = trials[trials["block_kind"] == "paradigm_oddball"].copy()
first_block = odd_t["block"].unique()[0]
ob = odd_t[odd_t["block"] == first_block]

trial_types = ["standard"] + sorted(ob[ob["is_deviant"]]["trial_type"].unique().tolist())
print(f"\n  Trial types in {first_block}: {trial_types}")

# Collect per-type trials
type_trials = {}
for tt in trial_types:
    subset = ob[ob["trial_type"] == tt]
    if tt == "standard" and len(subset) > N_STD_MAX:
        subset = subset.sample(N_STD_MAX, random_state=42)
    type_trials[tt] = subset.reset_index(drop=True)
    print(f"    {tt:20s}: {len(subset):5d} trials")

# ── Step 3: bin spikes for each type ─────────────────────────────────
print("\nBinning oddball spikes by type…")
type_arrays = {}
t1 = time.time()
# Bin all at once by pooling, then split — faster than separate calls
all_odd = ob.copy()
std_sub = type_trials["standard"]
# We bin per type separately (manageable given n_vis is small)
for tt, sub in type_trials.items():
    arr, centers = bin_spikes(
        vis_idx, all_spikes, spike_index,
        sub["start_time"].values, ODD_WINDOW, ODD_BIN,
    )
    type_arrays[tt] = arr
    print(f"  {tt:20s}: {arr.shape}")
print(f"  Done in {time.time()-t1:.1f}s")

# ── Step 4: z-score — baseline computed from standard trials ──────────
# Use the standard condition to set the baseline (most trials, most stable)
std_arr = type_arrays["standard"]
bl_t    = (centers >= -0.5) & (centers < 0.0)
mu  = std_arr[:, :, bl_t].mean(axis=(0, 2), keepdims=True)   # (1, unit, 1)
sig = std_arr[:, :, bl_t].std(axis=(0, 2), keepdims=True)
sig = np.where(sig > 0.1, sig, 1.0)

def apply_z(arr):
    return (arr - mu) / sig

type_z = {tt: apply_z(arr) for tt, arr in type_arrays.items()}

# ── Step 5: plot ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))

ax.axvspan(0, 0.25, color="gray", alpha=0.10)
ax.axhline(0,  color="gray", lw=0.6, ls="--")
ax.axvline(0,  color="gray", lw=0.8, ls=":")
ax.axvline(0.25, color="gray", lw=0.6, ls=":")

# Standard first (background)
m, s = mean_sem(type_z["standard"])
ax.fill_between(centers, m - s, m + s, color=COLORS["standard"], alpha=0.20)
ax.plot(centers, m, color=COLORS["standard"], lw=2.5,
        label=f"standard  (n={len(type_trials['standard'])})", zorder=3)

# Deviants
for tt in trial_types[1:]:
    color = COLORS.get(tt, "#888888")
    m, s  = mean_sem(type_z[tt])
    n     = len(type_trials[tt])
    ax.fill_between(centers, m - s, m + s, color=color, alpha=0.18)
    ax.plot(centers, m, color=color, lw=2,
            label=f"{tt}  (n={n})", zorder=4)

ax.set_xlabel("Time from stimulus onset (s)", fontsize=11)
ax.set_ylabel("Population response (z-score)", fontsize=11)
ax.set_title(
    f"Ecephys oddball PSTH — SEQUENCE block — deviant types split\n"
    f"Visual SUA: {n_vis}/{n_sua} ({100*n_vis/n_sua:.0f}% of SUA, "
    f"{100*n_vis/n_total:.0f}% of all units)",
    fontsize=10,
)
ax.set_xlim(ODD_WINDOW)
ax.legend(fontsize=8.5, framealpha=0.8, loc="upper right")
ax.spines[["top", "right"]].set_visible(False)

# Annotate sequence period
ax.annotate("↑ sequence echoes", xy=(0.35, 0.06), fontsize=7.5, color="gray",
            xycoords=("data", "axes fraction"))

fig.tight_layout()
out = "oddball_by_type_ecephys.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved → {out}")
handle.close()
