"""Plot trial-triggered oddball responses across all three techniques.

Saves: oddball_responses.png

Ecephys  : population PSTH (mean firing rate across SUA units, z-scored to baseline)
Mesoscope: mean z-scored ΔF/F across VISp_0 soma ROIs
SLAP2    : mean z-scored ΔF/F across DMD1 ROIs
"""
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from openscope_pp.loaders.streaming import open_nwb
from openscope_pp.loaders.trials import load_trials

# ── Configuration ────────────────────────────────────────────────────
WINDOW     = (-0.5, 1.0)
BASELINE   = (-0.5, 0.0)
N_STD_MAX  = 200   # max standard trials to use (keeps ecephys fast)

ASSETS = {
    "ecephys":   "cd175e65-8faa-4216-86af-c1fd30e571a1",  # sub-820459 SEQUENCE
    "mesoscope": "55babc82-9551-4df7-b64f-572d6ec21415",  # sub-832700 SENSORYMOTOR
    "slap2":     "98e54c75-2b4a-41ca-b502-b58d63b1f6d5",  # sub-776270 pilot
}

TECH_LABELS = {
    "ecephys":   "Ecephys\n(SUA population z-score)",
    "mesoscope": "Mesoscope\n(soma ΔF/F z-score)",
    "slap2":     "SLAP2\n(DMD1 ΔF/F z-score)",
}
COLORS = {"standard": "#4878CF", "deviant": "#D65F5F"}


# ── Helper: mean ± SEM across units, per time bin ────────────────────
def mean_sem(arr):
    """arr: (trial, unit, time) → mean (time,), sem (time,) averaged first over
    units then over trials, so each trial contributes one data point."""
    # trial-mean across units → (trial, time)
    per_trial = np.nanmean(arr, axis=1)
    m = np.nanmean(per_trial, axis=0)
    s = np.nanstd(per_trial, axis=0) / np.sqrt(per_trial.shape[0])
    return m, s


# ── Ecephys: fast spike-array reader ────────────────────────────────
def ecephys_responses_fast(handle, trials_df, window, bin_size=0.025):
    """Read all spike times in one HDF5 call, then bin per unit."""
    h5 = handle.h5
    units = h5["units"]

    # SUA mask
    dl = units["decoder_label"][:]
    if dl.dtype.kind in ("S", "O"):
        dl = np.array([v.decode() if isinstance(v, bytes) else str(v) for v in dl])
    sua_mask = dl == "sua"
    sua_idx  = np.where(sua_mask)[0]

    # Read ALL spike times in one contiguous read
    all_spikes  = units["spike_times"][:]
    spike_index = units["spike_times_index"][:]

    pre, post = window
    edges = np.arange(pre, post + bin_size, bin_size)
    centers = 0.5 * (edges[:-1] + edges[1:])
    n_bins = len(centers)

    t0_arr = trials_df["start_time"].values
    n_trials = len(t0_arr)
    n_units  = len(sua_idx)

    result = np.zeros((n_trials, n_units, n_bins), dtype=np.float32)

    for j, uid in enumerate(sua_idx):
        i0 = int(spike_index[uid - 1]) if uid > 0 else 0
        i1 = int(spike_index[uid])
        spikes = all_spikes[i0:i1]

        for i, t0 in enumerate(t0_arr):
            rel = spikes - t0
            in_win = rel[(rel >= pre) & (rel < post)]
            if len(in_win):
                counts, _ = np.histogram(in_win, bins=edges)
                result[i, j, :] = counts / bin_size   # → spikes/s

    return result.astype(np.float64), centers


def zscore_baseline(arr, time_centers, baseline):
    """Z-score each unit to its own baseline mean/std across all trials.
    arr: (trial, unit, time)
    """
    bl_pre, bl_post = baseline
    bl_mask = (time_centers >= bl_pre) & (time_centers < bl_post)
    bl = arr[:, :, bl_mask]                    # (trial, unit, n_bl)
    mu  = np.nanmean(bl, axis=(0, 2), keepdims=True)   # (1, unit, 1)
    sig = np.nanstd( bl, axis=(0, 2), keepdims=True)
    sig = np.where(sig > 1e-9, sig, 1.0)
    return (arr - mu) / sig


# ── Main ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)

for ax, (tech, asset_id) in zip(axes, ASSETS.items()):
    print(f"\n{'='*50}")
    print(f"  {tech.upper()}")
    print(f"{'='*50}")

    t_open = time.time()
    handle = open_nwb(asset_id)
    print(f"  opened in {time.time()-t_open:.1f}s")

    trials = load_trials(handle)
    oddball = trials[trials["block_kind"] == "paradigm_oddball"]

    # Use first oddball block only (cleanest signal, avoids block effects)
    first_block = oddball["block"].unique()[0]
    ob = oddball[oddball["block"] == first_block]

    std_trials = ob[ob["trial_type"] == "standard"]
    dev_trials = ob[ob["is_deviant"]]

    # Subsample standards to keep things fast
    if len(std_trials) > N_STD_MAX:
        std_trials = std_trials.sample(N_STD_MAX, random_state=42)

    print(f"  standards: {len(std_trials)}, deviants: {len(dev_trials)}")

    # ── Load responses ────────────────────────────────────────────
    t_resp = time.time()

    if tech == "ecephys":
        std_arr, t_centers = ecephys_responses_fast(handle, std_trials, WINDOW)
        dev_arr, _         = ecephys_responses_fast(handle, dev_trials, WINDOW)
        std_arr = zscore_baseline(std_arr, t_centers, BASELINE)
        dev_arr = zscore_baseline(dev_arr, t_centers, BASELINE)

    elif tech == "mesoscope":
        from openscope_pp.loaders.responses import load_responses
        resp_std = load_responses(
            handle, std_trials, WINDOW,
            signal_type="dff", soma_only=True, plane_filter=["VISp_0"],
            baseline_window=BASELINE, baseline_mode="zscore",
        )
        resp_dev = load_responses(
            handle, dev_trials, WINDOW,
            signal_type="dff", soma_only=True, plane_filter=["VISp_0"],
            baseline_window=BASELINE, baseline_mode="zscore",
        )
        std_arr    = resp_std.values
        dev_arr    = resp_dev.values
        t_centers  = resp_std.coords["time_sec"].values

    else:  # slap2
        from openscope_pp.loaders.responses import load_responses
        resp_std = load_responses(
            handle, std_trials, WINDOW,
            dmd_filter=["DMD1"],
            baseline_window=BASELINE, baseline_mode="zscore",
        )
        resp_dev = load_responses(
            handle, dev_trials, WINDOW,
            dmd_filter=["DMD1"],
            baseline_window=BASELINE, baseline_mode="zscore",
        )
        std_arr   = resp_std.values
        dev_arr   = resp_dev.values
        t_centers = resp_std.coords["time_sec"].values

    print(f"  responses loaded in {time.time()-t_resp:.1f}s")
    print(f"  std shape: {std_arr.shape}, dev shape: {dev_arr.shape}")

    # ── Plot ──────────────────────────────────────────────────────
    std_m, std_s = mean_sem(std_arr)
    dev_m, dev_s = mean_sem(dev_arr)

    ax.axvspan(0, 0.25, color="gray", alpha=0.12, label="stim")
    ax.axhline(0, color="gray", lw=0.6, ls="--")
    ax.axvline(0, color="gray", lw=0.8, ls=":")

    ax.fill_between(t_centers, std_m - std_s, std_m + std_s,
                    color=COLORS["standard"], alpha=0.25)
    ax.fill_between(t_centers, dev_m - dev_s, dev_m + dev_s,
                    color=COLORS["deviant"],  alpha=0.25)
    ax.plot(t_centers, std_m, color=COLORS["standard"], lw=2.0, label=f"standard (n={len(std_trials)})")
    ax.plot(t_centers, dev_m, color=COLORS["deviant"],  lw=2.0, label=f"deviant (n={len(dev_trials)})")

    ax.set_title(TECH_LABELS[tech], fontsize=11, fontweight="bold")
    ax.set_xlabel("Time from stimulus onset (s)", fontsize=9)
    ax.set_xlim(WINDOW)
    ax.legend(fontsize=7.5, framealpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)

    if ax is axes[0]:
        ax.set_ylabel("Population response (z-score)", fontsize=9)

    handle.close()
    print(f"  done.")

fig.suptitle("Oddball responses: standard vs deviant (mean ± SEM across units)",
             fontsize=12, fontweight="bold", y=1.01)
fig.tight_layout()

out = "oddball_responses.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved → {out}")
