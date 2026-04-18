"""Ecephys RF mapping — full analysis figure.

Figure 1 (rf_mapping_summary.png):
  Row 1 — Population PSTH  |  Unit response-strength histogram  |  Yield vs threshold
  Row 2+— RF maps for top N visually-responsive SUA units (one panel per unit)

Figure 2 (oddball_responsive_units.png):
  Oddball PSTH re-run using only visually-responsive units, to fix the flat-line issue.

Console output tracks unit counts and percentages at every filter step.
"""
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter

from openscope_pp.loaders.streaming import open_nwb
from openscope_pp.loaders.trials import load_trials

# ── Config ────────────────────────────────────────────────────────────
ASSET_ID     = "cd175e65-8faa-4216-86af-c1fd30e571a1"
RF_WINDOW    = (-0.1, 0.5)   # PSTH window around RF trial onset
RESP_WIN     = (0.03, 0.25)  # response scoring window
BIN_SIZE     = 0.01          # 10 ms bins
RESP_THRESH  = 5.0           # spk/s above baseline — primary filter
RF_SMOOTH    = 0.8           # gaussian sigma for RF map display
N_MAP_SHOW   = 16            # how many RF maps to display

ODD_WINDOW   = (-0.5, 1.0)   # window for oddball PSTH
ODD_BIN_SIZE = 0.025         # 25 ms bins for oddball
N_ODD_STD    = 300           # max standard trials to use


# ── Helpers ───────────────────────────────────────────────────────────
def decode(arr):
    """Decode byte-string arrays to Python str."""
    if arr.dtype.kind in ("S", "O"):
        return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in arr])
    return arr


def bin_spikes(sua_idx, all_spikes, spike_index, t0_arr, window, bin_size):
    """Return (n_trials, n_sua, n_bins) spike-count array in spk/s."""
    pre, post = window
    edges   = np.arange(pre, post + bin_size, bin_size)
    centers = 0.5 * (edges[:-1] + edges[1:])
    n_bins  = len(centers)
    result  = np.zeros((len(t0_arr), len(sua_idx), n_bins), dtype=np.float32)
    for j, uid in enumerate(sua_idx):
        i0  = int(spike_index[uid - 1]) if uid > 0 else 0
        i1  = int(spike_index[uid])
        spk = all_spikes[i0:i1]
        for i, t0 in enumerate(t0_arr):
            rel = spk - t0
            in_win = rel[(rel >= pre) & (rel < post)]
            if len(in_win):
                cnt, _ = np.histogram(in_win, bins=edges)
                result[i, j, :] = cnt / bin_size
    return result, centers


def mean_sem_units(arr):
    """(trial, unit, time) → per-trial unit-mean, then mean±SEM across trials."""
    per_trial = arr.mean(axis=1)           # (trial, time)
    return per_trial.mean(axis=0), per_trial.std(axis=0) / np.sqrt(per_trial.shape[0])


# ── Load ──────────────────────────────────────────────────────────────
print("=" * 60)
print("ECEPHYS RF MAPPING")
print("=" * 60)

print("\n[1/5] Opening NWB…")
t0 = time.time()
handle  = open_nwb(ASSET_ID)
trials  = load_trials(handle)
rf_t    = trials[trials["block_kind"] == "rf_mapping"].copy().reset_index(drop=True)
odd_t   = trials[trials["block_kind"] == "paradigm_oddball"].copy().reset_index(drop=True)
print(f"  opened in {time.time()-t0:.1f}s")
print(f"  RF trials: {len(rf_t)},  oddball trials: {len(odd_t)}")

h5          = handle.h5
units_grp   = h5["units"]
n_total     = int(units_grp["id"].shape[0])
dl          = decode(units_grp["decoder_label"][:])
sua_mask    = dl == "sua"
sua_idx     = np.where(sua_mask)[0]
n_sua       = len(sua_idx)

all_spikes  = units_grp["spike_times"][:]
spike_index = units_grp["spike_times_index"][:]

print(f"\n[2/5] Unit inventory")
print(f"  Total units:  {n_total}")
print(f"  SUA:          {n_sua:4d}  ({100*n_sua/n_total:.1f}% of all)")


# ── RF spike arrays ───────────────────────────────────────────────────
print("\n[3/5] Binning RF spikes…")
t0 = time.time()
rf_arr, rf_centers = bin_spikes(
    sua_idx, all_spikes, spike_index,
    rf_t["start_time"].values, RF_WINDOW, BIN_SIZE,
)
print(f"  {rf_arr.shape}  in {time.time()-t0:.1f}s")   # (trial, unit, bin)


# ── Per-unit RF scoring ───────────────────────────────────────────────
bl_mask   = rf_centers < 0
resp_mask = (rf_centers >= RESP_WIN[0]) & (rf_centers < RESP_WIN[1])

bl_rate   = rf_arr[:, :, bl_mask].mean(axis=2)    # (trial, unit)
resp_rate = rf_arr[:, :, resp_mask].mean(axis=2)
delta     = resp_rate - bl_rate                    # ΔFR per trial per unit

# Best position for each unit: mean ΔFR at the position it responds most to
xs = np.array(sorted(rf_t["x"].unique()))
ys = np.array(sorted(rf_t["y"].unique()))
nx, ny = len(xs), len(ys)

print("\n[4/5] Building RF maps and scoring units…")
rf_maps  = np.zeros((n_sua, ny, nx), dtype=np.float32)
peak_dfr = np.zeros(n_sua, dtype=np.float32)

for xi, x in enumerate(xs):
    for yi, y in enumerate(ys):
        mask = (rf_t["x"].values == x) & (rf_t["y"].values == y)
        rf_maps[:, yi, xi] = delta[mask, :].mean(axis=0)

for u in range(n_sua):
    smooth = gaussian_filter(rf_maps[u], sigma=RF_SMOOTH)
    peak_dfr[u] = smooth.max()   # peak response above baseline

# ── Unit selection cascade ────────────────────────────────────────────
thresholds  = np.arange(0, 31, 1)
yield_pct   = [(peak_dfr >= th).mean() * 100 for th in thresholds]

n_resp      = (peak_dfr >= RESP_THRESH).sum()
resp_mask_u = peak_dfr >= RESP_THRESH

print(f"\n  Unit selection (threshold = {RESP_THRESH} spk/s):")
print(f"  {'Step':<35} {'N':>5}   {'% of SUA':>9}   {'% of all':>9}")
print(f"  {'-'*60}")
print(f"  {'All recorded':35} {n_total:5d}   {'100.0':>9}   {'100.0':>9}")
sua_label = 'SUA (decoder_label=="sua")'
print(f"  {sua_label:35} {n_sua:5d}   {100*n_sua/n_sua:9.1f}   {100*n_sua/n_total:9.1f}")
print(f"  {f'Visual (peak ΔFR > {RESP_THRESH} spk/s)':35} {n_resp:5d}   {100*n_resp/n_sua:9.1f}   {100*n_resp/n_total:9.1f}")

# Sort responsive units by peak ΔFR
resp_order  = np.argsort(peak_dfr)[::-1]
show_units  = resp_order[:N_MAP_SHOW]


# ── Figure 1: RF summary ──────────────────────────────────────────────
print("\n[5/5] Plotting…")
fig = plt.figure(figsize=(18, 4 + 4 * int(np.ceil(N_MAP_SHOW / 4))), constrained_layout=False)
fig.suptitle(
    f"Ecephys RF mapping — sub-820459\n"
    f"{n_total} units → {n_sua} SUA → {n_resp} visual "
    f"({100*n_resp/n_sua:.0f}% of SUA, {100*n_resp/n_total:.0f}% of all)",
    fontsize=12, fontweight="bold", y=0.98,
)

outer = gridspec.GridSpec(2, 1, figure=fig, hspace=0.45,
                          top=0.93, bottom=0.06, left=0.06, right=0.97)

# ── Row 1: diagnostics ────────────────────────────────────────────────
top_gs = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0], wspace=0.38)

# 1a — Population PSTH
ax_psth = fig.add_subplot(top_gs[0])
pop_m, pop_s = mean_sem_units(rf_arr)
ax_psth.axvspan(0, 0.25, color="gray", alpha=0.12)
ax_psth.axhline(0, color="gray", lw=0.5, ls="--")
ax_psth.axvline(0, color="gray", lw=0.8, ls=":")
ax_psth.fill_between(rf_centers, pop_m - pop_s, pop_m + pop_s, color="#4878CF", alpha=0.3)
ax_psth.plot(rf_centers, pop_m, color="#4878CF", lw=2)
bl_mu = pop_m[bl_mask].mean(); bl_sd = pop_m[bl_mask].std()
first_resp = np.where((rf_centers > 0) & (pop_m > bl_mu + 2 * bl_sd))[0]
if len(first_resp):
    lat = rf_centers[first_resp[0]]
    ax_psth.axvline(lat, color="red", lw=1.2, ls="--")
    ax_psth.text(lat + 0.005, pop_m.max() * 0.98, f"latency\n≈{lat*1000:.0f} ms",
                 color="red", fontsize=7, va="top")
ax_psth.set(xlabel="Time from onset (s)", ylabel="Mean FR (spk/s)",
            title="Population PSTH\n(all SUA, all RF positions)", xlim=RF_WINDOW)
ax_psth.spines[["top", "right"]].set_visible(False)

# 1b — Distribution of peak ΔFR
ax_hist = fig.add_subplot(top_gs[1])
ax_hist.hist(peak_dfr, bins=60, color="#888", edgecolor="none", alpha=0.8)
ax_hist.axvline(RESP_THRESH, color="red", lw=1.5, ls="--",
                label=f"threshold = {RESP_THRESH} spk/s")
ax_hist.text(RESP_THRESH + 0.5, ax_hist.get_ylim()[1] * 0.85,
             f"{n_resp} units\n({100*n_resp/n_sua:.0f}% SUA)",
             color="red", fontsize=8)
ax_hist.set(xlabel="Peak ΔFR above baseline (spk/s)",
            ylabel="Number of SUA units",
            title="Response-strength distribution\n(best RF position, smoothed)")
ax_hist.legend(fontsize=8)
ax_hist.spines[["top", "right"]].set_visible(False)

# 1c — Yield vs threshold
ax_yield = fig.add_subplot(top_gs[2])
ax_yield.plot(thresholds, yield_pct, color="#4878CF", lw=2)
ax_yield.axvline(RESP_THRESH, color="red", lw=1.5, ls="--")
ax_yield.axhline(100 * n_resp / n_sua, color="red", lw=0.8, ls=":")
ax_yield.scatter([RESP_THRESH], [100 * n_resp / n_sua],
                 color="red", zorder=5, s=40)
ax_yield.set(xlabel="Response threshold (spk/s)",
             ylabel="% of SUA units retained",
             title="Yield vs threshold\n(how strict is the filter?)",
             ylim=(0, 105), xlim=(0, 30))
ax_yield.spines[["top", "right"]].set_visible(False)

# ── Row 2+: RF maps ───────────────────────────────────────────────────
n_cols = 4
n_rows = int(np.ceil(N_MAP_SHOW / n_cols))
bot_gs = gridspec.GridSpecFromSubplotSpec(
    n_rows, n_cols, subplot_spec=outer[1],
    hspace=0.55, wspace=0.3,
)

all_map_vals = np.array([gaussian_filter(rf_maps[u], RF_SMOOTH) for u in show_units])
vmax = np.percentile(np.abs(all_map_vals), 99)

dev_names = decode(units_grp["device_name"][:])
orig_ids  = units_grp["original_cluster_id"][:]

for plot_i, uid in enumerate(show_units):
    row, col = divmod(plot_i, n_cols)
    ax = fig.add_subplot(bot_gs[row, col])
    smooth = gaussian_filter(rf_maps[uid], RF_SMOOTH)
    im = ax.imshow(
        smooth,
        origin="lower",
        extent=[xs[0] - 5, xs[-1] + 5, ys[0] - 5, ys[-1] + 5],
        vmin=-vmax, vmax=vmax,
        cmap="RdBu_r", aspect="equal",
    )
    # Mark peak
    peak_pos = np.unravel_index(np.argmax(smooth), smooth.shape)
    ax.plot(xs[peak_pos[1]], ys[peak_pos[0]], "k+", ms=6, mew=1.2)
    ax.set_xticks(xs[::2]); ax.set_yticks(ys[::2])
    ax.tick_params(labelsize=5)
    rank = plot_i + 1
    probe = dev_names[sua_idx[uid]]
    oid   = int(orig_ids[sua_idx[uid]])
    ax.set_title(
        f"#{rank}  {probe}·{oid}\npeak={peak_dfr[uid]:.1f} spk/s",
        fontsize=6.5, pad=2,
    )
    if col == 0: ax.set_ylabel("Y (°)", fontsize=6)
    if row == n_rows - 1: ax.set_xlabel("X (°)", fontsize=6)

# Shared colorbar for RF maps
cbar_ax = fig.add_axes([0.975, 0.06, 0.012, 0.30])
fig.colorbar(im, cax=cbar_ax, label="ΔFR (spk/s)")

out1 = "rf_mapping_summary.png"
fig.savefig(out1, dpi=150, bbox_inches="tight")
print(f"  Saved → {out1}")


# ── Figure 2: Oddball PSTH — responsive units only ───────────────────
print("\n  Replotting oddball responses with visual units only…")

first_block = odd_t["block"].unique()[0]
ob          = odd_t[odd_t["block"] == first_block]
std_t = ob[ob["trial_type"] == "standard"].sample(
            min(N_ODD_STD, ob["trial_type"].eq("standard").sum()),
            random_state=42)
dev_t = ob[ob["is_deviant"]]

resp_sua_idx = sua_idx[resp_mask_u]
print(f"  Oddball: {len(std_t)} std, {len(dev_t)} deviant — {len(resp_sua_idx)} visual units")

t1 = time.time()
std_arr, odd_centers = bin_spikes(
    resp_sua_idx, all_spikes, spike_index,
    std_t["start_time"].values, ODD_WINDOW, ODD_BIN_SIZE,
)
dev_arr, _ = bin_spikes(
    resp_sua_idx, all_spikes, spike_index,
    dev_t["start_time"].values, ODD_WINDOW, ODD_BIN_SIZE,
)
print(f"  Binned in {time.time()-t1:.1f}s")

# Z-score each unit to its own baseline across all trials
def zscore_to_baseline(arr, centers, baseline=(-0.5, 0.0)):
    bl = (centers >= baseline[0]) & (centers < baseline[1])
    # pool both std and dev trials for baseline stats
    mu  = arr[:, :, bl].mean(axis=(0, 2), keepdims=True)   # (1, unit, 1)
    sig = arr[:, :, bl].std(axis=(0, 2), keepdims=True)
    sig = np.where(sig > 0.1, sig, 1.0)
    return (arr - mu) / sig

std_z = zscore_to_baseline(std_arr, odd_centers)
dev_z = zscore_to_baseline(dev_arr, odd_centers)

std_m, std_s = mean_sem_units(std_z)
dev_m, dev_s = mean_sem_units(dev_z)

fig2, ax = plt.subplots(figsize=(7, 4))
ax.axvspan(0, 0.25, color="gray", alpha=0.12, label="stim (250 ms)")
ax.axhline(0, color="gray", lw=0.6, ls="--")
ax.axvline(0, color="gray", lw=0.8, ls=":")
ax.fill_between(odd_centers, std_m - std_s, std_m + std_s, color="#4878CF", alpha=0.25)
ax.fill_between(odd_centers, dev_m - dev_s, dev_m + dev_s, color="#D65F5F",  alpha=0.25)
ax.plot(odd_centers, std_m, color="#4878CF", lw=2,
        label=f"standard  (n={len(std_t)} trials)")
ax.plot(odd_centers, dev_m, color="#D65F5F",  lw=2,
        label=f"deviant   (n={len(dev_t)} trials, all types pooled)")
ax.set(xlabel="Time from stimulus onset (s)",
       ylabel="Population response (z-score)",
       title=(f"Oddball PSTH — ecephys sub-820459  [SEQUENCE block]\n"
              f"Visual SUA only: {n_resp}/{n_sua} units "
              f"({100*n_resp/n_sua:.0f}% of SUA, {100*n_resp/n_total:.0f}% of all)"),
       xlim=ODD_WINDOW)
ax.legend(fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
fig2.tight_layout()

out2 = "oddball_responsive_units.png"
fig2.savefig(out2, dpi=150, bbox_inches="tight")
print(f"  Saved → {out2}")

handle.close()
print("\nDone.")
