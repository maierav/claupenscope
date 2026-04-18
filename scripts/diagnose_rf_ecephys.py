"""RF mapping alignment diagnostic — ecephys.

Two panels:
  Left : Population PSTH for all RF trials (alignment check).
         If timing is good, we should see a clear response onset
         at ~30–80 ms in mouse V1.
  Right: RF maps for the N_SHOW units with the strongest responses.
         Each cell shows mean spike rate in the response window at
         that (x, y) position.

Saves: rf_diagnostic_ecephys.png
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
ASSET_ID   = "cd175e65-8faa-4216-86af-c1fd30e571a1"   # sub-820459 SEQUENCE
WINDOW     = (-0.1, 0.5)    # seconds around stimulus onset
BIN_SIZE   = 0.01           # 10 ms bins
RESP_WIN   = (0.03, 0.25)   # window to score "response strength" for RF maps
N_SHOW     = 9              # RF maps to display (perfect square recommended)
RF_SMOOTH  = 0.8            # gaussian smoothing sigma (grid units) for RF maps

# ── Open file ─────────────────────────────────────────────────────────
print("Opening NWB…")
t0 = time.time()
handle = open_nwb(ASSET_ID)
trials  = load_trials(handle)
rf_trials = trials[trials["block_kind"] == "rf_mapping"].copy().reset_index(drop=True)
print(f"  {len(rf_trials)} RF trials, opened in {time.time()-t0:.1f}s")

# ── Fast spike reader (one big read) ─────────────────────────────────
print("Reading spike times…")
h5 = handle.h5
units = h5["units"]

dl = units["decoder_label"][:]
dl = np.array([v.decode() if isinstance(v, bytes) else str(v) for v in dl])
sua_mask = dl == "sua"
sua_idx  = np.where(sua_mask)[0]

all_spikes  = units["spike_times"][:]
spike_index = units["spike_times_index"][:]

pre, post = WINDOW
edges   = np.arange(pre, post + BIN_SIZE, BIN_SIZE)
centers = 0.5 * (edges[:-1] + edges[1:])
n_bins  = len(centers)
n_units = len(sua_idx)
n_trials = len(rf_trials)
t0_arr   = rf_trials["start_time"].values

print(f"  Binning {n_trials} trials × {n_units} SUA units…")
t1 = time.time()
result = np.zeros((n_trials, n_units, n_bins), dtype=np.float32)

for j, uid in enumerate(sua_idx):
    i0 = int(spike_index[uid - 1]) if uid > 0 else 0
    i1 = int(spike_index[uid])
    spikes = all_spikes[i0:i1]
    for i, t0_trial in enumerate(t0_arr):
        rel = spikes - t0_trial
        in_win = rel[(rel >= pre) & (rel < post)]
        if len(in_win):
            counts, _ = np.histogram(in_win, bins=edges)
            result[i, j, :] = counts / BIN_SIZE   # spikes/s

print(f"  Done in {time.time()-t1:.1f}s")

# ── Population PSTH ───────────────────────────────────────────────────
# Mean firing rate across units and trials
pop_rate  = result.mean(axis=(0, 1))       # (time,)
pop_sem   = result.mean(axis=1).std(axis=0) / np.sqrt(n_trials)  # SEM over trials

# ── RF maps ──────────────────────────────────────────────────────────
# Score each unit: mean rate in RESP_WIN minus mean rate in baseline
bl_mask   = centers < 0
resp_mask = (centers >= RESP_WIN[0]) & (centers < RESP_WIN[1])

bl_rate   = result[:, :, bl_mask].mean(axis=2)    # (trial, unit)
resp_rate = result[:, :, resp_mask].mean(axis=2)  # (trial, unit)
delta     = resp_rate - bl_rate                    # (trial, unit) — response above baseline

# x/y grid
xs = np.array(sorted(rf_trials["x"].unique()))
ys = np.array(sorted(rf_trials["y"].unique()))
nx, ny = len(xs), len(ys)

# Build (unit, ny, nx) RF maps
rf_maps = np.zeros((n_units, ny, nx), dtype=np.float32)
for xi, x in enumerate(xs):
    for yi, y in enumerate(ys):
        mask = (rf_trials["x"].values == x) & (rf_trials["y"].values == y)
        rf_maps[:, yi, xi] = delta[mask, :].mean(axis=0)

# Smooth
for u in range(n_units):
    rf_maps[u] = gaussian_filter(rf_maps[u], sigma=RF_SMOOTH)

# Rank units by peak response
peak_resp = rf_maps.max(axis=(1, 2)) - rf_maps.min(axis=(1, 2))   # dynamic range
top_units = np.argsort(peak_resp)[::-1][:N_SHOW]

print(f"  Top {N_SHOW} unit peak ΔFR: {peak_resp[top_units][:5].round(1)} spk/s")

# ── Plot ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 7))
gs  = fig.add_gridspec(1, 2, width_ratios=[1, 2], wspace=0.35)

# — Left: population PSTH —
ax_psth = fig.add_subplot(gs[0])
ax_psth.axvspan(0, 0.25, color="gray", alpha=0.12, label="stim (250 ms)")
ax_psth.axhline(0, color="gray", lw=0.6, ls="--")
ax_psth.axvline(0, color="gray", lw=0.8, ls=":")
ax_psth.fill_between(centers, pop_rate - pop_sem, pop_rate + pop_sem,
                     color="#4878CF", alpha=0.3)
ax_psth.plot(centers, pop_rate, color="#4878CF", lw=2)
ax_psth.set_xlabel("Time from stimulus onset (s)", fontsize=10)
ax_psth.set_ylabel("Population mean firing rate (spk/s)", fontsize=10)
ax_psth.set_title("Population PSTH — RF mapping\n(all SUA units, all positions)", fontsize=10)
ax_psth.set_xlim(WINDOW)
ax_psth.legend(fontsize=8)
ax_psth.spines[["top", "right"]].set_visible(False)

# Annotate response latency (first bin > mean + 2 SD baseline)
bl_mu  = pop_rate[bl_mask].mean()
bl_sd  = pop_rate[bl_mask].std()
thresh = bl_mu + 2 * bl_sd
resp_bins = np.where((centers > 0) & (pop_rate > thresh))[0]
if len(resp_bins):
    latency = centers[resp_bins[0]]
    ax_psth.axvline(latency, color="red", lw=1.2, ls="--", label=f"latency ≈{latency*1000:.0f} ms")
    ax_psth.legend(fontsize=8)

# — Right: RF maps grid —
ncols = int(np.ceil(np.sqrt(N_SHOW)))
nrows = int(np.ceil(N_SHOW / ncols))
gs2   = gs[1].subgridspec(nrows, ncols, hspace=0.4, wspace=0.3)

# Shared colour scale across all shown maps
all_vals = rf_maps[top_units]
vmax = np.percentile(np.abs(all_vals), 99)

for plot_i, uid in enumerate(top_units):
    row, col = divmod(plot_i, ncols)
    ax = fig.add_subplot(gs2[row, col])

    im = ax.imshow(
        rf_maps[uid],
        origin="lower",
        extent=[xs[0] - 5, xs[-1] + 5, ys[0] - 5, ys[-1] + 5],
        vmin=-vmax, vmax=vmax,
        cmap="RdBu_r",
        aspect="equal",
    )
    ax.set_xticks([]); ax.set_yticks([])
    # Unit label
    dev_name = units["device_name"][sua_idx[uid]]
    if isinstance(dev_name, bytes): dev_name = dev_name.decode()
    orig_id  = int(units["original_cluster_id"][sua_idx[uid]])
    ax.set_title(f"{dev_name}#{orig_id}\npeak={peak_resp[uid]:.1f}", fontsize=6.5, pad=2)

# Single colourbar
cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
fig.colorbar(im, cax=cbar_ax, label="ΔFR vs baseline (spk/s)")

fig.suptitle(
    f"Ecephys RF mapping — sub-820459\n"
    f"{n_units} SUA units, {len(rf_trials)} trials, "
    f"response window {RESP_WIN[0]*1000:.0f}–{RESP_WIN[1]*1000:.0f} ms",
    fontsize=11, fontweight="bold",
)

out = "rf_diagnostic_ecephys.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved → {out}")
handle.close()
