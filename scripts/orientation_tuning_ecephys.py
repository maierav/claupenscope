"""Ecephys orientation tuning — all quality-passing SUA, all 5 probes.

Uses sequential_control_block (14 directions × 80 trials each, 0.267s stimuli).
Unit pool: SUA + default_qc (~820 units across 5 probes).
No RF pre-filter — orientation selectivity is assessed directly from tuning curves.

Saves:
  ori_tuning_ecephys_polar.png  — polar curves for top 36 SUA by OSI
  ori_tuning_ecephys_stats.png  — OSI/DSI distributions + preferred orientation wheel
  ori_tuning_ecephys_probes.png — per-probe OSI/DSI breakdown
"""
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from openscope_pp.loaders.streaming import open_nwb
from openscope_pp.loaders.trials import load_trials

# ── Config ────────────────────────────────────────────────────────────
ASSET_ID     = "cd175e65-8faa-4216-86af-c1fd30e571a1"
ORI_WINDOW   = (-0.1, 0.5)
ORI_BIN      = 0.01
RESP_WIN     = (0.03, 0.25)
BL_WIN       = (-0.1, 0.0)
OSI_THRESH   = 0.2           # minimum OSI to count as "tuned"
N_POLAR      = 36            # top ROIs to show in polar grid

# ── Helpers ───────────────────────────────────────────────────────────
def decode(arr):
    if arr.dtype.kind in ("S", "O"):
        return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in arr])
    return arr

def bin_spikes(uid_arr, spikes, index, onsets, window, bin_size):
    """Fast spike binning using searchsorted — O(n_units × n_trials × log n_spikes)."""
    pre, post = window
    edges   = np.arange(pre, post + bin_size, bin_size)
    centers = 0.5 * (edges[:-1] + edges[1:])
    n_bins  = len(centers)
    out = np.zeros((len(onsets), len(uid_arr), n_bins), dtype=np.float32)
    for j, uid in enumerate(uid_arr):
        i0  = int(index[uid - 1]) if uid > 0 else 0
        i1  = int(index[uid])
        spk = spikes[i0:i1]
        if len(spk) == 0:
            continue
        for i, t0 in enumerate(onsets):
            lo = int(np.searchsorted(spk, t0 + pre))
            hi = int(np.searchsorted(spk, t0 + post))
            if lo < hi:
                local = spk[lo:hi] - t0
                cnt, _ = np.histogram(local, bins=edges)
                out[i, j, :] = cnt / bin_size
        if (j + 1) % 100 == 0:
            print(f"    {j+1}/{len(uid_arr)} units done…")
    return out, centers

# ── Load ──────────────────────────────────────────────────────────────
print("Opening NWB…")
t0 = time.time()
handle = open_nwb(ASSET_ID)
trials = load_trials(handle)
h5     = handle.h5
print(f"  Opened in {time.time()-t0:.1f}s")

units_grp  = h5["units"]
n_total    = int(units_grp["id"].shape[0])
dl         = decode(units_grp["decoder_label"][:])
qc         = units_grp["default_qc"][:].astype(bool)
dev        = decode(units_grp["device_name"][:])
all_spikes = units_grp["spike_times"][:]
spk_index  = units_grp["spike_times_index"][:]

# Quality gate: SUA + default_qc
qual_mask = (dl == "sua") & qc
qual_idx  = np.where(qual_mask)[0]
probes    = dev[qual_mask]
n_qual    = len(qual_idx)
print(f"  {n_total} total units → {n_qual} SUA + default_qc across 5 probes")
from collections import Counter
for p, cnt in sorted(Counter(probes).items()):
    print(f"    {p}: {cnt}")

# ── Orientation tuning block ──────────────────────────────────────────
ori_t    = trials[trials["block_kind"] == "sequential_control_block"].reset_index(drop=True)
dirs_rad = np.array(sorted(ori_t["orientation"].unique()))
dirs_deg = np.degrees(dirs_rad)
n_dirs   = len(dirs_rad)
oris_v   = ori_t["orientation"].values
print(f"\nOri block: {len(ori_t)} trials, {n_dirs} directions: {dirs_deg.round(1)}")

# ── Bin spikes ────────────────────────────────────────────────────────
print(f"\nBinning {n_qual} units × {len(ori_t)} trials…")
t1 = time.time()
ori_arr, centers = bin_spikes(qual_idx, all_spikes, spk_index,
                               ori_t["start_time"].values, ORI_WINDOW, ORI_BIN)
print(f"  Done in {time.time()-t1:.1f}s  shape: {ori_arr.shape}")

# ── Z-score ────────────────────────────────────────────────────────────
bl_m  = (centers >= BL_WIN[0])   & (centers < BL_WIN[1])
rsp_m = (centers >= RESP_WIN[0]) & (centers < RESP_WIN[1])
mu_z  = ori_arr[:, :, bl_m].mean(axis=(0, 2), keepdims=True)
sig_z = ori_arr[:, :, bl_m].std(axis=(0, 2), keepdims=True)
sig_z = np.where(sig_z > 0.1, sig_z, 1.0)
z     = (ori_arr - mu_z) / sig_z

# ── Tuning curves + OSI/DSI ────────────────────────────────────────────
curves = np.full((n_qual, n_dirs), np.nan, dtype=np.float32)
for di, d in enumerate(dirs_rad):
    m = oris_v == d
    resp = np.nanmean(z[m][:, :, rsp_m], axis=(0, 2))
    base = np.nanmean(z[m][:, :, bl_m ], axis=(0, 2))
    curves[:, di] = resp - base

angles = dirs_rad
r_sum  = np.nansum(np.abs(curves), axis=1).clip(1e-9)
r_ori  = np.nansum(curves * np.exp(2j * angles)[None, :], axis=1)
r_dir  = np.nansum(curves * np.exp(1j * angles)[None, :], axis=1)
OSI    = (np.abs(r_ori) / r_sum).real
DSI    = (np.abs(r_dir) / r_sum).real
pref_ori = (np.angle(r_ori) / 2) % np.pi
pref_dir = np.angle(r_dir) % (2 * np.pi)

n_tuned = (OSI >= OSI_THRESH).sum()
print(f"\nAll {n_qual} SUA:  OSI median={np.median(OSI):.3f}  DSI median={np.median(DSI):.3f}")
print(f"OSI >= {OSI_THRESH}: {n_tuned} units ({100*n_tuned/n_qual:.0f}%)")

# ── Figure 1: Polar curves (top 36 by OSI) ────────────────────────────
n_show = min(N_POLAR, n_qual)
order  = np.argsort(OSI)[::-1][:n_show]
n_cols, n_rows = 6, int(np.ceil(n_show / 6))
fig, axes = plt.subplots(n_rows, n_cols,
                          figsize=(n_cols * 2.0, n_rows * 2.0),
                          subplot_kw={"projection": "polar"}, squeeze=False)
theta = np.append(dirs_rad, dirs_rad[0])
cmap  = plt.cm.hsv

for plot_i in range(n_rows * n_cols):
    row, col = divmod(plot_i, n_cols)
    ax = axes[row, col]
    if plot_i >= n_show:
        ax.set_visible(False); continue
    uid = order[plot_i]
    c   = np.clip(np.append(curves[uid], curves[uid][0]), 0, None)
    color = cmap(pref_ori[uid] / np.pi)
    ax.plot(theta, c, lw=1.5, color=color)
    ax.fill(theta, c, alpha=0.25, color=color)
    ax.set(xticks=[], yticks=[])
    ax.set_title(f"{probes[uid]}\nOSI={OSI[uid]:.2f}", fontsize=6, pad=2)

fig.suptitle(
    f"Ecephys orientation tuning — top {n_show} SUA by OSI\n"
    f"{n_qual} SUA+QC total · {n_dirs} directions · sequential_control_block",
    fontsize=9, fontweight="bold"
)
fig.tight_layout()
fig.savefig("ori_tuning_ecephys_polar.png", dpi=150, bbox_inches="tight")
print("\nSaved → ori_tuning_ecephys_polar.png")

# ── Figure 2: OSI/DSI distributions + preferred orientation wheel ──────
fig = plt.figure(figsize=(14, 4))

ax1 = fig.add_subplot(1, 3, 1)
bins = np.linspace(0, 1, 16)
ax1.hist(OSI, bins=bins, color="#4878CF", edgecolor="white", lw=0.4)
ax1.axvline(np.median(OSI), color="k", lw=1.5, ls="--",
            label=f"median={np.median(OSI):.2f}")
ax1.axvline(OSI_THRESH, color="red", lw=1, ls=":", alpha=0.7,
            label=f"tuned thresh={OSI_THRESH}")
ax1.set(xlabel="OSI", ylabel="Unit count",
        title=f"Orientation Selectivity\n{n_tuned}/{n_qual} tuned (OSI≥{OSI_THRESH})")
ax1.legend(fontsize=7); ax1.spines[["top","right"]].set_visible(False)

ax2 = fig.add_subplot(1, 3, 2)
ax2.hist(DSI, bins=bins, color="#D65F5F", edgecolor="white", lw=0.4)
ax2.axvline(np.median(DSI), color="k", lw=1.5, ls="--",
            label=f"median={np.median(DSI):.2f}")
ax2.set(xlabel="DSI", ylabel="Unit count", title="Direction Selectivity")
ax2.legend(fontsize=7); ax2.spines[["top","right"]].set_visible(False)

ax3 = fig.add_subplot(1, 3, 3, projection="polar")
pori_deg = np.degrees(pref_ori[OSI >= OSI_THRESH])
bin_edges = np.linspace(0, 180, 13)
counts, _  = np.histogram(pori_deg, bins=bin_edges)
bc = np.radians(0.5 * (bin_edges[:-1] + bin_edges[1:]))
theta_full  = np.concatenate([bc, bc + np.pi])
counts_full = np.tile(counts, 2)
ax3.bar(theta_full, counts_full, width=np.pi/6,
        color=plt.cm.hsv(theta_full / (2*np.pi)), alpha=0.85)
ax3.set_title(f"Preferred orientation\n(OSI≥{OSI_THRESH}, n={n_tuned})", fontsize=8)
ax3.set_xticklabels(["0°","45°","90°","135°","180°","225°","270°","315°"], fontsize=6)

fig.suptitle(f"Ecephys orientation tuning — {n_qual} SUA+QC", fontsize=10, fontweight="bold")
fig.tight_layout()
fig.savefig("ori_tuning_ecephys_stats.png", dpi=150, bbox_inches="tight")
print("Saved → ori_tuning_ecephys_stats.png")

# ── Figure 3: Per-probe breakdown ─────────────────────────────────────
probe_names = sorted(set(probes))
fig, axes = plt.subplots(1, len(probe_names), figsize=(3.5 * len(probe_names), 4),
                          sharey=True)
bins = np.linspace(0, 1, 14)
for ax, pname in zip(axes, probe_names):
    pm = probes == pname
    ax.hist(OSI[pm], bins=bins, color="#4878CF", alpha=0.7, label="OSI")
    ax.hist(DSI[pm], bins=bins, color="#D65F5F", alpha=0.5, label="DSI")
    ax.axvline(np.median(OSI[pm]), color="#4878CF", lw=1.5, ls="--")
    ax.axvline(np.median(DSI[pm]), color="#D65F5F", lw=1.5, ls="--")
    n_p = pm.sum(); n_t = (OSI[pm] >= OSI_THRESH).sum()
    ax.set(title=f"{pname}  (n={n_p})\nOSI={np.median(OSI[pm]):.2f}  "
                 f"{n_t} tuned",
           xlabel="Index", xlim=(0,1))
    if ax is axes[0]:
        ax.set_ylabel("Unit count")
        ax.legend(fontsize=7)
    ax.spines[["top","right"]].set_visible(False)

fig.suptitle("Ecephys orientation tuning — per-probe breakdown", fontsize=10, fontweight="bold")
fig.tight_layout()
fig.savefig("ori_tuning_ecephys_probes.png", dpi=150, bbox_inches="tight")
print("Saved → ori_tuning_ecephys_probes.png")

handle.close()
print("\nDone.")
