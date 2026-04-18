"""SLAP2 orientation tuning — DMD1 + DMD2 polar curves + OSI/DSI.

Uses sub-803496 ori_tuning block (15 directions × ~134 trials each).
Per-DMD timing offsets applied: DMD1 +115 ms, DMD2 −165 ms.

Saves:
  ori_tuning_slap2_polar.png  — polar tuning curves for every ROI
  ori_tuning_slap2_stats.png  — OSI / DSI distributions + scatter
"""
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from openscope_pp.loaders.streaming import open_nwb
from openscope_pp.loaders.trials import load_trials

# ── Config ────────────────────────────────────────────────────────────
ASSET_ID  = "d23a03af-c3bd-4cf0-9492-6dca96fb201d"   # sub-803496
WINDOW    = (-0.3, 0.8)      # s around corrected onset
RESP_WIN  = (0.05, 0.35)     # response window post-onset
BL_WIN    = (-0.25, 0.0)     # baseline window
OFFSETS   = {"DMD1": +0.115, "DMD2": -0.165}

# ── Load ──────────────────────────────────────────────────────────────
print("Opening NWB (sub-803496)…")
t0 = time.time()
handle = open_nwb(ASSET_ID)
trials = load_trials(handle)
h5     = handle.h5
print(f"  Opened in {time.time()-t0:.1f}s")

ori_t = trials[trials["block_kind"] == "ori_tuning"].reset_index(drop=True)
dirs_rad = np.array(sorted(ori_t["orientation"].unique()))
dirs_deg = np.degrees(dirs_rad)
n_dirs   = len(dirs_rad)
print(f"\nOri tuning: {len(ori_t)} trials, {n_dirs} directions")
print(f"  Directions (deg): {dirs_deg.round(1)}")

# ── Extract dFF snippets per DMD ──────────────────────────────────────
def extract(dmd, ori_trials, offset_sec, window):
    path = f"processing/ophys/Fluorescence_{dmd}/{dmd}_dFF"
    ts   = h5[f"{path}/timestamps"][:]
    data = h5[f"{path}/data"]
    n_rois = data.shape[1]

    mid = len(ts) // 2
    dt  = float(np.median(np.diff(ts[mid:mid + 2000])))
    pre, post = window
    n_samp = int(round((post - pre) / dt))
    t_rel  = np.linspace(pre, pre + (n_samp - 1) * dt, n_samp)
    tc     = t_rel + dt / 2

    onsets = ori_trials["start_time"].values + offset_sec
    valid  = (onsets + pre >= ts[0]) & (onsets + post <= ts[-1])
    onsets = onsets[valid]
    oris_v = ori_trials["orientation"].values[valid]

    # Bulk read
    i0 = max(0, int(np.searchsorted(ts, onsets.min() + pre - 1.0)))
    i1 = min(data.shape[0], int(np.searchsorted(ts, onsets.max() + post + 1.0)) + 1)
    trace   = data[i0:i1, :].astype(np.float32)
    ts_span = ts[i0:i1]

    t_query = onsets[:, None] + t_rel[None, :]
    t_flat  = t_query.ravel()
    snip    = np.full((len(onsets), n_rois, n_samp), np.nan, dtype=np.float32)
    for roi in range(n_rois):
        y = trace[:, roi]; fin = np.isfinite(y)
        if fin.sum() < 10: continue
        vals = np.interp(t_flat, ts_span[fin], y[fin], left=np.nan, right=np.nan)
        snip[:, roi, :] = vals.reshape(len(onsets), n_samp)

    print(f"  {dmd}: {n_rois} ROIs, {valid.sum()}/{len(ori_trials)} trials, {1/dt:.0f} Hz")
    return snip, tc, oris_v, n_rois


print("\nExtracting snippets…")
snip1, tc1, oris1, n1 = extract("DMD1", ori_t, OFFSETS["DMD1"], WINDOW)
snip2, tc2, oris2, n2 = extract("DMD2", ori_t, OFFSETS["DMD2"], WINDOW)

# ── Z-score ────────────────────────────────────────────────────────────
def zscore(snip, tc, bl=BL_WIN):
    bl_mask = (tc >= bl[0]) & (tc < bl[1])
    mu  = np.nanmean(snip[:, :, bl_mask], axis=(0, 2), keepdims=True)
    sig = np.nanstd( snip[:, :, bl_mask], axis=(0, 2), keepdims=True)
    sig = np.where(sig > 1e-6, sig, 1.0)
    return (snip - mu) / sig

z1 = zscore(snip1, tc1)
z2 = zscore(snip2, tc2)

# ── Tuning curves + OSI/DSI ────────────────────────────────────────────
def tuning_stats(z, tc, oris_v, dirs_rad):
    """Return curves (n_rois, n_dirs), OSI, DSI, pref_dir (rad), pref_ori (rad)."""
    resp_mask = (tc >= RESP_WIN[0]) & (tc < RESP_WIN[1])
    bl_mask   = (tc >= BL_WIN[0])  & (tc < BL_WIN[1])
    n_rois = z.shape[1]
    n_dirs = len(dirs_rad)

    curves = np.full((n_rois, n_dirs), np.nan, dtype=np.float32)
    for di, d in enumerate(dirs_rad):
        m = oris_v == d
        if m.sum() == 0: continue
        resp = np.nanmean(z[m][:, :, resp_mask], axis=(0, 2))
        base = np.nanmean(z[m][:, :, bl_mask  ], axis=(0, 2))
        curves[:, di] = resp - base

    # Vector sum selectivity indices
    angles = dirs_rad  # already in radians
    # OSI: use 2θ (treats opposite dirs as same orientation)
    r_ori  = np.nansum(curves * np.exp(2j * angles)[None, :], axis=1)
    r_sum  = np.nansum(np.abs(curves), axis=1).clip(1e-9)
    OSI    = np.abs(r_ori) / r_sum
    pref_ori = (np.angle(r_ori) / 2) % np.pi   # 0–π

    # DSI: use 1θ (direction)
    r_dir  = np.nansum(curves * np.exp(1j * angles)[None, :], axis=1)
    DSI    = np.abs(r_dir) / r_sum
    pref_dir = np.angle(r_dir) % (2 * np.pi)   # 0–2π

    return curves, OSI.real, DSI.real, pref_dir, pref_ori

curves1, OSI1, DSI1, pdir1, pori1 = tuning_stats(z1, tc1, oris1, dirs_rad)
curves2, OSI2, DSI2, pdir2, pori2 = tuning_stats(z2, tc2, oris2, dirs_rad)

print(f"\nDMD1  OSI: {OSI1.mean():.3f} ± {OSI1.std():.3f}  "
      f"DSI: {DSI1.mean():.3f} ± {DSI1.std():.3f}")
print(f"DMD2  OSI: {OSI2.mean():.3f} ± {OSI2.std():.3f}  "
      f"DSI: {DSI2.mean():.3f} ± {DSI2.std():.3f}")

# ── Figure 1: Polar tuning curves ────────────────────────────────────
def polar_grid(curves, OSI, pref_ori, n_rois, title, filename):
    """One polar subplot per ROI, ordered by OSI descending."""
    n_cols = min(6, n_rois)
    n_rows = int(np.ceil(n_rois / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(n_cols * 2.0, n_rows * 2.0),
                              subplot_kw={"projection": "polar"},
                              squeeze=False)

    # Close the polar curve by appending first point
    theta = np.append(dirs_rad, dirs_rad[0])
    order = np.argsort(OSI)[::-1]

    cmap = plt.cm.hsv
    for plot_i in range(n_rows * n_cols):
        row, col = divmod(plot_i, n_cols)
        ax = axes[row, col]
        if plot_i >= n_rois:
            ax.set_visible(False); continue
        uid = order[plot_i]
        c   = np.append(curves[uid], curves[uid][0])
        c   = np.clip(c, 0, None)       # polar plots want non-negative
        color = cmap(pref_ori[uid] / np.pi)
        ax.plot(theta, c, lw=1.5, color=color)
        ax.fill(theta, c, alpha=0.25, color=color)
        ax.set(xticks=[], yticks=[])
        ax.set_title(f"ROI {uid}\nOSI={OSI[uid]:.2f}", fontsize=6, pad=2)

    fig.suptitle(title, fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Saved → {filename}")

polar_grid(curves1, OSI1, pori1, n1,
           f"SLAP2 DMD1 — orientation tuning  ({n1} ROIs, sub-803496)",
           "ori_tuning_slap2_dmd1_polar.png")
polar_grid(curves2, OSI2, pori2, n2,
           f"SLAP2 DMD2 — orientation tuning  ({n2} ROIs, sub-803496)",
           "ori_tuning_slap2_dmd2_polar.png")

# ── Figure 2: OSI / DSI distributions ────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4))

# Panel A: OSI histogram
ax = axes[0]
bins = np.linspace(0, 1, 16)
ax.hist(OSI1, bins=bins, color="#4878CF", alpha=0.7, label=f"DMD1 (n={n1})")
ax.hist(OSI2, bins=bins, color="#D65F5F", alpha=0.7, label=f"DMD2 (n={n2})")
ax.axvline(np.median(OSI1), color="#4878CF", lw=1.5, ls="--")
ax.axvline(np.median(OSI2), color="#D65F5F", lw=1.5, ls="--")
ax.set(xlabel="OSI", ylabel="ROI count", title="Orientation Selectivity Index")
ax.legend(fontsize=8)

# Panel B: DSI histogram
ax = axes[1]
ax.hist(DSI1, bins=bins, color="#4878CF", alpha=0.7, label=f"DMD1")
ax.hist(DSI2, bins=bins, color="#D65F5F", alpha=0.7, label=f"DMD2")
ax.axvline(np.median(DSI1), color="#4878CF", lw=1.5, ls="--")
ax.axvline(np.median(DSI2), color="#D65F5F", lw=1.5, ls="--")
ax.set(xlabel="DSI", ylabel="ROI count", title="Direction Selectivity Index")
ax.legend(fontsize=8)

# Panel C: OSI vs DSI scatter
ax = axes[2]
ax.scatter(OSI1, DSI1, c="#4878CF", s=40, alpha=0.8, label="DMD1", edgecolors="none")
ax.scatter(OSI2, DSI2, c="#D65F5F", s=40, alpha=0.8, label="DMD2", edgecolors="none")
ax.set(xlabel="OSI", ylabel="DSI", title="OSI vs DSI", xlim=(0, 1), ylim=(0, 1))
ax.plot([0, 1], [0, 1], "k--", lw=0.7, alpha=0.4)
ax.legend(fontsize=8)

for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("SLAP2 orientation tuning — sub-803496", fontsize=11, fontweight="bold")
fig.tight_layout()
fig.savefig("ori_tuning_slap2_stats.png", dpi=150, bbox_inches="tight")
print("Saved → ori_tuning_slap2_stats.png")

handle.close()
print("\nDone.")
