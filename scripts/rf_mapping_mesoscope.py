"""Mesoscope RF mapping — alignment PSTH + per-ROI RF maps.

Uses dandiset 001768 (sub-839909, smallest session).
9×9 spatial grid, 15 trials/position, 8 imaging planes (VISp & VISl).
Soma-only ROIs via is_soma filter.

Saves:
  rf_mapping_meso_psth.png   — population PSTH per plane (alignment check)
  rf_mapping_meso_maps.png   — top-ROI RF maps for VISp_0 (most ROIs)
  rf_mapping_meso_allplanes.png — ROI-averaged RF map per plane
"""
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from dandi.dandiapi import DandiAPIClient
import remfile
import h5py

# ── Config ────────────────────────────────────────────────────────────
DANDISET_ID = "001768"
WINDOW      = (-0.5, 1.5)    # seconds around stimulus onset (slow ~10 Hz imaging)
RESP_WIN    = (0.1, 0.8)     # response window post-onset (GCaMP dynamics)
RF_SMOOTH   = 0.8            # gaussian sigma (grid units) for display
N_TOP_ROIS  = 36             # individual ROI maps to show for best plane
PLANES      = ["VISp_0", "VISp_1", "VISp_2", "VISp_3",
               "VISl_4", "VISl_5", "VISl_6", "VISl_7"]

# ── Open file ─────────────────────────────────────────────────────────
print(f"Connecting to dandiset {DANDISET_ID}…")
t0 = time.time()
client  = DandiAPIClient()
ds      = client.get_dandiset(DANDISET_ID)
assets  = sorted(list(ds.get_assets()), key=lambda a: a.size)
asset   = assets[0]
print(f"  Using: {asset.path}  ({asset.size/1e9:.2f} GB)")
url = asset.get_content_url(follow_redirects=1, strip_query=True)
f   = h5py.File(remfile.File(url), "r")
print(f"  Opened in {time.time()-t0:.1f}s")

# ── Load RF mapping trials ─────────────────────────────────────────────
def decode_col(arr):
    """Decode byte-string columns to float array."""
    return np.array([float(v.decode() if isinstance(v, bytes) else v) for v in arr])

rf_grp = f["intervals"]["RF mapping_presentations"]
starts  = rf_grp["start_time"][:]
xs_raw  = decode_col(rf_grp["X"][:])
ys_raw  = decode_col(rf_grp["Y"][:])
xs_grid = np.array(sorted(np.unique(xs_raw)))
ys_grid = np.array(sorted(np.unique(ys_raw)))

print(f"\nRF mapping: {len(starts)} trials, {len(xs_grid)}×{len(ys_grid)} grid, "
      f"{len(starts)/(len(xs_grid)*len(ys_grid)):.0f} trials/position")
print(f"  X: {xs_grid}")
print(f"  Y: {ys_grid}")
print(f"  Time: {starts[0]:.0f}–{starts[-1]:.0f}s")

# ── Helpers ───────────────────────────────────────────────────────────
def load_plane_dff(plane):
    """Load dFF data, timestamps, and is_soma mask for one plane."""
    base = f"processing/{plane}"
    dff_path = f"{base}/dff_timeseries/dff_timeseries"
    ts   = f[f"{dff_path}/timestamps"][:]
    data = f[f"{dff_path}/data"]          # lazy HDF5 dataset
    is_soma = f[f"{base}/image_segmentation/roi_table/is_soma"][:]
    return data, ts, is_soma.astype(bool)


def extract_snippets(data, ts, is_soma, onsets, window):
    """
    Extract trial × soma-ROI × time snippets via linear interpolation.
    Reads the RF-block span as one bulk HDF5 read, then interpolates locally.
    """
    pre, post = window
    dt = float(np.median(np.diff(ts[len(ts)//2: len(ts)//2 + 500])))
    n_samples = int(round((post - pre) / dt))
    t_rel     = np.linspace(pre, pre + (n_samples - 1) * dt, n_samples)
    t_centers = t_rel + dt / 2

    # Bulk read: span covering all RF trials
    t0_span = onsets.min() + pre - 2.0
    t1_span = onsets.max() + post + 2.0
    i0 = max(0, int(np.searchsorted(ts, t0_span)))
    i1 = min(data.shape[0], int(np.searchsorted(ts, t1_span)) + 1)
    trace   = data[i0:i1, :].astype(np.float32)   # (span, all_rois)
    ts_span = ts[i0:i1]

    # Keep soma ROIs only
    trace_soma = trace[:, is_soma]
    n_trials = len(onsets)
    n_rois   = trace_soma.shape[1]

    # Pre-compute flat query array: (n_trials, n_samples) → ravel
    t_query = onsets[:, None] + t_rel[None, :]   # (n_trials, n_samples)
    t_flat  = t_query.ravel()

    out = np.full((n_trials, n_rois, n_samples), np.nan, dtype=np.float32)
    for roi in range(n_rois):
        y = trace_soma[:, roi]
        fin = np.isfinite(y)
        if fin.sum() < 10:
            continue
        vals = np.interp(t_flat, ts_span[fin], y[fin], left=np.nan, right=np.nan)
        out[:, roi, :] = vals.reshape(n_trials, n_samples)

    return out, t_centers, dt


def zscore(snip, centers, bl=(-0.5, 0.0)):
    bl_mask = (centers >= bl[0]) & (centers < bl[1])
    mu  = np.nanmean(snip[:, :, bl_mask], axis=(0, 2), keepdims=True)
    sig = np.nanstd( snip[:, :, bl_mask], axis=(0, 2), keepdims=True)
    sig = np.where(sig > 1e-6, sig, 1.0)
    return (snip - mu) / sig


def build_rf_maps(snip_z, onsets, xs_trial, ys_trial, resp_win, centers, smooth):
    n_rois = snip_z.shape[1]
    resp_mask = (centers >= resp_win[0]) & (centers < resp_win[1])
    bl_mask   = centers < 0

    rf_maps  = np.zeros((n_rois, len(ys_grid), len(xs_grid)), dtype=np.float32)
    for xi, x in enumerate(xs_grid):
        for yi, y in enumerate(ys_grid):
            m = (xs_trial == x) & (ys_trial == y)
            if m.sum() == 0:
                continue
            resp = np.nanmean(snip_z[m][:, :, resp_mask], axis=(0, 2))
            base = np.nanmean(snip_z[m][:, :, bl_mask  ], axis=(0, 2))
            rf_maps[:, yi, xi] = resp - base

    peak = np.array([gaussian_filter(rf_maps[u], smooth).max() for u in range(n_rois)])
    return rf_maps, peak

# ── Process each plane ────────────────────────────────────────────────
results = {}
for plane in PLANES:
    print(f"\n{plane}…")
    data, ts, is_soma = load_plane_dff(plane)
    n_soma = is_soma.sum()
    print(f"  {data.shape[1]} ROIs → {n_soma} soma")

    valid = (starts + WINDOW[0] >= ts[0]) & (starts + WINDOW[1] <= ts[-1])
    onsets = starts[valid]
    xs_v   = xs_raw[valid]
    ys_v   = ys_raw[valid]
    print(f"  {valid.sum()}/{len(starts)} RF trials within recording")

    snip, tc, dt = extract_snippets(data, ts, is_soma, onsets, WINDOW)
    print(f"  snippets: {snip.shape}  rate: {1/dt:.1f} Hz")

    z    = zscore(snip, tc)
    maps, peak = build_rf_maps(z, onsets, xs_v, ys_v, RESP_WIN, tc, RF_SMOOTH)

    # Population PSTH (mean across soma ROIs then trials)
    pop = np.nanmean(z, axis=1)       # (trial, time)
    m   = np.nanmean(pop, axis=0)
    s   = np.nanstd(pop, axis=0) / np.sqrt(np.isfinite(pop).sum(axis=0).clip(1))

    results[plane] = dict(
        tc=tc, m=m, s=s, maps=maps, peak=peak,
        n_soma=n_soma, n_trials=valid.sum(), dt=dt,
        xs_v=xs_v, ys_v=ys_v
    )
    print(f"  peak z-score range: [{peak.min():.3f}, {peak.max():.3f}]")

# ── Figure 1: population PSTH per plane ──────────────────────────────
n_planes = len(PLANES)
fig1, axes = plt.subplots(2, 4, figsize=(16, 7), sharey=False)
axes = axes.ravel()

for ax, plane in zip(axes, PLANES):
    r = results[plane]
    tc, m, s = r["tc"], r["m"], r["s"]

    ax.axvspan(0, 0.25, color="gray", alpha=0.12)
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.axvline(0, color="gray", lw=0.8, ls=":")
    ax.fill_between(tc, m - s, m + s, color="#4878CF", alpha=0.3)
    ax.plot(tc, m, color="#4878CF", lw=2)

    # Latency
    bl_mu = np.nanmean(m[tc < 0])
    bl_sd = np.nanstd(m[tc < 0])
    first = np.where((tc > 0) & (m > bl_mu + 2 * bl_sd))[0]
    if len(first):
        lat = tc[first[0]]
        ax.axvline(lat, color="red", lw=1.2, ls="--")
        ax.text(lat + 0.03, m.max(), f"≈{lat*1000:.0f} ms", color="red", fontsize=7)

    ax.set(title=f"{plane}  ({r['n_soma']} soma, {r['n_trials']} trials)",
           xlabel="Time from onset (s)", ylabel="Mean z-score", xlim=WINDOW)
    ax.spines[["top", "right"]].set_visible(False)

fig1.suptitle(f"Mesoscope RF mapping — population PSTH per plane\n"
              f"sub-839909 · {len(xs_grid)}×{len(ys_grid)} grid · "
              f"response window {RESP_WIN[0]*1000:.0f}–{RESP_WIN[1]*1000:.0f} ms",
              fontsize=11, fontweight="bold")
fig1.tight_layout()
fig1.savefig("rf_mapping_meso_psth.png", dpi=150, bbox_inches="tight")
print("\nSaved → rf_mapping_meso_psth.png")

# ── Figure 2: ROI-averaged map per plane ─────────────────────────────
fig2, axes2 = plt.subplots(2, 4, figsize=(16, 7))
axes2 = axes2.ravel()

for ax, plane in zip(axes2, PLANES):
    r = results[plane]
    avg_map = gaussian_filter(r["maps"].mean(axis=0), RF_SMOOTH)
    vmax = np.percentile(np.abs(avg_map), 99)
    im = ax.imshow(avg_map, origin="lower", cmap="RdBu_r",
                   extent=[xs_grid[0]-5, xs_grid[-1]+5,
                           ys_grid[0]-5, ys_grid[-1]+5],
                   vmin=-vmax, vmax=vmax, aspect="equal")
    pk = np.unravel_index(np.argmax(avg_map), avg_map.shape)
    ax.plot(xs_grid[pk[1]], ys_grid[pk[0]], "k+", ms=8, mew=1.5)
    ax.set(title=f"{plane}  (peak={r['peak'].max():.2f})",
           xlabel="X (°)", ylabel="Y (°)")
    plt.colorbar(im, ax=ax, label="z-score")

fig2.suptitle("Mesoscope RF mapping — ROI-averaged map per plane",
              fontsize=11, fontweight="bold")
fig2.tight_layout()
fig2.savefig("rf_mapping_meso_allplanes.png", dpi=150, bbox_inches="tight")
print("Saved → rf_mapping_meso_allplanes.png")

# ── Figure 3: Top individual ROI maps (best plane by peak z-score) ────
best_plane = max(PLANES, key=lambda p: results[p]["peak"].max())
r = results[best_plane]
print(f"\nBest plane for individual maps: {best_plane} "
      f"(max peak={r['peak'].max():.3f})")

n_cols = 6
n_rows = int(np.ceil(N_TOP_ROIS / n_cols))
fig3, axes3 = plt.subplots(n_rows, n_cols,
                            figsize=(n_cols * 2.2, n_rows * 2.2),
                            squeeze=False)

order    = np.argsort(r["peak"])[::-1][:N_TOP_ROIS]
all_maps = np.array([gaussian_filter(r["maps"][u], RF_SMOOTH) for u in range(r["maps"].shape[0])])
vmax     = np.percentile(np.abs(all_maps[order]), 98)

for plot_i in range(n_rows * n_cols):
    row, col = divmod(plot_i, n_cols)
    ax = axes3[row, col]
    if plot_i >= N_TOP_ROIS:
        ax.set_visible(False)
        continue
    uid = order[plot_i]
    sm  = gaussian_filter(r["maps"][uid], RF_SMOOTH)
    im  = ax.imshow(sm, origin="lower", cmap="RdBu_r",
                    extent=[xs_grid[0]-5, xs_grid[-1]+5,
                            ys_grid[0]-5, ys_grid[-1]+5],
                    vmin=-vmax, vmax=vmax, aspect="equal")
    pk  = np.unravel_index(np.argmax(sm), sm.shape)
    ax.plot(xs_grid[pk[1]], ys_grid[pk[0]], "k+", ms=7, mew=1.5)
    ax.set(xticks=xs_grid[::3], yticks=ys_grid[::3])
    ax.tick_params(labelsize=5)
    ax.set_title(f"ROI {uid}  z={r['peak'][uid]:.2f}", fontsize=7, pad=2)
    if col == 0: ax.set_ylabel("Y (°)", fontsize=6)
    if row == n_rows - 1: ax.set_xlabel("X (°)", fontsize=6)

cbar_ax = fig3.add_axes([0.92, 0.15, 0.015, 0.7])
fig3.colorbar(im, cax=cbar_ax, label="z-score vs baseline")
fig3.suptitle(
    f"Mesoscope {best_plane} RF maps — top {N_TOP_ROIS} soma ROIs\n"
    f"sub-839909 · {r['n_soma']} soma · {r['n_trials']} RF trials · "
    f"response window {RESP_WIN[0]*1000:.0f}–{RESP_WIN[1]*1000:.0f} ms",
    fontsize=10, fontweight="bold"
)
fig3.tight_layout(rect=[0, 0, 0.91, 0.93])
fig3.savefig("rf_mapping_meso_maps.png", dpi=150, bbox_inches="tight")
print(f"Saved → rf_mapping_meso_maps.png")

f.close()
print("\nDone.")
