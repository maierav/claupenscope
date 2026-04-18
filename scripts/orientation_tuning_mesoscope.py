"""Mesoscope orientation tuning — all 8 planes, soma ROIs.

Uses Control block 2_presentations (14 directions × 80 trials, 0.267s stimuli).
Orientations stored in radians in the NWB file.

Saves:
  ori_tuning_meso_polar.png   — top-24 polar curves for best plane (by median OSI)
  ori_tuning_meso_stats.png   — OSI / DSI distributions per plane
  ori_tuning_meso_orimap.png  — spatial map of preferred orientation (VISp_0)
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
PLANES      = ["VISp_0", "VISp_1", "VISp_2", "VISp_3",
               "VISl_4", "VISl_5", "VISl_6", "VISl_7"]
BLOCK_NAME  = "Control block 2_presentations"
WINDOW      = (-0.3, 1.2)     # wide to capture slow GCaMP response
RESP_WIN    = (0.1, 0.8)      # response window (GCaMP dynamics)
BL_WIN      = (-0.25, 0.0)
N_POLAR     = 24              # top ROIs to show in polar grid

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

# ── Load orientation block ─────────────────────────────────────────────
def decode_col(arr):
    return np.array([float(v.decode() if isinstance(v, bytes) else v) for v in arr])

g       = f["intervals"][BLOCK_NAME]
starts  = g["start_time"][:]
oris_rad = decode_col(g["Orientation"][:])          # stored in radians
dirs_rad = np.array(sorted(np.unique(oris_rad)))
dirs_deg = np.degrees(dirs_rad)
n_dirs   = len(dirs_rad)

print(f"\n{BLOCK_NAME}: {len(starts)} trials, {n_dirs} directions")
print(f"  Directions (deg): {dirs_deg.round(1)}")
print(f"  Trials per dir:   {len(starts)/n_dirs:.0f}")

# ── Helpers ───────────────────────────────────────────────────────────
def load_plane(plane):
    base     = f"processing/{plane}"
    dff_path = f"{base}/dff_timeseries/dff_timeseries"
    ts       = f[f"{dff_path}/timestamps"][:]
    data     = f[f"{dff_path}/data"]
    is_soma  = f[f"{base}/image_segmentation/roi_table/is_soma"][:].astype(bool)
    return data, ts, is_soma

def extract_snippets(data, ts, is_soma, onsets, oris_v, window):
    pre, post = window
    dt     = float(np.median(np.diff(ts[len(ts)//2 : len(ts)//2 + 200])))
    n_samp = int(round((post - pre) / dt))
    t_rel  = np.linspace(pre, pre + (n_samp - 1) * dt, n_samp)
    tc     = t_rel + dt / 2

    valid  = (onsets + pre >= ts[0]) & (onsets + post <= ts[-1])
    onsets = onsets[valid]; oris_v = oris_v[valid]

    i0 = max(0, int(np.searchsorted(ts, onsets.min() + pre - 2.0)))
    i1 = min(data.shape[0], int(np.searchsorted(ts, onsets.max() + post + 2.0)) + 1)
    trace      = data[i0:i1, :].astype(np.float32)
    ts_span    = ts[i0:i1]
    trace_soma = trace[:, is_soma]
    n_soma     = trace_soma.shape[1]

    t_query = onsets[:, None] + t_rel[None, :]
    t_flat  = t_query.ravel()
    snip    = np.full((len(onsets), n_soma, n_samp), np.nan, dtype=np.float32)
    for roi in range(n_soma):
        y = trace_soma[:, roi]; fin = np.isfinite(y)
        if fin.sum() < 10: continue
        vals = np.interp(t_flat, ts_span[fin], y[fin], left=np.nan, right=np.nan)
        snip[:, roi, :] = vals.reshape(len(onsets), n_samp)
    return snip, tc, oris_v, n_soma

def tuning_stats(snip, tc, oris_v, dirs_rad):
    bl_mask   = (tc >= BL_WIN[0])   & (tc < BL_WIN[1])
    resp_mask = (tc >= RESP_WIN[0]) & (tc < RESP_WIN[1])
    mu  = np.nanmean(snip[:, :, bl_mask], axis=(0, 2), keepdims=True)
    sig = np.nanstd( snip[:, :, bl_mask], axis=(0, 2), keepdims=True)
    sig = np.where(sig > 1e-6, sig, 1.0)
    z   = (snip - mu) / sig

    n_rois = snip.shape[1]
    curves = np.full((n_rois, len(dirs_rad)), np.nan, dtype=np.float32)
    for di, d in enumerate(dirs_rad):
        m = oris_v == d
        if m.sum() == 0: continue
        resp = np.nanmean(z[m][:, :, resp_mask], axis=(0, 2))
        base = np.nanmean(z[m][:, :, bl_mask  ], axis=(0, 2))
        curves[:, di] = resp - base

    angles = dirs_rad
    r_sum  = np.nansum(np.abs(curves), axis=1).clip(1e-9)
    r_ori  = np.nansum(curves * np.exp(2j * angles)[None, :], axis=1)
    r_dir  = np.nansum(curves * np.exp(1j * angles)[None, :], axis=1)
    OSI    = (np.abs(r_ori) / r_sum).real
    DSI    = (np.abs(r_dir) / r_sum).real
    pref_ori = (np.angle(r_ori) / 2) % np.pi
    pref_dir = np.angle(r_dir) % (2 * np.pi)
    return curves, OSI, DSI, pref_ori, pref_dir

# ── Process each plane ────────────────────────────────────────────────
results = {}
for plane in PLANES:
    print(f"\n{plane}…")
    data, ts, is_soma = load_plane(plane)
    snip, tc, oris_v, n_soma = extract_snippets(
        data, ts, is_soma, starts, oris_rad, WINDOW
    )
    print(f"  {is_soma.sum()} soma  snippets {snip.shape}  rate {1/float(np.median(np.diff(ts[len(ts)//2:len(ts)//2+100]))):.1f} Hz")
    curves, OSI, DSI, pref_ori, pref_dir = tuning_stats(snip, tc, oris_v, dirs_rad)
    results[plane] = dict(curves=curves, OSI=OSI, DSI=DSI,
                          pref_ori=pref_ori, pref_dir=pref_dir,
                          n_soma=n_soma, tc=tc)
    print(f"  OSI: {np.median(OSI):.3f} median  DSI: {np.median(DSI):.3f} median")

# ── Figure 1: Polar curves for best plane ────────────────────────────
best_plane = max(PLANES, key=lambda p: np.median(results[p]["OSI"]))
r   = results[best_plane]
n_show  = min(N_POLAR, r["n_soma"])
order   = np.argsort(r["OSI"])[::-1][:n_show]
n_cols  = 6
n_rows  = int(np.ceil(n_show / n_cols))

fig, axes = plt.subplots(n_rows, n_cols,
                          figsize=(n_cols * 2.0, n_rows * 2.0),
                          subplot_kw={"projection": "polar"},
                          squeeze=False)
theta = np.append(dirs_rad, dirs_rad[0])
cmap  = plt.cm.hsv

for plot_i in range(n_rows * n_cols):
    row, col = divmod(plot_i, n_cols)
    ax = axes[row, col]
    if plot_i >= n_show:
        ax.set_visible(False); continue
    uid = order[plot_i]
    c   = np.append(r["curves"][uid], r["curves"][uid][0])
    c   = np.clip(c, 0, None)
    color = cmap(r["pref_ori"][uid] / np.pi)
    ax.plot(theta, c, lw=1.5, color=color)
    ax.fill(theta, c, alpha=0.25, color=color)
    ax.set(xticks=[], yticks=[])
    ax.set_title(f"ROI {uid}\nOSI={r['OSI'][uid]:.2f}", fontsize=6, pad=2)

fig.suptitle(
    f"Mesoscope {best_plane} — top {n_show} soma ROIs by OSI\n"
    f"sub-839909  ·  {n_dirs} directions  ·  response {RESP_WIN[0]*1000:.0f}–{RESP_WIN[1]*1000:.0f} ms",
    fontsize=9, fontweight="bold"
)
fig.tight_layout()
fig.savefig("ori_tuning_meso_polar.png", dpi=150, bbox_inches="tight")
print("\nSaved → ori_tuning_meso_polar.png")

# ── Figure 2: OSI / DSI distributions per plane ───────────────────────
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
axes = axes.ravel()
bins = np.linspace(0, 1, 14)

for ax, plane in zip(axes, PLANES):
    r = results[plane]
    ax.hist(r["OSI"], bins=bins, color="#4878CF", alpha=0.7, label="OSI")
    ax.hist(r["DSI"], bins=bins, color="#D65F5F", alpha=0.5, label="DSI")
    ax.axvline(np.median(r["OSI"]), color="#4878CF", lw=1.5, ls="--")
    ax.axvline(np.median(r["DSI"]), color="#D65F5F", lw=1.5, ls="--")
    ax.set(title=f"{plane}  ({r['n_soma']} soma)\nOSI={np.median(r['OSI']):.2f}  DSI={np.median(r['DSI']):.2f}",
           xlabel="Index", ylabel="ROI count", xlim=(0, 1))
    ax.legend(fontsize=6, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("Mesoscope orientation tuning — OSI & DSI per plane  (sub-839909)",
             fontsize=11, fontweight="bold")
fig.tight_layout()
fig.savefig("ori_tuning_meso_stats.png", dpi=150, bbox_inches="tight")
print("Saved → ori_tuning_meso_stats.png")

# ── Figure 3: Spatial preferred-orientation map (VISp_0) ─────────────
# Load ROI centroids from segmentation mask
plane   = "VISp_0"
r       = results[plane]
base    = f"processing/{plane}/image_segmentation"
try:
    cx = f[f"{base}/roi_table/x"][:].astype(float)
    cy = f[f"{base}/roi_table/y"][:].astype(float)
    is_soma = f[f"{base}/roi_table/is_soma"][:].astype(bool)
    cx_s = cx[is_soma]; cy_s = cy[is_soma]

    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(cx_s, cy_s,
                    c=r["pref_ori"] / np.pi,      # 0–1 maps to 0–180°
                    cmap="hsv", vmin=0, vmax=1,
                    s=np.clip(r["OSI"] * 60, 4, 60),   # size ~ OSI
                    alpha=0.85, edgecolors="none")
    cb = plt.colorbar(sc, ax=ax, label="Preferred orientation (°)")
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1])
    cb.set_ticklabels(["0°", "45°", "90°", "135°", "180°"])
    ax.set(xlabel="X (px)", ylabel="Y (px)",
           title=f"VISp_0 — preferred orientation map  ({r['n_soma']} soma)\n"
                 f"dot size ∝ OSI  ·  color = preferred orientation")
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig("ori_tuning_meso_orimap.png", dpi=150, bbox_inches="tight")
    print("Saved → ori_tuning_meso_orimap.png")
except Exception as e:
    print(f"  Orientation map skipped (centroid data not available): {e}")

f.close()
print("\nDone.")
