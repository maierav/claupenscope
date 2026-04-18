"""Mesoscope orientation tuning — all 8 planes, soma ROIs, grouped by area.

Uses Control block 2_presentations (14 directions × 80 trials, 0.267s stimuli).
Orientations stored in radians in the NWB file.

Areas:
  VISp — planes 0–3 (primary visual cortex)
  VISl — planes 4–7 (lateral visual area)

Saves:
  ori_tuning_meso_polar.png    — top-24 polar curves per area (best plane per area)
  ori_tuning_meso_planes.png   — OSI / DSI per plane, colour-coded by area
  ori_tuning_meso_areas.png    — area-level OSI / DSI comparison (pooled across planes)
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
AREAS       = {
    "VISp": ["VISp_0", "VISp_1", "VISp_2", "VISp_3"],
    "VISl": ["VISl_4", "VISl_5", "VISl_6", "VISl_7"],
}
AREA_COLORS = {"VISp": "#4878CF", "VISl": "#D65F5F"}
BLOCK_NAME  = "Control block 2_presentations"
WINDOW      = (-0.3, 1.2)     # wide to capture slow GCaMP response
RESP_WIN    = (0.1, 0.8)      # response window (GCaMP dynamics)
BL_WIN      = (-0.25, 0.0)
N_POLAR     = 12              # top ROIs per area to show in polar grid

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

# ── Figure 1: Polar curves — top N_POLAR per area (best plane per area) ──
theta = np.append(dirs_rad, dirs_rad[0])
cmap  = plt.cm.hsv

fig, all_axes = plt.subplots(
    len(AREAS) * int(np.ceil(N_POLAR / 6)), 6,
    figsize=(6 * 2.0, len(AREAS) * int(np.ceil(N_POLAR / 6)) * 2.0),
    subplot_kw={"projection": "polar"}, squeeze=False
)

row_offset = 0
for area_name, area_planes in AREAS.items():
    best_plane = max(area_planes, key=lambda p: np.median(results[p]["OSI"]))
    r = results[best_plane]
    n_show = min(N_POLAR, r["n_soma"])
    order  = np.argsort(r["OSI"])[::-1][:n_show]
    n_rows_area = int(np.ceil(N_POLAR / 6))

    for plot_i in range(n_rows_area * 6):
        row = row_offset + plot_i // 6
        col = plot_i % 6
        ax  = all_axes[row, col]
        if plot_i >= n_show:
            ax.set_visible(False); continue
        uid   = order[plot_i]
        c     = np.clip(np.append(r["curves"][uid], r["curves"][uid][0]), 0, None)
        color = cmap(r["pref_ori"][uid] / np.pi)
        ax.plot(theta, c, lw=1.5, color=color)
        ax.fill(theta, c, alpha=0.25, color=color)
        ax.set(xticks=[], yticks=[])
        ax.set_title(f"{area_name} ROI {uid}\nOSI={r['OSI'][uid]:.2f}", fontsize=6, pad=2)

    row_offset += n_rows_area
    print(f"  {area_name}: best plane = {best_plane}  (median OSI={np.median(r['OSI']):.3f})")

fig.suptitle(
    f"Mesoscope orientation tuning — top {N_POLAR} ROIs per area\n"
    f"sub-839909  ·  {n_dirs} directions  ·  response {RESP_WIN[0]*1000:.0f}–{RESP_WIN[1]*1000:.0f} ms",
    fontsize=9, fontweight="bold"
)
fig.tight_layout()
fig.savefig("ori_tuning_meso_polar.png", dpi=150, bbox_inches="tight")
print("\nSaved → ori_tuning_meso_polar.png")

# ── Figure 2: OSI / DSI per plane, colour-coded by area ──────────────
bins = np.linspace(0, 1, 14)
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
axes = axes.ravel()

for ax, plane in zip(axes, PLANES):
    area = "VISp" if plane.startswith("VISp") else "VISl"
    col  = AREA_COLORS[area]
    r    = results[plane]
    ax.hist(r["OSI"], bins=bins, color=col,   alpha=0.75, label="OSI")
    ax.hist(r["DSI"], bins=bins, color="gray", alpha=0.45, label="DSI")
    ax.axvline(np.median(r["OSI"]), color=col,   lw=1.5, ls="--")
    ax.axvline(np.median(r["DSI"]), color="gray", lw=1.5, ls="--")
    ax.set(title=f"[{area}]  {plane}  ({r['n_soma']} soma)\n"
                 f"OSI={np.median(r['OSI']):.2f}  DSI={np.median(r['DSI']):.2f}",
           xlabel="Index", ylabel="ROI count", xlim=(0, 1))
    ax.legend(fontsize=6, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("Mesoscope orientation tuning — OSI & DSI per plane  (sub-839909)\n"
             "Blue = VISp · Red = VISl",
             fontsize=11, fontweight="bold")
fig.tight_layout()
fig.savefig("ori_tuning_meso_planes.png", dpi=150, bbox_inches="tight")
print("Saved → ori_tuning_meso_planes.png")

# ── Figure 3: Area-level comparison (pooled across planes) ────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
bins = np.linspace(0, 1, 16)

for area_name, area_planes in AREAS.items():
    color = AREA_COLORS[area_name]
    OSI_pool = np.concatenate([results[p]["OSI"] for p in area_planes])
    DSI_pool = np.concatenate([results[p]["DSI"] for p in area_planes])
    pori_pool = np.concatenate([results[p]["pref_ori"] for p in area_planes])
    n_pool = len(OSI_pool)

    # Panel 1: OSI
    axes[0].hist(OSI_pool, bins=bins, color=color, alpha=0.6,
                 label=f"{area_name}  n={n_pool}\nmedian={np.median(OSI_pool):.2f}")
    axes[0].axvline(np.median(OSI_pool), color=color, lw=2, ls="--")

    # Panel 2: DSI
    axes[1].hist(DSI_pool, bins=bins, color=color, alpha=0.6,
                 label=f"{area_name}  median={np.median(DSI_pool):.2f}")
    axes[1].axvline(np.median(DSI_pool), color=color, lw=2, ls="--")

axes[0].set(xlabel="OSI", ylabel="ROI count", title="Orientation Selectivity by Area")
axes[0].legend(fontsize=8); axes[0].spines[["top","right"]].set_visible(False)
axes[1].set(xlabel="DSI", ylabel="ROI count", title="Direction Selectivity by Area")
axes[1].legend(fontsize=8); axes[1].spines[["top","right"]].set_visible(False)

# Panel 3: preferred orientation wheel per area
ax3 = fig.add_subplot(1, 3, 3, projection="polar")
axes[2].set_visible(False)
bin_edges = np.linspace(0, 180, 13)
bc = np.radians(0.5 * (bin_edges[:-1] + bin_edges[1:]))
theta_full = np.concatenate([bc, bc + np.pi])

for area_name, area_planes in AREAS.items():
    color = AREA_COLORS[area_name]
    pori_pool = np.concatenate([results[p]["pref_ori"] for p in area_planes])
    counts, _ = np.histogram(np.degrees(pori_pool), bins=bin_edges)
    counts_full = np.tile(counts / counts.max(), 2)   # normalise for overlay
    ax3.plot(np.append(theta_full, theta_full[0]),
             np.append(counts_full, counts_full[0]),
             color=color, lw=2, label=area_name)
    ax3.fill(theta_full, counts_full, color=color, alpha=0.2)

ax3.set_title("Preferred orientation\n(normalised)", fontsize=8)
ax3.set_xticklabels(["0°","45°","90°","135°","180°","225°","270°","315°"], fontsize=6)
ax3.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1.3, 1.1))

fig.suptitle("Mesoscope orientation tuning — VISp vs VISl  (sub-839909)",
             fontsize=11, fontweight="bold")
fig.tight_layout()
fig.savefig("ori_tuning_meso_areas.png", dpi=150, bbox_inches="tight")
print("Saved → ori_tuning_meso_areas.png")

f.close()
print("\nDone.")
