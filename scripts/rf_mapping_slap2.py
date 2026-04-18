"""SLAP2 RF mapping — alignment diagnostic + per-ROI RF maps.

Uses sub-803496 (13 DMD1 ROIs, 39 DMD2 ROIs).
RF block falls at end of session; both DMDs stop before it ends, so we
filter trials to each DMD's actual recording window.

Alignment follows the reference notebook (OpenScope2Slap2Oddballs2.ipynb):
  - Per-DMD hardware timing offsets: DMD1 +115 ms, DMD2 −165 ms
  - np.interp over finite-only timestamps — robust to gaps, jitter, drift
  - dt estimated from mid-recording (avoids 2.9 MHz calibration burst)

Saves:
  rf_mapping_slap2_psth.png   — alignment PSTH for both DMDs
  rf_mapping_slap2_maps.png   — per-ROI RF maps for DMD2 (more ROIs)
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
ASSET_ID  = "d23a03af-c3bd-4cf0-9492-6dca96fb201d"   # sub-803496
WINDOW    = (-0.5, 0.8)     # seconds around (corrected) RF trial onset
RESP_WIN  = (0.05, 0.50)    # response scoring window (post-stim, after offset)
RF_SMOOTH = 0.8             # gaussian sigma (grid units) for display

# Per-DMD hardware timing offsets from reference notebook (oddball block estimate).
# Apply as: corrected_t0 = start_time + offset
OFFSETS = {"DMD1": +0.115, "DMD2": -0.165}

# ── Load ──────────────────────────────────────────────────────────────
print("Opening NWB (sub-803496)…")
t0 = time.time()
handle = open_nwb(ASSET_ID)
trials = load_trials(handle)
rf     = trials[trials["block_kind"] == "rf_mapping"].copy().reset_index(drop=True)
print(f"  opened in {time.time()-t0:.1f}s")
print(f"  RF trials (total): {len(rf)}")

h5 = handle.h5

# ── Per-DMD validity filter ───────────────────────────────────────────
dmds = {}
for dmd in ["DMD1", "DMD2"]:
    path = f"processing/ophys/Fluorescence_{dmd}/{dmd}_dFF"
    ts   = h5[f"{path}/timestamps"][:]
    data = h5[f"{path}/data"]
    n_rois = data.shape[1]

    pre, post = WINDOW
    offset = OFFSETS[dmd]
    # Corrected trial onsets must fit within recording window
    corrected_starts = rf["start_time"].values + offset
    valid = (corrected_starts + pre  >= ts[0]) & \
            (corrected_starts + post <= ts[-1])
    rf_valid = rf[valid].reset_index(drop=True)

    # dt from mid-recording — skips the 2.9 MHz calibration burst at start
    mid = len(ts) // 2
    dt  = float(np.median(np.diff(ts[mid:mid+2000])))

    dmds[dmd] = dict(ts=ts, data=data, n_rois=n_rois, dt=dt,
                     rf_valid=rf_valid, valid_mask=valid)
    print(f"  {dmd}: {n_rois} ROIs, recording {ts[0]:.0f}–{ts[-1]:.0f}s  "
          f"offset={offset*1000:+.0f}ms  "
          f"→ {valid.sum()}/{len(rf)} RF trials valid "
          f"({100*valid.mean():.0f}%)")

# ── Extract dFF snippets via linear interpolation ─────────────────────
def extract_snippets(dmd_info, window, offset_sec):
    """Return (n_valid_trials, n_rois, n_samples), time_centers.

    Uses np.interp over finite-only timestamps — robust to gaps, jitter,
    and slow clock drift (as characterised in the reference notebook).
    The per-DMD offset shifts trial onset into neural time before sampling.
    """
    ts, data, rf_valid = dmd_info["ts"], dmd_info["data"], dmd_info["rf_valid"]
    dt = dmd_info["dt"]
    pre, post = window

    n_samples = int(round((post - pre) / dt))
    t_rel = np.linspace(pre, pre + (n_samples - 1) * dt, n_samples)
    t_centers = t_rel + dt / 2

    # Corrected onset times: shift into neural / DMD clock
    onsets = rf_valid["start_time"].values + offset_sec

    # Read one contiguous HDF5 chunk covering all corrected onsets
    t_start_all = onsets.min() + pre - 1.0
    t_end_all   = onsets.max() + post + 1.0
    span_i0 = max(0, int(np.searchsorted(ts, t_start_all)))
    span_i1 = min(data.shape[0], int(np.searchsorted(ts, t_end_all)) + 1)

    print(f"    reading span [{span_i0}:{span_i1}] = "
          f"{span_i1-span_i0} samples × {data.shape[1]} ROIs …")
    trace   = data[span_i0:span_i1, :].astype(np.float32)
    ts_span = ts[span_i0:span_i1]

    n_trials = len(rf_valid)
    n_rois   = data.shape[1]
    out = np.full((n_trials, n_rois, n_samples), np.nan, dtype=np.float32)

    # Per-ROI interpolation: handles any gaps/NaNs in the trace
    # Vectorise over all trials at once for speed
    t_query = onsets[:, None] + t_rel[None, :]  # (n_trials, n_samples)
    t_flat  = t_query.ravel()                   # (n_trials * n_samples,)

    for roi in range(n_rois):
        y = trace[:, roi]
        finite = np.isfinite(y)
        ts_f = ts_span[finite]
        y_f  = y[finite]
        if len(ts_f) < 10:
            continue
        vals = np.interp(t_flat, ts_f, y_f, left=np.nan, right=np.nan)
        out[:, roi, :] = vals.reshape(n_trials, n_samples)

    return out, t_centers, dt


print("\nExtracting DMD1 snippets…")
snip1, tc1, dt1 = extract_snippets(dmds["DMD1"], WINDOW, OFFSETS["DMD1"])
print(f"  shape: {snip1.shape}  rate: {1/dt1:.1f} Hz")

print("Extracting DMD2 snippets…")
snip2, tc2, dt2 = extract_snippets(dmds["DMD2"], WINDOW, OFFSETS["DMD2"])
print(f"  shape: {snip2.shape}  rate: {1/dt2:.1f} Hz")

# ── Baseline z-score per ROI ──────────────────────────────────────────
def zscore(snip, centers, bl=(-0.2, 0.0)):
    bl_mask = (centers >= bl[0]) & (centers < bl[1])
    mu  = np.nanmean(snip[:, :, bl_mask], axis=(0, 2), keepdims=True)
    sig = np.nanstd( snip[:, :, bl_mask], axis=(0, 2), keepdims=True)
    sig = np.where(sig > 1e-6, sig, 1.0)
    return (snip - mu) / sig

z1 = zscore(snip1, tc1)
z2 = zscore(snip2, tc2)

# ── RF maps ───────────────────────────────────────────────────────────
def build_rf_maps(snip, rf_valid, resp_win, centers, smooth_sigma):
    xs = np.array(sorted(rf_valid["x"].unique()))
    ys = np.array(sorted(rf_valid["y"].unique()))
    n_rois = snip.shape[1]
    resp_mask = (centers >= resp_win[0]) & (centers < resp_win[1])
    bl_mask   = centers < 0

    rf_maps  = np.zeros((n_rois, len(ys), len(xs)), dtype=np.float32)
    peak_dfr = np.zeros(n_rois, dtype=np.float32)

    for xi, x in enumerate(xs):
        for yi, y in enumerate(ys):
            m = (rf_valid["x"].values == x) & (rf_valid["y"].values == y)
            if m.sum() == 0:
                continue
            resp = np.nanmean(snip[m][:, :, resp_mask], axis=(0, 2))  # (n_rois,)
            base = np.nanmean(snip[m][:, :, bl_mask  ], axis=(0, 2))
            rf_maps[:, yi, xi] = resp - base

    for u in range(n_rois):
        peak_dfr[u] = gaussian_filter(rf_maps[u], smooth_sigma).max()

    return rf_maps, peak_dfr, xs, ys


print("\nBuilding RF maps (from z-scored snippets)…")
maps1, peak1, xs, ys = build_rf_maps(z1, dmds["DMD1"]["rf_valid"],
                                      RESP_WIN, tc1, RF_SMOOTH)
maps2, peak2, xs, ys = build_rf_maps(z2, dmds["DMD2"]["rf_valid"],
                                      RESP_WIN, tc2, RF_SMOOTH)
print(f"  DMD1 peak ΔF/F range: [{peak1.min():.3f}, {peak1.max():.3f}]")
print(f"  DMD2 peak ΔF/F range: [{peak2.min():.3f}, {peak2.max():.3f}]")

# ── Figure 1: Alignment PSTH (both DMDs) ─────────────────────────────
fig1, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)

for ax, (dmd, z, tc) in zip(axes, [
        ("DMD1", z1, tc1),
        ("DMD2", z2, tc2),
]):
    n_valid = dmds[dmd]["rf_valid"].shape[0]
    pop = np.nanmean(z, axis=1)          # (trial, time)
    m   = np.nanmean(pop, axis=0)
    s   = np.nanstd( pop, axis=0) / np.sqrt(np.sum(np.isfinite(pop), axis=0).clip(min=1))

    ax.axvspan(0, 0.25, color="gray", alpha=0.12, label="stim (250 ms)")
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.axvline(0, color="gray", lw=0.8, ls=":", label="corrected onset")
    ax.fill_between(tc, m - s, m + s, color="#4878CF", alpha=0.3)
    ax.plot(tc, m, color="#4878CF", lw=2)

    # Latency estimate
    bl_vals = m[tc < 0]
    if len(bl_vals):
        bl_mu = np.nanmean(bl_vals)
        bl_sd = np.nanstd(bl_vals)
        first = np.where((tc > 0) & (m > bl_mu + 2 * bl_sd))[0]
        if len(first):
            lat = tc[first[0]]
            ax.axvline(lat, color="red", lw=1.2, ls="--")
            ax.text(lat + 0.01, m.max(), f"≈{lat*1000:.0f} ms",
                    color="red", fontsize=8)

    offset_ms = OFFSETS[dmd] * 1000
    ax.set(xlabel="Time from corrected onset (s)", ylabel="Mean ΔF/F (z-score)",
           title=f"{dmd} — population PSTH\n"
                 f"({dmds[dmd]['n_rois']} ROIs, {n_valid} valid RF trials, "
                 f"offset {offset_ms:+.0f} ms applied)",
           xlim=WINDOW)
    ax.legend(fontsize=7, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)

fig1.suptitle("SLAP2 RF mapping alignment — sub-803496\n"
              "(per-DMD offset corrected: DMD1 +115 ms, DMD2 −165 ms)",
              fontsize=11, fontweight="bold")
fig1.tight_layout()
out1 = "rf_mapping_slap2_psth.png"
fig1.savefig(out1, dpi=150, bbox_inches="tight")
print(f"\nSaved → {out1}")

# ── Figure 2: Per-ROI RF maps (DMD2, more ROIs) ───────────────────────
n_rois2 = dmds["DMD2"]["n_rois"]
n_cols   = 6
n_rows   = int(np.ceil(n_rois2 / n_cols))

fig2, axes2 = plt.subplots(n_rows, n_cols,
                            figsize=(n_cols * 2.2, n_rows * 2.2),
                            squeeze=False)

order2   = np.argsort(peak2)[::-1]   # best ROIs first
all_maps = np.array([gaussian_filter(maps2[u], RF_SMOOTH) for u in range(n_rois2)])
vmax     = np.percentile(np.abs(all_maps), 98)

for plot_i in range(n_rows * n_cols):
    row, col = divmod(plot_i, n_cols)
    ax = axes2[row, col]
    if plot_i >= n_rois2:
        ax.set_visible(False)
        continue
    uid = order2[plot_i]
    sm  = gaussian_filter(maps2[uid], RF_SMOOTH)
    im  = ax.imshow(sm, origin="lower",
                    extent=[xs[0]-5, xs[-1]+5, ys[0]-5, ys[-1]+5],
                    vmin=-vmax, vmax=vmax, cmap="RdBu_r", aspect="equal")
    pk = np.unravel_index(np.argmax(sm), sm.shape)
    ax.plot(xs[pk[1]], ys[pk[0]], "k+", ms=7, mew=1.5)
    ax.set(xticks=xs[::3], yticks=ys[::3])
    ax.tick_params(labelsize=5)
    ax.set_title(f"ROI {uid}  peak={peak2[uid]:.2f}", fontsize=7, pad=2)
    if col == 0: ax.set_ylabel("Y (°)", fontsize=6)
    if row == n_rows - 1: ax.set_xlabel("X (°)", fontsize=6)

cbar_ax = fig2.add_axes([0.92, 0.15, 0.015, 0.7])
fig2.colorbar(im, cax=cbar_ax, label="z-score vs baseline")

fig2.suptitle(
    f"SLAP2 DMD2 RF maps — sub-803496\n"
    f"{n_rois2} ROIs, {dmds['DMD2']['rf_valid'].shape[0]} valid trials "
    f"({100*dmds['DMD2']['valid_mask'].mean():.0f}% of RF block covered)\n"
    f"offset −165 ms applied · response window "
    f"{RESP_WIN[0]*1000:.0f}–{RESP_WIN[1]*1000:.0f} ms post corrected onset",
    fontsize=10, fontweight="bold"
)
fig2.tight_layout(rect=[0, 0, 0.91, 0.93])
out2 = "rf_mapping_slap2_maps.png"
fig2.savefig(out2, dpi=150, bbox_inches="tight")
print(f"Saved → {out2}")

handle.close()
print("\nDone.")
