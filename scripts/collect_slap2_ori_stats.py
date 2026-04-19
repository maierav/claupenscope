"""Collect SLAP2 orientation tuning statistics across all sessions.

Processes all 12 sessions in dandiset 001424 sequentially, computing per-DMD
OSI/DSI for every ROI in each ori_tuning block. Saves one row per DMD per
session to:

    results/slap2_ori_stats.csv

Resumes from where it left off if interrupted.
Run time: ~5–10 minutes total.

Note on DMD timing offsets: DMD1 stimulus onset leads NWB onset by +115 ms;
DMD2 lags by −165 ms (calibrated on sub-803496). These same offsets are
applied to all sessions as a first approximation — may need per-session
calibration in future iterations.

Usage:
    python scripts/collect_slap2_ori_stats.py
"""
import time
import csv
import os
import sys
import numpy as np
import requests
import remfile
import h5py
from dandi.dandiapi import DandiAPIClient

# Add package to path so we can use the SLAP2 trial loader
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from openscope_pp.loaders.streaming import open_nwb
from openscope_pp.loaders.trials import load_trials

# ── Config ────────────────────────────────────────────────────────────
DANDISET_ID = "001424"
WINDOW      = (-0.3, 0.8)
RESP_WIN    = (0.05, 0.35)
BL_WIN      = (-0.25, 0.0)
OFFSETS     = {"DMD1": +0.115, "DMD2": -0.165}
OSI_THRESH  = 0.2
OUT_CSV     = "results/slap2_ori_stats.csv"

CSV_FIELDS = [
    "asset_id", "subject_id", "session_date",
    "dmd", "n_rois",
    "mean_OSI", "median_OSI", "std_OSI", "q25_OSI", "q75_OSI",
    "mean_DSI", "median_DSI", "std_DSI", "q25_DSI", "q75_DSI",
    "n_tuned",
]

# ── Helper: extract dFF snippets for one DMD channel ──────────────────
def extract_dmd(h5, dmd, ori_trials, offset_sec, window):
    """Return (snip, tc, oris_v, n_rois) or None if channel absent."""
    path = f"processing/ophys/Fluorescence_{dmd}/{dmd}_dFF"
    if path not in h5 and f"processing/ophys/Fluorescence_{dmd}" not in h5:
        return None
    try:
        ts   = h5[f"{path}/timestamps"][:]
        data = h5[f"{path}/data"]
    except KeyError:
        return None

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

    return snip, tc, oris_v, n_rois


def compute_selectivity(snip, tc, oris_v, dirs_rad):
    """Return (OSI, DSI) arrays over ROIs."""
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
    return OSI, DSI


# ── Load asset list ───────────────────────────────────────────────────
print(f"Fetching asset list for dandiset {DANDISET_ID}…")
client = DandiAPIClient()
ds     = client.get_dandiset(DANDISET_ID)
assets = sorted(list(ds.get_assets()), key=lambda a: a.size)
print(f"  {len(assets)} assets")

# ── Set up output CSV ─────────────────────────────────────────────────
os.makedirs("results", exist_ok=True)
already_done = set()
if os.path.exists(OUT_CSV):
    with open(OUT_CSV) as fh:
        for row in csv.DictReader(fh):
            already_done.add(row["asset_id"])
    print(f"  Resuming — {len(already_done)} assets already in CSV")

write_header = not os.path.exists(OUT_CSV) or len(already_done) == 0
csv_fh = open(OUT_CSV, "a", newline="")
writer = csv.DictWriter(csv_fh, fieldnames=CSV_FIELDS)
if write_header:
    writer.writeheader()

# ── Main loop ─────────────────────────────────────────────────────────
t_total = time.time()
for si, asset in enumerate(assets):
    asset_id = asset.identifier
    if asset_id in already_done:
        print(f"[{si+1:2d}/{len(assets)}] SKIP {asset.path}")
        continue

    subject_id = asset.path.split("/")[0]
    # Date from session label e.g. SLAP2-803496-2025-07-02-11-50-06 → 2025-07-02
    parts = asset.path.split("SLAP2-")
    date_part = parts[1][7:17] if len(parts) > 1 else "unknown"

    print(f"[{si+1:2d}/{len(assets)}] {subject_id}  {date_part}  …", end=" ", flush=True)
    t0 = time.time()

    try:
        handle = open_nwb(asset_id)
        trials = load_trials(handle)
        h5     = handle.h5

        ori_t = trials[trials["block_kind"] == "ori_tuning"].reset_index(drop=True)
        if len(ori_t) == 0:
            print("no ori_tuning block — skip")
            handle.close()
            continue

        dirs_rad = np.array(sorted(ori_t["orientation"].unique()))

        wrote_any = False
        for dmd, offset in OFFSETS.items():
            result = extract_dmd(h5, dmd, ori_t, offset, WINDOW)
            if result is None:
                continue
            snip, tc, oris_v, n_rois = result
            OSI, DSI = compute_selectivity(snip, tc, oris_v, dirs_rad)

            row = {
                "asset_id":    asset_id,
                "subject_id":  subject_id,
                "session_date": date_part,
                "dmd":         dmd,
                "n_rois":      n_rois,
                "mean_OSI":    float(np.mean(OSI)),
                "median_OSI":  float(np.median(OSI)),
                "std_OSI":     float(np.std(OSI)),
                "q25_OSI":     float(np.percentile(OSI, 25)),
                "q75_OSI":     float(np.percentile(OSI, 75)),
                "mean_DSI":    float(np.mean(DSI)),
                "median_DSI":  float(np.median(DSI)),
                "std_DSI":     float(np.std(DSI)),
                "q25_DSI":     float(np.percentile(DSI, 25)),
                "q75_DSI":     float(np.percentile(DSI, 75)),
                "n_tuned":     int((OSI >= OSI_THRESH).sum()),
            }
            writer.writerow(row)
            wrote_any = True

        csv_fh.flush()
        handle.close()
        print(f"{time.time()-t0:.0f}s")

    except Exception as e:
        print(f"ERROR: {e}")

csv_fh.close()
print(f"\nDone. Total time: {(time.time()-t_total)/60:.1f} min")
print(f"Results saved → {OUT_CSV}")
