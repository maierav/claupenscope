"""Collect mesoscope orientation tuning statistics across all sessions.

Processes all 42 sessions in dandiset 001768 sequentially, computing per-plane
OSI/DSI for every soma ROI. Saves one row per plane per session to:

    results/meso_ori_stats.csv

Resumes from where it left off if interrupted (skips already-processed assets).
Run time: ~2–3 hours. Intended to be run in the background.

Usage:
    python scripts/collect_meso_ori_stats.py
"""
import time
import csv
import os
import numpy as np
import requests
import remfile
import h5py
from dandi.dandiapi import DandiAPIClient

# ── Config ────────────────────────────────────────────────────────────
DANDISET_ID = "001768"
BLOCK_NAME  = "Control block 2_presentations"
PLANES      = ["VISp_0", "VISp_1", "VISp_2", "VISp_3",
               "VISl_4", "VISl_5", "VISl_6", "VISl_7"]
AREAS       = {"VISp": ["VISp_0","VISp_1","VISp_2","VISp_3"],
               "VISl": ["VISl_4","VISl_5","VISl_6","VISl_7"]}
WINDOW      = (-0.3, 1.2)
RESP_WIN    = (0.1, 0.8)
BL_WIN      = (-0.25, 0.0)
OSI_THRESH  = 0.2
OUT_CSV     = "results/meso_ori_stats.csv"

CSV_FIELDS  = [
    "asset_id", "subject_id", "session_date",
    "area", "plane", "n_soma",
    "mean_OSI", "median_OSI", "std_OSI", "q25_OSI", "q75_OSI",
    "mean_DSI", "median_DSI", "std_DSI", "q25_DSI", "q75_DSI",
    "n_tuned",   # OSI >= OSI_THRESH
]

# ── Helpers ───────────────────────────────────────────────────────────
def decode_col(arr):
    return np.array([float(v.decode() if isinstance(v, bytes) else v) for v in arr])

def open_asset(asset_id):
    url_r = requests.get(
        f"https://api.dandiarchive.org/api/assets/{asset_id}/download/",
        allow_redirects=True, stream=True, timeout=30
    )
    url = url_r.url; url_r.close()
    return h5py.File(remfile.File(url), "r")

def extract_and_compute(f, plane, starts, oris_rad, dirs_rad):
    """Return (OSI array, DSI array, n_soma) for one plane, or None if unavailable."""
    try:
        base     = f"processing/{plane}"
        dff_path = f"{base}/dff_timeseries/dff_timeseries"
        ts       = f[f"{dff_path}/timestamps"][:]
        data     = f[f"{dff_path}/data"]
        is_soma  = f[f"{base}/image_segmentation/roi_table/is_soma"][:].astype(bool)
    except Exception as e:
        print(f"    Skipping {plane}: {e}")
        return None

    pre, post = WINDOW
    dt     = float(np.median(np.diff(ts[len(ts)//2 : len(ts)//2 + 200])))
    n_samp = int(round((post - pre) / dt))
    t_rel  = np.linspace(pre, pre + (n_samp - 1) * dt, n_samp)
    tc     = t_rel + dt / 2

    valid   = (starts + pre >= ts[0]) & (starts + post <= ts[-1])
    onsets  = starts[valid]
    oris_v  = oris_rad[valid]

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

    bl_mask   = (tc >= BL_WIN[0])   & (tc < BL_WIN[1])
    resp_mask = (tc >= RESP_WIN[0]) & (tc < RESP_WIN[1])
    mu  = np.nanmean(snip[:, :, bl_mask], axis=(0, 2), keepdims=True)
    sig = np.nanstd( snip[:, :, bl_mask], axis=(0, 2), keepdims=True)
    sig = np.where(sig > 1e-6, sig, 1.0)
    z   = (snip - mu) / sig

    curves = np.full((n_soma, len(dirs_rad)), np.nan, dtype=np.float32)
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
    return OSI, DSI, n_soma

def stats_row(arr):
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "q25": float(np.percentile(arr, 25)),
        "q75": float(np.percentile(arr, 75)),
    }

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

    parts      = asset.path.replace("sub-","").split("/")[0]
    subject_id = f"sub-{parts}"
    date_part  = asset.path.split("ophys-")[1][:10] if "ophys-" in asset.path else "unknown"

    print(f"[{si+1:2d}/{len(assets)}] {subject_id}  {date_part}  …", end=" ", flush=True)
    t0 = time.time()

    try:
        f = open_asset(asset_id)

        # Load orientation block
        g        = f["intervals"][BLOCK_NAME]
        starts   = g["start_time"][:]
        oris_rad = decode_col(g["Orientation"][:])
        dirs_rad = np.array(sorted(np.unique(oris_rad)))

        for plane in PLANES:
            result = extract_and_compute(f, plane, starts, oris_rad, dirs_rad)
            if result is None:
                continue
            OSI, DSI, n_soma = result
            area = "VISp" if plane.startswith("VISp") else "VISl"
            row = {
                "asset_id":     asset_id,
                "subject_id":   subject_id,
                "session_date": date_part,
                "area":         area,
                "plane":        plane,
                "n_soma":       n_soma,
                "mean_OSI":     np.mean(OSI),
                "median_OSI":   np.median(OSI),
                "std_OSI":      np.std(OSI),
                "q25_OSI":      np.percentile(OSI, 25),
                "q75_OSI":      np.percentile(OSI, 75),
                "mean_DSI":     np.mean(DSI),
                "median_DSI":   np.median(DSI),
                "std_DSI":      np.std(DSI),
                "q25_DSI":      np.percentile(DSI, 25),
                "q75_DSI":      np.percentile(DSI, 75),
                "n_tuned":      int((OSI >= OSI_THRESH).sum()),
            }
            writer.writerow(row)

        csv_fh.flush()
        f.close()
        print(f"{time.time()-t0:.0f}s")

    except Exception as e:
        print(f"ERROR: {e}")

csv_fh.close()
print(f"\nDone. Total time: {(time.time()-t_total)/60:.1f} min")
print(f"Results saved → {OUT_CSV}")
