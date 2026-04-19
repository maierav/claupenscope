"""Collect ecephys orientation tuning statistics across all sessions.

Processes all 6 sessions in dandiset 001637 sequentially, computing per-probe
OSI/DSI for every SUA + default_qc unit. Saves one row per probe per session to:

    results/ephys_ori_stats.csv

Resumes from where it left off if interrupted.
Run time: ~5–10 minutes total.

Usage:
    python scripts/collect_ephys_ori_stats.py
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
DANDISET_ID  = "001637"
BLOCK_NAME   = "Control block 2_presentations"  # BlockType = sequential_control_block
ORI_WINDOW   = (-0.1, 0.5)
ORI_BIN      = 0.01
RESP_WIN     = (0.03, 0.25)
BL_WIN       = (-0.1, 0.0)
OSI_THRESH   = 0.2
OUT_CSV      = "results/ephys_ori_stats.csv"

CSV_FIELDS = [
    "asset_id", "subject_id", "session_date",
    "probe", "n_units",
    "mean_OSI", "median_OSI", "std_OSI", "q25_OSI", "q75_OSI",
    "mean_DSI", "median_DSI", "std_DSI", "q25_DSI", "q75_DSI",
    "n_tuned",
]

# ── Helpers ───────────────────────────────────────────────────────────
def decode(arr):
    if arr.dtype.kind in ("S", "O"):
        return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in arr])
    return arr

def open_asset(asset_id):
    url_r = requests.get(
        f"https://api.dandiarchive.org/api/assets/{asset_id}/download/",
        allow_redirects=True, stream=True, timeout=30,
    )
    url = url_r.url; url_r.close()
    return h5py.File(remfile.File(url), "r")

def bin_spikes(uid_arr, spikes, index, onsets, window, bin_size):
    """Fast searchsorted spike binning."""
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
    return out, centers

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

    subject_id  = asset.path.split("/")[0]                    # e.g. sub-820459
    date_part   = asset.path.split("_ses-")[1][:10] if "_ses-" in asset.path else "unknown"

    print(f"[{si+1:2d}/{len(assets)}] {subject_id}  {date_part}  …", end=" ", flush=True)
    t0 = time.time()

    try:
        f = open_asset(asset_id)

        # ── Units ─────────────────────────────────────────────────────
        units_grp  = f["units"]
        dl         = decode(units_grp["decoder_label"][:])
        qc         = units_grp["default_qc"][:].astype(bool)
        dev        = decode(units_grp["device_name"][:])
        all_spikes = units_grp["spike_times"][:]
        spk_index  = units_grp["spike_times_index"][:]

        qual_mask = (dl == "sua") & qc
        qual_idx  = np.where(qual_mask)[0]
        probes    = dev[qual_mask]
        n_qual    = len(qual_idx)

        # ── Orientation block ─────────────────────────────────────────
        g        = f["intervals"][BLOCK_NAME]
        oris_rad = np.array([float(v.decode() if isinstance(v, bytes) else v)
                             for v in g["Orientation"][:]])
        onsets   = g["start_time"][:]
        dirs_rad = np.array(sorted(np.unique(oris_rad)))

        # ── Bin spikes ────────────────────────────────────────────────
        ori_arr, centers = bin_spikes(qual_idx, all_spikes, spk_index,
                                      onsets, ORI_WINDOW, ORI_BIN)

        # ── Z-score ───────────────────────────────────────────────────
        bl_m  = (centers >= BL_WIN[0])   & (centers < BL_WIN[1])
        rsp_m = (centers >= RESP_WIN[0]) & (centers < RESP_WIN[1])
        mu_z  = ori_arr[:, :, bl_m].mean(axis=(0, 2), keepdims=True)
        sig_z = ori_arr[:, :, bl_m].std(axis=(0, 2),  keepdims=True)
        sig_z = np.where(sig_z > 0.1, sig_z, 1.0)
        z     = (ori_arr - mu_z) / sig_z

        # ── Tuning curves + OSI/DSI ───────────────────────────────────
        curves = np.full((n_qual, len(dirs_rad)), np.nan, dtype=np.float32)
        for di, d in enumerate(dirs_rad):
            m    = oris_rad == d
            resp = np.nanmean(z[m][:, :, rsp_m], axis=(0, 2))
            base = np.nanmean(z[m][:, :, bl_m ], axis=(0, 2))
            curves[:, di] = resp - base

        angles = dirs_rad
        r_sum  = np.nansum(np.abs(curves), axis=1).clip(1e-9)
        r_ori  = np.nansum(curves * np.exp(2j * angles)[None, :], axis=1)
        r_dir  = np.nansum(curves * np.exp(1j * angles)[None, :], axis=1)
        OSI    = (np.abs(r_ori) / r_sum).real
        DSI    = (np.abs(r_dir) / r_sum).real

        # ── Write per-probe rows ──────────────────────────────────────
        for probe_name in sorted(set(probes)):
            pm = probes == probe_name
            o  = OSI[pm]; d_ = DSI[pm]
            row = {
                "asset_id":    asset_id,
                "subject_id":  subject_id,
                "session_date": date_part,
                "probe":       probe_name,
                "n_units":     int(pm.sum()),
                "mean_OSI":    float(np.mean(o)),
                "median_OSI":  float(np.median(o)),
                "std_OSI":     float(np.std(o)),
                "q25_OSI":     float(np.percentile(o, 25)),
                "q75_OSI":     float(np.percentile(o, 75)),
                "mean_DSI":    float(np.mean(d_)),
                "median_DSI":  float(np.median(d_)),
                "std_DSI":     float(np.std(d_)),
                "q25_DSI":     float(np.percentile(d_, 25)),
                "q75_DSI":     float(np.percentile(d_, 75)),
                "n_tuned":     int((o >= OSI_THRESH).sum()),
            }
            writer.writerow(row)

        csv_fh.flush()
        f.close()
        print(f"{n_qual} units  {time.time()-t0:.0f}s")

    except Exception as e:
        print(f"ERROR: {e}")

csv_fh.close()
print(f"\nDone. Total time: {(time.time()-t_total)/60:.1f} min")
print(f"Results saved → {OUT_CSV}")
