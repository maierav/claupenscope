"""Ecephys oddball responses — all 4 paradigms, properly controlled.

Processes all 6 sessions in dandiset 001637, groups by paradigm, and pairs
each paradigm block with its appropriate control block:

  SEQUENCE     → Control block 2 (sequential_control — spaced randomized)
  STANDARD     → Control block 1 (standard_control — spaced randomized)
  SENSORYMOTOR → Control block 4 (open_loop_prerecorded — replay)
  DURATION     → Control block 1 (standard_control — spaced randomized)

For each paradigm, computes population z-scored PSTHs for standard and all
deviant trial types, then plots:
  - oddball_paradigms_psth.png  — 4-panel PSTH grid (one per paradigm)
  - oddball_mismatch_index.png  — mismatch magnitude per paradigm × deviant type

Units: SUA + default_qc across all probes.
Mismatch index = mean deviant response − mean standard response (in RESP_WIN).

Usage:
    python scripts/oddball_by_paradigm_ecephys.py
"""
import time
import os
import sys
import numpy as np
import requests
import remfile
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from dandi.dandiapi import DandiAPIClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from openscope_pp.loaders.streaming import open_nwb
from openscope_pp.loaders.trials import load_trials

# ── Config ────────────────────────────────────────────────────────────
DANDISET_ID  = "001637"
ODD_WINDOW   = (-0.5, 1.0)    # s around trial onset
ODD_BIN      = 0.025          # 25 ms bins
RESP_WIN     = (0.05, 0.35)   # response scoring window
BL_WIN       = (-0.4, 0.0)    # baseline window
N_STD_MAX    = 400            # cap standard trials (SM has 45k+)

# Which control block to use per paradigm
CTRL_BLOCK_KIND = {
    "SEQUENCE":     "control_sequential",  # Control block 2
    "STANDARD":     "control_standard",    # Control block 1
    "SENSORYMOTOR": "control_replay",      # Control block 4
    "DURATION":     "control_standard",    # Control block 1
}

PARADIGM_LABELS = {
    "SEQUENCE":     "Sequence mismatch",
    "STANDARD":     "Standard mismatch",
    "SENSORYMOTOR": "Sensory-motor mismatch",
    "DURATION":     "Duration mismatch",
}

PARADIGM_COLORS = {
    "standard":  "#4878CF",
    "deviant":   "#D65F5F",
    "omission":  "#E07B39",
    "control":   "#888888",
}

# ── Helpers ───────────────────────────────────────────────────────────
def decode(arr):
    if arr.dtype.kind in ("S", "O"):
        return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in arr])
    return arr

def bin_spikes(uid_arr, spikes, index, onsets, window, bin_size):
    """Fast searchsorted spike binning → (n_trials, n_units, n_bins) firing rates."""
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

def pop_psth(arr, centers):
    """Population mean PSTH: average over units then trials → (n_bins,)."""
    per_unit = arr.mean(axis=2)   # (trial, unit) → mean over time? No.
    # arr: (trial, unit, time) → mean over units first → (trial, time)
    per_trial = arr.mean(axis=1)  # (trial, time)
    m = per_trial.mean(axis=0)
    s = per_trial.std(axis=0) / np.sqrt(per_trial.shape[0])
    return m, s

# ── Load asset list ───────────────────────────────────────────────────
print(f"Fetching asset list for dandiset {DANDISET_ID}…")
client = DandiAPIClient()
ds     = client.get_dandiset(DANDISET_ID)
assets = sorted(list(ds.get_assets()), key=lambda a: a.size)
print(f"  {len(assets)} assets")

# ── Collect per-paradigm PSTHs across sessions ────────────────────────
# paradigm_data[paradigm] = list of dicts with {trial_type: (means, sems, centers)}
paradigm_data  = defaultdict(list)
paradigm_mi    = defaultdict(list)   # mismatch indices per session

t_total = time.time()
for si, asset in enumerate(assets):
    asset_id   = asset.identifier
    subject_id = asset.path.split("/")[0]
    print(f"\n[{si+1}/{len(assets)}] {subject_id}  {asset_id[:8]}…")
    t0 = time.time()

    try:
        handle = open_nwb(asset_id)
        trials = load_trials(handle)
        h5     = handle.h5

        # ── Identify paradigm ─────────────────────────────────────────
        odd_trials = trials[trials["block_kind"] == "paradigm_oddball"]
        if len(odd_trials) == 0:
            print("  No paradigm_oddball block — skip")
            handle.close(); continue

        paradigms = odd_trials["paradigm"].dropna().unique()
        if len(paradigms) == 0:
            print("  Paradigm not identified — skip")
            handle.close(); continue
        paradigm = paradigms[0]
        print(f"  Paradigm: {paradigm}  ({len(odd_trials)} trials)")

        # ── Identify appropriate control block ────────────────────────
        ctrl_kind  = CTRL_BLOCK_KIND.get(paradigm, "control_sequential")
        ctrl_trials = trials[trials["block_kind"] == ctrl_kind]
        if len(ctrl_trials) == 0:
            print(f"  No control block ({ctrl_kind}) — skip")
            handle.close(); continue
        print(f"  Control: {ctrl_kind}  ({len(ctrl_trials)} trials)")

        # ── Load units ────────────────────────────────────────────────
        units_grp  = h5["units"]
        dl         = decode(units_grp["decoder_label"][:])
        qc         = units_grp["default_qc"][:].astype(bool)
        all_spikes = units_grp["spike_times"][:]
        spk_index  = units_grp["spike_times_index"][:]
        qual_idx   = np.where((dl == "sua") & qc)[0]
        n_qual     = len(qual_idx)
        print(f"  Units: {n_qual} SUA+QC")

        # ── Group oddball trials by type ──────────────────────────────
        trial_types_in_block = sorted(odd_trials["trial_type"].unique())
        std_rows = odd_trials[odd_trials["trial_type"] == "standard"]
        dev_rows = {tt: odd_trials[odd_trials["trial_type"] == tt]
                    for tt in trial_types_in_block if tt != "standard"}

        print(f"  Trial types: standard({len(std_rows)})",
              " ".join(f"{tt}({len(r)})" for tt, r in dev_rows.items()))

        # Cap standard trials
        if len(std_rows) > N_STD_MAX:
            std_rows = std_rows.sample(N_STD_MAX, random_state=42)

        # Cap control trials similarly
        if len(ctrl_trials) > N_STD_MAX:
            ctrl_trials = ctrl_trials.sample(N_STD_MAX, random_state=42)

        # ── Bin spikes ────────────────────────────────────────────────
        print("  Binning standard…", end=" ", flush=True)
        std_arr, centers = bin_spikes(qual_idx, all_spikes, spk_index,
                                      std_rows["start_time"].values,
                                      ODD_WINDOW, ODD_BIN)
        print(f"{time.time()-t0:.0f}s")

        # Z-score baseline from standard block
        bl_m  = (centers >= BL_WIN[0])   & (centers < BL_WIN[1])
        rsp_m = (centers >= RESP_WIN[0]) & (centers < RESP_WIN[1])
        mu_z  = std_arr[:, :, bl_m].mean(axis=(0, 2), keepdims=True)
        sig_z = std_arr[:, :, bl_m].std(axis=(0, 2),  keepdims=True)
        sig_z = np.where(sig_z > 0.1, sig_z, 1.0)

        session_psths = {}
        session_psths["standard"] = pop_psth((std_arr - mu_z) / sig_z, centers)

        # Deviants
        for tt, rows in dev_rows.items():
            if len(rows) < 5:
                continue
            print(f"  Binning {tt} ({len(rows)} trials)…", end=" ", flush=True)
            arr, _ = bin_spikes(qual_idx, all_spikes, spk_index,
                                rows["start_time"].values, ODD_WINDOW, ODD_BIN)
            z_arr = (arr - mu_z) / sig_z
            session_psths[tt] = pop_psth(z_arr, centers)
            # Mismatch index
            std_resp = ((std_arr - mu_z) / sig_z)[:, :, rsp_m].mean()
            dev_resp = z_arr[:, :, rsp_m].mean()
            paradigm_mi[paradigm].append({
                "trial_type": tt,
                "mismatch_index": float(dev_resp - std_resp),
                "session": asset_id[:8],
            })
            print(f"MI={dev_resp - std_resp:.3f}")

        # Control
        print(f"  Binning control ({len(ctrl_trials)} trials)…", end=" ", flush=True)
        ctrl_arr, _ = bin_spikes(qual_idx, all_spikes, spk_index,
                                  ctrl_trials["start_time"].values,
                                  ODD_WINDOW, ODD_BIN)
        session_psths["control"] = pop_psth((ctrl_arr - mu_z) / sig_z, centers)
        print(f"{time.time()-t0:.0f}s total")

        paradigm_data[paradigm].append({
            "psths": session_psths,
            "centers": centers,
            "trial_types": list(dev_rows.keys()),
            "n_units": n_qual,
            "session": asset_id[:8],
        })

        handle.close()

    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"  ERROR: {e}")

print(f"\nProcessed in {(time.time()-t_total)/60:.1f} min")
print("Paradigms found:", list(paradigm_data.keys()))

# ── Figure 1: 4-panel PSTH grid ───────────────────────────────────────
PARADIGM_ORDER = ["SEQUENCE", "STANDARD", "SENSORYMOTOR", "DURATION"]
present = [p for p in PARADIGM_ORDER if p in paradigm_data]
n_panels = len(present)

# Color palette for deviant types
DEV_PALETTE = [
    "#D65F5F", "#E07B39", "#6ACC65", "#9B59B6",
    "#C4AD66", "#77BEDB", "#E84393", "#17BECF",
]

fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5), sharey=False)
if n_panels == 1:
    axes = [axes]

for ax, paradigm in zip(axes, present):
    sessions = paradigm_data[paradigm]
    # Average PSTHs across sessions for each trial type
    all_tts = set()
    for s in sessions:
        all_tts.update(s["trial_types"])
    all_tts = sorted(all_tts)

    centers = sessions[0]["centers"]
    ax.axvspan(0, 0.267, color="gray", alpha=0.10)
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.axvline(0, color="gray", lw=0.8, ls=":")

    # Standard
    std_means = np.array([s["psths"]["standard"][0] for s in sessions
                           if "standard" in s["psths"]])
    if len(std_means):
        m = std_means.mean(axis=0)
        s_err = std_means.std(axis=0) / np.sqrt(len(std_means))
        ax.fill_between(centers, m - s_err, m + s_err,
                        color="#4878CF", alpha=0.20)
        ax.plot(centers, m, color="#4878CF", lw=2.5,
                label=f"standard (n={len(sessions)} sess)")

    # Control
    ctrl_means = np.array([s["psths"]["control"][0] for s in sessions
                            if "control" in s["psths"]])
    if len(ctrl_means):
        m = ctrl_means.mean(axis=0)
        ax.plot(centers, m, color="#888888", lw=1.5, ls="--",
                alpha=0.8, label="control block")

    # Deviants
    for ci, tt in enumerate(all_tts):
        color = DEV_PALETTE[ci % len(DEV_PALETTE)]
        dev_means = np.array([s["psths"][tt][0] for s in sessions
                               if tt in s["psths"]])
        if len(dev_means) == 0:
            continue
        m = dev_means.mean(axis=0)
        s_err = dev_means.std(axis=0) / np.sqrt(len(dev_means))
        ax.fill_between(centers, m - s_err, m + s_err, color=color, alpha=0.15)
        n_sess_tt = len(dev_means)
        ax.plot(centers, m, color=color, lw=2,
                label=f"{tt} (n={n_sess_tt})")

    ax.set(xlabel="Time from onset (s)", ylabel="z-score",
           title=f"{PARADIGM_LABELS[paradigm]}\n{len(sessions)} session(s)",
           xlim=ODD_WINDOW)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("Ecephys oddball responses — 4 paradigms\n"
             "SUA + default_qc · z-scored to standard baseline · shading = ±SEM across sessions",
             fontsize=11, fontweight="bold")
fig.tight_layout()
fig.savefig("oddball_paradigms_psth.png", dpi=150, bbox_inches="tight")
print("Saved → oddball_paradigms_psth.png")

# ── Figure 2: Mismatch index per paradigm × deviant type ──────────────
fig, ax = plt.subplots(figsize=(10, 5))
x_pos = 0
xtick_pos, xtick_lab = [], []

for pi, paradigm in enumerate(present):
    sessions_mi = paradigm_mi[paradigm]
    if not sessions_mi:
        continue
    # Group by trial type
    tt_map = defaultdict(list)
    for d in sessions_mi:
        tt_map[d["trial_type"]].append(d["mismatch_index"])

    for ci, (tt, vals) in enumerate(sorted(tt_map.items())):
        color = DEV_PALETTE[ci % len(DEV_PALETTE)]
        mn = np.mean(vals)
        sem = np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0
        ax.bar(x_pos, mn, yerr=sem, color=color, width=0.7,
               alpha=0.8, capsize=4, label=f"{paradigm}:{tt}" if pi == 0 else None)
        ax.scatter([x_pos] * len(vals), vals, color="k", s=20, alpha=0.7, zorder=3)
        xtick_pos.append(x_pos)
        xtick_lab.append(f"{tt}\n({PARADIGM_LABELS[paradigm][:6]}…)")
        x_pos += 1
    x_pos += 0.8   # gap between paradigms

    # Paradigm label above group
    group_start = xtick_pos[-len(tt_map)]; group_end = xtick_pos[-1]
    ax.annotate(PARADIGM_LABELS[paradigm], xy=((group_start + group_end) / 2, 0),
                xytext=(0, -55), textcoords="offset points",
                ha="center", fontsize=8, color="gray")

ax.axhline(0, color="k", lw=0.8, ls="--")
ax.set_xticks(xtick_pos)
ax.set_xticklabels(xtick_lab, fontsize=7, rotation=20, ha="right")
ax.set_ylabel("Mismatch index (deviant − standard z-score, in response window)")
ax.set_title("Ecephys mismatch response magnitude — all paradigms\n"
             f"Response window {RESP_WIN[0]*1000:.0f}–{RESP_WIN[1]*1000:.0f} ms  ·  "
             f"bars = mean ± SEM across sessions  ·  dots = individual sessions",
             fontsize=10, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("oddball_mismatch_index.png", dpi=150, bbox_inches="tight")
print("Saved → oddball_mismatch_index.png")

print("\nDone.")
