"""SLAP2 oddball mismatch responses.

Processes all non-794237 SLAP2 sessions in dandiset 001424.
SLAP2 has a single oddball paradigm: orientation deviants (0°, 45°, 90°)
and omissions embedded in a standard stream. Two DMDs are imaged in parallel.

Trial type naming in SLAP2: "standard", "0", "45", "90", "omission"
Control: ori_tuning block (same gratings, no oddball structure)

Produces:
  slap2_oddball_psth.png            — DMD1 vs DMD2 PSTHs per deviant type
  slap2_oddball_mismatch_index.png  — mismatch index per trial type × DMD

Usage:
    python scripts/oddball_by_paradigm_slap2.py
"""
import time, os, sys, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from dandi.dandiapi import DandiAPIClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from openscope_pp.loaders.streaming import open_nwb
from openscope_pp.loaders.trials import load_trials

# ── Config ────────────────────────────────────────────────────────────
DANDISET_ID = "001424"
SKIP_SUBJECT = "sub-794237"   # older NWB format — no intervals/gratings

# DMD timing offsets (stimulus onset relative to NWB trial start_time)
DMD_OFFSETS = {"DMD1": 0.115, "DMD2": -0.165}

ODD_WINDOW  = (-0.5, 1.0)
# SLAP2 oddball ISI = 700 ms, so the response window must stay well below
# the next trial onset. iGluSnFR is fast (tens of ms), so 0.05-0.35 s
# captures the transient cleanly without tail contamination.
RESP_WIN    = (0.05, 0.35)
BL_WIN      = (-0.35, 0.0)
N_STD_MAX   = 300
CACHE_DIR   = "results/oddball_cache_slap2"

DMD_COLORS  = {"DMD1": "#4878CF", "DMD2": "#D65F5F"}
DEV_PALETTE = ["#D65F5F","#E07B39","#6ACC65","#9B59B6","#C4AD66","#77BEDB"]

os.makedirs(CACHE_DIR, exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────
def extract_dmd_snippets(h5, dmd, onsets, window):
    """Extract dF/F snippets for one DMD via np.interp.
    Returns (n_trials, n_rois, n_t) or (None, None)."""
    try:
        grp  = h5[f"processing/ophys/Fluorescence_{dmd}/{dmd}_dFF"]
        ts   = grp["timestamps"][:]
        data = grp["data"]   # (T, N_rois)
    except Exception:
        return None, None

    offset   = DMD_OFFSETS[dmd]
    adj_ons  = onsets + offset      # correct for DMD timing
    pre, post = window

    valid = (adj_ons + pre >= ts[0]) & (adj_ons + post <= ts[-1])
    if valid.sum() < 5:
        return None, None
    good_ons = adj_ons[valid]

    dt     = float(np.median(np.diff(ts[len(ts)//2 : len(ts)//2 + 200])))
    t_rel  = np.arange(pre, post + dt, dt)
    n_samp = len(t_rel)
    n_rois = data.shape[1]

    i0 = max(0, int(np.searchsorted(ts, good_ons.min() + pre - 2.0)))
    i1 = min(data.shape[0], int(np.searchsorted(ts, good_ons.max() + post + 2.0)) + 1)
    trace   = data[i0:i1, :].astype(np.float32)
    ts_span = ts[i0:i1]

    t_query = good_ons[:, None] + t_rel[None, :]
    snip = np.full((len(good_ons), n_rois, n_samp), np.nan, dtype=np.float32)
    for roi in range(n_rois):
        y = trace[:, roi]; fin = np.isfinite(y)
        if fin.sum() < 10: continue
        vals = np.interp(t_query.ravel(), ts_span[fin], y[fin],
                         left=np.nan, right=np.nan)
        snip[:, roi, :] = vals.reshape(len(good_ons), n_samp)

    return snip, t_rel

def pop_psth(arr, mu_z, sig_z):
    z = (arr - mu_z) / sig_z
    per_trial = np.nanmean(z, axis=1)
    return np.nanmean(per_trial, axis=0), np.nanstd(per_trial, axis=0) / np.sqrt(per_trial.shape[0])

# ── Load assets ───────────────────────────────────────────────────────
print(f"Fetching asset list for dandiset {DANDISET_ID}…")
client = DandiAPIClient()
assets = sorted(list(client.get_dandiset(DANDISET_ID).get_assets()), key=lambda a: a.size)
assets = [a for a in assets if SKIP_SUBJECT not in a.path]
print(f"  {len(assets)} usable assets")

# dmd_data[dmd] = list of session dicts
dmd_data = defaultdict(list)
dmd_mi   = defaultdict(list)   # dmd_mi[dmd] = list of {trial_type, mismatch_index, session}

t_total = time.time()
for si, asset in enumerate(assets):
    asset_id   = asset.identifier
    subject_id = asset.path.split("/")[0]
    cache_path = os.path.join(CACHE_DIR, f"{asset_id[:8]}.pkl")
    print(f"\n[{si+1}/{len(assets)}] {subject_id}  {asset_id[:8]}…")

    if os.path.exists(cache_path):
        print("  CACHED")
        with open(cache_path, "rb") as fh:
            cached = pickle.load(fh)
        for dmd, entry in cached["dmd_entries"].items():
            dmd_data[dmd].append(entry)
        for dmd, mi_list in cached["mi_entries"].items():
            for e in mi_list:
                dmd_mi[dmd].append(e)
        continue

    t0 = time.time()
    try:
        handle = open_nwb(asset_id)
        trials = load_trials(handle)
        h5     = handle.h5

        odd_trials = trials[trials["block_kind"] == "paradigm_oddball"]
        if len(odd_trials) == 0:
            print("  No paradigm_oddball — skip")
            handle.close(); continue

        ctrl_trials = trials[trials["block_kind"] == "ori_tuning"]

        trial_types = sorted(odd_trials["trial_type"].unique())
        std_rows = odd_trials[odd_trials["trial_type"] == "standard"]
        dev_rows = {tt: odd_trials[odd_trials["trial_type"] == tt]
                    for tt in trial_types if tt != "standard"}

        print(f"  Trials: standard({len(std_rows)})",
              " ".join(f"{tt}({len(r)})" for tt, r in dev_rows.items()))
        print(f"  Control (ori_tuning): {len(ctrl_trials)} trials")

        if len(std_rows) > N_STD_MAX:
            std_rows = std_rows.sample(N_STD_MAX, random_state=42)
        if len(ctrl_trials) > N_STD_MAX:
            ctrl_trials = ctrl_trials.sample(N_STD_MAX, random_state=42)

        session_dmd_entries = {}
        session_mi_entries  = defaultdict(list)

        for dmd in ["DMD1", "DMD2"]:
            print(f"  [{dmd}] standard…", end=" ", flush=True)
            std_arr, centers = extract_dmd_snippets(
                h5, dmd, std_rows["start_time"].values, ODD_WINDOW)
            if std_arr is None:
                print("no data")
                continue
            print(f"{std_arr.shape[1]} ROIs  {time.time()-t0:.0f}s")

            bl_m  = (centers >= BL_WIN[0])   & (centers < BL_WIN[1])
            rsp_m = (centers >= RESP_WIN[0]) & (centers < RESP_WIN[1])
            mu_z  = np.nanmean(std_arr[:, :, bl_m], axis=(0, 2), keepdims=True)
            sig_z = np.nanstd( std_arr[:, :, bl_m], axis=(0, 2), keepdims=True)
            sig_z = np.where(sig_z > 1e-6, sig_z, 1.0)

            session_psths = {"standard": pop_psth(std_arr, mu_z, sig_z)}
            std_resp_win  = np.nanmean(((std_arr - mu_z) / sig_z)[:, :, rsp_m])

            for tt, rows in dev_rows.items():
                if len(rows) < 5: continue
                print(f"  [{dmd}] {tt} ({len(rows)})…", end=" ", flush=True)
                arr, _ = extract_dmd_snippets(
                    h5, dmd, rows["start_time"].values, ODD_WINDOW)
                if arr is None: print("skip"); continue
                session_psths[tt] = pop_psth(arr, mu_z, sig_z)
                mi = float(np.nanmean(((arr - mu_z)/sig_z)[:,:,rsp_m]) - std_resp_win)
                session_mi_entries[dmd].append(
                    {"trial_type": tt, "mismatch_index": mi, "session": asset_id[:8]})
                dmd_mi[dmd].append(session_mi_entries[dmd][-1])
                print(f"MI={mi:.3f}")

            # Control
            if len(ctrl_trials) > 0:
                ctrl_arr, _ = extract_dmd_snippets(
                    h5, dmd, ctrl_trials["start_time"].values, ODD_WINDOW)
                if ctrl_arr is not None:
                    session_psths["control"] = pop_psth(ctrl_arr, mu_z, sig_z)

            entry = {"psths": session_psths, "centers": centers,
                     "trial_types": list(dev_rows.keys()),
                     "n_rois": std_arr.shape[1], "session": asset_id[:8]}
            dmd_data[dmd].append(entry)
            session_dmd_entries[dmd] = entry

        print(f"  {time.time()-t0:.0f}s total")
        with open(cache_path, "wb") as fh:
            pickle.dump({"dmd_entries": session_dmd_entries,
                         "mi_entries": dict(session_mi_entries)}, fh)
        print(f"  Cached → {cache_path}")
        handle.close()

    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"  ERROR: {e}")

print(f"\nProcessed in {(time.time()-t_total)/60:.1f} min")
print("DMDs found:", list(dmd_data.keys()))

# ── Figures ───────────────────────────────────────────────────────────
dmds = [d for d in ["DMD1","DMD2"] if d in dmd_data]
all_tts = sorted(set(tt for dmd in dmds for s in dmd_data[dmd]
                     for tt in s["trial_types"]))

# Common time axis — sessions may have slightly different frame rates
COMMON_T = np.linspace(ODD_WINDOW[0], ODD_WINDOW[1], 300)

def interp_psth(session, tt):
    """Interpolate a session PSTH to COMMON_T, return mean only."""
    if tt not in session["psths"]: return None
    m, _ = session["psths"][tt]
    return np.interp(COMMON_T, session["centers"], m)

fig, axes = plt.subplots(len(dmds), len(all_tts),
                         figsize=(4*len(all_tts), 4*len(dmds)), sharey="row")
if len(dmds) == 1: axes = axes[np.newaxis, :]
if len(all_tts) == 1: axes = axes[:, np.newaxis]

for ri, dmd in enumerate(dmds):
    sessions = dmd_data[dmd]
    for ci, tt in enumerate(all_tts):
        ax = axes[ri, ci]
        ax.axvspan(0, 0.267, color="gray", alpha=0.08)
        ax.axhline(0, color="gray", lw=0.5, ls="--")
        ax.axvline(0, color="gray", lw=0.8, ls=":")

        std_means = np.array([interp_psth(s, "standard") for s in sessions
                              if interp_psth(s, "standard") is not None])
        if len(std_means):
            m = std_means.mean(0); se = std_means.std(0)/np.sqrt(len(std_means))
            ax.fill_between(COMMON_T, m-se, m+se, color="#4878CF", alpha=0.20)
            ax.plot(COMMON_T, m, color="#4878CF", lw=2, label="standard")

        dev_means = np.array([interp_psth(s, tt) for s in sessions
                              if interp_psth(s, tt) is not None])
        if len(dev_means):
            m = dev_means.mean(0); se = dev_means.std(0)/np.sqrt(len(dev_means))
            ax.fill_between(COMMON_T, m-se, m+se, color=DMD_COLORS[dmd], alpha=0.20)
            ax.plot(COMMON_T, m, color=DMD_COLORS[dmd], lw=2, label=f"{tt} (n={len(dev_means)})")

        ctrl_means = np.array([interp_psth(s, "control") for s in sessions
                               if interp_psth(s, "control") is not None])
        if len(ctrl_means):
            ax.plot(COMMON_T, ctrl_means.mean(0), color="#888888", lw=1.2, ls="--", alpha=0.7, label="ori_tuning")

        ax.set(xlabel="Time from onset (s)", xlim=ODD_WINDOW,
               title=f"{dmd} — deviant: {tt}")
        if ci == 0: ax.set_ylabel("z-score")
        ax.legend(fontsize=7); ax.spines[["top","right"]].set_visible(False)

fig.suptitle("SLAP2 oddball responses\ndF/F z-scored to standard baseline · ±SEM across sessions",
             fontsize=11, fontweight="bold")
fig.tight_layout()
fig.savefig("slap2_oddball_psth.png", dpi=150, bbox_inches="tight")
print("Saved → slap2_oddball_psth.png")

# Mismatch index
fig, axes = plt.subplots(1, len(dmds), figsize=(5*len(dmds), 5), sharey=True)
if len(dmds) == 1: axes = [axes]

for ax, dmd in zip(axes, dmds):
    tt_map = defaultdict(list)
    for d in dmd_mi[dmd]:
        tt_map[d["trial_type"]].append(d["mismatch_index"])

    tts = sorted(tt_map.keys())
    for xi, tt in enumerate(tts):
        vals = tt_map[tt]
        mn = np.mean(vals); sem = np.std(vals)/np.sqrt(len(vals)) if len(vals)>1 else 0
        ax.bar(xi, mn, yerr=sem, color=DMD_COLORS[dmd], alpha=0.8, width=0.6, capsize=4)
        ax.scatter([xi]*len(vals), vals, color="k", s=25, alpha=0.7, zorder=3)

    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xticks(range(len(tts))); ax.set_xticklabels(tts, fontsize=9)
    ax.set_title(dmd); ax.set_ylabel("Mismatch index")
    ax.spines[["top","right"]].set_visible(False)

fig.suptitle(f"SLAP2 mismatch index per deviant type\n"
             f"Response window {RESP_WIN[0]*1000:.0f}–{RESP_WIN[1]*1000:.0f} ms · "
             f"bars = mean±SEM · dots = individual sessions",
             fontsize=10, fontweight="bold")
fig.tight_layout()
fig.savefig("slap2_oddball_mismatch_index.png", dpi=150, bbox_inches="tight")
print("Saved → slap2_oddball_mismatch_index.png")

print("\nDone.")
