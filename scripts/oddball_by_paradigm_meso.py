"""Mesoscope oddball mismatch responses — all 4 paradigms.

Processes all 42 sessions in dandiset 001768, detects paradigm, extracts
dF/F population PSTHs (all soma ROIs pooled across 8 planes), and computes
mismatch index = deviant − standard response in the response window.

Control block pairing matches ecephys:
  SEQUENCE     → control_sequential
  STANDARD     → control_standard
  SENSORYMOTOR → control_replay
  DURATION     → control_standard

Results cached per session to results/oddball_cache_meso/ for resume.
Produces:
  meso_oddball_psth.png          — 4-panel PSTH grid
  meso_oddball_mismatch_index.png — mismatch magnitude bar chart

Usage:
    python scripts/oddball_by_paradigm_meso.py
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
DANDISET_ID = "001768"
PLANES      = ["VISp_0","VISp_1","VISp_2","VISp_3",
               "VISl_4","VISl_5","VISl_6","VISl_7"]
ODD_WINDOW  = (-0.5, 1.0)
ODD_BIN     = 0.05           # 50 ms — coarser than ephys, appropriate for dF/F
RESP_WIN    = (0.1,  0.8)    # wider than ephys due to calcium decay
BL_WIN      = (-0.25, 0.0)
N_STD_MAX   = 300
CACHE_DIR   = "results/oddball_cache_meso"

CTRL_BLOCK_KIND = {
    "SEQUENCE":     "control_sequential",
    "STANDARD":     "control_standard",
    "SENSORYMOTOR": "control_replay",
    "DURATION":     "control_standard",
}
PARADIGM_LABELS = {
    "SEQUENCE":     "Sequence mismatch",
    "STANDARD":     "Standard mismatch",
    "SENSORYMOTOR": "Sensory-motor mismatch",
    "DURATION":     "Duration mismatch",
}
DEV_PALETTE = ["#D65F5F","#E07B39","#6ACC65","#9B59B6",
               "#C4AD66","#77BEDB","#E84393","#17BECF"]

os.makedirs(CACHE_DIR, exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────
def extract_snippets(h5, plane, onsets, window):
    """Extract dF/F snippets for one plane via np.interp.
    Returns (n_trials, n_soma, n_timepoints) array or None if plane unavailable."""
    try:
        base = f"processing/{plane}"
        ts   = h5[f"{base}/dff_timeseries/dff_timeseries/timestamps"][:]
        data = h5[f"{base}/dff_timeseries/dff_timeseries/data"]
        is_soma = h5[f"{base}/image_segmentation/roi_table/is_soma"][:].astype(bool)
    except Exception as e:
        return None, None

    pre, post = window
    dt     = float(np.median(np.diff(ts[len(ts)//2 : len(ts)//2 + 200])))
    t_rel  = np.arange(pre, post + dt, dt)
    n_samp = len(t_rel)

    valid  = (onsets + pre >= ts[0]) & (onsets + post <= ts[-1])
    if valid.sum() < 5:
        return None, None
    good_onsets = onsets[valid]

    i0 = max(0, int(np.searchsorted(ts, good_onsets.min() + pre - 2.0)))
    i1 = min(data.shape[0], int(np.searchsorted(ts, good_onsets.max() + post + 2.0)) + 1)
    trace      = data[i0:i1, :].astype(np.float32)
    ts_span    = ts[i0:i1]
    trace_soma = trace[:, is_soma]
    n_soma     = trace_soma.shape[1]
    if n_soma == 0:
        return None, None

    t_query = good_onsets[:, None] + t_rel[None, :]
    snip = np.full((len(good_onsets), n_soma, n_samp), np.nan, dtype=np.float32)
    for roi in range(n_soma):
        y = trace_soma[:, roi]
        fin = np.isfinite(y)
        if fin.sum() < 10:
            continue
        vals = np.interp(t_query.ravel(), ts_span[fin], y[fin],
                         left=np.nan, right=np.nan)
        snip[:, roi, :] = vals.reshape(len(good_onsets), n_samp)

    return snip, t_rel   # valid-trial subset only; caller must handle

def collect_plane_snippets(h5, onsets, window):
    """Pool soma ROIs across all available planes. Returns (n_trials, n_rois, n_t)."""
    all_snips = []
    t_rel_out = None
    for plane in PLANES:
        snip, t_rel = extract_snippets(h5, plane, onsets, window)
        if snip is not None:
            all_snips.append(snip)
            if t_rel_out is None:
                t_rel_out = t_rel
    if not all_snips:
        return None, None
    # Align trial counts (take minimum valid set — should all be same since onsets fixed)
    min_trials = min(s.shape[0] for s in all_snips)
    all_snips  = [s[:min_trials] for s in all_snips]
    return np.concatenate(all_snips, axis=1), t_rel_out

def pop_psth(arr, mu_z, sig_z):
    """z-score and compute population mean ± SEM PSTH."""
    z = (arr - mu_z) / sig_z
    per_trial = np.nanmean(z, axis=1)   # (trial, time)
    m = np.nanmean(per_trial, axis=0)
    s = np.nanstd(per_trial, axis=0) / np.sqrt(per_trial.shape[0])
    return m, s

# ── Load assets ───────────────────────────────────────────────────────
print(f"Fetching asset list for dandiset {DANDISET_ID}…")
client = DandiAPIClient()
assets = sorted(list(client.get_dandiset(DANDISET_ID).get_assets()), key=lambda a: a.size)
print(f"  {len(assets)} assets")

paradigm_data = defaultdict(list)
paradigm_mi   = defaultdict(list)

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
        paradigm_data[cached["paradigm"]].append(cached["session_entry"])
        for e in cached["mi_entries"]:
            paradigm_mi[cached["paradigm"]].append(e)
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

        paradigms = odd_trials["paradigm"].dropna().unique()
        if len(paradigms) == 0:
            print("  Paradigm not identified — skip")
            handle.close(); continue
        paradigm = paradigms[0]

        ctrl_kind   = CTRL_BLOCK_KIND.get(paradigm, "control_sequential")
        ctrl_trials = trials[trials["block_kind"] == ctrl_kind]
        if len(ctrl_trials) == 0:
            print(f"  No control block ({ctrl_kind}) — skip")
            handle.close(); continue

        print(f"  Paradigm: {paradigm}  ({len(odd_trials)} trials)")
        print(f"  Control: {ctrl_kind}  ({len(ctrl_trials)} trials)")

        # Group trial types
        trial_types = sorted(odd_trials["trial_type"].unique())
        std_rows = odd_trials[odd_trials["trial_type"] == "standard"]
        dev_rows = {tt: odd_trials[odd_trials["trial_type"] == tt]
                    for tt in trial_types if tt != "standard"}

        print(f"  Trial types: standard({len(std_rows)})",
              " ".join(f"{tt}({len(r)})" for tt, r in dev_rows.items()))

        if len(std_rows) > N_STD_MAX:
            std_rows = std_rows.sample(N_STD_MAX, random_state=42)
        if len(ctrl_trials) > N_STD_MAX:
            ctrl_trials = ctrl_trials.sample(N_STD_MAX, random_state=42)

        # Standard snippets
        print("  Extracting standard snippets…", end=" ", flush=True)
        std_arr, centers = collect_plane_snippets(
            h5, std_rows["start_time"].values, ODD_WINDOW)
        if std_arr is None:
            print("no planes — skip")
            handle.close(); continue
        print(f"{std_arr.shape[1]} ROIs  {time.time()-t0:.0f}s")

        bl_m  = (centers >= BL_WIN[0])   & (centers < BL_WIN[1])
        rsp_m = (centers >= RESP_WIN[0]) & (centers < RESP_WIN[1])
        mu_z  = np.nanmean(std_arr[:, :, bl_m], axis=(0, 2), keepdims=True)
        sig_z = np.nanstd( std_arr[:, :, bl_m], axis=(0, 2), keepdims=True)
        sig_z = np.where(sig_z > 1e-6, sig_z, 1.0)

        session_psths = {"standard": pop_psth(std_arr, mu_z, sig_z)}
        std_resp_val  = np.nanmean((std_arr - mu_z) / sig_z * (centers[None,None,:] >= RESP_WIN[0]) * (centers[None,None,:] < RESP_WIN[1]))

        mi_entries = []
        for tt, rows in dev_rows.items():
            if len(rows) < 5:
                continue
            print(f"  Binning {tt} ({len(rows)} trials)…", end=" ", flush=True)
            arr, _ = collect_plane_snippets(h5, rows["start_time"].values, ODD_WINDOW)
            if arr is None:
                print("skip")
                continue
            session_psths[tt] = pop_psth(arr, mu_z, sig_z)
            dev_resp_val = np.nanmean(((arr - mu_z) / sig_z)[:, :, rsp_m])
            std_resp_win = np.nanmean(((std_arr - mu_z) / sig_z)[:, :, rsp_m])
            mi = float(dev_resp_val - std_resp_win)
            mi_entries.append({"trial_type": tt, "mismatch_index": mi,
                                "session": asset_id[:8]})
            paradigm_mi[paradigm].append(mi_entries[-1])
            print(f"MI={mi:.3f}")

        # Control
        print(f"  Extracting control ({len(ctrl_trials)} trials)…", end=" ", flush=True)
        ctrl_arr, _ = collect_plane_snippets(h5, ctrl_trials["start_time"].values, ODD_WINDOW)
        if ctrl_arr is not None:
            session_psths["control"] = pop_psth(ctrl_arr, mu_z, sig_z)
        print(f"{time.time()-t0:.0f}s total")

        session_entry = {
            "psths": session_psths, "centers": centers,
            "trial_types": list(dev_rows.keys()),
            "n_rois": std_arr.shape[1], "session": asset_id[:8],
        }
        paradigm_data[paradigm].append(session_entry)

        with open(cache_path, "wb") as fh:
            pickle.dump({"paradigm": paradigm, "session_entry": session_entry,
                         "mi_entries": mi_entries}, fh)
        print(f"  Cached → {cache_path}")
        handle.close()

    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"  ERROR: {e}")

print(f"\nProcessed in {(time.time()-t_total)/60:.1f} min")
print("Paradigms found:", list(paradigm_data.keys()))

# ── Figures ───────────────────────────────────────────────────────────
PARADIGM_ORDER = ["SEQUENCE","STANDARD","SENSORYMOTOR","DURATION"]
present = [p for p in PARADIGM_ORDER if p in paradigm_data]

# Common time axis — mesoscope sessions may have slightly different frame rates
COMMON_T = np.linspace(ODD_WINDOW[0], ODD_WINDOW[1], 300)

def interp_psth(session, tt):
    if tt not in session["psths"]: return None
    m, _ = session["psths"][tt]
    return np.interp(COMMON_T, session["centers"], m)

fig, axes = plt.subplots(1, len(present), figsize=(5 * len(present), 5), sharey=False)
if len(present) == 1: axes = [axes]

for ax, paradigm in zip(axes, present):
    sessions = paradigm_data[paradigm]
    all_tts  = sorted(set(tt for s in sessions for tt in s["trial_types"]))

    ax.axvspan(0, 0.267, color="gray", alpha=0.10)
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.axvline(0, color="gray", lw=0.8, ls=":")

    std_means = np.array([interp_psth(s, "standard") for s in sessions
                          if interp_psth(s, "standard") is not None])
    if len(std_means):
        m = std_means.mean(axis=0)
        se = std_means.std(axis=0) / np.sqrt(len(std_means))
        ax.fill_between(COMMON_T, m-se, m+se, color="#4878CF", alpha=0.20)
        ax.plot(COMMON_T, m, color="#4878CF", lw=2.5,
                label=f"standard (n={len(sessions)} sess)")

    ctrl_means = np.array([interp_psth(s, "control") for s in sessions
                           if interp_psth(s, "control") is not None])
    if len(ctrl_means):
        ax.plot(COMMON_T, ctrl_means.mean(axis=0), color="#888888",
                lw=1.5, ls="--", alpha=0.8, label="control block")

    for ci, tt in enumerate(all_tts):
        color = DEV_PALETTE[ci % len(DEV_PALETTE)]
        dev_means = np.array([interp_psth(s, tt) for s in sessions
                              if interp_psth(s, tt) is not None])
        if len(dev_means) == 0: continue
        m  = dev_means.mean(axis=0)
        se = dev_means.std(axis=0) / np.sqrt(len(dev_means))
        ax.fill_between(COMMON_T, m-se, m+se, color=color, alpha=0.15)
        ax.plot(COMMON_T, m, color=color, lw=2, label=f"{tt} (n={len(dev_means)})")

    ax.set(xlabel="Time from onset (s)", ylabel="z-score",
           title=f"{PARADIGM_LABELS.get(paradigm, paradigm)}\n{len(sessions)} session(s)",
           xlim=ODD_WINDOW)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.8)
    ax.spines[["top","right"]].set_visible(False)

fig.suptitle("Mesoscope oddball responses — 4 paradigms\n"
             "Soma ROIs pooled across 8 planes · z-scored to standard baseline · ±SEM across sessions",
             fontsize=11, fontweight="bold")
fig.tight_layout()
fig.savefig("meso_oddball_psth.png", dpi=150, bbox_inches="tight")
print("Saved → meso_oddball_psth.png")

# Mismatch index bar
fig, ax = plt.subplots(figsize=(10, 5))
x_pos = 0
xtick_pos, xtick_lab = [], []

for pi, paradigm in enumerate(present):
    tt_map = defaultdict(list)
    for d in paradigm_mi[paradigm]:
        tt_map[d["trial_type"]].append(d["mismatch_index"])
    for ci, (tt, vals) in enumerate(sorted(tt_map.items())):
        color = DEV_PALETTE[ci % len(DEV_PALETTE)]
        mn = np.mean(vals)
        sem = np.std(vals)/np.sqrt(len(vals)) if len(vals) > 1 else 0
        ax.bar(x_pos, mn, yerr=sem, color=color, width=0.7, alpha=0.8, capsize=4)
        ax.scatter([x_pos]*len(vals), vals, color="k", s=20, alpha=0.7, zorder=3)
        xtick_pos.append(x_pos); xtick_lab.append(f"{tt}\n({PARADIGM_LABELS.get(paradigm,'')[:6]}…)")
        x_pos += 1
    group_start = xtick_pos[-len(tt_map)]; group_end = xtick_pos[-1]
    ax.annotate(PARADIGM_LABELS.get(paradigm, paradigm),
                xy=((group_start+group_end)/2, 0), xytext=(0,-55),
                textcoords="offset points", ha="center", fontsize=8, color="gray")
    x_pos += 0.8

ax.axhline(0, color="k", lw=0.8, ls="--")
ax.set_xticks(xtick_pos)
ax.set_xticklabels(xtick_lab, fontsize=7, rotation=20, ha="right")
ax.set_ylabel("Mismatch index (deviant − standard z-score, response window)")
ax.set_title(f"Mesoscope mismatch response magnitude — all paradigms\n"
             f"Response window {RESP_WIN[0]*1000:.0f}–{RESP_WIN[1]*1000:.0f} ms · "
             f"bars = mean±SEM · dots = individual sessions",
             fontsize=10, fontweight="bold")
ax.spines[["top","right"]].set_visible(False)
fig.tight_layout()
fig.savefig("meso_oddball_mismatch_index.png", dpi=150, bbox_inches="tight")
print("Saved → meso_oddball_mismatch_index.png")

print("\nDone.")
