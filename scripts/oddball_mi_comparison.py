"""Cross-technique mismatch index comparison.

Aggregates MI data from all three technique caches and produces:
  results/oddball_mi_all.csv          — tidy long-format MI table
  mi_cross_technique.png              — 3-panel figure per technique
  mi_stats_summary.csv               — t-test results (one-sample vs 0)

Usage:
    python scripts/oddball_mi_comparison.py
"""
import os, sys, pickle
import numpy as np
import pandas as pd
from scipy import stats
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# ── Load caches ───────────────────────────────────────────────────────
rows = []

CACHE_E = "results/oddball_cache"
for f in sorted(os.listdir(CACHE_E)):
    if not f.endswith(".pkl"): continue
    with open(os.path.join(CACHE_E, f), "rb") as fh:
        c = pickle.load(fh)
    for e in c["mi_entries"]:
        rows.append(dict(technique="ecephys", paradigm=c["paradigm"],
                         trial_type=e["trial_type"], mi=e["mismatch_index"],
                         session=e["session"], dmd=None))

CACHE_M = "results/oddball_cache_meso"
for f in sorted(os.listdir(CACHE_M)):
    if not f.endswith(".pkl"): continue
    with open(os.path.join(CACHE_M, f), "rb") as fh:
        c = pickle.load(fh)
    for e in c["mi_entries"]:
        rows.append(dict(technique="mesoscope", paradigm=c["paradigm"],
                         trial_type=e["trial_type"], mi=e["mismatch_index"],
                         session=e["session"], dmd=None))

CACHE_S = "results/oddball_cache_slap2"
for f in sorted(os.listdir(CACHE_S)):
    if not f.endswith(".pkl"): continue
    with open(os.path.join(CACHE_S, f), "rb") as fh:
        c = pickle.load(fh)
    for dmd, entries in c["mi_entries"].items():
        for e in entries:
            rows.append(dict(technique="slap2", paradigm="SLAP2",
                             trial_type=e["trial_type"], mi=e["mismatch_index"],
                             session=e["session"], dmd=dmd))

df = pd.DataFrame(rows)
os.makedirs("results", exist_ok=True)
df.to_csv("results/oddball_mi_all.csv", index=False)
print(f"Saved {len(df)} rows → results/oddball_mi_all.csv")
print(df.groupby(["technique","paradigm","trial_type"])["mi"].count().to_string())

# ── One-sample t-tests (MI vs 0) ─────────────────────────────────────
stat_rows = []
for (tech, par, tt), grp in df.groupby(["technique","paradigm","trial_type"]):
    vals = grp["mi"].values
    n = len(vals)
    mn = vals.mean(); se = vals.std() / np.sqrt(n) if n > 1 else np.nan
    if n >= 3:
        t, p = stats.ttest_1samp(vals, 0)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    else:
        t, p, sig = np.nan, np.nan, "n.s.(low n)"
    stat_rows.append(dict(technique=tech, paradigm=par, trial_type=tt,
                          n=n, mean_mi=mn, sem_mi=se, t=t, p=p, sig=sig))

stat_df = pd.DataFrame(stat_rows)
stat_df.to_csv("results/mi_stats_summary.csv", index=False, float_format="%.4f")
print(f"\nStats saved → results/mi_stats_summary.csv")
print(stat_df[["technique","paradigm","trial_type","n","mean_mi","sem_mi","p","sig"]].to_string(index=False))

# ── Plotting helpers ──────────────────────────────────────────────────
PARADIGM_LABELS = {
    "SEQUENCE":     "Sequence",
    "STANDARD":     "Standard",
    "SENSORYMOTOR": "Sensory-motor",
    "DURATION":     "Duration",
    "SLAP2":        "Orientation/Omission",
}

# Consistent trial-type colors across panels
TT_COLORS = {
    "halt":             "#4878CF",
    "omission":         "#E07B39",
    "orientation_45":   "#6ACC65",
    "orientation_90":   "#9B59B6",
    "sequence_omission":"#C0392B",
    "jitter":           "#77BEDB",
    "motor_halt":       "#4878CF",
    "motor_omission":   "#E07B39",
    "motor_orientation_45": "#6ACC65",
    "motor_orientation_90": "#9B59B6",
    "0":                "#D65F5F",
    "45":               "#6ACC65",
    "90":               "#9B59B6",
    "static":           "#AAAAAA",
}

def plot_mi_panel(ax, vals_dict, sig_dict, color_map, title, ylabel=True):
    """Bar + scatter MI panel. vals_dict: {tt: [mi values]}"""
    tts = sorted(vals_dict.keys())
    for xi, tt in enumerate(tts):
        vals = np.array(vals_dict[tt])
        mn = vals.mean()
        se = vals.std() / np.sqrt(len(vals)) if len(vals) > 1 else 0
        color = color_map.get(tt, "#888888")
        ax.bar(xi, mn, yerr=se, color=color, alpha=0.80, width=0.6,
               capsize=4, error_kw={"lw": 1.5})
        jx = xi + np.random.default_rng(xi).uniform(-0.15, 0.15, len(vals))
        ax.scatter(jx, vals, color="k", s=18, alpha=0.65, zorder=3)
        # significance annotation
        sig = sig_dict.get(tt, "")
        if sig and sig not in ("ns", "n.s.(low n)"):
            ymax = mn + se + max(abs(vals.max() - mn), 0.01)
            ax.text(xi, ymax + 0.005, sig, ha="center", va="bottom",
                    fontsize=9, color="k", fontweight="bold")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xticks(range(len(tts)))
    ax.set_xticklabels([t.replace("motor_","").replace("orientation_","ori_")
                        for t in tts], fontsize=8, rotation=25, ha="right")
    ax.set_title(title, fontsize=9, fontweight="bold")
    if ylabel: ax.set_ylabel("Mismatch index\n(deviant − standard z-score)", fontsize=8)
    ax.spines[["top","right"]].set_visible(False)


# ── Figure 1: Per-technique, per-paradigm MI bars ─────────────────────
#   Rows = techniques (ecephys, meso, slap2)
#   Cols = paradigms (SEQUENCE, STANDARD, SENSORYMOTOR, DURATION, SLAP2)

PARADIGM_ORDER = ["SEQUENCE","STANDARD","SENSORYMOTOR","DURATION","SLAP2"]
TECH_ORDER     = ["ecephys","mesoscope","slap2"]
TECH_LABELS    = {"ecephys":"Ecephys (Neuropixels)",
                  "mesoscope":"Mesoscope (GCaMP)",
                  "slap2":"SLAP2"}
TECH_COLORS    = {"ecephys":"#555555","mesoscope":"#4878CF","slap2":"#D65F5F"}

# Determine which (tech, paradigm) combos exist
present = [(tech, par) for tech in TECH_ORDER for par in PARADIGM_ORDER
           if not df[(df.technique==tech) & (df.paradigm==par)].empty]

n_cols = len(PARADIGM_ORDER)
n_rows = len(TECH_ORDER)

fig, axes = plt.subplots(n_rows, n_cols,
                         figsize=(3.5 * n_cols, 3.8 * n_rows),
                         sharey="row")

for ri, tech in enumerate(TECH_ORDER):
    for ci, par in enumerate(PARADIGM_ORDER):
        ax = axes[ri, ci]
        sub = df[(df.technique == tech) & (df.paradigm == par)]
        if sub.empty:
            ax.set_visible(False); continue

        # For SLAP2 average DMD1+DMD2 per session/tt
        if tech == "slap2":
            sub = sub.groupby(["session","trial_type"])["mi"].mean().reset_index()

        by_tt = defaultdict(list)
        for _, row in sub.iterrows():
            by_tt[row["trial_type"]].append(row["mi"])

        sig_d = {}
        for tt, vals in by_tt.items():
            key = (tech, par, tt)
            match = stat_df[(stat_df.technique==tech) &
                            (stat_df.paradigm==par) &
                            (stat_df.trial_type==tt)]
            if not match.empty:
                sig_d[tt] = match.iloc[0]["sig"]

        plot_mi_panel(ax, by_tt, sig_d, TT_COLORS,
                      title=f"{PARADIGM_LABELS.get(par,par)}",
                      ylabel=(ci == 0))

        if ri == 0:
            ax.set_title(f"{PARADIGM_LABELS.get(par,par)}", fontsize=9, fontweight="bold")
        if ci == 0:
            ax.set_ylabel(f"{TECH_LABELS[tech]}\n\nMismatch index\n(deviant − standard z-score)",
                          fontsize=8)

fig.suptitle(
    "Mismatch index across techniques and paradigms\n"
    "bars = mean ± SEM · dots = individual sessions · * p<0.05  ** p<0.01  *** p<0.001 (1-sample t vs 0)",
    fontsize=11, fontweight="bold"
)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("mi_cross_technique.png", dpi=150, bbox_inches="tight")
print("\nSaved → mi_cross_technique.png")

# ── Figure 2: Omission MI across all 3 techniques (same trial type) ────
# Omissions appear in SEQUENCE, STANDARD, DURATION (meso/ephys) and SLAP2
fig2, ax2 = plt.subplots(figsize=(9, 5))

omission_groups = []   # list of (label, vals, color)

for tech, par in [("ecephys","SEQUENCE"),("ecephys","STANDARD"),("ecephys","DURATION"),
                  ("mesoscope","SEQUENCE"),("mesoscope","STANDARD"),("mesoscope","DURATION"),
                  ("slap2","SLAP2")]:
    tt = "omission" if par != "SLAP2" else "omission"
    sub = df[(df.technique==tech) & (df.paradigm==par) & (df.trial_type==tt)]
    if sub.empty: continue
    if tech == "slap2":
        vals = sub.groupby("session")["mi"].mean().values
    else:
        vals = sub["mi"].values
    label = f"{TECH_LABELS[tech]}\n{PARADIGM_LABELS.get(par,par)}"
    omission_groups.append((label, vals, TECH_COLORS[tech], par))

x_pos = np.arange(len(omission_groups))
for xi, (label, vals, color, par) in enumerate(omission_groups):
    mn = vals.mean(); se = vals.std()/np.sqrt(len(vals)) if len(vals)>1 else 0
    alpha = 0.85 if par != "SLAP2" else 0.6
    ax2.bar(xi, mn, yerr=se, color=color, alpha=alpha, width=0.6, capsize=4,
            error_kw={"lw":1.5})
    jx = xi + np.random.default_rng(xi).uniform(-0.12, 0.12, len(vals))
    ax2.scatter(jx, vals, color="k", s=22, alpha=0.7, zorder=3)
    # t-test vs 0
    if len(vals) >= 3:
        _, p = stats.ttest_1samp(vals, 0)
        sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else ""
        if sig:
            ymax = mn + se + max(abs(vals.max()-mn), 0.005)
            ax2.text(xi, ymax + 0.004, sig, ha="center", va="bottom",
                     fontsize=10, fontweight="bold")

ax2.axhline(0, color="k", lw=0.8, ls="--")
ax2.set_xticks(x_pos)
ax2.set_xticklabels([g[0] for g in omission_groups], fontsize=8)
ax2.set_ylabel("Mismatch index (omission − standard z-score)", fontsize=10)
ax2.set_title("Omission mismatch responses across techniques and paradigms\n"
              "bars = mean ± SEM · dots = individual sessions",
              fontsize=10, fontweight="bold")
ax2.spines[["top","right"]].set_visible(False)

# Technique separators
prev_tech = None
for xi, (_, _, _, _) in enumerate(omission_groups):
    tech = omission_groups[xi][2]
    if tech != prev_tech and prev_tech is not None:
        ax2.axvline(xi - 0.5, color="gray", lw=0.8, ls=":")
    prev_tech = tech

fig2.tight_layout()
fig2.savefig("mi_omission_comparison.png", dpi=150, bbox_inches="tight")
print("Saved → mi_omission_comparison.png")

print("\nDone.")
