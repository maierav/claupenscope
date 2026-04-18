"""Plot cross-session mesoscope orientation tuning statistics.

Reads results/meso_ori_stats.csv produced by collect_meso_ori_stats.py.
Produces four figures:

  meso_ori_pop_area.png        — VISp vs VISl violin plots (OSI & DSI),
                                  mean and median overlaid, all sessions pooled
  meso_ori_pop_subjects.png    — per-subject box plots, area split
  meso_ori_pop_sessions.png    — per-session trajectories per subject
                                  (intra-subject consistency across sessions)
  meso_ori_pop_summary.png     — summary table: mean ± SEM and median [IQR]
                                  per area, per subject, and overall
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

IN_CSV = "results/meso_ori_stats.csv"

# ── Load & aggregate ──────────────────────────────────────────────────
if not os.path.exists(IN_CSV):
    raise FileNotFoundError(f"{IN_CSV} not found — run collect_meso_ori_stats.py first")

df = pd.read_csv(IN_CSV)
print(f"Loaded {len(df)} rows  ({df['asset_id'].nunique()} sessions, "
      f"{df['subject_id'].nunique()} subjects)")
print(df.groupby(["area"])["n_soma"].agg(["count","sum","mean"]).round(1))

# Session-level aggregates: pool planes within each area per session
sess_area = (
    df.groupby(["asset_id", "subject_id", "session_date", "area"])
    .agg(
        n_soma      = ("n_soma",      "sum"),
        mean_OSI    = ("mean_OSI",    "mean"),
        median_OSI  = ("median_OSI",  "mean"),   # mean of per-plane medians
        mean_DSI    = ("mean_DSI",    "mean"),
        median_DSI  = ("median_DSI",  "mean"),
        n_tuned     = ("n_tuned",     "sum"),
    )
    .reset_index()
)

AREA_COLORS  = {"VISp": "#4878CF", "VISl": "#D65F5F"}
SUBJ_MARKERS = {s: m for s, m in zip(sorted(df["subject_id"].unique()),
                                      ["o","s","^","D","v","P"])}

# ── Figure 1: VISp vs VISl — violin + mean/median dots, all sessions ──
metrics = [
    ("mean_OSI",   "Mean OSI"),
    ("median_OSI", "Median OSI"),
    ("mean_DSI",   "Mean DSI"),
    ("median_DSI", "Median DSI"),
]
fig, axes = plt.subplots(1, 4, figsize=(14, 5), sharey=False)

for ax, (col, label) in zip(axes, metrics):
    parts = ax.violinplot(
        [sess_area.loc[sess_area["area"]==a, col].values for a in ["VISp","VISl"]],
        positions=[1, 2], showmedians=False, showextrema=False
    )
    for pc, area in zip(parts["bodies"], ["VISp","VISl"]):
        pc.set_facecolor(AREA_COLORS[area]); pc.set_alpha(0.5)

    for xi, area in enumerate(["VISp","VISl"], start=1):
        vals = sess_area.loc[sess_area["area"]==area, col].values
        # jitter
        jx = xi + np.random.default_rng(42).uniform(-0.08, 0.08, len(vals))
        ax.scatter(jx, vals, color=AREA_COLORS[area], s=25, alpha=0.7,
                   zorder=3, edgecolors="none")
        ax.plot([xi-0.2, xi+0.2], [np.mean(vals)]*2,
                color="k", lw=2.5, zorder=4, solid_capstyle="round")
        ax.plot([xi-0.12, xi+0.12], [np.median(vals)]*2,
                color="white", lw=1.5, zorder=5, ls="--",
                solid_capstyle="round")

    ax.set_xticks([1, 2]); ax.set_xticklabels(["VISp","VISl"])
    ax.set_ylabel(label); ax.set_title(label)
    ax.spines[["top","right"]].set_visible(False)

legend_elements = [
    Patch(facecolor="none", edgecolor="k",   label="— black line = mean"),
    Patch(facecolor="none", edgecolor="gray", label="-- white line = median"),
]
axes[-1].legend(handles=legend_elements, fontsize=7, loc="upper right")
fig.suptitle(
    f"Mesoscope orientation tuning — VISp vs VISl\n"
    f"{df['asset_id'].nunique()} sessions · {df['subject_id'].nunique()} subjects · "
    f"dandiset 001768",
    fontsize=10, fontweight="bold"
)
fig.tight_layout()
fig.savefig("meso_ori_pop_area.png", dpi=150, bbox_inches="tight")
print("Saved → meso_ori_pop_area.png")

# ── Figure 2: Per-subject box plots, area split ────────────────────────
subjects = sorted(sess_area["subject_id"].unique())
n_subj   = len(subjects)
fig, axes = plt.subplots(2, 2, figsize=(max(10, n_subj * 1.6), 8))
metric_pairs = [("mean_OSI","Mean OSI"), ("median_OSI","Median OSI"),
                ("mean_DSI","Mean DSI"), ("median_DSI","Median DSI")]

for ax, (col, label) in zip(axes.ravel(), metric_pairs):
    positions = []
    tick_labels = []
    for si, subj in enumerate(subjects):
        for ai, area in enumerate(["VISp","VISl"]):
            pos = si * 3 + ai
            vals = sess_area.loc[(sess_area["subject_id"]==subj) &
                                 (sess_area["area"]==area), col].values
            if len(vals) == 0: continue
            bp = ax.boxplot(vals, positions=[pos], widths=0.6,
                            patch_artist=True, showfliers=True,
                            medianprops=dict(color="white", lw=2),
                            whiskerprops=dict(color="gray"),
                            capprops=dict(color="gray"),
                            flierprops=dict(marker=".", color="gray", ms=4))
            for patch in bp["boxes"]:
                patch.set_facecolor(AREA_COLORS[area])
                patch.set_alpha(0.7)
            positions.append(pos)

        mid = si * 3 + 0.5
        tick_labels.append((mid, subj.replace("sub-","")))

    ax.set_xticks([t for t, _ in tick_labels])
    ax.set_xticklabels([l for _, l in tick_labels], fontsize=8)
    ax.set_ylabel(label)
    ax.set_title(label)
    ax.spines[["top","right"]].set_visible(False)

    legend_patches = [Patch(facecolor=c, alpha=0.7, label=a)
                      for a, c in AREA_COLORS.items()]
    ax.legend(handles=legend_patches, fontsize=7, loc="upper right")

fig.suptitle("Mesoscope orientation tuning — per-subject breakdown\n"
             "Box = IQR · line = median · whiskers = 1.5×IQR",
             fontsize=10, fontweight="bold")
fig.tight_layout()
fig.savefig("meso_ori_pop_subjects.png", dpi=150, bbox_inches="tight")
print("Saved → meso_ori_pop_subjects.png")

# ── Figure 3: Intra-subject session trajectories ──────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

for ax, (col, label) in zip(axes.ravel(), metric_pairs):
    for subj in subjects:
        for area in ["VISp","VISl"]:
            sub_df = sess_area.loc[(sess_area["subject_id"]==subj) &
                                   (sess_area["area"]==area)].copy()
            sub_df = sub_df.sort_values("session_date").reset_index(drop=True)
            if len(sub_df) < 2: continue
            ax.plot(sub_df.index, sub_df[col],
                    color=AREA_COLORS[area], alpha=0.5, lw=1.2,
                    marker=SUBJ_MARKERS[subj], ms=5)

    # Population mean ± SEM per session index
    for area in ["VISp","VISl"]:
        area_df = sess_area[sess_area["area"]==area].copy()
        # Align by session index within subject
        area_df["sess_idx"] = area_df.groupby("subject_id").cumcount()
        grp = area_df.groupby("sess_idx")[col]
        mn  = grp.mean(); se = grp.sem(); idx = mn.index
        ax.fill_between(idx, mn-se, mn+se, color=AREA_COLORS[area], alpha=0.15)
        ax.plot(idx, mn, color=AREA_COLORS[area], lw=2.5,
                label=f"{area} mean±SEM")

    ax.set(xlabel="Session index (within subject)", ylabel=label, title=label)
    ax.legend(fontsize=7); ax.spines[["top","right"]].set_visible(False)

# Subject legend
handles = [plt.Line2D([0],[0], marker=SUBJ_MARKERS[s], color="gray",
                       ms=6, lw=0, label=s.replace("sub-",""))
           for s in subjects]
axes[0,0].legend(handles=handles, title="Subject", fontsize=6,
                 loc="upper right", ncol=2)

fig.suptitle("Mesoscope orientation tuning — intra-subject session trajectories\n"
             "Thin lines = individual subjects · thick = cross-subject mean±SEM",
             fontsize=10, fontweight="bold")
fig.tight_layout()
fig.savefig("meso_ori_pop_sessions.png", dpi=150, bbox_inches="tight")
print("Saved → meso_ori_pop_sessions.png")

# ── Figure 4: Summary table (text figure) ─────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
ax.axis("off")

col_labels = ["", "n sessions", "n soma",
              "mean OSI", "median OSI", "IQR OSI",
              "mean DSI", "median DSI", "IQR DSI"]
rows_data  = []

for area in ["VISp","VISl"]:
    for subj in ["(all)"] + subjects:
        if subj == "(all)":
            sub_df = sess_area[sess_area["area"]==area]
            label  = f"{area} — ALL"
        else:
            sub_df = sess_area[(sess_area["area"]==area) &
                               (sess_area["subject_id"]==subj)]
            label  = f"  {subj.replace('sub-','')}"
        if len(sub_df) == 0: continue

        def fmt(vals, use_sem=False):
            mn = np.mean(vals); md = np.median(vals)
            se = np.std(vals)/np.sqrt(len(vals)) if use_sem else np.std(vals)
            q1, q3 = np.percentile(vals, [25,75])
            return f"{mn:.3f}±{se:.3f}", f"{md:.3f}", f"[{q1:.3f},{q3:.3f}]"

        m_osi, med_osi, iqr_osi = fmt(sub_df["mean_OSI"].values,   use_sem=True)
        m_dsi, med_dsi, iqr_dsi = fmt(sub_df["mean_DSI"].values,   use_sem=True)
        rows_data.append([
            label,
            str(sub_df["asset_id"].nunique()),
            f"{sub_df['n_soma'].sum():,}",
            m_osi, med_osi, iqr_osi,
            m_dsi, med_dsi, iqr_dsi,
        ])
    rows_data.append([""] * len(col_labels))  # spacer

tbl = ax.table(
    cellText=rows_data,
    colLabels=col_labels,
    loc="center", cellLoc="center"
)
tbl.auto_set_font_size(False); tbl.set_fontsize(8)
tbl.auto_set_column_width(list(range(len(col_labels))))

# Colour header + area rows
for (row, col), cell in tbl.get_celld().items():
    if row == 0:
        cell.set_facecolor("#CCCCCC"); cell.set_text_props(fontweight="bold")
    elif col == 0:
        txt = cell.get_text().get_text()
        if "VISp" in txt:
            cell.set_facecolor("#D0DEFF")
        elif "VISl" in txt:
            cell.set_facecolor("#FFD0D0")

ax.set_title("Cross-session summary — mean±SEM and median[IQR] per area per subject",
             fontsize=10, fontweight="bold", pad=20)
fig.tight_layout()
fig.savefig("meso_ori_pop_summary.png", dpi=150, bbox_inches="tight")
print("Saved → meso_ori_pop_summary.png")

print("\nDone.")
