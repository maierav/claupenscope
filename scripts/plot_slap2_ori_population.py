"""Plot cross-session SLAP2 orientation tuning statistics.

Reads results/slap2_ori_stats.csv produced by collect_slap2_ori_stats.py.
Produces three figures:

  slap2_ori_pop_dmd.png       — DMD1 vs DMD2 violin plots (OSI & DSI),
                                  mean and median overlaid, all sessions pooled
  slap2_ori_pop_subjects.png  — per-subject bar charts, DMD split
  slap2_ori_pop_summary.png   — summary table: mean±SEM and median[IQR]
                                  per DMD, per subject, and overall
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

IN_CSV = "results/slap2_ori_stats.csv"

if not os.path.exists(IN_CSV):
    raise FileNotFoundError(f"{IN_CSV} not found — run collect_slap2_ori_stats.py first")

df = pd.read_csv(IN_CSV)
print(f"Loaded {len(df)} rows  ({df['asset_id'].nunique()} sessions, "
      f"{df['subject_id'].nunique()} subjects)")
print(df.groupby(["dmd"])["n_rois"].agg(["count", "sum", "mean"]).round(1))

DMD_COLORS   = {"DMD1": "#4878CF", "DMD2": "#D65F5F"}
SUBJ_MARKERS = {s: m for s, m in zip(sorted(df["subject_id"].unique()),
                                      ["o", "s", "^", "D", "v", "P"])}
subjects = sorted(df["subject_id"].unique())

# ── Figure 1: DMD1 vs DMD2 violin + mean/median, all sessions ────────
metrics = [
    ("mean_OSI",   "Mean OSI"),
    ("median_OSI", "Median OSI"),
    ("mean_DSI",   "Mean DSI"),
    ("median_DSI", "Median DSI"),
]

dmds = sorted(df["dmd"].unique())  # might be just DMD1, or DMD1+DMD2
positions = list(range(1, len(dmds) + 1))

fig, axes = plt.subplots(1, 4, figsize=(14, 5), sharey=False)

for ax, (col, label) in zip(axes, metrics):
    data_per_dmd = [df.loc[df["dmd"] == d, col].values for d in dmds]
    if all(len(v) > 0 for v in data_per_dmd):
        parts = ax.violinplot(data_per_dmd, positions=positions,
                              showmedians=False, showextrema=False)
        for pc, dmd in zip(parts["bodies"], dmds):
            pc.set_facecolor(DMD_COLORS.get(dmd, "gray"))
            pc.set_alpha(0.5)

    for xi, dmd in zip(positions, dmds):
        vals = df.loc[df["dmd"] == dmd, col].values
        if len(vals) == 0:
            continue
        jx = xi + np.random.default_rng(42).uniform(-0.08, 0.08, len(vals))
        ax.scatter(jx, vals, color=DMD_COLORS.get(dmd, "gray"),
                   s=35, alpha=0.8, zorder=3, edgecolors="none")
        ax.plot([xi - 0.2, xi + 0.2], [np.mean(vals)] * 2,
                color="k", lw=2.5, zorder=4, solid_capstyle="round")
        ax.plot([xi - 0.12, xi + 0.12], [np.median(vals)] * 2,
                color="white", lw=1.5, zorder=5, ls="--",
                solid_capstyle="round")

    ax.set_xticks(positions)
    ax.set_xticklabels(dmds)
    ax.set_ylabel(label)
    ax.set_title(label)
    ax.spines[["top", "right"]].set_visible(False)

legend_elements = [
    Patch(facecolor="none", edgecolor="k",    label="— black line = mean"),
    Patch(facecolor="none", edgecolor="gray",  label="-- white line = median"),
]
axes[-1].legend(handles=legend_elements, fontsize=7, loc="upper right")
fig.suptitle(
    f"SLAP2 orientation tuning — DMD1 vs DMD2\n"
    f"{df['asset_id'].nunique()} sessions · {df['subject_id'].nunique()} subjects · "
    f"dandiset 001424",
    fontsize=10, fontweight="bold",
)
fig.tight_layout()
fig.savefig("slap2_ori_pop_dmd.png", dpi=150, bbox_inches="tight")
print("Saved → slap2_ori_pop_dmd.png")

# ── Figure 2: Per-subject bar charts, DMD split ────────────────────────
metric_pairs = [("mean_OSI", "Mean OSI"), ("median_OSI", "Median OSI"),
                ("mean_DSI", "Mean DSI"), ("median_DSI", "Median DSI")]

fig, axes = plt.subplots(2, 2, figsize=(max(10, len(subjects) * 2.0), 8))

for ax, (col, label) in zip(axes.ravel(), metric_pairs):
    positions_ticks = []
    for si, subj in enumerate(subjects):
        for ai, dmd in enumerate(dmds):
            pos = si * (len(dmds) + 1) + ai
            vals = df.loc[(df["subject_id"] == subj) & (df["dmd"] == dmd), col].values
            if len(vals) == 0:
                continue
            mean_v = np.mean(vals)
            sem_v  = np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0
            color  = DMD_COLORS.get(dmd, "gray")
            ax.bar(pos, mean_v, yerr=sem_v, color=color, alpha=0.75,
                   width=0.7, capsize=3, error_kw=dict(lw=1.5))
            # individual session dots
            jx = pos + np.random.default_rng(42).uniform(-0.15, 0.15, len(vals))
            ax.scatter(jx, vals, color="k", s=15, alpha=0.7, zorder=3)

        mid = si * (len(dmds) + 1) + (len(dmds) - 1) / 2
        positions_ticks.append((mid, subj.replace("sub-", "")))

    ax.set_xticks([t for t, _ in positions_ticks])
    ax.set_xticklabels([l for _, l in positions_ticks], fontsize=8)
    ax.set_ylabel(label)
    ax.set_title(label)
    ax.spines[["top", "right"]].set_visible(False)

    legend_patches = [Patch(facecolor=DMD_COLORS.get(d, "gray"), alpha=0.75, label=d)
                      for d in dmds]
    ax.legend(handles=legend_patches, fontsize=7, loc="upper right")

fig.suptitle("SLAP2 orientation tuning — per-subject breakdown\n"
             "Bar = mean ± SEM · dots = individual sessions",
             fontsize=10, fontweight="bold")
fig.tight_layout()
fig.savefig("slap2_ori_pop_subjects.png", dpi=150, bbox_inches="tight")
print("Saved → slap2_ori_pop_subjects.png")

# ── Figure 3: Summary table ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 4))
ax.axis("off")

col_labels = ["", "n sessions", "n ROIs",
              "mean OSI", "median OSI", "IQR OSI",
              "mean DSI", "median DSI", "IQR DSI"]
rows_data = []

for dmd in dmds:
    for subj in ["(all)"] + subjects:
        if subj == "(all)":
            sub_df = df[df["dmd"] == dmd]
            label  = f"{dmd} — ALL"
        else:
            sub_df = df[(df["dmd"] == dmd) & (df["subject_id"] == subj)]
            label  = f"  {subj.replace('sub-', '')}"
        if len(sub_df) == 0:
            continue

        def fmt(vals):
            mn = np.mean(vals); md = np.median(vals)
            se = np.std(vals) / np.sqrt(len(vals))
            q1, q3 = np.percentile(vals, [25, 75])
            return f"{mn:.3f}±{se:.3f}", f"{md:.3f}", f"[{q1:.3f},{q3:.3f}]"

        m_osi, med_osi, iqr_osi = fmt(sub_df["mean_OSI"].values)
        m_dsi, med_dsi, iqr_dsi = fmt(sub_df["mean_DSI"].values)
        rows_data.append([
            label,
            str(sub_df["asset_id"].nunique()),
            f"{sub_df['n_rois'].sum():,}",
            m_osi, med_osi, iqr_osi,
            m_dsi, med_dsi, iqr_dsi,
        ])
    rows_data.append([""] * len(col_labels))

tbl = ax.table(cellText=rows_data, colLabels=col_labels,
               loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.auto_set_column_width(list(range(len(col_labels))))

for (row, col), cell in tbl.get_celld().items():
    if row == 0:
        cell.set_facecolor("#CCCCCC")
        cell.set_text_props(fontweight="bold")
    elif col == 0:
        txt = cell.get_text().get_text()
        if "DMD1" in txt:
            cell.set_facecolor("#D0DEFF")
        elif "DMD2" in txt:
            cell.set_facecolor("#FFD0D0")

ax.set_title("SLAP2 cross-session summary — mean±SEM and median[IQR] per DMD per subject",
             fontsize=10, fontweight="bold", pad=20)
fig.tight_layout()
fig.savefig("slap2_ori_pop_summary.png", dpi=150, bbox_inches="tight")
print("Saved → slap2_ori_pop_summary.png")

print("\nDone.")
