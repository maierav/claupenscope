"""Plot cross-session ecephys orientation tuning statistics.

Reads results/ephys_ori_stats.csv produced by collect_ephys_ori_stats.py.
Produces three figures:

  ephys_ori_pop_probes.png    — per-probe OSI & DSI bar charts (mean±SEM),
                                  individual session dots overlaid, coloured by subject
  ephys_ori_pop_subjects.png  — per-subject breakdown across probes
  ephys_ori_pop_summary.png   — summary table: mean±SEM and median[IQR]
                                  per probe, per subject, and overall
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

IN_CSV = "results/ephys_ori_stats.csv"

if not os.path.exists(IN_CSV):
    raise FileNotFoundError(f"{IN_CSV} not found — run collect_ephys_ori_stats.py first")

df = pd.read_csv(IN_CSV)
print(f"Loaded {len(df)} rows  ({df['asset_id'].nunique()} sessions, "
      f"{df['subject_id'].nunique()} subjects)")
print(df.groupby(["probe"])["n_units"].agg(["count", "sum", "mean"]).round(1))

probes   = sorted(df["probe"].unique())
subjects = sorted(df["subject_id"].unique())

# Assign a colour per subject and a marker
SUBJ_COLORS = {s: c for s, c in zip(subjects,
    ["#4878CF", "#D65F5F", "#6ACC65", "#B47CC7", "#C4AD66", "#77BEDB"])}
SUBJ_MARKERS = {s: m for s, m in zip(subjects, ["o", "s", "^", "D", "v", "P"])}

# ── Figure 1: Per-probe bar chart (mean±SEM), coloured by subject ──────
fig, axes = plt.subplots(1, 2, figsize=(max(12, len(probes) * 1.8), 5))
metric_pairs = [("mean_OSI", "Mean OSI"), ("mean_DSI", "Mean DSI")]

for ax, (col, label) in zip(axes, metric_pairs):
    x_base = np.arange(len(probes))
    n_subj = len(subjects)
    width  = 0.7 / n_subj

    for si, subj in enumerate(subjects):
        offsets = (si - (n_subj - 1) / 2) * width
        for pi, probe in enumerate(probes):
            vals = df.loc[(df["subject_id"] == subj) & (df["probe"] == probe), col].values
            if len(vals) == 0:
                continue
            mean_v = np.mean(vals)
            sem_v  = np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0
            xpos   = x_base[pi] + offsets
            ax.bar(xpos, mean_v, width=width * 0.85,
                   color=SUBJ_COLORS[subj], alpha=0.75,
                   yerr=sem_v, capsize=2, error_kw=dict(lw=1.2))
            jx = xpos + np.random.default_rng(42).uniform(-width * 0.2,
                                                            width * 0.2, len(vals))
            ax.scatter(jx, vals, color="k", s=12, alpha=0.7,
                       zorder=4, marker=SUBJ_MARKERS[subj])

    # Cross-session mean ± SEM per probe (thick horizontal lines)
    for pi, probe in enumerate(probes):
        all_vals = df.loc[df["probe"] == probe, col].values
        if len(all_vals) == 0:
            continue
        mn = np.mean(all_vals)
        se = np.std(all_vals) / np.sqrt(len(all_vals))
        ax.plot([x_base[pi] - 0.38, x_base[pi] + 0.38], [mn, mn],
                color="k", lw=2.5, zorder=5, solid_capstyle="round")
        ax.fill_between([x_base[pi] - 0.38, x_base[pi] + 0.38],
                        [mn - se, mn - se], [mn + se, mn + se],
                        color="gray", alpha=0.15, zorder=4)

    ax.set_xticks(x_base)
    ax.set_xticklabels(probes, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel(label)
    ax.set_title(label)
    ax.spines[["top", "right"]].set_visible(False)

# Subject legend
legend_patches = [Patch(facecolor=SUBJ_COLORS[s], alpha=0.75,
                        label=s.replace("sub-", "")) for s in subjects]
axes[0].legend(handles=legend_patches, title="Subject", fontsize=7,
               loc="upper right")

fig.suptitle(
    f"Ecephys orientation tuning — per-probe OSI & DSI\n"
    f"{df['asset_id'].nunique()} sessions · {df['subject_id'].nunique()} subjects · "
    f"dandiset 001637  (black bar = cross-session mean±SEM)",
    fontsize=10, fontweight="bold",
)
fig.tight_layout()
fig.savefig("ephys_ori_pop_probes.png", dpi=150, bbox_inches="tight")
print("Saved → ephys_ori_pop_probes.png")

# ── Figure 2: Per-subject violin, probes pooled ───────────────────────
fig, axes = plt.subplots(1, 2, figsize=(max(8, len(subjects) * 2.5), 5))

for ax, (col, label) in zip(axes, metric_pairs):
    data_list = [df.loc[df["subject_id"] == s, col].values for s in subjects]
    # Violin only if enough data points
    valid = [(i, d) for i, d in enumerate(data_list) if len(d) >= 2]
    if valid:
        vp = ax.violinplot([d for _, d in valid],
                           positions=[i + 1 for i, _ in valid],
                           showmedians=False, showextrema=False)
        for pc, (i, _) in zip(vp["bodies"], valid):
            pc.set_facecolor(SUBJ_COLORS[subjects[i]])
            pc.set_alpha(0.45)

    for si, (subj, vals) in enumerate(zip(subjects, data_list)):
        xi = si + 1
        if len(vals) == 0:
            continue
        jx = xi + np.random.default_rng(42).uniform(-0.12, 0.12, len(vals))
        ax.scatter(jx, vals, color=SUBJ_COLORS[subj], s=35, alpha=0.85,
                   zorder=3, edgecolors="none", marker=SUBJ_MARKERS[subj])
        ax.plot([xi - 0.22, xi + 0.22], [np.mean(vals)] * 2,
                color="k", lw=2.5, zorder=4, solid_capstyle="round")
        ax.plot([xi - 0.13, xi + 0.13], [np.median(vals)] * 2,
                color="white", lw=1.5, ls="--", zorder=5,
                solid_capstyle="round")

    ax.set_xticks(range(1, len(subjects) + 1))
    ax.set_xticklabels([s.replace("sub-", "") for s in subjects], fontsize=8)
    ax.set_ylabel(label)
    ax.set_title(label)
    ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("Ecephys orientation tuning — per-subject (all probes pooled)\n"
             "black = mean · white dashed = median",
             fontsize=10, fontweight="bold")
fig.tight_layout()
fig.savefig("ephys_ori_pop_subjects.png", dpi=150, bbox_inches="tight")
print("Saved → ephys_ori_pop_subjects.png")

# ── Figure 3: Summary table ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, max(4, (len(probes) * (len(subjects) + 2) + 4) * 0.28)))
ax.axis("off")

col_labels = ["", "n sessions", "n units",
              "mean OSI", "median OSI", "IQR OSI",
              "mean DSI", "median DSI", "IQR DSI"]
rows_data = []
PROBE_BG = ["#D0DEFF", "#FFD0D0", "#D0FFD8", "#FFF0D0", "#F0D0FF", "#D0F0FF"]

for pi, probe in enumerate(probes):
    for subj in ["(all)"] + subjects:
        if subj == "(all)":
            sub_df = df[df["probe"] == probe]
            label  = f"{probe} — ALL"
        else:
            sub_df = df[(df["probe"] == probe) & (df["subject_id"] == subj)]
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
            f"{sub_df['n_units'].sum():,}",
            m_osi, med_osi, iqr_osi,
            m_dsi, med_dsi, iqr_dsi,
        ])
    rows_data.append([""] * len(col_labels))

tbl = ax.table(cellText=rows_data, colLabels=col_labels,
               loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.auto_set_column_width(list(range(len(col_labels))))

probe_bg_map = {probe: PROBE_BG[pi % len(PROBE_BG)] for pi, probe in enumerate(probes)}
for (row, col_), cell in tbl.get_celld().items():
    if row == 0:
        cell.set_facecolor("#CCCCCC")
        cell.set_text_props(fontweight="bold")
    elif col_ == 0:
        txt = cell.get_text().get_text()
        for probe, bg in probe_bg_map.items():
            if probe in txt:
                cell.set_facecolor(bg)
                break

ax.set_title("Ecephys cross-session summary — mean±SEM and median[IQR] per probe per subject",
             fontsize=10, fontweight="bold", pad=20)
fig.tight_layout()
fig.savefig("ephys_ori_pop_summary.png", dpi=150, bbox_inches="tight")
print("Saved → ephys_ori_pop_summary.png")

print("\nDone.")
