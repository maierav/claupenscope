"""Load behavioral signals (running speed, pupil) into a common DataFrame.

Verified paths (ecephys + mesoscope):
- ``processing/running/running_speed``  → (data, timestamps)
- ``processing/eye_tracking/pupil``     → (area, area_raw, angle, data_x, data_y,
                                            height, width, timestamps)

SLAP2 pilot behavior availability is unconfirmed — this loader will return
whatever is present and silently skip missing signals.

The returned DataFrame is indexed by time (seconds from session start) and
has one column per signal. It is *not* trial-aligned by default — use
:func:`align_behavior_to_trials` for that.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from openscope_pp.loaders.streaming import NWBHandle


def load_behavior(handle: "NWBHandle") -> pd.DataFrame:
    """Return a time-indexed DataFrame of behavioral signals.

    Columns may include: ``running_speed``, ``pupil_area``, ``pupil_x``,
    ``pupil_y``, ``pupil_angle``, ``pupil_width``, ``pupil_height``.
    Missing signals are omitted (no NaN padding).
    """
    h5 = handle.h5
    frames = {}

    # ── Running speed ────────────────────────────────────────────────
    rs_path = "processing/running/running_speed"
    if rs_path in h5:
        grp = h5[rs_path]
        ts = grp["timestamps"][:]
        data = grp["data"][:]
        frames["running_speed"] = pd.Series(data, index=ts, name="running_speed")

    # ── Pupil ────────────────────────────────────────────────────────
    pupil_path = "processing/eye_tracking/pupil"
    if pupil_path in h5:
        grp = h5[pupil_path]
        if "timestamps" in grp:
            ts = grp["timestamps"][:]
            col_map = {
                "area": "pupil_area",
                "area_raw": "pupil_area_raw",
                "angle": "pupil_angle",
                "data_x": "pupil_x",
                "data_y": "pupil_y",
                "width": "pupil_width",
                "height": "pupil_height",
            }
            for h5_col, df_col in col_map.items():
                if h5_col in grp:
                    frames[df_col] = pd.Series(
                        grp[h5_col][:], index=ts, name=df_col,
                    )

    if not frames:
        return pd.DataFrame()

    df = pd.DataFrame(frames)
    df.index.name = "time"
    return df


def align_behavior_to_trials(
    behavior: pd.DataFrame,
    trials: pd.DataFrame,
    window: tuple[float, float] = (-0.5, 1.0),
    *,
    signal: str = "running_speed",
    n_bins: int | None = None,
) -> np.ndarray:
    """Trial-align a behavioral signal.

    Parameters
    ----------
    behavior : DataFrame
        From :func:`load_behavior`, time-indexed.
    trials : DataFrame
        Must have ``start_time``.
    window : (pre, post) seconds
    signal : str
        Column name in *behavior*.
    n_bins : int, optional
        Number of output time bins.  Default: infer from the signal's
        native sampling rate.

    Returns
    -------
    np.ndarray
        Shape ``(n_trials, n_bins)``.
    """
    if signal not in behavior.columns:
        raise KeyError(f"{signal!r} not in behavior columns: {list(behavior.columns)}")

    times = behavior.index.values
    values = behavior[signal].values
    pre, post = window
    if n_bins is None:
        dt = np.median(np.diff(times[:1000]))
        n_bins = int(np.ceil((post - pre) / dt))

    bin_edges = np.linspace(pre, post, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    trial_starts = trials["start_time"].values
    n_trials = len(trial_starts)
    result = np.full((n_trials, n_bins), np.nan)

    for i, t0 in enumerate(trial_starts):
        mask = (times >= t0 + pre) & (times < t0 + post)
        if mask.sum() < 2:
            continue
        rel_t = times[mask] - t0
        vals = values[mask]
        # Simple binned mean
        for b in range(n_bins):
            in_bin = (rel_t >= bin_edges[b]) & (rel_t < bin_edges[b + 1])
            if in_bin.sum() > 0:
                result[i, b] = np.nanmean(vals[in_bin])

    return result
