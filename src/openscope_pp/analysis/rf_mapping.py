"""Receptive-field mapping — works across all three techniques.

Computes a spatial RF map per unit/ROI from the RF-mapping block, where small
gratings are flashed at a grid of (x, y) positions.

Usage
-----
>>> trials = load_trials(handle)
>>> rf_trials = trials[trials["block_kind"] == "rf_mapping"]
>>> responses = load_responses(handle, rf_trials, window=(0, 0.3))
>>> rf_maps = compute_rf_maps(rf_trials, responses)
>>> plot_rf_map(rf_maps, unit_idx=0)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr


def compute_rf_maps(
    trials: pd.DataFrame,
    responses: xr.DataArray,
    *,
    response_window: tuple[float, float] = (0.05, 0.3),
    baseline_window: tuple[float, float] = (-0.2, 0.0),
) -> xr.DataArray:
    """Compute spatial RF maps.

    Parameters
    ----------
    trials : DataFrame
        RF-mapping trials (filtered to ``block_kind == "rf_mapping"``).
    responses : DataArray
        ``(trial, unit, time)`` aligned to the same trials.
    response_window, baseline_window : tuples
        Windows for computing baseline-subtracted response amplitude.

    Returns
    -------
    xr.DataArray
        ``(unit, y, x)`` — the mean baseline-subtracted response at each
        spatial position.  Coordinates include ``unit_id``, ``x_deg``,
        ``y_deg``.
    """
    time = responses.coords["time_sec"].values
    unit_ids = responses.coords["unit_id"].values

    resp_mask = (time >= response_window[0]) & (time < response_window[1])
    base_mask = (time >= baseline_window[0]) & (time < baseline_window[1])

    resp_mean = responses.values[:, :, resp_mask].mean(axis=2)  # (trial, unit)
    base_mean = responses.values[:, :, base_mask].mean(axis=2)
    delta = resp_mean - base_mean

    x_vals = np.sort(trials["x"].dropna().unique())
    y_vals = np.sort(trials["y"].dropna().unique())

    n_units = len(unit_ids)
    rf = np.full((n_units, len(y_vals), len(x_vals)), np.nan)

    x_pos = trials["x"].values
    y_pos = trials["y"].values

    for xi, xv in enumerate(x_vals):
        for yi, yv in enumerate(y_vals):
            mask = (x_pos == xv) & (y_pos == yv)
            if mask.sum() > 0:
                rf[:, yi, xi] = np.nanmean(delta[mask, :], axis=0)

    return xr.DataArray(
        rf,
        dims=("unit", "y", "x"),
        coords={
            "unit_id": ("unit", list(unit_ids)),
            "y_deg": ("y", y_vals),
            "x_deg": ("x", x_vals),
        },
        attrs={"description": "Baseline-subtracted mean response at each (x,y) position"},
    )


def rf_center_of_mass(rf_maps: xr.DataArray) -> pd.DataFrame:
    """Compute center-of-mass RF location for each unit.

    Parameters
    ----------
    rf_maps : DataArray
        ``(unit, y, x)`` from :func:`compute_rf_maps`.

    Returns
    -------
    DataFrame
        Columns: ``unit_id``, ``rf_x``, ``rf_y``, ``rf_peak``, ``rf_area``
        (number of grid positions with response > 50% of peak).
    """
    unit_ids = rf_maps.coords["unit_id"].values
    x_grid = rf_maps.coords["x_deg"].values
    y_grid = rf_maps.coords["y_deg"].values

    records = []
    for j in range(len(unit_ids)):
        m = rf_maps.values[j, :, :]
        peak = np.nanmax(m)
        if np.isnan(peak) or peak <= 0:
            records.append({
                "unit_id": unit_ids[j],
                "rf_x": np.nan, "rf_y": np.nan,
                "rf_peak": np.nan, "rf_area": 0,
            })
            continue

        # Threshold at zero for center-of-mass (only positive responses)
        m_pos = np.maximum(m, 0)
        total = m_pos.sum()
        if total == 0:
            records.append({
                "unit_id": unit_ids[j],
                "rf_x": np.nan, "rf_y": np.nan,
                "rf_peak": float(peak), "rf_area": 0,
            })
            continue

        yy, xx = np.meshgrid(y_grid, x_grid, indexing="ij")
        cx = float(np.sum(xx * m_pos) / total)
        cy = float(np.sum(yy * m_pos) / total)
        area = int((m > 0.5 * peak).sum())

        records.append({
            "unit_id": unit_ids[j],
            "rf_x": cx, "rf_y": cy,
            "rf_peak": float(peak), "rf_area": area,
        })

    return pd.DataFrame(records)
