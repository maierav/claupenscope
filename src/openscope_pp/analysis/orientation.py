"""Orientation-tuning analysis — works across all three techniques.

Computes an orientation-tuning curve per unit/ROI from the orientation-tuning
block(s) in a session, and extracts standard metrics:

- Preferred orientation (circular mean of response-weighted orientations)
- Orientation selectivity index (OSI)
- Direction selectivity index (DSI)
- Tuning curve (mean response at each presented orientation)

Usage
-----
>>> trials = load_trials(handle)
>>> responses = load_responses(handle, trials, window=(0, 0.5))
>>> ori_tuning = compute_orientation_tuning(trials, responses)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr


def select_ori_tuning_trials(trials: pd.DataFrame) -> pd.DataFrame:
    """Filter trials to orientation-tuning blocks only."""
    return trials[trials["block_kind"] == "ori_tuning"].copy()


def compute_orientation_tuning(
    trials: pd.DataFrame,
    responses: xr.DataArray,
    *,
    response_window: tuple[float, float] = (0.05, 0.5),
    baseline_window: tuple[float, float] = (-0.3, 0.0),
) -> pd.DataFrame:
    """Compute orientation-tuning metrics for each unit.

    Parameters
    ----------
    trials : DataFrame
        Should be pre-filtered to orientation-tuning trials.
    responses : DataArray
        ``(trial, unit, time)`` from :func:`load_responses`, aligned to the
        same trials.
    response_window : (start, end)
        Post-stimulus window in seconds to average for the "response".
    baseline_window : (start, end)
        Pre-stimulus window for baseline subtraction.

    Returns
    -------
    DataFrame
        One row per unit with columns: ``unit_id``, ``pref_ori``,
        ``osi``, ``dsi``, ``peak_response``, ``mean_baseline``,
        ``tuning_curve`` (dict mapping orientation → mean response).
    """
    time = responses.coords["time_sec"].values
    unit_ids = responses.coords["unit_id"].values
    orientations = trials["orientation"].values

    # Time masks
    resp_mask = (time >= response_window[0]) & (time < response_window[1])
    base_mask = (time >= baseline_window[0]) & (time < baseline_window[1])

    # Mean response per trial per unit
    resp_mean = responses.values[:, :, resp_mask].mean(axis=2)  # (trial, unit)
    base_mean = responses.values[:, :, base_mask].mean(axis=2)  # (trial, unit)
    delta = resp_mean - base_mean  # baseline-subtracted

    unique_oris = np.sort(np.unique(orientations))
    n_units = len(unit_ids)

    records = []
    for j in range(n_units):
        tuning = {}
        for ori in unique_oris:
            mask = orientations == ori
            if mask.sum() > 0:
                tuning[float(ori)] = float(np.nanmean(delta[mask, j]))

        oris = np.array(list(tuning.keys()))
        resps = np.array(list(tuning.values()))

        if len(oris) < 2 or np.all(resps == 0):
            records.append({
                "unit_id": unit_ids[j],
                "pref_ori": np.nan,
                "osi": np.nan,
                "dsi": np.nan,
                "peak_response": np.nan,
                "mean_baseline": float(base_mean[:, j].mean()),
                "tuning_curve": tuning,
            })
            continue

        # Preferred orientation (circular mean weighted by response)
        # Orientations are in radians; double them for π-periodicity
        weights = np.maximum(resps, 0)  # only positive responses contribute
        if weights.sum() > 0:
            z = np.sum(weights * np.exp(2j * oris)) / weights.sum()
            pref_ori = (np.angle(z) / 2) % np.pi
        else:
            pref_ori = oris[np.argmax(resps)]

        # OSI = (R_pref - R_orth) / (R_pref + R_orth)
        r_pref = resps.max()
        # Find the orientation closest to 90° away from preferred
        orth_target = (pref_ori + np.pi / 2) % np.pi
        orth_idx = np.argmin(np.abs(np.angle(np.exp(2j * (oris - orth_target)))))
        r_orth = resps[orth_idx]
        osi = (r_pref - r_orth) / (r_pref + r_orth) if (r_pref + r_orth) > 0 else 0.0

        # DSI: treat orientations as directions (0..2π)
        # DSI = (R_pref_dir - R_null_dir) / (R_pref_dir + R_null_dir)
        null_target = (pref_ori + np.pi) % (2 * np.pi)
        null_dists = np.abs(np.angle(np.exp(1j * (oris - null_target))))
        null_idx = np.argmin(null_dists)
        r_null = resps[null_idx]
        dsi = (r_pref - r_null) / (r_pref + r_null) if (r_pref + r_null) > 0 else 0.0

        records.append({
            "unit_id": unit_ids[j],
            "pref_ori": float(pref_ori),
            "osi": float(osi),
            "dsi": float(dsi),
            "peak_response": float(r_pref),
            "mean_baseline": float(base_mean[:, j].mean()),
            "tuning_curve": tuning,
        })

    return pd.DataFrame(records)
