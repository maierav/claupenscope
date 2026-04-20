"""Oddball / mismatch response analysis — the preliminary cross-technique comparison.

Computes standard and deviant response averages, the mismatch response
(deviant − standard), and per-unit oddball indices, across any technique.

This works with the basic standard-vs-deviant contrast available from
all three techniques (ecephys P3, mesoscope P3, SLAP2 pilots).

Usage
-----
>>> trials = load_trials(handle)
>>> odd_trials = trials[trials["block_kind"] == "paradigm_oddball"]
>>> responses = load_responses(handle, odd_trials, window=(-0.5, 1.0))
>>> results = compute_oddball_index(odd_trials, responses)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr


def compute_oddball_responses(
    trials: pd.DataFrame,
    responses: xr.DataArray,
) -> dict[str, xr.DataArray]:
    """Compute trial-averaged responses for standards vs deviants.

    Parameters
    ----------
    trials : DataFrame
        Oddball-block trials (``block_kind == "paradigm_oddball"``).
    responses : DataArray
        ``(trial, unit, time)`` aligned to the same trials.

    Returns
    -------
    dict with keys:
        ``"standard_mean"`` : DataArray (unit, time)
        ``"deviant_mean"``  : DataArray (unit, time)
        ``"mismatch"``      : DataArray (unit, time) — deviant - standard
        ``"standard_sem"``  : DataArray (unit, time)
        ``"deviant_sem"``   : DataArray (unit, time)
    """
    is_dev = trials["is_deviant"].values

    std_mask = ~is_dev
    dev_mask = is_dev

    std_resp = responses.values[std_mask, :, :]  # (n_std, unit, time)
    dev_resp = responses.values[dev_mask, :, :]

    def _mean_sem(arr, axis=0):
        m = np.nanmean(arr, axis=axis)
        s = np.nanstd(arr, axis=axis, ddof=1) / np.sqrt(
            np.sum(~np.isnan(arr), axis=axis).clip(1)
        )
        return m, s

    std_m, std_s = _mean_sem(std_resp)
    dev_m, dev_s = _mean_sem(dev_resp)

    coords = {
        "unit_id": responses.coords["unit_id"],
        "time_sec": responses.coords["time_sec"],
    }
    dims = ("unit", "time")

    return {
        "standard_mean": xr.DataArray(std_m, dims=dims, coords=coords),
        "deviant_mean": xr.DataArray(dev_m, dims=dims, coords=coords),
        "mismatch": xr.DataArray(dev_m - std_m, dims=dims, coords=coords),
        "standard_sem": xr.DataArray(std_s, dims=dims, coords=coords),
        "deviant_sem": xr.DataArray(dev_s, dims=dims, coords=coords),
        "n_standard": int(std_mask.sum()),
        "n_deviant": int(dev_mask.sum()),
    }


def compute_oddball_index(
    trials: pd.DataFrame,
    responses: xr.DataArray,
    *,
    response_window: tuple[float, float] = (0.05, 0.5),
    baseline_window: tuple[float, float] = (-0.3, 0.0),
) -> pd.DataFrame:
    """Compute per-unit oddball / mismatch index.

    The oddball index (OI) is defined as::

        OI = (R_deviant - R_standard) / (|R_deviant| + |R_standard|)

    where R is the mean baseline-subtracted response in the response window.
    OI ranges from −1 to +1; positive means stronger deviant response.

    Parameters
    ----------
    trials : DataFrame
        Oddball-block trials.
    responses : DataArray
        ``(trial, unit, time)``.
    response_window, baseline_window : tuples
        Windows in seconds.

    Returns
    -------
    DataFrame
        One row per unit: ``unit_id``, ``oddball_index``,
        ``response_standard``, ``response_deviant``, ``p_value``
        (permutation test or Welch t-test).
    """
    time = responses.coords["time_sec"].values
    unit_ids = responses.coords["unit_id"].values
    is_dev = trials["is_deviant"].values

    resp_mask = (time >= response_window[0]) & (time < response_window[1])
    base_mask = (time >= baseline_window[0]) & (time < baseline_window[1])

    # NaN-aware: interp-derived snippets may carry edge NaNs on individual
    # trials (onsets near recording boundaries). Plain .mean would propagate
    # those to the whole (trial, unit) cell.
    resp_amp = np.nanmean(responses.values[:, :, resp_mask], axis=2)
    base_amp = np.nanmean(responses.values[:, :, base_mask], axis=2)
    delta = resp_amp - base_amp  # (trial, unit)

    std_delta = delta[~is_dev, :]
    dev_delta = delta[is_dev, :]

    r_std = np.nanmean(std_delta, axis=0)
    r_dev = np.nanmean(dev_delta, axis=0)

    denom = np.abs(r_dev) + np.abs(r_std)
    oi = np.where(denom > 0, (r_dev - r_std) / denom, 0.0)

    # Welch's t-test per unit
    from scipy import stats

    p_vals = np.full(len(unit_ids), np.nan)
    for j in range(len(unit_ids)):
        s = std_delta[:, j]
        d = dev_delta[:, j]
        s = s[~np.isnan(s)]
        d = d[~np.isnan(d)]
        if len(s) >= 2 and len(d) >= 2:
            _, p_vals[j] = stats.ttest_ind(d, s, equal_var=False)

    return pd.DataFrame({
        "unit_id": unit_ids,
        "oddball_index": oi,
        "response_standard": r_std,
        "response_deviant": r_dev,
        "p_value": p_vals,
    })
