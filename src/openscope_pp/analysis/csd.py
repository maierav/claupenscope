"""Laminar LFP and Current Source Density (CSD) analysis for ecephys.

Provides trial-aligned LFP extraction, common-average referencing, and CSD
computation via the second spatial derivative along probe depth.

The approach follows the reference notebook ``intro_to_ephys_nwbs_CSD.ipynb``
(maierav, Jan-2026).

Usage
-----
>>> from openscope_pp.loaders.streaming import open_nwb
>>> from openscope_pp.loaders.trials import load_trials
>>> from openscope_pp.analysis.csd import (
...     load_lfp_metadata, extract_trial_lfp, compute_csd,
... )
>>> handle = open_nwb(asset_id)
>>> trials = load_trials(handle)
>>> meta = load_lfp_metadata(handle, probe="ProbeC")
>>> lfp = extract_trial_lfp(handle, trials, meta, window=(-0.2, 0.8))
>>> csd = compute_csd(lfp, meta["depths_sorted"])
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    import pandas as pd
    from openscope_pp.loaders.streaming import NWBHandle


# ── Probe metadata ──────────────────────────────────────────────────

@dataclass
class LFPMeta:
    """Metadata for one probe's LFP data."""
    probe: str
    lfp_path: str           # HDF5 path to the ElectricalSeries
    n_channels: int
    depths: np.ndarray      # raw depth per channel (µm)
    depth_order: np.ndarray  # argsort of depths
    depths_sorted: np.ndarray  # depths[depth_order]
    electrode_ids: np.ndarray  # electrode indices


def load_lfp_metadata(
    handle: "NWBHandle",
    probe: str = "ProbeC",
    depth_col: str = "rel_y",
) -> LFPMeta:
    """Read electrode depths and build channel ordering for one probe.

    Parameters
    ----------
    handle : NWBHandle
        Must be an ecephys session.
    probe : str
        Probe name (e.g. ``"ProbeC"``).
    depth_col : str
        Column in the electrodes table to use as the depth axis.
        Default ``"rel_y"`` (position along probe shank, 0–3800 µm).

    Returns
    -------
    LFPMeta
    """
    h5 = handle.h5
    lfp_path = f"processing/ecephys/LFP/ElectricalSeries{probe}-LFP"
    if lfp_path not in h5:
        available = [k for k in h5["processing/ecephys/LFP"].keys()
                     if k.startswith("ElectricalSeries")]
        raise KeyError(
            f"LFP path {lfp_path!r} not found. "
            f"Available: {available}"
        )

    es = h5[lfp_path]
    n_channels = es["data"].shape[1]

    # The electrodes table is at /general/extracellular_ephys/electrodes
    # but the series references specific rows via its electrodes dataset.
    # With h5py we read the electrode indices from the series' electrodes ref.
    elec_table = h5["general/extracellular_ephys/electrodes"]

    # The ElectricalSeries stores its electrode references — read the indices
    if "electrodes" in es:
        elec_idx = es["electrodes"][:]
    else:
        # Fallback: assume first n_channels electrodes
        elec_idx = np.arange(n_channels)

    # Read depth column
    if depth_col not in elec_table:
        cols = [k for k in elec_table.keys() if k.startswith("rel")]
        raise KeyError(
            f"{depth_col!r} not in electrodes table. "
            f"Available rel_* columns: {cols}"
        )
    all_depths = elec_table[depth_col][:]
    depths = all_depths[elec_idx].astype(np.float64)

    order = np.argsort(depths)
    return LFPMeta(
        probe=probe,
        lfp_path=lfp_path,
        n_channels=n_channels,
        depths=depths,
        depth_order=order,
        depths_sorted=depths[order],
        electrode_ids=elec_idx,
    )


def list_probes(handle: "NWBHandle") -> list[str]:
    """Return available probe names that have LFP data."""
    h5 = handle.h5
    lfp_grp = h5.get("processing/ecephys/LFP")
    if lfp_grp is None:
        return []
    probes = []
    for k in lfp_grp.keys():
        if k.startswith("ElectricalSeries") and k.endswith("-LFP"):
            # "ElectricalSeriesProbeC-LFP" → "ProbeC"
            name = k[len("ElectricalSeries"):-len("-LFP")]
            probes.append(name)
    return sorted(probes)


# ── Trial-aligned LFP extraction ────────────────────────────────────

def extract_trial_lfp(
    handle: "NWBHandle",
    trials: "pd.DataFrame",
    meta: LFPMeta,
    window: tuple[float, float] = (-0.2, 0.8),
    *,
    car: bool = True,
    baseline: tuple[float, float] | None = None,
    downsample: int = 1,
    max_nan_frac: float = 0.2,
) -> xr.DataArray:
    """Extract trial-aligned LFP traces for one probe.

    Parameters
    ----------
    handle : NWBHandle
    trials : DataFrame
        Must have ``start_time`` column.
    meta : LFPMeta
        From :func:`load_lfp_metadata`.
    window : (pre, post)
        Time window in seconds relative to trial onset.
    car : bool
        Apply common-average reference (subtract mean across channels).
    baseline : (start, end) or None
        If provided, subtract the mean LFP in this time window per channel.
        E.g. ``(-0.2, 0.0)`` for pre-stimulus baseline.
    downsample : int
        Keep every nth sample within each window (default 1 = no downsampling).
    max_nan_frac : float
        Drop trials with more than this fraction of NaN samples.

    Returns
    -------
    xr.DataArray
        ``(trial, channel, time)`` — channels sorted by depth.
    """
    h5 = handle.h5
    es = h5[meta.lfp_path]
    data_ds = es["data"]

    # Read full timestamps (one read, ~tens of MB as float32)
    timestamps = es["timestamps"][:].astype(np.float32)

    # Estimate sample rate
    dt_est = float(np.median(np.diff(timestamps[:50000])))
    dt_eff = dt_est * downsample

    pre, post = window
    t_grid = np.arange(pre, post, dt_eff, dtype=np.float32)
    n_time = len(t_grid)

    trial_starts = trials["start_time"].values
    n_trials = len(trial_starts)
    n_chan = meta.n_channels
    order = meta.depth_order

    result = np.full((n_trials, n_chan, n_time), np.nan, dtype=np.float32)
    valid_mask = np.ones(n_trials, dtype=bool)

    for i, t0 in enumerate(trial_starts):
        i0 = int(np.searchsorted(timestamps, t0 + pre, side="left"))
        i1 = int(np.searchsorted(timestamps, t0 + post, side="right"))

        if i1 <= i0 + 10:
            valid_mask[i] = False
            continue

        # Read this trial's LFP slice
        V = np.array(data_ds[i0:i1, :], dtype=np.float32)
        tt = timestamps[i0:i1] - np.float32(t0)

        # Reorder by depth
        V = V[:, order]

        # Common-average reference
        if car:
            V -= V.mean(axis=1, keepdims=True)

        # Downsample
        if downsample > 1:
            V = V[::downsample, :]
            tt = tt[::downsample]

        # Resample onto uniform t_grid
        Vg = np.full((n_time, n_chan), np.nan, dtype=np.float32)
        for ch in range(n_chan):
            Vg[:, ch] = np.interp(
                t_grid, tt, V[:, ch], left=np.nan, right=np.nan,
            )

        # Check NaN fraction
        if np.mean(~np.isfinite(Vg)) > max_nan_frac:
            valid_mask[i] = False
            continue

        result[i, :, :] = np.nan_to_num(Vg, nan=0.0).T  # (chan, time)

    # Apply baseline correction
    if baseline is not None:
        b0 = int(np.searchsorted(t_grid, baseline[0], side="left"))
        b1 = int(np.searchsorted(t_grid, baseline[1], side="right"))
        if b1 > b0:
            bl_mean = np.nanmean(result[:, :, b0:b1], axis=2, keepdims=True)
            result -= bl_mean

    # Filter out invalid trials
    if not valid_mask.all():
        result = result[valid_mask]
        trial_starts = trial_starts[valid_mask]

    return xr.DataArray(
        result,
        dims=("trial", "channel", "time"),
        coords={
            "trial_start": ("trial", trial_starts),
            "depth_um": ("channel", meta.depths_sorted),
            "time_sec": ("time", t_grid.astype(np.float64)),
        },
        attrs={
            "probe": meta.probe,
            "sample_rate_hz": 1.0 / dt_eff,
            "car_applied": car,
            "baseline": baseline,
            "technique": "ecephys",
        },
    )


# ── CSD computation ─────────────────────────────────────────────────

def compute_csd(
    lfp: xr.DataArray,
    depths_um: np.ndarray | None = None,
    *,
    sigma: float = 1.0,
) -> xr.DataArray:
    """Compute Current Source Density from trial-aligned LFP.

    CSD = −σ · ∂²V/∂z² using finite differences along the depth axis.

    Parameters
    ----------
    lfp : xr.DataArray
        ``(trial, channel, time)`` or ``(channel, time)`` — e.g. from
        :func:`extract_trial_lfp` or a trial-averaged result.
    depths_um : array-like, optional
        Channel depths in µm.  If ``None``, reads from ``lfp.coords["depth_um"]``.
    sigma : float
        Conductivity scaling factor.  Default 1.0 (arbitrary units).

    Returns
    -------
    xr.DataArray
        Same shape as *lfp*, with CSD values.
    """
    if depths_um is None:
        depths_um = lfp.coords["depth_um"].values

    dz = float(np.median(np.diff(depths_um)))
    if not np.isfinite(dz) or dz == 0:
        raise ValueError(
            f"Invalid depth spacing dz={dz}. Check depth values."
        )

    data = lfp.values
    # Second spatial derivative along the channel (depth) axis
    # For (trial, channel, time) this is axis=1; for (channel, time) axis=0
    depth_axis = lfp.dims.index("channel")
    d2V = np.gradient(np.gradient(data, dz, axis=depth_axis), dz, axis=depth_axis)
    csd_data = -sigma * d2V

    result = lfp.copy(data=csd_data)
    result.attrs = dict(lfp.attrs)
    result.attrs["sigma"] = sigma
    result.attrs["dz_um"] = dz
    result.name = "CSD"
    return result


# ── Convenience: condition-averaged LFP and CSD ─────────────────────

def condition_average_lfp(
    lfp: xr.DataArray,
    trials: "pd.DataFrame",
    condition_col: str = "trial_type",
) -> dict[str, xr.DataArray]:
    """Average trial-aligned LFP by condition.

    Parameters
    ----------
    lfp : xr.DataArray
        ``(trial, channel, time)`` from :func:`extract_trial_lfp`.
    trials : DataFrame
        Must have the same number of rows as ``lfp.sizes["trial"]``
        (after any NaN-trial filtering, you may need to pass the filtered
        trials subset).
    condition_col : str
        Column in *trials* to group by.

    Returns
    -------
    dict[str, xr.DataArray]
        Keys are condition labels, values are ``(channel, time)`` averages.
    """
    n_trials_lfp = lfp.sizes["trial"]
    # trials may have been filtered; take the first n_trials_lfp rows
    conditions = trials[condition_col].values[:n_trials_lfp]
    unique_conds = np.unique(conditions)

    result = {}
    for cond in unique_conds:
        mask = conditions == cond
        mean_lfp = lfp.isel(trial=mask).mean("trial")
        mean_lfp.attrs = dict(lfp.attrs)
        mean_lfp.attrs["condition"] = str(cond)
        mean_lfp.attrs["n_trials"] = int(mask.sum())
        result[str(cond)] = mean_lfp

    return result
