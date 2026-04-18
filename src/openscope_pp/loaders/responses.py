"""Trial-aligned neural responses — common interface across techniques.

Each technique adapter reads data lazily from the open HDF5 handle and
returns an xarray.DataArray with dimensions ``(trial, unit, time)`` plus
metadata coords (``unit_id``, ``trial_start``, ``time_sec``).

Usage
-----
>>> responses = load_responses(handle, trials_df, window=(-0.5, 1.0))
>>> responses.sel(unit="ProbeA_42").mean("trial").plot()
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    import pandas as pd
    from openscope_pp.loaders.streaming import NWBHandle


def load_responses(
    handle: "NWBHandle",
    trials: "pd.DataFrame",
    window: tuple[float, float] = (-0.5, 1.0),
    *,
    bin_size: float | None = None,
    unit_filter: dict | None = None,
    plane_filter: list[str] | None = None,
    dmd_filter: list[str] | None = None,
    signal_type: str = "dff",
    soma_only: bool = True,
    baseline_window: tuple[float, float] | None = None,
    baseline_mode: str = "zscore",
) -> xr.DataArray:
    """Extract trial-aligned neural responses.

    Parameters
    ----------
    handle : NWBHandle
    trials : DataFrame
        Must have ``start_time`` column (used as alignment reference).
    window : (pre, post) in seconds
        Relative to each trial's ``start_time``.  *pre* should be negative.
    bin_size : float, optional
        Bin width in seconds.  Ecephys default = 0.01 (10 ms).
        Ignored for imaging techniques (they use native samples).
    unit_filter : dict, optional
        Ecephys only.  Keys are column names in ``/units`` (e.g.
        ``{"default_qc": True, "decoder_label": "sua"}``).
    plane_filter : list[str], optional
        Mesoscope only.  Subset of imaging plane names (e.g. ``["VISp_0"]``).
    dmd_filter : list[str], optional
        SLAP2 only.  Subset of ``["DMD1", "DMD2"]``.
    signal_type : str
        Mesoscope / SLAP2 only.  Which fluorescence signal to extract.
        Mesoscope options: ``"dff"`` (default), ``"events"``,
        ``"neuropil_corrected"``, ``"neuropil"``, ``"raw"``.
        SLAP2 always uses dFF (parameter ignored).
    soma_only : bool
        Mesoscope only.  If True (default), keep only ROIs where ``is_soma == 1``
        in the roi_table.  Set False to include dendrites.
    baseline_window : (pre, post) or None
        Mesoscope / SLAP2 only.  Time window (relative to trial onset, seconds)
        used to compute a per-trial, per-ROI baseline for normalisation.
        Typical value: ``(-0.5, 0.0)``.  If None, no baseline correction is applied.
    baseline_mode : str
        How to apply the baseline.  ``"zscore"`` (default): subtract mean and
        divide by std — produces z-scored ΔF/F or event rates.  ``"subtract"``:
        subtract baseline mean only.  ``"divide"``: divide by baseline mean
        (useful for raw F to get a ΔF/F approximation).

    Returns
    -------
    xr.DataArray
        ``(trial, unit, time)`` with float64 values.
    """
    tech = handle.technique
    if tech == "ecephys":
        return _ecephys_responses(
            handle, trials, window, bin_size or 0.01, unit_filter,
        )
    elif tech == "mesoscope":
        return _mesoscope_responses(
            handle, trials, window, plane_filter,
            signal_type=signal_type,
            soma_only=soma_only,
            baseline_window=baseline_window,
            baseline_mode=baseline_mode,
        )
    elif tech == "slap2":
        return _slap2_responses(
            handle, trials, window, dmd_filter,
            baseline_window=baseline_window,
            baseline_mode=baseline_mode,
        )
    else:
        raise ValueError(f"Unknown technique {tech!r}")


# ── Ecephys: binned spike counts ─────────────────────────────────────

def _ecephys_responses(handle, trials, window, bin_size, unit_filter):
    h5 = handle.h5
    units_grp = h5["units"]

    # Read unit metadata for filtering
    n_units_total = units_grp["id"].shape[0]
    keep = np.ones(n_units_total, dtype=bool)

    if unit_filter:
        for col, val in unit_filter.items():
            if col in units_grp:
                col_data = units_grp[col][:]
                # Handle byte-strings: dtype "S" (fixed-length) or "O" (object with bytes)
                if col_data.dtype.kind == "S":
                    col_data = np.char.decode(col_data, "utf-8")
                elif col_data.dtype.kind == "O" and len(col_data) > 0 and isinstance(col_data[0], bytes):
                    col_data = np.array([v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in col_data])
                if isinstance(val, (list, tuple, set)):
                    keep &= np.isin(col_data, list(val))
                else:
                    keep &= col_data == val

    unit_indices = np.where(keep)[0]
    n_units = len(unit_indices)

    # Unit identifiers
    device_names = units_grp["device_name"][:][unit_indices]
    if device_names.dtype.kind == "S":
        device_names = np.char.decode(device_names, "utf-8")
    elif device_names.dtype.kind == "O" and len(device_names) > 0 and isinstance(device_names[0], bytes):
        device_names = np.array([v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in device_names])
    orig_ids = units_grp["original_cluster_id"][:][unit_indices]
    unit_ids = [f"{d}_{int(c)}" for d, c in zip(device_names, orig_ids)]

    # Build time bins
    pre, post = window
    n_bins = int(np.ceil((post - pre) / bin_size))
    time_edges = np.linspace(pre, pre + n_bins * bin_size, n_bins + 1)
    time_centers = 0.5 * (time_edges[:-1] + time_edges[1:])

    # Read spike times — this can be big; read per unit via the index
    # /units/spike_times is a ragged array stored with spike_times + spike_times_index
    spike_times_ds = units_grp["spike_times"]
    spike_times_idx = units_grp["spike_times_index"][:]

    trial_starts = trials["start_time"].values
    n_trials = len(trial_starts)

    result = np.zeros((n_trials, n_units, n_bins), dtype=np.float64)

    for j, uid in enumerate(unit_indices):
        # Ragged array: spikes for unit uid are at [prev_idx : idx]
        idx_end = int(spike_times_idx[uid])
        idx_start = int(spike_times_idx[uid - 1]) if uid > 0 else 0
        spikes = spike_times_ds[idx_start:idx_end]

        for i, t0 in enumerate(trial_starts):
            # Spikes in the window
            rel = spikes - t0
            in_win = rel[(rel >= pre) & (rel < post)]
            if len(in_win) > 0:
                counts, _ = np.histogram(in_win, bins=time_edges)
                result[i, j, :] = counts

    return xr.DataArray(
        result,
        dims=("trial", "unit", "time"),
        coords={
            "unit_id": ("unit", unit_ids),
            "trial_start": ("trial", trial_starts),
            "time_sec": ("time", time_centers),
        },
        attrs={"bin_size_s": bin_size, "technique": "ecephys"},
    )


# ── Mesoscope: fluorescence traces ─────────────────────────────────
#
# Each imaging plane lives at processing/{plane_name}/ and stores multiple
# signal types.  Relative sub-paths within each plane:
_MESO_SIGNAL_PATHS = {
    "dff":                "dff_timeseries/dff_timeseries",
    "events":             "event_timeseries",
    "neuropil_corrected": "neuropil_corrected_timeseries",
    "neuropil":           "neuropil_fluorescence_timeseries",
    "raw":                "raw_timeseries/ROI_fluorescence_timeseries",
}

# Non-imaging processing modules to skip
_MESO_SKIP_MODULES = {"running", "eye_tracking", "stimulus"}


def _mesoscope_responses(
    handle, trials, window, plane_filter,
    *, signal_type, soma_only, baseline_window, baseline_mode,
):
    h5 = handle.h5
    proc = h5["processing"]

    if signal_type not in _MESO_SIGNAL_PATHS:
        raise ValueError(
            f"signal_type {signal_type!r} unknown. "
            f"Choose from: {list(_MESO_SIGNAL_PATHS)}"
        )
    sig_subpath = _MESO_SIGNAL_PATHS[signal_type]

    # ── Collect plane data ──────────────────────────────────────────
    # plane_data entries: (plane_name, timestamps, data_ds, soma_mask)
    plane_data = []
    for key in sorted(proc.keys()):
        if key in _MESO_SKIP_MODULES:
            continue
        if plane_filter and key not in plane_filter:
            continue
        grp = proc[key]
        if not isinstance(grp, h5py.Group):
            continue

        sig_path = f"{sig_subpath}"
        if sig_path not in grp:
            continue  # this plane doesn't have the requested signal

        sig_grp = grp[sig_path]
        if "data" not in sig_grp or "timestamps" not in sig_grp:
            continue

        # ROI quality mask: is_soma == 1
        soma_mask = None
        if soma_only:
            rt_path = "image_segmentation/roi_table/is_soma"
            if rt_path in grp:
                soma_mask = grp[rt_path][:].astype(bool)

        timestamps = sig_grp["timestamps"][:]
        plane_data.append((key, timestamps, sig_grp["data"], soma_mask))

    if not plane_data:
        raise ValueError(
            f"No {signal_type!r} data found in any imaging plane. "
            f"Available planes: {[k for k in proc.keys() if k not in _MESO_SKIP_MODULES]}"
        )

    # ── Build common time grid from first plane ─────────────────────
    ref_ts = plane_data[0][1]
    dt = float(np.median(np.diff(ref_ts[:1000])))

    pre, post = window
    n_samples = int(np.ceil((post - pre) / dt))
    time_centers = np.linspace(pre + dt / 2, pre + (n_samples - 0.5) * dt, n_samples)

    # Baseline sample indices (relative to window start)
    bl_idx0 = bl_idx1 = None
    if baseline_window is not None:
        bl_pre, bl_post = baseline_window
        if bl_pre < pre or bl_post > post:
            raise ValueError(
                f"baseline_window {baseline_window} must lie within response window {window}"
            )
        bl_idx0 = int(round((bl_pre - pre) / dt))
        bl_idx1 = int(round((bl_post - pre) / dt))
        if bl_idx1 <= bl_idx0:
            raise ValueError(f"baseline_window {baseline_window} too narrow for dt={dt:.4f}s")

    trial_starts = trials["start_time"].values
    n_trials = len(trial_starts)

    # ── Allocate output — count soma-filtered ROIs ──────────────────
    roi_counts = []
    for _, _, data_ds, soma_mask in plane_data:
        n_total = data_ds.shape[1]
        if soma_mask is not None:
            roi_counts.append(int(soma_mask.sum()))
        else:
            roi_counts.append(n_total)
    total_rois = sum(roi_counts)

    result = np.full((n_trials, total_rois, n_samples), np.nan, dtype=np.float64)
    unit_ids = []

    # ── Extract per plane ───────────────────────────────────────────
    roi_offset = 0
    for (plane_name, timestamps, data_ds, soma_mask), n_rois in zip(plane_data, roi_counts):
        # Read full trace once — (T × n_rois_total), float32 → float64
        trace = data_ds[:].astype(np.float64)  # (T, n_rois_total)

        # Apply soma mask: keep only soma columns
        if soma_mask is not None:
            trace = trace[:, soma_mask]
            roi_indices = np.where(soma_mask)[0]
        else:
            roi_indices = np.arange(trace.shape[1])

        for i, t0 in enumerate(trial_starts):
            idx0 = int(np.searchsorted(timestamps, t0 + pre))
            idx1 = idx0 + n_samples
            if idx1 > trace.shape[0]:
                continue
            snippet = trace[idx0:idx1, :].T  # (n_rois, n_samples)
            result[i, roi_offset:roi_offset + n_rois, :] = snippet

        # Baseline correction (per trial, per ROI)
        if bl_idx0 is not None:
            bl_data = result[:, roi_offset:roi_offset + n_rois, bl_idx0:bl_idx1]
            bl_mean = np.nanmean(bl_data, axis=2, keepdims=True)  # (trial, roi, 1)
            bl_std  = np.nanstd(bl_data, axis=2, keepdims=True)

            roi_slice = slice(roi_offset, roi_offset + n_rois)
            if baseline_mode == "zscore":
                # Avoid division by zero for silent ROIs
                safe_std = np.where(bl_std > 1e-9, bl_std, 1.0)
                result[:, roi_slice, :] = (result[:, roi_slice, :] - bl_mean) / safe_std
            elif baseline_mode == "subtract":
                result[:, roi_slice, :] -= bl_mean
            elif baseline_mode == "divide":
                safe_mean = np.where(np.abs(bl_mean) > 1e-9, bl_mean, 1.0)
                result[:, roi_slice, :] /= safe_mean
            else:
                raise ValueError(f"baseline_mode {baseline_mode!r} unknown. "
                                 f"Choose 'zscore', 'subtract', or 'divide'.")

        unit_ids.extend([f"{plane_name}_roi{int(r)}" for r in roi_indices])
        roi_offset += n_rois

    return xr.DataArray(
        result,
        dims=("trial", "unit", "time"),
        coords={
            "unit_id": ("unit", unit_ids),
            "trial_start": ("trial", trial_starts),
            "time_sec": ("time", time_centers),
        },
        attrs={
            "sample_rate_hz": 1.0 / dt,
            "signal_type": signal_type,
            "soma_only": soma_only,
            "baseline_window": baseline_window,
            "baseline_mode": baseline_mode if baseline_window else None,
            "technique": "mesoscope",
        },
    )


# ── SLAP2: ΔF/F traces per DMD ──────────────────────────────────────

def _slap2_responses(handle, trials, window, dmd_filter, *, baseline_window, baseline_mode):
    h5 = handle.h5
    dmds = dmd_filter or ["DMD1", "DMD2"]

    dmd_data = []
    for dmd in dmds:
        ts_path = f"processing/ophys/Fluorescence_{dmd}/{dmd}_dFF"
        if ts_path not in h5:
            continue
        grp = h5[ts_path]
        timestamps = grp["timestamps"][:]
        data_ds = grp["data"]
        dmd_data.append((dmd, timestamps, data_ds))

    if not dmd_data:
        raise ValueError("No SLAP2 dFF data found")

    # Use first DMD for timing reference
    ref_ts = dmd_data[0][1]
    dt = np.median(np.diff(ref_ts[:1000]))

    pre, post = window
    n_samples = int(np.ceil((post - pre) / dt))
    time_centers = np.linspace(pre + dt / 2, pre + (n_samples - 0.5) * dt, n_samples)

    trial_starts = trials["start_time"].values
    n_trials = len(trial_starts)

    total_rois = sum(ds.shape[1] for _, _, ds in dmd_data)
    result = np.full((n_trials, total_rois, n_samples), np.nan, dtype=np.float64)
    unit_ids = []

    roi_offset = 0
    for dmd, timestamps, data_ds in dmd_data:
        n_rois = data_ds.shape[1]
        # SLAP2 traces can be large (~195 Hz × long sessions).
        # Read only the windows we need to save memory.
        for i, t0 in enumerate(trial_starts):
            idx0 = np.searchsorted(timestamps, t0 + pre)
            idx1 = idx0 + n_samples
            if idx1 <= data_ds.shape[0]:
                chunk = data_ds[idx0:idx1, :]  # (n_samples, n_rois)
                result[i, roi_offset:roi_offset + n_rois, :] = chunk.T

        unit_ids.extend([f"{dmd}_roi{r}" for r in range(n_rois)])
        roi_offset += n_rois

    # Baseline correction
    if baseline_window is not None:
        bl_pre, bl_post = baseline_window
        bl_idx0 = int(round((bl_pre - pre) / dt))
        bl_idx1 = int(round((bl_post - pre) / dt))
        bl_data = result[:, :, bl_idx0:bl_idx1]
        bl_mean = np.nanmean(bl_data, axis=2, keepdims=True)
        bl_std  = np.nanstd(bl_data, axis=2, keepdims=True)
        if baseline_mode == "zscore":
            safe_std = np.where(bl_std > 1e-9, bl_std, 1.0)
            result = (result - bl_mean) / safe_std
        elif baseline_mode == "subtract":
            result -= bl_mean
        elif baseline_mode == "divide":
            safe_mean = np.where(np.abs(bl_mean) > 1e-9, bl_mean, 1.0)
            result /= safe_mean

    return xr.DataArray(
        result,
        dims=("trial", "unit", "time"),
        coords={
            "unit_id": ("unit", unit_ids),
            "trial_start": ("trial", trial_starts),
            "time_sec": ("time", time_centers),
        },
        attrs={
            "sample_rate_hz": 1.0 / dt,
            "baseline_window": baseline_window,
            "baseline_mode": baseline_mode if baseline_window else None,
            "technique": "slap2",
        },
    )


import h5py  # noqa: E402  (needed at module level for mesoscope adapter)
