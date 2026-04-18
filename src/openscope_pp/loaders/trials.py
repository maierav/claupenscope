"""Build a common trials DataFrame from any technique's NWB file.

The three techniques store stimulus intervals differently:

- **Ecephys / Mesoscope**: one named ``TimeIntervals`` group per stimulus
  block under ``/intervals`` (e.g. ``Sequence mismatch block_presentations``).
  Block identity is the group name.  The column schema is shared but ecephys
  stores every value as byte-strings while mesoscope stores floats.

- **SLAP2 pilots**: a single ``/intervals/gratings`` table.  Block boundaries
  are inferred from ``stim_id`` transitions and long ISIs.

``load_trials`` dispatches to the right adapter based on
``handle.technique`` and returns a single DataFrame with a uniform schema.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from openscope_pp.loaders.streaming import NWBHandle


# ── Shared column schema ─────────────────────────────────────────────
# After adapter runs, every trials DataFrame will have at least these columns.
# Adapters may add extras (e.g. ``sequence_number``).
_REQUIRED_COLS = [
    "start_time",
    "stop_time",
    "block",            # human-readable block label
    "block_kind",       # machine category: paradigm_oddball, rf_mapping, movie, …
    "paradigm",         # SENSORYMOTOR / STANDARD / SEQUENCE / DURATION / None
    "trial_type",       # TrialType string (standard, halt, omission, …)
    "is_deviant",       # bool
    "orientation",      # radians
    "spatial_frequency",
    "temporal_frequency",
    "contrast",
    "x",
    "y",
    "diameter",
]


# ── Public entry point ───────────────────────────────────────────────

def load_trials(handle: "NWBHandle") -> pd.DataFrame:
    """Return a trials DataFrame with the common schema.

    Parameters
    ----------
    handle : NWBHandle
        From :func:`~openscope_pp.loaders.streaming.open_nwb`.

    Returns
    -------
    pd.DataFrame
        Rows = individual stimulus presentations, sorted by ``start_time``.
    """
    tech = handle.technique
    if tech == "ecephys":
        df = _from_named_intervals(handle, decode_bytes=True)
    elif tech == "mesoscope":
        df = _from_named_intervals(handle, decode_bytes=False)
    elif tech == "slap2":
        df = _from_slap2_gratings(handle)
    else:
        raise ValueError(f"Unknown technique {tech!r}")

    df.sort_values("start_time", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ── Ecephys / Mesoscope adapter ─────────────────────────────────────

# Mapping from BlockType → block_kind for the common schema.
_BLOCK_TYPE_MAP = {
    "sequential_oddball": "paradigm_oddball",
    "sensorimotor_closed_loop": "paradigm_oddball",
    "motor_oddball": "paradigm_oddball",
    "duration_oddball": "paradigm_oddball",
    "standard_oddball": "paradigm_oddball",
    "open_loop_prerecorded": "control_replay",
    "rf_mapping": "rf_mapping",
    "movie": "movie",
    "trippy": "movie",
    "zebra": "movie",
    "spontaneous": "spontaneous",
    "None or Blank": "blank",
    "": "blank",
}

# BlockLabel substrings → paradigm assignment.
_LABEL_TO_PARADIGM = {
    "Sequence mismatch": "SEQUENCE",
    "Sensory-motor mismatch": "SENSORYMOTOR",
    "Duration mismatch": "DURATION",
    "Standard mismatch": "STANDARD",
    # Control blocks inherit paradigm from their TrialType prefixes later
}

# Columns to extract from the NWB intervals groups.
_NUMERIC_COLS = [
    "Orientation", "SpatialFrequency", "TemporalFrequency",
    "contrast", "X", "Y", "DiameterX", "DiameterY",
    "SequenceNumber", "TrialInSequence", "Duration", "Delay",
    "BlockNumber",
]
_STR_COLS = [
    "BlockLabel", "BlockType", "TrialType", "stim_name", "stim_type",
]


def _safe_read(group, col, n, decode_bytes):
    """Read a column from an h5py group, handling missing cols and byte decoding."""
    if col not in group:
        return np.full(n, np.nan) if col in _NUMERIC_COLS else np.full(n, "", dtype=object)
    raw = group[col][:]
    if decode_bytes and raw.dtype.kind == "S":
        raw = np.char.decode(raw, "utf-8")
    return raw


def _to_float_safe(arr):
    """Convert an array to float, returning NaN for un-parseable values."""
    try:
        return arr.astype(np.float64)
    except (ValueError, TypeError):
        out = np.empty(len(arr), dtype=np.float64)
        for i, v in enumerate(arr):
            try:
                out[i] = float(v)
            except (ValueError, TypeError):
                out[i] = np.nan
        return out


def _infer_paradigm(block_label: str, block_type: str) -> str | None:
    """Return canonical paradigm name or None."""
    for substr, paradigm in _LABEL_TO_PARADIGM.items():
        if substr in block_label:
            return paradigm
    # Control blocks that replay a paradigm
    if "Control" in block_label and block_type == "open_loop_prerecorded":
        return None  # control, not a paradigm itself
    return None


def _from_named_intervals(handle: "NWBHandle", decode_bytes: bool) -> pd.DataFrame:
    """Build trials from ecephys/mesoscope named interval groups."""
    h5 = handle.h5
    intervals = h5["intervals"]
    frames = []

    for group_name in intervals.keys():
        g = intervals[group_name]

        # Skip groups that aren't visual-stim intervals (e.g. optotagging)
        if not isinstance(g, h5py.Group):
            continue
        if "start_time" not in g:
            continue

        n = g["start_time"].shape[0]
        st = g["start_time"][:]
        sp = g["stop_time"][:]

        # Read metadata columns
        block_label = _safe_read(g, "BlockLabel", n, decode_bytes)
        block_type = _safe_read(g, "BlockType", n, decode_bytes)
        trial_type = _safe_read(g, "TrialType", n, decode_bytes)

        # Build the per-row block label — use the group name since it's unique
        block_clean = group_name.replace("_presentations", "")

        # Detect if this is an optotagging block (different schema entirely)
        if "condition" in g and "BlockLabel" not in g:
            # Optotagging block — include with block_kind="optotagging"
            row = {
                "start_time": st,
                "stop_time": sp,
                "block": np.full(n, block_clean, dtype=object),
                "block_kind": np.full(n, "optotagging", dtype=object),
                "paradigm": np.full(n, None, dtype=object),
                "trial_type": np.full(n, "optotagging", dtype=object),
                "is_deviant": np.zeros(n, dtype=bool),
                "orientation": np.full(n, np.nan),
                "spatial_frequency": np.full(n, np.nan),
                "temporal_frequency": np.full(n, np.nan),
                "contrast": np.full(n, np.nan),
                "x": np.full(n, np.nan),
                "y": np.full(n, np.nan),
                "diameter": np.full(n, np.nan),
            }
            frames.append(pd.DataFrame(row))
            continue

        # Standard visual-stim block
        # Ensure we have clean strings (not b'...' from byte arrays)
        def _clean_str(arr):
            v = arr[0] if len(arr) > 0 else ""
            if isinstance(v, bytes):
                return v.decode("utf-8")
            s = str(v)
            if s.startswith("b'") and s.endswith("'"):
                return s[2:-1]
            return s

        bl_str = _clean_str(block_label) if n > 0 else ""
        bt_str = _clean_str(block_type) if n > 0 else ""
        paradigm = _infer_paradigm(bl_str, bt_str)
        bk = _BLOCK_TYPE_MAP.get(bt_str, bt_str)

        # Read numeric stim params
        ori = _to_float_safe(_safe_read(g, "Orientation", n, decode_bytes))
        sf = _to_float_safe(_safe_read(g, "SpatialFrequency", n, decode_bytes))
        tf = _to_float_safe(_safe_read(g, "TemporalFrequency", n, decode_bytes))
        con = _to_float_safe(_safe_read(g, "contrast", n, decode_bytes))
        x = _to_float_safe(_safe_read(g, "X", n, decode_bytes))
        y = _to_float_safe(_safe_read(g, "Y", n, decode_bytes))
        dx = _to_float_safe(_safe_read(g, "DiameterX", n, decode_bytes))

        # TrialType → is_deviant
        def _clean_val(v):
            if isinstance(v, bytes):
                return v.decode("utf-8")
            s = str(v)
            if s.startswith("b'") and s.endswith("'"):
                return s[2:-1]
            return s
        tt_arr = np.array([_clean_val(v) for v in trial_type], dtype=object)
        is_dev = np.array([t not in ("standard", "prerecorded", "single",
                                      "rf_mapping", "spontaneous", "")
                           for t in tt_arr])

        row = {
            "start_time": st,
            "stop_time": sp,
            "block": np.full(n, block_clean, dtype=object),
            "block_kind": np.full(n, bk, dtype=object),
            "paradigm": np.full(n, paradigm, dtype=object),
            "trial_type": tt_arr,
            "is_deviant": is_dev,
            "orientation": ori,
            "spatial_frequency": sf,
            "temporal_frequency": tf,
            "contrast": con,
            "x": x,
            "y": y,
            "diameter": dx,
        }

        # Add sequence columns if present
        seq = _safe_read(g, "SequenceNumber", n, decode_bytes)
        tis = _safe_read(g, "TrialInSequence", n, decode_bytes)
        row["sequence_number"] = _to_float_safe(seq)
        row["trial_in_sequence"] = _to_float_safe(tis)

        frames.append(pd.DataFrame(row))

    if not frames:
        return pd.DataFrame(columns=_REQUIRED_COLS)
    return pd.concat(frames, ignore_index=True)


# ── SLAP2 pilot adapter ─────────────────────────────────────────────

import h5py  # noqa: E402 (already imported at top, but keeps the section self-contained)
from collections import Counter as _Counter

# SLAP2 pilot sessions store all stimulus presentations in a single
# ``intervals/gratings`` table with no explicit block labels.
#
# Block structure varies between sessions.  Examples:
#
#   sub-803496: Ori1(60) → OB1(~1500) → Ori2(33) → OB2(~1500) → Ori3(33) → RF(~150)
#   sub-776270: OB1(960) → Ori1(1533) → OB2(528) → Ori2(1533) → OB3(528) → RF(1470)
#
# Detection uses three heuristics applied sequentially:
#   1. RF mapping: small diameter (< 30°) OR multiple (x,y) positions
#   2. Oddball:    "low-entropy" orientation — contiguous stretch with ≤3
#                  unique orientations (mod 180°) where one dominates ≥80%.
#   3. Orientation tuning: everything else (typically short blocks with many
#                  orientations or many stim_ids, placed between oddball blocks)
#
# Within oddball blocks:
#   - The dominant orientation is "standard"
#   - contrast == 0 trials are "omission"  (blank screen)
#   - temporal_frequency == 0 trials are "static"
#   - everything else is a deviant, labeled by its orientation in degrees
#
# Per-DMD timing offsets exist (e.g. DMD1 +115 ms, DMD2 −165 ms on sub-803496)
# and must be applied during response extraction, not here.

_RF_DIAMETER_THRESHOLD = 30.0
_ODDBALL_MAX_UNIQUE_ORI = 3        # ≤ 3 orientations (mod 180°)
_ODDBALL_MIN_DOMINANT_FRAC = 0.80  # dominant must be ≥ 80%
_ODDBALL_MIN_LENGTH = 200          # minimum trials to qualify as oddball block


def _find_oddball_spans(ori_deg: np.ndarray) -> list[tuple[int, int]]:
    """Find ALL contiguous oddball-like spans via a low-entropy sweep.

    Returns a list of (start, end_inclusive) index pairs, non-overlapping,
    sorted by length descending then position ascending.  Spans that are
    subsets of a longer span are discarded.
    """
    n = len(ori_deg)
    candidates = []
    i = 0
    while i < n:
        counts: dict[int, int] = {}
        best_j = None
        j = i
        while j < n:
            v = int(ori_deg[j])
            counts[v] = counts.get(v, 0) + 1
            if len(counts) > _ODDBALL_MAX_UNIQUE_ORI:
                break
            total = j - i + 1
            dom_count = max(counts.values())
            if total >= _ODDBALL_MIN_LENGTH and (dom_count / total) >= _ODDBALL_MIN_DOMINANT_FRAC:
                best_j = j
            j += 1
        if best_j is not None:
            candidates.append((i, best_j))
            i = best_j + 1  # skip past this span
        else:
            i += 1

    # Remove spans that are subsets of others (keep longest first)
    candidates.sort(key=lambda x: -(x[1] - x[0]))
    kept = []
    for s, e in candidates:
        if not any(s >= ks and e <= ke for ks, ke in kept):
            kept.append((s, e))
    kept.sort()
    return kept


def _from_slap2_gratings(handle: "NWBHandle") -> pd.DataFrame:
    """Build trials from the monolithic ``intervals/gratings`` table."""
    h5 = handle.h5
    g = h5["intervals/gratings"]
    n = g["start_time"].shape[0]

    st = g["start_time"][:]
    sp = g["stop_time"][:]
    ori = g["orientation"][:]
    sf = g["spatial_frequency"][:]
    tf = g["temporal_frequency"][:]
    con = g["contrast"][:]
    dia = g["diameter"][:]
    x = g["x"][:]
    y = g["y"][:]
    sid = g["stim_id"][:] if "stim_id" in g else np.zeros(n, dtype=int)
    phase = g["phase"][:] if "phase" in g else np.full(n, np.nan)

    # Orientation in integer degrees (mod 180) for block detection
    ori_deg = (np.degrees(ori) % 180).round().astype(int)

    # ── Step 1: flag RF mapping rows (small diameter or multi-position) ──
    n_unique_pos = len(set(zip(x, y)))
    is_rf = dia < _RF_DIAMETER_THRESHOLD
    # If there are multiple positions but large diameter, also check per-row
    # (some sessions use (x,y) != 0 to flag RF)
    if not is_rf.any() and n_unique_pos > 4:
        is_rf = (x != 0) | (y != 0)

    # ── Step 2: find oddball spans in the non-RF rows ───────────────
    # Build a mapping: non-RF row indices → local array
    non_rf_idx = np.where(~is_rf)[0]
    non_rf_ori_deg = ori_deg[non_rf_idx]

    oddball_local_spans = _find_oddball_spans(non_rf_ori_deg)

    # Map local spans back to global indices
    oddball_global_sets: list[set[int]] = []
    for ls, le in oddball_local_spans:
        oddball_global_sets.append(set(non_rf_idx[ls:le + 1].tolist()))

    # ── Step 3: classify every row ──────────────────────────────────
    block_labels = np.empty(n, dtype=object)
    block_kinds = np.empty(n, dtype=object)
    paradigms = np.full(n, None, dtype=object)
    trial_types = np.empty(n, dtype=object)
    is_deviant = np.zeros(n, dtype=bool)

    # Mark RF rows
    block_labels[is_rf] = "RF mapping"
    block_kinds[is_rf] = "rf_mapping"
    trial_types[is_rf] = "rf_mapping"

    # Mark oddball rows
    for ob_idx, ob_set in enumerate(oddball_global_sets, start=1):
        ob_mask = np.array([i in ob_set for i in range(n)])
        ob_rows = np.where(ob_mask)[0]
        ob_ori_deg = ori_deg[ob_rows]
        ob_con = con[ob_rows]
        ob_tf = tf[ob_rows]

        # Find dominant orientation
        _ctr = _Counter(ob_ori_deg.tolist())
        dominant_deg = _ctr.most_common(1)[0][0]

        block_labels[ob_mask] = f"Oddball {ob_idx}"
        block_kinds[ob_mask] = "paradigm_oddball"

        for j, gi in enumerate(ob_rows):
            if np.isfinite(ob_con[j]) and ob_con[j] == 0.0:
                trial_types[gi] = "omission"
                is_deviant[gi] = True
            elif ob_tf[j] == 0.0:
                trial_types[gi] = "static"
                is_deviant[gi] = True
            elif ob_ori_deg[j] == dominant_deg:
                trial_types[gi] = "standard"
                is_deviant[gi] = False
            else:
                trial_types[gi] = str(ob_ori_deg[j])
                is_deviant[gi] = True

    # Mark remaining rows as orientation tuning
    labeled = is_rf.copy()
    for ob_set in oddball_global_sets:
        for i in ob_set:
            labeled[i] = True

    unlabeled = ~labeled
    if unlabeled.any():
        # Group contiguous unlabeled stretches
        tuning_counter = 0
        in_block = False
        for i in range(n):
            if unlabeled[i]:
                if not in_block:
                    tuning_counter += 1
                    in_block = True
                block_labels[i] = f"Orientation tuning {tuning_counter}"
                block_kinds[i] = "ori_tuning"
                trial_types[i] = "ori_tuning"
            else:
                in_block = False

    return pd.DataFrame({
        "start_time": st,
        "stop_time": sp,
        "block": block_labels,
        "block_kind": block_kinds,
        "paradigm": paradigms,
        "trial_type": trial_types,
        "is_deviant": is_deviant,
        "orientation": ori,
        "orientation_deg": ori_deg,
        "spatial_frequency": sf,
        "temporal_frequency": tf,
        "contrast": con,
        "x": x,
        "y": y,
        "diameter": dia,
        "stim_id": sid,
    })
