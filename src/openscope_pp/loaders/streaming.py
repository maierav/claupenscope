"""Stream NWB files from DANDI without downloading.

The main entry point is ``open_nwb``, which takes a DANDI asset ID (or a
full URL, or a dandiset_id + asset_id pair) and returns a lightweight handle
that keeps the remote HDF5 file open for lazy reads.

Usage
-----
>>> handle = open_nwb("55babc82-9551-4df7-b64f-572d6ec21415")
>>> handle.h5["intervals"].keys()
>>> handle.close()

Or as a context manager:
>>> with open_nwb(asset_id="55babc82-...", dandiset_id="001768") as h:
...     print(h.nwb.session_start_time)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import h5py
import pynwb
import remfile
import requests


@dataclass
class NWBHandle:
    """Thin wrapper around a streaming NWB file.

    Attributes
    ----------
    h5 : h5py.File
        Raw HDF5 handle — use when pynwb is awkward (e.g. byte-string decoding).
    nwb : pynwb.NWBFile
        High-level pynwb object — use for structured access (subject, devices, …).
    io : pynwb.NWBHDF5IO
        Keep alive; closing it invalidates *nwb*.
    url : str
        Resolved S3 URL that *remfile* is streaming from.
    asset_id : str
        DANDI asset UUID.
    technique : str
        One of ``"ecephys"``, ``"mesoscope"``, ``"slap2"`` — auto-detected from
        the NWB contents (see :func:`_detect_technique`).
    """
    h5: h5py.File
    nwb: pynwb.NWBFile
    io: pynwb.NWBHDF5IO
    url: str
    asset_id: str
    technique: str
    _rf: remfile.File = field(repr=False)

    # ------------------------------------------------------------------
    def close(self):
        try:
            self.io.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def open_nwb(
    asset_id: str,
    dandiset_id: Optional[str] = None,
    *,
    technique: Optional[str] = None,
) -> NWBHandle:
    """Stream an NWB file from DANDI and return a ready-to-use handle.

    Parameters
    ----------
    asset_id : str
        DANDI asset UUID.
    dandiset_id : str, optional
        Not required for resolution (the asset_id is globally unique on DANDI),
        but stored for provenance.
    technique : str, optional
        Override auto-detection.  One of ``"ecephys"``, ``"mesoscope"``,
        ``"slap2"``.

    Returns
    -------
    NWBHandle
    """
    url = _resolve_url(asset_id)
    rf = remfile.File(url)
    h5 = h5py.File(rf, "r")
    io = pynwb.NWBHDF5IO(file=h5, load_namespaces=True, mode="r")
    nwb = io.read()
    tech = technique or _detect_technique(h5, nwb)
    return NWBHandle(
        h5=h5, nwb=nwb, io=io, url=url,
        asset_id=asset_id, technique=tech, _rf=rf,
    )


# ------------------------------------------------------------------
# Internals
# ------------------------------------------------------------------

def _resolve_url(asset_id: str) -> str:
    """Follow the DANDI download redirect to get the signed S3 URL.

    The signed S3 URL itself returns 403 on HEAD (expected — S3 signed URLs
    are method-specific), so we follow the redirect chain manually and
    extract the final Location without hitting S3 with HEAD.
    """
    dl = f"https://api.dandiarchive.org/api/assets/{asset_id}/download/"
    resp = requests.get(dl, allow_redirects=True, stream=True, timeout=30)
    url = resp.url
    resp.close()
    return url


def _detect_technique(h5: h5py.File, nwb: pynwb.NWBFile) -> str:
    """Guess the recording technique from NWB contents.

    Heuristics (in order):
      1. ``/units`` exists → ecephys
      2. ``/processing/ophys`` has ``Fluorescence_DMD1`` → slap2
      3. ``/processing/ophys`` exists → mesoscope
      4. Fall back to ``"unknown"``
    """
    if "units" in h5:
        return "ecephys"
    proc = h5.get("processing")
    if proc is not None:
        proc_keys = list(proc.keys())
        # SLAP2: has a processing/ophys group with DMD subgroups
        ophys = proc.get("ophys")
        if ophys is not None and any("DMD" in k for k in ophys.keys()):
            return "slap2"
        # Mesoscope: imaging-plane modules directly under processing/
        # (named like VISp_0, VISl_4, etc.)
        if any(k.startswith(("VIS", "ophys")) for k in proc_keys):
            return "mesoscope"
    return "unknown"
