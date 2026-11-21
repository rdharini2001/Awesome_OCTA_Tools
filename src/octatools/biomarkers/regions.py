"""Analysis regions centred on the fovea (ETDRS-style rings)."""
from __future__ import annotations

import numpy as np


def etdrs_regions(shape: tuple[int, int], fov_mm: float, center_px: tuple[float, float] | None = None,
                  *, eye: str | None = None, nasal_side: str = "auto") -> dict[str, np.ndarray]:
    """Boolean masks for the whole image, the central 1 mm disc and the 1-3 / 3-6 mm rings.

    Names: whole, fovea (0-0.5 mm radius), parafovea (0.5-1.5 mm radius),
    perifovea (1.5-3 mm radius, only where the scan reaches). If `eye` is
    'OD' or 'OS' the parafovea is also split into superior / inferior /
    nasal / temporal quadrants.

    Image axes: x to the right, y downwards, superior at the top. For
    `nasal_side='auto'` the nasal side is right for OD and left for OS, which
    is how fundus-style en-face images are usually displayed. Devices differ,
    so check yours and pass 'left' or 'right' explicitly if needed.
    """
    h, w = shape
    cy, cx = center_px if center_px else ((h - 1) / 2, (w - 1) / 2)
    yy, xx = np.mgrid[:h, :w]
    mm = fov_mm / w
    r = np.hypot(yy - cy, xx - cx) * mm
    out = {
        "whole": np.ones(shape, bool),
        "fovea": r <= 0.5,
        "parafovea": (r > 0.5) & (r <= 1.5),
    }
    if fov_mm > 3.0:
        out["perifovea"] = (r > 1.5) & (r <= 3.0)

    if eye:
        eye = eye.upper()
        if eye not in ("OD", "OS"):
            raise ValueError("eye must be 'OD' or 'OS'")
        side = nasal_side if nasal_side != "auto" else ("right" if eye == "OD" else "left")
        dy, dx = yy - cy, xx - cx
        sector = {
            "superior": (-dy > np.abs(dx)),
            "inferior": (dy > np.abs(dx)),
            "right": (dx > np.abs(dy)),
            "left": (-dx > np.abs(dy)),
        }
        para = out["parafovea"]
        other = "left" if side == "right" else "right"
        out["parafovea_superior"] = para & sector["superior"]
        out["parafovea_inferior"] = para & sector["inferior"]
        out["parafovea_nasal"] = para & sector[side]
        out["parafovea_temporal"] = para & sector[other]
    return out
