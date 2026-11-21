"""FAZ from a vessel mask: the large capillary-free region near the centre.

Method. The background of the vessel mask is the set of capillary-free pixels.
Its distance transform gives, at each pixel, the radius of the largest
vessel-free disc centred there. The FAZ contains the point with the largest
such radius inside a central search disc. To stop the region leaking through
small gaps in the capillary ring, only background with at least a fraction
of that radius is kept (an opening) and the result is then grown back
inside the background by the same distance.

It needs a vessel segmentation but no intensity thresholds of its own, and
it works with the output of either vessel segmenter in this package.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import disk

from .._morph import fill_small_holes
from ..preprocessing.geometry import pixel_size_um


@dataclass
class FazResult:
    mask: np.ndarray
    found: bool
    center_px: tuple[float, float] | None = None
    max_inscribed_radius_um: float | None = None
    reason: str = ""


def faz_from_vessels(vessels: np.ndarray, fov_mm: float, *, center_px: tuple[float, float] | None = None,
                     search_radius_mm: float = 0.9, min_radius_um: float = 60.0,
                     leak_fraction: float = 0.3) -> FazResult:
    """Find the FAZ in a binary vessel mask.

    Args:
        center_px: (row, col) of the expected fovea; defaults to the image centre.
        search_radius_mm: how far from the centre to look for the FAZ core.
        min_radius_um: smallest inscribed radius accepted as a FAZ. Below this
            the largest vessel-free region is just an intercapillary gap.
        leak_fraction: share of the core radius used to close leaks.
    """
    h, w = vessels.shape
    px = pixel_size_um(vessels.shape, fov_mm)
    bg = ~vessels.astype(bool)
    dist = ndi.distance_transform_edt(bg)

    cy, cx = center_px if center_px else ((h - 1) / 2, (w - 1) / 2)
    yy, xx = np.mgrid[:h, :w]
    near = np.hypot(yy - cy, xx - cx) <= search_radius_mm * 1000.0 / px
    if not near.any():
        return FazResult(np.zeros_like(bg), False, reason="search region is outside the image")

    d_near = np.where(near, dist, 0)
    peak = np.unravel_index(np.argmax(d_near), dist.shape)
    r_core = float(dist[peak])
    if r_core * px < min_radius_um:
        return FazResult(np.zeros_like(bg), False, tuple(map(float, peak)), r_core * px,
                         "no vessel-free region large enough")

    r_keep = max(1.0, leak_fraction * r_core)
    core = dist >= r_keep
    lab, _ = ndi.label(core)
    core = lab == lab[peak]
    k = int(np.ceil(r_keep))
    grown = ndi.binary_dilation(core, structure=disk(k)) & bg
    grown = fill_small_holes(grown, 200)
    lab, _ = ndi.label(grown)
    grown = lab == lab[peak]
    grown = ndi.binary_opening(ndi.binary_closing(grown, disk(2)), disk(2))
    lab, _ = ndi.label(grown)
    if lab[peak] == 0:
        return FazResult(np.zeros_like(bg), False, tuple(map(float, peak)), r_core * px, "region vanished")
    mask = lab == lab[peak]
    return FazResult(mask, True, tuple(map(float, peak)), r_core * px)
