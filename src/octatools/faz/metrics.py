"""Shape measures for a FAZ mask."""
from __future__ import annotations

import numpy as np
from skimage.measure import label, perimeter_crofton, regionprops


def faz_metrics(mask: np.ndarray, fov_mm: float) -> dict[str, float]:
    """Area, perimeter and shape descriptors of the largest connected region.

    area_mm2          pixel count times pixel area
    perimeter_mm      Crofton estimate (less biased than counting boundary pixels)
    circularity       4*pi*A / P^2, 1 for a disc
    acircularity      P / (2*sqrt(pi*A)), 1 for a disc, larger for irregular outlines
                      (acircularity index, Krawitz et al. 2017)
    equiv_diameter_mm diameter of the disc with the same area
    axis_ratio        major / minor axis of the fitted ellipse
    solidity          area / convex hull area
    centroid_*_mm     offset of the centroid from the image centre (x right, y down)
    """
    lab = label(mask.astype(bool))
    if lab.max() == 0:
        return {k: float("nan") for k in (
            "area_mm2", "perimeter_mm", "circularity", "acircularity", "equiv_diameter_mm",
            "axis_ratio", "solidity", "centroid_x_mm", "centroid_y_mm")}
    sizes = np.bincount(lab.ravel())[1:]
    region = regionprops((lab == (np.argmax(sizes) + 1)).astype(np.uint8))[0]
    mm = fov_mm / mask.shape[1]
    area = region.area * mm ** 2
    per = float(perimeter_crofton(region.image, directions=4)) * mm
    h, w = mask.shape
    cy, cx = region.centroid
    return {
        "area_mm2": float(area),
        "perimeter_mm": per,
        "circularity": float(min(1.0, 4 * np.pi * area / per ** 2)) if per > 0 else float("nan"),
        "acircularity": float(max(1.0, per / (2 * np.sqrt(np.pi * area)))) if area > 0 else float("nan"),
        "equiv_diameter_mm": float(2 * np.sqrt(area / np.pi)),
        "axis_ratio": float(region.axis_major_length / max(region.axis_minor_length, 1e-6)),
        "solidity": float(region.solidity),
        "centroid_x_mm": float((cx - (w - 1) / 2) * mm),
        "centroid_y_mm": float((cy - (h - 1) / 2) * mm),
    }


def scale_area_by_axial_length(area_mm2: float, axial_length_mm: float,
                               reference_axial_length_mm: float) -> float:
    """Correct a lateral measurement for ocular magnification (Bennett et al. 1994).

    The device assumes a model eye with axial length `reference_axial_length_mm`
    (see the device manual). With q = 0.01306 * (AL - 1.82) mm per degree, lengths
    scale by q_eye / q_model and areas by the square of that. Linderman et al.
    (2017) and Sampson et al. (2017) show why this matters for FAZ area and
    vessel density in eyes far from average length.
    """
    q = lambda al: 0.01306 * (al - 1.82)
    return float(area_mm2 * (q(axial_length_mm) / q(reference_axial_length_mm)) ** 2)
