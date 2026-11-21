"""Vessel biomarkers from a binary vessel mask.

Naming differs between vendors and papers. What this module reports:

vessel_area_density     fraction of the region covered by vessel pixels. Called
                        vessel density (Optovue) or perfusion density (Zeiss).
vessel_skeleton_density skeleton pixels over region pixels (Chu et al. 2016, VSD).
vessel_length_density   skeleton length in mm per mm^2 of region.
mean_vessel_width_um    vessel area divided by skeleton length, a vessel
                        diameter index. It is biased by the image resolution
                        and by whatever threshold produced the mask.
fractal_dimension       box-counting dimension of the mask.
tortuosity              mean arc length over chord length of skeleton branches
                        longer than `min_branch_um`.
"""
from __future__ import annotations

import numpy as np

from .. import skeleton as sk
from .regions import etdrs_regions


def box_counting_dimension(mask: np.ndarray, min_box: int = 2) -> float:
    """Box-counting (Minkowski-Bouligand) dimension of a binary image.

    Counts occupied boxes at scales from `min_box` pixels up to a quarter of
    the shorter side, then fits log N against log(1/size).
    """
    m = mask.astype(bool)
    if m.sum() < 8:
        return float("nan")
    n = min(m.shape)
    sizes = np.unique(np.round(np.geomspace(min_box, max(min_box + 1, n // 4), 8)).astype(int))
    counts = []
    for s in sizes:
        hh, ww = (m.shape[0] // s) * s, (m.shape[1] // s) * s
        blocks = m[:hh, :ww].reshape(hh // s, s, ww // s, s).any(axis=(1, 3))
        counts.append(blocks.sum())
    counts = np.array(counts, float)
    ok = counts > 0
    if ok.sum() < 3:
        return float("nan")
    return float(np.polyfit(np.log(1.0 / sizes[ok]), np.log(counts[ok]), 1)[0])


def vessel_metrics(mask: np.ndarray, fov_mm: float, *, roi: np.ndarray | None = None,
                   exclude: np.ndarray | None = None, min_branch_um: float = 40.0,
                   with_topology: bool = True) -> dict[str, float]:
    """Biomarkers inside `roi`, leaving out `exclude` (typically the FAZ).

    The skeleton is computed on the whole mask first so vessels are not cut at
    the region border, and only afterwards restricted to the region.
    """
    mask = mask.astype(bool)
    region = np.ones(mask.shape, bool) if roi is None else roi.astype(bool)
    if exclude is not None:
        region = region & ~exclude.astype(bool)
    n_region = int(region.sum())
    if n_region == 0:
        raise ValueError("empty region")
    mm = fov_mm / mask.shape[1]
    area_mm2 = n_region * mm ** 2

    skel = sk.skeleton(mask)
    skel_r = skel & region
    length_mm = sk.length_px(skel_r) * mm

    out = {
        "region_area_mm2": area_mm2,
        "vessel_area_density": float((mask & region).sum() / n_region),
        "vessel_skeleton_density": float(skel_r.sum() / n_region),
        "vessel_length_density_mm_per_mm2": float(length_mm / area_mm2),
        "mean_vessel_width_um": float((mask & region).sum() * mm ** 2 / length_mm * 1000) if length_mm > 0 else float("nan"),
        "fractal_dimension": box_counting_dimension(mask & region),
    }
    if with_topology:
        brs, junc = sk.branches(skel_r)
        min_px = min_branch_um / 1000.0 / mm
        tort = [b.length / b.chord for b in brs if b.length >= min_px and b.chord > 0]
        out.update({
            "n_branches": float(len(brs)),
            "n_junction_pixels": float(junc.sum()),
            "tortuosity": float(np.mean(tort)) if tort else float("nan"),
        })
    return out


def quantify(mask: np.ndarray, fov_mm: float, *, faz: np.ndarray | None = None,
             center_px: tuple[float, float] | None = None, eye: str | None = None,
             nasal_side: str = "auto", with_topology: bool = False) -> dict[str, dict[str, float]]:
    """Metrics for the whole image and each ETDRS region.

    Every region is also reported with the FAZ removed ('<region>_noFAZ') when a
    FAZ mask is given, since the avascular zone itself pulls density down.
    """
    regions = etdrs_regions(mask.shape, fov_mm, center_px, eye=eye, nasal_side=nasal_side)
    result = {}
    for name, roi in regions.items():
        if not roi.any():
            continue
        result[name] = vessel_metrics(mask, fov_mm, roi=roi, with_topology=with_topology)
        if faz is not None and (roi & ~faz.astype(bool)).any() and name != "fovea":
            result[f"{name}_noFAZ"] = vessel_metrics(mask, fov_mm, roi=roi, exclude=faz,
                                                      with_topology=with_topology)
    return result
