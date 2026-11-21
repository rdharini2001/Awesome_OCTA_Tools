"""Small binary-morphology helpers (kept local so scikit-image API changes cannot break them)."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def remove_small(mask: np.ndarray, min_size: int, connectivity: int = 1) -> np.ndarray:
    """Drop connected components with fewer than `min_size` pixels."""
    mask = mask.astype(bool)
    if min_size <= 1 or not mask.any():
        return mask
    st = ndi.generate_binary_structure(2, connectivity)
    lab, n = ndi.label(mask, structure=st)
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    keep = sizes >= min_size
    keep[0] = False
    return keep[lab]


def fill_small_holes(mask: np.ndarray, max_area: int) -> np.ndarray:
    """Fill background holes of at most `max_area` pixels that do not touch the border."""
    mask = mask.astype(bool)
    lab, n = ndi.label(~mask)
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    edge = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    fill = sizes <= max_area
    fill[0] = False
    fill[edge] = False
    return mask | fill[lab]
