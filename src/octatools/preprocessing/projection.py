"""Projection-artifact suppression for a pair of en-face slabs.

True projection removal works on the volume, A-scan by A-scan (Zhang et al.
2016, Campbell et al. 2017). If you only have exported slabs, a regression of
the deep slab on the superficial one removes the part of the deep image that
simply repeats superficial vessels. It is a heuristic: it cannot tell a real
deep vessel that sits under a superficial one from its shadow. Prefer the
vendor's projection-removed export when it exists.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def regress_out_superficial(deep: np.ndarray, superficial: np.ndarray, *,
                            blur_sigma: float = 1.0, mask: np.ndarray | None = None):
    """Subtract the superficial-correlated component from the deep slab.

    Returns (cleaned_deep, beta). beta is the fitted coefficient; values near
    zero mean there was little projection to remove.
    """
    s = ndi.gaussian_filter(superficial, blur_sigma)
    d = ndi.gaussian_filter(deep, blur_sigma)
    sel = np.ones(deep.shape, bool) if mask is None else mask
    sv, dv = s[sel] - s[sel].mean(), d[sel] - d[sel].mean()
    beta = float(max(0.0, (sv @ dv) / max(sv @ sv, 1e-8)))
    cleaned = np.clip(deep - beta * superficial, 0, 1).astype(np.float32)
    return cleaned, beta


def projection_correlation(deep: np.ndarray, superficial: np.ndarray, *, sigma: float = 1.0,
                           mask: np.ndarray | None = None) -> float:
    """Pearson correlation between the slabs after a light blur.

    The slabs share some real anatomy (vessels run between plexuses), so a
    modest positive value is normal. Values close to 1 mean the deep slab is
    mostly a copy of the superficial one.
    """
    a, b = ndi.gaussian_filter(deep, sigma), ndi.gaussian_filter(superficial, sigma)
    if mask is not None:
        a, b = a[mask], b[mask]
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    denom = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / denom) if denom > 1e-12 else 0.0
