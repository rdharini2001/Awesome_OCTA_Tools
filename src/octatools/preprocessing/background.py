"""Slow intensity variation across the field of view.

Signal strength falls off towards the edges and under media opacities. The
trend is estimated with a masked Gaussian (normalised convolution) so black
borders do not pull it down, then divided out or subtracted.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def estimate_background(img: np.ndarray, sigma_px: float, mask: np.ndarray | None = None) -> np.ndarray:
    m = np.ones(img.shape, np.float32) if mask is None else mask.astype(np.float32)
    num = ndi.gaussian_filter(img * m, sigma_px, mode="nearest")
    den = ndi.gaussian_filter(m, sigma_px, mode="nearest")
    return num / np.maximum(den, 1e-3)


def flatten_background(img: np.ndarray, sigma_px: float, mask: np.ndarray | None = None,
                       mode: str = "divide") -> np.ndarray:
    """Remove the low-frequency trend.

    'divide' treats illumination as multiplicative (attenuated signal) and is
    the default. 'subtract' treats it as an additive offset. The overall mean
    is kept so thresholds stay meaningful.
    """
    bg = estimate_background(img, sigma_px, mask)
    level = float(bg[mask].mean()) if mask is not None and mask.any() else float(bg.mean())
    if mode == "divide":
        out = img / np.maximum(bg, 1e-3) * level
    elif mode == "subtract":
        out = img - bg + level
    else:
        raise ValueError(mode)
    out = np.clip(out, 0, 1).astype(np.float32)
    return np.where(mask, out, img) if mask is not None else out
