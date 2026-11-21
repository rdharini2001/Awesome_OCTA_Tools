"""Binarisation: global, local and hysteresis."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import apply_hysteresis_threshold, threshold_otsu


def otsu(img: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    vals = img[mask] if mask is not None and mask.any() else img.ravel()
    return img > threshold_otsu(vals)


def phansalkar(img: np.ndarray, window: int = 15, k: float = 0.25, r: float = 0.5,
               p: float = 2.0, q: float = 10.0) -> np.ndarray:
    """Phansalkar et al. (2011) local threshold for low-contrast images.

    T = m * (1 + p*exp(-q*m) + k*(s/r - 1)), with m and s the local mean and
    standard deviation. The exponential term raises the threshold in dark
    regions so background noise is not picked up there. Defaults follow the
    original paper for images scaled to [0, 1].
    """
    img = img.astype(np.float32)
    m = ndi.uniform_filter(img, window, mode="reflect")
    s = np.sqrt(np.maximum(ndi.uniform_filter(img * img, window, mode="reflect") - m * m, 0))
    return img > m * (1 + p * np.exp(-q * m) + k * (s / r - 1))


def hysteresis(score: np.ndarray, low: float, high: float) -> np.ndarray:
    """Keep weak responses only if they connect to a strong one."""
    return apply_hysteresis_threshold(score, low, high)
