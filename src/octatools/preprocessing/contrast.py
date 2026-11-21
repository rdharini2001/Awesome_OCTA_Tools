"""Intensity normalisation and local contrast."""
from __future__ import annotations

import numpy as np
from skimage import exposure


def percentile_stretch(img: np.ndarray, low: float = 1.0, high: float = 99.5,
                       mask: np.ndarray | None = None) -> np.ndarray:
    """Map the [low, high] percentiles of the valid pixels to [0, 1]."""
    vals = img[mask] if mask is not None and mask.any() else img.ravel()
    lo, hi = np.percentile(vals, [low, high])
    if hi - lo < 1e-6:
        return img.copy()
    return np.clip((img - lo) / (hi - lo), 0, 1).astype(np.float32)


def clahe(img: np.ndarray, tile_px: int = 64, clip_limit: float = 0.01) -> np.ndarray:
    """Contrast-limited adaptive histogram equalisation (Zuiderveld 1994)."""
    return exposure.equalize_adapthist(img, kernel_size=tile_px, clip_limit=clip_limit).astype(np.float32)


def gamma(img: np.ndarray, g: float) -> np.ndarray:
    return np.power(np.clip(img, 0, 1), g).astype(np.float32)
