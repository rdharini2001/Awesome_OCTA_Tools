"""Noise level estimate shared by the denoiser and the quality module."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

_IMMERKAER = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)


def noise_sigma(img: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Noise standard deviation, Immerkaer (1996) with a median instead of a mean.

    The 3x3 Laplacian-difference mask cancels smooth structure; what is left
    is mostly noise. The median makes the estimate far less sensitive to
    vessel edges than the original average. For Gaussian noise the response
    has std 6*sigma, hence the constant.
    """
    resp = np.abs(ndi.convolve(img.astype(np.float32), _IMMERKAER, mode="nearest"))
    if mask is not None:
        inner = ndi.binary_erosion(mask, iterations=1)
        resp = resp[inner] if inner.any() else resp.ravel()
    return float(np.median(resp) / (0.6745 * 6.0))
