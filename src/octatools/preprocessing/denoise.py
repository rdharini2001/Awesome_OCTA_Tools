"""Denoising choices for OCTA en-face images.

OCTA noise is not Gaussian: decorrelation noise and speckle fill the
background with a bright, grainy floor. Plain Gaussian smoothing is safe but
blurs capillaries. Non-local means and edge-preserving diffusion keep thin
vessels better at the same noise reduction.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.restoration import denoise_nl_means, denoise_tv_chambolle

from .._noise import noise_sigma


def anisotropic_diffusion(img: np.ndarray, n_iter: int = 10, kappa: float = 0.1,
                          step: float = 0.2) -> np.ndarray:
    """Perona-Malik diffusion with the exponential conduction function.

    `kappa` is the gradient size (in image units) above which smoothing stops.
    `step` must be at most 0.25 for stability on a 4-neighbour stencil.
    """
    u = img.astype(np.float32).copy()
    for _ in range(n_iter):
        p = np.pad(u, 1, mode="edge")
        d = (p[:-2, 1:-1] - u, p[2:, 1:-1] - u, p[1:-1, 2:] - u, p[1:-1, :-2] - u)
        u += step * sum(np.exp(-(g / kappa) ** 2) * g for g in d)
    return u


def denoise(img: np.ndarray, method: str = "nlm", **kw) -> np.ndarray:
    """Denoise with one of: gaussian, median, bilateral, nlm, tv, diffusion."""
    img = img.astype(np.float32)
    if method == "gaussian":
        return ndi.gaussian_filter(img, kw.get("sigma", 1.0))
    if method == "median":
        return ndi.median_filter(img, size=kw.get("size", 3))
    if method == "bilateral":
        return cv2.bilateralFilter(img, d=kw.get("d", 5), sigmaColor=kw.get("sigma_color", 0.1),
                                   sigmaSpace=kw.get("sigma_space", 2.0))
    if method == "nlm":
        sigma = kw.get("sigma") or max(noise_sigma(img), 1e-3)
        return denoise_nl_means(img, patch_size=kw.get("patch_size", 5),
                                patch_distance=kw.get("patch_distance", 6),
                                h=kw.get("h", 0.8) * sigma, sigma=sigma, fast_mode=True
                                ).astype(np.float32)
    if method == "tv":
        return denoise_tv_chambolle(img, weight=kw.get("weight", 0.05)).astype(np.float32)
    if method == "diffusion":
        return anisotropic_diffusion(img, kw.get("n_iter", 10), kw.get("kappa", 0.1))
    raise ValueError(f"unknown denoising method '{method}'")
