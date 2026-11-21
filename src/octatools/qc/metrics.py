"""No-reference image quality measures for en-face OCTA.

Each function takes a float [0, 1] image (and optionally a valid-pixel mask)
and returns one number. They are deliberately simple so they can be explained
and tested. None needs a vessel segmentation.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import blur_effect

from .._noise import noise_sigma  # noqa: F401  (re-exported)

def _vals(img: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    return img[mask] if mask is not None and mask.any() else img.ravel()


def signal_fraction(img: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Share of the image variance that is not noise, in [0, 1].

    Equal to 1 - sigma_noise^2 / var(image): a reliability-style signal to
    noise ratio that does not depend on how the device scales intensities.
    """
    v = float(np.var(_vals(img, mask)))
    if v < 1e-10:
        return 0.0
    return float(np.clip(1.0 - noise_sigma(img, mask) ** 2 / v, 0.0, 1.0))


def blur_score(img: np.ndarray, h_size: int = 11) -> float:
    """Crete et al. (2007) no-reference blur: 0 is sharp, 1 is very blurred."""
    return float(blur_effect(img, h_size=h_size))


def laplacian_variance(img: np.ndarray) -> float:
    """Variance of the Laplacian (Pech-Pacheco et al. 2000). Scale dependent, so diagnostic only."""
    return float(ndi.laplace(img.astype(np.float32)).var())


def robust_contrast(img: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Spread between the 5th and 95th percentile of valid pixels."""
    lo, hi = np.percentile(_vals(img, mask), [5, 95])
    return float(hi - lo)


def clipped_fraction(img: np.ndarray, mask: np.ndarray | None = None, level: float = 0.995) -> float:
    """Fraction of valid pixels at or near the top of the intensity range."""
    return float(np.mean(_vals(img, mask) >= level))


def coverage(mask: np.ndarray) -> float:
    return float(mask.mean())


def peripheral_dark_fraction(img: np.ndarray, mask: np.ndarray, px_per_mm: float, *,
                             exclude_radius_mm: float = 0.8, rel_level: float = 0.25,
                             sigma_px: float = 3.0) -> float:
    """Fraction of the periphery that is much darker than the typical signal.

    The central disc is excluded because the avascular zone is dark by design.
    Shadows, media opacities and non-perfusion all raise this number, and an
    en-face image alone cannot tell them apart, so treat it as a warning.
    """
    sm = ndi.gaussian_filter(img, sigma_px)
    ref = np.median(sm[mask]) if mask.any() else np.median(sm)
    h, w = img.shape
    yy, xx = np.mgrid[:h, :w]
    ring = np.hypot(yy - (h - 1) / 2, xx - (w - 1) / 2) > exclude_radius_mm * px_per_mm
    sel = ring & mask
    if not sel.any():
        return 0.0
    return float(np.mean(sm[sel] < rel_level * ref))
