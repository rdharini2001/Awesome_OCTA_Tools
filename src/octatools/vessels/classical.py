"""Classical multi-scale vessel enhancement followed by thresholding.

This is the baseline to beat: Frangi (1998), Sato (1998) or Meijering (2004)
ridge filters, hysteresis thresholding, small-object removal. It needs no
weights and runs anywhere.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage.filters import frangi, meijering, sato
from .._morph import remove_small
from ..preprocessing.geometry import pixel_size_um
from .threshold import hysteresis, otsu

FILTERS = {"frangi", "sato", "meijering"}


@dataclass
class VesselSegmentation:
    mask: np.ndarray        # bool
    vesselness: np.ndarray  # float [0, 1]
    method: str


def vesselness(img: np.ndarray, fov_mm: float, method: str = "sato",
               sigmas_um: tuple[float, ...] = (10.0, 20.0)) -> np.ndarray:
    """Multi-scale bright-ridge response, scaled so the 99.5th percentile is 1.

    Scales are given in micrometres so the same call works at any resolution.
    """
    px = pixel_size_um(img.shape, fov_mm)
    sig = sorted({max(0.7, s / px) for s in sigmas_um})
    if method == "frangi":
        v = frangi(img, sigmas=sig, black_ridges=False, beta=0.5, gamma=None)
    elif method == "sato":
        v = sato(img, sigmas=sig, black_ridges=False)
    elif method == "meijering":
        v = meijering(img, sigmas=sig, black_ridges=False)
    else:
        raise ValueError(f"method must be one of {sorted(FILTERS)}")
    top = np.percentile(v, 99.5)
    return np.clip(v / top, 0, 1).astype(np.float32) if top > 0 else v.astype(np.float32)


def segment_classical(img: np.ndarray, fov_mm: float, method: str = "sato", *,
                      sigmas_um: tuple[float, ...] = (10.0, 20.0),
                      thresholding: str = "hysteresis", low: float = 0.20, high: float = 0.40,
                      min_size_um2: float = 400.0) -> VesselSegmentation:
    """Vesselness, binarise, remove specks.

    thresholding: 'hysteresis' uses (low, high) on the normalised vesselness;
    'otsu' uses a single Otsu cut on it.

    The defaults (Sato, scales 10 and 20 um, hysteresis 0.2 / 0.4) came from a
    small grid search on three simulated images with known vessels. They are a
    reasonable start, not a tuned optimum for your device. Check the overlay.
    """
    v = vesselness(img, fov_mm, method, sigmas_um)
    mask = hysteresis(v, low, high) if thresholding == "hysteresis" else otsu(v)
    px = pixel_size_um(img.shape, fov_mm)
    min_px = max(1, int(round(min_size_um2 / px ** 2)))
    mask = remove_small(mask, min_px)
    return VesselSegmentation(mask, v, f"{method}+{thresholding}")
