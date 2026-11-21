"""Procedural OCTA phantoms and controlled degradations.

The phantom is a toy: smooth random vessels around a round avascular zone,
with speckle. It is good enough to test code against known answers (FAZ
area, vessel density) and the degradations let you check how a quality
measure responds as an image gets worse.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage.draw import line as draw_line


@dataclass
class Phantom:
    image: np.ndarray     # with speckle and noise
    clean: np.ndarray     # the same scene without noise
    vessels: np.ndarray   # bool ground truth
    faz: np.ndarray       # bool ground truth
    fov_mm: float
    faz_radius_mm: float


def _walk(skeleton: np.ndarray, rng, y: float, x: float, heading: float, steps: int, step: float,
          wiggle: float, keep_out: np.ndarray) -> None:
    size = skeleton.shape[0]
    for _ in range(steps):
        heading += rng.normal(0, wiggle)
        ny, nx = y + step * np.sin(heading), x + step * np.cos(heading)
        if not (0 <= ny < size - 1 and 0 <= nx < size - 1) or keep_out[int(ny), int(nx)]:
            break
        rr, cc = draw_line(int(y), int(x), int(ny), int(nx))
        skeleton[rr, cc] = True
        y, x = ny, nx


def phantom(size: int = 304, fov_mm: float = 3.0, faz_radius_mm: float = 0.3, n_vessels: int = 40,
            n_capillaries: int = 1800, noise: float = 0.04, seed: int = 0) -> Phantom:
    """Large vessels plus a capillary mesh around a round avascular zone, with speckle."""
    rng = np.random.default_rng(seed)
    px_mm = fov_mm / size
    yy, xx = np.mgrid[:size, :size]
    c = (size - 1) / 2
    r_mm = np.hypot(yy - c, xx - c) * px_mm
    faz = r_mm <= faz_radius_mm
    keep_out = r_mm <= faz_radius_mm + 0.01

    big = np.zeros((size, size), bool)
    for _ in range(n_vessels):
        ang = rng.uniform(0, 2 * np.pi)
        r0 = rng.uniform(faz_radius_mm + 0.05, fov_mm * 0.5)
        y, x = c + r0 / px_mm * np.sin(ang), c + r0 / px_mm * np.cos(ang)
        _walk(big, rng, y, x, ang + rng.normal(0, 0.8), int(rng.integers(30, 90)), 2.5, 0.15, keep_out)
    caps = np.zeros((size, size), bool)
    for _ in range(n_capillaries):
        y, x = rng.uniform(0, size - 1, 2)
        _walk(caps, rng, y, x, rng.uniform(0, 2 * np.pi), int(rng.integers(2, 7)), 2.0, 0.5, keep_out)
    vessels = (ndi.binary_dilation(big, iterations=1) | caps) & ~faz

    signal = ndi.gaussian_filter(vessels.astype(np.float32), 0.7)
    signal = 0.12 + 0.75 * signal / max(float(signal.max()), 1e-6)
    signal[faz] = 0.04
    speckle = rng.gamma(shape=6.0, scale=1 / 6.0, size=signal.shape).astype(np.float32)
    img = signal * speckle + rng.normal(0, noise, signal.shape)
    return Phantom(np.clip(img, 0, 1).astype(np.float32), signal.astype(np.float32), vessels, faz,
                   fov_mm, faz_radius_mm)


# --- degradations -----------------------------------------------------------

def add_motion_lines(img: np.ndarray, n: int, *, thickness: int = 2, kind: str = "dark",
                     seed: int = 0) -> np.ndarray:
    """Overwrite `n` random rows (thickness rows each) with dropout or bulk-motion signal."""
    rng = np.random.default_rng(seed)
    out = img.copy()
    for r in rng.choice(img.shape[0] - thickness, size=n, replace=False):
        out[r:r + thickness] = 0.0 if kind == "dark" else 0.95
    return out


def add_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    return ndi.gaussian_filter(img, sigma)


def add_noise(img: np.ndarray, sigma: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(img + rng.normal(0, sigma, img.shape), 0, 1).astype(np.float32)


def attenuate(img: np.ndarray, factor: float = 0.5, gamma: float = 1.0) -> np.ndarray:
    """Low signal strength: scale the flow signal down and compress its range."""
    return (np.clip(img, 0, 1) ** gamma * factor).astype(np.float32)


def add_black_border(img: np.ndarray, fraction: float, side: str = "top") -> np.ndarray:
    """Black out a band covering `fraction` of the image (blink or registration loss)."""
    out = img.copy()
    k = int(round(fraction * img.shape[0 if side in ("top", "bottom") else 1]))
    if k:
        if side == "top":
            out[:k] = 0
        elif side == "bottom":
            out[-k:] = 0
        elif side == "left":
            out[:, :k] = 0
        else:
            out[:, -k:] = 0
    return out


def saturate(img: np.ndarray, gain: float) -> np.ndarray:
    return np.clip(img * gain, 0, 1).astype(np.float32)
