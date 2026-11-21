"""Pixel size, cropping, resampling and the valid-pixel mask."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.transform import resize


def pixel_size_um(shape: tuple[int, ...], fov_mm: float) -> float:
    """Lateral pixel size in micrometres, assuming a square field of view."""
    return fov_mm * 1000.0 / shape[1]


def mm_to_px(mm: float, shape: tuple[int, ...], fov_mm: float) -> float:
    return mm * shape[1] / fov_mm


def resample_to_pixel_size(img: np.ndarray, fov_mm: float, target_um: float,
                           order: int = 1) -> np.ndarray:
    """Resample so that one pixel is `target_um` wide.

    Downsampling is anti-aliased. Use order=0 for masks.
    """
    n = max(8, int(round(fov_mm * 1000.0 / target_um)))
    if n == img.shape[1] and n == img.shape[0]:
        return img
    out_shape = (int(round(img.shape[0] * n / img.shape[1])), n)
    out = resize(img.astype(np.float32), out_shape, order=order,
                 anti_aliasing=order > 0 and n < img.shape[1],
                 preserve_range=True, mode="edge")
    return out.astype(img.dtype) if img.dtype == bool else out.astype(np.float32)


def center_crop_mm(img: np.ndarray, fov_mm: float, crop_mm: float):
    """Crop the central `crop_mm` square. Returns (crop, crop_mm)."""
    if crop_mm > fov_mm + 1e-6:
        raise ValueError(f"cannot crop {crop_mm} mm out of a {fov_mm} mm image")
    h, w = img.shape[:2]
    ch, cw = int(round(h * crop_mm / fov_mm)), int(round(w * crop_mm / fov_mm))
    top, left = (h - ch) // 2, (w - cw) // 2
    return img[top:top + ch, left:left + cw], crop_mm


def valid_mask(img: np.ndarray, eps: float = 2 / 255, min_thickness_frac: float = 0.015) -> np.ndarray:
    """Pixels that carry data.

    Registration and blinks leave black areas along the border. A dark region
    only counts as missing if it touches the image border and is at least
    `min_thickness_frac` of the image wide. Thinner black rows are motion
    lines, which belong to the stripe detector, and dark regions inside the
    scan (the avascular zone, genuine dropout) stay valid.
    """
    dark = img <= eps
    k = max(3, int(round(min_thickness_frac * min(img.shape))))
    core = ndi.binary_opening(dark, structure=np.ones((k, k), bool))
    lab, n = ndi.label(core)
    if n == 0:
        return np.ones(img.shape, bool)
    border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    border = border[border > 0]
    missing = np.isin(lab, border)
    missing = ndi.binary_dilation(missing, structure=np.ones((3, 3), bool), iterations=k // 2) & dark
    return ~missing
