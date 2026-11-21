"""Reading and writing en-face OCTA images.

Everything inside the package works on 2-D float32 arrays in [0, 1] where
flow (vessels) is bright. This module converts whatever is on disk to that.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from skimage import io as skio

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def to_float01(arr: np.ndarray) -> np.ndarray:
    """Convert an image array to 2-D float32 in [0, 1].

    RGB(A) is converted with the usual luma weights. uint8 and uint16 are
    scaled by their full range. Float input with values above 1.5 is assumed
    to be on a 0-255 scale (or divided by its maximum if it is larger).
    """
    arr = np.asarray(arr)
    if arr.ndim == 3:
        arr = arr[..., :3] if arr.shape[2] >= 3 else arr[..., :1]
        if arr.shape[2] == 3:
            arr = arr.astype(np.float64) @ np.array([0.299, 0.587, 0.114])
        else:
            arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"expected a 2-D image, got shape {arr.shape}")
    if arr.dtype == np.uint8:
        out = arr.astype(np.float32) / 255.0
    elif arr.dtype == np.uint16:
        out = arr.astype(np.float32) / 65535.0
    elif arr.dtype == bool:
        out = arr.astype(np.float32)
    else:
        out = arr.astype(np.float32)
        top = float(out.max()) if out.size else 0.0
        if top > 1.5:
            out /= 255.0 if top <= 255.0 else top
    return np.clip(out, 0.0, 1.0)


def read_image(path: str | Path) -> np.ndarray:
    """Read an image file as float32 in [0, 1]."""
    return to_float01(skio.imread(str(path)))


def save_image(path: str | Path, arr: np.ndarray) -> None:
    """Write a float [0, 1], bool or uint8 array as an 8-bit PNG/TIFF."""
    arr = np.asarray(arr)
    if arr.dtype == bool:
        arr = arr.astype(np.uint8) * 255
    elif arr.dtype != np.uint8:
        arr = (np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    skio.imsave(str(path), arr, check_contrast=False)


def list_images(path: str | Path) -> list[Path]:
    """Return one file, or every image directly inside a directory (sorted)."""
    p = Path(path)
    if p.is_file():
        return [p]
    if p.is_dir():
        return sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES)
    raise FileNotFoundError(p)
