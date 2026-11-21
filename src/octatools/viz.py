"""Overlays for checking results by eye. NumPy only, no plotting library."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def to_rgb(img: np.ndarray) -> np.ndarray:
    """Grayscale [0, 1] to RGB; an array that is already RGB is returned as a copy."""
    g = np.clip(img, 0, 1)
    if g.ndim == 3:
        return g.astype(np.float32).copy()
    return np.stack([g, g, g], axis=-1).astype(np.float32)


def overlay_mask(img: np.ndarray, mask: np.ndarray, color=(1.0, 0.2, 0.2), alpha: float = 0.45) -> np.ndarray:
    rgb = to_rgb(img)
    m = mask.astype(bool)
    rgb[m] = (1 - alpha) * rgb[m] + alpha * np.array(color, np.float32)
    return rgb


def overlay_outline(img: np.ndarray, mask: np.ndarray, color=(0.1, 0.9, 0.3), width: int = 1) -> np.ndarray:
    rgb = to_rgb(img)
    m = mask.astype(bool)
    edge = m & ~ndi.binary_erosion(m, iterations=width)
    rgb[edge] = np.array(color, np.float32)
    return rgb


def side_by_side(*panels: np.ndarray, gap: int = 4) -> np.ndarray:
    """Place same-height RGB or grayscale panels next to each other."""
    rgbs = [p if p.ndim == 3 else to_rgb(p) for p in panels]
    spacer = np.ones((rgbs[0].shape[0], gap, 3), np.float32)
    parts = []
    for p in rgbs:
        parts += [p, spacer]
    return np.concatenate(parts[:-1], axis=1)
