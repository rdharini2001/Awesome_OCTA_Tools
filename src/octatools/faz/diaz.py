"""FAZ detection by Diaz et al. (2019), ported to Python 3 / OpenCV 4.

Original method and code
    M. Diaz, J. Novo, P. Cutrin, F. Gomez-Ulla, M. G. Penedo, M. Ortega.
    "Automatic segmentation of the foveal avascular zone in ophthalmological
    OCT-A images." PLoS ONE 14(2): e0212364, 2019.
    https://github.com/macarenadiaz/FAZ_Extraction

The pipeline is: top-hat filtering, Canny edges closed morphologically, the
largest edge-free region near the centre as a seed, then a region growing
that can add pixels darker than the current region mean until it stops
changing. The steps and constants are those of the original code. What
changed in the port:

* Python 2 integer division is written as `//`, and the OpenCV 4
  `findContours` return value is handled.
* Per-pixel Python loops are replaced by array operations with the same
  result; the region growing visits all contour neighbours in one step,
  which is equivalent because the growth test does not depend on visit order.
* Failure returns a result with `found=False` instead of raising.
* Optional resizing to the ~320 px working size the constants were tuned for.

Two quirks of the original are kept so results stay comparable: the
`mm == 6` Canny setting is overridden by the next line, and the closing
step in `find_mask` is discarded.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

WORK_SIZE = 320  # image side the original constants were designed around (OCTAGON dataset)


@dataclass
class DiazResult:
    mask: np.ndarray            # bool, at the input resolution
    found: bool
    area_mm2: float | None      # contour area, as the original reports it
    contour: np.ndarray | None = None
    reason: str = ""


def _morph(op: str, im: np.ndarray, size: int) -> np.ndarray:
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    f = {"open": cv2.MORPH_OPEN, "closed": cv2.MORPH_CLOSE, "tophat": cv2.MORPH_TOPHAT}
    if op in f:
        return cv2.morphologyEx(im, f[op], se)
    src = im.astype(np.float32)
    return cv2.erode(src, se) if op == "erode" else cv2.dilate(src, se)


def _canny(im: np.ndarray, t2: float, sigma: float) -> np.ndarray:
    blur = cv2.GaussianBlur(im, (15, 15), sigma)
    blur = cv2.normalize(blur, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.Canny(blur.astype(np.uint8), 0, t2)


def _edges(im: np.ndarray, deep: bool, precision: float) -> np.ndarray:
    level = float(im.mean()) * 255
    if deep:
        edges = _canny(im, level + level * 0.2 * precision, 2.0)
    else:
        # the original also sets a 6 mm variant here and then overwrites it
        edges = _canny(im, level + level * 0.3 * precision, 1.9)
    size = max(im.shape[0] // 100 - 3, 3) * 2
    return _morph("closed", edges, size)


def _central_seed(closed: np.ndarray, fov_mm: float) -> np.ndarray:
    """Edge-free pixels in the central half of the image, after an opening."""
    free = (closed < 45).astype(np.float64)
    opened = _morph("open", free, 5 if fov_mm == 6 else 10)
    h, w = free.shape
    out = np.zeros_like(opened)
    out[h // 4:h - h // 4, w // 4:w - w // 4] = opened[h // 4:h - h // 4, w // 4:w - w // 4]
    return out


def _largest_contour(binary: np.ndarray):
    cnts = cv2.findContours(cv2.convertScaleAbs(binary), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
    if not cnts:
        return None
    return max(cnts, key=cv2.contourArea)


def _grow(im: np.ndarray, reg: np.ndarray, area_mm2: float, deep: bool, precision: float) -> np.ndarray:
    if deep:
        if area_mm2 > 0.085:
            for k in (25, 15):
                eroded = _morph("erode", reg, k)
                if eroded.max() == 1.0:
                    reg = eroded
                    break
    elif area_mm2 > 0.15:
        eroded = _morph("erode", reg, 15)
        if eroded.max() == 1.0:
            reg = eroded

    prec = precision / 2 if deep else precision
    h, w = im.shape
    count = int(np.count_nonzero(reg == 1))
    while True:
        reg = cv2.convertScaleAbs(reg)
        cnts = cv2.findContours(reg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
        if not cnts:
            break
        reg[reg != 0] = 1
        mean_level = float(im[reg == 1].sum()) / max(cv2.contourArea(cnts[0]), 0.001)
        pts = cnts[0][:, 0, :]                       # (x, y) = (col, row)
        rows, cols = pts[:, 1], pts[:, 0]
        for dr, dc in ((-1, 0), (0, -1), (0, 1), (1, 0)):
            r, c = rows + dr, cols + dc
            ok = (r > 0) & (c > 0) & (r < h) & (c < w)
            r, c = r[ok], c[ok]
            take = (im[r, c] < mean_level + 0.05 * prec) & (reg[r, c] != 1)
            reg[r[take], c[take]] = 1
        new = int(np.count_nonzero(reg == 1))
        if new == count:
            break
        count = new
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    out = cv2.dilate(reg.astype(np.float32), se, iterations=3)
    return cv2.erode(out, se, iterations=2)


def detect_faz(img: np.ndarray, fov_mm: float, *, deep: bool = False, precision: float = 0.7,
               work_size: int | None = None) -> DiazResult:
    """Segment the FAZ in a superficial (deep=False) or deep (deep=True) slab.

    Args:
        img: float image in [0, 1], vessels bright.
        fov_mm: 3 or 6 in the original paper; other values work but the
            constants were not tuned for them.
        precision: 0..1, how permissive the region growing is. 0.7 is the
            value used in the authors' example.
        work_size: optionally resize to this side length first. The default
            None preserves the native resolution, as in FAZ_Extraction.
            Set 320 explicitly to work at the OCTAGON reference size.
    """
    h0, w0 = img.shape
    work = img.astype(np.float64)
    if work_size and w0 != work_size:
        work = cv2.resize(work, (work_size, work_size), interpolation=cv2.INTER_AREA)
    top = float(work.max())
    if top <= 0:
        return DiazResult(np.zeros((h0, w0), bool), False, None, reason="empty image")
    im = work / top
    h, w = im.shape

    se = ((h // 80) * 2 - int(fov_mm) + 3) * 2 - 1
    im = _morph("tophat", im, max(se, 3))
    closed = _edges(im, deep, precision)
    seed = _central_seed(closed, fov_mm)

    cnt = _largest_contour(seed)
    if cnt is None:
        return DiazResult(np.zeros((h0, w0), bool), False, None, reason="no edge-free region near the centre")
    area = cv2.contourArea(cnt) * fov_mm ** 2 / (h * w)
    reg0 = np.zeros((h, w), np.uint8)
    cv2.drawContours(reg0, [cnt], 0, 1, -1)

    reg = _grow(im, reg0.astype(np.float64), area, deep, precision)
    reg = _morph("closed", _morph("open", reg, 3), 3)
    cnt = _largest_contour(reg)
    if cnt is None:
        return DiazResult(np.zeros((h0, w0), bool), False, None, reason="region vanished during growing")
    area = cv2.contourArea(cnt) * fov_mm ** 2 / (h * w)

    mask = reg > 0.5
    if (h, w) != (h0, w0):
        mask = cv2.resize(mask.astype(np.uint8), (w0, h0), interpolation=cv2.INTER_NEAREST).astype(bool)
    return DiazResult(mask, True, float(area), cnt)
