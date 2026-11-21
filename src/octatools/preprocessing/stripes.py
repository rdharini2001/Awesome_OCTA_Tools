"""Motion and blink lines.

Eye motion during the fast axis scan leaves whole rows that are too dark
(lost flow signal) or too bright (bulk-motion decorrelation) compared with
their neighbours (Spaide et al. 2015). Rows are located from a robust
row-level profile and then repaired by interpolating across them.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from scipy.interpolate import interp1d


@dataclass
class StripeResult:
    dark: np.ndarray    # bool per row
    bright: np.ndarray  # bool per row
    profile: np.ndarray
    baseline: np.ndarray

    @property
    def rows(self) -> np.ndarray:
        return self.dark | self.bright

    @property
    def fraction(self) -> float:
        return float(self.rows.mean())

    @property
    def longest_run(self) -> int:
        lab, n = ndi.label(self.rows)
        return int(np.bincount(lab.ravel())[1:].max()) if n else 0


def _row_profile(img: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    if mask is None:
        return img.mean(axis=1)
    w = mask.sum(axis=1)
    prof = np.where(w > 0, (img * mask).sum(axis=1) / np.maximum(w, 1), np.nan)
    # rows with almost no data are not evidence for or against a stripe
    prof[w < 0.2 * img.shape[1]] = np.nan
    return prof


def _across_row_agreement(img: np.ndarray, mask: np.ndarray | None, rows: np.ndarray,
                          sign: np.ndarray, gap: int = 4) -> np.ndarray:
    """Share of columns in which each candidate row differs from its neighbours in the same direction.

    A motion line deviates along the whole width. A long horizontal vessel
    raises the row mean too, but only where the vessel is, so it scores low.
    Neighbours are taken `gap` rows either side of the run the row belongs to.
    """
    n = len(rows)
    lab, k = ndi.label(rows)
    out = np.zeros(n)
    for i in range(1, k + 1):
        idx = np.flatnonzero(lab == i)
        lo, hi = max(0, idx[0] - gap), min(n, idx[-1] + gap + 1)
        around = [r for r in range(lo, hi) if not rows[r]]
        if not around:
            continue
        ref = np.median(img[around], axis=0)
        diff = img[idx].mean(axis=0) - ref
        valid = np.ones(img.shape[1], bool) if mask is None else mask[idx].all(axis=0)
        scale = 0.15 * max(float(np.median(ref[valid])) if valid.any() else 0.0, 0.05)
        agree = (np.sign(diff) == sign[idx[0]]) & (np.abs(diff) > scale)
        out[idx] = float(agree[valid].mean()) if valid.any() else 0.0
    return out


def detect_stripes(img: np.ndarray, mask: np.ndarray | None = None, *, axis: int = 0,
                   window: int = 41, z_thresh: float = 4.0, min_rel_dev: float = 0.25,
                   min_agreement: float = 0.75) -> StripeResult:
    """Find rows (axis=0) or columns (axis=1) that deviate from their surroundings.

    A row is flagged when its mean differs from the running median of the row
    profile by more than `z_thresh` robust standard deviations *and* by more
    than `min_rel_dev` of the local level, *and* when at least `min_agreement`
    of the columns agree with it. The second condition stops quiet images from
    producing false lines, the third stops long horizontal vessels from being
    mistaken for lines.
    """
    if axis == 1:
        img = img.T
        mask = None if mask is None else mask.T
    prof = _row_profile(img, mask)
    filled = prof.copy()
    bad = np.isnan(filled)
    if bad.all():
        z = np.zeros_like(prof, bool)
        return StripeResult(z, z.copy(), prof, prof)
    if bad.any():
        filled[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), filled[~bad])
    window = min(window | 1, (len(filled) // 2) * 2 - 1) if len(filled) > 4 else 1
    base = ndi.median_filter(filled, size=window, mode="reflect")
    dev = filled - base
    sigma = 1.4826 * np.median(np.abs(dev - np.median(dev))) + 1e-6
    rel = np.abs(dev) / np.maximum(base, 1e-3)
    hit = (np.abs(dev) > z_thresh * sigma) & (rel > min_rel_dev) & ~bad
    if hit.any():
        agree = _across_row_agreement(img, mask, hit, np.sign(dev))
        hit &= agree >= min_agreement
    return StripeResult(hit & (dev < 0), hit & (dev > 0), prof, base)


def repair_stripes(img: np.ndarray, rows: np.ndarray, *, axis: int = 0, grow: int = 1) -> np.ndarray:
    """Replace flagged rows with a linear interpolation of the nearest good ones."""
    if axis == 1:
        return repair_stripes(img.T, rows, axis=0, grow=grow).T
    bad = ndi.binary_dilation(rows, iterations=grow) if grow else rows.copy()
    if not bad.any() or bad.all():
        return img
    good = np.flatnonzero(~bad)
    f = interp1d(good, img[good], axis=0, kind="linear", bounds_error=False,
                 fill_value=(img[good[0]], img[good[-1]]))
    out = img.copy()
    idx = np.flatnonzero(bad)
    out[idx] = f(idx)
    return out


def remove_stripes(img: np.ndarray, mask: np.ndarray | None = None, *, axis: int = 0, **kw):
    """Detect and repair in one call. Returns (image, StripeResult)."""
    res = detect_stripes(img, mask, axis=axis, **kw)
    return repair_stripes(img, res.rows, axis=axis), res


def detect_jumps(img: np.ndarray, mask: np.ndarray | None = None, *, axis: int = 0, sigma: float = 1.5,
                 window: int = 41, z_thresh: float = 6.0, min_ratio: float = 1.5) -> np.ndarray:
    """Rows where the image content jumps from one row to the next.

    When strips of a registered scan are misplaced, vessels break and shift
    across a line, often without any dark or bright row. The mean absolute
    difference between neighbouring rows (after a light blur, so speckle does
    not dominate) is low and smooth in a continuous image and spikes at a jump.
    Returns a bool array per row marking the row above each detected jump.
    """
    if axis == 1:
        return detect_jumps(img.T, None if mask is None else mask.T, axis=0, sigma=sigma,
                            window=window, z_thresh=z_thresh, min_ratio=min_ratio)
    sm = ndi.gaussian_filter(img, sigma)
    diff = np.abs(np.diff(sm, axis=0))
    if mask is not None:
        ok = mask[:-1] & mask[1:]
        w = ok.sum(axis=1)
        d = np.where(w > 0.2 * img.shape[1], (diff * ok).sum(axis=1) / np.maximum(w, 1), np.nan)
    else:
        d = diff.mean(axis=1)
    out = np.zeros(img.shape[0], bool)
    good = ~np.isnan(d)
    if good.sum() < 8:
        return out
    d = np.interp(np.arange(len(d)), np.flatnonzero(good), d[good])
    window = min(window | 1, (len(d) // 2) * 2 - 1)
    base = ndi.median_filter(d, size=window, mode="reflect")
    dev = d - base
    sigma_d = 1.4826 * np.median(np.abs(dev - np.median(dev))) + 1e-6
    hit = (dev > z_thresh * sigma_d) & (d > min_ratio * np.maximum(base, 1e-6))
    out[:-1] = hit
    return out


def count_line_events(rows: np.ndarray, jumps: np.ndarray | None = None, gap: int = 6,
                      edge_frac: float = 0.02) -> int:
    """Number of separate line artifacts: flagged rows and jumps within `gap` rows count once.

    Rows in the outermost `edge_frac` of the image are ignored: a damaged border
    can be cropped away, so it should not count as an artifact through the scan.
    """
    allrows = rows.copy() if jumps is None else rows | jumps
    k = max(3, int(round(edge_frac * len(allrows))))
    allrows[:k] = False
    allrows[-k:] = False
    if not allrows.any():
        return 0
    idx = np.flatnonzero(allrows)
    return int(1 + (np.diff(idx) > gap).sum())
