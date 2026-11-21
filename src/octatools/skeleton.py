"""Skeleton length, branches and pruning on 8-connected binary skeletons."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

_K8 = np.ones((3, 3), int)
_K8[1, 1] = 0


def skeleton(mask: np.ndarray) -> np.ndarray:
    return skeletonize(mask.astype(bool))


def neighbor_count(skel: np.ndarray) -> np.ndarray:
    return ndi.convolve(skel.astype(int), _K8, mode="constant") * skel


def length_px(skel: np.ndarray) -> float:
    """Skeleton length in pixels.

    Counts orthogonal links as 1 and diagonal links as sqrt(2). A diagonal link
    is skipped when it only short-cuts an L-shaped corner that is already
    counted with two orthogonal links.
    """
    s = skel.astype(bool)
    horiz = (s[:, :-1] & s[:, 1:]).sum()
    vert = (s[:-1, :] & s[1:, :]).sum()
    d1 = s[:-1, :-1] & s[1:, 1:] & ~(s[:-1, 1:] | s[1:, :-1])
    d2 = s[:-1, 1:] & s[1:, :-1] & ~(s[:-1, :-1] | s[1:, 1:])
    return float(horiz + vert + np.sqrt(2) * (d1.sum() + d2.sum()))


@dataclass
class Branch:
    coords: np.ndarray        # ordered (row, col)
    touches_junction: int     # how many of its two ends meet a junction (0, 1 or 2)

    @property
    def length(self) -> float:
        d = np.diff(self.coords, axis=0)
        return float(np.hypot(d[:, 0], d[:, 1]).sum())

    @property
    def chord(self) -> float:
        return float(np.hypot(*(self.coords[-1] - self.coords[0])))


def _order(comp: np.ndarray) -> np.ndarray:
    """Walk the pixels of one branch from an end to the other."""
    pts = [tuple(p) for p in np.argwhere(comp)]
    pset = set(pts)
    nb = lambda p: [(p[0] + a, p[1] + b) for a in (-1, 0, 1) for b in (-1, 0, 1)
                    if (a or b) and (p[0] + a, p[1] + b) in pset]
    ends = [p for p in pts if len(nb(p)) <= 1]
    cur = ends[0] if ends else pts[0]
    seen, path = {cur}, [cur]
    while True:
        cand = [q for q in nb(cur) if q not in seen]
        if not cand:
            break
        # prefer orthogonal steps so the walk follows the line
        cand.sort(key=lambda q: abs(q[0] - cur[0]) + abs(q[1] - cur[1]))
        cur = cand[0]
        seen.add(cur)
        path.append(cur)
    return np.array(path)


def branches(skel: np.ndarray) -> tuple[list[Branch], np.ndarray]:
    """Split a skeleton at its junctions.

    Returns (branches, junction_mask). Junction pixels belong to no branch.
    """
    skel = skel.astype(bool)
    junction = skel & (neighbor_count(skel) >= 3)
    zone = ndi.binary_dilation(junction, structure=np.ones((3, 3))) & skel
    seg = skel & ~zone
    lab, n = ndi.label(seg, structure=np.ones((3, 3)))
    near_junction = ndi.binary_dilation(zone, structure=np.ones((3, 3)))
    out = []
    for i in range(1, n + 1):
        comp = lab == i
        if comp.sum() < 2:
            continue
        path = _order(comp)
        t = int(near_junction[tuple(path[0])]) + int(near_junction[tuple(path[-1])])
        out.append(Branch(path, t))
    return out, junction


def prune(skel: np.ndarray, min_len_px: float, passes: int = 2) -> np.ndarray:
    """Delete terminal branches (one free end, one junction end) shorter than `min_len_px`."""
    skel = skel.astype(bool).copy()
    for _ in range(passes):
        brs, _ = branches(skel)
        removed = False
        for b in brs:
            if b.touches_junction == 1 and b.length < min_len_px:
                skel[b.coords[:, 0], b.coords[:, 1]] = False
                removed = True
        if not removed:
            break
        skel = skeletonize(skel)
    return skel
