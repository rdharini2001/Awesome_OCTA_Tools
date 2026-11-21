"""Fit QC thresholds to your own graded images.

You supply a table of metrics (from `QCReport.metrics`) and a binary label
(1 = usable, 0 = not usable). For every metric this gives the AUC and the
cut-off that maximises Youden's J (sensitivity + specificity - 1).
"""
from __future__ import annotations

import numpy as np


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve via the rank-sum statistic (ties get mid-ranks)."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    pos, neg = labels == 1, labels == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


def youden_cutoff(scores: np.ndarray, labels: np.ndarray, higher_is_better: bool = True):
    """Return (cutoff, sensitivity, specificity) maximising J, with 'usable' as the positive class."""
    x = np.asarray(scores, float) * (1 if higher_is_better else -1)
    y = np.asarray(labels, int)
    best = (-1.0, None, None, None)
    for t in np.unique(x):
        pred = x >= t
        sens = float((pred & (y == 1)).sum() / max((y == 1).sum(), 1))
        spec = float((~pred & (y == 0)).sum() / max((y == 0).sum(), 1))
        if sens + spec - 1 > best[0]:
            best = (sens + spec - 1, t, sens, spec)
    cut = best[1] * (1 if higher_is_better else -1)
    return float(cut), best[2], best[3]


def fit_thresholds(table: list[dict[str, float]], labels: list[int]) -> list[dict]:
    """Rank metrics by how well they separate usable from unusable images."""
    y = np.asarray(labels, int)
    out = []
    for name in table[0]:
        x = np.array([row[name] for row in table], float)
        if np.isnan(x).any():
            continue
        a = auc(x, y)
        better_high = a >= 0.5
        cut, sens, spec = youden_cutoff(x, y, better_high)
        out.append({"metric": name, "auc": max(a, 1 - a), "usable_when": ">=" if better_high else "<=",
                    "cutoff": cut, "sensitivity": sens, "specificity": spec})
    return sorted(out, key=lambda r: -r["auc"])
