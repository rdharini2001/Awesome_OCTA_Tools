"""Segmentation agreement measures."""
from __future__ import annotations

import numpy as np

from .skeleton import skeleton


def confusion(pred: np.ndarray, truth: np.ndarray) -> tuple[int, int, int, int]:
    p, t = pred.astype(bool), truth.astype(bool)
    return int((p & t).sum()), int((p & ~t).sum()), int((~p & t).sum()), int((~p & ~t).sum())


def dice(pred: np.ndarray, truth: np.ndarray) -> float:
    tp, fp, fn, _ = confusion(pred, truth)
    return 2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) else 1.0


def iou(pred: np.ndarray, truth: np.ndarray) -> float:
    tp, fp, fn, _ = confusion(pred, truth)
    return tp / (tp + fp + fn) if (tp + fp + fn) else 1.0


def cldice(pred: np.ndarray, truth: np.ndarray) -> float:
    """Centreline Dice (Shit et al., CVPR 2021).

    Tprec is the share of the predicted skeleton inside the true mask and Tsens
    the share of the true skeleton inside the predicted mask. Unlike Dice it
    does not reward getting thick vessels right at the cost of capillaries.
    """
    p, t = pred.astype(bool), truth.astype(bool)
    sp, st = skeleton(p), skeleton(t)
    tprec = (sp & t).sum() / max(sp.sum(), 1)
    tsens = (st & p).sum() / max(st.sum(), 1)
    return float(2 * tprec * tsens / (tprec + tsens)) if (tprec + tsens) else 0.0


def segmentation_scores(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    tp, fp, fn, tn = confusion(pred, truth)
    return {
        "dice": dice(pred, truth), "iou": iou(pred, truth), "cldice": cldice(pred, truth),
        "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
    }
