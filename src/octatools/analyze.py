"""Quality check, preprocessing, segmentation, FAZ and biomarkers in one call."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .biomarkers import quantify
from .faz import detect_faz, faz_from_vessels, faz_metrics
from .preprocessing import preprocess, preset
from .qc import QCConfig, QCReport, assess
from .vessels import segment_classical


@dataclass
class Analysis:
    qc: QCReport
    image: np.ndarray                  # preprocessed
    vessels: np.ndarray                # bool
    faz: np.ndarray | None             # bool
    faz_stats: dict = field(default_factory=dict)
    biomarkers: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def analyze(img: np.ndarray, fov_mm: float, *, layer: str = "superficial", vessel_method: str = "classical",
            faz_method: str = "vessels", eye: str | None = None, segmenter=None, qc_config: QCConfig | None = None,
            skip_if_rejected: bool = False) -> Analysis:
    """Run the full chain on one slab.

    vessel_method: 'classical' (Sato + hysteresis) or 'learned' (Kreitner et al.,
        pass an OCTASegmenter via `segmenter` to reuse loaded weights).
    faz_method: 'vessels' (largest avascular region of the vessel mask) or
        'diaz' (Diaz et al. 2019 on the preprocessed image).
    """
    report = assess(img, fov_mm, config=qc_config)
    notes = []
    empty = np.zeros(img.shape, bool)
    if skip_if_rejected and not report.passed:
        return Analysis(report, img, empty, None, notes=["skipped: image rejected by QC"])

    pre = preprocess(img, preset("vessel", fov_mm))
    if vessel_method == "learned":
        from .vessels import OCTASegmenter
        seg = (segmenter or OCTASegmenter()).segment(pre.image, fov_mm, smooth_sigma_um=None)
    else:
        seg = segment_classical(pre.image, fov_mm)
    vessels = seg.mask

    if faz_method == "diaz":
        faz_in = preprocess(img, preset("faz", fov_mm)).image
        res = detect_faz(faz_in, fov_mm, deep=layer == "deep")
        faz, found, why = res.mask, res.found, res.reason
    else:
        res = faz_from_vessels(vessels, fov_mm)
        faz, found, why = res.mask, res.found, res.reason
    if not found:
        notes.append(f"FAZ not found: {why}")
        faz = None
    stats = faz_metrics(faz, fov_mm) if faz is not None else {}
    bio = quantify(vessels, fov_mm, faz=faz, eye=eye)
    return Analysis(report, pre.image, vessels, faz, stats, bio, notes)
