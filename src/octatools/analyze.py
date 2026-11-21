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
    image: np.ndarray                  # image supplied to the vessel segmenter
    vessels: np.ndarray                # bool
    faz: np.ndarray | None             # bool
    faz_stats: dict = field(default_factory=dict)
    biomarkers: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def analyze(img: np.ndarray, fov_mm: float, *, layer: str = "superficial", vessel_method: str = "learned",
            faz_method: str = "diaz", eye: str | None = None, segmenter=None, qc_config: QCConfig | None = None,
            skip_if_rejected: bool = False) -> Analysis:
    """Run the full chain on one slab.

    By default, vessels use the pretrained OCTA-autosegmentation network and
    the FAZ uses the Python 3 port of FAZ_Extraction. Both receive the source
    image and apply their own method-specific preparation. The learned model
    requires the deep extra; pass an OCTASegmenter through `segmenter` to reuse
    loaded weights. Select vessel_method="classical" or faz_method="vessels"
    explicitly to use the alternative methods.
    """
    if vessel_method not in ("learned", "classical"):
        raise ValueError("vessel_method must be 'learned' or 'classical'")
    if faz_method not in ("diaz", "vessels"):
        raise ValueError("faz_method must be 'diaz' or 'vessels'")
    if layer not in ("superficial", "deep"):
        raise ValueError("layer must be 'superficial' or 'deep'")
    report = assess(img, fov_mm, config=qc_config)
    notes = []
    empty = np.zeros(img.shape, bool)
    if skip_if_rejected and not report.passed:
        return Analysis(report, img, empty, None, notes=["skipped: image rejected by QC"])

    if vessel_method == "learned":
        from .vessels import OCTASegmenter
        vessel_image = img
        seg = (segmenter or OCTASegmenter()).segment(vessel_image, fov_mm)
    else:
        vessel_image = preprocess(img, preset("vessel", fov_mm)).image
        seg = segment_classical(vessel_image, fov_mm)
    vessels = seg.mask

    if faz_method == "diaz":
        res = detect_faz(img, fov_mm, deep=layer == "deep")
        faz, found, why = res.mask, res.found, res.reason
    else:
        res = faz_from_vessels(vessels, fov_mm)
        faz, found, why = res.mask, res.found, res.reason
    if not found:
        notes.append(f"FAZ not found: {why}")
        faz = None
    stats = faz_metrics(faz, fov_mm) if faz is not None else {}
    bio = quantify(vessels, fov_mm, faz=faz, eye=eye)
    return Analysis(report, vessel_image, vessels, faz, stats, bio, notes)
