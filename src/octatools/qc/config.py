"""Scoring configuration for the quality module.

Each metric is turned into a sub-score in [0, 1] with a linear ramp between a
"bad" anchor (score 0) and a "good" anchor (score 1). The anchors below are
starting points chosen from simulated degradations and visual review (see
docs/qc.md). They are not clinically validated cut-offs. Re-fit them on your
own graded images with `octatools fit-qc` before relying on them.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class QCConfig:
    target_um: float = 10.0               # pixel size used for the image-based measures
    stripe_axes: tuple[int, ...] = (0,)   # 0: horizontal lines, 1: vertical lines
    check_vessel_density: bool = True

    # metric name -> (bad, good). bad > good means "lower is better".
    anchors: dict[str, tuple[float, float]] = field(default_factory=lambda: {
        "coverage":         (0.75, 0.97),
        "line_events":      (2.0, 0.0),
        "motion_fraction":  (0.06, 0.005),
        "signal_fraction":  (0.60, 0.90),
        "blur":             (0.65, 0.40),
        "contrast":         (0.08, 0.25),
        "clipped_fraction": (0.10, 0.01),
        "peripheral_dark":  (0.50, 0.15),
        "projection_corr":  (0.95, 0.60),
    })
    # vessel density is scored by a plausible band: (zero, one, one, zero)
    vessel_density_band: tuple[float, float, float, float] = (0.08, 0.20, 0.70, 0.90)

    weights: dict[str, float] = field(default_factory=lambda: {
        "coverage": 1.0, "line_events": 2.0, "motion_fraction": 1.0, "signal_fraction": 2.0, "blur": 1.5,
        "contrast": 1.0, "clipped_fraction": 0.5, "peripheral_dark": 0.5,
        "projection_corr": 0.5, "vessel_density": 0.5,
    })

    # any of these forces a reject regardless of the overall score
    hard_fail: dict[str, tuple[str, float]] = field(default_factory=lambda: {
        "coverage": ("<", 0.60),
        "motion_fraction": (">", 0.15),
        "line_events": (">", 3.0),
        "signal_fraction": ("<", 0.25),
    })

    grade_cuts: tuple[float, float] = (0.45, 0.70)   # reject below, good at or above
    min_signal_strength: float | None = None         # device-reported index, if you have one
