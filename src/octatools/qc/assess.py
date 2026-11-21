"""Quality assessment of one en-face OCTA image."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..io import list_images, read_image, to_float01
from ..preprocessing.geometry import resample_to_pixel_size, valid_mask
from ..preprocessing.projection import projection_correlation
from ..preprocessing.stripes import count_line_events, detect_jumps, detect_stripes
from . import metrics as M
from .config import QCConfig
from .report import QCReport, write_csv


def ramp(x: float, bad: float, good: float) -> float:
    """1 at `good`, 0 at `bad`, linear in between (works in either direction)."""
    if good == bad:
        return float(x == good)
    return float(min(1.0, max(0.0, (x - bad) / (good - bad))))


def band(x: float, edges: tuple[float, float, float, float]) -> float:
    a, b, c, d = edges
    return ramp(x, a, b) if x < (b + c) / 2 else ramp(x, d, c)


def _quick_vessel_density(img: np.ndarray, valid: np.ndarray, fov_mm: float) -> float:
    from ..vessels.classical import segment_classical
    seg = segment_classical(img, fov_mm, min_size_um2=300.0)
    return float(seg.mask[valid].mean()) if valid.any() else 0.0


def assess(img: np.ndarray, fov_mm: float | None = None, *, config: QCConfig | None = None,
           signal_strength: float | None = None, other_slab: np.ndarray | None = None) -> QCReport:
    """Score one image.

    Args:
        img: en-face slab, any dtype `to_float01` understands.
        fov_mm: scan width. Needed to resample to a common pixel size; without
            it the image-based measures run at native resolution and are less
            comparable between devices.
        signal_strength: the device's own quality index, if exported.
        other_slab: the paired slab (e.g. deep when `img` is superficial), used
            for the projection check.
    """
    cfg = config or QCConfig()
    img = to_float01(img)
    notes: list[str] = []
    valid = valid_mask(img)

    m: dict[str, float] = {"coverage": M.coverage(valid)}

    # line artifacts are measured at native resolution: resampling smears thin lines
    worst, events = None, 0
    for ax in cfg.stripe_axes:
        s = detect_stripes(img, valid, axis=ax)
        n = count_line_events(s.rows, detect_jumps(img, valid, axis=ax))
        if worst is None or n > events or (n == events and s.fraction > worst.fraction):
            worst, events = s, n
    m["line_events"] = float(events)
    m["motion_fraction"] = worst.fraction
    m["motion_longest_run_px"] = float(worst.longest_run)

    pixel_um = None
    work, wmask = img, valid
    if fov_mm:
        pixel_um = fov_mm * 1000.0 / img.shape[1]
        # only ever downsample: interpolating up would smooth the noise and flatter the image
        if pixel_um < 0.85 * cfg.target_um:
            work = resample_to_pixel_size(img, fov_mm, cfg.target_um)
            wmask = resample_to_pixel_size(valid.astype(np.float32), fov_mm, cfg.target_um) > 0.5
        else:
            notes.append(f"native pixel size {pixel_um:.1f} um kept (target {cfg.target_um:g} um)")
    else:
        notes.append("fov_mm not given: measures computed at native resolution")

    m["signal_fraction"] = M.signal_fraction(work, wmask)
    m["noise_sigma"] = M.noise_sigma(work, wmask)
    m["blur"] = M.blur_score(work)
    m["contrast"] = M.robust_contrast(work, wmask)
    m["clipped_fraction"] = M.clipped_fraction(work, wmask)
    px_per_mm = work.shape[1] / fov_mm if fov_mm else work.shape[1] / 3.0
    m["peripheral_dark"] = M.peripheral_dark_fraction(work, wmask, px_per_mm)

    if cfg.check_vessel_density and fov_mm:
        m["vessel_density"] = _quick_vessel_density(work, wmask, fov_mm)
    if other_slab is not None:
        other = to_float01(other_slab)
        if other.shape != img.shape:
            raise ValueError("other_slab must have the same shape as img")
        m["projection_corr"] = projection_correlation(other, img, mask=valid)
    if signal_strength is not None:
        m["signal_strength"] = float(signal_strength)

    sub: dict[str, float] = {}
    for name, (bad, good) in cfg.anchors.items():
        if name in m:
            sub[name] = ramp(m[name], bad, good)
    if "vessel_density" in m:
        sub["vessel_density"] = band(m["vessel_density"], cfg.vessel_density_band)

    flags = [f"{k}: {m[k]:.3g} is poor" for k, v in sub.items() if v < 0.34]
    if m["motion_longest_run_px"] >= 8:
        flags.append(f"wide line artifact ({int(m['motion_longest_run_px'])} px)")

    forced = False
    for name, (op, val) in cfg.hard_fail.items():
        if name in m and ((op == "<" and m[name] < val) or (op == ">" and m[name] > val)):
            flags.append(f"hard fail: {name} {op} {val}")
            forced = True
    if cfg.min_signal_strength is not None and signal_strength is not None \
            and signal_strength < cfg.min_signal_strength:
        flags.append(f"hard fail: device signal strength {signal_strength} < {cfg.min_signal_strength}")
        forced = True

    wsum = sum(cfg.weights.get(k, 1.0) for k in sub)
    score = math.exp(sum(cfg.weights.get(k, 1.0) * math.log(max(v, 0.02)) for k, v in sub.items()) / wsum)

    lo, hi = cfg.grade_cuts
    clean = all(v >= 0.34 for v in sub.values())    # a single poor measure keeps an image out of "good"
    grade = "reject" if forced or score < lo else ("good" if score >= hi and clean else "acceptable")
    return QCReport(grade, score, m, sub, flags, pixel_um, notes)


def assess_file(path: str | Path, fov_mm: float | None, **kw) -> QCReport:
    return assess(read_image(path), fov_mm, **kw)


def assess_folder(path: str | Path, fov_mm: float | None, out_csv: str | Path | None = None, **kw):
    """Assess every image in a folder. Returns a list of (path, report)."""
    results = [(p, assess_file(p, fov_mm, **kw)) for p in list_images(path)]
    if out_csv:
        write_csv(out_csv, [{"file": p.name, **r.flat()} for p, r in results])
    return results
