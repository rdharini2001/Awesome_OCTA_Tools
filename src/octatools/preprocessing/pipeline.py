"""One configurable preprocessing recipe with a log of what was applied."""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .background import flatten_background
from .contrast import clahe, percentile_stretch
from .denoise import denoise
from .geometry import center_crop_mm, mm_to_px, pixel_size_um, resample_to_pixel_size, valid_mask
from .stripes import remove_stripes


@dataclass(frozen=True)
class PreprocessConfig:
    fov_mm: float
    crop_mm: float | None = None          # keep only the central square
    target_um: float | None = None        # resample to this pixel size
    destripe: bool = True
    denoise: str | None = "nlm"           # None to skip
    denoise_kw: dict = field(default_factory=dict)
    flatten_sigma_mm: float | None = 0.6  # None to skip
    stretch: tuple[float, float] | None = (1.0, 99.5)
    clahe: bool = False


# vessel detection wants capillaries kept, FAZ detection wants smooth, closed edges
VESSEL = dict(destripe=True, denoise="nlm", flatten_sigma_mm=0.6, stretch=(1.0, 99.5), clahe=False)
FAZ = dict(destripe=True, denoise="gaussian", denoise_kw={"sigma": 1.5}, flatten_sigma_mm=0.8,
           stretch=(1.0, 99.5), clahe=False)
MINIMAL = dict(destripe=False, denoise=None, flatten_sigma_mm=None, stretch=None, clahe=False)

PRESETS = {"vessel": VESSEL, "faz": FAZ, "minimal": MINIMAL}


def preset(name: str, fov_mm: float, **overrides) -> PreprocessConfig:
    return replace(PreprocessConfig(fov_mm=fov_mm, **PRESETS[name]), **overrides)


@dataclass
class Preprocessed:
    image: np.ndarray
    valid: np.ndarray
    fov_mm: float
    steps: list[str]


def preprocess(img: np.ndarray, cfg: PreprocessConfig) -> Preprocessed:
    """Run the recipe in a fixed order: crop, resample, destripe, denoise, flatten, stretch."""
    steps: list[str] = []
    fov = cfg.fov_mm
    out = img.astype(np.float32)

    if cfg.crop_mm is not None and cfg.crop_mm < fov:
        out, fov = center_crop_mm(out, fov, cfg.crop_mm)
        steps.append(f"center_crop {cfg.crop_mm} mm")
    if cfg.target_um is not None:
        out = resample_to_pixel_size(out, fov, cfg.target_um)
        steps.append(f"resample to {cfg.target_um:g} um/px")

    valid = valid_mask(out)
    if cfg.destripe:
        out, res = remove_stripes(out, valid)
        steps.append(f"destripe ({int(res.rows.sum())} rows)")
    if cfg.denoise:
        out = denoise(out, cfg.denoise, **cfg.denoise_kw)
        steps.append(f"denoise {cfg.denoise}")
    if cfg.flatten_sigma_mm:
        sigma = mm_to_px(cfg.flatten_sigma_mm, out.shape, fov)
        out = flatten_background(out, sigma, valid)
        steps.append(f"flatten background sigma={cfg.flatten_sigma_mm} mm")
    if cfg.stretch:
        out = percentile_stretch(out, *cfg.stretch, mask=valid)
        steps.append(f"percentile stretch {cfg.stretch}")
    if cfg.clahe:
        out = clahe(out)
        steps.append("clahe")
    return Preprocessed(out, valid, fov, steps)


__all__ = ["PreprocessConfig", "Preprocessed", "preprocess", "preset", "pixel_size_um"]
