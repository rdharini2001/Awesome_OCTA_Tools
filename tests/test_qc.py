import numpy as np
import pytest

from octatools import simulate
from octatools.qc import QCConfig, assess, auc, fit_thresholds, metrics as M, youden_cutoff
from octatools.qc.assess import band, ramp


def test_ramp_and_band():
    assert ramp(0.5, 0.0, 1.0) == 0.5 and ramp(2, 0, 1) == 1 and ramp(-1, 0, 1) == 0
    assert ramp(0.02, 0.10, 0.01) > 0.85            # lower-is-better direction
    assert band(0.5, (0.1, 0.2, 0.7, 0.9)) == 1 and band(0.05, (0.1, 0.2, 0.7, 0.9)) == 0
    assert 0 < band(0.8, (0.1, 0.2, 0.7, 0.9)) < 1


def test_noise_sigma_recovers_gaussian_sigma():
    rng = np.random.default_rng(0)
    flat = np.full((256, 256), 0.5, np.float32) + rng.normal(0, 0.05, (256, 256)).astype(np.float32)
    assert abs(M.noise_sigma(flat) - 0.05) < 0.005


def test_noise_sigma_ignores_structure(phantom):
    clean = M.noise_sigma(phantom.image)
    noisy = M.noise_sigma(simulate.add_noise(phantom.image, 0.2))
    assert noisy > 1.8 * clean


def test_blur_score_orders(phantom):
    sharp = M.blur_score(phantom.image)
    assert M.blur_score(simulate.add_blur(phantom.image, 1.5)) > sharp
    assert M.blur_score(simulate.add_blur(phantom.image, 3.0)) > M.blur_score(simulate.add_blur(phantom.image, 1.5))


def test_signal_fraction_drops_with_noise(phantom):
    assert M.signal_fraction(simulate.add_noise(phantom.image, 0.2)) < M.signal_fraction(phantom.image)


def test_clipped_fraction():
    img = np.zeros((10, 10), np.float32)
    img[:3] = 1.0
    assert abs(M.clipped_fraction(img) - 0.3) < 1e-9


def test_clean_sample_is_accepted(sample):
    img, _ = sample
    rep = assess(img, 3.0)
    assert rep.grade in ("good", "acceptable"), (rep.grade, rep.flags, rep.subscores)
    assert rep.metrics["motion_fraction"] == 0


@pytest.mark.parametrize("degrade", [
    lambda x: simulate.add_noise(x, 0.35),
    lambda x: simulate.add_blur(x, 5.0),
    lambda x: simulate.add_black_border(x, 0.45),
    lambda x: simulate.add_motion_lines(x, 45, thickness=2),
    lambda x: simulate.attenuate(x, 0.08),
])
def test_severe_degradation_is_rejected_or_scored_lower(sample, degrade):
    img, _ = sample
    good, bad = assess(img, 3.0), assess(degrade(img), 3.0)
    assert bad.score < good.score
    assert bad.grade != "good" and bad.flags


def test_score_decreases_with_severity(sample):
    img, _ = sample
    scores = [assess(simulate.add_noise(img, s), 3.0).score for s in (0.0, 0.1, 0.2, 0.4)]
    assert all(a >= b - 1e-6 for a, b in zip(scores, scores[1:])), scores
    scores = [assess(simulate.add_blur(img, s), 3.0).score for s in (0.01, 1.0, 2.0, 4.0)]
    assert all(a >= b - 1e-6 for a, b in zip(scores, scores[1:])), scores


def test_resolution_independence(sample):
    """The same scene at a different sampling should score similarly."""
    img, _ = sample
    from skimage.transform import resize
    big = resize(img, (608, 608), order=1, anti_aliasing=False).astype(np.float32)
    a, b = assess(img, 3.0), assess(big, 3.0)
    assert abs(a.score - b.score) < 0.2


def test_works_without_fov(sample):
    rep = assess(sample[0])
    assert any("fov_mm" in n for n in rep.notes)


def test_device_signal_strength_gate(sample):
    cfg = QCConfig(min_signal_strength=6)
    assert assess(sample[0], 3.0, config=cfg, signal_strength=4).grade == "reject"
    assert assess(sample[0], 3.0, config=cfg, signal_strength=8).grade != "reject"


def test_projection_check_uses_other_slab(sample):
    img, _ = sample
    rep = assess(img, 3.0, other_slab=img)
    assert rep.metrics["projection_corr"] > 0.99 and rep.subscores["projection_corr"] == 0


def test_report_serialises(sample):
    rep = assess(sample[0], 3.0)
    assert '"grade"' in rep.to_json()
    assert "m_blur" in rep.flat()


def test_calibration():
    y = np.array([1] * 20 + [0] * 20)
    x = np.concatenate([np.linspace(0.6, 1.0, 20), np.linspace(0.0, 0.4, 20)])
    assert auc(x, y) == 1.0 and auc(-x, y) == 0.0
    cut, sens, spec = youden_cutoff(x, y)
    assert 0.4 < cut <= 0.6 and sens == 1 and spec == 1
    ranked = fit_thresholds([{"a": float(v), "noise": float(i % 3)} for i, v in enumerate(x)], list(y))
    assert ranked[0]["metric"] == "a" and ranked[0]["auc"] == 1.0
