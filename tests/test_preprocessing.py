import numpy as np

from octatools import simulate
from octatools.preprocessing import (anisotropic_diffusion, denoise, detect_stripes, flatten_background,
                                     percentile_stretch, preprocess, preset, projection_correlation,
                                     regress_out_superficial, remove_stripes, valid_mask)
from octatools.qc import metrics as M


def test_no_false_stripes_on_clean_phantom(phantom):
    res = detect_stripes(phantom.image)
    assert res.rows.sum() == 0


def test_dark_and_bright_lines_are_found(phantom):
    for kind in ("dark", "bright"):
        bad = simulate.add_motion_lines(phantom.image, 6, thickness=2, kind=kind, seed=3)
        res = detect_stripes(bad)
        hit = res.dark if kind == "dark" else res.bright
        truth = np.flatnonzero((bad[:, :5] == (0.0 if kind == "dark" else 0.95)).all(axis=1))
        assert np.isin(truth, np.flatnonzero(hit)).mean() > 0.9
        assert (res.rows & ~np.isin(np.arange(len(res.rows)), truth)).sum() <= 2


def test_stripe_repair_restores_signal(phantom):
    bad = simulate.add_motion_lines(phantom.image, 6, thickness=2, seed=3)
    fixed, res = remove_stripes(bad)
    hit = np.flatnonzero(bad[:, 0] != phantom.image[:, 0])
    err_before = np.abs(bad[hit] - phantom.clean[hit]).mean()
    err_after = np.abs(fixed[hit] - phantom.clean[hit]).mean()
    assert err_after < 0.6 * err_before


def test_stripes_on_columns(phantom):
    bad = simulate.add_motion_lines(phantom.image.T, 5, seed=1).T
    assert detect_stripes(bad, axis=1).rows.sum() >= 5
    assert detect_stripes(bad, axis=0).rows.sum() == 0


def test_denoise_reduces_noise(phantom):
    noisy = simulate.add_noise(phantom.image, 0.12, seed=0)
    base = M.noise_sigma(noisy)
    for method in ("gaussian", "median", "bilateral", "nlm", "tv", "diffusion"):
        out = denoise(noisy, method)
        assert out.shape == noisy.shape and out.dtype == np.float32
        assert np.abs(out - phantom.clean).mean() < np.abs(noisy - phantom.clean).mean(), method
    assert M.noise_sigma(denoise(noisy, "nlm")) < base


def test_diffusion_keeps_edges():
    img = np.zeros((64, 64), np.float32)
    img[:, 32:] = 1
    out = anisotropic_diffusion(img, n_iter=20, kappa=0.1)
    assert out[:, 20].max() < 0.01 and out[:, 44].min() > 0.99


def test_flatten_removes_gradient():
    yy, xx = np.mgrid[:200, :200]
    img = (0.3 + 0.5 * xx / 200).astype(np.float32)
    flat = flatten_background(img, 30)
    assert flat.std() < 0.3 * img.std()


def test_flatten_ignores_black_border():
    img = np.full((200, 200), 0.5, np.float32)
    img[:, :40] = 0
    out = flatten_background(img, 20, valid_mask(img))
    assert abs(out[100, 120] - 0.5) < 0.02


def test_percentile_stretch_range():
    img = np.random.default_rng(0).random((50, 50)).astype(np.float32) * 0.3 + 0.2
    out = percentile_stretch(img, 1, 99)
    assert out.min() == 0 and out.max() == 1


def test_projection_helpers():
    rng = np.random.default_rng(0)
    sup = rng.random((128, 128)).astype(np.float32)
    own = rng.random((128, 128)).astype(np.float32)
    deep = np.clip(0.5 * sup + 0.5 * own, 0, 1)
    cleaned, beta = regress_out_superficial(deep, sup, blur_sigma=1.0)
    assert 0.3 < beta < 0.7
    assert projection_correlation(cleaned, sup) < projection_correlation(deep, sup)
    assert projection_correlation(sup, sup) > 0.999


def test_pipeline_logs_and_shapes(sample):
    img, _ = sample
    res = preprocess(img, preset("vessel", 3.0, target_um=10.0))
    assert res.image.shape == (300, 300) and 0 <= res.image.min() and res.image.max() <= 1
    assert any("denoise" in s for s in res.steps)
    res2 = preprocess(img, preset("minimal", 3.0, crop_mm=2.0))
    assert res2.image.shape[0] == 203 or res2.image.shape[0] == 202
