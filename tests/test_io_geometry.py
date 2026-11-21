import numpy as np

from octatools import io
from octatools.preprocessing import center_crop_mm, pixel_size_um, resample_to_pixel_size, valid_mask


def test_to_float01_types():
    assert io.to_float01(np.full((4, 4), 255, np.uint8)).max() == 1.0
    assert abs(io.to_float01(np.full((4, 4), 65535, np.uint16)).max() - 1.0) < 1e-6
    rgb = np.zeros((4, 4, 3), np.uint8)
    rgb[..., 1] = 255
    assert abs(io.to_float01(rgb).max() - 0.587) < 1e-3
    assert io.to_float01(np.full((3, 3), 200.0)).max() <= 1.0


def test_roundtrip(tmp_path):
    a = np.linspace(0, 1, 64 * 64, dtype=np.float32).reshape(64, 64)
    io.save_image(tmp_path / "a.png", a)
    b = io.read_image(tmp_path / "a.png")
    assert np.abs(a - b).max() < 1 / 255 + 1e-6


def test_pixel_size_and_resample():
    img = np.random.default_rng(0).random((304, 304)).astype(np.float32)
    assert abs(pixel_size_um(img.shape, 3.0) - 9.868) < 0.01
    out = resample_to_pixel_size(img, 3.0, 10.0)
    assert out.shape == (300, 300)
    assert resample_to_pixel_size(img > 0.5, 3.0, 10.0).dtype == bool


def test_center_crop():
    img = np.arange(600 * 600, dtype=np.float32).reshape(600, 600)
    crop, fov = center_crop_mm(img, 6.0, 3.0)
    assert crop.shape == (300, 300) and fov == 3.0
    assert crop[0, 0] == img[150, 150]


def test_valid_mask_ignores_inner_dark_regions():
    img = np.full((100, 100), 0.5, np.float32)
    img[:10] = 0          # border band: invalid
    img[45:55, 45:55] = 0  # interior hole (like a FAZ): valid
    v = valid_mask(img)
    assert not v[:10].any()
    assert v[45:55, 45:55].all()
    assert abs(v.mean() - 0.9) < 0.01
    thin = np.full((100, 100), 0.5, np.float32)
    thin[40:42] = 0                       # a two-row black line is not "missing border"
    assert valid_mask(thin).all()
