import numpy as np
import pytest

from octatools import simulate
from octatools.evaluation import dice
from octatools.faz import detect_faz, faz_from_vessels, faz_metrics, scale_area_by_axial_length
from octatools.preprocessing import preprocess, preset


def disk_mask(n, r_px, c=None):
    c = c or ((n - 1) / 2)
    yy, xx = np.mgrid[:n, :n]
    return np.hypot(yy - c, xx - c) <= r_px


def test_metrics_of_a_disc():
    n, fov, r_mm = 600, 6.0, 0.3
    m = disk_mask(n, r_mm / (fov / n))
    s = faz_metrics(m, fov)
    assert abs(s["area_mm2"] - np.pi * r_mm ** 2) / (np.pi * r_mm ** 2) < 0.02
    assert abs(s["equiv_diameter_mm"] - 2 * r_mm) < 0.01
    assert s["circularity"] > 0.9 and 1.0 <= s["acircularity"] < 1.1
    assert abs(s["axis_ratio"] - 1) < 0.05
    assert abs(s["centroid_x_mm"]) < 0.01


def test_metrics_of_an_ellipse_and_empty():
    yy, xx = np.mgrid[:400, :400]
    e = ((yy - 200) / 40) ** 2 + ((xx - 200) / 80) ** 2 <= 1
    s = faz_metrics(e, 4.0)
    assert 1.8 < s["axis_ratio"] < 2.2 and s["acircularity"] > 1.03
    assert np.isnan(faz_metrics(np.zeros((10, 10), bool), 3.0)["area_mm2"])


def test_axial_length_scaling():
    assert scale_area_by_axial_length(0.3, 24.0, 24.0) == pytest.approx(0.3)
    assert scale_area_by_axial_length(0.3, 26.0, 24.0) > 0.3 > scale_area_by_axial_length(0.3, 22.0, 24.0)


def test_faz_from_vessels_recovers_the_phantom(phantom):
    res = faz_from_vessels(phantom.vessels, 3.0)
    assert res.found
    assert dice(res.mask, phantom.faz) > 0.8
    area = faz_metrics(res.mask, 3.0)["area_mm2"]
    assert abs(area - np.pi * phantom.faz_radius_mm ** 2) / (np.pi * phantom.faz_radius_mm ** 2) < 0.25


def test_faz_from_vessels_on_labelled_sample(sample):
    from skimage.transform import resize
    _, lab = sample
    res = faz_from_vessels(lab, 3.0)
    assert res.found
    stats = faz_metrics(res.mask, 3.0)
    assert 0.1 < stats["area_mm2"] < 0.8 and abs(stats["centroid_x_mm"]) < 0.6


def test_faz_not_found_in_dense_vessels():
    m = np.ones((200, 200), bool)
    m[::7] = False
    assert not faz_from_vessels(m, 3.0).found


def test_faz_search_is_limited_to_the_centre():
    m = np.ones((300, 300), bool)
    m[:60, :60] = False         # large hole far from the centre
    assert not faz_from_vessels(m, 3.0, search_radius_mm=0.5).found


def test_diaz_on_phantom(phantom):
    img = preprocess(phantom.image, preset("faz", 3.0)).image
    res = detect_faz(img, 3.0)
    assert res.found and res.mask.shape == img.shape
    assert dice(res.mask, phantom.faz) > 0.5
    assert 0.05 < res.area_mm2 < 1.0


def test_diaz_deep_flag_and_failure_modes(phantom):
    res = detect_faz(phantom.image, 3.0, deep=True)
    assert res.mask.shape == phantom.image.shape
    empty = detect_faz(np.zeros((100, 100), np.float32), 3.0)
    assert not empty.found and empty.reason


def test_diaz_native_resolution_option(phantom):
    res = detect_faz(phantom.image, 3.0, work_size=None)
    assert res.mask.shape == phantom.image.shape
