import numpy as np
import pytest

from octatools import skeleton as sk
from octatools.biomarkers import box_counting_dimension, etdrs_regions, quantify, vessel_metrics
from octatools.evaluation import cldice, dice, iou, segmentation_scores
from octatools.vessels import hysteresis, otsu, phansalkar, segment_classical, vesselness


def test_skeleton_length():
    s = np.zeros((20, 20), bool)
    s[5, 2:12] = True
    assert abs(sk.length_px(s) - 9) < 1e-9
    d = np.eye(20, dtype=bool)
    assert abs(sk.length_px(d) - 19 * np.sqrt(2)) < 1e-6
    corner = np.zeros((5, 5), bool)
    corner[2, 1:3] = corner[3, 2] = True   # L shape: 2 links, not 2 + sqrt(2)
    assert abs(sk.length_px(corner) - 2) < 1e-9


def test_branches_and_tortuosity():
    s = np.zeros((60, 60), bool)
    s[30, 5:55] = True
    brs, junc = sk.branches(s)
    assert len(brs) == 1 and junc.sum() == 0
    assert abs(brs[0].length / brs[0].chord - 1) < 1e-6
    x = np.arange(5, 55)
    wavy = np.zeros((60, 60), bool)
    wavy[(30 + 8 * np.sin(x / 4)).round().astype(int), x] = True
    wavy = sk.skeleton(wavy | np.roll(wavy, 1, axis=0))
    b = sk.branches(wavy)[0]
    assert max(br.length / br.chord for br in b) > 1.15


def test_junction_and_prune():
    s = np.zeros((60, 60), bool)
    s[30, 5:55] = True
    s[20:30, 30] = True          # short spur
    brs, junc = sk.branches(s)
    assert junc.sum() >= 1 and len(brs) >= 3
    pruned = sk.prune(s, 15)
    assert not pruned[20:27, 30].any() and pruned[30, 10]


def test_box_counting_dimension():
    sq = np.ones((128, 128), bool)
    line = np.zeros((128, 128), bool)
    line[64] = True
    assert abs(box_counting_dimension(sq) - 2.0) < 0.05
    assert abs(box_counting_dimension(line) - 1.0) < 0.1


def test_vessel_metrics_known_geometry():
    mask = np.zeros((300, 300), bool)
    mask[::20, :] = True
    mask[1::20, :] = True        # 2 px wide horizontal lines, every 20 px
    m = vessel_metrics(mask, 3.0, with_topology=False)
    assert abs(m["vessel_area_density"] - 0.1) < 1e-9
    assert abs(m["mean_vessel_width_um"] - 2 * 10.0) < 3.0
    assert abs(m["vessel_length_density_mm_per_mm2"] - 15 * 299 * 0.01 / 9.0) < 0.3


def test_vessel_metrics_exclusion():
    mask = np.zeros((100, 100), bool)
    mask[:, 50:] = True
    faz = np.zeros_like(mask)
    faz[:, 50:] = True
    full = vessel_metrics(mask, 3.0, with_topology=False)
    ex = vessel_metrics(mask, 3.0, exclude=faz, with_topology=False)
    assert full["vessel_area_density"] == 0.5 and ex["vessel_area_density"] == 0.0


def test_regions_geometry():
    r = etdrs_regions((600, 600), 6.0, eye="OD")
    pix_mm2 = (6.0 / 600) ** 2
    assert abs(r["fovea"].sum() * pix_mm2 - np.pi * 0.25) < 0.02
    assert abs(r["parafovea"].sum() * pix_mm2 - np.pi * (1.5 ** 2 - 0.5 ** 2)) < 0.05
    quad = sum(r[f"parafovea_{q}"].sum() for q in ("superior", "inferior", "nasal", "temporal"))
    assert abs(quad - r["parafovea"].sum()) / r["parafovea"].sum() < 0.01
    assert r["parafovea_nasal"][300, 360:445].any()                      # OD: nasal at right
    assert etdrs_regions((600, 600), 6.0, eye="OS")["parafovea_nasal"][300, 155:240].any()
    assert "perifovea" not in etdrs_regions((300, 300), 3.0)


def test_quantify_reports_regions(phantom):
    res = quantify(phantom.vessels, 3.0, faz=phantom.faz)
    assert {"whole", "fovea", "parafovea", "whole_noFAZ"} <= set(res)
    assert res["whole_noFAZ"]["vessel_area_density"] > res["whole"]["vessel_area_density"]


def test_classical_segmentation_beats_chance(phantom):
    seg = segment_classical(phantom.image, 3.0)
    assert seg.mask.dtype == bool and seg.vesselness.max() <= 1
    assert dice(seg.mask, phantom.vessels) > 0.55


@pytest.mark.parametrize("method", ["sato", "frangi", "meijering"])
def test_filters_run(phantom, method):
    v = vesselness(phantom.image, 3.0, method)
    assert v.shape == phantom.image.shape and np.isfinite(v).all()
    assert v[phantom.vessels].mean() > 2 * v[~phantom.vessels].mean()


def test_thresholds(phantom):
    assert otsu(phantom.image).mean() < 0.5
    assert phansalkar(phantom.image).dtype == bool
    s = np.zeros((20, 20), float)
    s[5, 5] = 1.0
    s[5, 6:10] = 0.5
    s[15, 15] = 0.5
    h = hysteresis(s, 0.4, 0.9)
    assert h[5, 9] and not h[15, 15]


def test_evaluation_metrics():
    a = np.zeros((50, 50), bool)
    a[10:30, 10:30] = True
    assert dice(a, a) == 1 and iou(a, a) == 1
    b = np.roll(a, 5, axis=1)
    assert 0 < dice(a, b) < 1
    line = np.zeros((50, 50), bool)
    line[25, 5:45] = True
    assert cldice(line, line) == 1
    assert set(segmentation_scores(a, b)) >= {"dice", "iou", "cldice", "precision", "recall"}
