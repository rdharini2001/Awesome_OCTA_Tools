"""Needs torch, monai and the downloaded weights (about 30 MB)."""
import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("monai")

from octatools.evaluation import dice  # noqa: E402
from octatools.vessels import weights  # noqa: E402

pytestmark = pytest.mark.deep


@pytest.fixture(scope="module")
def segmenter():
    try:
        path = weights.ensure_weights()
    except Exception as e:  # offline machines
        pytest.skip(f"weights unavailable: {e}")
    from octatools.vessels.synthetic import OCTASegmenter
    return OCTASegmenter(path)


def test_segments_synthetic_sample(segmenter, sample):
    img, lab = sample
    seg = segmenter.segment(img, 3.0)
    assert seg.mask.shape == img.shape and seg.vesselness.shape == img.shape
    from skimage.transform import resize
    truth = resize(lab.astype(float), img.shape, order=1) > 0.5
    assert dice(seg.mask, truth) > 0.5


def test_tiling_for_larger_fields(segmenter, sample):
    from skimage.transform import resize
    img, _ = sample
    big = resize(np.tile(img, (2, 2)), (400, 400), order=1).astype(np.float32)   # pretend 6 x 6 mm
    prob = segmenter.predict_proba(big, 6.0)
    assert prob.shape == big.shape and 0 <= prob.min() and prob.max() <= 1


def test_tile_helpers():
    from octatools.vessels.synthetic import _tile_starts
    assert _tile_starts(100, 100, 0) == [0]
    assert _tile_starts(200, 100, 0) == [0, 100]
    assert _tile_starts(200, 100, 0.5) == [0, 50, 100]
    assert _tile_starts(250, 100, 0) == [0, 75, 150]


def test_default_analysis_uses_loaded_upstream_model(segmenter, sample):
    from octatools.analyze import analyze
    from octatools.faz import detect_faz

    img, _ = sample
    result = analyze(img, 3.0, segmenter=segmenter)
    expected = segmenter.segment(img, 3.0)
    expected_faz = detect_faz(img, 3.0)
    assert np.array_equal(result.vessels, expected.mask)
    assert expected_faz.found and np.array_equal(result.faz, expected_faz.mask)
    assert "whole" in result.biomarkers
