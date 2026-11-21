import csv

from octatools import io
from octatools.analyze import analyze
from octatools.cli import main


def test_analyze_end_to_end(sample):
    img, _ = sample
    res = analyze(img, 3.0, eye="OD", vessel_method="classical", faz_method="vessels")
    assert res.qc.passed
    assert res.vessels.any() and res.faz is not None
    assert "area_mm2" in res.faz_stats
    assert 0.05 < res.biomarkers["whole"]["vessel_area_density"] < 0.9
    assert "parafovea_nasal" in res.biomarkers


def test_analyze_diaz_branch(sample):
    res = analyze(sample[0], 3.0, vessel_method="classical", faz_method="diaz")
    assert res.vessels.any()


def test_skip_if_rejected():
    import numpy as np
    img = np.zeros((200, 200), np.float32)
    res = analyze(img, 3.0, skip_if_rejected=True)
    assert not res.qc.passed and not res.vessels.any() and res.notes


def test_cli_commands(tmp_path, capsys):
    from conftest import DATA
    src = DATA / "synthetic_20230216_232653.png"
    assert main(["qc", str(src), "--fov-mm", "3", "--csv", str(tmp_path / "qc.csv")]) == 0
    assert (tmp_path / "qc.csv").exists()
    assert main(["preprocess", str(src), "--fov-mm", "3", "--out", str(tmp_path / "pre")]) == 0
    assert main(["faz", str(src), "--fov-mm", "3", "--out", str(tmp_path / "faz")]) == 0
    assert main(["vessels", str(src), "--fov-mm", "3", "--out", str(tmp_path / "ves"), "--method", "classical"]) == 0
    assert main(["run", str(src), "--fov-mm", "3", "--out", str(tmp_path / "run"), "--vessels", "classical"]) == 0
    rows = list(csv.DictReader(open(tmp_path / "run" / "results.csv")))
    assert rows[0]["grade"] in ("good", "acceptable")
    assert (tmp_path / "run" / "synthetic_20230216_232653_summary.png").exists()
    assert io.read_image(tmp_path / "ves" / "synthetic_20230216_232653_vessels.png").max() == 1.0


def test_default_pipeline_uses_upstream_methods_on_source_image(sample, monkeypatch):
    import importlib
    from types import SimpleNamespace
    import numpy as np

    pipeline = importlib.import_module("octatools.analyze")
    image, _ = sample
    calls = []
    original_faz = pipeline.detect_faz

    class Segmenter:
        def segment(self, img, fov_mm):
            assert np.array_equal(img, image)
            assert fov_mm == 3.0
            calls.append("learned")
            return SimpleNamespace(mask=img > 0.4)

    def detect(img, fov_mm, **kwargs):
        assert np.array_equal(img, image)
        calls.append("diaz")
        return original_faz(img, fov_mm, **kwargs)

    def unexpected(*args, **kwargs):
        raise AssertionError("The default pipeline must use upstream preparation.")

    monkeypatch.setattr(pipeline, "detect_faz", detect)
    monkeypatch.setattr(pipeline, "preprocess", unexpected)
    monkeypatch.setattr(pipeline, "segment_classical", unexpected)
    result = analyze(image, 3.0, segmenter=Segmenter())
    assert calls == ["learned", "diaz"]
    assert np.array_equal(result.image, image)
    assert result.faz is not None


def test_cli_defaults_select_upstream_methods(sample, tmp_path, monkeypatch):
    from types import SimpleNamespace
    import numpy as np
    from conftest import DATA
    import octatools.faz as faz_module
    import octatools.vessels as vessel_module

    source = DATA / "synthetic_20230216_232653.png"
    image, _ = sample
    calls = []
    original_faz = faz_module.detect_faz

    class Segmenter:
        def __init__(self, *args, **kwargs):
            pass

        def segment(self, img, fov_mm, **kwargs):
            assert np.array_equal(img, image)
            calls.append("learned")
            return SimpleNamespace(mask=img > 0.4)

    def detect(img, fov_mm, **kwargs):
        assert np.array_equal(img, image)
        calls.append("diaz")
        return original_faz(img, fov_mm, **kwargs)

    monkeypatch.setattr(vessel_module, "OCTASegmenter", Segmenter)
    monkeypatch.setattr(faz_module, "detect_faz", detect)
    import importlib
    monkeypatch.setattr(importlib.import_module("octatools.analyze"), "detect_faz", detect)
    for command in ("vessels", "faz", "run"):
        assert main([command, str(source), "--fov-mm", "3", "--out", str(tmp_path / command)]) == 0
    assert calls == ["learned", "diaz", "learned", "diaz"]
    rows = list(csv.DictReader((tmp_path / "run" / "results.csv").open()))
    assert rows[0]["vessel_method"] == "learned"
    assert rows[0]["faz_method"] == "diaz"


def test_unknown_method_does_not_silently_select_an_alternative(sample):
    import pytest

    with pytest.raises(ValueError, match="vessel_method"):
        analyze(sample[0], 3.0, vessel_method="misspelled")
    with pytest.raises(ValueError, match="faz_method"):
        analyze(sample[0], 3.0, faz_method="misspelled")
