import csv

from octatools import io
from octatools.analyze import analyze
from octatools.cli import main


def test_analyze_end_to_end(sample):
    img, _ = sample
    res = analyze(img, 3.0, eye="OD")
    assert res.qc.passed
    assert res.vessels.any() and res.faz is not None
    assert "area_mm2" in res.faz_stats
    assert 0.05 < res.biomarkers["whole"]["vessel_area_density"] < 0.9
    assert "parafovea_nasal" in res.biomarkers


def test_analyze_diaz_branch(sample):
    res = analyze(sample[0], 3.0, faz_method="diaz")
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
    assert main(["vessels", str(src), "--fov-mm", "3", "--out", str(tmp_path / "ves")]) == 0
    assert main(["run", str(src), "--fov-mm", "3", "--out", str(tmp_path / "run")]) == 0
    rows = list(csv.DictReader(open(tmp_path / "run" / "results.csv")))
    assert rows[0]["grade"] in ("good", "acceptable")
    assert (tmp_path / "run" / "synthetic_20230216_232653_summary.png").exists()
    assert io.read_image(tmp_path / "ves" / "synthetic_20230216_232653_vessels.png").max() == 1.0
