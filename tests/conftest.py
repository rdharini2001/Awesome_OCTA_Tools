from pathlib import Path

import pytest

from octatools import io, simulate

DATA = Path(__file__).resolve().parents[1] / "examples" / "data"


@pytest.fixture(scope="session")
def phantom():
    return simulate.phantom(size=304, fov_mm=3.0, faz_radius_mm=0.3, seed=1)


@pytest.fixture(scope="session")
def sample():
    """A synthetic 3 x 3 mm image from Kreitner et al. (2024), with its vessel label."""
    img = io.read_image(DATA / "synthetic_20230216_232653.png")
    lab = io.read_image(DATA / "synthetic_20230216_232653_label.png") > 0.5
    return img, lab
