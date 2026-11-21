"""Compare the FAZ port with a local copy of the original FAZ_Extraction code.

    python scripts/verify_faz_against_upstream.py --upstream /path/to/FAZ_Extraction

The upstream files are read without modifying them. Compatibility changes cover
Python 2 integer division, OpenCV 3's findContours signature, and drawing calls
that do not affect the returned mask. The original segmentation code is then
executed on its four bundled images at their native resolution.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from octatools.evaluation import dice
from octatools.faz import detect_faz


def load_reference(upstream: Path):
    # Return the OpenCV 3 tuple expected by the original code, using OpenCV 4.
    cv3 = SimpleNamespace(**{name: getattr(cv2, name) for name in dir(cv2)})
    cv3.findContours = lambda *args, **kwargs: (None, *cv2.findContours(*args, **kwargs))
    namespace = {"cv2": cv3, "draw": SimpleNamespace(draw_in_image=lambda *args: None)}
    for name in ("util.py", "opsfaz.py"):
        source = (upstream / name).read_text()
        for statement in ("import cv2", "import drawfaz as draw", "from util import *"):
            source = source.replace(statement, "")
        for before, after in (("size[0]/80", "size[0]//80"),
                              ("size[0]/100", "size[0]//100"),
                              ("size[0]/4", "size[0]//4"),
                              ("size[1]/4", "size[1]//4")):
            source = source.replace(before, after)
        exec(compile(source, str(upstream / name), "exec"), namespace)
    return namespace["detectFAZ"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    args = parser.parse_args()
    reference = load_reference(args.upstream)
    for width in (3, 6):
        for deep, suffix in ((False, "superf"), (True, "prof")):
            path = args.upstream / "images" / f"angio{width}{suffix}.tif"
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(path)
            expected, area, _ = reference(image, width, int(deep), 0.7)
            result = detect_faz(image, width, deep=deep)
            score = dice(result.mask, expected > 0.5)
            print(f"{path.name}: Dice {score:.6f}; contour area difference "
                  f"{abs(result.area_mm2 - area):.8f} mm²")
            if not result.found or not np.array_equal(result.mask, expected > 0.5):
                raise AssertionError(f"FAZ masks differ for {path.name}")
            if not np.isclose(result.area_mm2, area, rtol=0, atol=1e-12):
                raise AssertionError(f"FAZ contour areas differ for {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
