"""Run the pretrained vessel model and the Díaz FAZ detector on one image.

Install with python -m pip install -e ".[deep]" before running this example.
The first run downloads the authors’ vessel checkpoint if it is not cached.

    python examples/quickstart.py [path/to/image.png] [fov_mm]
"""
import sys
from pathlib import Path

from octatools import io
from octatools.analyze import analyze

here = Path(__file__).parent
path = sys.argv[1] if len(sys.argv) > 1 else here / "data" / "synthetic_20230216_232653.png"
fov = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0

res = analyze(io.read_image(path), fov_mm=fov, eye="OD")

print("quality:", res.qc.grade, f"({res.qc.score:.2f})", res.qc.flags)
if res.faz is not None:
    print("FAZ area (mm2):", round(res.faz_stats["area_mm2"], 3), " acircularity:", round(res.faz_stats["acircularity"], 2))
for region in ("whole", "parafovea", "parafovea_nasal"):
    m = res.biomarkers[region]
    print(f"{region:16s} vessel area density {m['vessel_area_density']:.3f}   "
          f"length density {m['vessel_length_density_mm_per_mm2']:.1f} mm/mm2")
