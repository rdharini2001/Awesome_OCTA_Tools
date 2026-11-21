"""Check that octatools.vessels.OCTASegmenter reproduces the authors' own test.py.

Runs the upstream repository's test.py on a folder of images, runs the wrapper
on the same images, and reports the Dice overlap of the two masks at the model's own 1216 x 1216 resolution.

    python scripts/verify_against_upstream.py \
        --upstream /path/to/OCTA-autosegmentation --images examples/data --fov-mm 3
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from octatools import io
from octatools.evaluation import dice
from octatools._morph import remove_small
from octatools.vessels import OCTASegmenter
from octatools.vessels.synthetic import MODEL_MIN_SIZE_PX


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", required=True, help="clone of aiforvision/OCTA-autosegmentation")
    ap.add_argument("--images", required=True, help="folder of PNG images (3 mm scans)")
    ap.add_argument("--fov-mm", type=float, default=3.0)
    ap.add_argument("--glob", default="synthetic_*[0-9].png", help="which files in --images to use")
    args = ap.parse_args()

    up = Path(args.upstream).resolve()
    files = sorted(Path(args.images).glob(args.glob))
    if not files:
        sys.exit("no images matched")
    cfg = up / "docker/trained_models/ves_seg-S-GAN/config.yml"
    out = Path(tempfile.mkdtemp())
    pattern = str(Path(args.images).resolve() / args.glob)
    subprocess.run([sys.executable, "test.py", "--config_file", str(cfg), "--epoch", "30",
                    "--num_workers", "0", "--Test.data.image.files", pattern, "--Test.save_dir", str(out)], cwd=up, check=True)

    checkpoint = cfg.parent / "checkpoints/30_model.pth"
    seg = OCTASegmenter(str(checkpoint))
    scores = []
    for f in files:
        theirs = sorted(out.glob(f"*{f.stem}*"))
        if not theirs:
            print(f"{f.name}: no upstream output found in {out}")
            raise RuntimeError(f"Missing upstream output for {f.name}")
        t = io.read_image(theirs[0]) > 0.5
        prob = seg._predict_tile(io.read_image(f), out_size=t.shape)
        mine = remove_small(prob > 0.5, MODEL_MIN_SIZE_PX)
        d = dice(mine, t)
        scores.append(d)
        native = seg.segment(io.read_image(f), args.fov_mm).mask
        h, w = native.shape
        native_reference = cv2.resize(t.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        print(f"{f.name}: Dice at model resolution = {d:.6f}; "
              f"native masks identical = {np.array_equal(native, native_reference)}")
        if not np.array_equal(mine, t) or not np.array_equal(native, native_reference):
            raise AssertionError(f"Wrapper and upstream masks differ for {f.name}")
    if scores:
        print(f"mean {np.mean(scores):.4f} over {len(scores)} images")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
