"""Command line interface: octatools <command> ..."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import io
from .qc import QCConfig, assess_folder, fit_thresholds, write_csv


def _qc(a):
    res = assess_folder(a.input, a.fov_mm, a.csv, config=QCConfig(stripe_axes=tuple(a.stripe_axes)))
    for p, r in res:
        print(f"{p.name:40s} {r.grade:11s} score={r.score:.2f}  {'; '.join(r.flags)}")
    return 0


def _preprocess(a):
    from .preprocessing import preprocess, preset
    out = Path(a.out)
    for p in io.list_images(a.input):
        res = preprocess(io.read_image(p), preset(a.preset, a.fov_mm, crop_mm=a.crop_mm, target_um=a.target_um))
        io.save_image(out / f"{p.stem}_{a.preset}.png", res.image)
        print(p.name, "->", ", ".join(res.steps))
    return 0


def _vessels(a):
    from .preprocessing import preprocess, preset
    out = Path(a.out)
    seg = None
    if a.method == "learned":
        from .vessels import OCTASegmenter
        seg = OCTASegmenter(a.weights, a.device)
    else:
        from .vessels import segment_classical
    for p in io.list_images(a.input):
        img = io.read_image(p)
        if a.preprocess:
            img = preprocess(img, preset("vessel", a.fov_mm)).image
        res = seg.segment(img, a.fov_mm, smooth_sigma_um=a.smooth_um) if seg else segment_classical(img, a.fov_mm)
        io.save_image(out / f"{p.stem}_vessels.png", res.mask)
        print(p.name, f"vessel pixels: {res.mask.mean():.3f}")
    return 0


def _faz(a):
    from .faz import detect_faz, faz_from_vessels, faz_metrics
    from .preprocessing import preprocess, preset
    from .viz import overlay_outline
    out = Path(a.out)
    rows = []
    for p in io.list_images(a.input):
        img = io.read_image(p)
        if a.method == "diaz":
            r = detect_faz(img, a.fov_mm, deep=a.layer == "deep")
            mask, ok, why = r.mask, r.found, r.reason
        else:
            from .vessels import segment_classical
            v = segment_classical(preprocess(img, preset("vessel", a.fov_mm)).image, a.fov_mm)
            r = faz_from_vessels(v.mask, a.fov_mm)
            mask, ok, why = r.mask, r.found, r.reason
        row = {"file": p.name, "method": a.method, "found": ok, "reason": why}
        if ok:
            row.update(faz_metrics(mask, a.fov_mm))
            io.save_image(out / f"{p.stem}_faz.png", mask)
            io.save_image(out / f"{p.stem}_faz_overlay.png", overlay_outline(img, mask))
        rows.append(row)
        print(p.name, "area_mm2=%.3f" % row["area_mm2"] if ok else f"not found ({why})")
    write_csv(out / "faz.csv", rows)
    return 0


def _run(a):
    from .analyze import analyze
    from .viz import overlay_mask, overlay_outline, side_by_side
    out = Path(a.out)
    seg = None
    if a.vessels == "learned":
        from .vessels import OCTASegmenter
        seg = OCTASegmenter(a.weights, a.device)
    rows = []
    for p in io.list_images(a.input):
        img = io.read_image(p)
        res = analyze(img, a.fov_mm, layer=a.layer, vessel_method=a.vessels, faz_method=a.faz_method,
                      eye=a.eye, segmenter=seg, skip_if_rejected=a.skip_rejected)
        row = {"file": p.name, "vessel_method": a.vessels, "faz_method": a.faz_method,
               **res.qc.flat(), **res.faz_stats}
        for region in ("whole", "parafovea_noFAZ"):
            for k, v in res.biomarkers.get(region, {}).items():
                row[f"{region}.{k}"] = v
        rows.append(row)
        panel = overlay_mask(res.image, res.vessels, alpha=0.35)
        if res.faz is not None:
            panel = overlay_outline(panel, res.faz)
        summary = side_by_side(img, panel) if a.vessels == "learned" else side_by_side(img, res.image, panel)
        io.save_image(out / f"{p.stem}_summary.png", summary)
        print(f"{p.name}: QC {res.qc.grade} ({res.qc.score:.2f})", *res.notes)
    write_csv(out / "results.csv", rows)
    return 0


def _weights(a):
    from .vessels.weights import ensure_weights
    print(ensure_weights(a.path))
    return 0


def _fit_qc(a):
    import csv
    with open(a.table) as f:
        rows = list(csv.DictReader(f))
    labels = [int(r[a.label_column]) for r in rows]
    table = [{k[2:]: float(v) for k, v in r.items() if k.startswith("m_") and v != ""} for r in rows]
    for r in fit_thresholds(table, labels):
        print(json.dumps(r))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="octatools", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, need_fov=True):
        p.add_argument("input", help="image file or folder")
        p.add_argument("--fov-mm", type=float, required=need_fov, help="scan width in mm (3 or 6 for most macular scans)")

    p = sub.add_parser("qc", help="score image quality")
    common(p, need_fov=False)
    p.add_argument("--csv", help="write all measures to this CSV")
    p.add_argument("--stripe-axes", type=int, nargs="+", default=[0], help="0 horizontal lines, 1 vertical lines")
    p.set_defaults(fn=_qc)

    p = sub.add_parser("preprocess", help="destripe, denoise, flatten, stretch")
    common(p)
    p.add_argument("--out", required=True)
    p.add_argument("--preset", choices=["vessel", "faz", "minimal"], default="vessel")
    p.add_argument("--crop-mm", type=float)
    p.add_argument("--target-um", type=float)
    p.set_defaults(fn=_preprocess)

    p = sub.add_parser("vessels", help="segment vessels")
    common(p)
    p.add_argument("--out", required=True)
    p.add_argument("--method", choices=["learned", "classical"], default="learned",
                   help="vessel method (default: learned, from OCTA-autosegmentation)")
    p.add_argument("--weights")
    p.add_argument("--device", default="cpu")
    p.add_argument("--smooth-um", type=float, help="Gaussian pre-smoothing for the learned model")
    p.add_argument("--preprocess", action="store_true")
    p.set_defaults(fn=_vessels)

    p = sub.add_parser("faz", help="segment the foveal avascular zone")
    common(p)
    p.add_argument("--out", required=True)
    p.add_argument("--method", choices=["diaz", "vessels"], default="diaz",
                   help="FAZ method (default: diaz, from FAZ_Extraction)")
    p.add_argument("--layer", choices=["superficial", "deep"], default="superficial")
    p.set_defaults(fn=_faz)

    p = sub.add_parser("run", help="QC, preprocessing, vessels, FAZ and biomarkers")
    common(p)
    p.add_argument("--out", required=True)
    p.add_argument("--layer", choices=["superficial", "deep"], default="superficial")
    p.add_argument("--vessels", choices=["learned", "classical"], default="learned",
                   help="vessel method (default: learned, from OCTA-autosegmentation)")
    p.add_argument("--faz-method", choices=["diaz", "vessels"], default="diaz",
                   help="FAZ method (default: diaz, from FAZ_Extraction)")
    p.add_argument("--eye", choices=["OD", "OS"])
    p.add_argument("--weights")
    p.add_argument("--device", default="cpu")
    p.add_argument("--skip-rejected", action="store_true")
    p.set_defaults(fn=_run)

    p = sub.add_parser("weights", help="download the pretrained vessel network")
    p.add_argument("--path")
    p.set_defaults(fn=_weights)

    p = sub.add_parser("fit-qc", help="rank QC measures against your own usable/unusable labels")
    p.add_argument("table", help="CSV from `octatools qc --csv` with an added label column")
    p.add_argument("--label-column", default="usable")
    p.set_defaults(fn=_fit_qc)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
