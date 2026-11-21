"""Regenerate the figures in assets/.

    python scripts/make_figures.py            # all
    python scripts/make_figures.py layers qc  # some: layers, qc, pipeline

The example scan is one of the simulated images from Kreitner et al. (2024).
Style: journal-like, sans-serif, bold lowercase panel letters, no titles inside
the figure (descriptions live in the README captions).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from octatools import io, simulate
from octatools.analyze import analyze
from octatools.qc import assess

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples" / "data"
OUT = ROOT / "assets"
SAMPLE = DATA / "synthetic_20230216_232653.png"

INK, MUTED, RULE = "#1a1a1a", "#6b6b6b", "#bdbdbd"
NAVY, BLUE, SKY, PALE, SAGE, TAN, RED = "#1F3A7A", "#4A7EBB", "#A9C8E8", "#D9ECF2", "#6FA76F", "#E2BF6A", "#C8403B"
GREEN, ORANGE, PURPLE = SAGE, "#D9864A", "#7A5C9E"
GRADE = {"good": SAGE, "acceptable": "#D9A520", "reject": RED}
MM = 1 / 25.4

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "legend.fontsize": 6.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 300, "figure.dpi": 150, "pdf.fonttype": 42,
})


def letter(ax, s, dx=-0.02, dy=1.0):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="right", color=INK)


def scalebar(ax, shape, fov_mm, length_mm=0.5):
    h, w = shape[:2]
    px = length_mm / fov_mm * w
    x0, y0 = 0.05 * w, 0.93 * h
    ax.add_patch(Rectangle((x0, y0), px, 0.015 * h, fc="white", ec="none"))


def layers():
    """Schematic of retinal layers and capillary plexuses (not to scale)."""
    bands = [("NFL", 0.9, "#EBDDB0"), ("GCL", 0.8, "#E6C7A8"), ("IPL", 1.5, "#DCDCEB"), ("INL", 1.2, "#CBDDE6"),
             ("OPL", 0.9, "#DCDCEB"), ("ONL", 2.0, "#BBCBD8"), ("RPE", 0.45, "#B09A90"), ("Choriocapillaris", 0.6, "#E0A0A0")]
    fig, ax = plt.subplots(figsize=(120 * MM, 92 * MM))
    x1 = 3.6
    y, edges = 0.0, {}
    for name, h, c in bands:
        ax.add_patch(Rectangle((0, y), x1, h, fc=c, ec="white", lw=1.2))
        ax.text(0.08, y + h / 2, name, va="center", fontsize=7, color=INK)
        edges[name] = (y, y + h)
        y += h
    top = y
    rng = np.random.default_rng(3)
    svp = (edges["NFL"][0] + edges["GCL"][1]) / 2
    icp = edges["IPL"][1] - 0.05
    dcp = edges["INL"][1] + 0.05
    for y_mid, col in ((svp, RED), (icp, ORANGE), (dcp, PURPLE)):
        xs = np.linspace(1.55, 3.45, 12) + rng.normal(0, 0.02, 12)
        ax.plot([1.5, 3.5], [y_mid, y_mid], color=col, lw=0.9, alpha=0.5, zorder=3)
        ax.scatter(xs, y_mid + rng.normal(0, 0.05, 12), s=14, color=col, zorder=4, ec="white", lw=0.4)
    for y_mid, name, col in ((svp, "SVP", RED), (icp, "ICP", ORANGE), (dcp, "DCP", PURPLE)):
        ax.plot([x1, x1 + 0.12], [y_mid, y_mid], color=col, lw=0.9, clip_on=False)
        ax.text(x1 + 0.18, y_mid, name, color=col, fontsize=7, fontweight="bold", va="center")

    def bracket(y0, y1, label, sub, x=4.6):
        ax.plot([x, x + 0.1, x + 0.1, x], [y0, y0, y1, y1], color=INK, lw=0.8, clip_on=False)
        ax.text(x + 0.22, (y0 + y1) / 2 - 0.12, label, va="bottom", fontsize=7, fontweight="bold", color=INK)
        ax.text(x + 0.22, (y0 + y1) / 2 - 0.05, sub, va="top", fontsize=6.5, color=MUTED)

    bracket(edges["NFL"][0], edges["GCL"][1], "SVC", "superficial vascular complex")
    bracket(edges["IPL"][1] - 0.4, edges["INL"][1] + 0.4, "DVC", "deep vascular complex")
    ax.text(x1 + 0.18, edges["Choriocapillaris"][0] + 0.3, "CC", fontsize=7, color=INK, va="center")
    two = edges["INL"][0] + 0.6
    ax.plot([0.95, x1], [two, two], ls=(0, (4, 3)), color=INK, lw=0.8)
    ax.text(0.0, -0.25, "vitreous", fontsize=6.5, color=MUTED, va="bottom")
    ax.text(0.0, top + 0.25, "choroid", fontsize=6.5, color=MUTED, va="top")
    ax.legend(handles=[Line2D([0], [0], ls=(0, (4, 3)), color=INK, lw=0.8)],
              labels=["typical two-slab split between 'superficial' and 'deep' (differs by device)"],
              loc="upper left", bbox_to_anchor=(0.0, -0.02), frameon=False, handlelength=2.2)
    ax.set_xlim(0, 6.9)
    ax.set_ylim(top + 0.6, -0.6)
    ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.1)
    fig.savefig(OUT / "retina_layers.png", facecolor="white")
    plt.close(fig)


FAMILIES = {
    "noise": ([0.0, 0.1, 0.2, 0.3, 0.4], lambda x, s: simulate.add_noise(x, s) if s else x, "σ"),
    "blur": ([0.0, 1.0, 2.0, 3.0, 5.0], lambda x, s: simulate.add_blur(x, s) if s else x, "σ px"),
    "dark lines": ([0, 2, 5, 10, 20], lambda x, s: simulate.add_motion_lines(x, int(s), thickness=3) if s else x, "n"),
    "bright lines": ([0, 2, 5, 10, 20], lambda x, s: simulate.add_motion_lines(x, int(s), thickness=3, kind="bright") if s else x, "n"),
    "black band": ([0.0, 0.05, 0.15, 0.3, 0.5], lambda x, s: simulate.add_black_border(x, s) if s else x, "frac"),
    "low signal": ([1.0, 0.6, 0.3, 0.15, 0.08], lambda x, s: simulate.attenuate(x, s), "scale"),
}
CURVE_COLORS = [NAVY, BLUE, RED, "#E08A3C", SAGE, "#8A8A8A"]


def qc():
    img = io.read_image(SAMPLE)
    fig = plt.figure(figsize=(183 * MM, 108 * MM))
    top = fig.add_gridspec(1, 7, wspace=0.07, left=0.06, right=0.985, top=0.85, bottom=0.55)
    picks = [("noise", 0.3), ("blur", 3.0), ("dark lines", 10), ("bright lines", 10), ("black band", 0.3), ("low signal", 0.15)]
    panels = [("original", img)] + [(f"{n} {s:g}", FAMILIES[n][1](img, s)) for n, s in picks]
    for i, (name, x) in enumerate(panels):
        ax = fig.add_subplot(top[0, i])
        r = assess(x, 3.0)
        ax.imshow(x, cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([]), ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True), sp.set_color(GRADE[r.grade]), sp.set_linewidth(1.4)
        ax.set_title(f"{name}\n{r.grade}, {r.score:.2f}", fontsize=6.5, color=INK, pad=3, linespacing=1.3)
        if i == 0:
            ax.text(-0.22, 1.36, "a", transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")
    ax = fig.add_axes([0.075, 0.12, 0.60, 0.30])
    ax.axhspan(0, 0.45, color=GRADE["reject"], alpha=0.08, lw=0)
    ax.axhspan(0.45, 0.70, color=GRADE["acceptable"], alpha=0.10, lw=0)
    ax.axhspan(0.70, 1.05, color=GRADE["good"], alpha=0.08, lw=0)
    curves = [("noise", "σ 0.1, 0.2, 0.3, 0.4"), ("blur", "σ 1, 2, 3, 5 px"), ("dark lines", "2, 5, 10, 20 lines"),
              ("black band", "5, 15, 30, 50 % of image"), ("low signal", "scale 0.6, 0.3, 0.15, 0.08")]
    colors = {"noise": NAVY, "blur": BLUE, "dark lines": RED, "black band": "#E08A3C", "low signal": SAGE}
    names = {"dark lines": "dark or bright lines"}
    for fam, desc in curves:
        sev, fn, _ = FAMILIES[fam]
        sc = [assess(fn(img, v), 3.0).score for v in sev]
        ax.plot(range(5), sc, marker="o", ms=3.2, color=colors[fam], lw=1.2, label=f"{names.get(fam, fam)}\n{desc}")
    for y, t, c in ((0.22, "reject", GRADE["reject"]), (0.575, "acceptable", "#A87A00"), (0.88, "good", "#3E7D3E")):
        ax.text(1.015, y, t, color=c, fontsize=6.5, va="center", ha="left", transform=ax.get_yaxis_transform(), clip_on=False)
    ax.set_xlim(-0.15, 4.15)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(range(5))
    ax.set_xticklabels(["none", "mild", "moderate", "strong", "severe"])
    ax.set_ylabel("quality score")
    ax.legend(loc="center left", bbox_to_anchor=(1.17, 0.5), frameon=False, handlelength=2.0, labelspacing=0.8)
    ax.text(-0.10, 1.1, "b", transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")
    fig.savefig(OUT / "qc_degradations.png", facecolor="white")
    plt.close(fig)


def pipeline():
    from octatools.faz import detect_faz, faz_metrics
    from octatools.vessels import OCTASegmenter
    from octatools.viz import overlay_mask, overlay_outline
    img = io.read_image(SAMPLE)
    a = analyze(img, 3.0, eye="OD", vessel_method="classical", faz_method="vessels")
    seg = OCTASegmenter().segment(img, 3.0)
    diaz = detect_faz(img, 3.0)
    f_v = faz_metrics(a.faz, 3.0)["area_mm2"]
    f_d = faz_metrics(diaz.mask, 3.0)["area_mm2"]
    both = overlay_outline(overlay_outline(a.image, a.faz, color=(0.435, 0.655, 0.435), width=2),
                           diaz.mask, color=(0.85, 0.53, 0.29), width=2)
    panels = [
        ("Input", f"quality: {a.qc.grade}, {a.qc.score:.2f}", img),
        ("Preprocessed", "denoised, flattened", a.image),
        ("Classical vessels", f"area density {a.vessels.mean():.2f}", overlay_mask(a.image, a.vessels, color=(0.29, 0.49, 0.73), alpha=0.5)),
        ("Learned vessels", f"area density {seg.mask.mean():.2f}", overlay_mask(a.image, seg.mask, color=(0.29, 0.49, 0.73), alpha=0.5)),
        ("FAZ outlines", f"{f_v:.2f} mm² / {f_d:.2f} mm²", both),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(183 * MM, 54 * MM), gridspec_kw=dict(wspace=0.04, left=0.01, right=0.99, top=0.86, bottom=0.26))
    for i, (ax, (t, sub, p)) in enumerate(zip(axes, panels)):
        ax.imshow(p, cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([]), ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(t, fontsize=7, fontweight="bold", color=INK, pad=3)
        ax.text(0.5, -0.04, sub, transform=ax.transAxes, ha="center", va="top", fontsize=6.5, color=MUTED)
        ax.text(0.0, 1.075, "abcde"[i], transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")
    scalebar(axes[0], img.shape, 3.0)
    fig.legend(handles=[Line2D([0], [0], color=SAGE, lw=1.4), Line2D([0], [0], color=ORANGE, lw=1.4)],
               labels=["FAZ, vessel-mask method", "FAZ, Díaz et al."], loc="lower right", ncol=2,
               bbox_to_anchor=(0.99, 0.0), frameon=False, fontsize=6.5, handlelength=1.6, columnspacing=1.6)
    fig.savefig(OUT / "pipeline_example.png", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for name in sys.argv[1:] or ["layers", "qc", "pipeline"]:
        globals()[name]()
        print("wrote", name)
