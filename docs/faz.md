# Foveal avascular zone

The foveal avascular zone (FAZ) is the central capillary-free region of the macula. This module estimates its outline and measures its area and shape. Two detectors are available: one works on image intensities and edges, and the other works on a vessel mask. Comparing their outlines can help identify scans that need closer review.

| | `detect_faz` | `faz_from_vessels` |
|---|---|---|
| Idea | edges and region growing on the image | largest vessel-free region in a vessel mask |
| Input | grayscale slab | binary vessel mask |
| Origin | Díaz et al. 2019 (ported) | this repository |
| Tuned for | 3 and 6 mm scans, about 320 px, superficial and deep | any scan size, any vessel segmenter |
| Fails when | edges are weak, the FAZ is not dark enough | the vessel mask is poor around the fovea |

```python
from octatools import io
from octatools.faz import detect_faz, faz_from_vessels, faz_metrics
from octatools.preprocessing import preprocess, preset
from octatools.vessels import segment_classical

img = io.read_image("examples/data/synthetic_20230216_232653.png")
fov_mm = 3.0

# 1. Díaz et al. on the preprocessed image
res = detect_faz(preprocess(img, preset("faz", fov_mm)).image, fov_mm=fov_mm, deep=False)
res.found, res.area_mm2, res.mask

# 2. from any vessel mask
vessels = segment_classical(preprocess(img, preset("vessel", fov_mm)).image, fov_mm).mask
res2 = faz_from_vessels(vessels, fov_mm=fov_mm)

if res2.found:
    print(faz_metrics(res2.mask, fov_mm=fov_mm))
else:
    print("FAZ not found:", res2.reason)
```

Run the example from the repository root. Both detectors return a result with a `found` flag and a `reason` when detection fails. Check `found` before using a mask for measurements. A successful detection still needs visual review.

For a complete command-line example, run:

```bash
octatools faz examples/data/synthetic_20230216_232653.png --fov-mm 3 --method vessels --out outputs/faz
```

The command writes `faz.csv`, including a success flag and failure reason for each image. When detection succeeds, it also saves `<name>_faz.png` and `<name>_faz_overlay.png`. Use `--method diaz` to try the image-based detector, and add `--layer deep` when applying that detector to a deep-slab export. The command accepts source OCTA images; `--method vessels` generates its own classical vessel mask. To supply an existing vessel mask, use `faz_from_vessels` in Python.

## Díaz et al. method

Original paper: M. Díaz, J. Novo, P. Cutrín, F. Gómez-Ulla, M. G. Penedo, M. Ortega. *Automatic segmentation of the foveal avascular zone in ophthalmological OCT-A images.* PLoS ONE 14(2): e0212364, 2019. Original code: [macarenadiaz/FAZ_Extraction](https://github.com/macarenadiaz/FAZ_Extraction).

The detector follows the steps in the original code:

1. **Apply a top-hat filter.** The filter uses an elliptical element whose size depends on the image size and scan width. This flattens the background and keeps bright structures.
2. **Detect edges after smoothing.** The Canny detector uses a high threshold set from the mean intensity and the `precision` parameter. Edges are closed morphologically so the capillary ring around the FAZ forms a barrier.
3. **Choose a starting region.** Pixels without edges in the central half of the image form candidate regions after a morphological opening. The largest becomes the seed.
4. **Region growing.** The seed is optionally eroded when it is large, then grown. At each pass, pixels next to the border are added if they are darker than the region's mean level plus a small tolerance. It stops when the region stops changing.
5. **Clean the resulting mask.** After a small morphological opening and closing, the largest contour becomes the FAZ outline.

What changed in this port, and what did not, is listed at the top of [`diaz.py`](../src/octatools/faz/diaz.py). The port supports Python 3 and OpenCV 4, replaces loops with vectorised operations, reports failed detections explicitly, and can resize inputs to the 320 px working size used by the original constants. Two oddities of the original are kept so numbers stay comparable (a branch for 6 mm that is overwritten, and a morphological closing whose result is discarded).

**`precision`** (0 to 1) controls how freely the region grows. The authors' example uses 0.7. Lower is more conservative.

**`work_size`** (default 320) resizes the input before running. The kernel sizes and the edge threshold in the original code assume roughly that size. At very different sizes they act on a different physical scale. Set `work_size=None` if you have a reason to run at the native size.

### Know the limits

- It was developed and evaluated on OCTAGON images (3 and 6 mm, 320 × 320, one device) and was not tuned for other devices.
- It looks for edges. A shallow dark core with soft borders (common in some deep-slab images and in low-signal scans) gives weak edges and the region can grow past the visible FAZ.
- Always look at an overlay. `octatools faz` writes one whenever detection succeeds.

## Vessel-mask method

`faz_from_vessels` uses the spacing between segmented vessels to identify a central avascular region. It follows four steps:

1. Take the **distance transform** of the vessel-free pixels. Each pixel gets the radius of the largest vessel-free disc centred on it.
2. Find the pixel with the largest radius inside a **search disc** around the centre (default radius 0.9 mm). That is the core of the FAZ. If its radius is below a minimum (60 µm), report that no FAZ was found, since it is only an intercapillary gap.
3. Keep only background with at least 30 % of that radius. This removes thin leaks through gaps in the capillary ring.
4. Grow that region back by the same distance inside the background, fill small holes, and smooth with a small opening and closing.

The parameters are in the function signature: `search_radius_mm`, `min_radius_um`, `leak_fraction`, and `center_px` if you know where the fovea is. Pass `center_px` for off-centre scans. Without it the search is around the image centre.

Because it works on a mask, the result is only as good as the vessel segmentation near the fovea. Missed vessels around the FAZ can allow the estimated region to expand beyond its true boundary. False vessels within the region can make the estimate smaller or split it into separate parts.

## Measures — `faz_metrics`

| key | meaning |
|---|---|
| `area_mm2` | pixel count times pixel area |
| `perimeter_mm` | Crofton estimate, less biased than counting boundary pixels |
| `circularity` | 4πA / P², 1 for a disc |
| `acircularity` | P / (2√(πA)), 1 for a disc, larger for irregular outlines (Krawitz et al. 2017) |
| `equiv_diameter_mm` | diameter of the disc with the same area |
| `axis_ratio` | major over minor axis of the fitted ellipse |
| `solidity` | area over convex hull area |
| `centroid_x_mm`, `centroid_y_mm` | offset of the centroid from the image centre |

Only the largest connected region counts.

### Lateral scale and axial length

Area depends on the eye through ocular magnification. `scale_area_by_axial_length(area, axial_length_mm, reference_axial_length_mm)` applies the Bennett et al. (1994) relation, with lateral scale proportional to (AL − 1.82), squared for areas. The reference value is the axial length of the device's model eye; look it up in the device documentation. Linderman et al. (2017) and Sampson et al. (2017) show how large the effect can be.

## Checking results

- Look at the outline over the image. `octatools faz scan.png --fov-mm 6 --out outputs/` saves `<name>_faz_overlay.png`.
- Compare both methods on a sample. Large disagreements point to images worth reviewing.
- When you have manual outlines, [`octatools.evaluation.dice`](../src/octatools/evaluation.py) gives the overlap, and `faz_metrics` gives area differences to report next to it.
