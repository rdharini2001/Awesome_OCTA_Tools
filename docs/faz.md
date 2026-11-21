# Foveal avascular zone

Two detectors are included. They answer the same question in different ways, and it is worth running both when the answer matters.

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

img = io.read_image("scan.png")

# 1. Díaz et al. on the preprocessed image
res = detect_faz(preprocess(img, preset("faz", 6.0)).image, fov_mm=6.0, deep=False)
res.found, res.area_mm2, res.mask

# 2. from any vessel mask
vessels = segment_classical(preprocess(img, preset("vessel", 6.0)).image, 6.0).mask
res2 = faz_from_vessels(vessels, fov_mm=6.0)

faz_metrics(res2.mask, fov_mm=6.0)
```

Both return a result with `found`. When nothing sensible is there (a dense vessel mask, a dark image, nothing near the centre) they say so and give a reason, and they do not return a plausible-looking shape.

## Díaz et al. method

Original paper: M. Díaz, J. Novo, P. Cutrín, F. Gómez-Ulla, M. G. Penedo, M. Ortega. *Automatic segmentation of the foveal avascular zone in ophthalmological OCT-A images.* PLoS ONE 14(2): e0212364, 2019. Original code: [macarenadiaz/FAZ_Extraction](https://github.com/macarenadiaz/FAZ_Extraction).

The steps, as in the original code:

1. **Top-hat filter** with an elliptical element whose size depends on the image size and the scan width. This flattens the background and keeps bright structures.
2. **Canny edges** after Gaussian smoothing, with a high threshold set from the mean intensity and the `precision` parameter. Edges are closed morphologically so the capillary ring around the FAZ forms a barrier.
3. **Seed.** The pixels with no edge, restricted to the central half of the image and opened, give candidate regions. The largest becomes the seed.
4. **Region growing.** The seed is optionally eroded when it is large, then grown. At each pass, pixels next to the border are added if they are darker than the region's mean level plus a small tolerance. It stops when the region stops changing.
5. **Cleaning.** A small opening and closing, then the largest contour is the FAZ.

What changed in this port, and what did not, is listed at the top of [`diaz.py`](../src/octatools/faz/diaz.py). In short: Python 3 and OpenCV 4, vectorised loops that give the same result, a clean failure instead of an exception, and an option to resize to the 320 px the constants were tuned for. Two oddities of the original are kept so numbers stay comparable (a branch for 6 mm that is overwritten, and a morphological closing whose result is discarded).

**`precision`** (0 to 1) controls how freely the region grows. The authors' example uses 0.7. Lower is more conservative.

**`work_size`** (default 320) resizes the input before running. The kernel sizes and the edge threshold in the original code assume roughly that size. At very different sizes they act on a different physical scale. Set `work_size=None` if you have a reason to run at the native size.

### Know the limits

- It was developed and evaluated on OCTAGON images (3 and 6 mm, 320 × 320, one device) and was not tuned for other devices.
- It looks for edges. A shallow dark core with soft borders (common in some deep-slab images and in low-signal scans) gives weak edges and the region can grow past the visible FAZ.
- Always look at an overlay. `octatools faz` writes one for every image.

## Vessel-mask method

`faz_from_vessels` is simple on purpose.

1. Take the **distance transform** of the vessel-free pixels. Each pixel gets the radius of the largest vessel-free disc centred on it.
2. Find the pixel with the largest radius inside a **search disc** around the centre (default radius 0.9 mm). That is the core of the FAZ. If its radius is below a minimum (60 µm), report that no FAZ was found, since it is only an intercapillary gap.
3. Keep only background with at least 30 % of that radius. This removes thin leaks through gaps in the capillary ring.
4. Grow that region back by the same distance inside the background, fill small holes, and smooth with a small opening and closing.

The parameters are in the function signature: `search_radius_mm`, `min_radius_um`, `leak_fraction`, and `center_px` if you know where the fovea is. Pass `center_px` for off-centre scans. Without it the search is around the image centre.

Because it works on a mask, the result is only as good as the vessel segmentation near the fovea. Capillaries missed inside the FAZ make it smaller; false vessels from noise cut it up.

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
