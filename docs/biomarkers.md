# Biomarkers

[`octatools.biomarkers`](../src/octatools/biomarkers) turns a binary vessel mask into numbers. Nothing here segments vessels. If the mask is wrong the numbers are wrong, and the mask depends on the threshold or model that made it. Fix those, then report them with the result (Mehta et al. 2019, Rabiolo et al. 2018).

```python
from octatools.biomarkers import quantify, vessel_metrics, etdrs_regions

table = quantify(vessels, fov_mm=6.0, faz=faz_mask, eye="OD")
table["parafovea_noFAZ"]["vessel_area_density"]

vessel_metrics(vessels, 6.0, roi=my_region_mask, exclude=faz_mask)   # one region, your mask
```

## Measures

| key | definition | notes |
|---|---|---|
| `vessel_area_density` | vessel pixels / region pixels | called vessel density (Optovue), perfusion density (Zeiss), VAD (Chu et al. 2016) |
| `vessel_skeleton_density` | skeleton pixels / region pixels | VSD (Chu et al. 2016); less affected by vessel width than area density |
| `vessel_length_density_mm_per_mm2` | skeleton length in mm / region area in mm² | what Zeiss reports as vessel density |
| `mean_vessel_width_um` | vessel area / skeleton length | a vessel diameter index; depends on resolution and on the threshold |
| `fractal_dimension` | box-counting dimension of the mask | see Zahid et al. 2016 for use in OCTA |
| `tortuosity` | mean of arc length / chord length over skeleton branches | only branches at least 40 µm long |
| `n_branches`, `n_junction_pixels` | branch and junction counts from the skeleton | rough measures of network complexity |
| `region_area_mm2` | area of the region analysed | |

The topology measures (`tortuosity`, branch counts) are off by default in `quantify` because they are the slowest. Turn them on with `with_topology=True`.

### Details that change the numbers

- **Skeleton length.** Counts orthogonal links as 1 and diagonal links as √2, and skips a diagonal link that only short-cuts an L-shaped corner. A straight horizontal line of N pixels has length N − 1, a diagonal of N pixels (N − 1)·√2. The skeleton is computed on the *whole* mask and then restricted to the region, so vessels are not cut at the region border.
- **Tortuosity.** The skeleton is split at junctions (pixels with three or more neighbours, plus their neighbours). Each branch is walked from one end to the other. The result is arc length over end-to-end distance, so a straight branch gives 1.0. Very short branches are ignored because on a pixel grid their ratio is dominated by quantisation. Small spurs from skeletonising a ragged mask inflate branch counts; `octatools.skeleton.prune` removes terminal branches below a length you choose.
- **Fractal dimension.** Counts occupied boxes at sizes from 2 pixels to a quarter of the shorter side and fits log N against log(1/size). A filled square gives about 2 and a single line about 1; vessel masks fall between. The dimension depends on the field of view and resolution, so only compare it between scans of the same kind.
- **Width.** `mean_vessel_width_um` is only as accurate as the mask. Dilated masks give wider vessels; thin masks give thin ones. Use it to compare methods on the same data, not as a calibre measurement.

## Regions

`etdrs_regions(shape, fov_mm, center_px=None, eye=None)` returns boolean masks centred on the fovea (or on `center_px`):

| name | what |
|---|---|
| `whole` | the whole image |
| `fovea` | central disc, 1 mm diameter |
| `parafovea` | ring from 1 mm to 3 mm diameter |
| `perifovea` | ring from 3 mm to 6 mm diameter, only when the scan is wider than 3 mm |
| `parafovea_superior`, `_inferior`, `_nasal`, `_temporal` | quadrants of the parafovea, given `eye="OD"` or `"OS"` |

The rings follow the ETDRS grid. Parts of a ring that fall outside the image are simply missing, so for a 3 × 3 mm scan the 3 mm ring touches the image edge and the corners of the field are not in any ring.

**Nasal and temporal depend on the device's display convention.** The default assumes fundus-style orientation: for a right eye (OD) nasal is on the right of the image, for a left eye (OS) on the left. Some exports are mirrored. Check one image where the optic disc or a known landmark is visible and pass `nasal_side="left"` or `"right"` explicitly if the default is wrong.

`quantify` also reports each region with the FAZ removed, as `<region>_noFAZ`. A large FAZ extends into the parafoveal ring and lowers its density, so the "no FAZ" version separates capillary dropout from a larger avascular zone.

## Things to keep in mind

- **Quality first.** Vessel density falls with signal strength even in healthy eyes (Lim et al. 2018, Yu et al. 2019). Run [`octatools.qc`](qc.md) and drop or flag images below your floor before any group comparison.
- **Use the same slab and device.** Plexus boundaries differ between instruments (Spaide and Curcio 2017).
- **Axial length.** Lengths and areas scale with it. Correct, or at least record it (Linderman et al. 2017, Sampson et al. 2017). See [faz.md](faz.md).
- **Model trained on 3 mm.** The learned segmenter was trained on 3 × 3 mm scans. Measures from its tiled output on larger scans are plausible but should be validated against your own reference before use. See [vessel_segmentation.md](vessel_segmentation.md).
