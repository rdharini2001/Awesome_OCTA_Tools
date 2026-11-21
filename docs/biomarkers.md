# Biomarkers

[`octatools.biomarkers`](../src/octatools/biomarkers) measures the area, length, width, and structure of vessels in a binary mask. It takes an existing segmentation, so inspect that mask before calculating measurements. The threshold or model used to create it is part of the measurement method and should be reported with the results (Mehta et al. 2019, Rabiolo et al. 2018).

```python
from octatools import io
from octatools.vessels import OCTASegmenter
from octatools.faz import detect_faz
from octatools.biomarkers import quantify, vessel_metrics, etdrs_regions

img = io.read_image("examples/data/synthetic_20230216_232653.png")
fov_mm = 3.0
vessels = OCTASegmenter().segment(img, fov_mm).mask
faz = detect_faz(img, fov_mm)
faz_mask = faz.mask if faz.found else None

table = quantify(vessels, fov_mm=fov_mm, faz=faz_mask, eye="OD")
print(table["parafovea"]["vessel_area_density"])
if faz.found:
    print(table["parafovea_noFAZ"]["vessel_area_density"])

# Measure a single region using the same vessel mask.
regions = etdrs_regions(vessels.shape, fov_mm)
print(vessel_metrics(vessels, fov_mm, roi=regions["parafovea"], exclude=faz_mask))
```

Run the example from the repository root after installing the package with `python -m pip install -e ".[deep]"`. This example uses the same vessel and FAZ methods as the default pipeline. `vessels` must be a 2-D boolean mask in which `True` means vessel. Any FAZ or region mask must use the same shape and pixel coordinates. If you load a saved black-and-white mask, convert it explicitly with `io.read_image("vessels.png") > 0.5`.

`quantify` returns a dictionary of region names, each containing a dictionary of measurements. Density fractions are between 0 and 1; multiply by 100 to express them as percentages. Regions ending in `_noFAZ` are available only when a FAZ mask is supplied.

To save the standard measurements without writing Python code, run:

```bash
octatools run examples/data/synthetic_20230216_232653.png --fov-mm 3 --eye OD --out outputs/biomarkers
```

Open `outputs/biomarkers/results.csv` for whole-image and FAZ-excluded parafoveal measurements. Use the Python API for all regions, custom masks, or topology measures.

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

The topology measures, including tortuosity and branch counts, take more time to compute and are disabled by default in `quantify`. Set `with_topology=True` when you need them.

### Details that change the numbers

- **Skeleton length accounts for the direction of each link.** The calculation counts orthogonal links as 1 and diagonal links as √2, while skipping diagonal links that would cut across an L-shaped corner. A straight horizontal line of N pixels has length N − 1, while a diagonal line of N pixels has length (N − 1)·√2. The skeleton is computed on the *whole* mask and then restricted to the region, so vessels are not cut at the region border.
- **Tortuosity.** The skeleton is split at junctions (pixels with three or more neighbours, plus their neighbours). Each branch is walked from one end to the other. The result is arc length over end-to-end distance, so a straight branch gives 1.0. Very short branches are ignored because on a pixel grid their ratio is dominated by quantisation. Small spurs from skeletonising a ragged mask inflate branch counts; `octatools.skeleton.prune` removes terminal branches below a length you choose.
- **Fractal dimension describes how the mask fills space.** The calculation counts occupied boxes at sizes from 2 pixels to a quarter of the shorter image dimension, then fits log N against log(1/size). A filled square gives about 2 and a single line about 1; vessel masks fall between. The dimension depends on the field of view and resolution, so only compare it between scans of the same kind.
- **Width.** `mean_vessel_width_um` is only as accurate as the mask. Dilated masks give wider vessels; thin masks give thin ones. Use it to compare methods on the same data, not as a calibre measurement.

## Regions

`etdrs_regions(shape, fov_mm, center_px=None, eye=None)` returns boolean masks centred on the image by default. Set `center_px=(row, column)` to place them at a known foveal location:

| name | what |
|---|---|
| `whole` | the whole image |
| `fovea` | central disc, 1 mm diameter |
| `parafovea` | ring from 1 mm to 3 mm diameter |
| `perifovea` | ring from 3 mm to 6 mm diameter, only when the scan is wider than 3 mm |
| `parafovea_superior`, `_inferior`, `_nasal`, `_temporal` | quadrants of the parafovea, given `eye="OD"` or `"OS"` |

The rings follow the ETDRS grid. Parts of a ring that fall outside the image are simply missing, so for a 3 × 3 mm scan the 3 mm ring touches the image edge and the corners of the field are not in any ring.

**Nasal and temporal depend on the device's display convention.** The default assumes fundus-style orientation: for a right eye (OD) nasal is on the right of the image, for a left eye (OS) on the left. Some exports are mirrored. Check one image where the optic disc or a known landmark is visible and pass `nasal_side="left"` or `"right"` explicitly if the default is wrong.

`quantify` also reports each region with the FAZ removed, as `<region>_noFAZ`. A FAZ that extends into the parafoveal ring changes the area available for vessels. Reporting both versions helps show how excluding that region affects the density; it does not by itself establish the cause of a low measurement.

## Compare measurements consistently

- **Quality first.** Vessel density falls with signal strength even in healthy eyes (Lim et al. 2018, Yu et al. 2019). Run [`octatools.qc`](qc.md) and drop or flag images below your floor before any group comparison.
- **Use the same slab and device.** Plexus boundaries differ between instruments (Spaide and Curcio 2017).
- **Axial length affects physical scale.** Record it and account for ocular magnification when comparing lengths or areas (Linderman et al. 2017, Sampson et al. 2017). See [faz.md](faz.md).
- **The learned model was trained on 3 × 3 mm scans.** Check measurements from its tiled output on larger fields against your own reference masks before relying on them. See [vessel_segmentation.md](vessel_segmentation.md).
