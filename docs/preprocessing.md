# Preprocessing

[`octatools.preprocessing`](../src/octatools/preprocessing) prepares en-face images for segmentation. You can use each operation independently or apply a preset that combines them in a fixed order. The default upstream vessel and FAZ methods apply their own preparation and do not use these presets automatically. Use a preset for the classical alternatives, or when deliberately testing additional preparation. Inspect the result and keep only the steps that help your data.

```python
from octatools import io
from octatools.preprocessing import preprocess, preset

img = io.read_image("examples/data/synthetic_20230216_232653.png")
out = preprocess(img, preset("vessel", fov_mm=3.0))
io.save_image("outputs/preprocessed.png", out.image)
print(out.steps)  # Operations and settings applied to this image.
# out.valid is the boolean mask of pixels treated as valid.
```

Run the Python example from the repository root after [installing the package](../README.md#install-and-run-your-first-analysis). To use the command line instead:

```bash
octatools preprocess examples/data/synthetic_20230216_232653.png --fov-mm 3 --preset vessel --out outputs/pre
```

This saves `outputs/pre/synthetic_20230216_232653_vessel.png` and prints the applied steps. The CLI saves the processed image; use the Python result if you also need the valid-pixel mask or want to save the step history. For your own images, replace the path and field of view together.

The recipe runs enabled steps in this order: crop, resample, destripe, denoise, flatten, stretch, optional CLAHE. The order matters. Line repair comes before denoising so a smoothing filter does not spread a line into its neighbours, and background flattening comes after denoising so noise does not leak into the background estimate.

## Decide which preparation steps you need

Preprocessing can make vessel and FAZ boundaries easier to segment and reduce differences in image appearance between scans. It is not meant to prepare an image for **intensity-based quantification**. Brightness, contrast and binarisation choices change vessel density numbers (Mehta et al. 2019, Rabiolo et al. 2018), so measure on a mask from a fixed, documented pipeline and not on a contrast-stretched copy.

If the vendor has already denoised and normalised your images, begin with the `minimal` preset. Compare the results before adding further filtering or contrast adjustment.

## Steps

### Identify valid pixels with `valid_mask`

Registration of repeated scans and blinks leave black areas along the border. A dark area counts as missing only if it touches the image border and is larger than a small fraction of the image. The avascular zone and genuine dropout inside the field stay valid. Every later step that averages over the image (flattening, percentile stretch, the quality module) uses this mask so the black border does not pull the statistics down.

### Crop and resample with `center_crop_mm` and `resample_to_pixel_size`

Both take sizes in millimetres or micrometres. Cropping a 6 × 6 mm scan to its central 3 × 3 mm is the usual way to make it look like the scans many segmentation networks were trained on. Resampling downsamples with anti-aliasing. Use `order=0` for masks.

### Motion and blink lines — `detect_stripes`, `repair_stripes`

Eye movement during the fast scan leaves rows that are too dark or too bright (Spaide et al. 2015). The row-by-row mean is compared with its running median; a row is flagged if it differs by more than 4 robust standard deviations **and** by more than 25 % of the local level. The second condition keeps quiet images from producing false lines. Flagged rows (grown by one) are replaced by linear interpolation between the nearest good rows.

What to expect:

- Thin lines of one to a few rows are repaired well.
- A wide band cannot be recovered. Interpolating across it only hides it. The quality module reports the longest run so you can reject such images instead.
- The repair interpolates intensities across missing rows and cannot recover the original vessel structure. Avoid interpreting capillary measurements in repaired rows as observed anatomy.
- Pass `axis=1` if the fast axis is vertical in your export.

This detector looks at the exported image. Methods that correct motion from the raw volume, using orthogonal scans or per-A-scan registration, are better when you have the volume (Kraus et al. 2012).

### Denoising — `denoise`

| method | when to use |
|---|---|
| `gaussian` | safe and predictable; blurs capillaries, so keep σ small |
| `median` | removes isolated bright specks; slightly erodes thin vessels |
| `bilateral` | smooths flat background while keeping vessel edges |
| `nlm` | non-local means (Buades et al. 2005); best at keeping thin vessels, default in the `vessel` preset |
| `tv` | total variation; makes flat regions very clean, can leave a painted look |
| `diffusion` | Perona–Malik anisotropic diffusion (1990); stops smoothing at strong gradients |

OCTA noise is not Gaussian. Decorrelation noise gives the background a bright grainy floor, and signal loss makes it worse. These filters reduce the variation caused by noise, but they cannot restore flow information that was not captured. If a segmentation model has only seen clean images, even a light filter often helps; check by eye on your own data.

### Background flattening — `flatten_background`

Signal drops towards the edges and under opacities. The trend is estimated with a Gaussian that ignores invalid pixels (normalised convolution) and divided out (`mode="divide"`, the default, which treats attenuation as multiplicative) or subtracted. The mean level is kept. Use a scale much larger than a vessel and larger than the FAZ; the recipe uses 0.6 mm.

### Intensity stretch and CLAHE — `percentile_stretch`, `clahe`

`percentile_stretch` maps the 1st and 99.5th percentile of the valid pixels to 0 and 1. It makes thresholds and network inputs comparable between scans. `clahe` (Pizer et al. 1987, Zuiderveld 1994) adds local contrast but also amplifies noise in dark regions. It is off by default.

### Projection artifacts — `regress_out_superficial`

Real projection removal works on the volume (Zhang et al. 2016, Campbell et al. 2017). For two exported slabs, the function subtracts the part of the deep slab that is linearly explained by the superficial slab. This is a heuristic. It cannot tell a true deep vessel lying under a superficial one from its shadow, so some real deep vessels will be dimmed. Use the vendor's projection-removed export when there is one, and treat this as a diagnostic. `projection_correlation` gives a single number for how much the deep slab looks like a copy of the superficial one.

## Presets

| preset | destripe | denoise | flatten | stretch | for |
|---|---|---|---|---|---|
| `vessel` | yes | non-local means | 0.6 mm | 1–99.5 % | vessel segmentation |
| `faz` | yes | Gaussian σ 1.5 px | 0.8 mm | 1–99.5 % | Optional preparation for FAZ experiments; the default Díaz path uses the source image. |
| `minimal` | no | none | none | none | already-clean data |

Build your own with `PreprocessConfig(...)` or override one setting: `preset("vessel", 6.0, denoise=None)`. The recipe is a frozen dataclass, so you can print it and store it next to your results.

## Reproducibility

`preprocess` returns the list of applied steps with their parameters (`out.steps`). Save it with your outputs and keep the preset settings consistent across scans you plan to compare. If you crop from 6 mm to 3 mm, pass the new 3 mm field of view to downstream segmentation and measurement functions.
