# Vessel segmentation

Two segmenters, same output type (`VesselSegmentation` with `mask`, `vesselness` or probability, and `method`).

| | classical | learned |
|---|---|---|
| Call | `segment_classical(img, fov_mm)` | `OCTASegmenter().segment(img, fov_mm)` |
| Needs | numpy, scikit-image | torch, MONAI, 30 MB of weights |
| Speed | well under a second at 300 px | a few seconds per 3 mm tile on a CPU |
| Origin | Sato, Frangi, Meijering ridge filters | Kreitner et al. 2024 |
| Strength | transparent, no training data, any scale | much better on capillaries, trained without manual labels |
| Weakness | threshold is a guess; picks up speckle | trained on 3 × 3 mm scans from one device family; domain shift is possible |

```python
from octatools import io
from octatools.vessels import segment_classical, OCTASegmenter

img = io.read_image("scan.png")

seg = segment_classical(img, fov_mm=3.0)                 # classical
seg = OCTASegmenter().segment(img, fov_mm=3.0)           # learned; fetches weights on first use
seg.mask, seg.vesselness
```

## Classical

`vesselness` runs a multi-scale ridge filter. Scales are in micrometres, so `sigmas_um=(10, 20)` means the same physical widths at any resolution.

| filter | idea | reference |
|---|---|---|
| `sato` (default) | eigenvalues of the Hessian, tuned to bright tubes | Sato et al. 1998 |
| `frangi` | Hessian eigenvalue ratios to separate tubes from blobs and plates | Frangi et al. 1998 |
| `meijering` | a variant of Frangi with a neurite-tracing weighting | Meijering et al. 2004 |

The response is divided by its 99.5th percentile so values are comparable between images. `segment_classical` then thresholds with hysteresis (keep weak responses only when connected to strong ones) and removes components below an area in µm². `otsu` and `phansalkar` (Phansalkar et al. 2011, a local threshold made for low-contrast images) are in `octatools.vessels.threshold` if you want to try different binarisation.

**Where the defaults come from.** Sato, scales of 10 and 20 µm, and hysteresis at 0.2 and 0.4 were picked by a small grid search over filters, scales and thresholds on three simulated images with known vessels, and came out as a broad plateau rather than a sharp optimum. On those images the Dice score against the simulation's vessel map is about 0.73. That is a sanity check, not a validation. Real images need different settings often enough that you should look at an overlay (`octatools vessels ... ` writes the mask; `octatools.viz.overlay_mask` draws it).

Dice rewards getting wide vessels right and says little about capillaries. `octatools.evaluation.cldice` (Shit et al. 2021) is closer to what matters for thin networks.

## Learned: Kreitner et al. 2024

The network is a 2-D DynUNet trained **without human annotations**. The authors simulate retinal vessel networks by space colonisation, render them as OCTA-like images, and translate their contrast to match real scans with a GAN. The segmentation network is trained on the simulated labels and the translated images, then applied to real images. Details are in the paper (Kreitner et al. 2024, building on Menten et al. 2022) and the authors' repository, <https://github.com/aiforvision/OCTA-autosegmentation>.

What this package does with it:

- **Weights are downloaded, not stored here.** `octatools weights` (or the first call to `OCTASegmenter()`) fetches `ves_seg-S-GAN/checkpoints/30_model.pth` from the authors' repository at a fixed commit and checks its SHA-256. The cache directory is `~/.cache/octatools` or `$OCTATOOLS_CACHE`.
- **Same preprocessing and post-processing as the authors' test configuration.** Grayscale, min-max scaling, bilinear resize to 1216 × 1216, sigmoid, 0.5 threshold, removal of objects below 128 pixels at the model scale. Their rotate-and-flip step cancels out on a square image, so it is not repeated.
- **Tiling for bigger fields.** The network saw 3 × 3 mm at 1216 pixels. For a 6 × 6 mm scan the image is cut into 3 mm tiles, each predicted separately, and the probabilities are blended with a smooth window. `tile_overlap=0` (default) gives four tiles for 6 mm. Raise it to 0.25 or 0.5 to reduce seams at the cost of more tiles. Do not feed a whole 6 mm image into a model trained at 3 mm; vessels will be twice as thin as it expects.
- **Optional pre-smoothing.** `smooth_sigma_um=10` to 15 applies a Gaussian before prediction. Speckle-heavy images can be read as vessels by a network that has only seen clean simulated capillaries. This is a hint from visual inspection, not a tuned setting.

### Verifying the port

`scripts/verify_against_upstream.py` runs the authors' own `test.py` and this wrapper on the same images and reports the Dice agreement between the two masks. Run it after changing anything in `synthetic.py`.

### Limits, stated plainly

- Trained and evaluated on 3 × 3 mm macular scans, mainly from one device family. Other devices, other contrast, other slabs (deep, choriocapillaris) are outside what it was built for. The authors provide a GAN and training notebooks to adapt it to a new device; see their README.
- The sample images in `examples/data` are samples from the authors' *synthetic* dataset. They show the pipeline working, and they are in the model's own training domain. They say nothing about real-data accuracy.
- Detail is limited by the OCTA image. The model draws a plausible capillary network; where the image has no information, capillaries cannot be recovered.
- Vessel density from this model depends on its training-time label thickness. The authors note that the `min_radius` used for training should match the label detail of the evaluation dataset. Compare densities only between runs that use the same model and settings.

## Which to use

Use classical to get started and as a baseline you can explain. Use the learned model when capillary detail matters and your data resemble its training scans. When both agree on the FAZ and on large-vessel structure and disagree on capillaries, that disagreement is the uncertainty in your capillary measures.
