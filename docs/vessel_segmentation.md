# Vessel segmentation

This module identifies vessel pixels in an en-face image. The default command uses the pretrained network from OCTA-autosegmentation by Kreitner et al. A classical filter-based method remains available as an explicit alternative. Both return a `VesselSegmentation` result with a boolean `mask`, a `vesselness` response or probability map, and the method name. The shared output format lets you use either result with the FAZ and biomarker modules.

| | classical | learned |
|---|---|---|
| Call | `segment_classical(img, fov_mm)` | `OCTASegmenter().segment(img, fov_mm)` |
| Needs | numpy, scikit-image | torch, MONAI, 30 MB of weights |
| Speed | well under a second at 300 px | a few seconds per 3 mm tile on a CPU |
| Origin | Sato, Frangi, Meijering ridge filters | Kreitner et al. 2024 |
| Strength | Provides an interpretable baseline without training data. | Can capture fine capillary structure and was trained without manual labels. |
| Weakness | Thresholds need adjustment, and speckle can be mistaken for vessels. | Performance may change on scans from other devices or outside its 3 × 3 mm training domain. |

```python
from octatools import io
from octatools.vessels import OCTASegmenter

img = io.read_image("examples/data/synthetic_20230216_232653.png")
segmenter = OCTASegmenter(device="cpu")
seg = segmenter.segment(img, fov_mm=3.0)
io.save_image("outputs/vessels.png", seg.mask)
print("Vessel area fraction:", seg.mask.mean())
```

Install the model dependencies with `python -m pip install -e ".[deep]"`, then run the Python example from the repository root. The first model call downloads the weights if needed. For the equivalent command-line analysis:

```bash
octatools vessels examples/data/synthetic_20230216_232653.png --fov-mm 3 --out outputs/vessels
```

This saves `outputs/vessels/synthetic_20230216_232653_vessels.png`, with vessels in white and background in black. The terminal prints the fraction of vessel pixels. The CLI saves only the binary mask; use the Python result for the continuous response map. The model receives the source image and applies its own intensity scaling and resizing. Add `--preprocess` only if you deliberately want the package’s denoising and background correction before inference.

## Classical

Select this alternative explicitly:

```bash
octatools vessels examples/data/synthetic_20230216_232653.png --fov-mm 3 --method classical --preprocess --out outputs/classical
```

From Python, call `segment_classical` on an image prepared with `preprocess(img, preset("vessel", fov_mm))`. This method does not need PyTorch or downloaded weights.


`vesselness` runs a multi-scale ridge filter. Scales are in micrometres, so `sigmas_um=(10, 20)` means the same physical widths at any resolution.

| filter | idea | reference |
|---|---|---|
| `sato` (default) | eigenvalues of the Hessian, tuned to bright tubes | Sato et al. 1998 |
| `frangi` | Hessian eigenvalue ratios to separate tubes from blobs and plates | Frangi et al. 1998 |
| `meijering` | a variant of Frangi with a neurite-tracing weighting | Meijering et al. 2004 |

The response is divided by its 99.5th percentile so values are comparable between images. `segment_classical` then thresholds with hysteresis (keep weak responses only when connected to strong ones) and removes components below an area in µm². `otsu` and `phansalkar` (Phansalkar et al. 2011, a local threshold made for low-contrast images) are in `octatools.vessels.threshold` if you want to try different binarisation.

The default settings use the Sato filter at scales of 10 and 20 µm, with hysteresis thresholds of 0.2 and 0.4. They were selected by comparing filters, scales, and thresholds on three simulated images with known vessel masks. Several nearby settings performed similarly, and the selected settings achieved a Dice score of about 0.73 on those images. This small comparison helps check the implementation, but does not establish suitable settings for other datasets. Inspect an overlay on your own scans before using these defaults. The CLI saves the mask, and `octatools.viz.overlay_mask` can draw it over the image in Python.

Dice measures pixel overlap, so large vessels can dominate the score even when thin capillaries are missed. The `octatools.evaluation.cldice` measure (Shit et al. 2021) also evaluates the vessel skeleton and can help assess whether the network remains connected.

## Learned: Kreitner et al. 2024

The network is a 2-D DynUNet trained **without human annotations**. The authors simulate retinal vessel networks by space colonisation, render them as OCTA-like images, and translate their contrast to match real scans with a GAN. The segmentation network is trained on the simulated labels and the translated images, then applied to real images. Details are in the paper (Kreitner et al. 2024, building on Menten et al. 2022) and the authors' repository, <https://github.com/aiforvision/OCTA-autosegmentation>.

The wrapper handles the model inputs and outputs as follows:

- **Weights are downloaded, not stored here.** `octatools weights` (or the first call to `OCTASegmenter()`) fetches `ves_seg-S-GAN/checkpoints/30_model.pth` from the authors' repository at a fixed commit and checks its SHA-256. The cache directory is `~/.cache/octatools` or `$OCTATOOLS_CACHE`.
- **Inference follows the authors’ test configuration.** The wrapper uses a grayscale image, scales its intensities to [0, 1], and resizes it to 1216 × 1216 pixels using bilinear interpolation. It applies a sigmoid to the network output, thresholds probabilities at 0.5, and removes objects smaller than 128 pixels at the model scale. The authors’ image reader transposes the spatial axes; their rotation and flip restore the source orientation. Arrays loaded here already use source row and column coordinates, so the wrapper does not repeat those operations. For a 3 mm scan, thresholding and small-object removal happen at the model resolution before the binary mask is resized to the input grid with nearest-neighbour interpolation.
- **Larger fields are processed in tiles.** The network was trained on 3 × 3 mm images at 1216 pixels across. For a 6 × 6 mm scan the image is cut into 3 mm tiles, each predicted separately, and the probabilities are blended with a smooth window. `tile_overlap=0` (default) gives four tiles for 6 mm. Raise it to 0.25 or 0.5 to reduce seams at the cost of more tiles. Tiling is an extension provided by this package; it was not part of the upstream 3 mm test configuration. On tiled fields, small-object removal runs on the blended input-resolution mask with the cleanup area scaled to that grid. Do not feed a whole 6 mm image into a model trained at 3 mm; vessels will be twice as thin as it expects.
- **You can smooth an image before prediction.** Setting `smooth_sigma_um` to a value between 10 and 15 applies a Gaussian filter before inference. Speckle-heavy images can be read as vessels by a network that has only seen clean simulated capillaries. This option was motivated by visual inspection and needs evaluation on your own data.

### Run the learned model

Install its optional dependencies and fetch the weights:

```bash
python -m pip install -e ".[deep]"
octatools weights
octatools vessels examples/data/synthetic_20230216_232653.png --fov-mm 3 --method learned --out outputs/learned
```

The command uses the CPU by default. Add `--device cuda` to use an available CUDA GPU with a compatible PyTorch installation. For an existing local weight file, pass `--weights /path/to/30_model.pth`. Keep the default cache or set `OCTATOOLS_CACHE` to a writable directory before downloading. A cached, verified checkpoint lets you run inference without downloading it again.

From Python, reuse one model instance when processing multiple images:

```python
from octatools import io
from octatools.vessels import OCTASegmenter

img = io.read_image("examples/data/synthetic_20230216_232653.png")
segmenter = OCTASegmenter(device="cpu")
seg = segmenter.segment(img, fov_mm=3.0)
io.save_image("outputs/learned_vessels.png", seg.mask)
```

### Verifying the port

`scripts/verify_against_upstream.py` runs the authors' own `test.py` and this wrapper on the same images and reports the Dice agreement between the two masks. Run it after changing anything in `synthetic.py`.

### What to check on your data

- The model was trained and evaluated on 3 × 3 mm macular scans, mainly from one device family. Results on other devices, contrast settings, and slabs, including deep slabs and the choriocapillaris, require separate evaluation. The authors provide a GAN and training notebooks to adapt it to a new device; see their README.
- The sample images in `examples/data` are samples from the authors' *synthetic* dataset. They show the pipeline working, and they are in the model's own training domain. They say nothing about real-data accuracy.
- Detail is limited by the OCTA image. The model draws a plausible capillary network; where the image has no information, capillaries cannot be recovered.
- Vessel density from this model depends on its training-time label thickness. The authors note that the `min_radius` used for training should match the label detail of the evaluation dataset. Compare densities only between runs that use the same model and settings.

## Which to use

The pretrained model is the default analysis method. Evaluate it on your device, slab, and scan size before interpreting capillary measurements. Use the classical method when you need a filter-based comparison or an installation without deep-learning dependencies. When the two methods disagree on capillaries, inspect those regions and compare against reference masks where available. Their disagreement can reveal sensitivity to the segmentation method, but agreement alone does not establish accuracy.
