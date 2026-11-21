# Quality control

Poor images give wrong numbers without any warning: dim scans lose capillaries and look like low vessel density, motion lines break vessels, and a blink leaves a black band. Several studies show that vessel density falls with signal strength even in healthy eyes (Lim et al. 2018, Yu et al. 2019), and that artifacts are common, vary by disease and by device, and are often missed by simple signal-strength cut-offs (Spaide et al. 2015, Lauermann et al. 2018, Fenner et al. 2018). So every analysis should start by deciding which images to trust.

```python
from octatools import io
from octatools.qc import assess

report = assess(io.read_image("scan.png"), fov_mm=6.0)
report.grade      # "good", "acceptable" or "reject"
report.score      # 0..1
report.flags      # human-readable reasons
report.metrics    # every number behind the grade
```

```bash
octatools qc scans/ --fov-mm 6 --csv qc.csv
```

![Quality score under simulated degradation](../assets/qc_degradations.png)

**Quality score under simulated degradation.** **a**, A simulated scan and six degraded versions with the grade and score from `assess`. **b**, Score against degradation strength; shaded bands mark the grade ranges.

## What it measures

All measures come from the en-face image alone. They are computed on the valid pixels only (black borders left by registration are excluded) and, when `fov_mm` is given, at a common pixel size of about 10 µm. Resampling only ever goes down: interpolating a coarse image up would smooth its noise and make it look cleaner than it is.

| measure | what it is | what raises the alarm |
|---|---|---|
| `coverage` | share of the image that carries data | blink bands, registration loss at the border |
| `line_events` | number of separate line artifacts: rows that are far darker or brighter than their neighbours, plus abrupt jumps between neighbouring rows | motion, blinks, strip misalignment (Spaide et al. 2015) |
| `motion_fraction` | share of rows flagged as lines | the same, weighted by how much of the image is affected |
| `signal_fraction` | 1 − (noise variance) / (image variance) | grainy, low-signal images |
| `noise_sigma` | noise standard deviation, Immerkaer's estimate made robust with a median (Immerkaer 1996) | reported, not scored on its own |
| `blur` | no-reference blur score, 0 sharp to 1 blurred (Crete et al. 2007) | defocus, over-smoothing |
| `contrast` | 95th minus 5th percentile of intensity | dim, flat images |
| `clipped_fraction` | share of pixels at the top of the intensity range | saturation, pre-binarised exports |
| `peripheral_dark` | share of the periphery much darker than typical | shadows, opacities and non-perfusion alike |
| `vessel_density` | area density from a quick classical segmentation | values outside a plausible band mean something is off |
| `projection_corr` | correlation between the deep and superficial slabs, if you pass both | deep slab is mostly a copy of the superficial one (Zhang et al. 2016) |
| `signal_strength` | the device's own index, if you pass it | below a minimum you set |

Two design points worth stating:

- **Count lines, not row fractions.** On real scans a single dark line with a misregistered strip below it can ruin an image while touching only 1 % of the rows. The first version of this module scored the fraction and rated such images as good. It now counts events and treats two or more as a failure.
- **Jumps as well as lines.** Registration stitches strips together. When a strip is misplaced there may be no dark row at all, only vessels that shift across a seam. `detect_jumps` looks at the difference between neighbouring rows after a light blur; it spikes at a seam and is flat elsewhere.

## From measures to a grade

Each measure becomes a sub-score from 0 to 1 by a linear ramp between a "bad" and a "good" anchor (see `QCConfig.anchors`). The overall `score` is the weighted geometric mean of the sub-scores, so a failure in one respect cannot be averaged away by being fine in others. Then:

- **reject** if any hard-fail rule fires (coverage below 0.60, more than three line events, more than 15 % flagged rows, signal fraction below 0.25, or a device signal strength below your minimum) or the score is below 0.45;
- **good** if the score is at least 0.70 and no single measure is poor (sub-score under 0.34);
- **acceptable** otherwise.

`report.flags` says which measure did it.

### These cut-offs are starting points

The anchors were set from simulated degradations of one image (the figure above) and then checked by eye on 80 randomly chosen real 6 × 6 mm en-face images from a public dataset (AI-READI; no images are redistributed here). On those 80, 36 had no line events, 28 had one, and 16 had two or more; the images that scored lowest were visibly the worst, with broken and displaced strips. That is a sanity check, not a validation. Nobody has graded these images against this score.

For serious work, fit the thresholds to your own graded images:

```bash
octatools qc images/ --fov-mm 6 --csv qc.csv        # then add a 0/1 `usable` column by hand
octatools fit-qc qc.csv --label-column usable
```

`fit-qc` prints, for every measure, the AUC separating usable from unusable images and the cut-off that maximises Youden's J. `octatools.qc.calibrate` has the functions if you want to build your own rule. Apply the same QC rule to every group in a study, and report how many images were excluded and why.

## Device-reported signal strength

Every vendor exports some quality number. Use it as well; this module does not replace it. Pass it in and set a floor:

```python
from octatools.qc import QCConfig, assess
assess(img, 6.0, signal_strength=7, config=QCConfig(min_signal_strength=6))
```

Thresholds in the literature differ by device and study design. Many papers require at least 6 out of 10 on the Zeiss quality index and about 55 to 60 on the Optovue signal strength index; check the manual and the studies closest to yours (Lim et al. 2018 is a good start). The two are not interchangeable, and a good number does not rule out motion or segmentation artifacts, which is why image-based checks help.

## What this cannot see

- **Segmentation errors.** Boundaries drawn in the wrong place are visible only in the B-scans. An image can pass every check here and still show the wrong layer, especially in oedema, atrophy and high myopia (Spaide and Curcio 2017, Lauermann et al. 2018).
- **Real shadows versus real non-perfusion.** `peripheral_dark` flags dark areas. Without the volume the two cannot be told apart.
- **Noise in images that were interpolated.** An image that was upsampled from a coarser scan has smooth noise, so `signal_fraction` stays high. This is a property of the file, not the eye.
- **Vertical lines.** By default only horizontal lines (a horizontal fast axis) are checked. If your export is rotated, pass `QCConfig(stripe_axes=(1,))` or `(0, 1)`.
- **Subtle strip misalignment** with no intensity change can slip past the jump detector.
- **Clinical meaning.** A "good" image is one in which the measures here look fine. It is not a judgement about the eye.

## Deep-learning alternatives

Lauermann et al. 2019 trained a network to grade OCTA quality directly. If you have enough graded data, such a model can capture things hand-built measures miss. The measures here are transparent, need no training data and are easy to audit, and the CSV they produce is a convenient starting point for labelling.
