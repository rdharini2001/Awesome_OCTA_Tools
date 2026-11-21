# Testing with simulated images

`octatools.simulate` generates a small test image with a known vessel mask and a circular FAZ. It also provides functions that add controlled noise, blur, motion lines, and missing signal. These examples help you check how the code behaves when an image changes.

The procedural image is a simplified model of a vascular network. It is useful for testing calculations and failure cases, but it does not reproduce the full anatomy or acquisition process of a patient scan. It is separate from the published synthetic examples in `examples/data`.

## Create and save an image

After installing the core package, run this Python code from the repository root:

```python
from octatools import io
from octatools.simulate import phantom, add_motion_lines
from octatools.qc import assess

sample = phantom(size=304, fov_mm=3.0, seed=0)
degraded = add_motion_lines(sample.image, n=3, seed=0)

io.save_image("outputs/simulation/image.png", sample.image)
io.save_image("outputs/simulation/vessels.png", sample.vessels)
io.save_image("outputs/simulation/faz.png", sample.faz)
io.save_image("outputs/simulation/motion.png", degraded)

for label, image in [("Original", sample.image), ("With motion lines", degraded)]:
    report = assess(image, fov_mm=sample.fov_mm)
    print(label, report.grade, report.score, report.flags)
```

The result contains `image`, the noisy image; `clean`, the image before speckle and noise; and the boolean reference masks `vessels` and `faz`. It also records the field of view and FAZ radius. Use the same seed to reproduce an example.

## Compare measurements against known masks

You can measure the reference vessel mask directly to separate errors in measurement code from errors in segmentation:

```python
from octatools.biomarkers import vessel_metrics
from octatools.faz import faz_metrics

print(vessel_metrics(sample.vessels, sample.fov_mm))
print(faz_metrics(sample.faz, sample.fov_mm))
```

This snippet continues from the previous example. To test a segmentation method, run it on `sample.image` and compare its mask against `sample.vessels` using `octatools.evaluation.dice` or `cldice`.

Other available degradations include `add_noise`, `add_blur`, `attenuate`, and `add_black_border`. Their parameters are documented in [simulate.py](../src/octatools/simulate.py). Change one artifact at a time when you want to understand its effect on a quality score or measurement.
