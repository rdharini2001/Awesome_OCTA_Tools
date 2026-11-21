# Example OCTA images

This folder contains three simulated 3 × 3 mm OCTA images at 304 × 304 pixels and their corresponding vessel labels at 1216 × 1216 pixels. They are taken unchanged from the synthetic dataset released by [Kreitner et al. (2024)](https://github.com/aiforvision/OCTA-autosegmentation) and are redistributed under the authors’ MIT licence. Attribution and publication details are in [THIRD_PARTY.md](../../THIRD_PARTY.md).

These are generated images rather than patient scans. They come from the training domain of the learned vessel model, so use them to check installation and understand the outputs. They cannot establish performance on clinical data.

Install the package with `python -m pip install -e ".[deep]"`. From the repository root, run the default example, which uses the pretrained OCTA-autosegmentation model and the Díaz FAZ detector:

```bash
python examples/quickstart.py
```

To save measurements and a summary panel for one image, run:

```bash
octatools run examples/data/synthetic_20230216_232653.png --fov-mm 3 --eye OD --out outputs/example
```

Use `--fov-mm 3` for these images. The images and reference labels have different resolutions; align them to the same pixel grid before comparing masks, using nearest-neighbour interpolation for binary labels. Keep reference labels separate from scan inputs when processing folders.
