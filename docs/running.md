# Running the tools on your scans

This guide explains how the commands fit together and where to find their outputs. If you have not installed the package yet, follow the [installation instructions](../README.md#install-and-run-your-first-analysis). Run the examples below from the repository root with your virtual environment active.

## Start with one image

Use one of the bundled 3 × 3 mm simulated scans to check that the installation works:

```bash
octatools run examples/data/synthetic_20230216_232653.png --fov-mm 3 --eye OD --out outputs/example
```

The command checks quality, segments vessels with the pretrained OCTA-autosegmentation network, detects the FAZ with the Díaz method from FAZ_Extraction, and calculates vessel measurements. Both methods apply their own preparation to the source image. It writes a table to `outputs/example/results.csv` and a summary image to `outputs/example/synthetic_20230216_232653_summary.png`. The default summary places the original image and segmentation overlay side by side. When you select classical vessels, it also includes the preprocessed image used by that method.

Read the quality grade and inspect the vessel and FAZ outlines before using the measurements. If the FAZ cannot be found, the command prints a reason and leaves its measurements absent from that image's result. Vessel measurements are still calculated unless the image is skipped because of QC.

## Prepare a folder of scans

Each command accepts a single file or a folder containing PNG, JPEG, TIFF, or BMP images. Folder processing uses only the files directly inside that folder. Place source images in their own folder and save outputs elsewhere, so masks and summary panels are not mistaken for new inputs.

Supply an en-face slab with bright vessels against a dark background. Remove scanner interface elements and annotations. The software reads images as grayscale arrays; it does not interpret raw OCT volumes, extract retinal layers, or recover physical scale from scanner metadata.

Set `--fov-mm` to the width of the supplied image in millimetres. Process scans with different widths in separate commands. The package assumes equal horizontal and vertical pixel spacing. For cropped images, use the width of the crop rather than the original scan width.

For a folder of right-eye 6 × 6 mm scans, run:

```bash
octatools run scans/ --fov-mm 6 --eye OD --skip-rejected --out outputs/all
```

Use `OS` for left eyes. The eye setting controls nasal and temporal region labels; check that your export has the expected orientation before using those regions. The [biomarker guide](biomarkers.md#regions) explains how to override the nasal side in Python.

By default, `run` continues analysing images graded as rejected. Adding `--skip-rejected` keeps their QC results but skips segmentation and measurement. A summary image is still written for each input. The default QC thresholds need evaluation on your own data, as described in the [QC guide](qc.md#these-cut-offs-are-starting-points).

## Choose a command

Each command below can start from a source scan. The default vessel and FAZ methods apply their own preparation to the source image. You do not need to run `preprocess` first. The package’s general preprocessing preset is used for the classical alternatives or when explicitly requested with `vessels --preprocess`.

| Command | What it produces | Where to learn more |
|---|---|---|
| `qc` | Prints a grade, score, and flags. With `--csv`, it also saves the quality measurements. | [Quality control](qc.md) |
| `preprocess` | Saves a processed PNG for each image and prints the applied steps. | [Preprocessing](preprocessing.md) |
| `vessels` | Uses the pretrained OCTA-autosegmentation model and saves a binary vessel mask for each image. Additional preprocessing requires `--preprocess`. | [Vessel segmentation](vessel_segmentation.md) |
| `faz` | Uses the Díaz detector from FAZ_Extraction. It saves `faz.csv` and, when detection succeeds, a FAZ mask and an outline overlay. | [FAZ detection](faz.md) |
| `run` | Saves `results.csv` and a summary panel for each input. The CSV includes QC, available FAZ measurements, and selected vessel measurements. | [Biomarkers](biomarkers.md) |
| `weights` | Downloads and verifies the pretrained vessel checkpoint, then prints its local path. | [Learned vessel model](vessel_segmentation.md#run-the-learned-model) |
| `fit-qc` | Prints candidate thresholds from a QC CSV that you have labelled. It does not change the default configuration. | [QC calibration](qc.md#these-cut-offs-are-starting-points) |

Output directories are created as needed. Repeating a command with the same input names and output directory overwrites matching output files, so choose a different directory when comparing settings.

For the complete list of options for a command, use its help page:

```bash
octatools --help
octatools run --help
```

## Understand the full-pipeline table

`results.csv` contains one row per image and records the selected algorithms in `vessel_method` and `faz_method`. QC fields describe the input quality, FAZ fields describe a detected outline, and vessel fields use region prefixes such as `whole.` or `parafovea_noFAZ.`. Vessel area density is a fraction: a value of `0.40` means that vessels occupy 40% of the region.

The CLI exports measurements for the whole image and, when a FAZ is available, the parafoveal region with the FAZ excluded. Use `octatools.biomarkers.quantify` in Python for the complete regional table, custom regions, or topology measurements. The [biomarker guide](biomarkers.md) defines the units and calculation for each field.

The summary panel is a visual check rather than a saved collection of analysis masks. Use the individual `vessels` or `faz` commands when you need those masks, or access the arrays returned by `octatools.analyze.analyze` in Python. The Python result also contains notes about failed FAZ detection or skipped analysis.

## Resolve common setup problems

If your terminal cannot find `octatools`, activate the environment in which you installed the package. From the repository root, run `python -m pip install -e .` again with that environment's Python. You can also call the same command through `python -m octatools.cli --help`.

The default vessel analysis requires PyTorch and MONAI. If it reports missing dependencies, install them with `python -m pip install -e ".[deep]"`. The core package supports classical segmentation without PyTorch or MONAI; select it with `vessels --method classical` or `run --vessels classical`. To also use the alternative vessel-mask FAZ detector, add `--faz-method vessels` to `run`. Weight downloads need internet access and a writable cache directory; set `OCTATOOLS_CACHE` to a directory you can write to if the default cache is unavailable.

If measurements have unexpected units or scales, check the input's physical width before adjusting algorithm settings. If a folder run includes an unexpected file, check that the input folder contains only source scans. If an outline is visibly wrong, use the module guide to review the method and settings rather than interpreting the resulting number.
