# Third-party work

This repository builds on the work of others. Please cite them when you use the matching part.

## FAZ detection: Díaz et al.

`src/octatools/faz/diaz.py` is a Python 3 port of the FAZ extraction method and code by Macarena Díaz and colleagues.

- Code: <https://github.com/macarenadiaz/FAZ_Extraction>
- M. Díaz, J. Novo, P. Cutrín, F. Gómez-Ulla, M. G. Penedo, M. Ortega. *Automatic segmentation of the foveal avascular zone in ophthalmological OCT-A images.* PLoS ONE 14(2): e0212364, 2019. <https://doi.org/10.1371/journal.pone.0212364>
- M. Díaz, J. Novo, M. G. Penedo, M. Ortega. *Automatic extraction of vascularity measurements using OCT-A images.* Procedia Computer Science 126 (KES-2018): 273–281, 2018.

The OCTAGON dataset that goes with the method is available from the authors on request (see the original repository).

**Licence status.** The original repository carries no licence file (checked at commit `4b312ef`). The port keeps the method and constants and lists every change in the module docstring, but its legal status follows the original's. If you are one of the original authors and would like the port removed, relicensed or credited differently, please open an issue.

## Vessel segmentation: Kreitner et al.

`src/octatools/vessels/synthetic.py` runs the network of Kreitner et al. Nothing from that repository is copied into the package except a re-implementation of its inference steps. The weights are downloaded from the authors' repository at a fixed commit and verified by hash.

- Code and weights: <https://github.com/aiforvision/OCTA-autosegmentation> (MIT, © 2023 Linus Kreitner; licence copy in `third_party/LICENSE-OCTA-autosegmentation`)
- L. Kreitner, J. C. Paetzold, N. Rauch, C. Chen, A. M. Hagag, A. E. Fayed, S. Sivaprasad, S. Rausch, J. Weichsel, B. H. Menze, M. Harders, B. Knier, D. Rueckert, M. J. Menten. *Synthetic optical coherence tomography angiographs for detailed retinal vessel segmentation without human annotations.* IEEE Transactions on Medical Imaging 43(6): 2061–2073, 2024. <https://doi.org/10.1109/TMI.2024.3354408>
- M. J. Menten, J. C. Paetzold, A. Dima, B. H. Menze, B. Knier, D. Rueckert. *Physiology-based simulation of the retinal vasculature enables annotation-free segmentation of OCT angiographs.* MICCAI 2022, LNCS 13438: 330–340. <https://doi.org/10.1007/978-3-031-16452-1_32>

The three images and labels in `examples/data/` are samples from the authors' synthetic dataset, redistributed under their MIT licence. They are simulated, not patient images.

## Software this package depends on

NumPy, SciPy, scikit-image, OpenCV, and (optionally) PyTorch and MONAI. Their licences are permissive; see each project.

## Methods implemented from the literature

Each is cited where it is used in the code and listed in [docs/references.md](docs/references.md): Frangi, Sato and Meijering ridge filters; Phansalkar local thresholding; Perona–Malik diffusion; non-local means; CLAHE; the Immerkaer noise estimate; the Crete blur metric; clDice; box-counting dimension; the Bennett axial-length scaling.
