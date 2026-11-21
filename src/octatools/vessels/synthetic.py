"""Vessel segmentation with the network of Kreitner et al. (2024).

A 2-D DynUNet trained only on simulated OCTA images (after GAN-based contrast
adaptation) and run here on real ones. It was trained on 3 x 3 mm macular
scans resampled to 1216 x 1216, so larger fields are cut into 3 mm tiles.

Preprocessing follows the authors' test configuration: grayscale, min-max
scaling, bilinear resize to 1216, sigmoid, 0.5 threshold, removal of objects
below 128 pixels at the model resolution. The authors' rotate-and-flip step
cancels out for a square image, so it is not repeated here.

Original code: https://github.com/aiforvision/OCTA-autosegmentation (MIT).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from .._morph import remove_small
from ..preprocessing.geometry import pixel_size_um
from .classical import VesselSegmentation
from .weights import ensure_weights

MODEL_SIZE = 1216          # input side of the network
TRAIN_FOV_MM = 3.0         # field of view the network was trained at
MODEL_MIN_SIZE_PX = 128    # authors' small-object filter, in model pixels


def _import_torch():
    try:
        import torch
        from monai.networks.nets import DynUNet
    except ImportError as e:  # pragma: no cover
        raise ImportError("the learned segmenter needs torch and monai: pip install 'octatools[deep]'") from e
    return torch, DynUNet


def _tile_starts(n: int, tile: int, overlap: float) -> list[int]:
    if n <= tile:
        return [0]
    stride = max(1, int(round(tile * (1 - overlap))))
    k = int(np.ceil((n - tile) / stride)) + 1
    return [int(round(v)) for v in np.linspace(0, n - tile, k)]


def _window(n: int) -> np.ndarray:
    """1-D weight that fades towards the tile edge so overlaps blend smoothly."""
    w = np.hanning(n + 2)[1:-1]
    return np.maximum(w, 0.05).astype(np.float32)


class OCTASegmenter:
    def __init__(self, weights: str | None = None, device: str = "cpu"):
        torch, DynUNet = _import_torch()
        self.torch, self.device = torch, torch.device(device)
        self.net = DynUNet(spatial_dims=2, in_channels=1, out_channels=1, kernel_size=[3] * 5,
                           strides=[1, 2, 2, 2, 1], upsample_kernel_size=[1, 2, 2, 2, 1])
        ckpt = torch.load(ensure_weights(weights), map_location="cpu", weights_only=True)
        self.net.load_state_dict(ckpt["model"])
        self.net.to(self.device).eval()

    def _predict_tile(self, tile: np.ndarray, out_size: tuple[int, int] | None = None) -> np.ndarray:
        """Probability map for one square tile, returned at the tile's size (or `out_size`)."""
        torch = self.torch
        lo, hi = float(tile.min()), float(tile.max())
        x = (tile - lo) / (hi - lo) if hi > lo else np.zeros_like(tile)
        t = torch.from_numpy(x.astype(np.float32))[None, None].to(self.device)
        t = torch.nn.functional.interpolate(t, size=(MODEL_SIZE, MODEL_SIZE), mode="bilinear",
                                            align_corners=False)
        with torch.inference_mode():
            p = torch.sigmoid(self.net(t))
        p = torch.nn.functional.interpolate(p, size=out_size or tile.shape, mode="bilinear", align_corners=False)
        return p[0, 0].cpu().numpy()

    def predict_proba(self, img: np.ndarray, fov_mm: float = TRAIN_FOV_MM,
                      tile_overlap: float = 0.0) -> np.ndarray:
        """Vessel probability at the input resolution."""
        h, w = img.shape
        tile = int(round(w * TRAIN_FOV_MM / fov_mm)) if fov_mm > TRAIN_FOV_MM * 1.05 else None
        if tile is None or tile >= min(h, w):
            return self._predict_tile(img)
        acc = np.zeros((h, w), np.float32)
        wsum = np.zeros((h, w), np.float32)
        win = np.outer(_window(tile), _window(tile))
        for y in _tile_starts(h, tile, tile_overlap):
            for x in _tile_starts(w, tile, tile_overlap):
                acc[y:y + tile, x:x + tile] += self._predict_tile(img[y:y + tile, x:x + tile]) * win
                wsum[y:y + tile, x:x + tile] += win
        return acc / np.maximum(wsum, 1e-6)

    def segment(self, img: np.ndarray, fov_mm: float = TRAIN_FOV_MM, *, threshold: float = 0.5,
                tile_overlap: float = 0.0, smooth_sigma_um: float | None = None) -> VesselSegmentation:
        """Binary vessel mask plus probability.

        smooth_sigma_um: optional Gaussian pre-smoothing. Speckle-heavy images
        can be read as vessels by a network that has only seen clean simulated
        capillaries; a small blur (10-15 um) is worth trying on such data.
        """
        if smooth_sigma_um:
            img = ndi.gaussian_filter(img, smooth_sigma_um / pixel_size_um(img.shape, fov_mm))
        prob = self.predict_proba(img, fov_mm, tile_overlap)
        mask = prob > threshold
        # keep the authors' filter size, expressed as an area so it holds at any resolution
        eff_px = min(fov_mm, TRAIN_FOV_MM) * 1000.0 / MODEL_SIZE
        min_px = max(1, int(round(MODEL_MIN_SIZE_PX * eff_px ** 2 / pixel_size_um(img.shape, fov_mm) ** 2)))
        mask = remove_small(mask, min_px)
        return VesselSegmentation(mask, prob.astype(np.float32), "kreitner2024")
