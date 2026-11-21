"""Download the pretrained OCTA vessel segmentation weights.

The network and its weights come from Kreitner et al. (2024). They are not
copied into this repository; they are fetched from the authors' repository at
a fixed commit and checked against a SHA-256 hash.
"""
from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

UPSTREAM = "https://github.com/aiforvision/OCTA-autosegmentation"
COMMIT = "9cdc3137b6f55ae766dcae76c166ccf9774daf2b"
WEIGHTS_SHA256 = "e0c6e055f1740bcd40e7b9e7357619c1837f7715447203b6fb1bee7467828aec"
WEIGHTS_URL = f"{UPSTREAM}/raw/{COMMIT}/docker/trained_models/ves_seg-S-GAN/checkpoints/30_model.pth"


def cache_dir() -> Path:
    return Path(os.environ.get("OCTATOOLS_CACHE", Path.home() / ".cache" / "octatools"))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_weights(path: str | Path | None = None) -> Path:
    """Return the local path of the checkpoint, downloading it first if needed."""
    dest = Path(path) if path else cache_dir() / "ves_seg-S-GAN_epoch30.pth"
    if dest.exists() and _sha256(dest) == WEIGHTS_SHA256:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    urllib.request.urlretrieve(WEIGHTS_URL, tmp)
    if _sha256(tmp) != WEIGHTS_SHA256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("downloaded weights do not match the expected SHA-256")
    tmp.replace(dest)
    return dest
