from .classical import VesselSegmentation, segment_classical, vesselness
from .threshold import hysteresis, otsu, phansalkar


def OCTASegmenter(*args, **kwargs):
    """Learned segmenter (needs torch and monai). Imported lazily."""
    from .synthetic import OCTASegmenter as _S
    return _S(*args, **kwargs)
