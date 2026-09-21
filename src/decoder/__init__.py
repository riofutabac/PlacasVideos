"""
Decoder subpackage for ALPR Pipeline.
Provides unified access to CPU and GPU video decoders.
"""
from .base import (
    BaseVideoDecoder,
    probe_video_metadata,
    check_mp4_has_moov_atom,
    nv12_to_bgr_full_range,
    _read_exact
)
from .opencv_decoder import OpenCVDecoder
from .nvdec_decoder import (
    check_nvdec_available,
    FFmpegNVDECDecoder,
    PyNvVideoCodecDecoder,
    NVDECDecoder
)
from .factory import create_decoder

__all__ = [
    "BaseVideoDecoder",
    "OpenCVDecoder",
    "FFmpegNVDECDecoder",
    "PyNvVideoCodecDecoder",
    "NVDECDecoder",
    "create_decoder",
    "probe_video_metadata",
    "check_mp4_has_moov_atom",
    "check_nvdec_available",
    "nv12_to_bgr_full_range",
]
