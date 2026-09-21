"""
Decoupled Video Decoder Module for ALPR Pipeline (Backward Compatibility Facade).
Preserves identical public API and imports while delegating implementation to src.decoder.
"""
from src.decoder import (
    BaseVideoDecoder,
    OpenCVDecoder,
    FFmpegNVDECDecoder,
    PyNvVideoCodecDecoder,
    NVDECDecoder,
    create_decoder,
    probe_video_metadata,
    check_mp4_has_moov_atom,
    check_nvdec_available,
    nv12_to_bgr_full_range,
    _read_exact
)

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
