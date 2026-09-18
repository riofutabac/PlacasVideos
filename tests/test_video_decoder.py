import os
import numpy as np
import pytest

from src.video_decoder import (
    BaseVideoDecoder,
    OpenCVDecoder,
    FFmpegNVDECDecoder,
    NVDECDecoder,
    create_decoder,
    probe_video_metadata,
    check_nvdec_available,
    nv12_to_bgr_full_range
)
from src.decoder.base import _FULL_TO_BT601_Y_LUT

def test_decoder_imports_match():
    import src.video_decoder as vd
    import src.decoder as d
    assert vd.create_decoder is d.create_decoder
    assert vd.OpenCVDecoder is d.OpenCVDecoder
    assert vd.check_nvdec_available is d.check_nvdec_available
    assert vd.nv12_to_bgr_full_range is d.nv12_to_bgr_full_range

def test_check_nvdec_available_structure():
    info = check_nvdec_available()
    assert isinstance(info, dict)
    assert "cuda" in info
    assert "hevc_cuvid" in info
    assert "h264_cuvid" in info
    assert "pynv" in info
    for v in info.values():
        assert isinstance(v, bool)

def test_nv12_to_bgr_full_range_conversion():
    # Construct synthetic NV12 image: 40x40 resolution (height=40, width=40)
    # NV12 total height = 40 + 20 = 60
    w, h = 40, 40
    nv12 = np.zeros((h * 3 // 2, w), dtype=np.uint8)
    nv12[:h, :] = 180  # Luma Y
    nv12[h:, :] = 128  # Chroma UV neutral

    bgr = nv12_to_bgr_full_range(nv12, h=h, w=w)
    assert bgr.shape == (h, w, 3)
    assert bgr.dtype == np.uint8

def test_create_decoder_cpu_backend(tmp_path):
    # Create empty mock file
    mock_video = tmp_path / "mock.mp4"
    mock_video.write_bytes(b"0" * 100)

    # Backend 'opencv' should attempt OpenCVDecoder
    # Since mock.mp4 is invalid video bytes, OpenCVDecoder raises RuntimeError
    with pytest.raises(RuntimeError, match="OpenCV no pudo abrir el archivo de video"):
        create_decoder(str(mock_video), backend="opencv")

def test_lut_table_properties():
    assert len(_FULL_TO_BT601_Y_LUT) == 256
    assert _FULL_TO_BT601_Y_LUT[0] >= 0
    assert _FULL_TO_BT601_Y_LUT[255] <= 255
