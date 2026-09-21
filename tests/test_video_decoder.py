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

def test_check_mp4_has_moov_atom(tmp_path):
    from src.video_decoder import check_mp4_has_moov_atom
    # 1. Truncated or missing moov
    corrupt_mp4 = tmp_path / "corrupt.mp4"
    corrupt_mp4.write_bytes((16).to_bytes(4, 'big') + b'ftyp' + b'isom\x00\x00\x02\x00' + (100).to_bytes(4, 'big') + b'mdat' + b'1234')
    assert check_mp4_has_moov_atom(str(corrupt_mp4)) is False

    # 2. Valid moov atom present
    valid_mp4 = tmp_path / "valid.mp4"
    valid_mp4.write_bytes(
        (16).to_bytes(4, 'big') + b'ftyp' + b'isom\x00\x00\x02\x00' +
        (16).to_bytes(4, 'big') + b'mdat' + b'12345678' +
        (16).to_bytes(4, 'big') + b'moov' + b'head' + b'data'
    )
    assert check_mp4_has_moov_atom(str(valid_mp4)) is True

    # 3. Non-mp4 file (e.g. .avi) passes check
    avi_file = tmp_path / "video.avi"
    avi_file.write_bytes(b"RIFF" + b"0" * 50)
    assert check_mp4_has_moov_atom(str(avi_file)) is True

def test_create_decoder_cpu_backend(tmp_path):
    # Missing moov atom fails fast with ValueError
    corrupt_mp4 = tmp_path / "corrupt.mp4"
    corrupt_mp4.write_bytes(b"0" * 100)
    with pytest.raises(ValueError, match="Corrupted MP4 container"):
        create_decoder(str(corrupt_mp4), backend="opencv")

    # Non-mp4 invalid file passes container check but fails inside OpenCVDecoder
    mock_avi = tmp_path / "mock.avi"
    mock_avi.write_bytes(b"0" * 100)
    with pytest.raises(RuntimeError, match="OpenCV no pudo abrir el archivo de video"):
        create_decoder(str(mock_avi), backend="opencv")

def test_lut_table_properties():
    assert len(_FULL_TO_BT601_Y_LUT) == 256
    assert _FULL_TO_BT601_Y_LUT[0] >= 0
    assert _FULL_TO_BT601_Y_LUT[255] <= 255

def test_ffmpeg_nvdec_build_cmd(monkeypatch):
    # Mock metadata probe
    monkeypatch.setattr("src.decoder.nvdec_decoder.probe_video_metadata", lambda p: {
        "codec": "hevc", "width": 2960, "height": 1664, "fps": 25.0, "total_frames": 100
    })
    crop_rect = {"x_min": 300, "y_min": 600, "x_max": 2600, "y_max": 1664}
    dec = FFmpegNVDECDecoder("fake.mp4", crop_rect=crop_rect)

    # 1. Cuvid with hardware -crop
    cmd_cuvid_crop = dec._build_cmd(use_cuvid=True, use_hw_crop=True)
    assert "-c:v" in cmd_cuvid_crop
    assert "hevc_cuvid" in cmd_cuvid_crop
    assert "-crop" in cmd_cuvid_crop
    crop_idx = cmd_cuvid_crop.index("-crop")
    assert cmd_cuvid_crop[crop_idx + 1] == "600x0x300x360"

    # 2. CUDA hwaccel + hwdownload + crop
    cmd_cuda = dec._build_cmd(use_cuvid=False, use_hw_crop=False)
    assert "-hwaccel" in cmd_cuda
    assert "cuda" in cmd_cuda
    assert "crop=2300:1064:300:600" in "".join(cmd_cuda)

    # 3. Cuvid fallback with software crop filter
    cmd_cuvid_vf = dec._build_cmd(use_cuvid=True, use_hw_crop=False)
    assert "-c:v" in cmd_cuvid_vf
    assert "-crop" not in cmd_cuvid_vf
    assert "crop=2300:1064:300:600" in "".join(cmd_cuvid_vf)

