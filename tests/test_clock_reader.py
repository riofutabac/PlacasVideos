"""
Unit tests for OSD Clock Reader and Video Timestamp Extraction.
"""
import os
import numpy as np
from datetime import datetime
from src.clock_reader import (
    get_osd_templates,
    read_frame_timestamp,
    read_clip_start_datetime,
)
from src.pipeline_types import get_clip_start_datetime

def test_get_osd_templates():
    templates = get_osd_templates()
    assert isinstance(templates, dict)
    assert len(templates) == 10
    for d in range(10):
        assert d in templates
        tpl = templates[d]
        assert isinstance(tpl, np.ndarray)
        assert len(tpl.shape) == 2
        assert tpl.shape[0] > 0 and tpl.shape[1] > 0

def test_read_frame_timestamp_invalid_inputs():
    assert read_frame_timestamp(None) is None
    assert read_frame_timestamp(np.zeros((50, 50), dtype=np.uint8)) is None
    # Empty black frame of 2960x1664 has no text -> returns None
    black_frame = np.zeros((1664, 2960, 3), dtype=np.uint8)
    assert read_frame_timestamp(black_frame) is None
    # Pure noise -> returns None
    np.random.seed(42)
    noise_frame = np.random.randint(0, 256, (1664, 2960, 3), dtype=np.uint8)
    assert read_frame_timestamp(noise_frame) is None

def test_read_clip_start_datetime_nonexistent():
    assert read_clip_start_datetime("non_existent_file.mp4") is None
    assert read_clip_start_datetime("") is None

def test_get_clip_start_datetime_fallback():
    # When file does not exist, it falls back to filename-based extrapolation
    dt60 = get_clip_start_datetime("Camara Placas 2_20260909105651-20260909163038(60).mp4")
    assert isinstance(dt60, datetime)
    assert dt60.year == 2026
    assert dt60.month == 9
    assert dt60.day == 9
    assert dt60.hour == 16
    assert dt60.minute == 17
    assert dt60.second == 16

def test_read_real_clip_if_available():
    clip60 = "Camara Placas 2_20260909105651-20260909163038(60).mp4"
    if os.path.exists(clip60):
        dt = read_clip_start_datetime(clip60)
        assert dt is not None
        assert dt == datetime(2026, 9, 9, 16, 17, 16)
