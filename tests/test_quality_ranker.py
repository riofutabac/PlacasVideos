import pytest
import numpy as np
from src.quality_ranker import (
    QualityRanker,
    calculate_sharpness,
    calculate_overexposure_penalty
)

def test_sharpness_calculation():
    # Completely flat image -> 0 sharpness
    flat = np.ones((50, 50, 3), dtype=np.uint8) * 128
    assert calculate_sharpness(flat) == 0.0

    # High frequency checkerboard pattern -> high sharpness
    checker = np.zeros((50, 50, 3), dtype=np.uint8)
    checker[::2, ::2] = 255
    checker[1::2, 1::2] = 255
    sharpness = calculate_sharpness(checker)
    assert sharpness > 100.0

    # None or empty
    assert calculate_sharpness(None) == 0.0
    assert calculate_sharpness(np.array([])) == 0.0

def test_overexposure_penalty():
    # Pure white burned image -> max penalty (1.0)
    white = np.ones((50, 50, 3), dtype=np.uint8) * 255
    assert calculate_overexposure_penalty(white) == 1.0

    # Normal medium gray image -> 0 penalty
    gray = np.ones((50, 50, 3), dtype=np.uint8) * 120
    assert calculate_overexposure_penalty(gray) == 0.0

    # None
    assert calculate_overexposure_penalty(None) == 1.0

def test_score_vehicle_frame():
    ranker = QualityRanker()
    frame_shape = (1080, 1920)

    # Sharp crop in the center of the frame
    crop_good = np.zeros((200, 300, 3), dtype=np.uint8)
    crop_good[::2, ::2] = 200
    bbox_good = (500, 400, 800, 600)

    score_good = ranker.score_vehicle_frame(
        vehicle_crop=crop_good,
        bbox=bbox_good,
        frame_shape=frame_shape,
        detector_confidence=0.95
    )
    assert score_good > 0.4

    # Blurrier crop
    crop_blur = np.ones((200, 300, 3), dtype=np.uint8) * 100
    score_blur = ranker.score_vehicle_frame(
        vehicle_crop=crop_blur,
        bbox=bbox_good,
        frame_shape=frame_shape,
        detector_confidence=0.95
    )
    assert score_good > score_blur

    # Crop touching the border (x1=5 < 30) -> border penalty applied
    bbox_border = (5, 400, 305, 600)
    score_border = ranker.score_vehicle_frame(
        vehicle_crop=crop_good,
        bbox=bbox_border,
        frame_shape=frame_shape,
        detector_confidence=0.95
    )
    assert score_good > score_border

    # None crop returns 0
    assert ranker.score_vehicle_frame(None, bbox_good, frame_shape, 0.9) == 0.0

def test_score_plate_crop():
    ranker = QualityRanker()

    # High quality plate: aspect ratio ~ 2.0 (standard 100x50), good sharpness
    plate_good = np.zeros((50, 100, 3), dtype=np.uint8)
    plate_good[::2, :] = 180
    score_p_good = ranker.score_plate_crop(
        plate_crop=plate_good,
        plate_conf=0.90
    )
    assert score_p_good > 0.4

    # Overexposed plate
    plate_burned = np.ones((50, 100, 3), dtype=np.uint8) * 255
    score_burned = ranker.score_plate_crop(
        plate_crop=plate_burned,
        plate_conf=0.90
    )
    assert score_p_good > score_burned

def test_top_m_ordering():
    class DummyCandidate:
        def __init__(self, score, name):
            self.score = score
            self.name = name

    candidates = [
        DummyCandidate(0.35, "C1"),
        DummyCandidate(0.92, "C2"),
        DummyCandidate(0.68, "C3"),
        DummyCandidate(0.81, "C4")
    ]
    candidates.sort(key=lambda x: x.score, reverse=True)
    assert [c.name for c in candidates] == ["C2", "C4", "C3", "C1"]
    assert candidates[0].score == 0.92
    assert candidates[-1].score == 0.35
