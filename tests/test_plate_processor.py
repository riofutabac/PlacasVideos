import os
from unittest.mock import MagicMock
import numpy as np
import pytest

from src.pipeline_types import VehicleFrameCandidate
from src.quality_ranker import QualityRanker
from src.timer_profiler import PipelineProfiler
from src.plate_processor import PlateProcessor

@pytest.fixture
def processor(tmp_path):
    mock_alpr = MagicMock()
    ranker = QualityRanker()
    profiler = PipelineProfiler()
    veh_dir = str(tmp_path / "vehicles")
    plate_dir = str(tmp_path / "plates")
    return PlateProcessor(
        alpr=mock_alpr,
        ranker=ranker,
        profiler=profiler,
        top_k_crops=3,
        vehicle_evidence_dir=veh_dir,
        plate_evidence_dir=plate_dir
    )

def test_save_evidence(processor):
    veh_crop = np.zeros((100, 100, 3), dtype=np.uint8)
    plate_crop = np.zeros((30, 60, 3), dtype=np.uint8)

    veh_path, plate_path = processor.save_evidence("EVT_001", veh_crop, plate_crop)
    assert os.path.exists(veh_path)
    assert os.path.exists(plate_path)

    # Test when plate_crop is None
    veh_path2, plate_path2 = processor.save_evidence("EVT_002", veh_crop, None)
    assert os.path.exists(veh_path2)
    assert plate_path2 is None

def test_detect_plate_candidates(processor):
    # Empty frames list
    assert processor.detect_plate_candidates([]) == []

    # Mock detector output
    class DummyBBox:
        def __init__(self, x1, y1, x2, y2):
            self.x1 = x1
            self.y1 = y1
            self.x2 = x2
            self.y2 = y2

    class DummyDet:
        def __init__(self):
            self.bounding_box = DummyBBox(10, 10, 80, 40)
            self.confidence = 0.95

    processor.alpr.detector.predict.return_value = [DummyDet()]

    veh_crop = np.zeros((100, 150, 3), dtype=np.uint8)
    frame_cand = VehicleFrameCandidate(
        score=0.9,
        timestamp=12.5,
        vehicle_crop=veh_crop,
        bbox_in_full_frame=(100, 100, 250, 200)
    )

    candidates = processor.detect_plate_candidates([frame_cand])
    assert len(candidates) == 1
    assert candidates[0]["det_conf"] == 0.95
    assert candidates[0]["timestamp"] == 12.5
    # Padded crop dimensions (was 30x70, now 38x90)
    assert candidates[0]["crop"].shape == (38, 90, 3)

    # Test rejecting absurd vertical half-crop (aspect < 0.75)
    class VerticalSliverDet:
        def __init__(self):
            self.bounding_box = DummyBBox(10, 10, 30, 70)  # bw=20, bh=60 -> aspect=0.33
            self.confidence = 0.90

    processor.alpr.detector.predict.return_value = [VerticalSliverDet()]
    rejected = processor.detect_plate_candidates([frame_cand])
    assert len(rejected) == 0

def test_recognize_plate_candidates(processor):
    # Empty candidates
    res_empty = processor.recognize_plate_candidates([])
    assert res_empty[0] is None
    assert res_empty[1] is None

    # Mock OCR output
    class DummyOCRResult:
        def __init__(self, text, conf):
            self.text = text
            self.confidence = [conf]

    processor.alpr.ocr.predict.side_effect = [
        DummyOCRResult("PCW2492", 0.96),
        DummyOCRResult("PCW2497", 0.80)
    ]

    crop1 = np.zeros((30, 70, 3), dtype=np.uint8)
    crop2 = np.zeros((30, 70, 3), dtype=np.uint8)
    candidates = [
        {"crop": crop1, "det_conf": 0.92, "score": 0.85, "timestamp": 12.0},
        {"crop": crop2, "det_conf": 0.88, "score": 0.75, "timestamp": 12.5}
    ]

    plate_raw, plate_crop, plate_conf, ocr_conf, best_plate_ts, ocr_votes = processor.recognize_plate_candidates(candidates)
    assert plate_raw == "PCW2492"
    assert ocr_conf == 0.96
    assert best_plate_ts == 12.0
    assert len(ocr_votes) == 2
    assert ocr_votes[0][0] == "PCW2492"
