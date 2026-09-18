import os
from unittest.mock import MagicMock
import numpy as np
import pytest

from src.pipeline_types import VehicleFrameCandidate
from src.quality_ranker import QualityRanker
from src.timer_profiler import PipelineProfiler
from src.plate_processor import PlateProcessor, vote_plate_characters, try_two_tier_ocr

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
    assert 0.85 < ocr_conf <= 0.96
    assert best_plate_ts == 12.0
    assert len(ocr_votes) == 2
    assert ocr_votes[0][0] == "PCW2492"

def test_vote_plate_characters_consensus():
    # Candidate 1 mistook '2' for '7' at pos 6
    c1 = {
        'text': "PCW2497",
        'ocr_conf': 0.82,
        'char_confs': [0.9, 0.9, 0.85, 0.9, 0.9, 0.9, 0.60],
        'det_conf': 0.90,
        'plate_score': 1.2,
        'crop': np.ones((20, 50, 3), dtype=np.uint8),
        'timestamp': 10.0
    }
    # Candidate 2 mistook 'W' for 'H' at pos 2
    c2 = {
        'text': "PCH2492",
        'ocr_conf': 0.85,
        'char_confs': [0.9, 0.9, 0.60, 0.9, 0.9, 0.9, 0.90],
        'det_conf': 0.92,
        'plate_score': 1.3,
        'crop': np.ones((20, 50, 3), dtype=np.uint8) * 2,
        'timestamp': 11.0
    }
    # Candidate 3 is clean
    c3 = {
        'text': "PCW2492",
        'ocr_conf': 0.88,
        'char_confs': [0.9, 0.9, 0.88, 0.9, 0.9, 0.9, 0.92],
        'det_conf': 0.95,
        'plate_score': 1.4,
        'crop': np.ones((20, 50, 3), dtype=np.uint8) * 3,
        'timestamp': 12.0
    }

    text, conf, crop, ts, det_conf, needs_review = vote_plate_characters([c1, c2, c3])
    assert text == "PCW2492"
    assert conf > 0.85
    assert det_conf == 0.95
    assert ts == 12.0
    assert needs_review is False

def test_vote_plate_characters_prior_ecuador():
    # 0CW2492 (invalid ANT prefix '0') vs PCW2492 (valid ANT Pichincha)
    c_inv = {
        'text': "0CW2492",
        'ocr_conf': 0.85,
        'char_confs': [0.85] * 7,
        'det_conf': 0.90,
        'plate_score': 1.0,
        'crop': np.zeros((10, 10, 3)),
        'timestamp': 1.0
    }
    c_val = {
        'text': "PCW2492",
        'ocr_conf': 0.85,
        'char_confs': [0.85] * 7,
        'det_conf': 0.90,
        'plate_score': 1.0,
        'crop': np.zeros((10, 10, 3)),
        'timestamp': 2.0
    }
    text, conf, crop, ts, det_conf, needs_review = vote_plate_characters([c_inv, c_val])
    assert text == "PCW2492"

def test_vote_plate_characters_soft_pichincha_prior():
    # Cand 1 has 'A' at pos 0 with higher conf (0.88), Cand 2 has 'P' at pos 0 with lower conf (0.82)
    # Both are valid Ecuador provinces (A = Azuay, P = Pichincha)
    c_a = {
        'text': "AAC2573",
        'ocr_conf': 0.88,
        'char_confs': [0.88] * 7,
        'det_conf': 0.90,
        'plate_score': 1.0,
        'crop': np.zeros((10, 10, 3)),
        'timestamp': 1.0
    }
    c_p = {
        'text': "PAC2573",
        'ocr_conf': 0.82,
        'char_confs': [0.82] * 7,
        'det_conf': 0.90,
        'plate_score': 1.0,
        'crop': np.zeros((10, 10, 3)),
        'timestamp': 2.0
    }
    # Without prior: 'A' wins because 0.88 > 0.82
    text_no_prior, _, _, _, _, _ = vote_plate_characters([c_a, c_p], province_prior_p=0.0)
    assert text_no_prior == "AAC2573"

    # With soft Pichincha prior beta=0.10: 'P' gets +0.10 boost on pos 0 (0.82 + 0.10 = 0.92 > 0.88) -> 'P' wins!
    text_with_prior, _, _, _, _, _ = vote_plate_characters([c_a, c_p], province_prior_p=0.10)
    assert text_with_prior == "PAC2573"

def test_vote_plate_characters_manual_review_threshold():
    c_a = {
        'text': "AAC2573",
        'ocr_conf': 0.85,
        'char_confs': [0.85] * 7,
        'det_conf': 0.90,
        'plate_score': 1.0,
        'crop': np.zeros((10, 10, 3)),
        'timestamp': 1.0
    }
    c_p = {
        'text': "PAC2573",
        'ocr_conf': 0.84,
        'char_confs': [0.84] * 7,
        'det_conf': 0.90,
        'plate_score': 1.0,
        'crop': np.zeros((10, 10, 3)),
        'timestamp': 2.0
    }
    # Difference between A (0.85) and P (0.84) is 0.01 < manual_review_threshold (0.15)
    _, _, _, _, _, needs_review = vote_plate_characters([c_a, c_p], manual_review_threshold=0.15)
    assert needs_review is True

def test_try_two_tier_ocr_aspect_filter():
    mock_alpr = MagicMock()
    # Wide car plate (80x40 -> aspect 2.0 > 1.35)
    wide_crop = np.zeros((40, 80, 3), dtype=np.uint8)
    assert try_two_tier_ocr(mock_alpr, wide_crop) is None
    assert mock_alpr.ocr.predict.call_count == 0

    # Tiny crop (h < 24)
    tiny_crop = np.zeros((20, 20, 3), dtype=np.uint8)
    assert try_two_tier_ocr(mock_alpr, tiny_crop) is None

def test_try_two_tier_ocr_motorcycle_split():
    mock_alpr = MagicMock()
    class DummyRes:
        def __init__(self, text, conf):
            self.text = text
            self.confidence = [conf] * len(text)

    mock_alpr.ocr.predict.side_effect = [
        DummyRes("JU", 0.95),      # top crop
        DummyRes("436A", 0.90)     # bottom crop
    ]

    # Square motorcycle plate (50x50 -> aspect 1.0)
    moto_crop = np.zeros((50, 50, 3), dtype=np.uint8)
    res = try_two_tier_ocr(mock_alpr, moto_crop)

    assert res is not None
    text, confs, avg_conf = res
    assert text == "JU436A"
    assert len(confs) == 6
    assert avg_conf > 0.90

def test_recognize_plate_candidates_with_motorcycle(processor):
    class DummyRes:
        def __init__(self, text, conf):
            self.text = text
            self.confidence = [conf] * len(text)

    # When full crop is run: it only sees "JU" (conf 0.70)
    # When top crop is run: sees "JU" (conf 0.92)
    # When bottom crop is run: sees "436A" (conf 0.88)
    processor.alpr.ocr.predict.side_effect = [
        DummyRes("JU", 0.70),    # Full crop
        DummyRes("JU", 0.92),    # Top crop (two-tier)
        DummyRes("436A", 0.88),  # Bottom crop (two-tier)
    ]

    moto_crop = np.zeros((50, 50, 3), dtype=np.uint8)
    candidates = [{
        "crop": moto_crop,
        "det_conf": 0.85,
        "score": 0.80,
        "timestamp": 25.0
    }]

    plate_raw, plate_crop, plate_conf, ocr_conf, best_plate_ts, ocr_votes = processor.recognize_plate_candidates(candidates)
    assert plate_raw == "JU436A"
    assert ocr_conf > 0.85
    assert best_plate_ts == 25.0

def test_recognize_plate_short_plate_rejected(processor):
    class DummyRes:
        def __init__(self, text, conf):
            self.text = text
            self.confidence = [conf] * len(text)

    # Candidate with short text (<5 chars, like "4W")
    processor.alpr.ocr.predict.side_effect = [
        DummyRes("4W", 0.95),
        None,  # two-tier returns None
    ]
    crop = np.zeros((30, 70, 3), dtype=np.uint8)
    candidates = [{"crop": crop, "det_conf": 0.90, "score": 0.85, "timestamp": 10.0}]

    plate_raw, plate_crop, plate_conf, ocr_conf, best_plate_ts, ocr_votes = processor.recognize_plate_candidates(candidates)
    assert plate_raw is None  # Cleaned up to "sin placa"
    assert ocr_conf == 0.0

def test_detect_plate_candidates_save_manifest(tmp_path):
    mock_alpr = MagicMock()
    class DummyBBox:
        def __init__(self, x1, y1, x2, y2):
            self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2

    class DummyDet:
        def __init__(self):
            self.bounding_box = DummyBBox(10, 10, 80, 40)
            self.confidence = 0.95

    mock_alpr.detector.predict.return_value = [DummyDet()]
    ranker = QualityRanker()
    profiler = PipelineProfiler()
    veh_dir = str(tmp_path / "vehicles")
    plate_dir = str(tmp_path / "plates")

    proc = PlateProcessor(
        alpr=mock_alpr,
        ranker=ranker,
        profiler=profiler,
        top_k_crops=3,
        vehicle_evidence_dir=veh_dir,
        plate_evidence_dir=plate_dir,
        save_manifest=True
    )

    frame_cand = VehicleFrameCandidate(
        score=0.9,
        timestamp=12.5,
        vehicle_crop=np.zeros((100, 150, 3), dtype=np.uint8),
        bbox_in_full_frame=(100, 100, 250, 200)
    )

    cands = proc.detect_plate_candidates([frame_cand], event_id="EVT_TEST_001")
    assert len(cands) == 1
    assert len(proc.manifest_records) == 1
    assert proc.manifest_records[0]["event_id"] == "EVT_TEST_001"
    assert os.path.exists(proc.manifest_records[0]["plate_crop_path"])
    assert os.path.exists(proc.manifest_records[0]["vehicle_crop_path"])

    manifest_file = proc.write_manifest()
    assert os.path.exists(manifest_file)
