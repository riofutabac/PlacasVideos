import os
import json
import sqlite3
import numpy as np
import pytest
from datetime import time, datetime

from benchmarks.build_gt_candidates import parse_time_cell, match_with_database
from benchmarks.benchmark_ocr_offline import (
    apply_letterbox_resize,
    extract_plate_with_margin,
    logprob_fusion,
    run_ocr_matrix_benchmark
)
from benchmarks.benchmark_plate_detector import load_vehicle_crops, benchmark_detectors


def test_parse_time_cell():
    t = time(14, 30, 15)
    assert parse_time_cell(t) == t

    dt = datetime(2026, 9, 18, 10, 15, 0)
    assert parse_time_cell(dt) == time(10, 15, 0)

    assert parse_time_cell("12:45:30") == time(12, 45, 30)
    assert parse_time_cell("08:20") == time(8, 20, 0)
    assert parse_time_cell("invalid-time") is None
    assert parse_time_cell(None) is None


def test_match_with_database(tmp_path):
    db_path = str(tmp_path / "test_events.sqlite")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE events (
            event_id TEXT PRIMARY KEY,
            video_source TEXT,
            event_timestamp REAL,
            plate_corrected TEXT,
            plate_normalized TEXT,
            vehicle_crop_path TEXT,
            plate_crop_path TEXT,
            confidence_ocr REAL,
            duplicate_of TEXT
        )
    """)
    cur.execute("""
        INSERT INTO events VALUES (
            'EVT_1', 'clip60.mp4', 25.4, 'PCW2492', 'PCW2492', '/crop/v1.jpg', '/crop/p1.jpg', 0.95, NULL
        )
    """)
    conn.commit()
    conn.close()

    audit_entries = [
        {
            "row_idx": 2,
            "plate_text": "PCW2492",
            "vehicle_type": "car",
            "time_cam1": "10:57:15",
            "time_cam2": None,
            "note": "ok",
            "province": "Pichincha"
        },
        {
            "row_idx": 3,
            "plate_text": "TAA2204",
            "vehicle_type": "truck",
            "time_cam1": None,
            "time_cam2": None,
            "note": "",
            "province": "Tungurahua"
        }
    ]

    candidates = match_with_database(audit_entries, db_path=db_path)
    assert len(candidates) == 2
    c1 = candidates[0]
    assert c1["plate_text"] == "PCW2492"
    assert c1["matched_event_id"] == "EVT_1"
    assert c1["video_source"] == "clip60.mp4"

    c2 = candidates[1]
    assert c2["plate_text"] == "TAA2204"
    assert c2["matched_event_id"] is None


def test_apply_letterbox_resize():
    # Empty image
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    res_empty = apply_letterbox_resize(empty, 128, 64)
    assert res_empty.shape == (64, 128, 3)

    # Regular image
    img = np.zeros((50, 100, 3), dtype=np.uint8)
    res = apply_letterbox_resize(img, 128, 64)
    assert res.shape == (64, 128, 3)


def test_extract_plate_with_margin():
    veh_crop = np.zeros((200, 200, 3), dtype=np.uint8)
    bbox = [50, 50, 100, 100]

    # Margin 0.10: pad_x = 5, pad_y = 5 -> [45:105, 45:105]
    crop = extract_plate_with_margin(veh_crop, bbox, 0.10)
    assert crop.shape == (60, 60, 3)

    # Empty crop fallback
    crop_empty = extract_plate_with_margin(None, bbox, 0.10)
    assert crop_empty.shape == (30, 70, 3)


def test_logprob_fusion():
    assert logprob_fusion([]) == (None, 0.0)

    scored = [
        {"text": "PCW2492", "ocr_conf": 0.90, "plate_score": 1.0},
        {"text": "PCW2492", "ocr_conf": 0.85, "plate_score": 0.9},
        {"text": "POW2492", "ocr_conf": 0.50, "plate_score": 0.5},
    ]
    best_text, conf = logprob_fusion(scored, province_prior_p=0.10)
    assert best_text == "PCW2492"
    assert conf > 0.0


def test_load_real_event_crops():
    from benchmarks.benchmark_ocr_offline import load_real_event_crops
    crops = load_real_event_crops("evidence/plates/debug")
    assert len(crops) > 0
    for evt_id, paths in crops.items():
        assert evt_id.startswith("EVT_")
        assert len(paths) > 0

def test_map_events_to_ground_truth():
    from benchmarks.benchmark_ocr_offline import map_events_to_ground_truth
    evt_ids = ["EVT_(60)_0023_27", "EVT_(61)_0018_80"]
    mapping = map_events_to_ground_truth(evt_ids, "benchmarks/ground_truth.json")
    assert "EVT_(60)_0023_27" in mapping
    assert mapping["EVT_(60)_0023_27"]["plate_text"] == "PCW2492"

def test_run_ocr_matrix_benchmark_quick(tmp_path, monkeypatch):
    out_csv = str(tmp_path / "results.csv")
    results = run_ocr_matrix_benchmark(
        debug_dir="evidence/plates/debug",
        gt_path="benchmarks/ground_truth.json",
        output_csv=out_csv,
        quick_mode=True
    )
    assert len(results) > 0
    assert os.path.exists(out_csv)


def test_plate_detector_benchmark(tmp_path, monkeypatch):
    # Mock LicensePlateDetector to keep test fast and isolated
    import open_image_models
    class MockDet:
        def __init__(self, *args, **kwargs): pass
        def predict(self, img):
            class D: confidence = 0.95
            return [D()]
    monkeypatch.setattr(open_image_models, "LicensePlateDetector", MockDet)

    manifest = tmp_path / "empty_manifest.json"
    crops = load_vehicle_crops(str(manifest))
    assert len(crops) > 0

    results = benchmark_detectors(str(manifest))
    assert len(results) == 2
    assert results[0]["recall_pct"] >= 0.0


def test_pipeline_microbenchmarks(monkeypatch):
    from benchmarks.benchmark_pipeline_microbench import run_microbenchmarks
    # Test run_microbenchmarks runs without exceptions
    run_microbenchmarks()

def test_build_gt_candidates_matching_logic(tmp_path):
    import sqlite3
    from benchmarks.build_gt_candidates import match_with_database
    from src.db_manager import DatabaseManager

    db_file = str(tmp_path / "test_events.sqlite")
    db = DatabaseManager(db_file)

    # Insert test events
    with db._get_connection() as conn:
        conn.execute("""
            INSERT INTO events (
                event_id, processing_run_id, video_source, clip_hash, event_timestamp,
                datetime_str, direction, vehicle_type, plate_raw, plate_normalized,
                plate_corrected, plate_status, confidence_vehicle, confidence_ocr,
                vehicle_crop_path, track_id, line_id
            ) VALUES
            ('EVT_1', 'RUN_1', 'clip1.mp4', 'h1', 10.0, '2026-09-09 11:07:05', 'ENTRADA', 'truck', 'PBO4275', 'PBO4275', 'PBO4275', 'VALID', 0.95, 0.99, 'crop1.jpg', 1, 'L1'),
            ('EVT_2', 'RUN_1', 'clip2.mp4', 'h2', 20.0, '2026-09-09 11:28:10', 'ENTRADA', 'truck', 'PAC7438', 'PAC7438', 'PAC7438', 'VALID', 0.90, 0.85, 'crop2.jpg', 2, 'L1'),
            ('EVT_3', 'RUN_1', 'clip3.mp4', 'h3', 30.0, '2026-09-09 16:00:00', 'ENTRADA', 'car', 'FAR1234', 'FAR1234', 'FAR1234', 'VALID', 0.90, 0.90, 'crop3.jpg', 3, 'L1')
        """)
        conn.commit()

    audit_entries = [
        # Exact match
        {"plate_text": "PBO4275", "vehicle_type": "truck", "province": "Pichincha", "note": "", "time_cam1": None, "time_cam2": "11:07:00"},
        # Time window match (PAC7438 detected at 11:28:10 vs PAC7437 audited at 11:28:00)
        {"plate_text": "PAC7437", "vehicle_type": "truck", "province": "Pichincha", "note": "", "time_cam1": None, "time_cam2": "11:28:00"},
        # Exact plate fallback (FAR1234 audited at 12:00:00, but detected at 16:00:00 - outside 3min tolerance)
        {"plate_text": "FAR1234", "vehicle_type": "car", "province": "Pichincha", "note": "", "time_cam1": None, "time_cam2": "12:00:00"},
        # No match
        {"plate_text": "NOT9999", "vehicle_type": "car", "province": "Pichincha", "note": "", "time_cam1": None, "time_cam2": "10:00:00"}
    ]

    candidates = match_with_database(audit_entries, db_path=db_file, tolerance_minutes=3.0)
    assert len(candidates) == 4

    # 1. Exact match
    c0 = candidates[0]
    assert c0["match_type"] == "EXACT_PLATE_AND_TIME"
    assert c0["matched_event_id"] == "EVT_1"
    assert c0["time_diff_sec"] == 5.0
    assert c0["pipeline_detected_plate"] == "PBO4275"

    # 2. Time window match
    c1 = candidates[1]
    assert c1["match_type"] == "TIME_WINDOW"
    assert c1["matched_event_id"] == "EVT_2"
    assert c1["time_diff_sec"] == 10.0
    assert c1["pipeline_detected_plate"] == "PAC7438"

    # 3. Exact plate fallback
    c2 = candidates[2]
    assert c2["match_type"] == "EXACT_PLATE_FALLBACK"
    assert c2["matched_event_id"] == "EVT_3"
    assert c2["pipeline_detected_plate"] == "FAR1234"

    # 4. No match
    c3 = candidates[3]
    assert c3["match_type"] is None
    assert c3["matched_event_id"] is None


