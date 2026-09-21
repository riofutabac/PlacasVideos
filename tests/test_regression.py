import os
import json
import sqlite3
import pytest
from benchmarks.evaluate_pipeline import evaluate_run

def test_baseline_json_metrics():
    baseline_path = "benchmarks/baseline_yolov8n.json"
    assert os.path.exists(baseline_path), f"Baseline file {baseline_path} not found"
    with open(baseline_path, "r") as f:
        data = json.load(f)

    assert data["model"] == "yolov8n.onnx"
    metrics = data["metrics"]
    assert metrics["event_recall"] == 1.0
    assert metrics["direction_accuracy"] == 1.0
    assert metrics["plate_exact_match"] == 0.333
    assert metrics["false_positives"] == 0
    assert metrics["ground_truth_total"] == 8

def test_ground_truth_structure():
    gt_path = "benchmarks/ground_truth.json"
    assert os.path.exists(gt_path), f"Ground truth file {gt_path} not found"
    with open(gt_path, "r") as f:
        gt = json.load(f)

    assert len(gt) == 8
    for item in gt:
        assert "ground_truth_id" in item
        assert "video_source" in item
        assert "approx_timestamp" in item
        assert "direction" in item
        assert item["direction"] in ("ENTRADA", "SALIDA")

def test_evaluate_run_with_mock_db(tmp_path):
    db_file = str(tmp_path / "test_events.sqlite")
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE events (
            event_id TEXT PRIMARY KEY,
            processing_run_id TEXT,
            video_source TEXT,
            event_timestamp REAL,
            direction TEXT,
            plate_corrected TEXT,
            plate_normalized TEXT,
            duplicate_of TEXT
        )
    """)

    # Populate matching all 8 GT events
    gt_path = "benchmarks/ground_truth.json"
    with open(gt_path) as f:
        gt_data = json.load(f)

    run_id = "TEST_PERFECT_RUN"
    for item in gt_data:
        plate = item.get("plate_text")
        c.execute("""
            INSERT INTO events (event_id, processing_run_id, video_source, event_timestamp, direction, plate_corrected, plate_normalized, duplicate_of)
            VALUES (?, ?, ?, ?, ?, ?, ?, NULL)
        """, (
            f"EVT_{item['ground_truth_id']}",
            run_id,
            item["video_source"],
            item["approx_timestamp"],
            item["direction"],
            plate,
            plate
        ))
    conn.commit()
    conn.close()

    metrics = evaluate_run(db_path=db_file, run_id=run_id, gt_path=gt_path)
    assert metrics is not None
    assert metrics["event_recall"] == 1.0
    assert metrics["direction_accuracy"] == 1.0
    assert metrics["plate_exact_match"] == 1.0
    assert metrics["character_accuracy"] == 1.0
    assert metrics["unread_legible_plates"] == 0
    assert metrics["pos0_accuracy"] == 1.0
    assert metrics["total_real_gt_chars"] == 41
    assert metrics["total_correct_chars"] == 41
    assert metrics["false_positives"] == 0
    assert metrics["matched_gt_count"] == 8

def test_evaluate_run_detects_false_positives_and_mismatches(tmp_path):
    db_file = str(tmp_path / "test_fp.sqlite")
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE events (
            event_id TEXT PRIMARY KEY,
            processing_run_id TEXT,
            video_source TEXT,
            event_timestamp REAL,
            direction TEXT,
            plate_corrected TEXT,
            plate_normalized TEXT,
            duplicate_of TEXT
        )
    """)

    run_id = "TEST_FP_RUN"
    # Insert 1 true match (with inverted direction) + 1 false positive
    c.execute("""
        INSERT INTO events VALUES
        ('EVT_1', ?, 'Camara Placas 2_20260909105651-20260909163038(60).mp4', 25.0, 'SALIDA', 'PCW2492', 'PCW2492', NULL),
        ('EVT_FP', ?, 'Camara Placas 2_20260909105651-20260909163038(60).mp4', 500.0, 'ENTRADA', 'XYZ9999', 'XYZ9999', NULL)
    """, (run_id, run_id))
    conn.commit()
    conn.close()

    metrics = evaluate_run(db_path=db_file, run_id=run_id, gt_path="benchmarks/ground_truth.json")
    assert metrics is not None
    assert metrics["matched_gt_count"] == 1
    assert metrics["event_recall"] == 1 / 8
    # Direction was SALIDA instead of ENTRADA -> 0%
    assert metrics["direction_accuracy"] == 0.0
    # 2 detected - 1 matched = 1 false positive
    assert metrics["false_positives"] == 1
    # 5 other legible plates unread
    assert metrics["unread_legible_plates"] == 5
    assert metrics["total_real_gt_chars"] == 41

def test_evaluate_run_honest_unread_penalty(tmp_path):
    """Test that a matched vehicle with unread plate penalizes character accuracy over all 41 GT chars."""
    db_file = str(tmp_path / "test_unread.sqlite")
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE events (
            event_id TEXT PRIMARY KEY,
            processing_run_id TEXT,
            video_source TEXT,
            event_timestamp REAL,
            direction TEXT,
            plate_corrected TEXT,
            plate_normalized TEXT,
            duplicate_of TEXT
        )
    """)

    gt_path = "benchmarks/ground_truth.json"
    with open(gt_path) as f:
        gt_data = json.load(f)

    run_id = "TEST_UNREAD_RUN"
    for item in gt_data:
        # Simulate TAA2204 having no plate detected
        plate = item.get("plate_text")
        if item["ground_truth_id"] == "GT_60_3":
            plate = None

        c.execute("""
            INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?, NULL)
        """, (
            f"EVT_{item['ground_truth_id']}",
            run_id,
            item["video_source"],
            item["approx_timestamp"],
            item["direction"],
            plate,
            plate
        ))
    conn.commit()
    conn.close()

    metrics = evaluate_run(db_path=db_file, run_id=run_id, gt_path=gt_path)
    assert metrics["unread_legible_plates"] == 1
    assert metrics["plate_exact_match"] == 5 / 6
    assert metrics["total_real_gt_chars"] == 41
    # TAA2204 has 7 chars, so 41 - 7 = 34 correct
    assert metrics["total_correct_chars"] == 34
    assert round(metrics["character_accuracy"], 3) == round(34 / 41, 3)

