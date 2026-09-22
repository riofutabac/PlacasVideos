import os
import sqlite3
import pytest
from src.db_manager import DatabaseManager

@pytest.fixture
def db(tmp_path):
    db_file = str(tmp_path / "test_events.sqlite")
    return DatabaseManager(db_path=db_file)

def test_db_init_and_run_lifecycle(db):
    run_id = "RUN_TEST_123"
    db.start_run(
        run_id=run_id,
        pipeline_version="1.9.0",
        config_hash="abc123hash",
        model_versions={"detector": "yolov8n.onnx"},
        started_at="2026-09-18 10:00:00"
    )

    # Check clip completed false initially
    assert not db.is_clip_completed("clip_01", "hash1")

    # Record clip
    db.record_clip(
        clip_id="clip_01",
        run_id=run_id,
        file_path="/videos/test.mp4",
        file_hash="hash1",
        duration_sec=300.0,
        total_frames=7500,
        events_count=2,
        status="COMPLETED",
        started_at="2026-09-18 10:00:01",
        completed_at="2026-09-18 10:01:00",
        wall_clock_seconds=59.0,
        speed_ratio=5.08
    )
    assert db.is_clip_completed("clip_01", "hash1")

    # Finish run
    db.finish_run(
        run_id=run_id,
        finished_at="2026-09-18 10:01:05",
        total_videos=1,
        total_events=2,
        speed_ratio=5.08,
        status="COMPLETED"
    )

def test_db_insert_and_query_events(db):
    run_id = "RUN_EVENTS"
    event_1 = {
        'event_id': 'EVT_001',
        'processing_run_id': run_id,
        'video_source': 'clip_01.mp4',
        'clip_hash': 'hash1',
        'event_timestamp': 15.5,
        'datetime_str': '2026-09-18 10:00:15',
        'best_vehicle_timestamp': 15.2,
        'best_plate_timestamp': 15.4,
        'direction': 'ENTRADA',
        'vehicle_type': 'car',
        'plate_raw': 'PCW2492',
        'plate_normalized': 'PCW2492',
        'plate_corrected': 'PCW2492',
        'plate_correction_reason': 'exact_match',
        'plate_status': 'LEGIBLE',
        'confidence_vehicle': 0.92,
        'confidence_plate': 0.88,
        'confidence_ocr': 0.95,
        'confidence_consensus': 0.95,
        'vehicle_crop_path': 'data/crops/car.jpg',
        'plate_crop_path': 'data/crops/plate.jpg',
        'ocr_votes': ['PCW2492', 'PCW2492'],
        'track_id': 1,
        'line_id': 'L1',
        'dedup_key': 'PCW2492_ENTRADA',
        'duplicate_of': None
    }
    event_dup = dict(event_1)
    event_dup['event_id'] = 'EVT_002'
    event_dup['duplicate_of'] = 'EVT_001'

    db.insert_event(event_1)
    db.insert_event(event_dup)

    # Exclude duplicates
    events_unique = db.get_events_for_run(run_id=run_id, exclude_duplicates=True)
    assert len(events_unique) == 1
    assert events_unique[0]['event_id'] == 'EVT_001'

    # Include duplicates
    events_all = db.get_events_for_run(run_id=run_id, exclude_duplicates=False)
    assert len(events_all) == 2


def test_new_timing_columns_round_trip(db):
    run_id = "RUN_TIMING"
    db.start_run(
        run_id=run_id, pipeline_version="1.9.0", config_hash="abc",
        model_versions={"detector": "yolov8n.onnx"}, started_at="2026-09-22 10:00:00"
    )
    db.record_clip(
        clip_id="clipA.mp4", run_id=run_id, file_path="/videos/a.mp4", file_hash="hashA",
        duration_sec=330.0, total_frames=9900, events_count=3, status="COMPLETED",
        started_at="2026-09-22 10:00:01", completed_at="2026-09-22 10:01:00",
        wall_clock_seconds=59.0, speed_ratio=5.08,
        staging_seconds=4.2, file_hash_seconds=1.1, clock_read_seconds=0.3
    )
    db.record_clip(
        clip_id="clipB.mp4", run_id=run_id, file_path="/videos/b.mp4", file_hash="hashB",
        duration_sec=330.0, total_frames=9900, events_count=1, status="COMPLETED",
        started_at="2026-09-22 10:01:01", completed_at="2026-09-22 10:02:00",
        wall_clock_seconds=61.0, speed_ratio=4.9,
        staging_seconds=3.8, file_hash_seconds=0.9, clock_read_seconds=0.2
    )
    db.finish_run(
        run_id=run_id, finished_at="2026-09-22 10:02:05", total_videos=2, total_events=4,
        speed_ratio=5.0, status="COMPLETED", startup_seconds=12.5, export_seconds=7.3
    )

    totals = db.get_clip_timing_totals(run_id)
    assert totals["staging_seconds"] == pytest.approx(8.0)
    assert totals["file_hash_seconds"] == pytest.approx(2.0)
    assert totals["clock_read_seconds"] == pytest.approx(0.5)
    assert totals["wall_clock_seconds"] == pytest.approx(120.0)

    with sqlite3.connect(db.db_path) as conn:
        row = conn.execute(
            "SELECT startup_seconds, export_seconds FROM processing_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
    assert row == (12.5, 7.3)


def test_old_database_without_new_columns_still_opens(tmp_path):
    """Simulates a DB created before this instrumentation change and verifies
    DatabaseManager migrates it additively without breaking existing reads."""
    db_file = str(tmp_path / "legacy.sqlite")
    with sqlite3.connect(db_file) as conn:
        conn.execute("""
        CREATE TABLE processing_runs (
            run_id TEXT PRIMARY KEY,
            pipeline_version TEXT NOT NULL,
            config_hash TEXT NOT NULL,
            model_versions TEXT NOT NULL,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            total_videos INTEGER DEFAULT 0,
            total_events INTEGER DEFAULT 0,
            speed_ratio REAL DEFAULT 0.0,
            status TEXT NOT NULL
        );
        """)
        conn.execute("""
        CREATE TABLE processed_clips (
            clip_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_hash TEXT NOT NULL,
            duration_sec REAL NOT NULL,
            total_frames INTEGER NOT NULL,
            processed_events INTEGER DEFAULT 0,
            status TEXT NOT NULL,
            completed_at TEXT
        );
        """)
        conn.execute(
            "INSERT INTO processing_runs (run_id, pipeline_version, config_hash, model_versions, started_at, status) "
            "VALUES ('RUN_OLD', '1.0.0', 'oldhash', '{}', '2025-01-01 00:00:00', 'COMPLETED')"
        )
        conn.execute(
            "INSERT INTO processed_clips (clip_id, run_id, file_path, file_hash, duration_sec, total_frames, "
            "processed_events, status, completed_at) VALUES "
            "('clipOld.mp4', 'RUN_OLD', '/x.mp4', 'h', 300.0, 9000, 1, 'COMPLETED', '2025-01-01 00:05:00')"
        )
        conn.commit()

    # Opening with DatabaseManager should migrate additively, not raise
    db = DatabaseManager(db_path=db_file)
    totals = db.get_clip_timing_totals("RUN_OLD")
    assert totals["staging_seconds"] == 0.0
    assert totals["wall_clock_seconds"] == 0.0

    # New writes with new columns should work fine post-migration
    db.record_clip(
        clip_id="clipNew.mp4", run_id="RUN_OLD", file_path="/y.mp4", file_hash="h2",
        duration_sec=300.0, total_frames=9000, events_count=0, status="COMPLETED",
        started_at="2026-09-22 00:00:00", completed_at="2026-09-22 00:01:00",
        staging_seconds=2.0
    )
    assert db.get_clip_timing_totals("RUN_OLD")["staging_seconds"] == pytest.approx(2.0)
