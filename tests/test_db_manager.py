import os
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
