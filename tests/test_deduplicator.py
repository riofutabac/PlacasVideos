import pytest
import numpy as np
from src.deduplicator import (
    EventDeduplicator,
    levenshtein_distance,
    calculate_visual_similarity
)

def test_levenshtein_distance():
    assert levenshtein_distance("PCW2492", "PCW2492") == 0
    assert levenshtein_distance("PCW2492", "PCW2497") == 1
    assert levenshtein_distance("PBA1234", "PBA123") == 1
    assert levenshtein_distance("ABC", "DEF") == 3
    assert levenshtein_distance("", "TEST") == 4
    assert levenshtein_distance("TEST", "") == 4

def test_visual_similarity():
    # Identical images
    img1 = np.ones((50, 50, 3), dtype=np.uint8) * 128
    img2 = np.ones((50, 50, 3), dtype=np.uint8) * 128
    sim = calculate_visual_similarity(img1, img2)
    assert abs(sim - 1.0) < 1e-3

    # Empty or None images
    assert calculate_visual_similarity(None, img1) == 0.0
    assert calculate_visual_similarity(np.array([]), img1) == 0.0

def test_intra_clip_deduplication():
    dedup = EventDeduplicator(cooldown_seconds=25.0)

    # First emission for track 10 on LINE_A
    assert dedup.check_intra_clip(10, "LINE_A") is False

    # Second emission for track 10 on LINE_A -> must be blocked
    assert dedup.check_intra_clip(10, "LINE_A") is True

    # Track 10 on a different line -> not blocked
    assert dedup.check_intra_clip(10, "LINE_B") is False

    # New clip starts
    dedup.reset_clip_state()
    # Now track 10 on LINE_A can emit again in the new clip
    assert dedup.check_intra_clip(10, "LINE_A") is False

def test_inter_clip_exact_plate_match():
    dedup = EventDeduplicator(cooldown_seconds=25.0)

    # Register initial event at t=100s, ENTRADA, plate PCW2492
    event_1 = {
        'event_id': 'EVT_001',
        'event_timestamp': 100.0,
        'direction': 'ENTRADA',
        'plate_normalized': 'PCW2492',
        'confidence_ocr': 0.95
    }
    dedup.register_event(event_1)

    # Event 2 at t=110s (dt=10s < 25s), same direction, same plate
    dup_id = dedup.evaluate_inter_clip_duplicate(
        event_timestamp=110.0,
        direction='ENTRADA',
        plate_normalized='PCW2492',
        confidence_ocr=0.95
    )
    assert dup_id == 'EVT_001'

    # Event 3 with different direction (SALIDA) -> NOT a duplicate
    assert dedup.evaluate_inter_clip_duplicate(
        event_timestamp=112.0,
        direction='SALIDA',
        plate_normalized='PCW2492',
        confidence_ocr=0.95
    ) is None

    # Event 4 past cooldown (t=135s, dt=35s > 25s) -> NOT a duplicate
    assert dedup.evaluate_inter_clip_duplicate(
        event_timestamp=135.0,
        direction='ENTRADA',
        plate_normalized='PCW2492',
        confidence_ocr=0.95
    ) is None

def test_inter_clip_fuzzy_plate_match_with_low_confidence():
    dedup = EventDeduplicator(cooldown_seconds=25.0)

    event_orig = {
        'event_id': 'EVT_ORIG',
        'event_timestamp': 50.0,
        'direction': 'SALIDA',
        'plate_normalized': 'TAA2204',
        'confidence_ocr': 0.65
    }
    dedup.register_event(event_orig)

    # Plate with 1 char substitution and low confidence OCR (TAA2209 vs TAA2204)
    dup = dedup.evaluate_inter_clip_duplicate(
        event_timestamp=55.0,
        direction='SALIDA',
        plate_normalized='TAA2209',
        confidence_ocr=0.60
    )
    assert dup == 'EVT_ORIG'

def test_history_capacity_limit():
    dedup = EventDeduplicator(cooldown_seconds=10.0, max_history=5)
    for i in range(10):
        dedup.register_event({'event_id': f'EVT_{i}', 'event_timestamp': float(i * 10), 'direction': 'ENTRADA'})
    
    assert len(dedup.recent_events) == 5
    assert dedup.recent_events[0]['event_id'] == 'EVT_5'
    assert dedup.recent_events[-1]['event_id'] == 'EVT_9'
