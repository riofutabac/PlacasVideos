import numpy as np
import pytest

from src.pipeline_types import TrackState
from src.crossing_logic import CrossingFSM
from src.quality_ranker import QualityRanker
from src.timer_profiler import PipelineProfiler
from src.tracking_manager import (
    stitch_or_create_track,
    prune_inactive_tracks,
    update_tracks_and_fsm
)

def test_stitch_or_create_new_track():
    tracks = {}
    st = stitch_or_create_track(
        tracks=tracks,
        trk_id=1,
        cls_name="car",
        contact_pt=(500.0, 1000.0),
        timestamp=10.0
    )
    assert st.track_id == 1
    assert st.vehicle_class == "car"
    assert st.first_timestamp == 10.0
    assert 1 in tracks

def test_stitch_lost_track_proximity():
    tracks = {}
    # Create track 1 at t=10.0 at (500, 1000)
    old_st = TrackState(
        track_id=1,
        first_timestamp=10.0,
        last_timestamp=10.0,
        vehicle_class="car",
        trajectory=[(500.0, 1000.0)]
    )
    tracks[1] = old_st

    # New detection track 2 arrives at t=10.5 at (520, 1030) - close proximity and compatible class
    st = stitch_or_create_track(
        tracks=tracks,
        trk_id=2,
        cls_name="car",
        contact_pt=(520.0, 1030.0),
        timestamp=10.5
    )
    # Track 1 should be stitched into track 2
    assert st.track_id == 2
    assert 1 not in tracks
    assert 2 in tracks
    assert st.first_timestamp == 10.0
    assert st.last_timestamp == 10.5

def test_prune_inactive_tracks():
    tracks = {}
    # Track that crossed the virtual line (y=1200) from y=1100 to y=1300
    st_crossed = TrackState(
        track_id=1,
        first_timestamp=5.0,
        last_timestamp=6.0,
        vehicle_class="car",
        trajectory=[(500.0, 1100.0), (500.0, 1300.0)]
    )
    # Track that did NOT cross the line
    st_not_crossed = TrackState(
        track_id=2,
        first_timestamp=5.0,
        last_timestamp=6.0,
        vehicle_class="truck",
        trajectory=[(500.0, 900.0), (500.0, 950.0)]
    )
    tracks[1] = st_crossed
    tracks[2] = st_not_crossed

    # At t=10.0 (elapsed > 3.0s idle), both should be pruned, but only track 1 emitted
    emitted = prune_inactive_tracks(tracks, timestamp=10.0, line_y=1200.0, max_idle=3.0)
    assert len(emitted) == 1
    assert emitted[0].track_id == 1
    assert emitted[0].direction == "ENTRADA"  # increasing y (1100 -> 1300)
    assert len(tracks) == 0

def test_update_tracks_and_fsm_integration():
    class DummyDetections:
        def __init__(self):
            self.tracker_id = np.array([10])
            self.xyxy = np.array([[100, 100, 200, 200]], dtype=np.float32)
            self.confidence = np.array([0.92], dtype=np.float32)
            self.class_id = np.array([2], dtype=np.int32)
        def __len__(self):
            return 1

    tracks = {}
    fsm = CrossingFSM(line_p1=[0, 1200], line_p2=[3000, 1200])
    ranker = QualityRanker()
    profiler = PipelineProfiler()
    crop_roi = np.zeros((300, 300, 3), dtype=np.uint8)

    committed = update_tracks_and_fsm(
        crop_roi=crop_roi,
        tracked_dets=DummyDetections(),
        tracks=tracks,
        cx1=0,
        cy1=0,
        timestamp=1.0,
        vehicle_names={2: "car"},
        ranker=ranker,
        fsm=fsm,
        top_m_frames=5,
        full_frame_shape=(1664, 2960),
        profiler=profiler
    )
    assert 10 in tracks
    assert len(tracks[10].best_vehicle_frames) == 1
    assert tracks[10].vehicle_class == "car"
    assert len(committed) == 0  # Not crossed yet

def test_post_crossing_window_zero_emits_immediately():
    class DummyDet:
        def __init__(self, y1, y2):
            self.tracker_id = np.array([10])
            self.xyxy = np.array([[100, y1, 200, y2]], dtype=np.float32)
            self.confidence = np.array([0.90], dtype=np.float32)
            self.class_id = np.array([2], dtype=np.int32)
        def __len__(self):
            return 1

    tracks = {}
    fsm = CrossingFSM(line_p1=[0, 1200], line_p2=[3000, 1200])
    ranker = QualityRanker()
    profiler = PipelineProfiler()
    crop = np.zeros((1500, 500, 3), dtype=np.uint8)

    # Frame 1 at t=1.0, before line (y=1100)
    update_tracks_and_fsm(crop, DummyDet(1000, 1100), tracks, 0, 0, 1.0, {2: "car"}, ranker, fsm, 5, (1664, 2960), profiler, post_crossing_window_seconds=0.0)
    # Frame 2 at t=2.0, crossing line (y=1220)
    update_tracks_and_fsm(crop, DummyDet(1100, 1220), tracks, 0, 0, 2.0, {2: "car"}, ranker, fsm, 5, (1664, 2960), profiler, post_crossing_window_seconds=0.0)
    # Frame 3 at t=3.0, committing (y=1260)
    emitted = update_tracks_and_fsm(crop, DummyDet(1150, 1260), tracks, 0, 0, 3.0, {2: "car"}, ranker, fsm, 5, (1664, 2960), profiler, post_crossing_window_seconds=0.0)

    assert len(emitted) == 1
    assert emitted[0].track_id == 10
    assert emitted[0].crossing_timestamp == 2.0
    assert emitted[0].direction == "ENTRADA"
    assert emitted[0].has_emitted is True

def test_post_crossing_window_delays_emission_and_accumulates_better_frames():
    class DummyDet:
        def __init__(self, y1, y2):
            self.tracker_id = np.array([10])
            self.xyxy = np.array([[100, y1, 200, y2]], dtype=np.float32)
            self.confidence = np.array([0.90], dtype=np.float32)
            self.class_id = np.array([2], dtype=np.int32)
        def __len__(self):
            return 1

    tracks = {}
    fsm = CrossingFSM(line_p1=[0, 1200], line_p2=[3000, 1200])
    ranker = QualityRanker()
    profiler = PipelineProfiler()
    crop = np.zeros((1500, 500, 3), dtype=np.uint8)

    # Frame 1 at t=1.0
    update_tracks_and_fsm(crop, DummyDet(1000, 1100), tracks, 0, 0, 1.0, {2: "car"}, ranker, fsm, 5, (1664, 2960), profiler, post_crossing_window_seconds=1.5)
    # Frame 2 at t=2.0 (crossing occurs here, crossing_timestamp = 2.0)
    update_tracks_and_fsm(crop, DummyDet(1100, 1220), tracks, 0, 0, 2.0, {2: "car"}, ranker, fsm, 5, (1664, 2960), profiler, post_crossing_window_seconds=1.5)
    # Frame 3 at t=2.5 (committed occurs here, elapsed since crossing = 0.5s < 1.5s -> should NOT emit yet)
    emitted_3 = update_tracks_and_fsm(crop, DummyDet(1150, 1260), tracks, 0, 0, 2.5, {2: "car"}, ranker, fsm, 5, (1664, 2960), profiler, post_crossing_window_seconds=1.5)
    assert len(emitted_3) == 0
    assert tracks[10].state == "COMMITTED"
    assert tracks[10].has_emitted is False
    assert tracks[10].crossing_timestamp == 2.0

    # Frame 4 at t=3.0 (elapsed since crossing = 1.0s < 1.5s -> still should NOT emit)
    emitted_4 = update_tracks_and_fsm(crop, DummyDet(1200, 1300), tracks, 0, 0, 3.0, {2: "car"}, ranker, fsm, 5, (1664, 2960), profiler, post_crossing_window_seconds=1.5)
    assert len(emitted_4) == 0
    assert tracks[10].has_emitted is False

    # Frame 5 at t=3.6 (elapsed = 1.6s >= 1.5s -> SHOULD EMIT!)
    emitted_5 = update_tracks_and_fsm(crop, DummyDet(1250, 1350), tracks, 0, 0, 3.6, {2: "car"}, ranker, fsm, 5, (1664, 2960), profiler, post_crossing_window_seconds=1.5)
    assert len(emitted_5) == 1
    assert emitted_5[0].track_id == 10
    # Crucial: crossing_timestamp must remain 2.0, NOT 3.6!
    assert emitted_5[0].crossing_timestamp == 2.0
    assert emitted_5[0].direction == "ENTRADA"
    assert emitted_5[0].has_emitted is True
    # Frames were accumulated throughout all 5 steps
    assert len(emitted_5[0].best_vehicle_frames) == 5

def test_insert_vehicle_candidate_diversity():
    from src.tracking_manager import insert_vehicle_candidate
    from src.pipeline_types import VehicleFrameCandidate

    candidates = []
    dummy_crop = np.zeros((10, 10, 3), dtype=np.uint8)

    # 1. Insert first frame at t=1.00 with score=0.80
    c1 = VehicleFrameCandidate(score=0.80, timestamp=1.00, vehicle_crop=dummy_crop, bbox_in_full_frame=(0, 0, 10, 10))
    assert insert_vehicle_candidate(candidates, c1, top_m=3, min_separation=0.3) is True
    assert len(candidates) == 1
    assert candidates[0].score == 0.80

    # 2. Frame at t=1.10 (too close: diff=0.10 < 0.30) with lower score=0.75 -> rejected
    c2 = VehicleFrameCandidate(score=0.75, timestamp=1.10, vehicle_crop=dummy_crop, bbox_in_full_frame=(0, 0, 10, 10))
    assert insert_vehicle_candidate(candidates, c2, top_m=3, min_separation=0.3) is False
    assert len(candidates) == 1
    assert candidates[0].score == 0.80

    # 3. Frame at t=1.15 (too close: diff=0.15 < 0.30) with HIGHER score=0.90 -> replaces c1!
    c3 = VehicleFrameCandidate(score=0.90, timestamp=1.15, vehicle_crop=dummy_crop, bbox_in_full_frame=(0, 0, 10, 10))
    assert insert_vehicle_candidate(candidates, c3, top_m=3, min_separation=0.3) is True
    assert len(candidates) == 1
    assert candidates[0].score == 0.90
    assert candidates[0].timestamp == 1.15

    # 4. Frame at t=1.50 (diff=0.35 >= 0.30) -> accepted into empty slot
    c4 = VehicleFrameCandidate(score=0.70, timestamp=1.50, vehicle_crop=dummy_crop, bbox_in_full_frame=(0, 0, 10, 10))
    assert insert_vehicle_candidate(candidates, c4, top_m=3, min_separation=0.3) is True
    assert len(candidates) == 2

    # 5. Frame at t=1.90 (diff=0.40 >= 0.30) -> accepted into last slot
    c5 = VehicleFrameCandidate(score=0.85, timestamp=1.90, vehicle_crop=dummy_crop, bbox_in_full_frame=(0, 0, 10, 10))
    assert insert_vehicle_candidate(candidates, c5, top_m=3, min_separation=0.3) is True
    assert len(candidates) == 3
    # Sorted order
    assert [c.score for c in candidates] == [0.90, 0.85, 0.70]


