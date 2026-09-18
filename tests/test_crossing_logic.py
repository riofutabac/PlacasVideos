import pytest
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from src.crossing_logic import (
    CrossingFSM,
    point_line_side,
    point_line_distance,
    segments_intersect
)

@dataclass
class DummyTrackState:
    track_id: int
    trajectory: List[Tuple[float, float]] = field(default_factory=list)
    state: str = "OUTSIDE"
    direction: Optional[str] = None
    last_side: Optional[str] = None
    crossing_timestamp: Optional[float] = None
    crossing_point: Optional[Tuple[float, float]] = None

def test_point_line_geometry():
    p1 = (0.0, 100.0)
    p2 = (200.0, 100.0)

    # Point directly on line
    dist_on = point_line_distance((100.0, 100.0), p1, p2)
    assert abs(dist_on) < 1e-5

    # Point above line (y=50)
    dist_above = point_line_distance((100.0, 50.0), p1, p2)
    assert abs(dist_above - 50.0) < 1e-5

    # Point below line (y=150)
    dist_below = point_line_distance((100.0, 150.0), p1, p2)
    assert abs(dist_below - 50.0) < 1e-5

    side_above = point_line_side((100.0, 50.0), p1, p2)
    side_below = point_line_side((100.0, 150.0), p1, p2)
    assert side_above != side_below

def test_segments_intersect():
    # Crossing X
    s1_a, s1_b = (50.0, 50.0), (150.0, 150.0)
    s2_a, s2_b = (50.0, 150.0), (150.0, 50.0)
    assert segments_intersect(s1_a, s1_b, s2_a, s2_b) is True

    # Parallel horizontal lines
    p1_a, p1_b = (0.0, 100.0), (200.0, 100.0)
    p2_a, p2_b = (0.0, 150.0), (200.0, 150.0)
    assert segments_intersect(p1_a, p1_b, p2_a, p2_b) is False

    # Disjoint segments
    assert segments_intersect((0, 0), (10, 10), (20, 20), (30, 30)) is False

def test_crossing_fsm_entrada():
    # Line at y=1200 across road
    fsm = CrossingFSM(line_p1=(400, 1200), line_p2=(2250, 1200))
    track = DummyTrackState(track_id=1)

    # 1. Approach from distance (y=900)
    committed = fsm.update_track(track, (1000.0, 900.0), 1.0)
    assert committed is False
    assert track.state in ("OUTSIDE", "APPROACHING")

    # 2. Get closer (y=1100, dist=100 < 180)
    committed = fsm.update_track(track, (1000.0, 1100.0), 1.2)
    assert committed is False
    assert track.state == "APPROACHING"

    # 3. Cross the line to y=1250
    committed = fsm.update_track(track, (1000.0, 1250.0), 1.4)
    assert track.state == "CROSSED"
    assert track.direction == "ENTRADA"
    assert track.crossing_timestamp == 1.4

    # 4. Move further into foreground (y=1350) -> COMMITTED
    committed = fsm.update_track(track, (1000.0, 1350.0), 1.6)
    assert committed is True
    assert track.state == "COMMITTED"
    assert track.direction == "ENTRADA"

def test_crossing_fsm_salida():
    # Line at y=1200 across road
    fsm = CrossingFSM(line_p1=(400, 1200), line_p2=(2250, 1200))
    track = DummyTrackState(track_id=2)

    # 1. Approach from foreground (y=1500)
    committed = fsm.update_track(track, (1200.0, 1500.0), 10.0)
    assert committed is False

    # 2. Closer to line (y=1300)
    committed = fsm.update_track(track, (1200.0, 1300.0), 10.2)
    assert committed is False
    assert track.state == "APPROACHING"

    # 3. Cross the line towards background (y=1150)
    committed = fsm.update_track(track, (1200.0, 1150.0), 10.4)
    assert track.state == "CROSSED"
    assert track.direction == "SALIDA"

    # 4. Move further away (y=1000) -> COMMITTED
    committed = fsm.update_track(track, (1200.0, 1000.0), 10.6)
    assert committed is True
    assert track.state == "COMMITTED"
    assert track.direction == "SALIDA"

def test_non_crossing_vehicle_never_commits():
    fsm = CrossingFSM(line_p1=(400, 1200), line_p2=(2250, 1200))
    track = DummyTrackState(track_id=3)

    # Vehicle wandering only on top side (y=800..900), never crossing y=1200
    for y in [800.0, 830.0, 860.0, 880.0, 850.0, 820.0]:
        committed = fsm.update_track(track, (1000.0, y), 20.0)
        assert committed is False
        assert track.state != "CROSSED"
        assert track.state != "COMMITTED"
