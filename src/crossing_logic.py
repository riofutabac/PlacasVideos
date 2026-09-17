"""
Robust Crossing Logic and Finite State Machine for Direction Detection.
States:
[OUTSIDE] -> [APPROACHING] -> [CROSSED] -> [COMMITTED]
Uses vehicle bottom_center (road contact) to evaluate:
- Trajectory vector intersection with virtual line.
- Side of line (+/- signed distance).
- ENTRADA (increasing Y, towards camera) vs SALIDA (decreasing Y, towards distance).
- Net trajectory traversal fallback (min_y < line_y and max_y > line_y).
"""
from typing import Tuple, List, Optional
import numpy as np

def point_line_side(p: Tuple[float, float], p1: Tuple[float, float], p2: Tuple[float, float]) -> str:
    """Signed cross product of (p2 - p1) and (p - p1)."""
    val = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])
    return "SIDE_A" if val >= 0 else "SIDE_B"

def point_line_distance(p: Tuple[float, float], p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Perpendicular distance from point p to line segment p1-p2."""
    x0, y0 = p
    x1, y1 = p1
    x2, y2 = p2
    num = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
    den = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
    return float(num / den) if den > 0 else 0.0

def segments_intersect(a1: Tuple[float, float], a2: Tuple[float, float], b1: Tuple[float, float], b2: Tuple[float, float]) -> bool:
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
    return ccw(a1, b1, b2) != ccw(a2, b1, b2) and ccw(a1, a2, b1) != ccw(a1, a2, b2)

class CrossingFSM:
    def __init__(self, line_p1: Tuple[int, int], line_p2: Tuple[int, int], dir_increasing_y: str = "ENTRADA", dir_decreasing_y: str = "SALIDA"):
        self.p1 = (float(line_p1[0]), float(line_p1[1]))
        self.p2 = (float(line_p2[0]), float(line_p2[1]))
        self.dir_inc_y = dir_increasing_y
        self.dir_dec_y = dir_decreasing_y
        
        # Sample point with increasing Y (below line)
        mid_x = (self.p1[0] + self.p2[0]) / 2.0
        mid_y = (self.p1[1] + self.p2[1]) / 2.0
        sample_below = (mid_x, mid_y + 100.0)
        self.side_below = point_line_side(sample_below, self.p1, self.p2)
        self.line_y = mid_y

    def update_track(self, track_state, current_point: Tuple[float, float], timestamp: float) -> bool:
        """
        Updates the track state machine based on road contact point.
        Returns True if the event has reached COMMITTED.
        """
        track_state.trajectory.append(current_point)
        side = point_line_side(current_point, self.p1, self.p2)
        dist = point_line_distance(current_point, self.p1, self.p2)

        if track_state.last_side is None:
            track_state.last_side = side

        # 1. State: OUTSIDE / APPROACHING
        if track_state.state in ("OUTSIDE", "APPROACHING"):
            if dist < 180.0:
                track_state.state = "APPROACHING"

            # Check crossing
            prev_point = track_state.trajectory[-2] if len(track_state.trajectory) >= 2 else current_point
            side_changed = (side != track_state.last_side)
            intersected = segments_intersect(prev_point, current_point, self.p1, self.p2)

            if side_changed or intersected:
                track_state.state = "CROSSED"
                track_state.crossing_timestamp = timestamp
                track_state.crossing_point = current_point
                track_state.direction = self.dir_inc_y if (side == self.side_below or current_point[1] > prev_point[1]) else self.dir_dec_y

        # 2. State: CROSSED -> COMMITTED
        elif track_state.state == "CROSSED":
            prev_point = track_state.trajectory[0]
            # Confirm direction and commit
            if dist > 20.0 or len(track_state.trajectory) >= 4:
                # Confirm by net Y travel
                delta_y = current_point[1] - prev_point[1]
                if abs(delta_y) > 30.0:
                    track_state.direction = self.dir_inc_y if delta_y > 0 else self.dir_dec_y
                track_state.state = "COMMITTED"
                return True

        track_state.last_side = side
        return track_state.state == "COMMITTED"
