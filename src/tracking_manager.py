"""
Tracking Manager for ALPR Pipeline.
Handles track stitching, track state transitions, quality scoring insertion,
and inactive track pruning.
"""
from typing import Dict, List, Tuple, Optional
import numpy as np

from src.pipeline_types import TrackState, VehicleFrameCandidate
from src.crossing_logic import CrossingFSM
from src.quality_ranker import QualityRanker
from src.timer_profiler import PipelineProfiler

def stitch_or_create_track(
    tracks: Dict[int, TrackState],
    trk_id: int,
    cls_name: str,
    contact_pt: Tuple[float, float],
    timestamp: float
) -> TrackState:
    """Stitches newly assigned tracker ID with a recently lost track if spatio-temporally proximate."""
    if trk_id not in tracks:
        stitched_id = None
        best_dist = float('inf')
        for old_id, old_state in list(tracks.items()):
            if old_id != trk_id and not old_state.has_emitted:
                time_gap = timestamp - old_state.last_timestamp
                if 0.05 <= time_gap <= 3.0:
                    class_compat = (
                        old_state.vehicle_class == cls_name
                        or {old_state.vehicle_class, cls_name}.issubset({'truck', 'bus', 'car'})
                        or {old_state.vehicle_class, cls_name}.issubset({'motorcycle', 'person'})
                    )
                    if class_compat and old_state.trajectory:
                        last_pt = old_state.trajectory[-1]
                        dist = np.hypot(contact_pt[0] - last_pt[0], contact_pt[1] - last_pt[1])
                        if dist < 450.0 and dist < best_dist:
                            best_dist = dist
                            stitched_id = old_id

        if stitched_id is not None:
            old_state = tracks.pop(stitched_id)
            old_state.track_id = trk_id
            old_state.last_timestamp = timestamp
            if cls_name != 'vehicle':
                old_state.vehicle_class = cls_name
            tracks[trk_id] = old_state
        else:
            tracks[trk_id] = TrackState(
                track_id=trk_id,
                first_timestamp=timestamp,
                last_timestamp=timestamp,
                vehicle_class=cls_name
            )

    state = tracks[trk_id]
    state.last_timestamp = timestamp
    return state


def update_tracks_and_fsm(
    crop_roi: np.ndarray,
    tracked_dets,
    tracks: Dict[int, TrackState],
    cx1: int,
    cy1: int,
    timestamp: float,
    vehicle_names: Dict[int, str],
    ranker: QualityRanker,
    fsm: CrossingFSM,
    top_m_frames: int,
    full_frame_shape: Tuple[int, int],
    profiler: PipelineProfiler,
    post_crossing_window_seconds: float = 0.0
) -> List[TrackState]:
    """Updates tracks with new detections, updates FSM crossing, and returns committed tracks ready for emission."""
    committed_tracks = []
    full_h, full_w = full_frame_shape

    for i in range(len(tracked_dets)):
        trk_id = int(tracked_dets.tracker_id[i]) if tracked_dets.tracker_id is not None and len(tracked_dets.tracker_id) > i else None
        if trk_id is None:
            continue

        rx1, ry1, rx2, ry2 = tracked_dets.xyxy[i]
        fx1, fy1, fx2, fy2 = int(rx1 + cx1), int(ry1 + cy1), int(rx2 + cx1), int(ry2 + cy1)
        conf = float(tracked_dets.confidence[i])
        cls_id = int(tracked_dets.class_id[i])
        cls_name = vehicle_names.get(cls_id, 'vehicle')
        if cls_name in ('person', 'bicycle'):
            cls_name = 'motorcycle'

        contact_pt = ((fx1 + fx2) / 2.0, float(fy2 - 15))
        state = stitch_or_create_track(tracks, trk_id, cls_name, contact_pt, timestamp)

        # Only extract crop & rank if track has not yet been emitted
        if not state.has_emitted:
            veh_crop_view = crop_roi[max(0, int(ry1)):min(crop_roi.shape[0], int(ry2)), max(0, int(rx1)):min(crop_roi.shape[1], int(ry2))]
            q_score = ranker.score_vehicle_frame(
                vehicle_crop=veh_crop_view,
                bbox=(fx1, fy1, fx2, fy2),
                frame_shape=(full_h, full_w),
                detector_confidence=conf
            )
            # Only copy buffer if candidate qualifies for top_m
            if len(state.best_vehicle_frames) < top_m_frames or q_score > state.best_vehicle_frames[-1].score:
                cand = VehicleFrameCandidate(score=q_score, timestamp=timestamp, vehicle_crop=veh_crop_view.copy(), bbox_in_full_frame=(fx1, fy1, fx2, fy2))
                if len(state.best_vehicle_frames) < top_m_frames:
                    state.best_vehicle_frames.append(cand)
                else:
                    state.best_vehicle_frames[-1] = cand
                state.best_vehicle_frames.sort(key=lambda x: x.score, reverse=True)

        profiler.start_stage('crossing_fsm')
        fsm.update_track(state, contact_pt, timestamp)
        profiler.stop_stage('crossing_fsm')

    # Emit tracks whose post-crossing window has elapsed
    for state in tracks.values():
        if state.state == "COMMITTED" and not state.has_emitted:
            ref_t = state.crossing_timestamp if state.crossing_timestamp is not None else state.first_timestamp
            if (timestamp - ref_t) >= post_crossing_window_seconds:
                state.has_emitted = True
                committed_tracks.append(state)

    return committed_tracks


def prune_inactive_tracks(
    tracks: Dict[int, TrackState],
    timestamp: float,
    line_y: float,
    max_idle: float = 3.0,
    force_all: bool = False
) -> List[TrackState]:
    """Prunes tracks that have aged out; returns tracks that crossed the line before disappearing."""
    crossed_tracks = []
    for trk_id, state in list(tracks.items()):
        if force_all or (timestamp - state.last_timestamp) > max_idle:
            if not state.has_emitted and len(state.trajectory) >= 2:
                y_coords = [pt[1] for pt in state.trajectory]
                if (min(y_coords) < line_y and max(y_coords) > line_y) or state.state in ("CROSSED", "COMMITTED"):
                    state.has_emitted = True
                    if not state.direction:
                        state.direction = "ENTRADA" if state.trajectory[-1][1] > state.trajectory[0][1] else "SALIDA"
                    crossed_tracks.append(state)
            if not force_all:
                del tracks[trk_id]
    return crossed_tracks
