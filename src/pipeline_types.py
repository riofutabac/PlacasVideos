"""
Common Data Structures and Timestamp Utilities for ALPR Pipeline.
"""
import os
import re
import hashlib
from datetime import datetime, timedelta
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np

def get_clip_start_datetime(video_filename: str, fallback_filename: Optional[str] = None) -> datetime:
    """Derives realistic timestamp from camera OSD clock with extrapolation fallback."""
    # 1. Try reading hardware clock from frame pixels
    for target_path in (video_filename, fallback_filename):
        if target_path and os.path.exists(target_path):
            try:
                from src.clock_reader import read_clip_start_datetime
                osd_dt = read_clip_start_datetime(target_path)
                if osd_dt is not None:
                    return osd_dt
            except Exception:
                pass

    # 2. Extract timestamp info from filename
    ref_name = fallback_filename if fallback_filename else video_filename
    base = os.path.basename(ref_name) if ref_name else ""

    match_full = re.search(r'_(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})', base)
    match_date = re.search(r'_(\d{4})(\d{2})(\d{2})', base)
    if match_full:
        year, month, day = int(match_full.group(1)), int(match_full.group(2)), int(match_full.group(3))
        h, m, s = int(match_full.group(4)), int(match_full.group(5)), int(match_full.group(6))
        export_start_dt = datetime(year, month, day, h, m, s)
    elif match_date:
        year, month, day = int(match_date.group(1)), int(match_date.group(2)), int(match_date.group(3))
        export_start_dt = datetime(year, month, day, 10, 56, 51)
    else:
        year, month, day = 2026, 9, 9
        export_start_dt = datetime(year, month, day, 16, 0, 0)

    match_clip = re.search(r'\((\d+)\)\.mp4$', base)
    if match_clip:
        clip_num = int(match_clip.group(1))
        # Dahua NVR continuous export clips are 330.0s (5m30s) each
        # Calibrated against on-screen camera OSD clock:
        # Clip 60: 16:17:16, Clip 61: 16:22:46
        base_time = datetime(year, month, day, 16, 17, 16)
        return base_time + timedelta(seconds=(clip_num - 60) * 330.0)

    # Unnumbered clip uses derived export start time (Clip 0 = 10:56:51) or fallback
    return export_start_dt

@dataclass
class VehicleFrameCandidate:
    score: float
    timestamp: float
    vehicle_crop: np.ndarray
    bbox_in_full_frame: Tuple[int, int, int, int]

@dataclass
class TrackState:
    track_id: int
    first_timestamp: float
    last_timestamp: float
    vehicle_class: str
    trajectory: List[Tuple[float, float]] = field(default_factory=list)
    state: str = "OUTSIDE"  # OUTSIDE, APPROACHING, CROSSED, COMMITTED
    direction: Optional[str] = None  # 'ENTRADA', 'SALIDA'
    last_side: Optional[str] = None
    crossing_timestamp: Optional[float] = None
    crossing_point: Optional[Tuple[float, float]] = None
    crossing_line_id: Optional[str] = None
    has_emitted: bool = False
    best_vehicle_frames: List[VehicleFrameCandidate] = field(default_factory=list)

def compute_file_hash(filepath: str) -> str:
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()
