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

def get_clip_start_datetime(video_filename: str) -> datetime:
    """Derives realistic timestamp from camera OSD clock with extrapolation fallback."""
    try:
        from src.clock_reader import read_clip_start_datetime
        osd_dt = read_clip_start_datetime(video_filename)
        if osd_dt is not None:
            return osd_dt
    except Exception:
        pass

    base = os.path.basename(video_filename)
    match_date = re.search(r'_(\d{4})(\d{2})(\d{2})', base)
    if match_date:
        year, month, day = int(match_date.group(1)), int(match_date.group(2)), int(match_date.group(3))
    else:
        year, month, day = 2026, 9, 9

    match = re.search(r'\((\d+)\)\.mp4$', base)
    if match:
        clip_num = int(match.group(1))
        # Dahua NVR continuous export clips are 330.0s (5m30s) each
        # Calibrated against on-screen camera OSD clock:
        # Clip 60: 16:17:16, Clip 61: 16:22:46
        base_time = datetime(year, month, day, 16, 17, 16)
        return base_time + timedelta(seconds=(clip_num - 60) * 330.0)
    return datetime(year, month, day, 16, 0, 0)

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
