"""
OSD Clock Reader for CCTV Video Clips.
Extracts native hardware timestamp from camera on-screen display (OSD).
"""
import os
import io
import base64
from datetime import datetime
from typing import Optional, Dict, Tuple
from pathlib import Path
import numpy as np
import cv2

EMBEDDED_TEMPLATES_B64 = (
    "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"
    "CgAIAgAAuwkAAAAA"
)

_TEMPLATES_CACHE: Optional[Dict[int, np.ndarray]] = None

def get_osd_templates() -> Dict[int, np.ndarray]:
    """Loads digit templates 0-9 for OSD font matching."""
    global _TEMPLATES_CACHE
    if _TEMPLATES_CACHE is not None:
        return _TEMPLATES_CACHE
    
    npz_path = Path(__file__).resolve().parent.parent / 'data' / 'osd_templates.npz'
    if npz_path.exists():
        try:
            npz = np.load(str(npz_path))
            _TEMPLATES_CACHE = {int(k[1:]): npz[k] for k in npz.files}
            return _TEMPLATES_CACHE
        except Exception:
            pass
            
    raw_bytes = base64.b64decode(EMBEDDED_TEMPLATES_B64)
    npz = np.load(io.BytesIO(raw_bytes))
    _TEMPLATES_CACHE = {int(k[1:]): npz[k] for k in npz.files}
    return _TEMPLATES_CACHE

def _match_digit_ncc(patch: np.ndarray, templates: Dict[int, np.ndarray]) -> Tuple[Optional[int], float]:
    """Finds the digit with the highest normalized cross-correlation score."""
    best_d = None
    best_score = -1.0
    for d, tpl in templates.items():
        if patch.shape[0] < tpl.shape[0] or patch.shape[1] < tpl.shape[1]:
            mh = max(tpl.shape[0], patch.shape[0])
            mw = max(tpl.shape[1], patch.shape[1])
            p = np.zeros((mh, mw), dtype=np.uint8)
            p[:patch.shape[0], :patch.shape[1]] = patch
        else:
            p = patch
        res = cv2.matchTemplate(p, tpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        if max_val > best_score:
            best_score = max_val
            best_d = d
    return best_d, best_score

def read_frame_timestamp(frame: np.ndarray, min_confidence: float = 0.80) -> Optional[datetime]:
    """Reads camera OSD timestamp directly from a video frame."""
    if frame is None or len(frame.shape) < 2:
        return None
    h, w = frame.shape[:2]
    if h < 85 or w < 2000:
        return None
    
    # Standard Dahua 2960x1664 OSD header location
    # If dimensions slightly differ, scale region proportionally
    if w == 2960 and h == 1664:
        osd_crop = frame[20:81, 1850:2960]
    else:
        # Scale to match canonical 2960x1664 coordinate space
        y0, y1 = int(20 * h / 1664), int(81 * h / 1664)
        x0, x1 = int(1850 * w / 2960), int(2960 * w / 2960)
        raw_crop = frame[y0:y1, x0:x1]
        if raw_crop.size == 0:
            return None
        osd_crop = cv2.resize(raw_crop, (2960 - 1850, 61), interpolation=cv2.INTER_LINEAR)
        
    if len(osd_crop.shape) == 3:
        gray = cv2.cvtColor(osd_crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = osd_crop
    _, th = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    
    templates = get_osd_templates()
    
    # Slot offsets relative to x=1850
    slots = {
        'd1': 1860 - 1850, 'd2': 1918 - 1850,
        'm1': 2034 - 1850, 'm2': 2092 - 1850,
        'y1': 2208 - 1850, 'y2': 2266 - 1850, 'y3': 2324 - 1850, 'y4': 2382 - 1850,
        'h1': 2498 - 1850, 'h2': 2556 - 1850,
        'min1': 2672 - 1850, 'min2': 2730 - 1850,
        's1': 2846 - 1850, 's2': 2904 - 1850,
    }
    
    vals = {}
    for name, rel_x in slots.items():
        patch = th[:, rel_x:rel_x+50]
        d, score = _match_digit_ncc(patch, templates)
        if score < min_confidence or d is None:
            return None
        vals[name] = d
        
    try:
        year = vals['y1']*1000 + vals['y2']*100 + vals['y3']*10 + vals['y4']
        month = vals['m1']*10 + vals['m2']
        day = vals['d1']*10 + vals['d2']
        hour = vals['h1']*10 + vals['h2']
        minute = vals['min1']*10 + vals['min2']
        second = vals['s1']*10 + vals['s2']
        return datetime(year, month, day, hour, minute, second)
    except ValueError:
        return None

def read_clip_start_datetime(video_path: str, min_confidence: float = 0.80) -> Optional[datetime]:
    """Reads native OSD start timestamp from the first frame of a video file."""
    if not video_path or not os.path.exists(video_path):
        return None
    try:
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            return None
        return read_frame_timestamp(frame, min_confidence=min_confidence)
    except Exception:
        return None
