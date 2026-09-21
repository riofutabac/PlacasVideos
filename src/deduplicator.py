"""
Two-level Deduplication Engine for ALPR Events.
1. Intra-clip: Prevents duplicate emissions from the same (track_id, line_id).
2. Inter-clip: Deduplicates consecutive video boundaries using:
   - Same direction
   - Time window (delta_t < cooldown_seconds)
   - Exact plate match OR (Levenshtein <= 1 + uncertain OCR / visual similarity)
   - Non-destructive: sets duplicate_of = original_event_id for full auditability.
"""
from typing import List, Dict, Optional, Any
import numpy as np
import cv2

def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def calculate_visual_similarity(crop1: np.ndarray, crop2: np.ndarray) -> float:
    """Computes color histogram correlation between two vehicle crops."""
    if crop1 is None or crop2 is None or crop1.size == 0 or crop2.size == 0:
        return 0.0
    try:
        hsv1 = cv2.cvtColor(crop1, cv2.COLOR_BGR2HSV)
        hsv2 = cv2.cvtColor(crop2, cv2.COLOR_BGR2HSV)
        hist1 = cv2.calcHist([hsv1], [0, 1], None, [30, 32], [0, 180, 0, 256])
        hist2 = cv2.calcHist([hsv2], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return float(cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL))
    except Exception:
        return 0.0

class EventDeduplicator:
    def __init__(self, cooldown_seconds: float = 60.0, max_history: int = 50):
        self.cooldown_seconds = cooldown_seconds
        self.max_history = max_history
        self.emitted_tracks: set = set()  # (track_id, line_id) for current clip
        self.recent_events: List[Dict[str, Any]] = []

    def reset_clip_state(self):
        """Resets intra-clip track IDs between video files, retaining recent events for inter-clip."""
        self.emitted_tracks.clear()

    def check_intra_clip(self, track_id: int, line_id: str) -> bool:
        """Returns True if this track has already emitted for this line in this clip."""
        key = (track_id, line_id)
        if key in self.emitted_tracks:
            return True
        self.emitted_tracks.add(key)
        return False

    def evaluate_inter_clip_duplicate(
        self,
        event_timestamp: float,
        direction: str,
        plate_normalized: Optional[str],
        confidence_ocr: float,
        vehicle_crop: Optional[np.ndarray] = None,
        abs_timestamp: Optional[float] = None
    ) -> Optional[str]:
        """
        Checks if the event is a duplicate of a recent event from a preceding clip.
        Returns: original_event_id if duplicate, else None.
        """
        current_ts = abs_timestamp if abs_timestamp is not None else event_timestamp
        for prev in reversed(self.recent_events):
            prev_ts = prev.get('abs_timestamp', prev.get('event_timestamp', 0.0))
            dt = abs(current_ts - prev_ts)
            if dt > self.cooldown_seconds:
                continue

            # Direction must match
            if direction != prev['direction']:
                continue

            prev_plate = prev.get('plate_normalized')

            # 1. Exact plate match with good confidence
            if plate_normalized and prev_plate and plate_normalized == prev_plate:
                return prev['event_id']

            # 2. Approximate plate match (Levenshtein <= 1) with uncertain OCR or visual similarity
            if plate_normalized and prev_plate:
                lev = levenshtein_distance(plate_normalized, prev_plate)
                if lev <= 1:
                    if confidence_ocr < 0.75 or prev.get('confidence_ocr', 1.0) < 0.75:
                        return prev['event_id']
                    
                    if vehicle_crop is not None and prev.get('vehicle_crop') is not None:
                        sim = calculate_visual_similarity(vehicle_crop, prev['vehicle_crop'])
                        if sim > 0.65:
                            return prev['event_id']

            # 3. Vehicle without plate crossing within small window (< 6s) with high visual similarity
            if not plate_normalized and not prev_plate and dt < 6.0:
                if vehicle_crop is not None and prev.get('vehicle_crop') is not None:
                    sim = calculate_visual_similarity(vehicle_crop, prev['vehicle_crop'])
                    if sim > 0.75:
                        return prev['event_id']

            # 4. Same vehicle track fragment where one detected a plate and the other did not (dt < 6s, sim > 0.70)
            if ((plate_normalized and not prev_plate) or (not plate_normalized and prev_plate)) and dt < 6.0:
                if vehicle_crop is not None and prev.get('vehicle_crop') is not None:
                    sim = calculate_visual_similarity(vehicle_crop, prev['vehicle_crop'])
                    if sim > 0.70:
                        return prev['event_id']

        return None

    def register_event(self, event_dict: Dict[str, Any], vehicle_crop: Optional[np.ndarray] = None, abs_timestamp: Optional[float] = None):
        """Registers an emitted event into recent history."""
        stored = dict(event_dict)
        stored['abs_timestamp'] = abs_timestamp if abs_timestamp is not None else event_dict.get('event_timestamp', 0.0)
        stored['vehicle_crop'] = vehicle_crop
        self.recent_events.append(stored)
        if len(self.recent_events) > self.max_history:
            self.recent_events.pop(0)
