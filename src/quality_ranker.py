"""
Dual Quality Ranker for ALPR Pipeline.
Implements:
1. Vehicle Frame Quality Score (Top-M selection during track)
2. Plate Crop Quality Score (Top-K selection prior to OCR)
"""
import cv2
import numpy as np
from typing import Tuple

def calculate_sharpness(image_bgr: np.ndarray) -> float:
    if image_bgr is None or image_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def calculate_overexposure_penalty(image_bgr: np.ndarray) -> float:
    """Penalty if plate has burned highlights (pixel value > 250 in > 30% of pixels)."""
    if image_bgr is None or image_bgr.size == 0:
        return 1.0
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    overexposed = np.sum(gray > 248) / gray.size
    return float(min(1.0, overexposed * 2.0))

class QualityRanker:
    def __init__(self, weights_vehicle=None, weights_plate=None):
        self.w_v = weights_vehicle or {
            'sharpness': 0.35,
            'area': 0.35,
            'confidence': 0.20,
            'border_penalty': 0.10
        }
        self.w_p = weights_plate or {
            'conf': 0.30,
            'area': 0.25,
            'sharpness': 0.25,
            'aspect': 0.10,
            'overexposure': 0.10
        }

    def score_vehicle_frame(
        self,
        vehicle_crop: np.ndarray,
        bbox: Tuple[int, int, int, int],
        frame_shape: Tuple[int, int],
        detector_confidence: float
    ) -> float:
        """
        Calculates quality score for a vehicle frame candidate.
        bbox: (x1, y1, x2, y2)
        frame_shape: (height, width)
        """
        if vehicle_crop is None or vehicle_crop.size == 0:
            return 0.0

        fh, fw = frame_shape[:2]
        x1, y1, x2, y2 = bbox
        w, h = (x2 - x1), (y2 - y1)
        area = w * h

        # 1. Normalized Sharpness (sigmoid-like scaling: 50 to 500 var)
        var = calculate_sharpness(vehicle_crop)
        norm_sharpness = min(1.0, var / 400.0)

        # 2. Normalized Area (scaling up to 300x300 px)
        norm_area = min(1.0, area / (400.0 * 400.0))

        # 3. Border Proximity Penalty
        dist_left = x1
        dist_top = y1
        dist_right = fw - x2
        dist_bottom = fh - y2
        min_dist = min(dist_left, dist_top, dist_right, dist_bottom)
        border_penalty = 0.0 if min_dist > 30 else (1.0 - (min_dist / 30.0))

        score = (
            self.w_v['sharpness'] * norm_sharpness +
            self.w_v['area'] * norm_area +
            self.w_v['confidence'] * detector_confidence -
            self.w_v['border_penalty'] * border_penalty
        )
        return float(max(0.0, score))

    def score_plate_crop(
        self,
        plate_crop: np.ndarray,
        plate_conf: float
    ) -> float:
        """
        Calculates quality score for a plate candidate crop.
        """
        if plate_crop is None or plate_crop.size == 0:
            return 0.0

        h, w = plate_crop.shape[:2]
        if h == 0 or w == 0:
            return 0.0

        # 1. Aspect ratio plausibility (Ecuadorian plates are approx 2:1 to 2.5:1 ratio: 400x200mm)
        aspect = w / float(h)
        # Optimal aspect ~ 2.0; penalize if < 1.2 or > 3.5
        if 1.5 <= aspect <= 2.6:
            aspect_score = 1.0
        elif 1.2 <= aspect < 1.5 or 2.6 < aspect <= 3.2:
            aspect_score = 0.7
        else:
            aspect_score = 0.3

        # 2. Sharpness
        var = calculate_sharpness(plate_crop)
        norm_sharpness = min(1.0, var / 300.0)

        # 3. Area (scaling up to ~120x60 px)
        area = w * h
        norm_area = min(1.0, area / (120.0 * 60.0))

        # 4. Overexposure penalty
        overexp_penalty = calculate_overexposure_penalty(plate_crop)

        score = (
            self.w_p['conf'] * plate_conf +
            self.w_p['area'] * norm_area +
            self.w_p['sharpness'] * norm_sharpness +
            self.w_p['aspect'] * aspect_score -
            self.w_p['overexposure'] * overexp_penalty
        )
        return float(max(0.0, score))
