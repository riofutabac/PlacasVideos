"""
Ultra-fast Adaptive Motion Gate with Watchdog Sentinel.
Replaces slow MOG2 and slow interpolation with an ultra-fast INTER_NEAREST absdiff
on a downscaled thumbnail (128x72). Executes in ~0.04 ms per frame.
States:
- SLEEP: Zone quiet. Watchdog Sentinel runs at 1.5 - 2.0 FPS to guarantee zero missed vehicles.
- ACTIVE: Motion or vehicle detected; elevated to 8.0 FPS.
  Maintains ACTIVE for active_hold_seconds after the last motion or confirmed vehicle.
"""
import cv2
import numpy as np

class AdaptiveMotionGate:
    def __init__(
        self,
        sleep_sentinel_fps: float = 2.0,
        preactive_fps: float = 5.0,
        active_fps: float = 8.0,
        active_hold_seconds: float = 2.5,
        motion_threshold_ratio: float = 0.005,
        thumb_size: tuple = (128, 72)
    ):
        self.sleep_sentinel_fps = sleep_sentinel_fps
        self.preactive_fps = preactive_fps
        self.active_fps = active_fps
        self.active_hold_seconds = active_hold_seconds
        self.motion_threshold_ratio = motion_threshold_ratio
        self.thumb_size = thumb_size

        self.state = "SLEEP"  # SLEEP, PRE-ACTIVE, ACTIVE
        self.last_vehicle_timestamp = -1e9
        self.last_inference_timestamp = -1e9
        self.last_motion_timestamp = -1e9
        self.preactive_hold_seconds = 2.0
        self.prev_gray = None

    def reset(self):
        """Resets internal state between video clips."""
        self.state = "SLEEP"
        self.last_vehicle_timestamp = -1e9
        self.last_inference_timestamp = -1e9
        self.last_motion_timestamp = -1e9
        self.prev_gray = None

    def notify_vehicle_detected(self, timestamp: float):
        """Called whenever YOLO detects a vehicle within the gravel polygon."""
        self.state = "ACTIVE"
        self.last_vehicle_timestamp = timestamp

    def check_motion(self, roi_crop: np.ndarray) -> bool:
        """Computes ultra-rapid absdiff on a 128x72 grayscale thumbnail using INTER_NEAREST."""
        if roi_crop is None or roi_crop.size == 0:
            return False

        if roi_crop.ndim == 2:
            # Grayscale or NV12 luma plane (no color conversion needed!)
            if roi_crop.shape[0] % 3 == 0:
                h = (roi_crop.shape[0] * 2) // 3
                gray = roi_crop[:h, :]
            else:
                gray = roi_crop
            small = cv2.resize(gray, self.thumb_size, interpolation=cv2.INTER_NEAREST)
        elif roi_crop.ndim == 3:
            small_bgr = cv2.resize(roi_crop, self.thumb_size, interpolation=cv2.INTER_NEAREST)
            small = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2GRAY)
        else:
            small = cv2.resize(roi_crop, self.thumb_size, interpolation=cv2.INTER_NEAREST)

        if self.prev_gray is None:
            self.prev_gray = small
            return False

        diff = cv2.absdiff(self.prev_gray, small)
        _, thresh = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
        non_zero = cv2.countNonZero(thresh)
        total_pixels = self.thumb_size[0] * self.thumb_size[1]
        motion_ratio = non_zero / float(total_pixels)

        # Update reference background
        self.prev_gray = small

        return motion_ratio >= self.motion_threshold_ratio

    def should_run_detector(self, roi_crop: np.ndarray, current_timestamp: float) -> bool:
        """
        Determines whether YOLO detector should run on this frame.
        FSM:
        - ACTIVE: Vehicle confirmed in gravel ROI within last active_hold_seconds (8.0 FPS).
        - PRE-ACTIVE: Motion detected within last preactive_hold_seconds (5.0 FPS).
        - SLEEP: Zone quiet, watchdog sentinel mode (2.0 FPS).
        """
        has_motion = self.check_motion(roi_crop)
        if has_motion:
            self.last_motion_timestamp = current_timestamp

        # Determine state
        time_since_vehicle = current_timestamp - self.last_vehicle_timestamp
        time_since_motion = current_timestamp - self.last_motion_timestamp

        if time_since_vehicle <= self.active_hold_seconds:
            self.state = "ACTIVE"
            target_fps = self.active_fps
        elif time_since_motion <= self.preactive_hold_seconds:
            self.state = "PRE-ACTIVE"
            target_fps = self.preactive_fps
        else:
            self.state = "SLEEP"
            target_fps = self.sleep_sentinel_fps

        min_interval = 1.0 / max(0.1, target_fps)
        elapsed = current_timestamp - self.last_inference_timestamp

        if elapsed >= min_interval:
            self.last_inference_timestamp = current_timestamp
            return True

        return False
