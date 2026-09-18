"""
OpenCV Video Decoder for ALPR Pipeline (CPU Baseline and Fallback).
"""
from typing import Generator, Tuple, Optional, Dict, Any
import numpy as np
import cv2
from .base import BaseVideoDecoder

class OpenCVDecoder(BaseVideoDecoder):
    def __init__(self, video_path: str, profiler: Optional[Any] = None, crop_rect: Optional[Dict[str, int]] = None):
        super().__init__(video_path, profiler, crop_rect=crop_rect)
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"OpenCV no pudo abrir el archivo de video: {video_path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration_sec = (self.total_frames / self.fps) if self.fps > 0 else 0.0
        if not self.is_cropped:
            self.cx2 = self.width
            self.cy2 = self.height
            self.crop_w = self.width
            self.crop_h = self.height

    def __iter__(self) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        frame_idx = 0
        while self.cap.isOpened():
            if self.profiler:
                self.profiler.start_stage('decode')
            ret, frame = self.cap.read()
            if self.profiler:
                self.profiler.stop_stage('decode')
            if not ret or frame is None:
                break
            if self.is_cropped:
                frame = frame[self.cy1:self.cy2, self.cx1:self.cx2]
            timestamp = frame_idx / self.fps
            yield frame_idx, timestamp, frame
            frame_idx += 1

    def release(self):
        if self.cap:
            self.cap.release()
