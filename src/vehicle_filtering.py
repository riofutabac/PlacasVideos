"""
Vehicle Detection & Decoder Setup Mixin for ALPRPipeline.
Extracted from pipeline_runner.py to keep source files under the 400-line
maintainability ceiling. Behavior is unchanged; this is a pure code move.
"""
import time
from typing import Tuple

import numpy as np

from src.video_decoder import create_decoder
from src import fast_convert


class VehicleFilterMixin:
    """Mixin providing per-clip decoder setup and per-frame vehicle detection/filtering.
    Expects the host class (ALPRPipeline) to provide: self.cfg, self.profiler, self.crop_rect,
    self.vehicle_runner, self.vehicle_imgsz, self.vehicle_conf, self.vehicle_classes,
    self.rejection_logger, self._current_clip_id, self.is_point_in_gravel, self.motion_gate,
    self.fast_convert.
    """

    def _init_clip_decoder(self, video_path: str):
        backend = self.cfg.get('video', {}).get('decode_backend', 'auto')
        queue_sz = self.cfg.get('video', {}).get('decoder_queue_size', 128)
        decoder = create_decoder(
            video_path,
            backend=backend,
            profiler=self.profiler,
            crop_rect=self.crop_rect,
            queue_size=queue_sz
        )
        is_dec_cropped = getattr(decoder, 'is_cropped', False)
        is_nv12 = getattr(decoder, 'frame_format', 'bgr') == 'nv12'
        dec_h = decoder.crop_h if is_dec_cropped else decoder.height
        dec_w = decoder.crop_w if is_dec_cropped else decoder.width
        bgr_buf = np.empty((dec_h, dec_w, 3), dtype=np.uint8) if is_nv12 else None
        cx1, cy1 = self.crop_rect['x_min'], self.crop_rect['y_min']
        cx2, cy2 = self.crop_rect['x_max'], self.crop_rect['y_max']
        return decoder, bgr_buf, is_nv12, is_dec_cropped, dec_h, dec_w, cx1, cy1, cx2, cy2

    def _build_fast_detection_input(self, raw_roi: np.ndarray, is_nv12: bool, dec_h: int, dec_w: int):
        """Builds the small detection-only image for --fast-convert. Returns (small_img, scale)."""
        self.profiler.start_stage('frame_conversion')
        if is_nv12:
            small_img, scale = fast_convert.build_detection_image_nv12(raw_roi, dec_h, dec_w, self.vehicle_imgsz)
        else:
            small_img, scale = fast_convert.build_detection_image_bgr(raw_roi, self.vehicle_imgsz)
        self.profiler.stop_stage('frame_conversion')
        return small_img, scale

    def _maybe_fullres_crop(self, raw_roi: np.ndarray, is_nv12: bool, decoder, bgr_buffer, have_detections: bool):
        """Produces the full-resolution BGR ROI (today's conversion) only when at least one
        vehicle survived filtering this frame, since that is the only case quality ranking
        and evidence crops need full-res pixels. Returns None when skipped."""
        if not have_detections:
            return None
        self.profiler.start_stage('frame_conversion_fullres')
        crop_roi = decoder.to_bgr(raw_roi, dst=bgr_buffer) if is_nv12 else raw_roi
        self.profiler.stop_stage('frame_conversion_fullres')
        return crop_roi

    def _run_detection_stage(self, raw_roi: np.ndarray, is_nv12: bool, dec_h: int, dec_w: int,
                              decoder, bgr_buffer, cx1: int, cy1: int, timestamp: float):
        """Runs vehicle detection for one analyzed frame, dispatching to the fast-convert
        (downscaled detection + on-demand full-res crop) or legacy (always full-res) path.
        Returns (valid_boxes, valid_confs, valid_classes, crop_roi) where crop_roi is the
        full-resolution BGR ROI (None only in fast-convert mode with zero surviving detections)."""
        if self.fast_convert:
            small_img, det_scale = self._build_fast_detection_input(raw_roi, is_nv12, dec_h, dec_w)
            valid_boxes, valid_confs, valid_classes = self._detect_and_filter_vehicles(
                small_img, cx1, cy1, timestamp, scale=det_scale
            )
            crop_roi = self._maybe_fullres_crop(raw_roi, is_nv12, decoder, bgr_buffer, bool(valid_boxes))
            return valid_boxes, valid_confs, valid_classes, crop_roi

        self.profiler.start_stage('frame_conversion')
        crop_roi = decoder.to_bgr(raw_roi, dst=bgr_buffer) if is_nv12 else raw_roi
        self.profiler.stop_stage('frame_conversion')
        valid_boxes, valid_confs, valid_classes = self._detect_and_filter_vehicles(crop_roi, cx1, cy1, timestamp)
        return valid_boxes, valid_confs, valid_classes, crop_roi

    def _detect_and_filter_vehicles(self, crop_roi: np.ndarray, cx1: int, cy1: int, timestamp: float, scale: float = 1.0):
        self.profiler.start_stage('vehicle_detection')
        raw_boxes, raw_confs, raw_classes, timing = self.vehicle_runner.predict(
            crop_roi, imgsz=self.vehicle_imgsz, conf=self.vehicle_conf, classes=self.vehicle_classes
        )
        self.profiler.stop_stage('vehicle_detection')
        self.profiler.record_stage_time('vehicle_preprocess', timing.get('preprocess', 0.0))
        self.profiler.record_stage_time('vehicle_inference', timing.get('inference', 0.0))

        if scale != 1.0 and raw_boxes:
            raw_boxes = fast_convert.rescale_boxes(raw_boxes, scale)

        t_post_start = time.perf_counter()
        valid_boxes, valid_confs, valid_classes = [], [], []
        log_rejections = self.rejection_logger.enabled
        for (rx1, ry1, rx2, ry2), conf, cls_id in zip(raw_boxes, raw_confs, raw_classes):
            fx1, fy1, fx2, fy2 = int(rx1 + cx1), int(ry1 + cy1), int(rx2 + cx1), int(ry2 + cy1)
            # Exclude stationary parked van on far bottom-left curb (x <= 460)
            if fx2 <= 460 and fy2 > 1200:
                if log_rejections:
                    self.rejection_logger.log(
                        clip_id=self._current_clip_id, timestamp=timestamp, reason='parked_curb_filter',
                        bbox=[fx1, fy1, fx2, fy2], vehicle_class=cls_id, confidence=float(conf)
                    )
                continue
            if self.is_point_in_gravel(((fx1 + fx2) // 2, fy2)):
                valid_boxes.append([rx1, ry1, rx2, ry2])
                valid_confs.append(conf)
                valid_classes.append(3 if cls_id in (0, 1) else cls_id)
            elif log_rejections:
                self.rejection_logger.log(
                    clip_id=self._current_clip_id, timestamp=timestamp, reason='outside_gravel_polygon',
                    bbox=[fx1, fy1, fx2, fy2], vehicle_class=cls_id, confidence=float(conf)
                )

        post_extra = time.perf_counter() - t_post_start
        self.profiler.record_stage_time('vehicle_postprocess', timing.get('postprocess', 0.0) + post_extra)
        if valid_boxes:
            self.motion_gate.notify_vehicle_detected(timestamp)
        return valid_boxes, valid_confs, valid_classes
