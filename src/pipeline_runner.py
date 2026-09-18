"""
Pipeline Runner for Lightweight Event-Oriented ALPR Pipeline v1.9.0.
Coordinates decoupled decoding, adaptive motion gating, vehicle detection & tracking,
virtual line crossing FSM, quality ranking, and FastALPR plate recognition.
"""
import os
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any

import yaml
import numpy as np
import cv2
import torch
from ultralytics import YOLO
import supervision as sv
from fast_alpr import ALPR

from src.db_manager import DatabaseManager
from src.timer_profiler import PipelineProfiler
from src.crossing_logic import CrossingFSM
from src.motion_gate import AdaptiveMotionGate
from src.quality_ranker import QualityRanker
from src.deduplicator import EventDeduplicator
from src.ecuador_plate_validator import apply_ecuador_heuristics
from src.model_resolver import resolve_vehicle_model
from src.video_decoder import create_decoder
from src.pipeline_types import (
    VehicleFrameCandidate,
    TrackState,
    get_clip_start_datetime,
    compute_file_hash
)
from src.plate_processor import PlateProcessor
from src.tracking_manager import update_tracks_and_fsm, prune_inactive_tracks

PIPELINE_VERSION = "1.9.0"

class ALPRPipeline:
    def __init__(
        self,
        config_path: str = "config/camera_config.yaml",
        db_path: str = "data/events.sqlite",
        model_name_override: Optional[str] = None
    ):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)
            
        if model_name_override:
            self.cfg.setdefault('models', {}).setdefault('vehicle_detector', {})['model_name'] = model_name_override
            
        self.config_hash = hashlib.md5(yaml.dump(self.cfg).encode()).hexdigest()[:8]
        self.db = DatabaseManager(db_path)
        self.profiler = PipelineProfiler()
        self.ranker = QualityRanker(self.cfg.get('quality_ranking', {}).get('weights_vehicle'))
        
        # Geometries & Crossing FSM
        self.crop_rect = self.cfg['roi']['crop_rect']
        self.poly_gravel = np.array(self.cfg['roi']['polygon_gravel'], dtype=np.int32)
        line_cfg = self.cfg['roi']['crossing_line']
        self.crossing_line_id = line_cfg['line_id']
        self.fsm = CrossingFSM(
            line_p1=line_cfg['p1'],
            line_p2=line_cfg['p2'],
            dir_increasing_y=line_cfg.get('direction_increasing_y', 'ENTRADA'),
            dir_decreasing_y=line_cfg.get('direction_decreasing_y', 'SALIDA')
        )

        # Sampling & Motion Gate
        s_cfg = self.cfg.get('sampling', {})
        self.motion_gate = AdaptiveMotionGate(
            sleep_sentinel_fps=s_cfg.get('sleep_sentinel_fps', 2.0),
            preactive_fps=s_cfg.get('preactive_fps', 5.0),
            active_fps=s_cfg.get('active_fps', 8.0),
            active_hold_seconds=s_cfg.get('active_hold_seconds', 2.5),
            motion_threshold_ratio=s_cfg.get('motion_threshold_ratio', 0.005)
        )

        # Deduplicator
        d_cfg = self.cfg.get('deduplication', {})
        self.deduplicator = EventDeduplicator(
            cooldown_seconds=d_cfg.get('cooldown_seconds', 25.0)
        )

        # Execution Device & Vehicle Model Selection
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        alpr_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']

        m_veh = self.cfg.get('models', {}).get('vehicle_detector', {})
        req_model = m_veh.get('model_name', 'yolov8n.onnx')
        self.vehicle_imgsz = m_veh.get('imgsz', 416)
        model_name = resolve_vehicle_model(req_model, imgsz=self.vehicle_imgsz)

        print(f"Loading vehicle detector ({model_name}) on device='{self.device}'...")
        self.vehicle_model = YOLO(model_name)
        self.vehicle_classes = m_veh.get('classes', [0, 1, 2, 3, 5, 7])
        self.vehicle_conf = m_veh.get('conf_threshold', 0.25)

        # FastALPR Model Loading
        m_plate = self.cfg.get('models', {}).get('plate_detector', {})
        plate_model = m_plate.get('model_name', 'yolo-v9-t-512-license-plate-end2end')
        plate_conf = m_plate.get('conf_threshold', 0.20)

        m_ocr = self.cfg.get('models', {}).get('ocr', {})
        ocr_model = m_ocr.get('model', 'cct-xs-v2-global-model')

        print(f"Loading FastALPR on providers={alpr_providers} (device='{self.device}')...")
        self.alpr = ALPR(
            detector_model=plate_model,
            detector_providers=alpr_providers,
            detector_conf_thresh=plate_conf,
            ocr_model=ocr_model,
            ocr_providers=alpr_providers,
            ocr_device=self.device
        )

        self.plate_processor = PlateProcessor(
            alpr=self.alpr,
            ranker=self.ranker,
            profiler=self.profiler,
            top_k_crops=self.cfg.get('quality_ranking', {}).get('top_k_plate_crops', 7),
            vehicle_evidence_dir=self.cfg.get('storage', {}).get('vehicles_dir', 'evidence/vehicles'),
            plate_evidence_dir=self.cfg.get('storage', {}).get('plates_dir', 'evidence/plates'),
            save_manifest=self.cfg.get('quality_ranking', {}).get('save_candidate_manifest', False),
            province_prior=m_ocr.get('province_prior', {})
        )

        self.model_versions = {
            'vehicle_detector': f"{model_name} (imgsz={self.vehicle_imgsz}, device={self.device})",
            'plate_detector': f"{plate_model} ({self.device.upper()})",
            'ocr_engine': f"{ocr_model} ({self.device.upper()})"
        }

        cv2.setNumThreads(cv2.getNumberOfCPUs())

        if self.device == 'cuda':
            torch.backends.cudnn.benchmark = True
            try:
                dummy_frame = np.zeros((self.vehicle_imgsz, self.vehicle_imgsz, 3), dtype=np.uint8)
                self.vehicle_model(dummy_frame, imgsz=self.vehicle_imgsz, verbose=False, device=self.device)
                dummy_crop = np.zeros((120, 240, 3), dtype=np.uint8)
                self.alpr.predict(dummy_crop)
            except Exception:
                pass

    def is_point_in_gravel(self, point: Tuple[int, int]) -> bool:
        return cv2.pointPolygonTest(self.poly_gravel, (float(point[0]), float(point[1])), False) >= 0

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

    def _detect_and_filter_vehicles(self, crop_roi: np.ndarray, cx1: int, cy1: int, timestamp: float):
        self.profiler.start_stage('vehicle_detection')
        with torch.inference_mode():
            yolo_res = self.vehicle_model(
                crop_roi,
                imgsz=self.vehicle_imgsz,
                verbose=False,
                conf=self.vehicle_conf,
                classes=self.vehicle_classes,
                device=self.device
            )[0]
        self.profiler.stop_stage('vehicle_detection')

        valid_boxes, valid_confs, valid_classes = [], [], []
        for box in yolo_res.boxes:
            rx1, ry1, rx2, ry2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].item())
            cls_id = int(box.cls[0].item())
            fx1, fy1, fx2, fy2 = int(rx1 + cx1), int(ry1 + cy1), int(rx2 + cx1), int(ry2 + cy1)
            contact_point = ((fx1 + fx2) // 2, fy2)

            if fx1 < 500 and fy2 > 1300 and (fx2 - fx1) > 300:
                continue
            if self.is_point_in_gravel(contact_point):
                if cls_id in (0, 1):
                    cls_id = 3
                valid_boxes.append([rx1, ry1, rx2, ry2])
                valid_confs.append(conf)
                valid_classes.append(cls_id)

        if valid_boxes:
            self.motion_gate.notify_vehicle_detected(timestamp)
        return valid_boxes, valid_confs, valid_classes

    def _record_clip_summary(self, clip_id: str, file_hash: str, display_path: str, duration_sec: float, frame_idx: int, events_count: int, started_at_str: str, run_id: str):
        completed_at_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        clip_perf = self.profiler.finish_clip(frame_idx, duration_sec)
        self.db.record_clip(
            clip_id=clip_id,
            run_id=run_id,
            file_path=display_path,
            file_hash=file_hash,
            duration_sec=duration_sec,
            total_frames=frame_idx,
            events_count=events_count,
            status="COMPLETED",
            started_at=started_at_str,
            completed_at=completed_at_str,
            wall_clock_seconds=clip_perf['wall_clock_seconds'],
            speed_ratio=clip_perf['speed_ratio'],
            decode_seconds=clip_perf['decode_seconds'],
            vehicle_detection_seconds=clip_perf['vehicle_detection_seconds'],
            plate_detection_seconds=clip_perf['plate_detection_seconds'],
            ocr_seconds=clip_perf['ocr_seconds']
        )
        print(f"Finished {clip_id}: Recorded {events_count} transit events in {clip_perf['wall_clock_seconds']}s ({clip_perf['speed_ratio']}x realtime) [decode: {clip_perf['decode_seconds']}s, yolo: {clip_perf['vehicle_detection_seconds']}s].")

    def process_video_file(self, video_path: str, run_id: str, force_reprocess: bool = True, original_path: Optional[str] = None) -> List[Dict]:
        display_path = original_path if original_path else video_path
        clip_id = os.path.basename(display_path)
        clip_start_dt = get_clip_start_datetime(display_path)
        started_at_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting video processing: {clip_id} (Ref Time: {clip_start_dt.strftime('%Y-%m-%d %H:%M:%S')})")
        
        file_hash = compute_file_hash(video_path)
        if not force_reprocess and self.db.is_clip_completed(clip_id, file_hash):
            print(f"Checkpoint found: {clip_id} was already completed. Skipping without re-read.")
            return []

        self.profiler.start_clip()
        decoder, bgr_buffer, is_nv12, is_dec_cropped, dec_h, dec_w, cx1, cy1, cx2, cy2 = self._init_clip_decoder(video_path)
        duration_sec = decoder.duration_sec

        tracker = sv.ByteTrack(
            track_activation_threshold=0.20,
            lost_track_buffer=60,
            minimum_matching_threshold=0.35,
            frame_rate=int(round(self.cfg.get('sampling', {}).get('active_fps', 6.0)))
        )
        tracks: Dict[int, TrackState] = {}
        clip_events: List[Dict] = []
        self.deduplicator.reset_clip_state()
        self.motion_gate.reset()

        frame_idx = 0
        full_shape = (self.cfg.get('video', {}).get('height', 1664), self.cfg.get('video', {}).get('width', 2960))
        top_m = self.cfg.get('quality_ranking', {}).get('top_m_vehicle_frames', 5)

        try:
            for f_idx, timestamp, frame in decoder:
                frame_idx = f_idx + 1

                self.profiler.start_stage('motion_gate')
                # NV12 frames are (h*3/2, w): the motion gate only needs the luma plane,
                # but the BGR conversion needs the full Y+UV buffer.
                raw_roi = frame if is_dec_cropped else frame[cy1:cy2, cx1:cx2]
                gate_roi = frame[:dec_h, :] if (is_nv12 and is_dec_cropped) else raw_roi
                run_detector = self.motion_gate.should_run_detector(gate_roi, timestamp)
                self.profiler.stop_stage('motion_gate')

                if not run_detector:
                    continue

                self.profiler.start_stage('frame_conversion')
                crop_roi = decoder.to_bgr(raw_roi, dst=bgr_buffer) if is_nv12 else raw_roi
                self.profiler.stop_stage('frame_conversion')

                valid_boxes, valid_confs, valid_classes = self._detect_and_filter_vehicles(crop_roi, cx1, cy1, timestamp)

                self.profiler.start_stage('tracking')
                if valid_boxes:
                    detections = sv.Detections(
                        xyxy=np.array(valid_boxes, dtype=np.float32),
                        confidence=np.array(valid_confs, dtype=np.float32),
                        class_id=np.array(valid_classes, dtype=np.int32)
                    )
                    tracked_dets = tracker.update_with_detections(detections)
                else:
                    tracked_dets = tracker.update_with_detections(sv.Detections.empty())
                self.profiler.stop_stage('tracking')

                # Update Tracks & Virtual Line FSM
                committed_tracks = update_tracks_and_fsm(
                    crop_roi, tracked_dets, tracks, cx1, cy1, timestamp,
                    self.vehicle_model.names, self.ranker, self.fsm, top_m, full_shape, self.profiler
                )
                for st in committed_tracks:
                    evt = self.finalize_vehicle_event(st, clip_id, file_hash, run_id, clip_start_dt)
                    if evt:
                        clip_events.append(evt)

                # Prune inactive tracks
                expired_tracks = prune_inactive_tracks(tracks, timestamp, self.fsm.line_y, max_idle=3.0)
                for st in expired_tracks:
                    evt = self.finalize_vehicle_event(st, clip_id, file_hash, run_id, clip_start_dt)
                    if evt:
                        clip_events.append(evt)
        finally:
            decoder.release()

        # Final pass at clip end
        remaining_tracks = prune_inactive_tracks(tracks, 1e9, self.fsm.line_y, force_all=True)
        for st in remaining_tracks:
            evt = self.finalize_vehicle_event(st, clip_id, file_hash, run_id, clip_start_dt)
            if evt:
                clip_events.append(evt)

        self._record_clip_summary(clip_id, file_hash, display_path, duration_sec, frame_idx, len(clip_events), started_at_str, run_id)
        if self.plate_processor.save_manifest:
            self.plate_processor.write_manifest()
        return clip_events

    def finalize_vehicle_event(self, track: TrackState, video_source: str, clip_hash: str, run_id: str, clip_start_dt: datetime) -> Optional[Dict]:
        if not track.best_vehicle_frames or self.deduplicator.check_intra_clip(track.track_id, self.crossing_line_id):
            return None

        best_cand = track.best_vehicle_frames[0]
        crossing_ts = track.crossing_timestamp or track.first_timestamp
        event_dt = clip_start_dt + timedelta(seconds=crossing_ts)
        event_id = f"EVT_{os.path.splitext(video_source)[0][-4:]}_{track.track_id:04d}_{int(crossing_ts)}"

        candidates = self.plate_processor.detect_plate_candidates(track.best_vehicle_frames, event_id=event_id)
        plate_raw, plate_crop, plate_conf, ocr_conf, best_plate_ts, ocr_votes = self.plate_processor.recognize_plate_candidates(
            candidates, event_id=event_id
        )

        needs_review = getattr(self.plate_processor, 'last_needs_manual_review', False)
        heuristics = apply_ecuador_heuristics(plate_raw, ocr_conf, needs_manual_review=needs_review)
        veh_path, plate_path = self.plate_processor.save_evidence(event_id, best_cand.vehicle_crop, plate_crop)

        abs_ts = event_dt.timestamp()
        duplicate_of = self.deduplicator.evaluate_inter_clip_duplicate(
            event_timestamp=crossing_ts,
            direction=track.direction or "ENTRADA",
            plate_normalized=heuristics['plate_normalized'],
            confidence_ocr=ocr_conf,
            vehicle_crop=best_cand.vehicle_crop,
            abs_timestamp=abs_ts
        )

        event_record = {
            'event_id': event_id,
            'processing_run_id': run_id,
            'video_source': video_source,
            'clip_hash': clip_hash,
            'event_timestamp': round(crossing_ts, 2),
            'datetime_str': event_dt.strftime("%Y-%m-%d %H:%M:%S"),
            'best_vehicle_timestamp': round(best_cand.timestamp, 2) if best_cand else None,
            'best_plate_timestamp': round(best_plate_ts, 2) if best_plate_ts else None,
            'direction': track.direction or "ENTRADA",
            'vehicle_type': track.vehicle_class,
            'plate_raw': heuristics['plate_raw'],
            'plate_normalized': heuristics['plate_normalized'],
            'plate_corrected': heuristics['plate_corrected'],
            'plate_correction_reason': heuristics['plate_correction_reason'],
            'plate_status': heuristics['plate_status'],
            'confidence_vehicle': round(best_cand.score, 3),
            'confidence_plate': round(plate_conf, 3),
            'confidence_ocr': round(ocr_conf, 3),
            'confidence_consensus': round(ocr_conf, 3),
            'vehicle_crop_path': veh_path,
            'plate_crop_path': plate_path,
            'ocr_votes': ocr_votes,
            'track_id': track.track_id,
            'line_id': self.crossing_line_id,
            'dedup_key': f"{heuristics['plate_normalized']}_{track.direction}",
            'duplicate_of': duplicate_of
        }

        self.profiler.start_stage('sqlite_write')
        self.db.insert_event(event_record)
        self.profiler.stop_stage('sqlite_write')

        self.deduplicator.register_event(event_record, best_cand.vehicle_crop, abs_timestamp=abs_ts)

        status_flag = f"[{event_record['direction']}] {event_record['vehicle_type'].upper()}"
        plate_flag = f"Placa: {event_record['plate_corrected']} ({event_record['plate_status']})" if event_record['plate_corrected'] else "[SIN PLACA]"
        dup_flag = f"(DUPLICADO de {duplicate_of})" if duplicate_of else ""
        print(f"  -> {status_flag} {plate_flag} at {event_record['datetime_str']} {dup_flag}")

        return event_record
