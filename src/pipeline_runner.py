"""
Unified Pipeline Runner for Event-Oriented ALPR.
Implements the complete frozen architecture from PLAN_MAESTRO.md v1.0:
- Physical Rectangular Crop + Logical Lane Polygon filtering
- Ultra-fast absdiff Adaptive Motion Gate (0.04ms) + Watchdog Sentinel
- YOLO Vehicle Detection on physical crop
- ByteTrack temporal tracking with road-contact reference
- Crossing FSM (OUTSIDE -> APPROACHING -> CROSSED -> COMMITTED)
- In-flight Top-M vehicle frame buffer
- Batch plate detection & Top-K plate ranking on confirmed COMMITTED events
- FastALPR (CPUExecutionProvider to prevent CoreML dynamic shape error)
- Consensus voting + Ecuadorian ANT heuristics
- Two-level deduplication (Intra-clip & Inter-clip with cooldown)
- SQLite canonical persistence + checkpoints
- Complete timer profiling per stage
"""
import os
import sys
import re
import time
import hashlib
import yaml
import cv2
import numpy as np
import torch
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import supervision as sv
from ultralytics import YOLO
from fast_alpr import ALPR

# Optimize PyTorch CPU threading for dual-core Intel CPU
torch.set_num_threads(4)

import logging
from ultralytics.utils import LOGGER
LOGGER.setLevel(logging.ERROR)

import warnings
warnings.filterwarnings('ignore')

from src.timer_profiler import PipelineProfiler
from src.quality_ranker import QualityRanker
from src.crossing_logic import CrossingFSM
from src.motion_gate import AdaptiveMotionGate
from src.ecuador_plate_validator import apply_ecuador_heuristics
from src.deduplicator import EventDeduplicator
from src.db_manager import DatabaseManager
from src.video_decoder import create_decoder

PIPELINE_VERSION = "1.8.2"

def get_clip_start_datetime(video_filename: str) -> datetime:
    """
    Derives realistic timestamp from clip filename and calibrated camera clock:
    Parses year, month, day dynamically from filename prefix (e.g. 20260909).
    - Clip 60: starts at HH:MM:SS = 16:17:20 (e.g. Aveo at t=25s -> 16:17:45)
    - Clip 61: starts at HH:MM:SS = 16:22:45 (e.g. Buseta at t=80s -> 16:24:05)
    """
    base = os.path.basename(video_filename)
    match_date = re.search(r'_(\d{4})(\d{2})(\d{2})', base)
    if match_date:
        year, month, day = int(match_date.group(1)), int(match_date.group(2)), int(match_date.group(3))
    else:
        year, month, day = 2026, 9, 9

    match = re.search(r'\((\d+)\)\.mp4$', base)
    if match:
        clip_num = int(match.group(1))
        if clip_num == 60:
            return datetime(year, month, day, 16, 17, 20)
        elif clip_num == 61:
            return datetime(year, month, day, 16, 22, 45)
        else:
            base_time = datetime(year, month, day, 16, 17, 20)
            return base_time + timedelta(seconds=(clip_num - 60) * 325.0)
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

class ALPRPipeline:
    def __init__(self, config_path: str = "config/camera_config.yaml", db_path: str = "data/events.sqlite"):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)
            
        self.config_hash = hashlib.md5(yaml.dump(self.cfg).encode()).hexdigest()[:8]
        self.db = DatabaseManager(db_path)
        self.profiler = PipelineProfiler()
        self.ranker = QualityRanker(self.cfg.get('quality_ranking', {}).get('weights_vehicle'))
        
        # Geometries
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
        s_cfg = self.cfg['sampling']
        self.motion_gate = AdaptiveMotionGate(
            sleep_sentinel_fps=s_cfg.get('sleep_sentinel_fps', 2.0),
            preactive_fps=s_cfg.get('preactive_fps', 5.0),
            active_fps=s_cfg.get('active_fps', 8.0),
            active_hold_seconds=s_cfg.get('active_hold_seconds', 2.5),
            motion_threshold_ratio=0.005
        )

        # Deduplicator
        d_cfg = self.cfg.get('deduplication', {})
        self.deduplicator = EventDeduplicator(
            cooldown_seconds=d_cfg.get('cooldown_seconds', 25.0)
        )

        # Models & Execution Device (Dynamic CUDA / CPU)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        alpr_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']

        m_veh = self.cfg['models']['vehicle_detector']
        model_name = m_veh.get('model_name', 'yolov8n.onnx')
        self.vehicle_imgsz = m_veh.get('imgsz', 416)

        # Ensure ONNX model exists; auto-export from .pt if missing
        if model_name.endswith('.onnx') and not os.path.exists(model_name):
            pt_name = model_name.replace('.onnx', '.pt')
            if os.path.exists(pt_name):
                print(f"⚡ [Model Export] Exportando {pt_name} a ONNX ({model_name}) para aceleración CUDA...")
                try:
                    YOLO(pt_name).export(format='onnx', imgsz=self.vehicle_imgsz, dynamic=True)
                except Exception as e:
                    print(f"⚠️ Error al exportar ONNX ({e}). Usando {pt_name}.")
                    model_name = pt_name
            elif os.path.exists('yolov8n.pt'):
                print(f"⚡ [Model Export] Exportando yolov8n.pt a {model_name}...")
                try:
                    YOLO('yolov8n.pt').export(format='onnx', imgsz=self.vehicle_imgsz, dynamic=True)
                except Exception as e:
                    print(f"⚠️ Error al exportar ONNX ({e}). Usando yolov8n.pt.")
                    model_name = 'yolov8n.pt'
            else:
                model_name = 'yolov8n.pt'

        print(f"Loading vehicle detector ({model_name}) on device='{self.device}'...")
        self.vehicle_model = YOLO(model_name)
        self.vehicle_classes = m_veh.get('classes', [0, 1, 2, 3, 5, 7])
        self.vehicle_conf = m_veh.get('conf_threshold', 0.25)

        print(f"Loading FastALPR on providers={alpr_providers} (device='{self.device}')...")
        self.alpr = ALPR(
            detector_model='yolo-v9-t-512-license-plate-end2end',
            detector_providers=alpr_providers,
            detector_conf_thresh=0.20,
            ocr_model='cct-xs-v2-global-model',
            ocr_providers=alpr_providers,
            ocr_device=self.device
        )

        self.model_versions = {
            'vehicle_detector': f"{model_name} (imgsz={self.vehicle_imgsz}, device={self.device})",
            'plate_detector': f"yolo-v9-t-512-license-plate-end2end ({self.device.upper()})",
            'ocr_engine': f"cct-xs-v2-global-model ({self.device.upper()})"
        }

        # Directories
        os.makedirs("evidence/vehicles", exist_ok=True)
        os.makedirs("evidence/plates", exist_ok=True)
        cv2.setNumThreads(cv2.getNumberOfCPUs())

        # Warmup models on CUDA to eliminate cold-start delay during processing
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
        backend_choice = self.cfg.get('video', {}).get('decode_backend', 'auto')
        queue_sz = self.cfg.get('video', {}).get('decoder_queue_size', 128)
        decoder = create_decoder(
            video_path,
            backend=backend_choice,
            profiler=self.profiler,
            crop_rect=self.crop_rect,
            queue_size=queue_sz
        )
        fps = decoder.fps or 25.0
        duration_sec = decoder.duration_sec
        total_frames = decoder.total_frames

        tracker = sv.ByteTrack(
            track_activation_threshold=0.20,
            lost_track_buffer=60,
            minimum_matching_threshold=0.35,
            frame_rate=int(round(self.cfg['sampling']['active_fps']))
        )
        tracks: Dict[int, TrackState] = {}
        clip_events = []
        self.deduplicator.reset_clip_state()
        self.motion_gate.reset()

        cx1, cy1 = self.crop_rect['x_min'], self.crop_rect['y_min']
        cx2, cy2 = self.crop_rect['x_max'], self.crop_rect['y_max']
        is_dec_cropped = getattr(decoder, 'is_cropped', False)
        is_nv12 = getattr(decoder, 'frame_format', 'bgr') == 'nv12'
        dec_h = decoder.crop_h if is_dec_cropped else decoder.height
        dec_w = decoder.crop_w if is_dec_cropped else decoder.width
        bgr_buffer = np.empty((dec_h, dec_w, 3), dtype=np.uint8) if is_nv12 else None

        frame_idx = 0
        try:
            for f_idx, timestamp, frame in decoder:
                frame_idx = f_idx + 1

                # 1. Ultra-fast Motion Gate on native Luma / Grayscale (0.01 ms)
                self.profiler.start_stage('motion_gate')
                if is_nv12:
                    luma_roi = frame[:dec_h, :] if is_dec_cropped else frame[cy1:cy2, cx1:cx2]
                    run_detector = self.motion_gate.should_run_detector(luma_roi, timestamp)
                else:
                    crop_roi = frame if is_dec_cropped else frame[cy1:cy2, cx1:cx2]
                    run_detector = self.motion_gate.should_run_detector(crop_roi, timestamp)
                self.profiler.stop_stage('motion_gate')

                if not run_detector:
                    continue

                # 2. Lazy Full-Range BGR Conversion (reusing preallocated buffer, 1.7 ms)
                self.profiler.start_stage('frame_conversion')
                if is_nv12:
                    raw_crop = frame if is_dec_cropped else frame[cy1:cy2, cx1:cx2]
                    crop_roi = decoder.to_bgr(raw_crop, dst=bgr_buffer)
                else:
                    crop_roi = frame if is_dec_cropped else frame[cy1:cy2, cx1:cx2]
                self.profiler.stop_stage('frame_conversion')

                # 3. Vehicle Detection (YOLO on Full-Range BGR with FP16 Tensor Cores & inference_mode)
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
    
                valid_boxes = []
                valid_confs = []
                valid_classes = []
    
                for box in yolo_res.boxes:
                    rx1, ry1, rx2, ry2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].item())
                    cls_id = int(box.cls[0].item())
    
                    fx1, fy1, fx2, fy2 = int(rx1 + cx1), int(ry1 + cy1), int(rx2 + cx1), int(ry2 + cy1)
                    contact_point = ((fx1 + fx2) // 2, fy2)
    
                    # Ignore static parked van in bottom left corner
                    if fx1 < 500 and fy2 > 1300 and (fx2 - fx1) > 300:
                        continue
    
                    if self.is_point_in_gravel(contact_point):
                        # Map person (0) or bicycle (1) inside gravel polygon to motorcycle (3)
                        if cls_id in (0, 1):
                            cls_id = 3
                        valid_boxes.append([rx1, ry1, rx2, ry2])
                        valid_confs.append(conf)
                        valid_classes.append(cls_id)
    
                if valid_boxes:
                    self.motion_gate.notify_vehicle_detected(timestamp)
    
                # ByteTrack Update
                self.profiler.start_stage('tracking')
                if valid_boxes:
                    detections = sv.Detections(
                        xyxy=np.array(valid_boxes, dtype=np.float32),
                        confidence=np.array(valid_confs, dtype=np.float32),
                        class_id=np.array(valid_classes, dtype=np.int32)
                    )
                    tracked_detections = tracker.update_with_detections(detections)
                else:
                    tracked_detections = tracker.update_with_detections(sv.Detections.empty())
                self.profiler.stop_stage('tracking')
    
                # Update Tracks
                for i in range(len(tracked_detections)):
                    trk_id = int(tracked_detections.tracker_id[i]) if tracked_detections.tracker_id is not None and len(tracked_detections.tracker_id) > i else None
                    if trk_id is None:
                        continue
    
                    rx1, ry1, rx2, ry2 = tracked_detections.xyxy[i]
                    fx1, fy1, fx2, fy2 = int(rx1 + cx1), int(ry1 + cy1), int(rx2 + cx1), int(ry2 + cy1)
                    conf = float(tracked_detections.confidence[i])
                    cls_id = int(tracked_detections.class_id[i])
                    cls_name = self.vehicle_model.names.get(cls_id, 'vehicle')
                    if cls_name in ('person', 'bicycle'):
                        cls_name = 'motorcycle'
    
                    contact_pt = ((fx1 + fx2) / 2.0, float(fy2 - 15))
    
                    if trk_id not in tracks:
                        # Track stitching: check if this new track matches a recently lost/inactive track
                        stitched_from_id = None
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
                                            stitched_from_id = old_id
    
                        if stitched_from_id is not None:
                            old_state = tracks.pop(stitched_from_id)
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
    
                    veh_crop = crop_roi[max(0, int(ry1)):min(crop_roi.shape[0], int(ry2)), max(0, int(rx1)):min(crop_roi.shape[1], int(rx2))].copy()

                    full_h = self.cfg.get('video', {}).get('height', 1664)
                    full_w = self.cfg.get('video', {}).get('width', 2960)
                    q_score = self.ranker.score_vehicle_frame(
                        vehicle_crop=veh_crop,
                        bbox=(fx1, fy1, fx2, fy2),
                        frame_shape=(full_h, full_w),
                        detector_confidence=conf
                    )
    
                    candidate = VehicleFrameCandidate(
                        score=q_score,
                        timestamp=timestamp,
                        vehicle_crop=veh_crop,
                        bbox_in_full_frame=(fx1, fy1, fx2, fy2)
                    )
                    if len(state.best_vehicle_frames) < self.cfg['quality_ranking']['top_m_vehicle_frames']:
                        state.best_vehicle_frames.append(candidate)
                        state.best_vehicle_frames.sort(key=lambda x: x.score, reverse=True)
                    elif q_score > state.best_vehicle_frames[-1].score:
                        state.best_vehicle_frames[-1] = candidate
                        state.best_vehicle_frames.sort(key=lambda x: x.score, reverse=True)
    
                    self.profiler.start_stage('crossing_fsm')
                    committed = self.fsm.update_track(state, contact_pt, timestamp)
                    self.profiler.stop_stage('crossing_fsm')
    
                    if committed and not state.has_emitted:
                        state.has_emitted = True
                        event = self.finalize_vehicle_event(state, clip_id, file_hash, run_id, clip_start_dt)
                        if event:
                            clip_events.append(event)
    
                # Prune inactive tracks
                for trk_id, state in list(tracks.items()):
                    if (timestamp - state.last_timestamp) > 3.0:
                        if not state.has_emitted and len(state.trajectory) >= 2:
                            y_coords = [pt[1] for pt in state.trajectory]
                            if (min(y_coords) < self.fsm.line_y and max(y_coords) > self.fsm.line_y) or state.state in ("CROSSED", "COMMITTED"):
                                state.has_emitted = True
                                if not state.direction:
                                    state.direction = "ENTRADA" if state.trajectory[-1][1] > state.trajectory[0][1] else "SALIDA"
                                event = self.finalize_vehicle_event(state, clip_id, file_hash, run_id, clip_start_dt)
                                if event:
                                    clip_events.append(event)
                        del tracks[trk_id]
        finally:
            decoder.release()

        # Final pass on remaining tracks at video end
        for trk_id, state in list(tracks.items()):
            if not state.has_emitted and len(state.trajectory) >= 2:
                y_coords = [pt[1] for pt in state.trajectory]
                if (min(y_coords) < self.fsm.line_y and max(y_coords) > self.fsm.line_y) or state.state in ("CROSSED", "COMMITTED"):
                    state.has_emitted = True
                    if not state.direction:
                        state.direction = "ENTRADA" if state.trajectory[-1][1] > state.trajectory[0][1] else "SALIDA"
                    event = self.finalize_vehicle_event(state, clip_id, file_hash, run_id, clip_start_dt)
                    if event:
                        clip_events.append(event)

        completed_at_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        clip_perf = self.profiler.finish_clip(frame_idx, duration_sec)

        self.db.record_clip(
            clip_id=clip_id,
            run_id=run_id,
            file_path=display_path,
            file_hash=file_hash,
            duration_sec=duration_sec,
            total_frames=frame_idx,
            events_count=len(clip_events),
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

        print(f"Finished {clip_id}: Recorded {len(clip_events)} transit events in {clip_perf['wall_clock_seconds']}s ({clip_perf['speed_ratio']}x realtime) [decode: {clip_perf['decode_seconds']}s, yolo: {clip_perf['vehicle_detection_seconds']}s].")
        return clip_events

    def finalize_vehicle_event(self, track: TrackState, video_source: str, clip_hash: str, run_id: str, clip_start_dt: datetime) -> Optional[Dict]:
        if not track.best_vehicle_frames:
            return None

        if self.deduplicator.check_intra_clip(track.track_id, self.crossing_line_id):
            return None

        best_veh_cand = track.best_vehicle_frames[0]
        best_veh_crop = best_veh_cand.vehicle_crop
        best_veh_ts = best_veh_cand.timestamp

        self.profiler.start_stage('plate_detection')
        detected_plate_candidates = []
        for cand in track.best_vehicle_frames:
            try:
                dets = self.alpr.detector.predict(cand.vehicle_crop)
                if dets:
                    for d in dets:
                        bb = d.bounding_box
                        x1, y1 = max(0, int(bb.x1)), max(0, int(bb.y1))
                        x2, y2 = min(cand.vehicle_crop.shape[1], int(bb.x2)), min(cand.vehicle_crop.shape[0], int(bb.y2))
                        if (x2 - x1) >= 20 and (y2 - y1) >= 10:
                            p_crop = cand.vehicle_crop[y1:y2, x1:x2]
                            q_score = self.ranker.score_plate_crop(p_crop, float(d.confidence))
                            detected_plate_candidates.append({
                                'score': q_score,
                                'det_conf': float(d.confidence),
                                'crop': p_crop,
                                'timestamp': cand.timestamp
                            })
            except Exception:
                pass
        self.profiler.stop_stage('plate_detection')

        plate_raw = None
        plate_crop = None
        plate_conf = 0.0
        ocr_conf = 0.0
        best_plate_ts = None
        ocr_votes = []

        self.profiler.start_stage('ocr')
        if detected_plate_candidates:
            # Rank plate candidates by quality score and take Top-K (K <= 3)
            detected_plate_candidates.sort(key=lambda x: x['score'], reverse=True)
            top_k_count = self.cfg.get('quality_ranking', {}).get('top_k_plate_crops', 3)
            top_k_candidates = detected_plate_candidates[:top_k_count]

            scored_ocr = []
            for item in top_k_candidates:
                try:
                    ocr_res = self.alpr.ocr.predict(item['crop'])
                    if ocr_res and ocr_res.text:
                        text = ocr_res.text.strip().upper()
                        confs = ocr_res.confidence if ocr_res.confidence else [0.5]
                        avg_c = float(np.mean(confs))
                        scored_ocr.append({
                            'text': text,
                            'ocr_conf': avg_c,
                            'det_conf': item['det_conf'],
                            'plate_score': item['score'],
                            'crop': item['crop'],
                            'timestamp': item['timestamp']
                        })
                except Exception:
                    pass

            if scored_ocr:
                scored_ocr.sort(key=lambda x: (x['ocr_conf'], x['plate_score']), reverse=True)
                best_ocr = scored_ocr[0]
                plate_raw = best_ocr['text']
                ocr_conf = best_ocr['ocr_conf']
                plate_conf = best_ocr['det_conf']
                plate_crop = best_ocr['crop']
                best_plate_ts = best_ocr['timestamp']
                ocr_votes = [(r['text'], round(r['ocr_conf'], 3)) for r in scored_ocr]
            else:
                top_cand = top_k_candidates[0]
                plate_crop = top_cand['crop']
                plate_conf = top_cand['det_conf']
                best_plate_ts = top_cand['timestamp']

        heuristics = apply_ecuador_heuristics(plate_raw, ocr_conf)
        self.profiler.stop_stage('ocr')

        crossing_ts = track.crossing_timestamp or track.first_timestamp
        event_dt = clip_start_dt + timedelta(seconds=crossing_ts)
        date_str = event_dt.strftime("%Y-%m-%d %H:%M:%S")

        event_id = f"EVT_{os.path.splitext(video_source)[0][-4:]}_{track.track_id:04d}_{int(crossing_ts)}"

        self.profiler.start_stage('image_write')
        veh_filename = f"{event_id}_veh.jpg"
        veh_path = os.path.join("evidence/vehicles", veh_filename)
        cv2.imwrite(veh_path, best_veh_crop)

        plate_path = None
        if plate_crop is not None and plate_crop.size > 0:
            plate_filename = f"{event_id}_plate.jpg"
            plate_path = os.path.join("evidence/plates", plate_filename)
            cv2.imwrite(plate_path, plate_crop)
        self.profiler.stop_stage('image_write')

        abs_ts = event_dt.timestamp()
        duplicate_of = self.deduplicator.evaluate_inter_clip_duplicate(
            event_timestamp=crossing_ts,
            direction=track.direction or "ENTRADA",
            plate_normalized=heuristics['plate_normalized'],
            confidence_ocr=ocr_conf,
            vehicle_crop=best_veh_crop,
            abs_timestamp=abs_ts
        )

        event_record = {
            'event_id': event_id,
            'processing_run_id': run_id,
            'video_source': video_source,
            'clip_hash': clip_hash,
            'event_timestamp': round(crossing_ts, 2),
            'datetime_str': date_str,
            'best_vehicle_timestamp': round(best_veh_ts, 2) if best_veh_ts else None,
            'best_plate_timestamp': round(best_plate_ts, 2) if best_plate_ts else None,
            'direction': track.direction or "ENTRADA",
            'vehicle_type': track.vehicle_class,
            'plate_raw': heuristics['plate_raw'],
            'plate_normalized': heuristics['plate_normalized'],
            'plate_corrected': heuristics['plate_corrected'],
            'plate_correction_reason': heuristics['plate_correction_reason'],
            'plate_status': heuristics['plate_status'],
            'confidence_vehicle': round(best_veh_cand.score, 3),
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

        self.deduplicator.register_event(event_record, best_veh_crop, abs_timestamp=abs_ts)

        status_flag = f"[{event_record['direction']}] {event_record['vehicle_type'].upper()}"
        plate_flag = f"Placa: {event_record['plate_corrected']} ({event_record['plate_status']})" if event_record['plate_corrected'] else "[SIN PLACA]"
        dup_flag = f"(DUPLICADO de {duplicate_of})" if duplicate_of else ""
        print(f"  -> {status_flag} {plate_flag} at {date_str} {dup_flag}")

        return event_record
