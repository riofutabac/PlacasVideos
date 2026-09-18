"""
Plate Processing and Recognition Subsystem for ALPR Pipeline.
Handles plate bounding box detection, quality ranking, OCR candidate scoring,
and evidence artifact generation.
"""
import os
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import cv2
from fast_alpr import ALPR
from src.pipeline_types import VehicleFrameCandidate
from src.quality_ranker import QualityRanker
from src.timer_profiler import PipelineProfiler

class PlateProcessor:
    def __init__(
        self,
        alpr: ALPR,
        ranker: QualityRanker,
        profiler: PipelineProfiler,
        top_k_crops: int = 3,
        vehicle_evidence_dir: str = "evidence/vehicles",
        plate_evidence_dir: str = "evidence/plates"
    ):
        self.alpr = alpr
        self.ranker = ranker
        self.profiler = profiler
        self.top_k_crops = top_k_crops
        self.vehicle_evidence_dir = vehicle_evidence_dir
        self.plate_evidence_dir = plate_evidence_dir

        os.makedirs(self.vehicle_evidence_dir, exist_ok=True)
        os.makedirs(self.plate_evidence_dir, exist_ok=True)

    def detect_plate_candidates(self, best_frames: List[VehicleFrameCandidate]) -> List[Dict]:
        """Detects license plate bounding boxes in vehicle crops and ranks their visual quality."""
        self.profiler.start_stage('plate_detection')
        candidates = []
        for cand in best_frames:
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
                            candidates.append({
                                'score': q_score,
                                'det_conf': float(d.confidence),
                                'crop': p_crop,
                                'timestamp': cand.timestamp
                            })
            except Exception:
                pass
        self.profiler.stop_stage('plate_detection')
        return candidates

    def recognize_plate_candidates(
        self,
        candidates: List[Dict],
        event_id: Optional[str] = None
    ) -> Tuple[Optional[str], Optional[np.ndarray], float, float, Optional[float], list]:
        """Executes OCR on top-K ranked plate candidates and calculates consensus votes."""
        self.profiler.start_stage('ocr')
        plate_raw, plate_crop, plate_conf, ocr_conf, best_plate_ts, ocr_votes = None, None, 0.0, 0.0, None, []

        if candidates:
            candidates.sort(key=lambda x: x['score'], reverse=True)
            top_k = candidates[:self.top_k_crops]
            scored_ocr = []
            for item in top_k:
                try:
                    ocr_res = self.alpr.ocr.predict(item['crop'])
                    if ocr_res and ocr_res.text:
                        text = ocr_res.text.strip().upper()
                        confs = ocr_res.confidence if ocr_res.confidence else [0.5]
                        avg_c = float(np.mean(confs))
                        scored_ocr.append({
                            'text': text,
                            'ocr_conf': avg_c,
                            'char_confs': confs,
                            'det_conf': item['det_conf'],
                            'plate_score': item['score'],
                            'crop': item['crop'],
                            'timestamp': item['timestamp']
                        })
                except Exception:
                    pass

            if scored_ocr:
                scored_ocr.sort(key=lambda x: (x['ocr_conf'], x['plate_score']), reverse=True)
                best = scored_ocr[0]
                plate_raw, ocr_conf, plate_conf = best['text'], best['ocr_conf'], best['det_conf']
                plate_crop, best_plate_ts = best['crop'], best['timestamp']
                ocr_votes = [(r['text'], round(r['ocr_conf'], 3)) for r in scored_ocr]

                # Debug: save all candidate crops to evidence/plates/debug/
                debug_dir = os.path.join(self.plate_evidence_dir, "debug")
                os.makedirs(debug_dir, exist_ok=True)
                evt_prefix = event_id if event_id else "candidate"
                for idx, r in enumerate(scored_ocr):
                    clean_text = "".join(c for c in r['text'] if c.isalnum())
                    cand_filename = f"{evt_prefix}_rank{idx}_{clean_text}_c{int(r['ocr_conf']*100)}.jpg"
                    cv2.imwrite(os.path.join(debug_dir, cand_filename), r['crop'])

                cand_summary = " | ".join(f"{r['text']} (c={r['ocr_conf']:.2f}, s={r['plate_score']:.2f})" for r in scored_ocr)
                print(f"    🔎 [OCR Candidates] {cand_summary}")
            else:
                top = top_k[0]
                plate_crop, plate_conf, best_plate_ts = top['crop'], top['det_conf'], top['timestamp']

        self.profiler.stop_stage('ocr')
        return plate_raw, plate_crop, plate_conf, ocr_conf, best_plate_ts, ocr_votes

    def save_evidence(
        self,
        event_id: str,
        best_veh_crop: np.ndarray,
        plate_crop: Optional[np.ndarray]
    ) -> Tuple[str, Optional[str]]:
        """Saves vehicle and plate evidence images to disk."""
        self.profiler.start_stage('image_write')
        veh_path = os.path.join(self.vehicle_evidence_dir, f"{event_id}_veh.jpg")
        cv2.imwrite(veh_path, best_veh_crop)
        plate_path = None
        if plate_crop is not None and plate_crop.size > 0:
            plate_path = os.path.join(self.plate_evidence_dir, f"{event_id}_plate.jpg")
            cv2.imwrite(plate_path, plate_crop)
        self.profiler.stop_stage('image_write')
        return veh_path, plate_path
