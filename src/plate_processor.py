"""
Plate Processing and Recognition Subsystem for ALPR Pipeline.
Handles plate bounding box detection, quality ranking, OCR candidate scoring,
and evidence artifact generation.
"""
import os
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import cv2
import re
from collections import Counter
from fast_alpr import ALPR
from src.pipeline_types import VehicleFrameCandidate
from src.quality_ranker import QualityRanker
from src.timer_profiler import PipelineProfiler
from src.ecuador_plate_validator import PROVINCE_CODES

def vote_plate_characters(
    scored_candidates: List[Dict]
) -> Tuple[Optional[str], float, Optional[np.ndarray], Optional[float], float]:
    """
    Performs character-level weighted voting across multiple OCR candidate reads.
    Ponders each character vote by its character confidence, crop quality score,
    and a gentle prior boost if conforming to valid Ecuadorian plate formats.
    """
    if not scored_candidates:
        return None, 0.0, None, None, 0.0

    valid = []
    for c in scored_candidates:
        raw_text = c.get('text', '')
        clean_t = "".join(ch for ch in raw_text if ch.isalnum()).upper()
        if clean_t:
            confs = c.get('char_confs')
            if not confs or len(confs) != len(raw_text):
                confs = [c.get('ocr_conf', 0.5)] * len(clean_t)
            else:
                confs = [conf for ch, conf in zip(raw_text, confs) if ch.isalnum()]
            if len(confs) != len(clean_t):
                confs = [c.get('ocr_conf', 0.5)] * len(clean_t)

            # Step 3: Prior boost if candidate matches Ecuadorian format
            prior_boost = 0.0
            if re.match(r'^[A-Z]{3}[0-9]{3,4}$', clean_t):
                prior_boost += 0.10
                if clean_t[0] in PROVINCE_CODES:
                    prior_boost += 0.05
            elif re.match(r'^[A-Z]{2}[0-9]{3}[A-Z]$', clean_t):
                prior_boost += 0.10
                if clean_t[0] in PROVINCE_CODES:
                    prior_boost += 0.05

            valid.append({
                'clean_text': clean_t,
                'char_confs': confs,
                'plate_score': c.get('plate_score', 1.0) + prior_boost,
                'ocr_conf': c.get('ocr_conf', 0.5),
                'det_conf': c.get('det_conf', 0.0),
                'crop': c.get('crop'),
                'timestamp': c.get('timestamp')
            })

    if not valid:
        return None, 0.0, None, None, 0.0

    if len(valid) == 1:
        v0 = valid[0]
        return v0['clean_text'], v0['ocr_conf'], v0['crop'], v0['timestamp'], v0['det_conf']

    # 1. Determine modal length (prefer standard 7 and 6 char lengths on ties)
    lengths = [len(v['clean_text']) for v in valid]
    counts = Counter(lengths)
    modal_len = max(counts.keys(), key=lambda l: (counts[l], 2 if l == 7 else (1 if l == 6 else 0)))

    modal_cands = [v for v in valid if len(v['clean_text']) == modal_len]
    if not modal_cands:
        modal_cands = valid
        modal_len = len(modal_cands[0]['clean_text'])

    # 2. Position by position weighted voting
    consensus_chars = []
    consensus_char_confs = []

    for pos in range(modal_len):
        pos_votes: Dict[str, float] = {}
        char_raw_confs: Dict[str, List[float]] = {}

        for cand in modal_cands:
            char = cand['clean_text'][pos]
            c_conf = cand['char_confs'][pos] if pos < len(cand['char_confs']) else cand['ocr_conf']
            weight = c_conf * max(0.1, cand['plate_score'])

            pos_votes[char] = pos_votes.get(char, 0.0) + weight
            char_raw_confs.setdefault(char, []).append(c_conf)

        winning_char = max(pos_votes.keys(), key=lambda c: pos_votes[c])
        consensus_chars.append(winning_char)
        avg_winning_conf = sum(char_raw_confs[winning_char]) / len(char_raw_confs[winning_char])
        consensus_char_confs.append(avg_winning_conf)

    consensus_text = "".join(consensus_chars)
    consensus_conf = float(np.mean(consensus_char_confs)) if consensus_char_confs else 0.0

    # Best crop corresponds to candidate with highest agreement with consensus text
    best_cand = max(modal_cands, key=lambda c: (
        sum(1 for a, b in zip(c['clean_text'], consensus_text) if a == b),
        c['ocr_conf']
    ))

    return consensus_text, consensus_conf, best_cand['crop'], best_cand['timestamp'], best_cand['det_conf']

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
                        bw = x2 - x1
                        bh = y2 - y1
                        aspect = bw / float(bh) if bh > 0 else 0.0

                        # Reject extreme vertical slivers (partial half-plates) or tiny artifacts
                        if bw < 20 or bh < 10 or aspect < 0.75:
                            continue

                        # Step 1: Expand plate bbox by 15% margin per side to prevent cutting edge characters
                        pad_x = int(bw * 0.15)
                        pad_y = int(bh * 0.15)
                        px1 = max(0, x1 - pad_x)
                        py1 = max(0, y1 - pad_y)
                        px2 = min(cand.vehicle_crop.shape[1], x2 + pad_x)
                        py2 = min(cand.vehicle_crop.shape[0], y2 + pad_y)

                        p_crop = cand.vehicle_crop[py1:py2, px1:px2]
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
                ocr_votes = [(r['text'], round(r['ocr_conf'], 3)) for r in scored_ocr]
                c_text, c_conf, c_crop, c_ts, c_det_conf = vote_plate_characters(scored_ocr)
                if c_text is not None:
                    plate_raw = c_text
                    ocr_conf = c_conf
                    plate_crop = c_crop
                    best_plate_ts = c_ts
                    plate_conf = c_det_conf
                else:
                    best = scored_ocr[0]
                    plate_raw, ocr_conf, plate_conf = best['text'], best['ocr_conf'], best['det_conf']
                    plate_crop, best_plate_ts = best['crop'], best['timestamp']

                # Debug: save all candidate crops to evidence/plates/debug/
                debug_dir = os.path.join(self.plate_evidence_dir, "debug")
                os.makedirs(debug_dir, exist_ok=True)
                evt_prefix = event_id if event_id else "candidate"
                for idx, r in enumerate(scored_ocr):
                    clean_text = "".join(c for c in r['text'] if c.isalnum())
                    cand_filename = f"{evt_prefix}_rank{idx}_{clean_text}_c{int(r['ocr_conf']*100)}.jpg"
                    cv2.imwrite(os.path.join(debug_dir, cand_filename), r['crop'])

                cand_summary = " | ".join(f"{r['text']} (c={r['ocr_conf']:.2f}, s={r['plate_score']:.2f})" for r in scored_ocr)
                print(f"    🔎 [OCR Candidates] {cand_summary} => Consensus: {plate_raw} ({ocr_conf:.2f})")
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
