"""
Plate Processing and Recognition Subsystem for ALPR Pipeline.
Handles plate bounding box detection, quality ranking, OCR candidate scoring,
and evidence artifact generation.
"""
import os
import json
import logging
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
from src.deduplicator import levenshtein_distance

logger = logging.getLogger(__name__)

def is_plausible_plate_candidate(text: str) -> bool:
    """Filters out brand names, decals, stickers (e.g. T.U.GPS), logos, and OCR noise."""
    if not text or len(text) < 5 or len(text) > 8:
        return False
    # Must have both letters and digits
    if not any(c.isalpha() for c in text) or not any(c.isdigit() for c in text):
        return False
    # In Ecuador, plates start with at most 3 letters. 4 or more letters at start is a decal/brand (e.g. TUGP5, KENW0)
    if re.match(r'^[A-Z]{4,}', text):
        return False
    # Plates must have at least 2 digits (e.g. TUGP5 only has 1 digit)
    if sum(c.isdigit() for c in text) < 2:
        return False
    return True

def vote_plate_characters(
    scored_candidates: List[Dict],
    province_prior_p: float = 0.05,
    manual_review_threshold: float = 0.0
) -> Tuple[Optional[str], float, Optional[np.ndarray], Optional[float], float, bool]:
    """
    Selects winning plate strictly from real OCR candidate reads (never invents chimeras).
    Scores each unique candidate read by direct evidence (confidence * quality * format prior)
    plus cross-agreement support from other reads within Levenshtein distance <= 2.
    Returns:
        (consensus_text, consensus_conf, best_crop, best_timestamp, best_det_conf, needs_manual_review)
    """
    if not scored_candidates:
        return None, 0.0, None, None, 0.0, False

    valid = []
    for c in scored_candidates:
        raw_text = c.get('text', '')
        clean_t = "".join(ch for ch in raw_text if ch.isalnum()).upper()
        if clean_t:
            prior_boost = 0.0
            if re.match(r'^[A-Z]{3}[0-9]{3,4}$', clean_t):
                prior_boost += 0.15
                if clean_t[0] in PROVINCE_CODES:
                    prior_boost += 0.05
            elif re.match(r'^[A-Z]{2}[0-9]{3}[A-Z]$', clean_t):
                prior_boost += 0.15
                if clean_t[0] in PROVINCE_CODES:
                    prior_boost += 0.05

            valid.append({
                'clean_text': clean_t,
                'plate_score': c.get('plate_score', 1.0) + prior_boost,
                'ocr_conf': c.get('ocr_conf', 0.5),
                'det_conf': c.get('det_conf', 0.0),
                'crop': c.get('crop'),
                'timestamp': c.get('timestamp')
            })

    if not valid:
        return None, 0.0, None, None, 0.0, False

    # Filter out brand/body text (e.g. KENW0RRTH) if any plausible plate candidates exist
    needs_manual_review = False
    plausible = [v for v in valid if is_plausible_plate_candidate(v['clean_text'])]
    if plausible:
        valid = plausible
    else:
        needs_manual_review = True

    if len(valid) == 1:
        v0 = valid[0]
        return v0['clean_text'], v0['ocr_conf'], v0['crop'], v0['timestamp'], v0['det_conf'], needs_manual_review

    unique_texts = list(set(v['clean_text'] for v in valid))
    scores: Dict[str, float] = {}

    for text in unique_texts:
        instances = [v for v in valid if v['clean_text'] == text]
        direct_w = sum(inst['ocr_conf'] * max(0.1, inst['plate_score']) for inst in instances)
        if province_prior_p > 0.0 and text.startswith('P'):
            direct_w += province_prior_p * len(instances)

        cross_w = 0.0
        for other in valid:
            o_text = other['clean_text']
            if o_text == text:
                continue
            dist = levenshtein_distance(text, o_text)
            if dist <= 2:
                sim = 1.0 - (dist / max(len(text), len(o_text)))
                cross_w += (other['ocr_conf'] * max(0.1, other['plate_score'])) * (sim ** 2) * 0.5

        scores[text] = direct_w + cross_w

    ranked = sorted(scores.items(), key=lambda x: (x[1], x[0]), reverse=True)
    winner_text, winner_score = ranked[0]

    runner_up = ranked[1] if len(ranked) > 1 else None
    if runner_up and manual_review_threshold > 0.0:
        r_text, r_score = runner_up
        diff = winner_score - r_score
        if len(winner_text) > 0 and len(r_text) > 0 and winner_text[0] != r_text[0]:
            if diff < manual_review_threshold:
                needs_manual_review = True
        elif diff < (manual_review_threshold * 0.5):
            needs_manual_review = True

    winner_instances = [v for v in valid if v['clean_text'] == winner_text]
    best_cand = max(winner_instances, key=lambda v: (v['plate_score'], v['ocr_conf']))
    consensus_conf = float(np.mean([v['ocr_conf'] for v in winner_instances]))

    return winner_text, consensus_conf, best_cand['crop'], best_cand['timestamp'], best_cand['det_conf'], needs_manual_review

def try_two_tier_ocr(
    alpr: ALPR,
    plate_crop: np.ndarray
) -> Optional[Tuple[str, List[float], float]]:
    """
    Attempts two-tier (two-row) OCR for motorcycle or square-format plates.
    Splits the crop into overlapping top and bottom halves, runs OCR on each,
    and combines results if both rows produce plausible plate tokens.
    """
    if plate_crop is None or plate_crop.size == 0:
        return None
    h, w = plate_crop.shape[:2]
    aspect = w / max(1, h)
    if aspect > 1.35 or h < 24 or w < 24:
        return None

    try:
        # Overlapping split: top 55% and bottom 58%
        top_crop = plate_crop[0 : int(h * 0.55), :]
        bot_crop = plate_crop[int(h * 0.42) : h, :]

        top_res = alpr.ocr.predict(top_crop)
        bot_res = alpr.ocr.predict(bot_crop)

        if not top_res or not bot_res or not top_res.text or not bot_res.text:
            return None

        top_txt = "".join(c for c in top_res.text.strip().upper() if c.isalnum())
        bot_txt = "".join(c for c in bot_res.text.strip().upper() if c.isalnum())

        if len(top_txt) >= 2 and len(bot_txt) >= 3:
            combined_text = top_txt + bot_txt
            top_confs = top_res.confidence if top_res.confidence else [0.5] * len(top_txt)
            bot_confs = bot_res.confidence if bot_res.confidence else [0.5] * len(bot_txt)
            combined_confs = list(top_confs) + list(bot_confs)
            avg_conf = float(np.mean(combined_confs)) if combined_confs else 0.5
            return combined_text, combined_confs, avg_conf
    except Exception:
        pass

    return None

class PlateProcessor:
    def __init__(
        self,
        alpr: ALPR,
        ranker: QualityRanker,
        profiler: PipelineProfiler,
        top_k_crops: int = 7,
        vehicle_evidence_dir: str = "evidence/vehicles",
        plate_evidence_dir: str = "evidence/plates",
        save_manifest: bool = False,
        save_debug_crops: bool = False,
        province_prior: Optional[Dict[str, Any]] = None,
        min_ocr_confidence: float = 0.70
    ):
        self.alpr = alpr
        self.ranker = ranker
        self.profiler = profiler
        self.top_k_crops = top_k_crops
        self.vehicle_evidence_dir = vehicle_evidence_dir
        self.plate_evidence_dir = plate_evidence_dir
        self.save_manifest = save_manifest
        self.save_debug_crops = save_debug_crops
        self.min_ocr_confidence = min_ocr_confidence
        self.manifest_records: List[Dict[str, Any]] = []
        self.last_needs_manual_review = False

        province_cfg = province_prior or {}
        self.province_prior_enabled = province_cfg.get('enabled', False)
        self.province_prior_p = province_cfg.get('beta', 0.10) if self.province_prior_enabled else 0.0
        self.manual_review_threshold = province_cfg.get('manual_review_threshold', 0.15) if self.province_prior_enabled else 0.0

        os.makedirs(self.vehicle_evidence_dir, exist_ok=True)
        os.makedirs(self.plate_evidence_dir, exist_ok=True)

    def detect_plate_candidates(
        self,
        best_frames: List[VehicleFrameCandidate],
        event_id: Optional[str] = None
    ) -> List[Dict]:
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
                        cand_dict = {
                            'score': q_score, 'det_conf': float(d.confidence), 'crop': p_crop,
                            'timestamp': cand.timestamp, 'veh_crop': cand.vehicle_crop,
                            'bbox': [x1, y1, x2, y2], 'padded_bbox': [px1, py1, px2, py2]
                        }
                        candidates.append(cand_dict)

                        # Step A0: Save candidate crops and record manifest if enabled
                        if self.save_manifest and event_id:
                            cand_idx = len(self.manifest_records)
                            cand_sub, veh_sub = os.path.join(self.plate_evidence_dir, "candidates"), os.path.join(self.vehicle_evidence_dir, "candidates")
                            os.makedirs(cand_sub, exist_ok=True); os.makedirs(veh_sub, exist_ok=True)
                            p_path = os.path.join(cand_sub, f"{event_id}_cand{cand_idx}_plate.jpg")
                            v_path = os.path.join(veh_sub, f"{event_id}_cand{cand_idx}_veh.jpg")
                            cv2.imwrite(p_path, p_crop); cv2.imwrite(v_path, cand.vehicle_crop)
                            self.manifest_records.append({
                                "event_id": event_id, "candidate_idx": cand_idx, "timestamp": round(float(cand.timestamp), 3),
                                "bbox": [x1, y1, x2, y2], "padded_bbox": [px1, py1, px2, py2],
                                "quality_score": round(float(q_score), 4), "det_conf": round(float(d.confidence), 4),
                                "plate_crop_path": p_path, "vehicle_crop_path": v_path
                            })
            except Exception as e:
                logger.debug(f"Plate candidate detection exception: {e}", exc_info=True)
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
        self.last_needs_manual_review = False

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

                    # Step 4: Two-tier OCR attempt for near-square / motorcycle plates
                    two_tier = try_two_tier_ocr(self.alpr, item['crop'])
                    if two_tier is not None:
                        tt_text, tt_confs, tt_avg = two_tier
                        scored_ocr.append({
                            'text': tt_text,
                            'ocr_conf': tt_avg,
                            'char_confs': tt_confs,
                            'det_conf': item['det_conf'],
                            'plate_score': item['score'] * 1.1,
                            'crop': item['crop'],
                            'timestamp': item['timestamp']
                        })
                except Exception as e:
                    logger.debug(f"Candidate OCR exception: {e}", exc_info=True)

            if scored_ocr:
                scored_ocr.sort(key=lambda x: (x['ocr_conf'], x['plate_score']), reverse=True)
                ocr_votes = [(r['text'], round(r['ocr_conf'], 3)) for r in scored_ocr]
                c_text, c_conf, c_crop, c_ts, c_det_conf, needs_review = vote_plate_characters(
                    scored_ocr,
                    province_prior_p=self.province_prior_p,
                    manual_review_threshold=self.manual_review_threshold
                )
                self.last_needs_manual_review = needs_review
                if c_text is not None:
                    clean_check = "".join(ch for ch in c_text if ch.isalnum())
                    plate_crop = c_crop
                    best_plate_ts = c_ts
                    plate_conf = c_det_conf
                    # Reject reads with < 5 chars, conf < min_ocr_conf, or non-plausible (decals) as sin placa
                    if len(clean_check) < 5 or c_conf < self.min_ocr_confidence or not is_plausible_plate_candidate(clean_check):
                        logger.info(
                            f"Plate text '{c_text}' rejected (len={len(clean_check)}, conf={c_conf:.2f}, plausible={is_plausible_plate_candidate(clean_check)}, treated as sin placa)"
                        )
                        plate_raw = None
                        ocr_conf = 0.0
                    else:
                        plate_raw = c_text
                        ocr_conf = c_conf
                else:
                    best = scored_ocr[0]
                    clean_check = "".join(ch for ch in best['text'] if ch.isalnum())
                    plate_crop, best_plate_ts = best['crop'], best['timestamp']
                    plate_conf = best['det_conf']
                    if len(clean_check) < 5 or best['ocr_conf'] < self.min_ocr_confidence or not is_plausible_plate_candidate(clean_check):
                        plate_raw = None
                        ocr_conf = 0.0
                    else:
                        plate_raw, ocr_conf = best['text'], best['ocr_conf']

                # Debug: save candidate crops only if explicitly enabled
                if self.save_debug_crops:
                    debug_dir = os.path.join(self.plate_evidence_dir, "debug")
                    os.makedirs(debug_dir, exist_ok=True)
                    evt_prefix = event_id if event_id else "candidate"
                    for idx, r in enumerate(scored_ocr):
                        clean_text = "".join(c for c in r['text'] if c.isalnum())
                        cand_filename = f"{evt_prefix}_rank{idx}_{clean_text}_c{int(r['ocr_conf']*100)}.jpg"
                        cv2.imwrite(os.path.join(debug_dir, cand_filename), r['crop'])

                cand_summary = " | ".join(f"{r['text']} (c={r['ocr_conf']:.2f}, s={r['plate_score']:.2f})" for r in scored_ocr)
                rev_flag = " [REVISION_MANUAL]" if self.last_needs_manual_review else ""
                print(f"    🔎 [OCR Candidates] {cand_summary} => Consensus: {plate_raw} ({ocr_conf:.2f}){rev_flag}")
            else:
                top = top_k[0]
                plate_crop, plate_conf, best_plate_ts = top['crop'], top['det_conf'], top['timestamp']

        self.profiler.stop_stage('ocr')
        return plate_raw, plate_crop, plate_conf, ocr_conf, best_plate_ts, ocr_votes

    def write_manifest(self, output_path: Optional[str] = None) -> str:
        """Writes candidate manifest JSON to disk."""
        path = output_path or os.path.join(self.plate_evidence_dir, "manifest.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.manifest_records, f, indent=2, ensure_ascii=False)
        logger.info(f"Manifest written with {len(self.manifest_records)} records to {path}")
        return path

    def save_evidence(
        self,
        event_id: str,
        best_veh_crop: np.ndarray,
        plate_crop: Optional[np.ndarray]
    ) -> Tuple[str, Optional[str]]:
        """Saves vehicle and plate evidence images to disk."""
        self.profiler.start_stage('image_write')
        veh_path = os.path.join(self.vehicle_evidence_dir, f"{event_id}_veh.jpg")
        if best_veh_crop is not None and best_veh_crop.size > 0:
            cv2.imwrite(veh_path, best_veh_crop)
        plate_path = None
        if plate_crop is not None and plate_crop.size > 0:
            plate_path = os.path.join(self.plate_evidence_dir, f"{event_id}_plate.jpg")
            cv2.imwrite(plate_path, plate_crop)
        self.profiler.stop_stage('image_write')
        return veh_path, plate_path
