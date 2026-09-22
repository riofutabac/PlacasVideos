"""
Plate text voting and candidate plausibility helpers for the ALPR pipeline.
Extracted from plate_processor.py to keep source files under the 400-line
maintainability ceiling. Behavior is unchanged; this is a pure code move.
"""
import re
from typing import List, Dict, Tuple, Optional
import numpy as np
from fast_alpr import ALPR
from src.ecuador_plate_validator import PROVINCE_CODES
from src.deduplicator import levenshtein_distance

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
