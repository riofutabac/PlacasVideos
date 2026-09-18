#!/usr/bin/env python3
"""
Offline OCR Matrix Benchmark (Fase A1).
Runs REAL model inference across saved candidate plate crops using fast_plate_ocr.
Evaluates:
- OCR Models: cct-xs-v2-global-model, cct-s-v2-global-model, global-plates-mobile-vit-v2-model
- Top-K candidate crops: 3, 5, 7
- Fusion methods: character-level weighted voting vs logprob sum
- Pichincha Prior beta: 0.0, 0.05, 0.10, 0.20

All metrics (exact match %, character accuracy %, p50/p95 latency) are empirically
computed from real inference calls on real plate crops stored in evidence/plates/debug/.
"""

import os
import sys
import re
import csv
import time
import json
import argparse
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import cv2

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.plate_processor import vote_plate_characters
from src.ecuador_plate_validator import PROVINCE_CODES
from benchmarks.evaluate_pipeline import levenshtein_distance

OCR_MODELS = [
    "cct-xs-v2-global-model",
    "cct-s-v2-global-model",
    "global-plates-mobile-vit-v2-model"
]

TOP_K_FRAMES = [3, 5, 7]
FUSION_METHODS = ["char_voting", "logprob_sum"]
BETA_PRIORS = [0.0, 0.05, 0.10, 0.20]

def apply_letterbox_resize(image: np.ndarray, target_w: int = 128, target_h: int = 64) -> np.ndarray:
    """Resize image preserving aspect ratio with zero padding."""
    if image is None or image.size == 0:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)
    h, w = image.shape[:2]
    scale = min(target_w / w, target_h / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_w = target_w - nw
    pad_h = target_h - nh
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

def extract_plate_with_margin(veh_crop: np.ndarray, bbox: List[int], margin_ratio: float) -> np.ndarray:
    """Extract plate crop from vehicle crop using specified margin expansion."""
    if veh_crop is None or veh_crop.size == 0 or len(bbox) < 4:
        return np.zeros((30, 70, 3), dtype=np.uint8)
    vh, vw = veh_crop.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    pad_x = int(bw * margin_ratio)
    pad_y = int(bh * margin_ratio)
    px1 = max(0, x1 - pad_x)
    py1 = max(0, y1 - pad_y)
    px2 = min(vw, x2 + pad_x)
    py2 = min(vh, y2 + pad_y)
    crop = veh_crop[py1:py2, px1:px2]
    return crop if crop.size > 0 else veh_crop[y1:y2, x1:x2]


def logprob_fusion(
    scored_candidates: List[Dict],
    province_prior_p: float = 0.0
) -> Tuple[Optional[str], float]:
    """Calculates weighted log-probability sum across candidate reads."""
    if not scored_candidates:
        return None, 0.0
    valid = [c for c in scored_candidates if c.get('text')]
    if not valid:
        return None, 0.0

    scores: Dict[str, float] = {}
    for c in valid:
        t = "".join(ch for ch in c['text'] if ch.isalnum()).upper()
        if len(t) < 5:
            continue
        conf = max(0.01, min(0.99, c.get('ocr_conf', 0.5)))
        logprob = float(np.log(conf) - np.log(1.0 - conf))
        w = max(0.1, c.get('plate_score', 1.0)) * logprob
        if t.startswith('P') and province_prior_p > 0:
            w += province_prior_p * 2.0
        scores[t] = scores.get(t, 0.0) + w

    if not scores:
        return None, 0.0
    best_t = max(scores.keys(), key=lambda k: scores[k])
    return best_t, 0.85

def load_real_event_crops(debug_dir: str = "evidence/plates/debug") -> Dict[str, List[str]]:
    """Groups real saved plate crops by event ID."""
    if not os.path.exists(debug_dir):
        return {}
    event_crops: Dict[str, List[str]] = {}
    for fname in sorted(os.listdir(debug_dir)):
        if not fname.endswith(".jpg"):
            continue
        # Format: EVT_(clip)_idx_time_rankX_plate_conf.jpg
        m = re.match(r"(EVT_\([0-9]+\)_[0-9]+_[0-9]+)", fname)
        if m:
            evt_id = m.group(1)
            event_crops.setdefault(evt_id, []).append(os.path.join(debug_dir, fname))
    return event_crops

def map_events_to_ground_truth(
    event_ids: List[str],
    gt_path: str = "benchmarks/ground_truth.json"
) -> Dict[str, Dict[str, Any]]:
    """Maps event IDs to ground truth entries based on video clip and approximate timestamp."""
    gt_data = []
    if os.path.exists(gt_path):
        with open(gt_path, "r", encoding="utf-8") as f:
            gt_data = json.load(f)

    # Known event to GT mappings based on verified test timestamps
    event_mapping = {}
    for evt_id in event_ids:
        # Extract clip and timestamp from event ID: EVT_(60)_0023_27 -> clip 60, ~27s
        m = re.search(r"EVT_\(([0-9]+)\)_[0-9]+_([0-9]+)", evt_id)
        if not m:
            continue
        clip_num = m.group(1)
        evt_ts = float(m.group(2))

        best_gt = None
        min_dt = 15.0  # seconds window
        for gt in gt_data:
            if f"({clip_num})" in gt.get("video_source", ""):
                dt = abs(gt.get("approx_timestamp", 0.0) - evt_ts)
                if dt < min_dt:
                    min_dt = dt
                    best_gt = gt
        if best_gt:
            event_mapping[evt_id] = best_gt
    return event_mapping

def run_ocr_matrix_benchmark(
    debug_dir: str = "evidence/plates/debug",
    gt_path: str = "benchmarks/ground_truth.json",
    output_csv: str = "benchmarks/benchmark_ocr_matrix.csv",
    quick_mode: bool = False
) -> List[Dict[str, Any]]:
    print("=" * 80)
    print("BENCHMARK OFFLINE OCR REAL: MATRIZ DE MODELOS Y FUSIÓN")
    print("=" * 80)

    event_crops = load_real_event_crops(debug_dir)
    print(f"📦 Encontrados recortes de {len(event_crops)} eventos en {debug_dir}")

    gt_map = map_events_to_ground_truth(list(event_crops.keys()), gt_path)
    print(f"🔗 Mapeados {len(gt_map)}/{len(event_crops)} eventos contra {gt_path}")

    from fast_plate_ocr import LicensePlateRecognizer
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if sys.platform != 'darwin' else ['CPUExecutionProvider']

    models_to_test = [OCR_MODELS[0]] if quick_mode else OCR_MODELS
    k_frames_to_test = [7] if quick_mode else TOP_K_FRAMES
    betas_to_test = [0.0, 0.10] if quick_mode else BETA_PRIORS

    results = []

    for model_name in models_to_test:
        print(f"\n🧠 Evaluando modelo OCR: {model_name}...")
        try:
            recognizer = LicensePlateRecognizer(hub_ocr_model=model_name, providers=providers)
        except Exception as e:
            print(f"  ❌ Error cargando {model_name}: {e}")
            continue

        # Run REAL inference on all crops of all mapped events
        model_latencies = []
        event_candidates: Dict[str, List[Dict[str, Any]]] = {}

        for evt_id, crop_paths in event_crops.items():
            if evt_id not in gt_map:
                continue
            cand_list = []
            for cp in crop_paths:
                img_bgr = cv2.imread(cp)
                if img_bgr is None or img_bgr.size == 0:
                    continue

                # mobile-vit expects 1 channel grayscale
                feed_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if "mobile-vit" in model_name else img_bgr

                t0 = time.perf_counter()
                try:
                    preds = recognizer.run([feed_img], return_confidence=True)
                    pred = preds[0] if preds else None
                except Exception:
                    pred = None
                t1 = time.perf_counter()
                model_latencies.append((t1 - t0) * 1000.0)

                if pred and pred.plate:
                    clean_text = "".join(c for c in pred.plate.strip().upper() if c.isalnum())
                    char_confs = list(pred.char_probs) if pred.char_probs is not None else [0.85] * len(clean_text)
                    mean_conf = float(np.mean(char_confs)) if len(char_confs) > 0 else 0.5
                    cand_list.append({
                        'text': clean_text,
                        'ocr_conf': mean_conf,
                        'char_confs': char_confs,
                        'plate_score': 1.0,
                        'crop': img_bgr,
                        'timestamp': 0.0
                    })
            if cand_list:
                event_candidates[evt_id] = cand_list

        p50_lat = float(np.percentile(model_latencies, 50)) if model_latencies else 0.0
        p95_lat = float(np.percentile(model_latencies, 95)) if model_latencies else 0.0
        print(f"  ⚡ Inferencia real completada: {len(model_latencies)} recortes procesados. p50={p50_lat:.1f}ms, p95={p95_lat:.1f}ms")

        # Evaluate combinations of fusion, top-k, and beta on real candidates
        for k_frames in k_frames_to_test:
            for fusion in FUSION_METHODS:
                for beta in betas_to_test:
                    exact_matches = 0
                    total_chars = 0
                    correct_chars = 0
                    pos0_correct = 0
                    non_p_to_p_errors = 0
                    evaluated_events = 0

                    for evt_id, cands in event_candidates.items():
                        gt_info = gt_map.get(evt_id)
                        if not gt_info or not gt_info.get("plate_legible", True):
                            continue
                        gt_plate = (gt_info.get("plate_text") or "").replace("-", "").upper()
                        if not gt_plate:
                            continue

                        evaluated_events += 1
                        use_cands = cands[:k_frames]

                        if fusion == "char_voting":
                            pred_text, _, _, _, _, _ = vote_plate_characters(use_cands, province_prior_p=beta)
                        else:
                            pred_text, _ = logprob_fusion(use_cands, province_prior_p=beta)

                        clean_pred = (pred_text or "").replace("-", "").upper()
                        dist = levenshtein_distance(gt_plate, clean_pred)
                        if dist == 0:
                            exact_matches += 1
                        total_chars += len(gt_plate)
                        correct_chars += max(0, len(gt_plate) - dist)

                        if clean_pred:
                            if clean_pred[0] == gt_plate[0]:
                                pos0_correct += 1
                            elif clean_pred[0] == 'P' and gt_plate[0] != 'P':
                                non_p_to_p_errors += 1

                    exact_pct = (exact_matches / evaluated_events * 100.0) if evaluated_events else 0.0
                    char_pct = (correct_chars / total_chars * 100.0) if total_chars else 0.0
                    pos0_pct = (pos0_correct / evaluated_events * 100.0) if evaluated_events else 0.0

                    row = {
                        "model": model_name,
                        "top_k": k_frames,
                        "fusion": fusion,
                        "beta_p": beta,
                        "exact_match_pct": round(exact_pct, 1),
                        "char_acc_pct": round(char_pct, 1),
                        "pos0_acc_pct": round(pos0_pct, 1),
                        "non_p_corrupted": non_p_to_p_errors,
                        "evaluated_events": evaluated_events,
                        "latency_p50_ms": round(p50_lat, 1),
                        "latency_p95_ms": round(p95_lat, 1)
                    }
                    results.append(row)

    # Save to CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    if results:
        fieldnames = list(results[0].keys())
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\n📁 Matriz real de resultados guardada en {output_csv}")

    # Display results
    print("\n" + "=" * 100)
    print(f"{'Modelo':<33} | {'K':<3} | {'Fusion':<11} | {'Beta':<5} | {'Exact%':<7} | {'Char%':<7} | {'P0%':<5} | {'NonP_err':<8} | {'p50 (ms)'}")
    print("-" * 100)
    for r in sorted(results, key=lambda x: (x['exact_match_pct'], x['char_acc_pct']), reverse=True)[:15]:
        print(f"{r['model']:<33} | {r['top_k']:<3} | {r['fusion']:<11} | {r['beta_p']:<5.2f} | {r['exact_match_pct']:<7.1f} | {r['char_acc_pct']:<7.1f} | {r['pos0_acc_pct']:<5.1f} | {r['non_p_corrupted']:<8} | {r['latency_p50_ms']}")
    print("=" * 100 + "\n")

    return results

def main():
    parser = argparse.ArgumentParser(description="Run real offline OCR matrix benchmark")
    parser.add_argument("--debug-dir", default="evidence/plates/debug", help="Path to real crops dir")
    parser.add_argument("--gt", default="benchmarks/ground_truth.json", help="Path to ground truth JSON")
    parser.add_argument("--output", default="benchmarks/benchmark_ocr_matrix.csv", help="Output CSV path")
    parser.add_argument("--quick", action="store_true", help="Run quick subset")
    args = parser.parse_args()

    run_ocr_matrix_benchmark(
        debug_dir=args.debug_dir,
        gt_path=args.gt,
        output_csv=args.output,
        quick_mode=args.quick
    )

if __name__ == "__main__":
    main()
