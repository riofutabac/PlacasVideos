"""
Ground Truth Evaluation Tool for ALPR Pipeline.
Compares SQLite events against benchmarks/ground_truth.json and outputs metrics:
- Event Recall
- Direction Accuracy
- Plate Exact Match
- False Positive Count
"""
import os
import json
import sqlite3
from typing import Optional

def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculates Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def evaluate_run(db_path: str = "data/events.sqlite", run_id: Optional[str] = None, gt_path: str = "benchmarks/ground_truth.json"):
    if not os.path.exists(gt_path):
        print(f"Ground truth not found at {gt_path}")
        return

    with open(gt_path) as f:
        ground_truth = json.load(f)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    query = "SELECT * FROM events WHERE duplicate_of IS NULL"
    params = []
    if run_id:
        query += " AND processing_run_id = ?"
        params.append(run_id)
    query += " ORDER BY event_timestamp ASC"

    cursor.execute(query, params)
    detected_events = [dict(r) for r in cursor.fetchall()]
    conn.close()

    gt_videos = set(gt['video_source'] for gt in ground_truth)
    eval_detected = [d for d in detected_events if d['video_source'] in gt_videos]

    clips_list = sorted(list(set(os.path.basename(v) for v in gt_videos)))
    clips_display = ", ".join(clips_list) if len(clips_list) <= 4 else f"{len(clips_list)} clips"

    print("\n" + "="*60)
    print("GROUND TRUTH EVALUATION")
    print("="*60)
    print(f"Ground Truth Events:  {len(ground_truth)} (en {clips_display})")
    print(f"Total Eventos Lote:   {len(detected_events)}")
    if not eval_detected:
        print(f"ℹ️ Esta corrida no incluye videos del ground truth ({clips_display}). Evaluación omitida.")
        print("="*60 + "\n")
        return
    print(f"Eventos en Clips GT:  {len(eval_detected)}")

    legible_gt = [gt for gt in ground_truth if gt.get('plate_legible') and gt.get('plate_text')]
    legible_gt_plates = len(legible_gt)
    total_real_gt_chars = sum(len(gt['plate_text'].replace("-", "").strip().upper()) for gt in legible_gt)

    matched_gt = set()
    gt_to_det = {}
    matched_direction = 0
    exact_plate_matches = 0
    unread_legible_plates = 0
    pos0_matches = 0
    total_correct_chars = 0

    for det in eval_detected:
        det_ts = det['event_timestamp']
        det_video = det['video_source']
        det_dir = det['direction']

        best_gt = None
        best_dt = 1e9

        for i, gt in enumerate(ground_truth):
            if i in matched_gt:
                continue
            if gt['video_source'] == det_video:
                dt = abs(det_ts - gt['approx_timestamp'])
                if dt < 25.0 and dt < best_dt:
                    best_dt = dt
                    best_gt = (i, gt)

        if best_gt:
            gt_idx, gt = best_gt
            matched_gt.add(gt_idx)
            gt_to_det[gt_idx] = (det, best_dt)
            if det_dir == gt['direction']:
                matched_direction += 1

    for i, gt in enumerate(ground_truth):
        is_legible = bool(gt.get('plate_legible') and gt.get('plate_text'))
        if is_legible:
            gt_clean = gt['plate_text'].replace("-", "").strip().upper()
            n_chars = len(gt_clean)
            if i in gt_to_det:
                det, dt = gt_to_det[i]
                det_plate = det['plate_corrected'] or det['plate_normalized']
                if det_plate:
                    det_clean = det_plate.replace("-", "").strip().upper()
                    dist = levenshtein_distance(gt_clean, det_clean)
                    correct_chars = max(0, n_chars - dist)
                    total_correct_chars += correct_chars
                    if dist == 0:
                        exact_plate_matches += 1
                        print(f"  [MATCH] GT: {gt['ground_truth_id']} ({gt_clean}) == DET: {det['event_id']} ({det_clean}) [dist=0, {n_chars}/{n_chars} chars, dt={dt:.1f}s]")
                    else:
                        print(f"  [PARTIAL] GT: {gt['ground_truth_id']} ({gt_clean}) vs DET: {det['event_id']} ({det_clean}) [dist={dist}, {correct_chars}/{n_chars} chars, dt={dt:.1f}s]")
                    if len(det_clean) > 0 and len(gt_clean) > 0 and det_clean[0] == gt_clean[0]:
                        pos0_matches += 1
                else:
                    unread_legible_plates += 1
                    print(f"  [UNREAD LEGIBLE PLATE] GT: {gt['ground_truth_id']} ({gt_clean}) == DET: {det['event_id']} [sin lectura de placa, 0/{n_chars} chars, dt={dt:.1f}s]")
            else:
                unread_legible_plates += 1
                print(f"  [MISSED LEGIBLE GT] GT: {gt['ground_truth_id']} ({gt_clean}) at {gt.get('approx_timestamp')}s [evento no detectado, 0/{n_chars} chars]")
        else:
            if i in gt_to_det:
                det, dt = gt_to_det[i]
                det_plate = det['plate_corrected'] or det['plate_normalized']
                print(f"  [MATCH VEHICLE] GT: {gt['ground_truth_id']} == DET: {det['event_id']} [dir={det['direction']}, plate={det_plate}]")
            else:
                print(f"  [MISSED GT] {gt['ground_truth_id']} at {gt.get('approx_timestamp')}s ({gt.get('description')})")

    event_recall = len(matched_gt) / len(ground_truth) if ground_truth else 0.0
    dir_acc = matched_direction / len(matched_gt) if matched_gt else 0.0
    plate_acc = exact_plate_matches / legible_gt_plates if legible_gt_plates else 0.0
    char_acc = (total_correct_chars / total_real_gt_chars) if total_real_gt_chars > 0 else 0.0
    pos0_acc = (pos0_matches / legible_gt_plates) if legible_gt_plates > 0 else 0.0
    false_positives = len(eval_detected) - len(matched_gt)

    print("-"*60)
    print(f"EVENT RECALL:          {event_recall:.1%} ({len(matched_gt)}/{len(ground_truth)})")
    print(f"DIRECTION ACCURACY:    {dir_acc:.1%} ({matched_direction}/{len(matched_gt)})")
    print(f"PLATE EXACT MATCH:     {plate_acc:.1%} ({exact_plate_matches}/{legible_gt_plates} legible)")
    print(f"CHARACTER ACCURACY:    {char_acc:.1%} ({total_correct_chars}/{total_real_gt_chars} chars)")
    print(f"UNREAD LEGIBLE PLATES: {unread_legible_plates} ({unread_legible_plates}/{legible_gt_plates})")
    print(f"FIRST CHAR ACCURACY:   {pos0_acc:.1%} ({pos0_matches}/{legible_gt_plates})")
    print(f"FALSE POSITIVES:       {false_positives}")
    print("="*60 + "\n")

    return {
        "event_recall": event_recall,
        "direction_accuracy": dir_acc,
        "plate_exact_match": plate_acc,
        "character_accuracy": char_acc,
        "unread_legible_plates": unread_legible_plates,
        "pos0_accuracy": pos0_acc,
        "false_positives": false_positives,
        "matched_gt_count": len(matched_gt),
        "total_gt_count": len(ground_truth),
        "total_real_gt_chars": total_real_gt_chars,
        "total_correct_chars": total_correct_chars,
    }

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate pipeline against ground truth")
    parser.add_argument("--db", default="data/events.sqlite", help="Path to SQLite database")
    parser.add_argument("--run-id", default=None, help="Specific run_id to evaluate")
    parser.add_argument("--gt", default="benchmarks/ground_truth.json", help="Path to ground truth JSON")
    args = parser.parse_args()
    evaluate_run(db_path=args.db, run_id=args.run_id, gt_path=args.gt)
