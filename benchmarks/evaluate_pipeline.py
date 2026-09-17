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

    print("\n" + "="*60)
    print("GROUND TRUTH EVALUATION")
    print("="*60)
    print(f"Ground Truth Events:  {len(ground_truth)}")
    print(f"Detected Events:      {len(detected_events)}")

    matched_gt = set()
    matched_direction = 0
    exact_plate_matches = 0
    legible_gt_plates = sum(1 for gt in ground_truth if gt.get('plate_legible'))

    for det in detected_events:
        det_ts = det['event_timestamp']
        det_video = det['video_source']
        det_dir = det['direction']
        det_plate = det['plate_corrected'] or det['plate_normalized']

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
            if det_dir == gt['direction']:
                matched_direction += 1
            if gt.get('plate_legible') and gt.get('plate_text') and det_plate:
                # Compare without dashes
                gt_clean = gt['plate_text'].replace("-", "").strip().upper()
                det_clean = det_plate.replace("-", "").strip().upper()
                if gt_clean == det_clean:
                    exact_plate_matches += 1
                    print(f"  [MATCH] GT: {gt['ground_truth_id']} ({gt_clean}) == DET: {det['event_id']} ({det_clean}) [dt={best_dt:.1f}s, dir={det_dir}]")
                else:
                    print(f"  [PARTIAL] GT: {gt['ground_truth_id']} ({gt_clean}) vs DET: {det['event_id']} ({det_clean}) [dt={best_dt:.1f}s]")
            else:
                print(f"  [MATCH VEHICLE] GT: {gt['ground_truth_id']} == DET: {det['event_id']} [dir={det_dir}, plate={det_plate}]")

    for i, gt in enumerate(ground_truth):
        if i not in matched_gt:
            print(f"  [MISSED GT] {gt['ground_truth_id']} at {gt['approx_timestamp']}s ({gt['description']})")

    event_recall = len(matched_gt) / len(ground_truth) if ground_truth else 0.0
    dir_acc = matched_direction / len(matched_gt) if matched_gt else 0.0
    plate_acc = exact_plate_matches / legible_gt_plates if legible_gt_plates else 0.0
    false_positives = len(detected_events) - len(matched_gt)

    print("-"*60)
    print(f"EVENT RECALL:          {event_recall:.1%} ({len(matched_gt)}/{len(ground_truth)})")
    print(f"DIRECTION ACCURACY:    {dir_acc:.1%} ({matched_direction}/{len(matched_gt)})")
    print(f"PLATE EXACT MATCH:     {plate_acc:.1%} ({exact_plate_matches}/{legible_gt_plates} legible)")
    print(f"FALSE POSITIVES:       {false_positives}")
    print("="*60 + "\n")

if __name__ == '__main__':
    evaluate_run()
