#!/usr/bin/env python3
"""
Candidate Manifest Dataset Exporter (Fase A0).
Matches candidates from manifest.json against ground_truth.json and generates
an enriched dataset_manifest.json for offline OCR and detector benchmarks.
"""

import os
import re
import json
import argparse
from typing import List, Dict, Any, Optional

def load_ground_truth(gt_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(gt_path):
        return []
    with open(gt_path, "r", encoding="utf-8") as f:
        return json.load(f)

def find_matching_gt(
    event_id: str,
    timestamp: float,
    gt_list: List[Dict[str, Any]],
    tolerance_sec: float = 6.0
) -> Optional[Dict[str, Any]]:
    """Matches a candidate event to a ground truth record by clip and timestamp proximity."""
    clip_match = re.search(r'\((\d+)\)', event_id)
    clip_num = clip_match.group(1) if clip_match else None

    best_match = None
    min_diff = float('inf')

    for gt in gt_list:
        gt_source = gt.get("video_source", "")
        gt_clip_match = re.search(r'\((\d+)\)', gt_source)
        gt_clip = gt_clip_match.group(1) if gt_clip_match else None

        if clip_num and gt_clip and clip_num != gt_clip:
            continue

        diff = abs(timestamp - gt.get("approx_timestamp", 0.0))
        if diff <= tolerance_sec and diff < min_diff:
            min_diff = diff
            best_match = gt

    return best_match

def export_plate_dataset(
    manifest_path: str = "evidence/plates/manifest.json",
    gt_path: str = "benchmarks/ground_truth.json",
    output_path: str = "evidence/plates/dataset_manifest.json"
) -> List[Dict[str, Any]]:
    """Enriches candidate manifest with ground truth annotations."""
    if not os.path.exists(manifest_path):
        print(f"⚠️ [Dataset Export] No se encontró {manifest_path}")
        return []

    with open(manifest_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    gt_list = load_ground_truth(gt_path)
    enriched = []

    for r in records:
        evt = r.get("event_id", "")
        ts = r.get("timestamp", 0.0)
        gt = find_matching_gt(evt, ts, gt_list)

        rec = dict(r)
        if gt:
            rec["ground_truth_id"] = gt.get("ground_truth_id")
            rec["ground_truth_plate"] = gt.get("plate_text")
            rec["plate_legible"] = gt.get("plate_legible", True)
            rec["province"] = gt.get("province")
            rec["vehicle_type"] = gt.get("vehicle_type")
        else:
            rec["ground_truth_id"] = None
            rec["ground_truth_plate"] = None
            rec["plate_legible"] = False
            rec["province"] = None
            rec["vehicle_type"] = None

        enriched.append(rec)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(enriched, f, indent=2, ensure_ascii=False)

    print(f"✅ [Dataset Export] {len(enriched)} candidatos exportados a {output_path}")
    return enriched

def main():
    parser = argparse.ArgumentParser(description="Export candidate plate dataset with ground truth matching")
    parser.add_argument("--manifest", default="evidence/plates/manifest.json", help="Path to manifest.json")
    parser.add_argument("--gt", default="benchmarks/ground_truth.json", help="Path to ground_truth.json")
    parser.add_argument("--output", default="evidence/plates/dataset_manifest.json", help="Path to output json")
    args = parser.parse_args()

    export_plate_dataset(args.manifest, args.gt, args.output)

if __name__ == "__main__":
    main()
