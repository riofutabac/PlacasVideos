#!/usr/bin/env python3
"""
Ground Truth Candidate Builder (Fase B1).
Reads 'Revisión bypass Pintag.xlsx' and pipeline events from SQLite/manifest.
Generates an enriched ground truth candidate draft (benchmarks/ground_truth_draft.json)
associating audited vehicle crops and OCR reads for user review and confirmation.
"""

import os
import sys
import re
import json
import sqlite3
import argparse
import openpyxl
from datetime import datetime, date, time, timedelta
from typing import List, Dict, Any, Optional

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ecuador_plate_validator import PROVINCE_CODES
from src.pipeline_types import get_clip_start_datetime

def parse_time_cell(val: Any) -> Optional[time]:
    if isinstance(val, time):
        return val
    if isinstance(val, datetime):
        return val.time()
    if isinstance(val, str):
        val = val.strip()
        for fmt in ("%H:%M:%S", "%H:%M"):
            try:
                return datetime.strptime(val, fmt).time()
            except ValueError:
                pass
    return None

def load_audit_excel(excel_path: str = "Revisión bypass Pintag.xlsx") -> List[Dict[str, Any]]:
    if not os.path.exists(excel_path):
        print(f"⚠️ [GT Builder] No se encontró el archivo Excel en {excel_path}")
        return []

    wb = openpyxl.load_workbook(excel_path, data_only=True)
    ws = wb.active

    rows = list(ws.iter_rows(values_only=True))
    if not rows:
        return []

    entries = []
    for row_idx, row in enumerate(rows[1:], start=2):
        if not row or not row[0]:
            continue
        raw_plate = str(row[0]).strip().upper()
        clean_plate = re.sub(r'[^A-Z0-9]', '', raw_plate)
        if not clean_plate:
            continue

        v_type = str(row[1]).strip().lower() if len(row) > 1 and row[1] else "car"
        t_cam1 = parse_time_cell(row[3]) if len(row) > 3 else None
        t_cam2 = parse_time_cell(row[4]) if len(row) > 4 else None
        note = str(row[9]).strip() if len(row) > 9 and row[9] else ""

        first_char = clean_plate[0] if clean_plate else ""
        province = PROVINCE_CODES.get(first_char, "Desconocida")

        entries.append({
            "row_idx": row_idx,
            "plate_text": clean_plate,
            "vehicle_type": v_type,
            "time_cam1": t_cam1.strftime("%H:%M:%S") if t_cam1 else None,
            "time_cam2": t_cam2.strftime("%H:%M:%S") if t_cam2 else None,
            "note": note,
            "province": province
        })

    return entries

def match_with_database(
    audit_entries: List[Dict[str, Any]],
    db_path: str = "data/events.sqlite",
    tolerance_minutes: float = 3.0
) -> List[Dict[str, Any]]:
    db_events = []
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT * FROM events WHERE duplicate_of IS NULL ORDER BY event_timestamp ASC")
        db_events = [dict(r) for r in cur.fetchall()]
        conn.close()

    gt_candidates = []

    for idx, audit in enumerate(audit_entries):
        plate = audit["plate_text"]
        t2_val = parse_time_cell(audit.get("time_cam2"))

        cand_entry = {
            "ground_truth_id": f"GT_CAND_{idx+1:03d}",
            "plate_text": plate,
            "plate_legible": True,
            "vehicle_type": audit["vehicle_type"],
            "province": audit["province"],
            "audit_note": audit["note"],
            "audit_time_cam1": audit["time_cam1"],
            "audit_time_cam2": audit["time_cam2"],
            "matched_event_id": None,
            "match_type": None,
            "time_diff_sec": None,
            "video_source": None,
            "approx_timestamp": None,
            "pipeline_detected_plate": None,
            "vehicle_crop_path": None,
            "plate_crop_path": None,
            "confidence_ocr": None
        }

        # Match strategy:
        # 1. First priority: temporal proximity to audit_time_cam2 (camera 2 ground truth time) + plate/type match
        # 2. Second priority: exact plate match (fallback if timestamp not available or test database)
        best_match = None
        best_diff = 999999.0
        match_type = None

        if t2_val is not None:
            audit_dt = datetime(2026, 9, 9, t2_val.hour, t2_val.minute, t2_val.second)
            for evt in db_events:
                v_src = evt.get("video_source") or ""
                ts = float(evt.get("event_timestamp") or 0.0)
                evt_dt = get_clip_start_datetime(v_src) + timedelta(seconds=ts)
                diff = abs((evt_dt - audit_dt).total_seconds())
                det_p = (evt.get("plate_corrected") or evt.get("plate_normalized") or "").strip().upper()
                if diff <= tolerance_minutes * 60:
                    if det_p == plate:
                        best_match = evt
                        best_diff = diff
                        match_type = "EXACT_PLATE_AND_TIME"
                        break
                    elif diff < best_diff:
                        best_match = evt
                        best_diff = diff
                        match_type = "TIME_WINDOW"

        if not best_match:
            for evt in db_events:
                det_p = (evt.get("plate_corrected") or evt.get("plate_normalized") or "").strip().upper()
                if det_p == plate:
                    best_match = evt
                    match_type = "EXACT_PLATE_FALLBACK"
                    best_diff = None
                    break

        if best_match:
            cand_entry["matched_event_id"] = best_match.get("event_id")
            cand_entry["match_type"] = match_type
            cand_entry["time_diff_sec"] = round(best_diff, 2) if best_diff is not None and best_diff < 999999 else None
            cand_entry["video_source"] = best_match.get("video_source")
            cand_entry["approx_timestamp"] = best_match.get("event_timestamp")
            cand_entry["pipeline_detected_plate"] = best_match.get("plate_corrected")
            cand_entry["vehicle_crop_path"] = best_match.get("vehicle_crop_path")
            cand_entry["plate_crop_path"] = best_match.get("plate_crop_path")
            cand_entry["confidence_ocr"] = best_match.get("confidence_ocr")

        gt_candidates.append(cand_entry)

    return gt_candidates

def build_ground_truth_candidates(
    excel_path: str = "Revisión bypass Pintag.xlsx",
    db_path: str = "data/events.sqlite",
    output_path: str = "benchmarks/ground_truth_draft.json"
) -> List[Dict[str, Any]]:
    audit_entries = load_audit_excel(excel_path)
    print(f"📋 [GT Builder] Leídas {len(audit_entries)} placas auditadas del Excel.")

    candidates = match_with_database(audit_entries, db_path)
    matched_count = sum(1 for c in candidates if c["matched_event_id"])
    print(f"🔗 [GT Builder] {matched_count}/{len(candidates)} eventos cruzados con la base de datos local.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(candidates, f, indent=2, ensure_ascii=False)

    print(f"💾 [GT Builder] Borrador de Ground Truth guardado en {output_path}")
    return candidates

def main():
    parser = argparse.ArgumentParser(description="Build ground truth candidate dataset from audit Excel")
    parser.add_argument("--excel", default="Revisión bypass Pintag.xlsx", help="Audit Excel path")
    parser.add_argument("--db", default="data/events.sqlite", help="SQLite database path")
    parser.add_argument("--output", default="benchmarks/ground_truth_draft.json", help="Output draft JSON path")
    args = parser.parse_args()

    build_ground_truth_candidates(args.excel, args.db, args.output)

if __name__ == "__main__":
    main()
