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
import unicodedata
import openpyxl
from datetime import datetime, date, time, timedelta
from typing import List, Dict, Any, Optional

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ecuador_plate_validator import PROVINCE_CODES
from src.pipeline_types import get_clip_start_datetime
from src.deduplicator import levenshtein_distance

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

def resolve_audit_excel(excel_path: str) -> Optional[str]:
    """
    Resolves the audit workbook tolerating Unicode accent normalization.
    macOS stores the accent in 'Revision' decomposed (NFD) while Linux/Colab keeps it
    composed (NFC), so an exact path match fails on one of the two platforms.
    """
    if os.path.exists(excel_path):
        return excel_path

    target = unicodedata.normalize("NFC", os.path.basename(excel_path)).lower()
    search_dir = os.path.dirname(excel_path) or "."
    try:
        for name in os.listdir(search_dir):
            if unicodedata.normalize("NFC", name).lower() == target:
                return os.path.join(search_dir, name)
    except OSError:
        return None
    return None

def load_audit_excel(excel_path: str = "Revisión bypass Pintag.xlsx") -> List[Dict[str, Any]]:
    resolved = resolve_audit_excel(excel_path)
    if resolved is None:
        print(f"⚠️ [GT Builder] No se encontro el archivo Excel en {excel_path}")
        return []
    excel_path = resolved

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

def get_event_datetime(evt: Dict[str, Any]) -> Optional[datetime]:
    """Extracts datetime from event record, preferring OSD datetime_str with extrapolation fallback."""
    dt_str = evt.get("datetime_str")
    if dt_str:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
            try:
                return datetime.strptime(str(dt_str).strip(), fmt)
            except ValueError:
                pass
    v_src = evt.get("video_source") or ""
    ts = float(evt.get("event_timestamp") or 0.0)
    try:
        return get_clip_start_datetime(v_src) + timedelta(seconds=ts)
    except Exception:
        return None

NEAR_PLATE_MAX_DISTANCE = 2


def _load_db_events(db_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(db_path):
        return []
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM events WHERE duplicate_of IS NULL ORDER BY event_timestamp ASC")
    db_events = [dict(r) for r in cur.fetchall()]
    conn.close()
    return db_events


def _build_candidate_pairs(
    audit_entries: List[Dict[str, Any]],
    db_events: List[Dict[str, Any]],
    tolerance_seconds: float
) -> List[tuple]:
    """
    Builds every (audit_row, event) pair worth considering as a match, scored so a
    global greedy assignment (sorted by priority, then score) yields a one-to-one
    mapping. Priority: EXACT_PLATE (0) > NEAR_PLATE (1) > VEHICLE_ONLY (2). We do not
    depend on scipy for a true optimal assignment (linear_sum_assignment); sorting all
    candidate pairs by (priority, score) and greedily claiming unused rows/events is a
    documented approximation that is exact whenever priority tiers do not conflict,
    which holds for this dataset size (39 audit rows).
    """
    pairs = []
    for audit_idx, audit in enumerate(audit_entries):
        plate = audit["plate_text"]
        t2_val = parse_time_cell(audit.get("time_cam2"))
        t1_val = parse_time_cell(audit.get("time_cam1"))
        t_ref = t2_val or t1_val
        if t_ref is None:
            continue

        for evt_idx, evt in enumerate(db_events):
            evt_dt = get_event_datetime(evt)
            if evt_dt is None:
                continue
            audit_dt = datetime(evt_dt.year, evt_dt.month, evt_dt.day, t_ref.hour, t_ref.minute, t_ref.second)
            diff = abs((evt_dt - audit_dt).total_seconds())
            if diff > tolerance_seconds:
                continue

            det_p = (evt.get("plate_corrected") or evt.get("plate_normalized") or "").strip().upper()

            if det_p and det_p == plate:
                pairs.append((0, diff, audit_idx, evt_idx, diff, 0, "EXACT_PLATE"))
            elif det_p:
                dist = levenshtein_distance(plate, det_p)
                if dist <= NEAR_PLATE_MAX_DISTANCE:
                    pairs.append((1, dist * 1000.0 + diff, audit_idx, evt_idx, diff, dist, "NEAR_PLATE"))
            else:
                pairs.append((2, diff, audit_idx, evt_idx, diff, None, "VEHICLE_ONLY"))

    pairs.sort(key=lambda p: (p[0], p[1]))
    return pairs


def _assign_one_to_one(
    audit_entries: List[Dict[str, Any]],
    db_events: List[Dict[str, Any]],
    pairs: List[tuple]
) -> Dict[int, tuple]:
    """Greedily claims the best-scoring pair per audit row/event, one-to-one."""
    used_audit = set()
    used_evt = set()
    assignment: Dict[int, tuple] = {}

    for _priority, _score, audit_idx, evt_idx, diff, dist, match_type in pairs:
        if audit_idx in used_audit or evt_idx in used_evt:
            continue
        used_audit.add(audit_idx)
        used_evt.add(evt_idx)
        assignment[audit_idx] = (evt_idx, diff, dist, match_type)

    # Exact-plate fallback: plate matches literally but outside the time tolerance
    # (e.g. audit timestamp rounding error, clock drift). Still one-to-one.
    for audit_idx, audit in enumerate(audit_entries):
        if audit_idx in used_audit:
            continue
        plate = audit["plate_text"]
        for evt_idx, evt in enumerate(db_events):
            if evt_idx in used_evt:
                continue
            det_p = (evt.get("plate_corrected") or evt.get("plate_normalized") or "").strip().upper()
            if det_p == plate:
                used_audit.add(audit_idx)
                used_evt.add(evt_idx)
                assignment[audit_idx] = (evt_idx, None, 0, "EXACT_PLATE_FALLBACK")
                break

    return assignment


def match_with_database(
    audit_entries: List[Dict[str, Any]],
    db_path: str = "data/events.sqlite",
    tolerance_seconds: float = 75.0
) -> List[Dict[str, Any]]:
    db_events = _load_db_events(db_path)

    pairs = _build_candidate_pairs(audit_entries, db_events, tolerance_seconds)
    assignment = _assign_one_to_one(audit_entries, db_events, pairs)

    gt_candidates = []
    for idx, audit in enumerate(audit_entries):
        plate = audit["plate_text"]

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
            "edit_distance": None,
            "video_source": None,
            "approx_timestamp": None,
            "pipeline_detected_plate": None,
            "vehicle_crop_path": None,
            "plate_crop_path": None,
            "confidence_ocr": None
        }

        assigned = assignment.get(idx)
        if assigned is not None:
            evt_idx, diff, dist, match_type = assigned
            evt = db_events[evt_idx]
            cand_entry["matched_event_id"] = evt.get("event_id")
            cand_entry["match_type"] = match_type
            cand_entry["time_diff_sec"] = round(diff, 2) if diff is not None else None
            cand_entry["edit_distance"] = dist
            cand_entry["video_source"] = evt.get("video_source")
            cand_entry["approx_timestamp"] = evt.get("event_timestamp")
            cand_entry["pipeline_detected_plate"] = evt.get("plate_corrected") or evt.get("plate_normalized")
            cand_entry["vehicle_crop_path"] = evt.get("vehicle_crop_path")
            cand_entry["plate_crop_path"] = evt.get("plate_crop_path")
            cand_entry["confidence_ocr"] = evt.get("confidence_ocr")

        gt_candidates.append(cand_entry)

    return gt_candidates

def build_ground_truth_candidates(
    excel_path: str = "Revisión bypass Pintag.xlsx",
    db_path: str = "data/events.sqlite",
    output_path: str = "benchmarks/ground_truth_draft.json",
    tolerance_seconds: float = 75.0
) -> List[Dict[str, Any]]:
    audit_entries = load_audit_excel(excel_path)
    print(f"📋 [GT Builder] Leídas {len(audit_entries)} placas auditadas del Excel.")

    candidates = match_with_database(audit_entries, db_path, tolerance_seconds)
    n_total = len(candidates)
    n_exact = sum(1 for c in candidates if c["match_type"] in ("EXACT_PLATE", "EXACT_PLATE_FALLBACK"))
    n_near = sum(1 for c in candidates if c["match_type"] == "NEAR_PLATE")
    n_vehicle_only = sum(1 for c in candidates if c["match_type"] == "VEHICLE_ONLY")
    n_missed = sum(1 for c in candidates if not c["match_type"])

    db_total = 0
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM events WHERE duplicate_of IS NULL")
        db_total = cur.fetchone()[0]
        conn.close()

    # matched_event_id is unique per event thanks to the one-to-one assignment, so this
    # count of "events consumed by an audit row" is exact, not an estimate.
    matched_ids = set(c["matched_event_id"] for c in candidates if c["matched_event_id"])
    extra_detected = max(0, db_total - len(matched_ids))

    if n_total == 0:
        print(f"\n⚠️ [GT Builder] No hay filas auditadas que cruzar; revisa la ruta del Excel (--excel).")
        print(f"  • Total eventos unicos en el sistema: {db_total}")
        return candidates

    print(f"\n📊 === Resumen de Cruce con Base de Datos ({db_path}) ===")
    print(f"  • Tolerancia temporal: {tolerance_seconds:.0f}s")
    print(f"  • Total vehículos auditados en Excel: {n_total}")
    print(f"  • Total eventos únicos en el sistema: {db_total}")
    print(f"  • Coincidencia Exacta de Placa: {n_exact} ({n_exact/n_total*100:.1f}%)")
    print(f"  • Coincidencia Cercana (1-{NEAR_PLATE_MAX_DISTANCE} caracteres, mismo vehículo con error OCR): {n_near} ({n_near/n_total*100:.1f}%)")
    print(f"  • Solo Vehículo (detectado, placa no leída): {n_vehicle_only} ({n_vehicle_only/n_total*100:.1f}%)")
    print(f"  • No encontrados / Sin cruce: {n_missed} ({n_missed/n_total*100:.1f}%)")
    print(f"  • Eventos adicionales detectados (no listados en auditoría; esperado, la auditoría solo registra vehículos que cruzaron ambas cámaras): {extra_detected}\n")

    print(f"{'#':<4} {'Placa GT':<9} {'Tipo':<7} {'Hora Cam2':<10} {'Tipo Match':<16} {'Placa Pipeline':<15} {'Diff (s)':<9} {'Dist':<5} {'Conf OCR':<8}")
    print("-" * 96)
    for c in candidates:
        gid = c["ground_truth_id"].replace("GT_CAND_", "")
        plt_gt = c["plate_text"]
        v_type = c["vehicle_type"][:6]
        h_cam2 = c["audit_time_cam2"] or c["audit_time_cam1"] or "--:--:--"
        m_type = c["match_type"] or "NO_MATCH"
        p_pipe = c["pipeline_detected_plate"] or "-"
        diff_s = f"{c['time_diff_sec']}s" if c["time_diff_sec"] is not None else "-"
        dist_s = str(c["edit_distance"]) if c["edit_distance"] is not None else "-"
        conf_o = f"{c['confidence_ocr']:.2f}" if c["confidence_ocr"] is not None else "-"
        print(f"{gid:<4} {plt_gt:<9} {v_type:<7} {h_cam2:<10} {m_type:<16} {p_pipe:<15} {diff_s:<9} {dist_s:<5} {conf_o:<8}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(candidates, f, indent=2, ensure_ascii=False)

    print(f"\n💾 [GT Builder] Borrador de Ground Truth guardado en {output_path}")
    return candidates

def main():
    parser = argparse.ArgumentParser(description="Build ground truth candidate dataset from audit Excel")
    parser.add_argument("--excel", default="Revisión bypass Pintag.xlsx", help="Audit Excel path")
    parser.add_argument("--db", default="data/events.sqlite", help="SQLite database path")
    parser.add_argument("--output", default="benchmarks/ground_truth_draft.json", help="Output draft JSON path")
    parser.add_argument(
        "--time-tolerance", type=float, default=75.0, dest="time_tolerance",
        help=(
            "Max seconds between the audit's 'Hora cam 2' and a detected event's timestamp "
            "to be considered a match candidate (default: 75s). The audit time is rounded "
            "to the minute, so 60-90s is a sensible range; too large a tolerance produces "
            "false matches between unrelated vehicles."
        )
    )
    args = parser.parse_args()

    build_ground_truth_candidates(args.excel, args.db, args.output, args.time_tolerance)

if __name__ == "__main__":
    main()
