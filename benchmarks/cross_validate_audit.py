#!/usr/bin/env python3
"""
Cross-validation script: Compares pipeline detections against 'Revisión bypass Pintag.xlsx'.
Usage:
    python benchmarks/cross_validate_audit.py [--db data/events.sqlite] [--excel "Revisión bypass Pintag.xlsx"]
"""

import os
import re
import sys
import argparse
import sqlite3
from datetime import datetime, time, timedelta
import openpyxl

def get_clip_start_datetime(video_filename: str) -> datetime:
    base = os.path.basename(video_filename)
    match_date = re.search(r'_(\d{4})(\d{2})(\d{2})', base)
    if match_date:
        year, month, day = int(match_date.group(1)), int(match_date.group(2)), int(match_date.group(3))
    else:
        year, month, day = 2026, 9, 9

    match = re.search(r'\((\d+)\)\.mp4$', base)
    if match:
        clip_num = int(match.group(1))
        if clip_num == 60:
            return datetime(year, month, day, 16, 17, 20)
        elif clip_num == 61:
            return datetime(year, month, day, 16, 22, 45)
        else:
            base_time = datetime(year, month, day, 16, 17, 20)
            return base_time + timedelta(seconds=(clip_num - 60) * 325.0)
    return datetime(year, month, day, 16, 0, 0)

def parse_args():
    parser = argparse.ArgumentParser(description="Cross-validate pipeline events with manual bypass audit")
    parser.add_argument("--db", default="data/events.sqlite", help="Path to SQLite events database")
    parser.add_argument("--audit-excel", default="Revisión bypass Pintag.xlsx", help="Path to manual audit Excel")
    parser.add_argument("--time-tolerance-min", type=float, default=3.0, help="Time tolerance window in minutes for matching")
    return parser.parse_args()

def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

def plates_match(p1: str, p2: str) -> bool:
    if not p1 or not p2:
        return False
    p1, p2 = p1.strip().upper(), p2.strip().upper()
    if p1 == p2:
        return True
    if len(p1) >= 6 and len(p2) >= 6:
        # Check edit distance
        if levenshtein_distance(p1, p2) <= 1:
            return True
        # Check substring
        if p1 in p2 or p2 in p1:
            return True
        # Check last 4 digits match and length matches
        if p1[-4:] == p2[-4:] and abs(len(p1) - len(p2)) <= 1:
            return True
    return False

def load_audit_excel(filepath: str):
    if not os.path.exists(filepath):
        print(f"❌ Error: Archivo de auditoría no encontrado en '{filepath}'")
        return []

    wb = openpyxl.load_workbook(filepath, data_only=True)
    ws = wb.active
    audit_rows = []
    
    for idx, row in enumerate(ws.iter_rows(values_only=True)):
        if idx == 0:
            continue
        placa, tipo, fecha, h1, h2, perm, f1, f2, ejes, detalle, clip, arch, fila = row[:13]
        if not placa and not h2:
            continue
        
        # Parse h2 into a time or datetime
        h2_dt = None
        if isinstance(h2, time):
            h2_dt = datetime.combine(datetime(2026, 9, 9).date(), h2)
        elif isinstance(h2, datetime):
            h2_dt = h2
        elif isinstance(h2, str):
            try:
                parts = [int(p) for p in h2.strip().split(":")]
                if len(parts) == 2:
                    h2_dt = datetime(2026, 9, 9, parts[0], parts[1], 0)
                elif len(parts) == 3:
                    h2_dt = datetime(2026, 9, 9, parts[0], parts[1], parts[2])
            except Exception:
                pass

        audit_rows.append({
            "row_idx": idx + 1,
            "plate": str(placa).strip().upper() if placa else None,
            "vehicle_type": str(tipo).strip() if tipo else None,
            "ejes": ejes,
            "hora_cam1": str(h1) if h1 else None,
            "hora_cam2": str(h2) if h2 else None,
            "cam2_datetime": h2_dt,
            "detalle": str(detalle) if detalle else ""
        })
    return audit_rows

def load_detected_events(db_path: str):
    if not os.path.exists(db_path):
        print(f"⚠️ Base de datos '{db_path}' no existe todavía.")
        return []

    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
        SELECT event_id, video_source, datetime_str, direction, vehicle_type, 
               plate_raw, plate_normalized, plate_status, confidence_consensus
        FROM events
        ORDER BY datetime_str ASC
    """)
    events = []
    for r in c.fetchall():
        dt = None
        if r[2]:
            try:
                dt = datetime.strptime(r[2], "%Y-%m-%d %H:%M:%S")
            except Exception:
                pass
        events.append({
            "event_id": r[0],
            "video_source": r[1],
            "datetime_str": r[2],
            "datetime": dt,
            "direction": r[3],
            "vehicle_type": r[4],
            "plate_raw": r[5],
            "plate_normalized": r[6],
            "plate_status": r[7],
            "confidence": r[8]
        })
    conn.close()
    return events

def run_cross_validation(db_path: str, audit_path: str, tolerance_min: float = 3.0):
    audit_items = load_audit_excel(audit_path)
    detected_items = load_detected_events(db_path)

    print("=" * 80)
    print("CRUCE DE VALIDACIÓN: DETECCIONES DEL PIPELINE vs AUDITORÍA MANUAL PINTAG")
    print("=" * 80)
    print(f"📊 Registros en Excel humano: {len(audit_items)}")
    print(f"📹 Eventos en base de datos:  {len(detected_items)}")
    print(f"⏱️ Ventana de tolerancia temporal: ±{tolerance_min:.1f} minutos")
    print("-" * 80)

    if not audit_items:
        return

    # Determine which clips were actually processed
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT file_path, duration_sec FROM processed_clips")
    processed_clips = c.fetchall()
    conn.close()

    clip_windows = []
    for v_src, dur in processed_clips:
        start_dt = get_clip_start_datetime(v_src)
        end_dt = start_dt + timedelta(seconds=dur if dur else 330.0)
        clip_windows.append((v_src, start_dt, end_dt))

    if clip_windows:
        print("📹 Clips procesados y sus ventanas horarias:")
        for v_src, s, e in clip_windows:
            print(f"   • {v_src}: {s.strftime('%H:%M:%S')} ➔ {e.strftime('%H:%M:%S')}")
    else:
        print("⚠️ No hay clips registrados en processed_clips.")

    print("-" * 80)

    matches = []
    audit_in_window = []
    audit_outside_window = []

    for item in audit_items:
        dt2 = item["cam2_datetime"]
        if not dt2:
            audit_outside_window.append((item, "Sin hora en Cam 2"))
            continue

        in_window = False
        matching_clip = None
        for v_src, s, e in clip_windows:
            # Add tolerance window
            if (s - timedelta(minutes=tolerance_min)) <= dt2 <= (e + timedelta(minutes=tolerance_min)):
                in_window = True
                matching_clip = v_src
                break

        if not in_window:
            audit_outside_window.append((item, f"Hora Cam2 {item['hora_cam2']} no cubierta por clips actuales"))
            continue

        audit_in_window.append(item)

        # Try to find a match in detected_items
        best_match = None
        match_type = None

        for det in detected_items:
            # Match 1: plate match
            p_match = plates_match(item["plate"], det["plate_normalized"]) or plates_match(item["plate"], det["plate_raw"])
            
            # Match 2: time match
            time_match = False
            if det["datetime"]:
                diff_sec = abs((det["datetime"] - dt2).total_seconds())
                if diff_sec <= tolerance_min * 60:
                    time_match = True

            if p_match and time_match:
                best_match = det
                match_type = "PLACA_Y_HORA (EXACTO)"
                break
            elif p_match:
                best_match = det
                match_type = "SOLO_PLACA"
            elif time_match and best_match is None:
                best_match = det
                match_type = "SOLO_HORA"

        matches.append((item, best_match, match_type, matching_clip))

    print(f"\n🎯 [EVALUACIÓN EN LA VENTANA PROCESADA] ({len(audit_in_window)} vehículos de la auditoría humana):")
    if not audit_in_window:
        print("  Ningún vehículo del Excel humano cayó en el rango horario de los clips evaluados.")
    else:
        detected_count = 0
        for item, det, m_type, m_clip in matches:
            if det and "PLACA" in m_type:
                detected_count += 1
                placa_det = det["plate_normalized"] or det["plate_raw"] or "(Sin placa)"
                dt_det_str = det["datetime_str"].split(" ")[-1] if det["datetime_str"] else "N/A"
                print(f"  ✅ MATCH [{m_type}]:")
                print(f"     Humano:   Fila {item['row_idx']} | Placa: {item['plate']:<8} | Cam2: {item['hora_cam2']} | Tipo: {item['vehicle_type']} | {item['detalle'][:35]}")
                print(f"     Pipeline: Evento {det['event_id']} | Placa: {placa_det:<8} | Hora: {dt_det_str} | Tipo: {det['vehicle_type']} | Sentido: {det['direction']}")
            elif det and "SOLO_HORA" in m_type:
                placa_det = det["plate_normalized"] or det["plate_raw"] or "(Sin placa)"
                dt_det_str = det["datetime_str"].split(" ")[-1] if det["datetime_str"] else "N/A"
                print(f"  ⚠️ COINCIDENCIA POR HORA (Placa difiere):")
                print(f"     Humano:   Fila {item['row_idx']} | Placa: {item['plate']:<8} | Cam2: {item['hora_cam2']} | Tipo: {item['vehicle_type']} | {item['detalle'][:35]}")
                print(f"     Pipeline: Evento {det['event_id']} | Placa: {placa_det:<8} | Hora: {dt_det_str} | Tipo: {det['vehicle_type']} | Sentido: {det['direction']}")
            else:
                print(f"  ❌ NO DETECTADO:")
                print(f"     Humano:   Fila {item['row_idx']} | Placa: {item['plate']:<8} | Cam2: {item['hora_cam2']} | Tipo: {item['vehicle_type']} | {item['detalle'][:35]}")

        recall = (detected_count / len(audit_in_window)) * 100.0
        print(f"\n📈 RECALL AUDITORÍA EN LA VENTANA: {detected_count}/{len(audit_in_window)} ({recall:.1f}%)")

    print(f"\n⏳ [VEHÍCULOS DEL EXCEL EN OTROS CLIPS DEL DÍA] ({len(audit_outside_window)} registros):")
    # Group by approximate time
    for item, reason in audit_outside_window[:10]:
        print(f"  • Fila {item['row_idx']:2d}: Placa {item['plate']:<8} | Hora Cam 2: {str(item['hora_cam2']):<8} | {item['detalle'][:40]}")
    if len(audit_outside_window) > 10:
        print(f"  ... y {len(audit_outside_window) - 10} vehículos más que se verificarán al procesar el lote completo (clips 00 a 59).")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    args = parse_args()
    run_cross_validation(args.db, args.audit_excel, args.time_tolerance_min)
