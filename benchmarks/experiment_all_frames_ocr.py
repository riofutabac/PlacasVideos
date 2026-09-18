#!/usr/bin/env python3
"""
Experimento T-P1 y T-P6: Análisis exhaustivo de todos los cuadros de vehículos.
Extrae TODOS los cuadros de paso de los 3 vehículos con error de 1 carácter:
- PCW2492 (Clip 60, ~segundo 25 a 30) -> GT: PCW2492, Det: PCW2497
- PCG3981 (Clip 60, ~segundo 109 a 115) -> GT: PCG3981, Det: ECG3981
- PAC2573 (Clip 61, ~segundo 77 a 84) -> GT: PAC2573, Det: AAC2573
"""

import os
import sys
import cv2
import json
import numpy as np

# Ensure project root in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fast_alpr import ALPR

def run_all_frames_analysis():
    print("🔬 Iniciando Experimento T-P1: Extracción y OCR de TODOS los cuadros...")

    print("⏳ Cargando FastALPR...")
    alpr = ALPR(
        detector_model="yolo-v9-t-512-license-plate-end2end",
        detector_conf_thresh=0.20,
        ocr_model="cct-xs-v2-global-model"
    )

    targets = [
        {
            "id": "PCW2492",
            "video": "Camara Placas 2_20260909105651-20260909163038(60).mp4",
            "start_sec": 24.5,
            "end_sec": 30.5,
            "gt_plate": "PCW2492",
            "critical_char_idx": 6,  # 7th char: '2' vs '7'
            "target_char": "2",
            "roi": (0, 600, 2960, 1664) # (x1, y1, x2, y2)
        },
        {
            "id": "PCG3981",
            "video": "Camara Placas 2_20260909105651-20260909163038(60).mp4",
            "start_sec": 109.0,
            "end_sec": 115.0,
            "gt_plate": "PCG3981",
            "critical_char_idx": 0,  # 1st char: 'P' vs 'E'
            "target_char": "P",
            "roi": (0, 600, 2960, 1664)
        },
        {
            "id": "PAC2573",
            "video": "Camara Placas 2_20260909105651-20260909163038(61).mp4",
            "start_sec": 77.0,
            "end_sec": 83.5,
            "gt_plate": "PAC2573",
            "critical_char_idx": 0,  # 1st char: 'P' vs 'A'
            "target_char": "P",
            "roi": (0, 600, 2960, 1664)
        }
    ]

    out_dir = "benchmarks/experiments/all_frames_t_p1"
    os.makedirs(out_dir, exist_ok=True)

    summary_results = {}

    for tgt in targets:
        v_path = tgt["video"]
        if not os.path.exists(v_path):
            print(f"⚠️ Video {v_path} no encontrado.")
            continue

        cap = cv2.VideoCapture(v_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        start_frame = int(tgt["start_sec"] * fps)
        end_frame = int(tgt["end_sec"] * fps)

        print(f"\n🚗 Analizando {tgt['id']} (frames {start_frame} a {end_frame}, {end_frame - start_frame} cuadros)...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frame_idx = start_frame
        found_target_char = False
        best_exact_match = None
        analyzed_frames = 0
        detected_plates_count = 0

        target_records = []
        rx1, ry1, rx2, ry2 = tgt["roi"]

        while frame_idx <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            roi_frame = frame[ry1:ry2, rx1:rx2]
            analyzed_frames += 1

            try:
                dets = alpr.detector.predict(roi_frame)
            except Exception as e:
                dets = []

            for d_idx, d in enumerate(dets):
                bb = d.bounding_box
                bx1, by1, bx2, by2 = int(bb.x1), int(bb.y1), int(bb.x2), int(bb.y2)
                bw, bh = bx2 - bx1, by2 - by1
                if bw < 25 or bh < 12:
                    continue

                pad_x, pad_y = int(bw * 0.12), int(bh * 0.12)
                px1 = max(0, bx1 - pad_x)
                py1 = max(0, by1 - pad_y)
                px2 = min(roi_frame.shape[1], bx2 + pad_x)
                py2 = min(roi_frame.shape[0], by2 + pad_y)

                p_crop = roi_frame[py1:py2, px1:px2]
                if p_crop.size == 0:
                    continue

                detected_plates_count += 1

                try:
                    ocr_res = alpr.ocr.predict(p_crop)
                    read_text = ocr_res.text.strip().upper() if ocr_res and ocr_res.text else ""
                    confs = [float(c) for c in ocr_res.confidence] if ocr_res and ocr_res.confidence else []
                except Exception:
                    read_text = ""
                    confs = []

                has_char = False
                c_idx = tgt["critical_char_idx"]
                if len(read_text) > c_idx:
                    if read_text[c_idx] == tgt["target_char"]:
                        has_char = True
                        found_target_char = True

                is_exact = (read_text == tgt["gt_plate"])
                if is_exact and best_exact_match is None:
                    best_exact_match = (frame_idx, read_text)

                crop_fn = f"{tgt['id']}_f{frame_idx}_{d_idx}_{read_text}_c{int(d.confidence*100)}.jpg"
                crop_path = os.path.join(out_dir, crop_fn)
                cv2.imwrite(crop_path, p_crop)

                target_records.append({
                    "frame": frame_idx,
                    "time_sec": round(frame_idx / fps, 2),
                    "plate_text": read_text,
                    "confs": confs,
                    "det_conf": round(float(d.confidence), 3),
                    "crop_size": f"{p_crop.shape[1]}x{p_crop.shape[0]}",
                    "has_target_char": has_char,
                    "is_exact": is_exact,
                    "crop_path": crop_path
                })

            frame_idx += 1

        cap.release()

        summary_results[tgt["id"]] = {
            "gt_plate": tgt["gt_plate"],
            "analyzed_frames": analyzed_frames,
            "detected_plates": detected_plates_count,
            "found_target_char": found_target_char,
            "exact_match_found": (best_exact_match is not None),
            "best_exact_match": best_exact_match,
            "records": target_records
        }

        print(f"  📊 Resultados para {tgt['id']}:")
        print(f"     Cuadros analizados: {analyzed_frames}, Recortes de placa hallados: {detected_plates_count}")
        print(f"     ¿Apareció el carácter crítico '{tgt['target_char']}' en algún frame?: {'✅ SÍ' if found_target_char else '❌ NO'}")
        print(f"     ¿Hubo lectura 100% exacta?: {'✅ SÍ: ' + str(best_exact_match) if best_exact_match else '❌ NO'}")

        # Sorted by crop area
        sorted_by_size = sorted(target_records, key=lambda x: int(x["crop_size"].split('x')[0]) * int(x["crop_size"].split('x')[1]), reverse=True)
        print("     Top 5 recortes más grandes:")
        for r in sorted_by_size[:5]:
            print(f"       Frame {r['frame']} ({r['time_sec']}s) | Size: {r['crop_size']} | Det: {r['det_conf']} | Text: '{r['plate_text']}'")

    summary_file = os.path.join(out_dir, "summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        # Don't dump full records in summary to keep it clean
        clean_summary = {k: {sk: sv for sk, sv in v.items() if sk != "records"} for k, v in summary_results.items()}
        json.dump(clean_summary, f, indent=2)

    return summary_results

if __name__ == "__main__":
    run_all_frames_analysis()
