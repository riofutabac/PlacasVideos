#!/usr/bin/env python3
"""
Herramienta Forense de Video para Casos No Detectados (Fase 2).
Extrae fotogramas en intervalos de 1 segundo alrededor de las horas exactas
de los 3 vehículos no asociados (PAZ1513, PAB7630, PAB3439).
Genera hojas de contacto (contact sheets) para inspeccionar si los vehículos
aparecen físicamente en el video y por qué no fueron detectados por YOLO / FSM.
"""

import os
import cv2
import argparse
import numpy as np

CASES = [
    {
        'name': 'PAZ1513 (Camión JAC plataforma)',
        'clip_name': 'Camara Placas 2_20260909105651-20260909163038(31).mp4',
        'target_time_str': '13:39:00',
        'start_sec': 30.0,
        'end_sec': 110.0,
        'output_name': 'forense_PAZ1513_clip31.jpg'
    },
    {
        'name': 'PAB7630 (Mixer de concreto tambor)',
        'clip_name': 'Camara Placas 2_20260909105651-20260909163038(37).mp4',
        'target_time_str': '14:13:00',
        'start_sec': 140.0,
        'end_sec': 220.0,
        'output_name': 'forense_PAB7630_clip37.jpg'
    },
    {
        'name': 'PAB3439 (Camión furgón blanco)',
        'clip_name': 'Camara Placas 2_20260909105651-20260909163038(57).mp4',
        'target_time_str': '16:03:00',
        'start_sec': 110.0,
        'end_sec': 190.0,
        'output_name': 'forense_PAB3439_clip57.jpg'
    }
]

def make_contact_sheet(frames_with_labels, grid_cols=4, frame_w=360, frame_h=200):
    n_frames = len(frames_with_labels)
    if n_frames == 0:
        return None
    grid_rows = (n_frames + grid_cols - 1) // grid_cols
    
    sheet = np.zeros((grid_rows * frame_h, grid_cols * frame_w, 3), dtype=np.uint8)
    
    for idx, (frame, label) in enumerate(frames_with_labels):
        r = idx // grid_cols
        c = idx % grid_cols
        
        resized = cv2.resize(frame, (frame_w, frame_h))
        # Draw label banner
        cv2.rectangle(resized, (0, 0), (frame_w, 24), (0, 0, 0), -1)
        cv2.putText(resized, label, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        sheet[r*frame_h:(r+1)*frame_h, c*frame_w:(c+1)*frame_w] = resized
        
    return sheet

def process_case(case, video_dir, output_dir):
    clip_path = os.path.join(video_dir, case['clip_name'])
    if not os.path.exists(clip_path):
        print(f"⚠️ [Forense] No se encontró el video en: {clip_path}")
        return
        
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        print(f"❌ [Forense] Error abriendo video: {clip_path}")
        return
        
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_sec = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / fps
    print(f"🎬 [Forense] Procesando {case['name']} en {case['clip_name']} (FPS: {fps:.1f}, Duración: {total_sec:.1f}s)")
    
    frames_with_labels = []
    step_sec = 2.0  # Muestra cada 2 segundos para cubrir la ventana
    
    curr_t = case['start_sec']
    while curr_t <= case['end_sec']:
        f_idx = int(curr_t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret:
            break
        lbl = f"t={curr_t:.1f}s | {case['target_time_str']} ventana"
        frames_with_labels.append((frame, lbl))
        curr_t += step_sec
        
    cap.release()
    
    if frames_with_labels:
        sheet = make_contact_sheet(frames_with_labels, grid_cols=4, frame_w=400, frame_h=225)
        out_file = os.path.join(output_dir, case['output_name'])
        cv2.imwrite(out_file, sheet)
        print(f"  ✅ Guardada hoja de contacto con {len(frames_with_labels)} cuadros en: {out_file}")

def main():
    parser = argparse.ArgumentParser(description="Extrae hojas de contacto de video para los 3 casos no detectados")
    parser.add_argument("--video-dir", default="/content/drive/MyDrive/Cam PL", help="Directorio con los archivos MP4")
    parser.add_argument("--output-dir", default="reports/video_forensics", help="Directorio de salida")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    for case in CASES:
        process_case(case, args.video_dir, args.output_dir)

if __name__ == "__main__":
    main()
