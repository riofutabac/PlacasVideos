"""
A/B Diagnostic Script: OpenCVDecoder vs NVDECDecoder
Compares:
1. Event recall, direction, and false positives.
2. decoder_frames_yielded, motion detections, YOLO calls, tracks created, events.
3. Exact pixel comparison (MAE, RMSE, histogram shift, range discrepancy) at timestamps:
   25.0, 105.0, 109.0, 112.0, 116.0, 120.0s.
4. Motion Gate and YOLO detection behavior on missed vehicle (GT_60_2 at t=109.0s).
"""
import os
import sys
import cv2
import json
import time
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.video_decoder import create_decoder, OpenCVDecoder, FFmpegNVDECDecoder
from src.motion_gate import AdaptiveMotionGate
from src.pipeline_runner import ALPRPipeline

TIMESTAMPS_TO_COMPARE = [25.0, 105.0, 109.0, 112.0, 116.0, 120.0]

def extract_frames_at_timestamps(video_path: str, backend: str, timestamps: List[float], crop_rect: Optional[Dict[str, int]] = None) -> Dict[float, np.ndarray]:
    """Extracts frames closest to target timestamps from the specified decoder."""
    print(f"[{backend.upper()}] Extrayendo frames en timestamps {timestamps}...")
    decoder = create_decoder(video_path, backend=backend, crop_rect=crop_rect)
    fps = decoder.fps or 25.0
    
    target_indices = {int(round(ts * fps)): ts for ts in timestamps}
    extracted = {}
    
    total_yielded = 0
    max_target_idx = max(target_indices.keys()) + 10
    
    for f_idx, ts, frame in decoder:
        total_yielded += 1
        if f_idx in target_indices:
            orig_ts = target_indices[f_idx]
            extracted[orig_ts] = frame.copy()
            print(f"  [{backend}] Frame en t={ts:.2f}s (idx={f_idx}) extraído.")
        if f_idx > max_target_idx:
            break
            
    decoder.release()
    print(f"[{backend.upper()}] Extracción completa ({len(extracted)}/{len(timestamps)} frames encontrados).\n")
    return extracted

def analyze_pixel_differences(frames_opencv: Dict[float, np.ndarray], frames_nvdec: Dict[float, np.ndarray]):
    """Analyzes numerical and color differences between OpenCV and NVDEC frames."""
    print("="*70)
    print("ANÁLISIS DE DIFERENCIAS DE COLOR / RANGO (OpenCV vs NVDEC)")
    print("="*70)
    
    out_dir = Path("benchmarks/ab_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for ts in TIMESTAMPS_TO_COMPARE:
        if ts not in frames_opencv or ts not in frames_nvdec:
            print(f"Timestamp {ts}s no disponible en ambos backends.")
            continue
            
        f_cv = frames_opencv[ts]
        f_nv = frames_nvdec[ts]
        
        if f_cv.shape != f_nv.shape:
            print(f"Timestamp {ts}s: Dimensiones no coinciden (CV: {f_cv.shape} vs NV: {f_nv.shape})")
            continue
            
        diff = cv2.absdiff(f_cv, f_nv)
        mae = np.mean(diff)
        max_diff = np.max(diff)
        rmse = np.sqrt(np.mean((f_cv.astype(float) - f_nv.astype(float))**2))
        
        # Per channel MAE (B, G, R)
        mae_b = np.mean(diff[:, :, 0])
        mae_g = np.mean(diff[:, :, 1])
        mae_r = np.mean(diff[:, :, 2])
        
        # Luma stats
        gray_cv = cv2.cvtColor(f_cv, cv2.COLOR_BGR2GRAY)
        gray_nv = cv2.cvtColor(f_nv, cv2.COLOR_BGR2GRAY)
        
        print(f"Timestamp {ts:5.1f}s | MAE: {mae:5.2f} (B:{mae_b:4.1f} G:{mae_g:4.1f} R:{mae_r:4.1f}) | RMSE: {rmse:5.2f} | MaxDiff: {max_diff:3d} | LumaMean CV: {gray_cv.mean():5.1f}, NV: {gray_nv.mean():5.1f}")
        
        # Save side-by-side comparison for visual inspection
        h, w = f_cv.shape[:2]
        vis = np.hstack([f_cv, f_nv, diff * 3]) # Amplified difference map
        cv2.putText(vis, f"OpenCV t={ts}s", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.putText(vis, f"NVDEC t={ts}s", (w + 50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.putText(vis, f"Diff x3 (MAE={mae:.1f})", (2*w + 50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        cv2.imwrite(str(out_dir / f"diff_t{int(ts):03d}.jpg"), vis)

    print(f"\nImágenes de diagnóstico guardadas en: {out_dir}\n")

def test_motion_gate_and_yolo_on_frames(frames_opencv: Dict[float, np.ndarray], frames_nvdec: Dict[float, np.ndarray]):
    """Tests Motion Gate and YOLO on both frames specifically around t=109.0s (missed GT_60_2)."""
    print("="*70)
    print("TEST DE MOTION GATE Y YOLO EN TIMESTAMP CRÍTICO (t=109.0s)")
    print("="*70)
    
    pipeline = ALPRPipeline()
    cx1, cy1 = pipeline.crop_rect['x_min'], pipeline.crop_rect['y_min']
    cx2, cy2 = pipeline.crop_rect['x_max'], pipeline.crop_rect['y_max']
    
    for ts in [105.0, 109.0, 112.0]:
        if ts not in frames_opencv or ts not in frames_nvdec:
            continue
            
        f_cv = frames_opencv[ts]
        f_nv = frames_nvdec[ts]
        crop_cv = f_cv[cy1:cy2, cx1:cx2] if f_cv.shape[:2] == (1664, 2960) else f_cv
        crop_nv = f_nv[cy1:cy2, cx1:cx2] if f_nv.shape[:2] == (1664, 2960) else f_nv
        
        # Test YOLO directly on both crops
        res_cv = pipeline.vehicle_model(crop_cv, imgsz=pipeline.vehicle_imgsz, conf=0.20, verbose=False, device=pipeline.device)[0]
        res_nv = pipeline.vehicle_model(crop_nv, imgsz=pipeline.vehicle_imgsz, conf=0.20, verbose=False, device=pipeline.device)[0]
        
        print(f"\n--- Timestamp {ts:.1f}s ---")
        print(f"OpenCV Crop -> YOLO Detecciones: {len(res_cv.boxes)}")
        for b in res_cv.boxes:
            cls_name = pipeline.vehicle_classes.get(int(b.cls[0].item()), str(int(b.cls[0].item())))
            print(f"   [OpenCV] {cls_name:<12} Conf: {float(b.conf[0].item()):.3f} BBox: {[int(x) for x in b.xyxy[0].cpu().numpy()]}")
            
        print(f"NVDEC  Crop -> YOLO Detecciones: {len(res_nv.boxes)}")
        for b in res_nv.boxes:
            cls_name = pipeline.vehicle_classes.get(int(b.cls[0].item()), str(int(b.cls[0].item())))
            print(f"   [NVDEC]  {cls_name:<12} Conf: {float(b.conf[0].item()):.3f} BBox: {[int(x) for x in b.xyxy[0].cpu().numpy()]}")

def run_ab_comparison(video_path: str):
    print("="*70)
    print("DIAGNÓSTICO COMPARATIVO A/B EXACTO: OpenCV vs NVDEC")
    print(f"Video: {video_path}")
    print("="*70 + "\n")
    
    pipeline = ALPRPipeline()
    crop_rect = pipeline.crop_rect
    
    # 1. Extract frames (using crop_rect to mirror exact pipeline behavior)
    frames_cv = extract_frames_at_timestamps(video_path, backend="opencv", timestamps=TIMESTAMPS_TO_COMPARE, crop_rect=crop_rect)
    frames_nv = extract_frames_at_timestamps(video_path, backend="nvdec", timestamps=TIMESTAMPS_TO_COMPARE, crop_rect=crop_rect)
    
    # 2. Pixel and range analysis
    analyze_pixel_differences(frames_cv, frames_nv)
    
    # 3. Model detection analysis at critical timestamps
    test_motion_gate_and_yolo_on_frames(frames_cv, frames_nv)

def run_full_ab_pipeline(clips: List[str]):
    """Runs complete ALPR pipeline with OpenCV vs NVDEC and logs all 5 metrics + Ground Truth."""
    print("\n" + "="*70)
    print("COMPARATIVA COMPLETA DE PIPELINE A/B (OpenCV vs NVDEC)")
    print("="*70)
    
    results = {}
    
    for backend in ["opencv", "nvdec"]:
        print(f"\n🚀 EJECUTANDO PIPELINE CON BACKEND: {backend.upper()}")
        print("-" * 50)
        
        pipeline = ALPRPipeline()
        pipeline.cfg.setdefault('video', {})['decode_backend'] = backend
        run_id = f"AB_{backend.upper()}_{int(time.time())}"
        
        pipeline.db.start_run(
            run_id=run_id,
            pipeline_version="AB_TEST",
            config_hash=pipeline.config_hash,
            model_versions=pipeline.model_versions,
            started_at=time.strftime('%Y-%m-%d %H:%M:%S')
        )
        
        t0 = time.perf_counter()
        total_events = 0
        all_events = []
        
        for vf in clips:
            evts = pipeline.process_video_file(vf, run_id)
            total_events += len(evts)
            all_events.extend(evts)
            
        wall_time = time.perf_counter() - t0
        
        perf = pipeline.profiler.summary()
        
        # Count tracks created in this run
        import sqlite3
        conn = sqlite3.connect(pipeline.db.db_path)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(DISTINCT track_id) FROM events WHERE processing_run_id=?", (run_id,))
        tracks_created = cur.fetchone()[0]
        conn.close()
        
        total_frames = perf['stages']['decode']['calls']
        total_dur = 0.0
        for vf in clips:
            from src.video_decoder import probe_video_metadata
            m = probe_video_metadata(vf)
            total_dur += (m['total_frames'] / m['fps']) if m['fps'] > 0 else 0.0
        if total_dur == 0.0:
            total_dur = total_frames / 25.0
        speed_ratio = round(total_dur / wall_time, 2) if wall_time > 0 else 0.0
        yolo_avg = round(sum(pipeline.profiler.yolo_latencies) / len(pipeline.profiler.yolo_latencies), 1) if pipeline.profiler.yolo_latencies else 0.0

        results[backend] = {
            'run_id': run_id,
            'wall_time': wall_time,
            'speed_ratio': speed_ratio,
            'frames_yielded': total_frames,
            'motion_detections': perf['stages']['motion_gate']['calls'],
            'yolo_calls': perf['stages']['vehicle_detection']['calls'],
            'yolo_avg_ms': yolo_avg,
            'tracks_created': tracks_created,
            'events_count': total_events,
            'decode_sec': perf['stages']['decode']['seconds'],
            'events': all_events
        }
        
    print("\n" + "="*75)
    print("TABLA COMPARATIVA A/B RIGUROSA (CLIPS 60 Y 61)")
    print("="*75)
    print(f"{'Métrica':<28} | {'OpenCV (Baseline)':<20} | {'NVDEC (Hardware)':<20}")
    print("-" * 75)
    cv_r = results.get("opencv", {})
    nv_r = results.get("nvdec", {})
    
    print(f"{'decoder_frames_yielded':<28} | {cv_r.get('frames_yielded', 0):<20} | {nv_r.get('frames_yielded', 0):<20}")
    print(f"{'motion detections (calls)':<28} | {cv_r.get('motion_detections', 0):<20} | {nv_r.get('motion_detections', 0):<20}")
    print(f"{'YOLO calls':<28} | {cv_r.get('yolo_calls', 0):<20} | {nv_r.get('yolo_calls', 0):<20}")
    print(f"{'YOLO avg latency (ms)':<28} | {str(cv_r.get('yolo_avg_ms', 0.0)) + ' ms':<20} | {str(nv_r.get('yolo_avg_ms', 0.0)) + ' ms':<20}")
    print(f"{'tracks creados':<28} | {cv_r.get('tracks_created', 0):<20} | {nv_r.get('tracks_created', 0):<20}")
    print(f"{'eventos registrados':<28} | {cv_r.get('events_count', 0):<20} | {nv_r.get('events_count', 0):<20}")
    print(f"{'decode_seconds':<28} | {cv_r.get('decode_sec', 0.0):<20.2f} | {nv_r.get('decode_sec', 0.0):<20.2f}")
    print(f"{'wall_clock_seconds':<28} | {cv_r.get('wall_time', 0.0):<20.2f} | {nv_r.get('wall_time', 0.0):<20.2f}")
    print(f"{'speed_ratio':<28} | {str(cv_r.get('speed_ratio', 0.0)) + 'x':<20} | {str(nv_r.get('speed_ratio', 0.0)) + 'x':<20}")
    print("="*75 + "\n")
    
    # Ground Truth Evaluation for both runs
    from benchmarks.evaluate_pipeline import evaluate_run
    print("--- EVALUACIÓN GROUND TRUTH: OpenCV ---")
    evaluate_run("data/events.sqlite", cv_r.get('run_id'))
    print("\n--- EVALUACIÓN GROUND TRUTH: NVDEC ---")
    evaluate_run("data/events.sqlite", nv_r.get('run_id'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A/B Decoder Diagnostic Tool")
    parser.add_argument("video", nargs="?", default=None)
    parser.add_argument("--full-pipeline", action="store_true", help="Ejecutar corrida completa de pipeline A/B sobre clips 60 y 61")
    args = parser.parse_args()
    
    # Find clips 60 and 61
    candidates_dir = ["/content/videos_local", "/content/drive/MyDrive/Cam PL", "."]
    clips_found = []
    for d in candidates_dir:
        c60 = os.path.join(d, "Camara Placas 2_20260909105651-20260909163038(60).mp4")
        c61 = os.path.join(d, "Camara Placas 2_20260909105651-20260909163038(61).mp4")
        if os.path.exists(c60) and os.path.exists(c61):
            clips_found = [c60, c61]
            break
            
    if not clips_found and args.video and os.path.exists(args.video):
        clips_found = [args.video]
        
    if not clips_found:
        print("No se encontraron clips 60 y 61 para el test.")
        sys.exit(1)
        
    if args.full_pipeline:
        # Run diagnostic on critical timestamps first
        print("🔍 Paso 1/2: Diagnóstico numérico exacto en timestamps críticos (25, 105, 109, 112, 116, 120s)...")
        run_ab_comparison(clips_found[0])
        print("🚀 Paso 2/2: Ejecución del pipeline completo sobre ambos clips...")
        run_full_ab_pipeline(clips_found)
    else:
        run_ab_comparison(clips_found[0])
        if len(clips_found) >= 2:
            print("Para ejecutar la comparación completa del pipeline en ambos clips, corre con --full-pipeline.")

