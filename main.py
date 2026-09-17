"""
Main Execution Script for Event-Oriented ALPR Pipeline.
Usage:
  python main.py
Processes sample clips (60) and (61), outputs metrics summary, persists to SQLite,
and exports the audit-ready Excel report with embedded photos.
"""
import os
import sys
import glob
import time
from datetime import datetime
from src.pipeline_runner import ALPRPipeline, PIPELINE_VERSION
from src.excel_exporter import ExcelReportExporter

def run_full_pipeline():
    print("="*70)
    print(f"PIPELINE ALPR LIGERO ORIENTADO A EVENTOS v{PIPELINE_VERSION}")
    print(f"Inicio de corrida: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    pipeline = ALPRPipeline()
    run_id = f"RUN_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Start run in DB
    pipeline.db.start_run(
        run_id=run_id,
        pipeline_version=PIPELINE_VERSION,
        config_hash=pipeline.config_hash,
        model_versions=pipeline.model_versions,
        started_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    )

    # Find videos
    video_files = sorted(glob.glob("Camara Placas 2_*.mp4"))
    if not video_files:
        print("No se encontraron archivos de video con patrón 'Camara Placas 2_*.mp4'.")
        sys.exit(1)

    print(f"Videos a procesar ({len(video_files)} archivos):")
    for vf in video_files:
        print(f"  - {vf}")

    total_events_all = 0
    total_source_frames = 0
    total_duration_sec = 0.0

    t_start_wall = time.perf_counter()

    for vf in video_files:
        # Collect video duration
        import cv2
        cap = cv2.VideoCapture(vf)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        total_source_frames += frames
        total_duration_sec += (frames / fps)

        events = pipeline.process_video_file(vf, run_id)
        total_events_all += len(events)

    t_total_wall = time.perf_counter() - t_start_wall
    pipeline.profiler.finish(total_source_frames, total_duration_sec)

    speed_ratio = total_duration_sec / t_total_wall if t_total_wall > 0 else 0.0

    # Finish run in DB
    pipeline.db.finish_run(
        run_id=run_id,
        finished_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        total_videos=len(video_files),
        total_events=total_events_all,
        speed_ratio=round(speed_ratio, 2),
        status="COMPLETED"
    )

    # Print profiling table
    pipeline.profiler.print_summary()

    # Print thermal tracking table
    blocks = pipeline.profiler.get_yolo_block_latencies(block_size=200)
    if blocks:
        print("="*60)
        print("YOLO INFERENCE LATENCY EVOLUTION (Thermal Tracking)")
        print("="*60)
        for b in blocks:
            print(f"Inferences {b['range']:<14} (n={b['count']:<3}): Avg = {b['avg_ms']:5.1f} ms | Min = {b['min_ms']:5.1f} ms | Max = {b['max_ms']:5.1f} ms")
        print("="*60 + "\n")

    # Generate Excel Report
    print("\nGenerando Reporte Excel de Auditoría con Fotos Incrustadas...")
    exporter = ExcelReportExporter(pipeline.db.db_path)
    excel_path = exporter.export_report(
        output_path=f"reports/reporte_auditoria_{run_id}.xlsx",
        run_id=run_id,
        include_duplicates=False
    )
    # Also save as latest
    exporter.export_report(
        output_path="reports/reporte_auditoria.xlsx",
        run_id=run_id,
        include_duplicates=False
    )
    print(f"Reporte listo para auditoría: {excel_path}")

    # Evaluate against Ground Truth
    print("\nEvaluando contra Ground Truth (benchmarks/ground_truth.json)...")
    from benchmarks.evaluate_pipeline import evaluate_run
    evaluate_run(pipeline.db.db_path, run_id)

if __name__ == '__main__':
    run_full_pipeline()
