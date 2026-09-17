"""
Main Execution Script for Event-Oriented ALPR Pipeline.
Usage:
  python main.py
  python main.py "/content/drive/MyDrive/Cam PL"
  python main.py "/content/drive/MyDrive/Cam PL" --limit 2
  python main.py "video.mp4"
"""
import os
import sys
import glob
import time
import argparse
from pathlib import Path
from datetime import datetime

PIPELINE_VERSION = "1.0.0"

def parse_args():
    parser = argparse.ArgumentParser(
        description=f"Pipeline ALPR Ligero Orientado a Eventos v{PIPELINE_VERSION}",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "source",
        nargs="?",
        default=None,
        help="Ruta a un archivo de video, carpeta contenedora o patrón glob"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Límite máximo de videos a procesar en esta corrida"
    )
    return parser.parse_args()

def resolve_video_files(source_arg):
    extensions = (".mp4", ".avi", ".mkv", ".mov")
    video_files = []

    if source_arg:
        # Check if single file
        if os.path.isfile(source_arg):
            return [source_arg]

        # Check if directory
        if os.path.isdir(source_arg):
            for ext in extensions:
                video_files.extend(glob.glob(os.path.join(source_arg, f"*{ext}")))
                video_files.extend(glob.glob(os.path.join(source_arg, f"**/*{ext}"), recursive=True))
            video_files = sorted(list(set(video_files)))
            if video_files:
                return video_files

        # Check if glob pattern
        globbed = sorted(glob.glob(source_arg))
        if globbed:
            return globbed

    # Default fallbacks if no argument provided or not found
    # 1. Local Camara Placas 2_*.mp4
    local_cam = sorted(glob.glob("Camara Placas 2_*.mp4"))
    if local_cam:
        return local_cam

    # 2. Local any *.mp4
    local_mp4 = sorted(glob.glob("*.mp4"))
    if local_mp4:
        return local_mp4

    # 3. Google Colab standard Drive path
    colab_drive_path = "/content/drive/MyDrive/Cam PL"
    if os.path.isdir(colab_drive_path):
        drive_videos = []
        for ext in extensions:
            drive_videos.extend(glob.glob(os.path.join(colab_drive_path, f"*{ext}")))
            drive_videos.extend(glob.glob(os.path.join(colab_drive_path, f"**/*{ext}"), recursive=True))
        drive_videos = sorted(list(set(drive_videos)))
        if drive_videos:
            return drive_videos

    return []

def run_full_pipeline():
    args = parse_args()

    print("="*70)
    print(f"PIPELINE ALPR LIGERO ORIENTADO A EVENTOS v{PIPELINE_VERSION}")
    print(f"Inicio de corrida: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    # Resolve videos
    video_files = resolve_video_files(args.source)
    if not video_files:
        print("❌ Error: No se encontraron archivos de video para procesar.")
        print("Rutas verificadas:")
        print(" - Carpeta local actual:", os.getcwd())
        print(" - Google Drive por defecto: /content/drive/MyDrive/Cam PL")
        print("\nUso sugerido:")
        print("  python main.py \"/ruta/a/tus/videos\"")
        print("  python main.py \"/content/drive/MyDrive/Cam PL\" --limit 2")
        sys.exit(1)

    if args.limit and args.limit > 0:
        video_files = video_files[:args.limit]

    print(f"📹 Videos a procesar ({len(video_files)} archivos):")
    for vf in video_files:
        print(f"  - {vf}")

    # Lazy load heavy dependencies
    from src.pipeline_runner import ALPRPipeline, PIPELINE_VERSION as RUNNER_VER
    from src.excel_exporter import ExcelReportExporter

    pipeline = ALPRPipeline()
    run_id = f"RUN_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Start run in DB
    pipeline.db.start_run(
        run_id=run_id,
        pipeline_version=RUNNER_VER,
        config_hash=pipeline.config_hash,
        model_versions=pipeline.model_versions,
        started_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    )

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
    print("\n📊 Generando Reporte Excel de Auditoría con Fotos Incrustadas...")
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
    print(f"✅ Reporte listo para auditoría: {excel_path}")

    # Evaluate against Ground Truth if applicable
    try:
        from benchmarks.evaluate_pipeline import evaluate_run
        evaluate_run(pipeline.db.db_path, run_id)
    except Exception as e:
        print(f"Nota de evaluación: {e}")

if __name__ == '__main__':
    run_full_pipeline()
