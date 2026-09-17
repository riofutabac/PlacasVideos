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
import re
import shutil
import argparse
from pathlib import Path
from typing import Tuple, List, Optional
from datetime import datetime

PIPELINE_VERSION = "1.2.0"

def natural_clip_sort_key(filepath: str):
    """Sorts clip filenames numerically (1, 2, ... 10, ... 60, 61) rather than lexicographically."""
    m = re.search(r'\((\d+)\)\.[a-zA-Z0-9]+$', os.path.basename(filepath))
    if m:
        return (0, int(m.group(1)))
    return (1, [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', filepath)])

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
        "--clips",
        nargs="+",
        type=int,
        default=None,
        help="Números de clips específicos a procesar (ej. --clips 60 61)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Límite máximo de videos a procesar en esta corrida"
    )
    parser.add_argument(
        "--no-stage",
        action="store_true",
        help="Desactivar copiado temporal a SSD local antes de decodificar"
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

def stage_video_locally(src_path: str, enabled: bool = True, stage_dir: str = "/content/ssd_video_cache") -> Tuple[str, bool]:
    """
    If the video is on a Google Drive / remote FUSE mount, stages it to the local fast SSD
    to eliminate network decode stalls.
    Returns (path_to_process, is_temporary).
    """
    if not enabled:
        return src_path, False

    abs_src = os.path.abspath(src_path)
    is_drive = "/content/drive" in abs_src
    force_stage = os.environ.get("STAGE_SSD", "").lower() in ("1", "true", "yes")

    if is_drive or force_stage:
        try:
            os.makedirs(stage_dir, exist_ok=True)
            filename = os.path.basename(src_path)
            dst_path = os.path.join(stage_dir, filename)
            if abs_src != os.path.abspath(dst_path):
                t0 = time.perf_counter()
                size_bytes = os.path.getsize(src_path)
                size_mb = size_bytes / (1024 * 1024)
                print(f"⚡ [SSD Staging] Copiando {filename} ({size_mb:.1f} MB) a NVMe SSD...")
                shutil.copyfile(src_path, dst_path)
                dt = time.perf_counter() - t0
                speed_mb = size_mb / dt if dt > 0 else 0.0
                print(f"⚡ [SSD Staging] Listo en {dt:.2f}s ({speed_mb:.1f} MB/s). Decodificando en SSD local.")
                return dst_path, True
        except Exception as e:
            print(f"⚠️ [SSD Staging] No se pudo copiar a SSD ({e}), procesando directo desde Drive.")
            return src_path, False

    return src_path, False

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

    # Filter by specific clip numbers if requested (e.g. --clips 60 61)
    if args.clips:
        wanted_clips = set(args.clips)
        filtered = []
        for vf in video_files:
            m = re.search(r'\((\d+)\)\.[a-zA-Z0-9]+$', os.path.basename(vf))
            if m and int(m.group(1)) in wanted_clips:
                filtered.append(vf)
        video_files = filtered

    # Always sort naturally: (1), (2), ..., (9), (10), ..., (60), (61)
    video_files = sorted(video_files, key=natural_clip_sort_key)

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
        proc_vf, is_temp = stage_video_locally(vf, enabled=not args.no_stage)
        try:
            # Collect video duration
            import cv2
            cap = cv2.VideoCapture(proc_vf)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            total_source_frames += frames
            total_duration_sec += (frames / fps)

            events = pipeline.process_video_file(proc_vf, run_id, original_path=vf)
            total_events_all += len(events)
        finally:
            if is_temp and os.path.exists(proc_vf):
                try:
                    os.remove(proc_vf)
                except Exception:
                    pass

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
