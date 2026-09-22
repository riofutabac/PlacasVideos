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

PIPELINE_VERSION = "1.9.0"

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
        "--decoder",
        choices=["auto", "nvdec", "opencv"],
        default=None,
        help="Backend de decodificación de video: 'auto', 'nvdec' (GPU) u 'opencv' (CPU)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Límite máximo de videos a procesar en esta corrida"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Modelo YOLO para detección de vehículos (ej. 'yolo26n.onnx', 'yolov8n.onnx')"
    )
    parser.add_argument(
        "--no-stage",
        action="store_true",
        help="Desactivar copiado temporal a SSD local antes de decodificar"
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Activar el registro de detecciones descartadas (rejection log) en reports/diagnostics/<run_id>.jsonl"
    )
    parser.add_argument(
        "--prefetch",
        action="store_true",
        help="Adelantar el copiado a SSD del siguiente clip mientras se procesa el actual (opt-in, ver video.prefetch_next_clip)"
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

def stage_video_locally(src_path: str, enabled: bool = True, stage_dir: str = "/content/ssd_video_cache") -> Tuple[str, bool, float]:
    """
    If the video is on a Google Drive / remote FUSE mount, stages it to the local fast SSD
    to eliminate network decode stalls.
    Returns (path_to_process, is_temporary, staging_seconds).
    """
    if not enabled:
        return src_path, False, 0.0

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
                return dst_path, True, dt
        except Exception as e:
            print(f"⚠️ [SSD Staging] No se pudo copiar a SSD ({e}), procesando directo desde Drive.")
            return src_path, False, 0.0

    return src_path, False, 0.0

def run_full_pipeline():
    t_startup_start = time.perf_counter()
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

    pipeline = ALPRPipeline(model_name_override=args.model, diagnose_override=(True if args.diagnose else None))
    if args.decoder:
        pipeline.cfg.setdefault('video', {})['decode_backend'] = args.decoder
    run_id = f"RUN_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if pipeline.rejection_logger.enabled:
        print(f"🔍 [Diagnose] Rejection log habilitado: reports/diagnostics/{run_id}.jsonl")

    prefetch_enabled = bool(args.prefetch) or bool(
        pipeline.cfg.get('video', {}).get('prefetch_next_clip', False)
    )
    if prefetch_enabled:
        print("⚡ [Prefetch] Copiado anticipado del siguiente clip habilitado (--prefetch).")

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
    total_processing_wall = 0.0

    startup_seconds = time.perf_counter() - t_startup_start
    t_start_wall = time.perf_counter()

    total_wait_for_next_seconds = 0.0
    prefetcher = None
    if prefetch_enabled:
        from src.clip_prefetcher import ClipPrefetcher

        def _stage_fn(vf: str) -> Tuple[str, bool, float]:
            return stage_video_locally(vf, enabled=not args.no_stage)

        prefetcher = ClipPrefetcher(video_files, stage_fn=_stage_fn)
        prefetcher.start()

    try:
        for orig_vf in video_files:
            proc_vf = orig_vf
            is_temp = False
            try:
                if prefetcher is not None:
                    # Consume the clip the background worker already staged (or is
                    # staging); this is where processing "waits" for the next file.
                    t_wait_start = time.perf_counter()
                    staged_orig, proc_vf, is_temp, staging_seconds, stage_error = prefetcher.get_next()
                    total_wait_for_next_seconds += time.perf_counter() - t_wait_start
                    assert staged_orig == orig_vf, "Prefetcher returned clips out of order"
                    if stage_error is not None:
                        raise stage_error
                else:
                    # Stage just-in-time to local SSD cache (avoids disk filling and isolates I/O)
                    proc_vf, is_temp, staging_seconds = stage_video_locally(orig_vf, enabled=not args.no_stage)

                from src.video_decoder import probe_video_metadata
                meta = probe_video_metadata(proc_vf)
                fps = meta['fps']
                frames = meta['total_frames']
                cur_dur = (frames / fps) if fps > 0 else 0.0

                t_vid_start = time.perf_counter()
                events = pipeline.process_video_file(proc_vf, run_id, original_path=orig_vf, staging_seconds=staging_seconds)
                t_vid_elapsed = time.perf_counter() - t_vid_start

                total_events_all += len(events)
                total_source_frames += frames
                total_duration_sec += cur_dur
                total_processing_wall += t_vid_elapsed

            except Exception as e:
                print(f"\n❌ [ERROR EN VIDEO] {os.path.basename(orig_vf)}: {e}")
                print(f"⚠️ El archivo parece estar dañado o incompleto (ej. 'moov atom not found').")
                print(f"   Omitiendo este archivo y continuando automáticamente con los siguientes videos...\n")
                try:
                    pipeline.db.record_clip(
                        clip_id=os.path.basename(orig_vf),
                        run_id=run_id,
                        file_path=orig_vf,
                        file_hash="",
                        duration_sec=0.0,
                        total_frames=0,
                        events_count=0,
                        status=f"FAILED: {str(e)[:50]}",
                        started_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        completed_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        wall_clock_seconds=0.0,
                        speed_ratio=0.0,
                        decode_seconds=0.0,
                        vehicle_detection_seconds=0.0,
                        plate_detection_seconds=0.0,
                        ocr_seconds=0.0
                    )
                except Exception:
                    pass
            finally:
                if is_temp and os.path.exists(proc_vf):
                    try:
                        os.remove(proc_vf)
                    except Exception:
                        pass
    finally:
        # Ensure a clean shutdown of the prefetch worker on both the normal
        # path and interruptions (e.g. Ctrl-C) — no stray temp files, no hang.
        if prefetcher is not None:
            prefetcher.stop()

    t_total_wall = time.perf_counter() - t_start_wall
    pipeline.profiler.finish(total_source_frames, total_duration_sec)

    # Compute speed ratio based on pure video processing time
    speed_ratio = total_duration_sec / total_processing_wall if total_processing_wall > 0 else (total_duration_sec / t_total_wall if t_total_wall > 0 else 0.0)

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
    t_export_start = time.perf_counter()
    print("\n📊 Generando Reporte Excel de Auditoría con Fotos Incrustadas...")
    exporter = ExcelReportExporter(pipeline.db.db_path)
    excel_path = exporter.export_report(
        output_path=f"reports/reporte_auditoria_{run_id}.xlsx",
        run_id=run_id,
        include_duplicates=False
    )
    # Also save as latest
    latest_excel_path = "reports/reporte_auditoria.xlsx"
    try:
        shutil.copyfile(excel_path, latest_excel_path)
    except Exception as e:
        print(f"⚠️ No se pudo copiar a {latest_excel_path}: {e}")
    print(f"✅ Reporte listo para auditoría: {excel_path} (y actualizado en {latest_excel_path})")
    export_seconds = time.perf_counter() - t_export_start

    # Finish run in DB (finished_at now covers the Excel export, not just clip processing)
    pipeline.db.finish_run(
        run_id=run_id,
        finished_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        total_videos=len(video_files),
        total_events=total_events_all,
        speed_ratio=round(speed_ratio, 2),
        status="COMPLETED",
        startup_seconds=round(startup_seconds, 2),
        export_seconds=round(export_seconds, 2)
    )

    print_gap_summary(
        pipeline, run_id, startup_seconds, export_seconds, t_total_wall,
        prefetch_enabled=prefetch_enabled, wait_for_next_seconds=total_wait_for_next_seconds
    )
    pipeline.rejection_logger.close()

    # Evaluate against Ground Truth if applicable
    try:
        from benchmarks.evaluate_pipeline import evaluate_run
        evaluate_run(pipeline.db.db_path, run_id)
    except Exception as e:
        print(f"Nota de evaluación: {e}")


def print_gap_summary(
    pipeline, run_id: str, startup_seconds: float, export_seconds: float, t_total_wall: float,
    prefetch_enabled: bool = False, wait_for_next_seconds: float = 0.0
) -> None:
    """Prints a compact breakdown of copy/hash/clock/process/export time and their share of wall clock."""
    totals = pipeline.db.get_clip_timing_totals(run_id)
    copy_s = totals['staging_seconds']
    hash_s = totals['file_hash_seconds']
    clock_s = totals['clock_read_seconds']
    process_s = totals['wall_clock_seconds']
    wall = t_total_wall if t_total_wall > 0 else 1.0

    def pct(x: float) -> float:
        return round((x / wall) * 100, 1)

    print("\n" + "="*60)
    print("BATCH TIME BREAKDOWN (instrumentation only)")
    print("="*60)
    print(f"{'Segment':<22} {'Seconds':<12} {'% Wall Clock':<10}")
    print("-"*60)
    print(f"{'Startup':<22} {startup_seconds:<12.2f} {pct(startup_seconds):<10.1f}%")
    print(f"{'Staging/Copy':<22} {copy_s:<12.2f} {pct(copy_s):<10.1f}%")
    print(f"{'File Hash':<22} {hash_s:<12.2f} {pct(hash_s):<10.1f}%")
    print(f"{'OSD Clock Read':<22} {clock_s:<12.2f} {pct(clock_s):<10.1f}%")
    print(f"{'Clip Processing':<22} {process_s:<12.2f} {pct(process_s):<10.1f}%")
    if prefetch_enabled:
        print(f"{'Wait For Next File':<22} {wait_for_next_seconds:<12.2f} {pct(wait_for_next_seconds):<10.1f}%")
    print(f"{'Excel Export':<22} {export_seconds:<12.2f} {pct(export_seconds):<10.1f}%")
    print(f"{'Total Wall Clock':<22} {t_total_wall:<12.2f}")
    if prefetch_enabled:
        print(
            "Nota: con --prefetch, 'Staging/Copy' se solapa con 'Clip Processing' del clip anterior "
            "(copia en background), por lo que la suma de segmentos ya NO coincide con el wall clock. "
            "'Wait For Next File' es el tiempo real que el consumidor esperó a que terminara la copia."
        )
    print("="*60 + "\n")

if __name__ == '__main__':
    run_full_pipeline()
