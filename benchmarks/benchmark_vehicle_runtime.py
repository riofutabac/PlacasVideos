#!/usr/bin/env python3
"""
Vehicle Detector Runtime Benchmark: Ultralytics vs ORT I/O Binding vs TensorRT (Fase C2/C3).
Evaluates latency breakdown (preprocess, upload, inference, download, postprocess)
across candidate vehicle detection execution providers and runtimes.
"""

import os
import sys
import time
import csv
import argparse
from typing import List, Dict, Any, Optional
import numpy as np
import cv2

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.vehicle_runtime import VehicleDetectorRunner
from src.model_resolver import resolve_vehicle_model

RUNTIMES = [
    ("ultralytics", "Ultralytics YOLO (Default)"),
    ("ort_iobinding", "ONNX Runtime CUDA I/O Binding"),
    ("ort_trt", "ONNX Runtime TensorRT EP (FP16)"),
    ("trt_engine", "Native TensorRT Engine (.engine)")
]

def load_or_create_test_frames(n_frames: int = 20, imgsz: int = 416) -> List[np.ndarray]:
    """Loads realistic ROI test frames (2300x1064) or creates synthetic frames."""
    frames = []
    veh_dir = "evidence/vehicles"
    if os.path.exists(veh_dir):
        for fname in sorted(os.listdir(veh_dir)):
            if fname.endswith(".jpg"):
                img = cv2.imread(os.path.join(veh_dir, fname))
                if img is not None:
                    # Pad/resize to simulate 2300x1064 ROI
                    h, w = img.shape[:2]
                    canvas = np.zeros((1064, 2300, 3), dtype=np.uint8)
                    canvas[:min(h, 1064), :min(w, 2300)] = img[:min(h, 1064), :min(w, 2300)]
                    frames.append(canvas)
                    if len(frames) >= n_frames:
                        break

    if not frames:
        for _ in range(n_frames):
            frames.append(np.random.randint(0, 255, (1064, 2300, 3), dtype=np.uint8))

    return frames

def benchmark_runtime(
    runtime_key: str,
    runtime_label: str,
    model_name: str,
    frames: List[np.ndarray],
    imgsz: int = 416,
    warmup_iters: int = 5,
    runner_cache: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    print(f"⏱️ Probando backend: {runtime_label}...")

    # Hardware detection
    import torch
    has_cuda = torch.cuda.is_available()
    device = "cuda" if has_cuda else "cpu"

    if runtime_key == "trt_engine":
        print(f"  ℹ️ {runtime_label}: Motor nativo .engine no implementado aún.")
        return {
            "runtime": runtime_key,
            "label": runtime_label,
            "status": "NOT_IMPLEMENTED",
            "preprocess_ms": None,
            "inference_ms": None,
            "postprocess_ms": None,
            "total_p50_ms": None,
            "total_p95_ms": None,
            "fps_equivalent": None,
            "speedup_vs_baseline": None
        }

    if runtime_key in ("ort_iobinding", "ort_trt") and not has_cuda:
        print(f"  ⚠️ {runtime_label}: Requiere GPU NVIDIA con CUDA/TensorRT. No medido en host CPU.")
        return {
            "runtime": runtime_key,
            "label": runtime_label,
            "status": "REQUIRES_CUDA_GPU",
            "preprocess_ms": None,
            "inference_ms": None,
            "postprocess_ms": None,
            "total_p50_ms": None,
            "total_p95_ms": None,
            "fps_equivalent": None,
            "speedup_vs_baseline": None
        }

    if runtime_key == "ort_trt":
        import onnxruntime as ort
        if "TensorrtExecutionProvider" not in ort.get_available_providers():
            print(f"  ⚠️ {runtime_label}: TensorrtExecutionProvider no está disponible en este entorno.")
            return {
                "runtime": runtime_key,
                "label": runtime_label,
                "status": "NOT_AVAILABLE",
                "preprocess_ms": None,
                "inference_ms": None,
                "postprocess_ms": None,
                "total_p50_ms": None,
                "total_p95_ms": None,
                "fps_equivalent": None,
                "speedup_vs_baseline": None
            }

    effective_runtime = runtime_key
    cache_key = (effective_runtime, model_name, device)

    runner = None
    if runner_cache is not None and cache_key in runner_cache:
        runner = runner_cache[cache_key]
    else:
        try:
            runner = VehicleDetectorRunner(
                model_name=model_name,
                runtime=effective_runtime,
                imgsz=imgsz,
                device=device
            )
        except Exception as e:
            print(f"  ℹ️ Error al inicializar {runtime_label} ({e})")
            runner = None
        if runner_cache is not None:
            runner_cache[cache_key] = runner

    if runner is None:
        return {
            "runtime": runtime_key,
            "label": runtime_label,
            "status": "NOT_AVAILABLE",
            "preprocess_ms": None,
            "inference_ms": None,
            "postprocess_ms": None,
            "total_p50_ms": None,
            "total_p95_ms": None,
            "fps_equivalent": None,
            "speedup_vs_baseline": None
        }
    for i in range(min(warmup_iters, len(frames))):
        if runner:
            try:
                runner.predict(frames[i], imgsz=imgsz)
            except Exception:
                pass

    prep_times = []
    inf_times = []
    post_times = []
    total_times = []

    for frame in frames:
        _, _, _, timing = runner.predict(frame, imgsz=imgsz)
        p_ms = timing.get('preprocess', 0.0) * 1000.0
        i_ms = timing.get('inference', 0.0) * 1000.0
        o_ms = timing.get('postprocess', 0.0) * 1000.0
        tot_ms = timing.get('total', 0.0) * 1000.0

        prep_times.append(p_ms)
        inf_times.append(i_ms)
        post_times.append(o_ms)
        total_times.append(tot_ms)

    avg_prep = float(np.mean(prep_times))
    avg_inf = float(np.mean(inf_times))
    avg_post = float(np.mean(post_times))
    p50_tot = float(np.percentile(total_times, 50))
    p95_tot = float(np.percentile(total_times, 95))
    fps_equiv = 1000.0 / p50_tot if p50_tot > 0 else 0.0

    return {
        "runtime": runtime_key,
        "label": runtime_label,
        "status": "MEASURED",
        "preprocess_ms": round(avg_prep, 2),
        "inference_ms": round(avg_inf, 2),
        "postprocess_ms": round(avg_post, 2),
        "total_p50_ms": round(p50_tot, 2),
        "total_p95_ms": round(p95_tot, 2),
        "fps_equivalent": round(fps_equiv, 1),
        "speedup_vs_baseline": 1.0
    }

def run_vehicle_runtime_benchmark(
    model_name: str = "yolov8n.onnx",
    n_frames: int = 20,
    imgsz: int = 416,
    output_csv: str = "benchmarks/benchmark_vehicle_runtime.csv"
):
    print("=" * 80)
    print("BENCHMARK DE RUNTIMES DE VEHÍCULO YOLOv8n (Fase C2/C3)")
    print("=" * 80)

    resolved_model = resolve_vehicle_model(model_name, imgsz=imgsz)
    frames = load_or_create_test_frames(n_frames=n_frames, imgsz=imgsz)
    print(f"🖼️ Evaluando {len(frames)} cuadros ROI (2300x1064) @ imgsz={imgsz}...")

    results = []
    base_p50 = None
    runner_cache: Dict[str, Any] = {}
    for r_key, r_label in RUNTIMES:
        res = benchmark_runtime(r_key, r_label, resolved_model, frames, imgsz=imgsz, runner_cache=runner_cache)
        if res.get("status") == "REQUIRES_CUDA_GPU":
            results.append(res)
            continue
        if base_p50 is None and res.get("total_p50_ms"):
            base_p50 = res["total_p50_ms"]
        if base_p50 and res.get("total_p50_ms"):
            res["speedup_vs_baseline"] = round(base_p50 / res["total_p50_ms"], 2)
        else:
            res["speedup_vs_baseline"] = 1.0
        results.append(res)

    # Save to CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    fieldnames = [
        "runtime", "label", "status", "preprocess_ms", "inference_ms",
        "postprocess_ms", "total_p50_ms", "total_p95_ms", "fps_equivalent", "speedup_vs_baseline"
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"📁 Resultados guardados en {output_csv}")

    # Display Table
    print("\n" + "=" * 90)
    print(f"{'Backend / Runtime':<35} | {'Pre (ms)':<8} | {'Inf (ms)':<8} | {'Post (ms)':<9} | {'p50 (ms)':<8} | {'FPS':<6} | {'Speedup'}")
    print("-" * 90)
    for r in results:
        status = r.get("status")
        if status == "REQUIRES_CUDA_GPU":
            print(f"{r['label']:<35} | {'[Requiere GPU CUDA/TensorRT - No medido en CPU]':<50}")
        elif status == "NOT_IMPLEMENTED":
            print(f"{r['label']:<35} | {'[No implementado aún]':<50}")
        elif status == "NOT_AVAILABLE":
            print(f"{r['label']:<35} | {'[No disponible en este entorno]':<50}")
        else:
            print(f"{r['label']:<35} | {r['preprocess_ms']:<8.2f} | {r['inference_ms']:<8.2f} | {r['postprocess_ms']:<9.2f} | {r['total_p50_ms']:<8.2f} | {r['fps_equivalent']:<6.1f} | {r['speedup_vs_baseline']}x")
    print("=" * 90 + "\n")

    return results

def main():
    parser = argparse.ArgumentParser(description="Benchmark vehicle detector runtimes")
    parser.add_argument("--model", default="yolov8n.onnx", help="Vehicle detector model")
    parser.add_argument("--frames", type=int, default=20, help="Number of benchmark frames")
    parser.add_argument("--imgsz", type=int, default=416, help="Inference resolution")
    parser.add_argument("--output", default="benchmarks/benchmark_vehicle_runtime.csv", help="Output CSV")
    args = parser.parse_args()

    run_vehicle_runtime_benchmark(model_name=args.model, n_frames=args.frames, imgsz=args.imgsz, output_csv=args.output)

if __name__ == "__main__":
    main()
