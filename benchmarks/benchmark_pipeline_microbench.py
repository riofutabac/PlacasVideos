#!/usr/bin/env python3
"""
Pipeline Microbenchmarks: Empirical Validation of Hypotheses T-V3, T-V4, and T-C2.
Measures:
1. T-V3: Standalone NV12->BGR color conversion latency vs pipeline latency.
2. T-V4: YOLO standalone latency breakdown (preprocess, inference, postprocess).
3. T-C2: Thread scaling latency on host CPU (1, 2, 4, and max threads).
"""
import os
import sys
import time
import cv2
import numpy as np

# Ensure project root in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.threading_profile import configure_threading_profile
from src.model_resolver import resolve_vehicle_model

def run_microbenchmarks():
    print("=" * 65)
    print("ALPR PIPELINE EMPIRICAL MICROBENCHMARKS")
    print(f"Host CPUs: {os.cpu_count()} | Platform: {sys.platform}")
    print("=" * 65)

    # -------------------------------------------------------------
    # 1. T-V3: Standalone Color Conversion (NV12 -> BGR)
    # -------------------------------------------------------------
    print("\n[T-V3] Midiendo conversión de color NV12 -> BGR en ROI físico (2300x1064)...")
    h, w = 1064, 2300
    nv12_buffer = np.random.randint(0, 255, (int(h * 1.5), w), dtype=np.uint8)
    bgr_buffer = np.empty((h, w, 3), dtype=np.uint8)

    # Warmup
    for _ in range(5):
        cv2.cvtColor(nv12_buffer, cv2.COLOR_YUV2BGR_NV12, dst=bgr_buffer)

    t_conv = []
    for _ in range(50):
        t0 = time.perf_counter()
        cv2.cvtColor(nv12_buffer, cv2.COLOR_YUV2BGR_NV12, dst=bgr_buffer)
        t_conv.append(time.perf_counter() - t0)

    mean_conv_ms = np.mean(t_conv) * 1000.0
    min_conv_ms = np.min(t_conv) * 1000.0
    p95_conv_ms = np.percentile(t_conv, 95) * 1000.0
    print(f"  -> NV12 a BGR Standalone: Media = {mean_conv_ms:.2f} ms | Mín = {min_conv_ms:.2f} ms | P95 = {p95_conv_ms:.2f} ms")
    if mean_conv_ms < 3.0:
        print("  ✅ Hallazgo T-V3: La conversión aislada toma < 3 ms. La inflación a ~7.6 ms previa se debió a contención de CPU.")

    # -------------------------------------------------------------
    # 2. T-V4: YOLO Latency Breakdown
    # -------------------------------------------------------------
    print("\n[T-V4] Midiendo desglose de YOLOv8n (Preproceso, Inferencia, Postproceso NMS)...")
    model_path = resolve_vehicle_model("yolov8n.onnx", imgsz=416)
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        dummy_crop = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)

        # Warmup
        for _ in range(3):
            model(dummy_crop, imgsz=416, verbose=False)

        pre_times, inf_times, post_times = [], [], []
        for _ in range(30):
            res = model(dummy_crop, imgsz=416, verbose=False)
            sp = res[0].speed
            pre_times.append(sp['preprocess'])
            inf_times.append(sp['inference'])
            post_times.append(sp['postprocess'])

        print(f"  -> Preproceso:  {np.mean(pre_times):.2f} ms")
        print(f"  -> Inferencia:   {np.mean(inf_times):.2f} ms")
        print(f"  -> Postproceso:  {np.mean(post_times):.2f} ms (NMS estándar)")
    except Exception as e:
        print(f"  ⚠️ No se pudo ejecutar YOLO: {e}")

    # -------------------------------------------------------------
    # 3. T-C2: Thread Scaling Latency
    # -------------------------------------------------------------
    print("\n[T-C2] Midiendo impacto del número de hilos de OpenCV en procesamiento de recortes...")
    for t_count in [1, 2, 4, os.cpu_count() or 4]:
        cv2.setNumThreads(t_count)
        bench_times = []
        for _ in range(25):
            t0 = time.perf_counter()
            cv2.cvtColor(nv12_buffer, cv2.COLOR_YUV2BGR_NV12, dst=bgr_buffer)
            bench_times.append(time.perf_counter() - t0)
        print(f"  -> cv2.setNumThreads({t_count}): Media = {np.mean(bench_times)*1000.0:.2f} ms")

    print("\n" + "=" * 65)
    print("RESUMEN DE MICROBENCHMARKS COMPLETADO")
    print("=" * 65)

if __name__ == "__main__":
    run_microbenchmarks()
