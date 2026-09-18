#!/usr/bin/env python3
"""
Benchmark Comparativo A/B: YOLOv8n vs YOLO26n
Compara velocidad (latencia en ms), FPS de inferencia, compatibilidad ONNX y Supervision/ByteTrack
en GPU CUDA o CPU.

Uso en Colab:
    !pip install -U ultralytics
    !python benchmarks/benchmark_yolo_comparison.py [--device cuda] [--iterations 100]
"""

import os
import sys
import time
import argparse
import numpy as np
import cv2
import torch

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Comparativo YOLOv8n vs YOLO26n")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Dispositivo (cuda o cpu)")
    parser.add_argument("--imgsz", type=int, default=416, help="Tamaño de entrada para YOLO (default: 416)")
    parser.add_argument("--iterations", type=int, default=100, help="Número de inferencias de prueba")
    parser.add_argument("--warmup", type=int, default=15, help="Número de inferencias de calentamiento")
    return parser.parse_args()

def load_or_export_onnx(model_id: str, imgsz: int):
    """Loads a YOLO model from .pt and exports to .onnx if not present."""
    from ultralytics import YOLO
    
    pt_name = f"{model_id}.pt"
    onnx_name = f"{model_id}.onnx"
    
    # Check if ONNX already exists
    if os.path.exists(onnx_name):
        print(f"📦 ONNX encontrado: {onnx_name}")
        return onnx_name

    print(f"⬇️ Cargando {pt_name} de Ultralytics...")
    try:
        model = YOLO(pt_name)
    except Exception as e:
        print(f"⚠️ Error al cargar {pt_name}: {e}")
        # Try alternative name
        alt_name = pt_name.replace("yolo", "yolov") if "yolov" not in pt_name else pt_name.replace("yolov", "yolo")
        print(f"🔄 Intentando con nombre alternativo: {alt_name}...")
        model = YOLO(alt_name)
        pt_name = alt_name

    print(f"⚡ Exportando {pt_name} a formato ONNX (imgsz={imgsz})...")
    try:
        onnx_file = model.export(format="onnx", imgsz=imgsz, dynamic=True)
        print(f"✅ Exportación exitosa: {onnx_file}")
        return onnx_file
    except Exception as e:
        print(f"❌ Error al exportar {pt_name} a ONNX: {e}")
        return None

def test_model(model_path: str, crop_roi: np.ndarray, device: str, imgsz: int, iterations: int, warmup: int):
    """Runs speed benchmark and checks supervision compatibility."""
    from ultralytics import YOLO
    import supervision as sv

    print(f"\n────────────────────────────────────────────────────────────")
    print(f"🔬 Evaluando modelo: {model_path} en {device.upper()}")
    print(f"────────────────────────────────────────────────────────────")

    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"❌ No se pudo inicializar {model_path}: {e}")
        return None

    vehicle_classes = [0, 1, 2, 3, 5, 7] # person, bicycle, car, motorcycle, bus, truck

    # 1. Warmup
    print(f"🔥 Calentando GPU con {warmup} iteraciones...")
    for _ in range(warmup):
        _ = model(crop_roi, imgsz=imgsz, device=device, verbose=False, conf=0.25, classes=vehicle_classes)
    
    if device == "cuda":
        torch.cuda.synchronize()

    # 2. Benchmark runs
    print(f"⏱️ Midiendo latencia en {iterations} iteraciones...")
    latencies_ms = []
    
    for _ in range(iterations):
        t0 = time.perf_counter()
        results = model(crop_roi, imgsz=imgsz, device=device, verbose=False, conf=0.25, classes=vehicle_classes)
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    avg_ms = np.mean(latencies_ms)
    min_ms = np.min(latencies_ms)
    max_ms = np.max(latencies_ms)
    p95_ms = np.percentile(latencies_ms, 95)
    fps_equiv = 1000.0 / avg_ms if avg_ms > 0 else 0.0

    # 3. Compatibility test with Supervision / ByteTrack
    last_res = results[0]
    detections_count = len(last_res.boxes) if hasattr(last_res, "boxes") else 0
    
    supervision_ok = False
    try:
        sv_dets = sv.Detections.from_ultralytics(last_res)
        supervision_ok = len(sv_dets) == detections_count
    except Exception as e:
        print(f"⚠️ Incompatibilidad con Supervision: {e}")
        supervision_ok = False

    # Get file size
    size_mb = os.path.getsize(model_path) / (1024 * 1024) if os.path.exists(model_path) else 0.0

    print(f"  • Latencia Promedio: {avg_ms:5.2f} ms")
    print(f"  • Latencia Mínima:   {min_ms:5.2f} ms")
    print(f"  • Latencia P95:      {p95_ms:5.2f} ms")
    print(f"  • Throughput IA:     {fps_equiv:5.1f} inferencias/segundo")
    print(f"  • Objetos detectados:{detections_count}")
    print(f"  • Compatibilidad ByteTrack/Supervision: {'✅ COMPATIBLE' if supervision_ok else '❌ REQUIERE ADAPTADOR'}")

    return {
        "model": os.path.basename(model_path),
        "size_mb": size_mb,
        "avg_ms": avg_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "p95_ms": p95_ms,
        "fps": fps_equiv,
        "detections": detections_count,
        "supervision_ok": supervision_ok
    }

def main():
    args = parse_args()
    print("=" * 70)
    print("BENCHMARK COMPARATIVO: YOLOV8 vs YOLO26 (NMS-FREE)")
    print("=" * 70)
    print(f"💻 Dispositivo:     {args.device.upper()}")
    if args.device == "cuda":
        print(f"🎮 GPU Detectada:   {torch.cuda.get_device_name(0)}")
    print(f"📐 Resolución input: imgsz={args.imgsz}")
    print(f"🔄 Iteraciones:     {args.iterations} pruebas (+{args.warmup} warmup)")
    print("-" * 70)

    # Prepare sample frame
    img_path = "benchmarks/sample_frame.jpg"
    if os.path.exists(img_path):
        frame = cv2.imread(img_path)
        # Crop physical ROI (y: 600..1664, x: 300..2600)
        crop_roi = frame[600:1664, 300:2600]
    else:
        print("⚠️ No se encontró 'benchmarks/sample_frame.jpg'. Generando cuadro sintético 2300x1064...")
        crop_roi = np.random.randint(0, 255, (1064, 2300, 3), dtype=np.uint8)

    models_to_test = [
        ("yolov8n", "YOLOv8 Nano (Actual en Producción)"),
        ("yolo26n", "YOLO26 Nano (Nueva arquitectura 2026 NMS-Free)")
    ]

    benchmark_results = []

    for model_id, label in models_to_test:
        print(f"\n🚀 [Preparando {label}]")
        onnx_file = load_or_export_onnx(model_id, args.imgsz)
        
        # Test ONNX format
        if onnx_file and os.path.exists(onnx_file):
            res_onnx = test_model(onnx_file, crop_roi, args.device, args.imgsz, args.iterations, args.warmup)
            if res_onnx:
                res_onnx["label"] = f"{model_id} (ONNX)"
                benchmark_results.append(res_onnx)

    # Print Comparison Table
    if benchmark_results:
        print("\n" + "=" * 70)
        print("📊 TABLA COMPARATIVA FINAL DE RESULTADOS")
        print("=" * 70)
        print(f"{'Modelo':<18} | {'Tamaño':<8} | {'Promedio':<10} | {'Mínimo':<9} | {'FPS IA':<8} | {'ByteTrack':<10}")
        print("-" * 70)
        for r in benchmark_results:
            compat_str = "✅ OK" if r['supervision_ok'] else "❌ Revisa"
            print(f"{r['label']:<18} | {r['size_mb']:5.1f} MB | {r['avg_ms']:6.2f} ms | {r['min_ms']:6.2f} ms | {r['fps']:6.1f} | {compat_str}")
        print("=" * 70)

        # Conclusion
        if len(benchmark_results) >= 2:
            m1 = benchmark_results[0]
            m2 = benchmark_results[1]
            diff_pct = ((m1['avg_ms'] - m2['avg_ms']) / m1['avg_ms']) * 100.0
            if diff_pct > 0:
                print(f"🏆 GANADOR EN VELOCIDAD: {m2['label']} es {abs(diff_pct):.1f}% más rápido ({m2['avg_ms']:.1f} ms vs {m1['avg_ms']:.1f} ms).")
            else:
                print(f"🏆 GANADOR EN VELOCIDAD: {m1['label']} es {abs(diff_pct):.1f}% más rápido ({m1['avg_ms']:.1f} ms vs {m2['avg_ms']:.1f} ms).")
    else:
        print("❌ No se pudieron completar las pruebas de los modelos.")

if __name__ == "__main__":
    main()
