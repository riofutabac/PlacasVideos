"""
Benchmark YOLO inference speed across resolutions (640, 512, 416, 384) on Intel Mac CPU.
Uses real vehicle crops from benchmarks/ground_truth_candidates/.
"""
import time
import glob
import cv2
import torch
from ultralytics import YOLO

def run_benchmark():
    # Test setting thread count
    torch.set_num_threads(4)
    print(f"PyTorch threads set to: {torch.get_num_threads()}")

    model = YOLO("yolov8n.pt")
    
    # Load sample real crop (2300x1064 physical ROI crop)
    frame = cv2.imread("benchmarks/sample_frame.jpg")
    crop_roi = frame[600:1664, 300:2600]
    print(f"Input Crop ROI size: {crop_roi.shape}")

    # Warmup
    _ = model(crop_roi, imgsz=416, verbose=False)

    resolutions = [640, 512, 416, 384]
    results = {}

    for sz in resolutions:
        # Run 20 iterations
        times = []
        for _ in range(15):
            t0 = time.perf_counter()
            res = model(crop_roi, imgsz=sz, verbose=False, conf=0.30, classes=[2, 3, 5, 7])[0]
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)
            
        avg_ms = sum(times) / len(times)
        min_ms = min(times)
        boxes_count = len(res.boxes)
        results[sz] = {
            'avg_ms': round(avg_ms, 1),
            'min_ms': round(min_ms, 1),
            'boxes_found': boxes_count
        }
        print(f"imgsz={sz:3d} | Avg: {avg_ms:6.1f} ms | Min: {min_ms:6.1f} ms | Detections: {boxes_count}")

    # Also test ONNX export at 416
    print("\nExporting YOLO to ONNX at imgsz=416...")
    try:
        onnx_path = model.export(format="onnx", imgsz=416, verbose=False)
        print(f"Exported to: {onnx_path}")
        onnx_model = YOLO(onnx_path)
        # Warmup
        _ = onnx_model(crop_roi, imgsz=416, verbose=False)
        times_onnx = []
        for _ in range(15):
            t0 = time.perf_counter()
            res = onnx_model(crop_roi, imgsz=416, verbose=False, conf=0.30, classes=[2, 3, 5, 7])[0]
            t1 = time.perf_counter()
            times_onnx.append((t1 - t0) * 1000.0)
        avg_onnx = sum(times_onnx) / len(times_onnx)
        print(f"YOLO ONNX 416 | Avg: {avg_onnx:6.1f} ms | Min: {min(times_onnx):6.1f} ms | Detections: {len(res.boxes)}")
    except Exception as e:
        print(f"ONNX test error: {e}")

if __name__ == '__main__':
    run_benchmark()
