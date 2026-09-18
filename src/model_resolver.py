"""
Model Resolver and Fallback Logic for ALPR Vehicle Detector.
Ensures seamless switching between YOLOv8 and YOLO26, auto-exporting ONNX
when necessary and gracefully falling back to stable defaults if weights are missing.
"""
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "yolov8n.onnx"
FALLBACK_PYTORCH_MODEL = "yolov8n.pt"

def resolve_vehicle_model(
    model_name: str = DEFAULT_MODEL,
    imgsz: int = 416,
    default_model: str = DEFAULT_MODEL
) -> str:
    """
    Resolves a vehicle detector model path, exporting from .pt if .onnx is missing,
    or falling back to a safe default if the requested model is unavailable.

    Args:
        model_name: Name or path to requested model (e.g. 'yolo26n.onnx', 'yolov8n.onnx').
        imgsz: Inference input size used if ONNX export is needed.
        default_model: Fallback model if requested model cannot be found or exported.

    Returns:
        Resolved model path or name valid for YOLO() initialization.
    """
    # 1. Direct path check
    if os.path.exists(model_name):
        return model_name

    # 2. Attempt ONNX export if requested model is an ONNX file
    if model_name.endswith(".onnx"):
        pt_name = model_name.replace(".onnx", ".pt")
        # If .pt exists locally or is a known ultralytics model name (e.g., yolov8n.pt, yolo26n.pt)
        try:
            from ultralytics import YOLO
            print(f"⚡ [Model Resolver] Intentando exportar {pt_name} a ONNX ({model_name})...")
            m = YOLO(pt_name)
            m.export(format="onnx", imgsz=imgsz, dynamic=True)
            if os.path.exists(model_name):
                print(f"✅ [Model Resolver] Exportación exitosa: {model_name}")
                return model_name
        except Exception as e:
            print(f"⚠️ [Model Resolver] No se pudo exportar {model_name} desde {pt_name} ({e}).")
            if os.path.exists(pt_name):
                print(f"ℹ️ [Model Resolver] Usando modelo PyTorch {pt_name} directamente.")
                return pt_name

    # 3. If requested model is a .pt file and exists
    if os.path.exists(model_name):
        return model_name

    # 4. Fallback to default model if different from requested
    if model_name != default_model:
        print(f"⚠️ [Model Resolver] Modelo solicitado '{model_name}' no disponible. Cayendo a fallback '{default_model}'...")
        return resolve_vehicle_model(default_model, imgsz=imgsz, default_model=default_model)

    # 5. Last resort: return PyTorch default which ultralytics can download automatically
    print(f"ℹ️ [Model Resolver] Usando modelo PyTorch por defecto: {FALLBACK_PYTORCH_MODEL}")
    return FALLBACK_PYTORCH_MODEL
