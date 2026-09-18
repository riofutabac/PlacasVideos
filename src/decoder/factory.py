"""
Factory module for initializing appropriate video decoder instances based on configuration and hardware availability.
"""
from typing import Optional, Dict, Any
from .base import BaseVideoDecoder
from .opencv_decoder import OpenCVDecoder
from .nvdec_decoder import NVDECDecoder, check_nvdec_available

def create_decoder(
    video_path: str,
    backend: str = "auto",
    profiler: Optional[Any] = None,
    crop_rect: Optional[Dict[str, int]] = None,
    queue_size: int = 128
) -> BaseVideoDecoder:
    """
    Factory function for video decoders.
    backend: 'auto', 'nvdec', or 'opencv'
    crop_rect: Optional bounding rect dict {'x_min', 'y_min', 'x_max', 'y_max'}
    queue_size: Queue buffer depth for threaded producer
    """
    backend_clean = (backend or "auto").lower().strip()

    if backend_clean == "auto":
        info = check_nvdec_available()
        if info["cuda"] or info["hevc_cuvid"] or info["h264_cuvid"] or info["pynv"]:
            print(f"⚡ [VideoDecoder] Backend 'auto' -> Activando NVDEC (NVIDIA GPU Hardware Acceleration)")
            return NVDECDecoder(video_path, profiler, crop_rect=crop_rect, queue_size=queue_size)
        else:
            print(f"ℹ️ [VideoDecoder] Backend 'auto' -> Activando OpenCV (CPU Fallback)")
            return OpenCVDecoder(video_path, profiler, crop_rect=crop_rect)

    elif backend_clean in ("nvdec", "cuda", "cuvid"):
        print(f"⚡ [VideoDecoder] Backend explícito: NVDEC")
        return NVDECDecoder(video_path, profiler, crop_rect=crop_rect, queue_size=queue_size)

    elif backend_clean in ("opencv", "cpu"):
        print(f"ℹ️ [VideoDecoder] Backend explícito: OpenCV")
        return OpenCVDecoder(video_path, profiler, crop_rect=crop_rect)

    else:
        print(f"⚠️ [VideoDecoder] Backend desconocido '{backend}'. Usando OpenCV.")
        return OpenCVDecoder(video_path, profiler, crop_rect=crop_rect)
