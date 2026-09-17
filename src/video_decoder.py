"""
Decoupled Video Decoder Module for ALPR Pipeline.
Supports two backends:
  1. OpenCVDecoder: CPU baseline and fallback.
  2. NVDECDecoder: GPU hardware-accelerated H.264/HEVC decode via NVIDIA NVDEC (PyNvVideoCodec or FFmpeg NVDEC subprocess).

Auto-detects stream codec using ffprobe (e.g. HEVC vs H.264).
Maintains exact frame indexing and timestamps (frame_idx / fps).
Records granular profiling:
  - decode_nvdec
  - frame_download
  - frame_conversion
"""
import os
import sys
import json
import shutil
import subprocess
import time
from abc import ABC, abstractmethod
from typing import Generator, Tuple, Optional, Dict, Any
import numpy as np
import cv2

def probe_video_metadata(video_path: str) -> Dict[str, Any]:
    """Probes video metadata (codec, width, height, fps, total_frames) using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,r_frame_rate,nb_frames,duration",
        "-of", "json",
        video_path
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        data = json.loads(res.stdout)
        stream = data["streams"][0]
        codec = stream.get("codec_name", "").lower()
        width = int(stream.get("width", 2960))
        height = int(stream.get("height", 1664))
        
        r_fps = stream.get("r_frame_rate", "25/1")
        if "/" in r_fps:
            num, den = r_fps.split("/")
            fps = float(num) / float(den) if float(den) != 0 else 25.0
        else:
            fps = float(r_fps)
            
        nb_frames = stream.get("nb_frames")
        total_frames = int(nb_frames) if nb_frames and nb_frames.isdigit() else 0
        if total_frames == 0:
            dur = stream.get("duration")
            if dur:
                total_frames = int(float(dur) * fps)
                
        return {
            "codec": codec,
            "width": width,
            "height": height,
            "fps": fps,
            "total_frames": total_frames
        }
    except Exception:
        # Fallback to OpenCV metadata
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 2960
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1664
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        cap.release()
        return {
            "codec": "hevc",
            "width": w,
            "height": h,
            "fps": fps,
            "total_frames": total_frames
        }

def check_nvdec_available() -> Dict[str, bool]:
    """Checks if NVIDIA GPU and NVDEC/CUVID are available on the system."""
    has_nvidia = False
    try:
        res = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode == 0:
            has_nvidia = True
    except Exception:
        has_nvidia = False

    if not has_nvidia:
        return {"cuda": False, "hevc_cuvid": False, "h264_cuvid": False, "pynv": False}

    has_cuda_hwaccel = False
    try:
        res = subprocess.run(["ffmpeg", "-hide_banner", "-hwaccels"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if "cuda" in res.stdout:
            has_cuda_hwaccel = True
    except Exception:
        pass

    has_hevc_cuvid = False
    has_h264_cuvid = False
    try:
        res = subprocess.run(["ffmpeg", "-hide_banner", "-decoders"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if "hevc_cuvid" in res.stdout:
            has_hevc_cuvid = True
        if "h264_cuvid" in res.stdout:
            has_h264_cuvid = True
    except Exception:
        pass

    has_pynv = False
    try:
        import PyNvVideoCodec
        has_pynv = True
    except Exception:
        has_pynv = False

    return {
        "cuda": has_cuda_hwaccel,
        "hevc_cuvid": has_hevc_cuvid,
        "h264_cuvid": has_h264_cuvid,
        "pynv": has_pynv
    }

def _read_exact(stream, n_bytes: int) -> bytes:
    """Reads exactly n_bytes from stream, looping over pipe chunks."""
    buf = bytearray(n_bytes)
    view = memoryview(buf)
    pos = 0
    while pos < n_bytes:
        n = stream.readinto(view[pos:])
        if not n:
            break
        pos += n
    if pos < n_bytes:
        return bytes(view[:pos])
    return bytes(buf)


class BaseVideoDecoder(ABC):
    def __init__(self, video_path: str, profiler: Optional[Any] = None):
        self.video_path = video_path
        self.profiler = profiler
        self.fps: float = 25.0
        self.width: int = 2960
        self.height: int = 1664
        self.total_frames: int = 0
        self.duration_sec: float = 0.0

    @abstractmethod
    def __iter__(self) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        """Yields (frame_idx, timestamp_seconds, frame_bgr_ndarray)"""
        pass

    def release(self):
        pass


class OpenCVDecoder(BaseVideoDecoder):
    def __init__(self, video_path: str, profiler: Optional[Any] = None):
        super().__init__(video_path, profiler)
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"OpenCV no pudo abrir el archivo de video: {video_path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration_sec = (self.total_frames / self.fps) if self.fps > 0 else 0.0

    def __iter__(self) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        frame_idx = 0
        while self.cap.isOpened():
            if self.profiler:
                self.profiler.start_stage('decode')
            ret, frame = self.cap.read()
            if self.profiler:
                self.profiler.stop_stage('decode')
            if not ret or frame is None:
                break
            timestamp = frame_idx / self.fps
            yield frame_idx, timestamp, frame
            frame_idx += 1

    def release(self):
        if self.cap:
            self.cap.release()


class FFmpegNVDECDecoder(BaseVideoDecoder):
    """
    NVIDIA NVDEC Hardware-accelerated decoder using FFmpeg pipe.
    Decodes HEVC/H.264 on GPU via CUDA/CUVID, transfers NV12 to host memory (hwdownload),
    and converts to BGR via OpenCV AVX2 SIMD.
    """
    def __init__(self, video_path: str, profiler: Optional[Any] = None, nvdec_info: Optional[Dict[str, bool]] = None):
        super().__init__(video_path, profiler)
        meta = probe_video_metadata(video_path)
        self.codec = meta["codec"]
        self.width = meta["width"]
        self.height = meta["height"]
        self.fps = meta["fps"]
        self.total_frames = meta["total_frames"]
        self.duration_sec = (self.total_frames / self.fps) if self.fps > 0 else 0.0
        self.nvdec_info = nvdec_info or check_nvdec_available()
        self.proc: Optional[subprocess.Popen] = None

    def _build_cmd(self, use_cuvid_fallback: bool = False) -> list:
        if use_cuvid_fallback:
            cuvid_codec = "hevc_cuvid" if self.codec in ("hevc", "h265") else "h264_cuvid"
            return [
                "ffmpeg",
                "-hide_banner",
                "-nostdin",
                "-loglevel", "error",
                "-c:v", cuvid_codec,
                "-i", self.video_path,
                "-map", "0:v:0",
                "-an", "-sn", "-dn",
                "-f", "rawvideo",
                "-pix_fmt", "nv12",
                "-"
            ]
        
        # Primary benchmarked command (verified 21s in Colab Cell 10)
        return [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-loglevel", "error",
            "-hwaccel", "cuda",
            "-hwaccel_output_format", "cuda",
            "-i", self.video_path,
            "-map", "0:v:0",
            "-an", "-sn", "-dn",
            "-vf", "hwdownload,format=nv12",
            "-f", "rawvideo",
            "-pix_fmt", "nv12",
            "-"
        ]

    def __iter__(self) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        cmd = self._build_cmd(use_cuvid_fallback=False)
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=10485760  # 10 MB buffer
        )

        nv12_frame_size = self.width * self.height * 3 // 2
        nv12_h = self.height * 3 // 2
        w = self.width
        frame_idx = 0

        stdout = self.proc.stdout
        while True:
            # Stage 1: NVDEC hardware decode + DMA transfer wait
            if self.profiler:
                self.profiler.start_stage('decode')
                self.profiler.start_stage('decode_nvdec')

            raw_bytes = _read_exact(stdout, nv12_frame_size)

            if self.profiler:
                self.profiler.stop_stage('decode_nvdec')

            if len(raw_bytes) < nv12_frame_size:
                if self.profiler:
                    self.profiler.stop_stage('decode')
                
                # Check if it failed immediately at frame 0
                if frame_idx == 0:
                    err_msg = ""
                    try:
                        err_msg = self.proc.stderr.read().decode('utf-8', errors='replace').strip()
                    except Exception:
                        pass
                    print(f"⚠️ [FFmpegNVDECDecoder] NVDEC CUDA no entregó cuadros (stderr: {err_msg}).")
                    
                    # Try CUVID fallback
                    self.release()
                    cuvid_cmd = self._build_cmd(use_cuvid_fallback=True)
                    print(f"🔄 [FFmpegNVDECDecoder] Intentando fallback a CUVID...")
                    self.proc = subprocess.Popen(
                        cuvid_cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        bufsize=10485760
                    )
                    stdout = self.proc.stdout
                    test_bytes = _read_exact(stdout, nv12_frame_size)
                    if len(test_bytes) == nv12_frame_size:
                        print("✅ [FFmpegNVDECDecoder] CUVID funcionó correctamente.")
                        raw_bytes = test_bytes
                    else:
                        cuvid_err = ""
                        try:
                            cuvid_err = self.proc.stderr.read().decode('utf-8', errors='replace').strip()
                        except Exception:
                            pass
                        print(f"⚠️ [FFmpegNVDECDecoder] CUVID también falló (stderr: {cuvid_err}). Activando fallback a OpenCV...")
                        self.release()
                        opencv_dec = OpenCVDecoder(self.video_path, self.profiler)
                        yield from opencv_dec
                        return

                if len(raw_bytes) < nv12_frame_size:
                    break

            # Stage 2: Frame buffer download / shaping
            if self.profiler:
                self.profiler.start_stage('frame_download')
            nv12_arr = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((nv12_h, w))
            if self.profiler:
                self.profiler.stop_stage('frame_download')

            # Stage 3: Frame color conversion (NV12 -> BGR)
            if self.profiler:
                self.profiler.start_stage('frame_conversion')
            frame = cv2.cvtColor(nv12_arr, cv2.COLOR_YUV2BGR_NV12)
            if self.profiler:
                self.profiler.stop_stage('frame_conversion')
                self.profiler.stop_stage('decode')

            timestamp = frame_idx / self.fps
            yield frame_idx, timestamp, frame
            frame_idx += 1

        self.release()

    def release(self):
        if self.proc:
            try:
                if self.proc.stdout:
                    self.proc.stdout.close()
            except Exception:
                pass
            try:
                if self.proc.stderr:
                    self.proc.stderr.close()
            except Exception:
                pass
            try:
                self.proc.terminate()
                self.proc.wait(timeout=1.0)
            except Exception:
                pass
            self.proc = None


class PyNvVideoCodecDecoder(BaseVideoDecoder):
    """
    NVIDIA PyNvVideoCodec Decoder (if installed and compatible).
    """
    def __init__(self, video_path: str, profiler: Optional[Any] = None):
        super().__init__(video_path, profiler)
        import PyNvVideoCodec as nvc
        self.nvc = nvc
        meta = probe_video_metadata(video_path)
        self.codec = meta["codec"]
        self.width = meta["width"]
        self.height = meta["height"]
        self.fps = meta["fps"]
        self.total_frames = meta["total_frames"]
        self.duration_sec = (self.total_frames / self.fps) if self.fps > 0 else 0.0

    def __iter__(self) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        raise NotImplementedError("PyNvVideoCodec backend not fully initialized")


class NVDECDecoder(BaseVideoDecoder):
    """
    Composite NVDEC Decoder with priority:
      A. PyNvVideoCodec (if available and functional)
      B. FFmpeg NVDEC Subprocess (hardware CUVID/CUDA)
      C. Fallback to OpenCV if NVDEC unavailable or crashes
    """
    def __init__(self, video_path: str, profiler: Optional[Any] = None):
        super().__init__(video_path, profiler)
        self.nvdec_info = check_nvdec_available()
        self.active_decoder: BaseVideoDecoder

        # Priority A: PyNvVideoCodec
        if self.nvdec_info.get("pynv"):
            try:
                self.active_decoder = PyNvVideoCodecDecoder(video_path, profiler)
                self.fps = self.active_decoder.fps
                self.width = self.active_decoder.width
                self.height = self.active_decoder.height
                self.total_frames = self.active_decoder.total_frames
                self.duration_sec = self.active_decoder.duration_sec
                return
            except Exception:
                pass

        # Priority B: FFmpeg NVDEC Subprocess
        if self.nvdec_info.get("cuda") or self.nvdec_info.get("hevc_cuvid") or self.nvdec_info.get("h264_cuvid"):
            try:
                self.active_decoder = FFmpegNVDECDecoder(video_path, profiler, self.nvdec_info)
                self.fps = self.active_decoder.fps
                self.width = self.active_decoder.width
                self.height = self.active_decoder.height
                self.total_frames = self.active_decoder.total_frames
                self.duration_sec = self.active_decoder.duration_sec
                return
            except Exception as e:
                print(f"⚠️ Error al inicializar FFmpegNVDECDecoder ({e}). Fallback a OpenCV.")

        # Fallback to OpenCV
        self.active_decoder = OpenCVDecoder(video_path, profiler)
        self.fps = self.active_decoder.fps
        self.width = self.active_decoder.width
        self.height = self.active_decoder.height
        self.total_frames = self.active_decoder.total_frames
        self.duration_sec = self.active_decoder.duration_sec

    def __iter__(self) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        return self.active_decoder.__iter__()

    def release(self):
        if hasattr(self, 'active_decoder') and self.active_decoder:
            self.active_decoder.release()


def create_decoder(video_path: str, backend: str = "auto", profiler: Optional[Any] = None) -> BaseVideoDecoder:
    """
    Factory function for video decoders.
    backend: 'auto', 'nvdec', or 'opencv'
    """
    backend_clean = (backend or "auto").lower().strip()

    if backend_clean == "auto":
        info = check_nvdec_available()
        if info["cuda"] or info["hevc_cuvid"] or info["h264_cuvid"] or info["pynv"]:
            print(f"⚡ [VideoDecoder] Backend 'auto' -> Activando NVDEC (NVIDIA GPU Hardware Acceleration)")
            return NVDECDecoder(video_path, profiler)
        else:
            print(f"ℹ️ [VideoDecoder] Backend 'auto' -> Activando OpenCV (CPU Fallback)")
            return OpenCVDecoder(video_path, profiler)

    elif backend_clean in ("nvdec", "cuda", "cuvid"):
        print(f"⚡ [VideoDecoder] Backend explícito: NVDEC")
        return NVDECDecoder(video_path, profiler)

    elif backend_clean in ("opencv", "cpu"):
        print(f"ℹ️ [VideoDecoder] Backend explícito: OpenCV")
        return OpenCVDecoder(video_path, profiler)

    else:
        print(f"⚠️ [VideoDecoder] Backend desconocido '{backend}'. Usando OpenCV.")
        return OpenCVDecoder(video_path, profiler)
