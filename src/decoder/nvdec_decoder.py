"""
NVIDIA NVDEC Hardware Accelerated Video Decoders.
Includes FFmpegNVDECDecoder (with threaded producer queue and DMA hwdownload),
PyNvVideoCodecDecoder stub, and NVDECDecoder composite fallback.
"""
import subprocess
import queue
import threading
from typing import Generator, Tuple, Optional, Dict, Any
import numpy as np

from .base import BaseVideoDecoder, probe_video_metadata, _read_exact, check_nvdec_available
from .opencv_decoder import OpenCVDecoder


class FFmpegNVDECDecoder(BaseVideoDecoder):
    """
    NVIDIA NVDEC Hardware-accelerated decoder with decoupled threaded producer.
    1. HW NVDEC decodes HEVC/H.264 on GPU.
    2. hwdownload transfers NV12 to host memory via DMA.
    3. Producer slices raw NV12 directly to ROI.
    4. Queues frames into Queue(maxsize=queue_size) so NVDEC and YOLO overlap.
    """
    def __init__(
        self,
        video_path: str,
        profiler: Optional[Any] = None,
        nvdec_info: Optional[Dict[str, bool]] = None,
        crop_rect: Optional[Dict[str, int]] = None,
        queue_size: int = 128
    ):
        super().__init__(video_path, profiler, crop_rect=crop_rect)
        self.frame_format = "nv12"
        meta = probe_video_metadata(video_path)
        self.codec = meta["codec"]
        self.width = meta["width"]
        self.height = meta["height"]
        self.fps = meta["fps"]
        self.total_frames = meta["total_frames"]
        self.duration_sec = (self.total_frames / self.fps) if self.fps > 0 else 0.0
        self.is_full_range = meta.get("is_full_range", False)
        self.nvdec_info = nvdec_info or check_nvdec_available()
        self.queue_size = queue_size

        self.proc: Optional[subprocess.Popen] = None
        self._frame_queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._stop_event: threading.Event = threading.Event()
        self._producer_thread: Optional[threading.Thread] = None
        self._producer_error: Optional[Exception] = None

    def _build_cmd(self, use_cuvid: bool = True, use_hw_crop: bool = True) -> list:
        if use_cuvid:
            cuvid_codec = "hevc_cuvid" if self.codec in ("hevc", "h265") else "h264_cuvid"
            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-nostdin",
                "-loglevel", "error",
                "-c:v", cuvid_codec,
            ]
            if self.is_cropped and use_hw_crop:
                top = self.cy1
                bottom = self.height - self.cy2
                left = self.cx1
                right = self.width - self.cx2
                cmd.extend(["-crop", f"{top}x{bottom}x{left}x{right}"])
            cmd.extend([
                "-i", self.video_path,
                "-map", "0:v:0",
                "-an", "-sn", "-dn",
            ])
            if self.is_cropped and not use_hw_crop:
                cmd.extend(["-vf", f"crop={self.crop_w}:{self.crop_h}:{self.cx1}:{self.cy1}"])
            cmd.extend([
                "-pix_fmt", "nv12",
                "-vsync", "0",
                "-f", "rawvideo",
                "-"
            ])
            return cmd

        # Hardware NVDEC via CUDA hwaccel + DMA hwdownload + crop
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-loglevel", "error",
            "-hwaccel", "cuda",
            "-hwaccel_output_format", "cuda",
            "-i", self.video_path,
            "-map", "0:v:0",
            "-an", "-sn", "-dn",
        ]
        if self.is_cropped:
            cmd.extend(["-vf", f"hwdownload,format=nv12,crop={self.crop_w}:{self.crop_h}:{self.cx1}:{self.cy1}"])
        else:
            cmd.extend(["-vf", "hwdownload,format=nv12"])

        cmd.extend([
            "-pix_fmt", "nv12",
            "-vsync", "0",
            "-f", "rawvideo",
            "-"
        ])
        return cmd

    def _producer_worker(self):
        out_w = self.crop_w if self.is_cropped else self.width
        out_h = self.crop_h if self.is_cropped else self.height
        nv12_h = out_h * 3 // 2
        frame_bytes = out_w * nv12_h
        frame_idx = 0

        configs = [
            ("cuvid_hw_crop", True, True),
            ("cuda_hwdownload", False, False),
            ("cuvid_vf_crop", True, False)
        ]
        raw_buf = None
        last_err = ""
        for cfg_name, use_cuvid, use_hw in configs:
            cmd = self._build_cmd(use_cuvid=use_cuvid, use_hw_crop=use_hw)
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=10485760  # 10 MB buffer
            )
            stdout = self.proc.stdout
            if self.profiler:
                self.profiler.start_stage('decode_nvdec')
            raw_buf = _read_exact(stdout, frame_bytes)
            if self.profiler:
                self.profiler.stop_stage('decode_nvdec')

            if raw_buf is not None:
                break
            else:
                try:
                    last_err = self.proc.stderr.read().decode('utf-8', errors='replace').strip()
                except Exception:
                    pass
                try:
                    self.proc.terminate()
                    self.proc.wait(timeout=1.0)
                except Exception:
                    pass
                self.proc = None

        if raw_buf is None:
            raise RuntimeError(f"All NVDEC pipeline options failed: {last_err}")

        try:
            while not self._stop_event.is_set():
                if raw_buf is None:
                    if self.profiler:
                        self.profiler.start_stage('decode_nvdec')
                    raw_buf = _read_exact(self.proc.stdout, frame_bytes)
                    if self.profiler:
                        self.profiler.stop_stage('decode_nvdec')

                if raw_buf is None:
                    break

                if self.profiler:
                    self.profiler.start_stage('frame_download')
                frame = np.frombuffer(raw_buf, dtype=np.uint8).reshape((nv12_h, out_w))
                if self.profiler:
                    self.profiler.stop_stage('frame_download')

                timestamp = frame_idx / self.fps

                pushed = False
                while not self._stop_event.is_set():
                    try:
                        self._frame_queue.put((frame_idx, timestamp, frame), timeout=0.1)
                        pushed = True
                        break
                    except queue.Full:
                        continue

                if not pushed and self._stop_event.is_set():
                    break

                raw_buf = None
                frame_idx += 1

        except Exception as e:
            self._producer_error = e
        finally:
            try:
                self._frame_queue.put(None, timeout=0.5)
            except Exception:
                pass

    def __iter__(self) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        self._stop_event.clear()
        self._producer_error = None

        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except Exception:
                pass

        self._producer_thread = threading.Thread(target=self._producer_worker, daemon=True)
        self._producer_thread.start()

        while True:
            if self.profiler:
                self.profiler.start_stage('decode')
            try:
                item = self._frame_queue.get(timeout=30.0)
            except queue.Empty:
                if self.profiler:
                    self.profiler.stop_stage('decode')
                if self._producer_error:
                    raise self._producer_error
                break

            if self.profiler:
                self.profiler.stop_stage('decode')

            if item is None:
                if self._producer_error:
                    raise self._producer_error
                break

            frame_idx, timestamp, frame = item
            yield frame_idx, timestamp, frame

        self.release()

    def release(self):
        self._stop_event.set()
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except Exception:
                pass
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
        if self._producer_thread and self._producer_thread.is_alive():
            self._producer_thread.join(timeout=1.0)
            self._producer_thread = None


class PyNvVideoCodecDecoder(BaseVideoDecoder):
    """NVIDIA PyNvVideoCodec Decoder (if installed and compatible)."""
    def __init__(self, video_path: str, profiler: Optional[Any] = None, crop_rect: Optional[Dict[str, int]] = None):
        super().__init__(video_path, profiler, crop_rect=crop_rect)
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
      B. FFmpeg NVDEC Subprocess (hardware CUVID/CUDA with threaded producer and ROI crop)
      C. Fallback to OpenCV if NVDEC unavailable or crashes
    """
    def __init__(
        self,
        video_path: str,
        profiler: Optional[Any] = None,
        crop_rect: Optional[Dict[str, int]] = None,
        queue_size: int = 128
    ):
        super().__init__(video_path, profiler, crop_rect=crop_rect)
        self.nvdec_info = check_nvdec_available()
        self.active_decoder: BaseVideoDecoder

        # Priority A: PyNvVideoCodec
        if self.nvdec_info.get("pynv"):
            try:
                self.active_decoder = PyNvVideoCodecDecoder(video_path, profiler, crop_rect=crop_rect)
                self.fps = self.active_decoder.fps
                self.width = self.active_decoder.width
                self.height = self.active_decoder.height
                self.total_frames = self.active_decoder.total_frames
                self.duration_sec = self.active_decoder.duration_sec
                self.is_cropped = self.active_decoder.is_cropped
                return
            except Exception:
                pass

        # Priority B: FFmpeg NVDEC Subprocess
        if self.nvdec_info.get("cuda") or self.nvdec_info.get("hevc_cuvid") or self.nvdec_info.get("h264_cuvid"):
            try:
                self.active_decoder = FFmpegNVDECDecoder(
                    video_path, profiler, self.nvdec_info, crop_rect=crop_rect, queue_size=queue_size
                )
                self.fps = self.active_decoder.fps
                self.width = self.active_decoder.width
                self.height = self.active_decoder.height
                self.total_frames = self.active_decoder.total_frames
                self.duration_sec = self.active_decoder.duration_sec
                self.is_cropped = self.active_decoder.is_cropped
                return
            except Exception as e:
                print(f"⚠️ Error al inicializar FFmpegNVDECDecoder ({e}). Fallback a OpenCV.")

        # Fallback to OpenCV
        self.active_decoder = OpenCVDecoder(video_path, profiler, crop_rect=crop_rect)
        self.fps = self.active_decoder.fps
        self.width = self.active_decoder.width
        self.height = self.active_decoder.height
        self.total_frames = self.active_decoder.total_frames
        self.duration_sec = self.active_decoder.duration_sec
        self.is_cropped = self.active_decoder.is_cropped

    @property
    def frame_format(self) -> str:
        return getattr(self.active_decoder, 'frame_format', 'bgr') if hasattr(self, 'active_decoder') else 'bgr'

    @frame_format.setter
    def frame_format(self, value: str):
        if hasattr(self, 'active_decoder'):
            self.active_decoder.frame_format = value
        self._frame_format = value

    def to_bgr(self, frame: np.ndarray, dst: Optional[np.ndarray] = None) -> np.ndarray:
        return self.active_decoder.to_bgr(frame, dst=dst)

    def __iter__(self) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        return self.active_decoder.__iter__()

    def release(self):
        if hasattr(self, 'active_decoder') and self.active_decoder:
            self.active_decoder.release()
