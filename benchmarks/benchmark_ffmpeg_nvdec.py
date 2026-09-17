"""
FFmpeg NVDEC Benchmark Script for Colab.
Benchmarks different FFmpeg decoding, filtering, and pixel format pipelines
to identify the absolute fastest pipeline for 2960x1664 HEVC @ 25fps.
"""
import os
import sys
import time
import subprocess
from pathlib import Path

def find_clip():
    candidates = [
        "/content/videos_local/clip_60.mp4",
        "/content/drive/MyDrive/Cam PL/Camara Placas 2_20260909105651-20260909163038(60).mp4",
        "Camara Placas 2_20260909105651-20260909163038(60).mp4"
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

def run_bench(name: str, cmd: list) -> float:
    print(f"\nTesting: {name}...")
    t0 = time.perf_counter()
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    dt = time.perf_counter() - t0
    if p.returncode != 0:
        err = p.stderr.strip()[-300:]
        print(f"  ❌ FAILED ({dt:.2f}s): {err}")
        return -1.0
    else:
        fps_equiv = 8223 / dt if dt > 0 else 0
        speedup = 330.5 / dt if dt > 0 else 0
        print(f"  ✅ SUCCESS: {dt:.2f}s | {fps_equiv:.1f} fps ({speedup:.2f}x realtime)")
        return dt

def main():
    clip = find_clip()
    if not clip:
        print("No clip found to benchmark.")
        sys.exit(1)

    print("=" * 70)
    print("BENCHMARK DE PIPELINES FFMPEG NVDEC (Clip 60: 8223 frames, 330.5s)")
    print(f"Video: {clip}")
    print("=" * 70)

    # 1. Baseline: Pure NVDEC decode + hwdownload NV12 (from notebook cell 10)
    run_bench("1. NVDEC CUDA -> hwdownload -> NV12 (null sink)", [
        "ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error",
        "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
        "-i", clip,
        "-map", "0:v:0", "-an", "-sn", "-dn",
        "-vf", "hwdownload,format=nv12",
        "-f", "null", "-"
    ])

    # 2. NVDEC CUDA -> hwdownload -> crop NV12
    run_bench("2. NVDEC CUDA -> hwdownload -> crop NV12 (2300x1064)", [
        "ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error",
        "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
        "-i", clip,
        "-map", "0:v:0", "-an", "-sn", "-dn",
        "-vf", "hwdownload,format=nv12,crop=2300:1064:300:600",
        "-f", "null", "-"
    ])

    # 3. NVDEC CUDA -> hwdownload -> crop NV12 -> pix_fmt bgr24 (v1.4.0 current)
    run_bench("3. NVDEC CUDA -> hwdownload -> crop NV12 -> pix_fmt bgr24 (v1.4.0)", [
        "ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error",
        "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
        "-i", clip,
        "-map", "0:v:0", "-an", "-sn", "-dn",
        "-vf", "hwdownload,format=nv12,crop=2300:1064:300:600",
        "-pix_fmt", "bgr24",
        "-vsync", "0",
        "-f", "null", "-"
    ])

    # 4. NVDEC CUDA -> hwdownload -> crop NV12 -> pix_fmt bgr24 with -threads 0
    run_bench("4. NVDEC CUDA -> hwdownload -> crop NV12 -> bgr24 (-threads 0)", [
        "ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error",
        "-threads", "0",
        "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
        "-i", clip,
        "-map", "0:v:0", "-an", "-sn", "-dn",
        "-vf", "hwdownload,format=nv12,crop=2300:1064:300:600",
        "-pix_fmt", "bgr24",
        "-vsync", "0",
        "-f", "null", "-"
    ])

    # 5. NVDEC CUDA -> hwdownload -> crop NV12 -> pix_fmt rgb24
    run_bench("5. NVDEC CUDA -> hwdownload -> crop NV12 -> pix_fmt rgb24", [
        "ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error",
        "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
        "-i", clip,
        "-map", "0:v:0", "-an", "-sn", "-dn",
        "-vf", "hwdownload,format=nv12,crop=2300:1064:300:600",
        "-pix_fmt", "rgb24",
        "-vsync", "0",
        "-f", "null", "-"
    ])

    # 6. CUVID decoder direct to bgr24
    run_bench("6. HEVC_CUVID -> crop -> bgr24", [
        "ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error",
        "-c:v", "hevc_cuvid",
        "-i", clip,
        "-map", "0:v:0", "-an", "-sn", "-dn",
        "-vf", "crop=2300:1064:300:600",
        "-pix_fmt", "bgr24",
        "-vsync", "0",
        "-f", "null", "-"
    ])

    # 7. Check CUDA scale_cuda support
    run_bench("7. scale_cuda (GPU-resident scaling)", [
        "ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error",
        "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
        "-i", clip,
        "-map", "0:v:0", "-an", "-sn", "-dn",
        "-vf", "scale_cuda=format=yuv420p,hwdownload,format=yuv420p",
        "-f", "null", "-"
    ])

    print("\n" + "=" * 70)
    print("FIN DEL BENCHMARK")
    print("=" * 70)

if __name__ == '__main__':
    main()
