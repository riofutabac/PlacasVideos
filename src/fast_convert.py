"""
Fast-convert helpers for the opt-in --fast-convert vehicle-detection path.

Today the pipeline always converts the whole NV12 ROI to full-resolution BGR
(`decoder.to_bgr`) before handing the array to Ultralytics, which then
internally letterboxes it down to `imgsz` for the forward pass. That full-res
conversion (LUT + cvtColor over ~2.45 Mpx) is the single biggest slice of
`frame_conversion` time.

When fast-convert is enabled, we instead:
  1. Downscale the NV12 planes directly (cheap: resize on int8 data) to the
     same aspect-preserving scale Ultralytics would have computed internally,
     then run the (much smaller) NV12->BGR conversion for detection only.
  2. Feed that small BGR image to the detector. Ultralytics returns box
     coordinates already unletterboxed to the shape of the image we gave it
     (see `res.orig_shape` / `Boxes.xyxy`), so no manual pad bookkeeping is
     needed on our side for the detector call itself.
  3. Map returned boxes back to full-resolution ROI-local coordinates by
     dividing by the scale factor used in step 1.
  4. Only pay for the full-resolution NV12->BGR conversion (today's path) on
     frames where at least one vehicle survived filtering, since that is the
     only case where quality ranking / crop extraction needs full-res pixels.

All functions here are pure (no mutation of caller-owned arrays beyond an
explicit `dst` buffer, mirroring `nv12_to_bgr_full_range`).
"""
from typing import List, Tuple

import numpy as np
import cv2

from src.decoder.base import nv12_to_bgr_full_range


def compute_downscale_target(orig_h: int, orig_w: int, imgsz: int) -> Tuple[float, int, int]:
    """
    Replicates Ultralytics' own LetterBox scale computation (r = min(imgsz/h, imgsz/w))
    so the pre-downscale we do here matches, pixel-for-pixel in aspect ratio, the scale
    Ultralytics would apply internally to the full-resolution ROI. Returns (scale, new_w, new_h)
    with new_w/new_h rounded to even integers (required for NV12 4:2:0 chroma alignment).
    """
    if orig_h <= 0 or orig_w <= 0:
        raise ValueError(f"Invalid ROI dimensions: {orig_h}x{orig_w}")
    scale = min(imgsz / orig_w, imgsz / orig_h)
    new_w = max(2, int(round(orig_w * scale / 2.0)) * 2)
    new_h = max(2, int(round(orig_h * scale / 2.0)) * 2)
    return scale, new_w, new_h


def downscale_nv12(nv12_roi: np.ndarray, orig_h: int, orig_w: int, new_h: int, new_w: int) -> np.ndarray:
    """
    Downscales an NV12 buffer (Y plane followed by interleaved UV plane) to
    (new_h, new_w) while preserving the NV12 layout, without ever materializing
    a full-resolution BGR image.
    """
    y_plane = nv12_roi[:orig_h, :orig_w]
    uv_plane = nv12_roi[orig_h:orig_h + orig_h // 2, :orig_w]

    y_small = cv2.resize(y_plane, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    uv_pairs = uv_plane.reshape(orig_h // 2, orig_w // 2, 2)
    uv_small = cv2.resize(uv_pairs, (new_w // 2, new_h // 2), interpolation=cv2.INTER_LINEAR)
    uv_small = uv_small.reshape(new_h // 2, new_w)

    small_nv12 = np.empty((new_h * 3 // 2, new_w), dtype=np.uint8)
    small_nv12[:new_h, :] = y_small
    small_nv12[new_h:new_h + new_h // 2, :] = uv_small
    return small_nv12


def build_detection_image_nv12(nv12_roi: np.ndarray, orig_h: int, orig_w: int, imgsz: int) -> Tuple[np.ndarray, float]:
    """Full NV12 -> small BGR pipeline for detection input. Returns (small_bgr, scale)."""
    scale, new_w, new_h = compute_downscale_target(orig_h, orig_w, imgsz)
    small_nv12 = downscale_nv12(nv12_roi, orig_h, orig_w, new_h, new_w)
    small_bgr = nv12_to_bgr_full_range(small_nv12, new_h, new_w)
    return small_bgr, scale


def build_detection_image_bgr(bgr_roi: np.ndarray, imgsz: int) -> Tuple[np.ndarray, float]:
    """Downscale an already-decoded BGR ROI (non-NV12 decoders, e.g. OpenCV on CPU)."""
    orig_h, orig_w = bgr_roi.shape[:2]
    scale, new_w, new_h = compute_downscale_target(orig_h, orig_w, imgsz)
    small_bgr = cv2.resize(bgr_roi, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return small_bgr, scale


def rescale_boxes(boxes: List[List[float]], scale: float) -> List[List[float]]:
    """Maps detector boxes from the downscaled-image coordinate space back to
    full-resolution ROI-local coordinates. `boxes` and the return value are both
    new lists; the input is never mutated (immutability per project style)."""
    if scale <= 0:
        raise ValueError(f"Invalid scale factor: {scale}")
    inv = 1.0 / scale
    return [[x1 * inv, y1 * inv, x2 * inv, y2 * inv] for (x1, y1, x2, y2) in boxes]


def log_letterbox_shape_once(vehicle_runner, roi_h: int, roi_w: int, imgsz: int) -> None:
    """
    Measures (not assumes) the real tensor shape Ultralytics feeds the model today
    for a ROI of this size, by instrumenting BasePredictor.preprocess for a single
    dummy forward pass, and prints it once at startup. This is the shape the
    fast-convert downscale path is designed to reproduce (constraint: preserve
    aspect ratio, no squashing to a square imgsz x imgsz tensor).
    """
    if vehicle_runner.model is None:
        return
    try:
        import ultralytics.engine.predictor as predmod
        captured = {}
        original_preprocess = predmod.BasePredictor.preprocess

        def _patched(self, im):
            out = original_preprocess(self, im)
            captured['shape'] = tuple(out.shape)
            return out

        predmod.BasePredictor.preprocess = _patched
        try:
            dummy = np.zeros((roi_h, roi_w, 3), dtype=np.uint8)
            vehicle_runner.model(dummy, imgsz=imgsz, verbose=False, conf=0.25, classes=vehicle_runner.classes, device=vehicle_runner.device)
        finally:
            predmod.BasePredictor.preprocess = original_preprocess

        if 'shape' in captured:
            n, c, h, w = captured['shape']
            print(
                f"[fast-convert] Real Ultralytics preprocessed tensor for ROI {roi_w}x{roi_h} @ imgsz={imgsz}: "
                f"{w}x{h} (NCHW={captured['shape']}). fast-convert reproduces this scale exactly, no padding to a square tensor."
            )
    except Exception as e:
        print(f"[fast-convert] Could not measure real tensor shape ({e}); proceeding without startup log.")


def resolve_fast_convert_flag(cfg: dict, override) -> bool:
    """Applies an optional CLI override onto cfg['video']['fast_convert'] and returns the resolved bool."""
    if override is not None:
        cfg.setdefault('video', {})['fast_convert'] = override
    return bool(cfg.get('video', {}).get('fast_convert', False))


def announce_fast_convert(vehicle_runner, crop_rect: dict, imgsz: int) -> None:
    """One-line startup banner plus the measured letterbox shape (see log_letterbox_shape_once)."""
    roi_h = crop_rect['y_max'] - crop_rect['y_min']
    roi_w = crop_rect['x_max'] - crop_rect['x_min']
    print("⚡ [fast-convert] Habilitado: detección sobre imagen reducida, full-res reservado a ranking/evidencia.")
    log_letterbox_shape_once(vehicle_runner, roi_h, roi_w, imgsz)
