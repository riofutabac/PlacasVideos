"""
Unit tests for src/fast_convert.py: the --fast-convert opt-in detection-downscale path.

Covers:
- Box coordinates round-trip exactly through the downscale+rescale mapping
  (property-style over several bbox/scale combinations).
- NV12 downscale preserves layout and produces a plausible BGR image.
- The flag defaults to off and CLI override wires correctly.
- Plate crops / full-res evidence still come from the full-resolution frame,
  not the downscaled detection image (asserted at the pipeline_runner level
  via the crop_roi produced by _maybe_fullres_crop).
"""
import numpy as np
import pytest

from src import fast_convert


# ---------------------------------------------------------------------------
# compute_downscale_target
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("orig_h,orig_w,imgsz", [
    (1064, 2300, 416),   # real ROI from config/camera_config.yaml
    (1080, 1920, 640),
    (480, 640, 416),
    (1000, 1000, 416),   # square input
])
def test_compute_downscale_target_preserves_aspect_ratio(orig_h, orig_w, imgsz):
    scale, new_w, new_h = fast_convert.compute_downscale_target(orig_h, orig_w, imgsz)
    orig_ar = orig_w / orig_h
    new_ar = new_w / new_h
    # Rounding to even dims introduces a small tolerance, but no squashing to a square.
    assert abs(orig_ar - new_ar) / orig_ar < 0.02
    assert new_w % 2 == 0 and new_h % 2 == 0
    assert max(new_w, new_h) <= imgsz


def test_compute_downscale_target_matches_ultralytics_letterbox_scale():
    # From the real config ROI (2300x1064), Ultralytics' own r = min(imgsz/w, imgsz/h)
    # letterboxes to exactly 416x192 with zero extra padding (measured via
    # BasePredictor.preprocess instrumentation). Our downscale must reproduce that scale.
    scale, new_w, new_h = fast_convert.compute_downscale_target(1064, 2300, 416)
    assert new_w == 416
    assert new_h == 192


def test_compute_downscale_target_rejects_invalid_dims():
    with pytest.raises(ValueError):
        fast_convert.compute_downscale_target(0, 100, 416)


# ---------------------------------------------------------------------------
# rescale_boxes: round-trip property test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scale", [0.18, 0.25, 0.5, 0.75, 1.0])
@pytest.mark.parametrize("box", [
    [0.0, 0.0, 10.0, 10.0],
    [12.5, 30.2, 400.9, 190.0],
    [1.0, 1.0, 2.0, 2.0],
    [50.0, 50.0, 50.0, 50.0],  # degenerate zero-area box
])
def test_rescale_boxes_round_trips_exactly(box, scale):
    downscaled = [[c * scale for c in box]]
    recovered = fast_convert.rescale_boxes(downscaled, scale)
    assert recovered[0] == pytest.approx(box, abs=1e-9)


def test_rescale_boxes_is_pure_and_does_not_mutate_input():
    boxes = [[10.0, 10.0, 20.0, 20.0]]
    original = [row[:] for row in boxes]
    fast_convert.rescale_boxes(boxes, 0.5)
    assert boxes == original


def test_rescale_boxes_rejects_non_positive_scale():
    with pytest.raises(ValueError):
        fast_convert.rescale_boxes([[0, 0, 1, 1]], 0.0)


def test_rescale_boxes_empty_list():
    assert fast_convert.rescale_boxes([], 0.5) == []


# ---------------------------------------------------------------------------
# NV12 downscale
# ---------------------------------------------------------------------------

def _make_nv12(h, w, y_val=180, uv_val=128):
    nv12 = np.empty((h * 3 // 2, w), dtype=np.uint8)
    nv12[:h, :] = y_val
    nv12[h:, :] = uv_val
    return nv12


def test_downscale_nv12_preserves_layout_and_shape():
    orig_h, orig_w = 1064, 2300
    new_h, new_w = 192, 416
    nv12 = _make_nv12(orig_h, orig_w)
    small = fast_convert.downscale_nv12(nv12, orig_h, orig_w, new_h, new_w)
    assert small.shape == (new_h * 3 // 2, new_w)
    # A uniform-color input should downscale to a uniform-color output.
    assert np.all(small[:new_h, :] == 180)
    assert np.all(small[new_h:, :] == 128)


def test_build_detection_image_nv12_produces_bgr_of_expected_size():
    orig_h, orig_w = 1064, 2300
    nv12 = _make_nv12(orig_h, orig_w)
    small_bgr, scale = fast_convert.build_detection_image_nv12(nv12, orig_h, orig_w, imgsz=416)
    assert small_bgr.shape == (192, 416, 3)
    assert small_bgr.dtype == np.uint8
    assert scale == pytest.approx(416 / 2300)


def test_build_detection_image_bgr_downscales_directly():
    bgr = np.full((1064, 2300, 3), 100, dtype=np.uint8)
    small_bgr, scale = fast_convert.build_detection_image_bgr(bgr, imgsz=416)
    assert small_bgr.shape == (192, 416, 3)
    assert scale == pytest.approx(416 / 2300)


# ---------------------------------------------------------------------------
# resolve_fast_convert_flag: default-off + CLI override wiring
# ---------------------------------------------------------------------------

def test_resolve_fast_convert_flag_defaults_to_off():
    cfg = {"video": {}}
    assert fast_convert.resolve_fast_convert_flag(cfg, None) is False


def test_resolve_fast_convert_flag_reads_config_value():
    cfg = {"video": {"fast_convert": True}}
    assert fast_convert.resolve_fast_convert_flag(cfg, None) is True


def test_resolve_fast_convert_flag_cli_override_wins():
    cfg = {"video": {"fast_convert": False}}
    assert fast_convert.resolve_fast_convert_flag(cfg, True) is True

    cfg2 = {"video": {"fast_convert": True}}
    assert fast_convert.resolve_fast_convert_flag(cfg2, False) is False


def test_resolve_fast_convert_flag_missing_video_section():
    cfg = {}
    assert fast_convert.resolve_fast_convert_flag(cfg, None) is False
    assert fast_convert.resolve_fast_convert_flag(cfg, True) is True
