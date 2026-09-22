"""
Wiring-level tests for --fast-convert: verifies the flag defaults to off leaving
the legacy path untouched, and that when on, full-resolution crops (used for
quality ranking / evidence / plate detection) still come from the full-resolution
source, not the downscaled detection image, even though detection itself runs on
the downscaled image.
"""
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from src.pipeline_runner import ALPRPipeline


@pytest.fixture
def pipeline_factory(tmp_path):
    def _make(fast_convert_override=None):
        db_file = str(tmp_path / "test_events.sqlite")
        with patch("src.pipeline_runner.YOLO"), patch("src.pipeline_runner.ALPR"):
            return ALPRPipeline(
                config_path="config/camera_config.yaml",
                db_path=db_file,
                model_name_override="yolov8n.onnx",
                fast_convert_override=fast_convert_override
            )
    return _make


def test_fast_convert_defaults_to_off(pipeline_factory):
    pipeline = pipeline_factory()
    assert pipeline.fast_convert is False


def test_fast_convert_flag_on(pipeline_factory):
    pipeline = pipeline_factory(fast_convert_override=True)
    assert pipeline.fast_convert is True


def test_legacy_path_untouched_when_flag_off(pipeline_factory):
    """With the flag off, _run_detection_stage must call decoder.to_bgr exactly as
    today (full-res every analyzed frame) and never touch the fast-convert helpers."""
    pipeline = pipeline_factory(fast_convert_override=False)
    pipeline.vehicle_runner.predict = MagicMock(return_value=([], [], [], {}))

    decoder = MagicMock()
    full_res_bgr = np.full((1064, 2300, 3), 7, dtype=np.uint8)
    decoder.to_bgr.return_value = full_res_bgr
    raw_roi = np.zeros((1064 * 3 // 2, 2300), dtype=np.uint8)

    with patch("src.vehicle_filtering.fast_convert") as mocked_fc:
        boxes, confs, classes, crop_roi = pipeline._run_detection_stage(
            raw_roi, True, 1064, 2300, decoder, None, 300, 600, 1.0
        )
        mocked_fc.build_detection_image_nv12.assert_not_called()
        mocked_fc.build_detection_image_bgr.assert_not_called()

    decoder.to_bgr.assert_called_once()
    assert crop_roi is full_res_bgr


def test_fast_convert_skips_fullres_conversion_when_no_detections(pipeline_factory):
    """The whole point of --fast-convert: frames with zero surviving vehicle
    detections must never pay for the full-resolution NV12->BGR conversion."""
    pipeline = pipeline_factory(fast_convert_override=True)
    pipeline.vehicle_runner.predict = MagicMock(return_value=([], [], [], {}))

    decoder = MagicMock()
    raw_roi = np.full((1064 * 3 // 2, 2300), 128, dtype=np.uint8)

    boxes, confs, classes, crop_roi = pipeline._run_detection_stage(
        raw_roi, True, 1064, 2300, decoder, None, 300, 600, 1.0
    )

    assert boxes == []
    assert crop_roi is None
    decoder.to_bgr.assert_not_called()


def test_fast_convert_fullres_crop_comes_from_original_source_not_downscaled(pipeline_factory):
    """When a vehicle survives filtering, the crop passed downstream for quality
    ranking / evidence must be the full-resolution frame (matching dimensions and
    content of the original ROI), not the small detection image."""
    pipeline = pipeline_factory(fast_convert_override=True)

    # A box squarely inside the gravel polygon (see config/camera_config.yaml) so it
    # survives filtering. Coordinates are in downscaled-image space; scale below
    # matches compute_downscale_target(1064, 2300, 416).
    scale = 416 / 2300
    box_full_roi = [1500 - 300, 1000 - 600, 1600 - 300, 1100 - 600]  # ROI-local, full-res
    box_downscaled = [c * scale for c in box_full_roi]
    pipeline.vehicle_runner.predict = MagicMock(
        return_value=([box_downscaled], [0.9], [2], {})
    )

    full_res_bgr = np.full((1064, 2300, 3), 42, dtype=np.uint8)
    decoder = MagicMock()
    decoder.to_bgr.return_value = full_res_bgr
    raw_roi = np.zeros((1064 * 3 // 2, 2300), dtype=np.uint8)

    boxes, confs, classes, crop_roi = pipeline._run_detection_stage(
        raw_roi, True, 1064, 2300, decoder, None, 300, 600, 1.0
    )

    assert len(boxes) == 1
    # Box must be rescaled back to full-resolution ROI-local coordinates.
    assert boxes[0] == pytest.approx(box_full_roi, abs=1.0)
    # crop_roi is the full-resolution frame, dimensions match the original ROI, not
    # the small detection image (192x416).
    assert crop_roi.shape == (1064, 2300, 3)
    assert crop_roi is full_res_bgr
    decoder.to_bgr.assert_called_once()
