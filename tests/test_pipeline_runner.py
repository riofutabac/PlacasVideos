import os
from unittest.mock import MagicMock, patch
from datetime import datetime
import numpy as np
import pytest

from src.pipeline_types import (
    VehicleFrameCandidate,
    TrackState,
    get_clip_start_datetime,
    compute_file_hash
)
from src.pipeline_runner import ALPRPipeline

def test_get_clip_start_datetime():
    dt60 = get_clip_start_datetime("Camara Placas 2_20260909105651-20260909163038(60).mp4")
    assert dt60.hour == 16 and dt60.minute == 17 and dt60.second == 20

    dt61 = get_clip_start_datetime("Camara Placas 2_20260909105651-20260909163038(61).mp4")
    assert dt61.hour == 16 and dt61.minute == 22 and dt61.second == 45

    dt_other = get_clip_start_datetime("video_unknown.mp4")
    assert dt_other.hour == 16 and dt_other.minute == 0

def test_compute_file_hash(tmp_path):
    f = tmp_path / "sample.bin"
    f.write_bytes(b"hello world")
    h = compute_file_hash(str(f))
    assert len(h) == 32
    assert isinstance(h, str)

@pytest.fixture
def pipeline(tmp_path):
    db_file = str(tmp_path / "test_events.sqlite")
    # Patch YOLO and ALPR to avoid downloading / heavy GPU loads in unit tests
    with patch("src.pipeline_runner.YOLO") as mock_yolo, \
         patch("src.pipeline_runner.ALPR") as mock_alpr:
        instance = ALPRPipeline(
            config_path="config/camera_config.yaml",
            db_path=db_file,
            model_name_override="yolov8n.onnx"
        )
        return instance

def test_pipeline_initialization_and_roi(pipeline):
    assert pipeline.crop_rect["x_min"] == 300
    assert pipeline.crossing_line_id == "LINE_MAIN_LASTRE"

    # Gravel polygon test
    # Point inside gravel lane
    inside_pt = (1200, 1000)
    assert pipeline.is_point_in_gravel(inside_pt)

    # Point outside (top left corner)
    outside_pt = (100, 100)
    assert not pipeline.is_point_in_gravel(outside_pt)

def test_pipeline_finalize_vehicle_event(pipeline):
    veh_crop = np.zeros((100, 150, 3), dtype=np.uint8)
    cand = VehicleFrameCandidate(
        score=0.88,
        timestamp=25.0,
        vehicle_crop=veh_crop,
        bbox_in_full_frame=(1000, 800, 1200, 950)
    )
    track = TrackState(
        track_id=42,
        first_timestamp=20.0,
        last_timestamp=26.0,
        vehicle_class="car",
        direction="ENTRADA",
        crossing_timestamp=25.0,
        best_vehicle_frames=[cand]
    )

    # Mock plate processor to return plate text
    pipeline.plate_processor.detect_plate_candidates = MagicMock(return_value=[{"crop": veh_crop}])
    pipeline.plate_processor.recognize_plate_candidates = MagicMock(
        return_value=("PCW2492", veh_crop, 0.95, 0.92, 25.0, [("PCW2492", 0.92)])
    )
    pipeline.plate_processor.save_evidence = MagicMock(
        return_value=("evidence/vehicles/mock_veh.jpg", "evidence/plates/mock_plate.jpg")
    )

    start_dt = datetime(2026, 9, 9, 16, 17, 20)
    event = pipeline.finalize_vehicle_event(
        track=track,
        video_source="clip(60).mp4",
        clip_hash="hash123",
        run_id="RUN_TEST",
        clip_start_dt=start_dt
    )

    assert event is not None
    assert event["event_id"] == "EVT_(60)_0042_25"
    assert event["direction"] == "ENTRADA"
    assert event["plate_corrected"] == "PCW2492"
    assert event["confidence_ocr"] == 0.92

    # Verify event is in SQLite database
    stored = pipeline.db.get_events_for_run(run_id="RUN_TEST")
    assert len(stored) == 1
    assert stored[0]["event_id"] == "EVT_(60)_0042_25"


class _FakeCroppedNV12Decoder:
    """Mimics NVDECDecoder with hardware crop: yields raw (h*3/2, w) NV12 buffers."""

    def __init__(self, h: int, w: int, n_frames: int = 3):
        self.crop_h, self.crop_w = h, w
        self.height, self.width = h, w
        self.is_cropped = True
        self.frame_format = "nv12"
        self.duration_sec = float(n_frames)
        self._n = n_frames
        self.to_bgr_shapes = []

    def __iter__(self):
        for i in range(self._n):
            yield i, float(i), np.full((self.crop_h * 3 // 2, self.crop_w), 128, dtype=np.uint8)

    def to_bgr(self, frame, dst=None):
        from src.decoder import nv12_to_bgr_full_range
        self.to_bgr_shapes.append(frame.shape)
        return nv12_to_bgr_full_range(frame, self.crop_h, self.crop_w, dst=dst)

    def release(self):
        pass


def test_nvdec_cropped_frames_convert_full_nv12_buffer(pipeline, tmp_path):
    """Regression: v1.9 passed only the luma plane to to_bgr, crashing every NVDEC clip."""
    h, w = 64, 96
    decoder = _FakeCroppedNV12Decoder(h, w)
    bgr_buf = np.empty((h, w, 3), dtype=np.uint8)
    video = tmp_path / "Camara Placas 2_20260909105651-20260909163038(60).mp4"
    video.write_bytes(b"fake")

    with patch.object(pipeline, "_init_clip_decoder",
                      return_value=(decoder, bgr_buf, True, True, h, w, 0, 0, w, h)), \
         patch.object(pipeline.motion_gate, "should_run_detector", return_value=True) as gate, \
         patch.object(pipeline, "_detect_and_filter_vehicles", return_value=([], [], [])):
        pipeline.process_video_file(str(video), run_id="RUN_TEST")

    assert decoder.to_bgr_shapes == [(h * 3 // 2, w)] * 3
    assert all(call.args[0].shape == (h, w) for call in gate.call_args_list)
