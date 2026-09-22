import os
import json
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
    assert dt60.hour == 16 and dt60.minute == 17 and dt60.second == 16

    dt61 = get_clip_start_datetime("Camara Placas 2_20260909105651-20260909163038(61).mp4")
    assert dt61.hour == 16 and dt61.minute == 22 and dt61.second == 46

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


def test_detect_and_filter_vehicles_runs_postprocess(pipeline):
    """Regression: v2.1 used time.perf_counter() without importing time, crashing every clip."""
    box_in_gravel = [1300.0, 700.0, 1500.0, 900.0]  # ROI coords -> bottom-center (1700, 1500) in lane
    pipeline.vehicle_runner = MagicMock()
    pipeline.vehicle_runner.predict.return_value = ([box_in_gravel], [0.9], [2], {})

    boxes, confs, classes = pipeline._detect_and_filter_vehicles(
        np.zeros((1064, 2300, 3), dtype=np.uint8), 300, 600, timestamp=1.0
    )

    assert boxes == [box_in_gravel]
    assert confs == [0.9] and classes == [2]


def test_detect_and_filter_vehicles_preserves_large_trucks_in_salida(pipeline):
    """Regression: trucks crossing in SALIDA (e.g. PAB3439, PAB7630) must not be killed by parked van filter."""
    # Truck in SALIDA: fx1=438, fy1=600, fx2=1661, fy2=1349 (ROI: [138, 0, 1361, 749])
    truck_salida_box = [138.0, 0.0, 1361.0, 749.0]
    # Parked van on curb: fx1=300, fy1=850, fx2=420, fy2=1638 (ROI: [0.0, 250.0, 120.0, 1038.0])
    parked_van_box = [0.0, 250.0, 120.0, 1038.0]

    pipeline.vehicle_runner = MagicMock()
    pipeline.vehicle_runner.predict.return_value = (
        [truck_salida_box, parked_van_box],
        [0.85, 0.40],
        [7, 2],  # truck, car
        {}
    )

    boxes, confs, classes = pipeline._detect_and_filter_vehicles(
        np.zeros((1064, 2300, 3), dtype=np.uint8), 300, 600, timestamp=10.0
    )

    # Truck must be kept, parked van must be filtered out
    assert boxes == [truck_salida_box]
    assert confs == [0.85] and classes == [7]


def test_rejection_log_records_parked_curb_filter_only_when_enabled(pipeline, tmp_path):
    """Rejection records must only be produced when diagnostics.log_rejections is enabled,
    and must carry the correct reason (parked_curb_filter)."""
    parked_van_box = [0.0, 250.0, 120.0, 1038.0]
    pipeline.vehicle_runner = MagicMock()
    pipeline.vehicle_runner.predict.return_value = ([parked_van_box], [0.40], [2], {})
    pipeline._current_clip_id = "clipX.mp4"

    # Disabled by default: no records produced
    pipeline.rejection_logger.enabled = False
    pipeline.rejection_logger.output_dir = str(tmp_path)
    pipeline.rejection_logger.open("RUN_DISABLED")
    pipeline._detect_and_filter_vehicles(
        np.zeros((1064, 2300, 3), dtype=np.uint8), 300, 600, timestamp=10.0
    )
    assert pipeline.rejection_logger.path is None

    # Enabled: parked_curb_filter record produced
    pipeline.rejection_logger.enabled = True
    pipeline.rejection_logger.open("RUN_ENABLED")
    pipeline._detect_and_filter_vehicles(
        np.zeros((1064, 2300, 3), dtype=np.uint8), 300, 600, timestamp=10.0
    )
    pipeline.rejection_logger.close()

    with open(pipeline.rejection_logger.path) as f:
        records = [json.loads(line) for line in f if line.strip()]
    assert len(records) == 1
    assert records[0]["reason"] == "parked_curb_filter"
    assert records[0]["clip_id"] == "clipX.mp4"

