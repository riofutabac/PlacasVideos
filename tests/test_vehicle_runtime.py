import os
import numpy as np
import pytest
from src.vehicle_runtime import VehicleDetectorRunner

def test_vehicle_detector_runner_ultralytics_mock(monkeypatch):
    class MockBox:
        def __init__(self):
            import torch
            self.xyxy = [torch.tensor([10.0, 20.0, 100.0, 120.0])]
            self.conf = [torch.tensor(0.92)]
            self.cls = [torch.tensor(2.0)]

    class MockRes:
        def __init__(self):
            self.boxes = [MockBox()]
            self.speed = {'preprocess': 1.2, 'inference': 15.4, 'postprocess': 0.8}

    class MockYOLO:
        def __init__(self, model_name):
            self.names = {2: 'car', 3: 'motorcycle'}
        def __call__(self, *args, **kwargs):
            return [MockRes()]

    monkeypatch.setattr("ultralytics.YOLO", MockYOLO)

    runner = VehicleDetectorRunner(
        model_name="mock_yolo.onnx",
        runtime="ultralytics",
        imgsz=416,
        conf_threshold=0.25,
        classes=[2, 3],
        device="cpu"
    )

    assert runner.names == {2: 'car', 3: 'motorcycle'}

    crop = np.zeros((300, 400, 3), dtype=np.uint8)
    boxes, confs, cls_ids, timing = runner.predict(crop)

    assert len(boxes) == 1
    assert boxes[0] == [10.0, 20.0, 100.0, 120.0]
    assert confs[0] == pytest.approx(0.92, 0.01)
    assert cls_ids[0] == 2
    assert timing['preprocess'] == pytest.approx(0.0012, 0.0001)
    assert timing['inference'] == pytest.approx(0.0154, 0.0001)
    assert timing['postprocess'] == pytest.approx(0.0008, 0.0001)
    assert timing['total'] > 0.0

    # Test callable
    call_res = runner(crop)
    assert len(call_res) == 1

    # Test warmup
    runner.warmup()

def test_vehicle_detector_runner_fallback(monkeypatch):
    class MockYOLO:
        def __init__(self, model_name):
            self.names = {0: 'person'}
        def __call__(self, *args, **kwargs):
            class EmptyRes:
                boxes = []
                speed = {}
            return [EmptyRes()]

    monkeypatch.setattr("ultralytics.YOLO", MockYOLO)

    # Runtime ort_iobinding with non-existent file should fall back to Ultralytics
    runner = VehicleDetectorRunner(
        model_name="non_existent_model.onnx",
        runtime="ort_iobinding",
        device="cpu"
    )
    assert runner.model is not None
    boxes, confs, cls_ids, timing = runner.predict(np.zeros((100, 100, 3), dtype=np.uint8))
    assert boxes == []

def test_benchmark_vehicle_runtime(tmp_path, monkeypatch):
    from benchmarks.benchmark_vehicle_runtime import run_vehicle_runtime_benchmark
    out_csv = str(tmp_path / "runtime_bench.csv")
    results = run_vehicle_runtime_benchmark(
        model_name="yolov8n.onnx",
        n_frames=2,
        imgsz=416,
        output_csv=out_csv
    )
    assert len(results) == 4
    assert os.path.exists(out_csv)

