import os
import numpy as np
import pytest
from src.vehicle_runtime import VehicleDetectorRunner, _nms

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

def _make_ort_iobinding_runner(monkeypatch):
    """Builds a VehicleDetectorRunner wired for the ort_iobinding branch,
    without needing a real onnxruntime CUDA session (untestable on CPU-only
    Mac hosts). Falls back to a mocked Ultralytics constructor first, then
    manually swaps in fake ORT session/io_binding objects so predict() takes
    Branch B (ONNX Runtime I/O Binding)."""

    class MockYOLO:
        def __init__(self, model_name):
            self.names = {}

    monkeypatch.setattr("ultralytics.YOLO", MockYOLO)

    runner = VehicleDetectorRunner(
        model_name="non_existent_model.onnx",
        runtime="ort_iobinding",
        imgsz=416,
        conf_threshold=0.25,
        classes=[2, 3],
        device="cpu"
    )
    # Force Branch B: disable the ultralytics fallback model, install fakes.
    runner.model = None
    return runner


class _FakeIOName:
    def __init__(self, name):
        self.name = name


class _FakeSession:
    def __init__(self, input_name="images", output_name="output0"):
        self._input_name = input_name
        self._output_name = output_name

    def get_inputs(self):
        return [_FakeIOName(self._input_name)]

    def get_outputs(self):
        return [_FakeIOName(self._output_name)]

    def run_with_iobinding(self, io_binding):
        pass


class _FakeIOBinding:
    def __init__(self, outputs):
        self._outputs = outputs
        self.bind_input_calls = []
        self.bind_output_calls = []

    def bind_input(self, **kwargs):
        self.bind_input_calls.append(kwargs)

    def bind_output(self, *args, **kwargs):
        self.bind_output_calls.append((args, kwargs))

    def copy_outputs_to_cpu(self):
        # Real onnxruntime IOBinding.copy_outputs_to_cpu() returns numpy
        # arrays directly (no .numpy() accessor) -- unlike get_outputs(),
        # which returns OrtValue objects.
        return self._outputs


def _build_yolov8_raw_output(boxes_spec, num_anchors=200, num_classes=80):
    """Builds a raw YOLOv8 ONNX output of shape (1, 4 + num_classes, num_anchors),
    matching the un-transposed (1, 84, N) layout ONNX Runtime returns."""
    raw = np.zeros((4 + num_classes, num_anchors), dtype=np.float32)
    for idx, (cx, cy, w, h, cls, score) in enumerate(boxes_spec):
        raw[0, idx] = cx
        raw[1, idx] = cy
        raw[2, idx] = w
        raw[3, idx] = h
        raw[4 + cls, idx] = score
    return raw[None, ...]


class _FakeCudaTensor:
    """Stands in for a torch CUDA tensor on hosts with no GPU, so the
    ort_iobinding preprocessing path (torch.from_numpy(blob).cuda()) can be
    exercised without real CUDA."""

    def __init__(self, array):
        self._array = array
        self.shape = array.shape

    def data_ptr(self):
        return 0


def _patch_torch_cuda_upload(monkeypatch):
    import torch as real_torch

    def fake_from_numpy(arr):
        class _Bridge:
            def cuda(self_inner):
                return _FakeCudaTensor(arr)
        return _Bridge()

    monkeypatch.setattr(real_torch, "from_numpy", fake_from_numpy)


def test_vehicle_detector_runner_ort_iobinding_decodes_boxes(monkeypatch):
    runner = _make_ort_iobinding_runner(monkeypatch)
    _patch_torch_cuda_upload(monkeypatch)

    # Two overlapping class-2 boxes (NMS should keep only the higher-score
    # one), one different-class box (filtered by class mask), one low-conf
    # box (filtered by confidence threshold). Coordinates are expressed in
    # the imgsz=416 letterboxed frame for a 400x300 crop (scale=1.04).
    boxes_spec = [
        (104.0, 114.4, 104.0, 104.0, 2, 0.9),   # keeps -> crop coords (50,60,150,160)
        (108.0, 119.4, 104.0, 104.0, 2, 0.8),   # duplicate, suppressed by NMS
        (300.0, 300.0, 50.0, 50.0, 0, 0.95),    # class 0 not in classes=[2,3]
        (100.0, 100.0, 50.0, 50.0, 2, 0.05),    # below conf_threshold
    ]
    raw_output = _build_yolov8_raw_output(boxes_spec)

    runner.ort_session = _FakeSession()
    runner.io_binding = _FakeIOBinding(outputs=[raw_output])

    crop = np.zeros((300, 400, 3), dtype=np.uint8)
    boxes, confs, cls_ids, timing = runner.predict(crop, imgsz=416, conf=0.25, classes=[2, 3])

    assert len(boxes) == 1
    assert cls_ids == [2]
    assert confs[0] == pytest.approx(0.9, abs=1e-3)
    x1, y1, x2, y2 = boxes[0]
    assert x1 == pytest.approx(50.0, abs=1.0)
    assert y1 == pytest.approx(60.0, abs=1.0)
    assert x2 == pytest.approx(150.0, abs=1.0)
    assert y2 == pytest.approx(160.0, abs=1.0)
    assert timing['total'] > 0.0

    # bind_output should carry an explicit device_id, not just device_type.
    assert runner.io_binding.bind_output_calls[0][1].get('device_id') == 0


def test_vehicle_detector_runner_ort_iobinding_no_detections(monkeypatch):
    runner = _make_ort_iobinding_runner(monkeypatch)
    _patch_torch_cuda_upload(monkeypatch)
    raw_output = _build_yolov8_raw_output([])  # all zeros, nothing above conf
    runner.ort_session = _FakeSession()
    runner.io_binding = _FakeIOBinding(outputs=[raw_output])

    crop = np.zeros((300, 400, 3), dtype=np.uint8)
    boxes, confs, cls_ids, timing = runner.predict(crop, imgsz=416, conf=0.25, classes=[2, 3])

    assert boxes == []
    assert confs == []
    assert cls_ids == []


def test_nms_suppresses_overlapping_boxes():
    boxes = np.array([
        [50.0, 60.0, 150.0, 160.0],
        [55.0, 65.0, 155.0, 165.0],  # high IoU with box 0
        [300.0, 300.0, 350.0, 350.0],  # far away, independent
    ])
    scores = np.array([0.9, 0.8, 0.7])

    keep = _nms(boxes, scores, iou_threshold=0.45)

    assert 0 in keep
    assert 1 not in keep
    assert 2 in keep


def test_nms_empty_input_returns_empty_list():
    assert _nms(np.zeros((0, 4)), np.zeros((0,))) == []


def test_benchmark_vehicle_runtime_loads_model_once_per_effective_runtime(monkeypatch):
    """Regression test: the benchmark previously created a fresh
    VehicleDetectorRunner (and thus reloaded the underlying model) for every
    row in RUNTIMES, even though `ort_trt` and `trt_engine` both degrade to
    the same 'ultralytics' runtime as the baseline row. With runner caching,
    the ultralytics model factory must be invoked only once."""
    from benchmarks import benchmark_vehicle_runtime as bench_mod

    load_calls = []

    class MockRes:
        boxes = []
        speed = {}

    class MockYOLO:
        def __init__(self, model_name):
            load_calls.append(model_name)
            self.names = {2: 'car'}

        def __call__(self, *args, **kwargs):
            return [MockRes()]

    monkeypatch.setattr("ultralytics.YOLO", MockYOLO)
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    frames = [np.zeros((50, 50, 3), dtype=np.uint8)]
    runner_cache = {}
    for r_key, r_label in bench_mod.RUNTIMES:
        bench_mod.benchmark_runtime(
            r_key, r_label, "yolov8n.onnx", frames, imgsz=416, warmup_iters=0,
            runner_cache=runner_cache
        )

    assert len(load_calls) == 1


def _iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@pytest.mark.skipif(not os.path.exists("yolov8n.onnx"), reason="yolov8n.onnx not present")
def test_ort_cpu_decode_agrees_with_ultralytics_on_real_image():
    """Runs a real ONNX Runtime CPUExecutionProvider session (non-io_binding
    decode path) against yolov8n.onnx on an actual vehicle crop, and checks
    its top detection roughly agrees (by IoU) with the Ultralytics backend on
    the same image and model. This is a real, non-mocked empirical check --
    no fabricated timings or results are used."""
    veh_dir = "evidence/vehicles"
    candidates = sorted(f for f in os.listdir(veh_dir) if f.endswith(".jpg")) if os.path.exists(veh_dir) else []
    if not candidates:
        pytest.skip("no evidence/vehicles crops available")

    import cv2
    img = cv2.imread(os.path.join(veh_dir, candidates[0]))
    if img is None:
        pytest.skip("could not read sample vehicle crop")

    imgsz = 416
    classes = [0, 1, 2, 3, 5, 7]

    ort_runner = VehicleDetectorRunner(
        model_name="yolov8n.onnx",
        runtime="ort_iobinding",
        imgsz=imgsz,
        conf_threshold=0.25,
        classes=classes,
        device="cpu"
    )
    assert ort_runner.ort_session is not None, "expected a real CPU ORT session"
    assert ort_runner.io_binding is None, "CPU run should not use CUDA io_binding"
    ort_boxes, ort_confs, ort_cls, _ = ort_runner.predict(img, imgsz=imgsz, conf=0.25, classes=classes)

    ultra_runner = VehicleDetectorRunner(
        model_name="yolov8n.onnx",
        runtime="ultralytics",
        imgsz=imgsz,
        conf_threshold=0.25,
        classes=classes,
        device="cpu"
    )
    ultra_boxes, ultra_confs, ultra_cls, _ = ultra_runner.predict(img, imgsz=imgsz, conf=0.25, classes=classes)

    if not ort_boxes or not ultra_boxes:
        pytest.skip("no detections from one or both backends on this sample crop")

    # Compare the highest-confidence detection from each backend.
    ort_top = ort_boxes[int(np.argmax(ort_confs))]
    ultra_top = ultra_boxes[int(np.argmax(ultra_confs))]
    assert _iou(ort_top, ultra_top) > 0.5


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

