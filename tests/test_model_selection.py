import os
from unittest.mock import patch, MagicMock
import pytest
from src.model_resolver import resolve_vehicle_model, DEFAULT_MODEL, FALLBACK_PYTORCH_MODEL

def test_resolve_existing_model(tmp_path):
    fake_model = tmp_path / "custom_detector.onnx"
    fake_model.write_text("dummy onnx content")

    resolved = resolve_vehicle_model(str(fake_model))
    assert resolved == str(fake_model)

def test_resolve_fallback_on_nonexistent_model(tmp_path):
    # When nonexistent model is provided, it should fall back to default
    nonexistent = str(tmp_path / "phantom_model.onnx")

    # If default model exists in repo (yolov8n.onnx)
    if os.path.exists(DEFAULT_MODEL):
        resolved = resolve_vehicle_model(nonexistent)
        assert resolved == DEFAULT_MODEL
    else:
        # If neither exists, should fall back to FALLBACK_PYTORCH_MODEL
        resolved = resolve_vehicle_model(nonexistent)
        assert resolved in (DEFAULT_MODEL, FALLBACK_PYTORCH_MODEL)

def test_resolve_export_attempt():
    with patch("ultralytics.YOLO") as mock_yolo:
        mock_instance = MagicMock()
        mock_yolo.return_value = mock_instance
        # Simulate export succeeding by making target exist
        with patch("os.path.exists") as mock_exists:
            # First call: target onnx doesn't exist; second call after export: exists
            mock_exists.side_effect = [False, True]
            resolved = resolve_vehicle_model("test_detector.onnx", imgsz=416)
            assert resolved == "test_detector.onnx"
            mock_instance.export.assert_called_once_with(format="onnx", imgsz=416, dynamic=True)

def test_resolve_export_failure_falls_back():
    with patch("ultralytics.YOLO") as mock_yolo:
        mock_instance = MagicMock()
        mock_instance.export.side_effect = RuntimeError("Export failed")
        mock_yolo.return_value = mock_instance

        # Simulate neither .pt nor .onnx existing
        with patch("os.path.exists", return_value=False):
            resolved = resolve_vehicle_model("failing_model.onnx", imgsz=416, default_model="default.onnx")
            assert resolved == FALLBACK_PYTORCH_MODEL
