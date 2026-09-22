import json
import os
import pytest
from src.rejection_logger import RejectionLogger


def test_disabled_logger_writes_nothing(tmp_path):
    logger = RejectionLogger(enabled=False, output_dir=str(tmp_path))
    logger.open("RUN_TEST")
    logger.log(clip_id="clip1.mp4", timestamp=1.0, reason="parked_curb_filter", bbox=[0, 0, 10, 10])
    assert logger.path is None
    assert list(tmp_path.iterdir()) == []


def test_enabled_logger_writes_jsonl_with_reason(tmp_path):
    logger = RejectionLogger(enabled=True, output_dir=str(tmp_path))
    logger.open("RUN_TEST")
    logger.log(
        clip_id="clip1.mp4", timestamp=12.5, reason="parked_curb_filter",
        bbox=[100, 200, 300, 400], vehicle_class=2, confidence=0.55
    )
    logger.close()

    assert logger.path is not None
    assert os.path.exists(logger.path)
    with open(logger.path) as f:
        lines = [json.loads(line) for line in f if line.strip()]
    assert len(lines) == 1
    record = lines[0]
    assert record["reason"] == "parked_curb_filter"
    assert record["clip_id"] == "clip1.mp4"
    assert record["bbox"] == [100, 200, 300, 400]
    assert record["confidence"] == 0.55


def test_extra_fields_are_included(tmp_path):
    logger = RejectionLogger(enabled=True, output_dir=str(tmp_path))
    logger.open("RUN_TEST")
    logger.log(
        clip_id="clip2.mp4", timestamp=3.0, reason="plate_read_rejected",
        confidence=0.61, rejected_text="ABC123"
    )
    logger.close()
    with open(logger.path) as f:
        record = json.loads(f.readline())
    assert record["reason"] == "plate_read_rejected"
    assert record["rejected_text"] == "ABC123"


def test_open_is_noop_when_disabled(tmp_path):
    logger = RejectionLogger(enabled=False, output_dir=str(tmp_path))
    logger.open("RUN_TEST")
    assert logger.path is None
    logger.close()  # should not raise


def test_open_is_idempotent(tmp_path):
    logger = RejectionLogger(enabled=True, output_dir=str(tmp_path))
    logger.open("RUN_A")
    first_path = logger.path
    logger.open("RUN_B")  # should not reopen with a different run_id
    assert logger.path == first_path
    logger.close()
