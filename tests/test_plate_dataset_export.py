import os
import json
import pytest
from benchmarks.plate_dataset_export import export_plate_dataset, find_matching_gt

def test_find_matching_gt():
    gt_list = [
        {
            "ground_truth_id": "GT_60_1",
            "video_source": "Camara Placas 2_20260909105651-20260909163038(60).mp4",
            "approx_timestamp": 25.0,
            "plate_text": "PCW2492",
            "plate_legible": True,
            "province": "Pichincha"
        },
        {
            "ground_truth_id": "GT_61_1",
            "video_source": "Camara Placas 2_20260909105651-20260909163038(61).mp4",
            "approx_timestamp": 80.0,
            "plate_text": "PAC2573",
            "plate_legible": True,
            "province": "Pichincha"
        }
    ]

    # Matching clip 60 within 2.0s
    m1 = find_matching_gt("EVT_(60)_0023_27", 26.5, gt_list, tolerance_sec=5.0)
    assert m1 is not None
    assert m1["ground_truth_id"] == "GT_60_1"
    assert m1["plate_text"] == "PCW2492"

    # Not matching clip 61 when in clip 60
    m2 = find_matching_gt("EVT_(60)_0023_80", 80.0, gt_list, tolerance_sec=5.0)
    assert m2 is None

    # Matching clip 61
    m3 = find_matching_gt("EVT_(61)_0018_80", 80.2, gt_list, tolerance_sec=5.0)
    assert m3 is not None
    assert m3["ground_truth_id"] == "GT_61_1"

def test_export_plate_dataset(tmp_path):
    manifest_path = str(tmp_path / "manifest.json")
    gt_path = str(tmp_path / "ground_truth.json")
    output_path = str(tmp_path / "dataset_manifest.json")

    manifest_data = [
        {
            "event_id": "EVT_(60)_0023_27",
            "candidate_idx": 0,
            "timestamp": 25.4,
            "bbox": [10, 10, 50, 30],
            "quality_score": 0.85,
            "det_conf": 0.92,
            "plate_crop_path": "/fake/plate.jpg",
            "vehicle_crop_path": "/fake/veh.jpg"
        }
    ]
    gt_data = [
        {
            "ground_truth_id": "GT_60_1",
            "video_source": "Camara Placas 2_20260909105651-20260909163038(60).mp4",
            "approx_timestamp": 25.0,
            "plate_text": "PCW2492",
            "plate_legible": True,
            "province": "Pichincha",
            "vehicle_type": "car"
        }
    ]

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f)
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(gt_data, f)

    res = export_plate_dataset(manifest_path, gt_path, output_path)
    assert len(res) == 1
    assert res[0]["ground_truth_id"] == "GT_60_1"
    assert res[0]["ground_truth_plate"] == "PCW2492"
    assert res[0]["province"] == "Pichincha"
    assert os.path.exists(output_path)
