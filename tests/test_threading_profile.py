import os
import pytest
from unittest.mock import patch
from src.threading_profile import configure_threading_profile

def test_threading_profile_cuda_colab():
    with patch("os.cpu_count", return_value=2):
        profile = configure_threading_profile(device="cuda")
        assert profile["cv2_threads"] == 1
        assert profile["torch_threads"] == 1
        assert profile["ort_intra_threads"] == 1
        assert profile["ort_inter_threads"] == 1

def test_threading_profile_cuda_large():
    with patch("os.cpu_count", return_value=16):
        profile = configure_threading_profile(device="cuda")
        assert profile["cv2_threads"] == 2
        assert profile["torch_threads"] == 2
        assert profile["ort_intra_threads"] == 2

def test_threading_profile_cpu():
    with patch("os.cpu_count", return_value=8):
        profile = configure_threading_profile(device="cpu")
        assert profile["cv2_threads"] == 4
        assert profile["torch_threads"] == 8
        assert profile["ort_intra_threads"] == 8

def test_threading_profile_env_override(monkeypatch):
    monkeypatch.setenv("ALPR_NUM_THREADS", "3")
    profile = configure_threading_profile(device="cpu")
    assert profile["cv2_threads"] == 3
    assert profile["torch_threads"] == 3
    assert profile["ort_intra_threads"] == 3

def test_threading_profile_config_override():
    cfg = {"cv2": 2, "torch": 4, "ort_intra": 4}
    profile = configure_threading_profile(device="cuda", config_threads=cfg)
    assert profile["cv2_threads"] == 2
    assert profile["torch_threads"] == 4
    assert profile["ort_intra_threads"] == 4
