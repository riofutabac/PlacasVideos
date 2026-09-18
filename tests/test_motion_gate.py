import pytest
import numpy as np
from src.motion_gate import AdaptiveMotionGate

def test_motion_gate_initial_state_and_sentinel():
    gate = AdaptiveMotionGate(
        sleep_sentinel_fps=2.0,
        preactive_fps=5.0,
        active_fps=8.0,
        active_hold_seconds=2.5,
        motion_threshold_ratio=0.005
    )
    assert gate.state == "SLEEP"

    # Static frame (no motion)
    frame_static = np.ones((72, 128), dtype=np.uint8) * 100

    # First call at t=0.0 initializes background and runs initial frame 0 check
    res0 = gate.should_run_detector(frame_static, 0.0)
    assert res0 is True

    # Sentinel check: min_interval = 1.0 / 2.0 = 0.5s.
    # At t=0.2s: elapsed = 0.2s < 0.5s -> should return False
    assert gate.should_run_detector(frame_static, 0.2) is False

    # At t=0.6s: elapsed = 0.6s >= 0.5s -> Watchdog sentinel should fire (True)
    assert gate.should_run_detector(frame_static, 0.6) is True
    assert gate.state == "SLEEP"

def test_motion_gate_transitions_to_preactive_on_motion():
    gate = AdaptiveMotionGate(
        sleep_sentinel_fps=2.0,
        preactive_fps=5.0,
        active_fps=8.0,
        active_hold_seconds=2.5,
        motion_threshold_ratio=0.005
    )

    # Initial frame at t=0.0 (initializes background and runs)
    frame1 = np.ones((72, 128), dtype=np.uint8) * 50
    gate.should_run_detector(frame1, 0.0)

    # Subsequent frame at t=0.25s with large motion (half the frame changes color)
    frame2 = frame1.copy()
    frame2[0:36, :] = 220

    # Calling should_run_detector directly on frame2
    fired = gate.should_run_detector(frame2, 0.25)
    assert gate.state == "PRE-ACTIVE"
    assert fired is True

def test_motion_gate_active_state_and_hysteresis():
    gate = AdaptiveMotionGate(
        sleep_sentinel_fps=2.0,
        preactive_fps=5.0,
        active_fps=8.0,
        active_hold_seconds=2.5,
        motion_threshold_ratio=0.005
    )
    frame = np.ones((72, 128), dtype=np.uint8) * 100
    gate.should_run_detector(frame, 0.0)

    # YOLO detects a vehicle at t=10.0s
    gate.notify_vehicle_detected(10.0)
    assert gate.state == "ACTIVE"

    # Frame at t=10.15s (min_interval for 8 FPS = 0.125s)
    # Elapsed = 10.15 - 0.0 = 10.15 >= 0.125 -> fires True and maintains ACTIVE
    fired = gate.should_run_detector(frame, 10.15)
    assert fired is True
    assert gate.state == "ACTIVE"

    # Hysteresis test: at t=11.5s (1.5s after vehicle, < 2.5s active_hold)
    # Even without new motion, state must stay ACTIVE
    gate.should_run_detector(frame, 11.5)
    assert gate.state == "ACTIVE"

    # After active_hold_seconds expires (t=13.0s > 10.0 + 2.5s)
    # State should decay to SLEEP
    gate.should_run_detector(frame, 13.0)
    assert gate.state == "SLEEP"

def test_motion_gate_reset():
    gate = AdaptiveMotionGate()
    gate.notify_vehicle_detected(5.0)
    assert gate.state == "ACTIVE"

    gate.reset()
    assert gate.state == "SLEEP"
    assert gate.last_vehicle_timestamp == -1e9
    assert gate.prev_gray is None

def test_motion_gate_handles_different_input_shapes():
    gate = AdaptiveMotionGate()

    # 2D grayscale
    img_gray = np.ones((100, 200), dtype=np.uint8) * 80
    assert gate.check_motion(img_gray) is False

    # 3D BGR
    img_bgr = np.ones((100, 200, 3), dtype=np.uint8) * 80
    assert gate.check_motion(img_bgr) is False

    # None and empty
    assert gate.check_motion(None) is False
    assert gate.check_motion(np.array([])) is False
