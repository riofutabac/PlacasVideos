import time
import pytest
from src.timer_profiler import StageTimer, PipelineProfiler

def test_stage_timer():
    timer = StageTimer()
    assert timer.calls == 0
    assert timer.total_seconds == 0.0

    timer.start()
    time.sleep(0.01)
    elapsed = timer.stop()
    assert elapsed > 0.005
    assert timer.calls == 1
    assert timer.total_seconds >= elapsed

def test_pipeline_profiler_lifecycle():
    profiler = PipelineProfiler()
    assert "vehicle_detection" in profiler.stages

    profiler.start_clip()
    profiler.start_stage("vehicle_detection")
    time.sleep(0.005)
    profiler.stop_stage("vehicle_detection")

    assert len(profiler.yolo_latencies) == 1
    assert profiler.yolo_latencies[0] > 0.0

    clip_stats = profiler.finish_clip(total_frames=100, duration_sec=10.0)
    assert clip_stats["total_frames"] == 100
    assert clip_stats["duration_sec"] == 10.0
    assert clip_stats["speed_ratio"] > 0.0
    assert "vehicle_detection_seconds" in clip_stats

def test_yolo_block_latencies():
    profiler = PipelineProfiler()
    for i in range(15):
        profiler.yolo_latencies.append(float(i))
    blocks = profiler.get_yolo_block_latencies(block_size=5)
    assert len(blocks) == 3
    assert blocks[0]["count"] == 5
    assert blocks[0]["avg_ms"] == 2.0  # (0+1+2+3+4)/5
