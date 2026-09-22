"""
Timer Profiler and Execution Metrics for ALPR Pipeline.
Collects granular wall-clock time and invocation counts across every pipeline stage.
"""
import time
import threading
from typing import Dict

class StageTimer:
    def __init__(self):
        self.total_seconds: float = 0.0
        self.calls: int = 0
        self._start: float = 0.0
        self._lock = threading.Lock()

    def start(self):
        with self._lock:
            self._start = time.perf_counter()

    def stop(self):
        now = time.perf_counter()
        with self._lock:
            elapsed = now - self._start
            self.total_seconds += elapsed
            self.calls += 1
            return elapsed

    def add_duration(self, elapsed: float, calls: int = 1):
        with self._lock:
            self.total_seconds += elapsed
            self.calls += calls

class PipelineProfiler:
    def __init__(self):
        self.stages: Dict[str, StageTimer] = {
            'decode': StageTimer(),
            'decode_nvdec': StageTimer(),
            'frame_download': StageTimer(),
            'frame_conversion': StageTimer(),
            'frame_conversion_fullres': StageTimer(),
            'motion_gate': StageTimer(),
            'vehicle_detection': StageTimer(),
            'vehicle_preprocess': StageTimer(),
            'vehicle_inference': StageTimer(),
            'vehicle_postprocess': StageTimer(),
            'tracking': StageTimer(),
            'crossing_fsm': StageTimer(),
            'plate_detection': StageTimer(),
            'ocr': StageTimer(),
            'sqlite_write': StageTimer(),
            'image_write': StageTimer(),
        }
        self.wall_start = time.perf_counter()
        self.wall_end = 0.0
        self.source_frames = 0
        self.video_duration_seconds = 0.0
        self.yolo_latencies: list = []
        self._clip_start_perf = 0.0
        self._clip_baseline_stages = {}

    def start_clip(self):
        """Marks start of processing for an individual clip."""
        self._clip_start_perf = time.perf_counter()
        self._clip_baseline_stages = {
            name: st.total_seconds for name, st in self.stages.items()
        }

    def finish_clip(self, total_frames: int, duration_sec: float) -> Dict:
        """Returns performance metrics specific to the clip just processed."""
        clip_wall = time.perf_counter() - self._clip_start_perf
        clip_speed = duration_sec / clip_wall if clip_wall > 0 else 0.0
        stage_times = {
            name: round(st.total_seconds - self._clip_baseline_stages.get(name, 0.0), 3)
            for name, st in self.stages.items()
        }
        return {
            'wall_clock_seconds': round(clip_wall, 2),
            'duration_sec': round(duration_sec, 2),
            'speed_ratio': round(clip_speed, 2),
            'total_frames': total_frames,
            'decode_seconds': stage_times.get('decode', 0.0),
            'vehicle_detection_seconds': stage_times.get('vehicle_detection', 0.0),
            'plate_detection_seconds': stage_times.get('plate_detection', 0.0),
            'ocr_seconds': stage_times.get('ocr', 0.0),
            'stages': stage_times
        }

    def start_stage(self, stage_name: str):
        if stage_name in self.stages:
            self.stages[stage_name].start()

    def stop_stage(self, stage_name: str):
        if stage_name in self.stages:
            elapsed = self.stages[stage_name].stop()
            if stage_name == 'vehicle_detection':
                self.yolo_latencies.append(elapsed * 1000.0)
            return elapsed
        return 0.0

    def record_stage_time(self, stage_name: str, elapsed: float, calls: int = 1):
        if stage_name in self.stages:
            self.stages[stage_name].add_duration(elapsed, calls=calls)

    def get_yolo_block_latencies(self, block_size: int = 200) -> list:
        """Returns inference latency statistics grouped in temporal blocks (thermal tracking)."""
        blocks = []
        for i in range(0, len(self.yolo_latencies), block_size):
            chunk = self.yolo_latencies[i:i + block_size]
            if chunk:
                blocks.append({
                    'range': f"{i:4d} - {i+len(chunk):4d}",
                    'count': len(chunk),
                    'avg_ms': round(sum(chunk) / len(chunk), 1),
                    'min_ms': round(min(chunk), 1),
                    'max_ms': round(max(chunk), 1)
                })
        return blocks

    def finish(self, total_source_frames: int, video_duration_sec: float):
        self.wall_end = time.perf_counter()
        self.source_frames = total_source_frames
        self.video_duration_seconds = video_duration_sec

    def summary(self) -> Dict:
        wall_sec = (self.wall_end - self.wall_start) if self.wall_end > 0 else (time.perf_counter() - self.wall_start)
        speed_ratio = (self.video_duration_seconds / wall_sec) if wall_sec > 0 else 0.0
        
        breakdown = {
            'wall_clock_seconds': round(wall_sec, 2),
            'video_duration_seconds': round(self.video_duration_seconds, 2),
            'speed_ratio': round(speed_ratio, 2),
            'source_frames': self.source_frames,
            'source_fps_equivalent': round(self.source_frames / wall_sec, 1) if wall_sec > 0 else 0.0,
            'stages': {
                name: {
                    'seconds': round(st.total_seconds, 3),
                    'calls': st.calls,
                    'pct_wall': round((st.total_seconds / wall_sec) * 100, 1) if wall_sec > 0 else 0.0
                }
                for name, st in self.stages.items()
            },
            'yolo_blocks': self.get_yolo_block_latencies()
        }
        return breakdown

    def print_summary(self):
        s = self.summary()
        print("\n" + "="*60)
        print("PIPELINE PERFORMANCE SUMMARY")
        print("="*60)
        print(f"Video Duration:      {s['video_duration_seconds']}s")
        print(f"Wall Clock Time:     {s['wall_clock_seconds']}s")
        print(f"Speed Ratio:         {s['speed_ratio']}x Realtime")
        print(f"Source Frames:       {s['source_frames']} ({s['source_fps_equivalent']} fps equiv)")
        print("-"*60)
        print(f"{'Stage':<22} {'Time (s)':<10} {'Calls':<10} {'% Wall Clock':<10}")
        print("-"*60)
        for stage, data in s['stages'].items():
            print(f"{stage:<22} {data['seconds']:<10.2f} {data['calls']:<10} {data['pct_wall']:<10.1f}%")
        print("="*60 + "\n")
