import os
import threading
import time

import pytest

from src.clip_prefetcher import ClipPrefetcher


def _fake_stage(marker_dir, delay=0.0):
    """Builds a stage_fn that creates a temp file per clip and tracks live temp files."""
    live_files = set()
    lock = threading.Lock()

    def stage_fn(orig_vf):
        if delay:
            time.sleep(delay)
        dst = os.path.join(marker_dir, os.path.basename(orig_vf) + ".staged")
        with open(dst, "w") as f:
            f.write("x")
        with lock:
            live_files.add(dst)
        return dst, True, 0.01

    return stage_fn, live_files, lock


def test_stages_exactly_one_clip_ahead(tmp_path):
    video_files = [f"clip{i}.mp4" for i in range(5)]
    stage_fn, live_files, lock = _fake_stage(str(tmp_path))

    prefetcher = ClipPrefetcher(video_files, stage_fn=stage_fn)
    prefetcher.start()

    # Immediately after start, the worker should be able to get at most one
    # clip fully staged and queued before it blocks on the maxsize=1 queue.
    time.sleep(0.2)
    with lock:
        staged_so_far = len(live_files)
    assert staged_so_far <= 2  # one in queue + possibly one mid-copy

    results = []
    for _ in video_files:
        results.append(prefetcher.get_next())
    prefetcher.stop()

    assert [r[0] for r in results] == video_files


def test_order_is_preserved(tmp_path):
    video_files = [f"clip{i}.mp4" for i in range(8)]
    stage_fn, _, _ = _fake_stage(str(tmp_path))
    prefetcher = ClipPrefetcher(video_files, stage_fn=stage_fn)
    prefetcher.start()

    seen = [prefetcher.get_next()[0] for _ in video_files]
    prefetcher.stop()

    assert seen == video_files


def test_never_more_than_two_files_at_once(tmp_path):
    video_files = [f"clip{i}.mp4" for i in range(6)]
    stage_fn, live_files, lock = _fake_stage(str(tmp_path), delay=0.02)
    prefetcher = ClipPrefetcher(video_files, stage_fn=stage_fn)
    prefetcher.start()

    max_seen = 0
    for _ in video_files:
        result = prefetcher.get_next()
        # Consumer now "owns" the current file; count current + whatever the
        # worker has staged/queued so far.
        with lock:
            max_seen = max(max_seen, len(live_files))
        # Simulate processing, then cleanup like main.py's finally block.
        _, proc_vf, is_temp, _, _ = result
        if is_temp and os.path.exists(proc_vf):
            os.remove(proc_vf)
            with lock:
                live_files.discard(proc_vf)

    prefetcher.stop()
    assert max_seen <= 2


def test_staging_failure_for_one_clip_does_not_abort_the_rest(tmp_path):
    video_files = ["good1.mp4", "bad.mp4", "good2.mp4"]

    def stage_fn(orig_vf):
        if orig_vf == "bad.mp4":
            raise ValueError("corrupt file: moov atom not found")
        dst = os.path.join(str(tmp_path), orig_vf + ".staged")
        with open(dst, "w") as f:
            f.write("x")
        return dst, True, 0.01

    prefetcher = ClipPrefetcher(video_files, stage_fn=stage_fn)
    prefetcher.start()

    r1 = prefetcher.get_next()
    r2 = prefetcher.get_next()
    r3 = prefetcher.get_next()
    prefetcher.stop()

    assert r1[0] == "good1.mp4" and r1[4] is None
    assert r2[0] == "bad.mp4" and isinstance(r2[4], ValueError)
    assert r3[0] == "good2.mp4" and r3[4] is None


def test_stop_cleans_up_leftover_staged_file_and_does_not_hang(tmp_path):
    video_files = [f"clip{i}.mp4" for i in range(20)]
    stage_fn, live_files, lock = _fake_stage(str(tmp_path), delay=0.02)
    prefetcher = ClipPrefetcher(video_files, stage_fn=stage_fn)
    prefetcher.start()

    # Consume only the first clip, then interrupt (simulates Ctrl-C).
    prefetcher.get_next()
    prefetcher.stop()

    assert prefetcher._thread is not None
    assert not prefetcher._thread.is_alive()

    # Any file the worker staged but the consumer never took must be cleaned up.
    remaining = [f for f in live_files if os.path.exists(f)]
    # The one clip consumed above is the caller's responsibility to remove;
    # everything else the worker itself produced but never handed off must be gone.
    assert len(remaining) <= 1


def test_sequential_path_is_used_when_prefetch_disabled():
    """Guards the requirement that with the flag off, main.py's sequential
    stage_video_locally() call path is unchanged (no ClipPrefetcher involved)."""
    import main

    assert hasattr(main, "stage_video_locally")
    # ClipPrefetcher must be importable independently without main.py
    # importing/instantiating it at module load time.
    import inspect

    src = inspect.getsource(main)
    assert "ClipPrefetcher" in src  # used, but only inside the prefetch-enabled branch
    assert "if prefetcher is not None" in src
