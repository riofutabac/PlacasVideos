"""
Single-worker prefetch helper that overlaps staging (Drive -> local SSD copy)
of the NEXT clip with processing of the CURRENT one.

Opt-in only: callers decide whether to use this at all (see main.py's
`--prefetch` flag / `video.prefetch_next_clip` config). When unused, the
sequential `stage_video_locally()` call path in main.py is untouched.

Guarantees:
- Exactly one background worker thread stages clips strictly in the given
  order (a bounded queue with maxsize=1 means the worker can be at most one
  clip ahead of the consumer).
- At most two clips exist on local storage at once: the one the consumer is
  currently processing (already dequeued) plus the one the worker is
  staging or has already staged into the queue.
- A staging failure for one clip (corrupt/unreadable file, I/O error, etc.)
  is delivered to the consumer as an error alongside that clip instead of
  crashing the worker thread, so the existing "skip and continue" behavior
  in main.py keeps working unchanged.
- `stop()` unblocks the worker (even if it is parked on a full queue) and
  joins it, cleaning up any staged-but-unconsumed temp file so Ctrl-C does
  not leave stray files or a hung thread behind.
"""
import os
import queue
import threading
from typing import Callable, List, Optional, Tuple

StageFn = Callable[[str], Tuple[str, bool, float]]
StageResult = Tuple[str, str, bool, float, Optional[Exception]]

_SENTINEL = object()


class ClipPrefetcher:
    """Stages clips one-at-a-time, one clip ahead of the consumer."""

    def __init__(self, video_files: List[str], stage_fn: StageFn):
        self._video_files = list(video_files)
        self._stage_fn = stage_fn
        self._queue: "queue.Queue" = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="clip-prefetcher", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        for orig_vf in self._video_files:
            if self._stop_event.is_set():
                return
            try:
                proc_vf, is_temp, staging_seconds = self._stage_fn(orig_vf)
                result: StageResult = (orig_vf, proc_vf, is_temp, staging_seconds, None)
            except Exception as exc:  # staging failure must not kill the worker
                result = (orig_vf, orig_vf, False, 0.0, exc)

            if self._stop_event.is_set():
                self._cleanup_result(result)
                return

            self._queue.put(result)

        self._queue.put(_SENTINEL)

    @staticmethod
    def _cleanup_result(result: StageResult) -> None:
        _, proc_vf, is_temp, _, _ = result
        if is_temp and proc_vf and os.path.exists(proc_vf):
            try:
                os.remove(proc_vf)
            except OSError:
                pass

    def get_next(self) -> StageResult:
        """Blocks until the next staged clip (or its staging error) is ready.

        Returns (original_path, processed_path, is_temporary, staging_seconds, error).
        Raises StopIteration if the worker has exhausted the clip list.
        """
        item = self._queue.get()
        if item is _SENTINEL:
            raise StopIteration
        return item

    def stop(self) -> None:
        """Signals the worker to stop, unblocks it, and joins it.

        Cleans up any staged-but-unconsumed temp file so an interruption
        (e.g. Ctrl-C) never leaves stray SSD temp files or a hung thread.
        """
        self._stop_event.set()
        try:
            leftover = self._queue.get_nowait()
            if leftover is not _SENTINEL:
                self._cleanup_result(leftover)
        except queue.Empty:
            pass
        if self._thread is not None:
            self._thread.join(timeout=10)
