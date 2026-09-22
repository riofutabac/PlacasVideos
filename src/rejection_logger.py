"""
Rejection Logger for ALPR Pipeline diagnostics.
Off by default (near-zero overhead): records why a detection/track/plate-read was
discarded, so pipeline behavior can be investigated without changing thresholds.
Controlled by config `diagnostics.log_rejections` or the `--diagnose` CLI flag.
Writes one JSON object per line to reports/diagnostics/<run_id>.jsonl.
"""
import os
import json
import time
from typing import Any, Optional


def _json_default(obj: Any) -> Any:
    """Best-effort fallback serializer for numpy scalars and other non-JSON types."""
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    return str(obj)


class RejectionLogger:
    """Appends discarded-detection records to a per-run JSONL file when enabled."""

    def __init__(self, enabled: bool = False, output_dir: str = "reports/diagnostics"):
        self.enabled = bool(enabled)
        self.output_dir = output_dir
        self.path: Optional[str] = None
        self._fh = None

    def open(self, run_id: str) -> None:
        """Lazily opens the JSONL file for this run. No-op if disabled or already open."""
        if not self.enabled or self._fh is not None:
            return
        os.makedirs(self.output_dir, exist_ok=True)
        self.path = os.path.join(self.output_dir, f"{run_id}.jsonl")
        self._fh = open(self.path, "a", encoding="utf-8")

    def log(
        self,
        clip_id: str,
        timestamp: float,
        reason: str,
        bbox: Optional[Any] = None,
        vehicle_class: Optional[str] = None,
        confidence: Optional[float] = None,
        **extra: Any
    ) -> None:
        """Records one discarded detection/track/plate-read. No-op unless enabled and opened."""
        if not self.enabled or self._fh is None:
            return
        record = {
            "wall_time": time.time(),
            "clip_id": clip_id,
            "timestamp": timestamp,
            "reason": reason,
            "bbox": bbox,
            "vehicle_class": vehicle_class,
            "confidence": confidence,
        }
        record.update(extra)
        self._fh.write(json.dumps(record, default=_json_default) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
