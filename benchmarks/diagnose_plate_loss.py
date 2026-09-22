"""
Diagnostic helper (read-only, no fixes applied): given a clip and a wall-clock
time window, prints the plate candidates and OCR texts/confidences for events
recorded in that window, from a given SQLite events DB.

Intended to make plate-read regressions observable, e.g.:
  python benchmarks/diagnose_plate_loss.py --db data/events_137.sqlite --clip 14 --start "12:05:00" --end "12:08:00"
  python benchmarks/diagnose_plate_loss.py --db data/events_133.sqlite --clip 17 --start "12:20:00" --end "12:24:00"

This tool does NOT change any threshold or pipeline behavior; it only reads events.
"""
import argparse
import json
import re
import sqlite3
from typing import Optional


def _clip_matches(video_source: str, clip_arg: str) -> bool:
    """Matches a clip filter against video_source, accepting either a bare clip number
    (e.g. '14', matched against '(14).mp4') or a raw substring."""
    if clip_arg.isdigit():
        return re.search(rf'\({clip_arg}\)\.[a-zA-Z0-9]+$', video_source) is not None
    return clip_arg in video_source


def _time_in_window(datetime_str: str, start: Optional[str], end: Optional[str]) -> bool:
    """Compares only the time-of-day portion (HH:MM:SS) of datetime_str against the window,
    so callers don't need to know the exact date."""
    if not start and not end:
        return True
    time_part = datetime_str.split(" ")[-1] if " " in datetime_str else datetime_str
    if start and time_part < start:
        return False
    if end and time_part > end:
        return False
    return True


def diagnose(db_path: str, clip_arg: Optional[str], start: Optional[str], end: Optional[str], run_id: Optional[str] = None) -> None:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    query = "SELECT * FROM events"
    params = []
    if run_id:
        query += " WHERE processing_run_id = ?"
        params.append(run_id)
    query += " ORDER BY event_timestamp ASC"
    cursor.execute(query, params)
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()

    matches = []
    for row in rows:
        if clip_arg and not _clip_matches(row['video_source'], clip_arg):
            continue
        if not _time_in_window(row['datetime_str'], start, end):
            continue
        matches.append(row)

    print(f"DB: {db_path}")
    print(f"Filter: clip={clip_arg!r} window=[{start or '-'}, {end or '-'}] run_id={run_id or '-'}")
    print(f"Matched {len(matches)} event(s).\n")

    for row in matches:
        print("-" * 70)
        print(f"event_id={row['event_id']}  clip={row['video_source']}  datetime={row['datetime_str']}")
        print(f"  direction={row['direction']}  vehicle_type={row['vehicle_type']}  track_id={row['track_id']}")
        print(f"  plate_raw={row['plate_raw']}  plate_normalized={row['plate_normalized']}  "
              f"plate_corrected={row['plate_corrected']}  status={row['plate_status']}")
        print(f"  confidence_plate={row['confidence_plate']}  confidence_ocr={row['confidence_ocr']}  "
              f"confidence_consensus={row['confidence_consensus']}")
        print(f"  duplicate_of={row['duplicate_of']}  dedup_key={row['dedup_key']}")
        try:
            votes = json.loads(row['ocr_votes']) if row['ocr_votes'] else []
        except (json.JSONDecodeError, TypeError):
            votes = row['ocr_votes']
        print(f"  ocr_votes (candidate text, confidence): {votes}")
        print(f"  plate_crop_path={row['plate_crop_path']}")

    if not matches:
        print("No events found in that clip/window. Check --clip and --start/--end, "
              "or that the plate never reached the crossing FSM (see the rejection log "
              "from a --diagnose run for discarded detections/tracks/plate reads).")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnose plate-read regressions by inspecting recorded events in a clip/time window (read-only)."
    )
    parser.add_argument("--db", required=True, help="Path to the SQLite events DB (e.g. data/events_137.sqlite)")
    parser.add_argument("--clip", default=None, help="Clip number (e.g. 14) or substring of video_source")
    parser.add_argument("--start", default=None, help="Window start, HH:MM:SS (time-of-day only)")
    parser.add_argument("--end", default=None, help="Window end, HH:MM:SS (time-of-day only)")
    parser.add_argument("--run-id", default=None, help="Optional processing_run_id filter")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    diagnose(args.db, args.clip, args.start, args.end, args.run_id)
