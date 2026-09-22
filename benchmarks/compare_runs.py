"""
Event-by-event equivalence checker for two ALPR pipeline SQLite databases.

Used to prove that an isolated performance change (e.g. clip prefetching)
does not alter pipeline output: same events, same fields, same order —
not just matching aggregate counts.

Usage:
  python benchmarks/compare_runs.py DB_A.sqlite DB_B.sqlite [--run-a RUN_ID] [--run-b RUN_ID]

If --run-a / --run-b are omitted, the most recent COMPLETED run in each
database is used.
"""
import argparse
import sqlite3
import sys
from typing import Any, Dict, List, Optional

# Fields that define an event's identity/content for equivalence purposes.
# Deliberately excludes fields expected to differ between runs by design
# (event_id embeds a run-specific prefix in this pipeline; processing_run_id
# always differs between two separate runs; crop paths embed the run id too).
COMPARE_FIELDS = [
    "video_source",
    "event_timestamp",
    "direction",
    "vehicle_type",
    "plate_raw",
    "plate_normalized",
    "plate_corrected",
    "plate_status",
    "confidence_ocr",
    "duplicate_of_is_null",  # normalized boolean, see _load_events
]

FLOAT_TOLERANCE = 1e-6


def _latest_completed_run(conn: sqlite3.Connection) -> Optional[str]:
    cur = conn.cursor()
    cur.execute(
        "SELECT run_id FROM processing_runs WHERE status = 'COMPLETED' "
        "ORDER BY started_at DESC LIMIT 1"
    )
    row = cur.fetchone()
    return row[0] if row else None


def _load_events(db_path: str, run_id: Optional[str]) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    resolved_run_id = run_id or _latest_completed_run(conn)
    if resolved_run_id is None:
        conn.close()
        raise SystemExit(f"No COMPLETED run found in {db_path}")

    cur.execute(
        "SELECT * FROM events WHERE processing_run_id = ? ORDER BY video_source, event_timestamp",
        (resolved_run_id,),
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()

    for row in rows:
        # duplicate_of references another run's event_id, which is expected
        # to differ across separate runs even for equivalent output; what
        # matters for equivalence is whether the event WAS a duplicate.
        row["duplicate_of_is_null"] = row.get("duplicate_of") is None

    return rows, resolved_run_id


def _fields_differ(a: Dict[str, Any], b: Dict[str, Any]) -> List[str]:
    diffs = []
    for field in COMPARE_FIELDS:
        va, vb = a.get(field), b.get(field)
        if isinstance(va, float) or isinstance(vb, float):
            va_f = float(va) if va is not None else None
            vb_f = float(vb) if vb is not None else None
            if va_f is None or vb_f is None:
                if va_f != vb_f:
                    diffs.append(field)
            elif abs(va_f - vb_f) > FLOAT_TOLERANCE:
                diffs.append(field)
        elif va != vb:
            diffs.append(field)
    return diffs


def compare(db_a: str, db_b: str, run_a: Optional[str], run_b: Optional[str]) -> int:
    events_a, resolved_run_a = _load_events(db_a, run_a)
    events_b, resolved_run_b = _load_events(db_b, run_b)

    print(f"DB A: {db_a} (run {resolved_run_a}) -> {len(events_a)} events")
    print(f"DB B: {db_b} (run {resolved_run_b}) -> {len(events_b)} events")

    mismatches = []

    if len(events_a) != len(events_b):
        mismatches.append(
            f"EVENT COUNT MISMATCH: {len(events_a)} (A) vs {len(events_b)} (B)"
        )

    for idx, (ea, eb) in enumerate(zip(events_a, events_b)):
        diffs = _fields_differ(ea, eb)
        if diffs:
            mismatches.append(
                f"Event #{idx} ({ea.get('video_source')} @ {ea.get('event_timestamp')}): "
                f"fields differ {diffs} | A={ {k: ea.get(k) for k in diffs} } "
                f"B={ {k: eb.get(k) for k in diffs} }"
            )

    print("\n" + "=" * 60)
    print("PER-EVENT COMPARISON")
    print("=" * 60)
    if not mismatches:
        print("No differences found. Runs are event-for-event equivalent.")
    else:
        for m in mismatches:
            print(f"- {m}")
        print(f"\n{len(mismatches)} difference(s) found.")
    print("=" * 60 + "\n")

    return 1 if mismatches else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("db_a", help="Path to first SQLite database")
    parser.add_argument("db_b", help="Path to second SQLite database")
    parser.add_argument("--run-a", default=None, help="run_id to use in DB A (default: latest COMPLETED)")
    parser.add_argument("--run-b", default=None, help="run_id to use in DB B (default: latest COMPLETED)")
    args = parser.parse_args()

    exit_code = compare(args.db_a, args.db_b, args.run_a, args.run_b)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
