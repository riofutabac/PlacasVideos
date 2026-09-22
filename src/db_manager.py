"""
Canonical SQLite Database Manager for ALPR Pipeline.
Handles:
- processing_runs (experiments, benchmark tracking, hashes)
- processed_clips (checkpoints and resume capability)
- events (canonical vehicle & plate event records)
"""
import os
import sqlite3
import json
from typing import Optional, Dict, Any, List
from dataclasses import asdict

class DatabaseManager:
    def __init__(self, db_path: str = "data/events.sqlite"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # 1. Processing Runs
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS processing_runs (
                run_id TEXT PRIMARY KEY,
                pipeline_version TEXT NOT NULL,
                config_hash TEXT NOT NULL,
                model_versions TEXT NOT NULL,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                total_videos INTEGER DEFAULT 0,
                total_events INTEGER DEFAULT 0,
                speed_ratio REAL DEFAULT 0.0,
                status TEXT NOT NULL
            );
            """)

            # Automatic migration for existing databases (batch-level timing, additive/nullable)
            cursor.execute("PRAGMA table_info(processing_runs);")
            existing_run_cols = {row[1] for row in cursor.fetchall()}
            run_migrations = [
                ("startup_seconds", "REAL DEFAULT 0.0"),
                ("export_seconds", "REAL DEFAULT 0.0"),
            ]
            for col_name, col_type in run_migrations:
                if col_name not in existing_run_cols:
                    cursor.execute(f"ALTER TABLE processing_runs ADD COLUMN {col_name} {col_type};")

            # 2. Processed Clips (Checkpoints / Resume / Per-clip Telemetry)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS processed_clips (
                clip_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                file_path TEXT NOT NULL,
                file_hash TEXT NOT NULL,
                duration_sec REAL NOT NULL,
                total_frames INTEGER NOT NULL,
                processed_events INTEGER DEFAULT 0,
                status TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                wall_clock_seconds REAL DEFAULT 0.0,
                speed_ratio REAL DEFAULT 0.0,
                decode_seconds REAL DEFAULT 0.0,
                vehicle_detection_seconds REAL DEFAULT 0.0,
                plate_detection_seconds REAL DEFAULT 0.0,
                ocr_seconds REAL DEFAULT 0.0,
                FOREIGN KEY (run_id) REFERENCES processing_runs(run_id)
            );
            """)

            # Automatic migration for existing databases
            cursor.execute("PRAGMA table_info(processed_clips);")
            existing_cols = {row[1] for row in cursor.fetchall()}
            migrations = [
                ("started_at", "TEXT"),
                ("wall_clock_seconds", "REAL DEFAULT 0.0"),
                ("speed_ratio", "REAL DEFAULT 0.0"),
                ("decode_seconds", "REAL DEFAULT 0.0"),
                ("vehicle_detection_seconds", "REAL DEFAULT 0.0"),
                ("plate_detection_seconds", "REAL DEFAULT 0.0"),
                ("ocr_seconds", "REAL DEFAULT 0.0"),
                # Instrumentation-only additions: per-clip gap breakdown (staging, hashing, OSD clock read)
                ("staging_seconds", "REAL DEFAULT 0.0"),
                ("file_hash_seconds", "REAL DEFAULT 0.0"),
                ("clock_read_seconds", "REAL DEFAULT 0.0"),
            ]
            for col_name, col_type in migrations:
                if col_name not in existing_cols:
                    cursor.execute(f"ALTER TABLE processed_clips ADD COLUMN {col_name} {col_type};")

            # 3. Canonical Events Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                processing_run_id TEXT NOT NULL,
                video_source TEXT NOT NULL,
                clip_hash TEXT NOT NULL,
                event_timestamp REAL NOT NULL,
                datetime_str TEXT NOT NULL,
                best_vehicle_timestamp REAL,
                best_plate_timestamp REAL,
                direction TEXT NOT NULL,
                vehicle_type TEXT NOT NULL,
                plate_raw TEXT,
                plate_normalized TEXT,
                plate_corrected TEXT,
                plate_correction_reason TEXT,
                plate_status TEXT NOT NULL,
                confidence_vehicle REAL NOT NULL,
                confidence_plate REAL DEFAULT 0.0,
                confidence_ocr REAL DEFAULT 0.0,
                confidence_consensus REAL DEFAULT 0.0,
                vehicle_crop_path TEXT NOT NULL,
                plate_crop_path TEXT,
                ocr_votes TEXT,
                track_id INTEGER NOT NULL,
                line_id TEXT NOT NULL,
                dedup_key TEXT,
                duplicate_of TEXT,
                FOREIGN KEY (processing_run_id) REFERENCES processing_runs(run_id)
            );
            """)
            
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(event_timestamp);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_plate ON events(plate_normalized);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_run ON events(processing_run_id);")
            conn.commit()

    def start_run(self, run_id: str, pipeline_version: str, config_hash: str, model_versions: Dict, started_at: str):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT OR REPLACE INTO processing_runs (
                run_id, pipeline_version, config_hash, model_versions, started_at, status
            ) VALUES (?, ?, ?, ?, ?, 'RUNNING')
            """, (run_id, pipeline_version, config_hash, json.dumps(model_versions), started_at))
            conn.commit()

    def finish_run(
        self,
        run_id: str,
        finished_at: str,
        total_videos: int,
        total_events: int,
        speed_ratio: float,
        status: str = "COMPLETED",
        startup_seconds: float = 0.0,
        export_seconds: float = 0.0
    ):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            UPDATE processing_runs
            SET finished_at = ?, total_videos = ?, total_events = ?, speed_ratio = ?, status = ?,
                startup_seconds = ?, export_seconds = ?
            WHERE run_id = ?
            """, (finished_at, total_videos, total_events, speed_ratio, status, startup_seconds, export_seconds, run_id))
            conn.commit()

    def is_clip_completed(self, clip_id: str, file_hash: str) -> bool:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT status FROM processed_clips
            WHERE clip_id = ? AND file_hash = ? AND status = 'COMPLETED'
            """, (clip_id, file_hash))
            return cursor.fetchone() is not None

    def record_clip(
        self,
        clip_id: str,
        run_id: str,
        file_path: str,
        file_hash: str,
        duration_sec: float,
        total_frames: int,
        events_count: int,
        status: str,
        started_at: str,
        completed_at: str,
        wall_clock_seconds: float = 0.0,
        speed_ratio: float = 0.0,
        decode_seconds: float = 0.0,
        vehicle_detection_seconds: float = 0.0,
        plate_detection_seconds: float = 0.0,
        ocr_seconds: float = 0.0,
        staging_seconds: float = 0.0,
        file_hash_seconds: float = 0.0,
        clock_read_seconds: float = 0.0
    ):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT OR REPLACE INTO processed_clips (
                clip_id, run_id, file_path, file_hash, duration_sec, total_frames,
                processed_events, status, started_at, completed_at,
                wall_clock_seconds, speed_ratio,
                decode_seconds, vehicle_detection_seconds, plate_detection_seconds, ocr_seconds,
                staging_seconds, file_hash_seconds, clock_read_seconds
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                clip_id, run_id, file_path, file_hash, duration_sec, total_frames,
                events_count, status, started_at, completed_at,
                wall_clock_seconds, speed_ratio,
                decode_seconds, vehicle_detection_seconds, plate_detection_seconds, ocr_seconds,
                staging_seconds, file_hash_seconds, clock_read_seconds
            ))
            conn.commit()

    def get_clip_timing_totals(self, run_id: str) -> Dict[str, float]:
        """Aggregates per-clip instrumentation timers for a run (used for the batch gap summary)."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT
                COALESCE(SUM(staging_seconds), 0.0),
                COALESCE(SUM(file_hash_seconds), 0.0),
                COALESCE(SUM(clock_read_seconds), 0.0),
                COALESCE(SUM(wall_clock_seconds), 0.0)
            FROM processed_clips WHERE run_id = ?
            """, (run_id,))
            row = cursor.fetchone()
            return {
                'staging_seconds': row[0] if row else 0.0,
                'file_hash_seconds': row[1] if row else 0.0,
                'clock_read_seconds': row[2] if row else 0.0,
                'wall_clock_seconds': row[3] if row else 0.0,
            }

    def insert_event(self, event: Dict[str, Any]):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT OR REPLACE INTO events (
                event_id, processing_run_id, video_source, clip_hash,
                event_timestamp, datetime_str, best_vehicle_timestamp, best_plate_timestamp,
                direction, vehicle_type, plate_raw, plate_normalized, plate_corrected, plate_correction_reason,
                plate_status, confidence_vehicle, confidence_plate, confidence_ocr, confidence_consensus,
                vehicle_crop_path, plate_crop_path, ocr_votes, track_id, line_id, dedup_key, duplicate_of
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event['event_id'], event['processing_run_id'], event['video_source'], event['clip_hash'],
                event['event_timestamp'], event['datetime_str'], event.get('best_vehicle_timestamp'), event.get('best_plate_timestamp'),
                event['direction'], event['vehicle_type'], event.get('plate_raw'), event.get('plate_normalized'),
                event.get('plate_corrected'), event.get('plate_correction_reason'), event['plate_status'],
                event['confidence_vehicle'], event.get('confidence_plate', 0.0), event.get('confidence_ocr', 0.0),
                event.get('confidence_consensus', 0.0), event['vehicle_crop_path'], event.get('plate_crop_path'),
                json.dumps(event.get('ocr_votes', [])), event['track_id'], event['line_id'],
                event.get('dedup_key'), event.get('duplicate_of')
            ))
            conn.commit()

    def get_events_for_run(self, run_id: Optional[str] = None, exclude_duplicates: bool = True) -> List[Dict]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            query = "SELECT * FROM events"
            params = []
            conditions = []
            if run_id:
                conditions.append("processing_run_id = ?")
                params.append(run_id)
            if exclude_duplicates:
                conditions.append("duplicate_of IS NULL")
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            query += " ORDER BY event_timestamp ASC"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
