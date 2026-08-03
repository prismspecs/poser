#!/usr/bin/env python3
"""
Pose Database Module (SQLite Binary Vector Store)
Provides high-performance, compact storage and vector retrieval for millions of video pose keypoints.
"""

import sqlite3
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from utils.pose_utils import PoseData


class PoseDatabase:
    """SQLite-backed binary vector database for storing and querying video poses."""

    def __init__(self, db_path: str = "pose_library.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Initialize database schema with indexed binary pose BLOB tables."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Table for source films / video media
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS films (
                    film_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT UNIQUE NOT NULL,
                    filepath TEXT NOT NULL,
                    total_frames INTEGER DEFAULT 0,
                    fps REAL DEFAULT 24.0,
                    duration REAL DEFAULT 0.0
                )
            """)

            # Table for frame pose keypoints
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS poses (
                    pose_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    film_id INTEGER NOT NULL,
                    frame_idx INTEGER NOT NULL,
                    timestamp REAL NOT NULL,
                    bbox_x1 REAL, bbox_y1 REAL, bbox_x2 REAL, bbox_y2 REAL,
                    confidence REAL NOT NULL,
                    normalized_vector BLOB NOT NULL,
                    FOREIGN KEY(film_id) REFERENCES films(film_id) ON DELETE CASCADE
                )
            """)

            # Indexes for fast retrieval and spatial lookup
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_film_id ON poses(film_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_film_frame ON poses(film_id, frame_idx)")
            conn.commit()

    def register_film(self, title: str, filepath: str, total_frames: int = 0, fps: float = 24.0, duration: float = 0.0) -> int:
        """Register a source film in the database and return its film_id."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT film_id FROM films WHERE title = ?", (title,))
            row = cursor.fetchone()
            if row:
                return row["film_id"]

            cursor.execute("""
                INSERT INTO films (title, filepath, total_frames, fps, duration)
                VALUES (?, ?, ?, ?, ?)
            """, (title, str(filepath), total_frames, fps, duration))
            conn.commit()
            return cursor.lastrowid

    def add_poses_batch(self, film_id: int, pose_records: List[Dict[str, Any]]):
        """
        Batch insert poses for a given film.
        
        Each record in pose_records must have:
        - frame_idx: int
        - timestamp: float
        - bbox: Tuple[float, float, float, float]
        - confidence: float
        - keypoints: List[Optional[Tuple[float, float, float]]] (17 COCO keypoints)
        """
        if not pose_records:
            return

        db_rows = []
        for rec in pose_records:
            keypoints = rec["keypoints"]
            norm_vector = self.normalize_keypoints_to_vector(keypoints)
            if norm_vector is None:
                continue

            blob = norm_vector.astype(np.float16).tobytes()
            bbox = rec.get("bbox", (0.0, 0.0, 0.0, 0.0))

            db_rows.append((
                film_id,
                rec["frame_idx"],
                rec["timestamp"],
                bbox[0], bbox[1], bbox[2], bbox[3],
                rec["confidence"],
                blob
            ))

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany("""
                INSERT INTO poses (film_id, frame_idx, timestamp, bbox_x1, bbox_y1, bbox_x2, bbox_y2, confidence, normalized_vector)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, db_rows)
            conn.commit()

    @staticmethod
    def normalize_keypoints_to_vector(keypoints: List[Optional[Tuple[float, float, float]]]) -> Optional[np.ndarray]:
        """
        Normalize 17 COCO keypoints into a scale- and translation-invariant 34D float32 vector (x, y coords).
        """
        if not keypoints or len(keypoints) != 17:
            return None

        # COCO indices for shoulders and hips
        ls_idx, rs_idx, lh_idx, rh_idx = 5, 6, 11, 12
        critical = [keypoints[ls_idx], keypoints[rs_idx], keypoints[lh_idx], keypoints[rh_idx]]
        if any(kp is None for kp in critical):
            return None

        left_shoulder = np.array(critical[0][:2])
        right_shoulder = np.array(critical[1][:2])
        left_hip = np.array(critical[2][:2])
        right_hip = np.array(critical[3][:2])

        torso_center = (left_shoulder + right_shoulder + left_hip + right_hip) / 4.0
        shoulder_width = np.linalg.norm(right_shoulder - left_shoulder)
        torso_height = np.linalg.norm(((left_shoulder + right_shoulder) / 2.0) - ((left_hip + right_hip) / 2.0))
        torso_scale = max(shoulder_width, torso_height)

        if torso_scale < 1e-6:
            return None

        vector = np.zeros(34, dtype=np.float32)
        for i, kp in enumerate(keypoints):
            if kp is not None:
                vector[2 * i] = (kp[0] - torso_center[0]) / torso_scale
                vector[2 * i + 1] = (kp[1] - torso_center[1]) / torso_scale
            else:
                vector[2 * i] = 0.0
                vector[2 * i + 1] = 0.0

        return vector

    def get_all_poses(self) -> List[Dict[str, Any]]:
        """Fetch all poses from database with film details and unpacked 34D vectors."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT p.pose_id, p.film_id, f.title as film_title, f.filepath as film_path,
                       p.frame_idx, p.timestamp, p.bbox_x1, p.bbox_y1, p.bbox_x2, p.bbox_y2,
                       p.confidence, p.normalized_vector
                FROM poses p
                JOIN films f ON p.film_id = f.film_id
            """)
            
            results = []
            for row in cursor.fetchall():
                vector = np.frombuffer(row["normalized_vector"], dtype=np.float16).astype(np.float32)
                results.append({
                    "pose_id": row["pose_id"],
                    "film_id": row["film_id"],
                    "film_title": row["film_title"],
                    "film_path": row["film_path"],
                    "frame_idx": row["frame_idx"],
                    "timestamp": row["timestamp"],
                    "bbox": (row["bbox_x1"], row["bbox_y1"], row["bbox_x2"], row["bbox_y2"]),
                    "confidence": row["confidence"],
                    "vector": vector
                })
            return results

    def get_local_window_poses(self, film_id: int, center_frame_idx: int, window_sec: float = 2.0, fps: float = 24.0) -> List[Dict[str, Any]]:
        """Retrieve poses within a local +/- window_sec timeframe for motion refinement."""
        frame_window = int(window_sec * fps)
        min_frame = max(0, center_frame_idx - frame_window)
        max_frame = center_frame_idx + frame_window

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT p.pose_id, p.film_id, f.title as film_title, f.filepath as film_path,
                       p.frame_idx, p.timestamp, p.bbox_x1, p.bbox_y1, p.bbox_x2, p.bbox_y2,
                       p.confidence, p.normalized_vector
                FROM poses p
                JOIN films f ON p.film_id = f.film_id
                WHERE p.film_id = ? AND p.frame_idx BETWEEN ? AND ?
                ORDER BY p.frame_idx ASC
            """, (film_id, min_frame, max_frame))

            results = []
            for row in cursor.fetchall():
                vector = np.frombuffer(row["normalized_vector"], dtype=np.float16).astype(np.float32)
                results.append({
                    "pose_id": row["pose_id"],
                    "film_id": row["film_id"],
                    "film_title": row["film_title"],
                    "film_path": row["film_path"],
                    "frame_idx": row["frame_idx"],
                    "timestamp": row["timestamp"],
                    "bbox": (row["bbox_x1"], row["bbox_y1"], row["bbox_x2"], row["bbox_y2"]),
                    "confidence": row["confidence"],
                    "vector": vector
                })
            return results

    def get_stats(self) -> Dict[str, Any]:
        """Get summary statistics for the database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM films")
            total_films = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM poses")
            total_poses = cursor.fetchone()[0]
            
            db_size_bytes = self.db_path.stat().st_size if self.db_path.exists() else 0
            
            return {
                "db_path": str(self.db_path.resolve()),
                "total_films": total_films,
                "total_poses": total_poses,
                "db_size_bytes": db_size_bytes,
                "db_size_mb": round(db_size_bytes / (1024 * 1024), 2)
            }
