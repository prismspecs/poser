import pytest
import numpy as np
from pathlib import Path
import tempfile
from pose_db import PoseDatabase


def test_pose_database_init():
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test_poses.db"
        db = PoseDatabase(str(db_path))
        stats = db.get_stats()
        
        assert stats["total_films"] == 0
        assert stats["total_poses"] == 0
        assert db_path.exists()


def test_pose_database_ingest_and_query():
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test_poses.db"
        db = PoseDatabase(str(db_path))

        film_id = db.register_film("Test Movie 1", "/path/to/test.mp4", total_frames=100, fps=24.0, duration=4.16)
        assert film_id == 1

        dummy_kps = [
            (100.0, 100.0, 0.9), (105.0, 95.0, 0.9), (95.0, 95.0, 0.9),
            (110.0, 100.0, 0.9), (90.0, 100.0, 0.9), (120.0, 130.0, 0.9),
            (80.0, 130.0, 0.9), (130.0, 160.0, 0.9), (70.0, 160.0, 0.9),
            (140.0, 190.0, 0.9), (60.0, 190.0, 0.9), (115.0, 200.0, 0.9),
            (85.0, 200.0, 0.9), (120.0, 250.0, 0.9), (80.0, 250.0, 0.9),
            (125.0, 300.0, 0.9), (75.0, 300.0, 0.9)
        ]

        pose_records = [
            {
                "frame_idx": 0,
                "timestamp": 0.0,
                "bbox": (50.0, 50.0, 150.0, 310.0),
                "confidence": 0.92,
                "keypoints": dummy_kps
            }
        ]

        db.add_poses_batch(film_id, pose_records)
        stats = db.get_stats()
        assert stats["total_films"] == 1
        assert stats["total_poses"] == 1

        poses = db.get_all_poses()
        assert len(poses) == 1
        assert poses[0]["film_title"] == "Test Movie 1"
        assert len(poses[0]["vector"]) == 34
