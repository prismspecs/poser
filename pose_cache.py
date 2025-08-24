#!/usr/bin/env python3
"""
Pose caching system to avoid re-analyzing images.
Stores pose data in a JSON database with image hash-based keys.
"""

import json
import hashlib
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from utils.pose_utils import PoseData
import cv2
import numpy as np


class PoseCache:
    """Cache for storing and retrieving pose analysis results."""

    def __init__(self, cache_file: str = "pose_cache.json"):
        self.cache_file = Path(cache_file)
        self.cache: Dict[str, Any] = {}
        self.load_cache()

    def _calculate_image_hash(self, image_path: str) -> str:
        """Calculate a hash of the image content for unique identification."""
        try:
            # Read image and calculate hash
            img = cv2.imread(image_path)
            if img is None:
                return hashlib.md5(image_path.encode()).hexdigest()

            # Convert to grayscale and resize for consistent hashing
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (64, 64))

            # Calculate hash
            return hashlib.md5(resized.tobytes()).hexdigest()
        except Exception:
            # Fallback to filename hash if image processing fails
            return hashlib.md5(image_path.encode()).hexdigest()

    def _get_cache_key(self, image_path: str) -> str:
        """Generate a cache key for an image."""
        # Use both filename and content hash for uniqueness
        # Convert to absolute path to ensure consistency
        abs_path = str(Path(image_path).resolve())
        filename = Path(image_path).name
        content_hash = self._calculate_image_hash(abs_path)
        return f"{filename}_{content_hash}"

    def load_cache(self):
        """Load the cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r") as f:
                    self.cache = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load pose cache: {e}")
                self.cache = {}
        else:
            self.cache = {}

    def save_cache(self):
        """Save the cache to disk."""
        try:
            with open(self.cache_file, "w") as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save pose cache: {e}")

    def get_cached_poses(self, image_path: str) -> Optional[List[PoseData]]:
        """Retrieve cached poses for an image."""
        cache_key = self._get_cache_key(image_path)
        if cache_key in self.cache:
            cached_data = self.cache[cache_key]

            # Reconstruct PoseData objects from cached data
            poses = []
            for pose_dict in cached_data.get("poses", []):
                pose = PoseData(
                    keypoints=pose_dict["keypoints"],
                    confidence_score=pose_dict["confidence_score"],
                    bounding_box=pose_dict["bounding_box"],
                    pose_id=pose_dict["pose_id"],
                    image_path=pose_dict["image_path"],
                )
                poses.append(pose)

            return poses
        return None

    def cache_poses(self, image_path: str, poses: List[PoseData]):
        """Cache poses for an image."""
        cache_key = self._get_cache_key(image_path)

        # Convert PoseData objects to serializable format
        poses_data = []
        for pose in poses:
            pose_dict = {
                "keypoints": pose.keypoints,
                "confidence_score": pose.confidence_score,
                "bounding_box": pose.bounding_box,
                "pose_id": pose.pose_id,
                "image_path": pose.image_path,
            }
            poses_data.append(pose_dict)

        # Store in cache
        self.cache[cache_key] = {
            "image_path": image_path,
            "poses": poses_data,
            "timestamp": (
                str(Path(image_path).stat().st_mtime)
                if Path(image_path).exists()
                else "unknown"
            ),
        }

        # Save cache periodically (every 10 operations)
        if len(self.cache) % 10 == 0:
            self.save_cache()

    def clear_cache(self):
        """Clear the entire cache."""
        self.cache = {}
        if self.cache_file.exists():
            self.cache_file.unlink()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the cache."""
        return {
            "total_cached_images": len(self.cache),
            "cache_file_size": (
                self.cache_file.stat().st_size if self.cache_file.exists() else 0
            ),
            "cache_file_path": str(self.cache_file.absolute()),
        }

    def is_cached(self, image_path: str) -> bool:
        """Check if an image is cached."""
        cache_key = self._get_cache_key(image_path)
        return cache_key in self.cache

    def update_cache_timestamp(self, image_path: str):
        """Update the timestamp for a cached image."""
        cache_key = self._get_cache_key(image_path)
        if cache_key in self.cache:
            self.cache[cache_key]["timestamp"] = str(Path(image_path).stat().st_mtime)
            self.save_cache()
