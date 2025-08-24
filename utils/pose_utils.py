"""
Pose utility functions and data structures.
Handles pose data representation and utility functions.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class PoseData:
    """Data structure for pose information."""

    keypoints: List[Optional[Tuple[float, float, float]]]  # x, y, confidence
    bounding_box: Tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence_score: float
    image_path: str
    pose_id: str

    def __post_init__(self):
        """Validate pose data after initialization."""
        if len(self.keypoints) != 17:
            raise ValueError(f"Expected 17 keypoints, got {len(self.keypoints)}")

        if len(self.bounding_box) != 4:
            raise ValueError(
                f"Expected 4 bounding box coordinates, got {len(self.bounding_box)}"
            )

        if not (0.0 <= self.confidence_score <= 1.0):
            raise ValueError(
                f"Confidence score must be between 0 and 1, got {self.confidence_score}"
            )

    def get_valid_keypoints(self) -> List[Tuple[int, Tuple[float, float, float]]]:
        """Get list of valid keypoints with their indices."""
        return [(i, kp) for i, kp in enumerate(self.keypoints) if kp is not None]

    def get_keypoint_coordinates(self) -> List[Tuple[float, float]]:
        """Get list of keypoint coordinates (x, y) for valid keypoints."""
        return [(kp[0], kp[1]) for kp in self.keypoints if kp is not None]

    def get_keypoint_confidences(self) -> List[float]:
        """Get list of confidence scores for valid keypoints."""
        return [kp[2] for kp in self.keypoints if kp is not None]

    def get_center_point(self) -> Tuple[float, float]:
        """Calculate the center point of the pose."""
        valid_coords = self.get_keypoint_coordinates()
        if not valid_coords:
            return (0.0, 0.0)

        x_coords = [coord[0] for coord in valid_coords]
        y_coords = [coord[1] for coord in valid_coords]

        center_x = np.mean(x_coords)
        center_y = np.mean(y_coords)

        return (center_x, center_y)

    def normalize_keypoints(
        self, image_width: float, image_height: float
    ) -> "PoseData":
        """Create a new PoseData with normalized keypoints (0-1 range)."""
        normalized_keypoints = []

        for kp in self.keypoints:
            if kp is not None:
                x, y, conf = kp
                norm_x = x / image_width
                norm_y = y / image_height
                normalized_keypoints.append((norm_x, norm_y, conf))
            else:
                normalized_keypoints.append(None)

        # Normalize bounding box
        x1, y1, x2, y2 = self.bounding_box
        norm_bbox = (
            x1 / image_width,
            y1 / image_height,
            x2 / image_width,
            y2 / image_height,
        )

        return PoseData(
            keypoints=normalized_keypoints,
            bounding_box=norm_bbox,
            confidence_score=self.confidence_score,
            image_path=self.image_path,
            pose_id=self.pose_id,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert pose data to dictionary."""
        return {
            "keypoints": self.keypoints,
            "bounding_box": self.bounding_box,
            "confidence_score": self.confidence_score,
            "image_path": self.image_path,
            "pose_id": self.pose_id,
        }

    def to_json(self) -> str:
        """Convert pose data to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PoseData":
        """Create PoseData from dictionary."""
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> "PoseData":
        """Create PoseData from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)


@dataclass
class SimilarityResult:
    """Data structure for pose similarity results."""

    target_image: str
    comparison_image: str
    similarity_score: float
    keypoint_distances: List[float]
    rank: int
    comparison_pose: Optional["PoseData"] = (
        None  # The actual pose that gave this similarity score
    )

    def __post_init__(self):
        """Validate similarity result data."""
        if not (0.0 <= self.similarity_score <= 1.0):
            raise ValueError(
                f"Similarity score must be between 0 and 1, got {self.similarity_score}"
            )

        if self.rank < 1:
            raise ValueError(f"Rank must be >= 1, got {self.rank}")

    def get_average_keypoint_distance(self) -> float:
        """Calculate average keypoint distance."""
        valid_distances = [d for d in self.keypoint_distances if d >= 0]
        if not valid_distances:
            return 0.0
        return float(np.mean(valid_distances))

    def get_keypoint_distance_stats(self) -> Dict[str, float]:
        """Get statistics about keypoint distances."""
        valid_distances = [d for d in self.keypoint_distances if d >= 0]

        if not valid_distances:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0}

        return {
            "mean": float(np.mean(valid_distances)),
            "std": float(np.std(valid_distances)),
            "min": float(np.min(valid_distances)),
            "max": float(np.max(valid_distances)),
            "count": len(valid_distances),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert similarity result to dictionary."""
        return {
            "target_image": self.target_image,
            "comparison_image": self.comparison_image,
            "similarity_score": self.similarity_score,
            "keypoint_distances": self.keypoint_distances,
            "rank": self.rank,
            "avg_keypoint_distance": self.get_average_keypoint_distance(),
            "keypoint_distance_stats": self.get_keypoint_distance_stats(),
        }

    def to_json(self) -> str:
        """Convert similarity result to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


def calculate_pose_area(pose: PoseData) -> float:
    """Calculate the area covered by the pose bounding box."""
    x1, y1, x2, y2 = pose.bounding_box
    width = x2 - x1
    height = y2 - y1
    return width * height


def calculate_pose_aspect_ratio(pose: PoseData) -> float:
    """Calculate the aspect ratio of the pose bounding box."""
    x1, y1, x2, y2 = pose.bounding_box
    width = x2 - x1
    height = y2 - y1

    if height == 0:
        return 0.0

    return width / height


def get_pose_orientation(pose: PoseData) -> str:
    """Determine the general orientation of the pose."""
    # Use shoulders and hips to determine orientation
    left_shoulder = pose.keypoints[5]  # left shoulder
    right_shoulder = pose.keypoints[6]  # right shoulder
    left_hip = pose.keypoints[11]  # left hip
    right_hip = pose.keypoints[12]  # right hip

    if all(
        kp is not None for kp in [left_shoulder, right_shoulder, left_hip, right_hip]
    ):
        # Calculate center points
        shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) / 2
        hip_center_x = (left_hip[0] + right_hip[0]) / 2

        # Determine if person is facing left or right
        if shoulder_center_x > hip_center_x:
            return "facing_left"
        else:
            return "facing_right"

    return "unknown"


def filter_poses_by_confidence(
    poses: List[PoseData], min_confidence: float = 0.5
) -> List[PoseData]:
    """Filter poses by minimum confidence threshold."""
    return [pose for pose in poses if pose.confidence_score >= min_confidence]


def sort_poses_by_confidence(
    poses: List[PoseData], reverse: bool = True
) -> List[PoseData]:
    """Sort poses by confidence score."""
    return sorted(poses, key=lambda p: p.confidence_score, reverse=reverse)


def merge_poses(
    poses: List[PoseData], distance_threshold: float = 50.0
) -> List[PoseData]:
    """
    Merge poses that are close to each other.

    Args:
        poses: List of poses to merge
        distance_threshold: Maximum distance for merging

    Returns:
        List of merged poses
    """
    if len(poses) <= 1:
        return poses

    # Sort by confidence (highest first)
    sorted_poses = sort_poses_by_confidence(poses)
    merged = []
    used = set()

    for i, pose1 in enumerate(sorted_poses):
        if i in used:
            continue

        current_group = [pose1]
        used.add(i)

        for j, pose2 in enumerate(sorted_poses[i + 1 :], i + 1):
            if j in used:
                continue

            # Calculate distance between pose centers
            center1 = pose1.get_center_point()
            center2 = pose2.get_center_point()

            distance = np.sqrt(
                (center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2
            )

            if distance <= distance_threshold:
                current_group.append(pose2)
                used.add(j)

        # Merge poses in current group
        if len(current_group) > 1:
            merged_pose = _merge_pose_group(current_group)
            merged.append(merged_pose)
        else:
            merged.append(pose1)

    return merged


def _merge_pose_group(poses: List[PoseData]) -> PoseData:
    """Merge a group of poses into a single pose."""
    if len(poses) == 1:
        return poses[0]

    # Use the highest confidence pose as base
    base_pose = max(poses, key=lambda p: p.confidence_score)

    # Merge keypoints (use highest confidence for each keypoint)
    merged_keypoints = []
    for i in range(17):  # 17 COCO keypoints
        best_kp = None
        best_conf = 0.0

        for pose in poses:
            if pose.keypoints[i] is not None:
                kp = pose.keypoints[i]
                if kp[2] > best_conf:
                    best_kp = kp
                    best_conf = kp[2]

        merged_keypoints.append(best_kp)

    # Merge bounding box (union of all bounding boxes)
    all_x1 = [pose.bounding_box[0] for pose in poses]
    all_y1 = [pose.bounding_box[1] for pose in poses]
    all_x2 = [pose.bounding_box[2] for pose in poses]
    all_y2 = [pose.bounding_box[3] for pose in poses]

    merged_bbox = (min(all_x1), min(all_y1), max(all_x2), max(all_y2))

    # Calculate average confidence
    avg_confidence = np.mean([pose.confidence_score for pose in poses])

    return PoseData(
        keypoints=merged_keypoints,
        bounding_box=merged_bbox,
        confidence_score=avg_confidence,
        image_path=base_pose.image_path,
        pose_id=f"merged_{base_pose.pose_id}",
    )


def save_poses_to_file(poses: List[PoseData], output_path: str) -> bool:
    """Save poses to a JSON file."""
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        poses_data = [pose.to_dict() for pose in poses]

        with open(output_path, "w") as f:
            json.dump(poses_data, f, indent=2)

        return True
    except Exception as e:
        print(f"Error saving poses to {output_path}: {e}")
        return False


def load_poses_from_file(file_path: str) -> List[PoseData]:
    """Load poses from a JSON file."""
    try:
        with open(file_path, "r") as f:
            poses_data = json.load(f)

        return [PoseData.from_dict(data) for data in poses_data]
    except Exception as e:
        print(f"Error loading poses from {file_path}: {e}")
        return []
