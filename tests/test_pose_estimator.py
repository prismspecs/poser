"""
Tests for the pose estimator module.
"""

import unittest
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pose_estimator import PoseEstimator
from utils.pose_utils import PoseData


class TestPoseEstimator(unittest.TestCase):
    """Test cases for the PoseEstimator class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock image for testing
        self.test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        self.test_image_path = "test_image.jpg"

    @patch("ultralytics.YOLO")
    def test_initialization(self, mock_yolo):
        """Test PoseEstimator initialization."""
        # Mock the YOLO model
        mock_model = Mock()
        mock_yolo.return_value = mock_model

        estimator = PoseEstimator(confidence_threshold=0.7, model_size="s")

        self.assertEqual(estimator.confidence_threshold, 0.7)
        self.assertEqual(estimator.model_size, "s")
        self.assertIsNotNone(estimator.model)

    def test_confidence_threshold_validation(self):
        """Test confidence threshold validation."""
        # Test valid threshold
        estimator = PoseEstimator(confidence_threshold=0.8)
        self.assertEqual(estimator.confidence_threshold, 0.8)

        # Test threshold clamping
        estimator.set_confidence_threshold(1.5)  # Should be clamped to 1.0
        self.assertEqual(estimator.confidence_threshold, 1.0)

        estimator.set_confidence_threshold(-0.5)  # Should be clamped to 0.0
        self.assertEqual(estimator.confidence_threshold, 0.0)

    def test_keypoint_names(self):
        """Test keypoint names retrieval."""
        estimator = PoseEstimator()
        keypoint_names = estimator.get_keypoint_names()

        self.assertEqual(len(keypoint_names), 17)
        self.assertIn("nose", keypoint_names)
        self.assertIn("left_shoulder", keypoint_names)
        self.assertIn("right_ankle", keypoint_names)

    def test_bbox_calculation(self):
        """Test bounding box calculation from keypoints."""
        estimator = PoseEstimator()

        # Test with valid keypoints
        test_keypoints = [
            (100, 100, 0.9),
            (200, 100, 0.8),
            (150, 200, 0.7),  # Some valid keypoints
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,  # Missing keypoints
            None,
            None,
            None,
            None,
            None,
            None,
        ]

        bbox = estimator._calculate_bbox_from_keypoints(test_keypoints)
        self.assertEqual(len(bbox), 4)
        self.assertIsInstance(bbox[0], float)

        # Test with no valid keypoints
        empty_keypoints = [None] * 17
        bbox = estimator._calculate_bbox_from_keypoints(empty_keypoints)
        self.assertEqual(bbox, (0.0, 0.0, 0.0, 0.0))

    def test_model_info(self):
        """Test model information retrieval."""
        estimator = PoseEstimator()
        info = estimator.get_model_info()

        self.assertIn("status", info)
        self.assertIn("model_size", info)
        self.assertIn("confidence_threshold", info)
        self.assertIn("keypoint_count", info)
        self.assertIn("keypoint_names", info)


class TestPoseData(unittest.TestCase):
    """Test cases for the PoseData class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create test keypoints (17 COCO keypoints)
        self.test_keypoints = [
            (100, 100, 0.9),
            (200, 100, 0.8),
            (150, 200, 0.7),  # nose, left_eye, right_eye
            (120, 110, 0.6),
            (180, 110, 0.5),  # left_ear, right_ear
            (80, 150, 0.9),
            (220, 150, 0.8),  # left_shoulder, right_shoulder
            (60, 200, 0.7),
            (240, 200, 0.6),  # left_elbow, right_elbow
            (40, 250, 0.5),
            (260, 250, 0.4),  # left_wrist, right_wrist
            (100, 300, 0.8),
            (200, 300, 0.7),  # left_hip, right_hip
            (90, 400, 0.6),
            (210, 400, 0.5),  # left_knee, right_knee
            (80, 500, 0.4),
            (220, 500, 0.3),  # left_ankle, right_ankle
        ]

        self.test_bbox = (50, 50, 250, 550)
        self.test_image_path = "test_image.jpg"
        self.test_pose_id = "test_pose_001"

    def test_pose_data_creation(self):
        """Test PoseData object creation."""
        pose = PoseData(
            keypoints=self.test_keypoints,
            bounding_box=self.test_bbox,
            confidence_score=0.75,
            image_path=self.test_image_path,
            pose_id=self.test_pose_id,
        )

        self.assertEqual(len(pose.keypoints), 17)
        self.assertEqual(len(pose.bounding_box), 4)
        self.assertEqual(pose.confidence_score, 0.75)
        self.assertEqual(pose.image_path, self.test_image_path)
        self.assertEqual(pose.pose_id, self.test_pose_id)

    def test_pose_data_validation(self):
        """Test PoseData validation."""
        # Test invalid keypoint count
        with self.assertRaises(ValueError):
            PoseData(
                keypoints=[(100, 100, 0.9)],  # Only 1 keypoint
                bounding_box=self.test_bbox,
                confidence_score=0.75,
                image_path=self.test_image_path,
                pose_id=self.test_pose_id,
            )

        # Test invalid bounding box
        with self.assertRaises(ValueError):
            PoseData(
                keypoints=self.test_keypoints,
                bounding_box=(50, 50),  # Only 2 coordinates
                confidence_score=0.75,
                image_path=self.test_image_path,
                pose_id=self.test_pose_id,
            )

        # Test invalid confidence score
        with self.assertRaises(ValueError):
            PoseData(
                keypoints=self.test_keypoints,
                bounding_box=self.test_bbox,
                confidence_score=1.5,  # > 1.0
                image_path=self.test_image_path,
                pose_id=self.test_pose_id,
            )

    def test_valid_keypoints(self):
        """Test valid keypoint retrieval."""
        pose = PoseData(
            keypoints=self.test_keypoints,
            bounding_box=self.test_bbox,
            confidence_score=0.75,
            image_path=self.test_image_path,
            pose_id=self.test_pose_id,
        )

        valid_kps = pose.get_valid_keypoints()
        self.assertEqual(len(valid_kps), 17)

        # Check that all keypoints are valid
        for idx, kp in valid_kps:
            self.assertIsNotNone(kp)
            self.assertEqual(len(kp), 3)  # x, y, confidence

    def test_center_point_calculation(self):
        """Test center point calculation."""
        pose = PoseData(
            keypoints=self.test_keypoints,
            bounding_box=self.test_bbox,
            confidence_score=0.75,
            image_path=self.test_image_path,
            pose_id=self.test_pose_id,
        )

        center = pose.get_center_point()
        self.assertEqual(len(center), 2)
        self.assertIsInstance(center[0], float)
        self.assertIsInstance(center[1], float)

        # Center should be within bounding box
        x1, y1, x2, y2 = pose.bounding_box
        self.assertTrue(x1 <= center[0] <= x2)
        self.assertTrue(y1 <= center[1] <= y2)


if __name__ == "__main__":
    unittest.main()
