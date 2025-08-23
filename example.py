#!/usr/bin/env python3
"""
Example script showing programmatic usage of the pose estimation system.
This script demonstrates how to use the classes and functions directly in your code.
"""

import numpy as np
from pose_estimator import PoseEstimator
from pose_matcher import PoseMatcher
from utils.pose_utils import PoseData, SimilarityResult
from utils.image_utils import load_image, save_image


def example_basic_usage():
    """Example of basic pose estimation and matching."""
    print("🔍 Example: Basic Pose Estimation and Matching")
    print("=" * 50)

    # Initialize pose estimator
    estimator = PoseEstimator(confidence_threshold=0.5, model_size="n")

    # Initialize pose matcher
    matcher = PoseMatcher(distance_metric="euclidean", normalize_keypoints=True)

    print("✅ Initialized pose estimator and matcher")
    print(f"   Model size: {estimator.model_size}")
    print(f"   Confidence threshold: {estimator.confidence_threshold}")
    print(f"   Distance metric: {matcher.distance_metric}")
    print()


def example_pose_analysis():
    """Example of pose analysis and manipulation."""
    print("📊 Example: Pose Analysis and Manipulation")
    print("=" * 50)

    # Create sample pose data
    keypoints = [
        (100, 100, 0.9),
        (200, 100, 0.8),
        (150, 200, 0.7),  # nose, eyes
        (120, 110, 0.6),
        (180, 110, 0.5),  # ears
        (80, 150, 0.9),
        (220, 150, 0.8),  # shoulders
        (60, 200, 0.7),
        (240, 200, 0.6),  # elbows
        (40, 250, 0.5),
        (260, 250, 0.4),  # wrists
        (100, 300, 0.8),
        (200, 300, 0.7),  # hips
        (90, 400, 0.6),
        (210, 400, 0.5),  # knees
        (80, 500, 0.4),
        (220, 500, 0.3),  # ankles
    ]

    pose = PoseData(
        keypoints=keypoints,
        bounding_box=(30, 80, 270, 520),
        confidence_score=0.75,
        image_path="example.jpg",
        pose_id="example_001",
    )

    print(f"✅ Created pose: {pose.pose_id}")
    print(f"   Confidence: {pose.confidence_score:.3f}")
    print(f"   Valid keypoints: {len(pose.get_valid_keypoints())}/17")
    print(f"   Center point: {pose.get_center_point()}")
    print(f"   Bounding box: {pose.bounding_box}")
    print()


def example_similarity_calculation():
    """Example of pose similarity calculation."""
    print("📈 Example: Pose Similarity Calculation")
    print("=" * 50)

    # Create two similar poses
    pose1_keypoints = [(100 + i, 100 + i, 0.9) for i in range(17)]
    pose2_keypoints = [(105 + i, 105 + i, 0.9) for i in range(17)]

    pose1 = PoseData(
        keypoints=pose1_keypoints,
        bounding_box=(90, 90, 120, 120),
        confidence_score=0.9,
        image_path="pose1.jpg",
        pose_id="pose1",
    )

    pose2 = PoseData(
        keypoints=pose2_keypoints,
        bounding_box=(95, 95, 125, 125),
        confidence_score=0.9,
        image_path="pose2.jpg",
        pose_id="pose2",
    )

    # Calculate similarity
    matcher = PoseMatcher()
    similarity = matcher.calculate_similarity(pose1, pose2)

    print(f"✅ Calculated similarity between poses")
    print(f"   Pose 1: {pose1.pose_id}")
    print(f"   Pose 2: {pose2.pose_id}")
    print(f"   Similarity score: {similarity:.3f}")
    print()


def example_batch_processing():
    """Example of batch pose processing."""
    print("🔄 Example: Batch Pose Processing")
    print("=" * 50)

    # Create multiple poses
    poses = []
    for i in range(3):
        keypoints = [(100 + i * 10, 100 + i * 10, 0.9) for _ in range(17)]
        pose = PoseData(
            keypoints=keypoints,
            bounding_box=(90 + i * 10, 90 + i * 10, 120 + i * 10, 120 + i * 10),
            confidence_score=0.9 - i * 0.1,
            image_path=f"batch_pose_{i}.jpg",
            pose_id=f"batch_{i:03d}",
        )
        poses.append(pose)

    print(f"✅ Created {len(poses)} poses for batch processing")

    # Sort by confidence
    from utils.pose_utils import sort_poses_by_confidence

    sorted_poses = sort_poses_by_confidence(poses)

    print("   Sorted by confidence:")
    for i, pose in enumerate(sorted_poses):
        print(f"     {i+1}. {pose.pose_id}: {pose.confidence_score:.3f}")

    # Filter by confidence
    from utils.pose_utils import filter_poses_by_confidence

    high_conf_poses = filter_poses_by_confidence(poses, min_confidence=0.8)
    print(f"   High confidence poses (≥0.8): {len(high_conf_poses)}")
    print()


def example_pose_utilities():
    """Example of pose utility functions."""
    print("🛠️ Example: Pose Utility Functions")
    print("=" * 50)

    # Create a pose
    keypoints = [(100, 100, 0.9)] * 17
    pose = PoseData(
        keypoints=keypoints,
        bounding_box=(50, 50, 150, 150),
        confidence_score=0.9,
        image_path="utility_test.jpg",
        pose_id="utility_test",
    )

    # Test utility functions
    from utils.pose_utils import (
        calculate_pose_area,
        calculate_pose_aspect_ratio,
        get_pose_orientation,
        save_poses_to_file,
    )

    area = calculate_pose_area(pose)
    aspect_ratio = calculate_pose_aspect_ratio(pose)
    orientation = get_pose_orientation(pose)

    print(f"✅ Pose utility calculations:")
    print(f"   Area: {area:.0f} pixels²")
    print(f"   Aspect ratio: {aspect_ratio:.2f}")
    print(f"   Orientation: {orientation}")

    # Save pose to file
    success = save_poses_to_file([pose], "example_pose.json")
    if success:
        print(f"   ✅ Saved pose to example_pose.json")
    print()


def example_custom_matcher():
    """Example of custom matcher configuration."""
    print("⚙️ Example: Custom Matcher Configuration")
    print("=" * 50)

    # Create matcher with custom settings
    matcher = PoseMatcher(
        distance_metric="cosine", normalize_keypoints=False, weight_by_confidence=False
    )

    print(f"✅ Custom matcher configuration:")
    print(f"   Distance metric: {matcher.distance_metric}")
    print(f"   Normalize keypoints: {matcher.normalize_keypoints}")
    print(f"   Weight by confidence: {matcher.weight_by_confidence}")

    # Change settings dynamically
    matcher.set_distance_metric("manhattan")
    matcher.set_normalization(True)

    print(f"   Updated distance metric: {matcher.distance_metric}")
    print(f"   Updated normalization: {matcher.normalize_keypoints}")
    print()


def main():
    """Run all examples."""
    print("🚀 YOLO v13 Pose Estimation - Examples")
    print("=" * 60)
    print()

    try:
        example_basic_usage()
        example_pose_analysis()
        example_similarity_calculation()
        example_batch_processing()
        example_pose_utilities()
        example_custom_matcher()

        print("✅ All examples completed successfully!")
        print()
        print("💡 Next steps:")
        print("   - Try the demo: python demo.py")
        print("   - Run tests: python -m pytest tests/ -v")
        print("   - Use with real images: python main.py --help")

    except Exception as e:
        print(f"❌ Example failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
