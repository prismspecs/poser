#!/usr/bin/env python3
"""
Demo script for the YOLO v13 Pose Estimation project.
This script demonstrates the basic functionality without requiring real images.
"""

import numpy as np
from pose_estimator import PoseEstimator
from pose_matcher import PoseMatcher
from utils.pose_utils import (
    PoseData,
    SimilarityResult,
    calculate_pose_area,
    calculate_pose_aspect_ratio,
    get_pose_orientation,
    filter_poses_by_confidence,
    sort_poses_by_confidence,
)


def create_sample_pose_data():
    """Create sample pose data for demonstration purposes."""

    # Create sample keypoints for a standing person
    # Format: (x, y, confidence) for 17 COCO keypoints
    standing_keypoints = [
        (320, 100, 0.95),  # nose
        (310, 90, 0.90),  # left_eye
        (330, 90, 0.90),  # right_eye
        (300, 85, 0.85),  # left_ear
        (340, 85, 0.85),  # right_ear
        (280, 150, 0.95),  # left_shoulder
        (360, 150, 0.95),  # right_shoulder
        (250, 200, 0.90),  # left_elbow
        (390, 200, 0.90),  # right_elbow
        (220, 250, 0.85),  # left_wrist
        (420, 250, 0.85),  # right_wrist
        (300, 300, 0.95),  # left_hip
        (340, 300, 0.95),  # right_hip
        (280, 400, 0.90),  # left_knee
        (360, 400, 0.90),  # right_knee
        (270, 500, 0.85),  # left_ankle
        (370, 500, 0.85),  # right_ankle
    ]

    # Create sample keypoints for a similar but slightly different pose
    similar_keypoints = [
        (325, 105, 0.93),  # nose
        (315, 95, 0.88),  # left_eye
        (335, 95, 0.88),  # right_eye
        (305, 90, 0.83),  # left_ear
        (345, 90, 0.83),  # right_ear
        (285, 155, 0.93),  # left_shoulder
        (365, 155, 0.93),  # right_shoulder
        (255, 205, 0.88),  # left_elbow
        (395, 205, 0.88),  # right_elbow
        (225, 255, 0.83),  # left_wrist
        (425, 255, 0.83),  # right_wrist
        (305, 305, 0.93),  # left_hip
        (345, 305, 0.93),  # right_hip
        (285, 405, 0.88),  # left_knee
        (365, 405, 0.88),  # right_knee
        (275, 505, 0.83),  # left_ankle
        (375, 505, 0.83),  # right_ankle
    ]

    # Create sample keypoints for a very different pose (sitting)
    sitting_keypoints = [
        (320, 150, 0.92),  # nose
        (310, 140, 0.87),  # left_eye
        (330, 140, 0.87),  # right_eye
        (300, 135, 0.82),  # left_ear
        (340, 135, 0.82),  # right_ear
        (280, 200, 0.92),  # left_shoulder
        (360, 200, 0.92),  # right_shoulder
        (250, 250, 0.87),  # left_elbow
        (390, 250, 0.87),  # right_elbow
        (220, 300, 0.82),  # left_wrist
        (420, 300, 0.82),  # right_wrist
        (300, 350, 0.92),  # left_hip
        (340, 350, 0.92),  # right_hip
        (280, 450, 0.87),  # left_knee
        (360, 450, 0.87),  # right_knee
        (270, 550, 0.82),  # left_ankle
        (370, 550, 0.82),  # right_ankle
    ]

    # Create PoseData objects
    standing_pose = PoseData(
        keypoints=standing_keypoints,
        bounding_box=(200, 50, 440, 550),
        confidence_score=0.89,
        image_path="standing_person.jpg",
        pose_id="standing_001",
    )

    similar_pose = PoseData(
        keypoints=similar_keypoints,
        bounding_box=(205, 55, 445, 555),
        confidence_score=0.87,
        image_path="similar_person.jpg",
        pose_id="similar_001",
    )

    sitting_pose = PoseData(
        keypoints=sitting_keypoints,
        bounding_box=(200, 100, 440, 600),
        confidence_score=0.85,
        image_path="sitting_person.jpg",
        pose_id="sitting_001",
    )

    return standing_pose, similar_pose, sitting_pose


def demonstrate_pose_matching():
    """Demonstrate pose matching functionality."""
    print("🎯 YOLO v13 Pose Estimation Demo")
    print("=" * 50)

    # Create sample pose data
    print("📊 Creating sample pose data...")
    standing_pose, similar_pose, sitting_pose = create_sample_pose_data()

    # Initialize pose matcher
    print("🔍 Initializing pose matcher...")
    matcher = PoseMatcher(distance_metric="euclidean", normalize_keypoints=True)

    # Compare poses
    print("\n📈 Comparing poses...")

    # Compare standing pose with similar pose
    similarity_score = matcher.calculate_similarity(standing_pose, similar_pose)
    print(f"Standing vs Similar: {similarity_score:.3f}")

    # Compare standing pose with sitting pose
    similarity_score = matcher.calculate_similarity(standing_pose, sitting_pose)
    print(f"Standing vs Sitting: {similarity_score:.3f}")

    # Rank all poses
    print("\n🏆 Ranking poses by similarity to standing pose...")
    comparison_poses = [similar_pose, sitting_pose]
    ranked_results = matcher.rank_poses(standing_pose, comparison_poses)

    for i, result in enumerate(ranked_results, 1):
        print(f"{i}. {result.comparison_image}")
        print(f"   Similarity: {result.similarity_score:.3f}")
        print(f"   Rank: {result.rank}")
        print()

    # Get matching statistics
    print("📊 Matching Statistics:")
    stats = matcher.get_matching_statistics(standing_pose, comparison_poses)
    for key, value in stats.items():
        if key != "similarity_distribution":
            print(f"   {key}: {value}")

    print(f"   High similarity (≥0.8): {stats['similarity_distribution']['high']}")
    print(
        f"   Medium similarity (0.5-0.8): {stats['similarity_distribution']['medium']}"
    )
    print(f"   Low similarity (<0.5): {stats['similarity_distribution']['low']}")


def demonstrate_pose_analysis():
    """Demonstrate pose analysis functionality."""
    print("\n🔬 Pose Analysis Demo")
    print("=" * 30)

    standing_pose, _, _ = create_sample_pose_data()

    # Analyze pose properties
    print(f"📏 Pose Area: {calculate_pose_area(standing_pose):.0f} pixels²")
    print(f"📐 Aspect Ratio: {calculate_pose_aspect_ratio(standing_pose):.2f}")
    print(f"🧭 Orientation: {get_pose_orientation(standing_pose)}")

    # Keypoint analysis
    valid_keypoints = standing_pose.get_valid_keypoints()
    print(f"🎯 Valid Keypoints: {len(valid_keypoints)}/17")

    # Confidence analysis
    confidences = standing_pose.get_keypoint_confidences()
    print(f"📊 Average Confidence: {np.mean(confidences):.3f}")
    print(f"📊 Min Confidence: {np.min(confidences):.3f}")
    print(f"📊 Max Confidence: {np.max(confidences):.3f}")


def demonstrate_pose_utilities():
    """Demonstrate pose utility functions."""
    print("\n🛠️ Pose Utilities Demo")
    print("=" * 30)

    standing_pose, similar_pose, sitting_pose = create_sample_pose_data()

    # Filter poses by confidence
    all_poses = [standing_pose, similar_pose, sitting_pose]
    high_confidence_poses = filter_poses_by_confidence(all_poses, min_confidence=0.88)
    print(f"🔍 High confidence poses (≥0.88): {len(high_confidence_poses)}")

    # Sort poses by confidence
    sorted_poses = sort_poses_by_confidence(all_poses)
    print(f"📊 Top pose confidence: {sorted_poses[0].confidence_score:.3f}")

    # Pose normalization
    normalized_pose = standing_pose.normalize_keypoints(640, 640)
    print(f"📐 Normalized pose created: {normalized_pose.pose_id}")

    # Convert to JSON
    json_data = standing_pose.to_json()
    print(f"📄 JSON representation length: {len(json_data)} characters")


def main():
    """Main demo function."""
    try:
        demonstrate_pose_matching()
        demonstrate_pose_analysis()
        demonstrate_pose_utilities()

        print("\n✅ Demo completed successfully!")
        print("\n💡 To use with real images:")
        print(
            "   python main.py --target target.jpg --comparison-dir ./comparison_images/ --verbose"
        )

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
