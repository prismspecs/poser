#!/usr/bin/env python3
"""
YOLO v13 Pose Estimation - Main Entry Point
Finds the closest pose match between a target image and comparison images.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

from pose_estimator import PoseEstimator
from pose_matcher import PoseMatcher
from utils.image_utils import load_image, validate_image_path
from utils.pose_utils import PoseData, SimilarityResult


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Find closest pose match using YOLO v13 pose estimation"
    )
    parser.add_argument(
        "--target", required=True, help="Path to target image for pose matching"
    )
    parser.add_argument(
        "--comparison-dir", required=True, help="Directory containing comparison images"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for pose detection (default: 0.5)",
    )
    parser.add_argument("--output", help="Output file for results (JSON format)")
    parser.add_argument(
        "--max-results",
        type=int,
        default=10,
        help="Maximum number of results to return (default: 10)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    return parser.parse_args()


def main():
    """Main application entry point."""
    args = parse_arguments()

    # Validate target image
    if not validate_image_path(args.target):
        print(f"Error: Invalid target image path: {args.target}")
        sys.exit(1)

    # Validate comparison directory
    comparison_dir = Path(args.comparison_dir)
    if not comparison_dir.exists() or not comparison_dir.is_dir():
        print(f"Error: Invalid comparison directory: {args.comparison_dir}")
        sys.exit(1)

    try:
        # Initialize pose estimator
        if args.verbose:
            print("Initializing YOLO v13 pose estimator...")

        estimator = PoseEstimator(confidence_threshold=args.threshold)

        # Load target image and extract pose
        if args.verbose:
            print(f"Processing target image: {args.target}")

        target_image = load_image(args.target)
        target_poses = estimator.extract_poses(target_image, args.target)

        if not target_poses:
            print("Error: No poses detected in target image")
            sys.exit(1)

        # Use the highest confidence pose as target
        target_pose = max(target_poses, key=lambda p: p.confidence_score)
        if args.verbose:
            if len(target_poses) > 1:
                print(
                    f"Found {len(target_poses)} people in target image, using highest confidence person"
                )
                for i, pose in enumerate(target_poses):
                    print(f"  Person {i+1}: {pose.confidence_score:.3f} confidence")
                print(
                    f"Selected target pose: {target_pose.pose_id} with {target_pose.confidence_score:.3f} confidence"
                )
            else:
                print(f"Target pose confidence: {target_pose.confidence_score:.3f}")

        # Process comparison images
        if args.verbose:
            print(f"Processing comparison images from: {args.comparison_dir}")

        comparison_images = (
            list(comparison_dir.glob("*.jpg"))
            + list(comparison_dir.glob("*.jpeg"))
            + list(comparison_dir.glob("*.png"))
        )

        if not comparison_images:
            print("Error: No image files found in comparison directory")
            sys.exit(1)

        # Initialize pose matcher
        matcher = PoseMatcher()

        # Process each comparison image
        results = []
        for img_path in comparison_images:
            if args.verbose:
                print(f"Processing: {img_path.name}")

            try:
                comparison_image = load_image(str(img_path))
                comparison_poses = estimator.extract_poses(
                    comparison_image, str(img_path)
                )

                if comparison_poses:
                    if args.verbose and len(comparison_poses) > 1:
                        print(
                            f"   Found {len(comparison_poses)} people, comparing with all"
                        )

                    # Find best match among all people in this image
                    best_match = matcher.find_best_match(target_pose, comparison_poses)
                    if best_match:
                        results.append(best_match)

            except Exception as e:
                if args.verbose:
                    print(f"Warning: Failed to process {img_path.name}: {e}")
                continue

        # Sort results by similarity score (descending)
        results.sort(key=lambda x: x.similarity_score, reverse=True)

        # Limit results
        results = results[: args.max_results]

        # Display results
        print(f"\nFound {len(results)} pose matches:")
        print("-" * 80)

        for i, result in enumerate(results, 1):
            print(f"{i:2d}. {Path(result.comparison_image).name}")
            print(f"    Similarity Score: {result.similarity_score:.3f}")
            print(f"    Rank: {result.rank}")
            print()

        # Save results if output file specified
        if args.output:
            save_results(results, args.output, target_pose)
            print(f"Results saved to: {args.output}")

    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


def save_results(
    results: List[SimilarityResult], output_path: str, target_pose: PoseData
):
    """Save results to JSON file."""
    output_data = {
        "target_image": target_pose.image_path,
        "target_pose_confidence": target_pose.confidence_score,
        "results": [
            {
                "comparison_image": result.comparison_image,
                "similarity_score": result.similarity_score,
                "rank": result.rank,
                "keypoint_distances": result.keypoint_distances,
            }
            for result in results
        ],
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)


if __name__ == "__main__":
    main()
