#!/usr/bin/env python3
"""
YOLOv11 Pose Estimation - Main Entry Point
Finds the closest pose match between a target image and comparison images.
"""

import argparse
import json
import sys
import random
import time
from pathlib import Path
from typing import List
import cv2
import numpy as np

from pose_estimator import PoseEstimator
from pose_matcher import PoseMatcher
from pose_visualizer import PoseVisualizer
from utils.image_utils import load_image, validate_image_path
from utils.pose_utils import PoseData, SimilarityResult


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Find closest pose match using YOLOv11 pose estimation"
    )
    parser.add_argument(
        "--target",
        required=True,
        help="Path to target image OR directory of target images",
    )
    parser.add_argument(
        "--comparison-dir", required=True, help="Directory containing comparison images"
    )
    parser.add_argument(
        "--random-target",
        action="store_true",
        help="Randomly select target image if target is a directory",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Confidence threshold for pose detection (default: 0.7)",
    )
    parser.add_argument("--output", help="Output file for results (JSON format)")
    parser.add_argument(
        "--max-results",
        type=int,
        default=10,
        help="Maximum number of results to return (default: 10)",
    )
    parser.add_argument(
        "--relative-visibility-threshold",
        type=float,
        default=0.65,
        help="Minimum percentage of target's visible keypoints that must be shared (default: 0.65)",
    )
    parser.add_argument("--no-cache", action="store_true", help="Disable pose caching")
    parser.add_argument(
        "--clear-cache", action="store_true", help="Clear the pose cache before running"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate diagnostic visualization images",
    )
    parser.add_argument(
        "--no-mask",
        action="store_true",
        help="Disable body segmentation masks on comparison images (masks are on by default)",
    )
    parser.add_argument(
        "--no-skeleton",
        action="store_true",
        help="Disable skeleton drawing (lines + keypoints) on comparison images (target overlay always shows skeleton)",
    )
    parser.add_argument(
        "--layer-poses",
        action="store_true",
        help="Create layered visualizations overlaying comparison poses on target image with correct positioning",
    )

    parser.add_argument(
        "--model-size",
        choices=["n", "s", "m", "l", "x"],
        default="n",
        help="YOLOv11 model size: n=nano, s=small, m=medium, l=large, x=xlarge (default: n)",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory to save visualization outputs",
    )
    return parser.parse_args()


def get_target_image_path(args) -> str:
    """Get target image path from arguments."""
    target_path = Path(args.target)

    if target_path.is_file():
        if not validate_image_path(args.target):
            print(f"Error: Invalid target image path: {args.target}")
            sys.exit(1)
        return args.target

    elif target_path.is_dir():
        target_images = (
            list(target_path.glob("*.jpg"))
            + list(target_path.glob("*.jpeg"))
            + list(target_path.glob("*.png"))
            + list(target_path.glob("*.webp"))
        )
        if not target_images:
            print(f"Error: No image files found in target directory: {args.target}")
            sys.exit(1)

        if args.random_target:
            target_image_path = str(random.choice(target_images))
            if args.verbose:
                print(f"Randomly selected target: {Path(target_image_path).name}")
        else:
            target_image_path = str(target_images[0])
            if args.verbose:
                print(f"Using first target image: {Path(target_image_path).name}")
        return target_image_path

    else:
        print(f"Error: Target path does not exist: {args.target}")
        sys.exit(1)


def get_comparison_images(comparison_dir: Path) -> List[Path]:
    """Get list of comparison image paths."""
    comparison_images = (
        list(comparison_dir.glob("*.jpg"))
        + list(comparison_dir.glob("*.jpeg"))
        + list(comparison_dir.glob("*.png"))
        + list(comparison_dir.glob("*.webp"))
    )
    if not comparison_images:
        print(f"Error: No image files found in comparison directory: {comparison_dir}")
        sys.exit(1)
    return comparison_images


def process_comparison_images(
    estimator,
    matcher,
    target_pose,
    comparison_images,
    relative_visibility_threshold,
    verbose,
):
    """Process comparison images and find best matches."""
    results = []
    comparison_start_time = time.time()
    total_pose_extraction_time = 0.0
    total_pose_matching_time = 0.0

    for img_path in comparison_images:
        if verbose:
            print(f"Processing: {img_path.name}")

        try:
            comparison_image = load_image(str(img_path))
            start_time = time.time()
            comparison_poses = estimator.extract_poses(comparison_image, str(img_path))
            pose_time = time.time() - start_time
            total_pose_extraction_time += pose_time

            if comparison_poses:
                if verbose and len(comparison_poses) > 1:
                    print(
                        f"   Found {len(comparison_poses)} people, comparing with all"
                    )

                start_time = time.time()
                best_match = matcher.find_best_match(
                    target_pose, comparison_poses, relative_visibility_threshold
                )
                match_time = time.time() - start_time
                total_pose_matching_time += match_time

                if best_match:
                    results.append(best_match)

        except Exception as e:
            if verbose:
                print(f"Warning: Failed to process {img_path.name}: {e}")
            continue

    total_time = time.time() - comparison_start_time
    return results, total_time, total_pose_extraction_time, total_pose_matching_time


def print_timing_summary(
    init_time,
    target_time,
    total_time,
    pose_extraction_time,
    pose_matching_time,
    num_images,
):
    """Print timing summary."""
    print(f"\n=== TIMING SUMMARY ===")
    print(f"Model initialization: {init_time:.2f} seconds")
    print(f"Target pose extraction: {target_time:.2f} seconds")
    print(f"Comparison pose extraction: {pose_extraction_time:.2f} seconds")
    print(f"Pose matching: {pose_matching_time:.2f} seconds")
    print(f"Total comparison processing: {total_time:.2f} seconds")
    print(
        f"Total runtime (without viz): {init_time + target_time + total_time:.2f} seconds"
    )
    print(f"Average time per comparison image: {total_time/num_images:.3f} seconds")
    print(f"=" * 30)


def print_results(results, max_results):
    """Print results summary."""
    print(
        f"\nFound {len(results)} pose matches, showing top {min(len(results), max_results)}:"
    )
    print("-" * 80)

    for i, result in enumerate(results[:max_results], 1):
        print(f"{i:2d}. {Path(result.comparison_image).name}")
        print(f"    Similarity Score: {result.similarity_score:.3f}")
        print(f"    Rank: {result.rank}")
        print()


def generate_visualizations(
    estimator,
    target_image,
    target_pose,
    results,
    comparison_images,
    output_dir,
    verbose,
    show_skeleton=True,
    show_mask=True,
    layer_poses=False,
):
    """Generate diagnostic visualizations."""
    if verbose:
        print("Generating diagnostic visualizations...")
        viz_start_time = time.time()

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    visualizer = PoseVisualizer()

    # Prepare comparison images and poses for visualization
    comparison_image_arrays = []
    comparison_poses_data = []

    for img_path in comparison_images:
        try:
            img_array = load_image(str(img_path))
            comparison_image_arrays.append((str(img_path), img_array))

            # Extract poses for skeleton drawing
            comp_poses = estimator.extract_poses(img_array, str(img_path))
            if comp_poses:
                # Use the highest confidence pose for visualization
                best_pose = max(comp_poses, key=lambda p: p.confidence_score)
                comparison_poses_data.append(best_pose)
            else:
                comparison_poses_data.append(None)
        except Exception as e:
            comparison_image_arrays.append((str(img_path), None))
            comparison_poses_data.append(None)

    # Create main comparison visualization
    vis_output_path = (
        output_dir / f"pose_comparison_{Path(target_pose.image_path).stem}.jpg"
    )
    visualizer.create_comparison_visualization(
        target_image,
        target_pose,
        results,
        comparison_image_arrays,
        comparison_poses_data,
        str(vis_output_path),
        apply_body_mask=show_mask,
        pose_estimator=estimator,
        show_skeleton=show_skeleton,
    )

    # Create detailed analysis for top result
    if results:
        top_result = results[0]
        top_pose = None

        # Find the corresponding pose data
        for img_path in comparison_images:
            if str(img_path) == top_result.comparison_image:
                try:
                    comp_img = load_image(str(img_path))
                    comp_poses = estimator.extract_poses(comp_img, str(img_path))
                    if comp_poses:
                        # Find pose with matching similarity score
                        for pose in comp_poses:
                            score = visualizer.calculate_similarity(
                                target_pose, pose, 0.7, 0.6
                            )
                            if abs(score - top_result.similarity_score) < 0.001:
                                top_pose = pose
                                break
                        if not top_pose and comp_poses:
                            top_pose = max(comp_poses, key=lambda p: p.confidence_score)
                        break
                except Exception as e:
                    if verbose:
                        print(f"Warning: Failed to extract pose for visualization: {e}")
                    continue

        if top_pose:
            # Create keypoint analysis
            analysis_output_path = (
                output_dir
                / f"keypoint_analysis_{Path(target_pose.image_path).stem}.jpg"
            )
            analysis_vis = visualizer.create_keypoint_analysis(
                target_pose, top_pose, top_result
            )
            cv2.imwrite(str(analysis_output_path), analysis_vis)
            print(f"Keypoint analysis saved to: {analysis_output_path}")

            # Create winning pose overlay
            overlay_output_path = (
                output_dir / f"pose_overlay_{Path(target_pose.image_path).stem}.jpg"
            )
            overlay_vis = visualizer.create_winning_pose_overlay(
                target_image,
                target_pose,
                top_pose,
                top_result.similarity_score,
                str(overlay_output_path),
            )

    # Generate layered poses if requested
    if layer_poses and results:
        if verbose:
            print("Generating layered pose visualizations...")

        layer_output_dir = output_dir / "layered_poses"
        layer_output_dir.mkdir(exist_ok=True)

        # Create layered visualizations for top results (limit to avoid too many files)
        max_layers = min(5, len(results))

        for i, result in enumerate(results[:max_layers]):
            # Find corresponding pose data
            comp_pose = None
            comp_img = None

            for img_path in comparison_images:
                if str(img_path) == result.comparison_image:
                    try:
                        comp_img = load_image(str(img_path))
                        comp_poses = estimator.extract_poses(comp_img, str(img_path))
                        if comp_poses:
                            # Find pose with matching similarity score or use highest confidence
                            for pose in comp_poses:
                                if (
                                    hasattr(result, "comparison_pose")
                                    and result.comparison_pose
                                ):
                                    if pose.pose_id == result.comparison_pose.pose_id:
                                        comp_pose = pose
                                        break
                            if not comp_pose and comp_poses:
                                comp_pose = max(
                                    comp_poses, key=lambda p: p.confidence_score
                                )
                        break
                    except Exception as e:
                        if verbose:
                            print(
                                f"Warning: Failed to process {img_path.name} for layering: {e}"
                            )
                        continue

            if comp_pose and comp_img is not None:
                try:
                    # Create layered visualization
                    layered_output_path = (
                        layer_output_dir
                        / f"layer_{i+1}_{Path(result.comparison_image).stem}_on_{Path(target_pose.image_path).stem}.png"
                    )

                    layered_vis = create_layered_pose_visualization(
                        estimator,
                        target_image,
                        target_pose,
                        comp_img,
                        comp_pose,
                        result.similarity_score,
                        str(layered_output_path),
                    )

                    if verbose:
                        print(
                            f"Layered visualization {i+1} saved to: {layered_output_path}"
                        )

                except Exception as e:
                    if verbose:
                        print(
                            f"Warning: Failed to create layered visualization {i+1}: {e}"
                        )

    if verbose:
        viz_time = time.time() - viz_start_time
        print(f"Visualization generation took: {viz_time:.2f} seconds")
    print(f"Visualizations saved to: {output_dir}")


def create_layered_pose_visualization(
    estimator,
    target_image,
    target_pose,
    comparison_image,
    comparison_pose,
    similarity_score,
    output_path,
):
    """
    Create a layered visualization by overlaying a comparison pose on the target image.

    Uses transparency masking to extract the comparison pose and position it correctly
    on the target image using pose alignment.
    """
    try:
        # Create transparent mask of comparison pose
        comp_display, comp_alpha = (
            estimator.create_pose_specific_mask_with_transparency(
                comparison_image, comparison_pose
            )
        )

        # Create RGBA version of comparison image with transparency
        comp_rgba = estimator.create_rgba_image(comparison_image, comp_alpha)

        # Scale and align the comparison pose to the target pose
        # First, get target image dimensions
        target_h, target_w = target_image.shape[:2]
        comp_h, comp_w = comparison_image.shape[:2]

        # Calculate alignment transformation
        # This involves matching the pose centers and scales
        target_center = get_pose_center(target_pose)
        comp_center = get_pose_center(comparison_pose)

        if target_center is None or comp_center is None:
            print("Warning: Could not determine pose centers for alignment")
            # Fallback: just resize to target dimensions
            comp_resized = cv2.resize(comp_rgba, (target_w, target_h))
        else:
            # Calculate scale based on pose dimensions
            target_scale = get_pose_scale(target_pose)
            comp_scale = get_pose_scale(comparison_pose)

            if target_scale > 0 and comp_scale > 0:
                scale_factor = target_scale / comp_scale
            else:
                scale_factor = 1.0

            # Resize comparison image
            new_comp_w = int(comp_w * scale_factor)
            new_comp_h = int(comp_h * scale_factor)
            comp_resized = cv2.resize(comp_rgba, (new_comp_w, new_comp_h))

            # Calculate offset to align pose centers
            comp_center_scaled = (
                comp_center[0] * scale_factor,
                comp_center[1] * scale_factor,
            )
            offset_x = int(target_center[0] - comp_center_scaled[0])
            offset_y = int(target_center[1] - comp_center_scaled[1])

            # Create output canvas
            output_canvas = np.zeros((target_h, target_w, 4), dtype=np.uint8)

            # Place resized comparison image with offset
            start_x = max(0, offset_x)
            start_y = max(0, offset_y)
            end_x = min(target_w, offset_x + new_comp_w)
            end_y = min(target_h, offset_y + new_comp_h)

            comp_start_x = max(0, -offset_x)
            comp_start_y = max(0, -offset_y)
            comp_end_x = comp_start_x + (end_x - start_x)
            comp_end_y = comp_start_y + (end_y - start_y)

            if end_x > start_x and end_y > start_y:
                output_canvas[start_y:end_y, start_x:end_x] = comp_resized[
                    comp_start_y:comp_end_y, comp_start_x:comp_end_x
                ]

            comp_resized = output_canvas

        # Convert target to RGBA
        target_rgba = cv2.cvtColor(target_image, cv2.COLOR_BGR2BGRA)

        # Blend the images using alpha compositing
        alpha = comp_resized[:, :, 3:4].astype(float) / 255.0
        comp_rgb = comp_resized[:, :, :3]

        # Alpha blend: result = alpha * foreground + (1 - alpha) * background
        blended = alpha * comp_rgb + (1 - alpha) * target_rgba[:, :, :3]

        # Create final RGBA image
        final_rgba = target_rgba.copy()
        final_rgba[:, :, :3] = blended.astype(np.uint8)

        # Add text overlay showing similarity score
        cv2.putText(
            final_rgba,
            f"Similarity: {similarity_score:.3f}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (255, 255, 255, 255),
            3,
        )

        # Save as PNG to preserve transparency
        cv2.imwrite(output_path, final_rgba)

        return final_rgba

    except Exception as e:
        print(f"Error creating layered visualization: {e}")
        return None


def get_pose_center(pose):
    """Calculate the center point of a pose from its keypoints."""
    valid_keypoints = [kp for kp in pose.keypoints if kp is not None]
    if not valid_keypoints:
        return None

    x_coords = [kp[0] for kp in valid_keypoints]
    y_coords = [kp[1] for kp in valid_keypoints]

    center_x = sum(x_coords) / len(x_coords)
    center_y = sum(y_coords) / len(y_coords)

    return (center_x, center_y)


def get_pose_scale(pose):
    """Calculate a characteristic scale of the pose."""
    # Use torso keypoints for scale calculation
    torso_indices = [5, 6, 11, 12]  # shoulders and hips
    torso_points = []

    for idx in torso_indices:
        if idx < len(pose.keypoints) and pose.keypoints[idx] is not None:
            kp = pose.keypoints[idx]
            torso_points.append([kp[0], kp[1]])

    if len(torso_points) < 2:
        return 0.0

    # Calculate max distance between torso points as scale
    max_dist = 0.0
    for i in range(len(torso_points)):
        for j in range(i + 1, len(torso_points)):
            dist = np.sqrt(
                (torso_points[i][0] - torso_points[j][0]) ** 2
                + (torso_points[i][1] - torso_points[j][1]) ** 2
            )
            max_dist = max(max_dist, dist)

    return max_dist


def save_results(results, output_path, target_pose):
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


def main():
    """Main application entry point."""
    args = parse_arguments()

    try:
        # Get target and comparison paths
        target_image_path = get_target_image_path(args)
        comparison_dir = Path(args.comparison_dir)
        if not comparison_dir.exists() or not comparison_dir.is_dir():
            print(f"Error: Invalid comparison directory: {args.comparison_dir}")
            sys.exit(1)

        comparison_images = get_comparison_images(comparison_dir)

        # Initialize pose estimator
        if args.verbose:
            print("Initializing YOLOv11 pose estimator...")

        if args.clear_cache:
            from pose_cache import PoseCache

            PoseCache().clear_cache()
            print("Pose cache cleared.")

        start_time = time.time()
        estimator = PoseEstimator(
            confidence_threshold=args.threshold,
            model_size=args.model_size,
            use_cache=not args.no_cache,
        )
        init_time = time.time() - start_time

        if args.verbose:
            print(f"Model initialization took: {init_time:.2f} seconds")

        # Process target image
        if args.verbose:
            print(f"Processing target image: {target_image_path}")

        target_image = load_image(target_image_path)
        start_time = time.time()
        target_poses = estimator.extract_poses(target_image, target_image_path)
        target_time = time.time() - start_time

        if args.verbose:
            print(f"Target pose extraction took: {target_time:.2f} seconds")

        if not target_poses:
            print("Error: No poses detected in target image")
            sys.exit(1)

        # Use highest confidence pose as target
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

        matcher = PoseMatcher()
        results, total_time, pose_extraction_time, pose_matching_time = (
            process_comparison_images(
                estimator,
                matcher,
                target_pose,
                comparison_images,
                args.relative_visibility_threshold,
                args.verbose,
            )
        )

        # Sort and display results
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        print_results(results, args.max_results)

        # Print timing summary
        if args.verbose:
            print_timing_summary(
                init_time,
                target_time,
                total_time,
                pose_extraction_time,
                pose_matching_time,
                len(comparison_images),
            )

        # Generate visualizations
        if args.visualize:
            # Limit results for visualization based on max_results
            limited_results = (
                results[: args.max_results] if args.max_results else results
            )

            generate_visualizations(
                estimator,
                target_image,
                target_pose,
                limited_results,
                comparison_images,
                args.output_dir,
                args.verbose,
                show_skeleton=not args.no_skeleton,
                show_mask=not args.no_mask,
                layer_poses=args.layer_poses,
            )

        # Save results
        if args.output:
            save_results(results, args.output, target_pose)
            print(f"Results saved to: {args.output}")

    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
