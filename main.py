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
import subprocess
import shutil
from pathlib import Path
from typing import List, Optional, Tuple
import cv2
import numpy as np
from tqdm import tqdm

from pose_estimator import PoseEstimator
from pose_matcher import PoseMatcher
from pose_visualizer import PoseVisualizer
from utils.image_utils import load_image, validate_image_path
from utils.pose_utils import PoseData, SimilarityResult


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Poser: Pose Estimation, Compact Binary Database Storage, and Diversity-Constrained Video Art Synthesis"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands: ingest, reconstruct, db-stats (or run legacy CLI)")

    # Ingest subcommand
    ingest_parser = subparsers.add_parser("ingest", help="Ingest video files or image folders into binary SQLite pose database")
    ingest_parser.add_argument("--input-dir", help="Directory of source video files or image folders")
    ingest_parser.add_argument("--video-file", help="Single source video file to ingest")
    ingest_parser.add_argument("--db", default="pose_library.db", help="Path to SQLite database file (default: pose_library.db)")
    ingest_parser.add_argument("--fps", type=float, default=12.0, help="Frame rate sampling interval for ingestion (default: 12.0)")
    ingest_parser.add_argument("--model-size", choices=["n", "s", "m", "l", "x"], default="n", help="YOLOv11 model size (default: n)")

    # Reconstruct subcommand
    recon_parser = subparsers.add_parser("reconstruct", help="Reconstruct a target video clip using matching frames from ingested pose library")
    recon_parser.add_argument("--target", required=True, help="Path to target video file or frame directory")
    recon_parser.add_argument("--db", default="pose_library.db", help="Path to SQLite pose library database")
    recon_parser.add_argument("--output", default="output_art.mp4", help="Path for generated video art output")
    recon_parser.add_argument("--mode", choices=["diversity", "smooth"], default="diversity", help="Matching mode: 'diversity' (switch films) or 'smooth' (continuity)")
    recon_parser.add_argument("--exclude-same-film", action="store_true", help="Do not pick frames from the target clip's source film")
    recon_parser.add_argument("--max-clip-reuse", type=int, default=5, help="Max total seconds pulled from any single source film (default: 5)")
    recon_parser.add_argument("--cooldown", type=int, default=5, help="Frame cooldown before reusing a film in diversity mode (default: 5)")
    recon_parser.add_argument("--temporal-window", type=float, default=2.0, help="Local +/- refinement window in seconds (default: 2.0)")
    recon_parser.add_argument("--model-size", choices=["n", "s", "m", "l", "x"], default="n", help="YOLOv11 model size (default: n)")
    recon_parser.add_argument("--visualize", action="store_true", help="Generate diagnostic visualization frames")
    recon_parser.add_argument("--diagnostic", action="store_true", help="Generate side-by-side diagnostic frames (target left, match right with skeletons and scores)")

    # DB Stats subcommand
    stats_parser = subparsers.add_parser("db-stats", help="Inspect binary SQLite database stats")
    stats_parser.add_argument("--db", default="pose_library.db", help="Path to SQLite database file")

    # Legacy / top-level flags (backwards compatibility)
    parser.add_argument(
        "--target",
        help="Path to target image OR directory of target images (not required if using --video-input)",
    )
    parser.add_argument(
        "--comparison-dir", help="Directory containing comparison images (required unless using --clear-cache only)"
    )
    parser.add_argument(
        "--db", default="pose_library.db", help="Path to SQLite database file"
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
        "--batch-process",
        action="store_true",
        help="Process all images in target directory sequentially for video frame processing",
    )
    parser.add_argument(
        "--video-input",
        help="Input video file path (automatically extracts frames for processing)",
    )
    parser.add_argument(
        "--video-output",
        help="Output video file path (automatically combines processed frames into video)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Frame rate for video input/output (default: 30.0)",
    )
    parser.add_argument(
        "--cleanup-frames",
        action="store_true",
        help="Delete temporary frame files after video processing (default: keep frames)",
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


def get_target_images_for_batch(args) -> List[str]:
    """Get all target image paths for batch processing."""
    target_path = Path(args.target)

    if target_path.is_file():
        if not validate_image_path(args.target):
            print(f"Error: Invalid target image path: {args.target}")
            sys.exit(1)
        return [args.target]

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

        # Sort images naturally (so frame001.jpg comes before frame010.jpg)
        target_images.sort(key=lambda x: x.name)

        return [str(img) for img in target_images]

    else:
        print(f"Error: Target path does not exist: {args.target}")
        sys.exit(1)


def get_comparison_images(comparison_dir: Path) -> List[Path]:
    """Get list of comparison image paths, recursively searching nested folders."""
    # Recursively find all image files in comparison directory and subdirectories
    comparison_images = (
        list(comparison_dir.rglob("*.jpg"))
        + list(comparison_dir.rglob("*.jpeg"))
        + list(comparison_dir.rglob("*.png"))
        + list(comparison_dir.rglob("*.webp"))
    )
    
    if not comparison_images:
        print(f"Error: No image files found in comparison directory: {comparison_dir}")
        print("Tip: Checked all subdirectories recursively for .jpg, .jpeg, .png, .webp files")
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


def select_pose_matching_bbox(
    estimator, image, stored_bbox, image_path: str, min_iou: float = 0.3
):
    """
    Find the person in a retrieved source frame that matches an ingested pose.

    Args:
        estimator: PoseEstimator used to re-detect poses on the frame.
        image: Retrieved source frame array.
        stored_bbox: (x1, y1, x2, y2) bounding box recorded at ingest time.
        image_path: Path label passed through to the estimator.
        min_iou: Minimum overlap required to accept a detection.

    Returns:
        Matching PoseData, or None if the frame holds no corresponding person.
    """
    try:
        poses = estimator.extract_poses(image, image_path)
    except Exception:
        return None

    if not poses:
        return None

    if not stored_bbox:
        return max(poses, key=lambda p: p.confidence_score)

    sx1, sy1, sx2, sy2 = stored_bbox
    stored_area = max(1.0, (sx2 - sx1) * (sy2 - sy1))

    best_pose = None
    best_iou = 0.0
    for pose in poses:
        if not pose.bounding_box:
            continue
        px1, py1, px2, py2 = pose.bounding_box
        ix1, iy1 = max(sx1, px1), max(sy1, py1)
        ix2, iy2 = min(sx2, px2), min(sy2, py2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        pose_area = max(1.0, (px2 - px1) * (py2 - py1))
        iou = inter / (stored_area + pose_area - inter)
        if iou > best_iou:
            best_iou = iou
            best_pose = pose

    return best_pose if best_iou >= min_iou else None


def check_ffmpeg_available() -> bool:
    """Check if ffmpeg is available on the system."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def extract_frames_from_video(
    video_path: str, output_dir: str, fps: Optional[float] = None, verbose: bool = False
) -> Tuple[bool, int]:
    """
    Extract frames from video using ffmpeg.

    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted frames
        fps: Frame rate to extract (None = use video's native fps)
        verbose: Enable verbose output

    Returns:
        Tuple of (success, frame_count)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Build ffmpeg command
    cmd = ["ffmpeg", "-y", "-i", video_path]

    if fps is not None:
        cmd.extend(["-vf", f"fps={fps}"])

    frame_pattern = str(output_path / "frame_%04d.jpg")
    cmd.append(frame_pattern)

    if verbose:
        print(f" Extracting frames from video: {Path(video_path).name}")
        print(f" Output directory: {output_path}")
        if fps:
            print(f" Target FPS: {fps}")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300  # 5 minute timeout
        )

        if result.returncode != 0:
            print(f" FFmpeg error: {result.stderr}")
            return False, 0

        # Count extracted frames
        frame_files = list(output_path.glob("frame_*.jpg"))
        frame_count = len(frame_files)

        if verbose:
            print(f" Extracted {frame_count} frames successfully")

        return True, frame_count

    except subprocess.TimeoutExpired:
        print(" Video extraction timed out (>5 minutes)")
        return False, 0
    except Exception as e:
        print(f" Error extracting frames: {e}")
        return False, 0


def create_video_from_frames(
    frames_dir: str,
    output_video: str,
    fps: float = 30.0,
    frame_pattern: str = "frame_%04d_*.png",
    verbose: bool = False,
) -> bool:
    """
    Create video from processed frames using ffmpeg.

    Args:
        frames_dir: Directory containing processed frame images
        output_video: Path for output video file
        fps: Frame rate for output video
        frame_pattern: Pattern to match frame files
        verbose: Enable verbose output

    Returns:
        True if successful, False otherwise
    """
    frames_path = Path(frames_dir)

    if not frames_path.exists():
        print(f" Frames directory not found: {frames_dir}")
        return False

    # Get all frame files and sort them numerically
    frame_files = list(frames_path.glob("frame_*.png"))
    if not frame_files:
        print(f" No frame files found in {frames_dir}")
        return False

    # Sort by frame number extracted from filename
    def get_frame_number(filename):
        # Extract frame number from "frame_XXXX_something.png"
        try:
            parts = filename.stem.split("_")
            return int(parts[1])  # frame_0001_something -> 0001
        except (IndexError, ValueError):
            return 0

    frame_files.sort(key=get_frame_number)

    # Create a temporary directory with sequential frame names for FFmpeg
    temp_frames_dir = frames_path / "temp_sequential"
    temp_frames_dir.mkdir(exist_ok=True)

    try:
        # Create symbolic links with sequential names
        for i, frame_file in enumerate(frame_files, 1):
            temp_link = temp_frames_dir / f"frame_{i:04d}.png"
            if temp_link.exists():
                temp_link.unlink()
            temp_link.symlink_to(frame_file.resolve())

        # Build ffmpeg command with sequential pattern
        input_pattern = str(temp_frames_dir / "frame_%04d.png")
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            input_pattern,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",  # High quality
            output_video,
        ]

        if verbose:
            print(f" Creating video from {len(frame_files)} frames")
            print(f" Input pattern: {input_pattern}")
            print(f" Output video: {output_video}")
            print(f" FPS: {fps}")

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600  # 10 minute timeout
        )

        # Cleanup temporary directory
        import shutil

        shutil.rmtree(temp_frames_dir, ignore_errors=True)

        if result.returncode != 0:
            print(f" FFmpeg error creating video: {result.stderr}")
            return False

        if verbose:
            print(f" Video created successfully: {output_video}")

        return True

    except subprocess.TimeoutExpired:
        # Cleanup on timeout
        import shutil

        shutil.rmtree(temp_frames_dir, ignore_errors=True)
        print(" Video creation timed out (>10 minutes)")
        return False
    except Exception as e:
        # Cleanup on error
        import shutil

        shutil.rmtree(temp_frames_dir, ignore_errors=True)
        print(f" Error creating video: {e}")
        return False


def cleanup_temporary_frames(frames_dir: str, verbose: bool = False) -> bool:
    """
    Clean up temporary frame files.

    Args:
        frames_dir: Directory containing temporary frames
        verbose: Enable verbose output

    Returns:
        True if successful, False otherwise
    """
    frames_path = Path(frames_dir)

    if not frames_path.exists():
        return True  # Already clean

    try:
        # Only remove frame files, not the entire directory
        frame_patterns = ["frame_*.jpg", "frame_*.png"]
        removed_count = 0

        for pattern in frame_patterns:
            for frame_file in frames_path.glob(pattern):
                frame_file.unlink()
                removed_count += 1

        if verbose and removed_count > 0:
            print(f" Cleaned up {removed_count} temporary frame files")

        return True

    except Exception as e:
        print(f"️  Warning: Failed to cleanup frames: {e}")
        return False


def process_single_target(
    target_image_path, estimator, matcher, comparison_images, args, frame_number=None
):
    """Process a single target image and return results."""
    if args.verbose:
        frame_prefix = f"Frame {frame_number:04d}: " if frame_number is not None else ""
        print(f"{frame_prefix}Processing target image: {Path(target_image_path).name}")

    target_image = load_image(target_image_path)
    start_time = time.time()
    target_poses = estimator.extract_poses(target_image, target_image_path)
    target_time = time.time() - start_time

    if args.verbose:
        print(f"Target pose extraction took: {target_time:.2f} seconds")

    if not target_poses:
        if args.verbose:
            print("Warning: No poses detected in target image")
        return None, None, None, target_time

    # Use highest confidence pose as target
    target_pose = max(target_poses, key=lambda p: p.confidence_score)
    if args.verbose:
        if len(target_poses) > 1:
            print(
                f"Found {len(target_poses)} people in target image, using highest confidence person"
            )

    # Process comparison images using the normal flow (which uses cache automatically)
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

    # Sort results
    results.sort(key=lambda x: x.similarity_score, reverse=True)

    return target_image, target_pose, results, target_time


def process_single_target_fast(
    target_image_path,
    estimator,
    matcher,
    comparison_poses_data,
    args,
    frame_number=None,
):
    """Process a single target image using pre-loaded comparison poses."""
    if args.verbose:
        frame_prefix = f"Frame {frame_number:04d}: " if frame_number is not None else ""
        print(f"{frame_prefix}Processing target image: {Path(target_image_path).name}")

    target_image = load_image(target_image_path)
    start_time = time.time()
    target_poses = estimator.extract_poses(target_image, target_image_path)
    target_time = time.time() - start_time

    if args.verbose:
        print(f"Target pose extraction took: {target_time:.2f} seconds")

    if not target_poses:
        if args.verbose:
            print("Warning: No poses detected in target image")
        return None, None, None, target_time

    # Use highest confidence pose as target
    target_pose = max(target_poses, key=lambda p: p.confidence_score)
    if args.verbose:
        if len(target_poses) > 1:
            print(
                f"Found {len(target_poses)} people in target image, using highest confidence person"
            )

    # Fast comparison using pre-loaded poses (less verbose for speed)
    # Skip verbose output during matching for better performance

    results = []
    start_time = time.time()

    for img_path, pose_data in comparison_poses_data.items():
        comparison_poses = pose_data["poses"]

        if comparison_poses:
            best_match = matcher.find_best_match(
                target_pose, comparison_poses, args.relative_visibility_threshold
            )

            if best_match:
                results.append(best_match)

    match_time = time.time() - start_time

    # Only show timing for verbose mode and only occasionally for performance
    if args.verbose and frame_number and frame_number % 50 == 1:
        print(f"   Frame {frame_number}: Pose matching took: {match_time:.2f} seconds")

    # Sort results
    results.sort(key=lambda x: x.similarity_score, reverse=True)

    return target_image, target_pose, results, target_time


def process_video_workflow(args):
    """Process video input and output workflow."""
    # Check ffmpeg availability
    if not check_ffmpeg_available():
        print(" Error: FFmpeg is required for video processing but not found.")
        print(" Install FFmpeg: https://ffmpeg.org/download.html")
        print("   macOS: brew install ffmpeg")
        print("   Ubuntu: sudo apt install ffmpeg")
        print("   Windows: Download from https://ffmpeg.org/")
        sys.exit(1)

    # Validate video input
    video_input = Path(args.video_input)
    if not video_input.exists():
        print(f" Error: Video input file not found: {args.video_input}")
        sys.exit(1)

    # Setup temporary frame directory
    temp_frames_dir = Path("data/input_frames")
    temp_frames_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n VIDEO PROCESSING WORKFLOW")
    print(f" Input video: {video_input.name}")
    print(f" Comparison images: {args.comparison_dir}")
    if args.video_output:
        print(f" Output video: {args.video_output}")
    print(f" FPS: {args.fps}")
    print("=" * 50)

    try:
        # Step 1: Extract frames from video
        success, frame_count = extract_frames_from_video(
            str(video_input),
            str(temp_frames_dir),
            fps=(
                args.fps if args.fps != 30.0 else None
            ),  # Use video's native fps if default
            verbose=args.verbose,
        )

        if not success or frame_count == 0:
            print(" Failed to extract frames from video")
            sys.exit(1)

        # Step 2: Set up arguments for batch processing
        # Temporarily override target to use extracted frames
        original_target = args.target
        args.target = str(temp_frames_dir)
        args.batch_process = True
        args.layer_poses = True

        # Step 3: Process frames with pose matching
        print(f"\n️ Processing {frame_count} extracted frames...")
        process_batch_targets(args)

        # Step 4: Create output video if requested
        if args.video_output:
            layer_output_dir = Path(args.output_dir) / "batch_layered_poses"

            success = create_video_from_frames(
                str(layer_output_dir),
                args.video_output,
                fps=args.fps,
                frame_pattern="frame_%04d_*.png",
                verbose=args.verbose,
            )

            if success:
                print(f"\n Video processing complete!")
                print(f" Output video: {args.video_output}")
            else:
                print(f"\n️  Frame processing completed but video creation failed")
                print(f" Processed frames available in: {layer_output_dir}")

        # Step 5: Cleanup temporary frames if requested
        if args.cleanup_frames:
            cleanup_temporary_frames(str(temp_frames_dir), args.verbose)
        else:
            print(f"\n Extracted frames saved in: {temp_frames_dir}")
            print(f"   Use --cleanup-frames to auto-delete these files")

        # Restore original target
        args.target = original_target

    except Exception as e:
        print(f" Error in video processing workflow: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


def process_batch_targets(args):
    """Process all target images in batch mode for video frame processing."""
    # Get all target images
    target_images = get_target_images_for_batch(args)
    comparison_dir = Path(args.comparison_dir)

    if not comparison_dir.exists() or not comparison_dir.is_dir():
        print(f"Error: Invalid comparison directory: {args.comparison_dir}")
        sys.exit(1)

    comparison_images = get_comparison_images(comparison_dir)
    
    # Show organization info in verbose mode
    if args.verbose:
        # Find unique subdirectories that contain images
        subdirs = set()
        for img_path in comparison_images:
            relative_path = img_path.relative_to(comparison_dir)
            if relative_path.parent != Path('.'):
                subdirs.add(str(relative_path.parent))
        
        if subdirs:
            print(f" Found images in {len(subdirs)} subdirectories:")
            for subdir in sorted(subdirs):
                subdir_images = [img for img in comparison_images 
                               if str(img.relative_to(comparison_dir)).startswith(subdir)]
                print(f"    {subdir}: {len(subdir_images)} images")
        else:
            print(" All comparison images are in the root directory")

    print(f"\n BATCH PROCESSING MODE")
    print(f" Processing {len(target_images)} target frames")
    print(f" Using {len(comparison_images)} comparison images")
    print(f" Output directory: {args.output_dir}")
    print("=" * 50)

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
        verbose=args.verbose,
    )
    init_time = time.time() - start_time

    if args.verbose:
        print(f"Model initialization took: {init_time:.2f} seconds")

    # Pre-load all comparison poses efficiently using cache
    if args.verbose:
        print(f"\n Loading poses from {len(comparison_images)} comparison images...")
        print(
            f" Cache database contains {len(estimator.cache.cache) if estimator.cache else 0} cached images"
        )

    comparison_poses_data = {}
    cache_hits = 0
    cache_misses = 0
    start_time = time.time()

    # Optimize database loading by separating cached vs uncached
    if args.verbose:
        print(" Optimizing comparison database loading...")

    # Pre-check which images are cached vs need extraction
    cached_paths = []
    uncached_paths = []

    check_start = time.time()
    for img_path in comparison_images:
        if estimator.cache and estimator.cache.is_cached(str(img_path)):
            cached_paths.append(img_path)
        else:
            uncached_paths.append(img_path)

    if args.verbose:
        check_time = time.time() - check_start
        print(
            f" Cache check ({check_time:.1f}s): {len(cached_paths)} cached, {len(uncached_paths)} need extraction"
        )

    # Batch process cached images (should be very fast)
    if args.verbose:
        cached_iterator = enumerate(cached_paths, 1)
    else:
        cached_iterator = enumerate(
            tqdm(
                cached_paths, desc=" Loading cached poses", unit="pose", leave=False
            ),
            1,
        )

    for i, img_path in cached_iterator:
        if args.verbose and i % 2000 == 0:  # Less frequent for cached items
            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 else 0
            print(f"   Loaded {i}/{len(cached_paths)} cached poses ({rate:.0f}/sec)")

        try:
            poses = estimator.cache.get_cached_poses(str(img_path))
            comparison_poses_data[str(img_path)] = {
                "poses": poses,
                "image_path": str(img_path),
            }
            cache_hits += 1
        except Exception as e:
            if args.verbose:
                print(
                    f"   Warning: Failed to load cached poses from {img_path.name}: {e}"
                )
            comparison_poses_data[str(img_path)] = {
                "poses": [],
                "image_path": str(img_path),
            }

    # Process uncached images (slower but necessary)
    if args.verbose:
        uncached_iterator = enumerate(uncached_paths, 1)
    else:
        uncached_iterator = enumerate(
            tqdm(
                uncached_paths, desc=" Extracting new poses", unit="pose", leave=False
            ),
            1,
        )

    for i, img_path in uncached_iterator:
        if args.verbose:
            print(f"   Extracting {i}/{len(uncached_paths)}: {img_path.name}")

        try:
            img_array = load_image(str(img_path))
            poses = estimator.extract_poses(img_array, str(img_path))
            comparison_poses_data[str(img_path)] = {
                "poses": poses,
                "image_path": str(img_path),
            }
            cache_misses += 1
        except Exception as e:
            if args.verbose:
                print(f"   Warning: Failed to extract poses from {img_path.name}: {e}")
            comparison_poses_data[str(img_path)] = {
                "poses": [],
                "image_path": str(img_path),
            }

    if args.verbose:
        total_poses = sum(len(data["poses"]) for data in comparison_poses_data.values())
        elapsed = time.time() - start_time
        rate = len(comparison_images) / elapsed if elapsed > 0 else 0
        print(
            f" Loaded {total_poses} poses from {len(comparison_images)} images in {elapsed:.1f}s ({rate:.0f}/sec)"
        )
        print(f" Cache stats: {cache_hits} hits, {cache_misses} misses")

    matcher = PoseMatcher()

    # Create output directory structure
    output_dir = Path(args.output_dir)
    layer_output_dir = output_dir / "batch_layered_poses"

    # Clean up previous batch results to avoid conflicts
    if layer_output_dir.exists():
        if args.verbose:
            old_frames = list(layer_output_dir.glob("frame_*.png"))
            if old_frames:
                print(f" Cleaning up {len(old_frames)} old processed frames...")

        # Remove old frame files but keep directory
        for old_frame in layer_output_dir.glob("frame_*.png"):
            old_frame.unlink()

        # Also clean up any temporary sequential directories
        temp_seq_dir = layer_output_dir / "temp_sequential"
        if temp_seq_dir.exists():
            import shutil

            shutil.rmtree(temp_seq_dir, ignore_errors=True)

    layer_output_dir.mkdir(parents=True, exist_ok=True)

    batch_start_time = time.time()
    successful_frames = 0
    failed_frames = 0

    # Process each target image
    if args.verbose:
        # Verbose mode: show detailed progress for each frame
        frame_iterator = enumerate(target_images, 1)
    else:
        # Non-verbose mode: show progress bar
        frame_iterator = enumerate(
            tqdm(
                target_images,
                desc="️ Processing frames",
                unit="frame",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            ),
            1,
        )

    for frame_idx, target_image_path in frame_iterator:
        # Progress indicator for verbose mode only
        if args.verbose:
            print(
                f"\n️ Processing frame {frame_idx:04d}/{len(target_images):04d}: {Path(target_image_path).name}"
            )

        try:
            target_image, target_pose, results, target_time = (
                process_single_target_fast(
                    target_image_path,
                    estimator,
                    matcher,
                    comparison_poses_data,  # Use pre-loaded poses data
                    args,
                    frame_idx,
                )
            )

            if target_pose is None or not results:
                # Instead of skipping, use the original frame
                if args.verbose:
                    print(
                        f" Frame {frame_idx:04d}: No pose/matches found, using original frame"
                    )

                # Create output with original frame
                if args.layer_poses:
                    try:
                        # Load the original target image
                        original_image = load_image(target_image_path)

                        # Save as original frame without pose overlay
                        frame_output_path = (
                            layer_output_dir / f"frame_{frame_idx:04d}_original.png"
                        )

                        # Convert BGR to RGBA and save
                        import cv2

                        original_rgba = cv2.cvtColor(original_image, cv2.COLOR_BGR2BGRA)
                        cv2.imwrite(str(frame_output_path), original_rgba)

                        if args.verbose:
                            print(
                                f" Frame {frame_idx:04d} saved as original: {frame_output_path.name}"
                            )
                        successful_frames += 1

                    except Exception as e:
                        print(f" Error saving original frame {frame_idx:04d}: {e}")
                        failed_frames += 1
                else:
                    failed_frames += 1
                continue

            # Generate layered pose for the best match
            if results and args.layer_poses:
                best_result = results[0]

                # Get the comparison image and pose from pre-loaded data
                comp_img = None
                comp_pose = None

                pose_data = comparison_poses_data.get(best_result.comparison_image)
                if pose_data and pose_data["poses"]:
                    # Load image on-demand for layered visualization
                    try:
                        comp_img = load_image(pose_data["image_path"])
                    except Exception as e:
                        if args.verbose:
                            print(f"Warning: Failed to load comparison image: {e}")
                        continue

                    # Find the specific pose that gave this result
                    if (
                        hasattr(best_result, "comparison_pose")
                        and best_result.comparison_pose
                    ):
                        comp_pose = best_result.comparison_pose
                    else:
                        # Fallback to highest confidence pose
                        comp_pose = max(
                            pose_data["poses"], key=lambda p: p.confidence_score
                        )

                if comp_pose and comp_img is not None:
                    # Create sequential frame output name
                    frame_output_path = (
                        layer_output_dir
                        / f"frame_{frame_idx:04d}_{Path(best_result.comparison_image).stem}.png"
                    )

                    try:
                        layered_vis = create_layered_pose_visualization(
                            estimator,
                            target_image,
                            target_pose,
                            comp_img,
                            comp_pose,
                            best_result.similarity_score,
                            str(frame_output_path),
                        )

                        if layered_vis is not None:
                            if args.verbose:
                                print(
                                    f" Frame {frame_idx:04d} saved: {frame_output_path.name} (similarity: {best_result.similarity_score:.3f})"
                                )
                            successful_frames += 1
                        else:
                            if args.verbose:
                                print(
                                    f" Failed to create layered visualization for frame {frame_idx:04d}"
                                )
                            failed_frames += 1

                    except Exception as e:
                        if args.verbose:
                            print(
                                f" Error creating layered visualization for frame {frame_idx:04d}: {e}"
                            )
                        failed_frames += 1
                else:
                    if args.verbose:
                        print(
                            f"️  No suitable comparison pose found for frame {frame_idx:04d}"
                        )
                    failed_frames += 1
            else:
                if args.verbose:
                    print(f"️  No results found for frame {frame_idx:04d}")
                failed_frames += 1

        except Exception as e:
            if args.verbose:
                print(f" Error processing frame {frame_idx:04d}: {e}")
                import traceback

                traceback.print_exc()
            failed_frames += 1

    # Print batch summary
    total_time = time.time() - batch_start_time
    print(f"\n BATCH PROCESSING COMPLETE")
    print(f"=" * 50)
    print(f" Successful frames: {successful_frames}")
    print(f" Failed frames: {failed_frames}")
    print(f"️  Total time: {total_time:.2f} seconds")
    print(f" Average time per frame: {total_time/len(target_images):.2f} seconds")
    print(f" Output saved to: {layer_output_dir}")
    print(
        f" To create video: ffmpeg -framerate 30 -i {layer_output_dir}/frame_%04d_*.png -c:v libx264 -pix_fmt yuv420p output_video.mp4"
    )


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


def handle_db_stats(args):
    """Display SQLite database statistics."""
    from pose_db import PoseDatabase
    db = PoseDatabase(args.db)
    stats = db.get_stats()
    print("\n=== POSE DATABASE STATISTICS ===")
    print(f"Database file:  {stats['db_path']}")
    print(f"Total films:    {stats['total_films']}")
    print(f"Total poses:    {stats['total_poses']}")
    print(f"File size:      {stats['db_size_mb']} MB ({stats['db_size_bytes']} bytes)")
    print("=" * 35)


def handle_ingest(args):
    """Ingest source videos or frames into SQLite binary pose database."""
    from pose_db import PoseDatabase
    from pose_estimator import PoseEstimator
    from utils.image_utils import load_image, natural_frame_key

    if not args.input_dir and not args.video_file:
        print("Error: Specify --input-dir or --video-file for ingestion.")
        sys.exit(1)

    db = PoseDatabase(args.db)
    estimator = PoseEstimator(model_size=args.model_size)

    media_files = []
    if args.video_file:
        media_files.append(Path(args.video_file))
    if args.input_dir:
        input_path = Path(args.input_dir).expanduser()
        valid_exts = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".flv"}
        for f in input_path.rglob("*"):
            if f.is_file() and f.suffix.lower() in valid_exts:
                media_files.append(f)

    if not media_files:
        print("No video files found to ingest.")
        return

    print(f"Found {len(media_files)} video file(s) for ingestion.")
    for vid_path in media_files:
        print(f"Ingesting: {vid_path.name}")
        film_id = db.register_film(title=vid_path.stem, filepath=str(vid_path), fps=args.fps)

        # Extract frames to temp directory
        temp_dir = Path("data/temp_ingest") / vid_path.stem
        success, frame_count = extract_frames_from_video(str(vid_path), str(temp_dir), fps=args.fps)
        if not success:
            continue

        # Sort numerically, not lexicographically: ffmpeg's frame_%04d pattern
        # widens past 9999, and a plain sort would scramble every index beyond
        # that point, permanently corrupting the frame->timestamp mapping.
        frame_files = sorted(temp_dir.glob("frame_*.jpg"), key=natural_frame_key)
        pose_records = []

        skipped_no_person = 0
        for idx, fpath in enumerate(tqdm(frame_files, desc=f"Processing {vid_path.stem}")):
            img = load_image(str(fpath))
            poses = estimator.extract_poses(img, str(fpath))
            if poses:
                best_pose = max(poses, key=lambda p: p.confidence_score)
                # Enforce torso keypoint validation: left_shoulder(5), right_shoulder(6), left_hip(11), right_hip(12)
                kps = best_pose.keypoints
                if len(kps) == 17 and all(kps[i] is not None for i in [5, 6, 11, 12]):
                    # Person verification: confirm YOLO segmentation detects Class 0 (person) in this frame
                    has_person = False
                    try:
                        seg_results = estimator.segmentation_model(img, verbose=False)
                        for r in seg_results:
                            if r.boxes is not None and len(r.boxes.cls) > 0:
                                for i_cls, cls_id in enumerate(r.boxes.cls):
                                    if int(cls_id) == 0 and float(r.boxes.conf[i_cls]) >= 0.40:
                                        has_person = True
                                        break
                            if has_person:
                                break
                    except Exception:
                        has_person = True  # Fallback: allow if segmentation fails
                    
                    if not has_person:
                        skipped_no_person += 1
                        continue
                    
                    pose_records.append({
                        "frame_idx": idx,
                        "timestamp": idx / args.fps,
                        "bbox": best_pose.bounding_box,
                        "confidence": best_pose.confidence_score,
                        "keypoints": best_pose.keypoints
                    })

        db.add_poses_batch(film_id, pose_records)
        print(f"Successfully ingested {len(pose_records)} poses for {vid_path.stem}")
        if skipped_no_person > 0:
            print(f"  ⚠️  Skipped {skipped_no_person} frames (no person detected by segmentation model)")

        # Cleanup temp frames
        shutil.rmtree(temp_dir, ignore_errors=True)

    handle_db_stats(args)


def handle_reconstruct(args):
    """Reconstruct a target video clip using matching frames from SQLite pose database."""
    from pose_db import PoseDatabase
    from pose_estimator import PoseEstimator
    from pose_matcher import PoseMatcher
    from utils.image_utils import load_image, natural_frame_key

    db = PoseDatabase(args.db)
    stats = db.get_stats()
    if stats["total_poses"] == 0:
        print(f"Error: Database '{args.db}' contains 0 poses. Ingest films first using `python3 main.py ingest --input-dir /path/to/movies`.")
        sys.exit(1)

    estimator = PoseEstimator(model_size=args.model_size)
    matcher = PoseMatcher()

    target_path = Path(args.target)
    target_frames_dir = Path("results/temp_reconstruct_target")
    
    if target_path.is_file():
        print(f"Extracting target video frames: {target_path.name}")
        extract_frames_from_video(str(target_path), str(target_frames_dir), fps=30.0)
        target_frame_files = sorted(target_frames_dir.glob("frame_*.jpg"), key=natural_frame_key)
    else:
        target_frame_files = sorted(
            list(target_path.glob("*.jpg")) + list(target_path.glob("*.png")),
            key=natural_frame_key,
        )

    if not target_frame_files:
        print(f"Error: No target frames found at {args.target}")
        sys.exit(1)

    print(f"Processing target trajectory ({len(target_frame_files)} frames)...")
    target_vectors = []
    target_poses = []
    
    for fpath in tqdm(target_frame_files, desc="Extracting target trajectory"):
        img = load_image(str(fpath))
        poses = estimator.extract_poses(img, str(fpath))
        if poses:
            best_pose = max(poses, key=lambda p: p.confidence_score)
            vec = PoseDatabase.normalize_keypoints_to_vector(best_pose.keypoints)
            target_vectors.append(vec)
            target_poses.append((fpath, img, best_pose))
        else:
            target_vectors.append(None)
            target_poses.append((fpath, img, None))

    target_film_title = target_path.stem
    print(f"Running sequence matching (Mode: {args.mode}, Exclude target film: {args.exclude_same_film})...")
    matches = matcher.match_db_sequence(
        target_vectors=target_vectors,
        pose_db=db,
        mode=args.mode,
        exclude_same_film=args.exclude_same_film,
        max_clip_reuse=args.max_clip_reuse,
        cooldown=args.cooldown,
        target_film_title=target_film_title,
        top_k_candidates=20
    )

    # Render output frames
    render_dir = Path("results/reconstruct_render")
    render_dir.mkdir(parents=True, exist_ok=True)
    
    visualizer = PoseVisualizer()
    print("Rendering composite video art frames...")
    
    last_valid_vis = None
    composited_frames = 0
    no_target_pose = 0
    films_used = {}
    for idx, (match_entry, (target_fpath, target_img, target_pose)) in enumerate(
        tqdm(list(zip(matches, target_poses)), desc="Compositing frames")
    ):
        out_frame_path = render_dir / f"frame_{idx+1:04d}.png"

        if target_pose is None:
            no_target_pose += 1
        
        cand_list = match_entry if isinstance(match_entry, list) else ([match_entry] if match_entry else [])
        winning_match = None
        comp_vis = None

        # Walk the ranked candidates until one composites cleanly. A match can
        # fail late - the seek may land on a cut, or the person may resist
        # segmentation - so ranking alone is not enough to guarantee a frame.
        for cand in cand_list:
            if not cand or not cand.get("film_path") or not Path(cand["film_path"]).exists():
                continue
            source_frame_path = Path(cand["film_path"])
            if source_frame_path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}:
                # Seek by timestamp: frame_idx counts resampled ingest frames, so
                # using it as a native frame number lands on the wrong moment.
                from utils.image_utils import extract_frame_at_timestamp
                s_img = extract_frame_at_timestamp(str(source_frame_path), cand["timestamp"])
            else:
                s_img = load_image(str(source_frame_path))

            if s_img is None:
                continue

            # Re-detect on the retrieved frame and keep the person whose box best
            # overlaps the ingested one. This both verifies the seek landed on the
            # right shot and recovers pixel-space keypoints for pose alignment.
            # The identity must name the *source* frame: the pose cache keys on it,
            # so reusing the output path would serve every candidate the first
            # candidate's detection.
            source_identity = f"{source_frame_path}@{cand['timestamp']:.4f}"
            s_pose = select_pose_matching_bbox(estimator, s_img, cand.get("bbox"), source_identity)
            if s_pose is None:
                continue

            try:
                if getattr(args, "diagnostic", False):
                    # Side-by-side diagnostic comparison view (target left with skeleton, match right with skeleton & score)
                    candidate_vis = visualizer.create_side_by_side_diagnostic(
                        target_image=target_img,
                        target_pose=target_pose,
                        source_image=s_img,
                        source_pose=s_pose,
                        film_title=cand["film_title"],
                        frame_idx=cand["frame_idx"],
                        similarity_score=cand["similarity_score"],
                        frame_num=idx + 1
                    )
                elif getattr(args, "visualize", False) and target_pose is not None:
                    candidate_vis = visualizer.create_winning_pose_overlay(
                        target_img,
                        target_pose,
                        s_pose,
                        cand["similarity_score"],
                        winning_image=s_img
                    )
                else:
                    # Segment person from source video frame and composite directly onto target frame
                    candidate_vis = visualizer.create_person_cutout_composite(
                        target_image=target_img,
                        target_pose=target_pose,
                        source_image=s_img,
                        source_bbox=s_pose.bounding_box or cand["bbox"],
                        segmentation_model=estimator.segmentation_model,
                        source_pose=s_pose,
                    )
            except Exception as e:
                print(f"Warning rendering frame {idx+1}: {e}")
                continue

            if candidate_vis is not None:
                comp_vis = candidate_vis
                winning_match = cand
                break

        if comp_vis is not None:
            composited_frames += 1
            films_used[winning_match["film_title"]] = films_used.get(winning_match["film_title"], 0) + 1
        else:
            if target_img is not None:
                # Fall back to the untouched target frame, at its own resolution
                comp_vis = target_img
            elif last_valid_vis is not None:
                comp_vis = last_valid_vis

        if comp_vis is not None:
            cv2.imwrite(str(out_frame_path), comp_vis)
            last_valid_vis = comp_vis

    total = len(target_poses)
    print(
        f"\nComposited {composited_frames}/{total} frames "
        f"({composited_frames / max(1, total):.0%}); "
        f"{no_target_pose} frames had no detectable target pose."
    )
    if films_used:
        top = sorted(films_used.items(), key=lambda kv: -kv[1])
        print(f"Source films used ({len(films_used)}):")
        for title, count in top:
            print(f"  {count:>4} frames  {title[:64]}")

    # Combine frames into output video
    print(f"Encoding output video: {args.output}")
    create_video_from_frames(str(render_dir), args.output, fps=30.0)
    print(f" Video art synthesis complete! Saved to {args.output}")

    # Cleanup temp directories and any video handles held open for seeking
    from utils.image_utils import release_capture_cache
    release_capture_cache()
    shutil.rmtree(target_frames_dir, ignore_errors=True)


def main():
    """Main application entry point."""
    args = parse_arguments()

    if args.command == "db-stats":
        handle_db_stats(args)
        return
    elif args.command == "ingest":
        handle_ingest(args)
        return
    elif args.command == "reconstruct":
        handle_reconstruct(args)
        return

    # Validate required arguments for legacy mode
    if args.clear_cache and not args.target and not args.video_input:
        from pose_cache import PoseCache
        PoseCache().clear_cache()
        print(" Pose cache cleared.")
        sys.exit(0)

    if not args.target and not args.video_input:
        print(" Error: Either --target or --video-input must be specified")
        print("   Use --target for image/directory processing")
        print("   Use --video-input for video processing")
        sys.exit(1)

    # Validate comparison-dir is provided when not just clearing cache
    if not args.comparison_dir:
        print(" Error: --comparison-dir is required")
        sys.exit(1)

    try:
        # Check if video processing mode is enabled
        if args.video_input:
            if not args.comparison_dir:
                print(" Error: --comparison-dir is required for video processing")
                sys.exit(1)
            process_video_workflow(args)
            return

        # Check if batch processing mode is enabled
        if args.batch_process:
            args.layer_poses = True
            process_batch_targets(args)
            return

        # Single target image legacy flow
        target_image_path = get_target_image_path(args)
        comparison_dir = Path(args.comparison_dir)
        comparison_images = get_comparison_images(comparison_dir)

        estimator = PoseEstimator(
            confidence_threshold=args.threshold,
            model_size=args.model_size,
            use_cache=not args.no_cache,
            verbose=args.verbose,
        )

        target_image = load_image(target_image_path)
        target_poses = estimator.extract_poses(target_image, target_image_path)
        if not target_poses:
            print("Error: No poses detected in target image")
            sys.exit(1)

        target_pose = max(target_poses, key=lambda p: p.confidence_score)
        matcher = PoseMatcher()
        results, _, _, _ = process_comparison_images(
            estimator, matcher, target_pose, comparison_images, args.relative_visibility_threshold, args.verbose
        )

        results.sort(key=lambda x: x.similarity_score, reverse=True)
        print_results(results, args.max_results)

        if args.visualize:
            limited_results = results[: args.max_results] if args.max_results else results
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
