#!/usr/bin/env python3
"""
Pose Visualizer
Creates diagnostic images showing detected poses, keypoints, and similarity scores.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional

from utils.pose_utils import PoseData, SimilarityResult


class PoseVisualizer:
    """Visualizes poses, keypoints, and similarity scores on images."""

    def __init__(self):
        # COCO keypoint connections for drawing skeleton
        self.keypoint_connections = [
            (0, 1),
            (0, 2),
            (1, 3),
            (2, 4),  # head
            (5, 6),
            (5, 7),
            (6, 8),
            (7, 9),
            (8, 10),  # torso and arms
            (11, 12),
            (11, 13),
            (12, 14),
            (13, 15),
            (14, 16),  # legs
        ]

        # COCO keypoint names
        self.keypoint_names = [
            "nose",
            "left_eye",
            "right_eye",
            "left_ear",
            "right_ear",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
        ]

        # Colors for different elements
        self.colors = {
            "keypoint": (0, 255, 0),  # Green keypoints
            "skeleton": (255, 0, 0),  # Blue skeleton lines
            "bbox": (0, 0, 255),  # Red bounding boxes
            "text": (255, 255, 255),  # White text
            "background": (0, 0, 0),  # Black background for text
        }

    def draw_pose_on_image(
        self,
        image: np.ndarray,
        pose: PoseData,
        show_keypoints: bool = True,
        show_skeleton: bool = True,
        show_bbox: bool = True,
        show_confidence: bool = True,
        keypoint_size: int = 2,
        line_thickness: int = 2,
    ) -> np.ndarray:
        """
        Draw a single pose on an image.

        Args:
            image: Input image
            pose: Pose data to visualize
            show_keypoints: Whether to show keypoints
            show_skeleton: Whether to show skeleton connections
            show_bbox: Whether to show bounding box
            show_confidence: Whether to show confidence score
            keypoint_size: Size of keypoint circles
            line_thickness: Thickness of skeleton lines

        Returns:
            Image with pose visualization
        """
        img_copy = image.copy()

        if pose.keypoints is None:
            return img_copy

        # Draw skeleton connections
        if show_skeleton:
            for start_idx, end_idx in self.keypoint_connections:
                if (
                    start_idx < len(pose.keypoints)
                    and end_idx < len(pose.keypoints)
                    and pose.keypoints[start_idx] is not None
                    and pose.keypoints[end_idx] is not None
                ):

                    start_point = pose.keypoints[start_idx]
                    end_point = pose.keypoints[end_idx]

                    if start_point is not None and end_point is not None:
                        start_coord = (int(start_point[0]), int(start_point[1]))
                        end_coord = (int(end_point[0]), int(end_point[1]))

                        cv2.line(
                            img_copy,
                            start_coord,
                            end_coord,
                            self.colors["skeleton"],
                            line_thickness,
                        )

        # Draw keypoints
        if show_keypoints:
            for i, keypoint in enumerate(pose.keypoints):
                if keypoint is not None:
                    x, y, conf = keypoint
                    center = (int(x), int(y))

                    # Draw keypoint circle
                    cv2.circle(
                        img_copy, center, keypoint_size, self.colors["keypoint"], -1
                    )

                    # Draw keypoint index (small text)
                    cv2.putText(
                        img_copy,
                        str(i),
                        (center[0] + 5, center[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.3,
                        self.colors["text"],
                        1,
                    )

        # Draw bounding box
        if show_bbox and pose.bounding_box:
            x1, y1, x2, y2 = pose.bounding_box
            cv2.rectangle(
                img_copy, (int(x1), int(y1)), (int(x2), int(y2)), self.colors["bbox"], 2
            )

        # Draw confidence score
        if show_confidence:
            # Find top-left corner for text placement
            if pose.bounding_box:
                x1, y1, _, _ = pose.bounding_box
                text_x, text_y = int(x1), int(y1) - 10
            else:
                # Use first valid keypoint as reference
                valid_kp = next((kp for kp in pose.keypoints if kp is not None), None)
                if valid_kp:
                    text_x, text_y = int(valid_kp[0]), int(valid_kp[1]) - 20
                else:
                    text_x, text_y = 10, 30

            # Draw background rectangle for text
            text = f"Conf: {pose.confidence_score:.3f}"
            (text_width, text_height), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                img_copy,
                (text_x - 5, text_y - text_height - 5),
                (text_x + text_width + 5, text_y + 5),
                self.colors["background"],
                -1,
            )

            # Draw text
            cv2.putText(
                img_copy,
                text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                self.colors["text"],
                2,
            )

        return img_copy

    def create_comparison_visualization(
        self,
        target_image: np.ndarray,
        target_pose: PoseData,
        comparison_results: List[SimilarityResult],
        comparison_images: List[Tuple[str, np.ndarray]],
        comparison_poses_data: List[PoseData] = None,
        output_path: Optional[str] = None,
        max_images_per_row: int = 3,
        apply_body_mask: bool = False,
        pose_estimator=None,
        show_skeleton: bool = True,
    ) -> np.ndarray:
        """
        Create a comprehensive visualization showing target pose and comparison results.

        Args:
            target_image: Target image array
            target_pose: Target pose data
            comparison_results: List of similarity results
            comparison_images: List of (path, image_array) tuples
            comparison_poses_data: List of pose data for comparison images
            output_path: Optional path to save the visualization
            max_images_per_row: Maximum images per row in the grid
            apply_body_mask: Whether to apply body segmentation masks
            pose_estimator: Pose estimator instance for masking
            show_skeleton: Whether to draw skeletons on comparison images (target always shows skeleton)

        Returns:
            Visualization image
        """
        # Draw target pose and resize to HD
        target_vis = self.draw_pose_on_image(
            target_image,
            target_pose,
            show_keypoints=True,
            show_skeleton=True,
            show_bbox=True,
            show_confidence=True,
            keypoint_size=3,
            line_thickness=2,
        )
        target_vis = self._resize_to_hd_with_padding(target_vis)

        # Prepare comparison visualizations - ONLY for images that passed filtering
        comparison_vis = []
        comparison_poses = []  # Store pose data for potential overlay

        # Only process images that are in comparison_results (these passed the filtering)
        for i, result in enumerate(comparison_results):
            # Find corresponding image
            img_name = Path(result.comparison_image).name
            comp_img = None
            comp_pose = None

            for j, (img_path, img_array) in enumerate(comparison_images):
                if Path(img_path).name == img_name and img_array is not None:
                    comp_img = img_array
                    # Use the pose data directly from the result if available
                    if result.comparison_pose:
                        comp_pose = result.comparison_pose
                    # Fallback to comparison_poses_data if needed
                    elif comparison_poses_data and j < len(comparison_poses_data):
                        comp_pose = comparison_poses_data[j]
                    break

            if comp_img is not None:
                # Apply body mask if requested
                if apply_body_mask and pose_estimator:
                    if comp_pose:
                        try:
                            # ALWAYS use pose-specific mask when pose data is available
                            # This ensures the mask corresponds to the same person as the skeleton
                            comp_img = pose_estimator.create_pose_specific_mask(
                                comp_img, comp_pose
                            )
                        except Exception as e:
                            print(
                                f"Warning: Failed to apply pose-specific mask to {img_name}: {e}"
                            )
                            # Don't fallback to general mask - it might segment a different person
                    else:
                        # Only use general body mask as last resort when no pose data available
                        try:
                            comp_img = pose_estimator.create_body_mask(comp_img)
                        except Exception as e:
                            print(
                                f"Warning: Failed to apply body mask to {img_name}: {e}"
                            )

                # First resize to HD resolution with black padding
                hd_img = self._resize_to_hd_with_padding(comp_img)

                # Now draw similarity score text on the HD image
                text = f"{result.similarity_score:.3f}"
                cv2.putText(
                    hd_img,
                    text,
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    3,
                )
                cv2.putText(
                    hd_img, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4
                )

                # Draw skeleton if pose data is available (on HD image)
                if comp_pose and comp_pose.keypoints:
                    # Scale pose keypoints to HD resolution
                    scaled_pose = self._scale_pose_to_hd(
                        comp_pose, comp_img.shape, (1920, 1080)
                    )
                    if scaled_pose:
                        hd_img = self.draw_pose_on_image(
                            hd_img,
                            scaled_pose,
                            show_keypoints=True,
                            show_skeleton=show_skeleton,
                            show_bbox=False,
                            show_confidence=False,
                            keypoint_size=4,
                            line_thickness=2,
                        )

                comparison_vis.append(hd_img)
                comparison_poses.append(result)  # Store for potential use

        # All images are now HD resolution (1920x1080)
        target_width, target_height = 1920, 1080

        # Create grid layout with overlay image
        n_comparisons = len(comparison_vis)
        # Add 1 for the overlay image (target + winning pose skeleton)
        n_total_images = n_comparisons + 1

        # Dynamically adjust grid size for large numbers of images
        if n_comparisons > 20:
            # For large datasets, use more columns to keep grid manageable
            actual_max_cols = min(6, max_images_per_row * 2)
        else:
            actual_max_cols = max_images_per_row

        n_rows = (n_total_images + actual_max_cols - 1) // actual_max_cols

        if n_comparisons == 0:
            return target_vis

        # Use consistent dimensions from resized images
        img_height, img_width = target_height, target_width

        # Create grid
        grid_height = img_height + n_rows * img_height
        grid_width = actual_max_cols * img_width

        # Create grid image
        grid_img = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

        # Place target image at top (centered)
        target_start_x = (grid_width - img_width) // 2
        grid_img[:img_height, target_start_x : target_start_x + img_width] = target_vis

        # Create and place overlay image (target with winning pose skeleton)
        overlay_img = None
        if comparison_results and len(comparison_results) > 0:
            # Find the best matching pose for overlay
            top_result = comparison_results[0]
            overlay_pose = None

            # Use the pose data directly from the result if available
            if top_result.comparison_pose:
                overlay_pose = top_result.comparison_pose

            else:
                # Fallback: find corresponding pose data for overlay

                for i, (img_path, _) in enumerate(comparison_images):
                    if str(img_path) == top_result.comparison_image:
                        if comparison_poses_data and i < len(comparison_poses_data):
                            overlay_pose = comparison_poses_data[i]

                            break

            if overlay_pose and overlay_pose.keypoints:

                # Create overlay image
                overlay_img = self.create_winning_pose_overlay(
                    target_image, target_pose, overlay_pose, top_result.similarity_score
                )
                # Overlay is already HD resolution, no need to resize
            else:
                pass

        # Place overlay image first (if available)
        if overlay_img is not None:
            overlay_row = 0
            overlay_col = 0
            y_start = img_height + overlay_row * img_height
            y_end = y_start + img_height
            x_start = overlay_col * img_width
            x_end = x_start + img_width

            grid_img[y_start:y_end, x_start:x_end] = overlay_img

        # Place comparison images in grid (starting from second position)
        for i, comp_vis in enumerate(comparison_vis):
            # Skip first position if overlay is there
            grid_pos = i + 1 if overlay_img is not None else i
            row = grid_pos // actual_max_cols
            col = grid_pos % actual_max_cols

            y_start = img_height + row * img_height
            y_end = y_start + img_height
            x_start = col * img_width
            x_end = x_start + img_width

            grid_img[y_start:y_end, x_start:x_end] = comp_vis

        # Add title and labels (adjusted for HD resolution)
        title = "Pose Comparison Results"
        cv2.putText(
            grid_img, title, (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255, 255, 255), 6
        )
        cv2.putText(
            grid_img, title, (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 0), 10
        )

        # Add target label
        target_label = f"Target: {Path(target_pose.image_path).name} (Conf: {target_pose.confidence_score:.3f})"
        cv2.putText(
            grid_img,
            target_label,
            (40, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (255, 255, 255),
            4,
        )
        cv2.putText(
            grid_img,
            target_label,
            (40, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 0, 0),
            8,
        )

        # Add overlay label if overlay image exists (adjusted for HD resolution)
        if overlay_img is not None:
            overlay_label = f"Overlay: Target + Best Match Skeleton (Score: {comparison_results[0].similarity_score:.3f})"
            cv2.putText(
                grid_img,
                overlay_label,
                (40, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (255, 255, 255),
                4,
            )
            cv2.putText(
                grid_img,
                overlay_label,
                (40, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 0, 0),
                8,
            )

        # Save if output path provided
        if output_path:
            cv2.imwrite(output_path, grid_img)
            print(f"Visualization saved to: {output_path}")

        return grid_img

    def create_winning_pose_overlay(
        self,
        target_image: np.ndarray,
        target_pose: PoseData,
        winning_pose: PoseData,
        similarity_score: float,
        output_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        Create an overlay showing the winning comparison pose skeleton on the target image.

        Args:
            target_image: Target image array
            target_pose: Target pose data
            winning_pose: Winning comparison pose data
            similarity_score: Similarity score between poses
            output_path: Optional path to save the overlay

        Returns:
            Overlay visualization image
        """
        # Start with target image and resize to HD
        overlay_img = self._resize_to_hd_with_padding(target_image.copy())

        # Scale target pose to HD resolution before drawing
        target_h, target_w = target_image.shape[:2]
        hd_target_pose = self._scale_pose_to_hd(
            target_pose, (target_h, target_w), (1920, 1080)
        )

        # Draw target pose in blue on HD image using HD-scaled coordinates
        target_vis = self.draw_pose_on_image(
            overlay_img,
            hd_target_pose,
            show_keypoints=True,
            show_skeleton=True,
            show_bbox=False,
            show_confidence=False,
            keypoint_size=4,
            line_thickness=2,
        )

        # Draw winning pose skeleton aligned to target pose (overlay on target image)
        if winning_pose.keypoints and target_pose.keypoints:
            # Use different colors for overlay
            overlay_colors = {
                "keypoint": (0, 255, 255),  # Yellow keypoints for winning pose
                "skeleton": (255, 0, 0),  # Blue skeleton lines for winning pose
            }

            # The winning pose coordinates are from its own image, not the target image
            # We need to load the actual winning image to get the correct coordinate system
            try:
                from utils.image_utils import load_image

                winning_image = load_image(winning_pose.image_path)
                winning_h, winning_w = winning_image.shape[:2]

                # The winning pose needs to be scaled to match the target image's HD transformation
                # First, let's see how the target image was transformed to HD
                target_h, target_w = target_image.shape[:2]

                # Calculate how the target image was scaled to HD (same logic as _resize_to_hd_with_padding)
                target_scale_w = 1920 / target_w
                target_scale_h = 1080 / target_h
                target_scale = min(target_scale_w, target_scale_h)

                # Calculate target image's new dimensions and padding
                target_new_w = int(target_w * target_scale)
                target_new_h = int(target_h * target_scale)
                target_x_offset = (1920 - target_new_w) // 2
                target_y_offset = (1080 - target_new_h) // 2

                # Instead of just scaling, we need to ALIGN the winning pose to the target pose
                # First, scale both poses to HD using their respective transformations
                hd_target_pose = self._scale_pose_to_hd(
                    target_pose, (target_h, target_w), (1920, 1080)
                )
                hd_winning_pose = self._scale_pose_to_hd(
                    winning_pose, (winning_h, winning_w), (1920, 1080)
                )

                # Now align the winning pose to the target pose using the existing alignment function
                # This will transform the winning pose to match the target pose's position and orientation
                aligned_keypoints = self._align_pose_to_target(
                    hd_winning_pose, hd_target_pose
                )

                if aligned_keypoints is None:
                    print("Alignment failed, falling back to scaled coordinates")
                    # Fallback: use scaled coordinates but center them on the target pose
                    target_center = self._get_pose_center(hd_target_pose.keypoints)
                    winning_center = self._get_pose_center(hd_winning_pose.keypoints)

                    if target_center and winning_center:
                        # Calculate offset to move winning pose to target center
                        offset_x = target_center[0] - winning_center[0]
                        offset_y = target_center[1] - winning_center[1]

                        aligned_keypoints = []
                        for kp in hd_winning_pose.keypoints:
                            if kp is not None:
                                x, y, conf = kp
                                aligned_keypoints.append(
                                    (x + offset_x, y + offset_y, conf)
                                )
                            else:
                                aligned_keypoints.append(None)
                    else:
                        aligned_keypoints = hd_winning_pose.keypoints

            except Exception as e:
                print(f"Failed to load winning image: {e}")
                return target_vis

            if aligned_keypoints:
                # Draw skeleton connections for aligned winning pose
                for start_idx, end_idx in self.keypoint_connections:
                    if (
                        start_idx < len(aligned_keypoints)
                        and end_idx < len(aligned_keypoints)
                        and aligned_keypoints[start_idx] is not None
                        and aligned_keypoints[end_idx] is not None
                    ):

                        start_point = aligned_keypoints[start_idx]
                        end_point = aligned_keypoints[end_idx]

                        if start_point is not None and end_point is not None:
                            start_coord = (int(start_point[0]), int(start_point[1]))
                            end_coord = (int(end_point[0]), int(end_point[1]))

                            cv2.line(
                                target_vis,
                                start_coord,
                                end_coord,
                                overlay_colors["skeleton"],
                                2,
                            )

                # Draw keypoints for aligned winning pose
                for i, keypoint in enumerate(aligned_keypoints):
                    if keypoint is not None:
                        x, y, conf = keypoint
                        center = (int(x), int(y))

                        # Draw keypoint circle
                        cv2.circle(
                            target_vis, center, 3, overlay_colors["keypoint"], -1
                        )
                        cv2.circle(target_vis, center, 3, (0, 0, 0), 1)  # Black border

        # Add legend and similarity score (adjusted for HD resolution)
        legend_y = 80
        cv2.putText(
            target_vis,
            f"Similarity: {similarity_score:.3f}",
            (40, legend_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.0,
            (255, 255, 255),
            6,
        )
        cv2.putText(
            target_vis,
            f"Similarity: {similarity_score:.3f}",
            (40, legend_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.0,
            (0, 0, 0),
            3,
        )

        # Add legend
        legend_y += 120
        cv2.putText(
            target_vis,
            "Blue: Target pose",
            (40, legend_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (255, 0, 0),
            4,
        )  # Blue text
        legend_y += 80
        cv2.putText(
            target_vis,
            "Blue: Best match",
            (40, legend_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (255, 0, 0),
            4,
        )  # Blue text

        # Save if output path provided
        if output_path:
            cv2.imwrite(output_path, target_vis)
            print(f"Winning pose overlay saved to: {output_path}")

        return target_vis

    def _align_pose_to_target(
        self, source_pose: PoseData, target_pose: PoseData
    ) -> List:
        """
        Align source pose keypoints to target pose coordinate system.

        Args:
            source_pose: Pose to transform
            target_pose: Reference pose for alignment

        Returns:
            Transformed keypoints aligned to target pose
        """
        # Get torso keypoints for both poses (shoulders and hips)
        source_torso = self._get_torso_keypoints(source_pose.keypoints)
        target_torso = self._get_torso_keypoints(target_pose.keypoints)

        if not source_torso or not target_torso:
            return None

        # Calculate center and scale for both poses
        source_center = np.mean(source_torso, axis=0)
        target_center = np.mean(target_torso, axis=0)

        # Calculate scale based on torso size
        source_scale = self._calculate_torso_scale(source_torso)
        target_scale = self._calculate_torso_scale(target_torso)

        if source_scale == 0:
            return None

        scale_factor = target_scale / source_scale

        # Transform all keypoints
        aligned_keypoints = []
        for kp in source_pose.keypoints:
            if kp is not None:
                x, y, conf = kp
                # Translate to origin, scale, then translate to target center
                new_x = (x - source_center[0]) * scale_factor + target_center[0]
                new_y = (y - source_center[1]) * scale_factor + target_center[1]
                aligned_keypoints.append((new_x, new_y, conf))
            else:
                aligned_keypoints.append(None)

        return aligned_keypoints

    def _get_pose_center(self, keypoints: List) -> Optional[Tuple[float, float]]:
        """Get the center point of a pose from its keypoints."""
        valid_keypoints = [kp for kp in keypoints if kp is not None]
        if not valid_keypoints:
            return None

        # Calculate center from all valid keypoints
        x_coords = [kp[0] for kp in valid_keypoints]
        y_coords = [kp[1] for kp in valid_keypoints]

        center_x = sum(x_coords) / len(x_coords)
        center_y = sum(y_coords) / len(y_coords)

        return (center_x, center_y)

    def _get_torso_keypoints(self, keypoints: List) -> List:
        """Get valid torso keypoints (shoulders and hips)."""
        torso_indices = [
            5,
            6,
            11,
            12,
        ]  # left_shoulder, right_shoulder, left_hip, right_hip
        torso_points = []

        for idx in torso_indices:
            if idx < len(keypoints) and keypoints[idx] is not None:
                x, y, conf = keypoints[idx]
                torso_points.append([x, y])

        return torso_points if len(torso_points) >= 2 else None

    def _calculate_torso_scale(self, torso_points: List) -> float:
        """Calculate characteristic scale from torso keypoints."""
        if len(torso_points) < 2:
            return 0.0

        points = np.array(torso_points)
        # Use the maximum distance between any two torso points as scale
        distances = []
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                dist = np.linalg.norm(points[i] - points[j])
                distances.append(dist)

        return max(distances) if distances else 0.0

    def _resize_to_hd_with_padding(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image to HD resolution (1920x1080) with black padding to maintain aspect ratio.

        Args:
            image: Input image array

        Returns:
            HD image with black padding
        """
        target_width, target_height = 1920, 1080

        # Get original dimensions
        h, w = image.shape[:2]

        # Calculate scaling factor to fit image within HD bounds
        scale_w = target_width / w
        scale_h = target_height / h
        scale = min(scale_w, scale_h)

        # Calculate new dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resize image
        resized = cv2.resize(image, (new_w, new_h))

        # Create HD canvas with black background
        hd_canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)

        # Calculate position to center the resized image
        x_offset = (target_width - new_w) // 2
        y_offset = (target_height - new_h) // 2

        # Place resized image on canvas
        hd_canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

        return hd_canvas

    def _scale_pose_to_hd(
        self, pose: PoseData, original_shape: tuple, target_shape: tuple
    ) -> PoseData:
        """
        Scale pose keypoints from original image dimensions to HD dimensions.

        Args:
            pose: Original pose data
            original_shape: (height, width) of original image
            target_shape: (width, height) of target HD image

        Returns:
            Scaled pose data
        """
        if not pose.keypoints:
            return pose

        orig_h, orig_w = original_shape[:2]
        target_w, target_h = target_shape

        # Calculate scaling factors
        scale_w = target_w / orig_w
        scale_h = target_h / orig_h
        scale = min(scale_w, scale_h)

        # Calculate padding offsets
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2

        # Scale and offset keypoints
        scaled_keypoints = []
        for kp in pose.keypoints:
            if kp is not None:
                x, y, conf = kp
                # Scale coordinates
                new_x = x * scale + x_offset
                new_y = y * scale + y_offset
                scaled_keypoints.append((new_x, new_y, conf))
            else:
                scaled_keypoints.append(None)

        # Create new pose data with scaled keypoints
        scaled_pose = PoseData(
            keypoints=scaled_keypoints,
            bounding_box=pose.bounding_box,  # Bounding box not used in visualization
            confidence_score=pose.confidence_score,
            image_path=pose.image_path,
            pose_id=pose.pose_id,
        )

        return scaled_pose
