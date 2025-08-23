#!/usr/bin/env python3
"""
Pose Visualizer
Creates diagnostic images showing detected poses, keypoints, and similarity scores.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

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
            "skeleton": (255, 0, 0),  # Red skeleton lines
            "bbox": (0, 0, 255),  # Blue bounding boxes
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
        keypoint_size: int = 4,
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
    ) -> np.ndarray:
        """
        Create a comprehensive visualization showing target pose and comparison results.

        Args:
            target_image: Target image array
            target_pose: Target pose data
            comparison_results: List of similarity results
            comparison_images: List of (path, image_array) tuples
            output_path: Optional path to save the visualization
            max_images_per_row: Maximum images per row in the grid

        Returns:
            Visualization image
        """
        # Draw target pose
        target_vis = self.draw_pose_on_image(target_image, target_pose)

        # Prepare comparison visualizations
        comparison_vis = []
        comparison_poses = []  # Store pose data for potential overlay

        for i, result in enumerate(comparison_results):
            # Find corresponding image
            img_name = Path(result.comparison_image).name
            comp_img = None
            comp_pose = None

            for j, (img_path, img_array) in enumerate(comparison_images):
                if Path(img_path).name == img_name and img_array is not None:
                    comp_img = img_array
                    # Get corresponding pose data if available
                    if comparison_poses_data and j < len(comparison_poses_data):
                        comp_pose = comparison_poses_data[j]
                    break

            if comp_img is not None:
                # Draw similarity score on image
                vis_img = comp_img.copy()

                # Add similarity score text (smaller font)
                text = f"{result.similarity_score:.3f}"
                cv2.putText(
                    vis_img,
                    text,
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    vis_img, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3
                )

                # Draw skeleton if pose data is available
                if comp_pose and comp_pose.keypoints:
                    vis_img = self.draw_pose_on_image(
                        vis_img,
                        comp_pose,
                        show_keypoints=True,
                        show_skeleton=True,
                        show_bbox=False,
                        show_confidence=False,
                        keypoint_size=4,
                        line_thickness=2,
                    )

                comparison_vis.append(vis_img)
                comparison_poses.append(result)  # Store for potential use

        # Resize all comparison images to consistent higher resolution
        if comparison_vis:
            # Use a higher standard resolution (800x600)
            target_width, target_height = 800, 600

            # Resize target image
            target_vis = cv2.resize(target_vis, (target_width, target_height))

            # Resize all comparison images
            for i in range(len(comparison_vis)):
                comparison_vis[i] = cv2.resize(
                    comparison_vis[i], (target_width, target_height)
                )

        # Create grid layout with overlay image
        n_comparisons = len(comparison_vis)
        # Add 1 for the overlay image (target + winning pose skeleton)
        n_total_images = n_comparisons + 1
        n_rows = (n_total_images + max_images_per_row - 1) // max_images_per_row

        if n_comparisons == 0:
            return target_vis

        # Use consistent dimensions from resized images
        img_height, img_width = target_height, target_width

        # Create grid
        grid_height = img_height + n_rows * img_height
        grid_width = max_images_per_row * img_width

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

            # Find corresponding pose data for overlay
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
                # Resize overlay to match grid size
                overlay_img = cv2.resize(overlay_img, (img_width, img_height))

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
            row = grid_pos // max_images_per_row
            col = grid_pos % max_images_per_row

            y_start = img_height + row * img_height
            y_end = y_start + img_height
            x_start = col * img_width
            x_end = x_start + img_width

            grid_img[y_start:y_end, x_start:x_end] = comp_vis

        # Add title and labels
        title = "Pose Comparison Results"
        cv2.putText(
            grid_img, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3
        )
        cv2.putText(
            grid_img, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 5
        )

        # Add target label
        target_label = f"Target: {Path(target_pose.image_path).name} (Conf: {target_pose.confidence_score:.3f})"
        cv2.putText(
            grid_img,
            target_label,
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            grid_img,
            target_label,
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            4,
        )

        # Add overlay label if overlay image exists
        if overlay_img is not None:
            overlay_label = f"Overlay: Target + Best Match Skeleton (Score: {comparison_results[0].similarity_score:.3f})"
            cv2.putText(
                grid_img,
                overlay_label,
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                grid_img,
                overlay_label,
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 0),
                4,
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
        # Start with target image
        overlay_img = target_image.copy()

        # Draw target pose in blue
        target_vis = self.draw_pose_on_image(
            overlay_img,
            target_pose,
            show_keypoints=True,
            show_skeleton=True,
            show_bbox=False,
            show_confidence=False,
            keypoint_size=6,
            line_thickness=3,
        )

        # Draw winning pose skeleton aligned to target pose (overlay on target image)
        if winning_pose.keypoints and target_pose.keypoints:
            # Use different colors for overlay
            overlay_colors = {
                "keypoint": (0, 255, 255),  # Yellow keypoints for winning pose
                "skeleton": (0, 165, 255),  # Orange skeleton lines for winning pose
            }

            # Transform winning pose to align with target pose
            aligned_keypoints = self._align_pose_to_target(winning_pose, target_pose)

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
                                3,
                            )

                # Draw keypoints for aligned winning pose
                for i, keypoint in enumerate(aligned_keypoints):
                    if keypoint is not None:
                        x, y, conf = keypoint
                        center = (int(x), int(y))

                        # Draw keypoint circle
                        cv2.circle(
                            target_vis, center, 6, overlay_colors["keypoint"], -1
                        )
                        cv2.circle(target_vis, center, 6, (0, 0, 0), 2)  # Black border

        # Add legend and similarity score
        legend_y = 30
        cv2.putText(
            target_vis,
            f"Similarity: {similarity_score:.3f}",
            (10, legend_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            3,
        )
        cv2.putText(
            target_vis,
            f"Similarity: {similarity_score:.3f}",
            (10, legend_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            1,
        )

        # Add legend
        legend_y += 40
        cv2.putText(
            target_vis,
            "Blue: Target pose",
            (10, legend_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
        )  # Blue text
        legend_y += 30
        cv2.putText(
            target_vis,
            "Orange: Best match",
            (10, legend_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 165, 255),
            2,
        )  # Orange text

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

    def create_keypoint_analysis(
        self,
        target_pose: PoseData,
        comparison_pose: PoseData,
        similarity_result: SimilarityResult,
    ) -> np.ndarray:
        """
        Create a detailed keypoint analysis visualization.

        Args:
            target_pose: Target pose data
            comparison_pose: Comparison pose data
            similarity_result: Similarity result between poses

        Returns:
            Analysis visualization image
        """
        # Create figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

        # Plot 1: Target pose keypoints
        self._plot_keypoints(ax1, target_pose, "Target Pose", "green")

        # Plot 2: Comparison pose keypoints
        self._plot_keypoints(ax2, comparison_pose, "Comparison Pose", "red")

        # Plot 3: Keypoint distance comparison
        self._plot_keypoint_distances(ax3, similarity_result)

        plt.tight_layout()

        # Convert matplotlib figure to numpy array
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        img = np.asarray(buf)
        img = img[:, :, :3]  # Remove alpha channel

        plt.close(fig)

        # Convert RGB to BGR for OpenCV
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        return img

    def _plot_keypoints(self, ax, pose: PoseData, title: str, color: str):
        """Plot keypoints on a matplotlib axis."""
        if pose.keypoints is None:
            ax.text(
                0.5,
                0.5,
                "No keypoints",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(title)
            return

        # Extract valid keypoints
        valid_kps = [(i, kp) for i, kp in enumerate(pose.keypoints) if kp is not None]

        if not valid_kps:
            ax.text(
                0.5,
                0.5,
                "No valid keypoints",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(title)
            return

        # Plot keypoints
        x_coords = [kp[1][0] for kp in valid_kps]
        y_coords = [kp[1][1] for kp in valid_kps]
        confidences = [kp[1][2] for kp in valid_kps]

        # Normalize coordinates for better visualization
        x_norm = [
            (x - min(x_coords)) / (max(x_coords) - min(x_coords)) for x in x_coords
        ]
        y_norm = [
            (y - min(y_coords)) / (max(y_coords) - min(y_coords)) for y in y_coords
        ]

        # Plot keypoints with size based on confidence
        sizes = [conf * 100 for conf in confidences]
        ax.scatter(x_norm, y_norm, s=sizes, c=color, alpha=0.7)

        # Add keypoint labels
        for i, (idx, _) in enumerate(valid_kps):
            ax.annotate(
                str(idx),
                (x_norm[i], y_norm[i]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

        # Draw skeleton connections
        for start_idx, end_idx in self.keypoint_connections:
            start_kp = next((kp for kp in valid_kps if kp[0] == start_idx), None)
            end_kp = next((kp for kp in valid_kps if kp[0] == end_idx), None)

            if start_kp and end_kp:
                start_pos = start_kp[1][:2]
                end_pos = end_kp[1][:2]

                # Normalize positions
                start_norm = [
                    (start_pos[0] - min(x_coords)) / (max(x_coords) - min(x_coords)),
                    (start_pos[1] - min(y_coords)) / (max(y_coords) - min(y_coords)),
                ]
                end_norm = [
                    (end_pos[0] - min(x_coords)) / (max(x_coords) - min(x_coords)),
                    (end_pos[1] - min(y_coords)) / (max(y_coords) - min(y_coords)),
                ]

                ax.plot(
                    [start_norm[0], end_norm[0]],
                    [start_norm[1], end_norm[1]],
                    color=color,
                    alpha=0.5,
                    linewidth=1,
                )

        ax.set_title(f"{title}\nConfidence: {pose.confidence_score:.3f}")
        ax.set_xlabel("Normalized X")
        ax.set_ylabel("Normalized Y")
        ax.grid(True, alpha=0.3)

    def _plot_keypoint_distances(self, ax, similarity_result: SimilarityResult):
        """Plot keypoint distance comparison."""
        distances = similarity_result.keypoint_distances

        # Filter out invalid distances (-1)
        valid_distances = [(i, d) for i, d in enumerate(distances) if d >= 0]

        if not valid_distances:
            ax.text(
                0.5,
                0.5,
                "No distance data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Keypoint Distances")
            return

        indices, dist_values = zip(*valid_distances)

        # Create bar chart
        bars = ax.bar(indices, dist_values, alpha=0.7, color="blue")

        # Color bars based on distance magnitude
        for bar, dist in zip(bars, dist_values):
            if dist < 0.1:
                bar.set_color("green")  # Good match
            elif dist < 0.3:
                bar.set_color("orange")  # Moderate match
            else:
                bar.set_color("red")  # Poor match

        ax.set_title(
            f"Keypoint Distances\nSimilarity: {similarity_result.similarity_score:.3f}"
        )
        ax.set_xlabel("Keypoint Index")
        ax.set_ylabel("Distance")
        ax.grid(True, alpha=0.3)

        # Add keypoint name labels for major indices
        major_indices = [0, 5, 6, 11, 12]  # nose, shoulders, hips
        ax.set_xticks(major_indices)
        ax.set_xticklabels([self.keypoint_names[i] for i in major_indices], rotation=45)
