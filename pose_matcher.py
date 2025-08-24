"""
Pose Matcher
Handles pose similarity calculations and matching between target and comparison poses.
"""

import numpy as np
from typing import List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from utils.pose_utils import PoseData, SimilarityResult


class PoseMatcher:
    """Pose similarity matching and ranking."""

    def __init__(
        self,
        distance_metric: str = "euclidean",
        normalize_keypoints: bool = True,
        weight_by_confidence: bool = True,
    ):
        """
        Initialize the pose matcher.

        Args:
            distance_metric: Distance metric ('euclidean', 'cosine', 'manhattan')
            normalize_keypoints: Whether to normalize keypoint coordinates
            weight_by_confidence: Whether to weight distances by keypoint confidence
        """
        self.distance_metric = distance_metric
        self.normalize_keypoints = normalize_keypoints
        self.weight_by_confidence = weight_by_confidence
        self.scaler = StandardScaler() if normalize_keypoints else None

    def find_best_match(
        self,
        target_pose: PoseData,
        comparison_poses: List[PoseData],
        relative_visibility_threshold: float = 0.6,
    ) -> Optional[SimilarityResult]:
        """
        Find the best matching pose from a list of comparison poses.

        Args:
            target_pose: Target pose to match against
            comparison_poses: List of poses to compare with

        Returns:
            Best matching pose result or None if no valid matches
        """
        if not comparison_poses:
            return None

        best_match = None
        best_score = -1

        for comparison_pose in comparison_poses:
            similarity_score = self.calculate_similarity(
                target_pose,
                comparison_pose,
                relative_visibility_threshold=relative_visibility_threshold,
            )

            if similarity_score > best_score:
                best_score = similarity_score
                best_match = comparison_pose

        if best_match is None or best_score <= 0.0:
            return None

        # Calculate detailed similarity result
        keypoint_distances = self._calculate_keypoint_distances(target_pose, best_match)

        return SimilarityResult(
            target_image=target_pose.image_path,
            comparison_image=best_match.image_path,
            similarity_score=best_score,
            keypoint_distances=keypoint_distances,
            rank=1,  # Will be updated by caller
            comparison_pose=best_match,  # Include the actual pose that gave this score
        )

    def calculate_similarity(
        self,
        pose1: PoseData,
        pose2: PoseData,
        min_visibility_threshold: float = 0.6,
        relative_visibility_threshold: float = 0.6,
    ) -> float:
        """
        Calculate similarity between two poses using normalized pose comparison.

        Args:
            pose1: First pose
            pose2: Second pose
            min_visibility_threshold: Minimum percentage of visible keypoints required (0.0 to 1.0)

        Returns:
            Similarity score between 0 and 1 (higher is more similar)
        """
        # Check individual visibility thresholds first
        if not self._meets_visibility_threshold(pose1, pose2, min_visibility_threshold):
            return 0.0

        # Normalize poses to be position and scale invariant
        norm_pose1 = self._normalize_pose(pose1)
        norm_pose2 = self._normalize_pose(pose2)

        if norm_pose1 is None or norm_pose2 is None:
            return 0.0

        # Find common keypoints with meaningful confidence
        # A keypoint is only considered "visible" if it has confidence above a threshold
        confidence_threshold = (
            0.3  # Keypoints below this confidence are essentially invisible
        )

        common_indices = []
        for i in range(len(norm_pose1)):
            if (
                norm_pose1[i] is not None
                and norm_pose2[i] is not None
                and len(norm_pose1[i]) >= 3
                and len(norm_pose2[i]) >= 3
            ):
                # Check confidence of both keypoints
                conf1 = norm_pose1[i][2] if len(norm_pose1[i]) > 2 else 0.0
                conf2 = norm_pose2[i][2] if len(norm_pose2[i]) > 2 else 0.0

                # Only consider keypoints with meaningful confidence
                if conf1 >= confidence_threshold and conf2 >= confidence_threshold:
                    common_indices.append(i)

        if len(common_indices) < 3:  # Need at least 3 points for meaningful comparison
            return 0.0

        # CRITICAL: Check relative visibility - poses must share enough keypoints relative to the target
        # We want at least 60% of the target's visible keypoints to also be visible in the comparison
        # But now we only count keypoints with meaningful confidence
        target_visible_keypoints = sum(
            1
            for kp in pose1.keypoints
            if kp is not None and len(kp) >= 3 and kp[2] >= confidence_threshold
        )
        if target_visible_keypoints == 0:
            return 0.0

        relative_visibility = len(common_indices) / target_visible_keypoints

        # Instead of filtering, let's apply a quality penalty based on keypoint distribution
        # Poses with only shoulders visible should get penalized in the similarity calculation
        # We'll do this by reducing the similarity score based on missing keypoints

        if (
            relative_visibility < relative_visibility_threshold
        ):  # At least X% of target's keypoints must be shared
            return 0.0

        # NEW: Check for major body part completeness
        # A pose must have keypoints from ALL major body regions to be considered complete
        major_regions = {
            "head": [0, 1, 2, 3, 4],  # nose, eyes, ears
            "torso": [5, 6, 11, 12],  # shoulders, hips
            "arms": [7, 8, 9, 10],  # elbows, wrists
            "legs": [13, 14, 15, 16],  # knees, ankles
        }

        # Check if comparison pose has at least 2 keypoints from each major region
        region_completeness = {}
        for region_name, region_indices in major_regions.items():
            region_keypoints = [i for i in common_indices if i in region_indices]
            region_completeness[region_name] = len(region_keypoints)

        # Require at least 2 keypoints from each major region
        min_region_keypoints = 2
        for region_name, count in region_completeness.items():
            if count < min_region_keypoints:
                # print(
                #     f"REJECTING: {region_name} only has {count} keypoints (need {min_region_keypoints}+)"
                # )
                return 0.0

        # Uncomment for debugging if needed
        # print(f"REGION COMPLETENESS: {region_completeness}")

        # Calculate normalized distances for common keypoints
        distances = []
        confidences = []

        for idx in common_indices:
            kp1 = norm_pose1[idx]
            kp2 = norm_pose2[idx]

            # Extract normalized coordinates and confidence
            x1, y1, conf1 = kp1
            x2, y2, conf2 = kp2

            # Calculate Euclidean distance in normalized space
            dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

            distances.append(dist)
            confidences.append((conf1 + conf2) / 2)  # Average confidence

        if not distances:
            return 0.0

        # Calculate weighted mean squared distance
        if self.weight_by_confidence and confidences:
            weights = np.array(confidences)
            weights = weights / np.sum(weights)  # Normalize weights
            mse = np.average(np.array(distances) ** 2, weights=weights)
        else:
            mse = np.mean(np.array(distances) ** 2)

        # Convert MSE to similarity score
        # Use a decay function that gives meaningful discrimination
        # Scale factor chosen so that MSE of 0.1 gives ~60% similarity, MSE of 0.5 gives ~14% similarity
        scale_factor = 0.2
        similarity = np.exp(-mse / scale_factor)

        # Apply penalty for missing keypoints
        # The more keypoints we're missing, the more we penalize the similarity
        total_keypoints = len(pose1.keypoints)
        missing_penalty = (total_keypoints - len(common_indices)) / total_keypoints
        similarity *= (
            1.0 - missing_penalty * 0.5
        )  # Reduce similarity by up to 50% for missing keypoints

        # Additional quality penalty: penalize poses missing keypoints from important body regions
        # Critical regions: torso (shoulders + hips), arms (elbows), legs (knees)
        critical_indices = [
            5,
            6,
            11,
            12,
            7,
            8,
            13,
            14,
        ]  # shoulders, hips, elbows, knees
        critical_common = [i for i in common_indices if i in critical_indices]
        critical_penalty = (
            8 - len(critical_common)
        ) / 8  # Penalty for missing critical keypoints

        # Apply critical penalty much more aggressively (up to 95% reduction)
        # This should effectively filter out poses with only shoulders visible
        similarity *= 1.0 - critical_penalty * 0.95

        # Uncomment for debugging if needed
        # print(f"\n=== POSE COMPARISON DEBUG ===")
        # print(f"Target: {pose1.image_path}")
        # print(f"Comparison: {pose2.image_path}")
        # print(f"Missing keypoint penalty: {missing_penalty:.4f} -> Factor: {1.0 - missing_penalty * 0.5:.4f}")
        # print(f"Critical keypoints: {len(critical_common)}/8 (need 6+)")
        # print(f"Critical penalty: {critical_penalty:.4f} -> Factor: {1.0 - critical_penalty * 0.95:.4f}")
        # print(f"Final similarity: {similarity:.4f}")
        # print(f"Target visible keypoints: {target_visible_keypoints}")
        # print(f"Comparison visible keypoints: {sum(1 for kp in pose2.keypoints if kp is not None)}")
        # print(f"Common keypoints: {len(common_indices)}")
        # print(f"Relative visibility: {relative_visibility:.4f}")
        # print(f"Common keypoint indices: {sorted(common_indices)}")
        # print(f"Missing keypoint indices: {sorted(set(range(17)) - set(common_indices))}")
        # print("=" * 50)

        # Ensure similarity is between 0 and 1
        similarity = max(0.0, min(1.0, similarity))

        return float(similarity)

    def _calculate_keypoint_distances(
        self, pose1: PoseData, pose2: PoseData
    ) -> List[float]:
        """Calculate individual keypoint distances between two normalized poses."""
        # Normalize poses first
        norm_pose1 = self._normalize_pose(pose1)
        norm_pose2 = self._normalize_pose(pose2)

        distances = []
        max_keypoints = max(len(pose1.keypoints), len(pose2.keypoints))

        for i in range(max_keypoints):
            # Check if we have valid normalized keypoints
            if (
                norm_pose1
                and i < len(norm_pose1)
                and norm_pose1[i] is not None
                and norm_pose2
                and i < len(norm_pose2)
                and norm_pose2[i] is not None
            ):

                x1, y1, _ = norm_pose1[i]
                x2, y2, _ = norm_pose2[i]

                # Use Euclidean distance in normalized space
                dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                distances.append(float(dist))
            else:
                distances.append(-1.0)  # -1 indicates missing keypoint

        return distances

    def _normalize_pose(
        self, pose: PoseData
    ) -> Optional[List[Optional[Tuple[float, float, float]]]]:
        """
        Normalize pose to be position, scale, and orientation invariant.

        Uses the torso (shoulders and hips) as reference for normalization.

        Args:
            pose: Pose data to normalize

        Returns:
            Normalized keypoints or None if pose cannot be normalized
        """
        keypoints = pose.keypoints

        # COCO keypoint indices
        # 5: left_shoulder, 6: right_shoulder, 11: left_hip, 12: right_hip
        left_shoulder_idx, right_shoulder_idx = 5, 6
        left_hip_idx, right_hip_idx = 11, 12

        # Check if we have the critical keypoints for normalization
        critical_kps = [
            keypoints[left_shoulder_idx],
            keypoints[right_shoulder_idx],
            keypoints[left_hip_idx],
            keypoints[right_hip_idx],
        ]

        if any(kp is None for kp in critical_kps):
            return None

        # Extract torso keypoints
        left_shoulder = np.array(critical_kps[0][:2])  # x, y only
        right_shoulder = np.array(critical_kps[1][:2])
        left_hip = np.array(critical_kps[2][:2])
        right_hip = np.array(critical_kps[3][:2])

        # Calculate torso center (average of shoulders and hips)
        torso_center = (left_shoulder + right_shoulder + left_hip + right_hip) / 4.0

        # Calculate torso scale (distance between shoulders + distance between shoulder centers and hip centers)
        shoulder_width = np.linalg.norm(right_shoulder - left_shoulder)
        shoulder_center = (left_shoulder + right_shoulder) / 2.0
        hip_center = (left_hip + right_hip) / 2.0
        torso_height = np.linalg.norm(shoulder_center - hip_center)

        # Use the maximum of width and height as scale reference
        torso_scale = max(shoulder_width, torso_height)

        if torso_scale < 1e-6:  # Avoid division by zero
            return None

        # Normalize all keypoints
        normalized_keypoints = []
        for kp in keypoints:
            if kp is None:
                normalized_keypoints.append(None)
            else:
                x, y, conf = kp
                # Translate to torso center and scale by torso size
                norm_x = (x - torso_center[0]) / torso_scale
                norm_y = (y - torso_center[1]) / torso_scale
                normalized_keypoints.append((norm_x, norm_y, conf))

        return normalized_keypoints

    def _normalize_distances(self, distances: List[float]) -> List[float]:
        """Normalize distances using standardization."""
        if len(distances) < 2:
            return distances

        distances_array = np.array(distances).reshape(-1, 1)
        normalized = self.scaler.fit_transform(distances_array)
        return normalized.flatten().tolist()

    def rank_poses(
        self, target_pose: PoseData, comparison_poses: List[PoseData]
    ) -> List[SimilarityResult]:
        """
        Rank comparison poses by similarity to target pose.

        Args:
            target_pose: Target pose to match against
            comparison_poses: List of poses to rank

        Returns:
            List of similarity results ranked by similarity score
        """
        results = []

        for comparison_pose in comparison_poses:
            similarity_score = self.calculate_similarity(target_pose, comparison_pose)
            keypoint_distances = self._calculate_keypoint_distances(
                target_pose, comparison_pose
            )

            result = SimilarityResult(
                target_image=target_pose.image_path,
                comparison_image=comparison_pose.image_path,
                similarity_score=similarity_score,
                keypoint_distances=keypoint_distances,
                rank=1,  # Temporary rank, will be updated after sorting
            )

            results.append(result)

        # Sort by similarity score (descending)
        results.sort(key=lambda x: x.similarity_score, reverse=True)

        # Assign ranks
        for i, result in enumerate(results, 1):
            result.rank = i

        return results

    def get_matching_statistics(
        self, target_pose: PoseData, comparison_poses: List[PoseData]
    ) -> dict:
        """
        Get statistics about pose matching.

        Args:
            target_pose: Target pose
            comparison_poses: Comparison poses

        Returns:
            Dictionary with matching statistics
        """
        if not comparison_poses:
            return {"total_poses": 0, "avg_similarity": 0.0}

        similarities = [
            self.calculate_similarity(target_pose, pose) for pose in comparison_poses
        ]

        return {
            "total_poses": len(comparison_poses),
            "avg_similarity": float(np.mean(similarities)),
            "max_similarity": float(np.max(similarities)),
            "min_similarity": float(np.min(similarities)),
            "std_similarity": float(np.std(similarities)),
            "similarity_distribution": {
                "high": len([s for s in similarities if s >= 0.8]),
                "medium": len([s for s in similarities if 0.5 <= s < 0.8]),
                "low": len([s for s in similarities if s < 0.5]),
            },
        }

    def set_distance_metric(self, metric: str):
        """Change the distance metric."""
        valid_metrics = ["euclidean", "cosine", "manhattan"]
        if metric in valid_metrics:
            self.distance_metric = metric
        else:
            raise ValueError(f"Invalid distance metric. Choose from: {valid_metrics}")

    def set_normalization(self, normalize: bool):
        """Enable/disable keypoint normalization."""
        self.normalize_keypoints = normalize
        if normalize and self.scaler is None:
            self.scaler = StandardScaler()
        elif not normalize:
            self.scaler = None

    def _meets_visibility_threshold(
        self, pose1: PoseData, pose2: PoseData, min_threshold: float
    ) -> bool:
        """
        Check if both poses meet the minimum visibility threshold.

        Args:
            pose1: First pose
            pose2: Second pose
            min_threshold: Minimum percentage of visible keypoints required (0.0 to 1.0)

        Returns:
            True if both poses have enough visible keypoints for meaningful comparison
        """
        # Count visible keypoints in each pose
        visible1 = sum(1 for kp in pose1.keypoints if kp is not None)
        visible2 = sum(1 for kp in pose2.keypoints if kp is not None)

        total_keypoints = len(pose1.keypoints)

        # Calculate visibility percentages
        visibility1 = visible1 / total_keypoints if total_keypoints > 0 else 0.0
        visibility2 = visible2 / total_keypoints if total_keypoints > 0 else 0.0

        # Both poses must meet the threshold
        return visibility1 >= min_threshold and visibility2 >= min_threshold
